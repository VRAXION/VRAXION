use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use vraxion_runtime::{
    corrupt_crc, encode_frame, safe_filler, Action, LockedBodyRuntime, Proposal, ProposalField,
    ProposalFieldError, ProposalKind, DEFAULT_BODY, EXTENDED_BODY, OVERCAPACITY_AVOID_DEFAULT,
    PROPOSAL_WIDTH_64_CONTROL, RESEARCH_CEILING_BODY,
};

#[derive(Default)]
struct Metrics {
    cases: u64,
    success: u64,
    clean_commit_success: u64,
    invalid_reject_success: u64,
    false_commit: u64,
    missed_commit: u64,
    proposal_width_reject_success: u64,
    proposal_overflow_reject_success: u64,
}

impl Metrics {
    fn record(&mut self, ok: bool) {
        self.cases += 1;
        self.success += ok as u64;
    }

    fn passed(&self) -> bool {
        self.cases == self.success && self.false_commit == 0 && self.missed_commit == 0
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_millis()
}

fn append_jsonl(path: &PathBuf, line: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create progress parent");
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open progress file");
    writeln!(file, "{line}").expect("write progress line");
}

fn clean_stream(feature: u8, value: u8, nonce: u8) -> Vec<u8> {
    let mut stream = safe_filler(8);
    stream.extend(encode_frame(feature, value, 1, nonce));
    stream.extend(safe_filler(8));
    stream
}

fn wrong_feature_stream(feature: u8, value: u8, nonce: u8) -> Vec<u8> {
    let mut stream = safe_filler(8);
    stream.extend(encode_frame((feature + 11) & 31, value ^ 1, 1, nonce));
    stream.extend(safe_filler(8));
    stream
}

fn corrupt_stream(feature: u8, value: u8, nonce: u8) -> Vec<u8> {
    let mut stream = safe_filler(8);
    stream.extend(corrupt_crc(&encode_frame(feature, value, 1, nonce)));
    stream.extend(safe_filler(8));
    stream
}

fn run_round(idx: u64, metrics: &mut Metrics) {
    let feature = (idx % 31) as u8;
    let value = (idx & 1) as u8;

    let mut runtime = LockedBodyRuntime::default_body();
    let clean = runtime.process_binary_evidence(42, feature, &clean_stream(feature, value, 1));
    let clean_ok = clean.action == Action::CommitEvidence
        && clean.committed.map(|record| record.feature_id) == Some(feature)
        && clean.committed.map(|record| record.value) == Some(value)
        && clean.rendered.compact == "COMMIT_EVIDENCE";
    metrics.record(clean_ok);
    metrics.clean_commit_success += clean_ok as u64;
    metrics.missed_commit += (!clean_ok) as u64;

    let mut runtime = LockedBodyRuntime::default_body();
    let wrong =
        runtime.process_binary_evidence(42, feature, &wrong_feature_stream(feature, value, 2));
    let wrong_ok = wrong.action != Action::CommitEvidence
        && wrong.committed.is_none()
        && runtime.flow.active_count() == 0
        && runtime.ground.active_count() == 0;
    metrics.record(wrong_ok);
    metrics.invalid_reject_success += wrong_ok as u64;
    metrics.false_commit += (!wrong_ok) as u64;

    let mut runtime = LockedBodyRuntime::default_body();
    let corrupt = runtime.process_binary_evidence(42, feature, &corrupt_stream(feature, value, 3));
    let corrupt_ok = corrupt.action != Action::CommitEvidence
        && corrupt.committed.is_none()
        && runtime.flow.active_count() == 0
        && runtime.ground.active_count() == 0;
    metrics.record(corrupt_ok);
    metrics.invalid_reject_success += corrupt_ok as u64;
    metrics.false_commit += (!corrupt_ok) as u64;
}

fn run_proposal_capacity_cases(metrics: &mut Metrics) {
    let proposal = Proposal {
        kind: ProposalKind::EvidenceWrite,
        cycle_id: 1,
        source_pocket_id: 42,
        trace_valid: true,
        ground_compatible: true,
        target_feature: Some(7),
        value: Some(1),
    };
    let mut narrow = ProposalField::new(PROPOSAL_WIDTH_64_CONTROL);
    let width_ok = narrow.push(proposal) == Err(ProposalFieldError::SlotWidthTooSmall);
    metrics.record(width_ok);
    metrics.proposal_width_reject_success += width_ok as u64;

    let mut field = ProposalField::new(DEFAULT_BODY);
    let mut overflow_ok = true;
    for _ in 0..DEFAULT_BODY.proposal_slots {
        overflow_ok &= field.push(proposal).is_ok();
    }
    overflow_ok &= field.push(proposal) == Err(ProposalFieldError::SlotOverflow);
    metrics.record(overflow_ok);
    metrics.proposal_overflow_reject_success += overflow_ok as u64;
}

fn write_artifacts(out: &PathBuf, rounds: u64, metrics: &Metrics, seconds: f64) {
    fs::create_dir_all(out).expect("create output directory");
    let runtime_config = format!(
        concat!(
            "{{\n",
            "  \"default_body\": {{\"name\":\"{}\",\"flow_shape\":[{},{}],\"ground_shape\":[{},{}],\"proposal_slots\":{},\"proposal_bits\":{},\"agency_view_bits\":{},\"total_work_cells\":{}}},\n",
            "  \"extended_body\": {{\"name\":\"{}\",\"flow_shape\":[{},{}],\"ground_shape\":[{},{}],\"proposal_slots\":{},\"proposal_bits\":{},\"agency_view_bits\":{}}},\n",
            "  \"research_ceiling_body\": {{\"name\":\"{}\",\"flow_shape\":[{},{}],\"ground_shape\":[{},{}]}},\n",
            "  \"avoid_default_body\": {{\"name\":\"{}\",\"flow_shape\":[{},{}],\"ground_shape\":[{},{}]}}\n",
            "}}\n"
        ),
        DEFAULT_BODY.name,
        DEFAULT_BODY.flow_side,
        DEFAULT_BODY.flow_side,
        DEFAULT_BODY.ground_side,
        DEFAULT_BODY.ground_side,
        DEFAULT_BODY.proposal_slots,
        DEFAULT_BODY.proposal_bits,
        DEFAULT_BODY.agency_view_bits,
        DEFAULT_BODY.total_work_cells(),
        EXTENDED_BODY.name,
        EXTENDED_BODY.flow_side,
        EXTENDED_BODY.flow_side,
        EXTENDED_BODY.ground_side,
        EXTENDED_BODY.ground_side,
        EXTENDED_BODY.proposal_slots,
        EXTENDED_BODY.proposal_bits,
        EXTENDED_BODY.agency_view_bits,
        RESEARCH_CEILING_BODY.name,
        RESEARCH_CEILING_BODY.flow_side,
        RESEARCH_CEILING_BODY.flow_side,
        RESEARCH_CEILING_BODY.ground_side,
        RESEARCH_CEILING_BODY.ground_side,
        OVERCAPACITY_AVOID_DEFAULT.name,
        OVERCAPACITY_AVOID_DEFAULT.flow_side,
        OVERCAPACITY_AVOID_DEFAULT.flow_side,
        OVERCAPACITY_AVOID_DEFAULT.ground_side,
        OVERCAPACITY_AVOID_DEFAULT.ground_side,
    );
    fs::write(out.join("runtime_config.json"), runtime_config).expect("write runtime config");

    let results = format!(
        concat!(
            "{{\n",
            "  \"passed\": {},\n",
            "  \"rounds\": {},\n",
            "  \"cases\": {},\n",
            "  \"success\": {},\n",
            "  \"false_commit\": {},\n",
            "  \"missed_commit\": {},\n",
            "  \"clean_commit_success\": {},\n",
            "  \"invalid_reject_success\": {},\n",
            "  \"proposal_width_reject_success\": {},\n",
            "  \"proposal_overflow_reject_success\": {},\n",
            "  \"seconds\": {:.9},\n",
            "  \"rows_per_sec\": {:.3}\n",
            "}}\n"
        ),
        metrics.passed(),
        rounds,
        metrics.cases,
        metrics.success,
        metrics.false_commit,
        metrics.missed_commit,
        metrics.clean_commit_success,
        metrics.invalid_reject_success,
        metrics.proposal_width_reject_success,
        metrics.proposal_overflow_reject_success,
        seconds,
        metrics.cases as f64 / seconds.max(0.000_001),
    );
    fs::write(out.join("preflight_results.json"), results).expect("write results");
    fs::write(
        out.join("report.md"),
        format!(
            "# E65 Locked Body Runtime Integration Preflight\n\n\
             ```text\n\
             passed = {}\n\
             default_body = {}\n\
             Flow = {}x{}\n\
             Ground = {}x{}\n\
             Proposal = {}x{} bits\n\
             Agency View = {} bits\n\
             cases = {}\n\
             false_commit = {}\n\
             missed_commit = {}\n\
             ```\n",
            metrics.passed(),
            DEFAULT_BODY.name,
            DEFAULT_BODY.flow_side,
            DEFAULT_BODY.flow_side,
            DEFAULT_BODY.ground_side,
            DEFAULT_BODY.ground_side,
            DEFAULT_BODY.proposal_slots,
            DEFAULT_BODY.proposal_bits,
            DEFAULT_BODY.agency_view_bits,
            metrics.cases,
            metrics.false_commit,
            metrics.missed_commit,
        ),
    )
    .expect("write report");
}

fn main() {
    let rounds = env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(10_000);
    let out = env::args().nth(2).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("target/pilot_wave/e65_locked_body_runtime_integration_preflight")
    });
    fs::create_dir_all(&out).expect("create output directory");
    let progress = out.join("progress.jsonl");
    let _ = fs::remove_file(&progress);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"start\",\"rounds\":{},\"default_body\":\"{}\"}}",
            now_millis(),
            rounds,
            DEFAULT_BODY.name
        ),
    );

    let start = Instant::now();
    let mut last_write = Instant::now();
    let mut metrics = Metrics::default();
    for idx in 0..rounds {
        run_round(idx, &mut metrics);
        if last_write.elapsed() >= Duration::from_secs(20) {
            append_jsonl(
                &progress,
                &format!(
                    "{{\"timestamp_ms\":{},\"event\":\"progress\",\"round\":{},\"cases\":{},\"success\":{}}}",
                    now_millis(),
                    idx + 1,
                    metrics.cases,
                    metrics.success
                ),
            );
            last_write = Instant::now();
        }
    }
    run_proposal_capacity_cases(&mut metrics);
    let seconds = start.elapsed().as_secs_f64();
    write_artifacts(&out, rounds, &metrics, seconds);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"complete\",\"passed\":{},\"cases\":{},\"success\":{}}}",
            now_millis(),
            metrics.passed(),
            metrics.cases,
            metrics.success
        ),
    );
    println!(
        "{{\"passed\":{},\"rounds\":{},\"cases\":{},\"success\":{},\"false_commit\":{},\"missed_commit\":{},\"seconds\":{:.9},\"rows_per_sec\":{:.3},\"out\":\"{}\"}}",
        metrics.passed(),
        rounds,
        metrics.cases,
        metrics.success,
        metrics.false_commit,
        metrics.missed_commit,
        seconds,
        metrics.cases as f64 / seconds.max(0.000_001),
        out.display()
    );
}
