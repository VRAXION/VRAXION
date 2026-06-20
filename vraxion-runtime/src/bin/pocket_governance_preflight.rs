use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use vraxion_runtime::{
    active_pocket_set, resolve_pocket_call, LoadBlockReason, PocketLifecycle, PocketRegistryEntry,
    PocketToken,
};

#[derive(Default)]
struct Metrics {
    cases: u64,
    success: u64,
    allowed_success: u64,
    alias_rename_survival: u64,
    digest_mismatch_block: u64,
    token_swap_block: u64,
    lifecycle_block: u64,
    stale_token_block: u64,
    abi_mismatch_block: u64,
    capability_mismatch_block: u64,
    active_set_success: u64,
    unsafe_load: u64,
}

#[derive(Clone, Copy)]
struct TokenSpec {
    pocket_uid: &'static str,
    token_hash: &'static str,
    content_digest: &'static str,
    capability_signature: &'static str,
    utility_score: f32,
    safety_score: f32,
    reuse_score: f32,
    cost_score: f32,
}

impl Metrics {
    fn record(&mut self, ok: bool) {
        self.cases += 1;
        self.success += ok as u64;
    }

    fn record_allowed(&mut self, ok: bool) {
        self.record(ok);
        self.allowed_success += ok as u64;
        self.unsafe_load += (!ok) as u64;
    }

    fn record_block(&mut self, ok: bool) {
        self.record(ok);
        self.unsafe_load += (!ok) as u64;
    }

    fn passed(&self) -> bool {
        self.cases == self.success && self.unsafe_load == 0
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

fn token(spec: TokenSpec) -> PocketToken {
    PocketToken {
        pocket_uid: spec.pocket_uid,
        token_version: 4,
        min_token_version: 3,
        token_hash: spec.token_hash,
        content_digest: spec.content_digest,
        abi_version: "PocketABI-v1",
        capability_signature: spec.capability_signature,
        utility_score: spec.utility_score,
        safety_score: spec.safety_score,
        reuse_score: spec.reuse_score,
        cost_score: spec.cost_score,
    }
}

fn entry(
    pocket_uid: &'static str,
    human_alias: &'static str,
    token_hash: &'static str,
    content_digest: &'static str,
    capability_signature: &'static str,
    lifecycle: PocketLifecycle,
) -> PocketRegistryEntry {
    PocketRegistryEntry {
        pocket_uid,
        human_alias,
        artifact_path: "pockets/frozen/pocket.bin",
        content_digest,
        token_hash,
        abi_version: "PocketABI-v1",
        capability_signature,
        lifecycle,
    }
}

fn base_token() -> PocketToken {
    token(TokenSpec {
        pocket_uid: "pkt_binary_ingress_v001",
        token_hash: "tok_binary_ingress_v004",
        content_digest: "digest_binary_ingress_v001",
        capability_signature: "binary_frame_codec",
        utility_score: 0.94,
        safety_score: 0.99,
        reuse_score: 0.82,
        cost_score: 0.07,
    })
}

fn base_entry(alias: &'static str, lifecycle: PocketLifecycle) -> PocketRegistryEntry {
    entry(
        "pkt_binary_ingress_v001",
        alias,
        "tok_binary_ingress_v004",
        "digest_binary_ingress_v001",
        "binary_frame_codec",
        lifecycle,
    )
}

fn block_reason_ok(
    pocket_descriptor: PocketToken,
    registry: &[PocketRegistryEntry],
    reason: LoadBlockReason,
) -> bool {
    let decision = resolve_pocket_call(pocket_descriptor, registry);
    !decision.allowed && decision.reason == Some(reason)
}

fn run_round(idx: u64, metrics: &mut Metrics) {
    let tok = base_token();

    let allowed = resolve_pocket_call(
        tok,
        &[base_entry("binary_frame_codec", PocketLifecycle::Stable)],
    );
    metrics.record_allowed(allowed.allowed);

    let renamed = resolve_pocket_call(
        tok,
        &[base_entry(
            "protocol_framing_ingress",
            PocketLifecycle::Stable,
        )],
    );
    let alias_ok = renamed.allowed;
    metrics.record_allowed(alias_ok);
    metrics.alias_rename_survival += alias_ok as u64;

    let mut digest_bad = base_entry("binary_frame_codec", PocketLifecycle::Stable);
    digest_bad.content_digest = "digest_wrong";
    let digest_ok = block_reason_ok(tok, &[digest_bad], LoadBlockReason::ContentDigestMismatch);
    metrics.record_block(digest_ok);
    metrics.digest_mismatch_block += digest_ok as u64;

    let mut swapped = base_entry("binary_frame_codec", PocketLifecycle::Stable);
    swapped.token_hash = "tok_wrong";
    let swap_ok = block_reason_ok(tok, &[swapped], LoadBlockReason::TokenBindingMismatch);
    metrics.record_block(swap_ok);
    metrics.token_swap_block += swap_ok as u64;

    let lifecycle = if idx & 1 == 0 {
        PocketLifecycle::Quarantine
    } else {
        PocketLifecycle::Banned
    };
    let lifecycle_ok = block_reason_ok(
        tok,
        &[base_entry("binary_frame_codec", lifecycle)],
        LoadBlockReason::LifecycleBlocked,
    );
    metrics.record_block(lifecycle_ok);
    metrics.lifecycle_block += lifecycle_ok as u64;

    let stale = PocketToken {
        token_version: 2,
        ..tok
    };
    let stale_ok = block_reason_ok(
        stale,
        &[base_entry("binary_frame_codec", PocketLifecycle::Stable)],
        LoadBlockReason::StaleToken,
    );
    metrics.record_block(stale_ok);
    metrics.stale_token_block += stale_ok as u64;

    let mut abi_bad = base_entry("binary_frame_codec", PocketLifecycle::Stable);
    abi_bad.abi_version = "PocketABI-v0";
    let abi_ok = block_reason_ok(tok, &[abi_bad], LoadBlockReason::AbiMismatch);
    metrics.record_block(abi_ok);
    metrics.abi_mismatch_block += abi_ok as u64;

    let mut capability_bad = base_entry("binary_frame_codec", PocketLifecycle::Stable);
    capability_bad.capability_signature = "wrong_capability";
    let capability_ok =
        block_reason_ok(tok, &[capability_bad], LoadBlockReason::CapabilityMismatch);
    metrics.record_block(capability_ok);
    metrics.capability_mismatch_block += capability_ok as u64;
}

fn run_active_set_case(metrics: &mut Metrics) {
    let t1 = token(TokenSpec {
        pocket_uid: "pkt_binary_ingress_v001",
        token_hash: "tok_1",
        content_digest: "digest_1",
        capability_signature: "binary_frame_codec",
        utility_score: 0.94,
        safety_score: 0.99,
        reuse_score: 0.82,
        cost_score: 0.07,
    });
    let t2 = token(TokenSpec {
        pocket_uid: "pkt_text_lens_v001",
        token_hash: "tok_2",
        content_digest: "digest_2",
        capability_signature: "text_field_lens",
        utility_score: 0.86,
        safety_score: 0.96,
        reuse_score: 0.76,
        cost_score: 0.10,
    });
    let t3 = token(TokenSpec {
        pocket_uid: "pkt_route_helper_v001",
        token_hash: "tok_3",
        content_digest: "digest_3",
        capability_signature: "route_helper",
        utility_score: 0.89,
        safety_score: 0.98,
        reuse_score: 0.70,
        cost_score: 0.06,
    });
    let t4 = token(TokenSpec {
        pocket_uid: "pkt_rare_critical_v001",
        token_hash: "tok_4",
        content_digest: "digest_4",
        capability_signature: "rare_critical_guard",
        utility_score: 0.77,
        safety_score: 1.00,
        reuse_score: 0.40,
        cost_score: 0.04,
    });
    let blocked = token(TokenSpec {
        pocket_uid: "pkt_quarantine_v001",
        token_hash: "tok_5",
        content_digest: "digest_5",
        capability_signature: "unsafe_specialist",
        utility_score: 0.99,
        safety_score: 0.10,
        reuse_score: 0.90,
        cost_score: 0.01,
    });
    let registry = [
        entry(
            "pkt_binary_ingress_v001",
            "binary",
            "tok_1",
            "digest_1",
            "binary_frame_codec",
            PocketLifecycle::Core,
        ),
        entry(
            "pkt_text_lens_v001",
            "text",
            "tok_2",
            "digest_2",
            "text_field_lens",
            PocketLifecycle::Stable,
        ),
        entry(
            "pkt_route_helper_v001",
            "route",
            "tok_3",
            "digest_3",
            "route_helper",
            PocketLifecycle::Active,
        ),
        entry(
            "pkt_rare_critical_v001",
            "rare",
            "tok_4",
            "digest_4",
            "rare_critical_guard",
            PocketLifecycle::Specialist,
        ),
        entry(
            "pkt_quarantine_v001",
            "unsafe",
            "tok_5",
            "digest_5",
            "unsafe_specialist",
            PocketLifecycle::Quarantine,
        ),
    ];
    let active = active_pocket_set(&[t1, t2, t3, t4, blocked], &registry, 3);
    let ok = active.len() == 3
        && active
            .iter()
            .all(|pocket| pocket.pocket_uid != "pkt_quarantine_v001")
        && active.windows(2).all(|window| {
            window[0].routing_score >= window[1].routing_score
                || window[0].pocket_uid <= window[1].pocket_uid
        });
    metrics.record(ok);
    metrics.active_set_success += ok as u64;
    metrics.unsafe_load += (!ok) as u64;
}

fn write_artifacts(out: &PathBuf, rounds: u64, metrics: &Metrics, seconds: f64) {
    fs::create_dir_all(out).expect("create output directory");
    fs::write(
        out.join("runtime_governance_config.json"),
        concat!(
            "{\n",
            "  \"policy\": \"PocketToken + Registry + Manager/Agency Guard\",\n",
            "  \"identity\": \"pocket_uid\",\n",
            "  \"integrity\": \"content_digest\",\n",
            "  \"binding\": \"token_hash\",\n",
            "  \"abi\": \"PocketABI-v1\",\n",
            "  \"load_allowed_lifecycles\": [\"Active\", \"Stable\", \"Specialist\", \"Core\"],\n",
            "  \"blocked_lifecycles\": [\"Candidate\", \"Quarantine\", \"Deprecated\", \"Banned\"]\n",
            "}\n"
        ),
    )
    .expect("write governance config");

    let active_set_reduction = 1.0 - (3.0 / 5.0);
    let results = format!(
        concat!(
            "{{\n",
            "  \"passed\": {},\n",
            "  \"rounds\": {},\n",
            "  \"cases\": {},\n",
            "  \"success\": {},\n",
            "  \"unsafe_load\": {},\n",
            "  \"allowed_success\": {},\n",
            "  \"alias_rename_survival\": {},\n",
            "  \"digest_mismatch_block\": {},\n",
            "  \"token_swap_block\": {},\n",
            "  \"lifecycle_block\": {},\n",
            "  \"stale_token_block\": {},\n",
            "  \"abi_mismatch_block\": {},\n",
            "  \"capability_mismatch_block\": {},\n",
            "  \"active_set_success\": {},\n",
            "  \"active_set_reduction\": {:.6},\n",
            "  \"seconds\": {:.9},\n",
            "  \"rows_per_sec\": {:.3}\n",
            "}}\n"
        ),
        metrics.passed(),
        rounds,
        metrics.cases,
        metrics.success,
        metrics.unsafe_load,
        metrics.allowed_success,
        metrics.alias_rename_survival,
        metrics.digest_mismatch_block,
        metrics.token_swap_block,
        metrics.lifecycle_block,
        metrics.stale_token_block,
        metrics.abi_mismatch_block,
        metrics.capability_mismatch_block,
        metrics.active_set_success,
        active_set_reduction,
        seconds,
        metrics.cases as f64 / seconds.max(0.000_001),
    );
    fs::write(out.join("preflight_results.json"), results).expect("write results");
    fs::write(
        out.join("report.md"),
        format!(
            "# E66 Pocket Registry Runtime Governance Preflight\n\n\
             ```text\n\
             passed = {}\n\
             policy = PocketToken + Registry + Manager/Agency Guard\n\
             cases = {}\n\
             success = {}\n\
             unsafe_load = {}\n\
             alias_rename_survival = {}\n\
             digest_mismatch_block = {}\n\
             token_swap_block = {}\n\
             lifecycle_block = {}\n\
             stale_token_block = {}\n\
             active_set_reduction = {:.6}\n\
             ```\n",
            metrics.passed(),
            metrics.cases,
            metrics.success,
            metrics.unsafe_load,
            metrics.alias_rename_survival,
            metrics.digest_mismatch_block,
            metrics.token_swap_block,
            metrics.lifecycle_block,
            metrics.stale_token_block,
            active_set_reduction,
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
        PathBuf::from("target/pilot_wave/e66_pocket_registry_runtime_governance_preflight")
    });
    fs::create_dir_all(&out).expect("create output directory");
    let progress = out.join("progress.jsonl");
    let _ = fs::remove_file(&progress);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"start\",\"rounds\":{},\"policy\":\"PocketToken+Registry+Guard\"}}",
            now_millis(),
            rounds
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
                    "{{\"timestamp_ms\":{},\"event\":\"progress\",\"round\":{},\"cases\":{},\"success\":{},\"unsafe_load\":{}}}",
                    now_millis(),
                    idx + 1,
                    metrics.cases,
                    metrics.success,
                    metrics.unsafe_load
                ),
            );
            last_write = Instant::now();
        }
    }
    run_active_set_case(&mut metrics);
    let seconds = start.elapsed().as_secs_f64();
    write_artifacts(&out, rounds, &metrics, seconds);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"complete\",\"passed\":{},\"cases\":{},\"success\":{},\"unsafe_load\":{}}}",
            now_millis(),
            metrics.passed(),
            metrics.cases,
            metrics.success,
            metrics.unsafe_load
        ),
    );
    println!(
        "{{\"passed\":{},\"rounds\":{},\"cases\":{},\"success\":{},\"unsafe_load\":{},\"seconds\":{:.9},\"rows_per_sec\":{:.3},\"out\":\"{}\"}}",
        metrics.passed(),
        rounds,
        metrics.cases,
        metrics.success,
        metrics.unsafe_load,
        seconds,
        metrics.cases as f64 / seconds.max(0.000_001),
        out.display()
    );
}
