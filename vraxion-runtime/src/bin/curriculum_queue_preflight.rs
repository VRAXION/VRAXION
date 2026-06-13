use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use vraxion_runtime::{
    encode_frame, safe_filler, ChallengerEvidence, CurriculumBlockReason, CurriculumLesson,
    CurriculumQueueLesson, NextMutationEvidence, PocketLibraryStore, PocketLifecycle,
    PocketRegistryEntry, PocketToken, PromotionEvidence, RustCurriculumRunner, SafetyGate,
    ScoreVector, StoreGuardReason, StoreMutationStats, StorePromotionCandidate,
    StoredPocketArtifact,
};

#[derive(Default)]
struct QueueMetrics {
    queues: u64,
    lessons: u64,
    queue_success: u64,
    lesson_success: u64,
    promotion_success: u64,
    flow_ground_sync: u64,
    proposal_boundary: u64,
    reload_match: u64,
    quality_delta_positive: u64,
    adversarial_cases: u64,
    adversarial_blocks: u64,
    no_active_queue_block: u64,
    unsafe_queue_block: u64,
    bad_stream_block: u64,
    stale_token_block: u64,
    stale_write_block: u64,
    bad_commit: u64,
    unsafe_promotion: u64,
}

impl QueueMetrics {
    fn passed(&self) -> bool {
        self.queues == self.queue_success
            && self.lessons == self.lesson_success
            && self.lessons == self.promotion_success
            && self.lessons == self.flow_ground_sync
            && self.lessons == self.proposal_boundary
            && self.queues == self.reload_match
            && self.queues == self.quality_delta_positive
            && self.adversarial_cases == self.adversarial_blocks
            && self.bad_commit == 0
            && self.unsafe_promotion == 0
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

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn token() -> PocketToken {
    PocketToken {
        pocket_uid: "pkt_queue_ingress",
        token_version: 4,
        min_token_version: 3,
        token_hash: "tok_queue_ingress",
        content_digest: "digest_queue_ingress",
        abi_version: "PocketABI-v1",
        capability_signature: "binary_ingress",
        utility_score: 0.95,
        safety_score: 0.99,
        reuse_score: 0.86,
        cost_score: 0.06,
    }
}

fn entry(lifecycle: PocketLifecycle) -> PocketRegistryEntry {
    PocketRegistryEntry {
        pocket_uid: "pkt_queue_ingress",
        human_alias: "queue_binary_ingress",
        artifact_path: "persistent_library/artifacts/queue_binary_ingress.json",
        content_digest: "digest_queue_ingress",
        token_hash: "tok_queue_ingress",
        abi_version: "PocketABI-v1",
        capability_signature: "binary_ingress",
        lifecycle,
    }
}

fn artifact() -> StoredPocketArtifact {
    StoredPocketArtifact {
        pocket_uid: "pkt_queue_ingress",
        content_digest: "digest_queue_ingress",
        token_hash: "tok_queue_ingress",
        abi_version: "PocketABI-v1",
        quality_delta: 0.0,
        generation: 1,
    }
}

fn base_store(lifecycle: PocketLifecycle) -> PocketLibraryStore {
    let mut store = PocketLibraryStore::new();
    store.insert_pocket(entry(lifecycle), token(), artifact());
    store
}

fn lifecycle() -> NextMutationEvidence {
    NextMutationEvidence {
        active_slot_count: 1,
        sandbox_only: true,
        proposal_only: true,
        light_probe_passed: true,
        initial_quality: 0.89,
        refined_quality: 0.999,
        golden_disc_quality: 1.0,
        unique_value_score: 0.128,
        mutation_stats: StoreMutationStats {
            attempts: 256,
            accepted: 17,
            rejected: 239,
            rollback_count: 239,
            attempts_to_s_rank: Some(39),
        },
        prune_stability_passed: true,
        challenger_defense_passed: true,
        trace_replay_passed: true,
        wrong_commit_rate: 0.0,
        direct_flow_write_violation_rate: 0.0,
        pocket_uid_present: true,
        content_digest_present: true,
        token_metadata_present: true,
    }
}

fn promotion() -> PromotionEvidence {
    PromotionEvidence {
        score: ScoreVector {
            utility: 0.96,
            safety: 0.99,
            eligible_activation: 0.93,
            generality: 0.95,
            uniqueness: 0.95,
            transfer: 0.96,
            robustness: 0.97,
            cost: 0.08,
            stability: 0.98,
            scope_clarity: 0.96,
        },
        safety: SafetyGate {
            trace_safe: true,
            no_direct_flow_write: true,
            no_credit_hijack: true,
            no_delayed_poison: true,
            no_negative_transfer: true,
            no_unsafe_high_utility: true,
            no_scope_violation: true,
        },
        challenger: ChallengerEvidence {
            challenger_sweep_passed: true,
            counterfactual_unique: true,
            reload_shadow_import_passed: true,
            long_horizon_no_harm: true,
            redundant_clone_rejected: true,
            rare_critical_preserved: true,
        },
        rare_critical: false,
        global_scope_allowed: false,
    }
}

fn candidate(slot: usize) -> StorePromotionCandidate {
    match slot {
        0 => StorePromotionCandidate {
            pocket_uid: "gold_queue_frame",
            human_alias: "queue_frame_guard",
            content_digest: "digest_gold_queue_frame",
            token_hash: "tok_gold_queue_frame",
            capability_signature: "binary_queue_frame_guard",
            quality_delta: 0.031,
        },
        1 => StorePromotionCandidate {
            pocket_uid: "gold_queue_request_match",
            human_alias: "queue_requested_feature_guard",
            content_digest: "digest_gold_queue_request_match",
            token_hash: "tok_gold_queue_request_match",
            capability_signature: "binary_queue_request_guard",
            quality_delta: 0.033,
        },
        2 => StorePromotionCandidate {
            pocket_uid: "gold_queue_trace_commit",
            human_alias: "queue_trace_commit_guard",
            content_digest: "digest_gold_queue_trace_commit",
            token_hash: "tok_gold_queue_trace_commit",
            capability_signature: "binary_queue_trace_guard",
            quality_delta: 0.034,
        },
        _ => StorePromotionCandidate {
            pocket_uid: "gold_queue_reload_guard",
            human_alias: "queue_reload_guard",
            content_digest: "digest_gold_queue_reload_guard",
            token_hash: "tok_gold_queue_reload_guard",
            capability_signature: "binary_queue_reload_guard",
            quality_delta: 0.035,
        },
    }
}

fn queue_lessons(round: u64) -> [CurriculumQueueLesson; 4] {
    let families = [
        "frame_integrity",
        "requested_feature_match",
        "trace_commit",
        "reload_writeback",
    ];
    std::array::from_fn(|idx| CurriculumQueueLesson {
        family: families[idx],
        lesson: CurriculumLesson {
            requested_feature: (((round + idx as u64 * 5) % 23) + 1) as u8,
            value: ((round + idx as u64) % 2) as u8,
            source_pocket_id: 42,
            nonce: (((round + 3) * (idx as u64 + 5)) % 31) as u8,
        },
        candidate: candidate(idx),
        lifecycle_evidence: lifecycle(),
        promotion_evidence: promotion(),
    })
}

fn note_block(metrics: &mut QueueMetrics, ok: bool) {
    metrics.adversarial_cases += 1;
    metrics.adversarial_blocks += ok as u64;
}

fn run_primary_queue(metrics: &mut QueueMetrics, round: u64) {
    metrics.queues += 1;
    let lessons = queue_lessons(round);
    metrics.lessons += lessons.len() as u64;
    let mut runner = RustCurriculumRunner::default_body(base_store(PocketLifecycle::Stable));
    let report = runner.run_queue(&lessons);
    metrics.queue_success += report.passed() as u64;
    metrics.lesson_success += report.passed_count as u64;
    metrics.promotion_success += report.promoted_count as u64;
    metrics.flow_ground_sync += report.flow_ground_sync_count as u64;
    metrics.proposal_boundary += report.proposal_boundary_count as u64;
    metrics.reload_match += report.reload_match as u64;
    metrics.quality_delta_positive += (report.quality_delta > 0.0) as u64;
    metrics.bad_commit += (!report.passed()) as u64;
}

fn run_adversarial_queue(metrics: &mut QueueMetrics, round: u64) {
    let lessons = queue_lessons(round);

    let mut no_active = RustCurriculumRunner::default_body(base_store(PocketLifecycle::Quarantine));
    let no_active_report = no_active.run_queue(&lessons);
    let no_active_ok = !no_active_report.passed()
        && no_active_report.first_failure == CurriculumBlockReason::NoActivePocket
        && no_active.body.flow.active_count() == 0;
    note_block(metrics, no_active_ok);
    metrics.no_active_queue_block += no_active_ok as u64;

    let mut unsafe_lesson = lessons[0];
    let mut unsafe_lifecycle = lifecycle();
    unsafe_lifecycle.direct_flow_write_violation_rate = 1.0;
    unsafe_lesson.lifecycle_evidence = unsafe_lifecycle;
    let mut unsafe_runner = RustCurriculumRunner::default_body(base_store(PocketLifecycle::Stable));
    let unsafe_report = unsafe_runner.run_queue(&[unsafe_lesson]);
    let unsafe_ok = !unsafe_report.passed()
        && unsafe_report.first_failure
            == CurriculumBlockReason::PromotionBlocked(StoreGuardReason::UnsafePromotion);
    note_block(metrics, unsafe_ok);
    metrics.unsafe_queue_block += unsafe_ok as u64;
    metrics.unsafe_promotion += unsafe_report.promoted_count as u64;

    let row = lessons[1].lesson;
    let mut stream = safe_filler(8);
    stream.extend(encode_frame(
        row.requested_feature.wrapping_add(1),
        row.value,
        1,
        row.nonce,
    ));
    let mut bad_stream = RustCurriculumRunner::default_body(base_store(PocketLifecycle::Stable));
    let bad_stream_verdict = bad_stream.run_binary_lesson_with_stream(
        row,
        &stream,
        lessons[1].candidate,
        lessons[1].lifecycle_evidence,
        lessons[1].promotion_evidence,
    );
    let bad_stream_ok = !bad_stream_verdict.passed
        && bad_stream_verdict.reason == CurriculumBlockReason::RuntimeDidNotCommit
        && bad_stream.body.flow.active_count() == 0;
    note_block(metrics, bad_stream_ok);
    metrics.bad_stream_block += bad_stream_ok as u64;

    let mut stale_store = base_store(PocketLifecycle::Stable);
    stale_store.tokens[0].token_version = 2;
    let stale = stale_store.guarded_load(stale_store.tokens[0]);
    let stale_ok =
        stale.reason == StoreGuardReason::LoadBlocked(vraxion_runtime::LoadBlockReason::StaleToken);
    note_block(metrics, stale_ok);
    metrics.stale_token_block += stale_ok as u64;

    let mut stale_write_store = base_store(PocketLifecycle::Stable);
    let generation_before = stale_write_store.generation;
    assert!(stale_write_store.rename_alias("pkt_queue_ingress", "renamed_queue_ingress"));
    let stale_write_ok = stale_write_store
        .concurrent_write_guard(generation_before)
        .reason
        == StoreGuardReason::ConcurrentStaleWrite;
    note_block(metrics, stale_write_ok);
    metrics.stale_write_block += stale_write_ok as u64;
}

fn write_artifacts(out: &PathBuf, rounds: u64, metrics: &QueueMetrics, seconds: f64) {
    fs::create_dir_all(out).expect("create output directory");
    fs::write(
        out.join("curriculum_queue_config.json"),
        concat!(
            "{\n",
            "  \"runner\": \"rust_curriculum_queue_preflight\",\n",
            "  \"queue_length\": 4,\n",
            "  \"lesson_families\": [\"frame_integrity\", \"requested_feature_match\", \"trace_commit\", \"reload_writeback\"],\n",
            "  \"body\": \"near_28f_32g_20x80_default\",\n",
            "  \"active_set_limit\": 4,\n",
            "  \"heartbeat_seconds\": 20\n",
            "}\n"
        ),
    )
    .expect("write config");
    fs::write(
        out.join("queue_row_samples.json"),
        concat!(
            "{\n",
            "  \"rows\": [\n",
            "    {\"family\":\"frame_integrity\",\"feature\":1,\"value\":0,\"candidate\":\"gold_queue_frame\"},\n",
            "    {\"family\":\"requested_feature_match\",\"feature\":6,\"value\":1,\"candidate\":\"gold_queue_request_match\"},\n",
            "    {\"family\":\"trace_commit\",\"feature\":11,\"value\":0,\"candidate\":\"gold_queue_trace_commit\"},\n",
            "    {\"family\":\"reload_writeback\",\"feature\":16,\"value\":1,\"candidate\":\"gold_queue_reload_guard\"}\n",
            "  ],\n",
            "  \"direct_flow_write_allowed\": false,\n",
            "  \"promotion_requires_lifecycle_and_manager_gates\": true\n",
            "}\n"
        ),
    )
    .expect("write row samples");
    let results = format!(
        concat!(
            "{{\n",
            "  \"passed\": {},\n",
            "  \"rounds\": {},\n",
            "  \"queues\": {},\n",
            "  \"lessons\": {},\n",
            "  \"queue_success_rate\": {:.6},\n",
            "  \"lesson_success_rate\": {:.6},\n",
            "  \"promotion_success_rate\": {:.6},\n",
            "  \"flow_ground_sync_rate\": {:.6},\n",
            "  \"proposal_boundary_success_rate\": {:.6},\n",
            "  \"reload_match_rate\": {:.6},\n",
            "  \"quality_delta_positive_rate\": {:.6},\n",
            "  \"adversarial_block_rate\": {:.6},\n",
            "  \"no_active_queue_block_rate\": {:.6},\n",
            "  \"unsafe_queue_block_rate\": {:.6},\n",
            "  \"bad_stream_block_rate\": {:.6},\n",
            "  \"stale_token_block_rate\": {:.6},\n",
            "  \"stale_write_block_rate\": {:.6},\n",
            "  \"bad_commit_rate\": {:.6},\n",
            "  \"unsafe_promotion_rate\": {:.6},\n",
            "  \"seconds\": {:.9},\n",
            "  \"queues_per_sec\": {:.3},\n",
            "  \"lessons_per_sec\": {:.3}\n",
            "}}\n"
        ),
        metrics.passed(),
        rounds,
        metrics.queues,
        metrics.lessons,
        ratio(metrics.queue_success, metrics.queues),
        ratio(metrics.lesson_success, metrics.lessons),
        ratio(metrics.promotion_success, metrics.lessons),
        ratio(metrics.flow_ground_sync, metrics.lessons),
        ratio(metrics.proposal_boundary, metrics.lessons),
        ratio(metrics.reload_match, metrics.queues),
        ratio(metrics.quality_delta_positive, metrics.queues),
        ratio(metrics.adversarial_blocks, metrics.adversarial_cases),
        ratio(metrics.no_active_queue_block, metrics.queues),
        ratio(metrics.unsafe_queue_block, metrics.queues),
        ratio(metrics.bad_stream_block, metrics.queues),
        ratio(metrics.stale_token_block, metrics.queues),
        ratio(metrics.stale_write_block, metrics.queues),
        ratio(metrics.bad_commit, metrics.queues),
        ratio(metrics.unsafe_promotion, metrics.lessons),
        seconds,
        metrics.queues as f64 / seconds.max(0.000_001),
        metrics.lessons as f64 / seconds.max(0.000_001),
    );
    fs::write(out.join("preflight_results.json"), results).expect("write results");
    fs::write(
        out.join("report.md"),
        format!(
            "# E71 Rust Curriculum Queue Preflight\n\n\
             ```text\n\
             passed = {}\n\
             queue_success_rate = {:.6}\n\
             lesson_success_rate = {:.6}\n\
             promotion_success_rate = {:.6}\n\
             flow_ground_sync_rate = {:.6}\n\
             adversarial_block_rate = {:.6}\n\
             bad_commit_rate = {:.6}\n\
             unsafe_promotion_rate = {:.6}\n\
             ```\n",
            metrics.passed(),
            ratio(metrics.queue_success, metrics.queues),
            ratio(metrics.lesson_success, metrics.lessons),
            ratio(metrics.promotion_success, metrics.lessons),
            ratio(metrics.flow_ground_sync, metrics.lessons),
            ratio(metrics.adversarial_blocks, metrics.adversarial_cases),
            ratio(metrics.bad_commit, metrics.queues),
            ratio(metrics.unsafe_promotion, metrics.lessons),
        ),
    )
    .expect("write report");
}

fn main() {
    let rounds = env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(10_000);
    let out = env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/pilot_wave/e71_rust_curriculum_queue_preflight"));
    fs::create_dir_all(&out).expect("create output directory");
    let progress = out.join("progress.jsonl");
    let _ = fs::remove_file(&progress);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"start\",\"rounds\":{},\"policy\":\"Rust curriculum queue\"}}",
            now_millis(),
            rounds
        ),
    );

    let start = Instant::now();
    let mut last_write = Instant::now();
    let mut metrics = QueueMetrics::default();
    for round in 0..rounds {
        run_primary_queue(&mut metrics, round);
        run_adversarial_queue(&mut metrics, round);
        if last_write.elapsed() >= Duration::from_secs(20) {
            append_jsonl(
                &progress,
                &format!(
                    "{{\"timestamp_ms\":{},\"event\":\"progress\",\"round\":{},\"queues\":{},\"lessons\":{},\"adversarial_blocks\":{},\"bad_commit\":{},\"unsafe_promotion\":{}}}",
                    now_millis(),
                    round + 1,
                    metrics.queues,
                    metrics.lessons,
                    metrics.adversarial_blocks,
                    metrics.bad_commit,
                    metrics.unsafe_promotion
                ),
            );
            last_write = Instant::now();
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    write_artifacts(&out, rounds, &metrics, seconds);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"complete\",\"passed\":{},\"queues\":{},\"lessons\":{},\"adversarial_blocks\":{},\"bad_commit\":{},\"unsafe_promotion\":{}}}",
            now_millis(),
            metrics.passed(),
            metrics.queues,
            metrics.lessons,
            metrics.adversarial_blocks,
            metrics.bad_commit,
            metrics.unsafe_promotion
        ),
    );
    println!(
        "{{\"passed\":{},\"rounds\":{},\"queue_success_rate\":{:.6},\"lesson_success_rate\":{:.6},\"adversarial_block_rate\":{:.6},\"bad_commit_rate\":{:.6},\"unsafe_promotion_rate\":{:.6},\"seconds\":{:.9},\"queues_per_sec\":{:.3},\"lessons_per_sec\":{:.3},\"out\":\"{}\"}}",
        metrics.passed(),
        rounds,
        ratio(metrics.queue_success, metrics.queues),
        ratio(metrics.lesson_success, metrics.lessons),
        ratio(metrics.adversarial_blocks, metrics.adversarial_cases),
        ratio(metrics.bad_commit, metrics.queues),
        ratio(metrics.unsafe_promotion, metrics.lessons),
        seconds,
        metrics.queues as f64 / seconds.max(0.000_001),
        metrics.lessons as f64 / seconds.max(0.000_001),
        out.display()
    );
}
