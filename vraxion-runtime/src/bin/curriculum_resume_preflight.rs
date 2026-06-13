use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use vraxion_runtime::{
    audit_resume, ChallengerEvidence, CurriculumCheckpoint, CurriculumLesson,
    CurriculumQueueLesson, NextMutationEvidence, PocketLibraryStore, PocketLifecycle,
    PocketRegistryEntry, PocketToken, PromotionEvidence, RustCurriculumRunner, SafetyGate,
    ScoreVector, StoreMutationStats, StorePromotionCandidate, StoredPocketArtifact,
};

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

fn token() -> PocketToken {
    PocketToken {
        pocket_uid: "pkt_resume_ingress",
        token_version: 4,
        min_token_version: 3,
        token_hash: "tok_resume_ingress",
        content_digest: "digest_resume_ingress",
        abi_version: "PocketABI-v1",
        capability_signature: "binary_ingress",
        utility_score: 0.95,
        safety_score: 0.99,
        reuse_score: 0.86,
        cost_score: 0.06,
    }
}

fn entry() -> PocketRegistryEntry {
    PocketRegistryEntry {
        pocket_uid: "pkt_resume_ingress",
        human_alias: "resume_binary_ingress",
        artifact_path: "persistent_library/artifacts/resume_binary_ingress.json",
        content_digest: "digest_resume_ingress",
        token_hash: "tok_resume_ingress",
        abi_version: "PocketABI-v1",
        capability_signature: "binary_ingress",
        lifecycle: PocketLifecycle::Stable,
    }
}

fn artifact() -> StoredPocketArtifact {
    StoredPocketArtifact {
        pocket_uid: "pkt_resume_ingress",
        content_digest: "digest_resume_ingress",
        token_hash: "tok_resume_ingress",
        abi_version: "PocketABI-v1",
        quality_delta: 0.0,
        generation: 1,
    }
}

fn base_runner() -> RustCurriculumRunner {
    let mut store = PocketLibraryStore::new();
    store.insert_pocket(entry(), token(), artifact());
    RustCurriculumRunner::default_body(store)
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
            pocket_uid: "gold_resume_frame",
            human_alias: "resume_frame_guard",
            content_digest: "digest_gold_resume_frame",
            token_hash: "tok_gold_resume_frame",
            capability_signature: "binary_resume_frame_guard",
            quality_delta: 0.031,
        },
        1 => StorePromotionCandidate {
            pocket_uid: "gold_resume_request",
            human_alias: "resume_requested_feature_guard",
            content_digest: "digest_gold_resume_request",
            token_hash: "tok_gold_resume_request",
            capability_signature: "binary_resume_request_guard",
            quality_delta: 0.033,
        },
        2 => StorePromotionCandidate {
            pocket_uid: "gold_resume_trace",
            human_alias: "resume_trace_commit_guard",
            content_digest: "digest_gold_resume_trace",
            token_hash: "tok_gold_resume_trace",
            capability_signature: "binary_resume_trace_guard",
            quality_delta: 0.034,
        },
        _ => StorePromotionCandidate {
            pocket_uid: "gold_resume_reload",
            human_alias: "resume_reload_guard",
            content_digest: "digest_gold_resume_reload",
            token_hash: "tok_gold_resume_reload",
            capability_signature: "binary_resume_reload_guard",
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

fn run_range(
    runner: &mut RustCurriculumRunner,
    checkpoint: &mut CurriculumCheckpoint,
    start: u64,
    end: u64,
    progress: &PathBuf,
    phase: &str,
) {
    let mut last_heartbeat = Instant::now();
    for queue_index in start..end {
        let lessons = queue_lessons(queue_index);
        let report = runner.run_queue(&lessons);
        checkpoint.record_queue(queue_index, report, runner.store.generation);
        if last_heartbeat.elapsed() >= Duration::from_secs(20) {
            append_jsonl(
                progress,
                &format!(
                    "{{\"timestamp_ms\":{},\"event\":\"heartbeat\",\"phase\":\"{}\",\"queue_index\":{},\"range_end\":{},\"completed_queues\":{},\"completed_lessons\":{},\"checksum\":{}}}",
                    now_millis(),
                    phase,
                    queue_index + 1,
                    end,
                    checkpoint.completed_queues,
                    checkpoint.completed_lessons,
                    checkpoint.checksum
                ),
            );
            last_heartbeat = Instant::now();
        }
    }
}

fn write_checkpoint(path: &PathBuf, checkpoint: CurriculumCheckpoint) {
    fs::write(
        path,
        format!(
            concat!(
                "{{\n",
                "  \"run_id\": {},\n",
                "  \"completed_queues\": {},\n",
                "  \"completed_lessons\": {},\n",
                "  \"promoted_count\": {},\n",
                "  \"failed_count\": {},\n",
                "  \"bad_commit_count\": {},\n",
                "  \"unsafe_promotion_count\": {},\n",
                "  \"store_generation\": {},\n",
                "  \"quality_delta\": {:.6},\n",
                "  \"checksum\": {}\n",
                "}}\n"
            ),
            checkpoint.run_id,
            checkpoint.completed_queues,
            checkpoint.completed_lessons,
            checkpoint.promoted_count,
            checkpoint.failed_count,
            checkpoint.bad_commit_count,
            checkpoint.unsafe_promotion_count,
            checkpoint.store_generation,
            checkpoint.quality_delta,
            checkpoint.checksum,
        ),
    )
    .expect("write checkpoint");
}

fn main() {
    let rounds = env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(100_000);
    let out = env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/pilot_wave/e72_rust_curriculum_resume_preflight"));
    fs::create_dir_all(&out).expect("create output directory");
    let progress = out.join("progress.jsonl");
    let _ = fs::remove_file(&progress);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"start\",\"rounds\":{},\"policy\":\"Rust curriculum resume\"}}",
            now_millis(),
            rounds
        ),
    );

    let split = rounds / 2;
    let start = Instant::now();

    let mut reference_runner = base_runner();
    let mut reference = CurriculumCheckpoint::new(72);
    run_range(
        &mut reference_runner,
        &mut reference,
        0,
        rounds,
        &progress,
        "reference",
    );
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"reference_complete\",\"queues\":{},\"lessons\":{},\"checksum\":{}}}",
            now_millis(),
            reference.completed_queues,
            reference.completed_lessons,
            reference.checksum
        ),
    );

    let mut first_runner = base_runner();
    let mut mid = CurriculumCheckpoint::new(72);
    run_range(
        &mut first_runner,
        &mut mid,
        0,
        split,
        &progress,
        "checkpoint_first_half",
    );
    write_checkpoint(&out.join("checkpoint_mid.json"), mid);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"checkpoint_mid\",\"queues\":{},\"lessons\":{},\"checksum\":{}}}",
            now_millis(),
            mid.completed_queues,
            mid.completed_lessons,
            mid.checksum
        ),
    );

    let checkpoint_compatible = mid.resume_compatible(split);
    let mut resumed_runner = first_runner.clone();
    let mut resumed = mid;
    run_range(
        &mut resumed_runner,
        &mut resumed,
        split,
        rounds,
        &progress,
        "resume_second_half",
    );
    write_checkpoint(&out.join("checkpoint_final.json"), resumed);
    let audit = audit_resume(reference, resumed);
    let seconds = start.elapsed().as_secs_f64();

    fs::write(
        out.join("resume_config.json"),
        format!(
            concat!(
                "{{\n",
                "  \"runner\": \"rust_curriculum_resume_preflight\",\n",
                "  \"rounds\": {},\n",
                "  \"split\": {},\n",
                "  \"queue_length\": 4,\n",
                "  \"heartbeat_seconds\": 20,\n",
                "  \"checkpoint_files\": [\"checkpoint_mid.json\", \"checkpoint_final.json\"]\n",
                "}}\n"
            ),
            rounds, split
        ),
    )
    .expect("write config");
    let passed = checkpoint_compatible && audit.passed();
    fs::write(
        out.join("preflight_results.json"),
        format!(
            concat!(
                "{{\n",
                "  \"passed\": {},\n",
                "  \"rounds\": {},\n",
                "  \"split\": {},\n",
                "  \"reference_queues\": {},\n",
                "  \"resumed_queues\": {},\n",
                "  \"reference_lessons\": {},\n",
                "  \"resumed_lessons\": {},\n",
                "  \"checkpoint_resume_compatible\": {},\n",
                "  \"final_checksum_match\": {},\n",
                "  \"final_queue_match\": {},\n",
                "  \"final_lesson_match\": {},\n",
                "  \"final_promotion_match\": {},\n",
                "  \"bad_commit_rate\": {:.6},\n",
                "  \"unsafe_promotion_rate\": {:.6},\n",
                "  \"seconds\": {:.9},\n",
                "  \"queues_per_sec\": {:.3},\n",
                "  \"lessons_per_sec\": {:.3}\n",
                "}}\n"
            ),
            passed,
            rounds,
            split,
            reference.completed_queues,
            resumed.completed_queues,
            reference.completed_lessons,
            resumed.completed_lessons,
            checkpoint_compatible,
            audit.final_checksum_match,
            audit.final_queue_match,
            audit.final_lesson_match,
            audit.final_promotion_match,
            resumed.bad_commit_count as f64 / resumed.completed_queues.max(1) as f64,
            resumed.unsafe_promotion_count as f64 / resumed.completed_lessons.max(1) as f64,
            seconds,
            (reference.completed_queues + resumed.completed_queues) as f64 / seconds.max(0.000_001),
            (reference.completed_lessons + resumed.completed_lessons) as f64
                / seconds.max(0.000_001),
        ),
    )
    .expect("write results");
    fs::write(
        out.join("report.md"),
        format!(
            "# E72 Rust Curriculum Resume Preflight\n\n\
             ```text\n\
             passed = {}\n\
             checkpoint_resume_compatible = {}\n\
             final_checksum_match = {}\n\
             final_queue_match = {}\n\
             final_lesson_match = {}\n\
             final_promotion_match = {}\n\
             bad_commit_rate = {:.6}\n\
             unsafe_promotion_rate = {:.6}\n\
             ```\n",
            passed,
            checkpoint_compatible,
            audit.final_checksum_match,
            audit.final_queue_match,
            audit.final_lesson_match,
            audit.final_promotion_match,
            resumed.bad_commit_count as f64 / resumed.completed_queues.max(1) as f64,
            resumed.unsafe_promotion_count as f64 / resumed.completed_lessons.max(1) as f64,
        ),
    )
    .expect("write report");
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"complete\",\"passed\":{},\"resumed_queues\":{},\"resumed_lessons\":{},\"checksum_match\":{}}}",
            now_millis(),
            passed,
            resumed.completed_queues,
            resumed.completed_lessons,
            audit.final_checksum_match
        ),
    );

    println!(
        "{{\"passed\":{},\"rounds\":{},\"checkpoint_resume_compatible\":{},\"final_checksum_match\":{},\"bad_commit_rate\":{:.6},\"unsafe_promotion_rate\":{:.6},\"seconds\":{:.9},\"out\":\"{}\"}}",
        passed,
        rounds,
        checkpoint_compatible,
        audit.final_checksum_match,
        resumed.bad_commit_count as f64 / resumed.completed_queues.max(1) as f64,
        resumed.unsafe_promotion_count as f64 / resumed.completed_lessons.max(1) as f64,
        seconds,
        out.display()
    );
}
