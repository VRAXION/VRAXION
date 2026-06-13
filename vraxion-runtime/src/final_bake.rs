use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::{
    active_pocket_set, audit_resume, corrupt_crc, corrupt_length, encode_frame,
    evaluate_next_mutation_lifecycle, evaluate_promotion, reassemble_requested_frame,
    render_output, resolve_pocket_call, safe_filler, select_text_mode, Action, ChallengerEvidence,
    CurriculumCheckpoint, CurriculumLesson, CurriculumQueueLesson, EgressMode, LockedBodyRuntime,
    MutationBlockReason, MutationLifecycleStage, MutationStats, NextMutationEvidence,
    PocketLibraryStore, PocketLifecycle, PocketRegistryEntry, PocketToken, PromotionBlockReason,
    PromotionEvidence, PromotionLevel, RustCurriculumRunner, SafetyGate, ScoreVector,
    StoreGuardReason, StorePromotionCandidate, StoredPocketArtifact, TextMode, TextProfile,
};

#[derive(Debug, Clone, Copy, Default)]
struct GateMetrics {
    cases: u64,
    passed: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FinalBakeSummary {
    pub passed: bool,
    pub rounds: u64,
    pub resume_passed: bool,
    pub final_checksum_match: bool,
    pub bad_commit_rate: f64,
    pub unsafe_promotion_rate: f64,
    pub seconds: f64,
    pub out: PathBuf,
}

impl GateMetrics {
    fn note(&mut self, passed: bool) {
        self.cases += 1;
        self.passed += passed as u64;
    }

    fn all_passed(self) -> bool {
        self.cases > 0 && self.cases == self.passed
    }
}

pub(crate) fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_millis()
}

pub(crate) fn append_jsonl(path: &PathBuf, line: &str) {
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

fn token(uid: &'static str, digest: &'static str, token_hash: &'static str) -> PocketToken {
    PocketToken {
        pocket_uid: uid,
        token_version: 4,
        min_token_version: 3,
        token_hash,
        content_digest: digest,
        abi_version: "PocketABI-v1",
        capability_signature: "binary_ingress",
        utility_score: 0.95,
        safety_score: 0.99,
        reuse_score: 0.86,
        cost_score: 0.06,
    }
}

fn entry(
    uid: &'static str,
    digest: &'static str,
    token_hash: &'static str,
    lifecycle: PocketLifecycle,
) -> PocketRegistryEntry {
    PocketRegistryEntry {
        pocket_uid: uid,
        human_alias: "final_bake_ingress",
        artifact_path: "persistent_library/artifacts/final_bake_ingress.json",
        content_digest: digest,
        token_hash,
        abi_version: "PocketABI-v1",
        capability_signature: "binary_ingress",
        lifecycle,
    }
}

fn artifact(
    uid: &'static str,
    digest: &'static str,
    token_hash: &'static str,
) -> StoredPocketArtifact {
    StoredPocketArtifact {
        pocket_uid: uid,
        content_digest: digest,
        token_hash,
        abi_version: "PocketABI-v1",
        quality_delta: 0.0,
        generation: 1,
    }
}

pub(crate) fn base_store() -> PocketLibraryStore {
    let mut store = PocketLibraryStore::new();
    store.insert_pocket(
        entry(
            "pkt_final_bake_ingress",
            "digest_final_bake_ingress",
            "tok_final_bake_ingress",
            PocketLifecycle::Stable,
        ),
        token(
            "pkt_final_bake_ingress",
            "digest_final_bake_ingress",
            "tok_final_bake_ingress",
        ),
        artifact(
            "pkt_final_bake_ingress",
            "digest_final_bake_ingress",
            "tok_final_bake_ingress",
        ),
    );
    store
}

pub(crate) fn lifecycle() -> NextMutationEvidence {
    NextMutationEvidence {
        active_slot_count: 1,
        sandbox_only: true,
        proposal_only: true,
        light_probe_passed: true,
        initial_quality: 0.89,
        refined_quality: 0.999,
        golden_disc_quality: 1.0,
        unique_value_score: 0.128,
        mutation_stats: MutationStats {
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

pub(crate) fn promotion(global_scope_allowed: bool) -> PromotionEvidence {
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
        global_scope_allowed,
    }
}

fn candidate(slot: usize) -> StorePromotionCandidate {
    match slot {
        0 => StorePromotionCandidate {
            pocket_uid: "gold_final_frame",
            human_alias: "final_frame_guard",
            content_digest: "digest_gold_final_frame",
            token_hash: "tok_gold_final_frame",
            capability_signature: "binary_final_frame_guard",
            quality_delta: 0.031,
        },
        1 => StorePromotionCandidate {
            pocket_uid: "gold_final_request",
            human_alias: "final_requested_feature_guard",
            content_digest: "digest_gold_final_request",
            token_hash: "tok_gold_final_request",
            capability_signature: "binary_final_request_guard",
            quality_delta: 0.033,
        },
        2 => StorePromotionCandidate {
            pocket_uid: "gold_final_trace",
            human_alias: "final_trace_commit_guard",
            content_digest: "digest_gold_final_trace",
            token_hash: "tok_gold_final_trace",
            capability_signature: "binary_final_trace_guard",
            quality_delta: 0.034,
        },
        _ => StorePromotionCandidate {
            pocket_uid: "gold_final_reload",
            human_alias: "final_reload_guard",
            content_digest: "digest_gold_final_reload",
            token_hash: "tok_gold_final_reload",
            capability_signature: "binary_final_reload_guard",
            quality_delta: 0.035,
        },
    }
}

pub(crate) fn queue_lessons(round: u64) -> [CurriculumQueueLesson; 4] {
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
        promotion_evidence: promotion(false),
    })
}

pub(crate) fn run_range(
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

fn run_body_gate() -> GateMetrics {
    let mut metrics = GateMetrics::default();

    let mut runtime = LockedBodyRuntime::default_body();
    let mut valid = safe_filler(8);
    valid.extend(encode_frame(7, 1, 1, 3));
    let step = runtime.process_binary_evidence(42, 7, &valid);
    metrics.note(
        step.action == Action::CommitEvidence
            && step.committed.is_some()
            && step.rendered.compact == "COMMIT_EVIDENCE",
    );

    let mut wrong = safe_filler(8);
    wrong.extend(encode_frame(8, 1, 1, 3));
    let before_flow = runtime.flow.active_count();
    let step = runtime.process_binary_evidence(42, 7, &wrong);
    metrics
        .note(step.action != Action::CommitEvidence && runtime.flow.active_count() == before_flow);

    let mut corrupt = safe_filler(8);
    corrupt.extend(corrupt_crc(&encode_frame(4, 1, 1, 2)));
    let decoded = reassemble_requested_frame(&corrupt, 4);
    metrics.note(decoded.action == Action::Defer);

    let mut bad_length = safe_filler(8);
    bad_length.extend(corrupt_length(&encode_frame(4, 1, 1, 2)));
    let decoded = reassemble_requested_frame(&bad_length, 4);
    metrics.note(decoded.action == Action::Defer);

    let rendered = render_output(
        EgressMode::MultiResolution,
        runtime
            .process_binary_evidence(42, 9, &{
                let mut stream = safe_filler(8);
                stream.extend(encode_frame(9, 0, 1, 4));
                stream
            })
            .committed,
    );
    metrics.note(
        rendered.compact == "COMMIT_EVIDENCE"
            && rendered
                .long
                .as_deref()
                .is_some_and(|text| text.contains("Agency committed")),
    );

    metrics
}

fn run_text_gate() -> GateMetrics {
    let mut metrics = GateMetrics::default();
    let cases = [
        (
            TextProfile {
                byte_len: 200,
                evidence_available: true,
                boundary_risk: 0,
                integrity_risk: 0,
                requires_clean_long: false,
            },
            TextMode::FastDefault,
        ),
        (
            TextProfile {
                byte_len: 900,
                evidence_available: true,
                boundary_risk: 2,
                integrity_risk: 2,
                requires_clean_long: false,
            },
            TextMode::LongCapped,
        ),
        (
            TextProfile {
                byte_len: 1400,
                evidence_available: true,
                boundary_risk: 3,
                integrity_risk: 3,
                requires_clean_long: true,
            },
            TextMode::CleanLong,
        ),
        (
            TextProfile {
                byte_len: 2000,
                evidence_available: true,
                boundary_risk: 4,
                integrity_risk: 4,
                requires_clean_long: true,
            },
            TextMode::AskOrMultiCycle,
        ),
    ];
    for (profile, expected) in cases {
        metrics.note(select_text_mode(profile) == expected);
    }
    metrics
}

fn run_registry_gate() -> GateMetrics {
    let mut metrics = GateMetrics::default();
    let good_token = token("pkt_a", "digest_a", "tok_a");
    let good_entry = entry("pkt_a", "digest_a", "tok_a", PocketLifecycle::Stable);
    metrics.note(resolve_pocket_call(good_token, &[good_entry]).allowed);

    let bad_digest = entry("pkt_a", "digest_b", "tok_a", PocketLifecycle::Stable);
    metrics.note(!resolve_pocket_call(good_token, &[bad_digest]).allowed);

    let blocked = entry("pkt_a", "digest_a", "tok_a", PocketLifecycle::Quarantine);
    metrics.note(!resolve_pocket_call(good_token, &[blocked]).allowed);

    let mut lower = token("pkt_b", "digest_b", "tok_b");
    lower.utility_score = 0.70;
    let active = active_pocket_set(
        &[good_token, lower],
        &[
            good_entry,
            entry("pkt_b", "digest_b", "tok_b", PocketLifecycle::Stable),
        ],
        1,
    );
    metrics.note(active.len() == 1 && active[0].pocket_uid == "pkt_a");
    metrics
}

fn run_manager_and_mutation_gate() -> GateMetrics {
    let mut metrics = GateMetrics::default();

    let promoted = evaluate_promotion(promotion(false));
    metrics.note(
        promoted.reason == PromotionBlockReason::None && promoted.level == PromotionLevel::Core,
    );

    let true_golden = evaluate_promotion(promotion(true));
    metrics.note(
        true_golden.reason == PromotionBlockReason::None && true_golden.level.is_core_or_above(),
    );

    let mut unsafe_promotion = promotion(false);
    unsafe_promotion.safety.no_unsafe_high_utility = false;
    let unsafe_verdict = evaluate_promotion(unsafe_promotion);
    metrics.note(
        unsafe_verdict.reason == PromotionBlockReason::UnsafeHighUtility
            && unsafe_verdict.level == PromotionLevel::Quarantine,
    );

    let mutation = evaluate_next_mutation_lifecycle(lifecycle());
    metrics.note(
        mutation.stage == MutationLifecycleStage::GoldenDisc && mutation.golden_disc.is_some(),
    );

    let mut direct_write = lifecycle();
    direct_write.direct_flow_write_violation_rate = 1.0;
    let blocked = evaluate_next_mutation_lifecycle(direct_write);
    metrics.note(blocked.reason == MutationBlockReason::DirectFlowWrite);

    let mut rollback = lifecycle();
    rollback.mutation_stats.rollback_count = 1;
    let blocked = evaluate_next_mutation_lifecycle(rollback);
    metrics.note(blocked.reason == MutationBlockReason::RollbackMismatch);

    metrics
}

fn run_library_gate() -> GateMetrics {
    let mut metrics = GateMetrics::default();
    let mut store = PocketLibraryStore::new();
    store.insert_pocket(
        entry(
            "pkt_store",
            "digest_store",
            "tok_store",
            PocketLifecycle::Stable,
        ),
        token("pkt_store", "digest_store", "tok_store"),
        artifact("pkt_store", "digest_store", "tok_store"),
    );
    metrics.note(
        store
            .guarded_load(token("pkt_store", "digest_store", "tok_store"))
            .allowed,
    );

    store.insert_pocket(
        entry(
            "pkt_store",
            "digest_store",
            "tok_store",
            PocketLifecycle::Core,
        ),
        token("pkt_store", "digest_store", "tok_store"),
        artifact("pkt_store", "digest_store", "tok_store"),
    );
    let snapshot = store.snapshot();
    metrics.note(
        snapshot.registry_entry_count == 1
            && snapshot.token_count == 1
            && snapshot.artifact_count == 1,
    );

    let decision = store.promote_candidate(candidate(0), lifecycle(), promotion(false));
    metrics.note(decision.allowed);

    let mut unsafe_lifecycle = lifecycle();
    unsafe_lifecycle.direct_flow_write_violation_rate = 1.0;
    let decision = store.promote_candidate(candidate(1), unsafe_lifecycle, promotion(false));
    metrics.note(!decision.allowed && decision.reason == StoreGuardReason::UnsafePromotion);

    metrics
}

fn run_curriculum_resume_gate(
    rounds: u64,
    progress: &PathBuf,
) -> (bool, CurriculumCheckpoint, CurriculumCheckpoint, f64) {
    let split = rounds / 2;
    let start = Instant::now();

    let mut reference_runner = RustCurriculumRunner::default_body(base_store());
    let mut reference = CurriculumCheckpoint::new(73);
    run_range(
        &mut reference_runner,
        &mut reference,
        0,
        rounds,
        progress,
        "final_bake_reference",
    );
    append_jsonl(
        progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"reference_complete\",\"queues\":{},\"lessons\":{},\"checksum\":{}}}",
            now_millis(),
            reference.completed_queues,
            reference.completed_lessons,
            reference.checksum
        ),
    );

    let mut first_runner = RustCurriculumRunner::default_body(base_store());
    let mut mid = CurriculumCheckpoint::new(73);
    run_range(
        &mut first_runner,
        &mut mid,
        0,
        split,
        progress,
        "final_bake_checkpoint_first_half",
    );
    append_jsonl(
        progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"checkpoint_mid\",\"queues\":{},\"lessons\":{},\"checksum\":{}}}",
            now_millis(),
            mid.completed_queues,
            mid.completed_lessons,
            mid.checksum
        ),
    );

    let compatible = mid.resume_compatible(split);
    let mut resumed_runner = first_runner.clone();
    let mut resumed = mid;
    run_range(
        &mut resumed_runner,
        &mut resumed,
        split,
        rounds,
        progress,
        "final_bake_resume_second_half",
    );
    let audit = audit_resume(reference, resumed);
    (
        compatible && audit.passed(),
        reference,
        resumed,
        start.elapsed().as_secs_f64(),
    )
}

pub fn run_final_bake_preflight(rounds: u64, out: PathBuf) -> FinalBakeSummary {
    fs::create_dir_all(&out).expect("create output directory");
    let progress = out.join("progress.jsonl");
    let _ = fs::remove_file(&progress);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"start\",\"rounds\":{},\"policy\":\"Rust final bake preflight\"}}",
            now_millis(),
            rounds
        ),
    );

    let body = run_body_gate();
    let text = run_text_gate();
    let registry = run_registry_gate();
    let manager_mutation = run_manager_and_mutation_gate();
    let library = run_library_gate();
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"component_gates_complete\",\"body_passed\":{},\"text_passed\":{},\"registry_passed\":{},\"manager_mutation_passed\":{},\"library_passed\":{}}}",
            now_millis(),
            body.all_passed(),
            text.all_passed(),
            registry.all_passed(),
            manager_mutation.all_passed(),
            library.all_passed()
        ),
    );

    let (resume_passed, reference, resumed, seconds) =
        run_curriculum_resume_gate(rounds, &progress);
    let passed = body.all_passed()
        && text.all_passed()
        && registry.all_passed()
        && manager_mutation.all_passed()
        && library.all_passed()
        && resume_passed
        && resumed.bad_commit_count == 0
        && resumed.unsafe_promotion_count == 0;

    fs::write(
        out.join("final_bake_results.json"),
        format!(
            concat!(
                "{{\n",
                "  \"passed\": {},\n",
                "  \"rounds\": {},\n",
                "  \"body_cases\": {},\n",
                "  \"body_passed\": {},\n",
                "  \"text_cases\": {},\n",
                "  \"text_passed\": {},\n",
                "  \"registry_cases\": {},\n",
                "  \"registry_passed\": {},\n",
                "  \"manager_mutation_cases\": {},\n",
                "  \"manager_mutation_passed\": {},\n",
                "  \"library_cases\": {},\n",
                "  \"library_passed\": {},\n",
                "  \"resume_passed\": {},\n",
                "  \"reference_queues\": {},\n",
                "  \"resumed_queues\": {},\n",
                "  \"reference_lessons\": {},\n",
                "  \"resumed_lessons\": {},\n",
                "  \"final_checksum_match\": {},\n",
                "  \"bad_commit_rate\": {:.6},\n",
                "  \"unsafe_promotion_rate\": {:.6},\n",
                "  \"seconds\": {:.9},\n",
                "  \"queues_per_sec\": {:.3},\n",
                "  \"lessons_per_sec\": {:.3}\n",
                "}}\n"
            ),
            passed,
            rounds,
            body.cases,
            body.passed,
            text.cases,
            text.passed,
            registry.cases,
            registry.passed,
            manager_mutation.cases,
            manager_mutation.passed,
            library.cases,
            library.passed,
            resume_passed,
            reference.completed_queues,
            resumed.completed_queues,
            reference.completed_lessons,
            resumed.completed_lessons,
            reference.checksum == resumed.checksum,
            resumed.bad_commit_count as f64 / resumed.completed_queues.max(1) as f64,
            resumed.unsafe_promotion_count as f64 / resumed.completed_lessons.max(1) as f64,
            seconds,
            (reference.completed_queues + resumed.completed_queues) as f64 / seconds.max(0.000_001),
            (reference.completed_lessons + resumed.completed_lessons) as f64
                / seconds.max(0.000_001),
        ),
    )
    .expect("write final bake results");

    fs::write(
        out.join("report.md"),
        format!(
            "# E73 Rust Final Bake Preflight\n\n\
             ```text\n\
             passed = {}\n\
             component_gates = {}/{}/{}/{}/{}\n\
             resume_passed = {}\n\
             reference_queues = {}\n\
             resumed_queues = {}\n\
             final_checksum_match = {}\n\
             bad_commit_rate = {:.6}\n\
             unsafe_promotion_rate = {:.6}\n\
             ```\n",
            passed,
            body.all_passed(),
            text.all_passed(),
            registry.all_passed(),
            manager_mutation.all_passed(),
            library.all_passed(),
            resume_passed,
            reference.completed_queues,
            resumed.completed_queues,
            reference.checksum == resumed.checksum,
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
            reference.checksum == resumed.checksum
        ),
    );

    FinalBakeSummary {
        passed,
        rounds,
        resume_passed,
        final_checksum_match: reference.checksum == resumed.checksum,
        bad_commit_rate: resumed.bad_commit_count as f64 / resumed.completed_queues.max(1) as f64,
        unsafe_promotion_rate: resumed.unsafe_promotion_count as f64
            / resumed.completed_lessons.max(1) as f64,
        seconds,
        out,
    }
}
