use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use vraxion_runtime::{
    encode_frame, safe_filler, ChallengerEvidence, CurriculumBlockReason, CurriculumLesson,
    NextMutationEvidence, PocketLibraryStore, PocketLifecycle, PocketRegistryEntry, PocketToken,
    PromotionEvidence, RustCurriculumRunner, SafetyGate, ScoreVector, StoreGuardReason,
    StoreMutationStats, StorePromotionCandidate, StoredPocketArtifact,
};

#[derive(Default)]
struct Metrics {
    rows: u64,
    curriculum_success: u64,
    active_set_success: u64,
    commit_success: u64,
    flow_ground_sync: u64,
    trace_render_success: u64,
    proposal_boundary_success: u64,
    promotion_success: u64,
    reload_success: u64,
    quality_delta_positive: u64,
    adversarial_cases: u64,
    adversarial_blocks: u64,
    no_active_block: u64,
    bad_frame_block: u64,
    stale_token_block: u64,
    unsafe_candidate_block: u64,
    stale_write_block: u64,
    bad_commit: u64,
    unsafe_promotion: u64,
}

impl Metrics {
    fn passed(&self) -> bool {
        self.rows == self.curriculum_success
            && self.rows == self.active_set_success
            && self.rows == self.commit_success
            && self.rows == self.flow_ground_sync
            && self.rows == self.trace_render_success
            && self.rows == self.proposal_boundary_success
            && self.rows == self.promotion_success
            && self.rows == self.reload_success
            && self.rows == self.quality_delta_positive
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

fn token(
    pocket_uid: &'static str,
    token_hash: &'static str,
    content_digest: &'static str,
    capability_signature: &'static str,
    utility_score: f32,
) -> PocketToken {
    PocketToken {
        pocket_uid,
        token_version: 4,
        min_token_version: 3,
        token_hash,
        content_digest,
        abi_version: "PocketABI-v1",
        capability_signature,
        utility_score,
        safety_score: 0.99,
        reuse_score: 0.84,
        cost_score: 0.06,
    }
}

fn entry(
    pocket_uid: &'static str,
    token_hash: &'static str,
    content_digest: &'static str,
    capability_signature: &'static str,
    lifecycle: PocketLifecycle,
) -> PocketRegistryEntry {
    PocketRegistryEntry {
        pocket_uid,
        human_alias: "binary_ingress_curriculum",
        artifact_path: "persistent_library/artifacts/binary_ingress_curriculum.json",
        content_digest,
        token_hash,
        abi_version: "PocketABI-v1",
        capability_signature,
        lifecycle,
    }
}

fn artifact(
    pocket_uid: &'static str,
    token_hash: &'static str,
    content_digest: &'static str,
) -> StoredPocketArtifact {
    StoredPocketArtifact {
        pocket_uid,
        content_digest,
        token_hash,
        abi_version: "PocketABI-v1",
        quality_delta: 0.0,
        generation: 1,
    }
}

fn base_store(lifecycle: PocketLifecycle) -> PocketLibraryStore {
    let mut store = PocketLibraryStore::new();
    store.insert_pocket(
        entry(
            "pkt_ingress_curriculum",
            "tok_ingress_curriculum",
            "digest_ingress_curriculum",
            "binary_ingress",
            lifecycle,
        ),
        token(
            "pkt_ingress_curriculum",
            "tok_ingress_curriculum",
            "digest_ingress_curriculum",
            "binary_ingress",
            0.94,
        ),
        artifact(
            "pkt_ingress_curriculum",
            "tok_ingress_curriculum",
            "digest_ingress_curriculum",
        ),
    );
    store
}

fn lesson(round: u64) -> CurriculumLesson {
    CurriculumLesson {
        requested_feature: ((round % 23) + 1) as u8,
        value: (round % 2) as u8,
        source_pocket_id: 42,
        nonce: ((round * 7 + 3) % 31) as u8,
    }
}

fn lifecycle() -> NextMutationEvidence {
    NextMutationEvidence {
        active_slot_count: 1,
        sandbox_only: true,
        proposal_only: true,
        light_probe_passed: true,
        initial_quality: 0.88,
        refined_quality: 0.999,
        golden_disc_quality: 1.0,
        unique_value_score: 0.121,
        mutation_stats: StoreMutationStats {
            attempts: 192,
            accepted: 13,
            rejected: 179,
            rollback_count: 179,
            attempts_to_s_rank: Some(31),
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
            eligible_activation: 0.92,
            generality: 0.95,
            uniqueness: 0.94,
            transfer: 0.95,
            robustness: 0.96,
            cost: 0.08,
            stability: 0.98,
            scope_clarity: 0.95,
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

fn candidate() -> StorePromotionCandidate {
    StorePromotionCandidate {
        pocket_uid: "gold_curriculum_binary_ingress",
        human_alias: "curriculum_binary_ingress_guard",
        content_digest: "digest_curriculum_binary_ingress",
        token_hash: "tok_curriculum_binary_ingress",
        capability_signature: "binary_ingress_curriculum_guard",
        quality_delta: 0.044,
    }
}

fn note_block(metrics: &mut Metrics, blocked: bool) {
    metrics.adversarial_cases += 1;
    metrics.adversarial_blocks += blocked as u64;
}

fn run_primary_row(metrics: &mut Metrics, round: u64) {
    metrics.rows += 1;
    let mut runner = RustCurriculumRunner::default_body(base_store(PocketLifecycle::Stable));
    let verdict = runner.run_binary_lesson(lesson(round), candidate(), lifecycle(), promotion());
    metrics.active_set_success += (verdict.active_pocket_count > 0) as u64;
    metrics.commit_success += (verdict.action == vraxion_runtime::Action::CommitEvidence) as u64;
    metrics.flow_ground_sync += (verdict.flow_value == verdict.ground_value
        && verdict.flow_value == Some(lesson(round).value)) as u64;
    metrics.trace_render_success +=
        (verdict.reason != CurriculumBlockReason::TraceRenderMissing) as u64;
    metrics.proposal_boundary_success += (verdict.proposal_slots_used == 1) as u64;
    metrics.promotion_success += verdict.promoted as u64;
    let snapshot = runner.store.snapshot();
    metrics.reload_success += runner.store.reload_matches(snapshot) as u64;
    metrics.quality_delta_positive += (runner.store.quality_delta() > 0.0) as u64;
    metrics.bad_commit += (!verdict.passed) as u64;
    metrics.curriculum_success += verdict.passed as u64;
}

fn run_adversarial_row(metrics: &mut Metrics, round: u64) {
    let mut no_active = RustCurriculumRunner::default_body(base_store(PocketLifecycle::Quarantine));
    let no_active_verdict =
        no_active.run_binary_lesson(lesson(round), candidate(), lifecycle(), promotion());
    let no_active_ok = no_active_verdict.reason == CurriculumBlockReason::NoActivePocket
        && no_active.body.flow.active_count() == 0;
    note_block(metrics, no_active_ok);
    metrics.no_active_block += no_active_ok as u64;

    let mut bad_frame = RustCurriculumRunner::default_body(base_store(PocketLifecycle::Stable));
    let row = lesson(round);
    let mut stream = safe_filler(8);
    stream.extend(encode_frame(
        row.requested_feature.wrapping_add(1),
        row.value,
        1,
        row.nonce,
    ));
    let bad_frame_verdict = bad_frame.run_binary_lesson_with_stream(
        row,
        &stream,
        candidate(),
        lifecycle(),
        promotion(),
    );
    let bad_frame_ok = bad_frame_verdict.reason == CurriculumBlockReason::RuntimeDidNotCommit
        && bad_frame.body.flow.active_count() == 0;
    note_block(metrics, bad_frame_ok);
    metrics.bad_frame_block += bad_frame_ok as u64;

    let mut stale_store = base_store(PocketLifecycle::Stable);
    stale_store.tokens[0].token_version = 2;
    let stale_decision = stale_store.guarded_load(stale_store.tokens[0]);
    let stale_ok = stale_decision.reason
        == StoreGuardReason::LoadBlocked(vraxion_runtime::LoadBlockReason::StaleToken);
    note_block(metrics, stale_ok);
    metrics.stale_token_block += stale_ok as u64;

    let mut unsafe_lifecycle = lifecycle();
    unsafe_lifecycle.direct_flow_write_violation_rate = 1.0;
    let mut unsafe_candidate =
        RustCurriculumRunner::default_body(base_store(PocketLifecycle::Stable));
    let unsafe_verdict = unsafe_candidate.run_binary_lesson(
        lesson(round),
        candidate(),
        unsafe_lifecycle,
        promotion(),
    );
    let unsafe_ok = unsafe_verdict.reason
        == CurriculumBlockReason::PromotionBlocked(StoreGuardReason::UnsafePromotion);
    note_block(metrics, unsafe_ok);
    metrics.unsafe_candidate_block += unsafe_ok as u64;
    metrics.unsafe_promotion += unsafe_verdict.promoted as u64;

    let mut stale_write_store = base_store(PocketLifecycle::Stable);
    let generation_before = stale_write_store.generation;
    assert!(stale_write_store.rename_alias("pkt_ingress_curriculum", "renamed_curriculum"));
    let stale_write_ok = stale_write_store
        .concurrent_write_guard(generation_before)
        .reason
        == StoreGuardReason::ConcurrentStaleWrite;
    note_block(metrics, stale_write_ok);
    metrics.stale_write_block += stale_write_ok as u64;
}

fn write_artifacts(out: &PathBuf, rounds: u64, metrics: &Metrics, seconds: f64) {
    fs::create_dir_all(out).expect("create output directory");
    fs::write(
        out.join("curriculum_runner_config.json"),
        concat!(
            "{\n",
            "  \"runner\": \"rust_curriculum_runner_preflight\",\n",
            "  \"body\": \"near_28f_32g_20x80_default\",\n",
            "  \"active_set_limit\": 4,\n",
            "  \"path\": [\"active_set\", \"guarded_load\", \"locked_body_commit\", \"trace_egress\", \"next_mutation\", \"safe_promotion\", \"reload_snapshot\"],\n",
            "  \"heartbeat_seconds\": 20\n",
            "}\n"
        ),
    )
    .expect("write config");
    fs::write(
        out.join("curriculum_trace_sample.json"),
        concat!(
            "{\n",
            "  \"lesson\": {\"requested_feature\": 7, \"value\": 1, \"source_pocket_id\": 42},\n",
            "  \"expected_path\": \"PocketToken active set -> guarded load -> Proposal Field -> Agency commit -> Flow/Ground sync -> trace egress -> safe promotion\",\n",
            "  \"direct_flow_write_allowed\": false\n",
            "}\n"
        ),
    )
    .expect("write trace sample");
    let results = format!(
        concat!(
            "{{\n",
            "  \"passed\": {},\n",
            "  \"rounds\": {},\n",
            "  \"curriculum_success_rate\": {:.6},\n",
            "  \"active_set_success_rate\": {:.6},\n",
            "  \"commit_success_rate\": {:.6},\n",
            "  \"flow_ground_sync_rate\": {:.6},\n",
            "  \"trace_render_success_rate\": {:.6},\n",
            "  \"proposal_boundary_success_rate\": {:.6},\n",
            "  \"promotion_success_rate\": {:.6},\n",
            "  \"reload_success_rate\": {:.6},\n",
            "  \"quality_delta_positive_rate\": {:.6},\n",
            "  \"adversarial_block_rate\": {:.6},\n",
            "  \"no_active_block_rate\": {:.6},\n",
            "  \"bad_frame_block_rate\": {:.6},\n",
            "  \"stale_token_block_rate\": {:.6},\n",
            "  \"unsafe_candidate_block_rate\": {:.6},\n",
            "  \"stale_write_block_rate\": {:.6},\n",
            "  \"bad_commit_rate\": {:.6},\n",
            "  \"unsafe_promotion_rate\": {:.6},\n",
            "  \"seconds\": {:.9},\n",
            "  \"rows_per_sec\": {:.3}\n",
            "}}\n"
        ),
        metrics.passed(),
        rounds,
        ratio(metrics.curriculum_success, metrics.rows),
        ratio(metrics.active_set_success, metrics.rows),
        ratio(metrics.commit_success, metrics.rows),
        ratio(metrics.flow_ground_sync, metrics.rows),
        ratio(metrics.trace_render_success, metrics.rows),
        ratio(metrics.proposal_boundary_success, metrics.rows),
        ratio(metrics.promotion_success, metrics.rows),
        ratio(metrics.reload_success, metrics.rows),
        ratio(metrics.quality_delta_positive, metrics.rows),
        ratio(metrics.adversarial_blocks, metrics.adversarial_cases),
        ratio(metrics.no_active_block, metrics.rows),
        ratio(metrics.bad_frame_block, metrics.rows),
        ratio(metrics.stale_token_block, metrics.rows),
        ratio(metrics.unsafe_candidate_block, metrics.rows),
        ratio(metrics.stale_write_block, metrics.rows),
        ratio(metrics.bad_commit, metrics.rows),
        ratio(metrics.unsafe_promotion, metrics.rows),
        seconds,
        metrics.rows as f64 / seconds.max(0.000_001),
    );
    fs::write(out.join("preflight_results.json"), results).expect("write results");
    fs::write(
        out.join("report.md"),
        format!(
            "# E70 Rust Curriculum Runner Preflight\n\n\
             ```text\n\
             passed = {}\n\
             curriculum_success_rate = {:.6}\n\
             active_set_success_rate = {:.6}\n\
             commit_success_rate = {:.6}\n\
             flow_ground_sync_rate = {:.6}\n\
             trace_render_success_rate = {:.6}\n\
             promotion_success_rate = {:.6}\n\
             adversarial_block_rate = {:.6}\n\
             bad_commit_rate = {:.6}\n\
             unsafe_promotion_rate = {:.6}\n\
             ```\n",
            metrics.passed(),
            ratio(metrics.curriculum_success, metrics.rows),
            ratio(metrics.active_set_success, metrics.rows),
            ratio(metrics.commit_success, metrics.rows),
            ratio(metrics.flow_ground_sync, metrics.rows),
            ratio(metrics.trace_render_success, metrics.rows),
            ratio(metrics.promotion_success, metrics.rows),
            ratio(metrics.adversarial_blocks, metrics.adversarial_cases),
            ratio(metrics.bad_commit, metrics.rows),
            ratio(metrics.unsafe_promotion, metrics.rows),
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
        .unwrap_or_else(|| PathBuf::from("target/pilot_wave/e70_rust_curriculum_runner_preflight"));
    fs::create_dir_all(&out).expect("create output directory");
    let progress = out.join("progress.jsonl");
    let _ = fs::remove_file(&progress);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"start\",\"rounds\":{},\"policy\":\"Rust curriculum runner\"}}",
            now_millis(),
            rounds
        ),
    );

    let start = Instant::now();
    let mut last_write = Instant::now();
    let mut metrics = Metrics::default();
    for round in 0..rounds {
        run_primary_row(&mut metrics, round);
        run_adversarial_row(&mut metrics, round);
        if last_write.elapsed() >= Duration::from_secs(20) {
            append_jsonl(
                &progress,
                &format!(
                    "{{\"timestamp_ms\":{},\"event\":\"progress\",\"round\":{},\"curriculum_rows\":{},\"adversarial_blocks\":{},\"bad_commit\":{},\"unsafe_promotion\":{}}}",
                    now_millis(),
                    round + 1,
                    metrics.rows,
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
            "{{\"timestamp_ms\":{},\"event\":\"complete\",\"passed\":{},\"curriculum_rows\":{},\"adversarial_blocks\":{},\"bad_commit\":{},\"unsafe_promotion\":{}}}",
            now_millis(),
            metrics.passed(),
            metrics.rows,
            metrics.adversarial_blocks,
            metrics.bad_commit,
            metrics.unsafe_promotion
        ),
    );
    println!(
        "{{\"passed\":{},\"rounds\":{},\"curriculum_success_rate\":{:.6},\"adversarial_block_rate\":{:.6},\"bad_commit_rate\":{:.6},\"unsafe_promotion_rate\":{:.6},\"seconds\":{:.9},\"rows_per_sec\":{:.3},\"out\":\"{}\"}}",
        metrics.passed(),
        rounds,
        ratio(metrics.curriculum_success, metrics.rows),
        ratio(metrics.adversarial_blocks, metrics.adversarial_cases),
        ratio(metrics.bad_commit, metrics.rows),
        ratio(metrics.unsafe_promotion, metrics.rows),
        seconds,
        metrics.rows as f64 / seconds.max(0.000_001),
        out.display()
    );
}
