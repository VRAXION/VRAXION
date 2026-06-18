use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use vraxion_runtime::{
    corrupt_crc, encode_frame, safe_filler, Action, AtomicCommitAction, AtomicCommitDecision,
    AtomicCommitPolicy, AtomicCommitProposal, AtomicOverlayCanaryConfig, AtomicProposalRole,
    AtomicRejectReason, LockedBodyRuntime,
};

const ARTIFACT_CONTRACT: &str = "E136R_ATOMIC_MULTIWRITE_PREAPPLY_DECISION_GAUNTLET";
const DECISION_CONFIRMED: &str = "e136r_atomic_multiwrite_default_route_candidate_confirmed";
const DECISION_REJECTED: &str = "e136r_atomic_multiwrite_default_route_candidate_rejected";
const NEXT: &str = "E136S_ATOMIC_MULTIWRITE_DEFAULT_ROUTE_SWITCH_CANARY_GUARD";

#[derive(Default)]
struct Metrics {
    default_route_cases: u64,
    default_route_success: u64,
    default_false_commit: u64,
    default_missed_commit: u64,
    canary_cases: u64,
    canary_success: u64,
    canary_overlay_active: u64,
    default_route_unchanged: u64,
    rollback_snapshot: u64,
    production_apply_allowed: u64,
    seeded_default_state_preserved: u64,
    commit_single: u64,
    commit_multi: u64,
    commit_chunk: u64,
    defer_count: u64,
    reject_count: u64,
    atomic_write_total: u64,
    partial_write: u64,
    order_failure: u64,
    runtime_direct_write: u64,
    held_variant_promoted: u64,
    direct_flow_write_reject: u64,
    stale_snapshot_reject: u64,
    checksum_tamper_reject: u64,
    ambiguous_same_region_reject: u64,
    capacity_reject: u64,
    rollback_commit: u64,
    oracle_plan_feature_use: u64,
    destructive_delete: u64,
}

impl Metrics {
    fn record_default(&mut self, pass: bool, false_commit: bool, missed_commit: bool) {
        self.default_route_cases += 1;
        self.default_route_success += pass as u64;
        self.default_false_commit += false_commit as u64;
        self.default_missed_commit += missed_commit as u64;
    }

    fn record_canary(&mut self, result: &CanaryResult) {
        self.canary_cases += 1;
        self.canary_success += result.pass as u64;
        self.canary_overlay_active += result.canary_overlay_active as u64;
        self.default_route_unchanged += result.default_route_unchanged as u64;
        self.rollback_snapshot += result.rollback_snapshot_taken as u64;
        self.production_apply_allowed += result.production_apply_allowed_now as u64;
        self.seeded_default_state_preserved += result.seeded_default_state_preserved as u64;
        self.commit_single += (result.action == AtomicCommitAction::CommitSingle) as u64;
        self.commit_multi += (result.action == AtomicCommitAction::CommitMulti) as u64;
        self.commit_chunk += (result.action == AtomicCommitAction::CommitChunk) as u64;
        self.defer_count += (result.action == AtomicCommitAction::Defer) as u64;
        self.reject_count += (result.action == AtomicCommitAction::Reject) as u64;
        self.atomic_write_total += result.write_count as u64;
        self.partial_write += result.partial_write as u64;
        self.order_failure += (!result.order_independent) as u64;
        self.runtime_direct_write += result.runtime_direct_write_count as u64;
        self.held_variant_promoted += result.held_variant_promoted as u64;
        self.oracle_plan_feature_use += result.oracle_plan_feature_use_count as u64;
        self.direct_flow_write_reject += result.direct_flow_write_reject_count as u64;
        self.stale_snapshot_reject += result.stale_snapshot_reject_count as u64;
        self.checksum_tamper_reject += result.checksum_tamper_reject_count as u64;
        self.ambiguous_same_region_reject += result.ambiguous_same_region_reject_count as u64;
        self.capacity_reject += result.capacity_reject_count as u64;
        self.rollback_commit += result.rollback_commit as u64;
    }

    fn pass_gate(&self, default_rounds: u64, canary_rounds: u64) -> bool {
        self.default_route_cases == self.default_route_success
            && self.default_route_cases >= default_rounds * 3
            && self.default_false_commit == 0
            && self.default_missed_commit == 0
            && self.canary_cases == self.canary_success
            && self.canary_cases >= canary_rounds
            && self.canary_overlay_active == self.canary_cases
            && self.default_route_unchanged == self.canary_cases
            && self.rollback_snapshot == self.canary_cases
            && self.seeded_default_state_preserved == self.canary_cases
            && self.production_apply_allowed == 0
            && self.commit_single >= canary_rounds / 5
            && self.commit_multi >= canary_rounds / 10
            && self.commit_chunk >= canary_rounds / 10
            && self.reject_count + self.defer_count >= canary_rounds / 2
            && self.rollback_commit >= canary_rounds / 10
            && self.direct_flow_write_reject >= canary_rounds / 10
            && self.stale_snapshot_reject >= canary_rounds / 10
            && self.checksum_tamper_reject >= canary_rounds / 10
            && self.ambiguous_same_region_reject >= canary_rounds / 10
            && self.capacity_reject >= canary_rounds / 10
            && self.partial_write == 0
            && self.order_failure == 0
            && self.runtime_direct_write == 0
            && self.held_variant_promoted == 0
            && self.oracle_plan_feature_use == 0
            && self.destructive_delete == 0
    }
}

struct CanaryResult {
    case_id: String,
    pass: bool,
    canary_overlay_active: bool,
    default_route_unchanged: bool,
    rollback_snapshot_taken: bool,
    production_apply_allowed_now: bool,
    seeded_default_state_preserved: bool,
    action: AtomicCommitAction,
    expected_action: AtomicCommitAction,
    write_count: usize,
    expected_write_count: usize,
    partial_write: bool,
    order_independent: bool,
    runtime_direct_write_count: usize,
    held_variant_promoted: bool,
    oracle_plan_feature_use_count: usize,
    direct_flow_write_reject_count: usize,
    stale_snapshot_reject_count: usize,
    checksum_tamper_reject_count: usize,
    ambiguous_same_region_reject_count: usize,
    capacity_reject_count: usize,
    rollback_commit: bool,
    reject_reason: Option<AtomicRejectReason>,
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_millis()
}

fn clean_stream(feature: u8, value: u8, nonce: u8) -> Vec<u8> {
    let mut stream = safe_filler(8);
    stream.extend(encode_frame(feature, value, 1, nonce));
    stream.extend(safe_filler(8));
    stream
}

fn wrong_feature_stream(feature: u8, value: u8, nonce: u8) -> Vec<u8> {
    let mut stream = safe_filler(8);
    stream.extend(encode_frame((feature + 7) & 31, value ^ 1, 1, nonce));
    stream.extend(safe_filler(8));
    stream
}

fn corrupt_stream(feature: u8, value: u8, nonce: u8) -> Vec<u8> {
    let mut stream = safe_filler(8);
    stream.extend(corrupt_crc(&encode_frame(feature, value, 1, nonce)));
    stream.extend(safe_filler(8));
    stream
}

fn proposal(feature: u8, value: u8, source: u32, group: u8) -> AtomicCommitProposal {
    AtomicCommitProposal {
        cycle_id: 1,
        source_pocket_id: source,
        role: AtomicProposalRole::Primary,
        relation_group: group,
        trace_valid: true,
        ground_compatible: true,
        checksum_valid: true,
        direct_flow_write: false,
        unsupported_answer: false,
        hard_negative: false,
        primary_regression_signal: false,
        target_feature: Some(feature),
        value: Some(value),
        confidence: 200,
    }
}

fn action_name(action: AtomicCommitAction) -> &'static str {
    match action {
        AtomicCommitAction::CommitSingle => "commit_single",
        AtomicCommitAction::CommitMulti => "commit_multi",
        AtomicCommitAction::CommitChunk => "commit_chunk",
        AtomicCommitAction::Reject => "reject",
        AtomicCommitAction::Defer => "defer",
    }
}

fn reason_name(reason: AtomicRejectReason) -> &'static str {
    match reason {
        AtomicRejectReason::StaleSnapshot => "stale_snapshot",
        AtomicRejectReason::TraceOrGroundInvalid => "trace_or_ground_invalid",
        AtomicRejectReason::ChecksumInvalid => "checksum_invalid",
        AtomicRejectReason::DirectFlowWrite => "direct_flow_write",
        AtomicRejectReason::UnsafeAnswer => "unsafe_answer",
        AtomicRejectReason::PrimaryRegression => "primary_regression",
        AtomicRejectReason::MissingTarget => "missing_target",
        AtomicRejectReason::AmbiguousSameRegion => "ambiguous_same_region",
        AtomicRejectReason::AtomicBatchConflict => "atomic_batch_conflict",
        AtomicRejectReason::ProposalCapacityExceeded => "proposal_capacity_exceeded",
        AtomicRejectReason::MultiWritePolicyBlocked => "multi_write_policy_blocked",
        AtomicRejectReason::NoValidProposal => "no_valid_proposal",
    }
}

fn count_reject(decision: &AtomicCommitDecision, reason: AtomicRejectReason) -> usize {
    decision
        .rejected
        .iter()
        .filter(|row| row.reason == reason)
        .count()
        + usize::from(decision.reject_reason == Some(reason))
}

fn run_default_round(idx: u64, metrics: &mut Metrics) {
    let feature = (idx % 31) as u8;
    let value = (idx & 1) as u8;

    let mut clean = LockedBodyRuntime::default_body();
    let clean_step = clean.process_binary_evidence(900, feature, &clean_stream(feature, value, 1));
    let clean_pass = clean_step.action == Action::CommitEvidence
        && clean_step.committed.map(|record| record.feature_id) == Some(feature)
        && clean_step.committed.map(|record| record.value) == Some(value);
    metrics.record_default(clean_pass, false, !clean_pass);

    let mut wrong = LockedBodyRuntime::default_body();
    let wrong_step =
        wrong.process_binary_evidence(901, feature, &wrong_feature_stream(feature, value, 2));
    let wrong_pass = wrong_step.action != Action::CommitEvidence
        && wrong_step.committed.is_none()
        && wrong.flow.active_count() == 0
        && wrong.ground.active_count() == 0;
    metrics.record_default(wrong_pass, !wrong_pass, false);

    let mut corrupt = LockedBodyRuntime::default_body();
    let corrupt_step =
        corrupt.process_binary_evidence(902, feature, &corrupt_stream(feature, value, 3));
    let corrupt_pass = corrupt_step.action != Action::CommitEvidence
        && corrupt_step.committed.is_none()
        && corrupt.flow.active_count() == 0
        && corrupt.ground.active_count() == 0;
    metrics.record_default(corrupt_pass, !corrupt_pass, false);
}

fn build_canary_case(
    idx: u64,
) -> (
    String,
    Vec<AtomicCommitProposal>,
    AtomicCommitAction,
    usize,
    bool,
) {
    let base_feature = ((idx * 7) % 24) as u8;
    let base_source = 10_000 + idx as u32 * 16;
    match idx % 10 {
        0 => (
            format!("stress_{idx:04}_disjoint_atomic_multiwrite"),
            vec![
                proposal(base_feature, 1, base_source, 1),
                proposal(base_feature + 1, 0, base_source + 1, 2),
                proposal(base_feature + 2, 1, base_source + 2, 3),
            ],
            AtomicCommitAction::CommitMulti,
            3,
            false,
        ),
        1 => (
            format!("stress_{idx:04}_homogeneous_chunk_commit"),
            vec![
                proposal(base_feature, 1, base_source, 7),
                proposal(base_feature + 1, 0, base_source + 1, 7),
                proposal(base_feature + 2, 1, base_source + 2, 7),
            ],
            AtomicCommitAction::CommitChunk,
            3,
            false,
        ),
        2 => (
            format!("stress_{idx:04}_single_primary_commit"),
            vec![proposal(base_feature, 1, base_source, 1)],
            AtomicCommitAction::CommitSingle,
            1,
            false,
        ),
        3 => {
            let mut stale = proposal(base_feature, 1, base_source, 1);
            stale.cycle_id = 0;
            (
                format!("stress_{idx:04}_stale_snapshot_reject"),
                vec![stale],
                AtomicCommitAction::Defer,
                0,
                false,
            )
        }
        4 => {
            let mut checksum = proposal(base_feature, 1, base_source, 1);
            checksum.checksum_valid = false;
            (
                format!("stress_{idx:04}_checksum_tamper_reject"),
                vec![checksum],
                AtomicCommitAction::Defer,
                0,
                false,
            )
        }
        5 => {
            let mut direct = proposal(base_feature, 1, base_source, 1);
            direct.direct_flow_write = true;
            (
                format!("stress_{idx:04}_direct_flow_write_reject"),
                vec![direct, proposal(base_feature + 1, 1, base_source + 1, 1)],
                AtomicCommitAction::CommitSingle,
                1,
                false,
            )
        }
        6 => (
            format!("stress_{idx:04}_ambiguous_same_region_reject"),
            vec![
                proposal(base_feature, 0, base_source, 1),
                proposal(base_feature, 1, base_source + 1, 1),
            ],
            AtomicCommitAction::Reject,
            0,
            false,
        ),
        7 => {
            let mut held = proposal(base_feature, 1, base_source, 1);
            held.role = AtomicProposalRole::HeldChallenger;
            (
                format!("stress_{idx:04}_held_challenger_hold"),
                vec![held],
                AtomicCommitAction::Defer,
                0,
                false,
            )
        }
        8 => {
            let mut regressing = proposal(base_feature, 1, base_source, 1);
            regressing.primary_regression_signal = true;
            let mut rollback = proposal(base_feature, 0, base_source + 1, 1);
            rollback.role = AtomicProposalRole::Rollback;
            (
                format!("stress_{idx:04}_rollback_fallback_commit"),
                vec![regressing, rollback],
                AtomicCommitAction::CommitSingle,
                1,
                true,
            )
        }
        _ => {
            let mut proposals = Vec::new();
            for offset in 0..24 {
                proposals.push(proposal(
                    ((base_feature as usize + offset) % 31) as u8,
                    (offset & 1) as u8,
                    base_source + offset as u32,
                    1,
                ));
            }
            (
                format!("stress_{idx:04}_proposal_capacity_reject"),
                proposals,
                AtomicCommitAction::Reject,
                0,
                false,
            )
        }
    }
}

fn permute(mut proposals: Vec<AtomicCommitProposal>, idx: u64) -> Vec<AtomicCommitProposal> {
    if proposals.len() <= 1 {
        return proposals;
    }
    match idx % 3 {
        0 => proposals,
        1 => {
            proposals.reverse();
            proposals
        }
        _ => {
            let rotate = (idx as usize) % proposals.len();
            proposals.rotate_left(rotate);
            proposals
        }
    }
}

fn seeded_runtime(idx: u64) -> LockedBodyRuntime {
    let mut runtime = LockedBodyRuntime::default_body();
    let feature = ((idx * 5 + 3) % 31) as u8;
    let value = ((idx + 1) & 1) as u8;
    let step = runtime.process_binary_evidence(800, feature, &clean_stream(feature, value, 4));
    assert_eq!(step.action, Action::CommitEvidence);
    runtime
}

fn run_canary(idx: u64) -> CanaryResult {
    let (case_id, proposals, expected_action, expected_write_count, expect_rollback) =
        build_canary_case(idx);
    let proposals = permute(proposals, idx);
    let runtime = seeded_runtime(idx);
    let default_flow_before = runtime.flow.active_count();
    let default_ground_before = runtime.ground.active_count();
    let step = runtime.process_atomic_overlay_canary(
        AtomicOverlayCanaryConfig::e136q_canary(),
        AtomicCommitPolicy::e136p_preview(),
        &proposals,
    );
    let decision = step.overlay_step.decision;
    let held_ids: Vec<u32> = proposals
        .iter()
        .filter(|proposal| {
            matches!(
                proposal.role,
                AtomicProposalRole::HeldChallenger | AtomicProposalRole::HeldLineage
            )
        })
        .map(|proposal| proposal.source_pocket_id)
        .collect();
    let held_variant_promoted = decision
        .records
        .iter()
        .any(|record| held_ids.contains(&record.source_pocket_id));
    let partial_write = matches!(
        decision.action,
        AtomicCommitAction::Reject | AtomicCommitAction::Defer
    ) && (step.overlay_step.flow_active_cells != default_flow_before
        || step.overlay_step.ground_active_cells != default_ground_before);
    let rollback_commit = expect_rollback
        && decision.records.iter().any(|record| {
            proposals.iter().any(|proposal| {
                proposal.role == AtomicProposalRole::Rollback
                    && proposal.source_pocket_id == record.source_pocket_id
            })
        });
    let seeded_default_state_preserved = runtime.flow.active_count() == default_flow_before
        && runtime.ground.active_count() == default_ground_before;
    let pass = step.canary_overlay_active
        && step.default_route_unchanged
        && seeded_default_state_preserved
        && step.rollback_snapshot_taken
        && !step.production_apply_allowed_now
        && decision.action == expected_action
        && decision.records.len() == expected_write_count
        && step.overlay_step.committed.len() == expected_write_count
        && (!expect_rollback || rollback_commit)
        && !partial_write
        && decision.order_independent
        && decision.runtime_direct_write_count == 0
        && !held_variant_promoted
        && decision.oracle_plan_feature_use_count == 0;

    CanaryResult {
        case_id,
        pass,
        canary_overlay_active: step.canary_overlay_active,
        default_route_unchanged: step.default_route_unchanged,
        rollback_snapshot_taken: step.rollback_snapshot_taken,
        production_apply_allowed_now: step.production_apply_allowed_now,
        seeded_default_state_preserved,
        action: decision.action,
        expected_action,
        write_count: decision.records.len(),
        expected_write_count,
        partial_write,
        order_independent: decision.order_independent,
        runtime_direct_write_count: decision.runtime_direct_write_count,
        held_variant_promoted,
        oracle_plan_feature_use_count: decision.oracle_plan_feature_use_count,
        direct_flow_write_reject_count: count_reject(
            &decision,
            AtomicRejectReason::DirectFlowWrite,
        ),
        stale_snapshot_reject_count: count_reject(&decision, AtomicRejectReason::StaleSnapshot),
        checksum_tamper_reject_count: count_reject(&decision, AtomicRejectReason::ChecksumInvalid),
        ambiguous_same_region_reject_count: count_reject(
            &decision,
            AtomicRejectReason::AmbiguousSameRegion,
        ),
        capacity_reject_count: count_reject(
            &decision,
            AtomicRejectReason::ProposalCapacityExceeded,
        ),
        rollback_commit,
        reject_reason: decision.reject_reason,
    }
}

fn write_jsonl(path: PathBuf, rows: &[CanaryResult]) {
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .expect("open canary stress results");
    for row in rows {
        let reason = row
            .reject_reason
            .map(|value| format!("\"{}\"", reason_name(value)))
            .unwrap_or_else(|| "null".to_string());
        writeln!(
            file,
            concat!(
                "{{",
                "\"case_id\":\"{}\",",
                "\"pass\":{},",
                "\"canary_overlay_active\":{},",
                "\"default_route_unchanged\":{},",
                "\"rollback_snapshot_taken\":{},",
                "\"production_apply_allowed_now\":{},",
                "\"seeded_default_state_preserved\":{},",
                "\"action\":\"{}\",",
                "\"expected_action\":\"{}\",",
                "\"write_count\":{},",
                "\"expected_write_count\":{},",
                "\"partial_write\":{},",
                "\"order_independent\":{},",
                "\"runtime_direct_write_count\":{},",
                "\"held_variant_promoted\":{},",
                "\"oracle_plan_feature_use_count\":{},",
                "\"direct_flow_write_reject_count\":{},",
                "\"stale_snapshot_reject_count\":{},",
                "\"checksum_tamper_reject_count\":{},",
                "\"ambiguous_same_region_reject_count\":{},",
                "\"capacity_reject_count\":{},",
                "\"rollback_commit\":{},",
                "\"reject_reason\":{}",
                "}}"
            ),
            row.case_id,
            row.pass,
            row.canary_overlay_active,
            row.default_route_unchanged,
            row.rollback_snapshot_taken,
            row.production_apply_allowed_now,
            row.seeded_default_state_preserved,
            action_name(row.action),
            action_name(row.expected_action),
            row.write_count,
            row.expected_write_count,
            row.partial_write,
            row.order_independent,
            row.runtime_direct_write_count,
            row.held_variant_promoted,
            row.oracle_plan_feature_use_count,
            row.direct_flow_write_reject_count,
            row.stale_snapshot_reject_count,
            row.checksum_tamper_reject_count,
            row.ambiguous_same_region_reject_count,
            row.capacity_reject_count,
            row.rollback_commit,
            reason,
        )
        .expect("write canary stress result");
    }
}

fn write_artifacts(
    out: PathBuf,
    metrics: &Metrics,
    rows: &[CanaryResult],
    default_rounds: u64,
    canary_rounds: u64,
    seconds: f64,
) {
    fs::create_dir_all(&out).expect("create e136r artifact directory");
    let pass_gate = metrics.pass_gate(default_rounds, canary_rounds);
    let decision = if pass_gate {
        DECISION_CONFIRMED
    } else {
        DECISION_REJECTED
    };
    let default_route_candidate = pass_gate;
    fs::write(
        out.join("run_manifest.json"),
        format!(
            concat!(
                "{{\n",
                "  \"artifact_contract\": \"{}\",\n",
                "  \"boundary\": \"final pre-apply decision only; no production apply\",\n",
                "  \"generated_at_unix_ms\": {},\n",
                "  \"default_rounds\": {},\n",
                "  \"canary_rounds\": {}\n",
                "}}\n"
            ),
            ARTIFACT_CONTRACT,
            now_millis(),
            default_rounds,
            canary_rounds,
        ),
    )
    .expect("write run manifest");
    write_jsonl(out.join("canary_stress_results.jsonl"), rows);
    fs::write(
        out.join("preapply_decision.json"),
        format!(
            concat!(
                "{{\n",
                "  \"default_route_candidate\": {},\n",
                "  \"production_apply_allowed_now\": false,\n",
                "  \"recommended_next\": \"{}\",\n",
                "  \"decision_basis\": \"default regression plus seeded overlay canary stress passed\"\n",
                "}}\n"
            ),
            default_route_candidate, NEXT
        ),
    )
    .expect("write preapply decision");
    fs::write(
        out.join("rollback_manifest.json"),
        format!(
            concat!(
                "{{\n",
                "  \"rollback_snapshot_count\": {},\n",
                "  \"rollback_ready\": {},\n",
                "  \"rollback_action\": \"drop_canary_overlay_clone_keep_default_route\",\n",
                "  \"seeded_default_state_preserved_count\": {}\n",
                "}}\n"
            ),
            metrics.rollback_snapshot,
            metrics.rollback_snapshot == metrics.canary_cases,
            metrics.seeded_default_state_preserved
        ),
    )
    .expect("write rollback manifest");
    let summary = format!(
        concat!(
            "{{\n",
            "  \"artifact_contract\": \"{}\",\n",
            "  \"decision\": \"{}\",\n",
            "  \"next\": \"{}\",\n",
            "  \"pass_gate\": {},\n",
            "  \"default_route_candidate\": {},\n",
            "  \"production_apply_allowed_now\": false,\n",
            "  \"default_route_case_count\": {},\n",
            "  \"default_route_success_count\": {},\n",
            "  \"default_false_commit_count\": {},\n",
            "  \"default_missed_commit_count\": {},\n",
            "  \"canary_case_count\": {},\n",
            "  \"canary_success_count\": {},\n",
            "  \"canary_overlay_active_count\": {},\n",
            "  \"default_route_unchanged_count\": {},\n",
            "  \"rollback_snapshot_count\": {},\n",
            "  \"seeded_default_state_preserved_count\": {},\n",
            "  \"commit_single_count\": {},\n",
            "  \"commit_multi_count\": {},\n",
            "  \"commit_chunk_count\": {},\n",
            "  \"defer_count\": {},\n",
            "  \"reject_count\": {},\n",
            "  \"atomic_write_total\": {},\n",
            "  \"rollback_commit_count\": {},\n",
            "  \"partial_write_count\": {},\n",
            "  \"order_independence_failure_count\": {},\n",
            "  \"runtime_direct_write_count\": {},\n",
            "  \"held_variant_promoted_count\": {},\n",
            "  \"direct_flow_write_reject_count\": {},\n",
            "  \"stale_snapshot_reject_count\": {},\n",
            "  \"checksum_tamper_reject_count\": {},\n",
            "  \"ambiguous_same_region_reject_count\": {},\n",
            "  \"proposal_capacity_reject_count\": {},\n",
            "  \"oracle_plan_feature_use_count\": {},\n",
            "  \"destructive_delete_count\": {},\n",
            "  \"seconds\": {:.9}\n",
            "}}\n"
        ),
        ARTIFACT_CONTRACT,
        decision,
        NEXT,
        pass_gate,
        default_route_candidate,
        metrics.default_route_cases,
        metrics.default_route_success,
        metrics.default_false_commit,
        metrics.default_missed_commit,
        metrics.canary_cases,
        metrics.canary_success,
        metrics.canary_overlay_active,
        metrics.default_route_unchanged,
        metrics.rollback_snapshot,
        metrics.seeded_default_state_preserved,
        metrics.commit_single,
        metrics.commit_multi,
        metrics.commit_chunk,
        metrics.defer_count,
        metrics.reject_count,
        metrics.atomic_write_total,
        metrics.rollback_commit,
        metrics.partial_write,
        metrics.order_failure,
        metrics.runtime_direct_write,
        metrics.held_variant_promoted,
        metrics.direct_flow_write_reject,
        metrics.stale_snapshot_reject,
        metrics.checksum_tamper_reject,
        metrics.ambiguous_same_region_reject,
        metrics.capacity_reject,
        metrics.oracle_plan_feature_use,
        metrics.destructive_delete,
        seconds
    );
    fs::write(out.join("summary.json"), &summary).expect("write summary");
    fs::write(
        out.join("decision.json"),
        format!(
            concat!(
                "{{\n",
                "  \"artifact_contract\": \"{}\",\n",
                "  \"decision\": \"{}\",\n",
                "  \"next\": \"{}\",\n",
                "  \"pass_gate\": {}\n",
                "}}\n"
            ),
            ARTIFACT_CONTRACT, decision, NEXT, pass_gate
        ),
    )
    .expect("write decision");
    fs::write(
        out.join("checker_summary.json"),
        format!(
            concat!(
                "{{\n",
                "  \"artifact_contract\": \"{}\",\n",
                "  \"failure_count\": {},\n",
                "  \"failures\": [{}]\n",
                "}}\n"
            ),
            ARTIFACT_CONTRACT,
            if pass_gate { 0 } else { 1 },
            if pass_gate {
                ""
            } else {
                "\"e136r_preapply_decision_failed\""
            }
        ),
    )
    .expect("write checker summary");
    fs::write(
        out.join("report.md"),
        format!(
            "# E136R Atomic Multiwrite Pre-Apply Decision Gauntlet\n\n\
             ```text\n\
             decision = {}\n\
             next     = {}\n\
             ```\n\n\
             ```text\n\
             default_route_candidate = {}\n\
             production_apply_allowed_now = false\n\
             default_route_case_count = {}\n\
             default_route_success_count = {}\n\
             canary_case_count = {}\n\
             canary_success_count = {}\n\
             canary_overlay_active_count = {}\n\
             default_route_unchanged_count = {}\n\
             rollback_snapshot_count = {}\n\
             seeded_default_state_preserved_count = {}\n\
             partial_write_count = {}\n\
             order_independence_failure_count = {}\n\
             runtime_direct_write_count = {}\n\
             held_variant_promoted_count = {}\n\
             oracle_plan_feature_use_count = {}\n\
             ```\n\n\
             Boundary: final pre-apply decision only. No production apply.\n",
            decision,
            NEXT,
            default_route_candidate,
            metrics.default_route_cases,
            metrics.default_route_success,
            metrics.canary_cases,
            metrics.canary_success,
            metrics.canary_overlay_active,
            metrics.default_route_unchanged,
            metrics.rollback_snapshot,
            metrics.seeded_default_state_preserved,
            metrics.partial_write,
            metrics.order_failure,
            metrics.runtime_direct_write,
            metrics.held_variant_promoted,
            metrics.oracle_plan_feature_use,
        ),
    )
    .expect("write report");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let out = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("target/e136r_atomic_multiwrite_preapply_decision_gauntlet")
    });
    let default_rounds = args.get(2).and_then(|arg| arg.parse().ok()).unwrap_or(512);
    let canary_rounds = args.get(3).and_then(|arg| arg.parse().ok()).unwrap_or(2048);
    let start = Instant::now();
    let mut metrics = Metrics::default();
    for idx in 0..default_rounds {
        run_default_round(idx, &mut metrics);
    }
    let mut rows = Vec::with_capacity(canary_rounds as usize);
    for idx in 0..canary_rounds {
        let row = run_canary(idx);
        metrics.record_canary(&row);
        rows.push(row);
    }
    let seconds = start.elapsed().as_secs_f64();
    write_artifacts(out, &metrics, &rows, default_rounds, canary_rounds, seconds);
    println!(
        "{{\"artifact_contract\":\"{}\",\"decision\":\"{}\",\"next\":\"{}\",\"pass_gate\":{},\"default_route_candidate\":{},\"production_apply_allowed_now\":false,\"default_route_case_count\":{},\"canary_case_count\":{},\"canary_success_count\":{}}}",
        ARTIFACT_CONTRACT,
        if metrics.pass_gate(default_rounds, canary_rounds) {
            DECISION_CONFIRMED
        } else {
            DECISION_REJECTED
        },
        NEXT,
        metrics.pass_gate(default_rounds, canary_rounds),
        metrics.pass_gate(default_rounds, canary_rounds),
        metrics.default_route_cases,
        metrics.canary_cases,
        metrics.canary_success
    );
    if !metrics.pass_gate(default_rounds, canary_rounds) {
        std::process::exit(1);
    }
}
