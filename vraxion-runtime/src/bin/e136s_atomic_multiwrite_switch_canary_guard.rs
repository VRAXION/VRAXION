use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use vraxion_runtime::{
    corrupt_crc, encode_frame, flow_cell_for, ground_cell_for, safe_filler, Action,
    AtomicCommitAction, AtomicCommitDecision, AtomicCommitPolicy, AtomicCommitProposal,
    AtomicDefaultRouteSwitchCanaryConfig, AtomicProposalRole, AtomicRejectReason,
    LockedBodyRuntime,
};

const ARTIFACT_CONTRACT: &str = "E136S_ATOMIC_MULTIWRITE_DEFAULT_ROUTE_SWITCH_CANARY_GUARD";
const DECISION_CONFIRMED: &str =
    "e136s_atomic_multiwrite_default_route_switch_canary_guard_confirmed";
const DECISION_REJECTED: &str =
    "e136s_atomic_multiwrite_default_route_switch_canary_guard_rejected";
const NEXT: &str = "E136T_ATOMIC_MULTIWRITE_DEFAULT_ROUTE_PROBATION_ROLLOUT_DECISION";

#[derive(Default)]
struct Metrics {
    default_route_cases: u64,
    default_route_success: u64,
    default_false_commit: u64,
    default_missed_commit: u64,
    switch_cases: u64,
    switch_success: u64,
    switch_canary_active: u64,
    rollback_snapshot: u64,
    rollback_ready: u64,
    preview_checked: u64,
    preview_guard_passed: u64,
    preview_match: u64,
    default_route_applied: u64,
    blocked_no_apply: u64,
    default_route_unchanged_on_block: u64,
    production_apply_allowed: u64,
    commit_single: u64,
    commit_multi: u64,
    commit_chunk: u64,
    defer_count: u64,
    reject_count: u64,
    atomic_write_total: u64,
    rollback_commit: u64,
    guard_false_apply: u64,
    guard_missed_apply: u64,
    blocked_mutation: u64,
    preview_mismatch: u64,
    rollback_triggered: u64,
    partial_write: u64,
    runtime_direct_write: u64,
    held_variant_promoted: u64,
    direct_flow_write_reject: u64,
    stale_snapshot_reject: u64,
    checksum_tamper_reject: u64,
    ambiguous_same_region_reject: u64,
    capacity_reject: u64,
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

    fn record_switch(&mut self, result: &SwitchResult) {
        self.switch_cases += 1;
        self.switch_success += result.pass as u64;
        self.switch_canary_active += result.switch_canary_active as u64;
        self.rollback_snapshot += result.rollback_snapshot_taken as u64;
        self.rollback_ready += result.rollback_ready as u64;
        self.preview_checked += result.preview_checked as u64;
        self.preview_guard_passed += result.preview_guard_passed as u64;
        self.preview_match += result.preview_match as u64;
        self.default_route_applied += result.default_route_applied as u64;
        self.blocked_no_apply += result.blocked_no_apply as u64;
        self.default_route_unchanged_on_block += result.default_route_unchanged_on_block as u64;
        self.production_apply_allowed += result.production_apply_allowed_now as u64;
        self.commit_single += (result.action == AtomicCommitAction::CommitSingle) as u64;
        self.commit_multi += (result.action == AtomicCommitAction::CommitMulti) as u64;
        self.commit_chunk += (result.action == AtomicCommitAction::CommitChunk) as u64;
        self.defer_count += (result.action == AtomicCommitAction::Defer) as u64;
        self.reject_count += (result.action == AtomicCommitAction::Reject) as u64;
        self.atomic_write_total += result.write_count as u64;
        self.rollback_commit += result.rollback_commit as u64;
        self.guard_false_apply += result.guard_false_apply as u64;
        self.guard_missed_apply += result.guard_missed_apply as u64;
        self.blocked_mutation += result.blocked_mutation as u64;
        self.preview_mismatch += (!result.preview_match && result.expected_apply) as u64;
        self.rollback_triggered += result.default_route_rolled_back as u64;
        self.partial_write += result.partial_write as u64;
        self.runtime_direct_write += result.runtime_direct_write_count as u64;
        self.held_variant_promoted += result.held_variant_promoted as u64;
        self.oracle_plan_feature_use += result.oracle_plan_feature_use_count as u64;
        self.direct_flow_write_reject += result.direct_flow_write_reject_count as u64;
        self.stale_snapshot_reject += result.stale_snapshot_reject_count as u64;
        self.checksum_tamper_reject += result.checksum_tamper_reject_count as u64;
        self.ambiguous_same_region_reject += result.ambiguous_same_region_reject_count as u64;
        self.capacity_reject += result.capacity_reject_count as u64;
    }

    fn pass_gate(&self, default_rounds: u64, switch_rounds: u64) -> bool {
        self.default_route_cases == self.default_route_success
            && self.default_route_cases >= default_rounds * 3
            && self.default_false_commit == 0
            && self.default_missed_commit == 0
            && self.switch_cases == self.switch_success
            && self.switch_cases >= switch_rounds
            && self.switch_canary_active == self.switch_cases
            && self.rollback_snapshot == self.switch_cases
            && self.rollback_ready == self.switch_cases
            && self.preview_checked == self.switch_cases
            && self.production_apply_allowed == 0
            && self.default_route_applied >= switch_rounds * 2 / 5
            && self.blocked_no_apply >= switch_rounds * 2 / 5
            && self.default_route_unchanged_on_block == self.blocked_no_apply
            && self.preview_guard_passed == self.default_route_applied
            && self.preview_match == self.switch_cases
            && self.commit_single >= switch_rounds / 5
            && self.commit_multi >= switch_rounds / 10
            && self.commit_chunk >= switch_rounds / 10
            && self.reject_count + self.defer_count >= switch_rounds * 2 / 5
            && self.rollback_commit >= switch_rounds / 10
            && self.direct_flow_write_reject >= switch_rounds / 10
            && self.stale_snapshot_reject >= switch_rounds / 10
            && self.checksum_tamper_reject >= switch_rounds / 10
            && self.ambiguous_same_region_reject >= switch_rounds / 10
            && self.capacity_reject >= switch_rounds / 10
            && self.guard_false_apply == 0
            && self.guard_missed_apply == 0
            && self.blocked_mutation == 0
            && self.preview_mismatch == 0
            && self.rollback_triggered == 0
            && self.partial_write == 0
            && self.runtime_direct_write == 0
            && self.held_variant_promoted == 0
            && self.oracle_plan_feature_use == 0
            && self.destructive_delete == 0
    }
}

struct SwitchResult {
    case_id: String,
    pass: bool,
    expected_apply: bool,
    switch_canary_active: bool,
    default_route_apply_allowed: bool,
    production_apply_allowed_now: bool,
    rollback_snapshot_taken: bool,
    rollback_ready: bool,
    preview_checked: bool,
    preview_guard_passed: bool,
    preview_match: bool,
    default_route_applied: bool,
    default_route_rolled_back: bool,
    blocked_no_apply: bool,
    default_route_unchanged_on_block: bool,
    action: AtomicCommitAction,
    expected_action: AtomicCommitAction,
    write_count: usize,
    expected_write_count: usize,
    applied_records_present: bool,
    guard_false_apply: bool,
    guard_missed_apply: bool,
    blocked_mutation: bool,
    partial_write: bool,
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

fn build_switch_case(
    idx: u64,
) -> (
    String,
    Vec<AtomicCommitProposal>,
    AtomicCommitAction,
    usize,
    bool,
    bool,
) {
    let base_feature = ((idx * 7) % 24) as u8;
    let base_source = 20_000 + idx as u32 * 16;
    match idx % 10 {
        0 => (
            format!("switch_{idx:04}_disjoint_atomic_multiwrite"),
            vec![
                proposal(base_feature, 1, base_source, 1),
                proposal(base_feature + 1, 0, base_source + 1, 2),
                proposal(base_feature + 2, 1, base_source + 2, 3),
            ],
            AtomicCommitAction::CommitMulti,
            3,
            true,
            false,
        ),
        1 => (
            format!("switch_{idx:04}_homogeneous_chunk_commit"),
            vec![
                proposal(base_feature, 1, base_source, 7),
                proposal(base_feature + 1, 0, base_source + 1, 7),
                proposal(base_feature + 2, 1, base_source + 2, 7),
            ],
            AtomicCommitAction::CommitChunk,
            3,
            true,
            false,
        ),
        2 => (
            format!("switch_{idx:04}_single_primary_commit"),
            vec![proposal(base_feature, 1, base_source, 1)],
            AtomicCommitAction::CommitSingle,
            1,
            true,
            false,
        ),
        3 => {
            let mut stale = proposal(base_feature, 1, base_source, 1);
            stale.cycle_id = 0;
            (
                format!("switch_{idx:04}_stale_snapshot_reject"),
                vec![stale],
                AtomicCommitAction::Defer,
                0,
                false,
                false,
            )
        }
        4 => {
            let mut checksum = proposal(base_feature, 1, base_source, 1);
            checksum.checksum_valid = false;
            (
                format!("switch_{idx:04}_checksum_tamper_reject"),
                vec![checksum],
                AtomicCommitAction::Defer,
                0,
                false,
                false,
            )
        }
        5 => {
            let mut direct = proposal(base_feature, 1, base_source, 1);
            direct.direct_flow_write = true;
            (
                format!("switch_{idx:04}_direct_flow_write_reject"),
                vec![direct, proposal(base_feature + 1, 1, base_source + 1, 1)],
                AtomicCommitAction::CommitSingle,
                1,
                true,
                false,
            )
        }
        6 => (
            format!("switch_{idx:04}_ambiguous_same_region_reject"),
            vec![
                proposal(base_feature, 0, base_source, 1),
                proposal(base_feature, 1, base_source + 1, 1),
            ],
            AtomicCommitAction::Reject,
            0,
            false,
            false,
        ),
        7 => {
            let mut held = proposal(base_feature, 1, base_source, 1);
            held.role = AtomicProposalRole::HeldChallenger;
            (
                format!("switch_{idx:04}_held_challenger_hold"),
                vec![held],
                AtomicCommitAction::Defer,
                0,
                false,
                false,
            )
        }
        8 => {
            let mut regressing = proposal(base_feature, 1, base_source, 1);
            regressing.primary_regression_signal = true;
            let mut rollback = proposal(base_feature, 0, base_source + 1, 1);
            rollback.role = AtomicProposalRole::Rollback;
            (
                format!("switch_{idx:04}_rollback_fallback_commit"),
                vec![regressing, rollback],
                AtomicCommitAction::CommitSingle,
                1,
                true,
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
                format!("switch_{idx:04}_proposal_capacity_reject"),
                proposals,
                AtomicCommitAction::Reject,
                0,
                false,
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

fn committed_records_present(runtime: &LockedBodyRuntime, decision: &AtomicCommitDecision) -> bool {
    decision.records.iter().all(|record| {
        runtime
            .flow
            .read(flow_cell_for(record.feature_id, runtime.config))
            == Some(record.value)
            && runtime
                .ground
                .read(ground_cell_for(record.feature_id, runtime.config))
                == Some(record.value)
    })
}

fn run_switch(idx: u64) -> SwitchResult {
    let (
        case_id,
        proposals,
        expected_action,
        expected_write_count,
        expected_apply,
        expect_rollback,
    ) = build_switch_case(idx);
    let proposals = permute(proposals, idx);
    let mut runtime = seeded_runtime(idx);
    let before = runtime.clone();
    let step = runtime.process_atomic_default_route_switch_canary(
        AtomicDefaultRouteSwitchCanaryConfig::e136s_switch_canary(),
        AtomicCommitPolicy::e136p_preview(),
        &proposals,
    );
    let decision = &step.preview_step.decision;
    let blocked_no_apply = !expected_apply && !step.default_route_applied;
    let blocked_mutation = blocked_no_apply
        && (runtime.flow != before.flow
            || runtime.ground != before.ground
            || runtime.cycle_id != before.cycle_id);
    let guard_false_apply = step.default_route_applied && !expected_apply;
    let guard_missed_apply = !step.default_route_applied && expected_apply;
    let applied_records_present = !expected_apply
        || (step.default_route_applied && committed_records_present(&runtime, decision));
    let rollback_commit = expect_rollback
        && decision.records.iter().any(|record| {
            proposals.iter().any(|proposal| {
                proposal.role == AtomicProposalRole::Rollback
                    && proposal.source_pocket_id == record.source_pocket_id
            })
        });
    let partial_write = !step.default_route_applied
        && (runtime.flow != before.flow || runtime.ground != before.ground);
    let pass = step.switch_canary_active
        && step.default_route_apply_allowed
        && step.rollback_snapshot_taken
        && step.rollback_ready
        && step.preview_checked
        && !step.production_apply_allowed_now
        && decision.action == expected_action
        && decision.records.len() == expected_write_count
        && step.default_route_applied == expected_apply
        && step.preview_guard_passed == expected_apply
        && step.preview_match
        && applied_records_present
        && !guard_false_apply
        && !guard_missed_apply
        && !blocked_mutation
        && !partial_write
        && (!expected_apply || step.applied_step.is_some())
        && (expected_apply || step.applied_step.is_none())
        && (expected_apply || step.default_route_unchanged_on_block)
        && (!expect_rollback || rollback_commit)
        && decision.order_independent
        && decision.runtime_direct_write_count == 0
        && !step.held_variant_promoted
        && decision.oracle_plan_feature_use_count == 0;

    SwitchResult {
        case_id,
        pass,
        expected_apply,
        switch_canary_active: step.switch_canary_active,
        default_route_apply_allowed: step.default_route_apply_allowed,
        production_apply_allowed_now: step.production_apply_allowed_now,
        rollback_snapshot_taken: step.rollback_snapshot_taken,
        rollback_ready: step.rollback_ready,
        preview_checked: step.preview_checked,
        preview_guard_passed: step.preview_guard_passed,
        preview_match: step.preview_match,
        default_route_applied: step.default_route_applied,
        default_route_rolled_back: step.default_route_rolled_back,
        blocked_no_apply,
        default_route_unchanged_on_block: step.default_route_unchanged_on_block,
        action: decision.action,
        expected_action,
        write_count: decision.records.len(),
        expected_write_count,
        applied_records_present,
        guard_false_apply,
        guard_missed_apply,
        blocked_mutation,
        partial_write,
        runtime_direct_write_count: decision.runtime_direct_write_count,
        held_variant_promoted: step.held_variant_promoted,
        oracle_plan_feature_use_count: decision.oracle_plan_feature_use_count,
        direct_flow_write_reject_count: count_reject(decision, AtomicRejectReason::DirectFlowWrite),
        stale_snapshot_reject_count: count_reject(decision, AtomicRejectReason::StaleSnapshot),
        checksum_tamper_reject_count: count_reject(decision, AtomicRejectReason::ChecksumInvalid),
        ambiguous_same_region_reject_count: count_reject(
            decision,
            AtomicRejectReason::AmbiguousSameRegion,
        ),
        capacity_reject_count: count_reject(decision, AtomicRejectReason::ProposalCapacityExceeded),
        rollback_commit,
        reject_reason: decision.reject_reason,
    }
}

fn write_jsonl(path: PathBuf, rows: &[SwitchResult]) {
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .expect("open switch canary results");
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
                "\"expected_apply\":{},",
                "\"switch_canary_active\":{},",
                "\"default_route_apply_allowed\":{},",
                "\"production_apply_allowed_now\":{},",
                "\"rollback_snapshot_taken\":{},",
                "\"rollback_ready\":{},",
                "\"preview_checked\":{},",
                "\"preview_guard_passed\":{},",
                "\"preview_match\":{},",
                "\"default_route_applied\":{},",
                "\"default_route_rolled_back\":{},",
                "\"blocked_no_apply\":{},",
                "\"default_route_unchanged_on_block\":{},",
                "\"action\":\"{}\",",
                "\"expected_action\":\"{}\",",
                "\"write_count\":{},",
                "\"expected_write_count\":{},",
                "\"applied_records_present\":{},",
                "\"guard_false_apply\":{},",
                "\"guard_missed_apply\":{},",
                "\"blocked_mutation\":{},",
                "\"partial_write\":{},",
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
            row.expected_apply,
            row.switch_canary_active,
            row.default_route_apply_allowed,
            row.production_apply_allowed_now,
            row.rollback_snapshot_taken,
            row.rollback_ready,
            row.preview_checked,
            row.preview_guard_passed,
            row.preview_match,
            row.default_route_applied,
            row.default_route_rolled_back,
            row.blocked_no_apply,
            row.default_route_unchanged_on_block,
            action_name(row.action),
            action_name(row.expected_action),
            row.write_count,
            row.expected_write_count,
            row.applied_records_present,
            row.guard_false_apply,
            row.guard_missed_apply,
            row.blocked_mutation,
            row.partial_write,
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
        .expect("write switch canary result");
    }
}

fn write_artifacts(
    out: PathBuf,
    metrics: &Metrics,
    rows: &[SwitchResult],
    default_rounds: u64,
    switch_rounds: u64,
    seconds: f64,
) {
    fs::create_dir_all(&out).expect("create e136s artifact directory");
    let pass_gate = metrics.pass_gate(default_rounds, switch_rounds);
    let decision = if pass_gate {
        DECISION_CONFIRMED
    } else {
        DECISION_REJECTED
    };
    fs::write(
        out.join("run_manifest.json"),
        format!(
            concat!(
                "{{\n",
                "  \"artifact_contract\": \"{}\",\n",
                "  \"boundary\": \"default-route switch canary guard only; no production apply\",\n",
                "  \"generated_at_unix_ms\": {},\n",
                "  \"default_rounds\": {},\n",
                "  \"switch_rounds\": {}\n",
                "}}\n"
            ),
            ARTIFACT_CONTRACT,
            now_millis(),
            default_rounds,
            switch_rounds,
        ),
    )
    .expect("write run manifest");
    write_jsonl(out.join("switch_canary_results.jsonl"), rows);
    fs::write(
        out.join("switch_guard_manifest.json"),
        format!(
            concat!(
                "{{\n",
                "  \"default_route_switch_candidate\": {},\n",
                "  \"production_apply_allowed_now\": false,\n",
                "  \"guarded_default_route_apply_count\": {},\n",
                "  \"guarded_blocked_no_apply_count\": {},\n",
                "  \"rollback_snapshot_count\": {},\n",
                "  \"preview_match_count\": {},\n",
                "  \"recommended_next\": \"{}\"\n",
                "}}\n"
            ),
            pass_gate,
            metrics.default_route_applied,
            metrics.blocked_no_apply,
            metrics.rollback_snapshot,
            metrics.preview_match,
            NEXT
        ),
    )
    .expect("write switch guard manifest");
    let summary = format!(
        concat!(
            "{{\n",
            "  \"artifact_contract\": \"{}\",\n",
            "  \"decision\": \"{}\",\n",
            "  \"next\": \"{}\",\n",
            "  \"pass_gate\": {},\n",
            "  \"default_route_switch_candidate\": {},\n",
            "  \"production_apply_allowed_now\": false,\n",
            "  \"default_route_case_count\": {},\n",
            "  \"default_route_success_count\": {},\n",
            "  \"default_false_commit_count\": {},\n",
            "  \"default_missed_commit_count\": {},\n",
            "  \"switch_case_count\": {},\n",
            "  \"switch_success_count\": {},\n",
            "  \"switch_canary_active_count\": {},\n",
            "  \"rollback_snapshot_count\": {},\n",
            "  \"rollback_ready_count\": {},\n",
            "  \"preview_checked_count\": {},\n",
            "  \"preview_guard_passed_count\": {},\n",
            "  \"preview_match_count\": {},\n",
            "  \"default_route_applied_count\": {},\n",
            "  \"blocked_no_apply_count\": {},\n",
            "  \"default_route_unchanged_on_block_count\": {},\n",
            "  \"commit_single_count\": {},\n",
            "  \"commit_multi_count\": {},\n",
            "  \"commit_chunk_count\": {},\n",
            "  \"defer_count\": {},\n",
            "  \"reject_count\": {},\n",
            "  \"atomic_write_total\": {},\n",
            "  \"rollback_commit_count\": {},\n",
            "  \"guard_false_apply_count\": {},\n",
            "  \"guard_missed_apply_count\": {},\n",
            "  \"blocked_mutation_count\": {},\n",
            "  \"preview_mismatch_count\": {},\n",
            "  \"rollback_triggered_count\": {},\n",
            "  \"partial_write_count\": {},\n",
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
        pass_gate,
        metrics.default_route_cases,
        metrics.default_route_success,
        metrics.default_false_commit,
        metrics.default_missed_commit,
        metrics.switch_cases,
        metrics.switch_success,
        metrics.switch_canary_active,
        metrics.rollback_snapshot,
        metrics.rollback_ready,
        metrics.preview_checked,
        metrics.preview_guard_passed,
        metrics.preview_match,
        metrics.default_route_applied,
        metrics.blocked_no_apply,
        metrics.default_route_unchanged_on_block,
        metrics.commit_single,
        metrics.commit_multi,
        metrics.commit_chunk,
        metrics.defer_count,
        metrics.reject_count,
        metrics.atomic_write_total,
        metrics.rollback_commit,
        metrics.guard_false_apply,
        metrics.guard_missed_apply,
        metrics.blocked_mutation,
        metrics.preview_mismatch,
        metrics.rollback_triggered,
        metrics.partial_write,
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
                "\"e136s_switch_canary_guard_failed\""
            }
        ),
    )
    .expect("write checker summary");
    fs::write(
        out.join("report.md"),
        format!(
            "# E136S Atomic Multiwrite Default-Route Switch Canary Guard\n\n\
             ```text\n\
             decision = {}\n\
             next     = {}\n\
             ```\n\n\
             ```text\n\
             default_route_switch_candidate = {}\n\
             production_apply_allowed_now = false\n\
             default_route_case_count = {}\n\
             default_route_success_count = {}\n\
             switch_case_count = {}\n\
             switch_success_count = {}\n\
             default_route_applied_count = {}\n\
             blocked_no_apply_count = {}\n\
             rollback_snapshot_count = {}\n\
             preview_guard_passed_count = {}\n\
             preview_match_count = {}\n\
             guard_false_apply_count = {}\n\
             guard_missed_apply_count = {}\n\
             blocked_mutation_count = {}\n\
             partial_write_count = {}\n\
             runtime_direct_write_count = {}\n\
             held_variant_promoted_count = {}\n\
             oracle_plan_feature_use_count = {}\n\
             ```\n\n\
             Boundary: default-route switch canary guard only. No production apply.\n",
            decision,
            NEXT,
            pass_gate,
            metrics.default_route_cases,
            metrics.default_route_success,
            metrics.switch_cases,
            metrics.switch_success,
            metrics.default_route_applied,
            metrics.blocked_no_apply,
            metrics.rollback_snapshot,
            metrics.preview_guard_passed,
            metrics.preview_match,
            metrics.guard_false_apply,
            metrics.guard_missed_apply,
            metrics.blocked_mutation,
            metrics.partial_write,
            metrics.runtime_direct_write,
            metrics.held_variant_promoted,
            metrics.oracle_plan_feature_use,
        ),
    )
    .expect("write report");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let out = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/e136s_atomic_multiwrite_switch_canary_guard"));
    let default_rounds = args.get(2).and_then(|arg| arg.parse().ok()).unwrap_or(512);
    let switch_rounds = args.get(3).and_then(|arg| arg.parse().ok()).unwrap_or(2048);
    let start = Instant::now();
    let mut metrics = Metrics::default();
    for idx in 0..default_rounds {
        run_default_round(idx, &mut metrics);
    }
    let mut rows = Vec::with_capacity(switch_rounds as usize);
    for idx in 0..switch_rounds {
        let row = run_switch(idx);
        metrics.record_switch(&row);
        rows.push(row);
    }
    let seconds = start.elapsed().as_secs_f64();
    write_artifacts(out, &metrics, &rows, default_rounds, switch_rounds, seconds);
    let pass_gate = metrics.pass_gate(default_rounds, switch_rounds);
    println!(
        "{{\"artifact_contract\":\"{}\",\"decision\":\"{}\",\"next\":\"{}\",\"pass_gate\":{},\"default_route_switch_candidate\":{},\"production_apply_allowed_now\":false,\"default_route_case_count\":{},\"switch_case_count\":{},\"switch_success_count\":{},\"default_route_applied_count\":{},\"blocked_no_apply_count\":{}}}",
        ARTIFACT_CONTRACT,
        if pass_gate {
            DECISION_CONFIRMED
        } else {
            DECISION_REJECTED
        },
        NEXT,
        pass_gate,
        pass_gate,
        metrics.default_route_cases,
        metrics.switch_cases,
        metrics.switch_success,
        metrics.default_route_applied,
        metrics.blocked_no_apply
    );
    if !pass_gate {
        std::process::exit(1);
    }
}
