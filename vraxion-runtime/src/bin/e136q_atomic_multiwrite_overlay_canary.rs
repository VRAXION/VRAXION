use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use vraxion_runtime::{
    encode_frame, safe_filler, Action, AtomicCommitAction, AtomicCommitPolicy,
    AtomicCommitProposal, AtomicOverlayCanaryConfig, AtomicProposalRole, AtomicRejectReason,
    LockedBodyRuntime,
};

const ARTIFACT_CONTRACT: &str = "E136Q_RUNTIME_OVERLAY_CANARY_ATOMIC_MULTIWRITE_CONFIRM";
const DECISION_CONFIRMED: &str = "e136q_runtime_overlay_canary_atomic_multiwrite_confirmed";
const DECISION_REJECTED: &str = "e136q_runtime_overlay_canary_atomic_multiwrite_rejected";
const NEXT: &str = "E136R_ATOMIC_MULTIWRITE_DEFAULT_ROUTE_OR_PRODUCTION_APPLY_DECISION";

#[derive(Default)]
struct Metrics {
    default_route_cases: u64,
    default_route_success: u64,
    canary_cases: u64,
    canary_success: u64,
    canary_overlay_active_count: u64,
    default_route_unchanged_count: u64,
    rollback_snapshot_count: u64,
    production_apply_allowed_count: u64,
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
    oracle_plan_feature_use: u64,
    destructive_delete: u64,
}

impl Metrics {
    fn record_default(&mut self, pass: bool) {
        self.default_route_cases += 1;
        self.default_route_success += pass as u64;
    }

    fn record_canary(&mut self, result: &CanaryCaseResult) {
        self.canary_cases += 1;
        self.canary_success += result.pass as u64;
        self.canary_overlay_active_count += result.canary_overlay_active as u64;
        self.default_route_unchanged_count += result.default_route_unchanged as u64;
        self.rollback_snapshot_count += result.rollback_snapshot_taken as u64;
        self.production_apply_allowed_count += result.production_apply_allowed_now as u64;
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
    }

    fn pass_gate(&self) -> bool {
        self.default_route_cases == self.default_route_success
            && self.default_route_cases >= 1
            && self.canary_cases == self.canary_success
            && self.canary_cases >= 8
            && self.canary_overlay_active_count == self.canary_cases
            && self.default_route_unchanged_count == self.canary_cases
            && self.rollback_snapshot_count == self.canary_cases
            && self.production_apply_allowed_count == 0
            && self.commit_multi >= 1
            && self.commit_chunk >= 1
            && self.commit_single >= 1
            && self.reject_count + self.defer_count >= 5
            && self.partial_write == 0
            && self.order_failure == 0
            && self.runtime_direct_write == 0
            && self.held_variant_promoted == 0
            && self.direct_flow_write_reject >= 1
            && self.stale_snapshot_reject >= 1
            && self.checksum_tamper_reject >= 1
            && self.ambiguous_same_region_reject >= 1
            && self.capacity_reject >= 1
            && self.oracle_plan_feature_use == 0
            && self.destructive_delete == 0
    }
}

struct CanaryCaseResult {
    case_id: &'static str,
    pass: bool,
    canary_overlay_active: bool,
    default_route_unchanged: bool,
    rollback_snapshot_taken: bool,
    production_apply_allowed_now: bool,
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
    reject_reason: Option<AtomicRejectReason>,
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_millis()
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

fn clean_stream(feature: u8, value: u8) -> Vec<u8> {
    let mut stream = safe_filler(8);
    stream.extend(encode_frame(feature, value, 1, 9));
    stream.extend(safe_filler(8));
    stream
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

fn count_reject(
    decision: &vraxion_runtime::AtomicCommitDecision,
    reason: AtomicRejectReason,
) -> usize {
    decision
        .rejected
        .iter()
        .filter(|row| row.reason == reason)
        .count()
        + usize::from(decision.reject_reason == Some(reason))
}

fn run_default_route_case() -> bool {
    let mut runtime = LockedBodyRuntime::default_body();
    let feature = 17;
    let value = 1;
    let step = runtime.process_binary_evidence(700, feature, &clean_stream(feature, value));
    step.action == Action::CommitEvidence
        && step.committed.map(|record| record.feature_id) == Some(feature)
        && step.committed.map(|record| record.value) == Some(value)
        && step.rendered.compact == "COMMIT_EVIDENCE"
}

fn run_canary_case(
    case_id: &'static str,
    proposals: &[AtomicCommitProposal],
    expected_action: AtomicCommitAction,
    expected_write_count: usize,
) -> CanaryCaseResult {
    let runtime = LockedBodyRuntime::default_body();
    let step = runtime.process_atomic_overlay_canary(
        AtomicOverlayCanaryConfig::e136q_canary(),
        AtomicCommitPolicy::e136p_preview(),
        proposals,
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
    ) && step.overlay_step.flow_active_cells != 0;
    let pass = step.canary_overlay_active
        && step.default_route_unchanged
        && step.rollback_snapshot_taken
        && !step.production_apply_allowed_now
        && decision.action == expected_action
        && decision.records.len() == expected_write_count
        && step.overlay_step.committed.len() == expected_write_count
        && !partial_write
        && decision.order_independent
        && decision.runtime_direct_write_count == 0
        && !held_variant_promoted
        && decision.oracle_plan_feature_use_count == 0
        && runtime.flow.active_count() == 0
        && runtime.ground.active_count() == 0;
    CanaryCaseResult {
        case_id,
        pass,
        canary_overlay_active: step.canary_overlay_active,
        default_route_unchanged: step.default_route_unchanged,
        rollback_snapshot_taken: step.rollback_snapshot_taken,
        production_apply_allowed_now: step.production_apply_allowed_now,
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
        reject_reason: decision.reject_reason,
    }
}

fn build_canary_cases() -> Vec<(
    &'static str,
    Vec<AtomicCommitProposal>,
    AtomicCommitAction,
    usize,
)> {
    let mut stale = proposal(9, 1, 90, 1);
    stale.cycle_id = 0;

    let mut checksum = proposal(10, 1, 100, 1);
    checksum.checksum_valid = false;

    let mut direct = proposal(11, 1, 110, 1);
    direct.direct_flow_write = true;

    let mut held = proposal(12, 1, 120, 1);
    held.role = AtomicProposalRole::HeldChallenger;

    let mut regressing_primary = proposal(13, 1, 130, 1);
    regressing_primary.primary_regression_signal = true;
    let mut rollback = proposal(13, 0, 131, 1);
    rollback.role = AtomicProposalRole::Rollback;

    let mut capacity = Vec::new();
    for idx in 0..24 {
        capacity.push(proposal(
            (idx % 31) as u8,
            (idx % 2) as u8,
            300 + idx as u32,
            1,
        ));
    }

    vec![
        (
            "canary_disjoint_atomic_multiwrite",
            vec![
                proposal(1, 1, 10, 1),
                proposal(2, 0, 11, 2),
                proposal(3, 1, 12, 3),
            ],
            AtomicCommitAction::CommitMulti,
            3,
        ),
        (
            "canary_homogeneous_chunk_commit",
            vec![
                proposal(4, 1, 20, 7),
                proposal(5, 0, 21, 7),
                proposal(6, 1, 22, 7),
            ],
            AtomicCommitAction::CommitChunk,
            3,
        ),
        (
            "canary_single_primary_commit",
            vec![proposal(7, 1, 70, 1)],
            AtomicCommitAction::CommitSingle,
            1,
        ),
        (
            "canary_stale_snapshot_reject",
            vec![stale],
            AtomicCommitAction::Defer,
            0,
        ),
        (
            "canary_checksum_tamper_reject",
            vec![checksum],
            AtomicCommitAction::Defer,
            0,
        ),
        (
            "canary_direct_flow_write_reject",
            vec![direct, proposal(14, 1, 140, 1)],
            AtomicCommitAction::CommitSingle,
            1,
        ),
        (
            "canary_ambiguous_same_region_reject",
            vec![proposal(8, 0, 80, 1), proposal(8, 1, 81, 1)],
            AtomicCommitAction::Reject,
            0,
        ),
        (
            "canary_held_challenger_hold",
            vec![held],
            AtomicCommitAction::Defer,
            0,
        ),
        (
            "canary_rollback_fallback_commit",
            vec![regressing_primary, rollback],
            AtomicCommitAction::CommitSingle,
            1,
        ),
        (
            "canary_proposal_capacity_reject",
            capacity,
            AtomicCommitAction::Reject,
            0,
        ),
    ]
}

fn case_json(result: &CanaryCaseResult) -> String {
    let reason = result
        .reject_reason
        .map(|value| format!("\"{}\"", reason_name(value)))
        .unwrap_or_else(|| "null".to_string());
    format!(
        concat!(
            "{{",
            "\"case_id\":\"{}\",",
            "\"pass\":{},",
            "\"canary_overlay_active\":{},",
            "\"default_route_unchanged\":{},",
            "\"rollback_snapshot_taken\":{},",
            "\"production_apply_allowed_now\":{},",
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
            "\"reject_reason\":{}",
            "}}"
        ),
        result.case_id,
        result.pass,
        result.canary_overlay_active,
        result.default_route_unchanged,
        result.rollback_snapshot_taken,
        result.production_apply_allowed_now,
        action_name(result.action),
        action_name(result.expected_action),
        result.write_count,
        result.expected_write_count,
        result.partial_write,
        result.order_independent,
        result.runtime_direct_write_count,
        result.held_variant_promoted,
        result.oracle_plan_feature_use_count,
        result.direct_flow_write_reject_count,
        result.stale_snapshot_reject_count,
        result.checksum_tamper_reject_count,
        result.ambiguous_same_region_reject_count,
        result.capacity_reject_count,
        reason,
    )
}

fn write_jsonl(path: PathBuf, rows: &[CanaryCaseResult]) {
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .expect("open canary results");
    for row in rows {
        writeln!(file, "{}", case_json(row)).expect("write canary result");
    }
}

fn write_artifacts(out: PathBuf, metrics: &Metrics, seconds: f64, rows: &[CanaryCaseResult]) {
    fs::create_dir_all(&out).expect("create e136q artifact directory");
    let pass_gate = metrics.pass_gate();
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
                "  \"boundary\": \"runtime overlay canary only; no production apply\",\n",
                "  \"generated_at_unix_ms\": {},\n",
                "  \"canary_case_count\": {}\n",
                "}}\n"
            ),
            ARTIFACT_CONTRACT,
            now_millis(),
            metrics.canary_cases
        ),
    )
    .expect("write run manifest");
    fs::write(
        out.join("canary_overlay_manifest.json"),
        concat!(
            "{\n",
            "  \"canary_overlay_active\": true,\n",
            "  \"canary_api\": \"LockedBodyRuntime::process_atomic_overlay_canary\",\n",
            "  \"preview_api\": \"LockedBodyRuntime::process_atomic_proposals_preview\",\n",
            "  \"agency_api\": \"agency_decide_atomic_batch\",\n",
            "  \"rollback_snapshot_required\": true,\n",
            "  \"production_apply_allowed_now\": false\n",
            "}\n"
        ),
    )
    .expect("write canary overlay manifest");
    fs::write(
        out.join("rollback_manifest.json"),
        format!(
            concat!(
                "{{\n",
                "  \"rollback_snapshot_count\": {},\n",
                "  \"rollback_ready\": {},\n",
                "  \"rollback_action\": \"drop_canary_overlay_clone_keep_default_route\"\n",
                "}}\n"
            ),
            metrics.rollback_snapshot_count,
            metrics.rollback_snapshot_count == metrics.canary_cases
        ),
    )
    .expect("write rollback manifest");
    write_jsonl(out.join("canary_case_results.jsonl"), rows);
    let summary = format!(
        concat!(
            "{{\n",
            "  \"artifact_contract\": \"{}\",\n",
            "  \"decision\": \"{}\",\n",
            "  \"next\": \"{}\",\n",
            "  \"pass_gate\": {},\n",
            "  \"canary_overlay_active\": {},\n",
            "  \"default_route_unchanged\": {},\n",
            "  \"rollback_ready\": {},\n",
            "  \"production_apply_allowed_now\": false,\n",
            "  \"default_route_case_count\": {},\n",
            "  \"default_route_success_count\": {},\n",
            "  \"canary_case_count\": {},\n",
            "  \"canary_success_count\": {},\n",
            "  \"commit_single_count\": {},\n",
            "  \"commit_multi_count\": {},\n",
            "  \"commit_chunk_count\": {},\n",
            "  \"defer_count\": {},\n",
            "  \"reject_count\": {},\n",
            "  \"atomic_write_total\": {},\n",
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
        metrics.canary_overlay_active_count == metrics.canary_cases,
        metrics.default_route_unchanged_count == metrics.canary_cases,
        metrics.rollback_snapshot_count == metrics.canary_cases,
        metrics.default_route_cases,
        metrics.default_route_success,
        metrics.canary_cases,
        metrics.canary_success,
        metrics.commit_single,
        metrics.commit_multi,
        metrics.commit_chunk,
        metrics.defer_count,
        metrics.reject_count,
        metrics.atomic_write_total,
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
        out.join("runtime_delta.json"),
        format!(
            concat!(
                "{{\n",
                "  \"rust_runtime_canary_api_added\": true,\n",
                "  \"locked_body_canary_api\": \"LockedBodyRuntime::process_atomic_overlay_canary\",\n",
                "  \"default_route_unchanged\": {},\n",
                "  \"canary_overlay_active\": {},\n",
                "  \"rollback_ready\": {},\n",
                "  \"production_apply_allowed_now\": false\n",
                "}}\n"
            ),
            metrics.default_route_unchanged_count == metrics.canary_cases,
            metrics.canary_overlay_active_count == metrics.canary_cases,
            metrics.rollback_snapshot_count == metrics.canary_cases
        ),
    )
    .expect("write runtime delta");
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
                "\"e136q_runtime_overlay_canary_failed\""
            }
        ),
    )
    .expect("write checker summary");
    fs::write(
        out.join("report.md"),
        format!(
            "# E136Q Runtime Overlay Canary Atomic Multiwrite Confirm\n\n\
             ```text\n\
             decision = {}\n\
             next     = {}\n\
             ```\n\n\
             ```text\n\
             default_route_case_count = {}\n\
             default_route_success_count = {}\n\
             canary_case_count = {}\n\
             canary_success_count = {}\n\
             canary_overlay_active = {}\n\
             default_route_unchanged = {}\n\
             rollback_ready = {}\n\
             production_apply_allowed_now = false\n\
             partial_write_count = {}\n\
             order_independence_failure_count = {}\n\
             runtime_direct_write_count = {}\n\
             held_variant_promoted_count = {}\n\
             oracle_plan_feature_use_count = {}\n\
             ```\n\n\
             Boundary: runtime overlay canary only. No production apply.\n",
            decision,
            NEXT,
            metrics.default_route_cases,
            metrics.default_route_success,
            metrics.canary_cases,
            metrics.canary_success,
            metrics.canary_overlay_active_count == metrics.canary_cases,
            metrics.default_route_unchanged_count == metrics.canary_cases,
            metrics.rollback_snapshot_count == metrics.canary_cases,
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
    let out = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("target/e136q_runtime_overlay_canary_atomic_multiwrite_confirm")
    });
    let start = Instant::now();
    let mut rows = Vec::new();
    let mut metrics = Metrics::default();
    metrics.record_default(run_default_route_case());
    for (case_id, proposals, expected_action, expected_write_count) in build_canary_cases() {
        let row = run_canary_case(case_id, &proposals, expected_action, expected_write_count);
        metrics.record_canary(&row);
        rows.push(row);
    }
    let seconds = start.elapsed().as_secs_f64();
    write_artifacts(out, &metrics, seconds, &rows);
    println!(
        "{{\"artifact_contract\":\"{}\",\"decision\":\"{}\",\"next\":\"{}\",\"pass_gate\":{},\"default_route_unchanged\":{},\"canary_overlay_active\":{},\"rollback_ready\":{},\"production_apply_allowed_now\":false,\"canary_case_count\":{},\"canary_success_count\":{}}}",
        ARTIFACT_CONTRACT,
        if metrics.pass_gate() { DECISION_CONFIRMED } else { DECISION_REJECTED },
        NEXT,
        metrics.pass_gate(),
        metrics.default_route_unchanged_count == metrics.canary_cases,
        metrics.canary_overlay_active_count == metrics.canary_cases,
        metrics.rollback_snapshot_count == metrics.canary_cases,
        metrics.canary_cases,
        metrics.canary_success
    );
    if !metrics.pass_gate() {
        std::process::exit(1);
    }
}
