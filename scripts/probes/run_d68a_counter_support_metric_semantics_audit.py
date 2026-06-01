#!/usr/bin/env python3
"""D68A counter-support metric semantics audit.

D68 showed a useful support-cost reduction but failed the accuracy gates. Before
repairing the policy, D68A audits whether the D68 "unnecessary" and "missed"
counter-support metrics mean what they appeared to mean. In particular, D68's
metric counted "any internal counter would fix" rather than "the selected
concrete counter fixed", and counted external-test requests as unnecessary even
when no external support was actually used.
"""

import argparse
import copy
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d51_mutable_ecf_controller_prototype as d51
import run_d55_sparse_firing_ecf_controller_prototype as d55
import run_d59_rust_sparse_ecf_controller_with_mutation as d59
import run_d65_set_invariant_ipf_aggregation_prototype as d65
import run_d68_counter_support_triage_repair as d68

TASK = "D68A_COUNTER_SUPPORT_METRIC_SEMANTICS_AUDIT"
BOUNDARY = (
    "D68A only audits counter-support metric semantics for controlled symbolic "
    "joint formula discovery. It does not prove full VRAXION brain, raw visual "
    "Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, "
    "architecture superiority, or production readiness."
)

SUPPORT_COUNT = d68.SUPPORT_COUNT
REGIMES = d68.REGIMES
CORE_REGIMES = d68.CORE_REGIMES
INTERNAL_COUNTER_ACTIONS = d68.INTERNAL_COUNTER_ACTIONS
EXTERNAL_ACTION = d68.EXTERNAL_ACTION
COUNTER_ACTIONS = d68.COUNTER_ACTIONS
ROW_SAMPLE_PER_ARM_REGIME = 16

AUDIT_ARMS = [
    "D67_BEST_REPLAY",
    "D68_TRAINED_THRESHOLD_REPLAY",
    "COUNTER_REMOVAL_REPLAY",
    "CONCRETE_COUNTER_ORACLE_REFERENCE_ONLY",
    "CHEAPEST_CORRECT_ORACLE_REFERENCE_ONLY",
    "ALWAYS_COUNTER_CONTROL",
    "NEVER_COUNTER_CONTROL",
    "RANDOM_COUNTER_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]

REFERENCE_ONLY_ARMS = [
    "CONCRETE_COUNTER_ORACLE_REFERENCE_ONLY",
    "CHEAPEST_CORRECT_ORACLE_REFERENCE_ONLY",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]
FAIR_ARMS = [arm for arm in AUDIT_ARMS if arm not in REFERENCE_ONLY_ARMS]

SOURCE_FOR_ARM = {
    "D67_BEST_REPLAY": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "D68_TRAINED_THRESHOLD_REPLAY": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "COUNTER_REMOVAL_REPLAY": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "CONCRETE_COUNTER_ORACLE_REFERENCE_ONLY": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "CHEAPEST_CORRECT_ORACLE_REFERENCE_ONLY": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "ALWAYS_COUNTER_CONTROL": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "NEVER_COUNTER_CONTROL": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "RANDOM_COUNTER_CONTROL": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
}

POLICY_FOR_ACTION = {
    "DECIDE": "DECIDE_POLICY",
    "REQUEST_COUNTER_TOP1_TOP2": "COUNTERFACTUAL_POLICY",
    "REQUEST_JOINT_COUNTER": "ADVERSARIAL_REPAIR_POLICY",
    "REQUEST_EXTERNAL_TEST": "EXTERNAL_TEST_POLICY",
    "ABSTAIN": "ABSTAIN_POLICY",
}


def write_json(path, value):
    d51.write_json(path, value)


def append_jsonl(path, value):
    d51.append_jsonl(path, value)


def append_progress(out, event, started, data):
    d51.append_progress(out, event, started, data)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def parse_seeds(raw):
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def effective_correct(pack, action):
    item = pack["actions"].get(action, {})
    if bool(item.get("correct")):
        return True
    return pack.get("support_regime") == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT" and action == "ABSTAIN"


def action_support(pack, action):
    return int(pack["actions"].get(action, {}).get("total_support_used", SUPPORT_COUNT))


def action_external_used(pack, action):
    return int(pack["actions"].get(action, {}).get("external_test_used", 0))


def action_counter_used(pack, action):
    return int(pack["actions"].get(action, {}).get("counter_support_used", 0))


def cheapest_effective_action(pack):
    candidates = []
    for action in pack["actions"]:
        if effective_correct(pack, action):
            candidates.append((action_support(pack, action), action))
    if not candidates:
        return None, action_support(pack, "DECIDE")
    support, action = min(candidates)
    return action, support


def decorate_pack(pack, arm, source):
    out = copy.deepcopy(pack)
    out["d68a_arm"] = arm
    out["d65_source_arm"] = source
    out["support_budget_cap"] = None
    out["support_scoring_used"] = arm in {
        "D67_BEST_REPLAY",
        "D68_TRAINED_THRESHOLD_REPLAY",
        "COUNTER_REMOVAL_REPLAY",
    }
    return out


def build_audit_items(rows, bundle, rust_features, out, started, heartbeat_sec, split):
    items = []
    total = len(rows) * len(AUDIT_ARMS)
    completed = 0
    last = time.time()
    for idx, row in enumerate(rows):
        source_cache = {}
        for arm in AUDIT_ARMS:
            source = SOURCE_FOR_ARM[arm]
            if source not in source_cache:
                source_cache[source] = d65.build_pack(row, bundle, source, idx, rust_features)
            items.append({"arm": arm, "pack": decorate_pack(source_cache[source], arm, source)})
            completed += 1
        now = time.time()
        if now - last >= heartbeat_sec:
            last = now
            write_json(out / f"partial_{split}_d68a_pack_build.json", {"completed": completed, "total": total})
            append_progress(out, "d68a_pack_build_progress", started, {"split": split, "completed": completed, "total": total})
    append_progress(out, "d68a_pack_build_complete", started, {"split": split, "packs": len(items)})
    return items


def policy_from_action(action):
    return POLICY_FOR_ACTION.get(action, "DECIDE_POLICY")


def select_oracle_action(pack, mode):
    actions = pack["actions"]
    if mode == "concrete_counter":
        if effective_correct(pack, "DECIDE"):
            return "DECIDE", "CONCRETE_COUNTER_ORACLE_REFERENCE_ONLY", "decide_already_effective", True
        effective_counters = [
            action for action in ["REQUEST_COUNTER_TOP1_TOP2", "REQUEST_JOINT_COUNTER", "REQUEST_EXTERNAL_TEST", "ABSTAIN"]
            if effective_correct(pack, action)
        ]
        if effective_counters:
            return min(effective_counters, key=lambda action: (action_support(pack, action), action)), "CONCRETE_COUNTER_ORACLE_REFERENCE_ONLY", "reference_only_concrete_fix", True
        return "DECIDE", "CONCRETE_COUNTER_ORACLE_REFERENCE_ONLY", "reference_only_no_fix", True
    if mode == "cheapest":
        action, _ = cheapest_effective_action(pack)
        return action or "DECIDE", "CHEAPEST_CORRECT_ORACLE_REFERENCE_ONLY", "reference_only_cheapest_effective_action", True

    scored = []
    for action in actions:
        effective = effective_correct(pack, action)
        scored.append((1.0 if effective else 0.0, -action_support(pack, action), action))
    return max(scored)[2], "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY", "reference_only_truth_scored_action", True


def select_audit_policy(arm, pack, learned_triage):
    if arm == "D67_BEST_REPLAY":
        policy, features, basis = d68.support_scoring_policy(pack)
        return policy, d68.action_for_policy_name(policy), features, basis, False
    if arm == "D68_TRAINED_THRESHOLD_REPLAY":
        policy, features, basis = d68.counter_support_triage_policy(pack, "TRAINED_THRESHOLD_TRIAGE_GATE", learned_triage)
        return policy, d68.action_for_policy_name(policy), features, basis, False
    if arm == "COUNTER_REMOVAL_REPLAY":
        policy, features, basis = d68.counter_support_triage_policy(pack, "TRAINED_THRESHOLD_TRIAGE_GATE", learned_triage)
        action = d68.action_for_policy_name(policy)
        if action in COUNTER_ACTIONS:
            return "DECIDE_POLICY", "DECIDE", features, f"{basis}_counter_removed_to_decide", False
        return policy, action, features, basis, False
    if arm == "ALWAYS_COUNTER_CONTROL":
        features = d68.d68_gate_features(pack)
        return "ADVERSARIAL_REPAIR_POLICY", "REQUEST_JOINT_COUNTER", features, "forced_always_joint_counter", False
    if arm == "NEVER_COUNTER_CONTROL":
        features = d68.d68_gate_features(pack)
        preflight, reason = d68.triage_preflight(features)
        if preflight:
            return preflight, d68.action_for_policy_name(preflight), features, f"never_counter_keeps_{reason}", False
        return "DECIDE_POLICY", "DECIDE", features, "forced_never_counter", False
    if arm == "RANDOM_COUNTER_CONTROL":
        policy = d68.deterministic_random_counter(pack)
        return policy, d68.action_for_policy_name(policy), d68.d68_gate_features(pack), "deterministic_random_counter_control", False
    if arm == "CONCRETE_COUNTER_ORACLE_REFERENCE_ONLY":
        action, policy, basis, truth = select_oracle_action(pack, "concrete_counter")
        return policy_from_action(action), action, d68.d68_gate_features(pack), basis, truth
    if arm == "CHEAPEST_CORRECT_ORACLE_REFERENCE_ONLY":
        action, policy, basis, truth = select_oracle_action(pack, "cheapest")
        return policy_from_action(action), action, d68.d68_gate_features(pack), basis, truth
    if arm == "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY":
        action, policy, basis, truth = select_oracle_action(pack, "truth")
        return policy_from_action(action), action, d68.d68_gate_features(pack), basis, truth
    raise KeyError(f"unknown audit arm {arm}")


def concrete_action_audit(pack, selected_action):
    decide_eff = effective_correct(pack, "DECIDE")
    selected_eff = effective_correct(pack, selected_action)
    selected_internal = selected_action in INTERNAL_COUNTER_ACTIONS
    selected_external = selected_action == EXTERNAL_ACTION
    selected_counter = selected_action in COUNTER_ACTIONS
    selected_support = action_support(pack, selected_action)
    decide_support = action_support(pack, "DECIDE")
    cheapest_action, cheapest_support = cheapest_effective_action(pack)

    internal_effective = {action: effective_correct(pack, action) for action in INTERNAL_COUNTER_ACTIONS}
    any_internal_fix = (not decide_eff) and any(internal_effective.values())
    any_external_fix = (not decide_eff) and effective_correct(pack, EXTERNAL_ACTION) and action_external_used(pack, EXTERNAL_ACTION) > 0
    any_repair_fix = any_internal_fix or any_external_fix
    other_internal_fix = selected_internal and any(
        action != selected_action and ok for action, ok in internal_effective.items()
    )

    reported_unnecessary = selected_counter and decide_eff
    causal_unnecessary = reported_unnecessary and selected_eff and selected_support > cheapest_support
    no_cost_unnecessary_request = reported_unnecessary and selected_support <= cheapest_support
    harmful_unnecessary = reported_unnecessary and not selected_eff

    selected_concrete_counter_fixes = selected_internal and (not decide_eff) and selected_eff
    selected_external_fixes = selected_external and (not decide_eff) and selected_eff and action_external_used(pack, selected_action) > 0
    selected_repair_fixes = (selected_concrete_counter_fixes or selected_external_fixes)

    wrong_concrete_counter = selected_internal and (not selected_eff) and other_internal_fix
    weak_top1_top2_path_failure = (
        selected_action == "REQUEST_COUNTER_TOP1_TOP2"
        and not selected_eff
        and effective_correct(pack, "REQUEST_JOINT_COUNTER")
    )
    lost_external_possible = selected_action != EXTERNAL_ACTION and any_external_fix
    selected_external_unavailable = selected_external and action_external_used(pack, selected_action) == 0

    concrete_missed = any_repair_fix and not selected_repair_fixes and selected_action != "ABSTAIN"
    if pack.get("support_regime") == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT" and selected_action == "ABSTAIN":
        concrete_missed = False

    return {
        "decide_effective_correct": decide_eff,
        "selected_action_effective_correct": selected_eff,
        "selected_concrete_counter_fixes": selected_concrete_counter_fixes,
        "selected_external_fixes": selected_external_fixes,
        "selected_repair_fixes": selected_repair_fixes,
        "any_internal_counter_action_fixes": any_internal_fix,
        "any_external_action_fixes": any_external_fix,
        "any_repair_action_fixes": any_repair_fix,
        "other_internal_counter_would_fix": other_internal_fix,
        "wrong_concrete_counter": wrong_concrete_counter,
        "right_counter_family_wrong_concrete_action": wrong_concrete_counter,
        "weak_top1_top2_path_failure": weak_top1_top2_path_failure,
        "lost_external_path_possible": lost_external_possible,
        "reported_unnecessary_counter_request": reported_unnecessary,
        "causal_unnecessary_counter_support": causal_unnecessary,
        "no_cost_unnecessary_counter_request": no_cost_unnecessary_request,
        "harmful_unnecessary_counter": harmful_unnecessary,
        "reported_missed_counter_request": any_repair_fix and not selected_counter,
        "concrete_selected_counter_missed": concrete_missed,
        "selected_external_request_unavailable": selected_external_unavailable,
        "selected_counter_support_used": action_counter_used(pack, selected_action),
        "selected_external_test_used": action_external_used(pack, selected_action),
        "decide_support_used": decide_support,
        "selected_support_used": selected_support,
        "cheapest_effective_action": cheapest_action,
        "cheapest_effective_support_used": cheapest_support,
        "support_over_cheapest_effective": max(0, selected_support - cheapest_support),
        "request_top1_top2_effective": effective_correct(pack, "REQUEST_COUNTER_TOP1_TOP2"),
        "request_joint_counter_effective": effective_correct(pack, "REQUEST_JOINT_COUNTER"),
        "request_external_test_effective": effective_correct(pack, EXTERNAL_ACTION),
        "request_external_test_used_if_selected": action_external_used(pack, EXTERNAL_ACTION),
    }


def make_output_row(pack, arm, policy, selected_action, action_record, gate_features, gate_basis, used_truth):
    row = d68.output_row(pack, arm, policy, action_record, gate_features, gate_basis, used_truth)
    row.update(concrete_action_audit(pack, selected_action))
    row["selected_action"] = selected_action
    row["d68a_arm"] = arm
    row["effective_correct"] = row["selected_action_effective_correct"]
    row["diagnostic_margin"] = float(row.get("top1_top2_margin", 0.0))
    row["diagnostic_confidence"] = float(row.get("confidence", 0.0))
    return row


def record_row(row, outputs, sample_counts, path):
    outputs.append(row)
    key = (row["arm"], row["support_regime"])
    if sample_counts[key] < ROW_SAMPLE_PER_ARM_REGIME:
        append_jsonl(path, row)
        sample_counts[key] += 1


def summarize_audit_rows(rows):
    base = d68.summarize_rows(rows)
    names = [
        "reported_unnecessary_counter_request",
        "causal_unnecessary_counter_support",
        "no_cost_unnecessary_counter_request",
        "harmful_unnecessary_counter",
        "reported_missed_counter_request",
        "concrete_selected_counter_missed",
        "selected_concrete_counter_fixes",
        "selected_external_fixes",
        "selected_repair_fixes",
        "wrong_concrete_counter",
        "right_counter_family_wrong_concrete_action",
        "weak_top1_top2_path_failure",
        "lost_external_path_possible",
        "selected_external_request_unavailable",
        "decide_effective_correct",
        "selected_action_effective_correct",
        "any_internal_counter_action_fixes",
        "any_external_action_fixes",
    ]
    for name in names:
        base[f"{name}_rate"] = d51.mean([1.0 if row.get(name) else 0.0 for row in rows])
    base["support_over_cheapest_effective_mean"] = d51.mean([row["support_over_cheapest_effective"] for row in rows])
    base["selected_support_used_mean"] = d51.mean([row["selected_support_used"] for row in rows])
    base["selected_counter_support_used_mean"] = d51.mean([row["selected_counter_support_used"] for row in rows])
    base["selected_external_test_used_mean"] = d51.mean([row["selected_external_test_used"] for row in rows])
    base["diagnostic_margin_mean"] = d51.mean([row["diagnostic_margin"] for row in rows])
    base["diagnostic_confidence_mean"] = d51.mean([row["diagnostic_confidence"] for row in rows])
    base["effective_accuracy"] = d51.mean([1.0 if row["effective_correct"] else 0.0 for row in rows])
    return base


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_core = defaultdict(list)
    by_arm_regime = defaultdict(list)
    actions = defaultdict(Counter)
    for row in outputs:
        arm = row["arm"]
        by_arm[arm].append(row)
        by_arm_regime[(arm, row["support_regime"])].append(row)
        actions[arm][row["selected_action"]] += 1
        if row["support_regime"] in CORE_REGIMES:
            by_arm_core[arm].append(row)
    return {
        "by_arm": {arm: summarize_audit_rows(by_arm[arm]) for arm in AUDIT_ARMS if arm in by_arm},
        "by_arm_core": {arm: summarize_audit_rows(by_arm_core[arm]) for arm in AUDIT_ARMS if arm in by_arm_core},
        "by_arm_and_regime": {
            arm: {regime: summarize_audit_rows(by_arm_regime[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_arm_regime}
            for arm in AUDIT_ARMS
        },
        "action_distribution": {arm: dict(actions[arm]) for arm in AUDIT_ARMS},
    }


def write_partial(out, split, outputs, completed, started):
    recent = outputs[-min(len(outputs), 8000):]
    partial = summarize_outputs(recent)
    write_json(out / f"partial_{split}_d68a_metrics_snapshot.json", {"completed_outputs": completed, "recent_metrics": partial})
    append_progress(out, "d68a_eval_progress", started, {"split": split, "completed_outputs": completed})


def evaluate_split(rows, bundle, policy_controllers, learned_triage, out, split, started, heartbeat_sec, repo_root, row_output_path):
    rust_features, aggregation_report = d68.run_blocking_with_heartbeat(
        out,
        "d68a_rust_aggregation_bridge_wait",
        split,
        started,
        heartbeat_sec,
        d65.run_rust_aggregation_bridge,
        out,
        repo_root,
        rows,
        bundle,
        split,
        started,
        heartbeat_sec,
    )
    items = build_audit_items(rows, bundle, rust_features, out, started, heartbeat_sec, split)
    packs = [item["pack"] for item in items]
    policy_actions, policy_report = d68.run_blocking_with_heartbeat(
        out,
        "d68a_rust_policy_bridge_wait",
        split,
        started,
        heartbeat_sec,
        d59.run_rust_multi_bridge,
        out,
        repo_root,
        policy_controllers,
        packs,
        split,
        "d68a_policy_eval",
        started,
    )

    outputs = []
    sample_counts = Counter()
    last = 0.0
    for idx, item in enumerate(items):
        arm = item["arm"]
        pack = item["pack"]
        policy, selected_action, gate_features, basis, used_truth = select_audit_policy(arm, pack, learned_triage)
        action_record = policy_actions[policy][idx]
        # The Rust controller record is still used as the execution trace; the
        # selected action is the semantic action under audit.
        if action_record["action"] != selected_action:
            action_record = copy.deepcopy(action_record)
            action_record["action"] = selected_action
        row = make_output_row(pack, arm, policy, selected_action, action_record, gate_features, basis, used_truth)
        record_row(row, outputs, sample_counts, row_output_path)
        now = time.time()
        if now - last >= heartbeat_sec or len(outputs) == len(items):
            last = now
            write_partial(out, split, outputs, len(outputs), started)
    return outputs, {"aggregation": aggregation_report, "controller": policy_report}


def artifact_completeness_report(d68_root):
    required = [
        "aggregate_metrics.json",
        "decision.json",
        "summary.json",
        "support_cost_frontier_report.json",
        "row_outputs_test.jsonl",
        "row_outputs_ood.jsonl",
        "trained_threshold_triage_report.json",
    ]
    present = {name: (d68_root / name).exists() for name in required}
    row_fields = {}
    for name in ["row_outputs_test.jsonl", "row_outputs_ood.jsonl"]:
        path = d68_root / name
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                line = fh.readline()
            row_fields[name] = sorted(json.loads(line).keys()) if line else []
    alternatives_in_rows = all("actions" in fields or "action_alternatives" in fields for fields in row_fields.values())
    return {
        "d68_root": str(d68_root),
        "required_present": present,
        "row_outputs_are_sampled": True,
        "row_fields": row_fields,
        "action_alternatives_present_in_row_outputs": alternatives_in_rows,
        "deterministic_rebuild_required": not alternatives_in_rows,
        "reason": "D68 row outputs are sampled diagnostics and do not include full concrete action alternatives.",
    }


def d68_upstream_manifest(d68_root):
    return {
        "artifact_root": str(d68_root),
        "decision": load_json(d68_root / "decision.json"),
        "summary": load_json(d68_root / "summary.json"),
        "support_cost_frontier": load_json(d68_root / "support_cost_frontier_report.json"),
        "trained_threshold_triage_report": load_json(d68_root / "trained_threshold_triage_report.json"),
    }


def rebuild_parity_report(d68_root, metrics):
    upstream_frontier = load_json(d68_root / "support_cost_frontier_report.json", {})
    mapping = {
        "D67_BEST_REPLAY": "D67_BEST_REPLAY",
        "D68_TRAINED_THRESHOLD_REPLAY": "TRAINED_THRESHOLD_TRIAGE_GATE",
        "ALWAYS_COUNTER_CONTROL": "ALWAYS_COUNTER_CONTROL",
        "NEVER_COUNTER_CONTROL": "NEVER_COUNTER_CONTROL",
        "RANDOM_COUNTER_CONTROL": "RANDOM_COUNTER_CONTROL",
    }
    parity = {}
    for rebuilt, upstream in mapping.items():
        src = upstream_frontier.get(upstream, {})
        got = metrics["by_arm_core"].get(rebuilt, {})
        parity[rebuilt] = {
            "upstream_arm": upstream,
            "upstream_exact": src.get("exact"),
            "rebuilt_exact": got.get("exact_joint_accuracy"),
            "exact_delta": None if src.get("exact") is None else got.get("exact_joint_accuracy", 0.0) - src.get("exact", 0.0),
            "upstream_support": src.get("support"),
            "rebuilt_support": got.get("average_total_support_used"),
            "support_delta": None if src.get("support") is None else got.get("average_total_support_used", 0.0) - src.get("support", 0.0),
        }
    return {
        "parity_scope": "core test regimes only",
        "parity": parity,
        "max_abs_exact_delta": max(abs(item["exact_delta"] or 0.0) for item in parity.values()) if parity else None,
        "max_abs_support_delta": max(abs(item["support_delta"] or 0.0) for item in parity.values()) if parity else None,
    }


def classify_d68_harm(outputs):
    by_arm_row = defaultdict(dict)
    for row in outputs:
        by_arm_row[row["row_id"]][row["arm"]] = row
    counts = Counter()
    examples = []
    total_loss = 0
    for row_id, rows in by_arm_row.items():
        d67 = rows.get("D67_BEST_REPLAY")
        d68r = rows.get("D68_TRAINED_THRESHOLD_REPLAY")
        if not d67 or not d68r:
            continue
        if d67["effective_correct"] and not d68r["effective_correct"]:
            total_loss += 1
            if d68r["selected_action"] == "REQUEST_COUNTER_TOP1_TOP2" and d68r["request_joint_counter_effective"]:
                reason = "selected_top1_top2_failed_but_joint_counter_would_fix"
            elif d67["selected_action"] == "REQUEST_JOINT_COUNTER" and d68r["request_joint_counter_effective"]:
                reason = "lost_joint_counter_path"
            elif d67["selected_action"] == "REQUEST_EXTERNAL_TEST" and d68r["request_external_test_effective"]:
                reason = "external_test_removed_but_external_would_fix"
            elif d68r["wrong_concrete_counter"]:
                reason = "selected_counter_wrong_but_other_internal_counter_would_fix"
            elif d68r["selected_action"] == "DECIDE" and d68r["any_internal_counter_action_fixes"]:
                reason = "counter_removed_to_decide"
            elif not d68r["any_repair_action_fixes"]:
                reason = "no_available_repair"
            else:
                reason = "unknown"
            counts[reason] += 1
            if len(examples) < 40:
                examples.append({
                    "row_id": row_id,
                    "support_regime": d68r["support_regime"],
                    "d67_action": d67["selected_action"],
                    "d68_action": d68r["selected_action"],
                    "classification": reason,
                    "top1_effective": d68r["request_top1_top2_effective"],
                    "joint_effective": d68r["request_joint_counter_effective"],
                    "external_effective": d68r["request_external_test_effective"],
                })
    return {
        "d68_loss_rows_vs_d67": total_loss,
        "classification_counts": dict(counts),
        "classification_rates": {key: value / max(1, total_loss) for key, value in counts.items()},
        "examples": examples,
    }


def make_reports(out, d68_root, aggregate, decision):
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    d67 = core.get("D67_BEST_REPLAY", {})
    d68r = core.get("D68_TRAINED_THRESHOLD_REPLAY", {})

    definition_report = {
        "current_d68_definitions": {
            "unnecessary_internal": "selected internal counter and DECIDE already correct",
            "unnecessary_external": "selected external test and DECIDE already correct",
            "internal_needed": "DECIDE wrong and any internal counter action fixes",
            "counter_tp": "selected any internal counter and internal_needed",
            "weakness": "does not prove selected concrete counter action fixed the row; also mixes no-cost external requests with actual extra support cost",
        },
        "d68a_added_definitions": {
            "selected_concrete_counter_fixes": "selected internal counter action fixes when DECIDE does not",
            "wrong_concrete_counter": "selected internal counter wrong but another internal counter would fix",
            "causal_unnecessary_counter_support": "DECIDE already effective, selected repair effective, and selected support exceeds cheapest effective support",
            "no_cost_unnecessary_counter_request": "DECIDE already effective and a repair action was requested, but selected support does not exceed cheapest effective support",
            "harmful_unnecessary_counter": "DECIDE already effective but selected repair action is not effective",
        },
    }

    concrete_report = {
        arm: {
            key: values.get(key)
            for key in [
                "reported_unnecessary_counter_request_rate",
                "causal_unnecessary_counter_support_rate",
                "no_cost_unnecessary_counter_request_rate",
                "harmful_unnecessary_counter_rate",
                "reported_missed_counter_request_rate",
                "concrete_selected_counter_missed_rate",
                "selected_concrete_counter_fixes_rate",
                "wrong_concrete_counter_rate",
                "weak_top1_top2_path_failure_rate",
                "selected_external_request_unavailable_rate",
                "support_over_cheapest_effective_mean",
                "exact_joint_accuracy",
                "effective_accuracy",
                "average_total_support_used",
            ]
        }
        for arm, values in core.items()
    }

    support_accounting = {
        arm: {
            "original_support_mean": values.get("average_total_support_used", 0.0)
            - values.get("selected_counter_support_used_mean", 0.0)
            - values.get("selected_external_test_used_mean", 0.0),
            "selected_counter_support_used_mean": values.get("selected_counter_support_used_mean", 0.0),
            "selected_external_test_used_mean": values.get("selected_external_test_used_mean", 0.0),
            "selected_support_used_mean": values.get("selected_support_used_mean", 0.0),
            "support_over_cheapest_effective_mean": values.get("support_over_cheapest_effective_mean", 0.0),
        }
        for arm, values in core.items()
    }

    causal_removal = {
        "d68_trained_threshold_replay": d68r,
        "counter_removal_replay": core.get("COUNTER_REMOVAL_REPLAY", {}),
        "delta_exact": core.get("COUNTER_REMOVAL_REPLAY", {}).get("exact_joint_accuracy", 0.0)
        - d68r.get("exact_joint_accuracy", 0.0),
        "delta_support": core.get("COUNTER_REMOVAL_REPLAY", {}).get("average_total_support_used", 0.0)
        - d68r.get("average_total_support_used", 0.0),
    }

    margin_rows = []
    for split_outputs in [aggregate.get("test_outputs_sample", []), aggregate.get("ood_outputs_sample", [])]:
        for row in split_outputs:
            if row.get("decide_effective_correct") and row.get("selected_action") in COUNTER_ACTIONS:
                margin_rows.append(row)
    diagnostic_margin = {
        "note": "sampled row-output proxy only; aggregate reports carry the full support accounting",
        "sample_rows": len(margin_rows),
        "mean_margin": d51.mean([row.get("diagnostic_margin", 0.0) for row in margin_rows]),
        "mean_confidence": d51.mean([row.get("diagnostic_confidence", 0.0) for row in margin_rows]),
    }

    reports = {
        "d68_upstream_manifest.json": d68_upstream_manifest(d68_root),
        "artifact_completeness_and_rebuild_parity_report.json": aggregate["artifact_completeness_and_rebuild_parity_report"],
        "counter_metric_definition_report.json": definition_report,
        "concrete_counter_action_report.json": concrete_report,
        "causal_counter_removal_report.json": causal_removal,
        "cheapest_correct_support_report.json": {
            arm: {
                "support_over_cheapest_effective_mean": values.get("support_over_cheapest_effective_mean"),
                "selected_support_used_mean": values.get("selected_support_used_mean"),
                "effective_accuracy": values.get("effective_accuracy"),
            }
            for arm, values in core.items()
        },
        "d68_harm_classification_report.json": aggregate["d68_harm_classification_report"],
        "diagnostic_margin_stability_report.json": diagnostic_margin,
        "support_accounting_report.json": support_accounting,
        "regime_blind_audit_report.json": {
            "fair_arms": FAIR_ARMS,
            "reference_only_arms": REFERENCE_ONLY_ARMS,
            "fair_arms_using_truth_label": [],
            "fair_arms_using_regime_label": [],
            "truth_hidden_from_fair_arms": True,
            "oracle_labels_reference_only": True,
        },
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {
            "task": TASK,
            "decision": decision["decision"],
            "verdict": decision["verdict"],
            "next": decision["next"],
            "artifact_root": str(out),
            "failed_jobs": aggregate["failed_jobs"],
            "fallback_rows": aggregate["fallback_rows"],
        },
    }
    for name, value in reports.items():
        write_json(out / name, value)

    rows = [
        "# D68A Counter-Support Metric Semantics Audit Report",
        "",
        f"Decision: `{decision['decision']}`",
        f"Verdict: `{decision['verdict']}`",
        f"Next: `{decision['next']}`",
        "",
        "## Core Comparison",
        "",
        "| arm | exact | effective | support | reported unnecessary | causal unnecessary | no-cost request | concrete missed | wrong concrete |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ["D67_BEST_REPLAY", "D68_TRAINED_THRESHOLD_REPLAY", "COUNTER_REMOVAL_REPLAY", "ALWAYS_COUNTER_CONTROL", "NEVER_COUNTER_CONTROL"]:
        values = core.get(arm, {})
        rows.append(
            f"| {arm} | {values.get('exact_joint_accuracy', 0.0):.6f} | "
            f"{values.get('effective_accuracy', 0.0):.6f} | "
            f"{values.get('average_total_support_used', 0.0):.4f} | "
            f"{values.get('reported_unnecessary_counter_request_rate', 0.0):.6f} | "
            f"{values.get('causal_unnecessary_counter_support_rate', 0.0):.6f} | "
            f"{values.get('no_cost_unnecessary_counter_request_rate', 0.0):.6f} | "
            f"{values.get('concrete_selected_counter_missed_rate', 0.0):.6f} | "
            f"{values.get('wrong_concrete_counter_rate', 0.0):.6f} |"
        )
    rows.extend(
        [
            "",
            "## Interpretation",
            "",
            "D68A separates request-count metrics from causal support-cost metrics and concrete selected-action correctness.",
            "A D68 metric is not treated as confirmed unless reported unnecessary/missed rates agree with concrete action and support-cost evidence.",
            "",
            "## Boundary",
            "",
            BOUNDARY,
        ]
    )
    (out / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def load_upstream_trained_triage(d68_root):
    aggregate = load_json(d68_root / "aggregate_metrics.json", {})
    if aggregate.get("trained_threshold_triage"):
        return aggregate["trained_threshold_triage"]
    report = load_json(d68_root / "trained_threshold_triage_report.json", {})
    selected = report.get("selected", {})
    return selected.get("config", {"risk_threshold": 0.64, "joint_threshold": 0.66})


def read_jsonl_sample(path, limit=120):
    out = []
    path = Path(path)
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx >= limit:
                break
            out.append(json.loads(line))
    return out


def make_decision(aggregate):
    core = aggregate["test_metrics"]["by_arm_core"]
    d67 = core.get("D67_BEST_REPLAY", {})
    d68r = core.get("D68_TRAINED_THRESHOLD_REPLAY", {})
    failed = aggregate["failed_jobs"]
    if failed:
        decision = "counter_support_metric_pipeline_not_confirmed"
        verdict = "D68A_FAILED_JOBS"
        next_step = "D68A_REPAIR"
    elif aggregate["artifact_completeness_and_rebuild_parity_report"]["artifact_completeness"]["deterministic_rebuild_required"] and not aggregate["artifact_completeness_and_rebuild_parity_report"].get("rebuild_performed"):
        decision = "d68a_artifact_insufficient_for_metric_audit"
        verdict = "D68A_ARTIFACT_INSUFFICIENT_FOR_METRIC_AUDIT"
        next_step = "D68A_REBUILD_INSTRUMENTATION"
    else:
        reported_gap = abs(
            d67.get("reported_unnecessary_counter_request_rate", 0.0)
            - d67.get("causal_unnecessary_counter_support_rate", 0.0)
        )
        concrete_missed = d68r.get("concrete_selected_counter_missed_rate", 0.0)
        wrong_concrete = d68r.get("wrong_concrete_counter_rate", 0.0)
        no_cost = d67.get("no_cost_unnecessary_counter_request_rate", 0.0)
        if reported_gap > 0.10 or concrete_missed > 0.001 or wrong_concrete > 0.001:
            decision = "counter_support_metric_pipeline_not_confirmed"
            verdict = "D68A_COUNTER_SUPPORT_METRIC_PIPELINE_NOT_CONFIRMED"
            next_step = "D68R_COUNTER_METRIC_REPAIR"
        elif no_cost > 0.10:
            decision = "counter_metrics_valid_but_need_rename"
            verdict = "D68A_COUNTER_METRICS_VALID_BUT_NEED_RENAME"
            next_step = "D68M_METRIC_RENAME_AND_REPAIR"
        else:
            decision = "counter_support_metrics_confirmed"
            verdict = "D68A_COUNTER_SUPPORT_METRICS_CONFIRMED"
            next_step = "D68B_COUNTER_SUPPORT_TRIAGE_REPAIR_CAUSAL_METRICS"
    return {
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "reason": {
            "d67_reported_unnecessary": d67.get("reported_unnecessary_counter_request_rate"),
            "d67_causal_unnecessary": d67.get("causal_unnecessary_counter_support_rate"),
            "d67_no_cost_unnecessary_request": d67.get("no_cost_unnecessary_counter_request_rate"),
            "d68_reported_unnecessary": d68r.get("reported_unnecessary_counter_request_rate"),
            "d68_causal_unnecessary": d68r.get("causal_unnecessary_counter_support_rate"),
            "d68_concrete_selected_counter_missed": d68r.get("concrete_selected_counter_missed_rate"),
            "d68_wrong_concrete_counter": d68r.get("wrong_concrete_counter_rate"),
            "d68_weak_top1_top2_failure": d68r.get("weak_top1_top2_path_failure_rate"),
            "d68_exact": d68r.get("exact_joint_accuracy"),
            "d67_exact": d67.get("exact_joint_accuracy"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--d68-root", default="target/pilot_wave/d68_counter_support_triage_repair/smoke")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--test-rows-per-seed", type=int, default=0)
    parser.add_argument("--ood-rows-per-seed", type=int, default=0)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--scale-mode", default="artifact-rebuild-audit")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    d68_root = Path(args.d68_root)
    failed_jobs = []

    d68_queue = load_json(d68_root / "queue.json", {})
    d68_args = d68_queue.get("args", {})
    seeds = parse_seeds(args.seeds or d68_args.get("seeds", "12701,12702,12703,12704,12705"))
    test_rows_per_seed = args.test_rows_per_seed or int(d68_args.get("test_rows_per_seed", 240))
    ood_rows_per_seed = args.ood_rows_per_seed or int(d68_args.get("ood_rows_per_seed", 240))

    write_json(out / "queue.json", {"task": TASK, "args": vars(args), "seeds": seeds, "boundary": BOUNDARY})
    append_progress(out, "started", started, {"seeds": seeds, "scale_mode": args.scale_mode})
    write_json(out / "compute_probe.json", {"cpu_count": os.cpu_count(), "workers": d51.worker_count_from_arg(args.workers), "cpu_target": args.cpu_target})

    completeness = artifact_completeness_report(d68_root)
    write_json(out / "partial_artifact_completeness_report.json", completeness)
    append_progress(out, "artifact_completeness_checked", started, {"deterministic_rebuild_required": completeness["deterministic_rebuild_required"]})

    learned_triage = load_upstream_trained_triage(d68_root)
    policy_controllers, _ = d68.load_policy_modules(repo_root)
    bundle = d55.d49_bundle()

    test_rows = d51.make_rows_with_progress(seeds, test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)

    try:
        test_outputs, test_rust = evaluate_split(
            test_rows,
            bundle,
            policy_controllers,
            learned_triage,
            out,
            "test",
            started,
            args.heartbeat_sec,
            repo_root,
            out / "row_outputs_test.jsonl",
        )
        ood_outputs, ood_rust = evaluate_split(
            ood_rows,
            bundle,
            policy_controllers,
            learned_triage,
            out,
            "ood",
            started,
            args.heartbeat_sec,
            repo_root,
            out / "row_outputs_ood.jsonl",
        )
    except Exception as exc:
        failed_jobs.append({"stage": "evaluate", "error": str(exc)})
        write_json(out / "error.json", failed_jobs[-1])
        raise

    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    rust_invocation = {"test": test_rust, "ood": ood_rust}
    rust_aggregation_rows = sum(data.get("aggregation", {}).get("rows_returned", 0) for data in rust_invocation.values())
    rust_controller_rows = sum(data.get("controller", {}).get("rows_requested", 0) for data in rust_invocation.values())
    fallback_rows = sum(1 for row in test_outputs + ood_outputs if row.get("python_fallback_used"))

    parity = rebuild_parity_report(d68_root, test_metrics)
    artifact_and_parity = {
        "artifact_completeness": completeness,
        "rebuild_performed": True,
        "rebuild_reason": completeness["reason"],
        "rebuild_parity": parity,
    }

    aggregate = {
        "task": TASK,
        "boundary": BOUNDARY,
        "scale_mode": args.scale_mode,
        "d68_root": str(d68_root),
        "learned_triage_replayed": learned_triage,
        "artifact_completeness_and_rebuild_parity_report": artifact_and_parity,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "d68_harm_classification_report": classify_d68_harm(test_outputs),
        "rust_invocation_report": rust_invocation,
        "rust_path_invoked": rust_aggregation_rows > 0 and rust_controller_rows > 0,
        "rust_aggregation_rows": rust_aggregation_rows,
        "rust_controller_rows": rust_controller_rows,
        "fallback_rows": fallback_rows,
        "failed_jobs": failed_jobs,
        "test_outputs_sample": read_jsonl_sample(out / "row_outputs_test.jsonl"),
        "ood_outputs_sample": read_jsonl_sample(out / "row_outputs_ood.jsonl"),
    }
    decision = make_decision(aggregate)
    aggregate["decision"] = decision
    make_reports(out, d68_root, aggregate, decision)
    append_progress(out, "completed", started, {"decision": decision["decision"], "failed_jobs": failed_jobs})


if __name__ == "__main__":
    main()
