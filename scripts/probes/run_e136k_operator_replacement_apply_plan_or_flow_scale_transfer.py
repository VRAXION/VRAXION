#!/usr/bin/env python3
"""E136K operator replacement apply plan or Flow-scale transfer.

This probe consumes the E136I supersession ledger and the E136J shadow-apply
evidence. It does not mutate the runtime operator library. It produces an
explicit apply plan: direct runtime canary candidates, tightened challenger/OOD
required candidates, abstract lineage holds, rollback requirements, and the
recommended next track.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


ARTIFACT_CONTRACT = "E136K_OPERATOR_REPLACEMENT_APPLY_PLAN_OR_FLOW_SCALE_TRANSFER"
DECISION_CONFIRMED = "e136k_operator_replacement_apply_plan_confirmed"
DECISION_REJECTED = "e136k_operator_replacement_apply_plan_rejected"
NEXT = "E136L_RUNTIME_REPLACEMENT_CANARY_AND_TIGHTENED_CHALLENGER_CONFIRM"

DEFAULT_E136I = Path("docs/research/artifact_samples/e136i_operator_supersession_and_output_ledger_planning")
DEFAULT_E136J = Path("docs/research/artifact_samples/e136j_shadow_variant_apply_and_residual_prune_confirm")
DEFAULT_OUT = Path("target/e136k_operator_replacement_apply_plan_or_flow_scale_transfer")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136k_operator_replacement_apply_plan_or_flow_scale_transfer")

ARTIFACT_FILES = (
    "run_manifest.json",
    "apply_plan.json",
    "rollback_manifest.json",
    "flow_scale_transfer_decision.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_sample(out: Path, sample_out: Path | None) -> None:
    if not sample_out:
        return
    if sample_out.exists():
        shutil.rmtree(sample_out)
    sample_out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        src = out / name
        if src.exists():
            shutil.copy2(src, sample_out / name)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(e136i_dir: Path, e136j_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    e136i_summary = load_json(e136i_dir / "summary.json")
    e136j_summary = load_json(e136j_dir / "summary.json")
    e136i_rows = load_json(e136i_dir / "supersession_ledger.json")["rows"]
    e136j_rows = load_json(e136j_dir / "residual_prune_ledger.json")["rows"]
    return e136i_summary, e136j_summary, e136i_rows, e136j_rows


def choose_action(irow: dict[str, Any], jrow: dict[str, Any]) -> tuple[str, str]:
    tier = irow["supersession_tier"]
    if tier == "T0_KEEP_CURRENT_WITH_LIGHT_PRUNE":
        return (
            "DIRECT_CANARY_KEEP_CURRENT_WITH_LIGHT_PRUNE",
            "low-risk verified operator; apply selected trigger as canary with rollback snapshot",
        )
    if tier == "T1_VERIFIED_PRUNED_REPLACEMENT":
        return (
            "DIRECT_CANARY_REPLACE_WITH_VERIFIED_PRUNED_VARIANT",
            "verified pruned replacement survived long shadow replay with zero recall loss",
        )
    if tier == "T2_TIGHTENED_TRIGGER_REPLACEMENT":
        return (
            "CHALLENGER_OOD_REQUIRED_BEFORE_RUNTIME_REPLACEMENT",
            "tightened trigger prunes materially; require challenger/OOD canary before runtime replacement",
        )
    if tier == "T3_ABSTRACT_KERNEL_LINEAGE_REQUIRED":
        return (
            "RETAIN_ABSTRACT_KERNEL_NO_RUNTIME_REPLACEMENT",
            "useful abstract kernel; preserve lineage and inspect/narrow before replacement",
        )
    return ("HOLD_FOR_MANUAL_REVIEW", f"unrecognized tier {tier}")


def build_plan(e136i_rows: list[dict[str, Any]], e136j_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    j_by_id = {row["operator_id"]: row for row in e136j_rows}
    plan: list[dict[str, Any]] = []
    for irow in e136i_rows:
        operator_id = irow["operator_id"]
        if operator_id not in j_by_id:
            raise ValueError(f"operator missing from E136J residual ledger: {operator_id}")
        jrow = j_by_id[operator_id]
        action, reason = choose_action(irow, jrow)
        has_failures = any(
            int(jrow.get(field, 0)) != 0
            for field in ("strict_recall_miss", "wrong_scope_proxy")
        ) or any(
            int(irow.get(field, 0)) != 0
            for field in ("hard_negative", "wrong_scope_call", "unsupported_answer", "direct_flow_write")
        )
        direct_canary = action.startswith("DIRECT_CANARY") and not has_failures
        challenger_required = action.startswith("CHALLENGER")
        abstract_lineage = action.startswith("RETAIN_ABSTRACT")
        plan.append({
            "operator_id": operator_id,
            "display_name": irow["display_name"],
            "source": irow["source"],
            "selected_variant_id": irow["selected_variant_id"],
            "selected_variant_type": irow["selected_variant_type"],
            "supersession_tier": irow["supersession_tier"],
            "supersession_action": irow["supersession_action"],
            "readiness": irow["readiness"],
            "apply_action": action,
            "apply_reason": reason,
            "direct_canary_ready": direct_canary,
            "challenger_ood_required": challenger_required,
            "abstract_lineage_required": abstract_lineage,
            "runtime_mutation_allowed_now": False,
            "rollback_required": direct_canary,
            "current_activation": jrow["current_activation"],
            "selected_activation": jrow["selected_activation"],
            "shadow_pruned_activation": jrow["shadow_pruned_activation"],
            "shadow_prune_ratio": jrow["shadow_prune_ratio"],
            "strict_recall_miss": jrow["strict_recall_miss"],
            "wrong_scope_proxy": jrow["wrong_scope_proxy"],
            "hard_negative": irow["hard_negative"],
            "unsupported_answer": irow["unsupported_answer"],
            "direct_flow_write": irow["direct_flow_write"],
            "accepted_mutations": irow["accepted_mutations"],
            "mutation_attempts": irow["mutation_attempts"],
            "label_alignment_score": irow["label_alignment_score"],
            "kernel_value_score": irow["kernel_value_score"],
        })
    return plan


def summarize(plan: list[dict[str, Any]], e136i_summary: dict[str, Any], e136j_summary: dict[str, Any]) -> dict[str, Any]:
    direct_rows = [row for row in plan if row["direct_canary_ready"]]
    challenger_rows = [row for row in plan if row["challenger_ood_required"]]
    abstract_rows = [row for row in plan if row["abstract_lineage_required"]]
    failure_rows = [
        row for row in plan
        if row["strict_recall_miss"] or row["wrong_scope_proxy"] or row["hard_negative"] or row["unsupported_answer"] or row["direct_flow_write"]
    ]
    tier_counts = Counter(row["supersession_tier"] for row in plan)
    action_counts = Counter(row["apply_action"] for row in plan)
    current_total = sum(row["current_activation"] for row in plan)
    selected_total = sum(row["selected_activation"] for row in plan)
    pruned_total = sum(row["shadow_pruned_activation"] for row in plan)
    direct_pruned_total = sum(row["shadow_pruned_activation"] for row in direct_rows)
    challenger_pruned_total = sum(row["shadow_pruned_activation"] for row in challenger_rows)
    abstract_current_total = sum(row["current_activation"] for row in abstract_rows)
    mutation_attempt_total = sum(row["mutation_attempts"] for row in plan)
    accepted_mutation_total = sum(row["accepted_mutations"] for row in plan)
    pass_gate = (
        e136i_summary.get("decision") == "e136i_operator_supersession_and_output_ledger_confirmed"
        and e136j_summary.get("decision") == "e136j_shadow_variant_apply_and_residual_prune_confirmed"
        and bool(e136j_summary.get("pass_gate"))
        and not failure_rows
        and len(direct_rows) == int(e136j_summary.get("direct_runtime_candidate_count", -1))
        and len(challenger_rows) == int(e136j_summary.get("tightened_challenger_required_count", -1))
        and len(abstract_rows) == int(e136j_summary.get("abstract_lineage_required_count", -1))
    )
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": DECISION_CONFIRMED if pass_gate else DECISION_REJECTED,
        "next": NEXT,
        "pass_gate": pass_gate,
        "operator_count": len(plan),
        "direct_canary_ready_count": len(direct_rows),
        "challenger_ood_required_count": len(challenger_rows),
        "abstract_lineage_required_count": len(abstract_rows),
        "runtime_mutation_allowed_now_count": 0,
        "destructive_apply_count": 0,
        "rollback_manifest_count": len(direct_rows),
        "failure_row_count": len(failure_rows),
        "current_activation_total": current_total,
        "selected_activation_total": selected_total,
        "shadow_pruned_activation_total": pruned_total,
        "direct_canary_shadow_pruned_activation_total": direct_pruned_total,
        "challenger_shadow_pruned_activation_total": challenger_pruned_total,
        "abstract_current_activation_total": abstract_current_total,
        "shadow_prune_ratio": round(pruned_total / current_total, 6) if current_total else 0.0,
        "direct_canary_shadow_prune_ratio": round(
            direct_pruned_total / sum(row["current_activation"] for row in direct_rows), 6
        ) if direct_rows else 0.0,
        "challenger_shadow_prune_ratio": round(
            challenger_pruned_total / sum(row["current_activation"] for row in challenger_rows), 6
        ) if challenger_rows else 0.0,
        "strict_recall_miss_total": sum(row["strict_recall_miss"] for row in plan),
        "wrong_scope_proxy_total": sum(row["wrong_scope_proxy"] for row in plan),
        "hard_negative_total": sum(row["hard_negative"] for row in plan),
        "unsupported_answer_total": sum(row["unsupported_answer"] for row in plan),
        "direct_flow_write_total": sum(row["direct_flow_write"] for row in plan),
        "mutation_attempt_total": mutation_attempt_total,
        "accepted_mutation_total": accepted_mutation_total,
        "mutation_accept_rate": round(accepted_mutation_total / mutation_attempt_total, 6) if mutation_attempt_total else 0.0,
        "tier_counts": dict(sorted(tier_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "recommended_track": "operator_replacement_apply_plan",
        "flow_scale_transfer_decision": "defer_until_replacement_canary_plan_lands",
    }


def build_rollback_manifest(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in plan:
        if not row["direct_canary_ready"]:
            continue
        rows.append({
            "operator_id": row["operator_id"],
            "selected_variant_id": row["selected_variant_id"],
            "rollback_required": True,
            "rollback_trigger": "any strict recall miss, wrong-scope proxy, hard negative, unsupported answer, direct Flow write, or output regression",
            "canary_scope": "shadow-to-runtime canary only; not destructive library prune",
            "pre_apply_snapshot_required": True,
        })
    return rows


def build_flow_scale_decision(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommended_track": summary["recommended_track"],
        "flow_scale_transfer_decision": summary["flow_scale_transfer_decision"],
        "reason": (
            "E136J produced enough zero-failure evidence to plan direct canaries "
            "for the verified replacement subset. Flow-scale transfer remains useful, "
            "but it should not replace the immediate operator replacement canary step."
        ),
        "operator_replacement_ready_count": summary["direct_canary_ready_count"],
        "challenger_ood_required_count": summary["challenger_ood_required_count"],
        "abstract_lineage_required_count": summary["abstract_lineage_required_count"],
    }


def write_report(out: Path, summary: dict[str, Any]) -> None:
    report = f"""# E136K Operator Replacement Apply Plan Or Flow Scale Transfer

```text
decision = {summary['decision']}
next     = {summary['next']}
```

## Metrics

```text
operator_count = {summary['operator_count']}
direct_canary_ready_count = {summary['direct_canary_ready_count']}
challenger_ood_required_count = {summary['challenger_ood_required_count']}
abstract_lineage_required_count = {summary['abstract_lineage_required_count']}
runtime_mutation_allowed_now_count = {summary['runtime_mutation_allowed_now_count']}
destructive_apply_count = {summary['destructive_apply_count']}
rollback_manifest_count = {summary['rollback_manifest_count']}

current_activation_total = {summary['current_activation_total']}
selected_activation_total = {summary['selected_activation_total']}
shadow_pruned_activation_total = {summary['shadow_pruned_activation_total']}
shadow_prune_ratio = {summary['shadow_prune_ratio']}

strict_recall_miss_total = {summary['strict_recall_miss_total']}
wrong_scope_proxy_total = {summary['wrong_scope_proxy_total']}
hard_negative_total = {summary['hard_negative_total']}
unsupported_answer_total = {summary['unsupported_answer_total']}
direct_flow_write_total = {summary['direct_flow_write_total']}

accepted_mutation_total = {summary['accepted_mutation_total']}
mutation_attempt_total = {summary['mutation_attempt_total']}
mutation_accept_rate = {summary['mutation_accept_rate']}
```

## Boundary

This is an apply plan and canary manifest only. It does not destructively replace
or prune runtime operators.
"""
    (out / "report.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_out = None if args.sample_out == "" else Path(args.sample_out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        path = out / name
        if path.exists():
            path.unlink()

    e136i_dir = Path(args.e136i_artifact)
    e136j_dir = Path(args.e136j_artifact)
    e136i_summary, e136j_summary, e136i_rows, e136j_rows = load_inputs(e136i_dir, e136j_dir)
    plan = build_plan(e136i_rows, e136j_rows)
    summary = summarize(plan, e136i_summary, e136j_summary)
    rollback_manifest = build_rollback_manifest(plan)
    flow_scale_decision = build_flow_scale_decision(summary)

    checker_failures = []
    if not summary["pass_gate"]:
        checker_failures.append("e136k_pass_gate_failed")
    if summary["runtime_mutation_allowed_now_count"] != 0:
        checker_failures.append("runtime_mutation_allowed_in_plan")
    if summary["destructive_apply_count"] != 0:
        checker_failures.append("destructive_apply_present")
    if summary["failure_row_count"] != 0:
        checker_failures.append("failure_rows_present")

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "apply plan only; no destructive runtime replacement or prune",
        "e136i_artifact": str(e136i_dir),
        "e136j_artifact": str(e136j_dir),
    })
    write_json(out / "apply_plan.json", {"rows": plan})
    write_json(out / "rollback_manifest.json", {"rows": rollback_manifest})
    write_json(out / "flow_scale_transfer_decision.json", flow_scale_decision)
    write_json(out / "aggregate_metrics.json", {
        key: value
        for key, value in summary.items()
        if key not in {"decision", "next", "pass_gate", "tier_counts", "action_counts"}
    })
    write_json(out / "decision.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": summary["next"],
        "pass_gate": summary["pass_gate"],
    })
    write_json(out / "summary.json", summary)
    write_report(out, summary)
    write_json(out / "checker_summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": len(checker_failures),
        "failures": checker_failures,
    })
    copy_sample(out, sample_out)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e136i-artifact", default=str(DEFAULT_E136I))
    parser.add_argument("--e136j-artifact", default=str(DEFAULT_E136J))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": summary["next"],
        "pass_gate": summary["pass_gate"],
        "direct_canary_ready_count": summary["direct_canary_ready_count"],
        "challenger_ood_required_count": summary["challenger_ood_required_count"],
        "abstract_lineage_required_count": summary["abstract_lineage_required_count"],
        "runtime_mutation_allowed_now_count": summary["runtime_mutation_allowed_now_count"],
        "shadow_pruned_activation_total": summary["shadow_pruned_activation_total"],
        "strict_recall_miss_total": summary["strict_recall_miss_total"],
        "wrong_scope_proxy_total": summary["wrong_scope_proxy_total"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
