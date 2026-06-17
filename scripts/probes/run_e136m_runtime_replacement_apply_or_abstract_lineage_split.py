#!/usr/bin/env python3
"""E136M runtime replacement apply or abstract lineage split.

This probe consumes the E136L runtime-canary evidence and materializes the
first runtime-facing replacement overlay:

- the 16 direct canary-passed rows become an active overlay manifest;
- their legacy triggers are disabled only inside the overlay and retained for
  rollback;
- the 11 tightened challenger rows remain on a challenger/OOD queue;
- the 7 abstract kernels remain on an abstract-lineage split queue.

Boundary: this is a runtime-facing overlay/apply manifest, not destructive
deletion of the committed operator library.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


ARTIFACT_CONTRACT = "E136M_RUNTIME_REPLACEMENT_APPLY_OR_ABSTRACT_LINEAGE_SPLIT"
DECISION_CONFIRMED = "e136m_runtime_replacement_overlay_and_lineage_split_confirmed"
DECISION_REJECTED = "e136m_runtime_replacement_overlay_and_lineage_split_rejected"
NEXT = "E136N_CHALLENGER_OOD_RUNTIME_REPLACEMENT_GAUNTLET"

DEFAULT_E136L = Path("docs/research/artifact_samples/e136l_runtime_replacement_canary_and_tightened_challenger_confirm")
DEFAULT_OUT = Path("target/e136m_runtime_replacement_apply_or_abstract_lineage_split")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136m_runtime_replacement_apply_or_abstract_lineage_split")

ARTIFACT_FILES = (
    "run_manifest.json",
    "runtime_replacement_overlay.json",
    "replacement_apply_queue.json",
    "challenger_ood_queue.json",
    "abstract_lineage_split.json",
    "rollback_plan.json",
    "effective_runtime_delta.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_e136l(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = load_json(input_dir / "summary.json")
    if summary.get("decision") != "e136l_runtime_replacement_canary_and_tightened_challenger_confirmed":
        raise ValueError("E136L input is not confirmed")
    if not summary.get("pass_gate"):
        raise ValueError("E136L input pass gate is false")
    canary = load_json(input_dir / "canary_runtime_ledger.json")["rows"]
    challenger = load_json(input_dir / "challenger_ood_ledger.json")["rows"]
    abstract = load_json(input_dir / "abstract_lineage_hold_ledger.json")["rows"]
    rollback = load_json(input_dir / "rollback_audit.json")["rows"]
    return summary, canary, challenger, abstract, rollback


def overlay_action(row: dict[str, Any]) -> str:
    if row["apply_action"] == "DIRECT_CANARY_REPLACE_WITH_VERIFIED_PRUNED_VARIANT":
        return "activate_selected_variant_replace_legacy_trigger"
    if row["apply_action"] == "DIRECT_CANARY_KEEP_CURRENT_WITH_LIGHT_PRUNE":
        return "activate_selected_variant_light_prune_legacy_trigger"
    return "hold"


def build_runtime_overlay(canary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlay = []
    for index, row in enumerate(canary_rows, 1):
        overlay.append({
            "overlay_slot": index,
            "operator_id": row["operator_id"],
            "display_name": row["display_name"],
            "source": row["source"],
            "legacy_runtime_state": "disabled_in_overlay_retained_for_rollback",
            "selected_runtime_state": "active_in_overlay",
            "overlay_action": overlay_action(row),
            "apply_action": row["apply_action"],
            "selected_variant_id": row["selected_variant_id"],
            "selected_variant_type": row["selected_variant_type"],
            "current_activation": row["current_activation"],
            "overlay_activation": row["selected_activation"],
            "removed_activation": row["shadow_pruned_activation"],
            "removed_activation_ratio": row["shadow_prune_ratio"],
            "sample_legacy_activation": row["sample_replay"]["legacy_activation"],
            "sample_overlay_activation": row["sample_replay"]["canary_activation"],
            "sample_removed_activation": row["sample_replay"]["canary_removed_activation"],
            "strict_recall_miss": row["strict_recall_miss"] + row["sample_replay"]["strict_recall_miss"],
            "wrong_scope_proxy": row["wrong_scope_proxy"] + row["sample_replay"]["wrong_scope_proxy"],
            "hard_negative": row["hard_negative"] + row["sample_replay"]["hard_negative"],
            "unsupported_answer": row["unsupported_answer"] + row["sample_replay"]["unsupported_answer"],
            "direct_flow_write": row["direct_flow_write"] + row["sample_replay"]["direct_flow_write"],
            "rollback_snapshot_required": True,
            "destructive_delete": False,
        })
    return overlay


def build_replacement_queue(overlay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "operator_id": row["operator_id"],
            "selected_variant_id": row["selected_variant_id"],
            "queue_state": "runtime_overlay_active",
            "overlay_action": row["overlay_action"],
            "destructive_delete": False,
            "rollback_required": True,
            "removed_activation": row["removed_activation"],
        }
        for row in overlay_rows
    ]


def build_challenger_queue(challenger_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "operator_id": row["operator_id"],
            "display_name": row["display_name"],
            "selected_variant_id": row["selected_variant_id"],
            "selected_variant_type": row["selected_variant_type"],
            "queue_state": "challenger_ood_required_before_overlay",
            "runtime_overlay_active": False,
            "destructive_delete": False,
            "current_activation": row["current_activation"],
            "candidate_selected_activation": row["selected_activation"],
            "candidate_pruned_activation": row["shadow_pruned_activation"],
            "candidate_prune_ratio": row["shadow_prune_ratio"],
            "hold_reason": "tightened trigger prunes materially; needs OOD/challenger proof before runtime overlay",
        }
        for row in challenger_rows
    ]


def build_abstract_split(abstract_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "operator_id": row["operator_id"],
            "display_name": row["display_name"],
            "selected_variant_id": row["selected_variant_id"],
            "selected_variant_type": row["selected_variant_type"],
            "split_state": "abstract_lineage_hold",
            "runtime_overlay_active": False,
            "destructive_delete": False,
            "current_activation": row["current_activation"],
            "lineage_action": "preserve_kernel_and_trace_activation_family_before_rename_or_prune",
            "hold_reason": "abstract-but-useful kernel has value but insufficient human/operator lineage clarity for runtime replacement",
        }
        for row in abstract_rows
    ]


def build_rollback_plan(overlay_rows: list[dict[str, Any]], rollback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rollback_by_id = {row["operator_id"]: row for row in rollback_rows}
    plan = []
    for row in overlay_rows:
        source = rollback_by_id.get(row["operator_id"], {})
        plan.append({
            "operator_id": row["operator_id"],
            "selected_variant_id": row["selected_variant_id"],
            "rollback_action": "reactivate_legacy_trigger_disable_selected_overlay",
            "rollback_snapshot_required": True,
            "pre_apply_snapshot_required": bool(source.get("pre_apply_snapshot_required", True)),
            "rollback_triggered_in_e136l": bool(source.get("rollback_triggered", False)),
            "rollback_trigger": "any strict recall miss, wrong-scope proxy, hard negative, unsupported answer, direct Flow write, or output regression",
            "legacy_activation": row["current_activation"],
            "overlay_activation": row["overlay_activation"],
            "removed_activation": row["removed_activation"],
        })
    return plan


def build_effective_delta(
    overlay_rows: list[dict[str, Any]],
    challenger_rows: list[dict[str, Any]],
    abstract_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    direct_current = sum(row["current_activation"] for row in overlay_rows)
    direct_overlay = sum(row["overlay_activation"] for row in overlay_rows)
    direct_removed = sum(row["removed_activation"] for row in overlay_rows)
    challenger_current = sum(row["current_activation"] for row in challenger_rows)
    challenger_candidate_removed = sum(row["shadow_pruned_activation"] for row in challenger_rows)
    abstract_current = sum(row["current_activation"] for row in abstract_rows)
    effective_current = direct_current + challenger_current + abstract_current
    effective_overlay = direct_overlay + challenger_current + abstract_current
    return {
        "effective_runtime_current_activation_total": effective_current,
        "effective_runtime_overlay_activation_total": effective_overlay,
        "effective_runtime_removed_activation_total": direct_removed,
        "effective_runtime_removed_activation_ratio": round(direct_removed / effective_current, 6) if effective_current else 0.0,
        "direct_current_activation_total": direct_current,
        "direct_overlay_activation_total": direct_overlay,
        "direct_removed_activation_total": direct_removed,
        "challenger_current_activation_total": challenger_current,
        "challenger_candidate_removed_activation_total": challenger_candidate_removed,
        "challenger_candidate_removed_not_applied": challenger_candidate_removed,
        "abstract_current_activation_total": abstract_current,
    }


def summarize(
    e136l_summary: dict[str, Any],
    overlay_rows: list[dict[str, Any]],
    replacement_queue: list[dict[str, Any]],
    challenger_queue: list[dict[str, Any]],
    abstract_split: list[dict[str, Any]],
    rollback_plan: list[dict[str, Any]],
    effective_delta: dict[str, Any],
) -> dict[str, Any]:
    action_counts = Counter(row["apply_action"] for row in overlay_rows)
    overlay_failures = [
        row for row in overlay_rows
        if row["strict_recall_miss"]
        or row["wrong_scope_proxy"]
        or row["hard_negative"]
        or row["unsupported_answer"]
        or row["direct_flow_write"]
    ]
    rollback_trigger_count = sum(1 for row in rollback_plan if row["rollback_triggered_in_e136l"])
    destructive_delete_count = sum(1 for row in overlay_rows if row["destructive_delete"])
    pass_gate = (
        e136l_summary.get("decision") == "e136l_runtime_replacement_canary_and_tightened_challenger_confirmed"
        and bool(e136l_summary.get("pass_gate"))
        and len(overlay_rows) == int(e136l_summary.get("direct_canary_pass_count", -1))
        and len(replacement_queue) == 16
        and len(challenger_queue) == int(e136l_summary.get("challenger_hold_count", -1))
        and len(abstract_split) == int(e136l_summary.get("abstract_lineage_hold_count", -1))
        and len(rollback_plan) == int(e136l_summary.get("rollback_manifest_count", -1))
        and not overlay_failures
        and rollback_trigger_count == 0
        and destructive_delete_count == 0
        and effective_delta["effective_runtime_removed_activation_total"] == int(e136l_summary.get("direct_canary_removed_activation_total", -1))
    )
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": DECISION_CONFIRMED if pass_gate else DECISION_REJECTED,
        "next": NEXT,
        "pass_gate": pass_gate,
        "operator_count": 34,
        "runtime_overlay_active_count": len(overlay_rows),
        "runtime_overlay_apply_count": len(replacement_queue),
        "verified_replacement_apply_count": action_counts.get("DIRECT_CANARY_REPLACE_WITH_VERIFIED_PRUNED_VARIANT", 0),
        "light_prune_overlay_apply_count": action_counts.get("DIRECT_CANARY_KEEP_CURRENT_WITH_LIGHT_PRUNE", 0),
        "legacy_trigger_disabled_in_overlay_count": len(overlay_rows),
        "legacy_trigger_retained_for_rollback_count": len(rollback_plan),
        "rollback_snapshot_count": len(rollback_plan),
        "rollback_trigger_count": rollback_trigger_count,
        "challenger_ood_queue_count": len(challenger_queue),
        "challenger_runtime_overlay_active_count": 0,
        "abstract_lineage_split_count": len(abstract_split),
        "abstract_runtime_overlay_active_count": 0,
        "production_destructive_delete_count": destructive_delete_count,
        "runtime_mutation_allowed_now_count": len(replacement_queue),
        "current_activation_total": e136l_summary["current_activation_total"],
        "shadow_selected_activation_total": e136l_summary["selected_activation_total"],
        "shadow_pruned_activation_total": e136l_summary["shadow_pruned_activation_total"],
        "runtime_overlay_activation_total": effective_delta["effective_runtime_overlay_activation_total"],
        "runtime_overlay_removed_activation_total": effective_delta["effective_runtime_removed_activation_total"],
        "runtime_overlay_removed_activation_ratio": effective_delta["effective_runtime_removed_activation_ratio"],
        "direct_canary_removed_activation_total": effective_delta["direct_removed_activation_total"],
        "challenger_candidate_removed_not_applied": effective_delta["challenger_candidate_removed_not_applied"],
        "strict_recall_miss_total": sum(row["strict_recall_miss"] for row in overlay_rows),
        "wrong_scope_proxy_total": sum(row["wrong_scope_proxy"] for row in overlay_rows),
        "hard_negative_total": sum(row["hard_negative"] for row in overlay_rows),
        "unsupported_answer_total": sum(row["unsupported_answer"] for row in overlay_rows),
        "direct_flow_write_total": sum(row["direct_flow_write"] for row in overlay_rows),
        "overlay_failure_count": len(overlay_failures),
        "effective_runtime_delta": effective_delta,
        "action_counts": dict(sorted(action_counts.items())),
    }


def write_report(out: Path, summary: dict[str, Any]) -> None:
    report = f"""# E136M Runtime Replacement Apply Or Abstract Lineage Split

```text
decision = {summary['decision']}
next     = {summary['next']}
```

E136M materializes the first runtime-facing replacement overlay from E136L.
The direct canary-passed rows become active selected variants in the overlay.
Legacy triggers are disabled only inside the overlay and retained for rollback.
Challenger and abstract rows remain held.

## Result

```text
operator_count = {summary['operator_count']}
runtime_overlay_active_count = {summary['runtime_overlay_active_count']}
runtime_overlay_apply_count = {summary['runtime_overlay_apply_count']}
verified_replacement_apply_count = {summary['verified_replacement_apply_count']}
light_prune_overlay_apply_count = {summary['light_prune_overlay_apply_count']}
legacy_trigger_disabled_in_overlay_count = {summary['legacy_trigger_disabled_in_overlay_count']}
legacy_trigger_retained_for_rollback_count = {summary['legacy_trigger_retained_for_rollback_count']}

challenger_ood_queue_count = {summary['challenger_ood_queue_count']}
challenger_runtime_overlay_active_count = {summary['challenger_runtime_overlay_active_count']}
abstract_lineage_split_count = {summary['abstract_lineage_split_count']}
abstract_runtime_overlay_active_count = {summary['abstract_runtime_overlay_active_count']}

rollback_snapshot_count = {summary['rollback_snapshot_count']}
rollback_trigger_count = {summary['rollback_trigger_count']}
production_destructive_delete_count = {summary['production_destructive_delete_count']}
runtime_mutation_allowed_now_count = {summary['runtime_mutation_allowed_now_count']}

current_activation_total = {summary['current_activation_total']}
shadow_selected_activation_total = {summary['shadow_selected_activation_total']}
shadow_pruned_activation_total = {summary['shadow_pruned_activation_total']}

runtime_overlay_activation_total = {summary['runtime_overlay_activation_total']}
runtime_overlay_removed_activation_total = {summary['runtime_overlay_removed_activation_total']}
runtime_overlay_removed_activation_ratio = {summary['runtime_overlay_removed_activation_ratio']}
direct_canary_removed_activation_total = {summary['direct_canary_removed_activation_total']}
challenger_candidate_removed_not_applied = {summary['challenger_candidate_removed_not_applied']}

strict_recall_miss_total = {summary['strict_recall_miss_total']}
wrong_scope_proxy_total = {summary['wrong_scope_proxy_total']}
hard_negative_total = {summary['hard_negative_total']}
unsupported_answer_total = {summary['unsupported_answer_total']}
direct_flow_write_total = {summary['direct_flow_write_total']}
```

## Boundary

This is a runtime-facing overlay/apply manifest. It does not destructively
delete legacy operators. Challenger/OOD rows and abstract lineage rows remain
held.
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

    input_dir = Path(args.e136l_artifact)
    e136l_summary, canary_rows, challenger_rows, abstract_rows, rollback_rows = load_e136l(input_dir)
    overlay = build_runtime_overlay(canary_rows)
    replacement_queue = build_replacement_queue(overlay)
    challenger_queue = build_challenger_queue(challenger_rows)
    abstract_split = build_abstract_split(abstract_rows)
    rollback_plan = build_rollback_plan(overlay, rollback_rows)
    effective_delta = build_effective_delta(overlay, challenger_rows, abstract_rows)
    summary = summarize(
        e136l_summary,
        overlay,
        replacement_queue,
        challenger_queue,
        abstract_split,
        rollback_plan,
        effective_delta,
    )

    checker_failures = []
    if not summary["pass_gate"]:
        checker_failures.append("e136m_pass_gate_failed")
    if summary["production_destructive_delete_count"] != 0:
        checker_failures.append("destructive_delete_present")
    if summary["rollback_trigger_count"] != 0:
        checker_failures.append("rollback_triggered_before_apply")
    if summary["challenger_runtime_overlay_active_count"] != 0:
        checker_failures.append("challenger_overlay_activated_without_ood")
    if summary["abstract_runtime_overlay_active_count"] != 0:
        checker_failures.append("abstract_overlay_activated_without_lineage")

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "runtime-facing overlay only; no destructive runtime replacement or prune",
        "e136l_artifact": str(input_dir),
    })
    write_json(out / "runtime_replacement_overlay.json", {"rows": overlay})
    write_json(out / "replacement_apply_queue.json", {"rows": replacement_queue})
    write_json(out / "challenger_ood_queue.json", {"rows": challenger_queue})
    write_json(out / "abstract_lineage_split.json", {"rows": abstract_split})
    write_json(out / "rollback_plan.json", {"rows": rollback_plan})
    write_json(out / "effective_runtime_delta.json", effective_delta)
    write_json(out / "aggregate_metrics.json", {
        key: value
        for key, value in summary.items()
        if key not in {"decision", "next", "pass_gate", "effective_runtime_delta", "action_counts"}
    })
    write_json(out / "decision.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": NEXT,
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
    parser.add_argument("--e136l-artifact", default=str(DEFAULT_E136L))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": summary["next"],
        "pass_gate": summary["pass_gate"],
        "runtime_overlay_active_count": summary["runtime_overlay_active_count"],
        "verified_replacement_apply_count": summary["verified_replacement_apply_count"],
        "light_prune_overlay_apply_count": summary["light_prune_overlay_apply_count"],
        "challenger_ood_queue_count": summary["challenger_ood_queue_count"],
        "abstract_lineage_split_count": summary["abstract_lineage_split_count"],
        "runtime_overlay_removed_activation_total": summary["runtime_overlay_removed_activation_total"],
        "challenger_candidate_removed_not_applied": summary["challenger_candidate_removed_not_applied"],
        "rollback_trigger_count": summary["rollback_trigger_count"],
        "production_destructive_delete_count": summary["production_destructive_delete_count"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
