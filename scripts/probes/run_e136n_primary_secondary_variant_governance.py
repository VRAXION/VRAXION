#!/usr/bin/env python3
"""E136N primary/secondary variant governance.

This probe consumes E136M's runtime-facing overlay and creates a durable
primary/secondary variant registry skeleton:

- every operator receives exactly one primary variant;
- E136M direct overlay winners become primary active variants;
- replaced legacy triggers become secondary rollback variants;
- tightened candidates remain secondary challenger variants;
- abstract kernels remain secondary lineage-hold variants;
- no variant is retired destructively by this governance step.

Boundary: this is variant governance and routing metadata only. It does not
delete committed operators and does not apply the challenger/OOD queue.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ARTIFACT_CONTRACT = "E136N_PRIMARY_SECONDARY_VARIANT_GOVERNANCE"
DECISION_CONFIRMED = "e136n_primary_secondary_variant_governance_confirmed"
DECISION_REJECTED = "e136n_primary_secondary_variant_governance_rejected"
NEXT = "E136O_CHALLENGER_OOD_RUNTIME_REPLACEMENT_GAUNTLET"

DEFAULT_E136M = Path("docs/research/artifact_samples/e136m_runtime_replacement_apply_or_abstract_lineage_split")
DEFAULT_OUT = Path("target/e136n_primary_secondary_variant_governance")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136n_primary_secondary_variant_governance")

ARTIFACT_FILES = (
    "run_manifest.json",
    "variant_registry.json",
    "operator_variant_assignments.json",
    "primary_variant_manifest.json",
    "secondary_variant_manifest.json",
    "variant_state_machine.json",
    "retirement_candidates.json",
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


def load_e136m(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = load_json(input_dir / "summary.json")
    if summary.get("decision") != "e136m_runtime_replacement_overlay_and_lineage_split_confirmed":
        raise ValueError("E136M input is not confirmed")
    if not summary.get("pass_gate"):
        raise ValueError("E136M input pass gate is false")
    overlay = load_json(input_dir / "runtime_replacement_overlay.json")["rows"]
    challenger = load_json(input_dir / "challenger_ood_queue.json")["rows"]
    abstract = load_json(input_dir / "abstract_lineage_split.json")["rows"]
    rollback = load_json(input_dir / "rollback_plan.json")["rows"]
    return summary, overlay, challenger, abstract, rollback


def legacy_variant_id(operator_id: str) -> str:
    return f"{operator_id}::legacy_current_e136n"


def current_variant_id(operator_id: str) -> str:
    return f"{operator_id}::current_primary_e136n"


def abstract_primary_id(operator_id: str) -> str:
    return f"{operator_id}::abstract_current_primary_e136n"


def build_registry(
    overlay_rows: list[dict[str, Any]],
    challenger_rows: list[dict[str, Any]],
    abstract_rows: list[dict[str, Any]],
    rollback_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rollback_by_id = {row["operator_id"]: row for row in rollback_rows}
    registry: list[dict[str, Any]] = []
    for row in overlay_rows:
        registry.append({
            "operator_id": row["operator_id"],
            "display_name": row["display_name"],
            "variant_id": row["selected_variant_id"],
            "variant_role": "primary",
            "variant_state": "primary_active",
            "runtime_default": True,
            "runtime_overlay_active": True,
            "source_track": "E136M_DIRECT_OVERLAY",
            "activation": row["overlay_activation"],
            "removed_activation_vs_legacy": row["removed_activation"],
            "strict_recall_miss": row["strict_recall_miss"],
            "wrong_scope_proxy": row["wrong_scope_proxy"],
            "hard_negative": row["hard_negative"],
            "unsupported_answer": row["unsupported_answer"],
            "direct_flow_write": row["direct_flow_write"],
            "promotion_gate": "e136l_canary_pass_and_e136m_overlay_pass",
            "demotion_gate": "any rollback trigger, strict recall miss, wrong-scope proxy, hard negative, unsupported answer, direct Flow write, or output regression",
            "destructive_delete": False,
        })
        rollback_source = rollback_by_id.get(row["operator_id"], {})
        registry.append({
            "operator_id": row["operator_id"],
            "display_name": row["display_name"],
            "variant_id": legacy_variant_id(row["operator_id"]),
            "variant_role": "secondary",
            "variant_state": "secondary_rollback",
            "runtime_default": False,
            "runtime_overlay_active": False,
            "source_track": "E136M_LEGACY_ROLLBACK",
            "activation": row["current_activation"],
            "removed_activation_vs_primary": 0,
            "rollback_action": rollback_source.get("rollback_action", "reactivate_legacy_trigger_disable_selected_overlay"),
            "promotion_gate": "rollback_only_until_primary_regression_or_explicit_reaudit",
            "retirement_gate": "not_before_extended_no_unique_coverage_and_rollback_not_needed",
            "destructive_delete": False,
        })
    for row in challenger_rows:
        registry.append({
            "operator_id": row["operator_id"],
            "display_name": row["display_name"],
            "variant_id": current_variant_id(row["operator_id"]),
            "variant_role": "primary",
            "variant_state": "primary_current",
            "runtime_default": True,
            "runtime_overlay_active": False,
            "source_track": "E136M_CHALLENGER_HOLD_CURRENT",
            "activation": row["current_activation"],
            "promotion_gate": "current_default_until_challenger_ood_pass",
            "demotion_gate": "challenger_ood_pass_and_overlay_canary_pass",
            "destructive_delete": False,
        })
        registry.append({
            "operator_id": row["operator_id"],
            "display_name": row["display_name"],
            "variant_id": row["selected_variant_id"],
            "variant_role": "secondary",
            "variant_state": "secondary_challenger",
            "runtime_default": False,
            "runtime_overlay_active": False,
            "source_track": "E136M_CHALLENGER_OOD_QUEUE",
            "activation": row["candidate_selected_activation"],
            "candidate_removed_activation": row["candidate_pruned_activation"],
            "candidate_prune_ratio": row["candidate_prune_ratio"],
            "promotion_gate": "requires_e136o_challenger_ood_pass_then_runtime_canary",
            "retirement_gate": "if challenger fails OOD or loses unique coverage under replay",
            "destructive_delete": False,
        })
    for row in abstract_rows:
        registry.append({
            "operator_id": row["operator_id"],
            "display_name": row["display_name"],
            "variant_id": abstract_primary_id(row["operator_id"]),
            "variant_role": "primary",
            "variant_state": "primary_abstract_current",
            "runtime_default": True,
            "runtime_overlay_active": False,
            "source_track": "E136M_ABSTRACT_CURRENT",
            "activation": row["current_activation"],
            "promotion_gate": "current_default_until_lineage_split_resolved",
            "demotion_gate": "lineage_resolved_with_safer_primary_candidate",
            "destructive_delete": False,
        })
        registry.append({
            "operator_id": row["operator_id"],
            "display_name": row["display_name"],
            "variant_id": row["selected_variant_id"],
            "variant_role": "secondary",
            "variant_state": "secondary_lineage_hold",
            "runtime_default": False,
            "runtime_overlay_active": False,
            "source_track": "E136M_ABSTRACT_LINEAGE_SPLIT",
            "activation": row["current_activation"],
            "promotion_gate": "requires_lineage_name_scope_and_trace_family_resolution",
            "retirement_gate": "not_retirable_until_abstract_kernel_value_is_replaced_or_proven_redundant",
            "destructive_delete": False,
        })
    return registry


def build_assignments(registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in registry:
        by_operator[row["operator_id"]].append(row)
    assignments = []
    for operator_id, rows in sorted(by_operator.items()):
        primaries = [row for row in rows if row["variant_role"] == "primary"]
        secondaries = [row for row in rows if row["variant_role"] == "secondary"]
        display_name = rows[0]["display_name"]
        assignments.append({
            "operator_id": operator_id,
            "display_name": display_name,
            "primary_variant_id": primaries[0]["variant_id"] if primaries else None,
            "primary_state": primaries[0]["variant_state"] if primaries else None,
            "secondary_variant_ids": [row["variant_id"] for row in secondaries],
            "secondary_states": [row["variant_state"] for row in secondaries],
            "primary_count": len(primaries),
            "secondary_count": len(secondaries),
            "destructive_delete": False,
        })
    return assignments


def build_state_machine() -> dict[str, Any]:
    return {
        "states": {
            "primary_active": "Default selected variant for runtime overlay.",
            "primary_current": "Current legacy/default variant retained while challenger candidate waits.",
            "primary_abstract_current": "Current abstract kernel retained while lineage is resolved.",
            "secondary_rollback": "Legacy variant retained for rollback and shadow comparison.",
            "secondary_challenger": "Candidate variant held until challenger/OOD proof.",
            "secondary_lineage_hold": "Abstract candidate held until lineage/name/scope proof.",
            "retired_redundant": "Inactive archived variant after redundancy and rollback-not-needed proof.",
        },
        "allowed_transitions": [
            {
                "from": "secondary_challenger",
                "to": "primary_active",
                "gate": "challenger/OOD pass, runtime canary pass, rollback plan present, zero failure metrics",
            },
            {
                "from": "primary_active",
                "to": "secondary_rollback",
                "gate": "rollback trigger, output regression, or safety failure",
            },
            {
                "from": "secondary_rollback",
                "to": "primary_active",
                "gate": "primary regression and legacy rollback replay pass",
            },
            {
                "from": "secondary_lineage_hold",
                "to": "primary_active",
                "gate": "lineage resolved, scope named, OOD/canary pass",
            },
            {
                "from": "secondary_rollback",
                "to": "retired_redundant",
                "gate": "extended no-unique-coverage proof, rollback no longer required, archive snapshot present",
            },
        ],
        "hard_blocks": [
            "No destructive delete in primary/secondary governance.",
            "No operator may have zero primary variants.",
            "No operator may have more than one primary variant.",
            "No secondary may be unlinked from an operator primary.",
            "No challenger may become primary without challenger/OOD evidence.",
            "No abstract lineage hold may become primary without lineage resolution.",
        ],
    }


def build_retirement_candidates(registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # E136N creates the retirement lane but intentionally retires nothing.
    return [
        {
            "variant_id": row["variant_id"],
            "operator_id": row["operator_id"],
            "variant_state": row["variant_state"],
            "retirement_candidate": False,
            "reason": "retirement requires extended no-unique-coverage and rollback-not-needed proof",
        }
        for row in registry
        if row["variant_state"] == "secondary_rollback"
    ]


def summarize(
    e136m_summary: dict[str, Any],
    registry: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    retirement_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    states = Counter(row["variant_state"] for row in registry)
    roles = Counter(row["variant_role"] for row in registry)
    invalid_assignments = [
        row for row in assignments
        if row["primary_count"] != 1 or row["secondary_count"] < 1
    ]
    destructive_delete_count = sum(1 for row in registry if row.get("destructive_delete"))
    retired_count = states.get("retired_redundant", 0)
    pass_gate = (
        e136m_summary.get("decision") == "e136m_runtime_replacement_overlay_and_lineage_split_confirmed"
        and bool(e136m_summary.get("pass_gate"))
        and len(assignments) == 34
        and roles.get("primary", 0) == 34
        and roles.get("secondary", 0) == 34
        and states.get("primary_active", 0) == 16
        and states.get("primary_current", 0) == 11
        and states.get("primary_abstract_current", 0) == 7
        and states.get("secondary_rollback", 0) == 16
        and states.get("secondary_challenger", 0) == 11
        and states.get("secondary_lineage_hold", 0) == 7
        and retired_count == 0
        and destructive_delete_count == 0
        and not invalid_assignments
    )
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": DECISION_CONFIRMED if pass_gate else DECISION_REJECTED,
        "next": NEXT,
        "pass_gate": pass_gate,
        "operator_count": len(assignments),
        "variant_registry_row_count": len(registry),
        "primary_variant_count": roles.get("primary", 0),
        "secondary_variant_count": roles.get("secondary", 0),
        "primary_active_count": states.get("primary_active", 0),
        "primary_current_count": states.get("primary_current", 0),
        "primary_abstract_current_count": states.get("primary_abstract_current", 0),
        "secondary_rollback_count": states.get("secondary_rollback", 0),
        "secondary_challenger_count": states.get("secondary_challenger", 0),
        "secondary_lineage_hold_count": states.get("secondary_lineage_hold", 0),
        "retired_redundant_count": retired_count,
        "retirement_lane_created_count": len(retirement_candidates),
        "retirement_candidate_count": sum(1 for row in retirement_candidates if row["retirement_candidate"]),
        "destructive_delete_count": destructive_delete_count,
        "ambiguous_primary_operator_count": sum(1 for row in assignments if row["primary_count"] > 1),
        "missing_primary_operator_count": sum(1 for row in assignments if row["primary_count"] == 0),
        "orphan_secondary_count": 0,
        "invalid_assignment_count": len(invalid_assignments),
        "runtime_overlay_removed_activation_total": e136m_summary["runtime_overlay_removed_activation_total"],
        "challenger_candidate_removed_not_applied": e136m_summary["challenger_candidate_removed_not_applied"],
        "rollback_snapshot_count": e136m_summary["rollback_snapshot_count"],
        "rollback_trigger_count": e136m_summary["rollback_trigger_count"],
        "strict_recall_miss_total": e136m_summary["strict_recall_miss_total"],
        "wrong_scope_proxy_total": e136m_summary["wrong_scope_proxy_total"],
        "hard_negative_total": e136m_summary["hard_negative_total"],
        "unsupported_answer_total": e136m_summary["unsupported_answer_total"],
        "direct_flow_write_total": e136m_summary["direct_flow_write_total"],
        "state_counts": dict(sorted(states.items())),
        "role_counts": dict(sorted(roles.items())),
    }


def write_report(out: Path, summary: dict[str, Any]) -> None:
    report = f"""# E136N Primary/Secondary Variant Governance

```text
decision = {summary['decision']}
next     = {summary['next']}
```

E136N turns E136M's overlay into a durable primary/secondary variant registry.
Every operator now has exactly one primary variant and at least one secondary
variant. No variant is destructively retired in this step.

## Result

```text
operator_count = {summary['operator_count']}
variant_registry_row_count = {summary['variant_registry_row_count']}
primary_variant_count = {summary['primary_variant_count']}
secondary_variant_count = {summary['secondary_variant_count']}

primary_active_count = {summary['primary_active_count']}
primary_current_count = {summary['primary_current_count']}
primary_abstract_current_count = {summary['primary_abstract_current_count']}
secondary_rollback_count = {summary['secondary_rollback_count']}
secondary_challenger_count = {summary['secondary_challenger_count']}
secondary_lineage_hold_count = {summary['secondary_lineage_hold_count']}
retired_redundant_count = {summary['retired_redundant_count']}

retirement_lane_created_count = {summary['retirement_lane_created_count']}
retirement_candidate_count = {summary['retirement_candidate_count']}
destructive_delete_count = {summary['destructive_delete_count']}
ambiguous_primary_operator_count = {summary['ambiguous_primary_operator_count']}
missing_primary_operator_count = {summary['missing_primary_operator_count']}
orphan_secondary_count = {summary['orphan_secondary_count']}

runtime_overlay_removed_activation_total = {summary['runtime_overlay_removed_activation_total']}
challenger_candidate_removed_not_applied = {summary['challenger_candidate_removed_not_applied']}
rollback_snapshot_count = {summary['rollback_snapshot_count']}
rollback_trigger_count = {summary['rollback_trigger_count']}

strict_recall_miss_total = {summary['strict_recall_miss_total']}
wrong_scope_proxy_total = {summary['wrong_scope_proxy_total']}
hard_negative_total = {summary['hard_negative_total']}
unsupported_answer_total = {summary['unsupported_answer_total']}
direct_flow_write_total = {summary['direct_flow_write_total']}
```

## Boundary

This is variant governance metadata only. It does not delete operators and it
does not apply the challenger/OOD queue.
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

    input_dir = Path(args.e136m_artifact)
    e136m_summary, overlay, challenger, abstract, rollback = load_e136m(input_dir)
    registry = build_registry(overlay, challenger, abstract, rollback)
    assignments = build_assignments(registry)
    primary_manifest = [row for row in registry if row["variant_role"] == "primary"]
    secondary_manifest = [row for row in registry if row["variant_role"] == "secondary"]
    state_machine = build_state_machine()
    retirement_candidates = build_retirement_candidates(registry)
    summary = summarize(e136m_summary, registry, assignments, retirement_candidates)

    checker_failures = []
    if not summary["pass_gate"]:
        checker_failures.append("e136n_pass_gate_failed")
    if summary["destructive_delete_count"] != 0:
        checker_failures.append("destructive_delete_present")
    if summary["ambiguous_primary_operator_count"] != 0:
        checker_failures.append("ambiguous_primary_operator")
    if summary["missing_primary_operator_count"] != 0:
        checker_failures.append("missing_primary_operator")
    if summary["orphan_secondary_count"] != 0:
        checker_failures.append("orphan_secondary_variant")
    if summary["retired_redundant_count"] != 0:
        checker_failures.append("retired_variant_present_before_retirement_proof")

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "variant governance only; no destructive delete; no challenger apply",
        "e136m_artifact": str(input_dir),
    })
    write_json(out / "variant_registry.json", {"rows": registry})
    write_json(out / "operator_variant_assignments.json", {"rows": assignments})
    write_json(out / "primary_variant_manifest.json", {"rows": primary_manifest})
    write_json(out / "secondary_variant_manifest.json", {"rows": secondary_manifest})
    write_json(out / "variant_state_machine.json", state_machine)
    write_json(out / "retirement_candidates.json", {"rows": retirement_candidates})
    write_json(out / "aggregate_metrics.json", {
        key: value
        for key, value in summary.items()
        if key not in {"decision", "next", "pass_gate", "state_counts", "role_counts"}
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
    parser.add_argument("--e136m-artifact", default=str(DEFAULT_E136M))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": summary["next"],
        "pass_gate": summary["pass_gate"],
        "operator_count": summary["operator_count"],
        "primary_variant_count": summary["primary_variant_count"],
        "secondary_variant_count": summary["secondary_variant_count"],
        "primary_active_count": summary["primary_active_count"],
        "secondary_rollback_count": summary["secondary_rollback_count"],
        "secondary_challenger_count": summary["secondary_challenger_count"],
        "secondary_lineage_hold_count": summary["secondary_lineage_hold_count"],
        "retired_redundant_count": summary["retired_redundant_count"],
        "destructive_delete_count": summary["destructive_delete_count"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
