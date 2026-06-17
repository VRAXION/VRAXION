#!/usr/bin/env python3
"""E136I operator supersession and output ledger planning.

This probe consumes the E136H existing-operator refinement artifact and answers
the next governance question:

Which selected variants are ready to supersede their current operator trigger,
which require a challenger/OOD replay before runtime apply, and which are useful
abstract kernels that need lineage or naming work instead of destructive prune?

Boundary: planning evidence only. This script does not mutate the committed
operator library, write runtime state, or claim open-domain assistant behavior.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ARTIFACT_CONTRACT = "E136I_OPERATOR_SUPERSESSION_AND_OUTPUT_LEDGER_PLANNING"
DECISION_CONFIRMED = "e136i_operator_supersession_and_output_ledger_confirmed"
DECISION_REJECTED = "e136i_operator_supersession_and_output_ledger_rejected"
NEXT = "E136J_SHADOW_VARIANT_APPLY_AND_RESIDUAL_PRUNE_CONFIRM"

DEFAULT_E136H = Path("docs/research/artifact_samples/e136h_existing_operator_refinement_mutation_prune_night_cycle")
DEFAULT_OUT = Path("target/pilot_wave/e136i_operator_supersession_and_output_ledger_planning")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136i_operator_supersession_and_output_ledger_planning")

ARTIFACT_FILES = (
    "run_manifest.json",
    "input_e136h_report.json",
    "supersession_ledger.json",
    "output_impact_ledger.json",
    "mutation_transfer_ledger.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)


def now_ms() -> int:
    return int(time.time() * 1000)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_output_dir(out: Path) -> None:
    resolved = out.resolve()
    target_root = (REPO_ROOT / "target").resolve()
    try:
        resolved.relative_to(target_root)
    except ValueError as exc:
        raise ValueError(f"--out must resolve under {target_root}") from exc
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        path = out / name
        if path.exists():
            path.unlink()


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


def classify_operator(row: dict[str, Any]) -> dict[str, Any]:
    variant_type = str(row["selected_variant_type"])
    alignment = float(row["alignment"])
    prune_ratio = float(row["selected_prune_ratio"])
    kernel_value = float(row["kernel_value_score"])
    current_activation = int(row["current_activation"])
    selected_activation = int(row["selected_activation"])
    pruned_activation = int(row["pruned_activation"])

    if variant_type == "semantic_verified_pruned":
        if prune_ratio <= 0.02 and alignment >= 0.95:
            tier = "T0_KEEP_CURRENT_WITH_LIGHT_PRUNE"
            action = "apply_verified_light_prune_shadow"
            readiness = "ready_low_risk"
            next_gate = "shadow_apply_smoke"
        else:
            tier = "T1_VERIFIED_PRUNED_REPLACEMENT"
            action = "supersede_current_with_verified_pruned_variant"
            readiness = "ready_verified"
            next_gate = "shadow_apply_and_replay"
        replacement_ready = True
        direct_runtime_candidate = True
        lineage_required = False
        destructive_drop = False
    elif variant_type == "semantic_tightened_trigger":
        tier = "T2_TIGHTENED_TRIGGER_REPLACEMENT"
        action = "supersede_current_with_tightened_trigger_candidate"
        readiness = "ready_after_challenger"
        next_gate = "shadow_apply_challenger_and_ood_replay"
        replacement_ready = True
        direct_runtime_candidate = False
        lineage_required = False
        destructive_drop = False
    elif variant_type == "abstract_kernel_shadow":
        tier = "T3_ABSTRACT_KERNEL_LINEAGE_REQUIRED"
        action = "retain_as_abstract_kernel_shadow"
        readiness = "lineage_required_before_supersession"
        next_gate = "lineage_name_and_counterfactual_supersession"
        replacement_ready = False
        direct_runtime_candidate = False
        lineage_required = True
        destructive_drop = False
    else:
        tier = "T4_HOLD_FOR_MORE_EVIDENCE"
        action = "hold_for_more_evidence"
        readiness = "blocked_pending_evidence"
        next_gate = "more_seed_replay"
        replacement_ready = False
        direct_runtime_candidate = False
        lineage_required = False
        destructive_drop = False

    output_delta = selected_activation - current_activation
    if current_activation:
        output_risk_reduction_proxy = pruned_activation / current_activation
    else:
        output_risk_reduction_proxy = 0.0

    if pruned_activation == 0:
        supersession_pressure = "none"
    elif prune_ratio >= 0.35:
        supersession_pressure = "high"
    elif prune_ratio >= 0.15:
        supersession_pressure = "medium"
    else:
        supersession_pressure = "low"

    return {
        "operator_id": row["operator_id"],
        "display_name": row["display_name"],
        "source": row["source"],
        "selected_variant_id": row["selected_variant_id"],
        "selected_variant_type": variant_type,
        "label_status": row["label_status"],
        "supersession_tier": tier,
        "supersession_action": action,
        "readiness": readiness,
        "next_gate": next_gate,
        "replacement_ready": replacement_ready,
        "direct_runtime_candidate": direct_runtime_candidate,
        "lineage_required": lineage_required,
        "destructive_drop": destructive_drop,
        "current_activation": current_activation,
        "selected_activation": selected_activation,
        "pruned_activation": pruned_activation,
        "projected_output_activation_delta": output_delta,
        "selected_prune_ratio": round(prune_ratio, 6),
        "output_risk_reduction_proxy": round(output_risk_reduction_proxy, 6),
        "supersession_pressure": supersession_pressure,
        "label_alignment_score": round(alignment, 6),
        "kernel_value_score": round(kernel_value, 6),
        "accepted_mutations": int(row["accepted_mutations"]),
        "mutation_attempts": int(row["mutation_attempts"]),
        "hard_negative": int(row["hard_negative"]),
        "wrong_scope_call": int(row["wrong_scope_call"]),
        "unsupported_answer": int(row["unsupported_answer"]),
        "direct_flow_write": int(row["direct_flow_write"]),
    }


def write_report(out: Path, summary: dict[str, Any], ledger: list[dict[str, Any]]) -> None:
    by_tier = Counter(row["supersession_tier"] for row in ledger)
    by_pressure = Counter(row["supersession_pressure"] for row in ledger)
    lines = [
        "# E136I Operator Supersession And Output Ledger Planning",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"next     = {summary['next']}",
        "```",
        "",
        "## Metrics",
        "",
        "```text",
        f"operator_count = {summary['operator_count']}",
        f"replacement_ready_count = {summary['replacement_ready_count']}",
        f"direct_runtime_candidate_count = {summary['direct_runtime_candidate_count']}",
        f"tightened_challenger_required_count = {summary['tightened_challenger_required_count']}",
        f"abstract_lineage_required_count = {summary['abstract_lineage_required_count']}",
        f"destructive_drop_count = {summary['destructive_drop_count']}",
        f"projected_pruned_activation_total = {summary['projected_pruned_activation_total']}",
        f"projected_selected_activation_total = {summary['projected_selected_activation_total']}",
        f"projected_output_activation_delta_total = {summary['projected_output_activation_delta_total']}",
        f"accepted_mutation_total = {summary['accepted_mutation_total']}",
        f"mutation_attempt_total = {summary['mutation_attempt_total']}",
        f"hard_negative_total = {summary['hard_negative_total']}",
        f"wrong_scope_call_total = {summary['wrong_scope_call_total']}",
        f"unsupported_answer_total = {summary['unsupported_answer_total']}",
        f"direct_flow_write_total = {summary['direct_flow_write_total']}",
        "```",
        "",
        "## Tier Split",
        "",
        "```text",
    ]
    for tier, count in sorted(by_tier.items()):
        lines.append(f"{tier} = {count}")
    lines.extend([
        "```",
        "",
        "## Supersession Pressure",
        "",
        "```text",
    ])
    for pressure in ("high", "medium", "low", "none"):
        lines.append(f"{pressure} = {by_pressure.get(pressure, 0)}")
    lines.extend([
        "```",
        "",
        "## Highest Replacement Pressure",
        "",
    ])
    pressure_rows = sorted(
        [row for row in ledger if row["replacement_ready"]],
        key=lambda row: (row["pruned_activation"], row["selected_prune_ratio"]),
        reverse=True,
    )[:12]
    for row in pressure_rows:
        lines.extend([
            f"### {row['operator_id']}",
            "",
            "```text",
            f"tier = {row['supersession_tier']}",
            f"action = {row['supersession_action']}",
            f"readiness = {row['readiness']}",
            f"current_activation = {row['current_activation']}",
            f"selected_activation = {row['selected_activation']}",
            f"pruned_activation = {row['pruned_activation']}",
            f"selected_prune_ratio = {row['selected_prune_ratio']:.6f}",
            f"label_alignment_score = {row['label_alignment_score']:.6f}",
            "```",
            "",
        ])
    lines.extend([
        "## Boundary",
        "",
        "This is a supersession ledger and apply plan. It does not mutate the",
        "runtime library or destructively delete existing operators.",
        "",
    ])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_out = Path(args.sample_out) if args.sample_out else None
    input_dir = Path(args.e136h_artifact)
    prepare_output_dir(out)

    summary_h = load_json(input_dir / "summary.json")
    rows_h = load_json(input_dir / "operator_refinement_results.json")["rows"]
    selected_h = load_json(input_dir / "selected_variants.json")["rows"]
    selected_ids = {row["operator_id"] for row in selected_h}

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "supersession planning only; no runtime mutation and no destructive prune",
        "input_e136h_artifact": str(input_dir),
        "created_at_ms": now_ms(),
    })

    ledger = [classify_operator(row) for row in rows_h]
    output_impact = [
        {
            "operator_id": row["operator_id"],
            "supersession_tier": row["supersession_tier"],
            "readiness": row["readiness"],
            "current_activation": row["current_activation"],
            "selected_activation": row["selected_activation"],
            "pruned_activation": row["pruned_activation"],
            "projected_output_activation_delta": row["projected_output_activation_delta"],
            "output_risk_reduction_proxy": row["output_risk_reduction_proxy"],
        }
        for row in ledger
    ]
    mutation_transfer = [
        {
            "operator_id": row["operator_id"],
            "selected_variant_id": row["selected_variant_id"],
            "selected_variant_type": row["selected_variant_type"],
            "supersession_action": row["supersession_action"],
            "accepted_mutations": row["accepted_mutations"],
            "mutation_attempts": row["mutation_attempts"],
            "mutation_accept_rate": round(row["accepted_mutations"] / max(1, row["mutation_attempts"]), 6),
            "next_gate": row["next_gate"],
        }
        for row in ledger
    ]

    by_tier = Counter(row["supersession_tier"] for row in ledger)
    replacement_ready = sum(1 for row in ledger if row["replacement_ready"])
    direct_runtime = sum(1 for row in ledger if row["direct_runtime_candidate"])
    abstract_lineage = sum(1 for row in ledger if row["lineage_required"])
    destructive_drop = sum(1 for row in ledger if row["destructive_drop"])
    tightened_challenger = by_tier.get("T2_TIGHTENED_TRIGGER_REPLACEMENT", 0)

    summary = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "input_decision": summary_h.get("decision"),
        "operator_count": len(ledger),
        "input_selected_variant_count": len(selected_ids),
        "replacement_ready_count": replacement_ready,
        "direct_runtime_candidate_count": direct_runtime,
        "tightened_challenger_required_count": tightened_challenger,
        "abstract_lineage_required_count": abstract_lineage,
        "destructive_drop_count": destructive_drop,
        "hold_for_more_evidence_count": by_tier.get("T4_HOLD_FOR_MORE_EVIDENCE", 0),
        "projected_current_activation_total": sum(row["current_activation"] for row in ledger),
        "projected_selected_activation_total": sum(row["selected_activation"] for row in ledger),
        "projected_pruned_activation_total": sum(row["pruned_activation"] for row in ledger),
        "projected_output_activation_delta_total": sum(row["projected_output_activation_delta"] for row in ledger),
        "accepted_mutation_total": sum(row["accepted_mutations"] for row in ledger),
        "mutation_attempt_total": sum(row["mutation_attempts"] for row in ledger),
        "hard_negative_total": sum(row["hard_negative"] for row in ledger),
        "wrong_scope_call_total": sum(row["wrong_scope_call"] for row in ledger),
        "unsupported_answer_total": sum(row["unsupported_answer"] for row in ledger),
        "direct_flow_write_total": sum(row["direct_flow_write"] for row in ledger),
        "next": NEXT,
    }
    pass_gate = (
        summary["input_decision"] == "e136h_existing_operator_refinement_mutation_prune_confirmed"
        and summary["operator_count"] == 34
        and summary["input_selected_variant_count"] == 34
        and summary["replacement_ready_count"] == 27
        and summary["direct_runtime_candidate_count"] == 16
        and summary["tightened_challenger_required_count"] == 11
        and summary["abstract_lineage_required_count"] == 7
        and summary["hold_for_more_evidence_count"] == 0
        and summary["destructive_drop_count"] == 0
        and summary["projected_current_activation_total"] == int(summary_h["current_activation_total"])
        and summary["projected_selected_activation_total"] == int(summary_h["selected_activation_total"])
        and summary["projected_pruned_activation_total"] == int(summary_h["pruned_activation_total"])
        and summary["accepted_mutation_total"] == int(summary_h["accepted_mutation_total"])
        and summary["mutation_attempt_total"] == int(summary_h["mutation_attempt_total"])
        and summary["hard_negative_total"] == 0
        and summary["wrong_scope_call_total"] == 0
        and summary["unsupported_answer_total"] == 0
        and summary["direct_flow_write_total"] == 0
    )
    decision = DECISION_CONFIRMED if pass_gate else DECISION_REJECTED
    summary["decision"] = decision
    summary["pass_gate"] = pass_gate

    write_json(out / "input_e136h_report.json", {
        "input_artifact_contract": summary_h.get("artifact_contract"),
        "input_decision": summary_h.get("decision"),
        "input_pass_gate": summary_h.get("pass_gate"),
        "input_operator_count": summary_h.get("operator_count"),
        "input_current_activation_total": summary_h.get("current_activation_total"),
        "input_selected_activation_total": summary_h.get("selected_activation_total"),
        "input_pruned_activation_total": summary_h.get("pruned_activation_total"),
    })
    write_json(out / "supersession_ledger.json", {"rows": ledger})
    write_json(out / "output_impact_ledger.json", {"rows": output_impact})
    write_json(out / "mutation_transfer_ledger.json", {"rows": mutation_transfer})
    write_json(out / "aggregate_metrics.json", {key: value for key, value in summary.items() if key not in {"decision", "pass_gate", "next"}})
    write_json(out / "decision.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision,
        "next": NEXT,
        "pass_gate": pass_gate,
    })
    write_json(out / "summary.json", summary)
    write_json(out / "checker_summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": 0 if pass_gate else 1,
        "failures": [] if pass_gate else ["e136i_pass_gate_failed"],
    })
    write_report(out, summary, ledger)
    copy_sample(out, sample_out)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e136h-artifact", default=str(DEFAULT_E136H))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": summary["next"],
        "operator_count": summary["operator_count"],
        "replacement_ready_count": summary["replacement_ready_count"],
        "direct_runtime_candidate_count": summary["direct_runtime_candidate_count"],
        "tightened_challenger_required_count": summary["tightened_challenger_required_count"],
        "abstract_lineage_required_count": summary["abstract_lineage_required_count"],
        "projected_pruned_activation_total": summary["projected_pruned_activation_total"],
        "pass_gate": summary["pass_gate"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
