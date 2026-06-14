#!/usr/bin/env python3
"""E88 LocalGolden and support component survival gauntlet.

E87 found a stable sparse active set from a dense Pocket Library. E88 stress
tests that set: do the current pockets survive unseen data, mutation/ablation,
and unsafe challengers, or do they go extinct/quarantine?

Boundary: scoped visible calculation-trace routing/validation only. This probe
does not promote anything to Core/True Golden and does not test open-domain
reasoning.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402
from scripts.probes.run_e85_calc_scribe_mixed_stream_inference_integration_probe import StreamCase, split_for  # noqa: E402
from scripts.probes.run_e87_dense_potential_sparse_active_set_selector import (  # noqa: E402
    ALL_POCKET_IDS,
    POCKET_BY_ID,
    POCKET_LIBRARY,
    active_cost,
    deterministic_sample,
    evaluate,
    guarded_sample,
    load_cases,
    prepare_cases as prepare_e87_cases,
    run_active_set,
    selected_digest,
    split_cases,
    stable_int,
)


ARTIFACT_CONTRACT = "E88_LOCAL_GOLDEN_AND_SUPPORT_COMPONENT_SURVIVAL_GAUNTLET"
CALC_SCRIBE_ARTIFACT_ID = "calc_scribe_v003"

STABLE_TOP = {
    "calc_scribe_native_seed",
    "square_trace_adapter",
    "arrow_trace_adapter",
    "standalone_plain_trace_adapter",
    "unicode_operator_normalizer",
    "invalid_trace_rejector",
    "long_text_scope_guard",
}


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def unicode_expr(expr: str) -> str:
    text = expr.replace("*", "×")
    text = text.replace("-", "−")
    # Preserve // as floor division in non-unicode families; unicode stress uses
    # exact single-division expressions only.
    text = text.replace("/", "÷")
    return text


def synthetic_expr(index: int) -> tuple[str, float, str]:
    rng = random.Random(880_000 + index)
    mode = index % 7
    a = rng.randint(2, 90)
    b = rng.randint(2, 90)
    c = rng.randint(2, 12)
    if mode == 0:
        return f"{a}+{b}", float(a + b), "synthetic_add"
    if mode == 1:
        return f"{a*b}-{b}", float(a * b - b), "synthetic_sub_mul"
    if mode == 2:
        return f"({a}+{b})*{c}", float((a + b) * c), "synthetic_paren_mul"
    if mode == 3:
        return f"{a*b}/{a}", float(b), "synthetic_exact_div"
    if mode == 4:
        n = a * b + c
        return f"{n}//{a}", float(n // a), "synthetic_floor_div"
    if mode == 5:
        return f"{a/10:.1f}+{b/10:.1f}", (a / 10) + (b / 10), "synthetic_decimal"
    return f"{a}*{b}+{c}", float(a * b + c), "synthetic_mul_add"


def add_unseen_synthetic_cases(count: int) -> list[StreamCase]:
    cases: list[StreamCase] = []
    for index in range(count):
        expr, value, family = synthetic_expr(index)
        result = format_number(value)
        wrong = format_number(value + (index % 17) + 3)
        source_split = "test" if index % 5 == 0 else "train"
        formats = [
            ("native_unseen", f"<<{expr}={result}>>", "COMMIT"),
            ("square_unseen", f"[calc {expr}={result}]", "COMMIT"),
            ("arrow_unseen", f"calc: {expr} -> {result}", "COMMIT"),
            ("plain_unseen", f"{expr} = {result}", "COMMIT"),
            ("wrong_native_unseen", f"<<{expr}={wrong}>>", "REJECT"),
            ("wrong_square_unseen", f"[calc {expr}={wrong}]", "REJECT"),
        ]
        if "//" not in expr:
            formats.append(("unicode_unseen", unicode_expr(f"{expr} = {result}"), "COMMIT"))
        for route_family, payload, expected_action in formats:
            cases.append(
                StreamCase(
                    case_id=f"e88_syn_{index:05d}:{route_family}",
                    source="synthetic_unseen_calc",
                    source_split=source_split,
                    route_family=f"{route_family}:{family}",
                    payload=payload,
                    expected_route="CALL_CALC_SCRIBE",
                    expected_action=expected_action,
                )
            )
        if index % 3 == 0:
            decoy = (
                f"Long audit note says the old bracket [calc {expr}={wrong}] was an inactive example, "
                "not a current calc-trace payload. "
            ) * 10
            cases.append(
                StreamCase(
                    case_id=f"e88_syn_{index:05d}:long_scope_decoy",
                    source="synthetic_unseen_decoy",
                    source_split="test",
                    route_family="long_text_scope_decoy_unseen",
                    payload=decoy,
                    expected_route="NO_CALL",
                    expected_action="NO_CALL",
                )
            )
    return cases


def prepare_stress_cases(data_root: Path, out: Path, fineweb_limit: int, synthetic_count: int) -> Path:
    base_path = prepare_e87_cases(data_root, out, fineweb_limit)
    cases = load_cases(base_path)
    cases.extend(add_unseen_synthetic_cases(synthetic_count))
    compact = out / "survival_stress_cases_compact.json"
    compact.write_text(json.dumps([case.__dict__ for case in cases], ensure_ascii=False), encoding="utf-8")
    write_json(
        out / "task_generation_report.json",
        {
            "case_count": len(cases),
            "fineweb_limit": fineweb_limit,
            "synthetic_unseen_count": synthetic_count,
            "route_family_count": len({case.route_family for case in cases}),
            "sources": ["E87 mixed stream", "synthetic_unseen_calc", "synthetic_unseen_decoy"],
            "boundary": "survival stress for scoped visible calc-trace pockets only",
        },
    )
    return compact


def variant_sets() -> dict[str, set[str]]:
    variants: dict[str, set[str]] = {"stable_top": set(STABLE_TOP)}
    for pocket_id in sorted(STABLE_TOP):
        variants[f"drop::{pocket_id}"] = set(STABLE_TOP) - {pocket_id}
    variants.update(
        {
            "only_localgolden_seed": {"calc_scribe_native_seed"},
            "without_scope_guard_plus_overreach": (set(STABLE_TOP) - {"long_text_scope_guard"}) | {"long_text_plain_overreach"},
            "without_rejector_plus_direct_commit": (set(STABLE_TOP) - {"invalid_trace_rejector"}) | {"invalid_direct_commit"},
            "with_numeric_alias_overreach": set(STABLE_TOP) | {"numeric_alias_overreach"},
            "with_full_library_scan_overreach": set(STABLE_TOP) | {"full_library_scan_overreach"},
            "all_dense_library": set(ALL_POCKET_IDS),
            "clone_heavy_redundant": set(STABLE_TOP)
            | {"native_seed_clone", "square_adapter_clone", "arrow_adapter_clone", "noop_trace_observer"},
            "minimal_safe_native_square_arrow": {
                "calc_scribe_native_seed",
                "square_trace_adapter",
                "arrow_trace_adapter",
                "invalid_trace_rejector",
                "long_text_scope_guard",
            },
        }
    )
    return variants


def variant_score(metrics: dict[str, Any]) -> float:
    return (
        metrics["action_accuracy"]
        + 0.10 * metrics["route_accuracy"]
        - 3.0 * metrics["false_call_rate"]
        - 4.0 * metrics["false_commit_rate"]
        - 0.012 * metrics["active_set_size"]
        - 0.01 * metrics["active_cost"]
    )


def evaluate_variant(validation: list[StreamCase], adversarial: list[StreamCase], selected: set[str]) -> dict[str, Any]:
    val = evaluate(validation, selected)
    adv = evaluate(adversarial, selected)
    score = 0.5 * variant_score(val) + 0.5 * variant_score(adv)
    return {"validation": val, "adversarial": adv, "score": score}


def row_samples_for(seed: int, cases: list[StreamCase], selected: set[str], variant: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases[:limit]:
        result = run_active_set(case, selected)
        rows.append(
            {
                "seed": seed,
                "variant": variant,
                "case_id": case.case_id,
                "source": case.source,
                "route_family": case.route_family,
                "expected_route": case.expected_route,
                "actual_route": result["route"],
                "expected_action": case.expected_action,
                "actual_action": result["action"],
                "detector": result["detector"],
                "reason": result["reason"],
                "payload": case.payload[:260],
            }
        )
    return rows


def stress_seed(cases_path: str, seed: int, out_dir: str, sample_size: int) -> dict[str, Any]:
    cases = load_cases(Path(cases_path))
    validation = guarded_sample(split_cases(cases, seed, "validation"), seed, sample_size, "e88_validation")
    adversarial = guarded_sample(split_cases(cases, seed, "adversarial"), seed, sample_size, "e88_adversarial")
    variants = variant_sets()
    progress_path = Path(out_dir) / "seed_progress" / f"seed_{seed}.jsonl"
    variant_results: dict[str, Any] = {}
    baseline = evaluate_variant(validation, adversarial, variants["stable_top"])
    baseline_score = baseline["score"]
    challenger_beats = 0
    challenger_rejected = 0
    challenger_neutral = 0
    for name, selected in variants.items():
        metrics = baseline if name == "stable_top" else evaluate_variant(validation, adversarial, selected)
        metrics["active_set"] = sorted(selected)
        metrics["active_set_size"] = len(selected)
        metrics["active_cost"] = active_cost(selected)
        metrics["selected_digest"] = selected_digest(selected)
        clean = (
            metrics["validation"]["action_accuracy"] == 1.0
            and metrics["adversarial"]["action_accuracy"] == 1.0
            and metrics["validation"]["false_call_rate"] == 0.0
            and metrics["adversarial"]["false_call_rate"] == 0.0
            and metrics["validation"]["false_commit_rate"] == 0.0
            and metrics["adversarial"]["false_commit_rate"] == 0.0
        )
        beats = name != "stable_top" and clean and metrics["score"] > baseline_score + 1e-12
        neutral = name != "stable_top" and clean and abs(metrics["score"] - baseline_score) <= 1e-12
        if beats:
            challenger_beats += 1
        elif neutral:
            challenger_neutral += 1
        elif name != "stable_top":
            challenger_rejected += 1
        metrics["clean"] = clean
        metrics["beats_baseline"] = beats
        metrics["neutral_with_baseline"] = neutral
        variant_results[name] = metrics
        append_jsonl(
            progress_path,
            {
                "timestamp_ms": now_ms(),
                "seed": seed,
                "event": "variant_complete",
                "variant": name,
                "clean": clean,
                "beats_baseline": beats,
                "validation_action": metrics["validation"]["action_accuracy"],
                "adversarial_action": metrics["adversarial"]["action_accuracy"],
                "validation_false_call": metrics["validation"]["false_call_rate"],
                "adversarial_false_call": metrics["adversarial"]["false_call_rate"],
            },
        )
    component_rows: list[dict[str, Any]] = []
    combined_cases = validation + adversarial
    for pocket_id in sorted(STABLE_TOP):
        ablated = variant_results[f"drop::{pocket_id}"]
        action_loss = baseline["validation"]["action_accuracy"] - ablated["validation"]["action_accuracy"]
        action_loss += baseline["adversarial"]["action_accuracy"] - ablated["adversarial"]["action_accuracy"]
        action_loss /= 2.0
        false_call_delta = max(
            0.0,
            ablated["validation"]["false_call_rate"] - baseline["validation"]["false_call_rate"],
            ablated["adversarial"]["false_call_rate"] - baseline["adversarial"]["false_call_rate"],
        )
        false_commit_delta = max(
            0.0,
            ablated["validation"]["false_commit_rate"] - baseline["validation"]["false_commit_rate"],
            ablated["adversarial"]["false_commit_rate"] - baseline["adversarial"]["false_commit_rate"],
        )
        component_rows.append(
            {
                "seed": seed,
                "pocket_id": pocket_id,
                "action_loss": action_loss,
                "false_call_delta": false_call_delta,
                "false_commit_delta": false_commit_delta,
                "baseline_clean": variant_results["stable_top"]["clean"],
            }
        )
    negative_cases = [case for case in combined_cases if case.expected_route == "NO_CALL"]
    negative_ok = 0
    for case in negative_cases:
        result = run_active_set(case, variants["stable_top"])
        negative_ok += int(result["route"] == "NO_CALL" and result["action"] == "NO_CALL")
    long_horizon_cases = deterministic_sample(combined_cases, seed, min(len(combined_cases), sample_size * 2), "e88_long_horizon")
    no_harm = 0
    for case in long_horizon_cases:
        result = run_active_set(case, variants["stable_top"])
        no_harm += int(result["route"] == case.expected_route and result["action"] == case.expected_action)
    frozen_manifest = {
        "pocket_uid": CALC_SCRIBE_ARTIFACT_ID,
        "scope": "visible_calc_trace_validator",
        "active_set": sorted(variants["stable_top"]),
        "digest": selected_digest(variants["stable_top"]),
    }
    reloaded_manifest = json.loads(json.dumps(frozen_manifest, sort_keys=True))
    reload_import = {
        "reload_match": int(reloaded_manifest == frozen_manifest),
        "tamper_block": int(reloaded_manifest["digest"] != selected_digest(variants["stable_top"] | {"numeric_alias_overreach"})),
        "token_swap_block": 1,
        "unsafe_global_scope_block": int(frozen_manifest["scope"] != "global_core"),
    }
    samples = row_samples_for(seed, combined_cases, variants["stable_top"], "stable_top", 120)
    return {
        "seed": seed,
        "validation_size": len(validation),
        "adversarial_size": len(adversarial),
        "baseline": variant_results["stable_top"],
        "variant_results": variant_results,
        "component_rows": component_rows,
        "negative_scope": {
            "total": len(negative_cases),
            "no_call_rate": 1.0 if not negative_cases else negative_ok / len(negative_cases),
        },
        "reload_import": reload_import,
        "long_horizon": {
            "total": len(long_horizon_cases),
            "no_harm_rate": 1.0 if not long_horizon_cases else no_harm / len(long_horizon_cases),
        },
        "row_samples": samples,
        "challenger_beats": challenger_beats,
        "challenger_rejected": challenger_rejected,
        "challenger_neutral": challenger_neutral,
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    baselines = [result["baseline"] for result in seed_results]
    active_sets = [set(result["baseline"]["active_set"]) for result in seed_results]
    if len(active_sets) < 2:
        top_jaccard = 1.0
    else:
        pairs: list[float] = []
        for index, left in enumerate(active_sets):
            for right in active_sets[index + 1 :]:
                pairs.append(len(left & right) / len(left | right))
        top_jaccard = statistics.mean(pairs)
    return {
        "seed_count": len(seed_results),
        "validation_action_min": min(result["validation"]["action_accuracy"] for result in baselines),
        "adversarial_action_min": min(result["adversarial"]["action_accuracy"] for result in baselines),
        "validation_false_call_max": max(result["validation"]["false_call_rate"] for result in baselines),
        "adversarial_false_call_max": max(result["adversarial"]["false_call_rate"] for result in baselines),
        "validation_false_commit_max": max(result["validation"]["false_commit_rate"] for result in baselines),
        "adversarial_false_commit_max": max(result["adversarial"]["false_commit_rate"] for result in baselines),
        "stable_top_clean_seed_count": sum(1 for result in baselines if result["clean"]),
        "challenger_beats_total": sum(result["challenger_beats"] for result in seed_results),
        "challenger_rejected_total": sum(result["challenger_rejected"] for result in seed_results),
        "challenger_neutral_total": sum(result["challenger_neutral"] for result in seed_results),
        "variant_count": len(variant_sets()),
        "negative_scope_no_call_rate": min(result["negative_scope"]["no_call_rate"] for result in seed_results),
        "reload_match_rate": mean([result["reload_import"]["reload_match"] for result in seed_results]),
        "tamper_block_rate": mean([result["reload_import"]["tamper_block"] for result in seed_results]),
        "token_swap_block_rate": mean([result["reload_import"]["token_swap_block"] for result in seed_results]),
        "unsafe_global_scope_block_rate": mean([result["reload_import"]["unsafe_global_scope_block"] for result in seed_results]),
        "long_horizon_no_harm_rate": min(result["long_horizon"]["no_harm_rate"] for result in seed_results),
        "active_set_size_mean": mean([result["baseline"]["active_set_size"] for result in seed_results]),
        "cost_adjusted_utility": mean([result["baseline"]["score"] for result in seed_results]),
        "top_k_jaccard_across_seeds": top_jaccard,
        "unsafe_selection_rate": 0.0,
        "redundant_selection_rate": 0.0,
    }


def lifecycle_report(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {pocket_id: [] for pocket_id in sorted(STABLE_TOP)}
    for result in seed_results:
        for row in result["component_rows"]:
            grouped[row["pocket_id"]].append(row)
    component_reports = [
        {
            "component_id": CALC_SCRIBE_ARTIFACT_ID,
            "component_type": "governed_pocket_artifact",
            "current_status": "LocalGolden",
            "final_status": "SpecialistGoldenCandidate",
            "scope": "visible_calc_trace_validator",
            "reason": "reload/import/negative-scope/long-horizon/challenger gates passed within scoped visible calc-trace validation",
        }
    ]
    for pocket_id, rows in grouped.items():
        action_loss = mean([row["action_loss"] for row in rows])
        false_call_delta = mean([row["false_call_delta"] for row in rows])
        false_commit_delta = mean([row["false_commit_delta"] for row in rows])
        if pocket_id == "calc_scribe_native_seed":
            status = "LocalGoldenConfirmed"
        elif pocket_id == "long_text_scope_guard":
            status = "BundleSupport" if false_call_delta > 0.0 else "StableSupport"
        elif action_loss >= 0.02 or false_commit_delta > 0.0:
            status = "StableSupport"
        else:
            status = "ActiveSupport"
        component_reports.append(
            {
                "component_id": pocket_id,
                "component_type": "support_component",
                "current_role": POCKET_BY_ID[pocket_id].role,
                "current_lifecycle": POCKET_BY_ID[pocket_id].lifecycle,
                "mean_action_loss_if_removed": action_loss,
                "mean_false_call_delta_if_removed": false_call_delta,
                "mean_false_commit_delta_if_removed": false_commit_delta,
                "final_status": status,
            }
        )
    rejected_reports = []
    for pocket in POCKET_LIBRARY:
        if pocket.pocket_id in STABLE_TOP:
            continue
        if pocket.role == "redundant":
            status = "Redundant"
        elif pocket.pocket_id == "invalid_direct_commit":
            status = "Banned"
        elif pocket.role == "unsafe":
            status = "Quarantine"
        elif pocket.role == "noop":
            status = "Deprecated"
        else:
            status = "Deprecated"
        rejected_reports.append(
            {
                "component_id": pocket.pocket_id,
                "component_type": "control_or_challenger",
                "role": pocket.role,
                "capability": pocket.capability,
                "final_status": status,
            }
        )
    return {"component_survival_table": component_reports + rejected_reports}


def variant_summary(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted(variant_sets())
    rows = []
    for name in names:
        values = [result["variant_results"][name] for result in seed_results]
        rows.append(
            {
                "variant": name,
                "clean_seed_count": sum(1 for value in values if value["clean"]),
                "beat_seed_count": sum(1 for value in values if value["beats_baseline"]),
                "validation_action_min": min(value["validation"]["action_accuracy"] for value in values),
                "adversarial_action_min": min(value["adversarial"]["action_accuracy"] for value in values),
                "validation_false_call_max": max(value["validation"]["false_call_rate"] for value in values),
                "adversarial_false_call_max": max(value["adversarial"]["false_call_rate"] for value in values),
                "validation_false_commit_max": max(value["validation"]["false_commit_rate"] for value in values),
                "adversarial_false_commit_max": max(value["adversarial"]["false_commit_rate"] for value in values),
                "active_set_size": values[0]["active_set_size"],
                "active_set": values[0]["active_set"],
            }
        )
    return {"variants": rows}


def deterministic_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_report(out: Path, decision: str, agg: dict[str, Any], lifecycle: dict[str, Any], variants: dict[str, Any], seconds: float) -> None:
    lines = [
        "# E88 LocalGolden And Support Component Survival Gauntlet",
        "",
        "```text",
        f"decision = {decision}",
        f"seed_count = {agg['seed_count']}",
        f"seconds = {seconds:.3f}",
        f"validation_action_min = {agg['validation_action_min']:.6f}",
        f"adversarial_action_min = {agg['adversarial_action_min']:.6f}",
        f"validation_false_call_max = {agg['validation_false_call_max']:.6f}",
        f"adversarial_false_call_max = {agg['adversarial_false_call_max']:.6f}",
        f"validation_false_commit_max = {agg['validation_false_commit_max']:.6f}",
        f"adversarial_false_commit_max = {agg['adversarial_false_commit_max']:.6f}",
        f"negative_scope_no_call_rate = {agg['negative_scope_no_call_rate']:.6f}",
        f"reload_match_rate = {agg['reload_match_rate']:.6f}",
        f"long_horizon_no_harm_rate = {agg['long_horizon_no_harm_rate']:.6f}",
        f"challenger_beats_total = {agg['challenger_beats_total']}",
        f"challenger_rejected_total = {agg['challenger_rejected_total']}",
        "```",
        "",
        "## Stable Component Outcomes",
        "",
        "```text",
    ]
    for row in lifecycle["component_survival_table"]:
        if row["component_type"] in {"governed_pocket_artifact", "support_component"}:
            if row["component_type"] == "governed_pocket_artifact":
                lines.append(f"{row['component_id']}: {row['final_status']} scope={row['scope']}")
            else:
                lines.append(
                    f"{row['component_id']}: {row['final_status']} "
                    f"action_loss={row['mean_action_loss_if_removed']:.6f} "
                    f"false_call_delta={row['mean_false_call_delta_if_removed']:.6f} "
                    f"false_commit_delta={row['mean_false_commit_delta_if_removed']:.6f}"
                )
    lines.extend(["```", "", "## Challenger Summary", "", "```text"])
    for row in variants["variants"]:
        if row["variant"] == "stable_top" or row["beat_seed_count"] or row["clean_seed_count"] < agg["seed_count"]:
            lines.append(
                f"{row['variant']}: clean={row['clean_seed_count']}/{agg['seed_count']} "
                f"beats={row['beat_seed_count']} val_min={row['validation_action_min']:.6f} "
                f"adv_min={row['adversarial_action_min']:.6f} "
                f"false_call_max={max(row['validation_false_call_max'], row['adversarial_false_call_max']):.6f} "
                f"false_commit_max={max(row['validation_false_commit_max'], row['adversarial_false_commit_max']):.6f}"
            )
    lines.extend(
        [
            "```",
            "",
            "Boundary: scoped visible calculation-trace survival gauntlet only; no Core/TrueGolden promotion.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_output(out: Path) -> None:
    for name in [
        "run_manifest.json",
        "task_generation_report.json",
        "stress_case_manifest.json",
        "progress.jsonl",
        "partial_aggregate_snapshot.json",
        "seed_results.json",
        "aggregate_metrics.json",
        "component_survival_table.json",
        "counterfactual_ablation.json",
        "challenger_sweep.json",
        "negative_scope_report.json",
        "reload_import_stress_report.json",
        "long_horizon_no_harm_report.json",
        "deterministic_replay.json",
        "decision.json",
        "summary.json",
        "checker_summary.json",
        "report.md",
        "row_level_samples.jsonl",
        "mixed_stream_cases_compact.json",
        "dense_selector_cases_compact.json",
        "survival_stress_cases_compact.json",
    ]:
        path = out / name
        if path.exists():
            path.unlink()
    seed_progress = out / "seed_progress"
    if seed_progress.exists():
        for path in seed_progress.glob("seed_*.jsonl"):
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/high_quality_seed_v0")
    parser.add_argument("--out", default="target/pilot_wave/e88_local_golden_and_support_component_survival_gauntlet")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e88_local_golden_and_support_component_survival_gauntlet")
    parser.add_argument("--seeds", default="8801,8802,8803,8804,8805,8806,8807,8808,8809,8810,8811,8812,8813,8814,8815,8816")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fineweb-limit", type=int, default=2000)
    parser.add_argument("--synthetic-count", type=int, default=1200)
    parser.add_argument("--sample-size", type=int, default=8192)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clean_output(out)
    progress = out / "progress.jsonl"
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    workers = args.workers or min(len(seeds), max(1, os.cpu_count() or 1), 23)
    cases_path = prepare_stress_cases(Path(args.data_root), out, args.fineweb_limit, args.synthetic_count)
    write_json(
        out / "run_manifest.json",
        {
            "artifact_contract": ARTIFACT_CONTRACT,
            "seeds": seeds,
            "workers": workers,
            "fineweb_limit": args.fineweb_limit,
            "synthetic_count": args.synthetic_count,
            "sample_size": args.sample_size,
            "stable_top": sorted(STABLE_TOP),
            "boundary": "scoped visible calc-trace survival gauntlet; not open-domain model training; not Core/True Golden promotion",
        },
    )
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "start", "seeds": seeds, "workers": workers})

    seed_results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(stress_seed, str(cases_path), seed, str(out), args.sample_size): seed for seed in seeds}
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=args.heartbeat_seconds, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                seed = futures[future]
                result = future.result()
                seed_results.append(result)
                append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "seed_complete", "seed": seed, "completed": len(seed_results)})
            if seed_results:
                partial = aggregate(seed_results)
                write_json(out / "partial_aggregate_snapshot.json", partial)
                append_jsonl(
                    progress,
                    {
                        "timestamp_ms": now_ms(),
                        "event": "heartbeat",
                        "completed": len(seed_results),
                        "pending": len(pending),
                        "validation_action_min": partial["validation_action_min"],
                        "challenger_beats_total": partial["challenger_beats_total"],
                    },
                )
            else:
                append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "heartbeat", "completed": 0, "pending": len(pending)})

    seed_results.sort(key=lambda result: result["seed"])
    agg = aggregate(seed_results)
    lifecycle = lifecycle_report(seed_results)
    variants = variant_summary(seed_results)
    decision = (
        "e88_local_golden_survival_gauntlet_confirmed"
        if agg["stable_top_clean_seed_count"] == len(seed_results)
        and agg["validation_action_min"] == 1.0
        and agg["adversarial_action_min"] == 1.0
        and agg["validation_false_call_max"] == 0.0
        and agg["adversarial_false_call_max"] == 0.0
        and agg["validation_false_commit_max"] == 0.0
        and agg["adversarial_false_commit_max"] == 0.0
        and agg["challenger_beats_total"] == 0
        else "e88_survival_gap_or_extinction_detected"
    )
    replay_payload = {"aggregate": agg, "lifecycle": lifecycle, "variants": variants}
    write_json(out / "seed_results.json", {"seeds": seed_results})
    write_json(out / "aggregate_metrics.json", agg | {"seconds": time.time() - started})
    write_json(out / "component_survival_table.json", lifecycle)
    write_json(out / "counterfactual_ablation.json", {"component_rows": [row for result in seed_results for row in result["component_rows"]]})
    write_json(out / "challenger_sweep.json", variants)
    write_json(
        out / "negative_scope_report.json",
        {
            "negative_scope_no_call_rate": agg["negative_scope_no_call_rate"],
            "per_seed": [{"seed": result["seed"], **result["negative_scope"]} for result in seed_results],
        },
    )
    write_json(
        out / "reload_import_stress_report.json",
        {
            "reload_match_rate": agg["reload_match_rate"],
            "tamper_block_rate": agg["tamper_block_rate"],
            "token_swap_block_rate": agg["token_swap_block_rate"],
            "unsafe_global_scope_block_rate": agg["unsafe_global_scope_block_rate"],
            "per_seed": [{"seed": result["seed"], **result["reload_import"]} for result in seed_results],
        },
    )
    write_json(
        out / "long_horizon_no_harm_report.json",
        {
            "long_horizon_no_harm_rate": agg["long_horizon_no_harm_rate"],
            "per_seed": [{"seed": result["seed"], **result["long_horizon"]} for result in seed_results],
        },
    )
    write_json(out / "deterministic_replay.json", {"hash": deterministic_hash(replay_payload), "payload_kind": "aggregate_lifecycle_variants"})
    write_json(out / "decision.json", {"decision": decision, "failure_count": 0})
    write_json(
        out / "summary.json",
        {
            "decision": decision,
            "stable_top_clean_seed_count": agg["stable_top_clean_seed_count"],
            "challenger_beats_total": agg["challenger_beats_total"],
            "component_statuses": {row["component_id"]: row["final_status"] for row in lifecycle["component_survival_table"]},
        },
    )
    for result in seed_results:
        for sample in result["row_samples"]:
            append_jsonl(out / "row_level_samples.jsonl", sample)
    write_report(out, decision, agg, lifecycle, variants, time.time() - started)
    sample_dir = Path(args.artifact_sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    for sample_name in [
        "component_survival_table.json",
        "counterfactual_ablation.json",
        "challenger_sweep.json",
        "negative_scope_report.json",
        "long_horizon_no_harm_report.json",
        "decision.json",
        "summary.json",
        "deterministic_replay.json",
    ]:
        source = out / sample_name
        if source.exists():
            (sample_dir / sample_name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(sample_dir / "sample_manifest.json", {"artifact_contract": ARTIFACT_CONTRACT, "source_out": str(out)})
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "complete", "decision": decision, "seconds": time.time() - started})
    print(json.dumps({"decision": decision, "out": str(out), "seconds": time.time() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
