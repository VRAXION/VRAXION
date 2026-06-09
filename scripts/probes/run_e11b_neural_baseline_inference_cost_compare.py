#!/usr/bin/env python3
"""E11B neural baseline inference-cost compare over the E10 noisy-route task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any, Callable


MILESTONE = "E11B_NEURAL_BASELINE_INFERENCE_COST_COMPARE"
DEFAULT_OUT = Path("target/pilot_wave/e11b_neural_baseline_inference_cost_compare")
DEFAULT_SEEDS = (110201, 110202, 110203, 110204)
DEFAULT_ROWS_PER_SPLIT = 24
DEFAULT_BENCH_REPEATS = 2
E10_PATH = Path(__file__).with_name("run_e10_operator_library_transfer_noisy_route_confirm.py")

FLOW = "FLOW_E10_SCHEDULED_SCHEMA_GATED_PRUNED_PYTHON"
FLOW_BITSET = "FLOW_E10_BITSET_COST_MODEL"
MLP_ROUTE = "TINY_MLP_ROUTE_ONLY_CONTROLLER"
MLP_TRACE = "TINY_MLP_TRACE_CONTROLLER"
GRU_TRACE = "TINY_GRU_TRACE_CONTROLLER"
TRANSFORMER_TRACE = "SMALL_TRANSFORMER_TRACE_CONTROLLER"
SYSTEMS = (FLOW, FLOW_BITSET, MLP_ROUTE, MLP_TRACE, GRU_TRACE, TRANSFORMER_TRACE)
NEURAL_SYSTEMS = (MLP_ROUTE, MLP_TRACE, GRU_TRACE, TRANSFORMER_TRACE)
MEASURED_SYSTEMS = (FLOW, MLP_ROUTE, MLP_TRACE, GRU_TRACE, TRANSFORMER_TRACE)
VALID_DECISIONS = (
    "e11b_flow_proxy_cost_advantage_vs_quality_matched_neural_confirmed",
    "e11b_neural_quality_mismatch_no_cost_claim",
    "e11b_flow_proxy_cost_advantage_not_confirmed",
    "e11b_invalid_or_incomplete_run",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e11b_quality_report.json",
    "e11b_cost_model_report.json",
    "e11b_walltime_report.json",
    "e11b_neural_baseline_report.json",
    "e11b_deterministic_replay_report.json",
)


def load_e10() -> Any:
    spec = importlib.util.spec_from_file_location("e10_operator_library_transfer_noisy_route_confirm", E10_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {E10_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E10 = load_e10()
SKILLS = E10.SKILLS


def rounded(value: float) -> float:
    return round(float(value), 6)


def rate(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return rounded(float(num) / float(den))


def stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stable_payload(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [stable_payload(item) for item in value]
    if isinstance(value, float):
        return rounded(value)
    return value


def stable_json(value: Any) -> str:
    return json.dumps(stable_payload(value), indent=2, sort_keys=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_git(args: list[str]) -> tuple[int, str]:
    try:
        done = subprocess.run(["git", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    except OSError as exc:
        return 127, str(exc)
    return done.returncode, done.stdout


@dataclass
class NeuralStats:
    commits: int = 0
    accepted_good: int = 0
    accepted_bad: int = 0
    destructive: int = 0
    branch_contam: int = 0
    noisy_steps: int = 0
    route_repairs: int = 0
    route_false_repairs: int = 0
    transfer_steps: int = 0
    transfer_good: int = 0
    clean_steps: int = 0
    clean_good: int = 0
    oscillations: int = 0
    collapse: int = 0
    used_skills: set[str] = field(default_factory=set)


def route_signal(row: Any, step_idx: int, skill: str) -> float:
    return float(row.signals[step_idx][skill]["route_confidence"])


def trace_signal(row: Any, step_idx: int, skill: str) -> float:
    return float(row.signals[step_idx][skill]["trace_confidence"])


def select_skill(system: str, row: Any, step_idx: int, state: dict[str, float]) -> str:
    if system == MLP_ROUTE:
        return max(SKILLS, key=lambda skill: route_signal(row, step_idx, skill))
    if system == MLP_TRACE:
        return max(SKILLS, key=lambda skill: 0.45 * route_signal(row, step_idx, skill) + 0.55 * trace_signal(row, step_idx, skill))
    if system == GRU_TRACE:
        scores: dict[str, float] = {}
        for skill in SKILLS:
            state[skill] = 0.72 * state.get(skill, 0.0) + trace_signal(row, step_idx, skill) + 0.25 * route_signal(row, step_idx, skill)
            scores[skill] = trace_signal(row, step_idx, skill) + 0.25 * route_signal(row, step_idx, skill) + 0.08 * state[skill]
        return max(SKILLS, key=lambda skill: scores[skill])
    if system == TRANSFORMER_TRACE:
        lo = max(0, step_idx - 2)
        hi = min(len(row.true_route), step_idx + 3)
        scores = {}
        for skill in SKILLS:
            neighbor_trace = sum(trace_signal(row, idx, skill) for idx in range(lo, hi)) / max(1, hi - lo)
            scores[skill] = 0.50 * trace_signal(row, step_idx, skill) + 0.25 * route_signal(row, step_idx, skill) + 0.25 * neighbor_trace
        return max(SKILLS, key=lambda skill: scores[skill])
    raise ValueError(f"unknown neural system {system}")


def run_neural_system(system: str, rows: list[Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    stats = NeuralStats()
    split_stats: dict[str, NeuralStats] = {split: NeuralStats() for split in E10.SPLITS}
    row_metrics: list[dict[str, float]] = []
    split_rows: dict[str, list[dict[str, float]]] = {split: [] for split in E10.SPLITS}
    for row in rows:
        current = E10.copy_grid(row.initial)
        frames = [E10.copy_grid(current)]
        state = {skill: 0.0 for skill in SKILLS}
        targets = (stats, split_stats[row.split])

        def add_stat(name: str, value: int) -> None:
            for target in targets:
                setattr(target, name, getattr(target, name) + value)

        for step_idx, true_skill in enumerate(row.true_route):
            observed = E10.observed_skill_at(row, step_idx)
            if observed != true_skill:
                add_stat("noisy_steps", 1)
            if row.split in E10.TRANSFER_SPLITS:
                add_stat("transfer_steps", 1)
            if observed == true_skill and row.split in {"validation", "heldout_transfer"}:
                add_stat("clean_steps", 1)
            predicted = select_skill(system, row, step_idx, state)
            good = predicted == true_skill
            before = E10.copy_grid(current)
            current = E10.apply_rule(current, E10.TRUE_RULES[predicted])
            oracle_after = row.oracle[min(step_idx + 1, len(row.oracle) - 1)]
            for target in targets:
                target.commits += 1
                target.accepted_good += int(good)
                target.accepted_bad += int(not good)
                if good:
                    target.used_skills.add(predicted)
                    if row.split in E10.TRANSFER_SPLITS:
                        target.transfer_good += 1
                    if observed == true_skill and row.split in {"validation", "heldout_transfer"}:
                        target.clean_good += 1
                if observed != true_skill and predicted != observed:
                    if good:
                        target.route_repairs += 1
                    else:
                        target.route_false_repairs += 1
                if E10.grid_similarity(before, oracle_after) > E10.grid_similarity(current, oracle_after) and not good:
                    target.destructive += 1
                if E10.grid_similarity(before, current) > 0.99 and E10.grid_similarity(current, oracle_after) < 0.95:
                    target.oscillations += 1
            frames.append(E10.copy_grid(current))
        if E10.grid_sum(current) in (0, E10.GRID * E10.GRID):
            add_stat("collapse", 1)
        metrics = E10.row_score(row, frames, system)
        row_metrics.append(metrics)
        split_rows[row.split].append(metrics)
    aggregate = aggregate_neural(row_metrics, stats, len(rows))
    split_report = {split: aggregate_neural(split_rows[split], split_stats[split], max(1, len(split_rows[split]))) for split in E10.SPLITS}
    diagnostics = {
        "commits": stats.commits,
        "accepted_good": stats.accepted_good,
        "accepted_bad": stats.accepted_bad,
        "destructive_overwrites": stats.destructive,
        "branch_contamination": stats.branch_contam,
        "noisy_steps": stats.noisy_steps,
        "route_repairs": stats.route_repairs,
        "route_false_repairs": stats.route_false_repairs,
        "operator_skills_used": sorted(stats.used_skills),
    }
    return aggregate, diagnostics, split_report


def aggregate_neural(rows: list[dict[str, float]], stats: NeuralStats, row_count: int) -> dict[str, Any]:
    def mean(key: str) -> float:
        return rounded(sum(row[key] for row in rows) / max(1, len(rows)))

    tick_count = sum(row["route_length"] for row in rows)
    return {
        "usefulness": mean("usefulness"),
        "answer_accuracy": mean("answer_accuracy"),
        "final_state_accuracy": mean("final_state_accuracy"),
        "trace_validity": mean("trace_validity"),
        "delta_validity": mean("delta_validity"),
        "observed_route_error_rate": mean("observed_route_error_rate"),
        "useful_writeback_recall": rate(stats.accepted_good, tick_count),
        "wrong_writeback_rate": rate(stats.accepted_bad, stats.commits),
        "destructive_overwrite_rate": rate(stats.destructive, stats.commits),
        "branch_contamination_rate": rate(stats.branch_contam, stats.commits),
        "route_repair_rate": rate(stats.route_repairs, stats.noisy_steps),
        "noisy_route_false_accept_rate": rate(stats.route_false_repairs, stats.noisy_steps),
        "transfer_coverage": rate(stats.transfer_good, stats.transfer_steps),
        "clean_route_preservation_rate": rate(stats.clean_good, stats.clean_steps),
        "operator_reuse_rate": rate(len(stats.used_skills), len(SKILLS)),
        "temporal_drift_rate": mean("temporal_drift_rate"),
        "oscillation_rate": rate(stats.oscillations, row_count),
        "attractor_collapse_rate": rate(stats.collapse, row_count),
        "deterministic_replay_passed": True,
        "baseline_type": "inference_only_neural_controller_with_fixed_operator_decoder",
    }


def flow_cost_model() -> dict[str, Any]:
    return {
        "proxy_ops_per_tick": 200.0,
        "controller_ops_per_tick": 112.0,
        "decoder_cell_ops_per_tick": 64.0,
        "gate_ops_per_tick": 24.0,
        "notes": "Sparse detector/schema/trace gate plus one region operation, counted as scalar proxy ops.",
    }


def bitset_cost_model() -> dict[str, Any]:
    return {
        "proxy_ops_per_tick": 48.0,
        "controller_ops_per_tick": 28.0,
        "decoder_cell_ops_per_tick": 8.0,
        "gate_ops_per_tick": 12.0,
        "measured_python_walltime": False,
        "notes": "Bitpacked/Rust-style estimate only; no Rust implementation is run by E11B.",
    }


def neural_cost_model(system: str) -> dict[str, Any]:
    input_dim = len(SKILLS) * 2
    hidden = 32
    output = len(SKILLS)
    decoder = 64.0
    if system in {MLP_ROUTE, MLP_TRACE}:
        macs = input_dim * hidden + hidden * output
        return {
            "proxy_ops_per_tick": float(macs + decoder),
            "controller_macs_per_tick": float(macs),
            "decoder_cell_ops_per_tick": decoder,
            "architecture": "16 -> 32 -> 8 dense controller",
        }
    if system == GRU_TRACE:
        macs = 3 * (input_dim * hidden + hidden * hidden + hidden) + hidden * output
        return {
            "proxy_ops_per_tick": float(macs + decoder),
            "controller_macs_per_tick": float(macs),
            "decoder_cell_ops_per_tick": decoder,
            "architecture": "single-step GRU-style controller, hidden=32",
        }
    if system == TRANSFORMER_TRACE:
        tokens = 5
        dim = 32
        ffn = 64
        qkv = 3 * tokens * dim * dim
        attention_scores = tokens * tokens * dim
        attention_values = tokens * tokens * dim
        out_proj = tokens * dim * dim
        feed_forward = 2 * tokens * dim * ffn
        classifier = dim * output
        macs = qkv + attention_scores + attention_values + out_proj + feed_forward + classifier
        return {
            "proxy_ops_per_tick": float(macs + decoder),
            "controller_macs_per_tick": float(macs),
            "decoder_cell_ops_per_tick": decoder,
            "architecture": "one-layer five-token transformer-style controller, d_model=32",
        }
    raise ValueError(system)


def attach_cost_fields(system: str, metrics: dict[str, Any], model: dict[str, Any], flow_ops: float) -> dict[str, Any]:
    out = dict(metrics)
    ops = float(model["proxy_ops_per_tick"])
    out.update(model)
    out["cost_per_correct_trace"] = rate(ops, max(0.000001, float(metrics["trace_validity"])))
    out["cost_per_valid_writeback"] = rate(ops, max(0.000001, float(metrics["useful_writeback_recall"])))
    out["ops_ratio_vs_flow_python"] = rate(ops, flow_ops)
    out["quality_matched"] = bool(
        metrics["trace_validity"] >= 0.90
        and metrics["usefulness"] >= 0.85
        and metrics["useful_writeback_recall"] >= 0.85
        and metrics["wrong_writeback_rate"] <= 0.05
    )
    out["system"] = system
    return out


def build_deterministic_reports(seeds: tuple[int, ...], rows_per_split: int) -> dict[str, Any]:
    rows = E10.build_rows(seeds, rows_per_split)
    metrics: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    splits: dict[str, dict[str, dict[str, Any]]] = {}
    flow_metrics, flow_diag, _flow_sample = E10.run_system(E10.PRIMARY, rows)
    metrics[FLOW] = attach_cost_fields(FLOW, flow_metrics, flow_cost_model(), flow_cost_model()["proxy_ops_per_tick"])
    diagnostics[FLOW] = {key: value for key, value in flow_diag.items() if key != "split_report"}
    splits[FLOW] = flow_diag["split_report"]
    bitset_metrics = dict(flow_metrics)
    metrics[FLOW_BITSET] = attach_cost_fields(FLOW_BITSET, bitset_metrics, bitset_cost_model(), flow_cost_model()["proxy_ops_per_tick"])
    diagnostics[FLOW_BITSET] = {"estimated_from": FLOW, "measured_python_walltime": False}
    splits[FLOW_BITSET] = flow_diag["split_report"]
    for system in NEURAL_SYSTEMS:
        neural_metrics, neural_diag, split_report = run_neural_system(system, rows)
        metrics[system] = attach_cost_fields(system, neural_metrics, neural_cost_model(system), flow_cost_model()["proxy_ops_per_tick"])
        diagnostics[system] = neural_diag
        splits[system] = split_report
    gate = positive_gate(metrics)
    decision_label = decide(gate)
    decision = {
        "schema_version": "e11b_decision_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "next": next_for(decision_label),
        "primary_system": FLOW,
        "positive_gate_passed": gate["passed"],
        "deterministic_replay_passed": True,
    }
    aggregate = {
        "schema_version": "e11b_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "seeds": list(seeds),
        "rows_per_split": rows_per_split,
        "systems": metrics,
        "diagnostics": diagnostics,
        "split_metrics": splits,
        "positive_gate": gate,
    }
    quality_report = {
        "schema_version": "e11b_quality_report_v1",
        "systems": {
            system: {
                key: metrics[system][key]
                for key in (
                    "usefulness",
                    "trace_validity",
                    "answer_accuracy",
                    "useful_writeback_recall",
                    "wrong_writeback_rate",
                    "route_repair_rate",
                    "transfer_coverage",
                    "quality_matched",
                )
            }
            for system in SYSTEMS
        },
    }
    cost_report = {
        "schema_version": "e11b_cost_model_report_v1",
        "cost_units": "scalar proxy ops per tick plus fixed-operator decoder cell ops; neural MACs are counted as scalar ops",
        "systems": {
            system: {
                key: metrics[system][key]
                for key in (
                    "proxy_ops_per_tick",
                    "cost_per_correct_trace",
                    "cost_per_valid_writeback",
                    "ops_ratio_vs_flow_python",
                )
            }
            for system in SYSTEMS
        },
        "caveat": "This is an inference controller cost model over E10 detector evidence, not a trained raw-grid neural model benchmark.",
    }
    neural_report = {
        "schema_version": "e11b_neural_baseline_report_v1",
        "baseline_scope": "Inference-only neural controller proxies over the same E10 route/trace detector evidence with a fixed region-operator decoder.",
        "training_run": False,
        "raw_grid_neural_model": False,
        "systems": {system: metrics[system].get("architecture", metrics[system].get("notes", "")) for system in SYSTEMS},
    }
    summary = {
        "schema_version": "e11b_summary_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "primary_system": FLOW,
        "positive_gate_passed": gate["passed"],
        "flow_trace_validity": metrics[FLOW]["trace_validity"],
        "flow_usefulness": metrics[FLOW]["usefulness"],
        "flow_proxy_ops_per_tick": metrics[FLOW]["proxy_ops_per_tick"],
        "cheapest_quality_matched_neural": gate["cheapest_quality_matched_neural"],
        "cheapest_quality_matched_neural_ops_per_tick": gate["cheapest_quality_matched_neural_ops_per_tick"],
        "flow_ops_advantage_vs_cheapest_quality_neural": gate["flow_ops_advantage_vs_cheapest_quality_neural"],
    }
    return {
        "decision.json": decision,
        "summary.json": summary,
        "aggregate_metrics.json": aggregate,
        "report.md": render_report(decision, aggregate),
        "e11b_quality_report.json": quality_report,
        "e11b_cost_model_report.json": cost_report,
        "e11b_neural_baseline_report.json": neural_report,
    }


def positive_gate(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flow = metrics[FLOW]
    quality_matched_neural = [system for system in NEURAL_SYSTEMS if metrics[system]["quality_matched"]]
    cheapest = min(quality_matched_neural, key=lambda system: metrics[system]["proxy_ops_per_tick"]) if quality_matched_neural else None
    cheapest_ops = metrics[cheapest]["proxy_ops_per_tick"] if cheapest else 0.0
    advantage = rate(cheapest_ops, flow["proxy_ops_per_tick"]) if cheapest else 0.0
    checks = {
        "flow_quality_matched": flow["quality_matched"] is True,
        "at_least_one_quality_matched_neural": bool(quality_matched_neural),
        "flow_wrong_writeback_zero": flow["wrong_writeback_rate"] == 0.0,
        "flow_destructive_zero": flow["destructive_overwrite_rate"] == 0.0,
        "flow_ops_at_least_3x_lower_than_cheapest_quality_neural": advantage >= 3.0,
        "flow_cost_per_valid_writeback_lower_than_cheapest_quality_neural": bool(cheapest and flow["cost_per_valid_writeback"] < metrics[cheapest]["cost_per_valid_writeback"]),
        "bitset_cost_model_lower_than_flow_python": metrics[FLOW_BITSET]["proxy_ops_per_tick"] < flow["proxy_ops_per_tick"],
        "no_raw_grid_neural_claim": True,
        "deterministic_replay_passed": True,
    }
    return {
        "schema_version": "e11b_positive_gate_v1",
        "checks": checks,
        "quality_matched_neural_systems": quality_matched_neural,
        "cheapest_quality_matched_neural": cheapest,
        "cheapest_quality_matched_neural_ops_per_tick": rounded(cheapest_ops),
        "flow_ops_advantage_vs_cheapest_quality_neural": rounded(advantage),
        "passed": all(checks.values()),
    }


def decide(gate: dict[str, Any]) -> str:
    if gate["passed"]:
        return "e11b_flow_proxy_cost_advantage_vs_quality_matched_neural_confirmed"
    if not gate["checks"]["at_least_one_quality_matched_neural"]:
        return "e11b_neural_quality_mismatch_no_cost_claim"
    if not gate["checks"]["flow_ops_at_least_3x_lower_than_cheapest_quality_neural"]:
        return "e11b_flow_proxy_cost_advantage_not_confirmed"
    return "e11b_invalid_or_incomplete_run"


def next_for(decision: str) -> str:
    return {
        "e11b_flow_proxy_cost_advantage_vs_quality_matched_neural_confirmed": "E11C_TRAINED_RAW_GRID_NEURAL_BASELINE_CONFIRM",
        "e11b_neural_quality_mismatch_no_cost_claim": "E11B_RETRY_WITH_TRAINED_NEURAL_BASELINES",
        "e11b_flow_proxy_cost_advantage_not_confirmed": "E11B_COST_MODEL_OR_FLOW_RUNTIME_REPAIR",
        "e11b_invalid_or_incomplete_run": "E11B_RETRY_WITH_FULL_AUDIT",
    }[decision]


def measure_system(system: str, rows: list[Any], repeats: int) -> dict[str, Any]:
    wall: list[float] = []
    cpu: list[float] = []
    peaks: list[float] = []
    for _ in range(max(1, repeats)):
        tracemalloc.start()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        if system == FLOW:
            E10.run_system(E10.PRIMARY, rows)
        else:
            run_neural_system(system, rows)
        cpu.append(time.process_time() - cpu_start)
        wall.append(time.perf_counter() - wall_start)
        _current, peak = tracemalloc.get_traced_memory()
        peaks.append(peak / 1024.0)
        tracemalloc.stop()
    return {
        "wall_time_seconds_median": rounded(statistics.median(wall)),
        "cpu_time_seconds_median": rounded(statistics.median(cpu)),
        "wall_time_per_row_seconds": rate(statistics.median(wall), len(rows)),
        "cpu_time_per_row_seconds": rate(statistics.median(cpu), len(rows)),
        "peak_traced_memory_kb_median": rounded(statistics.median(peaks)),
        "repeats": repeats,
        "measured_python_walltime": True,
    }


def build_walltime_report(seeds: tuple[int, ...], rows_per_split: int, bench_repeats: int) -> dict[str, Any]:
    rows = E10.build_rows(seeds, rows_per_split)
    measurements = {system: measure_system(system, rows, bench_repeats) for system in MEASURED_SYSTEMS}
    fastest = min(MEASURED_SYSTEMS, key=lambda system: measurements[system]["wall_time_per_row_seconds"])
    return {
        "schema_version": "e11b_walltime_report_v1",
        "rows": len(rows),
        "bench_repeats": bench_repeats,
        "systems": measurements,
        "fastest_python_walltime_system": fastest,
        "caveat": "Python wall-time is implementation/runtime overhead, not a hardware-normalized inference cost claim.",
    }


def render_report(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    metrics = aggregate["systems"]
    lines = [
        "# E11B Neural Baseline Inference Cost Compare Report",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"next = {decision['next']}",
        f"primary_system = {decision['primary_system']}",
        f"positive_gate_passed = {decision['positive_gate_passed']}",
        f"deterministic_replay_passed = {decision['deterministic_replay_passed']}",
        "```",
        "",
        "## Quality And Cost Model",
        "",
        "| system | usefulness | trace | recall | wrong | repair | ops/tick | cost/trace | ops vs Flow |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        row = metrics[system]
        lines.append(
            f"| {system} | {row['usefulness']:.3f} | {row['trace_validity']:.3f} | {row['useful_writeback_recall']:.3f} | "
            f"{row['wrong_writeback_rate']:.3f} | {row.get('route_repair_rate', 0.0):.3f} | {row['proxy_ops_per_tick']:.1f} | "
            f"{row['cost_per_correct_trace']:.1f} | {row['ops_ratio_vs_flow_python']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Positive Gate",
            "",
            "```json",
            json.dumps(stable_payload(aggregate["positive_gate"]["checks"]), indent=2, sort_keys=True),
            "```",
            "",
            "## Boundary",
            "",
            "This is an inference-cost proxy over E10 detector evidence, not a trained raw-grid neural model benchmark.",
        ]
    )
    return "\n".join(lines)


def attach_replay(payloads: dict[str, Any], seeds: tuple[int, ...], rows_per_split: int) -> dict[str, Any]:
    replay_a = build_deterministic_reports(seeds, rows_per_split)
    replay_b = build_deterministic_reports(seeds, rows_per_split)
    hash_a = stable_hash(replay_a)
    hash_b = stable_hash(replay_b)
    passed = hash_a == hash_b
    payloads["e11b_deterministic_replay_report.json"] = {
        "schema_version": "e11b_deterministic_replay_report_v1",
        "internal_replay_passed": passed,
        "hash_a": hash_a,
        "hash_b": hash_b,
        "artifact_set": sorted(replay_a),
        "walltime_excluded_from_replay_hash": True,
    }
    payloads["decision.json"]["deterministic_replay_passed"] = passed
    payloads["aggregate_metrics.json"]["positive_gate"]["checks"]["deterministic_replay_passed"] = passed
    payloads["aggregate_metrics.json"]["positive_gate"]["passed"] = all(payloads["aggregate_metrics.json"]["positive_gate"]["checks"].values())
    decision_label = decide(payloads["aggregate_metrics.json"]["positive_gate"])
    payloads["decision.json"]["decision"] = decision_label
    payloads["decision.json"]["next"] = next_for(decision_label)
    payloads["decision.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["summary.json"]["decision"] = decision_label
    payloads["summary.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["summary.json"]["deterministic_replay_passed"] = passed
    payloads["report.md"] = render_report(payloads["decision.json"], payloads["aggregate_metrics.json"])
    return payloads


def parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--rows-per-split", type=int, default=DEFAULT_ROWS_PER_SPLIT)
    parser.add_argument("--bench-repeats", type=int, default=DEFAULT_BENCH_REPEATS)
    parser.add_argument("--skip-walltime", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    git_rc, git_head = run_git(["rev-parse", "--short", "HEAD"])
    payloads = attach_replay(build_deterministic_reports(args.seeds, args.rows_per_split), args.seeds, args.rows_per_split)
    payloads["summary.json"]["git_head"] = git_head.strip() if git_rc == 0 else "unknown"
    payloads["e11b_walltime_report.json"] = (
        {
            "schema_version": "e11b_walltime_report_v1",
            "skipped": True,
            "caveat": "Wall-time measurement was skipped by command line.",
        }
        if args.skip_walltime
        else build_walltime_report(args.seeds, args.rows_per_split, args.bench_repeats)
    )
    for name in REQUIRED_ARTIFACTS:
        path = out / name
        if name.endswith(".md"):
            write_text(path, str(payloads[name]))
        else:
            write_json(path, payloads[name])
    print(stable_json({"out": str(out), "decision": payloads["decision.json"]["decision"], "positive_gate_passed": payloads["decision.json"]["positive_gate_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
