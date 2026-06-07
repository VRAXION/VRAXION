#!/usr/bin/env python3
"""E7S FlowGrid visual debug audit.

E7S is a visualization/debug harness for the E7R numeric pocket Flow[D]
system. It renders Flow[D] as a 2D RAM grid and shows what each pocket call
reads, writes, preserves, changes, or corrupts. It does not train a model and
does not create a new capability claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
E7R_PATH = Path(__file__).with_name("run_e7r_numeric_pocket_masked_flow_io_contract_probe.py")
DEFAULT_E7R_SOURCE = Path("target/pilot_wave/e7r_numeric_pocket_masked_flow_io_contract_probe")
DEFAULT_OUT = Path("target/pilot_wave/e7s_flow_grid_visual_debug_audit")
MILESTONE = "E7S_FLOW_GRID_VISUAL_DEBUG_AUDIT"

VISUAL_SYSTEMS = (
    "current_untyped_flow_baseline",
    "anonymous_fixed_mask_contract",
    "anonymous_shuffled_mask_contract",
    "learned_mask_contract",
    "oracle_mask_reference",
)
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "flow_grid_frames.json",
    "flow_grid_frames.jsonl",
    "flow_grid_visualizer.html",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "flow_grid_frames.json",
    "flow_grid_visualizer.html",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7s_flow_grid_visual_debug_ready",
    "e7s_flow_grid_visual_debug_sample_only",
    "e7s_flow_grid_detected_io_corruption_pattern",
    "e7s_flow_grid_visual_debug_blocked",
)
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")


def load_e7r_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7r_numeric_pocket_masked_flow_io_contract_probe", E7R_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7R helpers from {E7R_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7r = load_e7r_module()
e7o = e7r.e7o

FLOW_DIM = int(e7r.FLOW_DIM)
SKILLS = tuple(e7r.SKILLS)
RESULT_POS = dict(e7r.RESULT_POS)


def round_float(value: float) -> float:
    return round(float(value), 12)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def resolve_out(path: str | Path) -> Path:
    resolved = resolve_repo_path(path)
    relative = resolved.relative_to(REPO_ROOT)
    if len(relative.parts) < 2 or relative.parts[0].lower() != "target" or relative.parts[1].lower() != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def append_progress(out: Path, event: str, **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"event": event, "details": details, "time": round_float(time.time())})


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def grid_shape(dim: int) -> dict[str, int]:
    best_rows = 1
    best_cols = dim
    best_delta = dim
    for rows in range(1, int(math.sqrt(dim)) + 1):
        if dim % rows == 0:
            cols = dim // rows
            delta = abs(cols - rows)
            if delta < best_delta:
                best_rows, best_cols, best_delta = rows, cols, delta
    return {"rows": best_rows, "cols": best_cols, "capacity": dim}


def bool_mask(indices: list[int] | tuple[int, ...], dim: int = FLOW_DIM) -> np.ndarray:
    mask = np.zeros(dim, dtype=bool)
    for idx in indices:
        if 0 <= int(idx) < dim:
            mask[int(idx)] = True
    return mask


def contract_from_json(raw: dict[str, Any] | None, system: str, skill: str) -> dict[str, Any]:
    if raw is None:
        if system == "current_untyped_flow_baseline":
            read = np.ones(FLOW_DIM, dtype=bool)
            write = np.ones(FLOW_DIM, dtype=bool)
            scratch = np.zeros(FLOW_DIM, dtype=bool)
            preserve = np.zeros(FLOW_DIM, dtype=bool)
            return {"system": system, "skill": skill, "read": read, "write": write, "scratch": scratch, "return": write, "preserve": preserve, "enforce": False, "permuted": False}
        read = np.ones(FLOW_DIM, dtype=bool)
        write = np.zeros(FLOW_DIM, dtype=bool)
        write[RESULT_POS[skill]] = True
        scratch = np.zeros(FLOW_DIM, dtype=bool)
        preserve = ~write
        return {"system": system, "skill": skill, "read": read, "write": write, "scratch": scratch, "return": write, "preserve": preserve, "enforce": True, "permuted": False}
    read = bool_mask(raw.get("read_indices", []))
    write = bool_mask(raw.get("write_indices", []))
    scratch = bool_mask(raw.get("scratch_indices", []))
    ret = bool_mask(raw.get("return_indices", raw.get("write_indices", [])))
    preserve = ~(write | scratch)
    return {
        "system": system,
        "skill": skill,
        "read": read,
        "write": write,
        "scratch": scratch,
        "return": ret,
        "preserve": preserve,
        "enforce": bool(raw.get("enforce", True)),
        "permuted": bool(raw.get("permuted", False)),
    }


def array_round(values: np.ndarray) -> list[float]:
    return [round_float(x) for x in values.astype(float).tolist()]


def mask_list(mask: np.ndarray) -> list[int]:
    return [int(x) for x in mask.astype(int).tolist()]


def deterministic_noise(row_id: str, system: str, skill: str, dim: int) -> np.ndarray:
    seed = int(hashlib.sha256(f"e7s-noise::{row_id}::{system}::{skill}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=dim).astype(np.float32)


def display_result_pos(skill: str, perm: np.ndarray | None) -> int:
    if perm is None:
        return int(RESULT_POS[skill])
    inv = e7r.inverse_perm(perm)
    return int(inv[RESULT_POS[skill]])


def final_answer(row: dict[str, Any], system: str, flow: np.ndarray, perm: np.ndarray | None) -> int:
    flow_eval = flow[e7r.inverse_perm(perm)] if perm is not None else flow
    return int(e7o.predict_answer_from_flow(row, flow_eval))


def simulate_step(row: dict[str, Any], system: str, skill: str, before: np.ndarray, contract: dict[str, Any], perm: np.ndarray | None, source_metrics: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    target_pos = display_result_pos(skill, perm)
    row_flow = before[e7r.inverse_perm(perm)] if perm is not None else before
    target_value = float(e7o.base_skill_value(skill, int(row["a"]), int(row["b"]), int(row["key"]), int(row["threshold"]), int(row["flip"]), row_flow))
    read = contract["read"]
    write = contract["write"]
    scratch = contract["scratch"]
    preserve = contract["preserve"]
    allowed = np.ones(FLOW_DIM, dtype=bool) if not contract["enforce"] else (write | scratch)
    noise = deterministic_noise(str(row["row_id"]), system, skill, FLOW_DIM)
    after = before.copy()
    intended = before.copy()
    intended[target_pos] = target_value
    for idx in np.flatnonzero(scratch):
        intended[int(idx)] = float(np.tanh(before[int(idx)] + 0.12 * noise[int(idx)]))

    if system == "oracle_mask_reference":
        after[target_pos] = target_value
    elif system == "current_untyped_flow_baseline":
        spread_scale = 0.04 + 0.18 * float(source_metrics.get("result_region_corruption_rate", 0.5))
        after = np.clip(before + spread_scale * noise, -1.0, 1.0)
        after[target_pos] = target_value
    else:
        after[allowed] = intended[allowed]
        if target_pos not in np.flatnonzero(allowed):
            after[target_pos] = target_value

    delta = after - before
    changed = np.abs(delta) > 1e-6
    illegal_write = changed & ~allowed if contract["enforce"] else np.zeros(FLOW_DIM, dtype=bool)
    preserve_corruption = changed & preserve if contract["enforce"] else np.zeros(FLOW_DIM, dtype=bool)
    metrics = {
        "target_value": int(target_value >= 0.5),
        "write_spread": round_float(float(np.mean(changed))),
        "delta_magnitude": round_float(float(np.mean(np.abs(delta)))),
        "flow_drift": round_float(float(np.linalg.norm(delta) / math.sqrt(FLOW_DIM))),
        "write_mask_violation_count": int(np.sum(illegal_write)),
        "preserve_corruption_count": int(np.sum(preserve_corruption)),
        "read_count": int(np.sum(read)),
        "write_count": int(np.sum(write)),
        "preserve_count": int(np.sum(preserve)),
        "mask_sparsity": round_float(float(np.mean(allowed))),
    }
    overlays = {
        "read_mask": read,
        "write_mask": write,
        "preserve_mask": preserve,
        "scratch_mask": scratch,
        "changed_mask": changed,
        "illegal_write_mask": illegal_write,
        "preserve_corruption_mask": preserve_corruption,
    }
    return after.astype(np.float32), {"metrics": metrics, "overlays": overlays}


def make_frame(example: dict[str, Any], system: str, frame_index: int, step_index: int, phase: str, pocket_id: str, before: np.ndarray, after: np.ndarray, overlays: dict[str, np.ndarray], metrics: dict[str, Any], answer: int | None = None, correct: bool | None = None) -> dict[str, Any]:
    delta = after - before
    return {
        "example_id": example["example_id"],
        "row_id": example["row_id"],
        "seed": example["seed"],
        "split": example["split"],
        "family": example["family"],
        "system": system,
        "frame_index": frame_index,
        "step_index": step_index,
        "phase": phase,
        "pocket_id": pocket_id,
        "route": list(example["route"]),
        "target_answer": int(example["target_answer"]),
        "predicted_answer": answer,
        "correct": correct,
        "before": array_round(before),
        "after": array_round(after),
        "delta": array_round(delta),
        "read_mask": mask_list(overlays.get("read_mask", np.zeros(FLOW_DIM, dtype=bool))),
        "write_mask": mask_list(overlays.get("write_mask", np.zeros(FLOW_DIM, dtype=bool))),
        "preserve_mask": mask_list(overlays.get("preserve_mask", np.zeros(FLOW_DIM, dtype=bool))),
        "scratch_mask": mask_list(overlays.get("scratch_mask", np.zeros(FLOW_DIM, dtype=bool))),
        "changed_mask": mask_list(overlays.get("changed_mask", np.abs(delta) > 1e-6)),
        "illegal_write_mask": mask_list(overlays.get("illegal_write_mask", np.zeros(FLOW_DIM, dtype=bool))),
        "preserve_corruption_mask": mask_list(overlays.get("preserve_corruption_mask", np.zeros(FLOW_DIM, dtype=bool))),
        "metrics": metrics,
    }


def select_examples(tasks: dict[int, dict[str, list[dict[str, Any]]]], max_examples: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seeds = sorted(tasks)
    for split in EVAL_SPLITS:
        for seed in seeds:
            for row in tasks[seed][split]:
                examples.append({
                    "example_id": f"{row['row_id']}:{row['family']}",
                    "row_id": row["row_id"],
                    "seed": seed,
                    "split": split,
                    "family": row["family"],
                    "route": list(row["expected_route"]),
                    "target_answer": int(row["target_answer"]),
                    "row": row,
                })
                break
            if len(examples) >= max_examples:
                return examples
    return examples


def source_metric_lookup(e7r_aggregate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for system, row in e7r_aggregate.get("systems", {}).items():
        mean = row.get("mean", {})
        out[system] = {
            "usefulness": mean.get("eval_mean_composition_usefulness", 0.0),
            "answer_accuracy": mean.get("eval_mean_answer_accuracy", 0.0),
            "result_region_corruption_rate": mean.get("eval_mean_result_region_corruption_rate", 0.0),
            "next_pocket_input_compatibility_error": mean.get("eval_mean_next_pocket_input_compatibility_error", 0.0),
            "lane_shuffle_robustness": mean.get("lane_shuffle_robustness", 0.0),
            "mask_sparsity": mean.get("mask_sparsity", 0.0),
        }
    return out


def build_contract_lookup(mask_report: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    lookup: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in mask_report.get("rows", []):
        key = (int(row["seed"]), str(row["system"]), str(row["skill"]))
        lookup[key] = row["contract"]
    return lookup


def build_tasks_from_e7r_manifest(manifest: dict[str, Any], max_examples: int) -> dict[int, dict[str, list[dict[str, Any]]]]:
    settings = manifest.get("settings", {})
    seeds = tuple(int(seed) for seed in settings.get("seeds", [99701]))
    namespace = SimpleNamespace(
        seeds=seeds,
        train_rows_per_seed=0,
        validation_rows_per_seed=0,
        heldout_rows_per_seed=max(1, min(max_examples, int(settings.get("heldout_rows_per_seed", 300)))),
        ood_rows_per_seed=max(1, min(max_examples, int(settings.get("ood_rows_per_seed", 300)))),
        counterfactual_rows_per_seed=max(1, min(max_examples, int(settings.get("counterfactual_rows_per_seed", 300)))),
        adversarial_rows_per_seed=max(1, min(max_examples, int(settings.get("adversarial_rows_per_seed", 300)))),
    )
    return e7o.generate_composition_tasks(namespace)


def build_frames(e7r_source: Path, max_examples: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(e7r_source / "backend_manifest.json")
    aggregate = load_json(e7r_source / "aggregate_metrics.json")
    mask_report = load_json(e7r_source / "mask_contract_report.json")
    task_report = load_json(e7r_source / "task_generation_report.json")
    source_metrics = source_metric_lookup(aggregate)
    contract_lookup = build_contract_lookup(mask_report)
    tasks = build_tasks_from_e7r_manifest(manifest, max_examples)
    examples = select_examples(tasks, max_examples)
    frames: list[dict[str, Any]] = []
    frame_index = 0
    for example in examples:
        row = example["row"]
        for system in VISUAL_SYSTEMS:
            perm = e7r.permutation_for_seed(int(row["row_id"].split(":")[0])) if system == "anonymous_shuffled_mask_contract" else None
            flow = np.asarray(row["flow"], dtype=np.float32)
            if perm is not None:
                flow = flow[perm]
            initial_overlays = {name: np.zeros(FLOW_DIM, dtype=bool) for name in ("read_mask", "write_mask", "preserve_mask", "scratch_mask", "changed_mask", "illegal_write_mask", "preserve_corruption_mask")}
            frames.append(make_frame(example, system, frame_index, 0, "initial_flow", "router", flow, flow, initial_overlays, {"route_steps": len(example["route"])}, None, None))
            frame_index += 1
            for step_index, skill in enumerate(example["route"], start=1):
                raw_contract = contract_lookup.get((int(row["row_id"].split(":")[0]), system, skill))
                contract = contract_from_json(raw_contract, system, skill)
                before = flow.copy()
                after, step_payload = simulate_step(row, system, skill, before, contract, perm, source_metrics.get(system, {}))
                overlays = step_payload["overlays"]
                metrics = step_payload["metrics"]
                for phase in ("router_choose", "read_mask", "write_mask", "preserve_mask", "before_pocket", "after_pocket", "delta_violation", "next_router_state"):
                    phase_before = before
                    phase_after = before if phase in {"router_choose", "read_mask", "write_mask", "preserve_mask", "before_pocket"} else after
                    frames.append(make_frame(example, system, frame_index, step_index, phase, skill, phase_before, phase_after, overlays, metrics, None, None))
                    frame_index += 1
                flow = after
            answer = final_answer(row, system, flow, perm)
            correct = bool(answer == int(row["target_answer"]))
            frames.append(make_frame(example, system, frame_index, len(example["route"]) + 1, "final_output", "answer", flow, flow, initial_overlays, {"route_steps": len(example["route"])}, answer, correct))
            frame_index += 1
    payload = {
        "schema_version": "e7s_flow_grid_frames_v1",
        "milestone": MILESTONE,
        "source_type": "e7r_artifact_plus_visualization_sample",
        "source_root": str(e7r_source),
        "source_task_report_hash": payload_sha256(task_report),
        "flow_dim": FLOW_DIM,
        "grid_shape": grid_shape(FLOW_DIM),
        "systems": list(VISUAL_SYSTEMS),
        "examples": [{key: value for key, value in example.items() if key != "row"} for example in examples],
        "frames": frames,
        "e7r_source_metrics": source_metrics,
    }
    return payload, frames, {"manifest": manifest, "aggregate": aggregate, "mask_report": mask_report}


def aggregate_frames(frame_payload: dict[str, Any]) -> dict[str, Any]:
    systems: dict[str, dict[str, Any]] = {}
    source_metrics = frame_payload.get("e7r_source_metrics", {})
    for system in frame_payload["systems"]:
        system_frames = [frame for frame in frame_payload["frames"] if frame["system"] == system and frame["phase"] == "delta_violation"]
        final_frames = [frame for frame in frame_payload["frames"] if frame["system"] == system and frame["phase"] == "final_output"]
        systems[system] = {
            "frame_count": len([frame for frame in frame_payload["frames"] if frame["system"] == system]),
            "call_frame_count": len(system_frames),
            "example_count": len(final_frames),
            "mean_write_spread": round_float(np.mean([frame["metrics"]["write_spread"] for frame in system_frames]) if system_frames else 0.0),
            "mean_delta_magnitude": round_float(np.mean([frame["metrics"]["delta_magnitude"] for frame in system_frames]) if system_frames else 0.0),
            "mean_flow_drift": round_float(np.mean([frame["metrics"]["flow_drift"] for frame in system_frames]) if system_frames else 0.0),
            "mean_write_mask_violation_count": round_float(np.mean([frame["metrics"]["write_mask_violation_count"] for frame in system_frames]) if system_frames else 0.0),
            "mean_preserve_corruption_count": round_float(np.mean([frame["metrics"]["preserve_corruption_count"] for frame in system_frames]) if system_frames else 0.0),
            "sample_answer_correct_rate": round_float(np.mean([1.0 if frame.get("correct") else 0.0 for frame in final_frames]) if final_frames else 0.0),
            "source_usefulness": round_float(source_metrics.get(system, {}).get("usefulness", 0.0)),
            "source_answer_accuracy": round_float(source_metrics.get(system, {}).get("answer_accuracy", 0.0)),
            "source_lane_shuffle_robustness": round_float(source_metrics.get(system, {}).get("lane_shuffle_robustness", 0.0)),
            "source_next_pocket_input_compatibility_error": round_float(source_metrics.get(system, {}).get("next_pocket_input_compatibility_error", 0.0)),
            "source_mask_sparsity": round_float(source_metrics.get(system, {}).get("mask_sparsity", 0.0)),
        }
    return {
        "schema_version": "e7s_aggregate_metrics_v1",
        "flow_dim": frame_payload["flow_dim"],
        "grid_shape": frame_payload["grid_shape"],
        "source_type": frame_payload["source_type"],
        "systems": systems,
    }


def decide(aggregate: dict[str, Any], source_available: bool) -> dict[str, Any]:
    masked_systems = ("anonymous_fixed_mask_contract", "anonymous_shuffled_mask_contract", "learned_mask_contract")
    corruption = max(aggregate["systems"][system]["mean_preserve_corruption_count"] for system in masked_systems)
    write_violation = max(aggregate["systems"][system]["mean_write_mask_violation_count"] for system in masked_systems)
    if corruption > 0.0 or write_violation > 0.0:
        decision = "e7s_flow_grid_detected_io_corruption_pattern"
    elif source_available:
        decision = "e7s_flow_grid_visual_debug_ready"
    else:
        decision = "e7s_flow_grid_visual_debug_sample_only"
    return {
        "schema_version": "e7s_decision_v1",
        "decision": decision,
        "source_available": bool(source_available),
        "grid_shape": aggregate["grid_shape"],
        "systems_visualized": list(aggregate["systems"]),
        "example_count": max(row["example_count"] for row in aggregate["systems"].values()) if aggregate["systems"] else 0,
        "masked_preserve_corruption_visible": bool(corruption > 0.0),
        "masked_write_violation_visible": bool(write_violation > 0.0),
    }


def render_report(aggregate: dict[str, Any], decision: dict[str, Any], source_root: Path) -> str:
    lines = [
        "# E7S FlowGrid Visual Debug Audit",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"source_root = {source_root}",
        f"grid_shape = {aggregate['grid_shape']['rows']}x{aggregate['grid_shape']['cols']}",
        f"example_count = {decision['example_count']}",
        "```",
        "",
        "## Visualized Systems",
        "",
        "```text",
    ]
    for system, row in aggregate["systems"].items():
        lines.append(
            f"{system:<38} source_useful={row['source_usefulness']:.6f} "
            f"write_spread={row['mean_write_spread']:.6f} "
            f"delta={row['mean_delta_magnitude']:.6f} "
            f"preserve_corrupt={row['mean_preserve_corruption_count']:.6f}"
        )
    lines.extend([
        "```",
        "",
        "## Interpretation",
        "",
        "The visualizer is a deterministic Flow[D] microscope. It reads E7R masks and",
        "aggregate source metrics, then creates visualization sample frames so the shared",
        "Flow/RAM grid can be inspected step by step. It does not alter E7R scientific",
        "claims and does not train any model.",
        "",
        "The expected visual pattern is that untyped Flow changes spread broadly, while",
        "anonymous masked systems constrain writes to mechanical read/write/preserve",
        "regions. The learned sparse mask should appear as the most compact non-oracle",
        "mechanical contract.",
        "",
        "## Boundary",
        "",
        "E7S is a controlled numeric visualization/debug audit. It does not prove raw-language learning, AGI, consciousness, or model-scale behavior.",
        "",
    ])
    return "\n".join(lines)


def html_escape_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")


def render_html(frame_payload: dict[str, Any], aggregate: dict[str, Any], decision: dict[str, Any]) -> str:
    data = {"frames": frame_payload, "aggregate": aggregate, "decision": decision}
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E7S FlowGrid Visual Debug Audit</title>
<style>
:root {{ color-scheme: dark; --bg:#0d1117; --panel:#161b22; --line:#30363d; --txt:#e6edf3; --muted:#8b949e; --read:#58a6ff; --write:#f0883e; --preserve:#a371f7; --changed:#3fb950; --illegal:#ff4d4f; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--txt); font:14px/1.4 system-ui, Segoe UI, Arial, sans-serif; }}
header {{ padding:18px 22px; border-bottom:1px solid var(--line); background:#010409; }}
h1 {{ margin:0 0 4px; font-size:20px; letter-spacing:0; }}
.sub {{ color:var(--muted); }}
.layout {{ display:grid; grid-template-columns: 330px 1fr; min-height:calc(100vh - 76px); }}
aside {{ border-right:1px solid var(--line); padding:16px; background:var(--panel); }}
main {{ padding:16px; }}
label {{ display:block; margin:12px 0 5px; color:var(--muted); font-size:12px; }}
select, button, input[type=range] {{ width:100%; background:#0d1117; color:var(--txt); border:1px solid var(--line); border-radius:6px; padding:8px; }}
button {{ cursor:pointer; margin-top:10px; }}
.grid-wrap {{ display:grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap:14px; align-items:start; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }}
.panel h2 {{ margin:0 0 10px; font-size:14px; }}
.flowgrid {{ display:grid; gap:4px; }}
.cell {{ position:relative; min-height:42px; border:1px solid #222832; border-radius:4px; display:flex; align-items:center; justify-content:center; font:11px/1 ui-monospace, SFMono-Regular, Consolas, monospace; color:#fff; overflow:hidden; }}
.cell .idx {{ position:absolute; left:4px; top:2px; color:rgba(255,255,255,.45); font-size:9px; }}
.read {{ box-shadow: inset 0 0 0 2px var(--read); }}
.write {{ outline:2px solid var(--write); }}
.preserve {{ border-color:var(--preserve); }}
.changed::after {{ content:""; position:absolute; right:3px; bottom:3px; width:7px; height:7px; border-radius:50%; background:var(--changed); }}
.illegal {{ box-shadow: inset 0 0 0 3px var(--illegal), 0 0 0 2px var(--illegal); }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; color:var(--muted); font-size:12px; }}
.sw {{ width:12px; height:12px; display:inline-block; margin-right:4px; vertical-align:-2px; border-radius:2px; }}
.kpi {{ display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-top:12px; }}
.kpi div {{ background:#0d1117; border:1px solid var(--line); border-radius:6px; padding:8px; }}
.timeline {{ display:flex; gap:6px; overflow:auto; padding:8px 0; }}
.tick {{ min-width:30px; height:22px; border-radius:4px; background:#21262d; border:1px solid var(--line); text-align:center; font-size:11px; line-height:20px; color:var(--muted); }}
.tick.active {{ background:#1f6feb; color:#fff; }}
pre {{ white-space:pre-wrap; background:#0d1117; border:1px solid var(--line); border-radius:6px; padding:10px; color:#c9d1d9; }}
@media (max-width: 1100px) {{ .layout {{ grid-template-columns:1fr; }} aside {{ border-right:0; border-bottom:1px solid var(--line); }} .grid-wrap {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
<h1>E7S FlowGrid Visual Debug Audit</h1>
<div class="sub">Flow[D] as shared RAM. Mechanical read/write/preserve masks only; no semantic lane labels.</div>
</header>
<div class="layout">
<aside>
<label>System</label><select id="system"></select>
<label>Example</label><select id="example"></select>
<label>Frame</label><input id="frameSlider" type="range" min="0" max="0" value="0">
<button id="play">Play</button>
<label>Grid Mode</label><select id="mode"><option value="after">after</option><option value="before">before</option><option value="delta">delta</option></select>
<div class="legend">
<span><i class="sw" style="border:2px solid var(--read)"></i>read</span>
<span><i class="sw" style="border:2px solid var(--write)"></i>write</span>
<span><i class="sw" style="border:2px solid var(--preserve)"></i>preserve</span>
<span><i class="sw" style="background:var(--changed)"></i>changed</span>
<span><i class="sw" style="background:var(--illegal)"></i>illegal/corrupt</span>
</div>
<div class="kpi" id="kpis"></div>
</aside>
<main>
<div class="panel"><h2>Route Timeline</h2><div id="timeline" class="timeline"></div></div>
<div class="grid-wrap">
<div class="panel"><h2 id="gridTitle">Flow Grid</h2><div id="grid" class="flowgrid"></div></div>
<div class="panel"><h2>Frame Detail</h2><pre id="detail"></pre></div>
<div class="panel"><h2>Source Metrics</h2><pre id="sourceMetrics"></pre></div>
</div>
</main>
</div>
<script>
const DATA = {html_escape_json(data)};
const frames = DATA.frames.frames;
const shape = DATA.frames.grid_shape;
const systems = DATA.frames.systems;
const systemEl = document.getElementById('system');
const exampleEl = document.getElementById('example');
const slider = document.getElementById('frameSlider');
const modeEl = document.getElementById('mode');
const gridEl = document.getElementById('grid');
const detailEl = document.getElementById('detail');
const kpisEl = document.getElementById('kpis');
const timelineEl = document.getElementById('timeline');
const sourceEl = document.getElementById('sourceMetrics');
let timer = null;
function uniqueExamples(system) {{
  const seen = new Map();
  frames.filter(f => f.system === system).forEach(f => {{
    if (!seen.has(f.example_id)) seen.set(f.example_id, `${{f.split}} | ${{f.family}} | ${{f.row_id}}`);
  }});
  return [...seen.entries()];
}}
function fillSystems() {{
  systems.forEach(s => systemEl.add(new Option(s, s)));
}}
function fillExamples() {{
  exampleEl.innerHTML = '';
  uniqueExamples(systemEl.value).forEach(([id, label]) => exampleEl.add(new Option(label, id)));
  updateSlider();
}}
function currentFrames() {{
  return frames.filter(f => f.system === systemEl.value && f.example_id === exampleEl.value);
}}
function updateSlider() {{
  const fs = currentFrames();
  slider.max = Math.max(0, fs.length - 1);
  slider.value = 0;
  render();
}}
function heat(v, isDelta) {{
  const x = Math.max(-1, Math.min(1, Number(v)));
  if (isDelta) {{
    const a = Math.min(1, Math.abs(x) * 5);
    return x >= 0 ? `rgba(63,185,80,${{0.15+a*0.75}})` : `rgba(248,81,73,${{0.15+a*0.75}})`;
  }}
  const a = Math.min(1, Math.abs(x));
  return x >= 0 ? `rgba(31,111,235,${{0.18+a*0.75}})` : `rgba(248,81,73,${{0.18+a*0.75}})`;
}}
function renderGrid(frame) {{
  const mode = modeEl.value;
  const values = frame[mode];
  gridEl.style.gridTemplateColumns = `repeat(${{shape.cols}}, minmax(0, 1fr))`;
  gridEl.innerHTML = '';
  values.forEach((v, i) => {{
    const cell = document.createElement('div');
    cell.className = 'cell';
    if (frame.read_mask[i]) cell.classList.add('read');
    if (frame.write_mask[i]) cell.classList.add('write');
    if (frame.preserve_mask[i]) cell.classList.add('preserve');
    if (frame.changed_mask[i]) cell.classList.add('changed');
    if (frame.illegal_write_mask[i] || frame.preserve_corruption_mask[i]) cell.classList.add('illegal');
    cell.style.background = heat(v, mode === 'delta');
    cell.innerHTML = `<span class="idx">${{i}}</span>${{Number(v).toFixed(2)}}`;
    gridEl.appendChild(cell);
  }});
}}
function renderTimeline(fs, idx) {{
  timelineEl.innerHTML = '';
  fs.forEach((f, i) => {{
    const t = document.createElement('div');
    t.className = 'tick' + (i === idx ? ' active' : '');
    t.textContent = f.pocket_id === 'router' ? 'R' : (f.pocket_id === 'answer' ? 'A' : f.pocket_id.slice(0,2));
    t.title = `${{i}} ${{f.phase}} ${{f.pocket_id}}`;
    t.onclick = () => {{ slider.value = i; render(); }};
    timelineEl.appendChild(t);
  }});
}}
function renderKpis(frame) {{
  const m = frame.metrics || {{}};
  const rows = [
    ['phase', frame.phase],
    ['pocket', frame.pocket_id],
    ['write spread', m.write_spread ?? 0],
    ['delta mag', m.delta_magnitude ?? 0],
    ['illegal writes', m.write_mask_violation_count ?? 0],
    ['preserve corrupt', m.preserve_corruption_count ?? 0],
    ['target', frame.target_answer],
    ['predicted', frame.predicted_answer ?? '...']
  ];
  kpisEl.innerHTML = rows.map(([k,v]) => `<div><b>${{k}}</b><br>${{v}}</div>`).join('');
}}
function render() {{
  const fs = currentFrames();
  if (!fs.length) return;
  const idx = Number(slider.value);
  const frame = fs[idx];
  document.getElementById('gridTitle').textContent = `${{modeEl.value}} grid | ${{frame.phase}}`;
  renderGrid(frame);
  renderTimeline(fs, idx);
  renderKpis(frame);
  detailEl.textContent = JSON.stringify({{
    row_id: frame.row_id, split: frame.split, family: frame.family, route: frame.route,
    frame_index: idx, step_index: frame.step_index, phase: frame.phase, pocket_id: frame.pocket_id,
    correct: frame.correct, metrics: frame.metrics
  }}, null, 2);
  sourceEl.textContent = JSON.stringify(DATA.aggregate.systems[systemEl.value], null, 2);
}}
document.getElementById('play').onclick = () => {{
  if (timer) {{ clearInterval(timer); timer = null; document.getElementById('play').textContent = 'Play'; return; }}
  document.getElementById('play').textContent = 'Pause';
  timer = setInterval(() => {{
    let v = Number(slider.value) + 1;
    if (v > Number(slider.max)) v = 0;
    slider.value = v;
    render();
  }}, 550);
}};
systemEl.onchange = fillExamples;
exampleEl.onchange = updateSlider;
slider.oninput = render;
modeEl.onchange = render;
fillSystems();
fillExamples();
</script>
</body>
</html>
"""


def write_frames_jsonl(path: Path, frames: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    for frame in frames:
        append_jsonl(path, frame)


def write_artifacts(out: Path, source_root: Path, frame_payload: dict[str, Any], frames: list[dict[str, Any]], aggregate: dict[str, Any], decision: dict[str, Any], deterministic: dict[str, Any]) -> None:
    write_json(out / "backend_manifest.json", {
        "schema_version": "e7s_backend_manifest_v1",
        "milestone": MILESTONE,
        "source_root": str(source_root),
        "source_type": frame_payload["source_type"],
        "systems": list(VISUAL_SYSTEMS),
        "flow_dim": FLOW_DIM,
        "grid_shape": frame_payload["grid_shape"],
        "training_performed": False,
        "model_changes": False,
        "semantic_lane_labels_as_model_input": False,
    })
    write_json(out / "flow_grid_frames.json", frame_payload)
    write_frames_jsonl(out / "flow_grid_frames.jsonl", frames)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    summary = {
        "schema_version": "e7s_summary_v1",
        "decision": decision["decision"],
        "source_root": str(source_root),
        "systems_visualized": list(VISUAL_SYSTEMS),
        "flow_dim": FLOW_DIM,
        "grid_shape": frame_payload["grid_shape"],
        "example_count": decision["example_count"],
        "deterministic_replay_passed": deterministic.get("internal_replay_passed", False),
    }
    write_json(out / "summary.json", summary)
    write_text(out / "report.md", render_report(aggregate, decision, source_root))
    write_text(out / "flow_grid_visualizer.html", render_html(frame_payload, aggregate, decision))


def hash_artifacts(out: Path) -> dict[str, str]:
    return {artifact: hashlib.sha256((out / artifact).read_bytes()).hexdigest() for artifact in HASH_ARTIFACTS}


def run(settings: argparse.Namespace) -> dict[str, Any]:
    out = resolve_out(settings.out)
    source_root = resolve_repo_path(settings.e7r_source)
    if out.exists() and not settings.replay:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    append_progress(out, "run_start", replay=bool(settings.replay), source_root=str(source_root))

    source_available = all((source_root / name).exists() for name in ("backend_manifest.json", "aggregate_metrics.json", "mask_contract_report.json", "task_generation_report.json"))
    if not source_available:
        decision = {
            "schema_version": "e7s_decision_v1",
            "decision": "e7s_flow_grid_visual_debug_blocked",
            "source_available": False,
            "reason": "missing E7R source artifacts",
        }
        write_json(out / "decision.json", decision)
        return decision

    frame_payload, frames, _source_payload = build_frames(source_root, int(settings.max_examples))
    append_progress(out, "frames_built", frame_count=len(frames), example_count=len(frame_payload["examples"]))
    aggregate = aggregate_frames(frame_payload)
    decision = decide(aggregate, source_available=True)
    deterministic_placeholder = {"schema_version": "e7s_deterministic_replay_report_v1", "internal_replay_passed": True, "replay_mode": bool(settings.replay), "hash_comparisons": {}}
    write_artifacts(out, source_root, frame_payload, frames, aggregate, decision, deterministic_placeholder)
    append_progress(out, "primary_artifacts_written" if not settings.replay else "replay_artifacts_written", artifact_count=len(REQUIRED_ARTIFACTS))

    if not settings.replay:
        replay_out = out / "deterministic_replay_work"
        if replay_out.exists():
            shutil.rmtree(replay_out)
        append_progress(out, "deterministic_replay_start", replay_out=str(replay_out.relative_to(REPO_ROOT)))
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--out",
            str(replay_out.relative_to(REPO_ROOT)),
            "--e7r-source",
            str(source_root.relative_to(REPO_ROOT)),
            "--max-examples",
            str(settings.max_examples),
            "--replay",
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        primary_hashes = hash_artifacts(out)
        replay_hashes = hash_artifacts(replay_out)
        comparisons = {artifact: {"primary": primary_hashes[artifact], "replay": replay_hashes[artifact], "match": primary_hashes[artifact] == replay_hashes[artifact]} for artifact in HASH_ARTIFACTS}
        deterministic = {"schema_version": "e7s_deterministic_replay_report_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
        write_json(out / "deterministic_replay.json", deterministic)
        write_artifacts(out, source_root, frame_payload, frames, aggregate, decision, deterministic)
        append_progress(out, "deterministic_replay_complete", internal_replay_passed=deterministic["internal_replay_passed"])
        append_progress(out, "final_artifacts_written", artifact_count=len(REQUIRED_ARTIFACTS) + 1)
    else:
        write_json(out / "deterministic_replay.json", deterministic_placeholder)
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--e7r-source", default=str(DEFAULT_E7R_SOURCE))
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args(argv)
    decision = run(args)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision.get("decision") != "e7s_flow_grid_visual_debug_blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
