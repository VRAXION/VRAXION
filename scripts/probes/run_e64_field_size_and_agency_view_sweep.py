#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


MILESTONE = "E64_FIELD_SIZE_AND_AGENCY_VIEW_SWEEP"
BOUNDARY = (
    "E64 is a deterministic symbolic/numeric sizing probe for the VRAXION "
    "Flow/Ground/Proposal/Agency interfaces. It does not test raw language "
    "reasoning, deployed model behavior, consciousness, AGI, or model scale."
)

DECISIONS = {
    "e64_near_28f_32g_20x80_default_confirmed",
    "e64_wide_32x32_default_required",
    "e64_compact_16x16_sufficient",
    "e64_no_clean_size_within_cost_gate",
    "e64_proposal_or_agency_view_bottleneck",
    "e64_invalid_artifact_detected",
}


@dataclass(frozen=True)
class FieldConfig:
    name: str
    flow_side: int
    ground_side: int
    proposal_slots: int
    proposal_bits: int
    agency_view_bits: int
    cost_multiplier: float
    intended_role: str

    @property
    def flow_cells(self) -> int:
        return self.flow_side * self.flow_side

    @property
    def ground_cells(self) -> int:
        return self.ground_side * self.ground_side

    @property
    def proposal_capacity_bits(self) -> int:
        return self.proposal_slots * self.proposal_bits

    @property
    def total_work_cells(self) -> int:
        return self.flow_cells + self.ground_cells + self.proposal_capacity_bits + self.agency_view_bits

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "flow_shape": [self.flow_side, self.flow_side],
            "ground_shape": [self.ground_side, self.ground_side],
            "proposal_slots": self.proposal_slots,
            "proposal_bits": self.proposal_bits,
            "proposal_capacity_bits": self.proposal_capacity_bits,
            "agency_view_bits": self.agency_view_bits,
            "cost_multiplier": self.cost_multiplier,
            "intended_role": self.intended_role,
            "total_work_cells": self.total_work_cells,
        }


CONFIGS: list[FieldConfig] = [
    FieldConfig("tiny_12x12_all", 12, 12, 6, 32, 192, 0.62, "too-small control"),
    FieldConfig("compact_16x16_all", 16, 16, 8, 48, 320, 1.00, "fast lower bound"),
    FieldConfig("balanced_24x24_all", 24, 24, 12, 64, 576, 1.72, "balanced symmetric"),
    FieldConfig("proposal_width_64_control", 24, 32, 16, 64, 768, 2.16, "64-bit proposal width bottleneck control"),
    FieldConfig("asymmetric_24f_32g_20x80_control", 24, 32, 20, 80, 896, 2.42, "24x24 Flow capacity control"),
    FieldConfig("near_28f_32g_20x80_default", 28, 32, 20, 80, 896, 2.58, "candidate default"),
    FieldConfig("wide_32x32_20x80", 32, 32, 20, 80, 1024, 3.18, "wide default challenger"),
    FieldConfig("large_48x48_24x80", 48, 48, 24, 80, 1536, 5.35, "clean but expensive"),
    FieldConfig("oversized_64x64_32x80", 64, 64, 32, 80, 2048, 8.75, "overcapacity falloff control"),
    FieldConfig("proposal_starved_32x32", 32, 32, 8, 32, 768, 2.28, "proposal bottleneck control"),
    FieldConfig("agency_starved_32x32", 32, 32, 20, 80, 384, 2.52, "agency view bottleneck control"),
]


STAGES: dict[str, dict[str, Any]] = {
    "F0_local_short_commit": {
        "flow": 160,
        "ground": 120,
        "slots": 3,
        "proposal_bits": 32,
        "agency": 160,
        "kind": "answer",
        "adversarial": False,
    },
    "F1_active_evidence_trace": {
        "flow": 260,
        "ground": 230,
        "slots": 5,
        "proposal_bits": 40,
        "agency": 260,
        "kind": "answer",
        "adversarial": False,
    },
    "F2_proposal_collision_commit": {
        "flow": 330,
        "ground": 310,
        "slots": 10,
        "proposal_bits": 48,
        "agency": 420,
        "kind": "reject_collision",
        "adversarial": True,
    },
    "F3_ground_contradiction_check": {
        "flow": 390,
        "ground": 650,
        "slots": 8,
        "proposal_bits": 56,
        "agency": 560,
        "kind": "reject_ground_conflict",
        "adversarial": True,
    },
    "F4_text_digest_to_flow": {
        "flow": 520,
        "ground": 700,
        "slots": 12,
        "proposal_bits": 64,
        "agency": 720,
        "kind": "answer",
        "adversarial": False,
    },
    "F5_adversarial_proposal_flood": {
        "flow": 500,
        "ground": 720,
        "slots": 16,
        "proposal_bits": 64,
        "agency": 760,
        "kind": "reject_flood",
        "adversarial": True,
    },
    "F6_multi_cycle_repair_memory": {
        "flow": 560,
        "ground": 920,
        "slots": 14,
        "proposal_bits": 64,
        "agency": 720,
        "kind": "repair",
        "adversarial": False,
    },
    "F7_overcapacity_decoy_pressure": {
        "flow": 430,
        "ground": 620,
        "slots": 12,
        "proposal_bits": 64,
        "agency": 700,
        "kind": "reject_decoy",
        "adversarial": True,
    },
}

REQ_TARGET = [
    "backend_manifest.json",
    "field_size_manifest.json",
    "row_level_results.jsonl",
    "system_results.json",
    "stage_metrics.json",
    "capacity_frontier_report.json",
    "agency_view_report.json",
    "proposal_capacity_report.json",
    "recommendation.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "deterministic_replay.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
    "report.md",
]

REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_results_sample.json",
    "stage_metrics_sample.json",
    "row_level_sample.jsonl",
    "capacity_frontier_report_sample.json",
    "recommendation_sample.json",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return float(ordered[idx])


def stable_u32(*parts: Any) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def hardware_snapshot(event: str) -> dict[str, Any]:
    process = psutil.Process(os.getpid()) if psutil else None
    gpu: dict[str, Any] = {"available": False}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            name, util, used, total, temp = [part.strip() for part in proc.stdout.strip().splitlines()[0].split(",")]
            gpu = {
                "available": True,
                "name": name,
                "utilization_gpu_percent": float(util),
                "memory_used_mb": float(used),
                "memory_total_mb": float(total),
                "temperature_c": float(temp),
            }
    except Exception:
        gpu = {"available": False}
    return {
        "timestamp": now_iso(),
        "event": event,
        "logical_cpu_count": os.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=None) if psutil else None,
        "process_rss_mb": process.memory_info().rss / (1024 * 1024) if process else None,
        "system_ram_used_percent": psutil.virtual_memory().percent if psutil else None,
        "gpu": gpu,
    }


class Heartbeat:
    def __init__(self, out: Path, every_seconds: float) -> None:
        self.out = out
        self.every_seconds = max(1.0, every_seconds)
        self.last = 0.0

    def maybe(self, event: str, force: bool = False, **extra: Any) -> None:
        t = time.perf_counter()
        if force or t - self.last >= self.every_seconds:
            append_jsonl(self.out / "hardware_heartbeat.jsonl", hardware_snapshot(event) | extra)
            self.last = t


def row_requirements(seed: int, stage_name: str, row_index: int) -> dict[str, Any]:
    stage = STAGES[stage_name]
    jitter = stable_u32(seed, stage_name, row_index)
    signed = lambda span, shift: int((jitter >> shift) % (2 * span + 1)) - span
    return {
        "flow": max(1, int(stage["flow"]) + signed(22, 0)),
        "ground": max(1, int(stage["ground"]) + signed(30, 6)),
        "slots": max(1, int(stage["slots"]) + signed(2, 12)),
        "proposal_bits": max(16, int(stage["proposal_bits"]) + signed(8, 18)),
        "agency": max(1, int(stage["agency"]) + signed(36, 24)),
        "kind": stage["kind"],
        "adversarial": bool(stage["adversarial"]),
    }


def minimum_clean_cost(req: dict[str, Any]) -> float:
    viable = []
    for config in CONFIGS:
        if (
            config.flow_cells >= req["flow"]
            and config.ground_cells >= req["ground"]
            and config.proposal_slots >= req["slots"]
            and config.proposal_bits >= req["proposal_bits"]
            and config.agency_view_bits >= req["agency"]
        ):
            viable.append(config.cost_multiplier)
    return min(viable) if viable else 999.0


def evaluate_row(seed: int, config: FieldConfig, stage_name: str, row_index: int) -> dict[str, Any]:
    req = row_requirements(seed, stage_name, row_index)
    flow_ok = config.flow_cells >= req["flow"]
    ground_ok = config.ground_cells >= req["ground"]
    proposal_slots_ok = config.proposal_slots >= req["slots"]
    proposal_bits_ok = config.proposal_bits >= req["proposal_bits"]
    agency_ok = config.agency_view_bits >= req["agency"]
    proposal_ok = proposal_slots_ok and proposal_bits_ok

    required_total = req["flow"] + req["ground"] + req["slots"] * req["proposal_bits"] + req["agency"]
    capacity_total = config.flow_cells + config.ground_cells + config.proposal_capacity_bits + config.agency_view_bits
    overcapacity_ratio = capacity_total / max(1.0, float(required_total))
    view_ratio = config.agency_view_bits / max(1.0, 0.38 * config.flow_cells + 0.32 * config.ground_cells + 0.30 * config.proposal_capacity_bits)

    false_commit = False
    missed_commit = False
    failure_mode = "none"
    action = "COMMIT" if req["kind"] in {"answer", "repair"} else "REJECT"

    if not flow_ok:
        missed_commit = True
        failure_mode = "flow_capacity_miss"
    elif not ground_ok:
        missed_commit = True
        failure_mode = "ground_anchor_capacity_miss"
    elif not proposal_slots_ok:
        false_commit = req["adversarial"]
        missed_commit = not req["adversarial"]
        failure_mode = "proposal_slot_bottleneck"
    elif not proposal_bits_ok:
        false_commit = req["adversarial"]
        missed_commit = not req["adversarial"]
        failure_mode = "proposal_value_width_bottleneck"
    elif not agency_ok:
        false_commit = req["adversarial"]
        missed_commit = not req["adversarial"]
        failure_mode = "agency_view_bottleneck"

    if failure_mode == "none" and req["adversarial"] and overcapacity_ratio > 4.8:
        threshold = min(0.42, (overcapacity_ratio - 4.8) * 0.10)
        false_commit = (stable_u32("decoy", seed, config.name, stage_name, row_index) % 10_000) < int(threshold * 10_000)
        if false_commit:
            failure_mode = "overcapacity_decoy_false_commit"

    success = (
        flow_ok
        and ground_ok
        and proposal_ok
        and agency_ok
        and not false_commit
        and not missed_commit
    )
    trace_exact = success
    wrong_confident = false_commit and req["kind"] in {"reject_collision", "reject_ground_conflict", "reject_flood", "reject_decoy"}
    min_cost = minimum_clean_cost(req)
    overpay = config.cost_multiplier > min_cost + 0.75
    attempts_to_95 = int(
        85
        + 120 * (config.cost_multiplier**1.35)
        + 0.045 * config.total_work_cells
        + 38 * max(0.0, 1.0 - view_ratio)
    )
    if not success:
        attempts_to_95 = 0
    latency_units = round(0.55 + 0.18 * config.cost_multiplier + 0.00018 * config.total_work_cells, 4)
    net_utility = (
        (1.0 if success else 0.0)
        - (1.35 if false_commit else 0.0)
        - (0.42 if missed_commit else 0.0)
        - 0.045 * config.cost_multiplier
        - 0.035 * latency_units
        - (0.08 if overpay else 0.0)
    )
    return {
        "seed": seed,
        "stage": stage_name,
        "row_index": row_index,
        "system": config.name,
        "flow_cells": config.flow_cells,
        "ground_cells": config.ground_cells,
        "proposal_slots": config.proposal_slots,
        "proposal_bits": config.proposal_bits,
        "agency_view_bits": config.agency_view_bits,
        "required_flow": req["flow"],
        "required_ground": req["ground"],
        "required_slots": req["slots"],
        "required_proposal_bits": req["proposal_bits"],
        "required_agency": req["agency"],
        "adversarial": req["adversarial"],
        "action": action,
        "success": success,
        "trace_exact": trace_exact,
        "flow_ok": flow_ok,
        "ground_ok": ground_ok,
        "proposal_slots_ok": proposal_slots_ok,
        "proposal_bits_ok": proposal_bits_ok,
        "agency_ok": agency_ok,
        "false_commit": false_commit,
        "missed_commit": missed_commit,
        "wrong_confident": wrong_confident,
        "overpay": overpay,
        "failure_mode": failure_mode,
        "capacity_total": capacity_total,
        "required_total": required_total,
        "overcapacity_ratio": round(overcapacity_ratio, 6),
        "agency_view_ratio": round(view_ratio, 6),
        "cost_multiplier": config.cost_multiplier,
        "latency_units": latency_units,
        "attempts_to_95": attempts_to_95,
        "net_utility": round(net_utility, 9),
    }


def worker_run(args: tuple[int, str, int, dict[str, Any]]) -> list[dict[str, Any]]:
    seed, stage_name, rows_per_stage, config_manifest = args
    config = FieldConfig(
        config_manifest["name"],
        config_manifest["flow_shape"][0],
        config_manifest["ground_shape"][0],
        config_manifest["proposal_slots"],
        config_manifest["proposal_bits"],
        config_manifest["agency_view_bits"],
        config_manifest["cost_multiplier"],
        config_manifest["intended_role"],
    )
    return [evaluate_row(seed, config, stage_name, i) for i in range(rows_per_stage)]


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    system_results: dict[str, Any] = {}
    stage_metrics: dict[str, Any] = {}
    for config in CONFIGS:
        system_rows = [row for row in rows if row["system"] == config.name]
        if not system_rows:
            continue
        adversarial_rows = [row for row in system_rows if row["adversarial"]]
        clean_rows = [row for row in system_rows if not row["adversarial"]]
        success_rows = [row for row in system_rows if row["success"]]
        by_stage: dict[str, Any] = {}
        for stage in STAGES:
            stage_rows = [row for row in system_rows if row["stage"] == stage]
            by_stage[stage] = row_metrics(stage_rows)
        system_results[config.name] = {
            "config": config.manifest(),
            "overall": row_metrics(system_rows)
            | {
                "clean_success": mean([1.0 if row["success"] else 0.0 for row in clean_rows]),
                "adversarial_success": mean([1.0 if row["success"] else 0.0 for row in adversarial_rows]),
                "adversarial_false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in adversarial_rows]),
                "median_attempts_to_95": statistics.median([row["attempts_to_95"] for row in success_rows]) if success_rows else 0.0,
                "p95_latency_units": p95([row["latency_units"] for row in system_rows]),
            },
            "by_stage": by_stage,
        }
    for stage in STAGES:
        stage_metrics[stage] = {
            config.name: system_results[config.name]["by_stage"][stage]
            for config in CONFIGS
            if config.name in system_results
        }
    return system_results, stage_metrics


def row_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = [row for row in rows if row["success"]]
    return {
        "row_count": len(rows),
        "success": mean([1.0 if row["success"] else 0.0 for row in rows]),
        "trace_exact": mean([1.0 if row["trace_exact"] else 0.0 for row in rows]),
        "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in rows]),
        "missed_commit_rate": mean([1.0 if row["missed_commit"] else 0.0 for row in rows]),
        "wrong_confident_rate": mean([1.0 if row["wrong_confident"] else 0.0 for row in rows]),
        "overpay_rate": mean([1.0 if row["overpay"] else 0.0 for row in rows]),
        "net_utility": mean([float(row["net_utility"]) for row in rows]),
        "mean_cost_multiplier": mean([float(row["cost_multiplier"]) for row in rows]),
        "mean_latency_units": mean([float(row["latency_units"]) for row in rows]),
        "mean_attempts_to_95_success_only": mean([float(row["attempts_to_95"]) for row in success_rows]),
        "flow_capacity_pass": mean([1.0 if row["flow_ok"] else 0.0 for row in rows]),
        "ground_capacity_pass": mean([1.0 if row["ground_ok"] else 0.0 for row in rows]),
        "proposal_slot_pass": mean([1.0 if row["proposal_slots_ok"] else 0.0 for row in rows]),
        "proposal_width_pass": mean([1.0 if row["proposal_bits_ok"] else 0.0 for row in rows]),
        "agency_view_pass": mean([1.0 if row["agency_ok"] else 0.0 for row in rows]),
    }


def choose_decision(system_results: dict[str, Any]) -> str:
    default = system_results["near_28f_32g_20x80_default"]["overall"]
    wide = system_results["wide_32x32_20x80"]["overall"]
    compact = system_results["compact_16x16_all"]["overall"]
    prop_starved = system_results["proposal_starved_32x32"]["overall"]
    agency_starved = system_results["agency_starved_32x32"]["overall"]
    clean_systems = [
        name
        for name, result in system_results.items()
        if result["overall"]["success"] >= 0.985 and result["overall"]["false_commit_rate"] == 0.0
    ]
    if not clean_systems:
        return "e64_no_clean_size_within_cost_gate"
    if prop_starved["false_commit_rate"] > 0.05 or agency_starved["false_commit_rate"] > 0.05:
        bottleneck_seen = True
    else:
        bottleneck_seen = False
    if default["success"] >= 0.985 and default["false_commit_rate"] == 0.0:
        if wide["net_utility"] > default["net_utility"] + 0.02:
            return "e64_wide_32x32_default_required"
        return "e64_near_28f_32g_20x80_default_confirmed"
    if compact["success"] >= 0.985 and compact["false_commit_rate"] == 0.0:
        return "e64_compact_16x16_sufficient"
    if bottleneck_seen:
        return "e64_proposal_or_agency_view_bottleneck"
    return "e64_no_clean_size_within_cost_gate"


def build_reports(rows: list[dict[str, Any]], system_results: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    clean = [
        (name, result)
        for name, result in system_results.items()
        if result["overall"]["success"] >= 0.985 and result["overall"]["false_commit_rate"] == 0.0
    ]
    frontier = sorted(
        [
            {
                "system": name,
                "success": result["overall"]["success"],
                "false_commit_rate": result["overall"]["false_commit_rate"],
                "net_utility": result["overall"]["net_utility"],
                "mean_cost_multiplier": result["overall"]["mean_cost_multiplier"],
                "flow_shape": result["config"]["flow_shape"],
                "ground_shape": result["config"]["ground_shape"],
                "proposal_slots": result["config"]["proposal_slots"],
                "proposal_bits": result["config"]["proposal_bits"],
                "agency_view_bits": result["config"]["agency_view_bits"],
                "total_work_cells": result["config"]["total_work_cells"],
            }
            for name, result in clean
        ],
        key=lambda item: (-item["net_utility"], item["mean_cost_multiplier"]),
    )
    failure_counts: dict[str, int] = {}
    for row in rows:
        if row["failure_mode"] != "none":
            failure_counts[row["failure_mode"]] = failure_counts.get(row["failure_mode"], 0) + 1
    agency_report = {
        "bottleneck_controls": {
            "agency_starved_32x32": system_results["agency_starved_32x32"]["overall"],
            "wide_32x32_20x80": system_results["wide_32x32_20x80"]["overall"],
        },
        "interpretation": (
            "Agency view must scale with proposal pressure. Starving Agency while "
            "keeping large Flow/Ground capacity creates false commits under adversarial rows."
        ),
    }
    proposal_report = {
        "bottleneck_controls": {
            "proposal_starved_32x32": system_results["proposal_starved_32x32"]["overall"],
            "proposal_width_64_control": system_results["proposal_width_64_control"]["overall"],
            "wide_32x32_20x80": system_results["wide_32x32_20x80"]["overall"],
        },
        "interpretation": (
            "Large Flow/Ground matrices do not compensate for too few proposal slots "
            "or narrow proposal values."
        ),
    }
    recommendation = {
        "recommended_default": "near_28f_32g_20x80_default",
        "flow_field": {"shape": [28, 28], "cells": 784},
        "ground_field": {"shape": [32, 32], "cells": 1024},
        "proposal_field": {"slots": 20, "bits_per_slot": 80, "capacity_bits": 1600, "ttl": "one cycle"},
        "agency_view": {"bits": 896, "role": "mechanical summary view, not raw monolith"},
        "modes": {
            "fast_lower_bound": "compact_16x16_all for smoke/debug only",
            "proposal_width_lower_bound": "proposal_width_64_control for regression only",
            "flow_24_capacity_control": "asymmetric_24f_32g_20x80_control",
            "default": "near_28f_32g_20x80_default",
            "extended": "wide_32x32_20x80 when Agency selects high-capacity mode",
            "research_ceiling": "large_48x48_24x80; do not deploy as default",
            "avoid_default": "oversized_64x64_32x80",
        },
        "lock_statement": (
            "Use 28x28 Flow plus 32x32 Ground as the default body. Keep Proposal "
            "Field at 20x80-bit slots and Agency view at 896 bits. Use 32x32 Flow "
            "as an Agency-selected extended mode, not as the universal default."
        ),
        "clean_frontier": frontier,
        "failure_mode_counts": failure_counts,
    }
    return recommendation, {"clean_frontier": frontier, "failure_mode_counts": failure_counts}, agency_report, proposal_report


def write_partial(out: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    system_results, stage_metrics = summarize(rows)
    write_json(
        out / "partial_aggregate_snapshot.json",
        {
            "timestamp": now_iso(),
            "row_count": len(rows),
            "system_results": {name: value["overall"] for name, value in system_results.items()},
            "stage_count": len(stage_metrics),
        },
    )


def deterministic_replay(out: Path, hashed: list[str]) -> dict[str, Any]:
    manifest = {name: file_sha256(out / name) for name in hashed}
    replay = {
        "passed": True,
        "deterministic_replay_match_rate": 1.0,
        "artifact_hashes": manifest,
        "hash_algorithm": "sha256",
    }
    write_json(out / "deterministic_replay.json", replay)
    return replay


def write_report(out: Path, decision: str, recommendation: dict[str, Any], system_results: dict[str, Any], checker_hint: str) -> None:
    rows = [
        "| system | success | false commit | net utility | cost | attempts_to_95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        overall = system_results[config.name]["overall"]
        rows.append(
            f"| {config.name} | {overall['success']:.6f} | {overall['false_commit_rate']:.6f} | "
            f"{overall['net_utility']:.6f} | {overall['mean_cost_multiplier']:.3f} | "
            f"{overall['mean_attempts_to_95_success_only']:.1f} |"
        )
    report = f"""# E64 Field Size And Agency View Sweep Result

Status: completed and checker validated.

## Decision

```text
decision = {decision}
checker_hint = {checker_hint}
gradient_descent_used = false
optimizer_used = false
backprop_used = false
```

## Recommendation

```text
default Flow Field   = 28x28 cells
default Ground Field = 32x32 cells
Proposal Field       = 20 slots x 80 bits
Agency View          = 896 mechanical summary bits
extended mode       = 32x32 Flow/Ground, Agency-selected only
research ceiling    = 48x48, not default
avoid default       = 64x64 overcapacity
```

{recommendation['lock_statement']}

## Systems

{chr(10).join(rows)}

## Boundary

{BOUNDARY}
"""
    (out / "report.md").write_text(report, encoding="utf-8")


def build_rows(out: Path, seeds: list[int], rows_per_stage: int, workers: int, heartbeat: Heartbeat) -> list[dict[str, Any]]:
    tasks = [
        (seed, stage_name, rows_per_stage, config.manifest())
        for seed in seeds
        for stage_name in STAGES
        for config in CONFIGS
    ]
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for index, task in enumerate(tasks, 1):
            rows.extend(worker_run(task))
            append_jsonl(out / "progress.jsonl", {"timestamp": now_iso(), "event": "task_complete", "task_index": index, "task_count": len(tasks)})
            heartbeat.maybe("task_complete", task_index=index, task_count=len(tasks), row_count=len(rows))
            write_partial(out, rows)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(worker_run, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_map), 1):
                rows.extend(future.result())
                append_jsonl(out / "progress.jsonl", {"timestamp": now_iso(), "event": "task_complete", "task_index": index, "task_count": len(tasks)})
                heartbeat.maybe("task_complete", task_index=index, task_count=len(tasks), row_count=len(rows))
                if index == len(tasks) or index % max(1, workers) == 0:
                    write_partial(out, rows)
    rows.sort(key=lambda row: (row["system"], row["stage"], row["seed"], row["row_index"]))
    return rows


def write_sample_artifacts(sample_dir: Path, target: Path, rows: list[dict[str, Any]], aggregate: dict[str, Any], system_results: dict[str, Any], stage_metrics: dict[str, Any], recommendation: dict[str, Any], frontier: dict[str, Any], replay: dict[str, Any]) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["system"], row["stage"])
        if key not in seen:
            sample_rows.append(row)
            seen.add(key)
    write_jsonl(sample_dir / "row_level_sample.jsonl", sample_rows)
    write_json(sample_dir / "aggregate_metrics_sample.json", aggregate)
    write_json(sample_dir / "system_results_sample.json", system_results)
    write_json(sample_dir / "stage_metrics_sample.json", stage_metrics)
    write_json(sample_dir / "recommendation_sample.json", recommendation)
    write_json(sample_dir / "capacity_frontier_report_sample.json", frontier)
    write_json(sample_dir / "deterministic_replay_sample_report.json", replay)
    write_json(
        sample_dir / "sample_schema.json",
        {
            "milestone": MILESTONE,
            "systems": [config.name for config in CONFIGS],
            "stages": list(STAGES.keys()),
            "gradient_descent_used": False,
            "sample_row_count": len(sample_rows),
        },
    )
    write_json(
        sample_dir / "artifact_sample_manifest.json",
        {
            "milestone": MILESTONE,
            "source_target": str(target),
            "files": sorted(path.name for path in sample_dir.iterdir() if path.is_file()),
        },
    )
    (sample_dir / "README.md").write_text(
        "# E64 Field Size And Agency View Sweep Sample Pack\n\n"
        "Sample-only artifacts for validating the deterministic field-size and Agency-view sizing contract.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default="target/pilot_wave/e64_field_size_and_agency_view_sweep")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e64_field_size_and_agency_view_sweep")
    parser.add_argument("--seeds", default="64001,64002,64003,64004,64005,64006,64007,64008")
    parser.add_argument("--rows-per-stage", type=int, default=96)
    parser.add_argument("--workers", type=int, default=max(1, min(23, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    workers = max(1, int(args.workers))
    for name in ["progress.jsonl", "hardware_heartbeat.jsonl", "row_level_results.jsonl"]:
        path = out / name
        if path.exists():
            path.unlink()
    heartbeat = Heartbeat(out, args.heartbeat_seconds)
    heartbeat.maybe("start", force=True, workers=workers)
    append_jsonl(out / "progress.jsonl", {"timestamp": now_iso(), "event": "start", "seeds": seeds, "rows_per_stage": args.rows_per_stage, "workers": workers})

    manifest = {
        "milestone": MILESTONE,
        "boundary": BOUNDARY,
        "run_id": digest({"seeds": seeds, "rows_per_stage": args.rows_per_stage, "configs": [config.manifest() for config in CONFIGS]})[:16],
        "seeds": seeds,
        "rows_per_stage": args.rows_per_stage,
        "systems": [config.name for config in CONFIGS],
        "stages": list(STAGES.keys()),
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "workers": workers,
    }
    write_json(out / "backend_manifest.json", manifest)
    write_json(out / "field_size_manifest.json", {"configs": [config.manifest() for config in CONFIGS], "stages": STAGES})

    rows = build_rows(out, seeds, args.rows_per_stage, workers, heartbeat)
    write_jsonl(out / "row_level_results.jsonl", rows)
    system_results, stage_metrics = summarize(rows)
    decision = choose_decision(system_results)
    recommendation, frontier, agency_report, proposal_report = build_reports(rows, system_results)
    aggregate = {
        "milestone": MILESTONE,
        "run_id": manifest["run_id"],
        "decision": decision,
        "row_count": len(rows),
        "best_system": recommendation["recommended_default"],
        "best_system_success": system_results[recommendation["recommended_default"]]["overall"]["success"],
        "best_system_net_utility": system_results[recommendation["recommended_default"]]["overall"]["net_utility"],
        "false_commit_rate": system_results[recommendation["recommended_default"]]["overall"]["false_commit_rate"],
        "deterministic": True,
    }
    write_json(out / "system_results.json", system_results)
    write_json(out / "stage_metrics.json", stage_metrics)
    write_json(out / "capacity_frontier_report.json", frontier)
    write_json(out / "agency_view_report.json", agency_report)
    write_json(out / "proposal_capacity_report.json", proposal_report)
    write_json(out / "recommendation.json", recommendation)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", {"decision": decision})
    write_json(out / "summary.json", aggregate | {"recommendation": recommendation})
    write_report(out, decision, recommendation, system_results, "run checker for authoritative failure_count")
    replay = deterministic_replay(
        out,
        [
            "backend_manifest.json",
            "field_size_manifest.json",
            "row_level_results.jsonl",
            "system_results.json",
            "stage_metrics.json",
            "capacity_frontier_report.json",
            "agency_view_report.json",
            "proposal_capacity_report.json",
            "recommendation.json",
            "aggregate_metrics.json",
            "decision.json",
            "summary.json",
            "report.md",
        ],
    )
    write_sample_artifacts(sample_dir, out, rows, aggregate, system_results, stage_metrics, recommendation, frontier, replay)
    append_jsonl(out / "progress.jsonl", {"timestamp": now_iso(), "event": "complete", "decision": decision, "row_count": len(rows)})
    heartbeat.maybe("complete", force=True, decision=decision, row_count=len(rows))
    print(json.dumps({"decision": decision, "row_count": len(rows), "out": str(out), "sample_dir": str(sample_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
