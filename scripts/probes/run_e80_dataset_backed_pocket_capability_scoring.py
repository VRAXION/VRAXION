#!/usr/bin/env python3
"""E80 dataset-backed Pocket capability scoring smoke/evidence runner.

This runner intentionally does not claim that VRAXION solves GSM8K or open
text. It tests the next needed bridge after E79: can local dataset rows be
converted into guarded, scoreable Pocket capability evidence with train /
validation / adversarial coverage and continuous artifacts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SYSTEMS = [
    "gsm8k_answer_marker_adapter",
    "gsm8k_rationale_calc_marker_adapter",
    "fineweb_text_mode_selector",
    "fineweb_byte_frame_boundary_adapter",
    "bad_answer_first_control",
    "bad_text_always_commit_control",
]


def now_ms() -> int:
    return int(time.time() * 1000)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def split_for(row_id: str, seed: int, source_split: str) -> str:
    if source_split == "test":
        return "adversarial"
    bucket = stable_hash(f"{seed}:{row_id}") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "adversarial"


def byte_len(text: str) -> int:
    return len(text.encode("utf-8", errors="replace"))


def select_text_mode(text: str) -> str:
    length = byte_len(text)
    replacement_count = text.count("\ufffd") + text.count("�")
    control_count = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t\r")
    evidence_available = bool(text.strip())
    integrity_risk = 0
    if replacement_count:
        integrity_risk += 2
    if control_count:
        integrity_risk += 2
    boundary_risk = 0 if length <= 416 else 1 if length <= 1024 else 2 if length <= 1664 else 4
    if not evidence_available:
        return "ASK_OR_MULTI_CYCLE"
    if length <= 416 and boundary_risk <= 1 and integrity_risk <= 1:
        return "FAST"
    if length <= 1024 and integrity_risk <= 2:
        return "LONG_CAPPED"
    if length <= 1664 and integrity_risk <= 3:
        return "CLEAN_LONG"
    return "ASK_OR_MULTI_CYCLE"


def mode_unique_capacity(mode: str) -> int:
    if mode == "FAST":
        return 416
    if mode == "LONG_CAPPED":
        return 1024
    if mode == "CLEAN_LONG":
        return 1664
    return 0


FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")
CALC_RE = re.compile(r"<<\s*([-+]?\d+(?:\.\d+)?)\s*([+\-*/])\s*([-+]?\d+(?:\.\d+)?)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*>>")


def parse_final_answer(answer: str) -> str | None:
    match = FINAL_ANSWER_RE.search(answer)
    return match.group(1) if match else None


def calc_ok(left: str, op: str, right: str, expected: str) -> bool:
    a = float(left)
    b = float(right)
    e = float(expected)
    if op == "+":
        got = a + b
    elif op == "-":
        got = a - b
    elif op == "*":
        got = a * b
    elif op == "/":
        if b == 0:
            return False
        got = a / b
    else:
        return False
    return abs(got - e) <= max(1e-6, abs(e) * 1e-6)


def evaluate_gsm(system: str, row: dict[str, Any], adversarial_variant: bool) -> dict[str, Any]:
    answer = str(row.get("answer", ""))
    question = str(row.get("question", ""))
    if adversarial_variant:
        answer = answer.replace("####", "###", 1)
    final = parse_final_answer(answer)
    calc_matches = list(CALC_RE.finditer(answer))
    if system == "gsm8k_answer_marker_adapter":
        commit = final is not None
        success = (not adversarial_variant and commit) or (adversarial_variant and not commit)
        false_commit = adversarial_variant and commit
        utility = 1.0 if success else 0.0
        trace_valid = success
    elif system == "gsm8k_rationale_calc_marker_adapter":
        checked = 0
        good = 0
        for match in calc_matches[:16]:
            checked += 1
            if calc_ok(match.group(1), match.group(2), match.group(3), match.group(4)):
                good += 1
        commit = final is not None and checked > 0 and good == checked
        success = (not adversarial_variant and commit) or (adversarial_variant and not commit)
        false_commit = adversarial_variant and commit
        utility = (good / checked) if checked else 0.0
        trace_valid = success and (checked > 0 or adversarial_variant)
    elif system == "bad_answer_first_control":
        commit = bool(question.strip())
        success = False if adversarial_variant else final is not None
        false_commit = adversarial_variant and commit
        utility = 0.15 if success else 0.0
        trace_valid = False
    else:
        return {"applicable": False}
    return {
        "applicable": True,
        "success": success,
        "commit": commit,
        "false_commit": false_commit,
        "trace_valid": trace_valid,
        "utility": utility,
        "final_answer_present": final is not None,
        "calc_marker_count": len(calc_matches),
        "byte_len": byte_len(question) + byte_len(answer),
    }


def evaluate_text(system: str, row: dict[str, Any], adversarial_variant: bool) -> dict[str, Any]:
    text = str(row.get("text", ""))
    if adversarial_variant:
        text = ("\x00" * 32) + text + ("X" * 1800)
    mode = select_text_mode(text)
    length = byte_len(text)
    capacity = mode_unique_capacity(mode)
    if system == "fineweb_text_mode_selector":
        commit = mode != "ASK_OR_MULTI_CYCLE"
        expected_commit = bool(text.strip()) and length <= 1664 and "\x00" not in text
        success = commit == expected_commit
        false_commit = commit and not expected_commit
        trace_valid = success
        utility = 1.0 if success else 0.0
    elif system == "fineweb_byte_frame_boundary_adapter":
        commit = mode != "ASK_OR_MULTI_CYCLE" and length <= capacity
        expected_commit = bool(text.strip()) and length <= capacity and "\x00" not in text
        success = commit == expected_commit
        false_commit = commit and not expected_commit
        trace_valid = success
        utility = 1.0 if success else 0.0
    elif system == "bad_text_always_commit_control":
        commit = True
        expected_commit = bool(text.strip()) and length <= 1664 and "\x00" not in text
        success = commit == expected_commit
        false_commit = commit and not expected_commit
        trace_valid = False
        utility = 0.15 if success else 0.0
    else:
        return {"applicable": False}
    return {
        "applicable": True,
        "success": success,
        "commit": commit,
        "false_commit": false_commit,
        "trace_valid": trace_valid,
        "utility": utility,
        "mode": mode,
        "byte_len": length,
        "capacity": capacity,
    }


@dataclass(frozen=True)
class SeedTask:
    seed: int
    data_root: str
    gsm_limit: int
    fineweb_limit: int


def evaluate_seed(task: SeedTask) -> dict[str, Any]:
    rng = random.Random(task.seed)
    data_root = Path(task.data_root)
    gsm_paths = [
        data_root / "gsm8k" / "train.jsonl",
        data_root / "gsm8k" / "test.jsonl",
    ]
    fineweb_path = data_root / "fineweb_edu" / "sample-10BT_sample_2000.jsonl"
    metrics: dict[str, dict[str, Any]] = {
        system: {
            "rows": 0,
            "success": 0,
            "commit": 0,
            "false_commit": 0,
            "trace_valid": 0,
            "utility_sum": 0.0,
            "splits": {"train": 0, "validation": 0, "adversarial": 0},
        }
        for system in SYSTEMS
    }
    samples: list[dict[str, Any]] = []

    def record(system: str, row_id: str, source: str, split: str, result: dict[str, Any]) -> None:
        if not result.get("applicable"):
            return
        m = metrics[system]
        m["rows"] += 1
        m["success"] += int(result["success"])
        m["commit"] += int(result["commit"])
        m["false_commit"] += int(result["false_commit"])
        m["trace_valid"] += int(result["trace_valid"])
        m["utility_sum"] += float(result["utility"])
        m["splits"][split] += 1
        if len(samples) < 160 and (m["rows"] <= 4 or result["false_commit"] or not result["success"]):
            slim = {k: v for k, v in result.items() if k not in {"applicable"}}
            samples.append(
                {
                    "seed": task.seed,
                    "system": system,
                    "row_id": row_id,
                    "source": source,
                    "split": split,
                    "result": slim,
                }
            )

    for path in gsm_paths:
        rows = list(iter_jsonl(path))
        if task.gsm_limit and len(rows) > task.gsm_limit:
            rows = rng.sample(rows, task.gsm_limit)
        for row in rows:
            row_id = str(row.get("row_id", "gsm_unknown"))
            split = split_for(row_id, task.seed, str(row.get("source_split", "train")))
            adversarial_variant = split == "adversarial"
            for system in SYSTEMS:
                result = evaluate_gsm(system, row, adversarial_variant)
                record(system, row_id, "gsm8k", split, result)

    fine_rows = list(iter_jsonl(fineweb_path))
    if task.fineweb_limit and len(fine_rows) > task.fineweb_limit:
        fine_rows = rng.sample(fine_rows, task.fineweb_limit)
    for row in fine_rows:
        row_id = str(row.get("row_id", "fineweb_unknown"))
        split = split_for(row_id, task.seed, "train")
        adversarial_variant = split == "adversarial"
        for system in SYSTEMS:
            result = evaluate_text(system, row, adversarial_variant)
            record(system, row_id, "fineweb_edu", split, result)

    return {"seed": task.seed, "metrics": metrics, "samples": samples}


def ratio(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def summarize(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, dict[str, Any]] = {}
    for system in SYSTEMS:
        agg = {
            "rows": 0,
            "success": 0,
            "commit": 0,
            "false_commit": 0,
            "trace_valid": 0,
            "utility_sum": 0.0,
            "splits": {"train": 0, "validation": 0, "adversarial": 0},
            "seed_success_rates": [],
        }
        for result in seed_results:
            m = result["metrics"][system]
            agg["rows"] += m["rows"]
            agg["success"] += m["success"]
            agg["commit"] += m["commit"]
            agg["false_commit"] += m["false_commit"]
            agg["trace_valid"] += m["trace_valid"]
            agg["utility_sum"] += m["utility_sum"]
            for split, count in m["splits"].items():
                agg["splits"][split] += count
            if m["rows"]:
                agg["seed_success_rates"].append(ratio(m["success"], m["rows"]))
        rows = agg["rows"]
        success_rate = ratio(agg["success"], rows)
        false_commit_rate = ratio(agg["false_commit"], rows)
        trace_validity = ratio(agg["trace_valid"], rows)
        mean_utility = ratio(agg["utility_sum"], rows)
        promoted = (
            rows > 0
            and success_rate >= 0.97
            and false_commit_rate == 0.0
            and trace_validity >= 0.97
            and all(agg["splits"][split] > 0 for split in ["train", "validation", "adversarial"])
            and not system.startswith("bad_")
        )
        systems[system] = {
            "rows": rows,
            "success_rate": success_rate,
            "false_commit_rate": false_commit_rate,
            "trace_validity": trace_validity,
            "mean_utility": mean_utility,
            "split_rows": agg["splits"],
            "seed_success_mean": statistics.mean(agg["seed_success_rates"]) if agg["seed_success_rates"] else 0.0,
            "seed_success_min": min(agg["seed_success_rates"]) if agg["seed_success_rates"] else 0.0,
            "promoted_candidate": promoted,
        }
    promoted_count = sum(1 for item in systems.values() if item["promoted_candidate"])
    bad_promotions = sum(
        1 for name, item in systems.items() if name.startswith("bad_") and item["promoted_candidate"]
    )
    return {
        "systems": systems,
        "promoted_candidate_count": promoted_count,
        "bad_promotion_count": bad_promotions,
        "all_promoted_safe": bad_promotions == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/high_quality_seed_v0")
    parser.add_argument("--out", default="target/pilot_wave/e80_dataset_backed_pocket_capability_scoring")
    parser.add_argument("--seeds", default="8001,8002,8003,8004,8005,8006,8007,8008")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gsm-limit", type=int, default=0)
    parser.add_argument("--fineweb-limit", type=int, default=0)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    if progress.exists():
        progress.unlink()
    started = time.time()
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    worker_count = args.workers or min(max(os.cpu_count() or 1, 1), len(seeds), 23)
    manifest_path = Path(args.data_root) / "manifest.json"
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    write_json(
        out / "backend_manifest.json",
        {
            "artifact_contract": "E80_DATASET_BACKED_POCKET_CAPABILITY_SCORING",
            "data_root": args.data_root,
            "data_manifest": str(manifest_path).replace("\\", "/"),
            "seeds": seeds,
            "workers": worker_count,
            "systems": SYSTEMS,
            "boundary": "dataset-backed scoring/probe; not GSM8K solver or open-domain assistant claim",
        },
    )
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "start", "seeds": seeds, "workers": worker_count})

    tasks = [SeedTask(seed, args.data_root, args.gsm_limit, args.fineweb_limit) for seed in seeds]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(evaluate_seed, task): task.seed for task in tasks}
        last_write = time.time()
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            result = future.result()
            results.append(result)
            append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "seed_complete", "seed": seed, "completed": len(results)})
            if time.time() - last_write >= args.heartbeat_seconds or len(results) == len(tasks):
                partial = summarize(results)
                write_json(out / "partial_aggregate_snapshot.json", partial)
                append_jsonl(
                    progress,
                    {
                        "timestamp_ms": now_ms(),
                        "event": "heartbeat",
                        "completed_seeds": len(results),
                        "promoted_candidate_count": partial["promoted_candidate_count"],
                    },
                )
                last_write = time.time()

    aggregate = summarize(results)
    samples: list[dict[str, Any]] = []
    for result in results:
        samples.extend(result["samples"])
    with (out / "row_level_samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples[:1000]:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")

    write_json(out / "system_results.json", aggregate["systems"])
    write_json(
        out / "aggregate_metrics.json",
        {
            "seed_count": len(seeds),
            "workers": worker_count,
            "promoted_candidate_count": aggregate["promoted_candidate_count"],
            "bad_promotion_count": aggregate["bad_promotion_count"],
            "all_promoted_safe": aggregate["all_promoted_safe"],
            "seconds": time.time() - started,
        },
    )
    decision = (
        "e80_dataset_backed_scoring_promotion_evidence_ready"
        if aggregate["promoted_candidate_count"] >= 3 and aggregate["bad_promotion_count"] == 0
        else "e80_dataset_backed_scoring_needs_curriculum_adapter_work"
    )
    write_json(out / "decision.json", {"decision": decision, "failure_count": 0})
    write_json(
        out / "summary.json",
        {
            "decision": decision,
            "source_files": data_manifest.get("files", []),
            "promoted_systems": [
                name for name, item in aggregate["systems"].items() if item["promoted_candidate"]
            ],
            "seconds": time.time() - started,
        },
    )
    lines = [
        "# E80 Dataset-Backed Pocket Capability Scoring",
        "",
        "```text",
        f"decision = {decision}",
        f"seeds = {len(seeds)}",
        f"workers = {worker_count}",
        f"promoted_candidate_count = {aggregate['promoted_candidate_count']}",
        f"bad_promotion_count = {aggregate['bad_promotion_count']}",
        "```",
        "",
        "| system | rows | success | false_commit | trace | promoted |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, item in aggregate["systems"].items():
        lines.append(
            f"| {name} | {item['rows']} | {item['success_rate']:.6f} | {item['false_commit_rate']:.6f} | {item['trace_validity']:.6f} | {item['promoted_candidate']} |"
        )
    lines.append("")
    lines.append("Boundary: this is dataset-backed scoring/conversion evidence, not a claim that the runtime solves GSM8K or open-domain text.")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "complete", "decision": decision, "seconds": time.time() - started})
    print(json.dumps({"decision": decision, "out": str(out), "seconds": time.time() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
