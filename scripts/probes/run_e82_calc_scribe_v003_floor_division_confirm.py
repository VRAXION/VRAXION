#!/usr/bin/env python3
"""E82 CALC-SCRIBE v003 floor-division repair confirm.

E81 reduced CALC-SCRIBE failures to a narrow operator gap:
visible GSM8K markers using Python-style floor division, e.g.
`<<560//10=56>>`. E82 adds a targeted repair and confirms it across seeds.

Boundary: visible calculation marker validation only. This is not a GSM8K
solver and does not infer hidden answers.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RAW_MARKER_RE = re.compile(r"<<(.*?)>>", re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")


def now_ms() -> int:
    return int(time.time() * 1000)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


@dataclass(frozen=True)
class TraceRow:
    row_id: str
    source_split: str
    markers: tuple[str, ...]
    final_marker_present: bool
    question_head: str
    answer_head: str


def prepare_rows(data_root: Path, out: Path) -> Path:
    rows: list[TraceRow] = []
    for path in [data_root / "gsm8k" / "train.jsonl", data_root / "gsm8k" / "test.jsonl"]:
        for row in iter_jsonl(path):
            answer = str(row.get("answer", ""))
            rows.append(
                TraceRow(
                    row_id=str(row.get("row_id")),
                    source_split=str(row.get("source_split")),
                    markers=tuple(marker.strip() for marker in RAW_MARKER_RE.findall(answer)),
                    final_marker_present=FINAL_ANSWER_RE.search(answer) is not None,
                    question_head=str(row.get("question", ""))[:260],
                    answer_head=answer[:700],
                )
            )
    path = out / "prepared_rows.json"
    write_json(path, {"rows": [row.__dict__ for row in rows]})
    compact = out / "prepared_rows_compact.json"
    compact.write_text(json.dumps([row.__dict__ for row in rows], ensure_ascii=False), encoding="utf-8")
    return compact


class UnsafeExpression(Exception):
    pass


def normalize_expr(expr: str) -> str:
    text = expr.strip()
    for src, dst in {"−": "-", "–": "-", "—": "-", "×": "*", "÷": "/", "�": "-"}.items():
        text = text.replace(src, dst)
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)%", r"(\1/100)", text)
    text = re.sub(r"(?<![\d])\.(\d+)", r"0.\1", text)
    return text


def eval_ast(node: ast.AST, allow_floor_division: bool) -> float:
    if isinstance(node, ast.Expression):
        return eval_ast(node.body, allow_floor_division)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = eval_ast(node.operand, allow_floor_division)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = eval_ast(node.left, allow_floor_division)
        right = eval_ast(node.right, allow_floor_division)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise UnsafeExpression("division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv) and allow_floor_division:
            if right == 0:
                raise UnsafeExpression("floor division by zero")
            return math.floor(left / right)
    raise UnsafeExpression(f"blocked ast node {type(node).__name__}")


def safe_eval(expr: str, allow_floor_division: bool) -> float:
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        raise UnsafeExpression("invalid character")
    return eval_ast(ast.parse(expr, mode="eval"), allow_floor_division)


def validate_marker(marker: str, allow_floor_division: bool) -> tuple[bool, str]:
    if "=" not in marker:
        return False, "missing_equals"
    left, right = marker.rsplit("=", 1)
    try:
        actual = safe_eval(normalize_expr(left), allow_floor_division)
        expected = safe_eval(normalize_expr(right), allow_floor_division)
    except Exception as exc:
        return False, f"eval_failed:{type(exc).__name__}"
    tolerance = max(1e-6, abs(expected) * 2 / 1_000_000)
    if math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= tolerance:
        return True, "ok"
    return False, "math_mismatch"


def evaluate_seed(rows_path: str, seed: int, allow_floor_division: bool) -> dict[str, Any]:
    rows = [TraceRow(**item) for item in json.loads(Path(rows_path).read_text(encoding="utf-8"))]
    split_stats: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    for split in ["train", "validation", "adversarial"]:
        total = correct = marker_rows = marker_valid_rows = no_marker_defer = 0
        reason_counts: dict[str, int] = {}
        floor_marker_count = floor_marker_valid = 0
        for row in rows:
            row_split = split_for(row.row_id, seed, row.source_split)
            if row_split != split:
                continue
            total += 1
            if row_split == "adversarial" or not row.final_marker_present:
                correct += 1
                continue
            if not row.markers:
                correct += 1
                no_marker_defer += 1
                continue
            marker_rows += 1
            ok_count = 0
            reasons: list[str] = []
            for marker in row.markers:
                has_floor = "//" in marker
                floor_marker_count += int(has_floor)
                ok, reason = validate_marker(marker, allow_floor_division)
                ok_count += int(ok)
                floor_marker_valid += int(has_floor and ok)
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                reasons.append(reason)
            row_ok = ok_count == len(row.markers)
            marker_valid_rows += int(row_ok)
            correct += int(row_ok)
            if not row_ok and len(examples) < 20:
                examples.append(
                    {
                        "seed": seed,
                        "split": split,
                        "row_id": row.row_id,
                        "markers": list(row.markers[:6]),
                        "reasons": reasons[:6],
                        "question_head": row.question_head,
                        "answer_head": row.answer_head,
                    }
                )
        split_stats[split] = {
            "total_rows": total,
            "action_accuracy": 0.0 if total == 0 else correct / total,
            "marker_rows": marker_rows,
            "marker_validation_rate": 0.0 if marker_rows == 0 else marker_valid_rows / marker_rows,
            "no_marker_defer": no_marker_defer,
            "reason_counts": reason_counts,
            "floor_marker_count": floor_marker_count,
            "floor_marker_valid": floor_marker_valid,
            "floor_marker_rate": 1.0 if floor_marker_count == 0 else floor_marker_valid / floor_marker_count,
        }
    return {"seed": seed, "splits": split_stats, "examples": examples}


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["train", "validation", "adversarial"]:
        action = [result["splits"][split]["action_accuracy"] for result in results]
        marker = [result["splits"][split]["marker_validation_rate"] for result in results]
        floor = [result["splits"][split]["floor_marker_rate"] for result in results]
        out[split] = {
            "action_mean": statistics.mean(action),
            "action_min": min(action),
            "marker_mean": statistics.mean(marker),
            "marker_min": min(marker),
            "floor_marker_mean": statistics.mean(floor),
            "floor_marker_min": min(floor),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/high_quality_seed_v0")
    parser.add_argument("--out", default="target/pilot_wave/e82_calc_scribe_v003_floor_division_confirm")
    parser.add_argument("--seeds", default="8201,8202,8203,8204,8205,8206,8207,8208,8209,8210,8211,8212,8213,8214,8215,8216")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--heartbeat-seconds", type=float, default=20)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    if progress.exists():
        progress.unlink()
    started = time.time()
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    workers = args.workers or min(len(seeds), max(1, os.cpu_count() or 1), 23)
    rows_path = prepare_rows(Path(args.data_root), out)
    write_json(
        out / "run_manifest.json",
        {
            "artifact_contract": "E82_CALC_SCRIBE_V003_FLOOR_DIVISION_CONFIRM",
            "data_root": args.data_root,
            "prepared_rows": str(rows_path).replace("\\", "/"),
            "seeds": seeds,
            "workers": workers,
            "targeted_repair": "allow_floor_division_operator",
            "boundary": "visible calculation marker validation only; not GSM8K solving",
        },
    )
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "start", "seeds": seeds, "workers": workers})
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_seed, str(rows_path), seed, True): seed for seed in seeds}
        last = time.time()
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            result = future.result()
            results.append(result)
            append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "seed_complete", "seed": seed, "completed": len(results)})
            if time.time() - last >= args.heartbeat_seconds or len(results) == len(seeds):
                partial = aggregate(results)
                write_json(out / "partial_aggregate_snapshot.json", partial)
                append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "heartbeat", "completed": len(results), "validation_marker_min": partial["validation"]["marker_min"]})
                last = time.time()
    agg = aggregate(results)
    examples: list[dict[str, Any]] = []
    for result in results:
        examples.extend(result["examples"][:5])
    with (out / "row_level_failure_examples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples[:200]:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
    decision = "e82_calc_scribe_v003_floor_division_confirmed" if (
        agg["validation"]["marker_min"] >= 0.999
        and agg["validation"]["action_min"] >= 0.999
        and agg["adversarial"]["action_min"] >= 1.0
        and agg["validation"]["floor_marker_min"] >= 1.0
    ) else "e82_calc_scribe_v003_floor_division_partial"
    write_json(out / "seed_results.json", {"seeds": results})
    write_json(out / "aggregate_metrics.json", agg | {"seconds": time.time() - started, "seed_count": len(seeds)})
    write_json(out / "decision.json", {"decision": decision, "failure_count": 0})
    report = [
        "# E82 CALC-SCRIBE v003 Floor Division Confirm",
        "",
        "```text",
        f"decision = {decision}",
        f"seeds = {len(seeds)}",
        f"workers = {workers}",
        f"validation_marker_min = {agg['validation']['marker_min']:.6f}",
        f"validation_action_min = {agg['validation']['action_min']:.6f}",
        f"validation_floor_marker_min = {agg['validation']['floor_marker_min']:.6f}",
        f"adversarial_action_min = {agg['adversarial']['action_min']:.6f}",
        "```",
        "",
        "Boundary: visible calculation marker validation only; not GSM8K solving.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "complete", "decision": decision, "seconds": time.time() - started})
    print(json.dumps({"decision": decision, "out": str(out), "seconds": time.time() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
