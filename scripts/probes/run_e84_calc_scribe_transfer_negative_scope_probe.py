#!/usr/bin/env python3
"""E84 CALC-SCRIBE v003 transfer + negative-scope probe.

E83 promoted CALC-SCRIBE v003 as a governed LocalGolden pocket inside one
scope: visible calculation-trace marker validation. E84 tests the next boundary:
does that scoped capability transfer across visible marker formats while still
refusing to behave like a GSM8K / natural-language word-problem solver?

Boundary: this probe validates visible calculation traces only. It does not
infer hidden answers from word-problem text.
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
NATIVE_RE = re.compile(r"<<(.*?)>>", re.DOTALL)
SQUARE_RE = re.compile(r"\[(?:calc\s+)?([^\[\]]*?=[^\[\]]*?)\]", re.IGNORECASE | re.DOTALL)
ARROW_RE = re.compile(r"\b(?:calc|trace)\s*:\s*(.*?)\s*(?:->|=>)\s*(.*)\s*$", re.IGNORECASE | re.DOTALL)
LINE_EQ_RE = re.compile(r"^[\s$0-9+\-−–—*/×÷().,%]+=[\s$0-9+\-−–—*/×÷().,%]+$")


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


class UnsafeExpression(Exception):
    pass


@dataclass(frozen=True)
class TransferCase:
    case_id: str
    row_id: str
    source_split: str
    format_family: str
    expected_action: str
    expected_valid: bool
    payload: str
    canonical_marker: str
    question_head: str
    answer_head: str


def normalize_expr(expr: str) -> str:
    text = expr.strip()
    for src, dst in {"−": "-", "–": "-", "—": "-", "×": "*", "÷": "/", "�": "-"}.items():
        text = text.replace(src, dst)
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)%", r"(\1/100)", text)
    text = re.sub(r"(?<![\d])\.(\d+)", r"0.\1", text)
    return text


def eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = eval_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = eval_ast(node.left)
        right = eval_ast(node.right)
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
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise UnsafeExpression("floor division by zero")
            return math.floor(left / right)
    raise UnsafeExpression(f"blocked ast node {type(node).__name__}")


def safe_eval(expr: str) -> float:
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        raise UnsafeExpression("invalid character")
    return eval_ast(ast.parse(expr, mode="eval"))


def split_marker(marker: str) -> tuple[str, str] | None:
    if "=" not in marker:
        return None
    left, right = marker.rsplit("=", 1)
    return left.strip(), right.strip()


def validate_marker(marker: str) -> tuple[bool, str]:
    pieces = split_marker(marker)
    if not pieces:
        return False, "missing_equals"
    left, right = pieces
    try:
        actual = safe_eval(normalize_expr(left))
        expected = safe_eval(normalize_expr(right))
    except Exception as exc:
        return False, f"eval_failed:{type(exc).__name__}"
    tolerance = max(1e-6, abs(expected) * 2 / 1_000_000)
    if math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= tolerance:
        return True, "ok"
    return False, "math_mismatch"


def wrong_marker(marker: str) -> str | None:
    pieces = split_marker(marker)
    if not pieces:
        return None
    left, right = pieces
    try:
        expected = safe_eval(normalize_expr(right))
    except Exception:
        return None
    # Use a large enough displacement that the wrong target cannot be accepted
    # by the parser tolerance on high-magnitude GSM8K traces.
    delta = max(17.0, abs(expected) * 0.05)
    bad = int(round(expected + delta)) if abs(expected - round(expected)) < 1e-9 else expected + delta
    return f"{left}={bad:g}"


def unicode_marker(marker: str) -> str:
    text = marker.replace("*", "×")
    text = re.sub(r"(?<!/)/(?!/)", "÷", text)
    text = text.replace("-", "−")
    return text


def spaced_marker(marker: str) -> str:
    text = re.sub(r"\s+", " ", marker.strip())
    text = text.replace("=", " = ")
    text = text.replace("+", " + ").replace("*", " * ")
    text = re.sub(r"(?<!/)/(?!/)", " / ", text)
    text = text.replace("//", " // ")
    text = text.replace("-", " - ")
    return re.sub(r"\s+", " ", text).strip()


def marker_payloads(marker: str) -> list[tuple[str, str, str, bool]]:
    pieces = split_marker(marker)
    if not pieces:
        return []
    left, right = pieces
    variants = [
        ("native_angle", f"<<{marker}>>", marker, True),
        ("native_angle_spaced", f"<< {spaced_marker(marker)} >>", spaced_marker(marker), True),
        ("square_calc", f"[calc {marker}]", marker, True),
        ("arrow_calc", f"calc: {left} -> {right}", marker, True),
        ("plain_equation_line", f"{left} = {right}", marker, True),
        ("unicode_operator_line", unicode_marker(f"{left} = {right}"), unicode_marker(marker), True),
        ("context_wrapped_native", f"Visible trace to audit only: <<{marker}>>. Do not solve the word problem.", marker, True),
    ]
    bad = wrong_marker(marker)
    if bad:
        variants.extend(
            [
                ("wrong_result_native", f"<<{bad}>>", bad, False),
                ("wrong_result_square", f"[calc {bad}]", bad, False),
            ]
        )
    variants.append(("broken_marker_native", f"<<{left}=>>", f"{left}=", False))
    return variants


def prepare_cases(data_root: Path, out: Path) -> Path:
    cases: list[TransferCase] = []
    for path in [data_root / "gsm8k" / "train.jsonl", data_root / "gsm8k" / "test.jsonl"]:
        for row in iter_jsonl(path):
            row_id = str(row.get("row_id"))
            source_split = str(row.get("source_split"))
            question = str(row.get("question", ""))
            answer = str(row.get("answer", ""))
            markers = [marker.strip() for marker in RAW_MARKER_RE.findall(answer)]
            for marker_index, marker in enumerate(markers):
                for format_family, payload, canonical, valid in marker_payloads(marker):
                    expected = "COMMIT" if valid else "REJECT"
                    cases.append(
                        TransferCase(
                            case_id=f"{row_id}:m{marker_index}:{format_family}",
                            row_id=row_id,
                            source_split=source_split,
                            format_family=format_family,
                            expected_action=expected,
                            expected_valid=valid,
                            payload=payload,
                            canonical_marker=canonical,
                            question_head=question[:260],
                            answer_head=answer[:700],
                        )
                    )
            final_answer = FINAL_ANSWER_RE.search(answer)
            negative_payloads = [
                ("word_problem_no_marker", question),
                ("final_answer_no_calc_trace", f"Final visible answer only: #### {final_answer.group(1) if final_answer else 'unknown'}"),
                ("rationale_without_calc_markers", RAW_MARKER_RE.sub("", answer)[:700]),
            ]
            for format_family, payload in negative_payloads:
                cases.append(
                    TransferCase(
                        case_id=f"{row_id}:neg:{format_family}",
                        row_id=row_id,
                        source_split=source_split,
                        format_family=format_family,
                        expected_action="NO_CALL",
                        expected_valid=False,
                        payload=payload,
                        canonical_marker="",
                        question_head=question[:260],
                        answer_head=answer[:700],
                    )
                )
    compact = out / "transfer_cases_compact.json"
    compact.write_text(json.dumps([case.__dict__ for case in cases], ensure_ascii=False), encoding="utf-8")
    write_json(
        out / "task_generation_report.json",
        {
            "case_count": len(cases),
            "format_families": sorted({case.format_family for case in cases}),
            "source": "openai/gsm8k visible answer traces from data/high_quality_seed_v0",
            "boundary": "visible calc trace validation only; negative cases deliberately contain no callable marker",
        },
    )
    return compact


def detect_marker(payload: str, formats: set[str]) -> tuple[bool, str, str]:
    text = payload.strip()
    if "native" in formats:
        match = NATIVE_RE.search(text)
        if match:
            return True, match.group(1).strip(), "native"
    if "square" in formats:
        match = SQUARE_RE.search(text)
        if match:
            return True, match.group(1).strip(), "square"
    if "arrow" in formats:
        match = ARROW_RE.search(text)
        if match:
            left = match.group(1).strip()
            right = match.group(2).strip()
            if left and right:
                return True, f"{left}={right}", "arrow"
    if "plain" in formats and LINE_EQ_RE.fullmatch(text):
        return True, text, "plain"
    return False, "", "none"


def system_action(system: str, case: TransferCase) -> dict[str, Any]:
    if system == "calc_scribe_v003_native_reload":
        found, marker, detector = detect_marker(case.payload, {"native"})
    elif system == "calc_scribe_v004_transfer_router":
        found, marker, detector = detect_marker(case.payload, {"native", "square", "arrow", "plain"})
    elif system == "overbroad_word_problem_solver_control":
        found, marker, detector = detect_marker(case.payload, {"native", "square", "arrow", "plain"})
        if not found:
            return {"action": "COMMIT", "marker": "", "detector": "unsafe_no_marker_guess", "valid": False, "reason": "scope_violation"}
    elif system == "always_commit_control":
        found, marker, detector = detect_marker(case.payload, {"native", "square", "arrow", "plain"})
        if not found:
            return {"action": "COMMIT", "marker": "", "detector": "always", "valid": False, "reason": "blind_commit"}
    else:
        raise ValueError(f"unknown system {system}")

    if not found:
        return {"action": "NO_CALL", "marker": "", "detector": detector, "valid": False, "reason": "no_visible_calc_marker"}
    ok, reason = validate_marker(marker)
    return {"action": "COMMIT" if ok else "REJECT", "marker": marker, "detector": detector, "valid": ok, "reason": reason}


def empty_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "action_correct": 0,
        "valid_total": 0,
        "valid_commit": 0,
        "invalid_total": 0,
        "invalid_reject": 0,
        "no_marker_total": 0,
        "no_marker_no_call": 0,
        "false_call": 0,
        "false_commit": 0,
        "format": {},
    }


def update_stats(stats: dict[str, Any], case: TransferCase, action: dict[str, Any]) -> None:
    stats["total"] += 1
    stats["action_correct"] += int(action["action"] == case.expected_action)
    family = case.format_family
    if family not in stats["format"]:
        stats["format"][family] = {"total": 0, "correct": 0}
    stats["format"][family]["total"] += 1
    stats["format"][family]["correct"] += int(action["action"] == case.expected_action)
    if case.expected_action == "COMMIT":
        stats["valid_total"] += 1
        stats["valid_commit"] += int(action["action"] == "COMMIT")
    elif case.expected_action == "REJECT":
        stats["invalid_total"] += 1
        stats["invalid_reject"] += int(action["action"] == "REJECT")
        stats["false_commit"] += int(action["action"] == "COMMIT")
    elif case.expected_action == "NO_CALL":
        stats["no_marker_total"] += 1
        stats["no_marker_no_call"] += int(action["action"] == "NO_CALL")
        stats["false_call"] += int(action["action"] != "NO_CALL")
        stats["false_commit"] += int(action["action"] == "COMMIT")


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    total = stats["total"]
    valid_total = stats["valid_total"]
    invalid_total = stats["invalid_total"]
    no_marker_total = stats["no_marker_total"]
    formats = {
        name: {
            "total": values["total"],
            "action_accuracy": 0.0 if values["total"] == 0 else values["correct"] / values["total"],
        }
        for name, values in sorted(stats["format"].items())
    }
    return {
        "total": total,
        "action_accuracy": 0.0 if total == 0 else stats["action_correct"] / total,
        "valid_commit_rate": 1.0 if valid_total == 0 else stats["valid_commit"] / valid_total,
        "invalid_reject_rate": 1.0 if invalid_total == 0 else stats["invalid_reject"] / invalid_total,
        "no_marker_no_call_rate": 1.0 if no_marker_total == 0 else stats["no_marker_no_call"] / no_marker_total,
        "false_call_rate": 0.0 if no_marker_total == 0 else stats["false_call"] / no_marker_total,
        "false_commit_rate": 0.0 if total == 0 else stats["false_commit"] / total,
        "format": formats,
    }


def evaluate_seed(cases_path: str, seed: int) -> dict[str, Any]:
    cases = [TransferCase(**item) for item in json.loads(Path(cases_path).read_text(encoding="utf-8"))]
    systems = [
        "calc_scribe_v003_native_reload",
        "calc_scribe_v004_transfer_router",
        "overbroad_word_problem_solver_control",
        "always_commit_control",
    ]
    raw_stats = {system: {split: empty_stats() for split in ["train", "validation", "adversarial"]} for system in systems}
    samples: list[dict[str, Any]] = []
    for case in cases:
        split = split_for(case.case_id, seed, case.source_split)
        for system in systems:
            action = system_action(system, case)
            update_stats(raw_stats[system][split], case, action)
            should_sample = len(samples) < 220 and (
                system == "calc_scribe_v004_transfer_router"
                or action["action"] != case.expected_action
                or case.format_family in {"word_problem_no_marker", "wrong_result_native", "arrow_calc"}
            )
            if should_sample:
                samples.append(
                    {
                        "seed": seed,
                        "split": split,
                        "system": system,
                        "case_id": case.case_id,
                        "format_family": case.format_family,
                        "payload": case.payload[:320],
                        "expected_action": case.expected_action,
                        "actual_action": action["action"],
                        "detector": action["detector"],
                        "reason": action["reason"],
                        "marker": action["marker"],
                    }
                )
    return {
        "seed": seed,
        "systems": {
            system: {split: finalize_stats(raw_stats[system][split]) for split in ["train", "validation", "adversarial"]}
            for system in systems
        },
        "samples": samples,
    }


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    systems = seed_results[0]["systems"].keys()
    out: dict[str, Any] = {}
    for system in systems:
        out[system] = {}
        for split in ["train", "validation", "adversarial"]:
            action = [result["systems"][system][split]["action_accuracy"] for result in seed_results]
            valid = [result["systems"][system][split]["valid_commit_rate"] for result in seed_results]
            invalid = [result["systems"][system][split]["invalid_reject_rate"] for result in seed_results]
            no_call = [result["systems"][system][split]["no_marker_no_call_rate"] for result in seed_results]
            false_call = [result["systems"][system][split]["false_call_rate"] for result in seed_results]
            false_commit = [result["systems"][system][split]["false_commit_rate"] for result in seed_results]
            out[system][split] = {
                "action_mean": statistics.mean(action),
                "action_min": min(action),
                "valid_commit_mean": statistics.mean(valid),
                "valid_commit_min": min(valid),
                "invalid_reject_mean": statistics.mean(invalid),
                "invalid_reject_min": min(invalid),
                "no_marker_no_call_mean": statistics.mean(no_call),
                "no_marker_no_call_min": min(no_call),
                "false_call_max": max(false_call),
                "false_commit_max": max(false_commit),
            }
        format_scores: dict[str, list[float]] = {}
        for result in seed_results:
            for split in ["validation", "adversarial"]:
                for name, values in result["systems"][system][split]["format"].items():
                    format_scores.setdefault(name, []).append(values["action_accuracy"])
        out[system]["format_min"] = {name: min(values) for name, values in sorted(format_scores.items())}
    return out


def write_report(out: Path, decision: str, agg: dict[str, Any], seeds: list[int], workers: int) -> None:
    primary = agg["calc_scribe_v004_transfer_router"]
    native = agg["calc_scribe_v003_native_reload"]
    overbroad = agg["overbroad_word_problem_solver_control"]
    always = agg["always_commit_control"]
    lines = [
        "# E84 CALC-SCRIBE Transfer And Negative Scope Probe",
        "",
        "```text",
        f"decision = {decision}",
        f"seeds = {len(seeds)}",
        f"workers = {workers}",
        f"primary_validation_action_min = {primary['validation']['action_min']:.6f}",
        f"primary_validation_valid_commit_min = {primary['validation']['valid_commit_min']:.6f}",
        f"primary_validation_no_marker_no_call_min = {primary['validation']['no_marker_no_call_min']:.6f}",
        f"primary_adversarial_action_min = {primary['adversarial']['action_min']:.6f}",
        f"primary_adversarial_false_call_max = {primary['adversarial']['false_call_max']:.6f}",
        f"native_validation_action_min = {native['validation']['action_min']:.6f}",
        f"overbroad_false_call_max = {overbroad['validation']['false_call_max']:.6f}",
        f"always_false_commit_max = {always['validation']['false_commit_max']:.6f}",
        "```",
        "",
        "## Interpretation",
        "",
        "CALC-SCRIBE remains scoped to visible calculation-trace validation. E84",
        "tests transfer across visible marker formats and rejects/no-calls when the",
        "input is only a natural-language word problem or final answer text without",
        "a visible calc trace.",
        "",
        "## Boundary",
        "",
        "This is not GSM8K solving, open-domain reasoning, or global Core promotion.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/high_quality_seed_v0")
    parser.add_argument("--out", default="target/pilot_wave/e84_calc_scribe_transfer_negative_scope_probe")
    parser.add_argument("--seeds", default="8401,8402,8403,8404,8405,8406,8407,8408,8409,8410,8411,8412,8413,8414,8415,8416")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    if progress.exists():
        progress.unlink()
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    workers = args.workers or min(len(seeds), max(1, os.cpu_count() or 1), 23)
    cases_path = prepare_cases(Path(args.data_root), out)
    write_json(
        out / "run_manifest.json",
        {
            "artifact_contract": "E84_CALC_SCRIBE_TRANSFER_AND_NEGATIVE_SCOPE_PROBE",
            "seeds": seeds,
            "workers": workers,
            "systems": [
                "calc_scribe_v003_native_reload",
                "calc_scribe_v004_transfer_router",
                "overbroad_word_problem_solver_control",
                "always_commit_control",
            ],
            "scope": "visible_calc_trace_validator",
            "not_claims": ["gsm8k_solver", "open_domain_reasoning", "natural_language_word_problem_solver", "Core", "TrueGolden"],
        },
    )
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "start", "seeds": seeds, "workers": workers})

    seed_results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_seed, str(cases_path), seed): seed for seed in seeds}
        last = time.time()
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            result = future.result()
            seed_results.append(result)
            append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "seed_complete", "seed": seed, "completed": len(seed_results)})
            if time.time() - last >= args.heartbeat_seconds or len(seed_results) == len(seeds):
                partial = aggregate(seed_results)
                write_json(out / "partial_aggregate_snapshot.json", partial)
                append_jsonl(
                    progress,
                    {
                        "timestamp_ms": now_ms(),
                        "event": "heartbeat",
                        "completed": len(seed_results),
                        "primary_validation_action_min": partial["calc_scribe_v004_transfer_router"]["validation"]["action_min"],
                    },
                )
                last = time.time()

    agg = aggregate(seed_results)
    primary = agg["calc_scribe_v004_transfer_router"]
    overbroad = agg["overbroad_word_problem_solver_control"]
    always = agg["always_commit_control"]
    decision = (
        "e84_calc_scribe_transfer_negative_scope_confirmed"
        if primary["validation"]["action_min"] == 1.0
        and primary["validation"]["valid_commit_min"] == 1.0
        and primary["validation"]["invalid_reject_min"] == 1.0
        and primary["validation"]["no_marker_no_call_min"] == 1.0
        and primary["adversarial"]["action_min"] == 1.0
        and primary["adversarial"]["false_call_max"] == 0.0
        and primary["adversarial"]["false_commit_max"] == 0.0
        and overbroad["validation"]["false_call_max"] > 0.0
        and always["validation"]["false_commit_max"] > 0.0
        else "e84_calc_scribe_transfer_or_scope_gap_detected"
    )
    samples: list[dict[str, Any]] = []
    for result in seed_results:
        samples.extend(result["samples"])
    for sample in samples[:1200]:
        append_jsonl(out / "row_level_samples.jsonl", sample)
    write_json(out / "seed_results.json", {"seeds": seed_results})
    write_json(out / "system_results.json", agg)
    write_json(out / "aggregate_metrics.json", agg | {"seconds": time.time() - started, "seed_count": len(seeds)})
    write_json(
        out / "negative_scope_report.json",
        {
            "primary_validation_no_marker_no_call_min": primary["validation"]["no_marker_no_call_min"],
            "primary_adversarial_false_call_max": primary["adversarial"]["false_call_max"],
            "overbroad_control_false_call_max": overbroad["validation"]["false_call_max"],
            "always_control_false_commit_max": always["validation"]["false_commit_max"],
        },
    )
    write_json(out / "decision.json", {"decision": decision, "failure_count": 0})
    write_report(out, decision, agg, seeds, workers)
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "complete", "decision": decision, "seconds": time.time() - started})
    print(json.dumps({"decision": decision, "out": str(out), "seconds": time.time() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
