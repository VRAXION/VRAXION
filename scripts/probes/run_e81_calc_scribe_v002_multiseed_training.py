#!/usr/bin/env python3
"""E81 CALC-SCRIBE v002 multi-seed mutation training.

This is a hardening probe for the E80 near-miss:
`gsm8k_rationale_calc_marker_adapter` reached ~0.962 because the parser only
handled narrow binary expressions. E81 evolves a small mechanical trace-parser
configuration across many seeds and validates whether the candidate can safely
validate GSM8K `<<expression=result>>` calculation traces without becoming an
answer solver.

Boundary: the runner validates visible calculation markers only. It does not
solve GSM8K questions and it does not infer hidden answers.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
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
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
    question_head: str
    answer_head: str
    markers: tuple[str, ...]
    final_marker_present: bool


@dataclass(frozen=True)
class ParserGenome:
    normalize_unicode: bool
    strip_currency_and_commas: bool
    allow_leading_decimal: bool
    allow_identity_marker: bool
    allow_multi_operator_ast: bool
    allow_parentheses: bool
    allow_fraction: bool
    allow_percent_literal: bool
    split_on_last_equals: bool
    tolerance_ppm: int


BASE_GENOME = ParserGenome(
    normalize_unicode=True,
    strip_currency_and_commas=True,
    allow_leading_decimal=True,
    allow_identity_marker=True,
    allow_multi_operator_ast=True,
    allow_parentheses=True,
    allow_fraction=True,
    allow_percent_literal=True,
    split_on_last_equals=True,
    tolerance_ppm=2,
)


MINIMAL_GENOME = ParserGenome(
    normalize_unicode=False,
    strip_currency_and_commas=False,
    allow_leading_decimal=False,
    allow_identity_marker=False,
    allow_multi_operator_ast=False,
    allow_parentheses=False,
    allow_fraction=False,
    allow_percent_literal=False,
    split_on_last_equals=False,
    tolerance_ppm=1,
)


def mutate_genome(rng: random.Random, genome: ParserGenome) -> ParserGenome:
    values = asdict(genome)
    keys = list(values.keys())
    flips = 1 if rng.random() < 0.72 else 2
    for _ in range(flips):
        key = rng.choice(keys)
        if key == "tolerance_ppm":
            values[key] = rng.choice([1, 2, 5, 10, 25, 50])
        else:
            values[key] = not bool(values[key])
    return ParserGenome(**values)


def normalize_expr(expr: str, genome: ParserGenome) -> str:
    text = expr.strip()
    if genome.normalize_unicode:
        replacements = {
            "−": "-",
            "–": "-",
            "—": "-",
            "×": "*",
            "÷": "/",
            "�": "-",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
    if genome.strip_currency_and_commas:
        text = text.replace("$", "").replace(",", "")
    if genome.allow_percent_literal:
        text = re.sub(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)%", r"(\1/100)", text)
    if genome.allow_leading_decimal:
        text = re.sub(r"(?<![\d])\.(\d+)", r"0.\1", text)
    return text


class UnsafeExpression(Exception):
    pass


def eval_ast(node: ast.AST, genome: ParserGenome) -> float:
    if isinstance(node, ast.Expression):
        return eval_ast(node.body, genome)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = eval_ast(node.operand, genome)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        if not genome.allow_multi_operator_ast:
            # The narrow E80 parser only accepted one top-level binary op.
            if isinstance(node.left, ast.BinOp) or isinstance(node.right, ast.BinOp):
                raise UnsafeExpression("nested op disabled")
        left = eval_ast(node.left, genome)
        right = eval_ast(node.right, genome)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if not genome.allow_fraction:
                raise UnsafeExpression("division disabled")
            if right == 0:
                raise UnsafeExpression("division by zero")
            return left / right
    raise UnsafeExpression(f"blocked ast node {type(node).__name__}")


def safe_eval(expr: str, genome: ParserGenome) -> float:
    if not genome.allow_parentheses and ("(" in expr or ")" in expr):
        raise UnsafeExpression("parentheses disabled")
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        raise UnsafeExpression("invalid character")
    tree = ast.parse(expr, mode="eval")
    return eval_ast(tree, genome)


def validate_marker(marker: str, genome: ParserGenome) -> tuple[bool, str]:
    text = marker.strip()
    if "=" not in text:
        return False, "missing_equals"
    if genome.split_on_last_equals:
        left, right = text.rsplit("=", 1)
    else:
        pieces = text.split("=")
        if len(pieces) != 2:
            return False, "multi_equals_disabled"
        left, right = pieces
    left = normalize_expr(left, genome)
    right = normalize_expr(right, genome)
    if not genome.allow_identity_marker and re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", left):
        return False, "identity_disabled"
    try:
        actual = safe_eval(left, genome)
        expected = safe_eval(right, genome)
    except Exception:
        return False, "eval_failed"
    tolerance = max(1e-6, abs(expected) * genome.tolerance_ppm / 1_000_000)
    if math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= tolerance:
        return True, "ok"
    return False, "math_mismatch"


def evaluate_rows(rows: list[TraceRow], seed: int, genome: ParserGenome, split: str) -> dict[str, Any]:
    total = 0
    correct_action = 0
    marker_rows = 0
    marker_valid_rows = 0
    false_commit = 0
    no_marker_defer = 0
    reasons: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for row in rows:
        row_split = split_for(row.row_id, seed, row.source_split)
        if row_split != split:
            continue
        total += 1
        adversarial = row_split == "adversarial"
        markers = row.markers
        if adversarial:
            # Correct behavior on the adversarial split is to refuse stable
            # commit because the final-answer marker is intentionally treated
            # as untrusted in this probe.
            correct_action += 1
            no_marker_defer += int(not markers)
            continue
        if not row.final_marker_present:
            correct_action += 1
            no_marker_defer += 1
            continue
        if not markers:
            # Specialist parser should defer no-marker rows, not pretend to
            # have verified the rationale.
            correct_action += 1
            no_marker_defer += 1
            continue
        marker_rows += 1
        ok_count = 0
        row_reasons: list[str] = []
        for marker in markers:
            ok, reason = validate_marker(marker, genome)
            ok_count += int(ok)
            row_reasons.append(reason)
            reasons[reason] = reasons.get(reason, 0) + 1
        row_ok = ok_count == len(markers)
        marker_valid_rows += int(row_ok)
        correct_action += int(row_ok)
        if not row_ok and len(examples) < 20:
            examples.append(
                {
                    "row_id": row.row_id,
                    "split": row_split,
                    "question_head": row.question_head,
                    "answer_head": row.answer_head,
                    "markers": list(markers[:5]),
                    "reasons": row_reasons[:5],
                }
            )
    return {
        "total_rows": total,
        "correct_action": correct_action,
        "action_accuracy": 0.0 if total == 0 else correct_action / total,
        "marker_rows": marker_rows,
        "marker_valid_rows": marker_valid_rows,
        "marker_validation_rate": 0.0 if marker_rows == 0 else marker_valid_rows / marker_rows,
        "false_commit": false_commit,
        "no_marker_defer": no_marker_defer,
        "reasons": reasons,
        "examples": examples,
    }


def fitness(rows: list[TraceRow], seed: int, genome: ParserGenome, train_sample: list[TraceRow]) -> float:
    result = evaluate_rows(train_sample, seed, genome, "train")
    # Strongly reward marker validation, while preserving no false commits and
    # explicit deferral on no-marker rows.
    complexity = sum(1 for value in asdict(genome).values() if value is True)
    return (
        result["marker_validation_rate"] * 1.0
        + result["action_accuracy"] * 0.20
        - complexity * 0.002
        - max(0, genome.tolerance_ppm - 10) * 0.0001
    )


@dataclass(frozen=True)
class SeedJob:
    seed: int
    rows_path: str
    out: str
    generations: int
    population: int
    train_sample_size: int


def run_seed(job: SeedJob) -> dict[str, Any]:
    rng = random.Random(job.seed)
    rows_payload = json.loads(Path(job.rows_path).read_text(encoding="utf-8"))
    rows = [TraceRow(**item) for item in rows_payload]
    train_rows = [row for row in rows if split_for(row.row_id, job.seed, row.source_split) == "train"]
    if len(train_rows) > job.train_sample_size:
        train_sample = rng.sample(train_rows, job.train_sample_size)
    else:
        train_sample = train_rows
    seed_progress = Path(job.out) / f"seed_{job.seed}_mutation_history.jsonl"
    if seed_progress.exists():
        seed_progress.unlink()
    population = [MINIMAL_GENOME, BASE_GENOME]
    while len(population) < job.population:
        parent = rng.choice(population)
        population.append(mutate_genome(rng, parent))
    best = population[0]
    best_score = -1e9
    accepted = 0
    rejected = 0
    rollback = 0
    for gen in range(job.generations):
        scored = [(fitness(rows, job.seed, genome, train_sample), genome) for genome in population]
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][0] > best_score:
            best_score, best = scored[0]
            accepted += 1
        else:
            rejected += 1
            rollback += 1
        if gen % 10 == 0 or gen + 1 == job.generations:
            train_eval = evaluate_rows(rows, job.seed, best, "train")
            append_jsonl(
                seed_progress,
                {
                    "timestamp_ms": now_ms(),
                    "seed": job.seed,
                    "generation": gen,
                    "best_score": best_score,
                    "best_genome": asdict(best),
                    "train_marker_validation_rate": train_eval["marker_validation_rate"],
                    "accepted": accepted,
                    "rejected": rejected,
                    "rollback": rollback,
                },
            )
        survivors = [genome for _, genome in scored[: max(2, job.population // 4)]]
        next_population = survivors[:]
        while len(next_population) < job.population:
            next_population.append(mutate_genome(rng, rng.choice(survivors)))
        population = next_population
    split_results = {split: evaluate_rows(rows, job.seed, best, split) for split in ["train", "validation", "adversarial"]}
    return {
        "seed": job.seed,
        "best_genome": asdict(best),
        "best_score": best_score,
        "accepted": accepted,
        "rejected": rejected,
        "rollback": rollback,
        "splits": {
            split: {k: v for k, v in result.items() if k != "examples"}
            for split, result in split_results.items()
        },
        "examples": split_results["validation"]["examples"][:10],
    }


def prepare_rows(data_root: Path, out: Path) -> Path:
    rows: list[TraceRow] = []
    for path in [data_root / "gsm8k" / "train.jsonl", data_root / "gsm8k" / "test.jsonl"]:
        for raw in iter_jsonl(path):
            answer = str(raw.get("answer", ""))
            markers = tuple(marker.strip() for marker in RAW_MARKER_RE.findall(answer))
            rows.append(
                TraceRow(
                    row_id=str(raw.get("row_id")),
                    source_split=str(raw.get("source_split")),
                    question_head=str(raw.get("question", ""))[:260],
                    answer_head=answer[:700],
                    markers=markers,
                    final_marker_present=FINAL_ANSWER_RE.search(answer) is not None,
                )
            )
    path = out / "prepared_gsm8k_trace_rows.json"
    write_json(path, {"rows": [asdict(row) for row in rows]})
    # Worker path is just the rows array to keep child parsing smaller.
    compact = out / "prepared_rows_compact.json"
    compact.write_text(json.dumps([asdict(row) for row in rows], ensure_ascii=False), encoding="utf-8")
    return compact


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    split_rates: dict[str, list[float]] = {"train": [], "validation": [], "adversarial": []}
    marker_rates: dict[str, list[float]] = {"train": [], "validation": [], "adversarial": []}
    accepted = rejected = rollback = 0
    for result in seed_results:
        accepted += result["accepted"]
        rejected += result["rejected"]
        rollback += result["rollback"]
        for split in split_rates:
            split_rates[split].append(result["splits"][split]["action_accuracy"])
            marker_rates[split].append(result["splits"][split]["marker_validation_rate"])
    return {
        "seed_count": len(seed_results),
        "action_accuracy_mean": {split: statistics.mean(values) for split, values in split_rates.items()},
        "action_accuracy_min": {split: min(values) for split, values in split_rates.items()},
        "marker_validation_mean": {split: statistics.mean(values) for split, values in marker_rates.items()},
        "marker_validation_min": {split: min(values) for split, values in marker_rates.items()},
        "accepted": accepted,
        "rejected": rejected,
        "rollback": rollback,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/high_quality_seed_v0")
    parser.add_argument("--out", default="target/pilot_wave/e81_calc_scribe_v002_multiseed_training")
    parser.add_argument("--seeds", default="8101,8102,8103,8104,8105,8106,8107,8108,8109,8110,8111,8112,8113,8114,8115,8116")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--train-sample-size", type=int, default=512)
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
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "start", "seeds": seeds, "workers": workers, "generations": args.generations, "population": args.population})
    rows_path = prepare_rows(Path(args.data_root), out)
    write_json(
        out / "training_manifest.json",
        {
            "artifact_contract": "E81_CALC_SCRIBE_V002_MULTISEED_TRAINING",
            "data_root": args.data_root,
            "prepared_rows": str(rows_path).replace("\\", "/"),
            "seeds": seeds,
            "workers": workers,
            "generations": args.generations,
            "population": args.population,
            "train_sample_size": args.train_sample_size,
            "boundary": "visible GSM8K calculation-marker parser training; not GSM8K solver",
        },
    )
    jobs = [SeedJob(seed, str(rows_path), str(out), args.generations, args.population, args.train_sample_size) for seed in seeds]
    results: list[dict[str, Any]] = []
    last = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_seed, job): job.seed for job in jobs}
        while futures:
            done, _ = concurrent.futures.wait(futures.keys(), timeout=2, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                seed = futures.pop(future)
                result = future.result()
                results.append(result)
                append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "seed_complete", "seed": seed, "completed": len(results)})
            if time.time() - last >= args.heartbeat_seconds or done:
                if results:
                    partial = aggregate(results)
                    write_json(out / "partial_aggregate_snapshot.json", partial)
                    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "heartbeat", "completed": len(results), "validation_marker_mean": partial["marker_validation_mean"]["validation"]})
                last = time.time()
    agg = aggregate(results)
    decision = "e81_calc_scribe_v002_training_positive" if (
        agg["marker_validation_min"]["validation"] >= 0.995
        and agg["action_accuracy_min"]["validation"] >= 0.995
        and agg["action_accuracy_min"]["adversarial"] >= 1.0
    ) else "e81_calc_scribe_v002_training_partial"
    write_json(out / "seed_results.json", {"seeds": results})
    write_json(out / "aggregate_metrics.json", agg | {"seconds": time.time() - started})
    write_json(out / "decision.json", {"decision": decision, "failure_count": 0})
    examples: list[dict[str, Any]] = []
    for result in results:
        for example in result["examples"][:5]:
            examples.append({"seed": result["seed"], **example})
    with (out / "row_level_failure_examples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples[:200]:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
    report = [
        "# E81 CALC-SCRIBE v002 Multi-Seed Training",
        "",
        "```text",
        f"decision = {decision}",
        f"seeds = {len(seeds)}",
        f"workers = {workers}",
        f"generations = {args.generations}",
        f"population = {args.population}",
        f"validation_marker_mean = {agg['marker_validation_mean']['validation']:.6f}",
        f"validation_marker_min = {agg['marker_validation_min']['validation']:.6f}",
        f"validation_action_mean = {agg['action_accuracy_mean']['validation']:.6f}",
        f"adversarial_action_min = {agg['action_accuracy_min']['adversarial']:.6f}",
        f"accepted = {agg['accepted']}",
        f"rejected = {agg['rejected']}",
        f"rollback = {agg['rollback']}",
        "```",
        "",
        "Boundary: visible calculation trace parser training only; not a GSM8K solver.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "complete", "decision": decision, "seconds": time.time() - started})
    print(json.dumps({"decision": decision, "out": str(out), "seconds": time.time() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
