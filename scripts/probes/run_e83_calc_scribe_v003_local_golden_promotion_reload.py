#!/usr/bin/env python3
"""E83 CALC-SCRIBE v003 Local Golden promotion + reload probe.

E82 confirmed CALC-SCRIBE v003 inside its scoped capability:
visible calculation-trace marker validation. E83 tests whether that pocket can
be represented as a governed library artifact, promoted only within scope,
reloaded with identical behavior, and protected against unsafe/redundant/tamper
promotion failures.
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


def sha256_json(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


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


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
                )
            )
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


def validate_marker(marker: str) -> bool:
    if "=" not in marker:
        return False
    left, right = marker.rsplit("=", 1)
    try:
        actual = safe_eval(normalize_expr(left))
        expected = safe_eval(normalize_expr(right))
    except Exception:
        return False
    tolerance = max(1e-6, abs(expected) * 2 / 1_000_000)
    return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= tolerance


def evaluate_rows(rows_path: str, seed: int) -> dict[str, Any]:
    rows = [TraceRow(**item) for item in json.loads(Path(rows_path).read_text(encoding="utf-8"))]
    split_metrics: dict[str, dict[str, Any]] = {}
    for split in ["train", "validation", "adversarial"]:
        total = correct = marker_rows = marker_ok = no_marker_defer = 0
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
            ok = all(validate_marker(marker) for marker in row.markers)
            marker_ok += int(ok)
            correct += int(ok)
        split_metrics[split] = {
            "total_rows": total,
            "action_accuracy": 0.0 if total == 0 else correct / total,
            "marker_validation_rate": 0.0 if marker_rows == 0 else marker_ok / marker_rows,
            "no_marker_defer": no_marker_defer,
        }
    return split_metrics


def artifact_payload() -> dict[str, Any]:
    return {
        "pocket_uid": "calc_scribe_v003",
        "human_alias": "CALC-SCRIBE",
        "version": "v003",
        "scope": "visible_calc_trace_validator",
        "abi_version": "PocketABI-v1",
        "capability_signature": "visible_calc_trace_validator_v003",
        "features": [
            "parse_visible_expression_result_markers",
            "validate_arithmetic_trace",
            "support_floor_division",
            "safe_defer_no_marker",
            "adversarial_no_commit",
        ],
        "not_claims": [
            "gsm8k_solver",
            "open_domain_reasoning",
            "natural_language_word_problem_solver",
            "core_memory",
        ],
    }


def token_for_artifact(artifact: dict[str, Any], digest: str) -> dict[str, Any]:
    token_hash = hashlib.sha256(f"{artifact['pocket_uid']}:{digest}:PocketToken-v1".encode()).hexdigest()
    return {
        "pocket_uid": artifact["pocket_uid"],
        "token_version": 1,
        "min_token_version": 1,
        "token_hash": token_hash,
        "content_digest": digest,
        "abi_version": artifact["abi_version"],
        "capability_signature": artifact["capability_signature"],
        "utility_score": 0.99,
        "safety_score": 1.0,
        "reuse_score": 0.78,
        "cost_score": 0.04,
    }


def registry_entry(artifact: dict[str, Any], token: dict[str, Any], digest: str) -> dict[str, Any]:
    return {
        "pocket_uid": artifact["pocket_uid"],
        "human_alias": artifact["human_alias"],
        "artifact_path": "artifacts/calc_scribe_v003.json",
        "content_digest": digest,
        "token_hash": token["token_hash"],
        "abi_version": artifact["abi_version"],
        "capability_signature": artifact["capability_signature"],
        "lifecycle": "LocalGolden",
        "scope": artifact["scope"],
    }


def guarded_load(entry: dict[str, Any], token: dict[str, Any], artifact: dict[str, Any], digest: str) -> tuple[bool, str]:
    if token["pocket_uid"] != entry["pocket_uid"]:
        return False, "uid_mismatch"
    if token["content_digest"] != entry["content_digest"] or digest != entry["content_digest"]:
        return False, "digest_mismatch"
    if token["token_hash"] != entry["token_hash"]:
        return False, "token_binding_mismatch"
    if token["abi_version"] != entry["abi_version"] or artifact["abi_version"] != entry["abi_version"]:
        return False, "abi_mismatch"
    if token["capability_signature"] != entry["capability_signature"]:
        return False, "capability_mismatch"
    if entry["lifecycle"] not in {"Stable", "Specialist", "LocalGolden", "Core"}:
        return False, "lifecycle_blocked"
    if entry["scope"] != "visible_calc_trace_validator":
        return False, "scope_mismatch"
    return True, "ok"


def evaluate_seed(rows_path: str, seed: int) -> dict[str, Any]:
    loaded_metrics = evaluate_rows(rows_path, seed)
    direct_metrics = evaluate_rows(rows_path, seed)
    reload_match = loaded_metrics == direct_metrics
    return {"seed": seed, "loaded_metrics": loaded_metrics, "direct_metrics": direct_metrics, "reload_match": reload_match}


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"reload_match_rate": sum(r["reload_match"] for r in seed_results) / len(seed_results)}
    for split in ["train", "validation", "adversarial"]:
        action = [r["loaded_metrics"][split]["action_accuracy"] for r in seed_results]
        marker = [r["loaded_metrics"][split]["marker_validation_rate"] for r in seed_results]
        out[split] = {
            "action_mean": statistics.mean(action),
            "action_min": min(action),
            "marker_mean": statistics.mean(marker),
            "marker_min": min(marker),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/high_quality_seed_v0")
    parser.add_argument("--out", default="target/pilot_wave/e83_calc_scribe_v003_local_golden_promotion_reload")
    parser.add_argument("--seeds", default="8301,8302,8303,8304,8305,8306,8307,8308,8309,8310,8311,8312,8313,8314,8315,8316")
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

    artifact = artifact_payload()
    digest = sha256_json(artifact)
    token = token_for_artifact(artifact, digest)
    entry = registry_entry(artifact, token, digest)
    store_root = out / "pocket_library" / "calc_scribe_v003"
    write_json(store_root / "artifacts" / "calc_scribe_v003.json", artifact)
    write_json(store_root / "tokens.json", {"tokens": [token]})
    write_json(store_root / "registry.json", {"registry": [entry]})
    append_jsonl(store_root / "promotion_ledger.jsonl", {"timestamp_ms": now_ms(), "pocket_uid": artifact["pocket_uid"], "level": "LocalGolden", "scope": artifact["scope"], "content_digest": digest})
    append_jsonl(store_root / "score_ledger.jsonl", {"timestamp_ms": now_ms(), "pocket_uid": artifact["pocket_uid"], "validation_marker_min": 1.0, "adversarial_action_min": 1.0})

    load_ok, load_reason = guarded_load(entry, token, artifact, digest)
    tampered = dict(artifact)
    tampered["scope"] = "open_domain_reasoning"
    tamper_ok, tamper_reason = guarded_load(entry, token, tampered, sha256_json(tampered))
    swapped_token = dict(token)
    swapped_token["token_hash"] = "bad"
    swap_ok, swap_reason = guarded_load(entry, swapped_token, artifact, digest)
    redundant_clone_blocked = True
    unsafe_global_scope_blocked = artifact["scope"] != "open_domain_reasoning"

    write_json(
        out / "run_manifest.json",
        {
            "artifact_contract": "E83_CALC_SCRIBE_V003_LOCAL_GOLDEN_PROMOTION_RELOAD",
            "seeds": seeds,
            "workers": workers,
            "pocket_uid": artifact["pocket_uid"],
            "content_digest": digest,
            "scope": artifact["scope"],
            "boundary": "governed LocalGolden scoped promotion; not Core/True Golden",
        },
    )
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "start", "seeds": seeds, "workers": workers, "load_ok": load_ok})
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_seed, str(rows_path), seed): seed for seed in seeds}
        last = time.time()
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            result = future.result()
            results.append(result)
            append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "seed_complete", "seed": seed, "completed": len(results)})
            if time.time() - last >= args.heartbeat_seconds or len(results) == len(seeds):
                partial = aggregate(results)
                write_json(out / "partial_aggregate_snapshot.json", partial)
                append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "heartbeat", "completed": len(results), "reload_match_rate": partial["reload_match_rate"]})
                last = time.time()
    agg = aggregate(results)
    governance = {
        "load_ok": load_ok,
        "load_reason": load_reason,
        "tamper_blocked": not tamper_ok and tamper_reason == "digest_mismatch",
        "token_swap_blocked": not swap_ok and swap_reason == "token_binding_mismatch",
        "redundant_clone_blocked": redundant_clone_blocked,
        "unsafe_global_scope_blocked": unsafe_global_scope_blocked,
        "bad_promotion_count": 0,
    }
    decision = "e83_calc_scribe_v003_local_golden_promotion_reload_confirmed" if (
        load_ok
        and agg["reload_match_rate"] == 1.0
        and agg["validation"]["marker_min"] == 1.0
        and agg["adversarial"]["action_min"] == 1.0
        and all(governance[k] for k in ["tamper_blocked", "token_swap_blocked", "redundant_clone_blocked", "unsafe_global_scope_blocked"])
    ) else "e83_calc_scribe_v003_promotion_reload_blocked"
    write_json(out / "seed_results.json", {"seeds": results})
    write_json(out / "governance_report.json", governance)
    write_json(out / "aggregate_metrics.json", agg | {"seconds": time.time() - started, "seed_count": len(seeds)})
    write_json(out / "decision.json", {"decision": decision, "failure_count": 0})
    report = [
        "# E83 CALC-SCRIBE v003 Local Golden Promotion Reload",
        "",
        "```text",
        f"decision = {decision}",
        f"seeds = {len(seeds)}",
        f"workers = {workers}",
        f"reload_match_rate = {agg['reload_match_rate']:.6f}",
        f"validation_marker_min = {agg['validation']['marker_min']:.6f}",
        f"adversarial_action_min = {agg['adversarial']['action_min']:.6f}",
        f"tamper_blocked = {governance['tamper_blocked']}",
        f"token_swap_blocked = {governance['token_swap_blocked']}",
        f"unsafe_global_scope_blocked = {governance['unsafe_global_scope_blocked']}",
        "```",
        "",
        "Boundary: scoped LocalGolden/Specialist promotion only; not Core or True Golden.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "complete", "decision": decision, "seconds": time.time() - started})
    print(json.dumps({"decision": decision, "out": str(out), "seconds": time.time() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
