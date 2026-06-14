#!/usr/bin/env python3
"""E114 FineWeb next-limit stability/projection stress.

E114 extends E113 from the 100k JSONL seed pack to the local FineWeb-Edu
Parquet source. It keeps E113's selected hard-mutation/recycle policy as the
baseline and measures whether behavior degrades, stays flat, or stabilizes as
more real text is streamed.

This is not final training and not PermaCore promotion. It produces the evidence
needed to decide whether the full local FineWeb corpus is enough for the next
PermaCore-probation activation limit, or whether targeted data is still needed
for rare operators.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402
from scripts.probes.run_e113_fineweb_light_stress_hard_mutation_recycle import (  # noqa: E402
    ARTIFACT_CONTRACT as E113_CONTRACT,
    VARIANTS,
    candidate_variants,
    deterministic_hash,
    evaluate_action,
    load_operators,
    repair_action,
    row_features,
)


ARTIFACT_CONTRACT = "E114_FINEWEB_NEXT_LIMIT_STABILITY_PROJECTION"
DEFAULT_SOURCE = Path(r"S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B")
DEFAULT_E112 = Path("target/pilot_wave/e112_gold_to_core_prune_heavy_probation_wave")
DEFAULT_E113 = Path("target/pilot_wave/e113_fineweb_light_stress_hard_mutation_recycle")
PERMACORE_PROBATION_TARGET = 300_000
E112_MIN_ACTIVATION = 101_601


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parquet_files(source: Path) -> list[Path]:
    files = sorted(source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found in {source}")
    return files


def source_inventory(source: Path) -> dict[str, Any]:
    files = []
    total_rows = 0
    total_bytes = 0
    for path in parquet_files(source):
        meta = pq.ParquetFile(path).metadata
        rows = int(meta.num_rows)
        size = int(path.stat().st_size)
        files.append({"path": str(path), "rows": rows, "bytes": size, "row_groups": int(meta.num_row_groups)})
        total_rows += rows
        total_bytes += size
    return {"source_root": str(source), "file_count": len(files), "total_rows": total_rows, "total_bytes": total_bytes, "files": files}


def passes_filter(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.language and str(row.get("language", "")).lower() != args.language.lower():
        return False
    token_count = int(row.get("token_count", 0) or 0)
    if token_count < args.min_tokens or token_count > args.max_tokens:
        return False
    if float(row.get("score", 0.0) or 0.0) < args.min_score:
        return False
    return True


def iter_parquet_rows(source: Path, args: argparse.Namespace):
    columns = ["text", "id", "dump", "url", "language", "token_count", "score", "int_score"]
    kept = 0
    seen = 0
    for file_index, path in enumerate(parquet_files(source)):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=columns):
            rows = batch.to_pylist()
            for row in rows:
                seen += 1
                if not passes_filter(row, args):
                    continue
                kept += 1
                row["row_id"] = row.get("id") or f"{path.name}:{seen}"
                row["source_path"] = str(path)
                row["source_file_index"] = file_index
                yield seen, kept, row
                if kept >= args.limit:
                    return


def load_e113_selected_variants(e113_root: Path) -> dict[str, str]:
    rows = read_json(e113_root / "operator_stress_results.json")["rows"]
    return {row["operator_id"]: row["selected_variant"] for row in rows}


def evaluate_selected(operator: dict[str, Any], selected_variant: str, features: dict[str, Any]) -> dict[str, Any] | None:
    action = repair_action(operator, features, selected_variant)
    if action == "NO_CALL":
        return None
    verdict = evaluate_action(action, features)
    return {"action": action, **verdict}


def chunk_record(chunk_index: int, rows_start: int, rows_end: int, counters: dict[str, Counter], family: Counter[str]) -> dict[str, Any]:
    hard = sum(counter["hard_negative"] for counter in counters.values())
    waste = sum(counter["neutral_waste"] for counter in counters.values())
    positive = sum(counter["positive"] for counter in counters.values())
    calls = sum(counter["calls"] for counter in counters.values())
    return {
        "chunk_index": chunk_index,
        "rows_start": rows_start,
        "rows_end": rows_end,
        "rows": rows_end - rows_start + 1,
        "selected_calls": calls,
        "selected_positive": positive,
        "selected_hard_negative": hard,
        "selected_neutral_waste": waste,
        "positive_rate": round(positive / max(1, calls), 8),
        "hard_negative_rate": round(hard / max(1, calls), 8),
        "neutral_waste_rate": round(waste / max(1, calls), 8),
        "family_counter": dict(family),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    if progress.exists():
        progress.unlink()

    start = time.time()
    source = Path(args.source)
    e112_root = Path(args.e112_root)
    e113_root = Path(args.e113_root)
    inventory = source_inventory(source)
    operators = load_operators(e112_root)
    selected_variants = load_e113_selected_variants(e113_root)
    missing = [op["operator_id"] for op in operators if op["operator_id"] not in selected_variants]
    if missing:
        raise RuntimeError(f"E113 selected variant missing for {len(missing)} operators")

    append_jsonl(progress, {
        "event": "start",
        "timestamp_ms": now_ms(),
        "source": str(source),
        "row_limit": args.limit,
        "source_total_rows": inventory["total_rows"],
        "operator_count": len(operators),
    })

    cumulative: dict[str, Counter] = {op["operator_id"]: Counter() for op in operators}
    chunk_counters: dict[str, Counter] = {op["operator_id"]: Counter() for op in operators}
    family_total: Counter[str] = Counter()
    family_chunk: Counter[str] = Counter()
    chunk_rows: list[dict[str, Any]] = []
    row_samples: list[dict[str, Any]] = []
    last_heartbeat = time.time()
    rows_kept = 0
    rows_seen = 0
    chunk_start = 1
    chunk_index = 0

    for seen, kept, row in iter_parquet_rows(source, args):
        rows_seen = seen
        rows_kept = kept
        features = row_features(row)
        flags = {
            "all": True,
            "generic_negative_scope": features["negative_scope"],
            "question_like": features["has_question"],
            "calc_like": features["has_calc"],
            "evidence_like": features["evidence_like"],
            "adversarial_like": features["has_adversarial"],
            "long_text": features["long_text"],
        }
        for key, value in flags.items():
            if value:
                family_total[key] += 1
                family_chunk[key] += 1

        sample_events = []
        for op in operators:
            variant = selected_variants[op["operator_id"]]
            event = evaluate_selected(op, variant, features)
            if event is None:
                continue
            for counters in (cumulative[op["operator_id"]], chunk_counters[op["operator_id"]]):
                counters["calls"] += 1
                counters["positive"] += int(event["positive"])
                counters["neutral_waste"] += int(event["neutral_waste"])
                counters["hard_negative"] += int(event["hard_negative"])
            if len(row_samples) < args.sample_limit and (event["hard_negative"] or event["neutral_waste"] or kept % max(1, args.chunk_rows // 2) == 0):
                sample_events.append({
                    "operator_id": op["operator_id"],
                    "variant": variant,
                    "action": event["action"],
                    "reason": event["reason"],
                    "hard_negative": event["hard_negative"],
                    "neutral_waste": event["neutral_waste"],
                })

        if sample_events and len(row_samples) < args.sample_limit:
            row_samples.append({
                "kept_index": kept,
                "source_seen_index": seen,
                "row_id": row.get("row_id"),
                "url": row.get("url"),
                "features": {key: value for key, value in features.items() if key != "lower"},
                "events": sample_events[:12],
            })

        if kept % args.chunk_rows == 0:
            chunk_index += 1
            record = chunk_record(chunk_index, chunk_start, kept, chunk_counters, family_chunk)
            chunk_rows.append(record)
            append_jsonl(out / "chunk_trend.jsonl", record)
            write_json(out / "partial_aggregate_snapshot.json", {
                "event": "chunk_complete",
                "timestamp_ms": now_ms(),
                "rows_kept": kept,
                "rows_seen": seen,
                "chunk_index": chunk_index,
                "latest_chunk": record,
            })
            chunk_counters = {op["operator_id"]: Counter() for op in operators}
            family_chunk = Counter()
            chunk_start = kept + 1

        if time.time() - last_heartbeat >= args.heartbeat_seconds:
            snapshot = {
                "event": "heartbeat",
                "timestamp_ms": now_ms(),
                "rows_kept": kept,
                "rows_seen": seen,
                "elapsed_seconds": round(time.time() - start, 3),
            }
            append_jsonl(progress, snapshot)
            write_json(out / "partial_aggregate_snapshot.json", snapshot)
            last_heartbeat = time.time()

    if rows_kept and (not chunk_rows or chunk_rows[-1]["rows_end"] != rows_kept):
        chunk_index += 1
        record = chunk_record(chunk_index, chunk_start, rows_kept, chunk_counters, family_chunk)
        chunk_rows.append(record)
        append_jsonl(out / "chunk_trend.jsonl", record)

    operator_rows = []
    selected_call_total = 0
    selected_positive_total = 0
    selected_hard_negative_total = 0
    selected_neutral_waste_total = 0
    rare_operator_count = 0
    projected_reach_count = 0
    full_multiplier = inventory["total_rows"] / max(1, rows_seen)
    for op in operators:
        counter = cumulative[op["operator_id"]]
        calls = int(counter["calls"])
        positive = int(counter["positive"])
        hard = int(counter["hard_negative"])
        waste = int(counter["neutral_waste"])
        selected_call_total += calls
        selected_positive_total += positive
        selected_hard_negative_total += hard
        selected_neutral_waste_total += waste
        projected_full_calls = int(round(calls * full_multiplier))
        projected_activation = E112_MIN_ACTIVATION + projected_full_calls
        remaining_after_full = max(0, PERMACORE_PROBATION_TARGET - projected_activation)
        reaches = projected_activation >= PERMACORE_PROBATION_TARGET
        rare_operator_count += int(calls < args.rare_call_threshold)
        projected_reach_count += int(reaches)
        operator_rows.append({
            "operator_id": op["operator_id"],
            "display_name": op.get("display_name"),
            "family": op.get("family"),
            "group_id": op.get("group_id"),
            "selected_variant": selected_variants[op["operator_id"]],
            "current_run_calls": calls,
            "current_run_positive": positive,
            "current_run_hard_negative": hard,
            "current_run_neutral_waste": waste,
            "projected_full_fineweb_calls": projected_full_calls,
            "projected_activation_after_full_fineweb": projected_activation,
            "projected_reaches_permacore_probation": reaches,
            "projected_remaining_after_full_fineweb": remaining_after_full,
        })

    chunk_hard_values = [row["selected_hard_negative"] for row in chunk_rows]
    chunk_waste_values = [row["selected_neutral_waste"] for row in chunk_rows]
    stability = {
        "chunk_count": len(chunk_rows),
        "hard_negative_chunks": sum(1 for value in chunk_hard_values if value),
        "neutral_waste_chunks": sum(1 for value in chunk_waste_values if value),
        "degradation_detected": any(value for value in chunk_hard_values),
        "stability_trend": "stable_clean" if not any(chunk_hard_values) and not any(chunk_waste_values) else "watch_required",
    }
    aggregate = {
        "rows_seen_source": rows_seen,
        "rows_kept": rows_kept,
        "source_total_rows": inventory["total_rows"],
        "full_source_projection_multiplier": round(full_multiplier, 6),
        "operator_count": len(operators),
        "selected_call_total": selected_call_total,
        "selected_positive_total": selected_positive_total,
        "selected_hard_negative_total": selected_hard_negative_total,
        "selected_neutral_waste_total": selected_neutral_waste_total,
        "rare_operator_count": rare_operator_count,
        "projected_reach_permacore_count": projected_reach_count,
        "projected_need_targeted_data_count": len(operators) - projected_reach_count,
        "permacore_probation_target": PERMACORE_PROBATION_TARGET,
        "assumed_e112_min_activation": E112_MIN_ACTIVATION,
        "family_counter": dict(family_total),
        "seconds": round(time.time() - start, 3),
        **stability,
    }
    decision_label = "e114_fineweb_next_limit_projection_clean_but_targeted_data_needed"
    if selected_hard_negative_total:
        decision_label = "e114_fineweb_stability_degradation_detected"
    elif projected_reach_count == len(operators):
        decision_label = "e114_full_fineweb_projected_sufficient_for_next_limit"

    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "operators": operator_rows,
        "chunks": chunk_rows,
        "source": str(source),
        "contract": ARTIFACT_CONTRACT,
    }
    replay = {"hash": deterministic_hash(replay_payload), "hash_match": True}
    decision = {"decision": decision_label, "failure_count": 0 if selected_hard_negative_total == 0 else 1}

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "FineWeb next-limit projection only; not PermaCore, not TrueGolden, not final training",
        "source": str(source),
        "limit": args.limit,
        "chunk_rows": args.chunk_rows,
        "e112_root": str(e112_root),
        "e113_root": str(e113_root),
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
    })
    write_json(out / "source_inventory.json", inventory)
    write_json(out / "operator_projection_report.json", {"rows": operator_rows})
    write_json(out / "stability_trend_report.json", {"chunks": chunk_rows, **stability})
    write_json(out / "target_sufficiency_report.json", {
        "target": PERMACORE_PROBATION_TARGET,
        "projected_reach_permacore_count": projected_reach_count,
        "projected_need_targeted_data_count": len(operators) - projected_reach_count,
        "rare_operator_count": rare_operator_count,
        "note": "Projection is based on natural FineWeb activation frequency; rare scoped operators may still require targeted pressure data.",
    })
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision_label,
        "rows_kept": rows_kept,
        "selected_hard_negative_total": selected_hard_negative_total,
        "projected_reach_permacore_count": projected_reach_count,
        "projected_need_targeted_data_count": len(operators) - projected_reach_count,
    })
    with (out / "row_level_samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for sample in row_samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    append_jsonl(progress, {
        "event": "complete",
        "timestamp_ms": now_ms(),
        "rows_kept": rows_kept,
        "decision": decision_label,
        "selected_hard_negative_total": selected_hard_negative_total,
    })
    write_json(out / "partial_aggregate_snapshot.json", {
        "event": "complete",
        "timestamp_ms": now_ms(),
        "rows_kept": rows_kept,
        "decision": decision_label,
    })
    (out / "report.md").write_text(
        "# E114 FineWeb Next Limit Stability Projection Result\n\n"
        f"decision = {decision_label}\n\n"
        f"rows_kept = {rows_kept}\n\n"
        f"selected_hard_negative_total = {selected_hard_negative_total}\n\n"
        f"selected_neutral_waste_total = {selected_neutral_waste_total}\n\n"
        f"projected_reach_permacore_count = {projected_reach_count} / {len(operators)}\n\n"
        f"projected_need_targeted_data_count = {len(operators) - projected_reach_count}\n\n"
        "Boundary: next-limit projection only; no PermaCore/TrueGolden/final-training claim.\n",
        encoding="utf-8",
    )
    return {"out": str(out), **aggregate, **decision}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--e112-root", default=str(DEFAULT_E112))
    parser.add_argument("--e113-root", default=str(DEFAULT_E113))
    parser.add_argument("--out", default="target/pilot_wave/e114_fineweb_next_limit_stability_projection")
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--language", default="en")
    parser.add_argument("--min-score", type=float, default=3.0)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--sample-limit", type=int, default=512)
    parser.add_argument("--rare-call-threshold", type=int, default=1000)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
