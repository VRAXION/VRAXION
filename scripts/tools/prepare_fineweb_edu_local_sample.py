#!/usr/bin/env python3
"""Prepare a local FineWeb-Edu sample pack from existing Parquet shards.

This tool indexes an existing local FineWeb-Edu Parquet directory and writes a
bounded JSONL sample plus manifest/progress artifacts. It does not download and
does not copy the raw shard set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def now_ms() -> int:
    return int(time.time() * 1000)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_ok(row: dict[str, Any], min_score: float, min_tokens: int, max_tokens: int) -> bool:
    if str(row.get("language") or "").lower() != "en":
        return False
    try:
        score = float(row.get("score") or 0.0)
        token_count = int(row.get("token_count") or 0)
    except (TypeError, ValueError):
        return False
    text = str(row.get("text") or "")
    if not text.strip():
        return False
    if score < min_score:
        return False
    if token_count < min_tokens or token_count > max_tokens:
        return False
    return True


def parquet_inventory(root: Path) -> tuple[list[Path], list[dict[str, Any]], int, int, str]:
    files = sorted(root.glob("*.parquet"))
    inventory: list[dict[str, Any]] = []
    total_rows = 0
    total_bytes = 0
    schema = ""
    for path in files:
        parquet = pq.ParquetFile(path)
        if not schema:
            schema = str(parquet.schema_arrow)
        total_rows += parquet.metadata.num_rows
        total_bytes += path.stat().st_size
        inventory.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "rows": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "last_write_time": int(path.stat().st_mtime),
        })
    return files, inventory, total_rows, total_bytes, schema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=r"S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B")
    parser.add_argument("--out", default="data/high_quality_seed_v1")
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--min-score", type=float, default=3.0)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "prepare_progress.jsonl"
    sample_dir = out / "fineweb_edu"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"local_fineweb_edu_sample_{args.limit}.jsonl"
    if sample_path.exists():
        sample_path.unlink()
    started = time.time()
    append_jsonl(progress_path, {
        "event": "start",
        "timestamp_ms": now_ms(),
        "source": str(source),
        "out": str(out),
        "limit": args.limit,
    })
    files, inventory, total_rows, total_bytes, schema = parquet_inventory(source)
    append_jsonl(progress_path, {
        "event": "inventory_done",
        "timestamp_ms": now_ms(),
        "file_count": len(files),
        "source_rows": total_rows,
        "source_bytes": total_bytes,
    })
    if not files:
        raise SystemExit(f"No parquet files found in {source}")

    kept = 0
    seen = 0
    rejected = 0
    last_heartbeat = time.time()
    columns = ["text", "id", "dump", "url", "language", "token_count", "score", "int_score"]
    with sample_path.open("w", encoding="utf-8", newline="\n") as output:
        for shard_index, path in enumerate(files):
            parquet = pq.ParquetFile(path)
            append_jsonl(progress_path, {
                "event": "shard_start",
                "timestamp_ms": now_ms(),
                "shard_index": shard_index,
                "path": str(path),
                "kept": kept,
                "seen": seen,
            })
            for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
                rows = batch.to_pylist()
                for row in rows:
                    seen += 1
                    if row_ok(row, args.min_score, args.min_tokens, args.max_tokens):
                        record = {
                            "source_dataset": "HuggingFaceFW/fineweb-edu",
                            "source_config": "sample-10BT",
                            "source_path": str(path),
                            "row_id": row.get("id"),
                            "dump": row.get("dump"),
                            "url": row.get("url"),
                            "language": row.get("language"),
                            "token_count": row.get("token_count"),
                            "score": row.get("score"),
                            "int_score": row.get("int_score"),
                            "text": row.get("text"),
                        }
                        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        kept += 1
                    else:
                        rejected += 1
                    now = time.time()
                    if now - last_heartbeat >= args.heartbeat_seconds:
                        append_jsonl(progress_path, {
                            "event": "heartbeat",
                            "timestamp_ms": now_ms(),
                            "seen": seen,
                            "kept": kept,
                            "rejected": rejected,
                            "elapsed_seconds": round(now - started, 3),
                        })
                        last_heartbeat = now
                    if kept >= args.limit:
                        break
                if kept >= args.limit:
                    break
            append_jsonl(progress_path, {
                "event": "shard_done",
                "timestamp_ms": now_ms(),
                "shard_index": shard_index,
                "path": str(path),
                "kept": kept,
                "seen": seen,
            })
            if kept >= args.limit:
                break

    sha = file_sha256(sample_path)
    manifest = {
        "name": "high_quality_seed_v1",
        "created_timestamp_ms": now_ms(),
        "root": str(out),
        "purpose": "Larger local FineWeb-Edu sample pack for VRAXION dataset-backed training/probing. Raw source shards remain external input-only data.",
        "sources": [
            {
                "dataset": "HuggingFaceFW/fineweb-edu",
                "config": "sample-10BT",
                "license": "odc-by",
                "source_root": str(source),
                "source_file_count": len(files),
                "source_rows": total_rows,
                "source_bytes": total_bytes,
                "schema": schema,
            }
        ],
        "filters": {
            "language": "en",
            "min_score": args.min_score,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
        },
        "files": [
            {
                "path": str(sample_path),
                "rows": kept,
                "sha256": sha,
                "bytes": sample_path.stat().st_size,
            }
        ],
        "source_inventory": inventory,
        "stats": {
            "seen": seen,
            "kept": kept,
            "rejected": rejected,
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "license_note": "FineWeb-Edu is ODC-By and also subject to Common Crawl terms; keep attribution/source metadata with any derived training artifacts. Do not commit raw data unless explicitly approved.",
    }
    write_json(out / "manifest.json", manifest)
    append_jsonl(progress_path, {
        "event": "complete",
        "timestamp_ms": now_ms(),
        "sample_path": str(sample_path),
        "rows": kept,
        "sha256": sha,
        "bytes": sample_path.stat().st_size,
        "elapsed_seconds": manifest["stats"]["elapsed_seconds"],
    })
    print(json.dumps({
        "manifest": str(out / "manifest.json"),
        "sample_path": str(sample_path),
        "rows": kept,
        "bytes": sample_path.stat().st_size,
        "elapsed_seconds": manifest["stats"]["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
