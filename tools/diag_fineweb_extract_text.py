"""Extract first ~100 MB of raw text from a FineWeb-EDU parquet shard.

Reads the `text` column row-group by row-group, concatenates documents with a
single newline separator, stops once the running byte count passes the target.
Writes:
  - output/data/fineweb_edu_100mb.txt       (raw UTF-8 text, ~100 MB)
  - output/data/fineweb_edu_100mb_meta.json (provenance + sizes)

Run:
    python3 tools/diag_fineweb_extract_text.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "fineweb_edu_sample_000_00000.parquet"
OUT_DIR = REPO_ROOT / "output" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT = OUT_DIR / "fineweb_edu_100mb.txt"
OUT_META = OUT_DIR / "fineweb_edu_100mb_meta.json"

TARGET_BYTES = 100 * 1024 * 1024  # 100 MB


def main():
    assert SRC.exists(), f"missing parquet: {SRC}"
    print(f"Source: {SRC}  ({SRC.stat().st_size / 1e9:.2f} GB)")

    pf = pq.ParquetFile(SRC)
    print(f"Row groups: {pf.num_row_groups}  rows total: {pf.metadata.num_rows:,}")
    print(f"Columns: {pf.schema_arrow.names}")

    t0 = time.time()
    written = 0
    rows = 0
    row_groups_used = 0
    with OUT_TXT.open("wb") as f:
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["text"])
            texts = tbl.column("text").to_pylist()
            for s in texts:
                if not s:
                    continue
                b = s.encode("utf-8", errors="ignore")
                f.write(b)
                f.write(b"\n")
                written += len(b) + 1
                rows += 1
                if written >= TARGET_BYTES:
                    break
            row_groups_used += 1
            print(f"  rg {rg:3d}  -> {written / 1e6:7.1f} MB  rows={rows:,}")
            if written >= TARGET_BYTES:
                break

    dt = time.time() - t0
    print(f"\nExtracted {written / 1e6:.1f} MB in {dt:.1f}s  "
          f"({rows:,} docs, {row_groups_used} row groups)")
    print(f"Wrote: {OUT_TXT}")

    meta = {
        "source_parquet": str(SRC.relative_to(REPO_ROOT)),
        "source_bytes": SRC.stat().st_size,
        "extracted_bytes": written,
        "docs": rows,
        "row_groups_used": row_groups_used,
        "row_groups_total": pf.num_row_groups,
        "rows_total": int(pf.metadata.num_rows),
        "extract_seconds": round(dt, 2),
        "target_bytes": TARGET_BYTES,
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"Wrote: {OUT_META}")


if __name__ == "__main__":
    main()
