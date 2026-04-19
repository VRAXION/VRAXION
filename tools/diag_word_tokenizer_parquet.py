"""Word tokenizer trained on FineWeb-EDU parquet samples.

Same exact, lossless, SentencePiece-style encoder/decoder as diag_word_tokenizer.py,
but builds the vocab from real natural English text, not Rust code.

Usage:
  python tools/diag_word_tokenizer_parquet.py \
      --parquet "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/000_00000.parquet" \
      --row-groups 20 \
      --vocab-size 32000 \
      --demo "The cat sleeps peacefully."
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from diag_word_tokenizer import (
    Tokenizer, build_tokenizer, MAX_WS_RUN, PUNCT_CHARS,
)

import pyarrow.parquet as pq


DEFAULT_PARQUET = r"S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\000_00000.parquet"
OUT_DIR = Path("output/word_tokenizer_parquet")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_parquet_text(path, row_groups=20, max_bytes=None):
    """Concatenate the 'text' column from the first N row groups."""
    pf = pq.ParquetFile(path)
    total_rg = pf.num_row_groups
    row_groups = min(row_groups, total_rg)
    chunks = []
    total = 0
    for i in range(row_groups):
        tbl = pf.read_row_group(i, columns=["text"])
        for text in tbl.column("text").to_pylist():
            if text is None:
                continue
            b = text.encode("utf-8")
            chunks.append(b)
            total += len(b)
            if max_bytes and total >= max_bytes:
                break
        if max_bytes and total >= max_bytes:
            break
    joined = b"\n".join(chunks)
    return joined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--row-groups", type=int, default=20,
                        help="How many parquet row groups to read (each ~1000 docs)")
    parser.add_argument("--max-bytes", type=int, default=100_000_000,
                        help="Cap on total corpus bytes (default 100 MB)")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--demo", type=str, default="The cat sleeps peacefully on the warm mat.")
    args = parser.parse_args()

    print("=" * 70)
    print("WORD TOKENIZER on FineWeb-EDU parquet")
    print("=" * 70)

    print(f"\n[1] Loading parquet...")
    print(f"    file: {args.parquet}")
    print(f"    row_groups: {args.row_groups}  max_bytes: {args.max_bytes}")
    t0 = time.time()
    corpus = load_parquet_text(args.parquet, row_groups=args.row_groups, max_bytes=args.max_bytes)
    print(f"    loaded {len(corpus):,} bytes  ({time.time()-t0:.1f}s)")
    print(f"    first 300 chars: {corpus[:300].decode('utf-8', errors='replace')!r}")

    print(f"\n[2] Building vocab (target {args.vocab_size} word tokens)...")
    t0 = time.time()
    tk, counter = build_tokenizer(corpus, vocab_size=args.vocab_size)
    print(f"    unique (word, prefix) combos: {len(counter):,}")
    print(f"    vocab size total: {tk.vocab_size:,} "
          f"(256 BYTE + {len(tk.punct_list)} PUNCT + {MAX_WS_RUN-1} WS_RUN + "
          f"{tk.vocab_size - tk.WORD_BASE} WORD)")
    print(f"    ({time.time()-t0:.1f}s)")

    print(f"\n[3] Top 30 WORD tokens:")
    sorted_items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
    for i, ((word, has_prefix), c) in enumerate(sorted_items):
        tid = tk.WORD_BASE + i
        shown = word.decode("utf-8", errors="replace")
        if not shown.isprintable():
            shown = "<bin:" + word.hex() + ">"
        marker = "▁" if has_prefix else " "
        print(f"    {tid:>5} {c:>10,}  {marker}{shown}")

    print(f"\n[4] Round-trip sanity on 1 KB corpus sample...")
    sample = corpus[1000:2000]
    ids = tk.encode(sample)
    back = tk.decode(ids)
    match = back == sample
    print(f"    {len(sample)} bytes -> {len(ids)} ids -> {len(back)} bytes  match={match}")

    print(f"\n[5] Demo: {args.demo!r}")
    demo_bytes = args.demo.encode("utf-8")
    demo_ids = tk.encode(demo_bytes)
    demo_back = tk.decode(demo_ids)
    print(f"    ids: {demo_ids}")
    print(f"    format: _" + "_".join(str(i) for i in demo_ids) + "_")
    print(f"    decoded: {demo_back!r}")
    print(f"    exact: {demo_back == demo_bytes}")
    print(f"    token breakdown:")
    for tid in demo_ids:
        kind, payload = tk.id_to_token[tid]
        if kind == "WORD":
            word, has_prefix = payload
            marker = "▁" if has_prefix else ""
            print(f"      {tid:>5}  WORD     {marker}{word.decode('utf-8','replace')}")
        elif kind == "BYTE":
            ch = chr(payload) if 32 <= payload < 127 else "."
            print(f"      {tid:>5}  BYTE     {payload:#04x} ({ch})")
        elif kind == "PUNCT":
            print(f"      {tid:>5}  PUNCT    {chr(payload)}")
        elif kind == "WS_RUN":
            print(f"      {tid:>5}  WS_RUN   {payload} spaces")

    print(f"\n[6] Corpus tokenization + coverage:")
    t0 = time.time()
    all_ids = tk.encode(corpus)
    dt = time.time() - t0
    # Count by kind
    kind_count = {}
    for tid in all_ids:
        kind = tk.id_to_token[tid][0]
        kind_count[kind] = kind_count.get(kind, 0) + 1
    import math
    bits_per_token = math.ceil(math.log2(tk.vocab_size))
    fixed_bytes = len(all_ids) * bits_per_token / 8
    print(f"    corpus bytes: {len(corpus):,}")
    print(f"    token count: {len(all_ids):,}  ({dt:.1f}s)")
    for kind, n in sorted(kind_count.items(), key=lambda x: -x[1]):
        pct = 100.0 * n / len(all_ids)
        print(f"      {kind:<8}: {n:>12,}  ({pct:5.2f}%)")
    print(f"    bits per token (fixed-width): {bits_per_token}")
    print(f"    tokens total size: {fixed_bytes:,.0f} bytes  ({100*fixed_bytes/len(corpus):.2f}% of raw)")
    print(f"    avg tokens per byte: {len(all_ids)/len(corpus):.4f}")

    vocab_path = OUT_DIR / f"vocab_{args.vocab_size}.json"
    vocab_path.write_text(json.dumps(tk.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Saved vocab: {vocab_path}  ({vocab_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
