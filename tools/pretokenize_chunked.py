"""Chunked pretokenizer for large corpora.

Tokenizing 1 GB in one shot via LexicalTokenizer + Python list builds a
massive int-object list (3+ GB RSS) and the GC/allocator slows down
super-linearly. Instead we read the corpus in fixed-byte chunks,
tokenize each independently, and concatenate.

Safety: the LexicalTokenizer is deterministic and stateless per-call, but
splitting on an arbitrary byte boundary can fracture a multi-byte UTF-8
character or a token at the seam. We therefore back off the boundary to
the last whitespace/newline within the last 4 KB of each chunk, so
tokens and UTF-8 codepoints always stay inside one chunk.

Run:
    python3 tools/pretokenize_chunked.py \\
        --corpus output/data/fineweb_edu_1gb.txt \\
        --chunk-mb 100 \\
        --out output/data/fineweb_edu_1gb.tokens.npy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
from diag_subword_tokenizer_exact import LexicalTokenizer  # noqa: E402


def _safe_boundary(data: memoryview, target_end: int) -> int:
    """Back off from target_end to the last whitespace/newline within 4 KB."""
    start = max(0, target_end - 4096)
    for i in range(target_end - 1, start - 1, -1):
        b = data[i]
        if b in (0x0A, 0x0D, 0x20, 0x09):  # \n \r space \t
            return i + 1
    return target_end  # no whitespace found, boundary may split


def build_tokenizer(vocab_json: Path) -> LexicalTokenizer:
    vocab = json.loads(vocab_json.read_text())
    learned = []
    for e in vocab:
        if e.get("kind") != "LEARNED":
            continue
        b = bytes.fromhex(e["bytes_hex"])
        if e.get("space_prefix"):
            b = b" " + b
        learned.append((b, bool(e.get("space_prefix"))))
    return LexicalTokenizer(learned_tokens=learned)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--chunk-mb", type=int, default=100,
                    help="Target chunk size in MB (default 100)")
    ap.add_argument("--max-bytes", type=int, default=0,
                    help="0 = read whole file")
    ap.add_argument("--vocab", type=Path,
                    default=REPO_ROOT / "output" / "word_tokenizer_champion"
                    / "champion_vocab.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tok = build_tokenizer(args.vocab)
    vocab_size = tok.vocab_size
    print(f"Vocab size = {vocab_size}")

    total_bytes = args.corpus.stat().st_size
    if args.max_bytes > 0:
        total_bytes = min(total_bytes, args.max_bytes)
    chunk_target = args.chunk_mb * 1024 * 1024
    print(f"Corpus: {total_bytes:,} bytes, chunk target: {chunk_target:,} "
          f"(~{total_bytes // chunk_target + 1} chunks)")

    # mmap-like read once into bytes
    with args.corpus.open("rb") as f:
        raw = f.read(total_bytes)
    view = memoryview(raw)

    chunks_ids: list[np.ndarray] = []
    chunk_idx = 0
    offset = 0
    t0 = time.time()
    while offset < total_bytes:
        end = min(offset + chunk_target, total_bytes)
        if end < total_bytes:
            end = _safe_boundary(view, end)
        chunk = bytes(view[offset:end])
        chunk_idx += 1
        t_chunk = time.time()
        ids = np.asarray(tok.encode(chunk), dtype=np.int32)
        dt = time.time() - t_chunk
        chunks_ids.append(ids)
        print(f"  chunk {chunk_idx:3d}  [{offset:>11,} - {end:>11,}]  "
              f"{len(chunk)/1e6:6.1f} MB -> {len(ids):>10,} tok  "
              f"{dt:6.1f}s", flush=True)
        offset = end

    dt_total = time.time() - t0
    print(f"\nTokenized {total_bytes/1e6:.1f} MB in {dt_total:.1f}s "
          f"({chunk_idx} chunks)")

    merged = np.concatenate(chunks_ids) if len(chunks_ids) > 1 else chunks_ids[0]
    print(f"Total tokens: {len(merged):,}  "
          f"(compression = {total_bytes/len(merged):.2f} bytes/token)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, merged)
    print(f"Saved: {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
