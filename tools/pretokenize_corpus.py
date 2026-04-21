"""Pre-tokenise a raw byte corpus to an int32 .npy for remote training.

Remote runners (Modal) don't have the champion LexicalTokenizer installed,
and re-tokenising on every Modal spawn is wasted time (~50s for 10 MB).
Pre-tokenise once locally, upload the .npy, and train from it.

Run:
    python3 tools/pretokenize_corpus.py \
        --corpus output/data/fineweb_edu_100mb.txt \
        --max-bytes 100000000 \
        --out output/data/fineweb_edu_100mb.tokens.npy
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--max-bytes", type=int, default=0,
                    help="0 = read whole file")
    ap.add_argument("--vocab", type=Path,
                    default=REPO_ROOT / "output" / "word_tokenizer_champion" / "champion_vocab.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    vocab = json.loads(args.vocab.read_text())
    learned = []
    for e in vocab:
        if e.get("kind") != "LEARNED":
            continue
        b = bytes.fromhex(e["bytes_hex"])
        if e.get("space_prefix"):
            b = b" " + b
        learned.append((b, bool(e.get("space_prefix"))))
    tok = LexicalTokenizer(learned_tokens=learned)
    vocab_size = tok.vocab_size

    with args.corpus.open("rb") as f:
        raw = f.read(args.max_bytes) if args.max_bytes > 0 else f.read()
    print(f"Corpus bytes: {len(raw):,}  vocab_size={vocab_size}  tokenising...",
          flush=True)
    t0 = time.time()
    ids = np.asarray(tok.encode(raw), dtype=np.int32)
    dt = time.time() - t0
    print(f"Tokenised to {len(ids):,} tokens in {dt:.1f}s  "
          f"(compression = {len(raw)/len(ids):.2f} bytes/token)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, ids)
    print(f"Saved: {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
