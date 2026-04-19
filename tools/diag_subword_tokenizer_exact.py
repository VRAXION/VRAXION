"""Exact, space-aware subword tokenizer with byte fallback.

Goal:
  - preserve exact round-trip: decode(encode(x)) == x
  - reduce BYTE fallback vs whole-word vocab by using learned subword pieces
  - keep explicit space-awareness via prefix tokens ("▁foo" style)

This script can build and compare:
  - whole-word tokenizer
  - subword tokenizer
on the same FineWeb-EDU parquet slice.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


DEFAULT_PARQUET = r"S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\000_00000.parquet"
OUT_DIR = Path("output/subword_tokenizer_exact")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PUNCT_CHARS = b".,;:!?()[]{}\"'/\\<>=+-*&|^~%$#@`"
WORD_CHARS = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
WS_CHARS = set(b" \t\n\r")
MAX_WS_RUN = 8


def is_word(b: int) -> bool:
    return b in WORD_CHARS


def is_ws(b: int) -> bool:
    return b in WS_CHARS


def is_punct(b: int) -> bool:
    return b in PUNCT_CHARS


def load_parquet_text(path: str, row_groups: int = 20, max_bytes: int | None = 100_000_000) -> bytes:
    pf = pq.ParquetFile(path)
    row_groups = min(row_groups, pf.num_row_groups)
    chunks: list[bytes] = []
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
    return b"\n".join(chunks)


def scan_words(corpus_bytes: bytes) -> Counter[tuple[bytes, bool]]:
    """Count exact words, tagged by whether they are prefixed by a single space."""
    counter: Counter[tuple[bytes, bool]] = Counter()
    n = len(corpus_bytes)
    i = 0
    while i < n:
        b = corpus_bytes[i]
        if b == 0x20 and i + 1 < n and is_word(corpus_bytes[i + 1]):
            j = i + 1
            k = j
            while k < n and is_word(corpus_bytes[k]):
                k += 1
            counter[(bytes(corpus_bytes[j:k]), True)] += 1
            i = k
            continue
        if is_word(b):
            j = i
            while j < n and is_word(corpus_bytes[j]):
                j += 1
            counter[(bytes(corpus_bytes[i:j]), False)] += 1
            i = j
            continue
        i += 1
    return counter


def build_whole_word_tokens(counter: Counter[tuple[bytes, bool]], vocab_size: int) -> list[tuple[bytes, bool]]:
    sorted_items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in sorted_items[:vocab_size]]


def build_subword_tokens(
    counter: Counter[tuple[bytes, bool]],
    vocab_size: int,
    max_sub_len: int = 12,
    min_count: int = 50,
) -> list[tuple[bytes, bool]]:
    """Build a space-aware subword inventory from weighted substring counts.

    Prefix tokens are allowed from length 1 because they also absorb the space.
    Non-prefix subwords start at length 2 because length-1 bytes are already covered.
    """
    scores: Counter[tuple[bytes, bool]] = Counter()

    for (word, has_prefix), freq in counter.items():
        L = len(word)
        if L == 0:
            continue

        # Prefix-aware subwords starting at the word boundary.
        if has_prefix:
            upto = min(L, max_sub_len)
            for ln in range(1, upto + 1):
                tok = word[:ln]
                # Prefix token saves the explicit space byte too.
                gain = ln
                scores[(tok, True)] += freq * gain

        # Generic subwords anywhere in the word.
        for start in range(L):
            upto = min(L - start, max_sub_len)
            for ln in range(2, upto + 1):
                tok = word[start:start + ln]
                gain = ln - 1
                scores[(tok, False)] += freq * gain

    # Filter tiny/noisy pieces.
    filtered: list[tuple[tuple[bytes, bool], int]] = []
    for tok, score in scores.items():
        raw_count = score // max(1, (len(tok[0]) - 1 + (1 if tok[1] else 0)))
        if raw_count >= min_count:
            filtered.append((tok, score))

    filtered.sort(key=lambda kv: (-kv[1], kv[0][1], kv[0][0]))
    chosen: list[tuple[bytes, bool]] = []
    seen = set()
    for tok, _ in filtered:
        if tok in seen:
            continue
        chosen.append(tok)
        seen.add(tok)
        if len(chosen) >= vocab_size:
            break
    return chosen


def build_hybrid_tokens(
    counter: Counter[tuple[bytes, bool]],
    vocab_size: int,
    whole_ratio: float = 0.75,
    max_sub_len: int = 12,
    min_count: int = 50,
) -> list[tuple[bytes, bool]]:
    n_whole = max(1, min(vocab_size, int(round(vocab_size * whole_ratio))))
    whole = build_whole_word_tokens(counter, n_whole)
    sub = build_subword_tokens(counter, vocab_size=vocab_size * 2, max_sub_len=max_sub_len, min_count=min_count)
    out = list(whole)
    used = set(out)
    for tok in sub:
        if tok in used:
            continue
        out.append(tok)
        used.add(tok)
        if len(out) >= vocab_size:
            break
    if len(out) < vocab_size:
        extra_whole = build_whole_word_tokens(counter, vocab_size)
        for tok in extra_whole:
            if tok in used:
                continue
            out.append(tok)
            used.add(tok)
            if len(out) >= vocab_size:
                break
    return out[:vocab_size]


@dataclass
class LexicalTokenizer:
    learned_tokens: list[tuple[bytes, bool]]

    def __post_init__(self) -> None:
        self.BYTE_BASE = 0
        self.PUNCT_BASE = 256
        self.punct_list = list(PUNCT_CHARS)
        self.WS_BASE = self.PUNCT_BASE + len(self.punct_list)
        self.LEARNED_BASE = self.WS_BASE + (MAX_WS_RUN - 1)

        self.id_to_token: dict[int, tuple[str, object]] = {}
        for i in range(256):
            self.id_to_token[i] = ("BYTE", i)
        for i, c in enumerate(self.punct_list):
            self.id_to_token[self.PUNCT_BASE + i] = ("PUNCT", c)
        for i, n in enumerate(range(2, MAX_WS_RUN + 1)):
            self.id_to_token[self.WS_BASE + i] = ("WS_RUN", n)

        self.punct_to_id = {c: self.PUNCT_BASE + i for i, c in enumerate(self.punct_list)}

        self.prefix_map: dict[bytes, int] = {}
        self.normal_map: dict[bytes, int] = {}
        self.max_prefix_len = 0
        self.max_normal_len = 0

        for i, tok in enumerate(self.learned_tokens):
            tid = self.LEARNED_BASE + i
            sub, has_prefix = tok
            self.id_to_token[tid] = ("LEARNED", (sub, has_prefix))
            if has_prefix:
                self.prefix_map[sub] = tid
                self.max_prefix_len = max(self.max_prefix_len, len(sub))
            else:
                self.normal_map[sub] = tid
                self.max_normal_len = max(self.max_normal_len, len(sub))

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def _longest_match(self, data: bytes, start: int, end: int, prefix: bool) -> tuple[int | None, int]:
        table = self.prefix_map if prefix else self.normal_map
        max_len = self.max_prefix_len if prefix else self.max_normal_len
        max_len = min(max_len, end - start)
        for ln in range(max_len, 0, -1):
            sub = data[start:start + ln]
            tid = table.get(sub)
            if tid is not None:
                return tid, ln
        return None, 0

    def _segment_word(self, word: bytes, has_prefix: bool) -> list[int]:
        """Exact DP segmentation of a single word.

        Objective:
          1. minimize emitted token count
          2. among ties, minimize BYTE fallback count
          3. among ties, prefer longer learned pieces
        """
        n = len(word)
        if n == 0:
            return [self.BYTE_BASE + 0x20] if has_prefix else []

        # best[(pos, need_space)] = (token_count, byte_count, learned_bonus, next_pos, next_need_space, tid)
        best: dict[tuple[int, bool], tuple[int, int, int, int, bool, int]] = {
            (n, False): (0, 0, 0, n, False, -1),
            (n, True): (1, 1, 0, n, False, self.BYTE_BASE + 0x20),
        }

        for pos in range(n, -1, -1):
            for need_space in (False, True):
                if pos == n:
                    continue
                candidates: list[tuple[int, int, int, int, bool, int]] = []

                if need_space:
                    # Emit raw leading space, then continue from the same word position.
                    tail = best.get((pos, False))
                    if tail is not None:
                        candidates.append(
                            (
                                1 + tail[0],
                                1 + tail[1],
                                tail[2],
                                pos,
                                False,
                                self.BYTE_BASE + 0x20,
                            )
                        )

                    # Or consume the leading space together with a prefix token.
                    max_len = min(self.max_prefix_len, n - pos)
                    for ln in range(max_len, 0, -1):
                        tid = self.prefix_map.get(word[pos:pos + ln])
                        if tid is None:
                            continue
                        tail = best.get((pos + ln, False))
                        if tail is None:
                            continue
                        candidates.append(
                            (
                                1 + tail[0],
                                tail[1],
                                tail[2] - ln,
                                pos + ln,
                                False,
                                tid,
                            )
                        )
                else:
                    max_len = min(self.max_normal_len, n - pos)
                    for ln in range(max_len, 0, -1):
                        tid = self.normal_map.get(word[pos:pos + ln])
                        if tid is None:
                            continue
                        tail = best.get((pos + ln, False))
                        if tail is None:
                            continue
                        candidates.append(
                            (
                                1 + tail[0],
                                tail[1],
                                tail[2] - ln,
                                pos + ln,
                                False,
                                tid,
                            )
                        )

                    tail = best.get((pos + 1, False))
                    if tail is not None:
                        candidates.append(
                            (
                                1 + tail[0],
                                1 + tail[1],
                                tail[2],
                                pos + 1,
                                False,
                                self.BYTE_BASE + word[pos],
                            )
                        )

                best[(pos, need_space)] = min(candidates)

        out: list[int] = []
        pos = 0
        need_space = has_prefix
        while True:
            state = best.get((pos, need_space))
            if state is None:
                break
            _, _, _, next_pos, next_need_space, tid = state
            if tid >= 0:
                out.append(tid)
            if pos == n and next_pos == n and not next_need_space:
                break
            pos, need_space = next_pos, next_need_space
            if pos == n and not need_space:
                break
        return out

    def encode(self, data: bytes | str) -> list[int]:
        if isinstance(data, str):
            data = data.encode("utf-8")
        ids: list[int] = []
        i = 0
        n = len(data)
        while i < n:
            b = data[i]

            # space-prefixed word boundary
            if b == 0x20 and i + 1 < n and is_word(data[i + 1]):
                j = i + 1
                k = j
                while k < n and is_word(data[k]):
                    k += 1
                ids.extend(self._segment_word(data[j:k], has_prefix=True))
                i = k
                continue

            # whitespace runs
            if is_ws(b):
                k = i
                while k < n and is_ws(data[k]) and (k - i) < MAX_WS_RUN:
                    k += 1
                run_len = k - i
                all_same = all(data[m] == b for m in range(i, k))
                if run_len >= 2 and all_same and b == 0x20:
                    ids.append(self.WS_BASE + (run_len - 2))
                    i = k
                    continue
                for m in range(i, k):
                    ids.append(self.BYTE_BASE + data[m])
                i = k
                continue

            # word without space prefix
            if is_word(b):
                k = i
                while k < n and is_word(data[k]):
                    k += 1
                ids.extend(self._segment_word(data[i:k], has_prefix=False))
                i = k
                continue

            if is_punct(b):
                ids.append(self.punct_to_id[b])
                i += 1
                continue

            ids.append(self.BYTE_BASE + b)
            i += 1
        return ids

    def decode(self, ids: list[int]) -> bytes:
        out = bytearray()
        for tid in ids:
            kind, payload = self.id_to_token[tid]
            if kind == "BYTE":
                out.append(int(payload))
            elif kind == "PUNCT":
                out.append(int(payload))
            elif kind == "WS_RUN":
                out.extend(b" " * int(payload))
            elif kind == "LEARNED":
                sub, has_prefix = payload
                if has_prefix:
                    out.append(0x20)
                out.extend(sub)
            else:
                raise ValueError(kind)
        return bytes(out)

    def kind_counts(self, ids: list[int]) -> Counter[str]:
        ctr: Counter[str] = Counter()
        for tid in ids:
            kind = self.id_to_token[tid][0]
            ctr[kind] += 1
        return ctr


def metrics_for_tokenizer(name: str, tk: LexicalTokenizer, corpus: bytes, demo: str) -> dict:
    sample = corpus[1000:2000]
    ids_sample = tk.encode(sample)
    exact_sample = tk.decode(ids_sample) == sample

    demo_bytes = demo.encode("utf-8")
    demo_ids = tk.encode(demo_bytes)
    demo_back = tk.decode(demo_ids)

    t0 = time.time()
    all_ids = tk.encode(corpus)
    dt = time.time() - t0
    ctr = Counter(all_ids)
    N = len(all_ids)
    H = -sum((c / N) * math.log2(c / N) for c in ctr.values())
    kinds = tk.kind_counts(all_ids)
    byte_share = kinds.get("BYTE", 0) / N
    punct_share = kinds.get("PUNCT", 0) / N
    ws_share = kinds.get("WS_RUN", 0) / N
    learned_share = kinds.get("LEARNED", 0) / N

    return {
        "name": name,
        "vocab_size": tk.vocab_size,
        "sample_exact": exact_sample,
        "demo_ids": demo_ids,
        "demo_exact": demo_back == demo_bytes,
        "token_count": N,
        "tokens_per_byte": N / len(corpus),
        "fixed_bits_per_token": math.ceil(math.log2(tk.vocab_size)),
        "fixed_total_bytes": N * math.ceil(math.log2(tk.vocab_size)) / 8,
        "entropy_bits_per_token": H,
        "entropy_total_bytes": N * H / 8,
        "share_byte": byte_share,
        "share_punct": punct_share,
        "share_wsrun": ws_share,
        "share_learned": learned_share,
        "encode_wall_s": dt,
    }


def summarize_result(r: dict, raw_bytes: int) -> None:
    print(f"\n[{r['name']}]")
    print(f"  vocab size         : {r['vocab_size']:,}")
    print(f"  sample exact       : {r['sample_exact']}")
    print(f"  demo exact         : {r['demo_exact']}")
    print(f"  token count        : {r['token_count']:,}")
    print(f"  tokens / byte      : {r['tokens_per_byte']:.4f}")
    print(f"  fixed-width        : {r['fixed_total_bytes']:,.0f} B  ({100*r['fixed_total_bytes']/raw_bytes:.2f}% of raw)")
    print(f"  entropy-coded est  : {r['entropy_total_bytes']:,.0f} B  ({100*r['entropy_total_bytes']/raw_bytes:.2f}% of raw)")
    print(f"  mix BYTE/PUNCT/WS/LEARNED : "
          f"{100*r['share_byte']:.1f}% / {100*r['share_punct']:.1f}% / "
          f"{100*r['share_wsrun']:.1f}% / {100*r['share_learned']:.1f}%")
    print(f"  encode wall        : {r['encode_wall_s']:.1f}s")


def parse_ratio_list(text: str | None) -> list[float]:
    if not text:
        return []
    out: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if not (0.0 < val <= 1.0):
            raise ValueError(f"whole_ratio out of range: {val}")
        out.append(val)
    # Preserve order, drop duplicates.
    seen = set()
    uniq: list[float] = []
    for val in out:
        if val in seen:
            continue
        uniq.append(val)
        seen.add(val)
    return uniq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--row-groups", type=int, default=20)
    parser.add_argument("--max-bytes", type=int, default=100_000_000)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-sub-len", type=int, default=12)
    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument("--whole-ratio", type=float, default=0.9375)
    parser.add_argument("--whole-ratios", type=str, default="")
    parser.add_argument("--demo", type=str, default="The cat sleeps peacefully on the warm mat.")
    args = parser.parse_args()

    print("=" * 70)
    print("EXACT SPACE-AWARE SUBWORD TOKENIZER")
    print("=" * 70)

    print(f"\n[1] Loading parquet corpus...")
    print(f"    file      : {args.parquet}")
    print(f"    row_groups: {args.row_groups}")
    print(f"    max_bytes : {args.max_bytes:,}")
    t0 = time.time()
    corpus = load_parquet_text(args.parquet, row_groups=args.row_groups, max_bytes=args.max_bytes)
    print(f"    loaded    : {len(corpus):,} bytes  ({time.time()-t0:.1f}s)")

    print(f"\n[2] Scanning words...")
    t0 = time.time()
    counter = scan_words(corpus)
    print(f"    unique (word,prefix): {len(counter):,}  ({time.time()-t0:.1f}s)")

    print(f"\n[3] Building token inventories...")
    t0 = time.time()
    word_tokens = build_whole_word_tokens(counter, vocab_size=args.vocab_size)
    sub_tokens = build_subword_tokens(
        counter,
        vocab_size=args.vocab_size,
        max_sub_len=args.max_sub_len,
        min_count=args.min_count,
    )
    whole = LexicalTokenizer(word_tokens)
    sub = LexicalTokenizer(sub_tokens)
    print(f"    whole-word learned tokens : {len(word_tokens):,}")
    print(f"    subword learned tokens    : {len(sub_tokens):,}")

    requested_hybrid_ratios = parse_ratio_list(args.whole_ratios)
    if not requested_hybrid_ratios:
        requested_hybrid_ratios = [args.whole_ratio]

    hybrid_tokenizers: list[tuple[float, LexicalTokenizer]] = []
    for ratio in requested_hybrid_ratios:
        hybrid_tokens = build_hybrid_tokens(
            counter,
            vocab_size=args.vocab_size,
            whole_ratio=ratio,
            max_sub_len=args.max_sub_len,
            min_count=args.min_count,
        )
        hybrid_tokenizers.append((ratio, LexicalTokenizer(hybrid_tokens)))
        print(f"    hybrid learned tokens     : {len(hybrid_tokens):,}  (whole_ratio={ratio:.4f})")
    print(f"    ({time.time()-t0:.1f}s)")

    print(f"\n[4] Demo string: {args.demo!r}")
    whole_demo = whole.encode(args.demo.encode('utf-8'))
    sub_demo = sub.encode(args.demo.encode('utf-8'))
    print(f"    whole-word ids: _{'_'.join(str(i) for i in whole_demo)}_")
    for ratio, hybrid in hybrid_tokenizers:
        hybrid_demo = hybrid.encode(args.demo.encode('utf-8'))
        print(f"    hybrid {ratio:.4f} ids: _{'_'.join(str(i) for i in hybrid_demo)}_")
    print(f"    subword ids   : _{'_'.join(str(i) for i in sub_demo)}_")

    print(f"\n[5] Measuring exactness + compression...")
    whole_res = metrics_for_tokenizer("whole_word", whole, corpus, args.demo)
    sub_res = metrics_for_tokenizer("subword", sub, corpus, args.demo)
    hybrid_results: list[dict] = []
    for ratio, hybrid in hybrid_tokenizers:
        res = metrics_for_tokenizer(f"hybrid_{ratio:.4f}", hybrid, corpus, args.demo)
        res["whole_ratio"] = ratio
        hybrid_results.append(res)

    summarize_result(whole_res, len(corpus))
    for res in hybrid_results:
        summarize_result(res, len(corpus))
    summarize_result(sub_res, len(corpus))

    print(f"\n[6] Delta vs whole-word")
    for label, res in [(f"hybrid_{r['whole_ratio']:.4f}", r) for r in hybrid_results] + [("subword", sub_res)]:
        print(f"  [{label}]")
        print(f"    token count delta   : {res['token_count'] - whole_res['token_count']:+,}")
        print(f"    fixed bytes delta   : {res['fixed_total_bytes'] - whole_res['fixed_total_bytes']:+,.0f} B")
        print(f"    entropy bytes delta : {res['entropy_total_bytes'] - whole_res['entropy_total_bytes']:+,.0f} B")
        print(f"    BYTE share delta    : {100*(res['share_byte'] - whole_res['share_byte']):+.2f} pp")
        print(f"    LEARNED share delta : {100*(res['share_learned'] - whole_res['share_learned']):+.2f} pp")

    if hybrid_results:
        best_fixed = min(hybrid_results, key=lambda r: (r["fixed_total_bytes"], r["entropy_total_bytes"]))
        best_entropy = min(hybrid_results, key=lambda r: (r["entropy_total_bytes"], r["fixed_total_bytes"]))
        print(f"\n[7] Best hybrid ratios")
        print(f"  fixed-width best  : whole_ratio={best_fixed['whole_ratio']:.4f}  -> {best_fixed['fixed_total_bytes']:,.0f} B")
        print(f"  entropy best      : whole_ratio={best_entropy['whole_ratio']:.4f}  -> {best_entropy['entropy_total_bytes']:,.0f} B")

    out = {
        "raw_bytes": len(corpus),
        "vocab_size_learned": args.vocab_size,
        "whole_word": whole_res,
        "hybrid": hybrid_results[0] if len(hybrid_results) == 1 else hybrid_results,
        "subword": sub_res,
    }
    summary_path = OUT_DIR / f"summary_{args.vocab_size}.json"
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
