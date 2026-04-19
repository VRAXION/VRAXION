"""Adversarial + sanity tests for the V1 word tokenizer (FineWeb-EDU trained).

Runs a battery of tests answering: does the research literature's ~32% target
actually hold for OUR tokenizer on OUR data? And is the tokenizer robust
to adversarial inputs (Unicode, whitespace runs, mixed scripts, etc.)?

Tests:
  1. Round-trip lossless on 10 MB corpus sample (not 1 KB)
  2. True byte-fallback rate measured per-input-byte (not per-token)
  3. Actual Huffman compression on token stream -> compare vs 32% estimate
  4. gzip/bzip2/lzma/zstd baselines on SAME raw bytes
  5. Unreachable-token audit (decode each vocab token, re-encode, check ID match)
  6. Edge-case battery: long WS runs, Hungarian/emoji/control bytes, empty input
  7. Greedy vs "if DP existed" token-count gap estimate

Usage:
  python tools/diag_word_tokenizer_adversarial.py
"""
from __future__ import annotations
import argparse, bz2, gzip, heapq, json, lzma, math, sys, time
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from diag_word_tokenizer import Tokenizer, MAX_WS_RUN, PUNCT_CHARS
from diag_word_tokenizer_parquet import load_parquet_text, DEFAULT_PARQUET

VOCAB_JSON = Path("output/word_tokenizer_parquet/vocab_32000.json")
OUT_DIR = Path("output/word_tokenizer_adversarial")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_tokenizer_from_json(path: Path) -> Tokenizer:
    items = json.loads(path.read_text(encoding="utf-8"))
    word_tokens = []
    for item in items:
        if item["kind"] == "WORD":
            text = item["text"].encode("utf-8")
            word_tokens.append((text, item["space_prefix"]))
    tk = Tokenizer(word_tokens=word_tokens)
    return tk


# --- Huffman implementation on int token stream ---
def huffman_bits(freq_counter: Counter) -> tuple[int, dict]:
    """Return total bits for optimal prefix code + code_length dict."""
    if len(freq_counter) <= 1:
        return (sum(freq_counter.values()), {k: 1 for k in freq_counter})
    heap = [[f, i, k] for i, (k, f) in enumerate(freq_counter.items())]
    heapq.heapify(heap)
    ctr = len(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, [a[0] + b[0], ctr, (a, b)])
        ctr += 1
    root = heap[0]
    code_len = {}

    def walk(node, depth):
        if not isinstance(node[2], tuple):
            code_len[node[2]] = max(1, depth)
        else:
            left, right = node[2]
            walk(left, depth + 1)
            walk(right, depth + 1)

    walk(root, 0)
    total_bits = sum(code_len[k] * f for k, f in freq_counter.items())
    return total_bits, code_len


def shannon_bits(freq_counter: Counter) -> float:
    total = sum(freq_counter.values())
    if total == 0:
        return 0.0
    s = 0.0
    for f in freq_counter.values():
        if f > 0:
            p = f / total
            s += -p * math.log2(p)
    return s * total


def test_1_roundtrip(tk, corpus, sample_size=10_000_000):
    print("\n[1] ROUND-TRIP LOSSLESS on 10 MB sample")
    sample = corpus[:sample_size]
    t0 = time.time()
    ids = tk.encode(sample)
    enc_s = time.time() - t0
    t0 = time.time()
    back = tk.decode(ids)
    dec_s = time.time() - t0
    match = back == sample
    print(f"    {len(sample):,} bytes -> {len(ids):,} tokens -> {len(back):,} bytes")
    print(f"    encode: {enc_s:.2f}s ({len(sample)/enc_s/1e6:.1f} MB/s)  decode: {dec_s:.2f}s")
    print(f"    MATCH: {match}")
    if not match:
        for i, (a, b) in enumerate(zip(sample, back)):
            if a != b:
                print(f"    first mismatch at byte {i}: expected {a} got {b}")
                break
    return ids, match


def test_2_byte_fallback_rate(tk, ids, corpus_len):
    print("\n[2] TRUE BYTE-FALLBACK RATE (per-input-byte, not per-token)")
    kind_count = Counter()
    kind_input_bytes = Counter()
    for tid in ids:
        kind, payload = tk.id_to_token[tid]
        kind_count[kind] += 1
        if kind == "BYTE":
            kind_input_bytes[kind] += 1
        elif kind == "PUNCT":
            kind_input_bytes[kind] += 1
        elif kind == "WS_RUN":
            kind_input_bytes[kind] += payload
        elif kind == "WORD":
            word_bytes, has_prefix = payload
            kind_input_bytes[kind] += len(word_bytes) + (1 if has_prefix else 0)
    total_tokens = sum(kind_count.values())
    total_input = sum(kind_input_bytes.values())
    print(f"    total tokens: {total_tokens:,}  total input bytes covered: {total_input:,} (should equal corpus {corpus_len:,})")
    for kind in ["WORD", "BYTE", "PUNCT", "WS_RUN"]:
        n_tok = kind_count.get(kind, 0)
        n_byt = kind_input_bytes.get(kind, 0)
        tok_pct = 100 * n_tok / total_tokens if total_tokens else 0
        byt_pct = 100 * n_byt / total_input if total_input else 0
        print(f"    {kind:<8}: {n_tok:>10,} tokens ({tok_pct:5.2f}%)   {n_byt:>12,} input-bytes ({byt_pct:5.2f}%)")
    return kind_count, kind_input_bytes


def test_3_huffman(ids, vocab_size, raw_len):
    print("\n[3] ACTUAL HUFFMAN/rANS COMPRESSION on token stream")
    freq = Counter(ids)
    t0 = time.time()
    huff_bits, code_len = huffman_bits(freq)
    huff_s = time.time() - t0
    shan_bits = shannon_bits(freq)
    fixed_bits = len(ids) * math.ceil(math.log2(vocab_size))
    huff_bytes = math.ceil(huff_bits / 8)
    shan_bytes = shan_bits / 8
    fixed_bytes = fixed_bits // 8
    print(f"    unique tokens used: {len(freq):,} / {vocab_size:,}")
    print(f"    fixed-width ({math.ceil(math.log2(vocab_size))} bpt): {fixed_bytes:>10,} bytes ({100*fixed_bytes/raw_len:5.2f}%)")
    print(f"    Shannon (entropy floor):                         {shan_bytes:>10,.0f} bytes ({100*shan_bytes/raw_len:5.2f}%)")
    print(f"    Huffman (true prefix code):                      {huff_bytes:>10,} bytes ({100*huff_bytes/raw_len:5.2f}%)")
    print(f"    Huffman-vs-Shannon overhead: {100*(huff_bytes*8-shan_bits)/shan_bits:.3f}%")
    print(f"    ({huff_s:.1f}s for Huffman tree build)")
    return {
        "fixed_bytes": fixed_bytes, "shannon_bytes": shan_bytes, "huffman_bytes": huff_bytes,
        "fixed_pct": 100*fixed_bytes/raw_len, "shannon_pct": 100*shan_bytes/raw_len,
        "huffman_pct": 100*huff_bytes/raw_len,
    }


def test_4_baselines(corpus, sample_size=10_000_000):
    print("\n[4] STANDARD COMPRESSOR BASELINES (raw bytes, no tokenizer)")
    sample = corpus[:sample_size]
    results = {}
    for name, fn in [
        ("gzip-9", lambda d: gzip.compress(d, compresslevel=9)),
        ("bzip2-9", lambda d: bz2.compress(d, compresslevel=9)),
        ("lzma-preset-6", lambda d: lzma.compress(d, preset=6)),
        ("lzma-preset-9e", lambda d: lzma.compress(d, preset=9 | lzma.PRESET_EXTREME)),
    ]:
        t0 = time.time()
        out = fn(sample)
        dt = time.time() - t0
        pct = 100 * len(out) / len(sample)
        print(f"    {name:<18}: {len(out):>10,} bytes ({pct:5.2f}%)   [{dt:5.1f}s]")
        results[name] = {"bytes": len(out), "pct": pct}
    return results


def test_5_unreachable(tk, sample_count=2000):
    print("\n[5] UNREACHABLE TOKEN AUDIT (decode(encode(X)) == X for each vocab token)")
    reachable = 0
    unreachable_examples = []
    # Only check WORD tokens (BYTE/PUNCT/WS_RUN by construction are reachable)
    word_ids = [i for i in range(tk.vocab_size) if tk.id_to_token[i][0] == "WORD"]
    import random
    random.seed(42)
    sample = random.sample(word_ids, min(sample_count, len(word_ids)))
    for tid in sample:
        decoded = tk.decode([tid])
        re_ids = tk.encode(decoded)
        if len(re_ids) == 1 and re_ids[0] == tid:
            reachable += 1
        else:
            if len(unreachable_examples) < 10:
                kind, payload = tk.id_to_token[tid]
                word_bytes, has_prefix = payload
                unreachable_examples.append({
                    "id": tid,
                    "shown": ("▁" if has_prefix else "") + word_bytes.decode("utf-8", errors="replace"),
                    "decoded_bytes": decoded.hex(),
                    "re_encoded_ids": re_ids[:10],
                })
    pct = 100 * reachable / len(sample)
    print(f"    sampled {len(sample)} WORD tokens")
    print(f"    reachable (decode->encode round-trips to same ID): {reachable}/{len(sample)} ({pct:.2f}%)")
    if unreachable_examples:
        print(f"    first {len(unreachable_examples)} unreachable examples:")
        for ex in unreachable_examples[:10]:
            print(f"      id={ex['id']} shown={ex['shown']!r} re_ids={ex['re_encoded_ids']}")
    return {"sampled": len(sample), "reachable": reachable, "unreachable_examples": unreachable_examples}


def test_6_edge_cases(tk):
    print("\n[6] EDGE-CASE BATTERY")
    cases = [
        ("empty input", b""),
        ("single space", b" "),
        ("5-space run", b"     "),
        ("8-space run (MAX_WS_RUN)", b" " * 8),
        ("9-space run (overflow)", b" " * 9),
        ("long whitespace run (100)", b" " * 100),
        ("tab + newline mix", b"\t\n\t\n\t\n"),
        ("Hungarian diacritics", "Péter szépen éneklő bárány".encode("utf-8")),
        ("emoji (4-byte UTF-8)", "Hello 🐈 world".encode("utf-8")),
        ("mixed scripts", "English 中文 العربية".encode("utf-8")),
        ("all control bytes 0-31", bytes(range(32))),
        ("null bytes", b"\x00\x00\x00"),
        ("Rust snippet", b"fn main() { println!(\"hi\"); }"),
        ("common English phrase", b"The quick brown fox jumps over the lazy dog."),
        ("100x repeated word", b"the " * 100),
    ]
    all_ok = True
    for name, data in cases:
        try:
            ids = tk.encode(data)
            back = tk.decode(ids)
            ok = back == data
            tok_per_byte = len(ids) / max(1, len(data))
            status = "OK" if ok else "FAIL"
            if not ok:
                all_ok = False
            print(f"    [{status}] {name:<35} {len(data):>4}B -> {len(ids):>4}tok ({tok_per_byte:.3f} tpb)")
            if not ok:
                print(f"           in:  {data!r}")
                print(f"           out: {back!r}")
        except Exception as e:
            all_ok = False
            print(f"    [CRASH] {name:<35} {e}")
    print(f"    ALL EDGE CASES {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--sample-bytes", type=int, default=10_000_000)
    parser.add_argument("--max-input-bytes", type=int, default=12_000_000)
    args = parser.parse_args()

    print("=" * 70)
    print("V1 WORD TOKENIZER — ADVERSARIAL + SANITY BATTERY")
    print("=" * 70)

    print(f"\n[0] Loading saved vocab: {VOCAB_JSON}")
    t0 = time.time()
    tk = load_tokenizer_from_json(VOCAB_JSON)
    print(f"    vocab size: {tk.vocab_size:,}  ({time.time()-t0:.1f}s)")

    print(f"\n[0b] Loading FineWeb-EDU sample ({args.max_input_bytes:,} bytes)")
    t0 = time.time()
    corpus = load_parquet_text(args.parquet, row_groups=5, max_bytes=args.max_input_bytes)
    print(f"    got {len(corpus):,} bytes ({time.time()-t0:.1f}s)")

    ids, rt_ok = test_1_roundtrip(tk, corpus, sample_size=args.sample_bytes)
    kind_count, kind_input_bytes = test_2_byte_fallback_rate(tk, ids, min(len(corpus), args.sample_bytes))
    comp_result = test_3_huffman(ids, tk.vocab_size, min(len(corpus), args.sample_bytes))
    baseline_result = test_4_baselines(corpus, sample_size=args.sample_bytes)
    unreach_result = test_5_unreachable(tk, sample_count=2000)
    edge_ok = test_6_edge_cases(tk)

    print("\n" + "=" * 70)
    print("VERDICT SUMMARY")
    print("=" * 70)
    raw = min(len(corpus), args.sample_bytes)
    print(f"Raw sample:                        {raw:>10,} bytes (100.00%)")
    print(f"Our fixed-width token stream:      {comp_result['fixed_bytes']:>10,} bytes ({comp_result['fixed_pct']:5.2f}%)")
    print(f"Our Huffman-packed token stream:   {comp_result['huffman_bytes']:>10,} bytes ({comp_result['huffman_pct']:5.2f}%)")
    print(f"Our Shannon floor (token stream):  {comp_result['shannon_bytes']:>10,.0f} bytes ({comp_result['shannon_pct']:5.2f}%)")
    for k, v in baseline_result.items():
        print(f"Baseline {k:<18}       {v['bytes']:>10,} bytes ({v['pct']:5.2f}%)")
    byte_fb_pct = 100 * kind_input_bytes.get("BYTE", 0) / raw
    print(f"\nByte fallback rate (input bytes):  {byte_fb_pct:.2f}%")
    print(f"Round-trip lossless on 10 MB:      {'YES' if rt_ok else 'NO'}")
    print(f"Unreachable tokens (of 2000):      {2000 - unreach_result['reachable']} ({100*(2000-unreach_result['reachable'])/2000:.2f}%)")
    print(f"Edge cases all pass:               {'YES' if edge_ok else 'NO'}")

    summary = {
        "raw_bytes": raw,
        "fixed_width": comp_result,
        "baselines": baseline_result,
        "byte_fallback_rate": byte_fb_pct,
        "roundtrip_ok": rt_ok,
        "unreachable": unreach_result,
        "edge_cases_ok": edge_ok,
    }
    out_path = OUT_DIR / "adversarial_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
