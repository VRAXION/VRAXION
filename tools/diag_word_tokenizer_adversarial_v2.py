"""V2 hybrid tokenizer adversarial + sanity battery.

Mirrors diag_word_tokenizer_adversarial.py but for the V2 LexicalTokenizer
from diag_subword_tokenizer_exact.py (hybrid whole-word + subword + byte).

Runs at whole_ratio 0.875 and 0.9375 so we can compare directly against V1
whole-word (32.86% real Huffman on 10 MB FineWeb-EDU).

Caches the built hybrid vocab to disk so reruns are fast.

Usage:
  python tools/diag_word_tokenizer_adversarial_v2.py
  python tools/diag_word_tokenizer_adversarial_v2.py --whole-ratio 0.875
"""
from __future__ import annotations
import argparse, bz2, gzip, json, lzma, math, pickle, sys, time
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from diag_subword_tokenizer_exact import (
    LexicalTokenizer, scan_words, build_hybrid_tokens, load_parquet_text, PUNCT_CHARS
)
from diag_word_tokenizer_adversarial import (
    huffman_bits, shannon_bits, test_4_baselines,
)

DEFAULT_PARQUET = r"S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\000_00000.parquet"
OUT_DIR = Path("output/word_tokenizer_adversarial")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("output/subword_tokenizer_exact")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def build_or_load_hybrid(corpus: bytes, vocab_size: int, whole_ratio: float) -> LexicalTokenizer:
    cache_path = CACHE_DIR / f"hybrid_tokens_v{vocab_size}_wr{whole_ratio:.4f}.pkl"
    if cache_path.exists():
        print(f"    [cache hit] {cache_path}")
        with open(cache_path, "rb") as f:
            learned = pickle.load(f)
    else:
        print(f"    [cache miss] building fresh...")
        t0 = time.time()
        counter = scan_words(corpus)
        print(f"    scan_words: {len(counter):,} unique (word, prefix) combos ({time.time()-t0:.1f}s)")
        t0 = time.time()
        learned = build_hybrid_tokens(counter, vocab_size=vocab_size, whole_ratio=whole_ratio)
        print(f"    build_hybrid_tokens: {len(learned):,} tokens ({time.time()-t0:.1f}s)")
        with open(cache_path, "wb") as f:
            pickle.dump(learned, f)
    return LexicalTokenizer(learned_tokens=learned)


def test_1_roundtrip(tk, corpus, sample_size):
    print("\n[1] ROUND-TRIP LOSSLESS")
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
                print(f"    first mismatch at byte {i}")
                break
    return ids, match


def test_2_byte_fallback_rate(tk, ids, corpus_len):
    print("\n[2] TRUE BYTE-FALLBACK RATE")
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
        elif kind == "LEARNED":
            sub, has_prefix = payload
            kind_input_bytes[kind] += len(sub) + (1 if has_prefix else 0)
    total_tokens = sum(kind_count.values())
    total_input = sum(kind_input_bytes.values())
    print(f"    total tokens: {total_tokens:,}  input bytes covered: {total_input:,} / corpus {corpus_len:,}")
    for kind in ["LEARNED", "BYTE", "PUNCT", "WS_RUN"]:
        n_tok = kind_count.get(kind, 0)
        n_byt = kind_input_bytes.get(kind, 0)
        tok_pct = 100 * n_tok / total_tokens if total_tokens else 0
        byt_pct = 100 * n_byt / total_input if total_input else 0
        print(f"    {kind:<8}: {n_tok:>10,} tokens ({tok_pct:5.2f}%)   {n_byt:>12,} input-bytes ({byt_pct:5.2f}%)")
    return kind_count, kind_input_bytes


def test_3_huffman(ids, vocab_size, raw_len):
    print("\n[3] REAL HUFFMAN COMPRESSION")
    freq = Counter(ids)
    t0 = time.time()
    huff_bits, _ = huffman_bits(freq)
    huff_s = time.time() - t0
    shan_bits = shannon_bits(freq)
    fixed_bits = len(ids) * math.ceil(math.log2(vocab_size))
    huff_bytes = math.ceil(huff_bits / 8)
    shan_bytes = shan_bits / 8
    fixed_bytes = fixed_bits // 8
    print(f"    unique tokens used: {len(freq):,} / {vocab_size:,}")
    print(f"    fixed-width ({math.ceil(math.log2(vocab_size))} bpt): {fixed_bytes:>10,} bytes ({100*fixed_bytes/raw_len:5.2f}%)")
    print(f"    Shannon floor:                                   {shan_bytes:>10,.0f} bytes ({100*shan_bytes/raw_len:5.2f}%)")
    print(f"    Huffman:                                         {huff_bytes:>10,} bytes ({100*huff_bytes/raw_len:5.2f}%)")
    print(f"    Huffman-vs-Shannon overhead: {100*(huff_bytes*8-shan_bits)/shan_bits:.3f}% ({huff_s:.1f}s)")
    return {
        "fixed_bytes": fixed_bytes, "shannon_bytes": shan_bytes, "huffman_bytes": huff_bytes,
        "fixed_pct": 100*fixed_bytes/raw_len, "shannon_pct": 100*shan_bytes/raw_len,
        "huffman_pct": 100*huff_bytes/raw_len,
    }


def test_5_unreachable(tk, sample_count=2000):
    print("\n[5] UNREACHABLE TOKEN AUDIT")
    learned_ids = [i for i in range(tk.vocab_size) if tk.id_to_token[i][0] == "LEARNED"]
    import random
    random.seed(42)
    sample = random.sample(learned_ids, min(sample_count, len(learned_ids)))
    reachable = 0
    examples = []
    for tid in sample:
        decoded = tk.decode([tid])
        re_ids = tk.encode(decoded)
        if len(re_ids) == 1 and re_ids[0] == tid:
            reachable += 1
        elif len(examples) < 10:
            kind, payload = tk.id_to_token[tid]
            sub, has_prefix = payload
            examples.append({
                "id": tid,
                "shown": ("▁" if has_prefix else "") + sub.decode("utf-8", errors="replace"),
                "re_ids": re_ids[:10],
            })
    pct = 100 * reachable / len(sample)
    print(f"    sampled {len(sample)} LEARNED tokens -> reachable: {reachable} ({pct:.2f}%)")
    if examples:
        print(f"    first unreachable examples:")
        for ex in examples[:10]:
            print(f"      id={ex['id']} shown={ex['shown']!r} re={ex['re_ids']}")
    return {"sampled": len(sample), "reachable": reachable, "examples": examples}


def test_6_edge_cases(tk):
    print("\n[6] EDGE-CASE BATTERY")
    cases = [
        ("empty", b""),
        ("single space", b" "),
        ("5-space run", b"     "),
        ("9-space overflow", b" " * 9),
        ("100 spaces", b" " * 100),
        ("tab+nl mix", b"\t\n\t\n\t\n"),
        ("Hungarian", "Péter szépen éneklő bárány".encode("utf-8")),
        ("emoji", "Hello 🐈 world".encode("utf-8")),
        ("mixed scripts", "English 中文 العربية".encode("utf-8")),
        ("control bytes", bytes(range(32))),
        ("null bytes", b"\x00\x00\x00"),
        ("Rust", b"fn main() { println!(\"hi\"); }"),
        ("common phrase", b"The quick brown fox jumps over the lazy dog."),
        ("100x the", b"the " * 100),
    ]
    all_ok = True
    for name, data in cases:
        try:
            ids = tk.encode(data)
            back = tk.decode(ids)
            ok = back == data
            if not ok:
                all_ok = False
            tpb = len(ids) / max(1, len(data))
            print(f"    [{'OK' if ok else 'FAIL'}] {name:<20} {len(data):>4}B -> {len(ids):>4}tok ({tpb:.3f} tpb)")
        except Exception as e:
            all_ok = False
            print(f"    [CRASH] {name:<20} {e}")
    print(f"    EDGE CASES {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def run_battery(tk, corpus, sample_size, name):
    print(f"\n{'='*70}")
    print(f"BATTERY: {name}  vocab={tk.vocab_size:,}")
    print(f"{'='*70}")
    ids, rt_ok = test_1_roundtrip(tk, corpus, sample_size)
    kind_count, kind_input_bytes = test_2_byte_fallback_rate(tk, ids, min(len(corpus), sample_size))
    comp = test_3_huffman(ids, tk.vocab_size, min(len(corpus), sample_size))
    unreach = test_5_unreachable(tk, sample_count=2000)
    edge_ok = test_6_edge_cases(tk)
    byte_fb_pct = 100 * kind_input_bytes.get("BYTE", 0) / min(len(corpus), sample_size)
    learned_pct = 100 * kind_input_bytes.get("LEARNED", 0) / min(len(corpus), sample_size)
    return {
        "name": name,
        "vocab_size": tk.vocab_size,
        "roundtrip_ok": rt_ok,
        "compression": comp,
        "byte_fallback_pct": byte_fb_pct,
        "learned_coverage_pct": learned_pct,
        "unreachable": 2000 - unreach["reachable"],
        "edge_ok": edge_ok,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--whole-ratios", type=str, default="0.875,0.9375")
    parser.add_argument("--sample-bytes", type=int, default=10_000_000)
    parser.add_argument("--build-bytes", type=int, default=100_000_000)
    args = parser.parse_args()
    ratios = [float(x) for x in args.whole_ratios.split(",")]

    print("=" * 70)
    print(f"V2 HYBRID TOKENIZER — ADVERSARIAL + SANITY BATTERY")
    print(f"  whole_ratios: {ratios}  vocab: {args.vocab_size}")
    print("=" * 70)

    print(f"\n[0] Loading FineWeb-EDU (build {args.build_bytes:,}B, test {args.sample_bytes:,}B)")
    t0 = time.time()
    corpus = load_parquet_text(args.parquet, row_groups=20, max_bytes=args.build_bytes)
    print(f"    got {len(corpus):,} bytes ({time.time()-t0:.1f}s)")

    results = []
    for wr in ratios:
        print(f"\n[build] whole_ratio={wr}")
        tk = build_or_load_hybrid(corpus, args.vocab_size, wr)
        r = run_battery(tk, corpus, args.sample_bytes, f"V2-hybrid wr={wr}")
        results.append(r)

    # Baselines (once)
    print(f"\n{'='*70}\nSTANDARD COMPRESSOR BASELINES (shared, 10 MB slice)\n{'='*70}")
    baselines = test_4_baselines(corpus, sample_size=args.sample_bytes)

    print(f"\n{'='*70}\nFINAL VERDICT\n{'='*70}")
    print(f"{'tokenizer':<30} {'Huffman%':>10} {'fixed%':>10} {'byte-fb%':>10} {'learned%':>10} {'unreach':>10}")
    print(f"{'V1 whole-word (prev run)':<30} {'32.86':>10} {'52.49':>10} {'9.72':>10} {'87.43':>10} {'0':>10}")
    for r in results:
        c = r["compression"]
        print(f"{r['name']:<30} {c['huffman_pct']:>10.2f} {c['fixed_pct']:>10.2f} "
              f"{r['byte_fallback_pct']:>10.2f} {r['learned_coverage_pct']:>10.2f} {r['unreachable']:>10}")
    for k, v in baselines.items():
        print(f"{'baseline '+k:<30} {v['pct']:>10.2f}")

    out = OUT_DIR / f"adversarial_v2_summary.json"
    out.write_text(json.dumps({"v2_results": results, "baselines": baselines}, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
