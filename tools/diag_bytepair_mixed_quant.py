"""Mixed-precision intelligent quantization for Block C byte-pair emb.

Hypothesis: only ~5K of 65,536 byte-pair rows got trained meaningfully;
the other ~60K are near init. Uniform quant wastes bits on noise rows
and clusters them to zero. Better: split by corpus frequency, use int4
for the hot tail and a cheap codec (binary, shared OOV) for the cold
tail.

Techniques tested:
  - tier_int4_binary : top-N at int4 α=0.5, rest binary-sign
  - tier_int4_ternary: top-N at int4, rest ternary
  - tier_int4_shared : top-N at int4, all cold rows -> ONE shared OOV vector
  - tier_int4_drop   : top-N at int4, cold rows dropped (vocab = top-N)
  - All compared to uniform int4 and uniform ternary from the previous sweep.

Reports effective deploy size (bytes) and cluster preservation on
anchors using a FREQUENCY-WEIGHTED overlap metric (only compares
neighbours among rows that ARE used in the corpus).

Run:
    python3 tools/diag_bytepair_mixed_quant.py \\
        --emb output/bytepair_ft_pull/seed1/seed_1/emb_E32_epoch10.npy \\
        --corpus output/data/fineweb_edu_1gb.txt \\
        --max-bytes 10000000 \\
        --out output/bytepair_mixed_quant/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ANCHOR_PAIRS = [
    " t", " a", " i", " o", " s",
    "th", "he", "in", "er", "an", "on",
    "ng", "ed", "ly", "es", "ti",
    ". ", ", ",
]


def pair_id(s: str) -> int:
    return (ord(s[0]) << 8) | ord(s[1])


def pair_label(pid: int) -> str:
    hi, lo = (pid >> 8) & 0xFF, pid & 0xFF
    def ch(b):
        if 32 <= b < 127:
            c = chr(b); return "\\\\" if c == "\\" else c
        return {0x20: "\\s", 0x0a: "\\n", 0x09: "\\t"}.get(b, f"\\x{b:02x}")
    return f"'{ch(hi)}{ch(lo)}'"


def byte_pair_freq(corpus_path: Path, max_bytes: int) -> np.ndarray:
    """Return freq array of shape (65536,) counting byte-pair occurrences."""
    raw = corpus_path.read_bytes()[:max_bytes]
    n = len(raw) // 2
    arr = np.frombuffer(raw[: n * 2], dtype=np.uint8).reshape(n, 2)
    ids = (arr[:, 0].astype(np.int64) << 8) | arr[:, 1].astype(np.int64)
    freq = np.bincount(ids, minlength=65536)
    return freq.astype(np.int64)


def quant_int4_sym(W: np.ndarray, alpha: float) -> np.ndarray:
    qmax = 7.0
    amax = alpha * np.max(np.abs(W), axis=0)
    safe = np.where(amax == 0, np.float32(1.0), amax)
    scale = safe / qmax
    q = np.round(W / scale).clip(-qmax, qmax)
    return (q * scale).astype(np.float32)


def quant_ternary(W: np.ndarray, alpha: float) -> np.ndarray:
    amax = alpha * np.max(np.abs(W), axis=0)
    safe = np.where(amax == 0, np.float32(1.0), amax)
    thresh = safe / 2.0
    q = np.zeros_like(W)
    q = np.where(W >  thresh, safe, q)
    q = np.where(W < -thresh, -safe, q)
    return q.astype(np.float32)


def quant_binary(W: np.ndarray) -> np.ndarray:
    amax = np.max(np.abs(W), axis=0)
    safe = np.where(amax == 0, np.float32(1.0), amax)
    return np.sign(W).astype(np.float32) * safe


def apply_mixed(emb: np.ndarray, hot_mask: np.ndarray, cold_scheme: str
                ) -> tuple[np.ndarray, int]:
    """Apply int4 α=0.5 to hot rows, the chosen scheme to cold rows.
    Returns (quantized_emb, deploy_bytes)."""
    V, E = emb.shape
    n_hot = int(hot_mask.sum())
    n_cold = V - n_hot

    # int4 on hot (per-channel scale computed on hot rows only, since cold
    # noise would pollute the amax otherwise)
    hot = emb[hot_mask]
    hot_q = quant_int4_sym(hot, alpha=0.5)

    # cold scheme
    cold = emb[~hot_mask]
    if cold_scheme == "binary":
        cold_q = quant_binary(cold)
        cold_bits = 1
    elif cold_scheme == "ternary":
        cold_q = quant_ternary(cold, alpha=0.5)
        cold_bits = 2
    elif cold_scheme == "shared":
        # One shared representative vector for all cold rows
        mean_cold = cold.mean(axis=0, keepdims=True)
        cold_q = np.tile(mean_cold, (cold.shape[0], 1)).astype(np.float32)
        cold_bits = 0  # no per-row storage, just one fp16 vector + bitmap
    elif cold_scheme == "drop":
        # Cold rows -> zero vector (effectively removed from vocab)
        cold_q = np.zeros_like(cold)
        cold_bits = 0
    else:
        raise ValueError(cold_scheme)

    out = np.empty_like(emb)
    out[hot_mask]  = hot_q
    out[~hot_mask] = cold_q

    # Deploy size:
    #   hot:  n_hot * (int4 4 bits × E) / 8 bytes/row
    #   cold: depends on scheme
    #   + bitmap of which rows are hot (V bits)
    hot_bytes = n_hot * ((4 * E) // 8)
    if cold_scheme in ("binary",):
        cold_bytes = n_cold * ((cold_bits * E) // 8)
    elif cold_scheme == "ternary":
        cold_bytes = n_cold * ((2 * E) // 8)
    elif cold_scheme == "shared":
        cold_bytes = 2 * E  # single fp16 shared vector
    elif cold_scheme == "drop":
        cold_bytes = 0
    bitmap_bytes = (V + 7) // 8
    # Per-channel scales: one fp16 per channel for hot, one for cold (if quantized)
    scale_bytes = 2 * E  # hot scales, fp16
    if cold_scheme in ("binary", "ternary"):
        scale_bytes += 2 * E
    total = hot_bytes + cold_bytes + bitmap_bytes + scale_bytes
    return out, total


def cluster_overlap(ref: np.ndarray, quant: np.ndarray,
                    anchor_ids: list[int], used_mask: np.ndarray, K: int = 5):
    """Compute mean top-K overlap, restricted to USED rows (not noise)."""
    used_idx_set = set(np.where(used_mask)[0].tolist())
    overlaps = []
    per_anchor = []
    for aid in anchor_ids:
        # Float neighbours restricted to used rows
        v = ref[aid]
        d2 = np.sum((ref - v) ** 2, axis=1)
        # Mask out unused rows so neighbours are always meaningful
        d2[~used_mask] = np.inf
        d2[aid] = np.inf
        nbrs_f = set(np.argpartition(d2, K)[:K])

        vq = quant[aid]
        d2q = np.sum((quant - vq) ** 2, axis=1)
        d2q[~used_mask] = np.inf
        d2q[aid] = np.inf
        nbrs_q = set(np.argpartition(d2q, K)[:K])
        overlap = len(nbrs_f & nbrs_q) / K
        overlaps.append(overlap)
        per_anchor.append((aid, overlap))
    return float(np.mean(overlaps)) * 100.0, per_anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--max-bytes", type=int, default=10_000_000)
    ap.add_argument("--top-n", type=int, default=0,
                    help="Hot-row count. 0 = auto-pick threshold at "
                         "freq >= 5 (appears at least 5 times in corpus)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    emb_f = np.load(args.emb).astype(np.float32)
    V, E = emb_f.shape
    print(f"Emb: shape={emb_f.shape}  float size={V*E*4/1e6:.2f} MB")

    freq = byte_pair_freq(args.corpus, args.max_bytes)
    print(f"Corpus: {args.max_bytes:,} bytes  "
          f"distinct pairs seen: {(freq > 0).sum():,}  "
          f"pairs w/ freq>=5: {(freq >= 5).sum():,}  "
          f"pairs w/ freq>=100: {(freq >= 100).sum():,}")

    # Pick hot threshold
    if args.top_n > 0:
        cutoff = np.sort(freq)[-args.top_n]
        hot_mask = freq >= cutoff
    else:
        hot_mask = freq >= 5
    n_hot = int(hot_mask.sum())
    print(f"Hot rows: {n_hot:,}  Cold rows: {V - n_hot:,}")

    anchor_ids = [pair_id(p) for p in ANCHOR_PAIRS]
    anchor_hot = [aid for aid in anchor_ids if hot_mask[aid]]
    print(f"Anchors in hot set: {len(anchor_hot)}/{len(anchor_ids)}")

    # Baseline float cluster overlap (against itself) = 100%
    results = []

    # 1. Uniform int4 (reference quant from previous sweep)
    q_u_int4 = quant_int4_sym(emb_f, alpha=0.5)
    ovl, per = cluster_overlap(emb_f, q_u_int4, anchor_hot, hot_mask, K=5)
    b_u_int4 = V * ((4 * E) // 8) + 2 * E   # per-channel scales fp16
    results.append({
        "label": "uniform_int4_a0.5",
        "cluster_overlap_pct": round(ovl, 1),
        "deploy_bytes": b_u_int4,
        "compression": round(V * E * 4 / b_u_int4, 1),
    })

    # 2. Uniform ternary
    q_u_ter = quant_ternary(emb_f, alpha=0.5)
    ovl, _ = cluster_overlap(emb_f, q_u_ter, anchor_hot, hot_mask, K=5)
    b_u_ter = V * ((2 * E) // 8) + 2 * E
    results.append({
        "label": "uniform_ternary_a0.5",
        "cluster_overlap_pct": round(ovl, 1),
        "deploy_bytes": b_u_ter,
        "compression": round(V * E * 4 / b_u_ter, 1),
    })

    # 3. Uniform binary
    q_u_bin = quant_binary(emb_f)
    ovl, _ = cluster_overlap(emb_f, q_u_bin, anchor_hot, hot_mask, K=5)
    b_u_bin = V * ((E) // 8) + 2 * E
    results.append({
        "label": "uniform_binary",
        "cluster_overlap_pct": round(ovl, 1),
        "deploy_bytes": b_u_bin,
        "compression": round(V * E * 4 / b_u_bin, 1),
    })

    # 4-7. Mixed precision variants
    for scheme in ["binary", "ternary", "shared", "drop"]:
        q_mixed, deploy_b = apply_mixed(emb_f, hot_mask, scheme)
        ovl, _ = cluster_overlap(emb_f, q_mixed, anchor_hot, hot_mask, K=5)
        results.append({
            "label": f"hot_int4_cold_{scheme}",
            "cluster_overlap_pct": round(ovl, 1),
            "deploy_bytes": deploy_b,
            "compression": round(V * E * 4 / deploy_b, 1),
        })

    # Sort + print
    print(f"\n{'='*85}")
    print(f"{'label':<32} {'clust%':>8} {'bytes':>10} {'compress':>10} {'B/hot':>8}")
    print("-" * 85)
    for r in results:
        bph = "-" if "uniform" in r["label"] else f"{r['deploy_bytes']/n_hot:.2f}"
        print(f"{r['label']:<32} {r['cluster_overlap_pct']:>7.1f}% "
              f"{r['deploy_bytes']:>10d} {r['compression']:>9.1f}x "
              f"{bph:>8}")

    (args.out / "mixed_quant_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {args.out / 'mixed_quant_results.json'}")


if __name__ == "__main__":
    main()
