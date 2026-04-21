"""Progressive intelligent quantization sweep for the Block C byte-pair champion.

Goal: find the most aggressive bit-width we can quantize the emb table to
while preserving uniqueness (100%), cluster geometry (sanity neighbours),
and mean pair distance.

Quant techniques tested (per-channel = separate scale per output column):
  - int4 symmetric (baseline, already proven for word-level champion)
  - int3 symmetric (7 levels: {-3,-2,-1,0,+1,+2,+3})
  - int2 ternary ({-1, 0, +1})
  - binary sign ({-1, +1})
  - int4 unsigned (for post-relu-style one-sided channels)
  - Alpha grid {0.3, 0.5, 0.7, 1.0} * amax for clipping threshold search

Output:
  - results table (bits, alpha, unique_pct, mean_pair, cluster_overlap)
  - sanity check on a fixed anchor set
  - per-row RMSE vs float

Run:
    python3 tools/diag_bytepair_quant_sweep.py \\
        --emb output/bytepair_ft_pull/seed1/seed_1/emb_E32_epoch10.npy \\
        --corpus output/data/fineweb_edu_1gb.txt \\
        --max-bytes 5000000 \\
        --out output/bytepair_quant_sweep/
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
    if len(s) != 2:
        raise ValueError(f"anchor must be 2 bytes, got {s!r}")
    return (ord(s[0]) << 8) | ord(s[1])


def pair_label(pid: int) -> str:
    hi, lo = (pid >> 8) & 0xFF, pid & 0xFF
    def ch(b):
        if 32 <= b < 127:
            c = chr(b); return "\\\\" if c == "\\" else c
        return {0x20: "\\s", 0x0a: "\\n", 0x09: "\\t"}.get(b, f"\\x{b:02x}")
    return f"'{ch(hi)}{ch(lo)}'"


# ---------- quantizers ----------

def quant_symmetric(W: np.ndarray, bits: int, alpha: float) -> np.ndarray:
    """Per-channel symmetric quantization, clip at alpha * amax."""
    qmax = float(2 ** (bits - 1) - 1)
    amax = alpha * np.max(np.abs(W), axis=0)
    safe = np.where(amax == 0, np.float32(1.0), amax)
    scale = safe / qmax
    q = np.round(W / scale).clip(-qmax, qmax)
    return (q * scale).astype(np.float32)


def quant_ternary(W: np.ndarray, alpha: float) -> np.ndarray:
    """Ternary {-s, 0, +s} per-channel. 3 levels = ~1.58 bits but padded to 2 bits in storage."""
    amax = alpha * np.max(np.abs(W), axis=0)
    safe = np.where(amax == 0, np.float32(1.0), amax)
    # Threshold at amax/2 — below is 0, above is +s, below is -s
    thresh = safe / 2.0
    q = np.zeros_like(W)
    q = np.where(W >  thresh, safe, q)
    q = np.where(W < -thresh, -safe, q)
    return q.astype(np.float32)


def quant_binary(W: np.ndarray) -> np.ndarray:
    """Binary sign {-s, +s} per-channel."""
    amax = np.max(np.abs(W), axis=0)
    safe = np.where(amax == 0, np.float32(1.0), amax)
    return np.sign(W).astype(np.float32) * safe


# ---------- metrics ----------

def used_mask(emb: np.ndarray, std_init: float = None) -> np.ndarray:
    """Rows that moved from init. Init was N(0, sqrt(1/E)). Unused rows have
    norm close to sqrt(E) * sqrt(1/E) ~ 1.0 approximately. A row that has
    been trained heavily diverges from that. Use z-score on norm distribution."""
    norms = np.linalg.norm(emb, axis=1)
    if std_init is None:
        std_init = norms.std()
    mean = norms.mean()
    # "Used" = norm outside 1 std of the bulk mean
    return np.abs(norms - mean) > 0.5 * std_init


def diagnostics(emb: np.ndarray, label: str, ref_emb: np.ndarray | None = None,
                anchors_ids: list[int] = None):
    """Return dict with uniqueness, min pair, cluster overlap vs ref."""
    n = emb.shape[0]

    # Uniqueness on rounded float (6 digits)
    rounded = np.round(emb, 6)
    _, inv = np.unique(rounded, axis=0, return_inverse=True)
    uniq = len(np.unique(inv)) / n * 100.0

    # Min pair distance on a 4K sample (bulk estimate)
    if n > 4096:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=4096, replace=False)
        L = emb[idx]
    else:
        L = emb
    sq = np.sum(L * L, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * (L @ L.T)
    np.fill_diagonal(d2, np.inf)
    min_pair = float(np.sqrt(max(d2.min(), 0.0)))

    # Per-row RMSE vs float reference
    rmse_to_ref = None
    if ref_emb is not None:
        rmse_to_ref = float(np.sqrt(np.mean((emb - ref_emb) ** 2)))

    # Anchor cluster overlap with reference (if ref supplied)
    cluster_overlap = None
    if ref_emb is not None and anchors_ids is not None:
        K = 5
        overlaps = []
        for aid in anchors_ids:
            # float neighbours
            v_f = ref_emb[aid]
            d2_f = np.sum((ref_emb - v_f) ** 2, axis=1)
            d2_f[aid] = np.inf
            nbrs_f = set(np.argpartition(d2_f, K)[:K])
            # quant neighbours
            v_q = emb[aid]
            d2_q = np.sum((emb - v_q) ** 2, axis=1)
            d2_q[aid] = np.inf
            nbrs_q = set(np.argpartition(d2_q, K)[:K])
            overlaps.append(len(nbrs_f & nbrs_q) / K)
        cluster_overlap = float(np.mean(overlaps)) * 100.0

    return {
        "label": label,
        "uniq_pct": round(uniq, 3),
        "min_pair": round(min_pair, 4),
        "rmse_to_ref": None if rmse_to_ref is None else round(rmse_to_ref, 5),
        "cluster_overlap_pct": (None if cluster_overlap is None
                                else round(cluster_overlap, 1)),
        "bytes_per_row": None,  # filled by caller
    }


def print_sanity_top5(emb: np.ndarray, label: str, anchor_pairs: list[str]):
    print(f"\n-- {label} --")
    for pair_s in anchor_pairs:
        pid = pair_id(pair_s)
        v = emb[pid]
        d2 = np.sum((emb - v) ** 2, axis=1)
        d2[pid] = np.inf
        nbrs = np.argpartition(d2, 5)[:5]
        nbrs = nbrs[np.argsort(d2[nbrs])]
        neigh = ", ".join(pair_label(int(n)) for n in nbrs)
        print(f"  {pair_label(pid):>8}  ->  {neigh}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True,
                    help="float32 (V, E) embedding champion to quantize")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    emb_f = np.load(args.emb).astype(np.float32)
    V, E = emb_f.shape
    bytes_per_row_float = E * 4
    print(f"Loaded champion: shape={emb_f.shape}  "
          f"float32 size = {V*E*4/1e6:.2f} MB  ({bytes_per_row_float} B/row)")

    anchors_ids = [pair_id(p) for p in ANCHOR_PAIRS]
    results = []

    # Baseline float
    res = diagnostics(emb_f, "float32", ref_emb=None, anchors_ids=None)
    res["bytes_per_row"] = bytes_per_row_float
    res["compression"] = 1.0
    results.append(res)
    print_sanity_top5(emb_f, "FLOAT (reference)", ANCHOR_PAIRS)

    # Quant sweep
    alphas = [1.0, 0.7, 0.5, 0.3]
    for bits in [4, 3]:
        for alpha in alphas:
            eq = quant_symmetric(emb_f, bits=bits, alpha=alpha)
            label = f"int{bits}_sym_a{alpha:.1f}"
            res = diagnostics(eq, label, ref_emb=emb_f, anchors_ids=anchors_ids)
            # Raw packed bytes: bits × E per row, round up
            res["bytes_per_row"] = (bits * E + 7) // 8
            res["compression"] = round(bytes_per_row_float / res["bytes_per_row"], 1)
            results.append(res)

    # Int2 ternary
    for alpha in alphas:
        eq = quant_ternary(emb_f, alpha=alpha)
        label = f"ternary_a{alpha:.1f}"
        res = diagnostics(eq, label, ref_emb=emb_f, anchors_ids=anchors_ids)
        res["bytes_per_row"] = (2 * E + 7) // 8  # 2 bits ternary packed
        res["compression"] = round(bytes_per_row_float / res["bytes_per_row"], 1)
        results.append(res)

    # Binary
    eq = quant_binary(emb_f)
    res = diagnostics(eq, "binary", ref_emb=emb_f, anchors_ids=anchors_ids)
    res["bytes_per_row"] = (E + 7) // 8  # 1 bit per dim packed
    res["compression"] = round(bytes_per_row_float / res["bytes_per_row"], 1)
    results.append(res)

    # Sort by compression descending within each acceptable-uniqueness group
    print(f"\n{'='*100}")
    print(f"{'label':<22} {'uniq':>8} {'pair':>8} {'rmse':>8} {'clust%':>8} "
          f"{'B/row':>6} {'compress':>10}")
    print("-" * 100)
    for r in results:
        uniq = r["uniq_pct"]
        pair = r["min_pair"]
        rmse = r["rmse_to_ref"] if r["rmse_to_ref"] is not None else float("nan")
        clust = r["cluster_overlap_pct"] if r["cluster_overlap_pct"] is not None else float("nan")
        print(f"{r['label']:<22} {uniq:>7.3f}% {pair:>8.4f} "
              f"{rmse:>8.5f} {clust:>7.1f}% {r['bytes_per_row']:>6d} "
              f"{r['compression']:>9.1f}x")

    # Dump full results
    (args.out / "quant_sweep_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {args.out / 'quant_sweep_results.json'}")

    # Also dump per-level best emb for deploy candidates
    best_per_bits = {}
    for r in results:
        if r["label"] == "float32":
            continue
        if r["cluster_overlap_pct"] is None:
            continue
        key = r["label"].split("_")[0]  # int4, int3, ternary, binary
        if (key not in best_per_bits
                or r["cluster_overlap_pct"] > best_per_bits[key][0]["cluster_overlap_pct"]):
            # need to reconstruct the emb for the best config
            best_per_bits[key] = (r, r["label"])

    print(f"\nBest config per bit-width (by cluster overlap):")
    for bw, (r, lbl) in best_per_bits.items():
        print(f"  {bw:<10} best: {lbl}  clust={r['cluster_overlap_pct']}%  "
              f"uniq={r['uniq_pct']}%  compress={r['compression']}x")


if __name__ == "__main__":
    main()
