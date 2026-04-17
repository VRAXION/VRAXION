"""Stacked TRUE ternary exhaustive clusters vs full float gradient network.

USER'S VISION PROPERLY:
  - Each cluster picks 16 random features from 64 total
  - On those 16, TRUE ternary exhaustive (3^16 = 43M configs)
  - Stack N clusters additively (boosting on residual)
  - Every weight is ternary {-1, 0, +1} -> mathematically optimal per cluster

COMPARE vs: full float gradient D=64, then int4/int8 quantize.

Proper trade-off analysis:
  - Accuracy: stacked-exhaustive vs float-then-quant
  - Total memory: N clusters × 16 ternary weights vs D=64 quantized
  - Total training time
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

CTX = 8
MASK_POS = CTX // 2
DIM = 8
VOCAB = 27
N_TRAIN = 5000
N_EVAL = 2000
D_FULL = (CTX - 1) * DIM  # 56 actually... let me recalc
# Wait: ctx=8, skip mask, = 7 positions, × DIM=8 = 56 features
D_CLUSTER = 14  # feature subset per cluster: 3^14 = 4.8M (fast)
# For 3^16 we need D=16, but let's try 14 for speed
N_CLUSTERS_LIST = [1, 5, 10, 20, 50]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_corpus(path):
    raw = Path(path).read_bytes()
    out = bytearray()
    for b in raw:
        if 97 <= b <= 122:
            out.append(b - 97)
        elif 65 <= b <= 90:
            out.append(b - 65)
        elif b in (32, 10, 9, 13):
            out.append(26)
    return torch.tensor(list(out), dtype=torch.long)


def sample_pairs(corpus, n, seed=42):
    gen = torch.Generator().manual_seed(seed)
    max_off = len(corpus) - CTX - 1
    offsets = torch.randint(0, max_off, (n,), generator=gen)
    idx_mat = offsets.unsqueeze(1) + torch.arange(CTX).unsqueeze(0)
    chunks = corpus[idx_mat]
    targets = chunks[:, MASK_POS]
    chunks = chunks.clone()
    chunks[:, MASK_POS] = 26
    return chunks, targets


def to_features(chunks, embed):
    B = chunks.shape[0]
    emb = embed[chunks]  # (B, CTX, DIM)
    # Skip mask position
    keep_mask = torch.ones(CTX, dtype=torch.bool, device=emb.device)
    keep_mask[MASK_POS] = False
    emb_kept = emb[:, keep_mask, :]  # (B, CTX-1, DIM)
    return emb_kept.reshape(B, -1)  # (B, (CTX-1)*DIM)


def generate_ternary_chunk(start, end, D_dim):
    n = end - start
    indices = torch.arange(start, end, device=DEVICE, dtype=torch.long)
    digits = torch.zeros(n, D_dim, device=DEVICE, dtype=torch.float32)
    for d in range(D_dim):
        digits[:, d] = ((indices // (3 ** d)) % 3).float()
    return digits - 1.0  # {-1, 0, +1}


def true_ternary_exhaustive_cluster(features_sub, residual, chunk_size=100000):
    """True ternary exhaustive on a feature SUBSET.

    features_sub: (N, D_cluster) - random subset selected
    residual: (N, 27) signed residual

    Returns: weights (27, D_cluster) ternary, score
    """
    N, D_dim = features_sub.shape
    n_classes = residual.shape[1]
    total_configs = 3 ** D_dim

    best_score = torch.full((n_classes,), -float("inf"), device=DEVICE)
    best_idx = torch.zeros(n_classes, dtype=torch.long, device=DEVICE)

    for chunk_start in range(0, total_configs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_configs)
        W_chunk = generate_ternary_chunk(chunk_start, chunk_end, D_dim)
        # W_chunk: (chunk, D_dim)

        outputs = features_sub @ W_chunk.t()  # (N, chunk)
        out_norms = outputs.pow(2).sum(dim=0).sqrt().clamp(min=1e-9)  # (chunk,)

        # Per-class score = residual[:, c].T @ outputs[:, conf] / out_norm[conf]
        scores = (residual.t() @ outputs) / out_norms.unsqueeze(0)  # (27, chunk)
        # Squared for positive magnitude (we want |correlation| not signed)
        scores_sq = scores.pow(2)

        chunk_best, chunk_idx = scores_sq.max(dim=1)
        improved = chunk_best > best_score
        if improved.any():
            global_idx = chunk_start + chunk_idx
            best_score = torch.where(improved, chunk_best, best_score)
            best_idx = torch.where(improved, global_idx, best_idx)

        del W_chunk, outputs, out_norms, scores, scores_sq

    # Decode best configs
    W_best = torch.zeros(n_classes, D_dim, device=DEVICE)
    for c in range(n_classes):
        idx = best_idx[c].item()
        for d in range(D_dim):
            digit = (idx // (3 ** d)) % 3
            W_best[c, d] = digit - 1.0

    return W_best


def run_stacked_exhaustive(train_features, train_targets, eval_features, eval_targets,
                           n_clusters_list, D_cluster, seed=42):
    """Stack N exhaustive clusters, each on random D_cluster subset of features."""
    gen = torch.Generator().manual_seed(seed)

    D_full = train_features.shape[1]
    N_train = train_features.shape[0]
    N_eval = eval_features.shape[0]

    class_counts = torch.bincount(train_targets, minlength=VOCAB).float()
    class_log_prior = (class_counts / class_counts.sum()).clamp(min=1e-9).log()

    cum_logits_train = class_log_prior.unsqueeze(0).expand(N_train, -1).clone()
    cum_logits_eval = class_log_prior.unsqueeze(0).expand(N_eval, -1).clone()

    targets_onehot = F.one_hot(train_targets, num_classes=VOCAB).float()

    results = {}
    max_clusters = max(n_clusters_list)
    t_start = time.time()

    for cluster_i in range(1, max_clusters + 1):
        # Pick random D_cluster feature subset
        perm = torch.randperm(D_full, generator=gen)[:D_cluster].to(DEVICE)

        feat_train_sub = train_features[:, perm]  # (N, D_cluster)
        feat_eval_sub = eval_features[:, perm]

        # Compute residual
        probs = F.softmax(cum_logits_train, dim=1)
        residual = targets_onehot - probs

        # True ternary exhaustive on subset
        W_c = true_ternary_exhaustive_cluster(feat_train_sub, residual)

        # Compute optimal alpha per class (boosting learning rate)
        with torch.no_grad():
            out_train = feat_train_sub @ W_c.t()  # (N, 27)
            out_eval = feat_eval_sub @ W_c.t()

            num = (residual * out_train).sum(dim=0)
            den = out_train.pow(2).sum(dim=0).clamp(min=1e-9)
            alpha = (num / den).clamp(min=0, max=5.0)

            cum_logits_train = cum_logits_train + alpha.unsqueeze(0) * out_train
            cum_logits_eval = cum_logits_eval + alpha.unsqueeze(0) * out_eval

        if cluster_i in n_clusters_list:
            train_pred = cum_logits_train.argmax(dim=1)
            eval_pred = cum_logits_eval.argmax(dim=1)
            train_acc = (train_pred == train_targets).float().mean().item() * 100
            eval_acc = (eval_pred == eval_targets).float().mean().item() * 100
            elapsed = time.time() - t_start
            results[cluster_i] = (train_acc, eval_acc, elapsed)
            print(f"  After {cluster_i:3d} clusters: train={train_acc:.2f}%  eval={eval_acc:.2f}%  [{elapsed:.1f}s total]")

    return results


def main():
    fineweb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print("=== STACKED TRUE-EXHAUSTIVE CLUSTERS vs FLOAT GRADIENT ===")
    print(f"   Device: {DEVICE}")
    print(f"   Full features D={D_FULL}, per-cluster subset D={D_CLUSTER}")
    print(f"   Cluster config space: 3^{D_CLUSTER} = {3**D_CLUSTER:,}")
    print()

    corpus = load_corpus(fineweb_path)

    torch.manual_seed(42)
    embed = torch.randn(VOCAB, DIM, device=DEVICE) * 0.5

    train_chunks, train_targets = sample_pairs(corpus, N_TRAIN, seed=42)
    eval_chunks, eval_targets = sample_pairs(corpus, N_EVAL, seed=99)

    train_chunks = train_chunks.to(DEVICE)
    train_targets = train_targets.to(DEVICE)
    eval_chunks = eval_chunks.to(DEVICE)
    eval_targets = eval_targets.to(DEVICE)

    train_features = to_features(train_chunks, embed)
    eval_features = to_features(eval_chunks, embed)
    print(f"   Train features: {train_features.shape}")
    print()

    # Baseline: Full float gradient
    print(">>> Baseline: Full float gradient on D=" + str(train_features.shape[1]))
    torch.manual_seed(42)
    D = train_features.shape[1]
    W = (torch.randn(VOCAB, D, device=DEVICE) * 0.1).detach().requires_grad_(True)
    b = torch.zeros(VOCAB, device=DEVICE).requires_grad_(True)
    opt = torch.optim.Adam([W, b], lr=0.01)
    t_fl = time.time()
    for ep in range(200):
        logits = train_features @ W.t() + b
        loss = F.cross_entropy(logits, train_targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    t_fl_elapsed = time.time() - t_fl
    logits_train = train_features @ W.detach().t() + b.detach()
    logits_eval = eval_features @ W.detach().t() + b.detach()
    train_fl = (logits_train.argmax(dim=1) == train_targets).float().mean().item() * 100
    eval_fl = (logits_eval.argmax(dim=1) == eval_targets).float().mean().item() * 100

    # Storage cost
    float_bytes = VOCAB * D * 4  # 4 bytes per float
    print(f"   Float gradient: train={train_fl:.2f}%  eval={eval_fl:.2f}%")
    print(f"   Storage: {VOCAB} x {D} x 4 bytes = {float_bytes:,} bytes = {float_bytes/1024:.1f} KB")
    print(f"   Training time: {t_fl_elapsed:.1f}s")
    print()

    # Simulate int4 quantization of float (naive PTQ)
    with torch.no_grad():
        max_w = W.detach().abs().max().clamp(min=1e-9)
        levels = 7.0
        W_q = (W.detach() / max_w * levels).round().clamp(-levels, levels) * max_w / levels
    logits_q = eval_features @ W_q.t() + b.detach()
    eval_int4 = (logits_q.argmax(dim=1) == eval_targets).float().mean().item() * 100
    int4_bytes = VOCAB * D * 4 // 8  # 4 bits per weight
    print(f"   Naive int4 PTQ: eval={eval_int4:.2f}%")
    print(f"   Int4 storage: {int4_bytes:,} bytes = {int4_bytes/1024:.1f} KB ({float_bytes/int4_bytes:.0f}x smaller)")
    print()

    # Stacked true-ternary-exhaustive clusters
    print(f">>> Stacked true-ternary-exhaustive clusters (D_cluster={D_CLUSTER}, random subsets)")
    results = run_stacked_exhaustive(train_features, train_targets,
                                      eval_features, eval_targets,
                                      N_CLUSTERS_LIST, D_CLUSTER)

    # Compute storage for each cluster count
    # Per cluster: D_CLUSTER positions (chosen from D_FULL=56), ternary weights
    # Position indexing: log2(D_FULL) bits per position = 6 bits for D=56
    bits_per_cluster = D_CLUSTER * 6 + VOCAB * D_CLUSTER * 2 + VOCAB * 32  # positions + ternary + alpha float
    # Simplified: ~((16×6) + (27×16×2) + 27×32) / 8 bytes = (96+864+864)/8 = 228 bytes per cluster

    print()
    print("=" * 75)
    print(f"  SUMMARY - Real trade-off (D_FULL={D_FULL}, D_CLUSTER={D_CLUSTER})")
    print("=" * 75)
    print(f"\n  Float baseline (gradient):")
    print(f"    Accuracy:       train={train_fl:.2f}%  eval={eval_fl:.2f}%")
    print(f"    Storage:        {float_bytes} bytes  ({float_bytes/1024:.2f} KB)")
    print(f"    Training time:  {t_fl_elapsed:.1f}s")

    print(f"\n  Float + naive int4 PTQ:")
    print(f"    Accuracy:       eval={eval_int4:.2f}%")
    print(f"    Storage:        {int4_bytes} bytes  ({int4_bytes/1024:.2f} KB)")
    print(f"    vs float:       {float_bytes/int4_bytes:.0f}x smaller, {eval_int4-eval_fl:+.2f}pp accuracy")

    print(f"\n  Stacked exhaustive clusters:")
    print(f"  {'Clusters':>10} {'Train':>8} {'Eval':>8} {'Time':>8} {'Storage':>12} {'vs float':>12}")
    for n in sorted(results.keys()):
        tr, ev, t = results[n]
        # Storage: per-cluster position indices + ternary weights + alpha scalar per class
        bytes_per_cluster = (D_CLUSTER * 6 + VOCAB * D_CLUSTER * 2 + VOCAB * 32) // 8
        total_bytes = n * bytes_per_cluster
        size_ratio = float_bytes / max(total_bytes, 1)
        print(f"  {n:>10} {tr:>7.2f}% {ev:>7.2f}% {t:>7.1f}s {total_bytes:>11}B "
              f"{size_ratio:>10.1f}x")


if __name__ == "__main__":
    main()
