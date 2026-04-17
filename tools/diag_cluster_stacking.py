"""Cluster-stacking exhaustive: add mini-clusters sequentially via residual boosting.

User's actual vision:
  Cluster 1: exhaustive-search a small classifier, predicts targets
  Cluster 2: exhaustive-search ANOTHER small classifier on RESIDUAL of C1
  Cluster 3: ANOTHER on residual of C1+C2
  ...

Each cluster has K=2 sparse binary weights per class row.
They stack additively on logits.

Tests whether multi-cluster stacking approaches dense float accuracy (34.2%).
"""

from __future__ import annotations

import sys
import time
import gc
from pathlib import Path
from itertools import combinations

import torch
import torch.nn.functional as F

CTX = 8
MASK_POS = CTX // 2
DIM = 8
VOCAB = 27
N_TRAIN = 5000
N_EVAL = 2000
K_PER_CLASS = 2  # sparsity per class row in each cluster
N_CLUSTERS_TEST = [50, 100, 200, 500, 1000]  # push to asymptote

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
    emb = embed[chunks]
    return emb.reshape(emb.shape[0], -1)


def exhaustive_cluster_search(features, residual_signal, K, chunk_size=20000):
    """Exhaustive search for a single cluster.

    features: (N, D)
    residual_signal: (N, n_classes) - target we want cluster to match
                     (e.g., normalized residual = 2*onehot - 1 for first cluster,
                      or y_true - y_pred_running for later clusters)

    Returns: (n_classes, D) binary weight matrix with K non-zero per row
    """
    N, D = features.shape
    n_classes = residual_signal.shape[1]

    position_combos = list(combinations(range(D), K))
    n_combos = len(position_combos)
    n_signs = 2 ** K
    total = n_combos * n_signs

    pos_np = torch.tensor(position_combos, device=DEVICE, dtype=torch.long)

    sign_combos = torch.zeros(n_signs, K, device=DEVICE, dtype=torch.float32)
    for i in range(n_signs):
        for j in range(K):
            sign_combos[i, j] = 1.0 if (i >> j) & 1 else -1.0

    best_score = torch.full((n_classes,), -float("inf"), device=DEVICE)
    best_pos = torch.zeros(n_classes, K, dtype=torch.long, device=DEVICE)
    best_sgn = torch.zeros(n_classes, K, device=DEVICE)

    pos_chunk = max(chunk_size // n_signs, 1)

    for pos_start in range(0, n_combos, pos_chunk):
        pos_end = min(pos_start + pos_chunk, n_combos)
        pos_batch = pos_np[pos_start:pos_end]
        B_pos = pos_batch.shape[0]

        gathered = features[:, pos_batch.flatten()].view(N, B_pos, K)
        outputs = torch.einsum('nbk,sk->nbs', gathered, sign_combos)
        outputs_flat = outputs.reshape(N, -1)

        # Score = correlation between config output and residual target per class
        scores = residual_signal.t() @ outputs_flat  # (n_classes, configs)

        chunk_best, chunk_idx = scores.max(dim=1)
        improved = chunk_best > best_score

        if improved.any():
            best_score[improved] = chunk_best[improved]
            for c in range(n_classes):
                if improved[c]:
                    ci = chunk_idx[c].item()
                    b = ci // n_signs
                    s = ci % n_signs
                    best_pos[c] = pos_batch[b]
                    best_sgn[c] = sign_combos[s]

        del gathered, outputs, outputs_flat, scores
        if pos_start % (pos_chunk * 50) == 0:
            torch.cuda.empty_cache()
            gc.collect()

    W = torch.zeros(n_classes, D, device=DEVICE)
    for c in range(n_classes):
        for k in range(K):
            W[c, best_pos[c, k]] = best_sgn[c, k]

    return W


def fit_cluster_scale(features, targets, W):
    """Find optimal scale for each class row.

    Given binary W, find scale alpha per class that minimizes cross entropy.
    Simple: alpha = 1 for all (or learn via brief gradient).
    """
    # Fit a single scalar per class via logistic regression-style
    # For simplicity: scale = 1 for all, bias = class log prior
    return torch.ones(W.shape[0], device=DEVICE)


def evaluate_linear(features, targets, W, bias):
    logits = features @ W.t() + bias
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item() * 100


def stacking_boost(train_features, train_targets, eval_features, eval_targets,
                    n_clusters, K):
    """Add clusters one at a time, each trained on residual."""
    N_train = train_features.shape[0]
    N_eval = eval_features.shape[0]
    D = train_features.shape[1]

    # Cumulative logits on train and eval
    cum_logits_train = torch.zeros(N_train, VOCAB, device=DEVICE)
    cum_logits_eval = torch.zeros(N_eval, VOCAB, device=DEVICE)

    # Class priors for initial bias
    class_counts = torch.bincount(train_targets, minlength=VOCAB).float()
    class_log_prior = (class_counts / class_counts.sum()).clamp(min=1e-9).log()

    # Add prior to cumulative logits (acts as initial "cluster 0")
    cum_logits_train = cum_logits_train + class_log_prior.unsqueeze(0)
    cum_logits_eval = cum_logits_eval + class_log_prior.unsqueeze(0)

    targets_onehot = F.one_hot(train_targets, num_classes=VOCAB).float()

    checkpoint_results = {}

    for cluster_idx in range(1, n_clusters + 1):
        # Compute residual: what we still need to predict
        # Using soft residual = targets_onehot - softmax(cum_logits)
        probs = F.softmax(cum_logits_train, dim=1)
        residual = targets_onehot - probs  # (N, 27) in [-1, 1]

        # Exhaustive search for this cluster's weights
        W_c = exhaustive_cluster_search(train_features, residual, K,
                                         chunk_size=20000)

        # Find scale per class (simple: unit scale + optimal bias)
        # For boosting, we can use a learning rate alpha
        # Fit: minimize || residual - alpha * (features @ W.t()) ||
        # Optimal alpha per class: (residual @ outputs) / (outputs @ outputs)
        with torch.no_grad():
            cluster_output_train = train_features @ W_c.t()  # (N, 27)
            cluster_output_eval = eval_features @ W_c.t()

            # Per-class optimal alpha via simple projection
            num = (residual * cluster_output_train).sum(dim=0)
            den = (cluster_output_train ** 2).sum(dim=0).clamp(min=1e-9)
            alpha = (num / den).clamp(min=0, max=5.0)  # (27,)

            # Also compute bias adjustment: keep constant log prior; add cluster output
            cum_logits_train = cum_logits_train + alpha.unsqueeze(0) * cluster_output_train
            cum_logits_eval = cum_logits_eval + alpha.unsqueeze(0) * cluster_output_eval

        # Evaluate at checkpoints
        if cluster_idx in N_CLUSTERS_TEST:
            train_pred = cum_logits_train.argmax(dim=1)
            eval_pred = cum_logits_eval.argmax(dim=1)
            train_acc = (train_pred == train_targets).float().mean().item() * 100
            eval_acc = (eval_pred == eval_targets).float().mean().item() * 100
            checkpoint_results[cluster_idx] = (train_acc, eval_acc)
            print(f"  After {cluster_idx:3d} clusters: train={train_acc:.2f}%  eval={eval_acc:.2f}%")

    return checkpoint_results


def main():
    fineweb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print("=== CLUSTER STACKING BOOST (exhaustive weak learners) ===")
    print(f"   Device: {DEVICE}")
    print(f"   Each cluster: 27-class linear, K={K_PER_CLASS} binary weights per class row")
    print(f"   Testing up to {max(N_CLUSTERS_TEST)} clusters stacked additively")
    print()

    corpus = load_corpus(fineweb_path)
    print(f"   Loaded: {len(corpus):,} bytes")

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
    D = train_features.shape[1]

    print(f"   Features: D={D} per sample")
    print(f"   Per cluster: C({D},{K_PER_CLASS}) x 2^{K_PER_CLASS} x 27 classes "
          f"= {len(list(combinations(range(D), K_PER_CLASS))) * (2**K_PER_CLASS) * VOCAB:,} configs")
    print()

    # Full float baseline
    print(">>> Baseline: full float linear")
    torch.manual_seed(42)
    W_fl = (torch.randn(VOCAB, D, device=DEVICE) * 0.1).detach().requires_grad_(True)
    b_fl = torch.zeros(VOCAB, device=DEVICE).requires_grad_(True)
    opt = torch.optim.Adam([W_fl, b_fl], lr=0.01)
    for ep in range(200):
        logits = train_features @ W_fl.t() + b_fl
        loss = F.cross_entropy(logits, train_targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    train_fl = evaluate_linear(train_features, train_targets, W_fl.detach(), b_fl.detach())
    eval_fl = evaluate_linear(eval_features, eval_targets, W_fl.detach(), b_fl.detach())
    print(f"   Full float train: {train_fl:.2f}%  eval: {eval_fl:.2f}%")
    print()

    # Run the stacking experiment
    print(f">>> Cluster stacking boost (K={K_PER_CLASS}, up to {max(N_CLUSTERS_TEST)} clusters)")
    t0 = time.time()
    results = stacking_boost(train_features, train_targets,
                              eval_features, eval_targets,
                              max(N_CLUSTERS_TEST), K_PER_CLASS)
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"  SUMMARY - cluster stacking (D={D}, K={K_PER_CLASS}, 27 classes)")
    print("=" * 60)
    print(f"  {'n_clusters':>11} {'Train':>8} {'Eval':>8} {'vs full float':>14}")
    print(f"  {'-'*11} {'-'*8} {'-'*8} {'-'*14}")
    for n in sorted(results.keys()):
        tr, ev = results[n]
        gap = ev - eval_fl
        print(f"  {n:>11} {tr:>7.2f}% {ev:>7.2f}% {gap:>+13.2f}pp")
    print()
    print(f"  Full float reference: train {train_fl:.2f}%  eval {eval_fl:.2f}%")
    print(f"  Total wallclock: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
