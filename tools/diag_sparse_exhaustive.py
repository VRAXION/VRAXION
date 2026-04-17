"""Quick test: sparse-K exhaustive binary vs gradient baseline.

Task: mini char-LM on FineWeb, predict target char from ctx=8 context.
Feature encoding: each char -> DIM=8 random embedding, total 64 features.

Model: 27-class linear predictor, but each class row has K=3 non-zero binary weights.

Comparison:
  A) Exhaustive: for each class, find best K=3 binary weights via brute force
  B) Gradient: standard SGD with L0 sparsity mask (STE)

Configs per class: C(64, 3) x 2^3 = 41,664 x 8 = 333,312

Run: python tools/diag_sparse_exhaustive.py <fineweb>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from itertools import combinations

import torch
import torch.nn.functional as F

CTX = 8
MASK_POS = CTX // 2
DIM = 8
VOCAB = 27
K = 5  # sparsity per class
N_TRAIN = 5000
N_EVAL = 2000
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
    chunks = corpus[idx_mat]  # (n, CTX)
    targets = chunks[:, MASK_POS]
    # Mask the middle position before feature extraction
    chunks = chunks.clone()
    chunks[:, MASK_POS] = 26  # treat masked position as whitespace
    return chunks, targets


def to_features(chunks, embed):
    """chunks: (N, CTX) -> features: (N, CTX * DIM)"""
    emb = embed[chunks]  # (N, CTX, DIM)
    return emb.reshape(emb.shape[0], -1)  # (N, CTX * DIM)


def exhaustive_search_per_class(features, targets_onehot, K, device, chunk_size=50000):
    """Find best K-sparse binary weights for each class.

    features: (N, D) float
    targets_onehot: (N, 27) float {0, 1}

    Returns: (27, D) weights where each row has K non-zero {-1, +1} values
    """
    N, D = features.shape
    n_classes = targets_onehot.shape[1]

    # Generate all K-combinations of positions
    position_combos = list(combinations(range(D), K))
    n_combos = len(position_combos)
    print(f"  Generated {n_combos} position combinations (C({D},{K}))")

    # Generate all sign combinations: 2^K
    n_signs = 2 ** K
    sign_combos = torch.zeros(n_signs, K, device=device)
    for i in range(n_signs):
        for j in range(K):
            sign_combos[i, j] = 1.0 if (i >> j) & 1 else -1.0

    # Total configs
    total_configs = n_combos * n_signs
    print(f"  Total configs to evaluate per class: {total_configs:,}")

    positions_tensor = torch.tensor(position_combos, device=device)  # (n_combos, K)
    positions_tensor = positions_tensor.unsqueeze(1).expand(-1, n_signs, -1).reshape(-1, K)  # (total_configs, K)
    signs_tensor = sign_combos.unsqueeze(0).expand(n_combos, -1, -1).reshape(-1, K)  # (total_configs, K)

    # Normalize targets: for each class, targets_normalized = (2 * onehot - 1) so +1/-1
    targets_normalized = (2 * targets_onehot - 1).to(device)  # (N, 27), values in {-1, +1}

    # Best tracking per class
    best_score = torch.full((n_classes,), -float("inf"), device=device)
    best_positions = torch.zeros(n_classes, K, dtype=torch.long, device=device)
    best_signs = torch.zeros(n_classes, K, device=device)

    # Process in chunks of configs for memory
    for chunk_start in range(0, total_configs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_configs)
        pos = positions_tensor[chunk_start:chunk_end]  # (chunk, K)
        sgn = signs_tensor[chunk_start:chunk_end]      # (chunk, K)

        # Gather features at positions: (N, chunk, K)
        # features: (N, D), pos: (chunk, K) -> features[:, pos.flatten()].reshape(N, chunk, K)
        gathered = features[:, pos.flatten()].reshape(N, -1, K)
        # Multiply by signs
        weighted = gathered * sgn.unsqueeze(0)  # (N, chunk, K)
        # Sum: (N, chunk)
        outputs = weighted.sum(dim=2)

        # For each class: score = correlation with class's targets_normalized
        # score[c, conf] = sum_n outputs[n, conf] * targets_normalized[n, c]
        # = (targets_normalized[:, c].T @ outputs)  for each class c
        # vectorized: (n_classes, N) @ (N, chunk) = (n_classes, chunk)
        scores = targets_normalized.t() @ outputs  # (27, chunk)

        # Update best per class
        chunk_best_score, chunk_best_idx = scores.max(dim=1)
        improved = chunk_best_score > best_score
        if improved.any():
            global_idx = chunk_start + chunk_best_idx
            best_score[improved] = chunk_best_score[improved]
            # Update positions & signs for improved classes
            for c in range(n_classes):
                if improved[c]:
                    idx = global_idx[c].item()
                    best_positions[c] = positions_tensor[idx]
                    best_signs[c] = signs_tensor[idx]

    # Build the 27 x D weight matrix
    W = torch.zeros(n_classes, D, device=device)
    for c in range(n_classes):
        for k in range(K):
            W[c, best_positions[c, k]] = best_signs[c, k]

    return W, best_score


def evaluate_linear(features, targets, W, bias):
    """Evaluate 27-class linear classifier."""
    logits = features @ W.t() + bias
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item() * 100


def train_gradient_sparse(features, targets, K, D, n_classes, seed=42):
    """Train gradient-based K-sparse binary classifier.
    Uses magnitude-based freezing similar to staged INQ but simpler."""
    torch.manual_seed(seed)
    W = (torch.randn(n_classes, D, device=DEVICE) * 0.1).detach().requires_grad_(True)
    bias = torch.zeros(n_classes, device=DEVICE).requires_grad_(True)
    opt = torch.optim.Adam([W, bias], lr=0.01)

    # Phase 1: train fully
    for ep in range(100):
        logits = features @ W.t() + bias
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Phase 2: identify top-K absolute weights per class, freeze to sign, zero rest
    with torch.no_grad():
        abs_W = W.abs()
        # For each class, find top K positions
        _, topk_idx = abs_W.topk(K, dim=1)  # (n_classes, K)
        # Create K-sparse mask
        mask = torch.zeros_like(W)
        for c in range(n_classes):
            mask[c, topk_idx[c]] = 1.0
        # Binarize signs (keep sign, magnitude = 1 for non-zero)
        W_sparse = W.sign() * mask
        # Apply to W
        W.data = W_sparse

    # Phase 3: fine-tune bias with W frozen
    for p in [W]:
        p.requires_grad_(False)
    opt = torch.optim.Adam([bias], lr=0.01)
    for ep in range(50):
        logits = features @ W.t() + bias
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return W.detach(), bias.detach()


def main():
    t0 = time.time()
    fineweb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print("=== SPARSE-K EXHAUSTIVE vs GRADIENT (quick test) ===")
    print(f"   Device: {DEVICE}")
    print(f"   Task: predict target char from CTX={CTX} context")
    print(f"   Features: DIM={DIM} per char, total D={CTX*DIM}")
    print(f"   Sparsity K={K} per class")
    print()

    corpus = load_corpus(fineweb_path)
    print(f"   Loaded corpus: {len(corpus):,} filtered bytes")

    # Random but fixed embedding (not learned)
    torch.manual_seed(42)
    embed = torch.randn(VOCAB, DIM, device=DEVICE) * 0.5

    # Sample train + eval
    train_chunks, train_targets = sample_pairs(corpus, N_TRAIN, seed=42)
    eval_chunks, eval_targets = sample_pairs(corpus, N_EVAL, seed=99)

    train_chunks = train_chunks.to(DEVICE)
    train_targets = train_targets.to(DEVICE)
    eval_chunks = eval_chunks.to(DEVICE)
    eval_targets = eval_targets.to(DEVICE)

    train_features = to_features(train_chunks, embed)  # (N, 64)
    eval_features = to_features(eval_chunks, embed)

    D = train_features.shape[1]
    print(f"   Train features: {train_features.shape}")
    print(f"   Eval features: {eval_features.shape}")
    print()

    # Method A: exhaustive
    print(">>> Method A: EXHAUSTIVE search")
    print(f"   Configs per class: C({D},{K}) x 2^{K} = "
          f"{len(list(combinations(range(D), K))) * (2**K):,}")
    t_ex = time.time()
    train_targets_onehot = F.one_hot(train_targets, num_classes=VOCAB).float()
    W_ex, scores = exhaustive_search_per_class(train_features, train_targets_onehot,
                                                 K, DEVICE)
    # Bias: per-class mean logit offset to balance
    with torch.no_grad():
        logits_train = train_features @ W_ex.t()
        # Use class priors for bias (log p(c))
        class_counts = torch.bincount(train_targets, minlength=VOCAB).float()
        class_log_prior = (class_counts / class_counts.sum()).log()
        bias_ex = class_log_prior - logits_train.mean(dim=0)

    train_acc_ex = evaluate_linear(train_features, train_targets, W_ex, bias_ex)
    eval_acc_ex = evaluate_linear(eval_features, eval_targets, W_ex, bias_ex)
    t_ex_elapsed = time.time() - t_ex
    print(f"   Train acc: {train_acc_ex:.2f}%")
    print(f"   Eval acc:  {eval_acc_ex:.2f}%")
    print(f"   Time: {t_ex_elapsed:.1f}s")
    print()

    # Method B: gradient-trained sparse
    print(">>> Method B: GRADIENT-trained sparse (top-K + sign)")
    t_gr = time.time()
    W_gr, bias_gr = train_gradient_sparse(train_features, train_targets, K, D,
                                            VOCAB, seed=42)
    train_acc_gr = evaluate_linear(train_features, train_targets, W_gr, bias_gr)
    eval_acc_gr = evaluate_linear(eval_features, eval_targets, W_gr, bias_gr)
    t_gr_elapsed = time.time() - t_gr
    print(f"   Train acc: {train_acc_gr:.2f}%")
    print(f"   Eval acc:  {eval_acc_gr:.2f}%")
    print(f"   Time: {t_gr_elapsed:.1f}s")
    print()

    # Method C: no-sparsity baseline (full float K=D)
    print(">>> Method C: FULL float baseline (no sparsity)")
    t_fl = time.time()
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
    train_acc_fl = evaluate_linear(train_features, train_targets, W_fl.detach(), b_fl.detach())
    eval_acc_fl = evaluate_linear(eval_features, eval_targets, W_fl.detach(), b_fl.detach())
    t_fl_elapsed = time.time() - t_fl
    print(f"   Train acc: {train_acc_fl:.2f}%")
    print(f"   Eval acc:  {eval_acc_fl:.2f}%")
    print(f"   Time: {t_fl_elapsed:.1f}s")
    print()

    # Summary
    print("=" * 70)
    print(f"  SUMMARY (D={D} features, K={K} per class, {VOCAB} classes)")
    print("=" * 70)
    print(f"  {'Method':<30} {'Train':>8} {'Eval':>8} {'Time':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Exhaustive K='+str(K)+' binary':<30} {train_acc_ex:>7.2f}% {eval_acc_ex:>7.2f}% "
          f"{t_ex_elapsed:>7.1f}s")
    print(f"  {'Gradient top-K='+str(K)+' binary':<30} {train_acc_gr:>7.2f}% {eval_acc_gr:>7.2f}% "
          f"{t_gr_elapsed:>7.1f}s")
    print(f"  {'Full float (K='+str(D)+')':<30} {train_acc_fl:>7.2f}% {eval_acc_fl:>7.2f}% "
          f"{t_fl_elapsed:>7.1f}s")
    print()
    random_baseline = 100.0 / VOCAB
    print(f"  Random baseline: {random_baseline:.2f}%")
    print(f"  Total wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
