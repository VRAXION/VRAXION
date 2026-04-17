"""Sparse-K exhaustive search V2: memory-managed.

V2 fixes:
  - Explicit del + empty_cache to prevent leak
  - Smaller chunks (10K instead of 50K)
  - Fused score computation (no large intermediate)
  - Targets float16 scores internally (not float32)

Tests K=3, K=4, K=5 on same task.
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


def exhaustive_search(features, targets_normalized, K, chunk_size=10000):
    """Memory-managed exhaustive sparse binary search.

    features: (N, D) float
    targets_normalized: (N, 27) in {-1, +1}
    Returns: (27, D) weight matrix
    """
    N, D = features.shape
    n_classes = targets_normalized.shape[1]

    position_combos = list(combinations(range(D), K))
    n_combos = len(position_combos)

    # Sign combos
    n_signs = 2 ** K
    sign_combos = torch.zeros(n_signs, K, device=DEVICE, dtype=torch.float16)
    for i in range(n_signs):
        for j in range(K):
            sign_combos[i, j] = 1.0 if (i >> j) & 1 else -1.0

    total = n_combos * n_signs
    print(f"  K={K}: {n_combos:,} pos combos x {n_signs} signs = {total:,} configs/class")
    print(f"  chunk_size={chunk_size}, {total//chunk_size} chunks/class to process")

    # Pre-pack positions (tensor form)
    pos_np = torch.tensor(position_combos, device=DEVICE, dtype=torch.long)  # (n_combos, K)

    # Best tracking
    best_score = torch.full((n_classes,), -float("inf"), device=DEVICE)
    best_pos = torch.zeros(n_classes, K, dtype=torch.long, device=DEVICE)
    best_sgn = torch.zeros(n_classes, K, device=DEVICE)

    # Process: iterate over POSITION chunks × all signs
    pos_chunk = chunk_size // n_signs  # number of position combos per chunk
    pos_chunk = max(pos_chunk, 1)

    for pos_start in range(0, n_combos, pos_chunk):
        pos_end = min(pos_start + pos_chunk, n_combos)
        pos_batch = pos_np[pos_start:pos_end]  # (B_pos, K)
        B_pos = pos_batch.shape[0]

        # Gather features at positions: (N, B_pos, K)
        # features[:, pos_batch.view(-1)].view(N, B_pos, K)
        gathered = features[:, pos_batch.flatten()].view(N, B_pos, K)

        # For each sign combo: apply and sum
        # Expand: (B_pos, n_signs, K) and multiply by gathered (N, B_pos, 1, K)
        # Get outputs (N, B_pos, n_signs) via einsum
        # gathered: (N, B_pos, K), sign_combos: (n_signs, K)
        # outputs[n, b, s] = sum_k gathered[n, b, k] * sign_combos[s, k]
        outputs = torch.einsum('nbk,sk->nbs',
                                gathered.half(),
                                sign_combos).to(torch.float32)
        # outputs: (N, B_pos, n_signs)

        # Flatten config dim: (N, B_pos*n_signs)
        outputs_flat = outputs.reshape(N, -1)

        # Score per class: (27, N) @ (N, configs) = (27, configs)
        scores = targets_normalized.t() @ outputs_flat

        # Find best per class in this chunk
        chunk_best, chunk_idx = scores.max(dim=1)
        improved = chunk_best > best_score

        if improved.any():
            # Update best
            global_idx = pos_start * n_signs + chunk_idx
            best_score[improved] = chunk_best[improved]
            # Decode position and sign indices
            for c in range(n_classes):
                if improved[c]:
                    ci = chunk_idx[c].item()
                    # ci = b * n_signs + s
                    b = ci // n_signs
                    s = ci % n_signs
                    best_pos[c] = pos_batch[b]
                    best_sgn[c] = sign_combos[s]

        # Explicit cleanup
        del gathered, outputs, outputs_flat, scores
        if pos_start % (pos_chunk * 50) == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Build weight matrix
    W = torch.zeros(n_classes, D, device=DEVICE)
    for c in range(n_classes):
        for k in range(K):
            W[c, best_pos[c, k]] = best_sgn[c, k]

    return W, best_score


def evaluate_linear(features, targets, W, bias):
    logits = features @ W.t() + bias
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item() * 100


def run_k(K, train_features, train_targets, eval_features, eval_targets,
          train_targets_normalized, class_log_prior):
    t = time.time()
    W, scores = exhaustive_search(train_features, train_targets_normalized, K)
    # Compute bias
    with torch.no_grad():
        logits_train = train_features @ W.t()
        bias = class_log_prior - logits_train.mean(dim=0)
    train_acc = evaluate_linear(train_features, train_targets, W, bias)
    eval_acc = evaluate_linear(eval_features, eval_targets, W, bias)
    elapsed = time.time() - t
    torch.cuda.empty_cache()
    gc.collect()
    return {"K": K, "train_acc": train_acc, "eval_acc": eval_acc, "sec": elapsed}


def main():
    fineweb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print("=== SPARSE-K EXHAUSTIVE V2 (memory-managed) ===")
    print(f"   Device: {DEVICE}")
    print(f"   Task: char-LM predict target from CTX={CTX} window")
    print(f"   Features: DIM={DIM}, total D={CTX*DIM}")
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

    train_targets_onehot = F.one_hot(train_targets, num_classes=VOCAB).float()
    train_targets_normalized = 2 * train_targets_onehot - 1

    class_counts = torch.bincount(train_targets, minlength=VOCAB).float()
    class_log_prior = (class_counts / class_counts.sum()).clamp(min=1e-9).log()

    # Also: full float baseline for reference
    print(">>> Baseline: full float linear (no sparsity)")
    torch.manual_seed(42)
    D = train_features.shape[1]
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

    results = []
    for K in [3, 4, 5]:
        print(f">>> Exhaustive K={K}")
        r = run_k(K, train_features, train_targets, eval_features, eval_targets,
                  train_targets_normalized, class_log_prior)
        print(f"   train: {r['train_acc']:.2f}%  eval: {r['eval_acc']:.2f}%  [{r['sec']:.1f}s]")
        print()
        results.append(r)

    print("=" * 60)
    print(f"  SUMMARY - sparse exhaustive sweep (D={D}, 27 classes)")
    print("=" * 60)
    print(f"  {'K':>4} {'Configs/class':>15} {'Train':>8} {'Eval':>8} {'Time':>8}")
    for r in results:
        n = (sum(1 for _ in combinations(range(D), r['K'])) * (2 ** r['K']))
        print(f"  {r['K']:>4} {n:>15,} {r['train_acc']:>7.2f}% "
              f"{r['eval_acc']:>7.2f}% {r['sec']:>7.1f}s")
    print()
    print(f"  Full float (K={D}): train {train_fl:.2f}%  eval {eval_fl:.2f}%")
    print(f"  Random baseline: {100/VOCAB:.2f}%")


if __name__ == "__main__":
    main()
