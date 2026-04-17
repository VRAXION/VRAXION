"""Bigger cluster stacking: Beukers-neuron clusters, joint exhaustive.

Each cluster = 1 Beukers-gate neuron:
  - Projection A: K=2 sparse binary weights (D inputs)
  - Projection B: K=2 sparse binary weights (D inputs)
  - Feature = A*B / (1 + |A*B|) (Beukers gate)
  - Contributes to 27-class logits via analytical optimal scaling

Joint exhaustive search: all (ws_A, ws_B) pairs.
Per cluster: ~65M joint configs = 1-2 min GPU exhaustive.
Stack 5-15 clusters, each trained on residual of previous.

Design choice: "N-neuron mini-cluster" = 1 Beukers neuron (with A, B substructures).
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
K = 2  # sparsity per projection
N_CLUSTERS_TOTAL = 15  # how many clusters to stack
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


def precompute_sparse_outputs(features, K):
    """Precompute outputs for all K-sparse binary weight configurations.

    features: (N, D)
    Returns: (N, n_configs), and (n_configs, K) positions, (n_configs, K) signs
    """
    N, D = features.shape
    position_combos = list(combinations(range(D), K))
    n_combos = len(position_combos)
    n_signs = 2 ** K
    total = n_combos * n_signs

    pos_tensor = torch.tensor(position_combos, device=DEVICE, dtype=torch.long)  # (n_combos, K)

    sign_combos = torch.zeros(n_signs, K, device=DEVICE, dtype=torch.float32)
    for i in range(n_signs):
        for j in range(K):
            sign_combos[i, j] = 1.0 if (i >> j) & 1 else -1.0

    # Gather features: (N, n_combos, K)
    gathered = features[:, pos_tensor.flatten()].view(N, n_combos, K)
    # Apply signs: outputs[n, c, s] = sum_k gathered[n, c, k] * sign_combos[s, k]
    outputs = torch.einsum('nck,sk->ncs', gathered, sign_combos).reshape(N, -1)
    # outputs: (N, n_combos * n_signs) = (N, total)

    # Store position+sign for each config
    pos_expanded = pos_tensor.unsqueeze(1).expand(-1, n_signs, -1).reshape(-1, K)  # (total, K)
    sgn_expanded = sign_combos.unsqueeze(0).expand(n_combos, -1, -1).reshape(-1, K)  # (total, K)

    return outputs, pos_expanded, sgn_expanded


def exhaustive_beukers_cluster(features, residual, K, chunk_size=256):
    """Joint exhaustive over (ws_A, ws_B), each K-sparse binary.

    features: (N, D)
    residual: (N, 27) signed residual target

    Returns: best (ws_A, ws_B, alpha_output) where:
      ws_A, ws_B: (D,) binary sparse weight vectors
      alpha_output: (27,) output scaling for the Beukers feature
    """
    N, D = features.shape
    n_classes = residual.shape[1]

    # Pre-compute all A outputs (same as B outputs, symmetric)
    A_outputs, A_positions, A_signs = precompute_sparse_outputs(features, K)
    # A_outputs: (N, n_A_configs)
    n_A = A_outputs.shape[1]
    B_outputs = A_outputs  # same (symmetric)
    n_B = n_A

    print(f"    Per-projection configs: {n_A:,}  Joint: {n_A * n_B:,}")

    best_score = -float("inf")
    best_a_idx = 0
    best_b_idx = 0

    # Residual precompute (for efficiency)
    # We want to maximize: sum over classes of (beukers_output . residual[:, c])^2 / (beukers . beukers)
    # simpler proxy: maximize sum over c of abs(beukers . residual[:, c]) / |beukers|

    # Iterate over A configs (outer loop)
    for a_start in range(0, n_A, chunk_size):
        a_end = min(a_start + chunk_size, n_A)
        A_batch = A_outputs[:, a_start:a_end]  # (N, B_a)

        # For each a in A_batch, compute joint Beukers with all B configs
        # beukers(a, b)[n] = A_batch[n, a] * B_outputs[n, b] / (1 + |A*B|)
        # shape: (N, B_a, n_B)
        # Too big to materialize at once — do inner loop over B too

        for b_start in range(0, n_B, chunk_size):
            b_end = min(b_start + chunk_size, n_B)
            B_batch = B_outputs[:, b_start:b_end]  # (N, B_b)

            # Compute product: (N, B_a, B_b)
            AB = A_batch.unsqueeze(2) * B_batch.unsqueeze(1)  # (N, B_a, B_b)
            # Beukers gate
            co = AB / (1.0 + AB.abs())  # (N, B_a, B_b)
            # Normalize by sqrt(sum co^2) per config for fair scoring
            co_norm = co.pow(2).sum(dim=0).sqrt().clamp(min=1e-9)  # (B_a, B_b)

            # Score: sum over classes of (co . residual[:, c])^2 / |co|^2
            # = sum_c (residual[:, c].T @ co[n, a, b])^2 / co_norm[a, b]^2
            # = ||(residual.T @ co_flat)||^2 / co_norm^2 (l2 norm over classes)
            co_flat = co.reshape(N, -1)  # (N, B_a*B_b)
            # residual: (N, 27) -> residual.T: (27, N) @ co_flat: (N, B_a*B_b) = (27, B_a*B_b)
            proj = residual.t() @ co_flat  # (27, B_a*B_b)
            # Score per config: sum over classes of proj^2 / co_norm^2
            # proj.pow(2).sum(dim=0): (B_a*B_b,)
            scores_flat = proj.pow(2).sum(dim=0) / co_norm.reshape(-1).pow(2)  # (B_a*B_b,)

            # Find best in this chunk
            best_in_chunk, idx_in_chunk = scores_flat.max(dim=0)
            if best_in_chunk.item() > best_score:
                best_score = best_in_chunk.item()
                b_idx_global = b_start + (idx_in_chunk.item() % (b_end - b_start))
                a_idx_global = a_start + (idx_in_chunk.item() // (b_end - b_start))
                best_a_idx = a_idx_global
                best_b_idx = b_idx_global

            del AB, co, co_flat, proj, scores_flat, co_norm

        # Periodic GC
        if a_start % (chunk_size * 10) == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Recompute best Beukers feature and optimal output alpha
    with torch.no_grad():
        A_out = A_outputs[:, best_a_idx]  # (N,)
        B_out = B_outputs[:, best_b_idx]  # (N,)
        AB = A_out * B_out
        beukers = AB / (1.0 + AB.abs())  # (N,)
        # Optimal per-class alpha: alpha_c = (beukers.T @ residual[:, c]) / (beukers.T @ beukers)
        denom = beukers.pow(2).sum().clamp(min=1e-9)
        alpha = (residual.t() @ beukers) / denom  # (27,)

        # Build full weight vectors
        ws_A = torch.zeros(D, device=DEVICE)
        for k in range(K):
            ws_A[A_positions[best_a_idx, k]] = A_signs[best_a_idx, k]
        ws_B = torch.zeros(D, device=DEVICE)
        for k in range(K):
            ws_B[A_positions[best_b_idx, k]] = A_signs[best_b_idx, k]

    return ws_A, ws_B, alpha, best_score


def compute_beukers_feature(features, ws_A, ws_B):
    """Compute Beukers gate feature for given weight vectors."""
    A_out = features @ ws_A  # (N,)
    B_out = features @ ws_B  # (N,)
    AB = A_out * B_out
    return AB / (1.0 + AB.abs())  # (N,)


def evaluate(cum_logits, targets):
    pred = cum_logits.argmax(dim=1)
    return (pred == targets).float().mean().item() * 100


def main():
    fineweb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print("=== BEUKERS-CLUSTER STACKING (bigger clusters, joint exhaustive) ===")
    print(f"   Device: {DEVICE}")
    print(f"   Each cluster: 1 Beukers neuron, K={K} sparse binary per projection")
    print(f"   Per cluster ~65M joint configs, expect 30-60s each on GPU")
    print(f"   Stacking: {N_CLUSTERS_TOTAL} clusters total")
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
    D = train_features.shape[1]
    print(f"   Features: D={D}")
    print()

    # Full float baseline for reference
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
    logits_train = train_features @ W_fl.detach().t() + b_fl.detach()
    logits_eval = eval_features @ W_fl.detach().t() + b_fl.detach()
    print(f"   Full float: train {evaluate(logits_train, train_targets):.2f}%  "
          f"eval {evaluate(logits_eval, eval_targets):.2f}%")
    print()

    # Class priors for initial logits
    class_counts = torch.bincount(train_targets, minlength=VOCAB).float()
    class_log_prior = (class_counts / class_counts.sum()).clamp(min=1e-9).log()
    cum_logits_train = class_log_prior.unsqueeze(0).expand(N_TRAIN, -1).clone()
    cum_logits_eval = class_log_prior.unsqueeze(0).expand(N_EVAL, -1).clone()

    targets_onehot = F.one_hot(train_targets, num_classes=VOCAB).float()

    # Stack clusters
    print(f">>> Stacking {N_CLUSTERS_TOTAL} Beukers-clusters")
    total_t = time.time()
    for cluster_i in range(1, N_CLUSTERS_TOTAL + 1):
        t_c = time.time()

        # Compute residual on training data
        probs = F.softmax(cum_logits_train, dim=1)
        residual = targets_onehot - probs

        print(f"  Cluster {cluster_i}/{N_CLUSTERS_TOTAL}: searching...")
        ws_A, ws_B, alpha, score = exhaustive_beukers_cluster(
            train_features, residual, K)

        # Update cumulative logits (train and eval)
        f_train = compute_beukers_feature(train_features, ws_A, ws_B)  # (N,)
        f_eval = compute_beukers_feature(eval_features, ws_A, ws_B)

        # Add alpha * feature contribution to each class
        cum_logits_train = cum_logits_train + alpha.unsqueeze(0) * f_train.unsqueeze(1)
        cum_logits_eval = cum_logits_eval + alpha.unsqueeze(0) * f_eval.unsqueeze(1)

        train_acc = evaluate(cum_logits_train, train_targets)
        eval_acc = evaluate(cum_logits_eval, eval_targets)
        dt = time.time() - t_c
        print(f"  Cluster {cluster_i}: train={train_acc:.2f}%  eval={eval_acc:.2f}%  [{dt:.1f}s]")

    total_dt = time.time() - total_t
    print()
    print(f"=== FINAL after {N_CLUSTERS_TOTAL} Beukers-clusters ===")
    final_train = evaluate(cum_logits_train, train_targets)
    final_eval = evaluate(cum_logits_eval, eval_targets)
    print(f"  train: {final_train:.2f}%")
    print(f"  eval:  {final_eval:.2f}%")
    print(f"  Full float reference eval: {evaluate(logits_eval, eval_targets):.2f}%")
    print(f"  Total wallclock: {total_dt:.1f}s")


if __name__ == "__main__":
    main()
