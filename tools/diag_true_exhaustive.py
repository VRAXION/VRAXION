"""TRUE exhaustive search: ternary per-position (topology IS the search).

Each of D positions independently chooses {-1, 0, +1}:
  -1: negative connection exists
   0: no connection (topology Y/N)
  +1: positive connection exists

Total configs: 3^D (NOT C(D,K) × 2^K — no K guessing!)

This finds the GLOBALLY OPTIMAL sparse pattern AND weights together.

Comparison:
  Method A: ternary exhaustive (this script, D=16)
  Method B: gradient-then-exhaustive (train gradient, pick top-K, exhaustive binary)
  Method C: full float baseline

Task: predict char from CTX=2 window (one before, one after mask).
Features: DIM=8 per char -> D=16 total.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Smaller CTX so D is small enough for true exhaustive
CTX = 3  # 3 chars: before, mask, after (mask is zeroed)
MASK_POS = 1  # middle
DIM = 8
VOCAB = 27
N_TRAIN = 5000
N_EVAL = 2000
D = (CTX - 1) * DIM  # exclude masked position: 2 chars × 8 dim = 16
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
    return chunks, targets


def to_features(chunks, embed):
    # Skip the mask position
    B = chunks.shape[0]
    keep_positions = [i for i in range(CTX) if i != MASK_POS]
    selected = chunks[:, keep_positions]  # (B, CTX-1)
    emb = embed[selected]  # (B, CTX-1, DIM)
    return emb.reshape(B, -1)  # (B, (CTX-1) * DIM) = (B, D)


def generate_ternary_chunk(start, end, D_dim):
    """Generate ternary weight vectors for indices [start, end).

    Returns: (end-start, D_dim) tensor with values in {-1, 0, +1}
    """
    n = end - start
    indices = torch.arange(start, end, device=DEVICE, dtype=torch.long)
    digits = torch.zeros(n, D_dim, device=DEVICE, dtype=torch.float32)
    for d in range(D_dim):
        digits[:, d] = ((indices // (3 ** d)) % 3).float()
    # Map 0/1/2 -> -1/0/+1
    return digits - 1.0


def true_ternary_exhaustive(features, residual, chunk_size=50000):
    """TRUE exhaustive ternary search over all 3^D weight configurations.

    features: (N, D)
    residual: (N, 27) signed residual

    For each of 3^D config:
      - compute output per sample: (N,)
      - score per class: correlation-squared / norm
    Find best config per class.

    Returns: (27, D) weight matrix (ternary)
    """
    N, D_dim = features.shape
    n_classes = residual.shape[1]
    total_configs = 3 ** D_dim

    print(f"    TRUE ternary exhaustive: 3^{D_dim} = {total_configs:,} configs")
    print(f"    Processing in chunks of {chunk_size:,}")

    best_score = torch.full((n_classes,), -float("inf"), device=DEVICE)
    best_config_idx = torch.zeros(n_classes, dtype=torch.long, device=DEVICE)

    for chunk_start in range(0, total_configs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_configs)
        # Generate weight vectors
        W_chunk = generate_ternary_chunk(chunk_start, chunk_end, D_dim)
        # W_chunk: (chunk, D)

        # Compute outputs: (N, chunk) = features @ W_chunk.T
        outputs = features @ W_chunk.t()  # (N, chunk)

        # Normalize outputs for scoring: ||output|| per config
        output_norms = outputs.pow(2).sum(dim=0).sqrt().clamp(min=1e-9)  # (chunk,)

        # Score: ||(residual.T @ outputs / output_norm)||^2 over classes
        # per-class score: (residual[:, c].T @ output[:, conf]) / output_norm[conf]
        # shape: (27, chunk)
        scores = (residual.t() @ outputs) / output_norms.unsqueeze(0)

        # Find best config per class in this chunk
        chunk_best, chunk_idx = scores.max(dim=1)
        improved = chunk_best > best_score
        if improved.any():
            global_idx = chunk_start + chunk_idx
            best_score = torch.where(improved, chunk_best, best_score)
            best_config_idx = torch.where(improved, global_idx, best_config_idx)

        del W_chunk, outputs, output_norms, scores

    # Decode best configs back to weight vectors
    W_best = torch.zeros(n_classes, D_dim, device=DEVICE)
    for c in range(n_classes):
        idx = best_config_idx[c].item()
        for d in range(D_dim):
            digit = (idx // (3 ** d)) % 3
            W_best[c, d] = digit - 1.0

    # Compute optimal bias (class log prior + optimal offset)
    with torch.no_grad():
        logits_train = features @ W_best.t()
        class_counts = residual.shape[0] * (F.one_hot(
            torch.arange(n_classes, device=DEVICE), num_classes=n_classes).float().mean(dim=0))
        # Use balanced classes... actually let's use 0 bias for now and add later

    return W_best


def gradient_then_exhaustive(features, targets_onehot, K=3):
    """Hybrid: gradient determines topology, then exhaustive ±1 on those positions."""
    N, D_dim = features.shape
    n_classes = targets_onehot.shape[1]

    # Phase 1: train dense gradient model
    torch.manual_seed(42)
    W = (torch.randn(n_classes, D_dim, device=DEVICE) * 0.1).detach().requires_grad_(True)
    b = torch.zeros(n_classes, device=DEVICE).requires_grad_(True)
    opt = torch.optim.Adam([W, b], lr=0.01)
    targets = targets_onehot.argmax(dim=1)
    for ep in range(200):
        logits = features @ W.t() + b
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Phase 2: for each class, find top-K |weights|, then exhaustive ±1 on those positions
    W_detached = W.detach()
    W_final = torch.zeros_like(W_detached)

    for c in range(n_classes):
        # Top-K positions by |weight|
        abs_w = W_detached[c].abs()
        _, topk_pos = abs_w.topk(K)
        # Exhaustive ±1 on those K positions: 2^K configs
        feats_sub = features[:, topk_pos]  # (N, K)
        residual_c = 2 * targets_onehot[:, c] - 1  # (N,) in {-1, +1}

        best_score = -float("inf")
        best_signs = None
        for sign_idx in range(2 ** K):
            signs = torch.tensor([(1.0 if (sign_idx >> k) & 1 else -1.0) for k in range(K)],
                                  device=DEVICE)
            out = feats_sub @ signs
            score = (out * residual_c).sum().item() / (out.pow(2).sum().sqrt().item() + 1e-9)
            if score > best_score:
                best_score = score
                best_signs = signs

        for k in range(K):
            W_final[c, topk_pos[k]] = best_signs[k]

    return W_final


def evaluate(features, targets, W, bias=None):
    if bias is None:
        bias = torch.zeros(W.shape[0], device=DEVICE)
    logits = features @ W.t() + bias
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item() * 100


def main():
    fineweb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print("=== TRUE TERNARY EXHAUSTIVE (topology is part of search) ===")
    print(f"   Device: {DEVICE}")
    print(f"   CTX={CTX} chars (mask at pos {MASK_POS}, {CTX-1} context chars)")
    print(f"   DIM={DIM}, D={D} features")
    print(f"   Total ternary configs: 3^{D} = {3**D:,}")
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
    print(f"   Train features: {train_features.shape}  (D={train_features.shape[1]})")
    print()

    train_targets_onehot = F.one_hot(train_targets, num_classes=VOCAB).float()
    train_targets_normalized = 2 * train_targets_onehot - 1

    class_counts = torch.bincount(train_targets, minlength=VOCAB).float()
    class_log_prior = (class_counts / class_counts.sum()).clamp(min=1e-9).log()

    # Method A: TRUE ternary exhaustive
    print(">>> Method A: TRUE ternary exhaustive")
    t_a = time.time()
    W_ter = true_ternary_exhaustive(train_features, train_targets_normalized)
    bias_ter = class_log_prior - (train_features @ W_ter.t()).mean(dim=0)
    train_a = evaluate(train_features, train_targets, W_ter, bias_ter)
    eval_a = evaluate(eval_features, eval_targets, W_ter, bias_ter)
    t_a_elapsed = time.time() - t_a
    print(f"   Train: {train_a:.2f}%  Eval: {eval_a:.2f}%  [{t_a_elapsed:.1f}s]")
    print()

    # Method B: Gradient-then-exhaustive (hybrid)
    for K in [3, 4, 5]:
        print(f">>> Method B-K{K}: gradient topology + binary exhaustive (K={K})")
        t_b = time.time()
        W_hybrid = gradient_then_exhaustive(train_features, train_targets_onehot, K=K)
        bias_hybrid = class_log_prior - (train_features @ W_hybrid.t()).mean(dim=0)
        train_b = evaluate(train_features, train_targets, W_hybrid, bias_hybrid)
        eval_b = evaluate(eval_features, eval_targets, W_hybrid, bias_hybrid)
        t_b_elapsed = time.time() - t_b
        print(f"   Train: {train_b:.2f}%  Eval: {eval_b:.2f}%  [{t_b_elapsed:.1f}s]")
        print()

    # Method C: full float baseline
    print(">>> Method C: full float linear (gradient, no sparsity constraint)")
    t_c = time.time()
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
    train_c = evaluate(train_features, train_targets, W_fl.detach(), b_fl.detach())
    eval_c = evaluate(eval_features, eval_targets, W_fl.detach(), b_fl.detach())
    t_c_elapsed = time.time() - t_c
    print(f"   Train: {train_c:.2f}%  Eval: {eval_c:.2f}%  [{t_c_elapsed:.1f}s]")
    print()

    # Summary
    print("=" * 70)
    print(f"  SUMMARY (D={D}, 27 classes, {N_TRAIN} train / {N_EVAL} eval)")
    print("=" * 70)
    print(f"  True ternary exhaustive 3^{D}: train={train_a:.2f}  eval={eval_a:.2f}  [{t_a_elapsed:.1f}s]")
    print(f"  Full float gradient:           train={train_c:.2f}  eval={eval_c:.2f}  [{t_c_elapsed:.1f}s]")
    print()
    print(f"  Random baseline: {100/VOCAB:.2f}%")


if __name__ == "__main__":
    main()
