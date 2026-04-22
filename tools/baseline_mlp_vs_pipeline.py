#!/usr/bin/env python3
"""Baseline: Vanilla MLP vs ABC Pipeline MLP on FineWeb EDU quality classification.

Task:   Predict text quality quartile (0-3) from a fixed-size byte window.
        Labels derived from FineWeb EDU float 'score' field (quartile binning).
        This is a DIFFERENT task from Block C's training objective (next-pair pred).

Fair comparison: same total effective parameter budget.
  - Pipeline MLP: ABC (frozen ~116K params equivalent) + small MLP head
  - Vanilla MLP: bigger MLP with 116K extra params to match

Usage:
    source .venv/bin/activate
    python3 tools/baseline_mlp_vs_pipeline.py
"""

import sys
import time
import collections
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Python"))
from block_c_embedder import L2Embedder

# ── Config ──────────────────────────────────────────────────────
CTX_BYTES   = 128        # bytes per sample (= 64 byte-pairs)
CTX_PAIRS   = CTX_BYTES // 2
HIDDEN      = 64
N_CLASSES   = 4          # quality quartiles
BATCH       = 256
EPOCHS      = 20
LR          = 1e-3
N_SAMPLES   = 40_000     # total (train 80% / test 20%)
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

PARQUET     = REPO / "data" / "fineweb_edu_sample_000_00000.parquet"

# Pipeline frozen param equivalents (float32 count)
PIPELINE_A_PARAMS = 4_480      # LUT 256×16 + W1 8×16 + W2 16×16
PIPELINE_B_PARAMS = 2_867      # W 32×81 + biases + C19 params
PIPELINE_C_PARAMS = 108_416    # 3386 hot × 32 + scales + OOV
PIPELINE_TOTAL    = PIPELINE_A_PARAMS + PIPELINE_B_PARAMS + PIPELINE_C_PARAMS


# ── Data ────────────────────────────────────────────────────────
def load_data(n=N_SAMPLES):
    """Load text + float score, bin into quartiles."""
    pf = pq.ParquetFile(str(PARQUET))
    texts, scores = [], []
    for batch in pf.iter_batches(batch_size=20_000, columns=["text", "score"]):
        for t, s in zip(batch["text"].to_pylist(), batch["score"].to_pylist()):
            if t and s is not None and len(t) >= CTX_BYTES:
                texts.append(t)
                scores.append(float(s))
                if len(texts) >= n * 2:    # oversample, then subsample balanced
                    break
        if len(texts) >= n * 2:
            break

    scores_arr = np.array(scores)
    edges = np.percentile(scores_arr, [25, 50, 75])
    labels = np.digitize(scores_arr, edges)   # 0,1,2,3

    # Subsample to balanced classes
    rng = np.random.RandomState(SEED)
    by_class = collections.defaultdict(list)
    for i, lab in enumerate(labels):
        by_class[lab].append(i)
    per_class = n // N_CLASSES
    selected = []
    for c in range(N_CLASSES):
        pool = by_class[c]
        rng.shuffle(pool)
        selected.extend(pool[:per_class])
    rng.shuffle(selected)

    texts_out = [texts[i] for i in selected]
    labels_out = [int(labels[i]) for i in selected]
    return texts_out, labels_out, edges


def prepare_tensors(texts, labels, embedder):
    """Build byte-level and pipeline-embedded tensors."""
    n = len(texts)
    E = embedder.E

    X_bytes = np.zeros((n, CTX_BYTES), dtype=np.int64)
    X_pipe  = np.zeros((n, CTX_PAIRS, E), dtype=np.float32)
    Y       = np.array(labels, dtype=np.int64)

    for i, text in enumerate(texts):
        raw = text[:CTX_BYTES].encode("utf-8", errors="replace")[:CTX_BYTES]
        if len(raw) < CTX_BYTES:
            raw = raw + b"\x00" * (CTX_BYTES - len(raw))

        X_bytes[i] = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)

        pairs = np.frombuffer(raw, dtype=np.uint8).reshape(CTX_PAIRS, 2)
        pair_ids = (pairs[:, 0].astype(np.int64) << 8) | pairs[:, 1].astype(np.int64)
        X_pipe[i] = embedder.embed_ids(pair_ids)

    return X_bytes, X_pipe, Y


# ── Models ──────────────────────────────────────────────────────
class VanillaByteMLP(nn.Module):
    """Raw bytes → learned embedding → MLP.  Gets extra params to match budget."""
    def __init__(self, emb_dim, ctx, hidden, n_classes):
        super().__init__()
        self.emb = nn.Embedding(256, emb_dim)
        self.fc1 = nn.Linear(ctx * emb_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        e = self.emb(x).flatten(1)
        return self.fc2(self.relu(self.fc1(e)))


class PipelineMLP(nn.Module):
    """Frozen ABC embeddings → small MLP head."""
    def __init__(self, E, ctx_pairs, hidden, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(ctx_pairs * E, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x.flatten(1))))


# ── Training loop ───────────────────────────────────────────────
def train_eval(model, X_tr, Y_tr, X_te, Y_te, name, epochs=EPOCHS):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)),
        batch_size=BATCH, shuffle=True,
    )
    test_dl = DataLoader(
        TensorDataset(torch.tensor(X_te), torch.tensor(Y_te)),
        batch_size=BATCH,
    )

    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Params: {n_params:>9,} total  {n_train:>9,} trainable")
    print(f"{'='*60}")

    best_acc, history = 0.0, []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += len(yb)
        acc = correct / total
        best_acc = max(best_acc, acc)
        history.append(acc)

        elapsed = time.time() - t0
        print(f"  ep {ep:2d}  loss {np.mean(losses):.4f}  "
              f"test_acc {acc:.4f}  best {best_acc:.4f}  [{elapsed:.1f}s]")

    return best_acc, history


# ── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BASELINE: Vanilla MLP vs ABC Pipeline MLP")
    print("  Task: FineWeb EDU quality quartile classification")
    print("=" * 60)

    # Load embedder
    print("\nLoading ABC pipeline (frozen)...")
    embedder = L2Embedder.load_default()
    print(f"  {embedder}")

    # Load data
    print(f"\nLoading FineWeb EDU ({N_SAMPLES:,} balanced samples)...")
    texts, labels, edges = load_data()
    dist = collections.Counter(labels)
    print(f"  Quartile edges: {[f'{e:.3f}' for e in edges]}")
    print(f"  Class dist: {sorted(dist.items())}")

    # Prepare tensors
    print("Preparing tensors...")
    X_bytes, X_pipe, Y = prepare_tensors(texts, labels, embedder)

    # Split
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(Y))
    split = int(0.8 * len(Y))
    tr, te = idx[:split], idx[split:]

    # ── Size the vanilla MLP to match total budget ──
    pipe_model = PipelineMLP(embedder.E, CTX_PAIRS, HIDDEN, N_CLASSES)
    pipe_head = sum(p.numel() for p in pipe_model.parameters())
    total_budget = pipe_head + PIPELINE_TOTAL

    # Vanilla: 256*E_v + CTX_BYTES*E_v*HIDDEN + HIDDEN + HIDDEN*N_CLASSES + N_CLASSES
    fixed = HIDDEN + HIDDEN * N_CLASSES + N_CLASSES
    emb_factor = 256 + CTX_BYTES * HIDDEN
    E_vanilla = max(1, (total_budget - fixed) // emb_factor)

    vanilla = VanillaByteMLP(E_vanilla, CTX_BYTES, HIDDEN, N_CLASSES)
    v_params = sum(p.numel() for p in vanilla.parameters())

    print(f"\n  Pipeline head params:     {pipe_head:>9,}")
    print(f"  Pipeline frozen (equiv):  {PIPELINE_TOTAL:>9,}")
    print(f"  Total budget:             {total_budget:>9,}")
    print(f"  Vanilla emb_dim:          {E_vanilla}")
    print(f"  Vanilla total params:     {v_params:>9,}")
    print(f"  Pipeline head params:     {pipe_head:>9,}")
    print(f"  Budget match: {abs(v_params - total_budget) < 500}")

    # ── Train ──
    print(f"\nTraining on {DEVICE}, {EPOCHS} epochs, seed={SEED}")

    v_best, v_hist = train_eval(
        vanilla,
        X_bytes[tr], Y[tr], X_bytes[te], Y[te],
        f"Vanilla Byte MLP (E={E_vanilla}, H={HIDDEN})"
    )

    p_best, p_hist = train_eval(
        pipe_model,
        X_pipe[tr], Y[tr], X_pipe[te], Y[te],
        f"Pipeline MLP (ABC frozen, H={HIDDEN})"
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  RESULTS — Quality Quartile Classification (4-class)")
    print(f"{'='*60}")
    print(f"  Context:   {CTX_BYTES} bytes = {CTX_PAIRS} byte-pairs")
    print(f"  Train:     {split:,}   Test: {len(Y)-split:,}")
    print(f"  Random baseline: {100/N_CLASSES:.1f}%")
    print()
    print(f"  {'Model':<35s}  {'Best Acc':>8s}  {'Params':>10s}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*10}")
    print(f"  {'Vanilla Byte MLP':<35s}  {v_best:>7.2%}  {v_params:>10,}")
    print(f"  {'Pipeline MLP (ABC frozen)':<35s}  {p_best:>7.2%}  {pipe_head:>10,}*")
    print(f"    * + {PIPELINE_TOTAL:,} frozen pipeline params")
    print()
    delta = p_best - v_best
    winner = "Pipeline" if delta > 0 else "Vanilla" if delta < 0 else "Tie"
    print(f"  Delta: {delta:+.2%}  →  {winner} wins")
    print()
    print(f"  Pipeline advantage = {PIPELINE_TOTAL:,} frozen params worth")
    print(f"  of pre-trained byte-pair structure (66 KB packed)")

    # Save curves
    curves_path = REPO / "output" / "baseline_mlp_vs_pipeline.txt"
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    with open(curves_path, "w") as f:
        f.write("epoch\tvanilla_acc\tpipeline_acc\n")
        for i in range(len(v_hist)):
            f.write(f"{i+1}\t{v_hist[i]:.4f}\t{p_hist[i]:.4f}\n")
    print(f"  Curves saved: {curves_path}")


if __name__ == "__main__":
    main()
