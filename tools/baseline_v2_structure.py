#!/usr/bin/env python3
"""Baseline v2: Vanilla MLP vs Pipeline MLP — text structure classification.

Task: Classify text window by structural type (4 classes derived from content):
  0 = list/bullets  (contains "\\n-", "\\n*", numbered items)
  1 = Q&A / dialogue (contains "?" with context)
  2 = technical      (high digit/symbol density)
  3 = narrative      (none of the above)

These structural labels are byte-pair-sensitive: list markers, question marks,
digits all have distinctive byte-pair patterns. If ABC's learned representations
capture structural semantics, Pipeline MLP should outperform.

Also tests longer context (512 bytes = 256 byte-pairs).
"""

import sys
import time
import collections
import re
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
CTX_BYTES   = 512
CTX_PAIRS   = CTX_BYTES // 2
HIDDEN      = 128
N_CLASSES   = 4
BATCH       = 256
EPOCHS      = 20
LR          = 1e-3
N_SAMPLES   = 40_000
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

PARQUET     = REPO / "data" / "fineweb_edu_sample_000_00000.parquet"

PIPELINE_TOTAL = 115_763   # A+B+C frozen float32 param equivalent

# ── Labeling heuristics ─────────────────────────────────────────
_LIST_PAT = re.compile(r'\n[\-\*•]|\n\d+[\.\)]')
_QA_PAT   = re.compile(r'\?')
_TECH_CHARS = set('0123456789=+<>{}[]()$%^&|\\/')

def classify_text(text: str) -> int:
    """Assign structural label based on content patterns."""
    window = text[:CTX_BYTES * 2]  # look a bit beyond context for labeling

    # List detection
    if _LIST_PAT.search(window):
        return 0

    # Q&A: at least 2 question marks
    if len(_QA_PAT.findall(window)) >= 2:
        return 1

    # Technical: >8% of chars are digits/symbols
    tech_count = sum(1 for c in window if c in _TECH_CHARS)
    if len(window) > 0 and tech_count / len(window) > 0.08:
        return 2

    return 3  # narrative


# ── Data ────────────────────────────────────────────────────────
def load_data(n=N_SAMPLES):
    pf = pq.ParquetFile(str(PARQUET))
    by_class = collections.defaultdict(list)
    per_class = n // N_CLASSES

    for batch in pf.iter_batches(batch_size=20_000, columns=["text"]):
        for t in batch["text"].to_pylist():
            if not t or len(t) < CTX_BYTES:
                continue
            label = classify_text(t)
            if len(by_class[label]) < per_class * 2:  # oversample
                by_class[label].append(t)
        if all(len(by_class[c]) >= per_class for c in range(N_CLASSES)):
            break

    rng = np.random.RandomState(SEED)
    texts, labels = [], []
    for c in range(N_CLASSES):
        pool = by_class[c]
        rng.shuffle(pool)
        for t in pool[:per_class]:
            texts.append(t)
            labels.append(c)

    idx = list(range(len(texts)))
    rng.shuffle(idx)
    return [texts[i] for i in idx], [labels[i] for i in idx]


def prepare_tensors(texts, labels, embedder):
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
    def __init__(self, emb_dim, ctx, hidden, n_classes):
        super().__init__()
        self.emb = nn.Embedding(256, emb_dim)
        self.fc1 = nn.Linear(ctx * emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, n_classes)

    def forward(self, x):
        e = self.emb(x).flatten(1)
        h = torch.relu(self.fc1(e))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


class PipelineMLP(nn.Module):
    def __init__(self, E, ctx_pairs, hidden, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(ctx_pairs * E, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, n_classes)

    def forward(self, x):
        h = torch.relu(self.fc1(x.flatten(1)))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


# ── Training ────────────────────────────────────────────────────
def train_eval(model, X_tr, Y_tr, X_te, Y_te, name, epochs=EPOCHS):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    train_dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)),
                          batch_size=BATCH, shuffle=True)
    test_dl  = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(Y_te)),
                          batch_size=BATCH)
    model.to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Params: {n_p:>9,} trainable")
    print(f"{'='*60}")

    best_acc, hist = 0.0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
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
        hist.append(acc)
        print(f"  ep {ep:2d}  loss {np.mean(losses):.4f}  "
              f"test_acc {acc:.4f}  best {best_acc:.4f}  [{time.time()-t0:.1f}s]")
    return best_acc, hist


# ── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BASELINE v2: Text Structure Classification")
    print("  Classes: list | Q&A | technical | narrative")
    print(f"  Context: {CTX_BYTES} bytes = {CTX_PAIRS} byte-pairs")
    print("=" * 60)

    embedder = L2Embedder.load_default()
    print(f"\n  Pipeline: {embedder}")

    print(f"\nLoading {N_SAMPLES:,} balanced samples...")
    texts, labels = load_data()
    dist = collections.Counter(labels)
    cls_names = {0: "list", 1: "Q&A", 2: "technical", 3: "narrative"}
    for c in sorted(dist):
        print(f"  [{c}] {cls_names[c]:>12s}: {dist[c]:,}")

    print("Preparing tensors...")
    X_bytes, X_pipe, Y = prepare_tensors(texts, labels, embedder)

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(Y))
    split = int(0.8 * len(Y))
    tr, te = idx[:split], idx[split:]

    # Size vanilla to match total budget
    pipe_model = PipelineMLP(embedder.E, CTX_PAIRS, HIDDEN, N_CLASSES)
    pipe_head = sum(p.numel() for p in pipe_model.parameters())
    total_budget = pipe_head + PIPELINE_TOTAL

    # Vanilla: 256*E + CTX*E*H + H + H*(H//2) + H//2 + (H//2)*4 + 4
    fixed = HIDDEN + HIDDEN * (HIDDEN // 2) + HIDDEN // 2 + (HIDDEN // 2) * N_CLASSES + N_CLASSES
    emb_factor = 256 + CTX_BYTES * HIDDEN
    E_vanilla = max(1, (total_budget - fixed) // emb_factor)

    vanilla = VanillaByteMLP(E_vanilla, CTX_BYTES, HIDDEN, N_CLASSES)
    v_params = sum(p.numel() for p in vanilla.parameters())

    print(f"\n  Pipeline head:    {pipe_head:>9,} trainable")
    print(f"  Pipeline frozen:  {PIPELINE_TOTAL:>9,} (equiv)")
    print(f"  Vanilla total:    {v_params:>9,} (all trainable)")
    print(f"  Vanilla E:        {E_vanilla}")

    print(f"\nTraining on {DEVICE}...")

    v_best, v_hist = train_eval(
        vanilla, X_bytes[tr], Y[tr], X_bytes[te], Y[te],
        f"Vanilla Byte MLP (E={E_vanilla})")

    p_best, p_hist = train_eval(
        pipe_model, X_pipe[tr], Y[tr], X_pipe[te], Y[te],
        "Pipeline MLP (ABC frozen)")

    # Per-class accuracy
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Random baseline: {100/N_CLASSES:.1f}%\n")
    print(f"  {'Model':<30s}  {'Best':>7s}  {'Params':>10s}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*10}")
    print(f"  {'Vanilla Byte MLP':<30s}  {v_best:>6.2%}  {v_params:>10,}")
    print(f"  {'Pipeline MLP (ABC)':<30s}  {p_best:>6.2%}  {pipe_head:>10,}*")
    print(f"    * + {PIPELINE_TOTAL:,} frozen pipeline")
    delta = p_best - v_best
    winner = "PIPELINE" if delta > 0.005 else "VANILLA" if delta < -0.005 else "TIE"
    print(f"\n  Delta: {delta:+.2%}  →  {winner}")

    # Save
    out = REPO / "output" / "baseline_v2_structure.txt"
    with open(out, "w") as f:
        f.write("epoch\tvanilla\tpipeline\n")
        for i in range(len(v_hist)):
            f.write(f"{i+1}\t{v_hist[i]:.4f}\t{p_hist[i]:.4f}\n")
    print(f"  Curves: {out}")


if __name__ == "__main__":
    main()
