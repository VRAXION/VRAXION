#!/usr/bin/env python3
"""Standard MLP baseline for next-char prediction on the same corpus as INSTNCT.

Task:   predict next character (27 classes: a-z + space) from context window.
        Identical to what INSTNCT Brain's evolve_language does.
Input:  lowercase a-z + space (same preprocessing as INSTNCT).
Metric: argmax accuracy on held-out portion.

Multiple model sizes to bracket INSTNCT's ~24.6% jackpot result.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parent.parent

# ── Config ──────────────────────────────────────────────────────
CHARS = 27                    # a-z (0..25) + space (26)
CONTEXT = 16                  # chars of context (match INSTNCT's temporal ticks)
BATCH = 512
EPOCHS = 30
LR = 1e-3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS_PATH = REPO / "instnct-core" / "tests" / "fixtures" / "alice_corpus.txt"


# ── Data ────────────────────────────────────────────────────────
def load_corpus(path):
    """Load corpus, map to 0-26 (a-z=0-25, space=26, skip rest)."""
    raw = path.read_text(encoding="utf-8", errors="replace").lower()
    ids = []
    for c in raw:
        if 'a' <= c <= 'z':
            ids.append(ord(c) - ord('a'))
        elif c == ' ':
            ids.append(26)
        # else skip
    return np.array(ids, dtype=np.int64)


def make_samples(corpus, context=CONTEXT):
    """Sliding window: (context, target) pairs."""
    n = len(corpus) - context
    X = np.zeros((n, context), dtype=np.int64)
    Y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        X[i] = corpus[i:i + context]
        Y[i] = corpus[i + context]
    return X, Y


# ── Models ──────────────────────────────────────────────────────
class CharMLP(nn.Module):
    """Embedding → flatten → hidden layers → 27 classes."""
    def __init__(self, emb_dim, context, hidden, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(CHARS, emb_dim)
        layers = [nn.Linear(context * emb_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, CHARS))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.emb(x).flatten(1))


class CharRNN(nn.Module):
    """Simple GRU for sequence modeling — gradient-trained recurrent baseline."""
    def __init__(self, emb_dim, hidden):
        super().__init__()
        self.emb = nn.Embedding(CHARS, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, CHARS)

    def forward(self, x):
        e = self.emb(x)
        _, h = self.gru(e)
        return self.fc(h.squeeze(0))


# ── Training ────────────────────────────────────────────────────
def train_eval(model, X_tr, Y_tr, X_te, Y_te, name, epochs=EPOCHS):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    train_dl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True)
    test_dl  = DataLoader(TensorDataset(X_te, Y_te), batch_size=BATCH)
    model.to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Params: {n_p:,}")
    print(f"{'='*60}")

    best_acc = 0.0
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
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:2d}  loss {np.mean(losses):.4f}  "
                  f"acc {acc:.2%}  best {best_acc:.2%}  [{time.time()-t0:.1f}s]")
    return best_acc, n_p


# ── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  STANDARD MODEL BASELINES — Next-Char Prediction")
    print("  Same task as INSTNCT evolve_language (27 classes, a-z+space)")
    print(f"  Corpus: {CORPUS_PATH.name} ({CORPUS_PATH.stat().st_size:,} bytes)")
    print(f"  Context: {CONTEXT} chars")
    print("=" * 60)

    corpus = load_corpus(CORPUS_PATH)
    print(f"\n  Corpus tokens: {len(corpus):,}")

    # Baselines
    from collections import Counter
    freq = Counter(corpus.tolist())
    most_common = freq.most_common(1)[0]
    freq_baseline = most_common[1] / len(corpus)
    random_baseline = 1.0 / CHARS
    print(f"  Random baseline:    {random_baseline:.2%}")
    print(f"  Frequency baseline: {freq_baseline:.2%} (char={most_common[0]})")

    # Bigram baseline
    bigram = np.zeros((CHARS, CHARS), dtype=np.float64)
    for i in range(len(corpus) - 1):
        bigram[corpus[i], corpus[i + 1]] += 1
    bigram_row_sum = bigram.sum(axis=1, keepdims=True).clip(min=1)
    bigram_pred = bigram.argmax(axis=1)
    bigram_correct = sum(1 for i in range(len(corpus) - 1)
                         if bigram_pred[corpus[i]] == corpus[i + 1])
    bigram_baseline = bigram_correct / (len(corpus) - 1)
    print(f"  Bigram baseline:    {bigram_baseline:.2%}")

    X, Y = make_samples(corpus)
    n = len(Y)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    tr, te = idx[:split], idx[split:]

    X_tr = torch.tensor(X[tr]); Y_tr = torch.tensor(Y[tr])
    X_te = torch.tensor(X[te]); Y_te = torch.tensor(Y[te])

    # ── Models at different scales ──
    # INSTNCT Brain: H=256 neurons, ~3000 edges after 30K steps
    # We test MLPs and a GRU at comparable scales

    results = []

    # Tiny MLP (~3K params — matches INSTNCT edge count)
    acc, n_p = train_eval(
        CharMLP(emb_dim=4, context=CONTEXT, hidden=32, n_layers=1),
        X_tr, Y_tr, X_te, Y_te, "Tiny MLP (E=4, H=32)")
    results.append(("Tiny MLP", acc, n_p))

    # Small MLP (~30K params)
    acc, n_p = train_eval(
        CharMLP(emb_dim=16, context=CONTEXT, hidden=64, n_layers=1),
        X_tr, Y_tr, X_te, Y_te, "Small MLP (E=16, H=64)")
    results.append(("Small MLP", acc, n_p))

    # Medium MLP (~100K params)
    acc, n_p = train_eval(
        CharMLP(emb_dim=32, context=CONTEXT, hidden=128, n_layers=2),
        X_tr, Y_tr, X_te, Y_te, "Medium MLP (E=32, H=128, 2-layer)")
    results.append(("Medium MLP", acc, n_p))

    # Large MLP (~500K params)
    acc, n_p = train_eval(
        CharMLP(emb_dim=64, context=CONTEXT, hidden=256, n_layers=2),
        X_tr, Y_tr, X_te, Y_te, "Large MLP (E=64, H=256, 2-layer)")
    results.append(("Large MLP", acc, n_p))

    # GRU (~30K params — recurrent, like INSTNCT)
    acc, n_p = train_eval(
        CharRNN(emb_dim=16, hidden=64),
        X_tr, Y_tr, X_te, Y_te, "Small GRU (E=16, H=64)")
    results.append(("Small GRU", acc, n_p))

    # GRU (~100K params)
    acc, n_p = train_eval(
        CharRNN(emb_dim=32, hidden=128),
        X_tr, Y_tr, X_te, Y_te, "Medium GRU (E=32, H=128)")
    results.append(("Medium GRU", acc, n_p))

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':<35s}  {'Acc':>7s}  {'Params':>10s}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*10}")
    print(f"  {'Random baseline':<35s}  {random_baseline:>6.2%}  {'—':>10s}")
    print(f"  {'Frequency baseline':<35s}  {freq_baseline:>6.2%}  {'—':>10s}")
    print(f"  {'Bigram baseline':<35s}  {bigram_baseline:>6.2%}  {'—':>10s}")
    print(f"  {'INSTNCT Brain (24.6% reported)':<35s}  {'~24.6%':>7s}  {'~3K edges':>10s}")
    for name, acc, n_p in results:
        print(f"  {name:<35s}  {acc:>6.2%}  {n_p:>10,}")

    # Save
    out = REPO / "output" / "baseline_standard_charpredict.txt"
    with open(out, "w") as f:
        f.write("model\taccuracy\tparams\n")
        f.write(f"random\t{random_baseline:.4f}\t0\n")
        f.write(f"frequency\t{freq_baseline:.4f}\t0\n")
        f.write(f"bigram\t{bigram_baseline:.4f}\t0\n")
        f.write(f"instnct_brain\t0.246\t3000\n")
        for name, acc, n_p in results:
            f.write(f"{name}\t{acc:.4f}\t{n_p}\n")
    print(f"\n  Results: {out}")


if __name__ == "__main__":
    main()
