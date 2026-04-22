#!/usr/bin/env python3
"""Baseline v3: Word-level next-token prediction — the natural "Block D" task.

Tests whether C's byte-pair embeddings compose into useful word representations.

Three models (same total param budget):
  1. Vanilla Byte:   learned byte embeddings → avg-pool per word → MLP → next word
  2. Pipeline (ABC):  frozen C byte-pair embs → avg-pool per word → MLP → next word
  3. Direct Word Emb: learned word embeddings (Embedding table) → MLP → next word

The "knee" should appear in the context sweep: as context grows, the pipeline
should maintain/grow its advantage because C's structured embeddings compose
better than raw byte embeddings.

Task: predict next word from context of K previous words.
Vocab: top-1000 most frequent words (+ UNK bucket).
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
VOCAB_SIZE    = 1001          # top 1000 + UNK
MAX_WORD_BYTES = 20           # max bytes per word (pad/truncate)
MAX_WORD_PAIRS = MAX_WORD_BYTES // 2  # = 10 byte-pairs per word
HIDDEN        = 64
BATCH         = 512
EPOCHS        = 15
LR            = 1e-3
N_SAMPLES     = 100_000       # total word-prediction samples
SEED          = 42
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

CONTEXT_SIZES = [2, 4, 8, 16]  # word context lengths to sweep

PARQUET       = REPO / "data" / "fineweb_edu_sample_000_00000.parquet"
PIPELINE_TOTAL = 115_763

# ── Build vocabulary ────────────────────────────────────────────
_WORD_PAT = re.compile(r'[a-zA-Z]+')

def build_vocab_and_samples(n_samples=N_SAMPLES, max_ctx=max(CONTEXT_SIZES)):
    """Build word vocab from corpus, extract (context, target) samples."""
    pf = pq.ParquetFile(str(PARQUET))

    # Pass 1: count word frequencies
    word_freq = collections.Counter()
    all_words = []
    print("  Pass 1: counting word frequencies...")
    for batch in pf.iter_batches(batch_size=10_000, columns=["text"]):
        for t in batch["text"].to_pylist():
            if not t:
                continue
            words = _WORD_PAT.findall(t.lower())
            word_freq.update(words)
            all_words.extend(words)
            all_words.append("<SEP>")  # document separator
            if len(all_words) > n_samples * 5:
                break
        if len(all_words) > n_samples * 5:
            break

    # Build vocab: top 1000 words
    UNK_ID = 0
    vocab = {"<UNK>": UNK_ID}
    for word, _ in word_freq.most_common(VOCAB_SIZE - 1):
        vocab[word] = len(vocab)
    inv_vocab = {v: k for k, v in vocab.items()}

    print(f"  Vocab: {len(vocab)} words, corpus: {len(all_words):,} tokens")
    print(f"  Top-10: {[w for w,_ in word_freq.most_common(10)]}")

    # Pass 2: extract (context, target) pairs for each context size
    word_ids = []
    for w in all_words:
        if w == "<SEP>":
            word_ids.append(-1)  # separator
        else:
            word_ids.append(vocab.get(w, UNK_ID))

    samples_by_ctx = {}
    for ctx_len in CONTEXT_SIZES:
        contexts, targets = [], []
        for i in range(ctx_len, len(word_ids)):
            target = word_ids[i]
            if target < 0:
                continue  # skip separator
            ctx = word_ids[i - ctx_len:i]
            if any(c < 0 for c in ctx):
                continue  # skip if context crosses document boundary
            if target == UNK_ID:
                continue  # skip UNK targets for cleaner signal
            contexts.append(ctx)
            targets.append(target)
            if len(contexts) >= n_samples:
                break
        samples_by_ctx[ctx_len] = (contexts, targets)
        print(f"  ctx={ctx_len}: {len(contexts):,} samples")

    return vocab, inv_vocab, samples_by_ctx


def word_to_byte_pairs(word: str) -> list:
    """Convert word to byte-pair IDs (hi<<8 | lo)."""
    raw = word.encode("utf-8", errors="replace")[:MAX_WORD_BYTES]
    if len(raw) % 2 == 1:
        raw = raw + b"\x00"  # pad odd byte
    pairs = []
    for j in range(0, len(raw), 2):
        pairs.append((raw[j] << 8) | raw[j + 1])
    return pairs


def word_to_bytes(word: str) -> list:
    """Convert word to raw byte values."""
    raw = word.encode("utf-8", errors="replace")[:MAX_WORD_BYTES]
    return list(raw)


# ── Models ──────────────────────────────────────────────────────
class VanillaByteMLP(nn.Module):
    """Byte embeddings → avg-pool per word → concat context → MLP → next word."""
    def __init__(self, emb_dim, ctx_len, word_dim, hidden, n_classes):
        super().__init__()
        self.emb = nn.Embedding(256, emb_dim)
        self.word_dim = word_dim
        self.fc1 = nn.Linear(ctx_len * word_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x: (B, ctx_len, max_bytes) long
        B, K, T = x.shape
        e = self.emb(x)                # (B, K, T, emb_dim)
        # mask zeros (padding)
        mask = (x > 0).float().unsqueeze(-1)  # (B, K, T, 1)
        counts = mask.sum(dim=2).clamp(min=1)  # (B, K, 1)
        word_vecs = (e * mask).sum(dim=2) / counts  # (B, K, emb_dim)
        # project to word_dim if needed
        flat = word_vecs.reshape(B, -1)
        return self.fc2(torch.relu(self.fc1(flat)))


class PipelineMLP(nn.Module):
    """Frozen C byte-pair embs → avg-pool per word → concat → MLP → next word."""
    def __init__(self, E, ctx_len, hidden, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(ctx_len * E, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x: (B, ctx_len, E) float — pre-pooled word vectors from C
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))


class DirectWordMLP(nn.Module):
    """Learned word embeddings → concat → MLP → next word."""
    def __init__(self, vocab_size, emb_dim, ctx_len, hidden, n_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(ctx_len * emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x: (B, ctx_len) long — word IDs
        e = self.emb(x).flatten(1)
        return self.fc2(torch.relu(self.fc1(e)))


# ── Training ────────────────────────────────────────────────────
def train_eval(model, X_tr, Y_tr, X_te, Y_te, name, epochs=EPOCHS):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    train_dl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True)
    test_dl  = DataLoader(TensorDataset(X_te, Y_te), batch_size=BATCH)
    model.to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  {name} ({n_p:,} trainable)")
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
        if ep % 5 == 0 or ep == 1:
            print(f"    ep {ep:2d}  loss {np.mean(losses):.4f}  "
                  f"acc {acc:.4f}  best {best_acc:.4f}  [{time.time()-t0:.1f}s]")
    return best_acc, hist


# ── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BASELINE v3: Word-Level Next-Token Prediction")
    print("  The natural 'Block D' task — composing byte-pairs into words")
    print("=" * 60)

    embedder = L2Embedder.load_default()
    E = embedder.E  # 32
    print(f"\n  Pipeline: {embedder}")

    vocab, inv_vocab, samples_by_ctx = build_vocab_and_samples()

    # Pre-compute word → C embedding (avg-pool of byte-pair embeddings)
    print("\n  Pre-computing word → C embedding lookup...")
    word_c_vecs = {}  # word_id → (E,) float32
    for word, wid in vocab.items():
        if word == "<UNK>":
            word_c_vecs[wid] = np.zeros(E, dtype=np.float32)
            continue
        pairs = word_to_byte_pairs(word)
        if len(pairs) == 0:
            word_c_vecs[wid] = np.zeros(E, dtype=np.float32)
        else:
            pair_arr = np.array(pairs, dtype=np.int64)
            embs = embedder.embed_ids(pair_arr)  # (n_pairs, E)
            word_c_vecs[wid] = embs.mean(axis=0)

    # Pre-compute word → bytes
    word_bytes = {}
    for word, wid in vocab.items():
        if word == "<UNK>":
            word_bytes[wid] = [0] * MAX_WORD_BYTES
            continue
        bs = word_to_bytes(word)
        bs = bs[:MAX_WORD_BYTES]
        bs = bs + [0] * (MAX_WORD_BYTES - len(bs))
        word_bytes[wid] = bs

    # ── Sweep context sizes ──
    results = {}

    for ctx_len in CONTEXT_SIZES:
        print(f"\n{'='*60}")
        print(f"  CONTEXT = {ctx_len} words")
        print(f"{'='*60}")

        contexts, targets = samples_by_ctx[ctx_len]
        n = min(len(contexts), N_SAMPLES)
        contexts = contexts[:n]
        targets = targets[:n]

        # Build tensors
        # Pipeline: (N, ctx_len, E) float
        X_pipe = np.zeros((n, ctx_len, E), dtype=np.float32)
        # Vanilla: (N, ctx_len, MAX_WORD_BYTES) long
        X_bytes = np.zeros((n, ctx_len, MAX_WORD_BYTES), dtype=np.int64)
        # Direct: (N, ctx_len) long
        X_word = np.zeros((n, ctx_len), dtype=np.int64)
        Y = np.array(targets[:n], dtype=np.int64)

        for i in range(n):
            for j, wid in enumerate(contexts[i]):
                X_pipe[i, j] = word_c_vecs[wid]
                X_bytes[i, j] = word_bytes[wid]
                X_word[i, j] = wid

        # Split
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(n)
        split = int(0.8 * n)
        tr, te = idx[:split], idx[split:]

        # Size models for same budget
        pipe_model = PipelineMLP(E, ctx_len, HIDDEN, VOCAB_SIZE)
        pipe_params = sum(p.numel() for p in pipe_model.parameters())
        total_budget = pipe_params + PIPELINE_TOTAL

        # Direct word: Embedding(V, E_w) + Linear(ctx*E_w, H) + Linear(H, V)
        # = V*E_w + ctx*E_w*H + H + H*V + V
        fixed_direct = HIDDEN + HIDDEN * VOCAB_SIZE + VOCAB_SIZE
        direct_factor = VOCAB_SIZE + ctx_len * HIDDEN
        E_direct = max(1, (total_budget - fixed_direct) // direct_factor)

        # Vanilla byte: Embedding(256, E_b) + Linear(ctx*E_b, H) + Linear(H, V)
        # word_dim = E_b (avg pool doesn't change dim)
        fixed_vanilla = HIDDEN + HIDDEN * VOCAB_SIZE + VOCAB_SIZE
        vanilla_factor = 256 + ctx_len * HIDDEN  # emb + fc1 scaling
        # But vanilla also has max_word_bytes dimension issue
        # Actually: fc1 input = ctx_len * emb_dim (after avg pool)
        E_vanilla = max(1, (total_budget - fixed_vanilla) // vanilla_factor)

        print(f"  Total budget:    {total_budget:>9,}")
        print(f"  Pipeline head:   {pipe_params:>9,} + {PIPELINE_TOTAL:,} frozen")
        print(f"  Vanilla E_byte:  {E_vanilla}")
        print(f"  Direct E_word:   {E_direct}")

        # Build models
        vanilla = VanillaByteMLP(E_vanilla, ctx_len, E_vanilla, HIDDEN, VOCAB_SIZE)
        direct  = DirectWordMLP(VOCAB_SIZE, E_direct, ctx_len, HIDDEN, VOCAB_SIZE)

        # Convert to tensors
        X_pipe_t  = torch.tensor(X_pipe, dtype=torch.float32)
        X_bytes_t = torch.tensor(X_bytes, dtype=torch.long)
        X_word_t  = torch.tensor(X_word, dtype=torch.long)
        Y_t       = torch.tensor(Y, dtype=torch.long)

        # Train all three
        v_best, v_hist = train_eval(
            vanilla, X_bytes_t[tr], Y_t[tr], X_bytes_t[te], Y_t[te],
            "Vanilla Byte MLP")

        p_best, p_hist = train_eval(
            pipe_model, X_pipe_t[tr], Y_t[tr], X_pipe_t[te], Y_t[te],
            "Pipeline MLP (ABC frozen)")

        d_best, d_hist = train_eval(
            direct, X_word_t[tr], Y_t[tr], X_word_t[te], Y_t[te],
            "Direct Word Embedding")

        results[ctx_len] = {
            "vanilla": v_best,
            "pipeline": p_best,
            "direct": d_best,
            "v_hist": v_hist,
            "p_hist": p_hist,
            "d_hist": d_hist,
        }

        print(f"\n  ctx={ctx_len}:  Vanilla {v_best:.2%}  "
              f"Pipeline {p_best:.2%}  Direct {d_best:.2%}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  CONTEXT SWEEP RESULTS — Next-Word Prediction (V={VOCAB_SIZE})")
    print(f"{'='*60}")
    print(f"  Random baseline: {1/VOCAB_SIZE:.2%}\n")
    print(f"  {'Ctx':>4s}  {'Vanilla':>8s}  {'Pipeline':>8s}  {'Direct':>8s}  {'Pipe-Van':>9s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}")
    for ctx_len in CONTEXT_SIZES:
        r = results[ctx_len]
        delta = r["pipeline"] - r["vanilla"]
        print(f"  {ctx_len:4d}  {r['vanilla']:>7.2%}  {r['pipeline']:>7.2%}  "
              f"{r['direct']:>7.2%}  {delta:>+8.2%}")

    print(f"\n  If Pipeline ≈ Direct: C's byte-pair embeddings compose into")
    print(f"  word representations as well as a dedicated word embedding table.")
    print(f"  If Pipeline >> Vanilla: the 66KB pipeline earns its budget at word level.")

    # Save
    out = REPO / "output" / "baseline_v3_word_prediction.txt"
    with open(out, "w") as f:
        f.write("ctx\tvanilla\tpipeline\tdirect\n")
        for ctx_len in CONTEXT_SIZES:
            r = results[ctx_len]
            f.write(f"{ctx_len}\t{r['vanilla']:.4f}\t{r['pipeline']:.4f}\t{r['direct']:.4f}\n")
    print(f"  Results: {out}")


if __name__ == "__main__":
    main()
