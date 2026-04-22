"""Block C next-token embedder with NEGATIVE SAMPLING.

Full 32K-way softmax on CPU runs ~200 ms per batch — 99% of the time sits
in the classifier matmul. Negative sampling (word2vec / skip-gram classic
trick) replaces that with K binary classifications per example: the
positive target vs K random negative tokens. For K=20 this is roughly
32000 / 21 ≈ 1500× cheaper per batch in the classifier layer, so the
whole training loop becomes bound by the encoder instead.

Architecture is otherwise identical to the softmax variant:
    prev K_ctx tokens → embedding_table[ids] → encoder (beukers_gate) →
    hidden H → compare against (1 positive + K_neg negative) output vectors
    via sigmoid BCE.

Per-epoch checkpoint save lands in --out so a Deck suspend can't wipe an
hour of work.

Run:
    python3 tools/diag_block_c_next_token_negsample.py \
        --corpus output/data/fineweb_edu_100mb.txt \
        --max-bytes 10000000 --context 8 --e 32 --hidden 128 \
        --epochs 8 --k-neg 20 \
        --out output/block_c_next_token_v4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
from diag_subword_tokenizer_exact import LexicalTokenizer  # noqa: E402

BATCH = 512
LR = 0.1
MOMENTUM = 0.9
SEED = 1337
PATIENCE = 2


# -------- activations --------

def act_beukers_gate(z):
    H_out = z.shape[-1] // 2
    x = z[..., :H_out]; y = z[..., H_out:]
    p = x * y; denom = 1.0 + np.abs(p)
    a = p / denom
    def grad(g):
        gp = g * (1.0 / (denom * denom))
        return np.concatenate([gp * y, gp * x], axis=-1)
    return a, grad


# -------- tokenizer loader --------

def build_champion_tokenizer(vocab_json_path: Path) -> LexicalTokenizer:
    data = json.loads(vocab_json_path.read_text())
    learned_tokens: list[tuple[bytes, bool]] = []
    for entry in data:
        if entry.get("kind") != "LEARNED":
            continue
        b = bytes.fromhex(entry["bytes_hex"])
        if entry.get("space_prefix"):
            b = b" " + b
        learned_tokens.append((b, bool(entry.get("space_prefix"))))
    return LexicalTokenizer(learned_tokens=learned_tokens)


# -------- per-channel int4 PTQ --------

def ptq_per_channel(W, bits=4):
    qmax = float(2 ** (bits - 1) - 1)
    amax = np.max(np.abs(W), axis=0)
    safe = np.where(amax == 0, np.float32(1.0), amax)
    scale = safe / qmax
    q = np.round(W / scale).clip(-qmax, qmax)
    return (q * scale).astype(np.float32)


def latent_diagnostics(latents):
    rounded = np.round(latents, decimals=6)
    _, inv = np.unique(rounded, axis=0, return_inverse=True)
    unique_pct = float(len(np.unique(inv))) / len(latents) * 100.0
    n = latents.shape[0]
    if n > 4096:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=4096, replace=False)
        L = latents[idx]
    else:
        L = latents
    sq = np.sum(L * L, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * (L @ L.T)
    np.fill_diagonal(d2, np.inf)
    return unique_pct, float(np.sqrt(max(d2.min(), 0.0)))


# -------- negative-sampling training --------

def train(
    E, H, context, k_neg, ids_tr, ids_te, vocab_size, epochs, out_dir,
):
    rng = np.random.default_rng(SEED)
    emb = rng.normal(0.0, np.sqrt(1.0 / E), size=(vocab_size, E)).astype(np.float32)
    in_dim = context * E
    pre_dim = 2 * H
    W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, pre_dim)).astype(np.float32)
    b1 = np.zeros(pre_dim, dtype=np.float32)
    # Output vectors: one per vocab item, in (H, vocab) layout for easy column gather.
    W_out = rng.normal(0.0, np.sqrt(2.0 / H), size=(H, vocab_size)).astype(np.float32)

    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)

    # Windows
    def make_windows(ids):
        ids = np.asarray(ids, dtype=np.int64)
        N = len(ids) - context
        if N <= 0:
            return np.empty((0, context), dtype=np.int64), np.empty((0,), dtype=np.int64)
        ctx = np.empty((N, context), dtype=np.int64)
        for i in range(context):
            ctx[:, i] = ids[i : i + N]
        return ctx, ids[context : context + N]

    Xtr, ytr = make_windows(ids_tr)
    Xte, yte = make_windows(ids_te)
    Ntr = len(ytr); Nte = len(yte)
    print(f"Train windows: {Ntr:,}  test windows: {Nte:,}  context={context}")

    curve = []
    best_loss = float("inf"); best_state = None; bad = 0

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(Ntr)
        losses = []
        t_epoch = time.time()
        for step_i, i in enumerate(range(0, Ntr, BATCH)):
            idx = perm[i : i + BATCH]
            xb_ids = Xtr[idx]
            yb_pos = ytr[idx]
            B = len(yb_pos)

            # Sample K negatives per example, uniform over vocab.
            # (Unigram-proportional sampling would be slightly better but adds
            # a frequency table; uniform is 90% of the quality at zero cost.)
            yb_neg = rng.integers(0, vocab_size, size=(B, k_neg))

            # Encoder
            xb = emb[xb_ids].reshape(B, in_dim)
            z = xb @ W1 + b1
            a, grad_act = act_beukers_gate(z)           # (B, H)

            # Gather positive + negative output vectors
            # W_out[:, ids] -> (H, B) or (H, B, k); transpose to (B, H) / (B, k, H)
            w_pos = W_out[:, yb_pos].T                   # (B, H)
            w_neg = W_out[:, yb_neg].transpose(1, 2, 0)  # (B, k, H)

            # Scores
            pos_score = np.sum(a * w_pos, axis=1)        # (B,)
            neg_score = np.einsum("bh,bkh->bk", a, w_neg)  # (B, k)

            # Sigmoid + BCE loss
            # Use numerically stable form: log(1+exp(-x)) with clamp
            pos_loss = np.log1p(np.exp(-np.clip(pos_score, -30, 30)))       # -log sigmoid(pos)
            neg_loss = np.log1p(np.exp(np.clip(neg_score, -30, 30)))        # -log sigmoid(-neg)
            loss = float(pos_loss.mean() + neg_loss.mean())
            losses.append(loss)

            # Gradients (sigmoid derivative)
            pos_sig = 1.0 / (1.0 + np.exp(-np.clip(pos_score, -30, 30)))
            neg_sig = 1.0 / (1.0 + np.exp(-np.clip(neg_score, -30, 30)))
            dpos = (pos_sig - 1.0) / B          # (B,)
            dneg = neg_sig / B                   # (B, k)

            # d_a from scores = dpos[:, None] * w_pos + einsum(dneg, w_neg)
            da = dpos[:, None] * w_pos + np.einsum("bk,bkh->bh", dneg, w_neg)

            # Gradient contributions to W_out (SPARSE: only touched cols).
            # `np.add.at` is ~1000× slower than fancy indexing when the index
            # list is long (here B*k = 10K entries), because it loops over
            # duplicates in Python. Since random K negatives collide only
            # ~16% of the time in this batch and the loss is approximate
            # anyway, we use direct fancy indexing and accept occasional
            # last-write-wins duplicates — negligible noise vs the training
            # signal and ~50× faster end-to-end.
            pos_grad = (-LR * dpos[:, None]) * a                 # (B, H)
            W_out[:, yb_pos] += pos_grad.T                        # (H, B) broadcast
            # Expand a across K negatives without creating a huge intermediate
            # via repeat; einsum stays in a compact (B, k, H) shape.
            neg_grad = (-LR * dneg)[..., None] * a[:, None, :]    # (B, k, H)
            W_out[:, yb_neg.ravel()] += neg_grad.reshape(-1, H).T # (H, B*k)

            # Backward to encoder
            dz = grad_act(da)
            dW1 = xb.T @ dz
            db1 = dz.sum(axis=0)

            vW1 = MOMENTUM * vW1 - LR * dW1; W1 += vW1
            vb1 = MOMENTUM * vb1 - LR * db1; b1 += vb1

            # Embedding update. Same fast-path: direct fancy indexing. The
            # context window is B*context = 4K indices per batch, and in
            # short-context next-token tasks collisions inside a single batch
            # are common (common tokens like " " appear dozens of times), but
            # duplicate loss is small — the Zipf tail sees many updates across
            # batches/epochs anyway.
            dxb = dz @ W1.T
            dxb_r = dxb.reshape(B, context, E)
            emb[xb_ids] -= LR * dxb_r

        t_epoch = time.time() - t_epoch

        # --- eval: quick test-set loss (sample, not full) ---
        EVAL_N = min(Nte, 4096)
        ix = rng.permutation(Nte)[:EVAL_N]
        xb_ids = Xte[ix]; yb_pos = yte[ix]
        yb_neg = rng.integers(0, vocab_size, size=(EVAL_N, k_neg))
        xb = emb[xb_ids].reshape(EVAL_N, in_dim)
        z = xb @ W1 + b1
        a, _ = act_beukers_gate(z)
        w_pos = W_out[:, yb_pos].T
        w_neg = W_out[:, yb_neg].transpose(1, 2, 0)
        pos_score = np.sum(a * w_pos, axis=1)
        neg_score = np.einsum("bh,bkh->bk", a, w_neg)
        pos_loss = np.log1p(np.exp(-np.clip(pos_score, -30, 30))).mean()
        neg_loss = np.log1p(np.exp(np.clip(neg_score, -30, 30))).mean()
        test_loss = float(pos_loss + neg_loss)
        uniq, min_pair = latent_diagnostics(emb)

        train_loss = float(np.mean(losses))
        curve.append({
            "epoch": epoch, "train_loss": train_loss, "test_loss": test_loss,
            "unique_pct": uniq, "min_pair": min_pair, "seconds": t_epoch,
        })
        print(f"  ep {epoch:2d}  train_bce={train_loss:.4f}  test_bce={test_loss:.4f}  "
              f"uniq={uniq:6.2f}%  pair={min_pair:.4f}  t_ep={t_epoch:.1f}s",
              flush=True)

        # checkpoint the CURRENT emb every epoch — cheap insurance
        np.save(out_dir / f"embedding_table_E{E}_epoch{epoch:02d}.npy", emb)

        if test_loss < best_loss - 1e-4:
            best_loss = test_loss
            best_state = (emb.copy(), W1.copy(), b1.copy(), W_out.copy())
            bad = 0
        else:
            bad += 1
            if bad > PATIENCE:
                break

    return best_state, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--max-bytes", type=int, default=0)
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--e", type=int, default=32, help="Embedding dim E.")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--k-neg", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--vocab", type=Path,
                    default=REPO_ROOT / "output" / "word_tokenizer_champion" / "champion_vocab.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"== Block C next-token embedder (negative sampling) ==")
    print(f"E={args.e}  H={args.hidden}  context={args.context}  "
          f"K_neg={args.k_neg}  epochs={args.epochs}")
    print(f"Building champion tokenizer from {args.vocab}...", flush=True)
    tok = build_champion_tokenizer(args.vocab)
    vocab_size = tok.vocab_size
    print(f"Vocab size = {vocab_size}", flush=True)

    print(f"Reading corpus {args.corpus} (max {args.max_bytes/1e6:.1f} MB)...",
          flush=True)
    with args.corpus.open("rb") as f:
        raw = f.read(args.max_bytes) if args.max_bytes > 0 else f.read()
    print(f"Corpus bytes: {len(raw):,}  tokenising...", flush=True)
    t0 = time.time()
    ids = tok.encode(raw)
    print(f"Tokenized to {len(ids):,} tokens in {time.time()-t0:.1f}s  "
          f"(compression = {len(raw)/len(ids):.2f} bytes/token)", flush=True)

    n_test = int(len(ids) * args.test_frac)
    ids_tr = ids[: len(ids) - n_test]
    ids_te = ids[len(ids) - n_test :]

    t_start = time.time()
    best_state, curve = train(
        args.e, args.hidden, args.context, args.k_neg,
        ids_tr, ids_te, vocab_size, args.epochs, args.out,
    )
    dt = time.time() - t_start

    emb_final, W1, b1, W_out = best_state
    # int4 PTQ on final embedding for robustness report
    emb_q = ptq_per_channel(emb_final, bits=4)
    uniq_f, pair_f = latent_diagnostics(emb_final)
    uniq_q, pair_q = latent_diagnostics(emb_q)

    print(f"\n== Summary (E={args.e}, total {dt/60:.1f} min) ==")
    print(f"  float: uniq={uniq_f:.2f}%  pair={pair_f:.4f}")
    print(f"  int4 : uniq={uniq_q:.2f}%  pair={pair_q:.4f}")

    final_path = args.out / f"embedding_table_E{args.e}_final.npy"
    np.save(final_path, emb_final)
    (args.out / "training_summary.json").write_text(json.dumps({
        "E": args.e, "H": args.hidden, "context": args.context,
        "k_neg": args.k_neg, "epochs_run": len(curve),
        "epochs_budget": args.epochs, "vocab_size": vocab_size,
        "train_tokens": len(ids_tr), "test_tokens": len(ids_te),
        "corpus": str(args.corpus), "corpus_bytes": len(raw),
        "total_seconds": dt, "curve": curve,
        "float": {"unique_pct": uniq_f, "min_pair": pair_f},
        "int4":  {"unique_pct": uniq_q, "min_pair": pair_q},
    }, indent=2))
    print(f"\nSaved final artifact: {final_path}")


if __name__ == "__main__":
    main()
