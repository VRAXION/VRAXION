"""Block C v4 — next-token prediction embedder (word2vec-style).

Architecture (B variant: separators-as-tokens):
    prev K tokens -> embedding_table[ids]  (K, E)  ← LOOKUP, trivially unique
                  -> flatten (K*E)
                  -> W1 (K*E, 2H) -> beukers_gate -> H
                  -> W_out (H, vocab)  -> softmax next-token
    loss: CE over next token
    deploy artifact: the trained embedding_table (vocab, E)

Uses the full champion vocab (BYTE + PUNCT + WS_RUN + LEARNED = 32,294 ids).
Separators are tokens just like words — the model naturally learns they mean
word/sentence boundaries because of their context co-occurrence.

Metrics logged each epoch:
  train_ce / test_ce — language-model loss
  acc_pct            — next-token top-1 accuracy (low target, language is hard)
  unique_pct         — distinct embedding rows
  min_pair           — smallest pairwise L2 in embedding_table
  int4 versions of all of the above

Output ranking criterion:
  best E such that float unique_pct == 100 AND int4 unique_pct == 100.

Run:
    python3 tools/diag_block_c_next_token_embedder.py \
        --corpus output/data/fineweb_edu_100mb.txt \
        --max-bytes 2000000 \
        --context 8 \
        --e-grid 16,24,32,48,64 \
        --hidden 128 \
        --epochs 8
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
# Import the canonical champion tokenizer (handles greedy DP + prefix/normal maps)
from diag_subword_tokenizer_exact import LexicalTokenizer  # noqa: E402

BATCH = 512
LR = 0.1
MOMENTUM = 0.9
SEED = 1337
PATIENCE = 2

VOCAB_PATH = REPO_ROOT / "output" / "word_tokenizer_champion" / "champion_vocab.json"
DEFAULT_OUT = REPO_ROOT / "output" / "block_c_next_token_embedder"


# -------- activation (beukers_gate, our v3 winner) --------

def act_beukers_gate(z):
    H_out = z.shape[-1] // 2
    x = z[..., :H_out]; y = z[..., H_out:]
    p = x * y; denom = 1.0 + np.abs(p)
    a = p / denom
    def grad(g):
        gp = g * (1.0 / (denom * denom))
        return np.concatenate([gp * y, gp * x], axis=-1)
    return a, grad


# -------- tokenizer loader (from champion vocab JSON) --------

def build_champion_tokenizer(vocab_json_path: Path) -> LexicalTokenizer:
    data = json.loads(vocab_json_path.read_text())
    learned_tokens: list[tuple[bytes, bool]] = []
    for entry in data:
        if entry.get("kind") != "LEARNED":
            continue
        b = bytes.fromhex(entry["bytes_hex"])
        learned_tokens.append((b, bool(entry.get("space_prefix"))))
    tok = LexicalTokenizer(learned_tokens=learned_tokens)
    # LexicalTokenizer is a @dataclass so __post_init__ already fired.
    return tok


# -------- per-channel int4 PTQ (v3 recipe) --------

def ptq_weight_per_channel(W, bits=4):
    qmax = float(2 ** (bits - 1) - 1)
    amax = np.max(np.abs(W), axis=0)
    safe = np.where(amax == 0, np.float32(1.0), amax)
    scale = safe / qmax
    q = np.round(W / scale).clip(-qmax, qmax)
    return (q * scale).astype(np.float32)


# -------- metrics --------

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
    min_pair = float(np.sqrt(max(d2.min(), 0.0)))
    return unique_pct, min_pair


# -------- softmax CE (chunked-safe; we still build (B, vocab) but only B=512) --------

def softmax_ce(logits, y):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m); p = e / e.sum(axis=1, keepdims=True)
    ll = -np.log(p[np.arange(len(y)), y] + 1e-12)
    return p, float(ll.mean())


# -------- training --------

def train_one(E, H, context, ids_tr, ids_te, vocab_size, epochs):
    rng = np.random.default_rng(SEED)
    # Small random init so the logit scale stays sane with vocab=32k.
    emb = rng.normal(0.0, np.sqrt(1.0 / E), size=(vocab_size, E)).astype(np.float32)
    in_dim = context * E
    pre_dim = 2 * H  # beukers_gate halves
    W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, pre_dim)).astype(np.float32)
    b1 = np.zeros(pre_dim, dtype=np.float32)
    W_out = rng.normal(0.0, np.sqrt(2.0 / H), size=(H, vocab_size)).astype(np.float32)
    b_out = np.zeros(vocab_size, dtype=np.float32)

    # Embedding updates go direct (no momentum) inline in the batch loop —
    # older code accumulated into a buffer and only applied once per epoch,
    # which effectively froze the embedding during an epoch and produced
    # un-learned random-init geometry at sanity check time. Direct inline
    # updates ensure each batch's lookups see the previous batch's learning.
    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vWo = np.zeros_like(W_out); vbo = np.zeros_like(b_out)
    EMB_LR = LR

    def make_windows(ids):
        # Build (N, context) input windows + (N,) targets
        ids = np.asarray(ids, dtype=np.int64)
        N = len(ids) - context
        if N <= 0:
            return np.empty((0, context), dtype=np.int64), np.empty((0,), dtype=np.int64)
        ctx = np.empty((N, context), dtype=np.int64)
        for i in range(context):
            ctx[:, i] = ids[i : i + N]
        y = ids[context : context + N]
        return ctx, y

    Xtr, ytr = make_windows(ids_tr)
    Xte, yte = make_windows(ids_te)
    Ntr = len(ytr); Nte = len(yte)

    best_ce = float("inf"); best_state = None; bad = 0
    curve = []
    # Per-epoch checkpoint save so a Deck suspend can't obliterate an hour of work.
    ckpt_dir = getattr(train_one, "_ckpt_dir", None)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(Ntr)
        losses = []
        for i in range(0, Ntr, BATCH):
            idx = perm[i : i + BATCH]
            xb_ids = Xtr[idx]            # (B, context) int64
            yb = ytr[idx]; B = len(yb)
            # Embed
            xb = emb[xb_ids].reshape(B, in_dim)
            # Encoder
            z = xb @ W1 + b1
            a, grad_act = act_beukers_gate(z)
            logits = a @ W_out + b_out
            p, loss = softmax_ce(logits, yb); losses.append(loss)
            # Backward
            dlogits = p.copy(); dlogits[np.arange(B), yb] -= 1.0; dlogits /= B
            dW_out = a.T @ dlogits
            db_out = dlogits.sum(axis=0)
            da = dlogits @ W_out.T
            dz = grad_act(da)
            dW1 = xb.T @ dz
            db1 = dz.sum(axis=0)
            dxb = dz @ W1.T              # (B, in_dim)
            # Scatter embedding gradient INLINE (direct update, no momentum).
            # np.add.at handles duplicate indices correctly when the same token
            # appears more than once in the batch — it sums the gradients.
            dxb_r = dxb.reshape(B, context, E)
            np.add.at(emb, xb_ids, -EMB_LR * dxb_r)

            vW1 = MOMENTUM * vW1 - LR * dW1; W1 += vW1
            vb1 = MOMENTUM * vb1 - LR * db1; b1 += vb1
            vWo = MOMENTUM * vWo - LR * dW_out; W_out += vWo
            vbo = MOMENTUM * vbo - LR * db_out; b_out += vbo
        # --- epoch eval (test set chunked to keep memory sane)
        EVAL_CHUNK = 512
        preds = np.empty(Nte, dtype=np.int64)
        ce_sum = 0.0
        for s in range(0, Nte, EVAL_CHUNK):
            e = min(s + EVAL_CHUNK, Nte)
            xb_ids = Xte[s:e]
            xb = emb[xb_ids].reshape(e - s, in_dim)
            z = xb @ W1 + b1
            a, _ = act_beukers_gate(z)
            logits = a @ W_out + b_out
            m = logits.max(axis=1, keepdims=True)
            ex = np.exp(logits - m)
            pr = ex / ex.sum(axis=1, keepdims=True)
            yb = yte[s:e]
            ce_sum += float(-np.log(pr[np.arange(e - s), yb] + 1e-12).sum())
            preds[s:e] = logits.argmax(axis=1)
        test_ce = ce_sum / Nte
        acc = float((preds == yte).mean()) * 100.0
        uniq, min_pair = latent_diagnostics(emb)
        train_ce = float(np.mean(losses))
        curve.append({"epoch": epoch, "train_ce": train_ce, "test_ce": test_ce,
                      "acc_pct": acc, "unique_pct": uniq, "min_pair": min_pair})
        print(f"    ep {epoch:2d}  train_ce={train_ce:.4f}  test_ce={test_ce:.4f}  "
              f"acc={acc:6.2f}%  uniq={uniq:6.2f}%  pair={min_pair:.4f}",
              flush=True)
        if test_ce < best_ce - 1e-4:
            best_ce = test_ce
            best_state = (emb.copy(), W1.copy(), b1.copy(), W_out.copy(), b_out.copy())
            bad = 0
        else:
            bad += 1
            if bad > PATIENCE:
                break
        # Per-epoch safety checkpoint. Writes the CURRENT embedding (not best)
        # so we never lose more than one epoch of work to a suspend/crash.
        if ckpt_dir is not None:
            ckpt_path = ckpt_dir / f"embedding_table_E{E}_epoch{epoch:02d}.npy"
            np.save(ckpt_path, emb)

    emb, W1, b1, W_out, b_out = best_state

    # Float emb diagnostics
    uniq_f, pair_f = latent_diagnostics(emb)
    # Int4 PTQ (per-channel along the E axis of the embedding table)
    emb_q = ptq_weight_per_channel(emb)
    uniq_q, pair_q = latent_diagnostics(emb_q)

    return {
        "E": E, "H": H, "context": context,
        "param_count": int(emb.size + W1.size + b1.size + W_out.size + b_out.size),
        "epochs_run": len(curve), "curve": curve,
        "final_test_ce": best_ce,
        "float": {"unique_pct": uniq_f, "min_pair": pair_f},
        "int4": {"unique_pct": uniq_q, "min_pair": pair_q},
        "embedding_shape": list(emb.shape),
    }, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--max-bytes", type=int, default=2_000_000)
    ap.add_argument("--context", type=int, default=8,
                    help="Number of previous tokens used as input.")
    ap.add_argument("--e-grid", type=str, default="16,24,32,48,64",
                    help="Embedding dim E grid.")
    ap.add_argument("--hidden", type=int, default=128, help="Hidden width H.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--vocab", type=Path, default=VOCAB_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    e_grid = [int(x) for x in args.e_grid.split(",")]

    print(f"== Block C v4 next-token embedder  (beukers_gate, H={args.hidden}) ==")
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
          f"(compression = {len(raw)/len(ids):.2f} bytes/token)",
          flush=True)

    # Train/test split at time-ordered boundary
    n_test = int(len(ids) * args.test_frac)
    ids_tr = ids[: len(ids) - n_test]
    ids_te = ids[len(ids) - n_test :]
    print(f"Train tokens: {len(ids_tr):,}  test tokens: {len(ids_te):,}")

    results = []
    best_E_pass = None
    t_start = time.time()
    # Let train_one know where to drop per-epoch safety checkpoints.
    train_one._ckpt_dir = args.out
    for E in e_grid:
        print(f"\n-- E={E}  H={args.hidden}  context={args.context} --")
        r, emb = train_one(E, args.hidden, args.context, ids_tr, ids_te,
                           vocab_size, args.epochs)
        results.append(r)
        f = r["float"]; q = r["int4"]
        print(f"  E={E}  float: uniq={f['unique_pct']:6.2f}%  "
              f"pair={f['min_pair']:.4f}")
        print(f"         int4 : uniq={q['unique_pct']:6.2f}%  "
              f"pair={q['min_pair']:.4f}  "
              f"params={r['param_count']:,}  ce={r['final_test_ce']:.4f}")
        if (best_E_pass is None
                and f["unique_pct"] >= 100.0 - 1e-9
                and q["unique_pct"] >= 100.0 - 1e-9):
            best_E_pass = E
            # Save the embedding table as the deploy artifact
            art_path = args.out / f"embedding_table_E{E}.npy"
            np.save(art_path, emb)
            print(f"  ★ Saved deploy artifact: {art_path}")

    dt = time.time() - t_start
    print(f"\n== Summary ==  total {dt/60:.1f} min")
    print(f"  {'E':>4}  {'ce':>7}  {'uniq_f':>7}  {'pair_f':>8}  "
          f"{'uniq_q':>7}  {'pair_q':>8}  params")
    for r in results:
        flag = "  ★" if r["E"] == best_E_pass else ""
        f = r["float"]; q = r["int4"]
        print(f"  {r['E']:4d}  {r['final_test_ce']:7.4f}  "
              f"{f['unique_pct']:7.2f}  {f['min_pair']:8.4f}  "
              f"{q['unique_pct']:7.2f}  {q['min_pair']:8.4f}  "
              f"{r['param_count']:,}{flag}")
    if best_E_pass is not None:
        print(f"\nMin E with 100% unique in float+int4 = {best_E_pass}")

    out_path = args.out / f"sweep_context{args.context}_H{args.hidden}.json"
    out_path.write_text(json.dumps({
        "context": args.context, "hidden": args.hidden, "e_grid": e_grid,
        "vocab_size": vocab_size, "epochs_budget": args.epochs,
        "corpus": str(args.corpus), "corpus_bytes": len(raw),
        "train_tokens": len(ids_tr), "test_tokens": len(ids_te),
        "best_E_pass": best_E_pass,
        "total_seconds": dt, "results": results,
    }, indent=2))
    print(f"\nSaved summary: {out_path}")


if __name__ == "__main__":
    main()
