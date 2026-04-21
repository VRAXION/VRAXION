"""Block C next-token embedder — PyTorch port with full 32K softmax CE.

Same architecture as tools/diag_block_c_next_token_negsample.py but:
  - Full 32K-way softmax CE loss (not negsample BCE) to restore the dense
    gradient signal on the embedding table. On CPU this is ~10× slower but
    BLAS-optimised; on a T4/L4 GPU the full softmax is nearly free.
  - Device-agnostic: runs on CPU or CUDA (`--device auto`).
  - Saves BOTH emb (input embedding) AND W_out (output embedding) so we can
    probe semantic structure in either table.
  - Multi-seed support via --seeds 1,2,3 runs each seed sequentially and
    aggregates a mean±std curve.
  - Accepts pre-tokenised `--tokens tokens.npy` to skip the tokenizer on
    remote runners (Modal) that don't have the LexicalTokenizer code.

Run (local):
    python3 tools/diag_block_c_torch.py \
        --tokens output/data/fineweb_edu_100mb.tokens.npy \
        --e 32 --hidden 128 --context 8 --epochs 8 --seeds 1,2,3 \
        --device auto --out output/block_c_torch_v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent

BATCH = 512
LR = 0.1
MOMENTUM = 0.9


class BlockCNextToken(nn.Module):
    """emb(ctx) -> flat -> W1 -> beukers_gate -> W_out -> softmax CE."""

    def __init__(self, vocab_size: int, E: int, H: int, context: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.E = E
        self.H = H
        self.context = context
        self.emb = nn.Embedding(vocab_size, E)
        self.W1 = nn.Linear(context * E, 2 * H)
        self.W_out = nn.Linear(H, vocab_size, bias=False)
        # Init matching the numpy reference: emb ~ N(0, sqrt(1/E)),
        # W1 ~ N(0, sqrt(2/in_dim)) [He], W_out ~ N(0, sqrt(2/H)) [He].
        with torch.no_grad():
            self.emb.weight.normal_(0.0, (1.0 / E) ** 0.5)
            self.W1.weight.normal_(0.0, (2.0 / (context * E)) ** 0.5)
            self.W1.bias.zero_()
            self.W_out.weight.normal_(0.0, (2.0 / H) ** 0.5)

    def forward(self, ctx_ids: torch.Tensor) -> torch.Tensor:
        # ctx_ids: (B, context) long
        B = ctx_ids.shape[0]
        x = self.emb(ctx_ids).reshape(B, self.context * self.E)
        z = self.W1(x)  # (B, 2H)
        x_half, y_half = z.chunk(2, dim=-1)
        p = x_half * y_half
        a = p / (1.0 + p.abs())  # beukers_gate, (B, H)
        return self.W_out(a)     # (B, vocab)


def make_windows(ids: np.ndarray, context: int):
    ids = np.asarray(ids, dtype=np.int64)
    N = len(ids) - context
    if N <= 0:
        return (np.empty((0, context), dtype=np.int64),
                np.empty((0,), dtype=np.int64))
    ctx = np.empty((N, context), dtype=np.int64)
    for i in range(context):
        ctx[:, i] = ids[i : i + N]
    return ctx, ids[context : context + N]


def ptq_per_channel(W: np.ndarray, bits: int = 4) -> np.ndarray:
    qmax = float(2 ** (bits - 1) - 1)
    amax = np.max(np.abs(W), axis=0)
    safe = np.where(amax == 0, np.float32(1.0), amax)
    scale = safe / qmax
    q = np.round(W / scale).clip(-qmax, qmax)
    return (q * scale).astype(np.float32)


def latent_diagnostics(latents: np.ndarray):
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


def train_one_seed(
    seed: int,
    E: int, H: int, context: int,
    Xtr: np.ndarray, ytr: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray,
    vocab_size: int, epochs: int,
    device: torch.device,
    out_dir: Path,
    resume_dir: Path | None = None,
    commit_hook: Path | None = None,
    patience: int = -1,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = BlockCNextToken(vocab_size, E, H, context).to(device)

    # Optional: resume from a previous best-state snapshot. We load emb,
    # W_out, W1 weights + bias. SGD momentum is NOT restored (not saved
    # during the original run); a fresh momentum buffer causes a brief
    # transient for ~10 batches which is negligible vs the training run.
    if resume_dir is not None:
        rd = Path(resume_dir)
        emb_f   = rd / f"emb_E{E}_final.npy"
        wout_f  = rd / f"W_out_E{E}_final.npy"
        w1_f    = rd / f"W1_weight_E{E}_final.npy"
        b1_f    = rd / f"W1_bias_E{E}_final.npy"
        missing = [p for p in (emb_f, wout_f, w1_f, b1_f) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"resume_dir {rd} missing: {[str(p) for p in missing]}")
        with torch.no_grad():
            model.emb.weight.copy_(torch.from_numpy(np.load(emb_f)))
            model.W_out.weight.copy_(torch.from_numpy(np.load(wout_f)))
            model.W1.weight.copy_(torch.from_numpy(np.load(w1_f)))
            model.W1.bias.copy_(torch.from_numpy(np.load(b1_f)))
        print(f"    seed={seed}  resumed from {rd}", flush=True)

    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    Xtr_t = torch.from_numpy(Xtr)  # keep on CPU, move per-batch
    ytr_t = torch.from_numpy(ytr)
    Xte_t = torch.from_numpy(Xte)
    yte_t = torch.from_numpy(yte)

    Ntr = len(ytr); Nte = len(yte)
    curve = []
    best_loss = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(Ntr)
        losses = []
        t_epoch = time.time()
        for i in range(0, Ntr, BATCH):
            idx = perm[i : i + BATCH]
            if len(idx) < 2:
                continue
            xb = Xtr_t[idx].to(device, non_blocking=True)
            yb = ytr_t[idx].to(device, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        t_epoch = time.time() - t_epoch

        # Test-set eval (sample)
        model.eval()
        with torch.no_grad():
            EVAL_N = min(Nte, 4096)
            ix = rng.permutation(Nte)[:EVAL_N]
            xb = Xte_t[ix].to(device)
            yb = yte_t[ix].to(device)
            logits = model(xb)
            test_loss = F.cross_entropy(logits, yb).item()
            # top-1 accuracy on eval sample
            acc = (logits.argmax(dim=-1) == yb).float().mean().item() * 100.0

        emb_np = model.emb.weight.detach().cpu().numpy().astype(np.float32)
        uniq, min_pair = latent_diagnostics(emb_np)
        train_loss = float(np.mean(losses))
        curve.append({
            "epoch": epoch, "train_ce": train_loss, "test_ce": test_loss,
            "acc_top1": acc, "unique_pct": uniq, "min_pair": min_pair,
            "seconds": t_epoch,
        })
        print(f"    seed={seed}  ep {epoch:2d}  train_ce={train_loss:.4f}  "
              f"test_ce={test_loss:.4f}  acc1={acc:5.2f}%  uniq={uniq:6.2f}%  "
              f"pair={min_pair:.4f}  t_ep={t_epoch:.1f}s", flush=True)

        # Per-epoch checkpoint: save FULL state so we can resume from any
        # epoch without losing W_out / W1 context (lesson from v1).
        ckpt_dir = out_dir / f"epoch{epoch:02d}_state"
        ckpt_dir.mkdir(exist_ok=True)
        np.save(ckpt_dir / f"emb_E{E}_final.npy", emb_np)
        np.save(ckpt_dir / f"W_out_E{E}_final.npy",
                model.W_out.weight.detach().cpu().numpy().astype(np.float32))
        np.save(ckpt_dir / f"W1_weight_E{E}_final.npy",
                model.W1.weight.detach().cpu().numpy().astype(np.float32))
        np.save(ckpt_dir / f"W1_bias_E{E}_final.npy",
                model.W1.bias.detach().cpu().numpy().astype(np.float32))

        # Live progress: append the current curve to a progress.json that
        # is committed to the volume via the commit_hook. External pollers
        # (including the user) can read this at any time.
        prog_path = out_dir / f"progress_seed{seed}.json"
        prog_path.write_text(json.dumps({
            "seed": seed, "E": E, "H": H, "context": context,
            "epochs_done": epoch, "epochs_budget": epochs,
            "curve": curve,
        }, indent=2))
        if commit_hook is not None:
            try:
                commit_hook.write_text(str(epoch))
            except Exception:
                pass

        if test_loss < best_loss - 1e-4:
            best_loss = test_loss
            best_state = {
                "emb": emb_np.copy(),
                "W_out": model.W_out.weight.detach().cpu().numpy().astype(np.float32),
                "W1_weight": model.W1.weight.detach().cpu().numpy().astype(np.float32),
                "W1_bias":   model.W1.bias.detach().cpu().numpy().astype(np.float32),
            }
            bad = 0
        else:
            bad += 1
            if patience >= 0 and bad > patience:
                print(f"    seed={seed}  early stop at epoch {epoch}", flush=True)
                break

    return best_state, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, default=None,
                    help="Pre-tokenised int32 npy (skip tokenizer)")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="Raw byte corpus; tokenised via LexicalTokenizer")
    ap.add_argument("--max-bytes", type=int, default=0)
    ap.add_argument("--vocab", type=Path,
                    default=REPO_ROOT / "output" / "word_tokenizer_champion" / "champion_vocab.json")
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--e", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=str, default="1",
                    help="Comma-separated seed list, e.g. '1,2,3,4,5'")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--commit-hook", type=Path, default=None,
                    help="Per-epoch, write the current epoch number to this "
                         "file. Used by the Modal wrapper to trigger a "
                         "vol.commit() so per-epoch checkpoints become "
                         "visible from outside the container mid-run.")
    ap.add_argument("--patience", type=int, default=-1,
                    help="Early stop after N non-improving epochs (on "
                         "test_ce). Negative disables early stop entirely.")
    ap.add_argument("--resume-dir", type=Path, default=None,
                    help="Directory containing emb_E<E>_final.npy, "
                         "W_out_E<E>_final.npy, W1_weight_E<E>_final.npy, "
                         "W1_bias_E<E>_final.npy from a previous run. If "
                         "multiple seeds are trained in one invocation, "
                         "all of them resume from the same checkpoint.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"== Block C torch (full softmax CE) ==")
    print(f"Device: {device}  E={args.e}  H={args.hidden}  "
          f"context={args.context}  epochs={args.epochs}", flush=True)

    # --- Load tokens ---
    if args.tokens is not None:
        print(f"Loading tokens from {args.tokens}", flush=True)
        ids = np.load(args.tokens).astype(np.int64)
        # Need vocab_size; read from champion_vocab if available
        vocab = json.loads(args.vocab.read_text())
        vocab_size = max(e["id"] for e in vocab) + 1
    elif args.corpus is not None:
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from diag_subword_tokenizer_exact import LexicalTokenizer
        vocab = json.loads(args.vocab.read_text())
        learned = []
        for e in vocab:
            if e.get("kind") != "LEARNED":
                continue
            b = bytes.fromhex(e["bytes_hex"])
            if e.get("space_prefix"):
                b = b" " + b
            learned.append((b, bool(e.get("space_prefix"))))
        tok = LexicalTokenizer(learned_tokens=learned)
        vocab_size = tok.vocab_size
        with args.corpus.open("rb") as f:
            raw = f.read(args.max_bytes) if args.max_bytes > 0 else f.read()
        print(f"Corpus bytes: {len(raw):,}  tokenising...", flush=True)
        t0 = time.time()
        ids = np.asarray(tok.encode(raw), dtype=np.int64)
        print(f"Tokenized to {len(ids):,} tokens in {time.time()-t0:.1f}s  "
              f"(compression = {len(raw)/len(ids):.2f} bytes/token)", flush=True)
    else:
        ap.error("must provide either --tokens or --corpus")

    print(f"Total tokens: {len(ids):,}  vocab_size={vocab_size}", flush=True)

    # Train/test split
    n_test = int(len(ids) * args.test_frac)
    ids_tr = ids[: len(ids) - n_test]
    ids_te = ids[len(ids) - n_test :]

    Xtr, ytr = make_windows(ids_tr, args.context)
    Xte, yte = make_windows(ids_te, args.context)
    print(f"Train windows: {len(ytr):,}  test windows: {len(yte):,}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"Seeds: {seeds}", flush=True)

    t_start = time.time()
    all_curves = {}
    finals = {}
    for seed in seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        best_state, curve = train_one_seed(
            seed, args.e, args.hidden, args.context,
            Xtr, ytr, Xte, yte, vocab_size, args.epochs, device, args.out,
            resume_dir=args.resume_dir,
            commit_hook=args.commit_hook,
            patience=args.patience,
        )
        all_curves[seed] = curve
        finals[seed] = best_state

        # Save per-seed final artifacts
        seed_out = args.out / f"seed_{seed}"
        seed_out.mkdir(exist_ok=True)
        np.save(seed_out / f"emb_E{args.e}_final.npy", best_state["emb"])
        np.save(seed_out / f"W_out_E{args.e}_final.npy", best_state["W_out"])
        np.save(seed_out / f"W1_weight_E{args.e}_final.npy", best_state["W1_weight"])
        np.save(seed_out / f"W1_bias_E{args.e}_final.npy", best_state["W1_bias"])

        emb_q = ptq_per_channel(best_state["emb"], bits=4)
        uniq_f, pair_f = latent_diagnostics(best_state["emb"])
        uniq_q, pair_q = latent_diagnostics(emb_q)
        print(f"    seed={seed}  float: uniq={uniq_f:.2f}% pair={pair_f:.4f}  "
              f"int4: uniq={uniq_q:.2f}% pair={pair_q:.4f}", flush=True)

    dt = time.time() - t_start

    # Aggregate
    print(f"\n== Summary (total {dt/60:.1f} min, seeds={len(seeds)}) ==", flush=True)
    last_test_ces = [all_curves[s][-1]["test_ce"] for s in seeds]
    last_accs     = [all_curves[s][-1]["acc_top1"] for s in seeds]
    if len(seeds) >= 2:
        print(f"  test_ce final:   mean={np.mean(last_test_ces):.4f}  "
              f"std={np.std(last_test_ces):.4f}")
        print(f"  acc@1 final:     mean={np.mean(last_accs):.2f}%  "
              f"std={np.std(last_accs):.2f}%")
    else:
        print(f"  test_ce final: {last_test_ces[0]:.4f}  acc@1: {last_accs[0]:.2f}%")

    (args.out / "training_summary.json").write_text(json.dumps({
        "E": args.e, "H": args.hidden, "context": args.context,
        "epochs_budget": args.epochs, "vocab_size": int(vocab_size),
        "train_tokens": int(len(ids_tr)), "test_tokens": int(len(ids_te)),
        "device": str(device), "seeds": seeds,
        "total_seconds": dt,
        "curves": {str(s): all_curves[s] for s in seeds},
        "last_test_ce_mean": float(np.mean(last_test_ces)),
        "last_test_ce_std":  float(np.std(last_test_ces)),
        "last_acc_mean":     float(np.mean(last_accs)),
        "last_acc_std":      float(np.std(last_accs)),
    }, indent=2))
    print(f"Saved summary: {args.out / 'training_summary.json'}")


if __name__ == "__main__":
    main()
