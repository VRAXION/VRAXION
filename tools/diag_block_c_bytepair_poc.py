"""Block B -> Block C proof-of-concept: train Block C over BYTE-PAIR IDs.

Hypothesis: the current Block C consumes word-level lexical tokens. A
truly canonical pipeline would run on top of the frozen Block A (byte
unit) + Block B (byte-pair merger). Block B is a lossless identity
autoencoder over 65,536 byte pairs, so its output is information-
equivalent to the byte-pair index itself. We therefore use raw byte-pair
IDs as the token stream and ask: does a small Block C learn usable
structure over this representation?

Input:   raw bytes (fineweb_edu_10mb.txt or similar small corpus)
Pairs:   bytes grouped 2-at-a-time -> id in [0, 65535]
Vocab:   65,536 (capped; real distinct pairs ~ 25-30K in English)
Arch:    nn.Embedding(65536, E) + context*E flat -> W1 -> beukers_gate -> W_out
Loss:    full softmax CE over 65,536 next-pair prediction
Sanity:  nearest-neighbor probe over common pairs (' t', 'th', 'he',
         'in', 'ng', 'ed', 'ly' etc.)

Run:
    python3 tools/diag_block_c_bytepair_poc.py \\
        --corpus output/data/fineweb_edu_10mb_raw.bin \\
        --e 32 --hidden 128 --context 16 --epochs 4 --seeds 1 \\
        --out output/block_c_bytepair_poc --device cpu
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent

VOCAB_SIZE = 65536     # all possible byte pairs
BATCH = 512
MOMENTUM = 0.9


class BlockCBytePair(nn.Module):
    """emb(ctx) -> flat -> W1 -> beukers_gate -> W_out -> softmax."""

    def __init__(self, vocab_size: int, E: int, H: int, context: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.E = E; self.H = H; self.context = context
        self.emb = nn.Embedding(vocab_size, E)
        self.W1 = nn.Linear(context * E, 2 * H)
        self.W_out = nn.Linear(H, vocab_size, bias=False)
        with torch.no_grad():
            self.emb.weight.normal_(0.0, (1.0 / E) ** 0.5)
            self.W1.weight.normal_(0.0, (2.0 / (context * E)) ** 0.5)
            self.W1.bias.zero_()
            self.W_out.weight.normal_(0.0, (2.0 / H) ** 0.5)

    def forward(self, ctx_ids: torch.Tensor) -> torch.Tensor:
        B = ctx_ids.shape[0]
        x = self.emb(ctx_ids).reshape(B, self.context * self.E)
        z = self.W1(x)
        x_half, y_half = z.chunk(2, dim=-1)
        p = x_half * y_half
        a = p / (1.0 + p.abs())
        return self.W_out(a)


def bytes_to_pair_ids(raw: bytes) -> np.ndarray:
    """Group bytes 2-at-a-time, encode as uint16 ID ((hi << 8) | lo)."""
    n = len(raw) // 2
    arr = np.frombuffer(raw[: n * 2], dtype=np.uint8).reshape(n, 2)
    ids = (arr[:, 0].astype(np.int64) << 8) | arr[:, 1].astype(np.int64)
    return ids


def pair_id_to_label(pid: int) -> str:
    hi = (pid >> 8) & 0xFF
    lo = pid & 0xFF
    def ch(b):
        if 32 <= b < 127:
            c = chr(b)
            if c == "\\":
                return "\\\\"
            return c
        if b == 0x20:
            return "\\s"
        if b == 0x0a:
            return "\\n"
        if b == 0x09:
            return "\\t"
        return f"\\x{b:02x}"
    return f"'{ch(hi)}{ch(lo)}'"


def make_windows(ids: np.ndarray, context: int):
    ids = np.asarray(ids, dtype=np.int64)
    N = len(ids) - context
    ctx = np.empty((N, context), dtype=np.int64)
    for i in range(context):
        ctx[:, i] = ids[i : i + N]
    return ctx, ids[context : context + N]


def top_k_neighbors(emb, aid, k=5):
    v = emb[aid]
    d2 = np.sum((emb - v) ** 2, axis=1)
    d2[aid] = np.inf
    idx = np.argpartition(d2, k)[:k]
    return idx[np.argsort(d2[idx])]


def train(seed, E, H, context, Xtr, ytr, Xte, yte, epochs, device,
          out_seed_dir: Path | None = None, commit_hook: Path | None = None,
          lr: float = 0.1, warm_emb: Path | None = None,
          lr_decay: bool = False):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = BlockCBytePair(VOCAB_SIZE, E, H, context).to(device)
    if warm_emb is not None:
        emb_init = np.load(warm_emb).astype(np.float32)
        if emb_init.shape != (VOCAB_SIZE, E):
            raise ValueError(f"warm_emb shape {emb_init.shape} != ({VOCAB_SIZE}, {E})")
        with torch.no_grad():
            model.emb.weight.copy_(torch.from_numpy(emb_init))
        print(f"    warm-started emb from {warm_emb}", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)
    sched = None
    if lr_decay:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
    Xte_t = torch.from_numpy(Xte); yte_t = torch.from_numpy(yte)
    Ntr, Nte = len(ytr), len(yte)
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(Ntr)
        losses = []
        t0 = time.time()
        for i in range(0, Ntr, BATCH):
            idx = perm[i : i + BATCH]
            if len(idx) < 2: continue
            xb = Xtr_t[idx].to(device); yb = ytr_t[idx].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
        dt = time.time() - t0

        model.eval()
        with torch.no_grad():
            K = min(Nte, 4096)
            ix = rng.permutation(Nte)[:K]
            xb = Xte_t[ix].to(device); yb = yte_t[ix].to(device)
            logits = model(xb)
            test_loss = F.cross_entropy(logits, yb).item()
            acc = (logits.argmax(dim=-1) == yb).float().mean().item() * 100.0
        tr_loss = float(np.mean(losses))
        lr_now = opt.param_groups[0]["lr"]
        curve.append({"epoch": epoch, "train_ce": tr_loss, "test_ce": test_loss,
                      "acc1": acc, "lr": lr_now, "sec": dt})
        print(f"  seed={seed}  ep{epoch}  lr={lr_now:.4f}  train_ce={tr_loss:.4f}  "
              f"test_ce={test_loss:.4f}  acc1={acc:5.2f}%  t={dt:.1f}s", flush=True)

        if sched is not None:
            sched.step()

        # Per-epoch FULL state snapshot to volume (live pollable + resumable)
        if out_seed_dir is not None:
            out_seed_dir.mkdir(parents=True, exist_ok=True)
            emb_now = model.emb.weight.detach().cpu().numpy().astype(np.float32)
            np.save(out_seed_dir / f"emb_E{E}_epoch{epoch:02d}.npy", emb_now)
            np.save(out_seed_dir / f"W_out_E{E}_epoch{epoch:02d}.npy",
                    model.W_out.weight.detach().cpu().numpy().astype(np.float32))
            np.save(out_seed_dir / f"W1_weight_E{E}_epoch{epoch:02d}.npy",
                    model.W1.weight.detach().cpu().numpy().astype(np.float32))
            np.save(out_seed_dir / f"W1_bias_E{E}_epoch{epoch:02d}.npy",
                    model.W1.bias.detach().cpu().numpy().astype(np.float32))
            (out_seed_dir / "progress.json").write_text(json.dumps({
                "seed": seed, "E": E, "H": H, "context": context,
                "epochs_done": epoch, "curve": curve,
            }, indent=2))
            if commit_hook is not None:
                try: commit_hook.write_text(str(epoch))
                except Exception: pass
    return model, curve


def sanity_print(emb_np: np.ndarray, label: str, anchor_pairs: list[str]):
    print(f"\n== {label}  emb shape={emb_np.shape} ==")
    for pair_s in anchor_pairs:
        # pair_s is a 2-char string like " t" or "th"
        if len(pair_s) != 2: continue
        pid = (ord(pair_s[0]) << 8) | ord(pair_s[1])
        if pid >= VOCAB_SIZE: continue
        nbrs = top_k_neighbors(emb_np, pid, 5)
        neigh = ", ".join(pair_id_to_label(int(n)) for n in nbrs)
        print(f"  {pair_id_to_label(pid):>8}  ->  {neigh}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True,
                    help="Raw byte corpus (text file, no tokenization)")
    ap.add_argument("--max-bytes", type=int, default=10_000_000)
    ap.add_argument("--e", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seeds", type=str, default="1")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--commit-hook", type=Path, default=None)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--warm-emb-template", type=str, default="",
                    help="Path template with {seed} placeholder, e.g. "
                         "'output/bytepair_100mb_pull/seed{seed}/seed_{seed}/"
                         "emb_E32_epoch03.npy'. Loaded before training.")
    ap.add_argument("--lr-decay", action="store_true",
                    help="Use cosine LR decay over --epochs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else "cpu" if args.device in ("auto", "cpu") else args.device)
    print(f"Device: {device}  E={args.e} H={args.hidden} context={args.context} epochs={args.epochs}")

    raw = args.corpus.read_bytes()[: args.max_bytes]
    print(f"Raw bytes: {len(raw):,}")
    ids = bytes_to_pair_ids(raw)
    print(f"Byte pairs: {len(ids):,}  distinct: {len(np.unique(ids)):,}")

    n_test = int(len(ids) * 0.1)
    ids_tr = ids[:-n_test]; ids_te = ids[-n_test:]
    Xtr, ytr = make_windows(ids_tr, args.context)
    Xte, yte = make_windows(ids_te, args.context)
    print(f"Train windows: {len(ytr):,}  test windows: {len(yte):,}")

    anchors = [
        " t", " a", " i", " o", " s", " w",       # space-prefixed starts
        "th", "he", "in", "er", "an", "on",       # common bigrams
        "ng", "ed", "ly", "ti", "es",             # morphological endings
        ". ", ", ",                                # punct+space
        "\n\n",                                    # paragraph break
    ]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        out_seed = args.out / f"seed_{seed}"
        warm_emb = None
        if args.warm_emb_template:
            warm_emb = Path(args.warm_emb_template.format(seed=seed))
            if not warm_emb.exists():
                print(f"    WARN: warm_emb {warm_emb} missing; using random init")
                warm_emb = None
        model, curve = train(seed, args.e, args.hidden, args.context,
                             Xtr, ytr, Xte, yte, args.epochs, device,
                             out_seed_dir=out_seed,
                             commit_hook=args.commit_hook,
                             lr=args.lr,
                             warm_emb=warm_emb,
                             lr_decay=args.lr_decay)
        emb_np = model.emb.weight.detach().cpu().numpy().astype(np.float32)
        np.save(out_seed / f"emb_E{args.e}_final.npy", emb_np)
        (out_seed / "curve.json").write_text(json.dumps(curve, indent=2))
        sanity_print(emb_np, f"seed={seed} FINAL", anchors)


if __name__ == "__main__":
    main()
