"""GPU quantization sweep on BIG network (nf=1024).

Mirrors the Rust CPU sweep architecture:
  - Beukers gate: co = ab / (1 + |ab|), a and b are two parallel projections
  - Char-LM mask-center task on 27-class alphabet (a-z + whitespace)
  - Staged INQ quantization (10 rounds × 20 ep, easiest-to-grid first)

GPU advantage: nf=1024 (8x bigger than CPU peak) with batched sample processing
is where GPU matmul is worth it. CPU (rayon) would be ~10x slower.

Runs: 2 tasks x 4 modes (float, int4, ternary, binary) = 8 configs.

Run: python tools/diag_quant_sweep_gpu.py <fineweb_path> <code_path>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Config matching Rust version ---
VOCAB = 2000
DIM = 16
CTX = 32
MASK_POS = CTX // 2
K = 7
HK = 3
N_PROJ = 2
FAN = K * DIM  # 112
N_CLASSES = 27  # a-z + whitespace

ROUNDS = 10
EPOCHS_PER_ROUND = 20
MAX_EP_INIT = 200
PATIENCE = 30
LOG_EVERY = 10

NF_GPU = 1024  # GPU size — 8x bigger than CPU peak (128)
BATCH_SIZE = 4096  # big batch for GPU throughput
SAMPLES_PER_EP = 16384  # 8x bigger than CPU (2000) since GPU is cheap
EVAL_SAMPLES = 2000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Data loading (matching Rust load_corpus) ---
def load_corpus(path: str) -> torch.Tensor:
    raw = Path(path).read_bytes()
    out = bytearray()
    for b in raw:
        if 97 <= b <= 122:        # a-z
            out.append(b - 97)
        elif 65 <= b <= 90:        # A-Z
            out.append(b - 65)
        elif b in (32, 10, 9, 13): # whitespace
            out.append(26)
    return torch.tensor(list(out), dtype=torch.long)


# --- Quantization primitives ---
INT4_LEVELS = 7.0

def q_float(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x

def q_int4(x: torch.Tensor, scale: float) -> torch.Tensor:
    q = (x / scale * INT4_LEVELS).round().clamp(-INT4_LEVELS, INT4_LEVELS)
    return q * scale / INT4_LEVELS

def q_binary(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.where(x >= 0, torch.full_like(x, scale), torch.full_like(x, -scale))

def q_ternary(x: torch.Tensor, scale: float) -> torch.Tensor:
    thr = scale * 0.5
    out = torch.zeros_like(x)
    out = torch.where(x > thr, torch.full_like(x, scale), out)
    out = torch.where(x < -thr, torch.full_like(x, -scale), out)
    return out

QUANTIZERS = {
    "float": q_float,
    "int4": q_int4,
    "binary": q_binary,
    "ternary": q_ternary,
}


# --- Model: explicit tensors (not nn.Module) so we can freeze per-weight ---
class BeukersCharLM:
    """Two-projection Beukers-gate char-LM matching the Rust architecture."""

    def __init__(self, nf: int, seed: int = 42):
        torch.manual_seed(seed)
        self.nf = nf
        sc_e = (1.0 / DIM) ** 0.5
        sc_c = (2.0 / FAN) ** 0.5
        sc_h = (2.0 / nf) ** 0.5
        self.embed = (torch.randn(VOCAB, DIM) * sc_e).to(DEVICE).requires_grad_(True)
        # ws: (n_proj, nf, fan)
        self.ws = (torch.randn(N_PROJ, nf, FAN) * sc_c).to(DEVICE).requires_grad_(True)
        self.bs = torch.zeros(N_PROJ, nf, device=DEVICE, requires_grad=True)
        self.hw = (torch.randn(N_CLASSES, nf) * sc_h).to(DEVICE).requires_grad_(True)
        self.hb = torch.zeros(N_CLASSES, device=DEVICE, requires_grad=True)

        # Masks: 1.0 = trainable, 0.0 = frozen
        self.ws_mask = torch.ones_like(self.ws)
        self.hw_mask = torch.ones_like(self.hw)

    def parameters(self):
        return [self.embed, self.ws, self.bs, self.hw, self.hb]

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        """chunks: (B, CTX) char ids -> logits (B, 27)."""
        B = chunks.shape[0]
        emb = self.embed[chunks]                          # (B, CTX, DIM)
        emb = emb.clone()
        emb[:, MASK_POS, :] = 0                            # mask center

        # Gather k=7 window centered at MASK_POS: positions [13..19]
        # Shape: (B, K, DIM) -> flatten to (B, FAN)
        window = emb[:, MASK_POS - HK : MASK_POS - HK + K, :]  # (B, K, DIM)
        window_flat = window.reshape(B, FAN)               # (B, FAN)

        # Two projections: each ws[p] is (nf, fan) -> (B, nf)
        pv0 = F.linear(window_flat, self.ws[0], self.bs[0])
        pv1 = F.linear(window_flat, self.ws[1], self.bs[1])

        # Beukers gate: ab / (1 + |ab|)
        p = pv0 * pv1
        co = p / (1.0 + p.abs())
        co = co.clamp(-10.0, 10.0)

        # Output: hw (27, nf), hb (27,)
        logits = F.linear(co, self.hw, self.hb)            # (B, 27)
        return logits


def sample_chunks(corpus: torch.Tensor, start: int, end: int,
                  n_samples: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n_samples (chunk[CTX], target_char) tuples uniformly."""
    max_off = end - CTX - 1
    offsets = torch.randint(start, max_off, (n_samples,), generator=generator)
    # Build all chunks
    idx_mat = offsets.unsqueeze(1) + torch.arange(CTX).unsqueeze(0)  # (n, CTX)
    chunks = corpus[idx_mat]  # (n, CTX)
    targets = chunks[:, MASK_POS]  # (n,)
    return chunks.to(DEVICE), targets.to(DEVICE)


def train_one_epoch(model: BeukersCharLM, optimizer: torch.optim.Optimizer,
                    corpus: torch.Tensor, split: int,
                    n_samples: int, generator: torch.Generator):
    """One epoch = n_samples samples processed in batches of BATCH_SIZE."""
    for batch_start in range(0, n_samples, BATCH_SIZE):
        batch_n = min(BATCH_SIZE, n_samples - batch_start)
        chunks, targets = sample_chunks(corpus, 0, split, batch_n, generator)
        logits = model.forward(chunks)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        # Apply freeze masks to gradients before step
        if model.ws.grad is not None:
            model.ws.grad *= model.ws_mask
        if model.hw.grad is not None:
            model.hw.grad *= model.hw_mask
        optimizer.step()


@torch.no_grad()
def evaluate(model: BeukersCharLM, corpus: torch.Tensor,
             start: int, end: int, n_samples: int = EVAL_SAMPLES) -> float:
    gen = torch.Generator().manual_seed(999)
    ok = 0
    total = 0
    for batch_start in range(0, n_samples, BATCH_SIZE):
        batch_n = min(BATCH_SIZE, n_samples - batch_start)
        chunks, targets = sample_chunks(corpus, start, end, batch_n, gen)
        logits = model.forward(chunks)
        pred = logits.argmax(dim=-1)
        ok += (pred == targets).sum().item()
        total += batch_n
    return 100.0 * ok / max(total, 1)


def run_staged_inq(task: str, corpus: torch.Tensor, split: int,
                   nf: int, mode: str, seed: int = 42) -> dict:
    """Full training run: float plateau -> staged quantization (if not float)."""
    t0 = time.time()
    model = BeukersCharLM(nf, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    gen = torch.Generator().manual_seed(seed)

    # Phase 1: float plateau
    print(f"[{task} nf={nf} {mode}] Phase 1: float plateau on {DEVICE}")
    best_test = 0.0
    no_improve = 0
    final_ep = 0
    for ep in range(MAX_EP_INIT):
        lr = 0.01 * (1.0 - ep / MAX_EP_INIT * 0.8)
        for g in optimizer.param_groups:
            g["lr"] = lr
        train_one_epoch(model, optimizer, corpus, split, SAMPLES_PER_EP, gen)
        final_ep = ep + 1
        if (ep + 1) % LOG_EVERY == 0:
            te = evaluate(model, corpus, split, len(corpus), EVAL_SAMPLES)
            if te > best_test + 0.01:
                best_test = te
                no_improve = 0
            else:
                no_improve += LOG_EVERY
            if no_improve >= PATIENCE:
                break

    acc_float = evaluate(model, corpus, split, len(corpus), EVAL_SAMPLES)
    print(f"[{task} nf={nf} {mode}] float done te={acc_float:.2f} @ ep={final_ep}")

    if mode == "float":
        elapsed = time.time() - t0
        return {"task": task, "nf": nf, "mode": mode,
                "float_acc": acc_float, "final_acc": acc_float, "ep": final_ep,
                "seconds": elapsed}

    # Phase 2: staged quantization
    quantize = QUANTIZERS[mode]
    with torch.no_grad():
        max_ws = model.ws.abs().max().clamp(min=1e-9).item()
        max_hw = model.hw.abs().max().clamp(min=1e-9).item()

    total_ws = model.ws.numel()
    total_hw = model.hw.numel()
    per_round = (total_ws + total_hw) // ROUNDS

    print(f"[{task} nf={nf} {mode}] Phase 2: staged ({ROUNDS} rounds x {EPOCHS_PER_ROUND} ep)")

    # Per-weight frozen masks (flat for easy sort)
    ws_frozen_flat = torch.zeros(total_ws, dtype=torch.bool, device=DEVICE)
    hw_frozen_flat = torch.zeros(total_hw, dtype=torch.bool, device=DEVICE)

    for round_i in range(1, ROUNDS + 1):
        with torch.no_grad():
            # Compute quantization error per weight
            ws_flat = model.ws.view(-1)
            hw_flat = model.hw.view(-1)
            ws_err = (ws_flat - quantize(ws_flat, max_ws)).abs()
            hw_err = (hw_flat - quantize(hw_flat, max_hw)).abs()
            # Mask already-frozen weights to +inf so they don't get picked again
            ws_err = ws_err.masked_fill(ws_frozen_flat, float("inf"))
            hw_err = hw_err.masked_fill(hw_frozen_flat, float("inf"))
            # Concatenate and sort ascending (smallest error = easiest to grid)
            all_err = torch.cat([ws_err, hw_err])
            _, sorted_idx = torch.sort(all_err)
            to_freeze = sorted_idx[:per_round]

            # Apply quantization + freeze in vectorized form
            ws_idx = to_freeze[to_freeze < total_ws]
            hw_idx = to_freeze[to_freeze >= total_ws] - total_ws
            if ws_idx.numel() > 0:
                vals = quantize(ws_flat[ws_idx], max_ws)
                ws_flat[ws_idx] = vals
                ws_frozen_flat[ws_idx] = True
            if hw_idx.numel() > 0:
                vals = quantize(hw_flat[hw_idx], max_hw)
                hw_flat[hw_idx] = vals
                hw_frozen_flat[hw_idx] = True

            # Update masks for gradient zeroing (1=trainable, 0=frozen)
            model.ws_mask = (~ws_frozen_flat).view_as(model.ws).float()
            model.hw_mask = (~hw_frozen_flat).view_as(model.hw).float()

        # Retrain remaining weights
        for ep in range(EPOCHS_PER_ROUND):
            lr = 0.005 * (1.0 - ep / EPOCHS_PER_ROUND * 0.5)
            for g in optimizer.param_groups:
                g["lr"] = lr
            train_one_epoch(model, optimizer, corpus, split, SAMPLES_PER_EP, gen)

    acc_final = evaluate(model, corpus, split, len(corpus), EVAL_SAMPLES)
    elapsed = time.time() - t0
    print(f"[{task} nf={nf} {mode}] DONE final te={acc_final:.2f} "
          f"(vs float {acc_final - acc_float:+.2f}pp) [{elapsed:.1f}s]")
    return {"task": task, "nf": nf, "mode": mode,
            "float_acc": acc_float, "final_acc": acc_final, "ep": final_ep,
            "seconds": elapsed}


def main():
    t0 = time.time()
    fineweb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"
    code_path = sys.argv[2] if len(sys.argv) > 2 else \
        "instnct-core/tests/fixtures/code_corpus.txt"

    print("=== GPU QUANT SWEEP: nf=1024 on RTX 4070 Ti Super ===")
    print(f"   Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"   FineWeb: {fineweb_path}")
    print(f"   Code:    {code_path}")
    print()

    print("Loading corpora...")
    fineweb = load_corpus(fineweb_path)
    code = load_corpus(code_path)
    fw_split = len(fineweb) * 80 // 100
    code_split = len(code) * 80 // 100
    print(f"   FineWeb: {len(fineweb):,} filtered bytes, split {fw_split:,}/{len(fineweb)-fw_split:,}")
    print(f"   Code:    {len(code):,} filtered bytes, split {code_split:,}/{len(code)-code_split:,}")
    print()

    modes = ["float", "int4", "ternary", "binary"]
    tasks = [("FineWeb", fineweb, fw_split), ("Code", code, code_split)]

    results = []
    for task_tag, corpus, split in tasks:
        for mode in modes:
            r = run_staged_inq(task_tag, corpus, split, NF_GPU, mode)
            results.append(r)
            print()

    print("=" * 70)
    print(f"  SUMMARY - GPU sweep (nf={NF_GPU})")
    print("=" * 70)
    print(f"  {'task':<8} {'nf':>4} {'mode':>10} {'final_acc':>10} {'delta_vs_float':>16} {'seconds':>10}")
    print(f"  {'-'*8} {'-'*4} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    # Group by task; find float baseline per task
    for task_tag, _, _ in tasks:
        task_rows = [r for r in results if r["task"] == task_tag]
        float_acc = next(r["final_acc"] for r in task_rows if r["mode"] == "float")
        for r in task_rows:
            delta = r["final_acc"] - float_acc
            delta_str = f"{delta:+.2f}pp" if r["mode"] != "float" else "—"
            print(f"  {r['task']:<8} {r['nf']:>4} {r['mode']:>10} "
                  f"{r['final_acc']:>10.2f} {delta_str:>12} {r['seconds']:>10.1f}")

    print()
    print(f"  Total wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
