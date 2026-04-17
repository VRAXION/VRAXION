"""GPU mid-precision fill sweep: int5, int8, fp16 at nf=1024.

Completes the precision ladder between int4 and float32:
  - int5 (32 levels)  - Zhou et al. INQ paper's "sweet spot"
  - int8 (256 levels) - standard deploy precision
  - fp16 (half prec)  - GPU-native 16-bit float

Matches architecture/protocol of diag_quant_sweep_gpu.py exactly.

Run: python tools/diag_quant_sweep_gpu_mid.py <fineweb> <code_corpus>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# --- Config (identical to diag_quant_sweep_gpu.py) ---
VOCAB = 2000
DIM = 16
CTX = 32
MASK_POS = CTX // 2
K = 7
HK = 3
N_PROJ = 2
FAN = K * DIM
N_CLASSES = 27

ROUNDS = 10
EPOCHS_PER_ROUND = 20
MAX_EP_INIT = 200
PATIENCE = 30
LOG_EVERY = 10

NF_GPU = 1024
BATCH_SIZE = 4096
SAMPLES_PER_EP = 16384
EVAL_SAMPLES = 2000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_corpus(path: str) -> torch.Tensor:
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


# --- New quantizers for mid-precision ladder ---

def q_int5(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Int5: 32 levels, symmetric, levels=-15..+15 (31 values + 0 = 32)."""
    levels = 15.0
    q = (x / scale * levels).round().clamp(-levels, levels)
    return q * scale / levels

def q_int8(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Int8: 256 levels, symmetric, levels=-127..+127."""
    levels = 127.0
    q = (x / scale * levels).round().clamp(-levels, levels)
    return q * scale / levels

def q_fp16(x: torch.Tensor, scale: float) -> torch.Tensor:
    """FP16: half-precision IEEE 754. Just cast to fp16 and back.
    Note: 'scale' is unused but kept for signature compatibility."""
    return x.to(torch.float16).to(x.dtype)

QUANTIZERS = {
    "int5": q_int5,
    "int8": q_int8,
    "fp16": q_fp16,
}


# --- Model (identical to main GPU script) ---
class BeukersCharLM:
    def __init__(self, nf: int, seed: int = 42):
        torch.manual_seed(seed)
        self.nf = nf
        sc_e = (1.0 / DIM) ** 0.5
        sc_c = (2.0 / FAN) ** 0.5
        sc_h = (2.0 / nf) ** 0.5
        self.embed = (torch.randn(VOCAB, DIM) * sc_e).to(DEVICE).requires_grad_(True)
        self.ws = (torch.randn(N_PROJ, nf, FAN) * sc_c).to(DEVICE).requires_grad_(True)
        self.bs = torch.zeros(N_PROJ, nf, device=DEVICE, requires_grad=True)
        self.hw = (torch.randn(N_CLASSES, nf) * sc_h).to(DEVICE).requires_grad_(True)
        self.hb = torch.zeros(N_CLASSES, device=DEVICE, requires_grad=True)
        self.ws_mask = torch.ones_like(self.ws)
        self.hw_mask = torch.ones_like(self.hw)

    def parameters(self):
        return [self.embed, self.ws, self.bs, self.hw, self.hb]

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        B = chunks.shape[0]
        emb = self.embed[chunks].clone()
        emb[:, MASK_POS, :] = 0
        window = emb[:, MASK_POS - HK : MASK_POS - HK + K, :]
        window_flat = window.reshape(B, FAN)
        pv0 = F.linear(window_flat, self.ws[0], self.bs[0])
        pv1 = F.linear(window_flat, self.ws[1], self.bs[1])
        p = pv0 * pv1
        co = (p / (1.0 + p.abs())).clamp(-10.0, 10.0)
        return F.linear(co, self.hw, self.hb)


def sample_chunks(corpus, start, end, n, gen):
    max_off = end - CTX - 1
    offsets = torch.randint(start, max_off, (n,), generator=gen)
    idx_mat = offsets.unsqueeze(1) + torch.arange(CTX).unsqueeze(0)
    chunks = corpus[idx_mat]
    targets = chunks[:, MASK_POS]
    return chunks.to(DEVICE), targets.to(DEVICE)


def train_one_epoch(model, optimizer, corpus, split, n_samples, gen):
    for batch_start in range(0, n_samples, BATCH_SIZE):
        batch_n = min(BATCH_SIZE, n_samples - batch_start)
        chunks, targets = sample_chunks(corpus, 0, split, batch_n, gen)
        logits = model.forward(chunks)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        if model.ws.grad is not None:
            model.ws.grad *= model.ws_mask
        if model.hw.grad is not None:
            model.hw.grad *= model.hw_mask
        optimizer.step()


@torch.no_grad()
def evaluate(model, corpus, start, end, n_samples=EVAL_SAMPLES):
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


def run_staged_inq(task, corpus, split, nf, mode, seed=42):
    t0 = time.time()
    model = BeukersCharLM(nf, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    gen = torch.Generator().manual_seed(seed)

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

    quantize = QUANTIZERS[mode]
    with torch.no_grad():
        max_ws = model.ws.abs().max().clamp(min=1e-9).item()
        max_hw = model.hw.abs().max().clamp(min=1e-9).item()

    total_ws = model.ws.numel()
    total_hw = model.hw.numel()
    per_round = (total_ws + total_hw) // ROUNDS

    print(f"[{task} nf={nf} {mode}] Phase 2: staged ({ROUNDS} rounds x {EPOCHS_PER_ROUND} ep)")
    ws_frozen = torch.zeros(total_ws, dtype=torch.bool, device=DEVICE)
    hw_frozen = torch.zeros(total_hw, dtype=torch.bool, device=DEVICE)

    for round_i in range(1, ROUNDS + 1):
        with torch.no_grad():
            ws_flat = model.ws.view(-1)
            hw_flat = model.hw.view(-1)
            ws_err = (ws_flat - quantize(ws_flat, max_ws)).abs()
            hw_err = (hw_flat - quantize(hw_flat, max_hw)).abs()
            ws_err = ws_err.masked_fill(ws_frozen, float("inf"))
            hw_err = hw_err.masked_fill(hw_frozen, float("inf"))
            all_err = torch.cat([ws_err, hw_err])
            _, sorted_idx = torch.sort(all_err)
            to_freeze = sorted_idx[:per_round]

            ws_idx = to_freeze[to_freeze < total_ws]
            hw_idx = to_freeze[to_freeze >= total_ws] - total_ws
            if ws_idx.numel() > 0:
                ws_flat[ws_idx] = quantize(ws_flat[ws_idx], max_ws)
                ws_frozen[ws_idx] = True
            if hw_idx.numel() > 0:
                hw_flat[hw_idx] = quantize(hw_flat[hw_idx], max_hw)
                hw_frozen[hw_idx] = True

            model.ws_mask = (~ws_frozen).view_as(model.ws).float()
            model.hw_mask = (~hw_frozen).view_as(model.hw).float()

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

    print("=== GPU MID-PRECISION SWEEP: int5 / int8 / fp16 at nf=1024 ===")
    print(f"   Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()

    fineweb = load_corpus(fineweb_path)
    code = load_corpus(code_path)
    fw_split = len(fineweb) * 80 // 100
    code_split = len(code) * 80 // 100
    print(f"   FineWeb: {len(fineweb):,} bytes")
    print(f"   Code:    {len(code):,} bytes")
    print()

    modes = ["int5", "int8", "fp16"]
    tasks = [("FineWeb", fineweb, fw_split), ("Code", code, code_split)]

    results = []
    for task_tag, corpus, split in tasks:
        for mode in modes:
            r = run_staged_inq(task_tag, corpus, split, NF_GPU, mode)
            results.append(r)
            print()

    print("=" * 70)
    print(f"  SUMMARY - mid-precision GPU sweep (nf={NF_GPU})")
    print("=" * 70)
    print(f"  {'task':<8} {'mode':>8} {'final_acc':>10} {'delta_vs_float':>16} {'seconds':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*16} {'-'*10}")
    for r in results:
        delta = r["final_acc"] - r["float_acc"]
        print(f"  {r['task']:<8} {r['mode']:>8} {r['final_acc']:>10.2f} "
              f"{delta:>+14.2f}pp {r['seconds']:>10.1f}")

    print()
    print(f"  Total wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
