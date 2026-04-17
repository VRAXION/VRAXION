"""QAT (Quantization-Aware Training) with STE (Straight-Through Estimator).

The key trick: quantize weights in forward pass, pretend gradient is 1 in backward.
Shadow float weights get updated; effective weights are int4/int8/ternary/binary.

This is fundamentally different from staged INQ (post-training gradual freeze)
and from naive PTQ (post-training round). BitNet b1.58 and modern LLM
quantization use STE-based QAT.

Tests 4 modes at nf=1024 on GPU, 2 tasks:
  - STE binary
  - STE ternary
  - STE int4
  - STE int8

Run: python tools/diag_qat_ste.py <fineweb> <code_corpus>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

VOCAB = 2000
DIM = 16
CTX = 32
MASK_POS = CTX // 2
K = 7
HK = 3
N_PROJ = 2
FAN = K * DIM
N_CLASSES = 27

NF = 1024
BATCH_SIZE = 4096
SAMPLES_PER_EP = 16384
EVAL_SAMPLES = 2000
MAX_EP = 400  # match float_long control from earlier
PATIENCE = 60
LOG_EVERY = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_corpus(path):
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


# --- STE quantizer: custom autograd Function ---
class STERound(torch.autograd.Function):
    """Round to nearest integer multiple of step, pass gradient through unchanged."""
    @staticmethod
    def forward(ctx, x, step):
        return torch.round(x / step) * step

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pretend derivative is 1 (identity)
        return grad_output, None


class STESign(torch.autograd.Function):
    """Sign function (binary), gradient passes through as clipped identity."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.full_like(x, scale), torch.full_like(x, -scale))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # STE: pass gradient through, but clip for weights that are saturated
        # (standard BNN / XNOR-Net trick)
        grad = grad_output.clone()
        grad[x.abs() > 1.0] = 0
        return grad, None


class STETernary(torch.autograd.Function):
    """Ternary: {-scale, 0, +scale} with threshold scale/2."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x)
        thr = scale * 0.5
        out = torch.zeros_like(x)
        out = torch.where(x > thr, torch.full_like(x, scale), out)
        out = torch.where(x < -thr, torch.full_like(x, -scale), out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = grad_output.clone()
        grad[x.abs() > 1.0] = 0
        return grad, None


def qat_int4(x, scale):
    """QAT int4: weights in [-scale, +scale], 16 levels, STE backprop."""
    levels = 7.0
    step = scale / levels
    # Clamp then STE-round
    x_c = x.clamp(-scale, scale)
    return STERound.apply(x_c, step)


def qat_int8(x, scale):
    """QAT int8: 256 levels, STE backprop."""
    levels = 127.0
    step = scale / levels
    x_c = x.clamp(-scale, scale)
    return STERound.apply(x_c, step)


def qat_ternary(x, scale):
    return STETernary.apply(x, scale)


def qat_binary(x, scale):
    return STESign.apply(x, scale)


QAT_MODES = {
    "binary": qat_binary,
    "ternary": qat_ternary,
    "int4": qat_int4,
    "int8": qat_int8,
}


# --- Forward pass with QAT: quantize weights before use ---
def forward_qat(embed, ws, bs, hw, hb, chunks, qat_fn, max_ws, max_hw):
    """Forward pass that applies QAT to ws and hw (but not embed/bs/hb).

    max_ws and max_hw are live scale factors recomputed from the shadow weights.
    """
    B = chunks.shape[0]
    emb = embed[chunks].clone()
    emb[:, MASK_POS, :] = 0
    window = emb[:, MASK_POS - HK : MASK_POS - HK + K, :]
    window_flat = window.reshape(B, FAN)

    # Apply QAT to ws and hw
    ws_q = qat_fn(ws, max_ws)
    hw_q = qat_fn(hw, max_hw)

    pv0 = F.linear(window_flat, ws_q[0], bs[0])
    pv1 = F.linear(window_flat, ws_q[1], bs[1])
    p = pv0 * pv1
    co = (p / (1.0 + p.abs())).clamp(-10.0, 10.0)
    return F.linear(co, hw_q, hb)


def forward_float(embed, ws, bs, hw, hb, chunks):
    """Plain float forward pass (for evaluation without QAT)."""
    B = chunks.shape[0]
    emb = embed[chunks].clone()
    emb[:, MASK_POS, :] = 0
    window = emb[:, MASK_POS - HK : MASK_POS - HK + K, :]
    window_flat = window.reshape(B, FAN)
    pv0 = F.linear(window_flat, ws[0], bs[0])
    pv1 = F.linear(window_flat, ws[1], bs[1])
    p = pv0 * pv1
    co = (p / (1.0 + p.abs())).clamp(-10.0, 10.0)
    return F.linear(co, hw, hb)


def sample_chunks(corpus, start, end, n, gen):
    max_off = end - CTX - 1
    offsets = torch.randint(start, max_off, (n,), generator=gen)
    idx_mat = offsets.unsqueeze(1) + torch.arange(CTX).unsqueeze(0)
    chunks = corpus[idx_mat]
    targets = chunks[:, MASK_POS]
    return chunks.to(DEVICE), targets.to(DEVICE)


@torch.no_grad()
def evaluate_qat(embed, ws, bs, hw, hb, qat_fn, corpus, start, end,
                 n_samples=EVAL_SAMPLES):
    """Evaluate with quantized weights (the actual deployed network)."""
    max_ws = ws.abs().max().clamp(min=1e-9).item()
    max_hw = hw.abs().max().clamp(min=1e-9).item()
    gen = torch.Generator().manual_seed(999)
    ok = 0
    total = 0
    for batch_start in range(0, n_samples, BATCH_SIZE):
        batch_n = min(BATCH_SIZE, n_samples - batch_start)
        chunks, targets = sample_chunks(corpus, start, end, batch_n, gen)
        logits = forward_qat(embed, ws, bs, hw, hb, chunks, qat_fn, max_ws, max_hw)
        pred = logits.argmax(dim=-1)
        ok += (pred == targets).sum().item()
        total += batch_n
    return 100.0 * ok / max(total, 1)


def run_qat(task, corpus, split, mode, seed=42):
    t0 = time.time()
    print(f"[{task} nf={NF} QAT_{mode}] Training with STE on {DEVICE}")

    torch.manual_seed(seed)
    sc_e = (1.0 / DIM) ** 0.5
    sc_c = (2.0 / FAN) ** 0.5
    sc_h = (2.0 / NF) ** 0.5
    embed = (torch.randn(VOCAB, DIM) * sc_e).to(DEVICE).requires_grad_(True)
    ws = (torch.randn(N_PROJ, NF, FAN) * sc_c).to(DEVICE).requires_grad_(True)
    bs = torch.zeros(N_PROJ, NF, device=DEVICE, requires_grad=True)
    hw = (torch.randn(N_CLASSES, NF) * sc_h).to(DEVICE).requires_grad_(True)
    hb = torch.zeros(N_CLASSES, device=DEVICE, requires_grad=True)

    optimizer = torch.optim.Adam([embed, ws, bs, hw, hb], lr=0.01)
    gen = torch.Generator().manual_seed(seed)
    qat_fn = QAT_MODES[mode]

    best = 0.0
    no_imp = 0
    final_ep = 0
    for ep in range(MAX_EP):
        lr = 0.01 * (1.0 - ep / MAX_EP * 0.8)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Recompute scale each epoch (from live shadow weights)
        with torch.no_grad():
            max_ws = ws.abs().max().clamp(min=1e-9).item()
            max_hw = hw.abs().max().clamp(min=1e-9).item()

        for batch_start in range(0, SAMPLES_PER_EP, BATCH_SIZE):
            batch_n = min(BATCH_SIZE, SAMPLES_PER_EP - batch_start)
            chunks, targets = sample_chunks(corpus, 0, split, batch_n, gen)
            logits = forward_qat(embed, ws, bs, hw, hb, chunks, qat_fn, max_ws, max_hw)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_ep = ep + 1
        if (ep + 1) % LOG_EVERY == 0:
            te = evaluate_qat(embed, ws, bs, hw, hb, qat_fn,
                              corpus, split, len(corpus), EVAL_SAMPLES)
            if te > best + 0.01:
                best = te
                no_imp = 0
            else:
                no_imp += LOG_EVERY
            if no_imp >= PATIENCE:
                break

    acc = evaluate_qat(embed, ws, bs, hw, hb, qat_fn,
                       corpus, split, len(corpus), EVAL_SAMPLES)
    elapsed = time.time() - t0
    print(f"[{task} nf={NF} QAT_{mode}] DONE te={acc:.2f} @ ep={final_ep} [{elapsed:.1f}s]")
    return {"task": task, "mode": mode, "acc": acc, "ep": final_ep, "sec": elapsed}


def main():
    t0 = time.time()
    fineweb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"
    code_path = sys.argv[2] if len(sys.argv) > 2 else \
        "instnct-core/tests/fixtures/code_corpus.txt"

    print(f"=== QAT with STE: binary / ternary / int4 / int8 (nf={NF}) ===")
    print(f"   Device: {DEVICE}")
    print(f"   Max epochs: {MAX_EP}, patience {PATIENCE}")
    print()

    fineweb = load_corpus(fineweb_path)
    code = load_corpus(code_path)
    fw_split = len(fineweb) * 80 // 100
    code_split = len(code) * 80 // 100

    modes = ["int8", "int4", "ternary", "binary"]
    tasks = [("FineWeb", fineweb, fw_split), ("Code", code, code_split)]

    results = []
    for task_tag, corpus, split in tasks:
        for mode in modes:
            r = run_qat(task_tag, corpus, split, mode)
            results.append(r)
            print()

    print("=" * 70)
    print(f"  SUMMARY - QAT/STE training (nf={NF})")
    print("=" * 70)
    print(f"  {'task':<8} {'mode':>10} {'acc':>10} {'epochs':>8} {'seconds':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for r in results:
        print(f"  {r['task']:<8} {r['mode']:>10} {r['acc']:>10.2f} {r['ep']:>8} {r['sec']:>10.1f}")

    print()
    print(f"  Total wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
