"""Progressive growing + per-neuron int4 quantization.

User's design: add one neuron between I/O, train normally with gradient,
round to int4 at end. Then add next neuron, same way. Repeat.

This is cascade-correlation style growth + per-step quantization.

Compares three variants at nf=128 on GPU:
  1. progressive_int4:  grow 1 at a time, int4-quantize each neuron as added
  2. batch_int4:        full nf=128 trained, int4-quantize at end (baseline)
  3. batch_float:       full nf=128 trained, no quantization (upper bound)

Each variant trained for similar total epochs (~1920).

Run: python tools/diag_progressive_quant.py <fineweb> <code_corpus>
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

NF_TARGET = 128
BATCH_SIZE = 4096
SAMPLES_PER_EP = 16384
EVAL_SAMPLES = 2000

# Progressive: 1 neuron × 15 epochs each → 1920 total epochs
EP_PER_NEURON = 15

# Batch: match total epochs for fair comparison (1920 = ~2000)
EP_BATCH = 2000
PATIENCE = 60

# Quantizer
INT4_LEVELS = 7.0

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


def q_int4(x, scale):
    q = (x / scale * INT4_LEVELS).round().clamp(-INT4_LEVELS, INT4_LEVELS)
    return q * scale / INT4_LEVELS


def forward(embed, ws, bs, hw, hb, chunks):
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
def evaluate(embed, ws, bs, hw, hb, corpus, start, end, n_samples=EVAL_SAMPLES):
    gen = torch.Generator().manual_seed(999)
    ok = 0
    total = 0
    for batch_start in range(0, n_samples, BATCH_SIZE):
        batch_n = min(BATCH_SIZE, n_samples - batch_start)
        chunks, targets = sample_chunks(corpus, start, end, batch_n, gen)
        logits = forward(embed, ws, bs, hw, hb, chunks)
        pred = logits.argmax(dim=-1)
        ok += (pred == targets).sum().item()
        total += batch_n
    return 100.0 * ok / max(total, 1)


def init_params(nf, seed):
    torch.manual_seed(seed)
    sc_e = (1.0 / DIM) ** 0.5
    sc_c = (2.0 / FAN) ** 0.5
    sc_h = (2.0 / max(nf, 1)) ** 0.5
    embed = (torch.randn(VOCAB, DIM) * sc_e).to(DEVICE).requires_grad_(True)
    ws = (torch.randn(N_PROJ, nf, FAN) * sc_c).to(DEVICE).requires_grad_(True)
    bs = torch.zeros(N_PROJ, nf, device=DEVICE, requires_grad=True)
    hw = (torch.randn(N_CLASSES, nf) * sc_h).to(DEVICE).requires_grad_(True)
    hb = torch.zeros(N_CLASSES, device=DEVICE, requires_grad=True)
    return embed, ws, bs, hw, hb


def run_progressive_int4(task, corpus, split, nf_target=NF_TARGET, seed=42):
    """Grow 1 neuron at a time; quantize each to int4 when added."""
    t0 = time.time()
    print(f"[{task} progressive_int4] growing 1 neuron at a time (target nf={nf_target})")

    # Pre-allocate target-size tensors (we'll only use the first k at step k)
    torch.manual_seed(seed)
    sc_e = (1.0 / DIM) ** 0.5
    sc_c = (2.0 / FAN) ** 0.5
    sc_h = (2.0 / nf_target) ** 0.5

    embed = (torch.randn(VOCAB, DIM) * sc_e).to(DEVICE).requires_grad_(True)
    ws = (torch.randn(N_PROJ, nf_target, FAN) * sc_c).to(DEVICE).requires_grad_(True)
    bs = torch.zeros(N_PROJ, nf_target, device=DEVICE, requires_grad=True)
    hw = (torch.randn(N_CLASSES, nf_target) * sc_h).to(DEVICE).requires_grad_(True)
    hb = torch.zeros(N_CLASSES, device=DEVICE, requires_grad=True)

    # Mask: 1 = trainable, 0 = frozen
    ws_mask = torch.zeros_like(ws)
    hw_mask = torch.zeros_like(hw)
    bs_mask = torch.zeros_like(bs)
    # hb and embed are always trainable (shared across neurons)

    optimizer = torch.optim.Adam([embed, ws, bs, hw, hb], lr=0.01)
    gen = torch.Generator().manual_seed(seed)

    for step in range(nf_target):
        # Unfreeze the new neuron (step-th column)
        ws_mask[:, step, :] = 1.0
        bs_mask[:, step] = 1.0
        hw_mask[:, step] = 1.0

        # Train for EP_PER_NEURON epochs
        for ep in range(EP_PER_NEURON):
            # Use ws[:, :step+1, :] (only the up-to-now neurons are active)
            # This is done by zeroing out the unused portion via forward's behavior,
            # but easier: just use slices
            ws_active = ws[:, :step + 1, :]
            bs_active = bs[:, :step + 1]
            hw_active = hw[:, :step + 1]

            lr = 0.01 * (1.0 - ep / EP_PER_NEURON * 0.5)
            for g in optimizer.param_groups:
                g["lr"] = lr

            for batch_start in range(0, SAMPLES_PER_EP, BATCH_SIZE):
                batch_n = min(BATCH_SIZE, SAMPLES_PER_EP - batch_start)
                chunks, targets = sample_chunks(corpus, 0, split, batch_n, gen)
                logits = forward(embed, ws_active, bs_active, hw_active, hb, chunks)
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                # Zero gradient for frozen neurons
                if ws.grad is not None:
                    ws.grad *= ws_mask
                if bs.grad is not None:
                    bs.grad *= bs_mask
                if hw.grad is not None:
                    hw.grad *= hw_mask
                optimizer.step()

        # Quantize the newly trained neuron to int4 and freeze it
        with torch.no_grad():
            # Find scale across the new neuron's ws and hw columns
            new_ws_slice = ws[:, step, :]  # (N_PROJ, FAN)
            new_hw_slice = hw[:, step]     # (N_CLASSES,)
            max_ws = new_ws_slice.abs().max().clamp(min=1e-9).item()
            max_hw = new_hw_slice.abs().max().clamp(min=1e-9).item()

            ws[:, step, :] = q_int4(new_ws_slice, max_ws)
            hw[:, step] = q_int4(new_hw_slice, max_hw)
            # bs is kept as-is (biases often better in float for stability)

            # Freeze: the new neuron is no longer trainable
            ws_mask[:, step, :] = 0.0
            bs_mask[:, step] = 0.0
            hw_mask[:, step] = 0.0

        if (step + 1) % 16 == 0 or step == nf_target - 1:
            acc_now = evaluate(embed, ws[:, :step + 1, :], bs[:, :step + 1],
                                hw[:, :step + 1], hb, corpus, split, len(corpus))
            print(f"  [{task} progressive] step={step+1}/{nf_target} te={acc_now:.2f}")

    acc = evaluate(embed, ws, bs, hw, hb, corpus, split, len(corpus))
    elapsed = time.time() - t0
    print(f"[{task} progressive_int4] DONE te={acc:.2f} [{elapsed:.1f}s]")
    return {"task": task, "variant": "progressive_int4", "acc": acc, "sec": elapsed}


def run_batch_float(task, corpus, split, seed=42):
    """Full nf=128 trained from scratch, no quantization."""
    t0 = time.time()
    print(f"[{task} batch_float] nf={NF_TARGET} normal training")

    embed, ws, bs, hw, hb = init_params(NF_TARGET, seed)
    optimizer = torch.optim.Adam([embed, ws, bs, hw, hb], lr=0.01)
    gen = torch.Generator().manual_seed(seed)

    best = 0.0
    no_imp = 0
    final_ep = 0
    for ep in range(EP_BATCH):
        lr = 0.01 * (1.0 - ep / EP_BATCH * 0.8)
        for g in optimizer.param_groups:
            g["lr"] = lr
        for batch_start in range(0, SAMPLES_PER_EP, BATCH_SIZE):
            batch_n = min(BATCH_SIZE, SAMPLES_PER_EP - batch_start)
            chunks, targets = sample_chunks(corpus, 0, split, batch_n, gen)
            logits = forward(embed, ws, bs, hw, hb, chunks)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final_ep = ep + 1
        if (ep + 1) % 10 == 0:
            te = evaluate(embed, ws, bs, hw, hb, corpus, split, len(corpus))
            if te > best + 0.01:
                best = te
                no_imp = 0
            else:
                no_imp += 10
            if no_imp >= PATIENCE:
                break

    acc = evaluate(embed, ws, bs, hw, hb, corpus, split, len(corpus))
    elapsed = time.time() - t0
    print(f"[{task} batch_float] DONE te={acc:.2f} @ ep={final_ep} [{elapsed:.1f}s]")
    return {"task": task, "variant": "batch_float", "acc": acc, "ep": final_ep, "sec": elapsed}


def run_batch_int4_end(task, corpus, split, seed=42):
    """Full nf=128 trained as float, then int4-quantized at end (PTQ)."""
    t0 = time.time()
    print(f"[{task} batch_int4_PTQ] nf={NF_TARGET} float training + end quantize")

    embed, ws, bs, hw, hb = init_params(NF_TARGET, seed)
    optimizer = torch.optim.Adam([embed, ws, bs, hw, hb], lr=0.01)
    gen = torch.Generator().manual_seed(seed)

    best = 0.0
    no_imp = 0
    final_ep = 0
    for ep in range(EP_BATCH):
        lr = 0.01 * (1.0 - ep / EP_BATCH * 0.8)
        for g in optimizer.param_groups:
            g["lr"] = lr
        for batch_start in range(0, SAMPLES_PER_EP, BATCH_SIZE):
            batch_n = min(BATCH_SIZE, SAMPLES_PER_EP - batch_start)
            chunks, targets = sample_chunks(corpus, 0, split, batch_n, gen)
            logits = forward(embed, ws, bs, hw, hb, chunks)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final_ep = ep + 1
        if (ep + 1) % 10 == 0:
            te = evaluate(embed, ws, bs, hw, hb, corpus, split, len(corpus))
            if te > best + 0.01:
                best = te
                no_imp = 0
            else:
                no_imp += 10
            if no_imp >= PATIENCE:
                break

    acc_pre = evaluate(embed, ws, bs, hw, hb, corpus, split, len(corpus))

    # Post-training quantization (naive): round all weights to int4
    with torch.no_grad():
        max_ws = ws.abs().max().clamp(min=1e-9).item()
        max_hw = hw.abs().max().clamp(min=1e-9).item()
        ws_q = q_int4(ws, max_ws)
        hw_q = q_int4(hw, max_hw)

    acc_post = evaluate(embed, ws_q, bs, hw_q, hb, corpus, split, len(corpus))
    elapsed = time.time() - t0
    print(f"[{task} batch_int4_PTQ] DONE pre={acc_pre:.2f} post_q={acc_post:.2f} "
          f"(delta={acc_post-acc_pre:+.2f}pp) [{elapsed:.1f}s]")
    return {"task": task, "variant": "batch_int4_PTQ", "acc": acc_post,
            "acc_pre_quant": acc_pre, "ep": final_ep, "sec": elapsed}


def main():
    t0 = time.time()
    fineweb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"
    code_path = sys.argv[2] if len(sys.argv) > 2 else \
        "instnct-core/tests/fixtures/code_corpus.txt"

    print(f"=== PROGRESSIVE GROWING + INT4 QUANT (nf={NF_TARGET}) ===")
    print(f"   Device: {DEVICE}")
    print(f"   Progressive: 1 neuron x {EP_PER_NEURON} ep = {NF_TARGET * EP_PER_NEURON} total ep")
    print(f"   Batch: up to {EP_BATCH} ep with early stop")
    print()

    fineweb = load_corpus(fineweb_path)
    code = load_corpus(code_path)
    fw_split = len(fineweb) * 80 // 100
    code_split = len(code) * 80 // 100

    tasks = [("FineWeb", fineweb, fw_split), ("Code", code, code_split)]
    results = []

    for task_tag, corpus, split in tasks:
        # Baseline 1: pure float (upper bound)
        r = run_batch_float(task_tag, corpus, split)
        results.append(r)
        print()

        # Baseline 2: float train + int4 PTQ at end (naive)
        r = run_batch_int4_end(task_tag, corpus, split)
        results.append(r)
        print()

        # Main experiment: progressive 1-at-a-time with int4
        r = run_progressive_int4(task_tag, corpus, split)
        results.append(r)
        print()

    print("=" * 70)
    print(f"  SUMMARY - progressive growing experiment (nf={NF_TARGET})")
    print("=" * 70)
    print(f"  {'task':<8} {'variant':>20} {'acc':>10} {'extras':>20} {'seconds':>10}")
    print(f"  {'-'*8} {'-'*20} {'-'*10} {'-'*20} {'-'*10}")
    for r in results:
        extras = ""
        if "ep" in r:
            extras = f"ep={r['ep']}"
        if "acc_pre_quant" in r:
            extras = f"pre={r['acc_pre_quant']:.1f}"
        print(f"  {r['task']:<8} {r['variant']:>20} {r['acc']:>10.2f} {extras:>20} {r['sec']:>10.1f}")

    print()
    print(f"  Total wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
