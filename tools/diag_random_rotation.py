"""Random-rotation sparse training on int4 backbone.

User's idea: model stored as int4, with a small 'hot buffer' of weights
temporarily promoted to float for training. Each iteration, a RANDOM
subset becomes hot, trains briefly, then freezes back to int4. Rotate.

Tests: does this actually converge? Simulate on nf=1024 GPU.

Variants:
  - hot_frac=1%  (very sparse, closest to memory-optimal scheme)
  - hot_frac=5%  (moderate)
  - hot_frac=20% (liberal)

Compare against: QAT STE int4 (84.50%) and float_long (86.20%).

Run: python tools/diag_random_rotation.py <fineweb> <code_corpus>
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
def evaluate(embed, ws, bs, hw, hb, corpus, start, end):
    gen = torch.Generator().manual_seed(999)
    ok = 0
    total = 0
    for batch_start in range(0, EVAL_SAMPLES, BATCH_SIZE):
        batch_n = min(BATCH_SIZE, EVAL_SAMPLES - batch_start)
        chunks, targets = sample_chunks(corpus, start, end, batch_n, gen)
        logits = forward(embed, ws, bs, hw, hb, chunks)
        pred = logits.argmax(dim=-1)
        ok += (pred == targets).sum().item()
        total += batch_n
    return 100.0 * ok / max(total, 1)


def run_random_rotation(task, corpus, split, hot_frac=0.05,
                        rotation_every=5, max_ep=400, seed=42):
    """Random-rotation sparse training.

    At each rotation:
      - select `hot_frac` fraction of weights randomly
      - those weights are trainable (float)
      - rest are quantized to int4 and frozen
    After `rotation_every` epochs, re-sample the mask and rotate.
    """
    t0 = time.time()
    print(f"[{task} hot_frac={hot_frac*100:.1f}% rot_every={rotation_every}] "
          f"Starting on {DEVICE}")

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
    mask_gen = torch.Generator(device=DEVICE).manual_seed(seed + 1)

    # Snapshot weights: these are the "cold" int4-quantized values
    # At each rotation: hot positions are unfrozen (revealed from cold state + can be trained)
    #                   cold positions are pinned to their int4 value (forced)
    # Masks: 1.0 = hot (trainable), 0.0 = cold (frozen)

    best = 0.0
    no_imp = 0

    # Phase 1: warm-up float training to get a decent starting point
    WARMUP_EP = 50
    print(f"  Phase 1 warmup: {WARMUP_EP} ep full float")
    for ep in range(WARMUP_EP):
        lr = 0.01 * (1.0 - ep / max_ep * 0.8)
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

    acc_warmup = evaluate(embed, ws, bs, hw, hb, corpus, split, len(corpus))
    print(f"  Phase 1 done: te={acc_warmup:.2f} @ ep={WARMUP_EP}")

    # Phase 2: rotation training
    print(f"  Phase 2: random rotation ({max_ep - WARMUP_EP} ep)")

    for ep in range(WARMUP_EP, max_ep):
        # Rotate mask every `rotation_every` epochs
        if (ep - WARMUP_EP) % rotation_every == 0:
            with torch.no_grad():
                # Quantize ALL weights to int4 snapshot first
                max_ws = ws.abs().max().clamp(min=1e-9).item()
                max_hw = hw.abs().max().clamp(min=1e-9).item()
                ws_cold = q_int4(ws, max_ws)
                hw_cold = q_int4(hw, max_hw)

                # Force cold values onto frozen positions
                ws_hot_mask = (torch.rand(ws.shape, generator=mask_gen, device=DEVICE) < hot_frac).float()
                hw_hot_mask = (torch.rand(hw.shape, generator=mask_gen, device=DEVICE) < hot_frac).float()

                # Set frozen positions to their int4 values
                ws.data = ws * ws_hot_mask + ws_cold * (1 - ws_hot_mask)
                hw.data = hw * hw_hot_mask + hw_cold * (1 - hw_hot_mask)

                # Reset Adam state for newly-hot weights (they just became trainable)
                # Actually simpler: just use the mask to zero gradients of cold weights

        # Train epoch
        lr = 0.005 * (1.0 - (ep - WARMUP_EP) / (max_ep - WARMUP_EP) * 0.5)
        for g in optimizer.param_groups:
            g["lr"] = lr

        for batch_start in range(0, SAMPLES_PER_EP, BATCH_SIZE):
            batch_n = min(BATCH_SIZE, SAMPLES_PER_EP - batch_start)
            chunks, targets = sample_chunks(corpus, 0, split, batch_n, gen)
            logits = forward(embed, ws, bs, hw, hb, chunks)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            # Apply mask: only hot weights get updated
            if ws.grad is not None:
                ws.grad *= ws_hot_mask
            if hw.grad is not None:
                hw.grad *= hw_hot_mask
            optimizer.step()

        if (ep + 1) % 10 == 0:
            te = evaluate(embed, ws, bs, hw, hb, corpus, split, len(corpus))
            print(f"  ep={ep+1}  te={te:.2f} (hot_frac={hot_frac*100:.1f}%)")
            if te > best + 0.01:
                best = te
                no_imp = 0
            else:
                no_imp += 10

    # Phase 3: final quantization to int4 and eval (the "deployed" model)
    with torch.no_grad():
        max_ws = ws.abs().max().clamp(min=1e-9).item()
        max_hw = hw.abs().max().clamp(min=1e-9).item()
        ws_final = q_int4(ws, max_ws)
        hw_final = q_int4(hw, max_hw)

    acc_final = evaluate(embed, ws_final, bs, hw_final, hb,
                         corpus, split, len(corpus))
    acc_pre = evaluate(embed, ws, bs, hw, hb, corpus, split, len(corpus))
    elapsed = time.time() - t0
    print(f"[{task} hot_frac={hot_frac*100:.1f}%] DONE "
          f"pre_quant={acc_pre:.2f} post_int4={acc_final:.2f} "
          f"[{elapsed:.1f}s]")
    return {"task": task, "hot_frac": hot_frac, "rotation_every": rotation_every,
            "acc_pre": acc_pre, "acc_final": acc_final, "sec": elapsed}


def main():
    t0 = time.time()
    fineweb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"
    code_path = sys.argv[2] if len(sys.argv) > 2 else \
        "instnct-core/tests/fixtures/code_corpus.txt"

    print(f"=== RANDOM ROTATION TRAINING (nf={NF}) ===")
    print(f"   Device: {DEVICE}")
    print()

    fineweb = load_corpus(fineweb_path)
    code = load_corpus(code_path)
    fw_split = len(fineweb) * 80 // 100
    code_split = len(code) * 80 // 100

    # Run variants: different hot fractions on FineWeb + Code
    variants = [
        # (hot_frac, rotation_every)
        (0.01, 5),    # very sparse: 1% at a time, rotate every 5 ep
        (0.05, 5),    # 5% sparse
        (0.20, 5),    # moderate: 20%
        (0.50, 10),   # heavy: 50% at once, rotate less often
    ]
    tasks = [("FineWeb", fineweb, fw_split), ("Code", code, code_split)]

    results = []
    for task_tag, corpus, split in tasks:
        for hot_frac, rot_every in variants:
            r = run_random_rotation(task_tag, corpus, split,
                                     hot_frac=hot_frac,
                                     rotation_every=rot_every)
            results.append(r)
            print()

    print("=" * 80)
    print(f"  SUMMARY - random rotation training (nf={NF})")
    print("=" * 80)
    print(f"  {'task':<8} {'hot %':>8} {'rot ep':>8} {'pre-q':>8} "
          f"{'post-q':>10} {'vs QAT int4':>14} {'seconds':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*14} {'-'*10}")
    QAT_INT4_BASELINE = {"FineWeb": 84.50, "Code": 89.95}
    for r in results:
        baseline = QAT_INT4_BASELINE.get(r["task"], 85.0)
        delta = r["acc_final"] - baseline
        print(f"  {r['task']:<8} {r['hot_frac']*100:>7.1f}% "
              f"{r['rotation_every']:>8} {r['acc_pre']:>8.2f} "
              f"{r['acc_final']:>10.2f} {delta:>+12.2f}pp "
              f"{r['sec']:>10.1f}")

    print()
    print(f"  Reference (from previous sweeps):")
    print(f"    FineWeb float_long:  86.20  |  QAT int4:  84.50  |  staged int4:  84.75")
    print(f"    Code float_long:     91.35  |  QAT int4:  89.95  |  staged int4:  91.90")
    print()
    print(f"  Total wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
