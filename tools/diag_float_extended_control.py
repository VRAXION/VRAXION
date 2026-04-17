"""Control: float32 with EXTENDED training matching the staged INQ protocol duration.

Tests hypothesis: the '+2.5pp win' of int4/int5/int8/fp16 over float32 baseline
is NOT a quantization effect, but a PROTOCOL artifact (extra 200 epochs in Phase 2).

If true: float32 trained for matching wallclock/epochs should also hit ~85% on FineWeb.

Three variants per task:
  1. float_short:    max_ep=200, patience=30 (= the baseline we compared against)
  2. float_long:     max_ep=400, patience=60 (2x longer training, no quant phase)
  3. float_staged:   max_ep=200 -> then 10 rounds x 20 ep of same continued training
                     (without freezing anything — pure extra training to match quant wallclock)

Run: python tools/diag_float_extended_control.py <fineweb> <code_corpus>
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
LOG_EVERY = 10

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

    def parameters(self):
        return [self.embed, self.ws, self.bs, self.hw, self.hb]

    def forward(self, chunks):
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


def run_float_training(task, corpus, split, max_ep, patience, variant, seed=42):
    """Run plain float training with configurable total epochs.

    Variants:
      'short'  - LR schedule over max_ep=200 (baseline)
      'long'   - LR schedule over max_ep=400, patience=60
      'staged' - 200 ep phase1 (LR schedule) + 10 x 20 ep phase2 (lower LR, same as quant)
    """
    t0 = time.time()
    model = BeukersCharLM(NF, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    gen = torch.Generator().manual_seed(seed)

    print(f"[{task} nf={NF} float_{variant}] Training on {DEVICE}")

    if variant in ("short", "long"):
        # Single-phase with early stop
        best_test = 0.0
        no_improve = 0
        final_ep = 0
        for ep in range(max_ep):
            lr = 0.01 * (1.0 - ep / max_ep * 0.8)
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
                if no_improve >= patience:
                    break
        acc = evaluate(model, corpus, split, len(corpus), EVAL_SAMPLES)
        print(f"[{task} nf={NF} float_{variant}] DONE te={acc:.2f} @ ep={final_ep} [{time.time()-t0:.1f}s]")
        return {"task": task, "variant": variant, "acc": acc, "ep": final_ep, "sec": time.time()-t0}

    elif variant == "staged":
        # Phase 1: standard 200 ep + early stop
        best_test = 0.0
        no_improve = 0
        final_ep = 0
        for ep in range(200):
            lr = 0.01 * (1.0 - ep / 200 * 0.8)
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
                if no_improve >= 30:
                    break
        acc_phase1 = evaluate(model, corpus, split, len(corpus), EVAL_SAMPLES)
        print(f"  Phase 1 done: te={acc_phase1:.2f} @ ep={final_ep}")

        # Phase 2: 10 x 20 extra epochs at lower LR (exactly mimicking quant phase)
        for round_i in range(1, 11):
            for ep in range(20):
                lr = 0.005 * (1.0 - ep / 20 * 0.5)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                train_one_epoch(model, optimizer, corpus, split, SAMPLES_PER_EP, gen)

        acc_final = evaluate(model, corpus, split, len(corpus), EVAL_SAMPLES)
        print(f"[{task} nf={NF} float_staged] DONE te={acc_final:.2f} "
              f"(phase1 {acc_phase1:.2f} -> +{acc_final-acc_phase1:.2f}pp after 200 more ep) "
              f"[{time.time()-t0:.1f}s]")
        return {"task": task, "variant": variant, "acc": acc_final,
                "acc_phase1": acc_phase1, "ep": final_ep + 200,
                "sec": time.time()-t0}


def main():
    t0 = time.time()
    fineweb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"
    code_path = sys.argv[2] if len(sys.argv) > 2 else \
        "instnct-core/tests/fixtures/code_corpus.txt"

    print("=== FLOAT EXTENDED CONTROL: does long training match +2.5pp 'win'? ===")
    print(f"   Device: {DEVICE}")
    print()

    fineweb = load_corpus(fineweb_path)
    code = load_corpus(code_path)
    fw_split = len(fineweb) * 80 // 100
    code_split = len(code) * 80 // 100

    tasks = [("FineWeb", fineweb, fw_split), ("Code", code, code_split)]

    results = []
    for task_tag, corpus, split in tasks:
        # Variant 1: short (baseline from earlier)
        r = run_float_training(task_tag, corpus, split, max_ep=200, patience=30, variant="short")
        results.append(r)
        print()

        # Variant 2: long (400 ep, patience=60)
        r = run_float_training(task_tag, corpus, split, max_ep=400, patience=60, variant="long")
        results.append(r)
        print()

        # Variant 3: staged (phase1 + phase2 WITHOUT freezing)
        r = run_float_training(task_tag, corpus, split, max_ep=0, patience=0, variant="staged")
        results.append(r)
        print()

    print("=" * 70)
    print(f"  SUMMARY - float control (nf={NF})")
    print("=" * 70)
    print(f"  {'task':<8} {'variant':>10} {'acc':>10} {'epochs':>8} {'seconds':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for r in results:
        print(f"  {r['task']:<8} {r['variant']:>10} {r['acc']:>10.2f} {r['ep']:>8} {r['sec']:>10.1f}")

    print()
    print("  HYPOTHESIS CHECK:")
    print("  If float_long or float_staged >= ~85.0 on FineWeb, the '+2.5pp quant win'")
    print("  was a PROTOCOL ARTIFACT (extra training epochs), not a quantization effect.")
    print()
    print(f"  Total wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
