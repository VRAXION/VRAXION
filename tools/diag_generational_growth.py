"""Generational growth: train float → LUT-freeze → add new generation → repeat.

User's idea (staying in the staged INQ framework):
  Gen 1: train nf=N float, staged INQ to int4, "LUT-ify"
  Gen 2: ADD nf=N fresh float neurons, keep Gen 1 frozen, train Gen 2
  Gen 3: LUT-ify Gen 2, ADD another nf=N, train...

Each new generation contributes additively to output logits, learning
what previous generations couldn't capture.

Compares:
  - nf=512 single-shot (baseline)
  - Gen1(256) + Gen2(256) generational = effective nf=512
  - Gen1(256) + Gen2(256) + Gen3(256) = effective nf=768
  - nf=768 single-shot (for comparison)

Run: python tools/diag_generational_growth.py <fineweb> <code_corpus>
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


def compute_features(embed, ws_gen, bs_gen, chunks):
    """Compute the Beukers-gate features from given ws/bs tensors.

    ws_gen shape: (N_PROJ, nf_this_gen, FAN)
    bs_gen shape: (N_PROJ, nf_this_gen)
    Returns: (B, nf_this_gen) features
    """
    B = chunks.shape[0]
    emb = embed[chunks].clone()
    emb[:, MASK_POS, :] = 0
    window = emb[:, MASK_POS - HK : MASK_POS - HK + K, :]
    window_flat = window.reshape(B, FAN)
    pv0 = F.linear(window_flat, ws_gen[0], bs_gen[0])
    pv1 = F.linear(window_flat, ws_gen[1], bs_gen[1])
    p = pv0 * pv1
    return (p / (1.0 + p.abs())).clamp(-10.0, 10.0)


def sample_chunks(corpus, start, end, n, gen):
    max_off = end - CTX - 1
    offsets = torch.randint(start, max_off, (n,), generator=gen)
    idx_mat = offsets.unsqueeze(1) + torch.arange(CTX).unsqueeze(0)
    chunks = corpus[idx_mat]
    targets = chunks[:, MASK_POS]
    return chunks.to(DEVICE), targets.to(DEVICE)


class GenerationalModel:
    """Model with multiple generations.

    - Older generations: frozen int4 weights, fixed forward contribution
    - Current generation: trainable float weights, contributes additively
    """

    def __init__(self, seed=42):
        torch.manual_seed(seed)
        sc_e = (1.0 / DIM) ** 0.5
        self.embed = (torch.randn(VOCAB, DIM) * sc_e).to(DEVICE).requires_grad_(True)
        self.hb = torch.zeros(N_CLASSES, device=DEVICE, requires_grad=True)

        # Per-generation storage
        self.frozen_gens = []  # list of {ws, bs, hw} dicts (quantized, fixed)
        self.current_ws = None
        self.current_bs = None
        self.current_hw = None

    def add_generation(self, nf, seed=42):
        """Add a new trainable generation with nf neurons."""
        torch.manual_seed(seed + len(self.frozen_gens) * 100)
        sc_c = (2.0 / FAN) ** 0.5
        sc_h = (2.0 / nf) ** 0.5
        self.current_ws = (torch.randn(N_PROJ, nf, FAN) * sc_c).to(DEVICE).requires_grad_(True)
        self.current_bs = torch.zeros(N_PROJ, nf, device=DEVICE, requires_grad=True)
        self.current_hw = (torch.randn(N_CLASSES, nf) * sc_h).to(DEVICE).requires_grad_(True)
        self.current_nf = nf

    def freeze_current(self):
        """Quantize current generation to int4 and move to frozen list."""
        with torch.no_grad():
            max_ws = self.current_ws.abs().max().clamp(min=1e-9).item()
            max_hw = self.current_hw.abs().max().clamp(min=1e-9).item()
            frozen = {
                "ws": q_int4(self.current_ws.detach(), max_ws),
                "bs": self.current_bs.detach().clone(),
                "hw": q_int4(self.current_hw.detach(), max_hw),
                "nf": self.current_nf,
            }
            self.frozen_gens.append(frozen)
        self.current_ws = None
        self.current_bs = None
        self.current_hw = None

    def forward(self, chunks):
        """Forward: sum contributions from all frozen gens + current gen."""
        logits = self.hb.unsqueeze(0).expand(chunks.shape[0], -1).clone()
        # Contribution from each frozen generation
        for gen in self.frozen_gens:
            with torch.no_grad():
                co = compute_features(self.embed, gen["ws"], gen["bs"], chunks)
            # hw is frozen (quantized), add its contribution
            logits = logits + F.linear(co, gen["hw"])
        # Contribution from current (trainable) generation
        if self.current_ws is not None:
            co_cur = compute_features(self.embed, self.current_ws, self.current_bs, chunks)
            logits = logits + F.linear(co_cur, self.current_hw)
        return logits

    def trainable_params(self):
        p = [self.embed, self.hb]
        if self.current_ws is not None:
            p.extend([self.current_ws, self.current_bs, self.current_hw])
        return p

    def total_nf(self):
        t = sum(g["nf"] for g in self.frozen_gens)
        if self.current_ws is not None:
            t += self.current_nf
        return t


def train_current_gen(model, corpus, split, max_ep=200, patience=30, seed=42):
    """Train the current (active) generation to plateau."""
    optimizer = torch.optim.Adam(model.trainable_params(), lr=0.01)
    gen_rng = torch.Generator().manual_seed(seed + model.total_nf())
    best = 0.0
    no_imp = 0
    final_ep = 0
    for ep in range(max_ep):
        lr = 0.01 * (1.0 - ep / max_ep * 0.8)
        for g in optimizer.param_groups:
            g["lr"] = lr
        for batch_start in range(0, SAMPLES_PER_EP, BATCH_SIZE):
            batch_n = min(BATCH_SIZE, SAMPLES_PER_EP - batch_start)
            chunks, targets = sample_chunks(corpus, 0, split, batch_n, gen_rng)
            logits = model.forward(chunks)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final_ep = ep + 1
        if (ep + 1) % 10 == 0:
            te = evaluate(model, corpus, split, len(corpus))
            if te > best + 0.01:
                best = te
                no_imp = 0
            else:
                no_imp += 10
            if no_imp >= patience:
                break
    return final_ep


def staged_inq_current(model, corpus, split, rounds=10, eps_per_round=20, seed=42):
    """Staged INQ on the current generation.

    Gradually freezes individual weights in the current gen to int4.
    """
    if model.current_ws is None:
        return

    ws = model.current_ws
    hw = model.current_hw
    optimizer = torch.optim.Adam(model.trainable_params(), lr=0.005)
    gen_rng = torch.Generator().manual_seed(seed + 500 + model.total_nf())

    with torch.no_grad():
        max_ws = ws.abs().max().clamp(min=1e-9).item()
        max_hw = hw.abs().max().clamp(min=1e-9).item()

    ws_frozen = torch.zeros(ws.numel(), dtype=torch.bool, device=DEVICE)
    hw_frozen = torch.zeros(hw.numel(), dtype=torch.bool, device=DEVICE)
    total_params = ws.numel() + hw.numel()
    per_round = total_params // rounds

    ws_mask = torch.ones_like(ws)
    hw_mask = torch.ones_like(hw)

    for round_i in range(1, rounds + 1):
        with torch.no_grad():
            ws_flat = ws.view(-1)
            hw_flat = hw.view(-1)
            ws_err = (ws_flat - q_int4(ws_flat, max_ws)).abs()
            hw_err = (hw_flat - q_int4(hw_flat, max_hw)).abs()
            ws_err = ws_err.masked_fill(ws_frozen, float("inf"))
            hw_err = hw_err.masked_fill(hw_frozen, float("inf"))
            all_err = torch.cat([ws_err, hw_err])
            _, sorted_idx = torch.sort(all_err)
            to_freeze = sorted_idx[:per_round]
            ws_idx = to_freeze[to_freeze < ws.numel()]
            hw_idx = to_freeze[to_freeze >= ws.numel()] - ws.numel()
            if ws_idx.numel() > 0:
                ws_flat[ws_idx] = q_int4(ws_flat[ws_idx], max_ws)
                ws_frozen[ws_idx] = True
            if hw_idx.numel() > 0:
                hw_flat[hw_idx] = q_int4(hw_flat[hw_idx], max_hw)
                hw_frozen[hw_idx] = True
            ws_mask = (~ws_frozen).view_as(ws).float()
            hw_mask = (~hw_frozen).view_as(hw).float()

        for ep in range(eps_per_round):
            lr = 0.005 * (1.0 - ep / eps_per_round * 0.5)
            for g in optimizer.param_groups:
                g["lr"] = lr
            for batch_start in range(0, SAMPLES_PER_EP, BATCH_SIZE):
                batch_n = min(BATCH_SIZE, SAMPLES_PER_EP - batch_start)
                chunks, targets = sample_chunks(corpus, 0, split, batch_n, gen_rng)
                logits = model.forward(chunks)
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                if ws.grad is not None:
                    ws.grad *= ws_mask
                if hw.grad is not None:
                    hw.grad *= hw_mask
                optimizer.step()


@torch.no_grad()
def evaluate(model, corpus, start, end):
    gen = torch.Generator().manual_seed(999)
    ok = 0
    total = 0
    for batch_start in range(0, EVAL_SAMPLES, BATCH_SIZE):
        batch_n = min(BATCH_SIZE, EVAL_SAMPLES - batch_start)
        chunks, targets = sample_chunks(corpus, start, end, batch_n, gen)
        logits = model.forward(chunks)
        pred = logits.argmax(dim=-1)
        ok += (pred == targets).sum().item()
        total += batch_n
    return 100.0 * ok / max(total, 1)


def run_generational(task, corpus, split, gen_sizes=[256, 256, 256], seed=42):
    """Train generations sequentially."""
    t0 = time.time()
    print(f"[{task} generational] gen_sizes={gen_sizes}")
    model = GenerationalModel(seed=seed)

    gen_results = []
    for i, nf in enumerate(gen_sizes):
        print(f"  --- Gen {i+1}: adding nf={nf} ---")
        model.add_generation(nf, seed=seed)

        # Phase 1: train float to plateau
        ep_p1 = train_current_gen(model, corpus, split, max_ep=200, patience=30, seed=seed)
        acc_p1 = evaluate(model, corpus, split, len(corpus))
        print(f"    Gen {i+1} float done: te={acc_p1:.2f} @ ep={ep_p1}")

        # Phase 2: staged INQ on this generation
        staged_inq_current(model, corpus, split, rounds=10, eps_per_round=20, seed=seed)
        acc_q = evaluate(model, corpus, split, len(corpus))
        print(f"    Gen {i+1} int4 done: te={acc_q:.2f} (total nf={model.total_nf()})")

        # Freeze this generation, next will add on top
        model.freeze_current()
        gen_results.append({"gen": i+1, "nf_added": nf, "total_nf": model.total_nf(),
                            "acc_float": acc_p1, "acc_int4": acc_q})

    acc_final = evaluate(model, corpus, split, len(corpus))
    elapsed = time.time() - t0
    print(f"[{task} generational] FINAL te={acc_final:.2f} "
          f"(total nf={model.total_nf()}) [{elapsed:.1f}s]")
    return {"task": task, "method": "generational", "gens": gen_results,
            "final_acc": acc_final, "total_nf": model.total_nf(),
            "sec": elapsed}


def run_single_shot(task, corpus, split, nf, seed=42):
    """Baseline: train single-shot at target nf, with staged INQ."""
    t0 = time.time()
    print(f"[{task} single_shot nf={nf}] starting")
    model = GenerationalModel(seed=seed)
    model.add_generation(nf, seed=seed)

    ep = train_current_gen(model, corpus, split, max_ep=400, patience=60, seed=seed)
    acc_p1 = evaluate(model, corpus, split, len(corpus))
    print(f"  float done: te={acc_p1:.2f} @ ep={ep}")

    staged_inq_current(model, corpus, split, rounds=10, eps_per_round=20, seed=seed)
    acc_q = evaluate(model, corpus, split, len(corpus))
    elapsed = time.time() - t0
    print(f"[{task} single_shot nf={nf}] FINAL te={acc_q:.2f} "
          f"(float was {acc_p1:.2f}) [{elapsed:.1f}s]")
    return {"task": task, "method": f"single_shot_nf{nf}",
            "total_nf": nf, "acc_float": acc_p1, "final_acc": acc_q,
            "sec": elapsed}


def main():
    t0 = time.time()
    fineweb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"
    code_path = sys.argv[2] if len(sys.argv) > 2 else \
        "instnct-core/tests/fixtures/code_corpus.txt"

    print("=== GENERATIONAL GROWTH: float -> INT4 LUT -> add next gen ===")
    print(f"   Device: {DEVICE}")
    print()

    fineweb = load_corpus(fineweb_path)
    code = load_corpus(code_path)
    fw_split = len(fineweb) * 80 // 100
    code_split = len(code) * 80 // 100

    tasks = [("FineWeb", fineweb, fw_split), ("Code", code, code_split)]

    all_results = []
    for task_tag, corpus, split in tasks:
        # Baseline 1: single-shot nf=512 (total gen 1+2)
        r = run_single_shot(task_tag, corpus, split, nf=512)
        all_results.append(r)
        print()

        # Baseline 2: single-shot nf=768 (total gen 1+2+3)
        r = run_single_shot(task_tag, corpus, split, nf=768)
        all_results.append(r)
        print()

        # Main experiment: 3 generations of 256
        r = run_generational(task_tag, corpus, split, gen_sizes=[256, 256, 256])
        all_results.append(r)
        print()

    print("=" * 80)
    print(f"  SUMMARY - generational growth")
    print("=" * 80)
    print(f"  {'task':<8} {'method':<20} {'total_nf':>8} {'final_acc':>10} {'seconds':>10}")
    print(f"  {'-'*8} {'-'*20} {'-'*8} {'-'*10} {'-'*10}")
    for r in all_results:
        print(f"  {r['task']:<8} {r['method']:<20} {r['total_nf']:>8} "
              f"{r['final_acc']:>10.2f} {r['sec']:>10.1f}")

    # Per-generation progression
    print()
    print("  Per-generation progression (generational method):")
    for r in all_results:
        if r.get("method") == "generational":
            print(f"    {r['task']}:")
            for g in r["gens"]:
                print(f"      Gen {g['gen']}: +{g['nf_added']} -> total {g['total_nf']}  "
                      f"float={g['acc_float']:.2f}  int4={g['acc_int4']:.2f}")

    print()
    print(f"  Total wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
