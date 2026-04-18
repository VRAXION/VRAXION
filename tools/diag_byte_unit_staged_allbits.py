"""Staged L-BFGS freeze at ALL bitwidths: int4, int3, ternary, binary.

Same protocol that gave 100% on int4 — does it hold at lower precisions?

Protocol per bitwidth:
  1. Float32 L-BFGS training → 100% lossless
  2. Staged freeze: 6 weights/step, L-BFGS retrain after each
  3. Final: all weights at target precision

Architecture: C19, H=24, out=16, L-BFGS.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BITS = 8
H = 24
OUT_DIM = 16
SEED = 42
VOCAB = 256
CTX = 8
MASK_POS = 4
BATCH_SIZE = 6


def byte_to_bits(b):
    bits = torch.zeros(b.shape[0], N_BITS, device=b.device)
    for i in range(N_BITS):
        bits[:, i] = (b >> i) & 1
    return bits

def load_bigrams(path, n):
    raw = Path(path).read_bytes()
    arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    gen = torch.Generator().manual_seed(SEED)
    offs = torch.randint(0, len(raw) - 2, (n,), generator=gen)
    return arr[offs].long(), arr[offs + 1].long()

def c19_vec(x, c, rho):
    c_s = c.clamp(min=0.1)
    rho_s = rho.clamp(min=0.0)
    L = 6.0 * c_s
    scaled = x / c_s
    n = scaled.floor()
    t = scaled - n
    h_val = t * (1.0 - t)
    sgn = torch.where(n.long() % 2 == 0, torch.ones_like(n), -torch.ones_like(n))
    interior = c_s * (sgn * h_val + rho_s * h_val * h_val)
    return torch.where(x >= L, x - L, torch.where(x <= -L, x + L, interior))


class ByteUnitStaged(torch.nn.Module):
    def __init__(self, target_bits):
        super().__init__()
        self.target_bits = target_bits
        self.max_val = max((2 ** (target_bits - 1)) - 1, 1)
        if target_bits == 1:
            self.max_val = 1

        self.W1 = torch.nn.Parameter(torch.randn(N_BITS, H) * 0.3)
        self.b1 = torch.nn.Parameter(torch.zeros(H))
        self.W2 = torch.nn.Parameter(torch.randn(H, OUT_DIM) * 0.3)
        self.b2 = torch.nn.Parameter(torch.zeros(OUT_DIM))
        self.c = torch.nn.Parameter(torch.ones(H))
        self.rho = torch.nn.Parameter(torch.full((H,), 4.0))

        n1 = N_BITS * H
        n2 = H * OUT_DIM
        self.register_buffer('frozen_mask', torch.zeros(n1 + n2))
        self.register_buffer('frozen_vals', torch.zeros(n1 + n2))
        self.n_w1 = n1
        self.n_total = n1 + n2

    def _get_scales(self):
        s1 = self.W1.abs().max().detach().clamp(min=1e-8) / self.max_val
        s2 = self.W2.abs().max().detach().clamp(min=1e-8) / self.max_val
        return s1, s2

    def _effective_weights(self):
        s1, s2 = self._get_scales()
        W1_flat = self.W1.reshape(-1)
        W2_flat = self.W2.reshape(-1)
        all_w = torch.cat([W1_flat, W2_flat])
        scales = torch.cat([s1.expand(self.n_w1), s2.expand(self.n_total - self.n_w1)])
        frozen_w = self.frozen_vals * scales
        effective = torch.where(self.frozen_mask.bool(), frozen_w, all_w)
        return effective[:self.n_w1].reshape(N_BITS, H), effective[self.n_w1:].reshape(H, OUT_DIM)

    def encode(self, x):
        W1, W2 = self._effective_weights()
        return c19_vec(x @ W1 + self.b1, self.c, self.rho) @ W2 + self.b2

    def decode(self, z):
        W1, W2 = self._effective_weights()
        return z @ W2.t() @ W1.t()

    def freeze_closest(self, n_to_freeze):
        s1, s2 = self._get_scales()
        W1_flat = self.W1.detach().reshape(-1)
        W2_flat = self.W2.detach().reshape(-1)
        all_w = torch.cat([W1_flat, W2_flat])
        scales = torch.cat([s1.expand(self.n_w1), s2.expand(self.n_total - self.n_w1)])
        q = torch.clamp(torch.round(all_w / scales), -self.max_val, self.max_val)
        dist = (all_w - q * scales).abs()
        dist[self.frozen_mask.bool()] = float('inf')
        n_avail = int((~self.frozen_mask.bool()).sum().item())
        k = min(n_to_freeze, n_avail)
        if k == 0:
            return int(self.frozen_mask.sum().item())
        _, idx = dist.topk(k, largest=False)
        for i in idx:
            self.frozen_mask[i.item()] = 1
            self.frozen_vals[i.item()] = q[i.item()]
        return int(self.frozen_mask.sum().item())

    def zero_frozen_grads(self):
        if self.W1.grad is not None:
            self.W1.grad[self.frozen_mask[:self.n_w1].reshape(N_BITS, H).bool()] = 0
        if self.W2.grad is not None:
            self.W2.grad[self.frozen_mask[self.n_w1:].reshape(H, OUT_DIM).bool()] = 0

    def frozen_pct(self):
        return self.frozen_mask.sum().item() / self.n_total * 100


def eval_lossless(unit):
    with torch.no_grad():
        all_b = torch.arange(256, device=DEVICE)
        bits = byte_to_bits(all_b).float()
        inp = bits * 2.0 - 1.0
        pred = (unit.decode(unit.encode(inp)) > 0).float()
        byte_acc = (pred == bits).all(dim=1).float().mean().item() * 100
        missed = int((~(pred == bits).all(dim=1)).sum().item())
    return byte_acc, missed


def eval_downstream(unit, corpus_path):
    raw_bytes = Path(corpus_path).read_bytes()
    arr = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
    def sample(n, seed):
        gen = torch.Generator().manual_seed(seed)
        offs = torch.randint(0, len(arr) - CTX - 1, (n,), generator=gen)
        idx_mat = offs.unsqueeze(1) + torch.arange(CTX).unsqueeze(0)
        chunks = arr[idx_mat].long()
        targets = chunks[:, MASK_POS].clone()
        chunks[:, MASK_POS] = 32
        return chunks.to(DEVICE), targets.to(DEVICE)
    train_x, train_y = sample(20000, 42)
    eval_x, eval_y = sample(5000, 99)
    with torch.no_grad():
        def embed(chunks):
            flat = chunks.flatten()
            b = byte_to_bits(flat).float() * 2.0 - 1.0
            lat = unit.encode(b)
            return lat.view(chunks.shape[0], CTX, -1).reshape(chunks.shape[0], -1)
        train_feat = embed(train_x)
        eval_feat = embed(eval_x)
    D = train_feat.shape[1]
    torch.manual_seed(SEED)
    P = torch.nn.Parameter(torch.randn(D, VOCAB, device=DEVICE) * 0.01)
    pb = torch.nn.Parameter(torch.zeros(VOCAB, device=DEVICE))
    opt = torch.optim.Adam([P, pb], lr=0.005)
    for _ in range(100):
        loss = F.cross_entropy(train_feat @ P + pb, train_y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (eval_feat @ P + pb).argmax(1).eq(eval_y).float().mean().item() * 100


def lbfgs_retrain(unit, cur_d, nxt_d, V, max_iter=30):
    all_params = list(unit.parameters()) + [V]
    opt = torch.optim.LBFGS(all_params, lr=0.5, max_iter=15, line_search_fn="strong_wolfe",
                             history_size=30, tolerance_grad=1e-9, tolerance_change=1e-12)
    all_b = torch.arange(256, device=DEVICE)
    all_bits = byte_to_bits(all_b).float()
    all_inp = all_bits * 2.0 - 1.0
    n_ctx = min(50000, cur_d.shape[0])
    ctx_inp = byte_to_bits(cur_d[:n_ctx]).float() * 2.0 - 1.0
    ctx_nxt = nxt_d[:n_ctx]

    for outer in range(max_iter):
        def closure():
            opt.zero_grad()
            latent_all = unit.encode(all_inp)
            loss_rec = F.binary_cross_entropy_with_logits(unit.decode(latent_all), all_bits)
            latent_ctx = unit.encode(ctx_inp)
            loss_ctx = F.cross_entropy(latent_ctx @ V, ctx_nxt)
            loss = loss_rec + 0.1 * loss_ctx
            loss.backward()
            unit.zero_frozen_grads()
            return loss
        opt.step(closure)


def run_bitwidth(bits, name, cur_d, nxt_d, corpus):
    torch.manual_seed(SEED)
    unit = ByteUnitStaged(bits).to(DEVICE)
    V = torch.nn.Parameter(torch.randn(OUT_DIM, VOCAB, device=DEVICE) * 0.1)

    # Phase 1: float32
    t0 = time.time()
    lbfgs_retrain(unit, cur_d, nxt_d, V, max_iter=200)
    ba0, _ = eval_lossless(unit)

    # Phase 2: staged freeze
    first_drop_pct = None
    step = 0
    while unit.frozen_pct() < 100.0:
        unit.freeze_closest(BATCH_SIZE)
        lbfgs_retrain(unit, cur_d, nxt_d, V, max_iter=30)
        ba, mi = eval_lossless(unit)
        step += 1
        if ba < 100.0 and first_drop_pct is None:
            first_drop_pct = unit.frozen_pct()

    elapsed = time.time() - t0
    ba_final, mi_final = eval_lossless(unit)
    ds_final = eval_downstream(unit, corpus)

    n_weights = unit.n_total
    if bits == 1:
        storage = (n_weights + 7) // 8
    else:
        storage = (n_weights * bits + 7) // 8

    return {
        "bits": bits, "name": name, "storage": storage,
        "byte_acc": ba_final, "missed": mi_final,
        "eval_acc": ds_final, "time": elapsed,
        "first_drop": first_drop_pct, "steps": step,
    }


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}")
    print(f"Staged L-BFGS freeze: int4 / int3 / ternary / binary")
    print(f"C19, H={H}, out={OUT_DIM}, {BATCH_SIZE} weights/step\n")

    cur, nxt = load_bigrams(corpus, 200_000)
    cur_d, nxt_d = cur.to(DEVICE), nxt.to(DEVICE)

    configs = [
        (4, "int4"),
        (3, "int3"),
        (2, "ternary"),
        (1, "binary"),
    ]

    results = []
    for bits, name in configs:
        print(f">>> {name} (max_val={max((2**(bits-1))-1, 1)})...")
        r = run_bitwidth(bits, name, cur_d, nxt_d, corpus)
        ll = "PASS" if r["byte_acc"] == 100.0 else f"{r['byte_acc']:.1f}%"
        drop_str = f"first drop at {r['first_drop']:.1f}%" if r['first_drop'] else "never dropped"
        print(f"    lossless={ll}  missed={r['missed']}  downstream={r['eval_acc']:.2f}%  "
              f"storage={r['storage']}B  [{r['time']:.1f}s]  ({drop_str})\n")
        results.append(r)

    print(f"\n{'='*78}")
    print(f"  STAGED L-BFGS FREEZE: FULL BITWIDTH COMPARISON")
    print(f"{'='*78}")
    print(f"{'precision':<10} {'storage':>8} {'compress':>10} {'lossless':>10} {'downstream':>12} {'drop_at':>10}")
    print(f"{'-'*78}")

    # Add reference rows
    refs = [
        {"name": "float32",     "storage": 2304, "byte_acc": 100.0, "eval_acc": 41.78, "first_drop": None},
        {"name": "int8 bulk",   "storage": 576,  "byte_acc": 100.0, "eval_acc": 41.74, "first_drop": None},
        {"name": "int5 bulk",   "storage": 360,  "byte_acc": 100.0, "eval_acc": 38.08, "first_drop": None},
        {"name": "int4 bulk",   "storage": 288,  "byte_acc": 95.3,  "eval_acc": 34.54, "first_drop": None},
    ]

    for ref in refs:
        ll = "100%" if ref["byte_acc"] == 100.0 else f"{ref['byte_acc']:.1f}%"
        comp = f"{2304/ref['storage']:.0f}x"
        print(f"{ref['name']:<10} {ref['storage']:>7}B {comp:>10} {ll:>10} {ref['eval_acc']:>11.2f}% {'—':>10}")

    print(f"{'-'*78}")
    for r in results:
        ll = "100%" if r["byte_acc"] == 100.0 else f"{r['byte_acc']:.1f}%"
        comp = f"{2304/r['storage']:.0f}x"
        drop = f"{r['first_drop']:.0f}%" if r['first_drop'] else "never"
        star = " ***" if r["byte_acc"] == 100.0 else ""
        print(f"{r['name']+'*':<10} {r['storage']:>7}B {comp:>10} {ll:>10} {r['eval_acc']:>11.2f}% {drop:>10}{star}")

    print(f"\n  * = staged L-BFGS (not bulk QAT)")


if __name__ == "__main__":
    main()
