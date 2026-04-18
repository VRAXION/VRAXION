"""Staged int4 freeze with L-BFGS: one weight at a time.

Protocol:
  1. Train float32 with L-BFGS → 100% lossless
  2. Loop (576 weights, ~6 per batch = ~96 steps):
     a. Find unfrozen weight closest to int4 grid
     b. Freeze it (clamp to {-7..+7} * scale)
     c. Re-optimize remaining with L-BFGS (quick, 30 iters)
     d. Check lossless — report if it drops
  3. Final: all weights int4, measure everything

Compare with bulk int4 QAT (95.3%) — can staged beat it?
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
BATCH_SIZE = 6  # freeze 6 weights per step (~1%)


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
    def __init__(self):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.randn(N_BITS, H) * 0.3)
        self.b1 = torch.nn.Parameter(torch.zeros(H))
        self.W2 = torch.nn.Parameter(torch.randn(H, OUT_DIM) * 0.3)
        self.b2 = torch.nn.Parameter(torch.zeros(OUT_DIM))
        self.c = torch.nn.Parameter(torch.ones(H))
        self.rho = torch.nn.Parameter(torch.full((H,), 4.0))

        n1 = N_BITS * H
        n2 = H * OUT_DIM
        self.register_buffer('frozen_mask', torch.zeros(n1 + n2))  # flat mask
        self.register_buffer('frozen_vals', torch.zeros(n1 + n2))  # frozen int4 values
        self.n_w1 = n1
        self.n_total = n1 + n2

    def _get_scales(self):
        s1 = self.W1.abs().max().detach().clamp(min=1e-8) / 7.0
        s2 = self.W2.abs().max().detach().clamp(min=1e-8) / 7.0
        return s1, s2

    def _effective_weights(self):
        s1, s2 = self._get_scales()
        W1_flat = self.W1.reshape(-1)
        W2_flat = self.W2.reshape(-1)
        all_w = torch.cat([W1_flat, W2_flat])

        scales = torch.cat([s1.expand(self.n_w1), s2.expand(self.n_total - self.n_w1)])
        frozen_w = self.frozen_vals * scales

        effective = torch.where(self.frozen_mask.bool(), frozen_w, all_w)
        W1_eff = effective[:self.n_w1].reshape(N_BITS, H)
        W2_eff = effective[self.n_w1:].reshape(H, OUT_DIM)
        return W1_eff, W2_eff

    def encode(self, x):
        W1, W2 = self._effective_weights()
        return c19_vec(x @ W1 + self.b1, self.c, self.rho) @ W2 + self.b2

    def decode(self, z):
        W1, W2 = self._effective_weights()
        return z @ W2.t() @ W1.t()

    def freeze_closest(self, n_to_freeze):
        """Freeze n weights closest to int4 grid."""
        s1, s2 = self._get_scales()
        W1_flat = self.W1.detach().reshape(-1)
        W2_flat = self.W2.detach().reshape(-1)
        all_w = torch.cat([W1_flat, W2_flat])
        scales = torch.cat([s1.expand(self.n_w1), s2.expand(self.n_total - self.n_w1)])

        # Distance to nearest int4 grid point
        q = torch.clamp(torch.round(all_w / scales), -7, 7)
        dist = (all_w - q * scales).abs()

        # Only consider unfrozen
        dist[self.frozen_mask.bool()] = float('inf')

        # Freeze closest n
        _, idx = dist.topk(min(n_to_freeze, int((~self.frozen_mask.bool()).sum().item())), largest=False)
        for i in idx:
            i = i.item()
            self.frozen_mask[i] = 1
            self.frozen_vals[i] = q[i]

        # Zero gradients on frozen positions
        return int(self.frozen_mask.sum().item())

    def zero_frozen_grads(self):
        """Zero out gradients for frozen weights."""
        if self.W1.grad is not None:
            mask1 = self.frozen_mask[:self.n_w1].reshape(N_BITS, H)
            self.W1.grad[mask1.bool()] = 0
        if self.W2.grad is not None:
            mask2 = self.frozen_mask[self.n_w1:].reshape(H, OUT_DIM)
            self.W2.grad[mask2.bool()] = 0

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
    """Quick L-BFGS re-optimization of unfrozen weights."""
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


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}")
    print(f"Staged int4 freeze: C19 + L-BFGS, H={H}, out={OUT_DIM}")
    print(f"Freeze {BATCH_SIZE} weights/step, L-BFGS retrain after each\n")

    cur, nxt = load_bigrams(corpus, 200_000)
    cur_d, nxt_d = cur.to(DEVICE), nxt.to(DEVICE)

    torch.manual_seed(SEED)
    unit = ByteUnitStaged().to(DEVICE)
    V = torch.nn.Parameter(torch.randn(OUT_DIM, VOCAB, device=DEVICE) * 0.1)

    # Phase 1: float32 training
    print("Phase 1: Float32 L-BFGS training...")
    t0 = time.time()
    lbfgs_retrain(unit, cur_d, nxt_d, V, max_iter=200)
    ba, mi = eval_lossless(unit)
    t1 = time.time()
    print(f"  Float32: lossless={ba:.2f}%  missed={mi}  [{t1-t0:.1f}s]")

    if ba < 100.0:
        print("  WARNING: float32 didn't reach 100%, trying harder...")
        lbfgs_retrain(unit, cur_d, nxt_d, V, max_iter=300)
        ba, mi = eval_lossless(unit)
        print(f"  Float32 retry: lossless={ba:.2f}%  missed={mi}")

    # Phase 2: staged int4 freeze
    print(f"\nPhase 2: Staged int4 freeze ({unit.n_total} weights, {BATCH_SIZE}/step)...")
    print(f"{'step':>5} {'frozen%':>8} {'frozen_n':>9} {'lossless':>10} {'missed':>8}")
    print("-" * 50)

    step = 0
    t2 = time.time()
    history = []

    while unit.frozen_pct() < 100.0:
        n_frozen = unit.freeze_closest(BATCH_SIZE)
        pct = unit.frozen_pct()

        # Retrain remaining
        lbfgs_retrain(unit, cur_d, nxt_d, V, max_iter=30)

        ba, mi = eval_lossless(unit)
        step += 1
        history.append({"step": step, "pct": pct, "frozen": n_frozen,
                        "byte_acc": ba, "missed": mi})

        if step % 10 == 0 or pct >= 99 or ba < 100.0:
            ll = "PASS" if ba == 100.0 else f"{ba:.1f}%"
            print(f"{step:>5} {pct:>7.1f}% {n_frozen:>9} {ll:>10} {mi:>8}")

    t3 = time.time()

    # Final eval
    ba_final, mi_final = eval_lossless(unit)
    ds_final = eval_downstream(unit, corpus)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULT: Staged int4")
    print(f"{'='*60}")
    ll = "PASS" if ba_final == 100.0 else f"{ba_final:.2f}%"
    print(f"  Lossless:    {ll}  (missed={mi_final})")
    print(f"  Downstream:  {ds_final:.2f}%")
    print(f"  Phase 1:     {t1-t0:.1f}s")
    print(f"  Phase 2:     {t3-t2:.1f}s ({step} steps)")
    print(f"  Total:       {t3-t0:.1f}s")
    print()
    print(f"  COMPARISON:")
    print(f"  {'method':<25} {'lossless':>10} {'downstream':>12} {'storage':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Bulk int4 QAT':25} {'95.3%':>10} {'34.54%':>12} {'288B':>10}")
    print(f"  {'Staged int4 L-BFGS':25} {ll:>10} {ds_final:>11.2f}% {'288B':>10}")
    print(f"  {'Bulk int5 QAT':25} {'100.0%':>10} {'38.08%':>12} {'360B':>10}")
    print(f"  {'Bulk int8 QAT':25} {'100.0%':>10} {'41.74%':>12} {'576B':>10}")

    # Show where lossless first dropped below 100% (if ever)
    drops = [h for h in history if h["byte_acc"] < 100.0]
    if drops:
        first_drop = drops[0]
        print(f"\n  First drop below 100%: step {first_drop['step']} at {first_drop['pct']:.1f}% frozen")
    else:
        print(f"\n  >>> 100% LOSSLESS MAINTAINED THROUGHOUT ALL {step} STEPS!")


if __name__ == "__main__":
    main()
