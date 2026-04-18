"""Extract the winner int4 weights from staged L-BFGS byte unit.

Train → staged freeze → print all weights in int4 grid values.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
import json

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BITS = 8
H = 24
OUT_DIM = 16
SEED = 42
VOCAB = 256
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
    def __init__(self):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.randn(N_BITS, H) * 0.3)
        self.b1 = torch.nn.Parameter(torch.zeros(H))
        self.W2 = torch.nn.Parameter(torch.randn(H, OUT_DIM) * 0.3)
        self.b2 = torch.nn.Parameter(torch.zeros(OUT_DIM))
        self.c = torch.nn.Parameter(torch.ones(H))
        self.rho = torch.nn.Parameter(torch.full((H,), 4.0))
        self.max_val = 7
        n1 = N_BITS * H
        n2 = H * OUT_DIM
        self.register_buffer('frozen_mask', torch.zeros(n1 + n2))
        self.register_buffer('frozen_vals', torch.zeros(n1 + n2))
        self.n_w1 = n1
        self.n_total = n1 + n2

    def _get_scales(self):
        s1 = self.W1.abs().max().detach().clamp(min=1e-8) / 7.0
        s2 = self.W2.abs().max().detach().clamp(min=1e-8) / 7.0
        return s1, s2

    def _effective_weights(self):
        s1, s2 = self._get_scales()
        all_w = torch.cat([self.W1.reshape(-1), self.W2.reshape(-1)])
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
        all_w = torch.cat([self.W1.detach().reshape(-1), self.W2.detach().reshape(-1)])
        scales = torch.cat([s1.expand(self.n_w1), s2.expand(self.n_total - self.n_w1)])
        q = torch.clamp(torch.round(all_w / scales), -7, 7)
        dist = (all_w - q * scales).abs()
        dist[self.frozen_mask.bool()] = float('inf')
        n_avail = int((~self.frozen_mask.bool()).sum().item())
        k = min(n_to_freeze, n_avail)
        if k == 0: return
        _, idx = dist.topk(k, largest=False)
        for i in idx:
            self.frozen_mask[i.item()] = 1
            self.frozen_vals[i.item()] = q[i.item()]

    def zero_frozen_grads(self):
        if self.W1.grad is not None:
            self.W1.grad[self.frozen_mask[:self.n_w1].reshape(N_BITS, H).bool()] = 0
        if self.W2.grad is not None:
            self.W2.grad[self.frozen_mask[self.n_w1:].reshape(H, OUT_DIM).bool()] = 0

    def frozen_pct(self):
        return self.frozen_mask.sum().item() / self.n_total * 100

    def get_int4_weights(self):
        """Extract final int4 grid values."""
        vals = self.frozen_vals
        W1_int = vals[:self.n_w1].reshape(N_BITS, H).long().cpu()
        W2_int = vals[self.n_w1:].reshape(H, OUT_DIM).long().cpu()
        return W1_int, W2_int


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


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}")
    print(f"Training winner byte unit and extracting int4 weights...\n")

    cur, nxt = load_bigrams(corpus, 200_000)
    cur_d, nxt_d = cur.to(DEVICE), nxt.to(DEVICE)

    torch.manual_seed(SEED)
    unit = ByteUnitStaged().to(DEVICE)
    V = torch.nn.Parameter(torch.randn(OUT_DIM, VOCAB, device=DEVICE) * 0.1)

    # Phase 1
    print("Phase 1: Float32 L-BFGS...")
    lbfgs_retrain(unit, cur_d, nxt_d, V, max_iter=200)
    with torch.no_grad():
        all_b = torch.arange(256, device=DEVICE)
        bits = byte_to_bits(all_b).float()
        inp = bits * 2.0 - 1.0
        pred = (unit.decode(unit.encode(inp)) > 0).float()
        ba = (pred == bits).all(dim=1).float().mean().item() * 100
    print(f"  Float32 lossless: {ba:.2f}%")

    # Phase 2: staged freeze
    print("Phase 2: Staged int4 freeze...")
    step = 0
    while unit.frozen_pct() < 100.0:
        unit.freeze_closest(BATCH_SIZE)
        lbfgs_retrain(unit, cur_d, nxt_d, V, max_iter=30)
        step += 1
        if step % 20 == 0:
            pct = unit.frozen_pct()
            print(f"  step {step}: {pct:.1f}% frozen")

    # Final check
    with torch.no_grad():
        pred = (unit.decode(unit.encode(inp)) > 0).float()
        ba = (pred == bits).all(dim=1).float().mean().item() * 100
        missed = int((~(pred == bits).all(dim=1)).sum().item())
    print(f"\nFinal lossless: {ba:.2f}%  missed={missed}")

    # Extract weights
    W1_int, W2_int = unit.get_int4_weights()
    s1, s2 = unit._get_scales()
    b1 = unit.b1.detach().cpu()
    b2 = unit.b2.detach().cpu()
    c = unit.c.detach().cpu()
    rho = unit.rho.detach().cpu()

    print(f"\n{'='*70}")
    print(f"  WINNER BYTE UNIT: int4 weights")
    print(f"{'='*70}")

    print(f"\n  Scale W1: {s1.item():.6f}")
    print(f"  Scale W2: {s2.item():.6f}")

    print(f"\n  W1 (8 x 24) — int4 grid values [-7..+7]:")
    print(f"  {'':>6}", end="")
    for j in range(H):
        print(f" n{j:02d}", end="")
    print()
    for i in range(N_BITS):
        print(f"  b{i}: ", end="")
        for j in range(H):
            v = W1_int[i, j].item()
            print(f" {v:>3}", end="")
        print()

    print(f"\n  W2 (24 x 16) — int4 grid values [-7..+7]:")
    print(f"  {'':>6}", end="")
    for j in range(OUT_DIM):
        print(f"  z{j:02d}", end="")
    print()
    for i in range(H):
        print(f"  n{i:02d}: ", end="")
        for j in range(OUT_DIM):
            v = W2_int[i, j].item()
            print(f" {v:>3}", end="")
        print()

    print(f"\n  Biases b1 (24 neurons):")
    print(f"  ", [f"{v:.4f}" for v in b1.tolist()])

    print(f"\n  Biases b2 (16 outputs):")
    print(f"  ", [f"{v:.4f}" for v in b2.tolist()])

    print(f"\n  C19 c (24 neurons):")
    print(f"  ", [f"{v:.3f}" for v in c.tolist()])

    print(f"\n  C19 rho (24 neurons):")
    print(f"  ", [f"{v:.3f}" for v in rho.tolist()])

    # Stats
    w1_flat = W1_int.flatten().tolist()
    w2_flat = W2_int.flatten().tolist()
    all_w = w1_flat + w2_flat
    print(f"\n  Weight distribution:")
    for v in range(-7, 8):
        cnt = all_w.count(v)
        bar = "#" * (cnt // 3)
        print(f"    {v:>3}: {cnt:>4} {bar}")

    n_zero = all_w.count(0)
    print(f"\n  Sparsity: {n_zero}/{len(all_w)} zeros ({n_zero/len(all_w)*100:.1f}%)")
    print(f"  Total weights: {len(all_w)}")
    print(f"  Storage: {len(all_w) * 4 // 8} bytes (int4 packed)")

    # Save as JSON
    export = {
        "architecture": "C19 1H, 8->24->16, tied mirror",
        "precision": "int4",
        "lossless": f"{ba:.2f}%",
        "scale_W1": s1.item(),
        "scale_W2": s2.item(),
        "W1_int4": W1_int.tolist(),
        "W2_int4": W2_int.tolist(),
        "bias1": b1.tolist(),
        "bias2": b2.tolist(),
        "c19_c": c.tolist(),
        "c19_rho": rho.tolist(),
    }
    out_path = "tools/byte_unit_winner_int4.json"
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n  Exported to: {out_path}")


if __name__ == "__main__":
    main()
