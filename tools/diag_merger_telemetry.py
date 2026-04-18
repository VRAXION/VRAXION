"""Telemetry check: are we actually training different models or is
something stuck/baked in?

Checks:
  1. Are init weights actually different for different H?
  2. Do trained models produce different outputs for same input?
  3. Is the LUT loaded correctly (matches expected)?
  4. Is the seed causing some degenerate init?
  5. Are ALL H values really hitting the same 19.2% minimum?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LUT_PATH = Path(__file__).with_name("byte_embedder_lut_int8_nozero.json")


class C19Activation(nn.Module):
    def __init__(self, dim, c_init=1.0, rho_init=8.0):
        super().__init__()
        self.c_raw = nn.Parameter(torch.full((dim,), c_init))
        self.rho_raw = nn.Parameter(torch.full((dim,), rho_init))
    def forward(self, x):
        c = self.c_raw.clamp(min=0.1)
        rho = self.rho_raw.clamp(min=0.0)
        L = 6.0 * c
        scaled = x / c
        n = scaled.floor()
        t = scaled - n
        h = t * (1.0 - t)
        sgn = torch.where(n.long() % 2 == 0,
                          torch.ones_like(n), -torch.ones_like(n))
        interior = c * (sgn * h + rho * h * h)
        return torch.where(x >= L, x - L,
               torch.where(x <= -L, x + L, interior))


class TiedMerger(nn.Module):
    def __init__(self, hidden, output_dim, input_dim=32):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.randn(hidden, output_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(output_dim))
        self.db1 = nn.Parameter(torch.zeros(hidden))
        self.db2 = nn.Parameter(torch.zeros(input_dim))
        self.c19 = C19Activation(hidden)
    def encode(self, x):
        return self.c19(x @ self.W1 + self.b1) @ self.W2 + self.b2
    def decode(self, z):
        return (z @ self.W2.t() + self.db1) @ self.W1.t() + self.db2
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def quick_train(m, data, epochs=50):
    opt = torch.optim.LBFGS(m.parameters(), lr=1.0, max_iter=20,
                            history_size=50, line_search_fn="strong_wolfe")
    for _ in range(epochs):
        def c():
            opt.zero_grad()
            x_hat, _ = m(data)
            loss = F.mse_loss(x_hat, data)
            loss.backward()
            return loss
        opt.step(c)
    with torch.no_grad():
        x_hat, z = m(data)
        sign_match = (torch.sign(x_hat) == torch.sign(data))
        lossless = sign_match.all(dim=1).float().mean().item() * 100
    return lossless, x_hat, z


def main():
    # Load data
    print("=== TELEMETRY CHECK ===\n")
    with open(LUT_PATH) as f:
        blob = json.load(f)
    scale = blob["scale"]
    lut_int8 = np.array(blob["lut"])
    lut = torch.tensor(blob["lut"], dtype=torch.float32) * scale

    print(f"LUT scale: {scale}")
    print(f"LUT shape: {lut.shape}")
    print(f"LUT zeros: {(lut_int8 == 0).sum()}  <-- should be 0 (nozero LUT)")
    print(f"LUT range: [{lut.min():.4f}, {lut.max():.4f}]")

    idx_a = torch.arange(256).unsqueeze(1).expand(256, 256).reshape(-1)
    idx_b = torch.arange(256).unsqueeze(0).expand(256, 256).reshape(-1)
    data = torch.cat([lut[idx_a], lut[idx_b]], dim=1).to(DEVICE)
    print(f"Pairs shape: {data.shape}")
    print(f"Pairs unique rows: {torch.unique(data, dim=0).shape[0]}")

    # --- Test 1: Different seeds, same H ---
    print("\n--- TEST 1: Does seed matter at H=64 out=16? ---")
    seeds = [0, 1, 42, 123, 999]
    for s in seeds:
        torch.manual_seed(s)
        m = TiedMerger(hidden=64, output_dim=16).to(DEVICE)
        # Check init
        w1_norm = m.W1.norm().item()
        ll_before = ((torch.sign(m(data)[0]) == torch.sign(data))
                     .all(dim=1).float().mean().item() * 100)
        ll, _, _ = quick_train(m, data, epochs=50)
        print(f"  seed={s:>4}: W1_norm_init={w1_norm:.4f}, "
              f"lossless_init={ll_before:.2f}%, lossless_trained={ll:.2f}%")

    # --- Test 2: Different H, same seed — are init weights actually different? ---
    print("\n--- TEST 2: Different H, init weight stats ---")
    for H in [32, 64, 128, 256, 512]:
        torch.manual_seed(42)
        m = TiedMerger(hidden=H, output_dim=16).to(DEVICE)
        print(f"  H={H:>4}: W1 shape={tuple(m.W1.shape)}, "
              f"W1 mean={m.W1.mean().item():.5f}, "
              f"W1 std={m.W1.std().item():.5f}, "
              f"W2 shape={tuple(m.W2.shape)}")

    # --- Test 3: Do trained models actually differ? ---
    print("\n--- TEST 3: Train 2 different H values, compare outputs ---")
    torch.manual_seed(42)
    m_small = TiedMerger(hidden=64, output_dim=16).to(DEVICE)
    ll_small, x_small, z_small = quick_train(m_small, data, epochs=100)

    torch.manual_seed(42)
    m_big = TiedMerger(hidden=512, output_dim=16).to(DEVICE)
    ll_big, x_big, z_big = quick_train(m_big, data, epochs=100)

    print(f"  H=64 lossless: {ll_small:.2f}%")
    print(f"  H=512 lossless: {ll_big:.2f}%")
    print(f"  Output L2 diff: {(x_small - x_big).norm().item():.4f}")
    print(f"  Latent L2 diff: {(z_small - z_big).norm().item():.4f}")
    print(f"  Same failing pairs? Comparing sign-match masks...")
    fail_small = ~(torch.sign(x_small) == torch.sign(data)).all(dim=1)
    fail_big = ~(torch.sign(x_big) == torch.sign(data)).all(dim=1)
    overlap = (fail_small & fail_big).sum().item()
    print(f"  H=64 fails: {fail_small.sum().item()} pairs")
    print(f"  H=512 fails: {fail_big.sum().item()} pairs")
    print(f"  Overlap: {overlap} ({overlap/max(fail_small.sum().item(),1)*100:.1f}% "
          f"of H=64 fails are also H=512 fails)")

    # --- Test 4: Are the same BYTES involved in failing pairs? ---
    print("\n--- TEST 4: Which bytes are involved in failing pairs? ---")
    # Indices of failing pairs
    fail_mask = fail_small.cpu().numpy()
    fail_idx = np.where(fail_mask)[0]
    # Each pair index = byte_a * 256 + byte_b
    byte_a = fail_idx // 256
    byte_b = fail_idx % 256
    from collections import Counter
    a_cnt = Counter(byte_a)
    b_cnt = Counter(byte_b)
    print(f"  Failing pairs: {len(fail_idx)}")
    print(f"  Top 10 bytes as 'A' in failing pairs:")
    for b, c in sorted(a_cnt.items(), key=lambda x: -x[1])[:10]:
        print(f"    byte {b:>3}: {c} failures")
    print(f"  Top 10 bytes as 'B' in failing pairs:")
    for b, c in sorted(b_cnt.items(), key=lambda x: -x[1])[:10]:
        print(f"    byte {b:>3}: {c} failures")

    # --- Test 5: Are the failing pairs similar in some way? ---
    print("\n--- TEST 5: Structure of failing pairs ---")
    fail_pairs = data[fail_mask].cpu().numpy()
    succ_pairs = data[~fail_mask].cpu().numpy()
    print(f"  Failing pairs abs-mean per dim: "
          f"[{np.abs(fail_pairs).mean():.4f}]")
    print(f"  Success pairs abs-mean per dim:  "
          f"[{np.abs(succ_pairs).mean():.4f}]")
    # Are failing pairs 'smaller' in magnitude on average?
    print(f"  Failing pairs L2 norm:  {np.linalg.norm(fail_pairs, axis=1).mean():.4f}")
    print(f"  Success pairs L2 norm:  {np.linalg.norm(succ_pairs, axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
