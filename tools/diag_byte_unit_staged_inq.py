"""Byte unit with staged INQ: float32 first, then gradually freeze to int8.

Protocol:
  Phase 1: Train fully float32 (no quantization) until 100% lossless + best downstream
  Phase 2: Staged INQ — 10 rounds, each round:
    - Sort unfrozen weights by distance to nearest int8 grid point
    - Freeze closest 10% to int8
    - Retrain remaining float weights for N epochs
  Final: all weights int8, measure lossless + downstream

Architecture: 1H SiLU 8->32->16 and 1H GELU 8->32->16.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BITS = 8
OUT_DIM = 16
H = 32
SEED = 42
BATCH = 8192
N_BIGRAMS = 200_000
LR = 0.01
VOCAB = 256
CTX = 8
MASK_POS = 4

# Phase 1 config
P1_EPOCHS = 60
# Phase 2 config
INQ_ROUNDS = 10
INQ_EPOCHS_PER_ROUND = 15
FREEZE_FRAC = 0.10  # freeze 10% of remaining per round


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


class ByteUnit(torch.nn.Module):
    def __init__(self, act="silu"):
        super().__init__()
        self.act_name = act
        self.W1 = torch.nn.Parameter(torch.randn(N_BITS, H) * 0.3)
        self.b1 = torch.nn.Parameter(torch.zeros(H))
        self.W2 = torch.nn.Parameter(torch.randn(H, OUT_DIM) * 0.3)
        self.b2 = torch.nn.Parameter(torch.zeros(OUT_DIM))

        # Frozen masks (0 = float, 1 = frozen to int8)
        self.register_buffer('mask_W1', torch.zeros(N_BITS, H))
        self.register_buffer('mask_W2', torch.zeros(H, OUT_DIM))
        # Frozen int8 values
        self.register_buffer('frozen_W1', torch.zeros(N_BITS, H))
        self.register_buffer('frozen_W2', torch.zeros(H, OUT_DIM))
        # Scale factors (computed once at freeze time)
        self.register_buffer('scale_W1', torch.ones(1))
        self.register_buffer('scale_W2', torch.ones(1))

    def act(self, x):
        if self.act_name == "silu": return F.silu(x)
        if self.act_name == "gelu": return F.gelu(x)
        return F.relu(x)

    def get_effective_W(self, W, mask, frozen, scale):
        """Mix frozen int8 and live float weights."""
        # frozen positions use quantized value, unfrozen use live float
        return torch.where(mask.bool(), frozen * scale, W)

    def encode(self, x):
        W1_eff = self.get_effective_W(self.W1, self.mask_W1, self.frozen_W1, self.scale_W1)
        W2_eff = self.get_effective_W(self.W2, self.mask_W2, self.frozen_W2, self.scale_W2)
        return self.act(x @ W1_eff + self.b1) @ W2_eff + self.b2

    def decode(self, z):
        W1_eff = self.get_effective_W(self.W1, self.mask_W1, self.frozen_W1, self.scale_W1)
        W2_eff = self.get_effective_W(self.W2, self.mask_W2, self.frozen_W2, self.scale_W2)
        return z @ W2_eff.t() @ W1_eff.t()

    def freeze_closest_n(self, n_to_freeze):
        """Freeze the n weights closest to their int8 grid point."""
        candidates = []

        for name, W, mask, frozen, scale_buf in [
            ("W1", self.W1, self.mask_W1, self.frozen_W1, "scale_W1"),
            ("W2", self.W2, self.mask_W2, self.frozen_W2, "scale_W2"),
        ]:
            scale = W.abs().max().detach().clamp(min=1e-8) / 127.0
            setattr(self, scale_buf, scale.unsqueeze(0))

            q = torch.clamp(torch.round(W.detach() / scale), -127, 127)
            dist = (W.detach() - q * scale).abs()

            unfrozen = (mask == 0).nonzero(as_tuple=False)
            for idx in unfrozen:
                i, j = idx[0].item(), idx[1].item()
                candidates.append((dist[i, j].item(), name, i, j, q[i, j].item()))

        # Sort by distance (closest to grid = easiest to freeze)
        candidates.sort(key=lambda x: x[0])

        n_to_freeze = min(n_to_freeze, len(candidates))
        for k in range(n_to_freeze):
            _, name, i, j, q_val = candidates[k]
            if name == "W1":
                self.mask_W1[i, j] = 1
                self.frozen_W1[i, j] = q_val
            else:
                self.mask_W2[i, j] = 1
                self.frozen_W2[i, j] = q_val

        total_w = self.W1.numel() + self.W2.numel()
        total_frozen = self.mask_W1.sum().item() + self.mask_W2.sum().item()
        return int(total_frozen), total_w

    def frozen_pct(self):
        total = self.W1.numel() + self.W2.numel()
        frozen = self.mask_W1.sum().item() + self.mask_W2.sum().item()
        return frozen / total * 100


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


def run_staged_inq(act, cur, nxt, corpus):
    print(f"\n{'='*65}")
    print(f"  STAGED INQ: {act.upper()} 1H 8->{H}->16")
    print(f"{'='*65}")

    torch.manual_seed(SEED)
    unit = ByteUnit(act).to(DEVICE)
    V = torch.nn.Parameter(torch.randn(OUT_DIM, VOCAB, device=DEVICE) * 0.1)
    cur_d, nxt_d = cur.to(DEVICE), nxt.to(DEVICE)
    N = cur_d.shape[0]

    # ── Phase 1: full float training ──
    print(f"\n  Phase 1: float32 training ({P1_EPOCHS} epochs, dual loss)...")
    opt = torch.optim.Adam(list(unit.parameters()) + [V], lr=LR)
    t0 = time.time()

    for ep in range(P1_EPOCHS):
        perm = torch.randperm(N, device=DEVICE)
        for start in range(0, N, BATCH):
            idx = perm[start:start + BATCH]
            bits = byte_to_bits(cur_d[idx]).float()
            inp = bits * 2.0 - 1.0
            latent = unit.encode(inp)
            loss_rec = F.binary_cross_entropy_with_logits(unit.decode(latent), bits)
            loss_ctx = F.cross_entropy(latent @ V, nxt_d[idx])
            loss = loss_rec + 0.1 * loss_ctx
            opt.zero_grad(); loss.backward(); opt.step()

        if ep % 10 == 0 or ep == P1_EPOCHS - 1:
            ba, mi = eval_lossless(unit)
            print(f"    ep={ep:3d}  lossless={ba:.2f}%  missed={mi}")

    ba_float, mi_float = eval_lossless(unit)
    ds_float = eval_downstream(unit, corpus)
    t_p1 = time.time() - t0
    print(f"  Phase 1 DONE: lossless={ba_float:.2f}%  downstream={ds_float:.2f}%  [{t_p1:.1f}s]")

    # ── Phase 2: staged INQ ──
    print(f"\n  Phase 2: Staged INQ ({INQ_ROUNDS} rounds, {FREEZE_FRAC*100:.0f}% per round)...")
    total_w = unit.W1.numel() + unit.W2.numel()
    per_round = max(1, int(total_w * FREEZE_FRAC))

    t1 = time.time()
    for rnd in range(INQ_ROUNDS):
        # Freeze closest weights
        n_frozen, n_total = unit.freeze_closest_n(per_round)
        pct = n_frozen / n_total * 100

        # Retrain unfrozen weights (only unfrozen params get gradients)
        # We need to zero gradients for frozen positions manually
        opt_inq = torch.optim.Adam(
            [p for p in unit.parameters() if p.requires_grad] + [V],
            lr=LR * 0.5  # lower LR during INQ
        )

        for ep in range(INQ_EPOCHS_PER_ROUND):
            perm = torch.randperm(N, device=DEVICE)
            for start in range(0, N, BATCH):
                idx = perm[start:start + BATCH]
                bits = byte_to_bits(cur_d[idx]).float()
                inp = bits * 2.0 - 1.0
                latent = unit.encode(inp)
                loss_rec = F.binary_cross_entropy_with_logits(unit.decode(latent), bits)
                loss_ctx = F.cross_entropy(latent @ V, nxt_d[idx])
                loss = loss_rec + 0.1 * loss_ctx
                opt_inq.zero_grad()
                loss.backward()
                # Zero gradients on frozen positions
                if unit.W1.grad is not None:
                    unit.W1.grad[unit.mask_W1.bool()] = 0
                if unit.W2.grad is not None:
                    unit.W2.grad[unit.mask_W2.bool()] = 0
                opt_inq.step()

        ba, mi = eval_lossless(unit)
        print(f"    round={rnd+1:2d}  frozen={pct:5.1f}%  lossless={ba:.2f}%  missed={mi}")

    # Final: freeze ALL remaining to int8
    remaining = total_w - int(unit.mask_W1.sum().item() + unit.mask_W2.sum().item())
    if remaining > 0:
        unit.freeze_closest_n(remaining)
        print(f"    FINAL freeze: all {total_w} weights now int8")

    t_p2 = time.time() - t1
    ba_final, mi_final = eval_lossless(unit)
    ds_final = eval_downstream(unit, corpus)

    print(f"\n  RESULT: lossless={ba_final:.2f}%  downstream={ds_final:.2f}%")
    print(f"  Phase 2 time: {t_p2:.1f}s")
    print(f"  Total: {t_p1 + t_p2:.1f}s")

    return {
        "act": act,
        "float_lossless": ba_float, "float_downstream": ds_float,
        "final_lossless": ba_final, "final_downstream": ds_final,
        "final_missed": mi_final,
    }


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}")
    print(f"Staged INQ for byte unit: float32 first, then gradually freeze to int8")

    cur, nxt = load_bigrams(corpus, N_BIGRAMS)

    results = []
    for act in ["silu", "gelu"]:
        r = run_staged_inq(act, cur, nxt, corpus)
        results.append(r)

    print(f"\n{'='*70}")
    print(f"  SUMMARY: Staged INQ vs previous results")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'Lossless':>10} {'Downstream':>12}")
    print(f"{'-'*70}")
    print(f"{'Linear tied (no neuron)':<30} {'100.0%':>10} {'35.54%':>12}")
    print(f"{'SiLU rw=10 (brute force)':<30} {'100.0%':>10} {'39.64%':>12}")
    for r in results:
        ll = f"{r['final_lossless']:.1f}%"
        if r['final_lossless'] == 100.0: ll = "100.0%"
        print(f"{'Staged INQ ' + r['act'].upper():<30} {ll:>10} {r['final_downstream']:>11.2f}%")
    print(f"{'SiLU rw=1 (no fix)':<30} {'99.6%':>10} {'42.56%':>12}")
    print(f"{'GELU rw=1 (no fix)':<30} {'97.7%':>10} {'43.24%':>12}")


if __name__ == "__main__":
    main()
