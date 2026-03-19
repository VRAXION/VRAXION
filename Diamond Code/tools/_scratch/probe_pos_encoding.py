#!/usr/bin/env python3
"""
Probe: Position Encoding A/B Test — TOT-H020 Confirmation
==========================================================
Tests whether sinusoidal position encoding enables a phi-EMA RNN
to learn echo256 (BLOCK=4). Two configs, same seed, same data.

Config A (no_pe):  No position encoding  → expect ~50% (random)
Config B (sin_pe): Sinusoidal PE added   → expect >>50%

Verdict:
  B > 70% AND A < 55%  → CONFIRMED (PE is the fix)
  Both < 55%           → INSUFFICIENT (need deeper changes)
  Both > 70%           → UNEXPECTED (phi-EMA can learn echo alone)
"""

import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── CONFIG ──────────────────────────────────────────────────
D             = 128       # embedding dim
DEPTH         = 4         # processing layers
SEQ_LEN       = 32        # 8 full BLOCK=4 cycles, ceiling 90.6%
BLOCK_SIZE    = 4         # echo256 block size
BATCH         = 16        # batch size
LR            = 1e-3      # Adam learning rate
STEPS         = 500       # training steps
SEEDS         = [42, 137] # two seeds for reproducibility
NUM_BITS      = 8         # binary bits mode
STEP_TIMEOUT  = 30        # seconds — kill if one step takes longer
DEVICE        = torch.device('cpu')

CONFIGS = [
    ('no_pe',  False),   # A: no position encoding
    ('sin_pe', True),    # B: sinusoidal position encoding
]

# ─── PATHS ───────────────────────────────────────────────────
DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_pos_encoding_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

# Clear live log
with open(LIVE_LOG, 'w') as f:
    f.write(f'# probe_pos_encoding — {time.strftime("%Y-%m-%d %H:%M:%S")}\n')


# ─── C19 ACTIVATION (from swarm_model.py:29) ────────────────
def c19_activation(x, rho=4.0):
    """C19 periodic parabolic wave activation."""
    l = 6.0 * math.pi
    inv_pi = 1.0 / math.pi
    scaled = x * inv_pi
    n = torch.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    is_even = torch.remainder(n, 2.0) < 1.0
    sgn = torch.where(is_even, torch.ones_like(x), -torch.ones_like(x))
    core = math.pi * (sgn * h + (rho * h * h))
    return torch.where(x >= l, x - l, torch.where(x <= -l, x + l, core))


# ─── DATA GENERATION ────────────────────────────────────────
def byte_to_bits(byte_seq, num_bits=8):
    """Convert byte sequence to binary bit tensor [T, num_bits]."""
    t = torch.tensor(byte_seq, dtype=torch.uint8)
    return ((t.unsqueeze(-1) >> torch.arange(num_bits)) & 1).float()


def make_echo_batch(batch_size, seq_len, block_size=4, num_bits=8):
    """Generate echo256-style data: random block repeated.

    Pattern: [B0 B1 B2 B3 | B0 B1 B2 B3 | ...] (block repeats)
    x[t] = data[t], y[t] = data[t+1] (next byte prediction)

    Positions 0 to block_size-2: unpredictable (first block)
    Positions block_size-1 onward: predictable (block repeats)
    Ceiling at seq_len=32, block=4: 29/32 = 90.6%
    """
    xs, ys = [], []
    for _ in range(batch_size):
        block = [random.randint(0, 255) for _ in range(block_size)]
        # Repeat block enough times to cover seq_len + 1
        repeats = (seq_len + 2) // block_size + 1
        data = (block * repeats)[:seq_len + 1]
        xs.append(byte_to_bits(data[:seq_len], num_bits))
        ys.append(byte_to_bits(data[1:seq_len + 1], num_bits))
    return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)


# ─── MINI PHI-EMA RNN ───────────────────────────────────────
class MiniPhiRNN(nn.Module):
    """Minimal RNN matching VRAXION core: phi-EMA + C19 processing layers.

    Replicates the signal path of SwarmByteRingModel without ring memory,
    LCX, beings, or pointers — isolating the position encoding variable.
    """

    def __init__(self, num_bits=8, d=128, depth=4, seq_len=32, use_pos_enc=False):
        super().__init__()
        self.d = d
        self.seq_len = seq_len
        self.use_pos_enc = use_pos_enc

        # Input/output projections (matches VRAXION input_proj / output_proj)
        self.input_proj = nn.Linear(num_bits, d)
        self.output_proj = nn.Linear(d, num_bits)

        # State normalization (matches VRAXION state_norm)
        self.state_norm = nn.LayerNorm(d)

        # Processing layers: Pre-LN + C19 residual (matches VRAXION processing_layers)
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(depth - 1)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth - 1)])

        # Sinusoidal position encoding (zero parameters)
        if use_pos_enc:
            self.register_buffer('pos_enc', self._make_sinusoidal_pe(seq_len, d))

    @staticmethod
    def _make_sinusoidal_pe(seq_len, d):
        """Standard transformer sinusoidal position encoding."""
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(seq_len, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # [seq_len, d]

    def forward(self, x):
        """Forward pass: autoregressive phi-EMA RNN.

        Args:
            x: [B, T, num_bits] input bits

        Returns:
            [B, T, num_bits] output logits
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.d, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            # 1. Input projection
            inp = self.input_proj(x[:, t, :])  # [B, D]

            # 2. Position encoding (THE VARIABLE BEING TESTED)
            if self.use_pos_enc:
                inp = inp + self.pos_enc[t]  # [B, D] + [D] broadcast

            # 3. Phi-EMA state update (matches swarm_model.py:1414)
            state = 0.618 * self.state_norm(h) + 0.382 * inp

            # 4. Processing layers: residual C19 (matches swarm_model.py:1416-1417)
            for norm, layer in zip(self.norms, self.layers):
                state = state + c19_activation(layer(norm(state)))

            # 5. Update hidden state
            h = state

            # 6. Output projection
            outputs.append(self.output_proj(h))

        return torch.stack(outputs, dim=1)  # [B, T, num_bits]


# ─── TRAINING LOOP ──────────────────────────────────────────
def run_config(label, use_pe, seed):
    """Run one config for one seed. Returns (tail_median, status, total_time)."""
    torch.manual_seed(seed)
    random.seed(seed)

    model = MiniPhiRNN(
        num_bits=NUM_BITS, d=D, depth=DEPTH,
        seq_len=SEQ_LEN, use_pos_enc=use_pe,
    ).to(DEVICE)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tail_accs = []
    had_nan = False
    had_div = False
    t_start = time.time()

    print(f'\n  [{label} seed={seed}] params={n_params:,}  pe={use_pe}', flush=True)

    for step in range(STEPS):
        t0 = time.time()

        # Generate fresh batch
        x, y = make_echo_batch(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS)

        # Forward
        opt.zero_grad()
        out = model(x)  # [B, T, num_bits]
        loss = F.binary_cross_entropy_with_logits(out, y)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        elapsed = time.time() - t0

        # TIMEOUT GUARD
        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT: step {step} took {elapsed:.0f}s, aborting')
            sys.exit(1)

        # Metrics
        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        # Divergence checks
        if math.isnan(loss.item()):
            had_nan = True
            print(f'  NaN at step {step}, aborting config')
            break
        if loss.item() > 3.0 and step > 200:
            had_div = True
            print(f'  Divergence at step {step} (loss={loss.item():.4f}), aborting config')
            break

        # Tail accumulation (last 100 steps)
        if step >= STEPS - 100:
            tail_accs.append(acc)

        # Progress logging
        if step % 50 == 0 or step == STEPS - 1:
            print(f'    step {step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | {elapsed:.2f}s',
                  flush=True)

        # Live log (dashboard compatible)
        with open(LIVE_LOG, 'a') as lf:
            lf.write(f'[{label} s={seed}] step {step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} RD:{elapsed:.4f}\n')

    total_time = time.time() - t_start

    if had_nan:
        return 0.0, 'NAN', total_time
    if had_div:
        return 0.0, 'DIVERGED', total_time
    if not tail_accs:
        return 0.0, 'NO_TAIL', total_time

    tail_median = sorted(tail_accs)[len(tail_accs) // 2]
    return tail_median, 'OK', total_time


# ─── MAIN ────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('PROBE: Position Encoding A/B Test (TOT-H020)')
    print('=' * 60)
    print(f'  D={D}  depth={DEPTH}  seq_len={SEQ_LEN}  block={BLOCK_SIZE}')
    print(f'  batch={BATCH}  lr={LR}  steps={STEPS}')
    print(f'  seeds={SEEDS}')
    print(f'  ceiling at seq_len={SEQ_LEN}: {(SEQ_LEN - BLOCK_SIZE + 1)/SEQ_LEN:.1%}')
    print(f'  live log: {LIVE_LOG}')
    print('=' * 60)

    results = {}

    for label, use_pe in CONFIGS:
        seed_tails = []
        seed_times = []

        for seed in SEEDS:
            tail, status, elapsed = run_config(label, use_pe, seed)
            seed_tails.append(tail)
            seed_times.append(elapsed)
            print(f'    -> tail_median={tail:.4f}  status={status}  time={elapsed:.1f}s')

        mean_tail = sum(seed_tails) / len(seed_tails)
        seed_gap = max(seed_tails) - min(seed_tails)
        results[label] = {
            'mean_tail': mean_tail,
            'seed_gap': seed_gap,
            'tails': seed_tails,
            'times': seed_times,
        }

    # ─── RESULTS TABLE ───────────────────────────────────────
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'  {"Config":<12} {"Mean Tail":>10} {"Gap":>8} {"Seeds":>20}')
    print(f'  {"-"*12} {"-"*10} {"-"*8} {"-"*20}')
    for label, r in results.items():
        seeds_str = ', '.join(f'{t:.4f}' for t in r['tails'])
        print(f'  {label:<12} {r["mean_tail"]:>10.4f} {r["seed_gap"]:>8.4f} {seeds_str:>20}')

    a = results.get('no_pe', {}).get('mean_tail', 0)
    b = results.get('sin_pe', {}).get('mean_tail', 0)
    delta = b - a

    print(f'\n  Delta (sin_pe - no_pe): {delta:+.4f}')

    # ─── VERDICT ─────────────────────────────────────────────
    print('\n' + '=' * 60)
    if b > 0.70 and a < 0.55:
        verdict = 'CONFIRMED'
        print('  VERDICT: CONFIRMED')
        print('  Position encoding IS the fix. Proceed to implement in SwarmByteRingModel.')
    elif a < 0.55 and b < 0.55:
        verdict = 'INSUFFICIENT'
        print('  VERDICT: INSUFFICIENT')
        print('  PE alone is not enough. Need deeper changes (learned gating, ring attention).')
    elif a > 0.70 and b > 0.70:
        verdict = 'UNEXPECTED'
        print('  VERDICT: UNEXPECTED')
        print('  Phi-EMA can learn echo WITHOUT PE. Analysis needs revision.')
    else:
        verdict = 'AMBIGUOUS'
        print(f'  VERDICT: AMBIGUOUS (A={a:.4f}, B={b:.4f})')
        print('  Results don\'t fit expected patterns. Inspect per-seed data.')
    print('=' * 60)

    # Write summary to live log
    with open(LIVE_LOG, 'a') as lf:
        lf.write(f'\n# VERDICT: {verdict}\n')
        lf.write(f'# no_pe  mean_tail={a:.4f}\n')
        lf.write(f'# sin_pe mean_tail={b:.4f}\n')
        lf.write(f'# delta={delta:+.4f}\n')
