"""
probe_activation_cold.py
Probe Ladder Level 1: activation function on cold start (tt=0, no LCX)
Issue: #100

Question: Does C19 beat GELU in processing layers on cold start?
Task: copy_echo256, binary bits mode, tt=0 (no LCX), 300 steps
Configs: gelu, c19, tanh, softsign
Signal: convergence speed + tail bit_acc (steps 250-300)

Harness: D=128, depth=4, seeds=[42,137], batch=16, seq_len=32, lr=1e-3, CPU fp32

NOTE: Diamond Code hardcodes c19_activation in processing layers.
We monkey-patch swarm_model.c19_activation to swap in other activations.
"""

import sys, os, time, random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import swarm_model as sm
from swarm_model import SwarmByteRingModel

# ── config ──────────────────────────────────────────────────────────────────
STEPS    = 300
SEEDS    = [42, 137]
BATCH    = 16
SEQ_LEN  = 32
LR       = 1e-3
TAIL     = 50   # last 50 steps for tail metric
LOG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'probe', 'probe_activation_cold_live.log')

_original_c19 = sm.c19_activation  # capture before any patching

ACTIVATION_FNS = {
    'c19':      lambda x: _original_c19(x),
    'gelu':     lambda x: F.gelu(x),
    'tanh':     lambda x: torch.tanh(x),
    'softsign': lambda x: F.softsign(x),
}
CONFIGS = list(ACTIVATION_FNS.keys())

BASE_CFG = dict(
    embedding_dim=128,
    depth=4,
    num_beings=1,
    num_bits=16,          # 16-bit input: 8 real bits + 8 zero-padded (GEM path needs num_bits*2 wide)
    num_memory_positions=SEQ_LEN,
    attention_radius=6,
    attention_temperature=8.0,
    byte_token_mode=False,
    use_lcx=False,        # no LCX — pure brain test
    think_ticks=0,
)

# ── data ─────────────────────────────────────────────────────────────────────
def make_batch(batch_size, seq_len, device):
    """copy_echo256: copy current byte — trivial 1-byte context task.
    GEM path needs num_bits*2=32 wide input (bits + GEM concat).
    We use num_bits=16, so input is [B, T, 32]: 8 real bits + 8 zeros padded.
    Output is [B, T, 16]: model predicts 16 bits, we score only first 8."""
    xs, ys = [], []
    for _ in range(batch_size):
        data = [random.randint(0, 255) for _ in range(seq_len)]
        x_bits, y_bits = [], []
        for b in data:
            bits = [(b >> i) & 1 for i in range(8)]
            x_bits.extend(bits + [0]*8)   # 8 real + 8 zero pad → 16 total
            y_bits.extend(bits + [0]*8)   # target: same
        xs.append(x_bits)
        ys.append(y_bits)
    x = torch.tensor(xs, dtype=torch.float32, device=device).view(batch_size, seq_len, 16)
    y = torch.tensor(ys, dtype=torch.float32, device=device).view(batch_size, seq_len, 16)
    return x, y

# ── runner ───────────────────────────────────────────────────────────────────
def run_config(name, activation_fn, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Monkey-patch: swap c19_activation in swarm_model's global namespace
    # (the function is called as a bare name inside the class, so patching
    #  sm.c19_activation patches the module-level name that the class uses)
    original_c19 = sm.__dict__['c19_activation']
    sm.__dict__['c19_activation'] = activation_fn

    model = SwarmByteRingModel(**BASE_CFG)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    device = torch.device('cpu')

    tail_accs = []
    t_start = time.time()

    for step in range(STEPS):
        x, y = make_batch(BATCH, SEQ_LEN, device)
        opt.zero_grad()
        out = model(x)  # [B, T, 8]
        loss = F.binary_cross_entropy_with_logits(out.float(), y)
        loss.backward()
        opt.step()

        pred = (out > 0).float()
        acc = (pred == y).float().mean().item()
        if step >= STEPS - TAIL:
            tail_accs.append(acc)

        s_per_step = (time.time() - t_start) / (step + 1)
        line = f"[{name} s={seed}] step {step} | loss {loss.item():.6f} | acc={acc:.4f} RD:{s_per_step:.4f}"
        print(line, flush=True)
        with open(LOG_PATH, 'a') as f:
            f.write(line + '\n')

    # Restore original
    sm.__dict__['c19_activation'] = original_c19

    mean_tail = float(np.mean(tail_accs))
    return mean_tail, s_per_step

# ── main ─────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
open(LOG_PATH, 'w').close()

print("=" * 60)
print("probe_activation_cold — Level 1: activation function (tt=0)")
print(f"Configs: {CONFIGS}")
print(f"Steps: {STEPS}, Seeds: {SEEDS}, Task: copy_echo256, no LCX")
print("=" * 60)

results = {}
for act_name in CONFIGS:
    act_fn = ACTIVATION_FNS[act_name]
    seed_tails, seed_speeds = [], []
    for seed in SEEDS:
        label = f"{act_name}"
        print(f"\n--- {act_name} seed={seed} ---")
        tail, speed = run_config(label, act_fn, seed)
        seed_tails.append(tail)
        seed_speeds.append(speed)
        print(f"  tail_acc={tail:.4f}  s/step={speed:.3f}")
    results[act_name] = dict(
        mean_tail=float(np.mean(seed_tails)),
        seed_gap=float(np.std(seed_tails)),
        mean_speed=float(np.mean(seed_speeds)),
    )

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Config':<12} {'mean_tail':>10} {'seed_gap':>10} {'s/step':>8}")
print("-" * 44)

ranked = sorted(results.items(), key=lambda x: -x[1]['mean_tail'])
for name, r in ranked:
    marker = " <-- WINNER" if name == ranked[0][0] else ""
    print(f"{name:<12} {r['mean_tail']:>10.4f} {r['seed_gap']:>10.4f} {r['mean_speed']:>8.3f}s{marker}")

print("=" * 60)
winner = ranked[0][0]
baseline = results['gelu']['mean_tail']
delta = results[winner]['mean_tail'] - baseline
print(f"\nWinner: {winner}  ({'+' if delta>=0 else ''}{delta:.4f} vs gelu baseline)")
print(f"Log: {LOG_PATH}")
