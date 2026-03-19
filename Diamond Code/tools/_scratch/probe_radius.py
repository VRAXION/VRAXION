"""
probe_radius.py -- attention_radius adversarial probe (#92)

Tests 3 configs: radius={2, 6, 10} at fixed temperature=8.0
Adversarial design: extremes vs current default.
  - radius=2 wins -> writes too spread at 6, tighter better
  - radius=10 wins -> writes too tight at 6, wider better
  - both lose -> phi was right, close ticket

CPU only -- does NOT touch GPU training or probe_live.log.
"""

import sys, os, time, math, random
sys.path.insert(0, r'S:\AI\work\VRAXION_DEV\Diamond Code')
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU only

import torch
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_PATH     = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_radius_console.log'
RESULTS_PATH = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_radius_results.txt'
LIVE_LOG     = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_radius_live.log'

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# clear live log
open(LIVE_LOG, 'w').close()

# ── Tee logger ────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, msg):
        self.stdout.write(msg)
        self.f.write(msg)
        self.f.flush()
    def flush(self):
        self.stdout.flush()
        self.f.flush()

sys.stdout = Tee(LOG_PATH)

# ── Import real model ─────────────────────────────────────────────────────────
print('Loading swarm_model.py...', flush=True)
from swarm_model import SwarmByteRingModel
print('OK', flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
STEPS        = 500
SEEDS        = [42, 137]
BATCH        = 16
SEQ_LEN      = 32
STEP_TIMEOUT = 60   # seconds per step before abort
DEVICE       = torch.device('cpu')

# Adversarial: two extremes vs current default
RADIUS_CONFIGS = [2, 6, 10]   # 6 = current (phi-derived baseline)
BASELINE_RADIUS = 6

BASE_MODEL_CFG = dict(
    embedding_dim        = 128,
    depth                = 4,
    num_beings           = 1,
    num_bits             = 8,
    num_memory_positions = SEQ_LEN,
    use_lcx              = True,
    lcx_mode             = 'hash',
    lcx_num_levels       = 1,
    lcx_level_slots      = [500],
    lcx_top_k            = 2,
    lcx_key_dim          = 13,
    attention_temperature = 8.0,   # fixed
    think_ticks          = 1,
    byte_token_mode      = True,
)

print('=' * 62)
print('probe_radius -- attention_radius adversarial test (#92)')
print('CPU ONLY -- separate log, GPU training unaffected')
print('=' * 62)
print(f'  D=128, depth=4, batch={BATCH}, seq_len={SEQ_LEN}')
print(f'  radii={RADIUS_CONFIGS}, baseline={BASELINE_RADIUS}')
print(f'  steps={STEPS}, seeds={SEEDS}')
print(f'  win: delta > 0.010 vs baseline, seed_gap < 0.010')
print()

# ── Data: echo256 -- 16-byte block pattern, fits in 32-byte window ────────────
def make_batch(batch, seq_len, device):
    """echo256: 16-byte random block repeated. Next byte = byte 16 positions back."""
    BLOCK = 16
    xs, ys = [], []
    for _ in range(batch):
        # generate enough data for seq_len+1 bytes
        n_blocks = (seq_len + BLOCK * 2) // BLOCK + 1
        data = []
        for _ in range(n_blocks):
            block = [random.randint(0, 255) for _ in range(BLOCK)]
            data.extend(block * 2)   # repeat each block once
        data = data[:seq_len + 1]
        xs.append(data[:seq_len])
        ys.append(data[1:seq_len + 1])
    x = torch.tensor(xs, dtype=torch.long, device=device)   # [B, T]
    y = torch.tensor(ys, dtype=torch.long, device=device)   # [B, T]
    return x, y

# ── Single config run ─────────────────────────────────────────────────────────
def run_config(radius, seed):
    torch.manual_seed(seed)
    random.seed(seed)

    cfg = {**BASE_MODEL_CFG, 'attention_radius': radius}
    model = SwarmByteRingModel(**cfg).to(DEVICE)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    tail_accs = []
    had_nan = False
    had_div = False

    for step in range(STEPS):
        t0 = time.time()

        if step % 100 == 0:
            print(f'    step {step}...', flush=True)

        x, y = make_batch(BATCH, SEQ_LEN, DEVICE)

        opt.zero_grad()
        out, stats = model(x, return_stats=True)

        # byte_token_mode: out is [B, T, 256]
        loss = F.cross_entropy(out.float().reshape(-1, 256), y.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # detach LCX buffers between steps
        with torch.no_grad():
            for lvl in range(model._lcx_num_levels):
                k_buf, v_buf = model._lcx_level_bufs(lvl)
                if hasattr(v_buf, 'requires_grad') and v_buf.requires_grad:
                    setattr(model, f'lcx_values_{lvl}', v_buf.detach())
                if hasattr(k_buf, 'requires_grad') and k_buf.requires_grad:
                    setattr(model, f'lcx_keys_{lvl}', k_buf.detach())

        elapsed = time.time() - t0
        if elapsed > STEP_TIMEOUT:
            print(f'TIMEOUT: step {step} took {elapsed:.0f}s, aborting')
            sys.exit(1)

        with torch.no_grad():
            pred = out.argmax(dim=-1)
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            had_nan = True; break
        if loss.item() > 5.0 and step > 100:
            had_div = True; break

        # score margin from LCX accumulator
        margin = 0.0
        if hasattr(model, '_lcx_score_margin_accum') and model._lcx_score_margin_accum:
            margin = model._lcx_score_margin_accum[-1][0]

        if step >= 400:
            tail_accs.append(acc)

        with open(LIVE_LOG, 'a', encoding='utf-8') as lf:
            lf.write(f'[r={radius} s={seed}] step {step} | loss {loss.item():.6f} | acc={acc:.4f} margin={margin:.4f} RD:{elapsed:.4f}\n')

    tail = float('nan') if not tail_accs else sorted(tail_accs)[len(tail_accs)//2]
    status = 'NaN' if had_nan else ('DIV' if had_div else 'OK')
    return tail, status

# ── Main sweep ────────────────────────────────────────────────────────────────
results = {}   # radius -> {seed -> tail}

for radius in RADIUS_CONFIGS:
    label = f'radius={radius}{"  <-- BASELINE" if radius == BASELINE_RADIUS else ""}'
    print(f'\n{"=" * 62}')
    print(f'  CONFIG: {label}')
    print(f'{"=" * 62}')
    seed_tails = []
    for seed in SEEDS:
        print(f'  seed={seed}:')
        tail, status = run_config(radius, seed)
        seed_tails.append(tail)
        print(f'    DONE | tail={tail:.4f} | status={status}')
    mean_tail = float('nan') if any(math.isnan(t) for t in seed_tails) else sum(seed_tails)/len(seed_tails)
    seed_gap  = max(seed_tails) - min(seed_tails) if not any(math.isnan(t) for t in seed_tails) else float('nan')
    results[radius] = dict(mean=mean_tail, gap=seed_gap, seeds=seed_tails)
    print(f'  -> mean_tail={mean_tail:.4f}  seed_gap={seed_gap:.4f}')

# ── Final verdict ─────────────────────────────────────────────────────────────
print(f'\n{"=" * 62}')
print('  FINAL RESULTS')
print(f'{"=" * 62}')

baseline = results[BASELINE_RADIUS]['mean']
best_radius = max(results, key=lambda r: results[r]['mean'])
best_mean   = results[best_radius]['mean']
delta       = best_mean - baseline

for r in RADIUS_CONFIGS:
    d = results[r]['mean'] - baseline
    marker = ' <-- BASELINE' if r == BASELINE_RADIUS else f' ({d:+.4f})'
    print(f'  radius={r:2d}: mean_tail={results[r]["mean"]:.4f}  seed_gap={results[r]["gap"]:.4f}{marker}')

print()
if best_radius != BASELINE_RADIUS and delta > 0.010 and results[best_radius]['gap'] < 0.010:
    print(f'  VERDICT: WINNER radius={best_radius} (+{delta:.4f}) -- run full GSS sweep')
    if best_radius < BASELINE_RADIUS:
        print(f'  DIRECTION: tighter writes better -- sweep downward [{best_radius}, {BASELINE_RADIUS}]')
    else:
        print(f'  DIRECTION: wider writes better -- sweep upward [{BASELINE_RADIUS}, {best_radius}]')
else:
    print(f'  VERDICT: FLAT -- phi-derived radius=6 holds, close #92')

# write results file
with open(RESULTS_PATH, 'w') as f:
    f.write('probe_radius results\n')
    f.write(f'configs: {RADIUS_CONFIGS}, baseline={BASELINE_RADIUS}\n\n')
    for r in RADIUS_CONFIGS:
        f.write(f'radius={r}: mean={results[r]["mean"]:.4f} gap={results[r]["gap"]:.4f} seeds={results[r]["seeds"]}\n')
    f.write(f'\nbest={best_radius} delta={delta:+.4f}\n')

print(f'\nResults: {RESULTS_PATH}')
print('Done.')
