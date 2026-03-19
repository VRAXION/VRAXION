"""
probe_lcx_bootstrap.py -- LCX bootstrapping probe

Question: does LCX help or hurt on a cold model on copy_echo256?
Tests: tt=0 (no LCX) vs tt=1 (LCX on) on the easiest solvable task.

If tt=0 learns faster/higher -> LCX is net negative cold, bootstrapping is a burden
If tt=1 learns faster/higher -> LCX helps even cold
If flat -> LCX is neutral, not blocking but not helping either

Also tests slot count: 50 vs 500 slots at tt=1.
Fewer slots = easier to fill, might bootstrap faster.

CPU only -- does NOT touch GPU training or probe_live.log.
"""

import sys, os, time, math, random
sys.path.insert(0, r'S:\AI\work\VRAXION_DEV\Diamond Code')
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU only

import torch
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_PATH     = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_lcx_bootstrap_console.log'
RESULTS_PATH = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_lcx_bootstrap_results.txt'
LIVE_LOG     = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_lcx_bootstrap_live.log'

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
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

print('Loading swarm_model.py...', flush=True)
from swarm_model import SwarmByteRingModel
print('OK', flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
STEPS        = 500
SEEDS        = [42, 137]
BATCH        = 16
SEQ_LEN      = 32
STEP_TIMEOUT = 60
DEVICE       = torch.device('cpu')

BASE_CFG = dict(
    embedding_dim        = 128,
    depth                = 4,
    num_beings           = 1,
    num_bits             = 8,
    num_memory_positions = SEQ_LEN,
    attention_radius     = 6,
    attention_temperature = 8.0,
    byte_token_mode      = False,   # binary bits mode -- 8 bits per byte, easier cold start
)

# Configs to test
CONFIGS = [
    # label,         tt, use_lcx, slots
    ('tt0_noLCX',    0,  False,   0  ),   # pure brain, no LCX at all
    ('tt1_50slots',  1,  True,    50 ),   # LCX on, very few slots (easy to fill)
    ('tt1_500slots', 1,  True,    500),   # LCX on, current default slot count
]

print('=' * 62)
print('probe_lcx_bootstrap -- does LCX help or hurt cold? (#92)')
print('CPU ONLY -- separate log, GPU training unaffected')
print('=' * 62)
print(f'  task: copy_echo256 (copy current byte, trivially solvable)')
print(f'  D=128, depth=4, batch={BATCH}, seq_len={SEQ_LEN}')
print(f'  steps={STEPS}, seeds={SEEDS}')
print(f'  configs: {[c[0] for c in CONFIGS]}')
print()

# ── Data: copy_echo256 in binary bits mode ────────────────────────────────────
def byte_to_bits(byte_seq, num_bits=8):
    """Convert list of byte values to float tensor [T, num_bits]."""
    t = torch.tensor(byte_seq, dtype=torch.uint8)
    bits = ((t.unsqueeze(-1) >> torch.arange(num_bits)) & 1).float()
    return bits

def make_batch(batch, seq_len, device):
    """copy_echo256: current byte repeats. y[t] = x[t] almost always.
    Returns x [B, T, 8] float bits, y [B, T, 8] float bits."""
    xs, ys = [], []
    for _ in range(batch):
        data = []
        while len(data) < seq_len + 1:
            val = random.randint(0, 255)
            data.extend([val] * 128)
        data = data[:seq_len + 1]
        xs.append(byte_to_bits(data[:seq_len]))
        ys.append(byte_to_bits(data[1:seq_len + 1]))
    x = torch.stack(xs).to(device)   # [B, T, 8]
    y = torch.stack(ys).to(device)   # [B, T, 8]
    return x, y

# ── Single config run ─────────────────────────────────────────────────────────
def run_config(label, tt, use_lcx, slots, seed):
    torch.manual_seed(seed)
    random.seed(seed)

    cfg = {**BASE_CFG, 'think_ticks': tt, 'use_lcx': use_lcx}
    if use_lcx:
        cfg['lcx_mode']        = 'hash'
        cfg['lcx_num_levels']  = 1
        cfg['lcx_level_slots'] = [slots]
        cfg['lcx_top_k']       = 2
        cfg['lcx_key_dim']     = 13

    model = SwarmByteRingModel(**cfg).to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    tail_accs = []
    had_nan = False
    had_div = False
    step_times = []

    for step in range(STEPS):
        t0 = time.time()

        if step % 100 == 0:
            print(f'    step {step}...', flush=True)

        x, y = make_batch(BATCH, SEQ_LEN, DEVICE)

        opt.zero_grad()
        out, stats = model(x, return_stats=True)

        # binary bits mode: out is [B, T, 8] logits, y is [B, T, 8] float bits
        loss = F.binary_cross_entropy_with_logits(out.float(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # detach LCX buffers between steps
        if use_lcx:
            with torch.no_grad():
                for lvl in range(model._lcx_num_levels):
                    k_buf, v_buf = model._lcx_level_bufs(lvl)
                    if hasattr(v_buf, 'requires_grad') and v_buf.requires_grad:
                        setattr(model, f'lcx_values_{lvl}', v_buf.detach())
                    if hasattr(k_buf, 'requires_grad') and k_buf.requires_grad:
                        setattr(model, f'lcx_keys_{lvl}', k_buf.detach())

        elapsed = time.time() - t0
        step_times.append(elapsed)

        if elapsed > STEP_TIMEOUT:
            print(f'TIMEOUT: step {step} took {elapsed:.0f}s, aborting')
            sys.exit(1)

        with torch.no_grad():
            pred_bits = (out > 0).float()
            acc = (pred_bits == y).float().mean().item()  # bit accuracy

        if math.isnan(loss.item()):
            had_nan = True; break
        if loss.item() > 2.0 and step > 150:   # baseline BCE = ln(2)=0.693
            had_div = True; break

        margin = 0.0
        if use_lcx and hasattr(model, '_lcx_score_margin_accum') and model._lcx_score_margin_accum:
            margin = model._lcx_score_margin_accum[-1][0]

        if step >= 400:
            tail_accs.append(acc)

        with open(LIVE_LOG, 'a', encoding='utf-8') as lf:
            lf.write(f'[{label} s={seed}] step {step} | loss {loss.item():.6f} | acc={acc:.4f} margin={margin:.4f} RD:{elapsed:.4f}\n')

    tail   = float('nan') if not tail_accs else sorted(tail_accs)[len(tail_accs)//2]
    status = 'NaN' if had_nan else ('DIV' if had_div else 'OK')
    avg_step = sum(step_times) / len(step_times) if step_times else 0
    return tail, status, avg_step

# ── Main ──────────────────────────────────────────────────────────────────────
results = {}

for label, tt, use_lcx, slots in CONFIGS:
    print(f'\n{"=" * 62}')
    print(f'  CONFIG: {label}  (tt={tt}, lcx={use_lcx}, slots={slots})')
    print(f'{"=" * 62}')
    seed_tails = []
    seed_times = []
    for seed in SEEDS:
        print(f'  seed={seed}:')
        tail, status, avg_step = run_config(label, tt, use_lcx, slots, seed)
        seed_tails.append(tail)
        seed_times.append(avg_step)
        print(f'    DONE | tail={tail:.4f} | status={status} | {avg_step:.2f}s/step')

    mean_tail = float('nan') if any(math.isnan(t) for t in seed_tails) else sum(seed_tails)/len(seed_tails)
    seed_gap  = (max(seed_tails) - min(seed_tails)) if not any(math.isnan(t) for t in seed_tails) else float('nan')
    mean_time = sum(seed_times) / len(seed_times)
    results[label] = dict(mean=mean_tail, gap=seed_gap, time=mean_time, tt=tt, slots=slots)
    print(f'  -> mean_tail={mean_tail:.4f}  seed_gap={seed_gap:.4f}  {mean_time:.2f}s/step')

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f'\n{"=" * 62}')
print('  FINAL RESULTS')
print(f'{"=" * 62}')

baseline = results['tt0_noLCX']['mean']
for label, _, _, _ in CONFIGS:
    r = results[label]
    delta = r['mean'] - baseline
    marker = ' <-- BASELINE (no LCX)' if label == 'tt0_noLCX' else f'  delta={delta:+.4f} vs no-LCX'
    print(f'  {label:20s}: mean={r["mean"]:.4f}  gap={r["gap"]:.4f}  {r["time"]:.2f}s/step{marker}')

print()
tt1_500 = results['tt1_500slots']['mean']
tt1_50  = results['tt1_50slots']['mean']
tt0     = results['tt0_noLCX']['mean']

if tt0 > tt1_500 + 0.010:
    print('  VERDICT: LCX HURTS cold -- net negative, bootstrapping is a burden')
    print('  ACTION: train tt=0 (Alpha) longer before enabling LCX')
elif tt1_500 > tt0 + 0.010:
    print('  VERDICT: LCX HELPS even cold -- bootstrapping works on easy tasks')
    if tt1_50 > tt1_500 + 0.010:
        print('  BONUS: fewer slots bootstrap faster -- consider starting with 50-100 slots')
else:
    print('  VERDICT: LCX NEUTRAL cold on copy_echo256 -- neither helps nor hurts')
    print('  NOTE: task too easy to stress LCX either way')

with open(RESULTS_PATH, 'w') as f:
    f.write('probe_lcx_bootstrap results\n\n')
    for label, _, _, _ in CONFIGS:
        r = results[label]
        f.write(f'{label}: mean={r["mean"]:.4f} gap={r["gap"]:.4f} time={r["time"]:.2f}s/step\n')
    f.write(f'\nbaseline (tt0): {baseline:.4f}\n')

print(f'\nResults: {RESULTS_PATH}')
print('Done.')
