"""
probe_echo256.py -- does LCX actually help on a task that needs memory?

echo256: 16-byte random block repeated twice. To predict byte[t],
model needs byte[t-16] which is outside the brain's local context.
Brain alone (tt=0) cannot solve this. LCX (tt=1) should retrieve it.

This is the core question: does LCX content integration work at all?

  tt=0 wins  -> impossible, brain can't look back 16
  tt=1 wins  -> LCX content works, Phase B is real
  flat       -> something deeper broken in LCX content path

CPU only -- separate log, GPU training unaffected.
"""

import sys, os, time, math, random
sys.path.insert(0, r'S:\AI\work\VRAXION_DEV\Diamond Code')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn.functional as F

LOG_PATH     = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_echo256_console.log'
RESULTS_PATH = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_echo256_results.txt'
LIVE_LOG     = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_echo256_live.log'

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
open(LIVE_LOG, 'w').close()

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
STEPS        = 1000   # more steps -- echo256 harder than copy_echo
SEEDS        = [42, 137]
BATCH        = 16
SEQ_LEN      = 40    # needs to see 16 bytes back + current position
BLOCK        = 16
STEP_TIMEOUT = 60
DEVICE       = torch.device('cpu')

BASE_CFG = dict(
    embedding_dim         = 128,
    depth                 = 4,
    num_beings            = 1,
    num_bits              = 8,
    num_memory_positions  = SEQ_LEN,
    attention_radius      = 6,
    attention_temperature = 8.0,
    byte_token_mode       = False,  # binary bits mode
)

CONFIGS = [
    # label,      tt, use_lcx, slots
    ('tt0_noLCX', 0,  False,   0  ),  # brain alone -- theoretical ceiling ~50% on echo256
    ('tt1_LCX',   1,  True,    500),  # LCX on -- should retrieve 16-step-old bytes
]

print('=' * 62)
print('probe_echo256 -- does LCX help on memory-requiring task?')
print('CPU ONLY -- separate log, GPU training unaffected')
print('=' * 62)
print(f'  task: echo256 (16-byte block repeat, needs 16-step lookback)')
print(f'  D=128, depth=4, batch={BATCH}, seq_len={SEQ_LEN}, block={BLOCK}')
print(f'  steps={STEPS}, seeds={SEEDS}')
print(f'  tt0 ceiling: ~50% (brain cant look back 16)')
print(f'  tt1 target:  >60% (LCX retrieves old block)')
print()

# ── Data: echo256 in binary bits ──────────────────────────────────────────────
def byte_to_bits(byte_seq, num_bits=8):
    t = torch.tensor(byte_seq, dtype=torch.uint8)
    return ((t.unsqueeze(-1) >> torch.arange(num_bits)) & 1).float()

def make_batch(batch, seq_len, device):
    """echo256: 16-byte random block repeated twice.
    y[t] = x[t-16] for positions 16-31 (second occurrence of block).
    Brain can't do this at seq_len=40 -- needs external memory."""
    xs, ys = [], []
    for _ in range(batch):
        data = []
        while len(data) < seq_len + 1:
            block = [random.randint(0, 255) for _ in range(BLOCK)]
            data.extend(block * 2)   # each block repeated once
        data = data[:seq_len + 1]
        xs.append(byte_to_bits(data[:seq_len]))
        ys.append(byte_to_bits(data[1:seq_len + 1]))
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
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
        loss = F.binary_cross_entropy_with_logits(out.float(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

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
            acc = (pred_bits == y).float().mean().item()

        if math.isnan(loss.item()):
            had_nan = True; break
        if loss.item() > 2.0 and step > 200:
            had_div = True; break

        margin = 0.0
        if use_lcx and hasattr(model, '_lcx_score_margin_accum') and model._lcx_score_margin_accum:
            margin = model._lcx_score_margin_accum[-1][0]

        if step >= 800:
            tail_accs.append(acc)

        with open(LIVE_LOG, 'a', encoding='utf-8') as lf:
            lf.write(f'[{label} s={seed}] step {step} | loss {loss.item():.6f} | acc={acc:.4f} margin={margin:.4f} RD:{elapsed:.4f}\n')

    tail   = float('nan') if not tail_accs else sorted(tail_accs)[len(tail_accs)//2]
    status = 'NaN' if had_nan else ('DIV' if had_div else 'OK')
    avg_t  = sum(step_times) / len(step_times) if step_times else 0
    return tail, status, avg_t

# ── Main ──────────────────────────────────────────────────────────────────────
results = {}

for label, tt, use_lcx, slots in CONFIGS:
    print(f'\n{"=" * 62}')
    print(f'  CONFIG: {label}  (tt={tt}, lcx={use_lcx})')
    print(f'{"=" * 62}')
    seed_tails = []
    for seed in SEEDS:
        print(f'  seed={seed}:')
        tail, status, avg_t = run_config(label, tt, use_lcx, slots, seed)
        seed_tails.append(tail)
        print(f'    DONE | tail={tail:.4f} | status={status} | {avg_t:.2f}s/step')
    mean_tail = float('nan') if any(math.isnan(t) for t in seed_tails) else sum(seed_tails)/len(seed_tails)
    seed_gap  = (max(seed_tails) - min(seed_tails)) if not any(math.isnan(t) for t in seed_tails) else float('nan')
    results[label] = dict(mean=mean_tail, gap=seed_gap)
    print(f'  -> mean_tail={mean_tail:.4f}  seed_gap={seed_gap:.4f}')

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f'\n{"=" * 62}')
print('  FINAL RESULTS')
print(f'{"=" * 62}')

tt0 = results['tt0_noLCX']['mean']
tt1 = results['tt1_LCX']['mean']
delta = tt1 - tt0

print(f'  tt0 (no LCX): {tt0:.4f}')
print(f'  tt1 (LCX on): {tt1:.4f}')
print(f'  delta:        {delta:+.4f}')
print()

if delta > 0.010 and results['tt1_LCX']['gap'] < 0.010:
    print(f'  VERDICT: LCX HELPS on echo256 (+{delta:.4f})')
    print(f'  LCX content integration works. Phase B is real.')
elif delta < -0.010:
    print(f'  VERDICT: LCX HURTS on echo256 ({delta:.4f})')
    print(f'  Something broken in LCX content path.')
else:
    print(f'  VERDICT: FLAT -- LCX not helping on echo256')
    print(f'  Content integration not working on CPU cold start.')
    print(f'  Phase B unlock requires trained model (real GPU run).')

with open(RESULTS_PATH, 'w') as f:
    f.write(f'probe_echo256 results\n\n')
    f.write(f'tt0_noLCX: {tt0:.4f}\n')
    f.write(f'tt1_LCX:   {tt1:.4f}\n')
    f.write(f'delta:     {delta:+.4f}\n')

print(f'\nResults: {RESULTS_PATH}')
print('Done.')
