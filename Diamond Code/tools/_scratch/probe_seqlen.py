"""
probe_seqlen.py -- sequence length sweep (#PF-017)

Tests 5 configs: seq_len={16, 32, 62, 100, 150}
Always co-varies memory_size = seq_len (production match).
Everything else frozen: D=128, depth=2, batch=16, tt=1, LCX hash.

Goal: does longer seq_len help learning (echo pattern)?
  - seq_len=16: baseline (D=128 mini current)
  - seq_len=32: first viable echo window
  - seq_len=62: current production (phi scale)
  - seq_len=100: upper phi bracket
  - seq_len=150: aggressive

CPU only -- does NOT touch GPU training or probe_live.log.
"""

import sys, os, time, math, random
sys.path.insert(0, r'S:\AI\work\VRAXION_DEV\Diamond Code')
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU only

import torch
import torch.nn.functional as F

# -- Paths -----------------------------------------------------------------
LOG_PATH     = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_seqlen_console.log'
RESULTS_PATH = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_seqlen_results.txt'
LIVE_LOG     = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_seqlen_live.log'

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# clear live log
open(LIVE_LOG, 'w').close()

# -- Tee logger -------------------------------------------------------------
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

# -- Import real model -------------------------------------------------------
print('Loading swarm_model.py...', flush=True)
from swarm_model import SwarmByteRingModel
print('OK', flush=True)

# -- Config ------------------------------------------------------------------
STEPS        = 500
SEEDS        = [42, 137]
BATCH        = 16
STEP_TIMEOUT = 60   # seconds per step before abort
DEVICE       = torch.device('cpu')

# Sweep: seq_len = memory_size (co-varied, production match)
SEQLEN_CONFIGS  = [16, 32, 62, 100, 150]
BASELINE_SEQLEN = 62   # current production phi-derived

REPEAT_LEN = 4   # copy_echo: 4-byte pattern repeating (trivially solvable)

def base_model_cfg(seq_len):
    """Model config with seq_len-dependent memory_size."""
    return dict(
        embedding_dim        = 128,
        depth                = 2,       # locked (PF-016)
        num_beings           = 1,
        num_bits             = 8,
        num_memory_positions = seq_len,  # co-varies with seq_len
        use_lcx              = True,
        lcx_mode             = 'hash',
        lcx_num_levels       = 1,
        lcx_level_slots      = [500],
        lcx_top_k            = 2,       # locked (probe #91)
        lcx_key_dim          = 13,
        attention_radius     = 6,       # fixed
        attention_temperature = 8.0,
        think_ticks          = 1,
        byte_token_mode      = True,
    )

print('=' * 62)
print('probe_seqlen -- sequence length sweep (#PF-017)')
print('CPU ONLY -- separate log, GPU training unaffected')
print('=' * 62)
print(f'  D=128, depth=2, batch={BATCH}, LCX=hash(500)')
print(f'  seq_lens={SEQLEN_CONFIGS}, baseline={BASELINE_SEQLEN}')
print(f'  task=copy_echo (repeat_len={REPEAT_LEN})')
print(f'  steps={STEPS}, seeds={SEEDS}')
print(f'  win: delta > 0.010 vs baseline, seed_gap < 0.020')
print()

# -- Data: copy_echo -- short pattern repeating forever ----------------------
def make_batch(batch, seq_len, device):
    """Copy-echo: REPEAT_LEN random bytes repeating forever.
    Trivially solvable at any seq_len >= REPEAT_LEN+1.
    Longer seq_len = more repetitions = more training signal per step."""
    xs, ys = [], []
    for _ in range(batch):
        pattern = [random.randint(0, 255) for _ in range(REPEAT_LEN)]
        # tile pattern to fill seq_len+1
        n_tiles = (seq_len + 1) // REPEAT_LEN + 1
        data = (pattern * n_tiles)[:seq_len + 1]
        xs.append(data[:seq_len])
        ys.append(data[1:seq_len + 1])
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y

# -- Single config run -------------------------------------------------------
def run_config(seq_len, seed):
    torch.manual_seed(seed)
    random.seed(seed)

    cfg = base_model_cfg(seq_len)
    model = SwarmByteRingModel(**cfg).to(DEVICE)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    tail_accs = []
    losses = []
    had_nan = False
    had_div = False

    for step in range(STEPS):
        t0 = time.time()

        if step % 100 == 0:
            print(f'    step {step}...', flush=True)

        x, y = make_batch(BATCH, seq_len, DEVICE)

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

        lv = loss.item()
        if math.isnan(lv):
            had_nan = True; break
        if lv > 8.0 and step > 100:
            had_div = True; break

        losses.append(lv)

        if step >= 400:
            tail_accs.append(acc)

        with open(LIVE_LOG, 'a', encoding='utf-8') as lf:
            lf.write(f'[sl={seq_len} s={seed}] step {step} | loss {lv:.6f} | acc={acc:.4f} RD:{elapsed:.4f}\n')

    tail = float('nan') if not tail_accs else sorted(tail_accs)[len(tail_accs)//2]
    final_loss = float('nan') if not losses else losses[-1]
    status = 'NaN' if had_nan else ('DIV' if had_div else 'OK')
    avg_step_time = elapsed  # last step time (approximate)
    return tail, final_loss, status, avg_step_time

# -- Main sweep --------------------------------------------------------------
results = {}   # seq_len -> {mean, gap, seeds, loss, time}

for sl in SEQLEN_CONFIGS:
    label = f'seq_len={sl}{"  <-- BASELINE" if sl == BASELINE_SEQLEN else ""}'
    print(f'\n{"=" * 62}')
    print(f'  CONFIG: {label}  (memory_size={sl})')
    print(f'{"=" * 62}')
    seed_tails = []
    seed_losses = []
    seed_times = []
    for seed in SEEDS:
        print(f'  seed={seed}:')
        tail, final_loss, status, step_time = run_config(sl, seed)
        seed_tails.append(tail)
        seed_losses.append(final_loss)
        seed_times.append(step_time)
        print(f'    DONE | tail_acc={tail:.4f} | loss={final_loss:.4f} | time/step={step_time:.3f}s | status={status}')
    mean_tail = float('nan') if any(math.isnan(t) for t in seed_tails) else sum(seed_tails)/len(seed_tails)
    seed_gap  = max(seed_tails) - min(seed_tails) if not any(math.isnan(t) for t in seed_tails) else float('nan')
    mean_loss = sum(seed_losses)/len(seed_losses) if not any(math.isnan(l) for l in seed_losses) else float('nan')
    mean_time = sum(seed_times)/len(seed_times)
    results[sl] = dict(mean=mean_tail, gap=seed_gap, seeds=seed_tails, loss=mean_loss, time=mean_time)
    print(f'  -> mean_tail={mean_tail:.4f}  seed_gap={seed_gap:.4f}  loss={mean_loss:.4f}  time/step={mean_time:.3f}s')

# -- Final verdict -----------------------------------------------------------
print(f'\n{"=" * 62}')
print('  FINAL RESULTS')
print(f'{"=" * 62}')
print(f'  {"seq_len":>7}  {"mean_acc":>9}  {"seed_gap":>9}  {"loss":>7}  {"s/step":>7}  {"delta":>7}')
print(f'  {"-"*7}  {"-"*9}  {"-"*9}  {"-"*7}  {"-"*7}  {"-"*7}')

baseline = results[BASELINE_SEQLEN]['mean']
best_sl = max(results, key=lambda s: results[s]['mean'])
best_mean = results[best_sl]['mean']
delta = best_mean - baseline

for sl in SEQLEN_CONFIGS:
    d = results[sl]['mean'] - baseline
    marker = '  BASE' if sl == BASELINE_SEQLEN else f' {d:+.4f}'
    r = results[sl]
    print(f'  {sl:>7}  {r["mean"]:>9.4f}  {r["gap"]:>9.4f}  {r["loss"]:>7.4f}  {r["time"]:>7.3f}  {marker}')

print()

# Efficiency: acc per second of training
print('  Efficiency (acc / wall_time for 500 steps):')
for sl in SEQLEN_CONFIGS:
    r = results[sl]
    wall = r['time'] * STEPS
    eff = r['mean'] / wall if wall > 0 else 0
    print(f'    seq_len={sl:>3}: {r["mean"]:.4f} acc / {wall:.0f}s = {eff:.6f} acc/s')

print()
if best_sl != BASELINE_SEQLEN and delta > 0.010 and results[best_sl]['gap'] < 0.020:
    print(f'  VERDICT: WINNER seq_len={best_sl} (+{delta:.4f} over baseline={BASELINE_SEQLEN})')
    if best_sl > BASELINE_SEQLEN:
        print(f'  DIRECTION: longer context helps -- consider seq_len={best_sl} for production')
    else:
        print(f'  DIRECTION: shorter is better -- efficiency gain with less compute')
else:
    print(f'  VERDICT: FLAT -- phi-derived seq_len=62 holds')
    if best_sl > BASELINE_SEQLEN:
        print(f'  NOTE: seq_len={best_sl} was best but delta={delta:.4f} < 0.010 threshold')
    elif best_sl < BASELINE_SEQLEN:
        print(f'  NOTE: seq_len={best_sl} was best -- shorter might be more efficient')

# write results file
with open(RESULTS_PATH, 'w') as f:
    f.write('probe_seqlen results (#PF-017)\n')
    f.write(f'configs: {SEQLEN_CONFIGS}, baseline={BASELINE_SEQLEN}\n')
    f.write(f'D=128, depth=2, batch={BATCH}, copy_echo repeat={REPEAT_LEN}\n\n')
    for sl in SEQLEN_CONFIGS:
        r = results[sl]
        f.write(f'seq_len={sl}: mean={r["mean"]:.4f} gap={r["gap"]:.4f} loss={r["loss"]:.4f} time={r["time"]:.3f}s seeds={r["seeds"]}\n')
    f.write(f'\nbest={best_sl} delta={delta:+.4f}\n')

print(f'\nResults: {RESULTS_PATH}')
print('Done.')
