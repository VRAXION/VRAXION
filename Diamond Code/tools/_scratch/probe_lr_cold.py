"""
probe_lr_cold.py — Level 3 of CPU probe ladder (#100)
Sweeps lr={1e-4, 3e-4, 1e-3, 3e-3} on copy_echo256, pure brain (tt=0, no LCX).
CPU only, deterministic, quarantined from GPU training.
Mini harness: D=128, depth=4 (locked Level 2).
"""

import sys
import time
import math
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

# ── CPU QUARANTINE ──────────────────────────────────────────────────────────
DEVICE = torch.device('cpu')

# ── Config ──────────────────────────────────────────────────────────────────
SEEDS        = [42, 137]
STEPS        = 300
BATCH_SIZE   = 16
SEQ_LEN      = 32
NUM_BITS     = 16       # 8 real + 8 zero-padded (GEM path needs num_bits*2)
STEP_TIMEOUT = 30
LOG_PATH     = Path('logs/probe/probe_lr_cold_live.log')

LR_VALUES = [1e-4, 3e-4, 1e-3, 3e-3]

BASE_CFG = dict(
    embedding_dim=128,
    depth=4,             # locked: Level 2
    num_beings=1,
    num_bits=NUM_BITS,
    num_memory_positions=SEQ_LEN,
    attention_radius=6,
    attention_temperature=8.0,
    byte_token_mode=False,
    use_lcx=False,
    think_ticks=0,
)

# ── Imports ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import swarm_model as sm

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
log_fh = open(LOG_PATH, 'w', buffering=1)

def log(msg):
    print(msg, flush=True)
    log_fh.write(msg + '\n')

# ── Data ─────────────────────────────────────────────────────────────────────
def make_batch(batch_size, seq_len, seed_offset=0):
    torch.manual_seed(seed_offset)
    x_list, y_list = [], []
    for _ in range(batch_size):
        byte_vals = torch.randint(0, 256, (seq_len,))
        x_bits, y_bits = [], []
        for b in byte_vals.tolist():
            bits = [(b >> i) & 1 for i in range(8)]
            x_bits.extend(bits + [0]*8)
            y_bits.extend(bits)
        x_list.append(x_bits)
        y_list.append(y_bits)
    x = torch.tensor(x_list, dtype=torch.float32).reshape(batch_size, seq_len, NUM_BITS)
    y = torch.tensor(y_list, dtype=torch.float32).reshape(batch_size, seq_len, 8)
    return x.to(DEVICE), y.to(DEVICE)

# ── Single run ───────────────────────────────────────────────────────────────
def run_lr(lr, seed):
    lr_str = f'{lr:.0e}'
    label = f'[lr={lr_str} s={seed}]'

    torch.manual_seed(seed)
    random.seed(seed)

    model = sm.SwarmByteRingModel(**BASE_CFG).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    accs, losses = [], []

    for step in range(STEPS):
        log(f'{label} starting step {step}...')
        t0 = time.time()

        x, y = make_batch(BATCH_SIZE, SEQ_LEN, seed_offset=step*1000+seed)
        optimizer.zero_grad()
        out = model(x)
        logits = out[:, :, :8]
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        elapsed = time.time() - t0
        if elapsed > STEP_TIMEOUT:
            log(f'TIMEOUT: step {step} took {elapsed:.0f}s — aborting')
            sys.exit(1)

        with torch.no_grad():
            preds = (logits.sigmoid() > 0.5).float()
            acc = (preds == y).float().mean().item()

        accs.append(acc)
        losses.append(loss.item())
        log(f'{label} step {step} | loss {loss.item():.6f} | acc={acc:.4f} RD:{elapsed:.4f} traction={acc:.4f} shard=0/0')

    tail_accs = accs[250:]
    return {
        'lr':         lr,
        'seed':       seed,
        'mean_tail':  statistics.mean(tail_accs),
        'median_tail':statistics.median(tail_accs),
        'last_acc':   accs[-1],
        'last_loss':  losses[-1],
        'loss_50':    losses[49] if len(losses) > 49 else None,
        'had_nan':    any(math.isnan(a) for a in accs),
        'had_inf':    any(math.isinf(a) for a in accs),
        'diverged':   losses[-1] > 3 * losses[49] if len(losses) > 49 else False,
    }

# ── Main sweep ───────────────────────────────────────────────────────────────
def main():
    log('=== probe_lr_cold.py — Level 3 CPU probe ladder ===')
    log(f'lr values: {LR_VALUES}  seeds: {SEEDS}')
    log(f'D=128, depth=4, tt=0, no LCX, copy_echo256, {STEPS} steps')
    log('')

    all_results = []
    for lr in LR_VALUES:
        for seed in SEEDS:
            log(f'\n--- lr={lr:.0e} seed={seed} ---')
            r = run_lr(lr, seed)
            all_results.append(r)
            log(f'  DONE: mean_tail={r["mean_tail"]:.4f} median={r["median_tail"]:.4f} '
                f'last={r["last_acc"]:.4f} diverged={r["diverged"]} nan={r["had_nan"]}')

    from collections import defaultdict
    by_lr = defaultdict(list)
    for r in all_results:
        by_lr[r['lr']].append(r['mean_tail'])

    log('\n\n=== RANKED RESULTS ===')
    log(f'{"lr":<10} {"mean_tail":>10} {"seed_gap":>10}')
    log('-' * 35)
    ranked = sorted(by_lr.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
    for lr, tails in ranked:
        mean = sum(tails)/len(tails)
        gap  = max(tails) - min(tails)
        log(f'{lr:<10.0e} {mean:>10.4f} {gap:>10.4f}')

    winner = ranked[0][0]
    log(f'\nWINNER: lr={winner:.0e}')

    summary_path = Path('logs/probe/probe_lr_results.txt')
    with open(summary_path, 'w') as f:
        f.write('=== probe_lr_cold.py results ===\n')
        for r in all_results:
            f.write(f"lr={r['lr']:.0e} seed={r['seed']} mean_tail={r['mean_tail']:.4f} "
                    f"median={r['median_tail']:.4f} last={r['last_acc']:.4f} "
                    f"diverged={r['diverged']} nan={r['had_nan']}\n")
        f.write(f'\nWINNER: lr={winner:.0e}\n')
    log(f'Summary written to {summary_path}')
    log_fh.close()

if __name__ == '__main__':
    main()
