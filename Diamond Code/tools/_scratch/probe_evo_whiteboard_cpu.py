#!/usr/bin/env python3
"""
Probe: Evolutionary Whiteboard — Tournament Selection on LCX Memory
====================================================================
Hypothesis: LCX writes are noisy. If we only keep writes that improve
loss (tournament selection), the whiteboard should accumulate useful
content faster.

Three configs compared:
  1. brain_only        — no LCX (baseline)
  2. brain+whiteboard  — normal LCX (writes always kept)
  3. brain+evo_wb      — evolutionary LCX (writes kept only if they improve loss)

Evo logic per step:
  1. Snapshot LCX state
  2. Normal training step (forward+backward+optimizer, LCX gets written)
  3. Fresh eval batch (unbiased comparison)
  4. no_grad: loss_new = forward(eval) with new LCX
  5. Restore snapshot, loss_old = forward(eval) with old LCX
  6. Keep whichever LCX gave lower loss

Cost: 2 extra forward passes per step (no gradient). ~3x compute.
"""

import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR  = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG = os.path.join(LOG_DIR, 'probe_evo_whiteboard_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

from swarm_model import SwarmByteRingModel

DEVICE = torch.device('cpu')

# ── Shared hyperparams ──────────────────────────────────────────
D          = 128
DEPTH      = 4
BATCH      = 32
NUM_BITS   = 8
SEQ_LEN    = 32
BLOCK_SIZE = 16
RADIUS     = 8
LCX_SLOTS  = 500
TOP_K      = 2
KEY_DIM    = 12
STEPS      = 400
LR         = 1e-4
WARMUP     = 30
SEED       = 42
STEP_TIMEOUT = 60
WINDOW     = 50


def make_echo_batch(batch_size, seq_len, block_size, num_bits, device):
    """Copy-echo task: repeating block pattern, predict next bit-vector."""
    xs, ys = [], []
    for _ in range(batch_size):
        block = torch.randint(0, 2, (block_size, num_bits)).float()
        repeats = (seq_len + 2) // block_size + 1
        data = block.repeat(repeats, 1)[:seq_len + 1]
        xs.append(data[:seq_len])
        ys.append(data[1:seq_len + 1])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def build_model(use_lcx):
    """Build a fresh SwarmByteRingModel with deterministic seed."""
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = SwarmByteRingModel(
        embedding_dim=D,
        num_memory_positions=SEQ_LEN,
        num_beings=1,
        depth=DEPTH,
        num_bits=NUM_BITS,
        attention_radius=RADIUS,
        attention_temperature=8.0,
        think_ticks=1,
        use_lcx=use_lcx,
        lcx_mode='hash',
        lcx_num_levels=1,
        lcx_level_slots=[LCX_SLOTS],
        lcx_key_dim=KEY_DIM,
        lcx_top_k=TOP_K,
        num_pointers=1,
    ).to(DEVICE)
    model.train()
    return model


def snapshot_lcx(model):
    """Clone all LCX buffer state (keys, values, heat, valid for level 0)."""
    return {
        'keys':  model.lcx_keys_0.clone(),
        'values': model.lcx_values_0.clone(),
        'heat':  model.lcx_heat_0.clone(),
        'valid': model.lcx_valid_0.clone(),
    }


def restore_lcx(model, snap):
    """Restore LCX state from a snapshot."""
    model.lcx_keys_0.copy_(snap['keys'])
    model.lcx_values_0.copy_(snap['values'])
    model.lcx_heat_0.copy_(snap['heat'])
    model.lcx_valid_0.copy_(snap['valid'])


def lcx_norm(model):
    """Mean L2 norm of LCX value vectors."""
    return model.lcx_values_0.norm(dim=-1).mean().item()


# ── Config definitions ──────────────────────────────────────────
CONFIGS = [
    {'name': 'brain_only',       'use_lcx': False, 'evo': False},
    {'name': 'brain+whiteboard', 'use_lcx': True,  'evo': False},
    {'name': 'brain+evo_wb',     'use_lcx': True,  'evo': True},
]


def run_config(cfg):
    name = cfg['name']
    use_lcx = cfg['use_lcx']
    evo = cfg['evo']

    print(f'\n{"="*60}')
    print(f'  CONFIG: {name}')
    print(f'  D={D} depth={DEPTH} seq={SEQ_LEN} block={BLOCK_SIZE}')
    print(f'  LCX={use_lcx} evo={evo} slots={LCX_SLOTS} top_k={TOP_K}')
    print(f'  batch={BATCH} lr={LR} steps={STEPS} seed={SEED}')
    print(f'{"="*60}')

    model = build_model(use_lcx)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params={n_params:,}')

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    window_accs = []
    checkpoints = {}
    wins = 0
    rejects = 0
    t_start = time.time()

    for step in range(STEPS):
        t0 = time.time()
        if step < 3:
            print(f'  starting step {step}...', end='', flush=True)

        # ── 1. Snapshot LCX (evo only) ──
        if evo and use_lcx:
            snap_old = snapshot_lcx(model)

        # ── 2. Normal training step ──
        x, y = make_echo_batch(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS, DEVICE)
        opt.zero_grad()
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        # ── 3. Evolutionary selection (evo only) ──
        if evo and use_lcx:
            # Save the new LCX state (after training step wrote to it)
            snap_new = snapshot_lcx(model)

            # Fresh eval batch for unbiased comparison
            x_eval, y_eval = make_echo_batch(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS, DEVICE)

            with torch.no_grad():
                # Evaluate with NEW whiteboard (already in place)
                restore_lcx(model, snap_new)
                out_new = model(x_eval)
                if isinstance(out_new, tuple):
                    out_new = out_new[0]
                loss_new = F.binary_cross_entropy_with_logits(out_new, y_eval).item()

                # Evaluate with OLD whiteboard
                restore_lcx(model, snap_old)
                out_old = model(x_eval)
                if isinstance(out_old, tuple):
                    out_old = out_old[0]
                loss_old = F.binary_cross_entropy_with_logits(out_old, y_eval).item()

            # Tournament: keep whichever whiteboard gives lower loss
            if loss_new < loss_old:
                restore_lcx(model, snap_new)  # keep mutation
                wins += 1
            else:
                # old is already restored, keep it
                rejects += 1

        elapsed = time.time() - t0
        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT at step {step} ({elapsed:.0f}s)')
            return {'name': name, 'tail': 0.0, 'error': 'TIMEOUT'}

        # ── Metrics ──
        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            print(f'  NaN at step {step}')
            return {'name': name, 'tail': 0.0, 'error': 'NaN'}

        window_accs.append(acc)
        if len(window_accs) > WINDOW:
            window_accs.pop(0)
        smooth_acc = sum(window_accs) / len(window_accs)

        # Win rate (evo only)
        total_sel = wins + rejects
        win_rate = wins / total_sel if total_sel > 0 else 0.0

        # Whiteboard norm (LCX configs only)
        wb_norm = lcx_norm(model) if use_lcx else 0.0

        if step % 50 == 0 or step == STEPS - 1:
            checkpoints[step] = smooth_acc
            evo_str = ''
            if evo:
                evo_str = f' | win_rate={win_rate:.3f} ({wins}W/{rejects}R)'
            wb_str = f' | wb_norm={wb_norm:.3f}' if use_lcx else ''
            print(f'  step {step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f}{evo_str}{wb_str} | {elapsed:.2f}s',
                  flush=True)

        # Dashboard-compatible log line
        extra = f' win_rate={win_rate:.4f}' if evo else ''
        extra += f' wb_norm={wb_norm:.4f}' if use_lcx else ''
        with open(LIVE_LOG, 'a') as lf:
            lf.write(f'step {step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 {name}{extra}\n')

    total_time = time.time() - t_start
    tail_start = int(STEPS * 0.75)
    tail_accs = [a for s, a in checkpoints.items() if s >= tail_start]
    tail_avg = sum(tail_accs) / len(tail_accs) if tail_accs else 0.5

    result = {'name': name, 'tail': tail_avg, 'time': total_time}
    if evo:
        result['win_rate'] = win_rate
        result['wins'] = wins
        result['rejects'] = rejects

    print(f'\n  {name}: tail={tail_avg*100:.2f}%  time={total_time:.0f}s')
    if evo:
        print(f'  final win_rate={win_rate:.3f} ({wins}W/{rejects}R)')
    return result


if __name__ == '__main__':
    print('=' * 60)
    print('PROBE: EVOLUTIONARY WHITEBOARD — TOURNAMENT SELECTION ON LCX')
    print('=' * 60)
    print(f'  device={DEVICE}')
    print(f'  D={D} depth={DEPTH} seq={SEQ_LEN} block={BLOCK_SIZE} bits={NUM_BITS}')
    print(f'  LCX: {LCX_SLOTS} slots, hash mode, top_k={TOP_K}, key_dim={KEY_DIM}')
    print(f'  Steps: {STEPS}, LR={LR}, seed={SEED}')
    print(f'  3 configs: brain_only, brain+whiteboard, brain+evo_wb')
    print()

    with open(LIVE_LOG, 'w') as f:
        f.write(f'# probe_evo_whiteboard -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    results = []
    for cfg in CONFIGS:
        r = run_config(cfg)
        results.append(r)

    # ── Summary ──────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'SUMMARY: EVOLUTIONARY WHITEBOARD')
    print(f'{"="*60}')
    for r in results:
        err = r.get('error', '')
        evo_str = f'  win_rate={r["win_rate"]:.3f}' if 'win_rate' in r else ''
        t_str = f'  {r["time"]:.0f}s' if 'time' in r else ''
        print(f'  {r["name"]:25s}  tail={r["tail"]*100:.2f}%{evo_str}{t_str}  {err}')

    brain = next((r for r in results if r['name'] == 'brain_only'), None)
    wb    = next((r for r in results if r['name'] == 'brain+whiteboard'), None)
    evo   = next((r for r in results if r['name'] == 'brain+evo_wb'), None)

    if brain and wb and evo and all('error' not in r for r in [brain, wb, evo]):
        diff_wb  = wb['tail'] - brain['tail']
        diff_evo = evo['tail'] - brain['tail']
        diff_sel = evo['tail'] - wb['tail']

        print(f'\n  whiteboard vs brain:  {diff_wb*100:+.2f}%')
        print(f'  evo_wb vs brain:     {diff_evo*100:+.2f}%')
        print(f'  evo_wb vs whiteboard: {diff_sel*100:+.2f}%')

        # Verdicts
        if diff_sel > 0.03:
            print(f'\n  VERDICT: EVO_WINS — selection helps (+{diff_sel*100:.1f}%), noisy writes ARE the problem')
        elif diff_sel > 0.01:
            print(f'\n  VERDICT: EVO_MODERATE — selection helps slightly (+{diff_sel*100:.1f}%)')
        elif diff_sel > -0.01:
            print(f'\n  VERDICT: EVO_NO_EFFECT — selection doesn\'t matter, writes aren\'t the bottleneck')
        else:
            print(f'\n  VERDICT: EVO_HURTS — selection is harmful ({diff_sel*100:.1f}%), model needs write diversity')

        wr = evo.get('win_rate', 0.5)
        if wr > 0.70:
            print(f'  WIN RATE: {wr:.1%} — writes are mostly good, selection barely filters')
        elif wr > 0.40:
            print(f'  WIN RATE: {wr:.1%} — writes are coin-flip noise')
        else:
            print(f'  WIN RATE: {wr:.1%} — writes are mostly BAD, selection is saving the model')

    print(f'\n{"="*60}')
    print(f'Done. Log: {LIVE_LOG}')
