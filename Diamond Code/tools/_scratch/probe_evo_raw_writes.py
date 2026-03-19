#!/usr/bin/env python3
"""
Probe: Raw Writes — Full overwrite instead of 1% EMA
======================================================
Same 3 configs as evo_whiteboard probe, but writes are full replacement
instead of EMA blend. This makes each write bold enough for evo selection
to actually measure the difference.

Hypothesis: evo selection failed because 1% EMA changes were too small
to detect. With full overwrites, the signal should be clear enough to
select on.

Comparison to previous probe (EMA writes):
  EMA:  brain_only=53.43%  wb=53.80%  evo=53.76%  win_rate=50.5%
  RAW:  ???
"""

import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR  = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG = os.path.join(LOG_DIR, 'probe_evo_raw_writes_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

from swarm_model import SwarmByteRingModel

DEVICE = torch.device('cpu')

# ── Hyperparams ─────────────────────────────────────────────────
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
    xs, ys = [], []
    for _ in range(batch_size):
        block = torch.randint(0, 2, (block_size, num_bits)).float()
        repeats = (seq_len + 2) // block_size + 1
        data = block.repeat(repeats, 1)[:seq_len + 1]
        xs.append(data[:seq_len])
        ys.append(data[1:seq_len + 1])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def build_model(use_lcx):
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


def patch_raw_writes(model):
    """Monkey-patch _lcx_flat_write to do full overwrite on top-1 slot
    instead of gated EMA. Also kills the anti-rut random write."""
    original_write = model._lcx_flat_write

    def raw_write(state, write_content, level=0):
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)
            write_content = write_content.unsqueeze(0)
        B = state.shape[0]

        # Write gate (still needed for aux loss tracking)
        with torch.amp.autocast('cuda', enabled=False):
            _wg_dtype = model.lcx_write_gate.weight.dtype
            _wc_norm = torch.nn.functional.normalize(write_content.to(_wg_dtype), dim=-1)
            gate_for_aux = torch.sigmoid(model.lcx_write_gate(_wc_norm))
        if model.training:
            if not hasattr(model, '_lcx_write_gate_accum'):
                model._lcx_write_gate_accum = []
            model._lcx_write_gate_accum.append(gate_for_aux)

        with torch.no_grad():
            query = model.lcx_route_query(state)
            query = torch.nn.functional.normalize(query, dim=-1)

            keys, values = model._lcx_level_bufs(level)
            scores = query @ keys.T
            # Top-1 only: full overwrite to the best-matching slot
            top_idx = scores.argmax(dim=-1)  # [B]

            heat = getattr(model, f'lcx_heat_{level}', None)
            valid = getattr(model, f'lcx_valid_{level}', None)

            for b in range(B):
                idx = top_idx[b].item()
                # FULL OVERWRITE — no EMA
                values[idx] = write_content[b]
                keys[idx] = torch.nn.functional.normalize(
                    0.5 * keys[idx] + 0.5 * query[b], dim=-1)
                if heat is not None:
                    heat[idx] = min(heat[idx] + 1, 32767)
                if valid is not None:
                    valid[idx] = True

            # NO anti-rut random write

    model._lcx_flat_write = raw_write
    return model


def snapshot_lcx(model):
    return {
        'keys':   model.lcx_keys_0.clone(),
        'values': model.lcx_values_0.clone(),
        'heat':   model.lcx_heat_0.clone(),
        'valid':  model.lcx_valid_0.clone(),
    }


def restore_lcx(model, snap):
    model.lcx_keys_0.copy_(snap['keys'])
    model.lcx_values_0.copy_(snap['values'])
    model.lcx_heat_0.copy_(snap['heat'])
    model.lcx_valid_0.copy_(snap['valid'])


def lcx_norm(model):
    return model.lcx_values_0.norm(dim=-1).mean().item()


CONFIGS = [
    {'name': 'brain_only',     'use_lcx': False, 'evo': False, 'raw': False},
    {'name': 'wb_ema',         'use_lcx': True,  'evo': False, 'raw': False},
    {'name': 'wb_raw',         'use_lcx': True,  'evo': False, 'raw': True},
    {'name': 'wb_raw+evo',     'use_lcx': True,  'evo': True,  'raw': True},
]


def run_config(cfg):
    name = cfg['name']
    use_lcx = cfg['use_lcx']
    evo = cfg['evo']
    raw = cfg['raw']

    print(f'\n{"="*60}')
    print(f'  CONFIG: {name}')
    print(f'  D={D} depth={DEPTH} seq={SEQ_LEN} block={BLOCK_SIZE}')
    print(f'  LCX={use_lcx} evo={evo} raw_writes={raw}')
    print(f'  batch={BATCH} lr={LR} steps={STEPS} seed={SEED}')
    print(f'{"="*60}')

    model = build_model(use_lcx)
    if raw and use_lcx:
        patch_raw_writes(model)
        print(f'  [PATCHED] Raw writes enabled (full overwrite, no EMA)')

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

        if evo and use_lcx:
            snap_old = snapshot_lcx(model)

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

        if evo and use_lcx:
            snap_new = snapshot_lcx(model)
            x_eval, y_eval = make_echo_batch(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS, DEVICE)

            with torch.no_grad():
                restore_lcx(model, snap_new)
                out_new = model(x_eval)
                if isinstance(out_new, tuple):
                    out_new = out_new[0]
                loss_new = F.binary_cross_entropy_with_logits(out_new, y_eval).item()

                restore_lcx(model, snap_old)
                out_old = model(x_eval)
                if isinstance(out_old, tuple):
                    out_old = out_old[0]
                loss_old = F.binary_cross_entropy_with_logits(out_old, y_eval).item()

            if loss_new < loss_old:
                restore_lcx(model, snap_new)
                wins += 1
            else:
                rejects += 1

        elapsed = time.time() - t0
        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT at step {step} ({elapsed:.0f}s)')
            return {'name': name, 'tail': 0.0, 'error': 'TIMEOUT'}

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

        total_sel = wins + rejects
        win_rate = wins / total_sel if total_sel > 0 else 0.0
        wb_norm = lcx_norm(model) if use_lcx else 0.0

        if step % 50 == 0 or step == STEPS - 1:
            checkpoints[step] = smooth_acc
            evo_str = ''
            if evo:
                evo_str = f' | wr={win_rate:.3f} ({wins}W/{rejects}R)'
            wb_str = f' | wb_norm={wb_norm:.3f}' if use_lcx else ''
            print(f'  step {step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f}{evo_str}{wb_str} | {elapsed:.2f}s',
                  flush=True)

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
    print('PROBE: RAW WRITES vs EMA — does bold writing help?')
    print('=' * 60)
    print(f'  device={DEVICE}')
    print(f'  D={D} depth={DEPTH} seq={SEQ_LEN} block={BLOCK_SIZE} bits={NUM_BITS}')
    print(f'  LCX: {LCX_SLOTS} slots, top_k={TOP_K}, key_dim={KEY_DIM}')
    print(f'  Steps: {STEPS}, LR={LR}, seed={SEED}')
    print(f'  4 configs: brain_only, wb_ema, wb_raw, wb_raw+evo')
    print()

    with open(LIVE_LOG, 'w') as f:
        f.write(f'# probe_evo_raw_writes -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    results = []
    for cfg in CONFIGS:
        r = run_config(cfg)
        results.append(r)

    print(f'\n{"="*60}')
    print(f'SUMMARY: RAW WRITES vs EMA')
    print(f'{"="*60}')
    for r in results:
        err = r.get('error', '')
        evo_str = f'  wr={r["win_rate"]:.3f}' if 'win_rate' in r else ''
        t_str = f'  {r["time"]:.0f}s' if 'time' in r else ''
        print(f'  {r["name"]:20s}  tail={r["tail"]*100:.2f}%{evo_str}{t_str}  {err}')

    brain = next((r for r in results if r['name'] == 'brain_only'), None)
    ema   = next((r for r in results if r['name'] == 'wb_ema'), None)
    raw   = next((r for r in results if r['name'] == 'wb_raw'), None)
    revo  = next((r for r in results if r['name'] == 'wb_raw+evo'), None)

    if all(r and 'error' not in r for r in [brain, ema, raw, revo]):
        print(f'\n  COMPARISONS:')
        print(f'  wb_ema vs brain:     {(ema["tail"]-brain["tail"])*100:+.2f}%  (baseline LCX advantage)')
        print(f'  wb_raw vs brain:     {(raw["tail"]-brain["tail"])*100:+.2f}%  (raw writes alone)')
        print(f'  wb_raw vs wb_ema:    {(raw["tail"]-ema["tail"])*100:+.2f}%  (raw vs EMA)')
        print(f'  wb_raw+evo vs brain: {(revo["tail"]-brain["tail"])*100:+.2f}%  (raw + selection)')
        print(f'  wb_raw+evo vs wb_raw:{(revo["tail"]-raw["tail"])*100:+.2f}%  (selection effect)')

        print(f'\n  EMA BASELINE (previous probe):')
        print(f'    brain=53.43%  wb_ema=53.80%  evo_ema=53.76%  wr=50.5%')

    print(f'\n{"="*60}')
    print(f'Done. Log: {LIVE_LOG}')
