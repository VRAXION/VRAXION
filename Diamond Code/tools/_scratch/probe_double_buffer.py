#!/usr/bin/env python3
"""
Probe: Double-Buffer LCX -- Golden (read-only) + Scratch (write-only) + Sleep Selection
========================================================================================
Hypothesis: Cross-batch contamination kills whiteboard signal because reads return
content written by a different batch. Solution: keep a proven-good "golden" LCX for
reads. Writes go to a separate scratch buffer. Periodically "sleep": test accumulated
snapshots on multiple fresh eval batches, promote the winner as new golden.

3 configs compared:
  1. brain_only       -- no LCX (baseline)
  2. normal_lcx       -- standard LCX (read+write same buffer)
  3. double_buffer     -- golden reads + scratch writes + periodic sleep

Sleep cycle:
  - Every SLEEP_INTERVAL steps, freeze weights
  - Test all accumulated scratch snapshots on EVAL_BATCHES fresh batches
  - Rank by average loss across batches
  - Promote winner to golden, clear scratch snapshots
  - Resume training

Device: auto-detect (cuda if available, else cpu)
"""

import argparse
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
os.makedirs(LOG_DIR, exist_ok=True)

from swarm_model import SwarmByteRingModel

# -- Hyperparams --
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
STEPS      = 1000
LR         = 1e-4
WARMUP     = 30
SEED       = 42
STEP_TIMEOUT = 60
WINDOW     = 50

# -- Double-buffer params --
SLEEP_INTERVAL  = 200    # steps between sleep cycles
SNAP_EVERY      = 10     # snapshot scratch every N steps within a cycle
EVAL_BATCHES    = 3      # number of fresh eval batches per sleep cycle


def make_echo_batch(batch_size, seq_len, block_size, num_bits, device):
    xs, ys = [], []
    for _ in range(batch_size):
        block = torch.randint(0, 2, (block_size, num_bits)).float()
        repeats = (seq_len + 2) // block_size + 1
        data = block.repeat(repeats, 1)[:seq_len + 1]
        xs.append(data[:seq_len])
        ys.append(data[1:seq_len + 1])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def build_model(use_lcx, device):
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
    ).to(device)
    model.train()
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


def patch_double_buffer(model, golden_snap):
    """Monkey-patch the model so reads come from golden, writes go to live buffers.

    We intercept _lcx_level_bufs to return golden keys/values for reads,
    and the real (scratch) buffers for writes. The distinction is tracked
    via a flag: model._db_reading.
    """
    model._db_golden = golden_snap
    model._db_reading = False

    original_bufs = model._lcx_level_bufs

    def patched_bufs(level):
        if model._db_reading:
            # Return golden copies (read-only path)
            g = model._db_golden
            return g['keys'], g['values']
        else:
            # Return live scratch buffers (write path)
            return original_bufs(level)

    model._lcx_level_bufs = patched_bufs

    # Patch _lcx_flat_read to set the flag
    original_read = model._lcx_flat_read

    def patched_read(state, level=0):
        model._db_reading = True
        try:
            return original_read(state, level=level)
        finally:
            model._db_reading = False

    model._lcx_flat_read = patched_read
    return model


def update_golden(model, new_snap):
    """Update the golden snapshot that reads pull from."""
    model._db_golden = {
        'keys':   new_snap['keys'].clone(),
        'values': new_snap['values'].clone(),
        'heat':   new_snap['heat'].clone(),
        'valid':  new_snap['valid'].clone(),
    }


def run_sleep_cycle(model, snapshots, device, cycle_num, log_file):
    """Sleep cycle: test all snapshots on EVAL_BATCHES fresh batches.
    Returns the best snapshot and its average loss."""
    n_snaps = len(snapshots)
    print(f'\n  --- SLEEP CYCLE {cycle_num} ({n_snaps} snapshots, {EVAL_BATCHES} eval batches) ---')

    # Generate fresh eval batches with different seeds
    eval_data = []
    for i in range(EVAL_BATCHES):
        torch.manual_seed(SEED * 1000 + cycle_num * 100 + i)
        ex, ey = make_echo_batch(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS, device)
        eval_data.append((ex, ey))

    # Save current scratch state to restore after
    current_scratch = snapshot_lcx(model)

    # Test each snapshot
    losses = []
    model.eval()
    with torch.no_grad():
        for si, snap in enumerate(snapshots):
            restore_lcx(model, snap)
            batch_losses = []
            for ex, ey in eval_data:
                out = model(ex)
                if isinstance(out, tuple):
                    out = out[0]
                loss = F.binary_cross_entropy_with_logits(out, ey).item()
                batch_losses.append(loss)
            avg_loss = sum(batch_losses) / len(batch_losses)
            losses.append(avg_loss)
    model.train()

    # Also test the current golden
    golden_losses = []
    model.eval()
    with torch.no_grad():
        restore_lcx(model, model._db_golden)
        for ex, ey in eval_data:
            out = model(ex)
            if isinstance(out, tuple):
                out = out[0]
            loss = F.binary_cross_entropy_with_logits(out, ey).item()
            golden_losses.append(loss)
    model.train()
    golden_avg = sum(golden_losses) / len(golden_losses)

    # Restore scratch
    restore_lcx(model, current_scratch)

    # Find winner
    best_idx = min(range(n_snaps), key=lambda i: losses[i])
    best_loss = losses[best_idx]
    worst_loss = max(losses)
    spread = worst_loss - best_loss

    promoted = best_loss < golden_avg
    result_str = 'PROMOTED' if promoted else 'GOLDEN_KEPT'

    print(f'  spread={spread:.6f}  best={best_loss:.6f} (snap {best_idx})'
          f'  worst={worst_loss:.6f}  golden={golden_avg:.6f}')
    print(f'  -> {result_str}')

    with open(log_file, 'a') as lf:
        lf.write(f'SLEEP {cycle_num} | n_snaps={n_snaps} spread={spread:.6f} '
                 f'best={best_loss:.6f} worst={worst_loss:.6f} '
                 f'golden={golden_avg:.6f} {result_str}\n')

    if promoted:
        return snapshots[best_idx], best_loss, True
    else:
        return model._db_golden, golden_avg, False


def run_config(cfg, device, log_file):
    name = cfg['name']
    use_lcx = cfg['use_lcx']
    double_buf = cfg.get('double_buffer', False)

    print(f'\n{"="*60}')
    print(f'  CONFIG: {name}')
    print(f'  D={D} depth={DEPTH} seq={SEQ_LEN} block={BLOCK_SIZE}')
    print(f'  LCX={use_lcx} double_buffer={double_buf}')
    if double_buf:
        print(f'  sleep_interval={SLEEP_INTERVAL} snap_every={SNAP_EVERY}'
              f'  eval_batches={EVAL_BATCHES}')
    print(f'  batch={BATCH} lr={LR} steps={STEPS} seed={SEED}')
    print(f'  device={device}')
    print(f'{"="*60}')

    model = build_model(use_lcx, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params={n_params:,}')

    # Setup double buffer
    if double_buf and use_lcx:
        golden_snap = snapshot_lcx(model)
        patch_double_buffer(model, golden_snap)
        print(f'  [PATCHED] Double-buffer mode: golden reads + scratch writes')

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    window_accs = []
    checkpoints = {}
    scratch_snapshots = []
    sleep_results = []
    promotions = 0
    cycle_num = 0
    t_start = time.time()

    for step in range(STEPS):
        t0 = time.time()
        if step < 3:
            print(f'  starting step {step}...', end='', flush=True)

        # Normal training step
        x, y = make_echo_batch(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS, device)
        opt.zero_grad()
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        elapsed = time.time() - t0
        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT at step {step} ({elapsed:.0f}s)')
            return {'name': name, 'tail': 0.0, 'error': 'TIMEOUT'}

        # Metrics
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

        wb_norm = lcx_norm(model) if use_lcx else 0.0

        # Snapshot scratch (double buffer only)
        if double_buf and use_lcx and (step + 1) % SNAP_EVERY == 0:
            scratch_snapshots.append(snapshot_lcx(model))

        # Sleep cycle (double buffer only)
        if double_buf and use_lcx and (step + 1) % SLEEP_INTERVAL == 0:
            cycle_num += 1
            winner_snap, winner_loss, was_promoted = run_sleep_cycle(
                model, scratch_snapshots, device, cycle_num, log_file)

            if was_promoted:
                update_golden(model, winner_snap)
                promotions += 1

            scratch_snapshots = []  # clear for next cycle

        if step % 50 == 0 or step == STEPS - 1:
            checkpoints[step] = smooth_acc
            db_str = ''
            if double_buf:
                db_str = f' | sleeps={cycle_num} promos={promotions}'
            wb_str = f' | wb_norm={wb_norm:.3f}' if use_lcx else ''
            print(f'  step {step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f}{wb_str}{db_str} | {elapsed:.2f}s',
                  flush=True)

        extra = ''
        if use_lcx:
            extra += f' wb_norm={wb_norm:.4f}'
        if double_buf:
            extra += f' sleeps={cycle_num} promos={promotions}'
        with open(log_file, 'a') as lf:
            lf.write(f'step {step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 {name}{extra}\n')

    total_time = time.time() - t_start
    tail_start = int(STEPS * 0.75)
    tail_accs = [a for s, a in checkpoints.items() if s >= tail_start]
    tail_avg = sum(tail_accs) / len(tail_accs) if tail_accs else 0.5

    result = {'name': name, 'tail': tail_avg, 'time': total_time}
    if double_buf:
        result['promotions'] = promotions
        result['sleep_cycles'] = cycle_num

    # Save checkpoint for Phase 2 continuation
    ckpt_dir = os.path.join(LOG_DIR, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'probe_db_{name}.pt')
    save_dict = {
        'model_state': model.state_dict(),
        'optimizer_state': opt.state_dict(),
        'step': STEPS,
        'config': cfg,
        'tail': tail_avg,
        'window_accs': window_accs,
    }
    if use_lcx:
        save_dict['lcx_snap'] = snapshot_lcx(model)
        if double_buf:
            save_dict['golden_snap'] = {
                k: v.cpu().clone() for k, v in model._db_golden.items()
            }
    torch.save(save_dict, ckpt_path)
    print(f'  checkpoint saved: {ckpt_path}')
    result['ckpt_path'] = ckpt_path

    print(f'\n  {name}: tail={tail_avg*100:.2f}%  time={total_time:.0f}s')
    if double_buf:
        print(f'  sleep_cycles={cycle_num}  promotions={promotions}')
    return result


PHASE2_STEPS = 500  # continuation steps after Phase 1


def run_phase2(ckpt_path, device, log_file, keep_sleep=False):
    """Phase 2: Load checkpoint, continue training.
    If keep_sleep=False: normal LCX (no double-buffer, no sleep).
    If keep_sleep=True: continue with double-buffer + sleep cycles.
    Both forks start from the same checkpoint (same weights + golden LCX)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    name = cfg['name']
    use_lcx = cfg['use_lcx']
    start_step = ckpt['step']
    fork_tag = 'with_sleep' if keep_sleep else 'no_sleep'
    fork_name = f'{name}_{fork_tag}'

    print(f'\n{"="*60}')
    print(f'  PHASE 2: {fork_name}')
    print(f'  Continue {PHASE2_STEPS} steps, sleep={"ON" if keep_sleep else "OFF"}')
    print(f'  Loaded from step {start_step}, tail was {ckpt["tail"]*100:.2f}%')
    print(f'{"="*60}')

    model = build_model(use_lcx, device)
    model.load_state_dict(ckpt['model_state'])
    model.train()

    # Restore golden LCX as starting state
    if 'golden_snap' in ckpt:
        golden = {k: v.to(device) for k, v in ckpt['golden_snap'].items()}
        restore_lcx(model, golden)
        print(f'  Restored golden LCX (sleep-selected best)')
    elif 'lcx_snap' in ckpt:
        snap = {k: v.to(device) for k, v in ckpt['lcx_snap'].items()}
        restore_lcx(model, snap)

    # Setup double-buffer if keeping sleep
    if keep_sleep and use_lcx:
        golden_snap = snapshot_lcx(model)
        patch_double_buffer(model, golden_snap)
        print(f'  [PATCHED] Double-buffer mode continues')

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    opt.load_state_dict(ckpt['optimizer_state'])

    window_accs = list(ckpt.get('window_accs', []))
    checkpoints = {}
    scratch_snapshots = []
    promotions = 0
    cycle_num = 0
    t_start = time.time()

    for step in range(PHASE2_STEPS):
        global_step = start_step + step
        t0 = time.time()

        x, y = make_echo_batch(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS, device)
        opt.zero_grad()
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        elapsed = time.time() - t0

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            print(f'  NaN at step {global_step}')
            return {'name': fork_name, 'tail': 0.0, 'error': 'NaN'}

        window_accs.append(acc)
        if len(window_accs) > WINDOW:
            window_accs.pop(0)
        smooth_acc = sum(window_accs) / len(window_accs)
        wb_norm = lcx_norm(model) if use_lcx else 0.0

        # Snapshot + sleep (keep_sleep fork only)
        if keep_sleep and use_lcx and (step + 1) % SNAP_EVERY == 0:
            scratch_snapshots.append(snapshot_lcx(model))

        if keep_sleep and use_lcx and (step + 1) % SLEEP_INTERVAL == 0:
            cycle_num += 1
            winner_snap, winner_loss, was_promoted = run_sleep_cycle(
                model, scratch_snapshots, device, cycle_num, log_file)
            if was_promoted:
                update_golden(model, winner_snap)
                promotions += 1
            scratch_snapshots = []

        if step % 50 == 0 or step == PHASE2_STEPS - 1:
            checkpoints[step] = smooth_acc
            wb_str = f' | wb_norm={wb_norm:.3f}' if use_lcx else ''
            sleep_str = f' | sleeps={cycle_num} promos={promotions}' if keep_sleep else ''
            print(f'  step {global_step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f}{wb_str}{sleep_str} | {elapsed:.2f}s', flush=True)

        with open(log_file, 'a') as lf:
            extra = f' wb_norm={wb_norm:.4f}' if use_lcx else ''
            lf.write(f'step {global_step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 {fork_name}{extra}\n')

    total_time = time.time() - t_start
    tail_start = int(PHASE2_STEPS * 0.75)
    tail_accs = [a for s, a in checkpoints.items() if s >= tail_start]
    tail_avg = sum(tail_accs) / len(tail_accs) if tail_accs else 0.5

    result = {'name': fork_name, 'tail': tail_avg, 'time': total_time,
              'p1_tail': ckpt['tail']}
    if keep_sleep:
        result['promotions'] = promotions
        result['sleep_cycles'] = cycle_num
    print(f'\n  {fork_name}: tail={tail_avg*100:.2f}%  (was {ckpt["tail"]*100:.2f}% at end of P1)')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cpu, cuda, or auto')
    parser.add_argument('--configs', type=str, default='all',
                        help='Configs to run: all, baselines, double_buffer')
    parser.add_argument('--phase2', action='store_true',
                        help='Run Phase 2 continuation from saved checkpoints')
    parser.add_argument('--phase2-fork', type=str, default=None,
                        help='Run Phase 2 from a specific checkpoint file')
    parser.add_argument('--keep-sleep', action='store_true',
                        help='Phase 2: continue WITH double-buffer sleep (default: no sleep)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Phase 2 fork mode: load specific checkpoint and continue
    if args.phase2_fork:
        log_file = os.path.join(LOG_DIR, 'probe_double_buffer_live.log')
        tag = 'with_sleep' if args.keep_sleep else 'no_sleep'
        print(f'PHASE 2 FORK: {tag} from {args.phase2_fork}')
        print(f'  device={device}')
        r = run_phase2(args.phase2_fork, device, log_file, keep_sleep=args.keep_sleep)
        print(f'\n  RESULT: {r["name"]}  tail={r["tail"]*100:.2f}%')
        return

    ALL_CONFIGS = [
        {'name': 'brain_only',     'use_lcx': False, 'double_buffer': False},
        {'name': 'normal_lcx',     'use_lcx': True,  'double_buffer': False},
        {'name': 'double_buffer',  'use_lcx': True,  'double_buffer': True},
    ]

    if args.configs == 'baselines':
        configs = [c for c in ALL_CONFIGS if not c.get('double_buffer')]
    elif args.configs == 'double_buffer':
        configs = [c for c in ALL_CONFIGS if c.get('double_buffer')]
    else:
        configs = ALL_CONFIGS

    log_file = os.path.join(LOG_DIR, 'probe_double_buffer_live.log')

    print('=' * 60)
    print('PROBE: DOUBLE-BUFFER LCX -- GOLDEN READS + SCRATCH WRITES')
    print('=' * 60)
    print(f'  device={device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={D} depth={DEPTH} seq={SEQ_LEN} block={BLOCK_SIZE} bits={NUM_BITS}')
    print(f'  LCX: {LCX_SLOTS} slots, top_k={TOP_K}, key_dim={KEY_DIM}')
    print(f'  Steps: {STEPS}, LR={LR}, seed={SEED}')
    print(f'  Sleep: every {SLEEP_INTERVAL} steps, {SNAP_EVERY}-step snapshots, '
          f'{EVAL_BATCHES} eval batches')
    print(f'  Configs: {[c["name"] for c in configs]}')
    print()

    with open(log_file, 'w') as f:
        f.write(f'# probe_double_buffer -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    results = []
    for cfg in configs:
        r = run_config(cfg, device, log_file)
        results.append(r)

    # Summary
    print(f'\n{"="*60}')
    print(f'SUMMARY: DOUBLE-BUFFER LCX')
    print(f'{"="*60}')
    for r in results:
        err = r.get('error', '')
        db_str = f'  promos={r["promotions"]}/{r["sleep_cycles"]}' if 'promotions' in r else ''
        t_str = f'  {r["time"]:.0f}s' if 'time' in r else ''
        print(f'  {r["name"]:20s}  tail={r["tail"]*100:.2f}%{db_str}{t_str}  {err}')

    brain = next((r for r in results if r['name'] == 'brain_only'), None)
    normal = next((r for r in results if r['name'] == 'normal_lcx'), None)
    dbuf = next((r for r in results if r['name'] == 'double_buffer'), None)

    if brain and normal and dbuf and all('error' not in r for r in [brain, normal, dbuf]):
        print(f'\n  COMPARISONS:')
        print(f'  normal_lcx vs brain:     {(normal["tail"]-brain["tail"])*100:+.2f}%  (baseline LCX)')
        print(f'  double_buf vs brain:     {(dbuf["tail"]-brain["tail"])*100:+.2f}%  (double-buffer)')
        print(f'  double_buf vs normal:    {(dbuf["tail"]-normal["tail"])*100:+.2f}%  (sleep selection effect)')

        if dbuf['tail'] > normal['tail'] + 0.01:
            print(f'\n  VERDICT: DOUBLE_BUFFER_WINS -- sleep selection helps! (+{(dbuf["tail"]-normal["tail"])*100:.1f}%)')
        elif dbuf['tail'] > normal['tail'] - 0.005:
            print(f'\n  VERDICT: DOUBLE_BUFFER_NEUTRAL -- no clear advantage')
        else:
            print(f'\n  VERDICT: DOUBLE_BUFFER_HURTS -- golden reads worse than live reads')

    elif brain and normal and all('error' not in r for r in [brain, normal]):
        print(f'\n  normal_lcx vs brain: {(normal["tail"]-brain["tail"])*100:+.2f}%')

    elif dbuf and 'error' not in dbuf:
        print(f'\n  double_buffer tail: {dbuf["tail"]*100:.2f}%')

    # Phase 2: auto-continuation (only if --phase2 flag or all Phase 1 succeeded)
    if args.phase2:
        ckpt_dir = os.path.join(LOG_DIR, 'checkpoints')
        import glob as globmod
        ckpt_files = sorted(globmod.glob(os.path.join(ckpt_dir, 'probe_db_*.pt')))
        if not ckpt_files:
            print('  No checkpoints found!')
        else:
            print(f'\n{"="*60}')
            print(f'PHASE 2: Loading from saved checkpoints (no_sleep mode)')
            print(f'{"="*60}')
            p2_results = []
            for ckpt_path in ckpt_files:
                r2 = run_phase2(ckpt_path, device, log_file, keep_sleep=args.keep_sleep)
                p2_results.append(r2)

            print(f'\n{"="*60}')
            print(f'PHASE 2 SUMMARY')
            print(f'{"="*60}')
            for r2 in p2_results:
                p1_tail = r2.get('p1_tail', 0)
                delta = r2['tail'] - p1_tail
                sleep_str = f'  promos={r2["promotions"]}' if 'promotions' in r2 else ''
                print(f'  {r2["name"]:30s}  p1={p1_tail*100:.2f}% -> p2={r2["tail"]*100:.2f}%  '
                      f'delta={delta*100:+.2f}%{sleep_str}  {r2["time"]:.0f}s')

    print(f'\n{"="*60}')
    print(f'Done. Log: {log_file}')


if __name__ == '__main__':
    main()
