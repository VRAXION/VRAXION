#!/usr/bin/env python3
"""
Probe: LCX Snapshot Tournament — Which Whiteboard State Actually Helps?
========================================================================
Phase 1: Train 400 steps with LCX, snapshot every 4 steps -> 100 snapshots
Phase 2: Freeze weights, one fresh eval batch, test all 100 snapshots
Phase 3: Rank, analyze top vs bottom, build consensus, test it

Isolates whiteboard contribution: same weights, same eval data,
only the whiteboard content varies.
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
LIVE_LOG = os.path.join(LOG_DIR, 'probe_lcx_snapshot_tournament_live.log')
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
SNAP_EVERY = 4  # snapshot every N steps -> 100 snapshots


def make_echo_batch(batch_size, seq_len, block_size, num_bits, device):
    xs, ys = [], []
    for _ in range(batch_size):
        block = torch.randint(0, 2, (block_size, num_bits)).float()
        repeats = (seq_len + 2) // block_size + 1
        data = block.repeat(repeats, 1)[:seq_len + 1]
        xs.append(data[:seq_len])
        ys.append(data[1:seq_len + 1])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def build_model():
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
        use_lcx=True,
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


if __name__ == '__main__':
    print('=' * 70)
    print('PROBE: LCX SNAPSHOT TOURNAMENT')
    print('=' * 70)
    print(f'  D={D} depth={DEPTH} seq={SEQ_LEN} block={BLOCK_SIZE} bits={NUM_BITS}')
    print(f'  LCX: {LCX_SLOTS} slots, top_k={TOP_K}, key_dim={KEY_DIM}')
    print(f'  Steps: {STEPS}, snapshot every {SNAP_EVERY} -> {STEPS // SNAP_EVERY} snapshots')
    print()

    with open(LIVE_LOG, 'w') as f:
        f.write(f'# probe_lcx_snapshot_tournament -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Train + Collect Snapshots
    # ═══════════════════════════════════════════════════════════════
    print('PHASE 1: Training + collecting snapshots...')
    print('-' * 70)

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params={n_params:,}')

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    snapshots = []  # list of (step, snapshot_dict, training_loss)
    t_start = time.time()

    for step in range(STEPS):
        t0 = time.time()
        if step < 3:
            print(f'  starting step {step}...', end='', flush=True)

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

        elapsed = time.time() - t0
        loss_val = loss.item()

        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT at step {step} ({elapsed:.0f}s)')
            sys.exit(1)

        if math.isnan(loss_val):
            print(f'  NaN at step {step}')
            sys.exit(1)

        # Snapshot every SNAP_EVERY steps
        if step % SNAP_EVERY == 0:
            snap = snapshot_lcx(model)
            wb_norm = snap['values'].norm(dim=-1).mean().item()
            snapshots.append({
                'step': step,
                'snap': snap,
                'train_loss': loss_val,
                'wb_norm': wb_norm,
            })

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if step % 50 == 0 or step == STEPS - 1:
            wb_norm = model.lcx_values_0.norm(dim=-1).mean().item()
            print(f'  step {step:4d} | loss {loss_val:.6f} | acc {acc:.4f} | '
                  f'wb_norm={wb_norm:.3f} | snaps={len(snapshots)} | {elapsed:.2f}s',
                  flush=True)

        with open(LIVE_LOG, 'a') as lf:
            lf.write(f'step {step} | loss {loss_val:.6f} | '
                     f'acc={acc:.4f} RD:{elapsed:.4f} traction={acc:.4f} '
                     f'shard=0/0 phase1_train\n')

    phase1_time = time.time() - t_start
    print(f'\n  Phase 1 done: {len(snapshots)} snapshots collected in {phase1_time:.0f}s')

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Tournament — test all snapshots on one eval batch
    # ═══════════════════════════════════════════════════════════════
    print(f'\n{"="*70}')
    print('PHASE 2: Tournament — testing all snapshots on fresh eval batch...')
    print('-' * 70)

    model.eval()
    # Fresh eval batch with different seed
    torch.manual_seed(9999)
    eval_x, eval_y = make_echo_batch(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS, DEVICE)
    print(f'  Eval batch generated (seed=9999)')

    tournament_results = []
    t2_start = time.time()

    for i, entry in enumerate(snapshots):
        restore_lcx(model, entry['snap'])
        with torch.no_grad():
            out = model(eval_x)
            if isinstance(out, tuple):
                out = out[0]
            eval_loss = F.binary_cross_entropy_with_logits(out, eval_y).item()
            pred = (out > 0).float()
            eval_acc = (pred == eval_y).float().mean().item()

        tournament_results.append({
            'idx': i,
            'step': entry['step'],
            'eval_loss': eval_loss,
            'eval_acc': eval_acc,
            'train_loss': entry['train_loss'],
            'wb_norm': entry['wb_norm'],
        })

        if i % 20 == 0:
            print(f'  tested {i+1}/{len(snapshots)}: '
                  f'step={entry["step"]} eval_loss={eval_loss:.6f} acc={eval_acc:.4f}',
                  flush=True)

    phase2_time = time.time() - t2_start
    print(f'\n  Phase 2 done: {len(tournament_results)} snapshots tested in {phase2_time:.0f}s')

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Analysis
    # ═══════════════════════════════════════════════════════════════
    print(f'\n{"="*70}')
    print('PHASE 3: Analysis')
    print('=' * 70)

    # Sort by eval loss (lower = better)
    ranked = sorted(tournament_results, key=lambda r: r['eval_loss'])
    losses = [r['eval_loss'] for r in tournament_results]
    accs = [r['eval_acc'] for r in tournament_results]

    # ── Spread test ──
    spread = max(losses) - min(losses)
    acc_spread = max(accs) - min(accs)
    mean_loss = sum(losses) / len(losses)
    std_loss = (sum((l - mean_loss)**2 for l in losses) / len(losses)) ** 0.5

    print(f'\n--- SPREAD TEST ---')
    print(f'  Loss:  min={min(losses):.6f}  max={max(losses):.6f}  spread={spread:.6f}')
    print(f'  Acc:   min={min(accs):.4f}  max={max(accs):.4f}  spread={acc_spread:.4f}')
    print(f'  Loss mean={mean_loss:.6f}  std={std_loss:.6f}  CoV={std_loss/mean_loss*100:.2f}%')

    if spread > 0.01:
        print(f'  VERDICT: WHITEBOARD_MATTERS — {spread:.4f} loss spread across snapshots')
    elif spread > 0.001:
        print(f'  VERDICT: WHITEBOARD_WEAK — {spread:.6f} loss spread (marginal)')
    else:
        print(f'  VERDICT: WHITEBOARD_INVISIBLE — {spread:.6f} loss spread (noise floor)')

    # ── Trend test ──
    print(f'\n--- TREND TEST (loss vs snapshot index) ---')
    # Split into 5 quintiles and show mean loss per quintile
    q_size = len(tournament_results) // 5
    for qi in range(5):
        start = qi * q_size
        end = start + q_size if qi < 4 else len(tournament_results)
        q_losses = [tournament_results[i]['eval_loss'] for i in range(start, end)]
        q_accs = [tournament_results[i]['eval_acc'] for i in range(start, end)]
        q_steps = [tournament_results[i]['step'] for i in range(start, end)]
        q_mean_loss = sum(q_losses) / len(q_losses)
        q_mean_acc = sum(q_accs) / len(q_accs)
        step_range = f'{q_steps[0]:3d}-{q_steps[-1]:3d}'
        bar_len = int((q_mean_acc - 0.5) * 200)  # scale for visibility
        bar = '=' * max(0, bar_len)
        print(f'  steps {step_range}: loss={q_mean_loss:.6f}  acc={q_mean_acc:.4f}  [{bar}]')

    # Correlation: step vs eval_loss
    steps_list = [r['step'] for r in tournament_results]
    n = len(steps_list)
    mean_step = sum(steps_list) / n
    cov = sum((s - mean_step) * (l - mean_loss) for s, l in zip(steps_list, losses)) / n
    std_step = (sum((s - mean_step)**2 for s in steps_list) / n) ** 0.5
    corr = cov / (std_step * std_loss) if std_step > 0 and std_loss > 0 else 0
    print(f'  Step-loss correlation: r={corr:.3f}')
    if corr < -0.3:
        print(f'  TREND: LATER_IS_BETTER (whiteboard improves over training)')
    elif corr > 0.3:
        print(f'  TREND: LATER_IS_WORSE (whiteboard degrades over training)')
    else:
        print(f'  TREND: NO_TREND (whiteboard quality is random across training)')

    # ── Ranked table (top 10 + bottom 10) ──
    print(f'\n--- TOP 10 SNAPSHOTS (lowest eval loss) ---')
    print(f'  {"rank":>4s}  {"step":>5s}  {"eval_loss":>10s}  {"eval_acc":>9s}  {"train_loss":>11s}  {"wb_norm":>8s}')
    print(f'  {"-"*55}')
    for i, r in enumerate(ranked[:10]):
        print(f'  {i+1:4d}  {r["step"]:5d}  {r["eval_loss"]:10.6f}  {r["eval_acc"]:9.4f}  '
              f'{r["train_loss"]:11.6f}  {r["wb_norm"]:8.3f}')

    print(f'\n--- BOTTOM 10 SNAPSHOTS (highest eval loss) ---')
    print(f'  {"rank":>4s}  {"step":>5s}  {"eval_loss":>10s}  {"eval_acc":>9s}  {"train_loss":>11s}  {"wb_norm":>8s}')
    print(f'  {"-"*55}')
    for i, r in enumerate(ranked[-10:]):
        print(f'  {n-9+i:4d}  {r["step"]:5d}  {r["eval_loss"]:10.6f}  {r["eval_acc"]:9.4f}  '
              f'{r["train_loss"]:11.6f}  {r["wb_norm"]:8.3f}')

    # ── X-ray overlay: top-10 vs bottom-10 slot analysis ──
    print(f'\n--- X-RAY: TOP-10 vs BOTTOM-10 slot comparison ---')

    top10_snaps = [snapshots[r['idx']]['snap'] for r in ranked[:10]]
    bot10_snaps = [snapshots[r['idx']]['snap'] for r in ranked[-10:]]

    # Stack values: [10, 500, 128]
    top_vals = torch.stack([s['values'] for s in top10_snaps])
    bot_vals = torch.stack([s['values'] for s in bot10_snaps])

    # Per-slot mean across top-10 and bottom-10
    top_mean = top_vals.mean(dim=0)  # [500, 128]
    bot_mean = bot_vals.mean(dim=0)  # [500, 128]

    # Per-slot L2 distance between top and bottom means
    slot_diff = (top_mean - bot_mean).norm(dim=-1)  # [500]

    # Per-slot variance within top-10 (consensus measure)
    top_var = top_vals.var(dim=0).mean(dim=-1)  # [500] mean var across dims
    bot_var = bot_vals.var(dim=0).mean(dim=-1)  # [500]

    # Per-slot norm in top-10 mean
    top_norms = top_mean.norm(dim=-1)  # [500]

    # Summary stats
    print(f'  Slot diff (top vs bot):  mean={slot_diff.mean():.4f}  '
          f'max={slot_diff.max():.4f}  min={slot_diff.min():.4f}')
    print(f'  Top-10 internal var:     mean={top_var.mean():.4f}  '
          f'max={top_var.max():.4f}')
    print(f'  Bottom-10 internal var:  mean={bot_var.mean():.4f}  '
          f'max={bot_var.max():.4f}')

    # Find the most-different slots (potential signal carriers)
    top_diff_slots = slot_diff.topk(10)
    print(f'\n  Most-different slots (potential signal carriers):')
    print(f'  {"slot":>5s}  {"diff":>8s}  {"top_norm":>9s}  {"top_var":>8s}  {"bot_var":>8s}')
    for si, sv in zip(top_diff_slots.indices, top_diff_slots.values):
        print(f'  {si.item():5d}  {sv.item():8.4f}  {top_norms[si].item():9.4f}  '
              f'{top_var[si].item():8.4f}  {bot_var[si].item():8.4f}')

    # Find most-consistent slots in top-10 (lowest variance)
    top_consistent = top_var.topk(10, largest=False)
    print(f'\n  Most-consistent slots in top-10 (lowest variance = consensus):')
    print(f'  {"slot":>5s}  {"var":>8s}  {"norm":>9s}  {"diff":>8s}')
    for si, sv in zip(top_consistent.indices, top_consistent.values):
        print(f'  {si.item():5d}  {sv.item():8.6f}  {top_norms[si].item():9.4f}  '
              f'{slot_diff[si].item():8.4f}')

    # ── Consensus build: average top-10 into one "dream" whiteboard ──
    print(f'\n--- CONSENSUS BUILD ---')

    consensus_snap = {
        'keys':   torch.stack([s['keys'] for s in top10_snaps]).mean(dim=0),
        'values': top_mean,
        'heat':   top10_snaps[0]['heat'],  # heat doesn't affect forward pass
        'valid':  top10_snaps[0]['valid'],
    }

    # Test consensus
    restore_lcx(model, consensus_snap)
    with torch.no_grad():
        out = model(eval_x)
        if isinstance(out, tuple):
            out = out[0]
        consensus_loss = F.binary_cross_entropy_with_logits(out, eval_y).item()
        pred = (out > 0).float()
        consensus_acc = (pred == eval_y).float().mean().item()

    best_loss = ranked[0]['eval_loss']
    worst_loss = ranked[-1]['eval_loss']
    median_loss = ranked[len(ranked)//2]['eval_loss']

    print(f'  Consensus (avg top-10):  loss={consensus_loss:.6f}  acc={consensus_acc:.4f}')
    print(f'  Best individual (#1):    loss={best_loss:.6f}  acc={ranked[0]["eval_acc"]:.4f}  (step {ranked[0]["step"]})')
    print(f'  Median individual:       loss={median_loss:.6f}')
    print(f'  Worst individual (#100): loss={worst_loss:.6f}  acc={ranked[-1]["eval_acc"]:.4f}  (step {ranked[-1]["step"]})')

    if consensus_loss < best_loss:
        print(f'  VERDICT: CONSENSUS_WINS — averaging beats best individual by {best_loss - consensus_loss:.6f}')
        print(f'  -> Distillation works! Averaging filters noise.')
    elif consensus_loss < median_loss:
        print(f'  VERDICT: CONSENSUS_GOOD — better than median, worse than best')
        print(f'  -> Partial signal in averaging, but best snapshot has unique value')
    else:
        print(f'  VERDICT: CONSENSUS_FAILS — worse than median')
        print(f'  -> Snapshots are too different to average meaningfully')

    # ── Also test: empty whiteboard (all zeros) as absolute baseline ──
    print(f'\n--- BASELINE: empty whiteboard (zeros) ---')
    zero_snap = {
        'keys':   torch.zeros_like(snapshots[0]['snap']['keys']),
        'values': torch.zeros_like(snapshots[0]['snap']['values']),
        'heat':   snapshots[0]['snap']['heat'],
        'valid':  torch.zeros_like(snapshots[0]['snap']['valid']),
    }
    restore_lcx(model, zero_snap)
    with torch.no_grad():
        out = model(eval_x)
        if isinstance(out, tuple):
            out = out[0]
        zero_loss = F.binary_cross_entropy_with_logits(out, eval_y).item()
        pred = (out > 0).float()
        zero_acc = (pred == eval_y).float().mean().item()

    print(f'  Empty whiteboard:  loss={zero_loss:.6f}  acc={zero_acc:.4f}')
    print(f'  vs best snapshot:  {zero_loss - best_loss:+.6f} loss')
    print(f'  vs consensus:      {zero_loss - consensus_loss:+.6f} loss')

    # ── Final summary ──
    print(f'\n{"="*70}')
    print(f'FINAL SUMMARY')
    print(f'{"="*70}')
    print(f'  Snapshots:   {len(snapshots)}')
    print(f'  Loss spread: {spread:.6f} ({"MATTERS" if spread > 0.01 else "WEAK" if spread > 0.001 else "INVISIBLE"})')
    print(f'  Trend:       r={corr:.3f} ({"later better" if corr < -0.3 else "later worse" if corr > 0.3 else "no trend"})')
    print(f'  Best snap:   step {ranked[0]["step"]:3d}  loss={best_loss:.6f}')
    print(f'  Consensus:   loss={consensus_loss:.6f}  ({"BETTER" if consensus_loss < best_loss else "WORSE"} than best)')
    print(f'  Empty:       loss={zero_loss:.6f}')
    print(f'  Time:        phase1={phase1_time:.0f}s  phase2={phase2_time:.0f}s')
    print(f'\n{"="*70}')
    print(f'Done. Log: {LIVE_LOG}')
