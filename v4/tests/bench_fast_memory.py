"""Fast iteration bench: repeating pattern task for long-range memory testing.

Task: a random block of P bytes repeats forever. The model must predict
each byte from the one P positions ago. When P > seq_len, this requires
cross-sequence memory (ring buffer state carry).

Dense supervision: ALL positions after the first period are supervised (~97%).
Random chance = 0.39% (1/256). Perfect memory = 100%.

Usage:
    python tests/bench_fast_memory.py                           # defaults: N=2, period=128
    python tests/bench_fast_memory.py --N 1                     # single expert
    python tests/bench_fast_memory.py --period 32 64 128 256    # sweep periods
    python tests/bench_fast_memory.py --model transformer       # transformer baseline
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Path setup (same as bench_ablation.py) ──
ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

from instnct import INSTNCT, set_ring_trace_enabled
from tiny_transformer import TinyTransformer


# ═══════════════════════════════════════════════════════════
#  DATA GENERATION
# ═══════════════════════════════════════════════════════════

def generate_repeating_pattern(B, length, period, seed=42):
    """Generate continuous byte streams with repeating patterns.

    A random block of `period` bytes repeats forever. The model must predict
    each byte from the one `period` positions ago.

    When period > seq_len: requires cross-sequence memory (ring state carry).
    Dense supervision: all positions after the first period are supervised.

    Args:
        B: batch size (independent streams)
        length: total bytes per stream
        period: repeat cycle length in bytes
        seed: random seed

    Returns:
        data: (B, length) long tensor — byte values 0-255
        mask: (B, length) float tensor — 1.0 on supervised positions
    """
    data = np.zeros((B, length), dtype=np.int64)
    mask = np.zeros((B, length), dtype=np.float32)

    for b in range(B):
        rng = np.random.RandomState(seed + b)
        pattern = rng.randint(0, 256, size=period)
        for pos in range(length):
            data[b, pos] = pattern[pos % period]
        # First period is unsupervised (model hasn't seen the pattern yet)
        mask[b, period:] = 1.0

    return torch.from_numpy(data), torch.from_numpy(mask)


# ═══════════════════════════════════════════════════════════
#  LOSS & ACCURACY
# ═══════════════════════════════════════════════════════════

def masked_ce_loss(logits, targets, mask):
    """Cross-entropy loss on mask=1 positions only.

    Args:
        logits: (B, T, 256)
        targets: (B, T) long
        mask: (B, T) float — 1.0 on supervised positions

    Returns:
        loss: scalar (mean over supervised positions)
        n_supervised: number of supervised positions
    """
    B, T, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)

    per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')

    n_sup = flat_mask.sum().clamp(min=1)
    loss = (per_token_loss * flat_mask).sum() / n_sup
    return loss, int(n_sup.item())


def masked_accuracy(logits, targets, mask):
    """Byte-match accuracy on mask=1 positions only.

    Returns:
        accuracy: float (0-1)
        n_supervised: int
    """
    preds = logits.argmax(dim=-1)  # (B, T)
    correct = (preds == targets).float()

    flat_correct = correct.reshape(-1)
    flat_mask = mask.reshape(-1)

    n_sup = flat_mask.sum().clamp(min=1)
    acc = (flat_correct * flat_mask).sum() / n_sup
    return acc.item(), int(n_sup.item())


TOPK_DIAG_KEYS = (
    'topk_mean_abs_circ_dist',
    'topk_outside_local_frac',
    'topk_attn_entropy',
    'topk_unique_slot_frac',
    'write_topk_mean_abs_circ_dist',
    'write_topk_outside_local_frac',
    'write_topk_unique_slot_frac',
)


def _circ_dist(a, b, M):
    delta = abs(int(a) - int(b))
    return min(delta, M - delta)


def _summarize_ring_trace(trace, M):
    if not trace or not trace.get('ptr_trace'):
        return None
    ptr_trace = trace['ptr_trace']
    read_idx_trace = trace['read_idx_trace']
    tap_idx_trace = trace.get('tap_idx_trace', [])
    write_idx_trace = trace['write_idx_trace']
    overlap_trace = trace['read_write_overlap_trace']
    ptr_jump = []
    read_center_dist = []
    tap_center_dist = []
    write_center_dist = []
    for i in range(1, len(ptr_trace)):
        ptr_jump.append(_circ_dist(ptr_trace[i - 1], ptr_trace[i], M))
    for step_idx, (center, read_idx, write_idx) in enumerate(zip(ptr_trace, read_idx_trace, write_idx_trace)):
        if read_idx:
            read_center_dist.append(sum(_circ_dist(center, idx, M) for idx in read_idx) / len(read_idx))
        if step_idx < len(tap_idx_trace):
            tap_idx = tap_idx_trace[step_idx]
            if tap_idx:
                tap_center_dist.append(sum(_circ_dist(center, idx, M) for idx in tap_idx) / len(tap_idx))
        if write_idx:
            write_center_dist.append(sum(_circ_dist(center, idx, M) for idx in write_idx) / len(write_idx))
    return {
        'steps_traced': len(ptr_trace),
        'ptr_unique_frac': sum(1 for v in trace['center_hist'] if v > 0) / max(len(trace['center_hist']), 1),
        'read_unique_frac': sum(1 for v in trace['read_hist'] if v > 0) / max(len(trace['read_hist']), 1),
        'tap_unique_frac': (
            sum(1 for v in trace.get('tap_hist', []) if v > 0) / max(len(trace.get('tap_hist', [])), 1)
            if trace.get('tap_hist') is not None else 0.0
        ),
        'write_unique_frac': sum(1 for v in trace['write_hist'] if v > 0) / max(len(trace['write_hist']), 1),
        'ptr_jump_mean': (sum(ptr_jump) / len(ptr_jump)) if ptr_jump else 0.0,
        'read_center_dist_mean': (sum(read_center_dist) / len(read_center_dist)) if read_center_dist else 0.0,
        'tap_center_dist_mean': (sum(tap_center_dist) / len(tap_center_dist)) if tap_center_dist else 0.0,
        'write_center_dist_mean': (sum(write_center_dist) / len(write_center_dist)) if write_center_dist else 0.0,
        'read_write_overlap_mean': (sum(overlap_trace) / len(overlap_trace)) if overlap_trace else 0.0,
    }


def _default_json_path() -> Path:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = ROOT / 'dev_notes' / 'telemetry'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f'{Path(__file__).stem}_{stamp}.json'


def _mean(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def _set_determinism(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════

def fresh_state_eval(model, data, mask, seq, period, device, n_seqs=10, context_mode='dotprod'):
    """Evaluate with fresh state (no carry) — tests if model cheats via state.

    Picks a random start offset aligned to period boundary, runs n_seqs
    consecutive sequences from scratch (state=None), measures accuracy.
    """
    model.eval()
    total_len = data.shape[1]
    # Start at a period boundary so the pattern is aligned
    max_start = total_len - (n_seqs + 1) * seq - 1
    if max_start < period:
        max_start = period
    start = (max_start // period) * period  # align to period

    all_correct = 0
    all_sup = 0
    with torch.no_grad():
        for s in range(n_seqs):
            pos = start + s * seq
            if pos + seq + 1 >= total_len:
                break
            x = data[:, pos:pos + seq]
            y = data[:, pos + 1:pos + seq + 1]
            m = mask[:, pos + 1:pos + seq + 1]
            logits, _ = model(x, S=context_mode, state=None)  # fresh state each time
            preds = logits.argmax(dim=-1)
            correct = (preds == y).float()
            all_correct += (correct * m).sum().item()
            all_sup += m.sum().item()

    model.train()
    return all_correct / max(all_sup, 1)


def s_zero_probe(model, data, mask, seq, period, device, n_seqs=10):
    """S=0 probe: evaluate with context scale forced to zero.

    If accuracy drops significantly: ring is truly being used.
    If accuracy stays: model bypasses ring (hidden shortcut).
    """
    model.eval()
    total_len = data.shape[1]
    max_start = total_len - (n_seqs + 1) * seq - 1
    if max_start < period:
        max_start = period
    start = (max_start // period) * period

    all_correct = 0
    all_sup = 0
    with torch.no_grad():
        state = None
        for s in range(n_seqs):
            pos = start + s * seq
            if pos + seq + 1 >= total_len:
                break
            x = data[:, pos:pos + seq]
            y = data[:, pos + 1:pos + seq + 1]
            m = mask[:, pos + 1:pos + seq + 1]
            logits, new_state = model(x, S=0.0, state=state)  # S=0 kills ring read
            if new_state is not None:
                state = {k: v.detach() for k, v in new_state.items()}
            preds = logits.argmax(dim=-1)
            correct = (preds == y).float()
            all_correct += (correct * m).sum().item()
            all_sup += m.sum().item()

    model.train()
    return all_correct / max(all_sup, 1)


def ring_diagnostics(model, state, device):
    """Measure ring health: SVD rank, adjacent cos_sim, slot diversity."""
    if state is None or 'ring' not in state:
        return {}
    ring = state['ring']  # (B, M, slot_dim)
    B, M_actual, sd = ring.shape
    slot_norms = ring.norm(dim=-1)
    active = (slot_norms > 0.1).float().mean().item()

    # Adjacent cosine similarity
    cos_adj = F.cosine_similarity(ring[:, :-1], ring[:, 1:], dim=-1)

    # SVD effective rank (on first batch element)
    ring_flat = ring[0]  # (M, slot_dim)
    try:
        U, S_vals, V = torch.svd(ring_flat)
        total_var = (S_vals ** 2).sum()
        cumvar = (S_vals ** 2).cumsum(0) / total_var
        rank_90 = (cumvar < 0.90).sum().item() + 1
        rank_95 = (cumvar < 0.95).sum().item() + 1
    except:
        rank_90, rank_95 = -1, -1

    return {
        'ring_active_pct': active * 100,
        'ring_adj_cos': cos_adj.mean().item(),
        'ring_svd_rank90': rank_90,
        'ring_svd_rank95': rank_95,
        'ring_slot_norm_mean': slot_norms.mean().item(),
    }


def run_one(N, period, steps, batch, seq, hidden_dim, M, slot_dim,
            model_type, device, io_split_mode='off', gated_write=False, lr=1e-3,
            log_every=100, seed=42, read_kernel_mode='vshape',
            write_address_mode='pointer', topk_k=2, ring_trace=False,
            pointer_mode='sequential', pointer_interp_mode='off', pointer_seam_mode='mod',
            mtaps_enabled=False, mtaps_lags=(1, 2, 4, 8, 16, 32),
            context_mode='dotprod', heartbeat_cb=None,
            R=1, embed_encoding='learned'):
    """Train one configuration and return results.

    Returns:
        dict with peak_acc, final_acc, fresh_acc, s0_acc, wall_time, n_params, history
    """
    _set_determinism(seed)

    # ── Generate data ──
    total_len = (steps + 20) * seq  # margin to avoid wrapping
    data, mask = generate_repeating_pattern(
        B=batch, length=total_len, period=period, seed=seed,
    )
    data = data.to(device)
    mask = mask.to(device)

    # ── Create model ──
    if model_type == 'instnct':
        use_topk_diag = read_kernel_mode == 'topk' or write_address_mode == 'content_topk'
        set_ring_trace_enabled(ring_trace)
        model = INSTNCT(
            M=M, hidden_dim=hidden_dim, slot_dim=slot_dim,
            N=N, R=R, embed_mode=True,
            kernel_mode='vshape',
            read_kernel_mode=read_kernel_mode,
            embed_encoding=embed_encoding,
            output_encoding='lowrank_c19',
            expert_weighting=False,
            checkpoint_chunks=0,
            bb_enabled=False,
            io_split_mode=io_split_mode,
            io_writer_count=1,
            io_output_from_readers_only=(io_split_mode == 'strict'),
            gated_write=gated_write,
            pointer_mode=pointer_mode,
            write_address_mode=write_address_mode,
            topk_K=topk_k,
            read_topk_K=topk_k,
            write_topk_K=topk_k,
            pointer_interp_mode=pointer_interp_mode,
            pointer_seam_mode=pointer_seam_mode,
            mtaps_enabled=mtaps_enabled,
            mtaps_lags=mtaps_lags,
        ).to(device)
        split_tag = f' io={io_split_mode}' if io_split_mode != 'off' else ''
        gw_tag = ' gated_write' if gated_write else ''
        rk_tag = f' read={read_kernel_mode}'
        wa_tag = f' write={write_address_mode}'
        pm_tag = f' ptr={pointer_mode}'
        pi_tag = '' if pointer_interp_mode == 'off' else f' interp={pointer_interp_mode}'
        ps_tag = '' if pointer_seam_mode == 'mod' else f' seam={pointer_seam_mode}'
        mt_tag = '' if not mtaps_enabled else f' mtaps={list(mtaps_lags)}'
        model_label = f'INSTNCT N={N}{split_tag}{gw_tag}{rk_tag}{wa_tag}{pm_tag}{pi_tag}{ps_tag}{mt_tag}'
        model._diag_enabled = use_topk_diag
    elif model_type == 'transformer':
        set_ring_trace_enabled(False)
        model = TinyTransformer(
            embed_mode=True, d_model=64, n_layers=2, n_heads=2, d_ff=256,
            max_seq=seq + 16,
        ).to(device)
        model_label = 'Transformer'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ── LR schedule: warmup then constant ──
    warmup = min(50, steps // 10)

    def get_lr(step):
        if step < warmup:
            return lr * step / max(warmup, 1)
        return lr  # constant after warmup

    # ── Print header ──
    sup_ratio = max(0, total_len - period) / total_len * 100
    print(f"\n  Model: {model_label} | hidden={hidden_dim} | M={M} | {n_params:,} params")
    print(f"  Data: period={period}, seq={seq}, supervised~{sup_ratio:.0f}%")
    print(f"  Training: {steps} steps, batch={batch}, lr={lr}")
    print(f"  Device: {device}\n")

    # ── Training loop ──
    state = None
    pos = 0
    peak_acc = 0.0
    peak_step = 0
    history = []
    diag_rows = {key: [] for key in TOPK_DIAG_KEYS}
    ring_trace_rows = None
    if ring_trace and model_type == 'instnct':
        ring_trace_rows = {
            'ptr_trace': [],
            'read_idx_trace': [],
            'read_weight_trace': [],
            'tap_idx_trace': [],
            'write_idx_trace': [],
            'write_weight_trace': [],
            'read_write_overlap_trace': [],
            'center_hist': [0 for _ in range(M)],
            'read_hist': [0 for _ in range(M)],
            'tap_hist': [0 for _ in range(M)],
            'write_hist': [0 for _ in range(M)],
        }
    t0 = time.perf_counter()
    if heartbeat_cb is not None:
        heartbeat_cb('start', 0, steps, {'model_type': model_type, 'period': period})

    for step in range(1, steps + 1):
        # Update LR
        current_lr = get_lr(step)
        for pg in opt.param_groups:
            pg['lr'] = current_lr

        # Get chunk
        x = data[:, pos:pos + seq]             # (B, seq)
        y = data[:, pos + 1:pos + seq + 1]     # (B, seq) — next byte
        m = mask[:, pos + 1:pos + seq + 1]     # (B, seq) — supervision

        # Forward
        logits, new_state = model(x, S=context_mode, state=state)

        # State carry (TBPTT for INSTNCT, None for transformer)
        if new_state is not None:
            state = {k: v.detach() for k, v in new_state.items()}
        else:
            state = None

        # Loss (only on supervised echo positions)
        loss, n_sup = masked_ce_loss(logits, y, m)

        # Backward
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        # Advance position
        pos += seq
        if pos + seq + 1 >= total_len:
            pos = 0
            state = None  # reset on wrap

        for key in TOPK_DIAG_KEYS:
            value = model._diag.get(key) if model_type == 'instnct' else None
            if value is not None:
                diag_rows[key].append(float(value))
        if ring_trace_rows is not None:
            trace = getattr(model, '_ring_trace', None)
            if trace is not None:
                for key in ('ptr_trace', 'read_idx_trace', 'read_weight_trace', 'tap_idx_trace', 'write_idx_trace', 'write_weight_trace', 'read_write_overlap_trace'):
                    ring_trace_rows[key].extend(trace.get(key, []))
                for key in ('center_hist', 'read_hist', 'tap_hist', 'write_hist'):
                    vals = trace.get(key, [])
                    ring_trace_rows[key] = [a + int(b) for a, b in zip(ring_trace_rows[key], vals)]

        # Log
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                acc, _ = masked_accuracy(logits, y, m)
            elapsed = time.perf_counter() - t0
            if acc > peak_acc:
                peak_acc = acc
                peak_step = step
            history.append((step, acc, loss.item(), elapsed))
            diag_suffix = ''
            if diag_rows['topk_mean_abs_circ_dist']:
                diag_suffix += (
                    f" rdist={diag_rows['topk_mean_abs_circ_dist'][-1]:.2f}"
                    f" rout={diag_rows['topk_outside_local_frac'][-1]:.3f}"
                )
            if diag_rows['write_topk_mean_abs_circ_dist']:
                diag_suffix += (
                    f" wdist={diag_rows['write_topk_mean_abs_circ_dist'][-1]:.2f}"
                    f" wout={diag_rows['write_topk_outside_local_frac'][-1]:.3f}"
                )
            print(f"  Step {step:5d} | echo_acc={acc*100:5.1f}% | "
                  f"loss={loss.item():.3f} | {elapsed:6.1f}s | "
                  f"sup={n_sup}{diag_suffix}")
            if heartbeat_cb is not None:
                heartbeat_cb(
                    'progress',
                    step,
                    steps,
                    {
                        'echo_acc': float(acc),
                        'loss': float(loss.item()),
                        'elapsed_s': float(elapsed),
                    },
                )

    wall = time.perf_counter() - t0
    final_acc = history[-1][1] if history else 0.0

    # ── Post-training eval ──
    fresh_acc = fresh_state_eval(model, data, mask, seq, period, device, context_mode=context_mode)
    s0_acc = s_zero_probe(model, data, mask, seq, period, device) if model_type == 'instnct' else -1.0

    # ── Ring diagnostics ──
    ring_diag = {}
    if model_type == 'instnct' and state is not None:
        ring_diag = ring_diagnostics(model, state, device)

    print(f"\n  Peak: {peak_acc*100:.1f}% @ step {peak_step} | "
          f"Final: {final_acc*100:.1f}% | Time: {wall:.1f}s | Speed: {wall/steps:.3f} s/step")
    print(f"  Fresh-state eval: {fresh_acc*100:.1f}% | S=0 probe: {s0_acc*100:.1f}%")
    if s0_acc >= 0:
        ring_dep = final_acc - s0_acc
        print(f"  Ring dependency: {ring_dep*100:+.1f}pp (final - S0)")
    if ring_diag:
        print(f"  Ring health: adj_cos={ring_diag.get('ring_adj_cos', -1):.4f} | "
              f"SVD rank(90%)={ring_diag.get('ring_svd_rank90', -1)} | "
              f"rank(95%)={ring_diag.get('ring_svd_rank95', -1)} | "
              f"slot_norm={ring_diag.get('ring_slot_norm_mean', -1):.3f}")
    if model_type == 'instnct' and gated_write:
        with torch.no_grad():
            erase_vals = [torch.sigmoid(model.erase_raw[i]).item() for i in range(model.N)]
            wgate_vals = [torch.sigmoid(model.write_gate_raw[i]).item() for i in range(model.N)]
        print(f"  Gated write: erase={erase_vals} write_gate={wgate_vals}")
    diag_means = {}
    for key in TOPK_DIAG_KEYS:
        avg = _mean(diag_rows[key])
        if avg is not None:
            diag_means[key] = avg
    ring_trace_summary = _summarize_ring_trace(ring_trace_rows, M) if ring_trace_rows is not None else None
    set_ring_trace_enabled(False)

    result = {
        'peak_acc': peak_acc,
        'peak_step': peak_step,
        'final_acc': final_acc,
        'fresh_acc': fresh_acc,
        's0_acc': s0_acc,
        'ring_dependency_pp': (final_acc - s0_acc) * 100.0 if s0_acc >= 0 else None,
        'wall_time': wall,
        'sec_per_step': wall / steps,
        'n_params': n_params,
        'period': period,
        'read_kernel_mode': read_kernel_mode,
        'write_address_mode': write_address_mode,
        'topk_k': topk_k,
        'mtaps_enabled': bool(mtaps_enabled),
        'mtaps_lags': list(mtaps_lags),
        'context_mode': context_mode,
        'history': history,
        'topk_diag_means': diag_means,
        **ring_diag,
    }
    result.update(diag_means)
    if ring_trace_rows is not None:
        result['ring_trace_summary'] = ring_trace_summary
        result['ring_trace'] = ring_trace_rows
    if heartbeat_cb is not None:
        heartbeat_cb(
            'done',
            steps,
            steps,
            {
                'final_acc': float(result['final_acc']),
                'time_s': float(result['wall_time']),
            },
        )
    return result


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Fast memory bench: repeating pattern')
    parser.add_argument('--N', type=int, default=2, help='Expert count (1 or 2)')
    parser.add_argument('--period', type=int, nargs='+', default=[128],
                        help='Pattern repeat period in bytes (sweep if multiple)')
    parser.add_argument('--steps', type=int, default=500, help='Training steps')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--seq', type=int, default=64, help='Sequence length')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden state width')
    parser.add_argument('--M', type=int, default=256, help='Ring slots')
    parser.add_argument('--slot-dim', type=int, default=64, help='Ring slot width')
    parser.add_argument('--model', choices=['instnct', 'transformer'], default='instnct')
    parser.add_argument('--io-split', choices=['off', 'strict'], default='off',
                        help='Hourglass I/O split mode (strict = writer/reader roles)')
    parser.add_argument('--gated-write', action='store_true',
                        help='Enable gated write (erase+write_gate anti-blob)')
    parser.add_argument('--device', default='auto', help='cuda, cpu, or auto')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--log-every', type=int, default=100, help='Log interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--read-kernel-mode', choices=['vshape', 'topk'], default='vshape',
                        help='Read addressing mode for INSTNCT.')
    parser.add_argument('--write-address-mode', choices=['pointer', 'content_topk'], default='pointer',
                        help='Write addressing mode for INSTNCT.')
    parser.add_argument('--topk-k', type=int, default=2, help='TopK for topk read/write modes.')
    parser.add_argument('--pointer-mode', choices=['sequential', 'learned', 'pilot'], default='sequential',
                        help='Pointer movement mode for INSTNCT.')
    parser.add_argument('--pointer-interp-mode', choices=['off', 'linear'], default='off',
                        help='Fractional pointer center mode for local read/write.')
    parser.add_argument('--pointer-seam-mode', choices=['mod', 'shortest_arc'], default='mod',
                        help='Wrap-seam handling for pointer updates.')
    parser.add_argument('--ring-trace', action='store_true', help='Capture full ring trace/histograms.')
    parser.add_argument('--json-out', default=None, help='Optional JSON results path.')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("  FAST MEMORY BENCH — Repeating Pattern, Sequential Mode")
    print("=" * 60)

    results = {}
    for period in args.period:
        results[period] = run_one(
            N=args.N, period=period, steps=args.steps,
            batch=args.batch, seq=args.seq,
            hidden_dim=args.hidden_dim, M=args.M, slot_dim=args.slot_dim,
            model_type=args.model, device=device,
            io_split_mode=args.io_split,
            gated_write=args.gated_write,
            lr=args.lr, log_every=args.log_every, seed=args.seed,
            read_kernel_mode=args.read_kernel_mode,
            write_address_mode=args.write_address_mode,
            topk_k=args.topk_k,
            ring_trace=args.ring_trace,
            pointer_mode=args.pointer_mode,
            pointer_interp_mode=args.pointer_interp_mode,
            pointer_seam_mode=args.pointer_seam_mode,
        )

    # Summary table (if sweep)
    if len(args.period) > 1:
        print("\n" + "=" * 60)
        print("  SWEEP SUMMARY")
        print("=" * 60)
        print(f"  {'Period':>6} | {'Peak':>6} | {'Final':>6} | {'Fresh':>6} | {'S=0':>6} | {'Time':>5} | {'Params':>8}")
        print(f"  {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*5} | {'-'*8}")
        for period in args.period:
            r = results[period]
            s0_str = f"{r['s0_acc']*100:5.1f}%" if r['s0_acc'] >= 0 else "  n/a "
            print(f"  {period:>6} | {r['peak_acc']*100:5.1f}% | "
                  f"{r['final_acc']*100:5.1f}% | {r['fresh_acc']*100:5.1f}% | "
                  f"{s0_str} | {r['wall_time']:>4.0f}s | "
                  f"{r['n_params']:>8,}")

    json_out = Path(args.json_out) if args.json_out else _default_json_path()
    payload = {
        'script': Path(__file__).name,
        'device': device,
        'seed': args.seed,
        'steps': args.steps,
        'batch': args.batch,
        'seq': args.seq,
        'hidden_dim': args.hidden_dim,
        'M': args.M,
        'slot_dim': args.slot_dim,
        'N': args.N,
        'model': args.model,
        'io_split_mode': args.io_split,
        'gated_write': bool(args.gated_write),
        'read_kernel_mode': args.read_kernel_mode,
        'write_address_mode': args.write_address_mode,
        'topk_k': args.topk_k,
        'pointer_mode': args.pointer_mode,
        'pointer_interp_mode': args.pointer_interp_mode,
        'pointer_seam_mode': args.pointer_seam_mode,
        'ring_trace': bool(args.ring_trace),
        'results': results,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"\n  JSON: {json_out}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
