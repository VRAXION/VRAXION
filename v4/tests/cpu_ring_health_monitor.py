"""CPU ring health monitor — watches write_gate, alpha, ring_delta for collapse.

Usage:
    python tests/cpu_ring_health_monitor.py [--steps 5000] [--log-every 25]

Designed to catch the write-gate collapse that was killing ring memory by step 7K.
Uses synthetic random byte data (no WikiText dependency needed).
Reports ring vitals at regular intervals with ALERT markers for collapse signals.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
for subdir in ("model", "training"):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import instnct  # type: ignore[import-not-found]
from model_factory import build_model_from_spec  # type: ignore[import-not-found]


def build_cpu_model(seed: int = 42, hidden_dim: int = 64, M: int = 64,
                    slot_dim: int = 16, min_write_strength: float = 0.002) -> instnct.INSTNCT:
    """Build a small INSTNCT model for CPU ring health testing."""
    torch.manual_seed(seed)
    spec = {
        'embed_mode': True,
        'hidden_dim': hidden_dim,
        'M': M,
        'slot_dim': slot_dim,
        'N': 1,
        'R': 1,
        'gated_write': False,
        'bb_enabled': False,
        'bb_gate_bias': 0.0,
        'bb_scale': 0.1,
        'bb_tau': 4.0,
        'bb_gate_mode': 'learned',
        'topk_K': 8,
        'read_topk_K': 2,
        'write_topk_K': 2,
        'write_address_mode': 'pointer',
        'pointer_mode': 'sequential',
        'kernel_mode': 'vshape',
        'read_kernel_mode': 'vshape',
        'write_mode': 'replace',
        'replace_impl': 'dense',
        'checkpoint_chunks': 0,
        'expert_weighting': False,
        'embed_encoding': 'learned',
        'output_encoding': 'learned',
        'min_write_strength': min_write_strength,
    }
    record = {'type': 'instnct', 'build_spec': spec}
    model = build_model_from_spec(record, 'cpu')
    return model


def ring_snapshot(ring: torch.Tensor) -> torch.Tensor:
    """Detached clone of ring state for delta computation."""
    return ring.detach().clone()


def main():
    parser = argparse.ArgumentParser(description='CPU ring health monitor')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seq', type=int, default=16)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--M', type=int, default=64)
    parser.add_argument('--slot-dim', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-write-strength', type=float, default=0.002)
    args = parser.parse_args()

    print(f'VRAXION Ring Health Monitor (CPU)')
    print(f'=' * 70)
    print(f'  steps={args.steps}  batch={args.batch}  seq={args.seq}')
    print(f'  hidden={args.hidden_dim}  M={args.M}  slot={args.slot_dim}')
    print(f'  min_write_strength={args.min_write_strength}  lr={args.lr}')
    print(f'  xavier write_gate init: YES (gain=0.01)')
    print(f'=' * 70)

    model = build_cpu_model(
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        M=args.M,
        slot_dim=args.slot_dim,
        min_write_strength=args.min_write_strength,
    )
    model._diag_enabled = True
    model.train()

    # Print initial write_gate weights info
    for i, wg in enumerate(model.write_gate):
        w = wg.weight.data
        b = wg.bias.data
        print(f'  write_gate[{i}] init: weight_norm={w.norm():.6f}  '
              f'weight_std={w.std():.6f}  bias={b.item():.4f}  '
              f'sigmoid(bias)={torch.sigmoid(b).item():.4f}')

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  params: {param_count:,} total, {trainable:,} trainable')
    print()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    state = None
    t0 = time.time()

    # Track history for trend detection
    alpha_history = []
    write_str_history = []
    ring_delta_history = []
    loss_history = []
    prev_ring = None

    header = (
        f'{"step":>6}  {"loss":>7}  {"acc":>5}  '
        f'{"alpha":>6} {"a_min":>6} {"a_max":>6}  '
        f'{"ws_mean":>7} {"ws_min":>7} {"wg_logit":>8}  '
        f'{"r_norm":>7} {"r_delta":>7} {"r_slot":>7}  '
        f'{"h_norm":>7} {"gnorm":>6}  {"status"}'
    )
    print(header)
    print('-' * len(header))

    alert_count = 0

    for step in range(1, args.steps + 1):
        # Random byte data — ring dynamics are what matters, not the task
        xb = torch.randint(0, 256, (args.batch, args.seq))
        yb = torch.randint(0, 256, (args.batch, args.seq))
        mask = torch.ones(args.batch, args.seq)

        logits, new_state = model(xb, S='dotprod', state=state)
        if new_state is not None:
            state = {k: v.detach() for k, v in new_state.items()}

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.view(-1),
            reduction='none',
        )
        loss = (loss * mask.view(-1)).sum() / mask.sum()

        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item()
        opt.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = ((preds == yb).float() * mask).sum() / mask.sum()

        loss_val = loss.item()
        acc_val = acc.item()
        loss_history.append(loss_val)

        # Get diag values
        d = model._diag
        alpha_mean = d.get('alpha_0_mean', 0.0)
        alpha_min = d.get('alpha_0_min', 0.0)
        alpha_max = d.get('alpha_0_max', 0.0)
        ws_mean = d.get('write_strength_0_mean', 0.0)
        ws_min = d.get('write_strength_0_min', 0.0)
        wg_logit = d.get('write_gate_logit_0_mean', 0.0)
        ring_norm = d.get('ring_norm', 0.0)
        ring_slot = d.get('ring_slot_mean', 0.0)
        h_norm = d.get('hidden_final_norm_0', 0.0)

        alpha_history.append(alpha_mean)
        write_str_history.append(ws_mean)

        # Ring delta: how much did ring change this step?
        ring_state = state.get('ring') if state else None
        ring_delta = 0.0
        if ring_state is not None:
            if prev_ring is not None:
                ring_delta = (ring_state - prev_ring).norm().item()
            prev_ring = ring_state.detach().clone()
        ring_delta_history.append(ring_delta)

        if step % args.log_every == 0 or step == 1:
            # Detect collapse signals
            status = 'OK'
            alerts = []

            # Check write gate collapse
            if ws_mean < 0.05:
                alerts.append('LOW_WRITE')
            if len(write_str_history) >= 200:
                recent = write_str_history[-100:]
                older = write_str_history[-200:-100]
                if sum(recent) / len(recent) < sum(older) / len(older) * 0.5:
                    alerts.append('WRITE_DECAY')

            # Check alpha saturation (stuck near 0 or 1)
            if alpha_mean < 0.05:
                alerts.append('ALPHA_DEAD')
            if alpha_max - alpha_min < 0.01 and step > 100:
                alerts.append('ALPHA_FLAT')

            # Check ring frozen
            if step > 100 and ring_delta < 1e-6:
                alerts.append('RING_FROZEN')
            if len(ring_delta_history) >= 200:
                recent_rd = ring_delta_history[-100:]
                older_rd = ring_delta_history[-200:-100]
                if sum(recent_rd) / max(len(recent_rd), 1) < sum(older_rd) / max(len(older_rd), 1) * 0.1:
                    alerts.append('RING_DYING')

            if alerts:
                status = ' | '.join(f'ALERT:{a}' for a in alerts)
                alert_count += len(alerts)

            elapsed = time.time() - t0
            print(
                f'{step:6d}  {loss_val:7.4f}  {acc_val:5.3f}  '
                f'{alpha_mean:6.4f} {alpha_min:6.4f} {alpha_max:6.4f}  '
                f'{ws_mean:7.4f} {ws_min:7.4f} {wg_logit:8.4f}  '
                f'{ring_norm:7.2f} {ring_delta:7.4f} {ring_slot:7.4f}  '
                f'{h_norm:7.4f} {gn:6.2f}  {status}'
            )

    # Final summary
    elapsed = time.time() - t0
    print()
    print(f'=' * 70)
    print(f'RING HEALTH SUMMARY after {args.steps} steps ({elapsed:.1f}s)')
    print(f'=' * 70)

    last_100_ws = write_str_history[-100:] if len(write_str_history) >= 100 else write_str_history
    last_100_alpha = alpha_history[-100:] if len(alpha_history) >= 100 else alpha_history
    last_100_rd = ring_delta_history[-100:] if len(ring_delta_history) >= 100 else ring_delta_history
    first_100_ws = write_str_history[:100] if len(write_str_history) >= 100 else write_str_history
    first_100_rd = ring_delta_history[:100] if len(ring_delta_history) >= 100 else ring_delta_history

    print(f'  Write strength:  first100_avg={sum(first_100_ws)/len(first_100_ws):.6f}  '
          f'last100_avg={sum(last_100_ws)/len(last_100_ws):.6f}')
    print(f'  Alpha:           last100_avg={sum(last_100_alpha)/len(last_100_alpha):.6f}  '
          f'min={min(last_100_alpha):.6f}  max={max(last_100_alpha):.6f}')
    print(f'  Ring delta:      first100_avg={sum(first_100_rd)/max(len(first_100_rd),1):.6f}  '
          f'last100_avg={sum(last_100_rd)/max(len(last_100_rd),1):.6f}')
    print(f'  Final loss:      {sum(loss_history[-100:])/min(100,len(loss_history)):.4f}')
    print(f'  Total alerts:    {alert_count}')

    # Write gate weight analysis
    for i, wg in enumerate(model.write_gate):
        w = wg.weight.data
        b = wg.bias.data
        print(f'  write_gate[{i}] final: weight_norm={w.norm():.6f}  '
              f'weight_std={w.std():.6f}  bias={b.item():.4f}  '
              f'sigmoid(bias)={torch.sigmoid(b).item():.4f}')

    # Verdict
    ws_ratio = (sum(last_100_ws) / len(last_100_ws)) / max(sum(first_100_ws) / len(first_100_ws), 1e-9)
    rd_ratio = (sum(last_100_rd) / max(len(last_100_rd), 1)) / max(sum(first_100_rd) / max(len(first_100_rd), 1), 1e-9)

    print()
    if ws_ratio > 0.5 and rd_ratio > 0.1:
        print('  VERDICT: RING HEALTHY - write gate and ring delta sustained')
    elif ws_ratio > 0.3:
        print('  VERDICT: RING MARGINAL - some write decay but still active')
    else:
        print('  VERDICT: RING COLLAPSED - write gate died, ring is frozen memory')


if __name__ == '__main__':
    main()
