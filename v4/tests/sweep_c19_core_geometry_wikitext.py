"""Deterministic sweep of C19 core geometry on WikiText-103.

This isolates the core geometry by forcing a fixed C inside the activation,
instead of letting the model's learnable C drift. It also compares whether
the standard linear tail matters at all in the current regime.

Core:
  dual-phi periodic parabolic C19, rho fixed at 4.0

Search axes:
  - fixed C values
  - tail mode:
      linear   -> standard linear tail after |x| > K*C
      periodic -> no tail; pure periodic core everywhere

Usage:
  python tests/sweep_c19_core_geometry_wikitext.py
  python tests/sweep_c19_core_geometry_wikitext.py --steps 100 --c-values 2.618,3.1415926535,6.283185307 --tail-modes linear,periodic
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

V4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V4_ROOT / 'model'))
sys.path.insert(0, str(V4_ROOT / 'training'))

from train import ByteDataset, func_discover_dat, func_maskloss_ce
from model_factory import build_model_from_spec

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = (math.sqrt(5) - 1) / 2
TOPK_READ_DIAG_KEYS = (
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
    write_idx_trace = trace['write_idx_trace']
    read_write_overlap_trace = trace['read_write_overlap_trace']
    ptr_jump = []
    read_center_dist = []
    write_center_dist = []
    for i in range(1, len(ptr_trace)):
        ptr_jump.append(_circ_dist(ptr_trace[i - 1], ptr_trace[i], M))
    for center, read_idx, write_idx in zip(ptr_trace, read_idx_trace, write_idx_trace):
        if read_idx:
            read_center_dist.append(sum(_circ_dist(center, idx, M) for idx in read_idx) / len(read_idx))
        if write_idx:
            write_center_dist.append(sum(_circ_dist(center, idx, M) for idx in write_idx) / len(write_idx))
    center_hist = trace['center_hist']
    read_hist = trace['read_hist']
    write_hist = trace['write_hist']
    return {
        'steps_traced': len(ptr_trace),
        'ptr_unique_frac': sum(1 for v in center_hist if v > 0) / max(len(center_hist), 1),
        'read_unique_frac': sum(1 for v in read_hist if v > 0) / max(len(read_hist), 1),
        'write_unique_frac': sum(1 for v in write_hist if v > 0) / max(len(write_hist), 1),
        'ptr_jump_mean': (sum(ptr_jump) / len(ptr_jump)) if ptr_jump else 0.0,
        'read_center_dist_mean': (sum(read_center_dist) / len(read_center_dist)) if read_center_dist else 0.0,
        'write_center_dist_mean': (sum(write_center_dist) / len(write_center_dist)) if write_center_dist else 0.0,
        'read_write_overlap_mean': (sum(read_write_overlap_trace) / len(read_write_overlap_trace)) if read_write_overlap_trace else 0.0,
    }


def _default_json_path():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = V4_ROOT / 'dev_notes' / 'telemetry'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f'{Path(__file__).stem}_{stamp}.json'


def _set_determinism(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _label_c(c_value):
    named = [
        ('phi', PHI),
        ('phi2', PHI * PHI),
        ('pi/phi', math.pi / PHI),
        ('pi', math.pi),
        ('2pi', 2 * math.pi),
    ]
    for name, ref in named:
        if abs(c_value - ref) < 1e-5:
            return name
    return f'{c_value:.3f}'


class ActivationTelemetry:
    def __init__(self, sample_per_call=1024, max_sample_size=65536):
        self.sample_per_call = sample_per_call
        self.max_sample_size = max_sample_size
        self.call_count = 0
        self.total_values = 0
        self.tail_values = 0
        self.abs_sum = 0.0
        self.scaled_abs_sum = 0.0
        self.max_abs_x = 0.0
        self.max_abs_over_c = 0.0
        self.abs_samples = []
        self.scaled_abs_samples = []
        self.ring_idx_samples = []

    def _append_sample(self, bucket, sample):
        bucket.append(sample)
        merged = torch.cat(bucket)
        if merged.numel() > self.max_sample_size:
            idx = torch.randint(0, merged.numel(), (self.max_sample_size,), device=merged.device)
            merged = merged[idx]
        bucket[:] = [merged]

    def observe(self, x, c_value, tail_k):
        with torch.no_grad():
            abs_x = x.detach().abs()
            scaled_abs = abs_x / c_value
            ring_idx = torch.floor(scaled_abs)

            self.call_count += 1
            self.total_values += abs_x.numel()
            self.tail_values += int((scaled_abs > tail_k).sum().item())
            self.abs_sum += float(abs_x.sum().item())
            self.scaled_abs_sum += float(scaled_abs.sum().item())
            self.max_abs_x = max(self.max_abs_x, float(abs_x.max().item()))
            self.max_abs_over_c = max(self.max_abs_over_c, float(scaled_abs.max().item()))

            flat_abs = abs_x.reshape(-1)
            flat_scaled = scaled_abs.reshape(-1)
            flat_ring = ring_idx.reshape(-1)
            k = min(self.sample_per_call, flat_abs.numel())
            if k > 0:
                if flat_abs.numel() > k:
                    idx = torch.randint(0, flat_abs.numel(), (k,), device=flat_abs.device)
                    abs_sample = flat_abs[idx].float().cpu()
                    scaled_sample = flat_scaled[idx].float().cpu()
                    ring_sample = flat_ring[idx].float().cpu()
                else:
                    abs_sample = flat_abs.float().cpu()
                    scaled_sample = flat_scaled.float().cpu()
                    ring_sample = flat_ring.float().cpu()
                self._append_sample(self.abs_samples, abs_sample)
                self._append_sample(self.scaled_abs_samples, scaled_sample)
                self._append_sample(self.ring_idx_samples, ring_sample)

    def summary(self):
        if self.total_values == 0:
            return {
                'activation_calls': 0,
                'activation_values': 0,
                'tail_hit_pct': 0.0,
                'mean_abs_x': 0.0,
                'mean_abs_over_c': 0.0,
                'p95_abs_x': 0.0,
                'p99_abs_x': 0.0,
                'max_abs_x': 0.0,
                'p95_abs_over_c': 0.0,
                'p99_abs_over_c': 0.0,
                'max_abs_over_c': 0.0,
                'p95_ring_idx': 0.0,
                'p99_ring_idx': 0.0,
                'max_ring_idx': 0.0,
                'quantile_sample_size': 0,
            }

        abs_sample = torch.cat(self.abs_samples) if self.abs_samples else torch.empty(0)
        scaled_sample = (
            torch.cat(self.scaled_abs_samples) if self.scaled_abs_samples else torch.empty(0)
        )
        ring_sample = (
            torch.cat(self.ring_idx_samples) if self.ring_idx_samples else torch.empty(0)
        )
        sample_size = int(abs_sample.numel())

        def q(sample, quant):
            if sample.numel() == 0:
                return 0.0
            return float(torch.quantile(sample, quant).item())

        return {
            'activation_calls': int(self.call_count),
            'activation_values': int(self.total_values),
            'tail_hit_pct': 100.0 * self.tail_values / self.total_values,
            'mean_abs_x': self.abs_sum / self.total_values,
            'mean_abs_over_c': self.scaled_abs_sum / self.total_values,
            'p95_abs_x': q(abs_sample, 0.95),
            'p99_abs_x': q(abs_sample, 0.99),
            'max_abs_x': self.max_abs_x,
            'p95_abs_over_c': q(scaled_sample, 0.95),
            'p99_abs_over_c': q(scaled_sample, 0.99),
            'max_abs_over_c': self.max_abs_over_c,
            'p95_ring_idx': q(ring_sample, 0.95),
            'p99_ring_idx': q(ring_sample, 0.99),
            'max_ring_idx': float(torch.max(ring_sample).item()) if ring_sample.numel() else 0.0,
            'quantile_sample_size': sample_size,
        }


def make_c19_dualphi_fixed_c(c_value, tail_mode='linear', tail_k=6.0, telemetry=None):
    def c19_dualphi_fixed(x, rho=4.0, C=None):
        if telemetry is not None:
            telemetry.observe(x, c_value, tail_k)
        inv_c = 1.0 / c_value
        scaled = x * inv_c
        n = torch.floor(scaled)
        t = scaled - n
        h = t - t * t
        odd = torch.remainder(n, 2.0)
        sgn = 1.0 - 2.0 * odd
        gain = odd * (PHI - PHI_INV) + PHI_INV
        core = c_value * h * (sgn + 4.0 * h) * gain

        if tail_mode == 'periodic':
            return core
        if tail_mode == 'linear':
            limit = tail_k * c_value
            return torch.where(x.abs() > limit, x - x.sign() * limit, core)
        raise ValueError(f'Unknown tail_mode: {tail_mode!r}')

    return c19_dualphi_fixed


def build_model(
    seed,
    replace_impl='dense',
    kernel_mode='vshape',
    topk_k=8,
    read_kernel_mode=None,
    write_address_mode='pointer',
    write_topk_k=None,
    pointer_mode='sequential',
    pointer_interp_mode='off',
    pointer_seam_mode='mod',
    mtaps_enabled=False,
    mtaps_lags=(1, 2, 4, 8, 16, 32),
    mtaps_mixer_mode='current',
    device='cuda',
    hidden_dim=2048,
    M=1024,
    slot_dim=128,
    N=1,
    R=1,
):
    _set_determinism(seed)
    effective_read_kernel_mode = read_kernel_mode or kernel_mode
    effective_write_topk_k = topk_k if write_topk_k is None else write_topk_k
    spec = {
        'M': M,
        'embed_dim': None,
        'hidden_dim': hidden_dim,
        'slot_dim': slot_dim,
        'N': N,
        'R': R,
        'B': 8,
        'embed_mode': True,
        'kernel_mode': kernel_mode,
        'read_kernel_mode': effective_read_kernel_mode,
        'checkpoint_chunks': 0,
        'expert_weighting': False,
        'embed_encoding': 'learned',
        'output_encoding': 'learned',
        'pointer_mode': pointer_mode,
        'pointer_seam_mode': pointer_seam_mode,
        'write_mode': 'replace',
        'replace_impl': replace_impl,
        'pointer_interp_mode': pointer_interp_mode,
        'mtaps_enabled': bool(mtaps_enabled),
        'mtaps_lags': list(mtaps_lags),
        'mtaps_mixer_mode': mtaps_mixer_mode,
        'bb_enabled': False,
        'bb_gate_bias': 0.0,
        'bb_scale': 0.1,
        'bb_tau': 4.0,
        'bb_gate_mode': 'learned',
        'topk_K': topk_k,
        'read_topk_K': topk_k,
        'write_address_mode': write_address_mode,
        'write_topk_K': effective_write_topk_k,
        's_constraint': 'softplus',
        'c19_mode': 'standard',  # nightly monkey-patches activation for telemetry
    }
    record = {'type': 'instnct', 'build_spec': spec}
    return build_model_from_spec(record, device)


def run_one(
    variant_name,
    act_fn,
    dataset,
    steps,
    batch_size,
    seed,
    kernel_mode='vshape',
    topk_k=8,
    replace_impl='dense',
    topk_read_diag=False,
    read_kernel_mode=None,
    write_address_mode='pointer',
    write_topk_k=None,
    pointer_mode='sequential',
    pointer_interp_mode='off',
    pointer_seam_mode='mod',
    mtaps_enabled=False,
    mtaps_lags=(1, 2, 4, 8, 16, 32),
    mtaps_mixer_mode='current',
    context_mode='dotprod',
    ring_trace=False,
    device='cuda',
    hidden_dim=2048,
    M=1024,
    slot_dim=128,
    N=1,
    R=1,
    heartbeat_cb=None,
):
    import instnct

    orig_fn = instnct._c19_activation
    instnct._c19_activation = act_fn
    instnct.set_ring_trace_enabled(ring_trace)

    model = build_model(
        seed,
        replace_impl=replace_impl,
        kernel_mode=kernel_mode,
        topk_k=topk_k,
        read_kernel_mode=read_kernel_mode,
        write_address_mode=write_address_mode,
        write_topk_k=write_topk_k,
        pointer_mode=pointer_mode,
        pointer_interp_mode=pointer_interp_mode,
        pointer_seam_mode=pointer_seam_mode,
        mtaps_enabled=mtaps_enabled,
        mtaps_lags=mtaps_lags,
        mtaps_mixer_mode=mtaps_mixer_mode,
        device=device,
        hidden_dim=hidden_dim,
        M=M,
        slot_dim=slot_dim,
        N=N,
        R=R,
    )

    # We want fixed-C geometry. Freeze the learnable C/rho carriers so the
    # optimizer doesn't waste effort on parameters the activation ignores.
    model._diag_enabled = topk_read_diag
    for name, param in model.named_parameters():
        if any(key in name for key in ('c19_C_', 'c19_rho_')):
            param.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    amp_enabled = device == 'cuda'
    scaler = torch.amp.GradScaler(device, enabled=amp_enabled)

    losses = []
    accs = []
    grad_norms = []
    topk_diag_rows = {key: [] for key in TOPK_READ_DIAG_KEYS}
    ring_trace_rows = None
    ring_trace_summary = None
    if ring_trace:
        ring_trace_rows = {
            'ptr_trace': [],
            'read_idx_trace': [],
            'read_weight_trace': [],
            'write_idx_trace': [],
            'write_weight_trace': [],
            'read_write_overlap_trace': [],
            'center_hist': [0 for _ in range(M)],
            'read_hist': [0 for _ in range(M)],
            'write_hist': [0 for _ in range(M)],
        }
    max_grad = 0.0
    t0 = time.time()
    if heartbeat_cb is not None:
        heartbeat_cb('start', 0, steps, {'variant_name': variant_name})

    for step in range(1, steps + 1):
        xb, yb, mask = dataset.sample_batch(batch_size, device)

        with torch.amp.autocast(device, enabled=amp_enabled):
            pred, _state = model(xb, S=context_mode, state=None)
            _, masked_loss = func_maskloss_ce(pred, yb, mask)

        opt.zero_grad()
        scaler.scale(masked_loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item()
        scaler.step(opt)
        scaler.update()

        lv = masked_loss.item()
        losses.append(lv)
        grad_norms.append(gn)
        max_grad = max(max_grad, gn)

        with torch.no_grad():
            preds = pred.argmax(dim=-1)
            correct = (preds == yb).float() * mask
            acc = correct.sum() / mask.sum().clamp(min=1)
            accs.append(acc.item())
        if topk_read_diag:
            for key in TOPK_READ_DIAG_KEYS:
                value = model._diag.get(key)
                if value is not None:
                    topk_diag_rows[key].append(float(value))
        if ring_trace:
            trace = getattr(model, '_ring_trace', None)
            if trace is not None:
                for key in ('ptr_trace', 'read_idx_trace', 'read_weight_trace', 'write_idx_trace', 'write_weight_trace', 'read_write_overlap_trace'):
                    ring_trace_rows[key].extend(trace.get(key, []))
                for key in ('center_hist', 'read_hist', 'write_hist'):
                    vals = trace.get(key, [])
                    ring_trace_rows[key] = [a + int(b) for a, b in zip(ring_trace_rows[key], vals)]

        if step % 100 == 0 or step == 1:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            avg_acc = sum(accs[-100:]) / len(accs[-100:])
            avg_gn = sum(grad_norms[-100:]) / len(grad_norms[-100:])
            tele = getattr(act_fn, '_telemetry').summary()
            elapsed = time.time() - t0
            spike = '*SPIKE*' if max(grad_norms[-100:]) > 50 else ''
            diag_suffix = ''
            if topk_read_diag:
                dist = model._diag.get('topk_mean_abs_circ_dist')
                outside = model._diag.get('topk_outside_local_frac')
                wdist = model._diag.get('write_topk_mean_abs_circ_dist')
                woutside = model._diag.get('write_topk_outside_local_frac')
                if dist is not None and outside is not None:
                    diag_suffix = f'  topk_dist={dist:.2f}  outside={outside:.3f}'
                if wdist is not None and woutside is not None:
                    diag_suffix += f'  wdist={wdist:.2f}  wout={woutside:.3f}'
            print(
                f'  [{variant_name}] step {step:4d}/{steps}  '
                f'loss={avg_loss:.4f}  bpc={avg_loss*1.4427:.3f}  '
                f'acc={avg_acc:.3f}  gnorm={avg_gn:.1f}  '
                f'tail={tele["tail_hit_pct"]:.3f}%  '
                f'p99|x|/C={tele["p99_abs_over_c"]:.2f}  '
                f'p99-ring={tele["p99_ring_idx"]:.2f}  '
                f'{elapsed:.0f}s {spike}{diag_suffix}'
            )
            if heartbeat_cb is not None:
                heartbeat_cb(
                    'progress',
                    step,
                    steps,
                    {
                        'avg_loss': float(avg_loss),
                        'avg_acc': float(avg_acc),
                        'elapsed_s': float(elapsed),
                    },
                )

    elapsed = time.time() - t0
    instnct._c19_activation = orig_fn
    model._diag_enabled = False
    instnct.set_ring_trace_enabled(False)

    tail = min(100, len(losses))
    spikes = sum(1 for g in grad_norms if g > 50)
    result = {
        'variant': variant_name,
        'seed': seed,
        'steps': steps,
        'params': n_params,
        'final_loss': sum(losses[-tail:]) / tail,
        'final_bpc': sum(losses[-tail:]) / tail * 1.4427,
        'final_acc': sum(accs[-tail:]) / tail,
        'best_loss': min(losses),
        'best_acc': max(accs),
        'time_s': elapsed,
        's_per_step': elapsed / steps,
        'max_grad': max_grad,
        'grad_spikes': spikes,
        'loss_curve': losses,
        'acc_curve': accs,
    }
    result.update(act_fn._telemetry.summary())
    for key in TOPK_READ_DIAG_KEYS:
        rows = topk_diag_rows[key]
        result[key] = (sum(rows) / len(rows)) if rows else None
    if ring_trace_rows is not None:
        ring_trace_summary = _summarize_ring_trace(ring_trace_rows, M)
        result['ring_trace_summary'] = ring_trace_summary
        result['ring_trace'] = ring_trace_rows
    if heartbeat_cb is not None:
        heartbeat_cb(
            'done',
            steps,
            steps,
            {
                'final_acc': float(result['final_acc']),
                'final_bpc': float(result['final_bpc']),
                'time_s': float(result['time_s']),
            },
        )
    return result


def parse_floats(text):
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def parse_modes(text):
    return [x.strip() for x in text.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seq', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--c-values', type=str, default=f'{PHI*PHI},{math.pi},{2*math.pi}')
    parser.add_argument('--tail-modes', type=str, default='linear,periodic')
    parser.add_argument('--kernel-modes', type=str, default='vshape')
    parser.add_argument('--topk-k', type=int, default=8)
    parser.add_argument('--write-topk-k', type=int, default=0)
    parser.add_argument('--topk-read-diag', action='store_true')
    parser.add_argument('--ring-trace', action='store_true')
    parser.add_argument('--replace-impl', type=str, default='dense', choices=['dense', 'proxy_overlay'])
    parser.add_argument('--read-kernel-mode', type=str, default='', choices=['', 'vshape', 'topk'])
    parser.add_argument('--write-address-mode', type=str, default='pointer', choices=['pointer', 'content_topk'])
    parser.add_argument('--tail-k', type=float, default=6.0)
    parser.add_argument('--sample-per-call', type=int, default=1024)
    parser.add_argument('--json-out', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--hidden-dim', type=int, default=2048)
    parser.add_argument('--M', type=int, default=1024)
    parser.add_argument('--slot-dim', type=int, default=128)
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--R', type=int, default=1)
    args = parser.parse_args()

    c_values = parse_floats(args.c_values)
    tail_modes = parse_modes(args.tail_modes)
    kernel_modes = parse_modes(args.kernel_modes)

    _set_determinism(args.seed)

    print('=== C19 Core Geometry Sweep ===')
    print(f'Steps: {args.steps}  Seed: {args.seed}  Batch: {args.batch}x{args.seq}')
    if args.device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('Device: cpu')
    print(f'C values: {[round(c, 4) for c in c_values]}')
    print(f'Tail modes: {tail_modes}  tail_k={args.tail_k}')
    print(f'Kernel modes: {kernel_modes}  topk_K={args.topk_k}  replace={args.replace_impl}')
    if args.read_kernel_mode or args.write_address_mode != 'pointer' or args.write_topk_k:
        print(
            f'Addressing override: read={args.read_kernel_mode or "default"}  '
            f'write={args.write_address_mode}  write_topk_K={args.write_topk_k or args.topk_k}'
        )
    print(
        f'Model: hidden={args.hidden_dim} M={args.M} slot={args.slot_dim} N={args.N} R={args.R}'
    )
    print(f'topk_read_diag: {args.topk_read_diag}')
    print(f'ring_trace: {args.ring_trace}')
    print()

    data_dir = V4_ROOT / 'training_data'
    if not data_dir.exists():
        fallback_dir = Path(r'S:\AI\work\VRAXION_DEV\v4\training_data')
        if fallback_dir.exists():
            data_dir = fallback_dir
    files = func_discover_dat(str(data_dir))
    dataset = ByteDataset(files, args.seq, embed_mode=True, seed=args.seed)
    print(f'Data: {len(files)} shards, {dataset.total_bytes / 1e6:.0f} MB')
    print()

    variants = []
    for kernel_mode in kernel_modes:
        for c_value in c_values:
            for tail_mode in tail_modes:
                telemetry = ActivationTelemetry(sample_per_call=args.sample_per_call)
                act_fn = make_c19_dualphi_fixed_c(
                    c_value=c_value,
                    tail_mode=tail_mode,
                    tail_k=args.tail_k,
                    telemetry=telemetry,
                )
                act_fn._telemetry = telemetry
                variants.append(
                    {
                        'name': f'{kernel_mode}-{_label_c(c_value)}-{tail_mode}',
                        'kernel_mode': kernel_mode,
                        'c_value': c_value,
                        'tail_mode': tail_mode,
                        'act_fn': act_fn,
                    }
                )

    results = []
    for item in variants:
        dataset.rng = np.random.default_rng(args.seed)
        r = run_one(
            item['name'],
            item['act_fn'],
            dataset,
            args.steps,
            args.batch,
            args.seed,
            kernel_mode=item['kernel_mode'],
            topk_k=args.topk_k,
            replace_impl=args.replace_impl,
            topk_read_diag=args.topk_read_diag and item['kernel_mode'] == 'topk',
            read_kernel_mode=args.read_kernel_mode or None,
            write_address_mode=args.write_address_mode,
            write_topk_k=(args.write_topk_k or None),
            ring_trace=args.ring_trace,
            device=args.device,
            hidden_dim=args.hidden_dim,
            M=args.M,
            slot_dim=args.slot_dim,
            N=args.N,
            R=args.R,
        )
        r['kernel_mode'] = item['kernel_mode']
        r['fixed_C'] = item['c_value']
        r['tail_mode'] = item['tail_mode']
        results.append(r)
        print(
            f'  -> {item["name"]}: loss={r["final_loss"]:.4f} '
            f'bpc={r["final_bpc"]:.3f} '
            f'acc={r["final_acc"]:.3f} '
            f'best_acc={r["best_acc"]:.3f} '
            f'max_gnorm={r["max_grad"]:.1f} '
            f'tail={r["tail_hit_pct"]:.4f}% '
            f'p99|x|/C={r["p99_abs_over_c"]:.2f} '
            f'p99-ring={r["p99_ring_idx"]:.2f} '
            f'({r["time_s"]:.0f}s)'
        )
        if args.ring_trace and r.get('ring_trace_summary'):
            s = r['ring_trace_summary']
            print(
                f'     trace: ptr_unique={s["ptr_unique_frac"]:.3f} '
                f'read_unique={s["read_unique_frac"]:.3f} '
                f'write_unique={s["write_unique_frac"]:.3f} '
                f'ptr_jump={s["ptr_jump_mean"]:.2f} '
                f'rdist={s["read_center_dist_mean"]:.2f} '
                f'wdist={s["write_center_dist_mean"]:.2f} '
                f'overlap={s["read_write_overlap_mean"]:.3f}'
            )
        print()

    print('=' * 118)
    print(
        f'{"Variant":16s} {"C":>7s} {"Tail":>8s} {"Final Acc":>10s} {"Best Acc":>10s} '
        f'{"Loss":>10s} {"BPC":>8s} {"Tail%":>8s} {"p99|x|/C":>10s} {"p99-ring":>10s}'
    )
    print('-' * 118)
    for r in results:
        print(
            f'{r["variant"]:16s} {r["fixed_C"]:7.3f} {r["tail_mode"]:>8s} '
            f'{r["final_acc"]:10.3f} {r["best_acc"]:10.3f} {r["final_loss"]:10.4f} '
            f'{r["final_bpc"]:8.3f} {r["tail_hit_pct"]:8.4f} '
            f'{r["p99_abs_over_c"]:10.2f} {r["p99_ring_idx"]:10.2f}'
        )

    best = max(results, key=lambda r: r['final_acc'])
    print(f'\nBest variant: {best["variant"]}')

    linear_by_c = {r['fixed_C']: r for r in results if r['tail_mode'] == 'linear'}
    periodic_by_c = {r['fixed_C']: r for r in results if r['tail_mode'] == 'periodic'}
    if linear_by_c and periodic_by_c:
        print('\nTail necessity check')
        print(f'{"C":>7s} {"Linear Acc":>10s} {"Periodic Acc":>12s} {"Delta":>8s} {"Linear Tail%":>13s}')
        print('-' * 62)
        for c_value in c_values:
            if c_value in linear_by_c and c_value in periodic_by_c:
                rl = linear_by_c[c_value]
                rp = periodic_by_c[c_value]
                delta = (rp['final_acc'] - rl['final_acc']) * 100
                print(
                    f'{c_value:7.3f} {rl["final_acc"]:10.3f} {rp["final_acc"]:12.3f} '
                    f'{delta:8.2f} {rl["tail_hit_pct"]:13.4f}'
                )

    json_out = Path(args.json_out) if args.json_out else _default_json_path()
    payload = {
        'script': Path(__file__).name,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'steps': args.steps,
            'batch': args.batch,
            'seq': args.seq,
            'seed': args.seed,
            'c_values': c_values,
            'tail_modes': tail_modes,
            'kernel_modes': kernel_modes,
            'topk_k': args.topk_k,
            'write_topk_k': args.write_topk_k,
            'topk_read_diag': args.topk_read_diag,
            'ring_trace': args.ring_trace,
            'replace_impl': args.replace_impl,
            'read_kernel_mode': args.read_kernel_mode or None,
            'write_address_mode': args.write_address_mode,
            'tail_k': args.tail_k,
            'sample_per_call': args.sample_per_call,
            'gpu': torch.cuda.get_device_name(0),
            'device': args.device,
            'hidden_dim': args.hidden_dim,
            'M': args.M,
            'slot_dim': args.slot_dim,
            'N': args.N,
            'R': args.R,
        },
        'results': results,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'\nSaved telemetry JSON: {json_out}')
    print('=' * 118)


if __name__ == '__main__':
    main()
