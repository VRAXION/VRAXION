"""Profile one training step of the fixed-C WikiText sweep config.

This measures the short-run proxy used by the recent C19 fixed-C sweeps:
  - WikiText random batch
  - seq=256, batch=32 by default
  - N=1, R=1, write_mode=replace
  - dual-phi fixed-C activation

Outputs:
  - coarse stage timing (sample / forward+loss / backward / optimizer)
  - logical function timing for C19/ring helpers
  - torch.profiler top CUDA ops
  - optional source-map scope breakdown + per-scope op attribution
  - optional Chrome trace JSON
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

V4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V4_ROOT / 'model'))
sys.path.insert(0, str(V4_ROOT / 'training'))

import instnct
from train import ByteDataset, func_discover_dat, func_maskloss_ce
from sweep_c19_core_geometry_wikitext import (
    ActivationTelemetry,
    PHI,
    PHI_INV,
    _set_determinism,
    build_model,
    make_c19_dualphi_fixed_c,
)

SOURCE_SCOPE_NAMES = (
    'state_init',
    'window_prepare',
    'softread',
    'write_prepare',
    'write_replace',
    'pointer_update',
    'output_head',
)


def _default_artifact_paths() -> tuple[Path, Path, Path, Path, Path]:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = V4_ROOT / 'dev_notes' / 'telemetry'
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f'{Path(__file__).stem}_{stamp}'
    return (
        base.with_suffix('.json'),
        base.with_name(base.name + '_ops.txt'),
        base.with_name(base.name + '_trace.json'),
        base.with_name(base.name + '_scopes.json'),
        base.with_name(base.name + '_scope_ops.json'),
    )


class FunctionTimer:
    def __init__(self):
        self.rows: dict[str, dict[str, object]] = {}

    def wrap(self, module, name: str):
        orig = getattr(module, name)
        self.rows[name] = {'count': 0, 'events': [], 'orig': orig}

        def wrapped(*args, **kwargs):
            row = self.rows[name]
            row['count'] = int(row['count']) + 1
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = orig(*args, **kwargs)
                end.record()
                row['events'].append((start, end))
                return out
            t0 = time.perf_counter()
            out = orig(*args, **kwargs)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            row['events'].append(dt_ms)
            return out

        setattr(module, name, wrapped)

    def restore_all(self, module):
        for name, row in self.rows.items():
            setattr(module, name, row['orig'])

    def summary(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out = []
        for name, row in self.rows.items():
            count = int(row['count'])
            events = row['events']
            if torch.cuda.is_available():
                total_ms = float(sum(start.elapsed_time(end) for start, end in events))
            else:
                total_ms = float(sum(events))
            avg_ms = total_ms / max(count, 1)
            out.append(
                {
                    'name': name,
                    'count': count,
                    'total_ms': total_ms,
                    'avg_ms': avg_ms,
                }
            )
        out.sort(key=lambda r: r['total_ms'], reverse=True)
        return out

    def reset(self):
        for row in self.rows.values():
            row['count'] = 0
            row['events'] = []


def _reset_activation_telemetry(telemetry):
    telemetry.call_count = 0
    telemetry.total_values = 0
    telemetry.tail_values = 0
    telemetry.abs_sum = 0.0
    telemetry.scaled_abs_sum = 0.0
    telemetry.max_abs_x = 0.0
    telemetry.max_abs_over_c = 0.0
    telemetry.abs_samples = []
    telemetry.scaled_abs_samples = []
    telemetry.ring_idx_samples = []


def _freeze_c_carriers(model):
    for name, param in model.named_parameters():
        if any(key in name for key in ('c19_C_', 'c19_rho_')):
            param.requires_grad_(False)


def _load_dataset(seq: int, seed: int):
    data_dir = V4_ROOT / 'training_data'
    files = func_discover_dat(str(data_dir))
    dataset = ByteDataset(files, seq, embed_mode=True, seed=seed)
    return files, dataset


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_stage(fn):
    _sync()
    t0 = time.perf_counter()
    out = fn()
    _sync()
    dt = time.perf_counter() - t0
    return out, dt


def _self_device_us_from_keyavg(row) -> float:
    value = getattr(row, 'self_device_time_total', None)
    if value is None:
        value = getattr(row, 'self_cuda_time_total', 0.0)
    return float(value or 0.0)


def _self_device_us_from_event(event) -> float:
    value = getattr(event, 'self_device_time_total', None)
    if value is None:
        value = getattr(event, 'self_cuda_time_total', 0.0)
    return float(value or 0.0)


def _self_cpu_us_from_event(event) -> float:
    return float(getattr(event, 'self_cpu_time_total', 0.0) or 0.0)


def _nearest_scope_name(event) -> str | None:
    parent = getattr(event, 'cpu_parent', None)
    while parent is not None:
        name = getattr(parent, 'name', '')
        if name.startswith('scope::'):
            return name.split('::', 1)[1]
        parent = getattr(parent, 'cpu_parent', None)
    return None


def _build_source_map(prof) -> tuple[list[dict], dict]:
    scope_rows = {
        scope: {
            'scope': scope,
            'calls': 0,
            'self_device_us': 0.0,
            'self_cpu_us': 0.0,
            'pct_scoped_device': 0.0,
        }
        for scope in SOURCE_SCOPE_NAMES
    }
    per_scope_ops = {scope: {} for scope in SOURCE_SCOPE_NAMES}

    keyavg_rows = list(prof.key_averages())
    aten_totals = {}
    total_aten_device_us = 0.0
    for row in keyavg_rows:
        name = getattr(row, 'key', '')
        if not name.startswith('aten::'):
            continue
        device_us = _self_device_us_from_keyavg(row)
        aten_totals[name] = device_us
        total_aten_device_us += device_us

    for event in prof.events():
        name = getattr(event, 'name', '')
        if name.startswith('scope::'):
            scope = name.split('::', 1)[1]
            if scope in scope_rows:
                scope_rows[scope]['calls'] += 1
            continue
        if not name.startswith('aten::'):
            continue
        scope = _nearest_scope_name(event)
        if scope not in scope_rows:
            continue
        device_us = _self_device_us_from_event(event)
        cpu_us = _self_cpu_us_from_event(event)
        scope_rows[scope]['self_device_us'] += device_us
        scope_rows[scope]['self_cpu_us'] += cpu_us
        per_scope_ops[scope][name] = per_scope_ops[scope].get(name, 0.0) + device_us

    scoped_total = sum(row['self_device_us'] for row in scope_rows.values())
    scope_rows_list = []
    for scope in SOURCE_SCOPE_NAMES:
        row = scope_rows[scope]
        row['pct_scoped_device'] = 100.0 * row['self_device_us'] / max(scoped_total, 1e-12)
        scope_rows_list.append(row)
    scope_rows_list.sort(key=lambda row: row['self_device_us'], reverse=True)

    significant_ops = []
    complete = True
    for name, total_us in sorted(aten_totals.items(), key=lambda item: item[1], reverse=True):
        pct_total = 100.0 * total_us / max(total_aten_device_us, 1e-12)
        if pct_total < 1.0:
            continue
        contributions = []
        attributed_us = 0.0
        for scope in SOURCE_SCOPE_NAMES:
            scoped_us = per_scope_ops[scope].get(name, 0.0)
            if scoped_us <= 0.0:
                continue
            contributions.append(
                {
                    'scope': scope,
                    'self_device_us': scoped_us,
                    'pct_of_op': 100.0 * scoped_us / max(total_us, 1e-12),
                }
            )
            attributed_us += scoped_us
        contributions.sort(key=lambda row: row['self_device_us'], reverse=True)
        if not contributions:
            complete = False
        significant_ops.append(
            {
                'name': name,
                'self_device_us': total_us,
                'pct_total_device': pct_total,
                'primary_scope': contributions[0]['scope'] if contributions else '',
                'coverage_pct': 100.0 * attributed_us / max(total_us, 1e-12),
                'contributions': contributions,
            }
        )

    per_scope_top_ops = {}
    for scope in SOURCE_SCOPE_NAMES:
        rows = [
            {
                'name': name,
                'self_device_us': total_us,
                'pct_of_scope': 100.0 * total_us / max(scope_rows[scope]['self_device_us'], 1e-12),
            }
            for name, total_us in per_scope_ops[scope].items()
        ]
        rows.sort(key=lambda row: row['self_device_us'], reverse=True)
        per_scope_top_ops[scope] = rows[:10]

    scope_ops_payload = {
        'source_map_complete': complete,
        'significant_ops': significant_ops,
        'per_scope_top_ops': per_scope_top_ops,
    }
    return scope_rows_list, scope_ops_payload


def make_c19_dualphi_fixed_c_impl(c_value, impl, tail_k, telemetry):
    if impl == 'current':
        return make_c19_dualphi_fixed_c(
            c_value=c_value,
            tail_mode='linear',
            tail_k=tail_k,
            telemetry=telemetry,
        )

    def c19_dualphi_fixed_impl(x, rho=4.0, C=None):
        if telemetry is not None:
            telemetry.observe(x, c_value, tail_k)
        limit = tail_k * c_value
        scaled = x / c_value
        n = torch.floor(scaled)
        t = scaled - n
        h = t - t * t
        if impl == 'gain_v2':
            odd = torch.remainder(n, 2.0)
            sgn = 1.0 - 2.0 * odd
            gain = PHI_INV + odd
        elif impl == 'bitwise_v3':
            odd = (n.to(torch.int64) & 1).to(x.dtype)
            sgn = 1.0 - 2.0 * odd
            gain = PHI_INV + odd
        else:
            raise ValueError(f'Unknown impl: {impl}')
        core = c_value * h * (sgn + 4.0 * h) * gain
        return torch.where(x.abs() > limit, x - x.sign() * limit, core)

    return c19_dualphi_fixed_impl


def make_hdd_write_impl(impl):
    if impl == 'current':
        return instnct.func_hdd_write_tns

    if impl == 'lerp_v2':
        def hdd_write_lerp_v2(ring_tns, write_vec_tns, expanded_idx_tns, weights_tns, write_strength=None):
            w = weights_tns.unsqueeze(-1)
            if write_strength is not None:
                w = w * write_strength.unsqueeze(1)
            current = ring_tns.gather(1, expanded_idx_tns)
            write_val = write_vec_tns.unsqueeze(1).expand(-1, weights_tns.size(1), -1)
            updated = torch.lerp(
                current.float(),
                write_val.float(),
                w.float(),
            ).to(current.dtype)
            ring_new = ring_tns.clone()
            ring_new.scatter_(1, expanded_idx_tns, updated)
            return ring_new

        return hdd_write_lerp_v2

    if impl == 'delta_v3':
        def hdd_write_delta_v3(ring_tns, write_vec_tns, expanded_idx_tns, weights_tns, write_strength=None):
            w = weights_tns.unsqueeze(-1)
            if write_strength is not None:
                w = w * write_strength.unsqueeze(1)
            current = ring_tns.gather(1, expanded_idx_tns)
            write_val = write_vec_tns.unsqueeze(1).expand(-1, weights_tns.size(1), -1)
            updated = current + w * (write_val - current)
            ring_new = ring_tns.clone()
            ring_new.scatter_(1, expanded_idx_tns, updated)
            return ring_new

        return hdd_write_delta_v3

    raise ValueError(f'Unknown write impl: {impl}')


def run_profile(args):
    _set_determinism(args.seed)
    torch.set_float32_matmul_precision('high')

    _, dataset = _load_dataset(args.seq, args.seed)
    telemetry = ActivationTelemetry(sample_per_call=args.sample_per_call)
    act_fn = make_c19_dualphi_fixed_c_impl(
        c_value=args.c_value,
        impl=args.impl,
        tail_k=args.tail_k,
        telemetry=telemetry,
    )
    act_fn._telemetry = telemetry

    orig_c19 = instnct._c19_activation
    orig_hdd = instnct.func_hdd_write_tns
    instnct._c19_activation = act_fn
    instnct.func_hdd_write_tns = make_hdd_write_impl(args.write_impl)
    instnct.set_source_map_enabled(args.source_map)

    fn_timer = FunctionTimer()
    for fn_name in (
        '_c19_activation',
        'func_softread_tns',
        'func_hdd_write_tns',
        'func_softwrit_tns',
        'func_gated_write_tns',
        'func_movepntr_tns',
    ):
        fn_timer.wrap(instnct, fn_name)

    try:
        model = build_model(args.seed)
        _freeze_c_carriers(model)
        opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        scaler = torch.amp.GradScaler('cuda', enabled=True)

        for _ in range(args.warmup_steps):
            xb, yb, mask = dataset.sample_batch(args.batch, 'cuda')
            with torch.amp.autocast('cuda', enabled=True):
                pred, _ = model(xb, state=None)
                _, loss = func_maskloss_ce(pred, yb, mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()

        fn_timer.reset()
        _reset_activation_telemetry(telemetry)

        stage = {}
        dataset.rng = np.random.default_rng(args.seed)

        (batch, stage['sample_s']) = _measure_stage(
            lambda: dataset.sample_batch(args.batch, 'cuda')
        )
        xb, yb, mask = batch

        def forward_loss():
            with torch.amp.autocast('cuda', enabled=True):
                pred, _ = model(xb, state=None)
                _, loss = func_maskloss_ce(pred, yb, mask)
            return pred, loss

        (fw_out, stage['forward_loss_s']) = _measure_stage(forward_loss)
        pred, loss = fw_out

        opt.zero_grad(set_to_none=True)
        (_, stage['backward_s']) = _measure_stage(lambda: scaler.scale(loss).backward())
        (_, stage['unscale_clip_s']) = _measure_stage(
            lambda: (scaler.unscale_(opt), torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        )
        (_, stage['optim_s']) = _measure_stage(lambda: (scaler.step(opt), scaler.update()))
        stage['total_s'] = sum(stage.values())

        coarse_rows = []
        for key in ('sample_s', 'forward_loss_s', 'backward_s', 'unscale_clip_s', 'optim_s'):
            coarse_rows.append(
                {
                    'stage': key.replace('_s', ''),
                    'seconds': stage[key],
                    'pct_total': 100.0 * stage[key] / max(stage['total_s'], 1e-12),
                }
            )

        fn_rows = fn_timer.summary()
        tele = telemetry.summary()
        fn_timer.reset()
        _reset_activation_telemetry(telemetry)

        dataset.rng = np.random.default_rng(args.seed)
        xb, yb, mask = dataset.sample_batch(args.batch, 'cuda')
        opt.zero_grad(set_to_none=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with record_function('sample_batch'):
                pass
            with record_function('forward_loss'):
                with torch.amp.autocast('cuda', enabled=True):
                    pred, _ = model(xb, state=None)
                    _, loss = func_maskloss_ce(pred, yb, mask)
            with record_function('backward'):
                scaler.scale(loss).backward()
            with record_function('unscale_clip'):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            with record_function('optim_step'):
                scaler.step(opt)
                scaler.update()

        ops_table = prof.key_averages().table(
            sort_by='self_cuda_time_total',
            row_limit=args.row_limit,
        )

        payload = {
            'script': Path(__file__).name,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'seed': args.seed,
                'batch': args.batch,
                'seq': args.seq,
                'c_value': args.c_value,
                'impl': args.impl,
                'write_impl': args.write_impl,
                'tail_k': args.tail_k,
                'warmup_steps': args.warmup_steps,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                'source_map': bool(args.source_map),
            },
            'coarse_stage_breakdown': coarse_rows,
            'function_breakdown': fn_rows,
            'activation_telemetry': tele,
        }

        scope_rows = []
        scope_ops_payload = {}
        if args.source_map:
            scope_rows, scope_ops_payload = _build_source_map(prof)
            payload['source_scope_breakdown'] = scope_rows
            payload['source_map_complete'] = scope_ops_payload['source_map_complete']

        return payload, ops_table, prof, scope_rows, scope_ops_payload
    finally:
        fn_timer.restore_all(instnct)
        instnct._c19_activation = orig_c19
        instnct.func_hdd_write_tns = orig_hdd
        instnct.set_source_map_enabled(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seq', type=int, default=256)
    parser.add_argument('--c-value', type=float, default=math.pi)
    parser.add_argument('--impl', type=str, default='current', choices=['current', 'gain_v2', 'bitwise_v3'])
    parser.add_argument('--write-impl', type=str, default='current', choices=['current', 'lerp_v2', 'delta_v3'])
    parser.add_argument('--tail-k', type=float, default=6.0)
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--sample-per-call', type=int, default=1024)
    parser.add_argument('--row-limit', type=int, default=25)
    parser.add_argument('--source-map', action='store_true')
    parser.add_argument('--json-out', type=str, default='')
    parser.add_argument('--ops-out', type=str, default='')
    parser.add_argument('--trace-out', type=str, default='')
    parser.add_argument('--scope-json-out', type=str, default='')
    parser.add_argument('--scope-ops-json-out', type=str, default='')
    args = parser.parse_args()

    json_path, ops_path, trace_path, scope_json_path, scope_ops_json_path = _default_artifact_paths()
    if args.json_out:
        json_path = Path(args.json_out)
    if args.ops_out:
        ops_path = Path(args.ops_out)
    if args.trace_out:
        trace_path = Path(args.trace_out)
    if args.scope_json_out:
        scope_json_path = Path(args.scope_json_out)
    if args.scope_ops_json_out:
        scope_ops_json_path = Path(args.scope_ops_json_out)

    payload, ops_table, prof, scope_rows, scope_ops_payload = run_profile(args)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    with open(ops_path, 'w', encoding='utf-8') as f:
        f.write(ops_table)
        f.write('\n')

    if args.source_map:
        with open(scope_json_path, 'w', encoding='utf-8') as f:
            json.dump(scope_rows, f, indent=2)
        with open(scope_ops_json_path, 'w', encoding='utf-8') as f:
            json.dump(scope_ops_payload, f, indent=2)

    prof.export_chrome_trace(str(trace_path))

    print('=== Sweep Step Profile ===')
    print(f'GPU: {payload["config"]["gpu"]}')
    print(
        f'Batch: {args.batch}x{args.seq}  C={args.c_value:.4f}  impl={args.impl}  '
        f'write={args.write_impl}  warmup={args.warmup_steps}  source_map={args.source_map}'
    )
    print()
    print('Coarse stage breakdown')
    for row in payload['coarse_stage_breakdown']:
        print(
            f'  {row["stage"]:14s} {row["seconds"]:7.3f}s  {row["pct_total"]:6.1f}%'
        )
    print()
    print('Logical function breakdown')
    for row in payload['function_breakdown']:
        print(
            f'  {row["name"]:18s} calls={row["count"]:4d}  '
            f'total={row["total_ms"]:8.1f}ms  avg={row["avg_ms"]:6.3f}ms'
        )
    print()
    if args.source_map:
        print('Source scope breakdown')
        for row in payload['source_scope_breakdown']:
            print(
                f'  {row["scope"]:16s} calls={row["calls"]:4d}  '
                f'total={row["self_device_us"] / 1000.0:8.1f}ms  pct={row["pct_scoped_device"]:5.1f}%'
            )
        print(f'  source_map_complete={payload["source_map_complete"]}')
        print()
    print('Activation telemetry')
    print(
        f'  tail={payload["activation_telemetry"]["tail_hit_pct"]:.4f}%  '
        f'p99|x|/C={payload["activation_telemetry"]["p99_abs_over_c"]:.2f}  '
        f'max|x|/C={payload["activation_telemetry"]["max_abs_over_c"]:.2f}'
    )
    print()
    print(f'Saved JSON: {json_path}')
    print(f'Saved ops table: {ops_path}')
    if args.source_map:
        print(f'Saved scope JSON: {scope_json_path}')
        print(f'Saved scope-op JSON: {scope_ops_json_path}')
    print(f'Saved trace: {trace_path}')


if __name__ == '__main__':
    main()
