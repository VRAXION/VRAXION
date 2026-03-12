"""A/B Test: dual-phi vs dual-phi + outer-envelope damping on WikiText-103.

This tests whether gently damping farther C19 arches helps learning without
adding more periodic structure before the linear tail.

Baseline:
  dual-phi core, rho=4.0, C=pi, linear tail at 6C

Variants:
  dual-phi-envelope(alpha): same core, multiplied by
      envelope = 1 / (1 + alpha * floor(|x| / C))
  inside the periodic core only. Tail remains unchanged.

Usage:
    python tests/ab_c19_dualphi_envelope_wikitext.py
    python tests/ab_c19_dualphi_envelope_wikitext.py --steps 300 --alphas 0.02,0.05,0.1
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
C19_C = math.pi


def _default_json_path():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = V4_ROOT / 'dev_notes' / 'telemetry'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f'{Path(__file__).stem}_{stamp}.json'


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
        self.c_sum = 0.0
        self.c_min = float('inf')
        self.c_max = 0.0
        self.abs_samples = []
        self.scaled_abs_samples = []

    def _append_sample(self, bucket, sample):
        bucket.append(sample)
        merged = torch.cat(bucket)
        if merged.numel() > self.max_sample_size:
            idx = torch.randint(0, merged.numel(), (self.max_sample_size,), device=merged.device)
            merged = merged[idx]
        bucket[:] = [merged]

    def observe(self, x, C):
        with torch.no_grad():
            abs_x = x.detach().abs()
            c_tns = torch.as_tensor(C, device=abs_x.device, dtype=abs_x.dtype)
            scaled_abs = abs_x / c_tns

            self.call_count += 1
            self.total_values += abs_x.numel()
            self.tail_values += int((scaled_abs > 6.0).sum().item())
            self.abs_sum += float(abs_x.sum().item())
            self.scaled_abs_sum += float(scaled_abs.sum().item())
            self.max_abs_x = max(self.max_abs_x, float(abs_x.max().item()))
            self.max_abs_over_c = max(self.max_abs_over_c, float(scaled_abs.max().item()))

            c_mean = float(c_tns.float().mean().item())
            c_min = float(c_tns.float().min().item())
            c_max = float(c_tns.float().max().item())
            self.c_sum += c_mean
            self.c_min = min(self.c_min, c_min)
            self.c_max = max(self.c_max, c_max)

            flat_abs = abs_x.reshape(-1)
            flat_scaled = scaled_abs.reshape(-1)
            k = min(self.sample_per_call, flat_abs.numel())
            if k > 0:
                if flat_abs.numel() > k:
                    idx = torch.randint(0, flat_abs.numel(), (k,), device=flat_abs.device)
                    abs_sample = flat_abs[idx].float().cpu()
                    scaled_sample = flat_scaled[idx].float().cpu()
                else:
                    abs_sample = flat_abs.float().cpu()
                    scaled_sample = flat_scaled.float().cpu()
                self._append_sample(self.abs_samples, abs_sample)
                self._append_sample(self.scaled_abs_samples, scaled_sample)

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
                'c_mean': 0.0,
                'c_min': 0.0,
                'c_max': 0.0,
                'quantile_sample_size': 0,
            }

        abs_sample = torch.cat(self.abs_samples) if self.abs_samples else torch.empty(0)
        scaled_sample = (
            torch.cat(self.scaled_abs_samples) if self.scaled_abs_samples else torch.empty(0)
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
            'c_mean': self.c_sum / max(self.call_count, 1),
            'c_min': 0.0 if self.c_min == float('inf') else self.c_min,
            'c_max': self.c_max,
            'quantile_sample_size': sample_size,
        }


def _wrap_with_telemetry(act_fn, telemetry):
    def wrapped(x, rho=4.0, C=None):
        actual_c = C19_C if C is None else C
        telemetry.observe(x, actual_c)
        return act_fn(x, rho=rho, C=C)

    return wrapped


def c19_dualphi(x, rho=4.0, C=None):
    if C is None:
        C = C19_C
    l = 6.0 * C
    inv_c = 1.0 / C
    scaled = x * inv_c
    n = torch.floor(scaled)
    t = scaled - n
    h = t - t * t
    odd = torch.remainder(n, 2.0)
    sgn = 1.0 - 2.0 * odd
    gain = odd * (PHI - PHI_INV) + PHI_INV
    core = C * h * (sgn + rho * h) * gain
    return torch.where(x.abs() > l, x - x.sign() * l, core)


def make_c19_dualphi_envelope(alpha):
    def c19_dualphi_envelope(x, rho=4.0, C=None):
        if C is None:
            C = C19_C
        l = 6.0 * C
        inv_c = 1.0 / C
        scaled = x * inv_c
        n = torch.floor(scaled)
        t = scaled - n
        h = t - t * t
        odd = torch.remainder(n, 2.0)
        sgn = 1.0 - 2.0 * odd
        gain = odd * (PHI - PHI_INV) + PHI_INV
        rings = torch.floor(x.abs() * inv_c)
        envelope = 1.0 / (1.0 + alpha * rings)
        core = C * h * (sgn + rho * h) * gain * envelope
        return torch.where(x.abs() > l, x - x.sign() * l, core)

    return c19_dualphi_envelope


def build_model(seed):
    torch.manual_seed(seed)
    spec = {
        'M': 1024,
        'embed_dim': None,
        'hidden_dim': 2048,
        'slot_dim': 128,
        'N': 1,
        'R': 1,
        'B': 8,
        'embed_mode': True,
        'kernel_mode': 'vshape',
        'checkpoint_chunks': 0,
        'expert_weighting': False,
        'embed_encoding': 'learned',
        'output_encoding': 'learned',
        'pointer_mode': 'sequential',
        'write_mode': 'replace',
        'bb_enabled': False,
        'bb_gate_bias': 0.0,
        'bb_scale': 0.1,
        'bb_tau': 4.0,
        'bb_gate_mode': 'learned',
        'topk_K': 8,
        's_constraint': 'softplus',
    }
    record = {'type': 'instnct', 'build_spec': spec}
    return build_model_from_spec(record, 'cuda')


def run_one(variant_name, act_fn, dataset, steps, batch_size, seed, sample_per_call=1024):
    import instnct

    orig_fn = instnct._c19_activation
    telemetry = ActivationTelemetry(sample_per_call=sample_per_call)
    instnct._c19_activation = _wrap_with_telemetry(act_fn, telemetry)

    model = build_model(seed)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    losses = []
    accs = []
    grad_norms = []
    max_grad = 0.0
    t0 = time.time()

    for step in range(1, steps + 1):
        xb, yb, mask = dataset.sample_batch(batch_size, 'cuda')

        with torch.amp.autocast('cuda', enabled=True):
            pred, _state = model(xb, state=None)
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
        if gn > max_grad:
            max_grad = gn

        with torch.no_grad():
            preds = pred.argmax(dim=-1)
            correct = (preds == yb).float() * mask
            acc = correct.sum() / mask.sum().clamp(min=1)
            accs.append(acc.item())

        if step % 100 == 0 or step == 1:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            avg_acc = sum(accs[-100:]) / len(accs[-100:])
            avg_gn = sum(grad_norms[-100:]) / len(grad_norms[-100:])
            tele = telemetry.summary()
            elapsed = time.time() - t0
            spike = '*SPIKE*' if max(grad_norms[-100:]) > 50 else ''
            print(
                f'  [{variant_name}] step {step:4d}/{steps}  '
                f'loss={avg_loss:.4f}  bpc={avg_loss*1.4427:.3f}  '
                f'acc={avg_acc:.3f}  gnorm={avg_gn:.1f}  '
                f'tail={tele["tail_hit_pct"]:.3f}%  '
                f'p99|x|/C={tele["p99_abs_over_c"]:.2f}  '
                f'max|x|/C={tele["max_abs_over_c"]:.2f}  '
                f'{elapsed:.0f}s {spike}'
            )

    elapsed = time.time() - t0
    instnct._c19_activation = orig_fn

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
    result.update(telemetry.summary())
    return result


def parse_alphas(text):
    if not text.strip():
        return []
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seq', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alphas', type=str, default='0.02,0.05,0.10')
    parser.add_argument('--sample-per-call', type=int, default=1024)
    parser.add_argument('--json-out', type=str, default='')
    args = parser.parse_args()

    alphas = parse_alphas(args.alphas)

    print('=== Dual-Phi Envelope Sweep ===')
    print(f'Steps: {args.steps}  Seed: {args.seed}  Batch: {args.batch}x{args.seq}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Alphas: {[0.0] + alphas}')
    print()

    data_dir = V4_ROOT / 'training_data'
    files = func_discover_dat(str(data_dir))
    dataset = ByteDataset(files, args.seq, embed_mode=True, seed=args.seed)
    print(f'Data: {len(files)} shards, {dataset.total_bytes / 1e6:.0f} MB')
    print()

    variants = [('dual-phi', c19_dualphi)]
    for alpha in alphas:
        variants.append((f'dual-env-{alpha:.2f}', make_c19_dualphi_envelope(alpha)))

    results = []
    for name, fn in variants:
        dataset.rng = np.random.default_rng(args.seed)
        r = run_one(
            name,
            fn,
            dataset,
            args.steps,
            args.batch,
            args.seed,
            sample_per_call=args.sample_per_call,
        )
        results.append(r)
        print(
            f'  -> {name}: loss={r["final_loss"]:.4f} '
            f'bpc={r["final_bpc"]:.3f} '
            f'acc={r["final_acc"]:.3f} '
            f'best_acc={r["best_acc"]:.3f} '
            f'max_gnorm={r["max_grad"]:.1f} '
            f'tail={r["tail_hit_pct"]:.4f}% '
            f'p99|x|/C={r["p99_abs_over_c"]:.2f} '
            f'spikes={r["grad_spikes"]} '
            f'({r["time_s"]:.0f}s)'
        )
        print()

    print('=' * 100)
    print(
        f'{"Variant":14s} {"Final Acc":>10s} {"Best Acc":>10s} {"Final Loss":>11s} '
        f'{"BPC":>8s} {"Time":>8s} {"MaxGrad":>8s} {"Tail%":>8s} {"p99|x|/C":>10s}'
    )
    print('-' * 100)
    for r in results:
        print(
            f'{r["variant"]:14s} {r["final_acc"]:10.3f} {r["best_acc"]:10.3f} '
            f'{r["final_loss"]:11.4f} {r["final_bpc"]:8.3f} '
            f'{r["time_s"]:7.0f}s {r["max_grad"]:8.1f} {r["tail_hit_pct"]:8.4f} '
            f'{r["p99_abs_over_c"]:10.2f}'
        )

    best = max(results, key=lambda r: r['final_acc'])
    baseline = next(r for r in results if r['variant'] == 'dual-phi')
    delta = (best['final_acc'] - baseline['final_acc']) * 100
    print(f'\nBest variant: {best["variant"]}  ({delta:+.2f}% vs dual-phi baseline)')

    print('\nActivation telemetry')
    print(
        f'{"Variant":14s} {"Tail%":>8s} {"p95|x|/C":>10s} {"p99|x|/C":>10s} '
        f'{"Max|x|/C":>10s} {"p99|x|":>10s} {"Cmean":>8s}'
    )
    print('-' * 100)
    for r in results:
        print(
            f'{r["variant"]:14s} {r["tail_hit_pct"]:8.4f} {r["p95_abs_over_c"]:10.2f} '
            f'{r["p99_abs_over_c"]:10.2f} {r["max_abs_over_c"]:10.2f} '
            f'{r["p99_abs_x"]:10.2f} {r["c_mean"]:8.3f}'
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
            'alphas': alphas,
            'sample_per_call': args.sample_per_call,
            'gpu': torch.cuda.get_device_name(0),
        },
        'results': results,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'\nSaved telemetry JSON: {json_out}')
    print('=' * 100)


if __name__ == '__main__':
    main()
