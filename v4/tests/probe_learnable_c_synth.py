"""Quick synthetic probe for learnable C behavior.

Goal:
  Watch what learnable C does on small deterministic synthetic tasks while
  keeping rho fixed, so C motion is easier to interpret.

Tasks:
  - count1: arithmetic progression mod 256
  - alternate2: A/B/A/B/... per sequence
  - echo8: 8-byte seed block repeated; first block unsupervised

Usage:
  python tests/probe_learnable_c_synth.py
  python tests/probe_learnable_c_synth.py --steps 150 --tasks count1,echo8
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
import torch.nn.functional as F

V4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V4_ROOT / 'model'))
sys.path.insert(0, str(V4_ROOT / 'training'))

import instnct as instnct_mod
from instnct import _C_from_raw
from model_factory import build_model_from_spec


def _default_json_path():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = V4_ROOT / 'dev_notes' / 'telemetry'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f'{Path(__file__).stem}_{stamp}.json'


def _set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class ActivationTelemetry:
    def __init__(self, sample_per_call=1024, max_sample_size=65536):
        self.sample_per_call = sample_per_call
        self.max_sample_size = max_sample_size
        self.call_count = 0
        self.total_values = 0
        self.tail_values = 0
        self.max_abs_over_c = 0.0
        self.scaled_samples = []

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
            self.max_abs_over_c = max(self.max_abs_over_c, float(scaled_abs.max().item()))

            flat = scaled_abs.reshape(-1)
            k = min(self.sample_per_call, flat.numel())
            if k > 0:
                if flat.numel() > k:
                    idx = torch.randint(0, flat.numel(), (k,), device=flat.device)
                    sample = flat[idx].float().cpu()
                else:
                    sample = flat.float().cpu()
                self._append_sample(self.scaled_samples, sample)

    def summary(self):
        if self.total_values == 0:
            return {
                'tail_hit_pct': 0.0,
                'p95_abs_over_c': 0.0,
                'p99_abs_over_c': 0.0,
                'max_abs_over_c': 0.0,
                'quantile_sample_size': 0,
            }

        sample = torch.cat(self.scaled_samples) if self.scaled_samples else torch.empty(0)

        def q(quant):
            if sample.numel() == 0:
                return 0.0
            return float(torch.quantile(sample, quant).item())

        return {
            'tail_hit_pct': 100.0 * self.tail_values / self.total_values,
            'p95_abs_over_c': q(0.95),
            'p99_abs_over_c': q(0.99),
            'max_abs_over_c': self.max_abs_over_c,
            'quantile_sample_size': int(sample.numel()),
        }


def _c_stats(model):
    with torch.no_grad():
        c_in = _C_from_raw(model.c19_C_input).detach()
        c_h = _C_from_raw(model.c19_C_hidden).detach()
        return {
            'c_in_mean': float(c_in.mean().item()),
            'c_in_std': float(c_in.std().item()),
            'c_in_min': float(c_in.min().item()),
            'c_in_max': float(c_in.max().item()),
            'c_h_mean': float(c_h.mean().item()),
            'c_h_std': float(c_h.std().item()),
            'c_h_min': float(c_h.min().item()),
            'c_h_max': float(c_h.max().item()),
        }


def _build_model(seed, device, embed_encoding='learned'):
    _set_seed(seed)
    spec = {
        'M': 128,
        'embed_dim': None,
        'hidden_dim': 512,
        'slot_dim': 64,
        'N': 1,
        'R': 1,
        'B': 8,
        'embed_mode': True,
        'kernel_mode': 'vshape',
        'checkpoint_chunks': 0,
        'expert_weighting': False,
        'embed_encoding': embed_encoding,
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
    return build_model_from_spec(record, device)


def _make_count1(batch, seq, rng, device):
    start = rng.integers(0, 256, size=(batch, 1), dtype=np.int64)
    offsets = np.arange(seq + 1, dtype=np.int64)[None, :]
    data = (start + offsets) % 256
    x = torch.from_numpy(data[:, :seq].copy()).to(device)
    y = torch.from_numpy(data[:, 1:seq + 1].copy()).to(device)
    mask = torch.ones(batch, seq, device=device)
    return x, y, mask


def _make_alternate2(batch, seq, rng, device):
    a = rng.integers(0, 256, size=(batch, 1), dtype=np.int64)
    b = rng.integers(0, 256, size=(batch, 1), dtype=np.int64)
    pattern = np.where(np.arange(seq + 1)[None, :] % 2 == 0, a, b)
    x = torch.from_numpy(pattern[:, :seq].copy()).to(device)
    y = torch.from_numpy(pattern[:, 1:seq + 1].copy()).to(device)
    mask = torch.ones(batch, seq, device=device)
    return x, y, mask


def _make_echo8(batch, seq, rng, device):
    block = 8
    reps = (seq + 1 + block - 1) // block
    seed_block = rng.integers(0, 256, size=(batch, block), dtype=np.int64)
    data = np.tile(seed_block, (1, reps))[:, :seq + 1]
    x = torch.from_numpy(data[:, :seq].copy()).to(device)
    y = torch.from_numpy(data[:, 1:seq + 1].copy()).to(device)
    mask = torch.ones(batch, seq, device=device)
    mask[:, : block - 1] = 0.0
    return x, y, mask


TASKS = {
    'count1': _make_count1,
    'alternate2': _make_alternate2,
    'echo8': _make_echo8,
}


def _wrap_activation(telemetry):
    orig = instnct_mod._c19_activation

    def wrapped(x, rho=4.0, C=None):
        actual_c = math.pi if C is None else C
        telemetry.observe(x, actual_c)
        return orig(x, rho=rho, C=C)

    return wrapped


def _masked_ce(pred, y, mask):
    loss_flat = F.cross_entropy(pred.reshape(-1, pred.size(-1)), y.reshape(-1), reduction='none')
    mask_flat = mask.reshape(-1)
    masked_loss = (loss_flat * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)

    with torch.no_grad():
        preds = pred.argmax(dim=-1)
        correct = ((preds == y).float() * mask).sum()
        acc = correct / mask.sum().clamp(min=1.0)
    return masked_loss, float(acc.item())


def run_task(
    task_name,
    steps,
    batch,
    seq,
    seed,
    device,
    embed_encoding='learned',
    lr=1e-3,
    log_every=25,
):
    rng = np.random.default_rng(seed)
    model = _build_model(seed, device, embed_encoding=embed_encoding)

    # Freeze rho to isolate C motion.
    model.c19_rho_input.requires_grad_(False)
    model.c19_rho_hidden.requires_grad_(False)

    telemetry = ActivationTelemetry()
    orig_fn = instnct_mod._c19_activation
    instnct_mod._c19_activation = _wrap_activation(telemetry)

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    scaler = torch.amp.GradScaler(device, enabled=(device == 'cuda'))

    init_stats = _c_stats(model)
    history = []
    t0 = time.time()

    try:
        for step in range(1, steps + 1):
            xb, yb, mask = TASKS[task_name](batch, seq, rng, device)

            with torch.amp.autocast(device, enabled=(device == 'cuda')):
                pred, _state = model(xb, state=None)
                loss, acc = _masked_ce(pred, yb, mask)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gin = model.c19_C_input.grad.norm().item() if model.c19_C_input.grad is not None else 0.0
            ghid = model.c19_C_hidden.grad.norm().item() if model.c19_C_hidden.grad is not None else 0.0
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item()
            scaler.step(opt)
            scaler.update()

            c = _c_stats(model)
            tele = telemetry.summary()
            row = {
                'step': step,
                'loss': float(loss.item()),
                'acc': acc,
                'grad_norm': float(grad_norm),
                'grad_c_in': float(gin),
                'grad_c_h': float(ghid),
            }
            row.update(c)
            row.update(tele)
            history.append(row)

            if step == 1 or step % log_every == 0 or step == steps:
                elapsed = time.time() - t0
                print(
                    f'  [{task_name}] step {step:4d}/{steps}  '
                    f'loss={row["loss"]:.4f}  acc={row["acc"]:.3f}  '
                    f'C_in={row["c_in_mean"]:.3f} (Δ{row["c_in_mean"] - init_stats["c_in_mean"]:+.3f})  '
                    f'C_h={row["c_h_mean"]:.3f} (Δ{row["c_h_mean"] - init_stats["c_h_mean"]:+.3f})  '
                    f'gC=({row["grad_c_in"]:.3e},{row["grad_c_h"]:.3e})  '
                    f'p99|x|/C={row["p99_abs_over_c"]:.2f}  tail={row["tail_hit_pct"]:.3f}%  '
                    f'{elapsed:.0f}s'
                )
    finally:
        instnct_mod._c19_activation = orig_fn

    final = history[-1]
    result = {
        'task': task_name,
        'steps': steps,
        'batch': batch,
        'seq': seq,
        'seed': seed,
        'device': device,
        'embed_encoding': embed_encoding,
        'init': init_stats,
        'final': final,
        'history': history,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seq', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tasks', type=str, default='count1,alternate2,echo8')
    parser.add_argument('--embed-encoding', type=str, default='bitlift')
    parser.add_argument('--json-out', type=str, default='')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tasks = [x.strip() for x in args.tasks.split(',') if x.strip()]

    print('=== Learnable C Synthetic Probe ===')
    print(f'Device: {device}')
    print(f'Tasks: {tasks}')
    print(f'Steps: {args.steps}  Batch: {args.batch}  Seq: {args.seq}  Seed: {args.seed}')
    print(f'embed_encoding: {args.embed_encoding}')
    print('rho: frozen  |  C: learnable')
    print()

    results = []
    for task_name in tasks:
        if task_name not in TASKS:
            raise ValueError(f'Unknown task: {task_name!r}')
        print(f'--- Task: {task_name} ---')
        result = run_task(
            task_name,
            args.steps,
            args.batch,
            args.seq,
            args.seed,
            device,
            embed_encoding=args.embed_encoding,
        )
        results.append(result)
        f = result['final']
        i = result['init']
        print(
            f'  -> {task_name}: '
            f'C_in {i["c_in_mean"]:.3f}->{f["c_in_mean"]:.3f} '
            f'({f["c_in_mean"] - i["c_in_mean"]:+.3f}), '
            f'C_h {i["c_h_mean"]:.3f}->{f["c_h_mean"]:.3f} '
            f'({f["c_h_mean"] - i["c_h_mean"]:+.3f}), '
            f'acc={f["acc"]:.3f}, p99|x|/C={f["p99_abs_over_c"]:.2f}'
        )
        print()

    print('=' * 110)
    print(
        f'{"Task":12s} {"C_in Δ":>8s} {"C_h Δ":>8s} {"Final Acc":>10s} '
        f'{"p99|x|/C":>10s} {"Tail%":>8s} {"gC_in":>10s} {"gC_h":>10s}'
    )
    print('-' * 110)
    for r in results:
        i = r['init']
        f = r['final']
        print(
            f'{r["task"]:12s} '
            f'{f["c_in_mean"] - i["c_in_mean"]:8.3f} '
            f'{f["c_h_mean"] - i["c_h_mean"]:8.3f} '
            f'{f["acc"]:10.3f} {f["p99_abs_over_c"]:10.2f} '
            f'{f["tail_hit_pct"]:8.3f} {f["grad_c_in"]:10.3e} {f["grad_c_h"]:10.3e}'
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
            'tasks': tasks,
            'device': device,
            'embed_encoding': args.embed_encoding,
        },
        'results': results,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'\nSaved telemetry JSON: {json_out}')
    print('=' * 110)


if __name__ == '__main__':
    main()
