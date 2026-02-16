"""
Checkpoint Autopsy — Deep eval at key points in the giant ant's lifeline.

Loads checkpoints at anomaly points and runs:
  1. Per-task accuracy (add, and, or, xor) — 2048 samples each
  2. Per-bit accuracy breakdown per task
  3. Think-tick sensitivity (0, 1, 2 ticks at inference)
  4. LCX state analysis (64-cell dump, norm, entropy, active/dead cells)
  5. Weight norm forensics (key layers)
  6. Robustness: out-of-distribution inputs (large values, edge cases)

Usage:
  python tools/checkpoint_autopsy.py
"""
import sys, math, json, time
from pathlib import Path

sys.path.insert(0, r"S:\AI\work\VRAXION_DEV\Diamond Code")

import torch
import torch.nn.functional as F
from swarm_model import SwarmByteRingModel

# ─── Config ──────────────────────────────────────────────────
CHECKPOINT_DIR = Path(r"S:\AI\work\VRAXION_DEV\Diamond Code\checkpoints\swarm")
NUM_BITS = 8
SEQ_LEN = 16
EVAL_SAMPLES = 2048
DEVICE = 'cpu'

# Key checkpoints from telemetry analysis
CHECKPOINTS = {
    500:  "Emergence — 84.7% bit_acc, bit3 bottleneck",
    1000: "Consolidation — just crossed 95%",
    2000: "Stable plateau — 98% pre-crisis",
    3000: "Peak — all bits perfect, pre-catastrophe",
    3500: "AT catastrophe (step 3559 imminent)",
    4000: "Mid-recovery — LCX norm spike to 6.1",
    4500: "Late recovery",
    5000: "Post-recovery — bit7 scarring",
}


# ─── Helpers ─────────────────────────────────────────────────
def int_to_bits(val, num_bits=8):
    return torch.tensor([float((val >> i) & 1) for i in range(num_bits)])

def bits_to_int(bits):
    val = 0
    for i, b in enumerate(bits):
        if b > 0.0:
            val |= (1 << i)
    return val

def generate_task_batch(task, n_samples, num_bits=8, seq_len=16, seed=None, max_value=None):
    """Generate eval batch for a single task."""
    if seed is not None:
        import random
        random.seed(seed)
        torch.manual_seed(seed)
    else:
        import random

    max_val = (2 ** num_bits - 1) if max_value is None else min(max_value, 2 ** num_bits - 1)

    op_codes = {
        'add': int_to_bits(1 << 0, num_bits),
        'and': int_to_bits(1 << 1, num_bits),
        'or':  int_to_bits(1 << 2, num_bits),
        'xor': int_to_bits(1 << 3, num_bits),
    }

    x_batch, y_batch = [], []
    for _ in range(n_samples):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)

        if task == 'add':
            result = (a + b) % (2 ** num_bits)
        elif task == 'and':
            result = a & b
        elif task == 'or':
            result = a | b
        elif task == 'xor':
            result = a ^ b
        else:
            raise ValueError(f"Unknown task: {task}")

        x_seq = torch.zeros(seq_len, num_bits)
        x_seq[0, :] = int_to_bits(a, num_bits)
        x_seq[1, :] = int_to_bits(b, num_bits)
        x_seq[2, :] = op_codes[task]

        y_seq = torch.zeros(seq_len, num_bits)
        y_seq[0, :] = int_to_bits(a, num_bits)
        y_seq[1, :] = int_to_bits(b, num_bits)
        y_seq[2, :] = int_to_bits(result, num_bits)

        x_batch.append(x_seq)
        y_batch.append(y_seq)

    return torch.stack(x_batch), torch.stack(y_batch)


def generate_edge_cases(num_bits=8, seq_len=16):
    """Generate OOD / edge-case inputs for robustness testing."""
    max_val = 2 ** num_bits - 1
    cases = []

    # Edge values: 0, 1, max, max-1, powers of 2
    edge_vals = [0, 1, max_val, max_val - 1, 128, 127, 64, 255]
    for task in ['add', 'and', 'or', 'xor']:
        op_codes = {
            'add': int_to_bits(1 << 0, num_bits),
            'and': int_to_bits(1 << 1, num_bits),
            'or':  int_to_bits(1 << 2, num_bits),
            'xor': int_to_bits(1 << 3, num_bits),
        }
        for a in edge_vals:
            for b in edge_vals:
                a_c = min(a, max_val)
                b_c = min(b, max_val)
                if task == 'add':
                    result = (a_c + b_c) % (2 ** num_bits)
                elif task == 'and':
                    result = a_c & b_c
                elif task == 'or':
                    result = a_c | b_c
                else:
                    result = a_c ^ b_c

                x_seq = torch.zeros(seq_len, num_bits)
                x_seq[0, :] = int_to_bits(a_c, num_bits)
                x_seq[1, :] = int_to_bits(b_c, num_bits)
                x_seq[2, :] = op_codes[task]

                y_seq = torch.zeros(seq_len, num_bits)
                y_seq[0, :] = int_to_bits(a_c, num_bits)
                y_seq[1, :] = int_to_bits(b_c, num_bits)
                y_seq[2, :] = int_to_bits(result, num_bits)

                cases.append((task, a_c, b_c, result, x_seq, y_seq))

    x_all = torch.stack([c[4] for c in cases])
    y_all = torch.stack([c[5] for c in cases])
    return cases, x_all, y_all


def load_model(checkpoint_path, think_ticks=0):
    """Load model from checkpoint, optionally override think_ticks."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']

    model = SwarmByteRingModel(
        num_memory_positions=cfg['memory_size'],
        embedding_dim=cfg['embedding_dim'],
        num_beings=cfg['num_beings'],
        depth=cfg['depth'],
        combiner_mode=cfg.get('combiner', 'mean'),
        num_bits=cfg['num_bits'],
        bits_per_being=cfg.get('bits_per_being', 0),
        min_coverage=cfg.get('min_coverage', 2),
        mask_seed=cfg.get('mask_seed', 42),
        fibonacci=cfg.get('fibonacci', False),
        combinatorial=cfg.get('combinatorial', False),
        think_ticks=think_ticks,
        temporal_fibonacci=cfg.get('temporal_fibonacci', False),
        capacity_fibonacci=cfg.get('capacity_fibonacci', False),
        max_hidden=cfg.get('max_hidden', 4096),
        min_hidden=cfg.get('min_hidden', 128),
        full_view=cfg.get('full_view', False),
        use_lcx=cfg.get('use_lcx', True),
        slots_per_being=-1,  # giant ant = global write
    )

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(DEVICE)
    model.eval()
    return model, ckpt


def eval_task(model, task, n_samples=EVAL_SAMPLES, seed=42):
    """Run eval on a single task. Returns dict of metrics."""
    x, y = generate_task_batch(task, n_samples, seed=seed)
    x, y = x.to(DEVICE), y.to(DEVICE)

    with torch.no_grad():
        out = model(x)

    # Position 2 = result position
    pred = (out[:, 2, :] > 0.0).float()
    target = y[:, 2, :]

    # Per-bit accuracy
    bit_accs = []
    for b in range(NUM_BITS):
        acc = (pred[:, b] == target[:, b]).float().mean().item()
        bit_accs.append(acc)

    # Byte accuracy
    byte_match = (pred == target).all(dim=-1).float().mean().item()
    bit_acc = sum(bit_accs) / NUM_BITS
    hamming = NUM_BITS - sum(bit_accs)

    return {
        'bit_acc': bit_acc,
        'byte_match': byte_match,
        'hamming': hamming,
        'per_bit': bit_accs,
        'weakest_bit': min(range(NUM_BITS), key=lambda i: bit_accs[i]),
        'weakest_val': min(bit_accs),
    }


def analyze_lcx(model):
    """Deep analysis of LCX state."""
    if model.lcx is None:
        return {'status': 'no_lcx'}

    lcx = model.lcx.detach().cpu()
    vals = lcx.numpy()

    # Basic stats
    norm = lcx.norm().item()
    mean = lcx.mean().item()
    std = lcx.std().item()
    min_val = lcx.min().item()
    max_val = lcx.max().item()

    # Entropy (treat abs values as distribution)
    abs_vals = lcx.abs()
    if abs_vals.sum() > 0:
        probs = abs_vals / abs_vals.sum()
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
    else:
        entropy = 0.0

    # Active vs dead cells
    active = (lcx.abs() > 0.1).sum().item()
    dead = (lcx.abs() < 0.01).sum().item()
    hot = (lcx.abs() > 0.8).sum().item()

    # Sign distribution
    positive = (lcx > 0).sum().item()
    negative = (lcx < 0).sum().item()

    # Top 5 most active cells
    top_idx = lcx.abs().argsort(descending=True)[:5]
    top_cells = [(int(i), float(lcx[i])) for i in top_idx]

    # Bottom 5 (weakest)
    bot_idx = lcx.abs().argsort()[:5]
    bot_cells = [(int(i), float(lcx[i])) for i in bot_idx]

    # Quadrant analysis (8x8 grid)
    grid = lcx.view(8, 8)
    row_norms = grid.norm(dim=1).tolist()
    col_norms = grid.norm(dim=0).tolist()

    return {
        'norm': norm,
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'entropy': entropy,
        'active_cells': int(active),
        'dead_cells': int(dead),
        'hot_cells': int(hot),
        'positive': int(positive),
        'negative': int(negative),
        'top_5': top_cells,
        'bottom_5': bot_cells,
        'row_norms': [round(x, 4) for x in row_norms],
        'col_norms': [round(x, 4) for x in col_norms],
        'full_state': [round(float(v), 6) for v in lcx],
    }


def analyze_weights(model):
    """Weight norm forensics for key layers."""
    results = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            results[name] = {
                'shape': list(param.shape),
                'norm': round(param.data.norm().item(), 6),
                'mean': round(param.data.mean().item(), 6),
                'std': round(param.data.std().item(), 6),
                'max_abs': round(param.data.abs().max().item(), 6),
            }
    return results


# ─── Main ────────────────────────────────────────────────────
def main():
    tasks = ['add', 'and', 'or', 'xor']
    think_tick_values = [0, 1, 2]

    all_results = {}

    for step, description in CHECKPOINTS.items():
        ckpt_path = CHECKPOINT_DIR / f"checkpoint_step_{step}.pt"
        if not ckpt_path.exists():
            print(f"\n  [SKIP] Step {step}: checkpoint not found")
            continue

        print(f"\n{'='*80}")
        print(f"  CHECKPOINT: Step {step}")
        print(f"  {description}")
        print(f"{'='*80}")

        step_results = {'description': description}

        # ─── 1. Per-task accuracy (think_ticks=0 baseline) ───
        print(f"\n  --- Per-Task Accuracy (think_ticks=0, {EVAL_SAMPLES} samples) ---")
        model, ckpt = load_model(ckpt_path, think_ticks=0)

        task_results = {}
        for task in tasks:
            r = eval_task(model, task)
            task_results[task] = r
            bit_str = ' '.join([f'b{i}={r["per_bit"][i]:.3f}' for i in range(NUM_BITS)])
            print(f"    {task:4s}: byte={r['byte_match']:.4f}  bit={r['bit_acc']:.4f}  weak=bit{r['weakest_bit']}({r['weakest_val']:.3f})  | {bit_str}")

        # Mixed
        mixed_byte = sum(r['byte_match'] for r in task_results.values()) / len(tasks)
        mixed_bit = sum(r['bit_acc'] for r in task_results.values()) / len(tasks)
        print(f"    {'MIX':4s}: byte={mixed_byte:.4f}  bit={mixed_bit:.4f}")
        step_results['tasks_tt0'] = task_results

        # ─── 2. Think-tick sensitivity ───
        print(f"\n  --- Think-Tick Sensitivity ---")
        tt_results = {}
        for tt in think_tick_values:
            model_tt, _ = load_model(ckpt_path, think_ticks=tt)
            # Eval on mixed (all tasks)
            tt_task_res = {}
            for task in tasks:
                r = eval_task(model_tt, task)
                tt_task_res[task] = r
            mixed_b = sum(r['byte_match'] for r in tt_task_res.values()) / len(tasks)
            mixed_a = sum(r['bit_acc'] for r in tt_task_res.values()) / len(tasks)
            tt_results[tt] = {'mixed_byte': mixed_b, 'mixed_bit': mixed_a, 'per_task': tt_task_res}

            task_summary = '  '.join([f"{t}={tt_task_res[t]['byte_match']:.3f}" for t in tasks])
            print(f"    tt={tt}: mixed_byte={mixed_b:.4f}  mixed_bit={mixed_a:.4f}  | {task_summary}")
            del model_tt

        step_results['think_tick_sensitivity'] = tt_results

        # ─── 3. LCX State Analysis ───
        print(f"\n  --- LCX Analysis ---")
        lcx = analyze_lcx(model)
        if lcx.get('status') != 'no_lcx':
            print(f"    norm={lcx['norm']:.4f}  mean={lcx['mean']:.4f}  std={lcx['std']:.4f}")
            print(f"    range=[{lcx['min']:.4f}, {lcx['max']:.4f}]  entropy={lcx['entropy']:.4f}")
            print(f"    active={lcx['active_cells']}/64  dead={lcx['dead_cells']}  hot={lcx['hot_cells']}  +{lcx['positive']}/-{lcx['negative']}")
            print(f"    top5: {lcx['top_5']}")
            print(f"    bot5: {lcx['bottom_5']}")
            print(f"    row_norms: {lcx['row_norms']}")
        step_results['lcx'] = lcx

        # ─── 4. Edge-case robustness ───
        print(f"\n  --- Edge-Case Robustness ---")
        cases, x_edge, y_edge = generate_edge_cases()
        with torch.no_grad():
            out_edge = model(x_edge.to(DEVICE))

        pred_edge = (out_edge[:, 2, :] > 0.0).float()
        target_edge = y_edge[:, 2, :]

        # Group by task
        cases_per_task = {}
        for i, (task, a, b, result, _, _) in enumerate(cases):
            if task not in cases_per_task:
                cases_per_task[task] = {'correct': 0, 'total': 0, 'failures': []}
            correct = (pred_edge[i] == target_edge[i]).all().item()
            cases_per_task[task]['total'] += 1
            if correct:
                cases_per_task[task]['correct'] += 1
            else:
                pred_val = bits_to_int(pred_edge[i])
                cases_per_task[task]['failures'].append(
                    f"{a} {task} {b} = {result} (got {pred_val})"
                )

        for task in tasks:
            info = cases_per_task.get(task, {'correct': 0, 'total': 0, 'failures': []})
            acc = info['correct'] / max(info['total'], 1)
            n_fail = info['total'] - info['correct']
            print(f"    {task:4s}: {info['correct']}/{info['total']} ({acc:.1%})", end='')
            if n_fail > 0 and n_fail <= 5:
                print(f"  FAILS: {info['failures']}")
            elif n_fail > 5:
                print(f"  {n_fail} failures (showing 3): {info['failures'][:3]}")
            else:
                print()

        step_results['edge_cases'] = {
            t: {'accuracy': cases_per_task[t]['correct'] / max(cases_per_task[t]['total'], 1),
                'n_failures': cases_per_task[t]['total'] - cases_per_task[t]['correct'],
                'sample_failures': cases_per_task[t]['failures'][:5]}
            for t in tasks if t in cases_per_task
        }

        # ─── 5. Weight norms ───
        weights = analyze_weights(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        top_norms = sorted(weights.items(), key=lambda x: x[1]['norm'], reverse=True)[:5]
        print(f"\n  --- Weight Norms ({total_params:,} params) ---")
        for name, info in top_norms:
            print(f"    {name}: norm={info['norm']:.4f}  shape={info['shape']}")
        step_results['weight_norms'] = weights
        step_results['total_params'] = total_params

        all_results[step] = step_results
        del model

    # ─── Summary Table ───
    print(f"\n\n{'='*80}")
    print(f"  CROSS-CHECKPOINT SUMMARY")
    print(f"{'='*80}")

    print(f"\n  {'Step':>5} | {'Phase':<35} | {'Byte%':>6} | {'Bit%':>6} | {'LCX norm':>8} | {'Active':>6} | {'Dead':>4} | {'Weak bit':>8}")
    print(f"  {'-'*5}-+-{'-'*35}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}-+-{'-'*8}")

    for step in sorted(all_results.keys()):
        r = all_results[step]
        tasks_r = r.get('tasks_tt0', {})
        if tasks_r:
            mixed_byte = sum(t['byte_match'] for t in tasks_r.values()) / len(tasks_r)
            mixed_bit = sum(t['bit_acc'] for t in tasks_r.values()) / len(tasks_r)
        else:
            mixed_byte = mixed_bit = 0.0

        lcx = r.get('lcx', {})
        lcx_norm = lcx.get('norm', 0)
        active = lcx.get('active_cells', 0)
        dead = lcx.get('dead_cells', 0)

        # Find overall weakest bit
        all_bits = [0.0] * NUM_BITS
        n_tasks = 0
        for t in tasks_r.values():
            for i in range(NUM_BITS):
                all_bits[i] += t['per_bit'][i]
            n_tasks += 1
        if n_tasks > 0:
            all_bits = [x / n_tasks for x in all_bits]
        weakest = min(range(NUM_BITS), key=lambda i: all_bits[i])
        weakest_val = all_bits[weakest]

        desc = r['description'][:35]
        print(f"  {step:>5} | {desc:<35} | {mixed_byte:>5.1%} | {mixed_bit:>5.1%} | {lcx_norm:>8.4f} | {active:>6} | {dead:>4} | bit{weakest}={weakest_val:.3f}")

    # ─── Think-tick impact table ───
    print(f"\n  THINK-TICK SENSITIVITY (mixed byte accuracy)")
    print(f"  {'Step':>5} | {'tt=0':>8} | {'tt=1':>8} | {'tt=2':>8} | {'d(0->1)':>9} | {'d(0->2)':>9}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}")

    for step in sorted(all_results.keys()):
        r = all_results[step]
        tt = r.get('think_tick_sensitivity', {})
        v0 = tt.get(0, {}).get('mixed_byte', 0)
        v1 = tt.get(1, {}).get('mixed_byte', 0)
        v2 = tt.get(2, {}).get('mixed_byte', 0)
        d1 = v1 - v0
        d2 = v2 - v0
        print(f"  {step:>5} | {v0:>7.1%} | {v1:>7.1%} | {v2:>7.1%} | {d1:>+8.1%} | {d2:>+8.1%}")

    # ─── LCX evolution ───
    print(f"\n  LCX EVOLUTION")
    print(f"  {'Step':>5} | {'Norm':>8} | {'Entropy':>8} | {'Active':>6} | {'Dead':>4} | {'Hot':>3} | {'+/-':>5} | {'Top cell':>20}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}-+-{'-'*3}-+-{'-'*5}-+-{'-'*20}")

    for step in sorted(all_results.keys()):
        lcx = all_results[step].get('lcx', {})
        if lcx.get('status') == 'no_lcx':
            continue
        top = lcx.get('top_5', [(0, 0)])[0]
        pn = f"+{lcx.get('positive',0)}/-{lcx.get('negative',0)}"
        top_str = f"cell{top[0]}={top[1]:.4f}"
        print(f"  {step:>5} | {lcx.get('norm',0):>8.4f} | {lcx.get('entropy',0):>8.4f} | {lcx.get('active_cells',0):>6} | {lcx.get('dead_cells',0):>4} | {lcx.get('hot_cells',0):>3} | {pn:>5} | {top_str:>20}")

    # ─── Save full results ───
    out_path = Path(r"S:\AI\work\VRAXION_DEV\Diamond Code\logs\swarm\autopsy_results.json")
    # Convert for JSON
    def jsonify(obj):
        if isinstance(obj, dict):
            return {str(k): jsonify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [jsonify(v) for v in obj]
        elif isinstance(obj, tuple):
            return [jsonify(v) for v in obj]
        elif isinstance(obj, float):
            return round(obj, 6)
        return obj

    with open(out_path, 'w') as f:
        json.dump(jsonify(all_results), f, indent=2)
    print(f"\n  Full results saved: {out_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\n  Total time: {time.time() - t0:.1f}s")
