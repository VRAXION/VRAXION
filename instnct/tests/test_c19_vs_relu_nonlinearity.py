"""
C19 vs ReLU Nonlinearity Battery
=================================
Replicates the GPT-designed test: XOR and 3-bit parity tasks.
Small network (hidden=24), evolutionary search, 5 seeds.

Goal: verify whether C19's periodic wave helps on discrete nonlinear tasks
where ReLU's monotonicity is a structural disadvantage.

Conditions:
  A) ReLU:             act = max(charge - theta, 0)
  B) C19 fixed C=1.0:  act = c19(charge - theta, C=1.0)
  C) C19 learnable C:  act = c19(charge - theta, C), C mutated

Config (matching GPT): hidden=24, 8000 attempts, stale_limit=2500, 5 seeds.
"""
import numpy as np
import random
import time
import sys

_RHO = 4.0

def c19_activation(x, C):
    """Periodic parabolic wave. rho=4.0 fixed, C per-neuron."""
    inv_c = 1.0 / C
    l = 6.0 * C
    scaled = x * inv_c
    n = np.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    sgn = np.where(np.remainder(n, 2.0) < 1.0, 1.0, -1.0)
    core = C * (sgn * h + _RHO * h * h)
    return np.where(x >= l, x - l, np.where(x <= -l, x + l, core))


def make_task(name):
    """Returns (inputs, targets) for a boolean task. inputs/targets are lists of (tuple, int)."""
    if name == 'xor':
        return [
            (np.array([0.0, 0.0]), 0),
            (np.array([0.0, 1.0]), 1),
            (np.array([1.0, 0.0]), 1),
            (np.array([1.0, 1.0]), 0),
        ]
    elif name == 'parity3':
        patterns = []
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    inp = np.array([float(a), float(b), float(c)])
                    target = (a + b + c) % 2
                    patterns.append((inp, target))
        return patterns
    else:
        raise ValueError(f"Unknown task: {name}")


def forward(inp, mask, theta, decay, C_param, input_w, output_w, H, ticks, use_c19):
    """Single forward pass through the tiny SWG."""
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)

    # Inject input
    injected = inp @ input_w  # (H,)

    for t in range(ticks):
        act = state.copy()
        if t < 2:
            act = act + injected
        raw = np.zeros(H, dtype=np.float32)
        if len(rs):
            np.add.at(raw, cs, act[rs] * sp_vals)
        charge += raw
        charge *= ret
        if use_c19:
            act = c19_activation(charge - theta, C_param)
        else:
            act = np.maximum(charge - theta, 0.0)
        charge = np.maximum(charge, 0.0)
        state = act.copy()

    # Output: charge -> scalar via output weights
    out = charge @ output_w  # scalar
    return out


def eval_task(patterns, mask, theta, decay, C_param, input_w, output_w, H, ticks, use_c19):
    """Evaluate all patterns. Returns (n_correct, total)."""
    correct = 0
    for inp, target in patterns:
        out = forward(inp, mask, theta, decay, C_param, input_w, output_w, H, ticks, use_c19)
        pred = 1 if out > 0.5 else 0
        if pred == target:
            correct += 1
    return correct, len(patterns)


def run_evo_search(task_name, H, mode, seed, max_attempts=8000, stale_limit=2500, ticks=6):
    """
    Evolutionary search for a tiny SWG that solves the task.
    mode: 'relu', 'c19_fixed', 'c19_learnable'
    Returns: (solved, edges_used, attempts_used, final_C_stats)
    """
    use_c19 = mode in ('c19_fixed', 'c19_learnable')
    learnable_c = mode == 'c19_learnable'

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    patterns = make_task(task_name)
    n_inputs = len(patterns[0][0])

    # Fixed random projections
    proj_rng = np.random.RandomState(seed + 1000)
    input_w = proj_rng.randn(n_inputs, H).astype(np.float32) * 0.5
    output_w = proj_rng.randn(H).astype(np.float32) * 0.5

    # Init
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, 0.03, dtype=np.float32)
    decay = np.full(H, 0.15, dtype=np.float32)
    C_param = np.full(H, 1.0, dtype=np.float32)

    best_correct, _ = eval_task(patterns, mask, theta, decay, C_param,
                                 input_w, output_w, H, ticks, use_c19)
    stale = 0

    # Mutation types
    if learnable_c:
        mut_types = ['add', 'add', 'flip', 'theta', 'decay', 'cparam']
    else:
        mut_types = ['add', 'add', 'flip', 'theta', 'decay']

    for attempt in range(1, max_attempts + 1):
        mtype = mut_types[rng.randint(0, len(mut_types) - 1)]

        new_mask = mask.copy()
        new_theta = theta.copy()
        new_decay = decay.copy()
        new_C = C_param.copy()

        if mtype == 'add':
            r = rng.randint(0, H - 1); c = rng.randint(0, H - 1)
            if r == c:
                continue
            val = 0.6 if rng.random() < 0.5 else -0.6
            new_mask[r, c] = val
        elif mtype == 'flip':
            alive = list(zip(*np.where(mask != 0)))
            if not alive:
                continue
            r, c = alive[rng.randint(0, len(alive) - 1)]
            new_mask[r, c] = -mask[r, c]
        elif mtype == 'theta':
            idx = rng.randint(0, H - 1)
            new_theta[idx] = max(0.0, min(1.0, theta[idx] + rng.uniform(-0.1, 0.1)))
        elif mtype == 'decay':
            idx = rng.randint(0, H - 1)
            new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.05, 0.05)))
        elif mtype == 'cparam':
            idx = rng.randint(0, H - 1)
            new_C[idx] = max(0.2, min(5.0, C_param[idx] + rng.uniform(-0.5, 0.5)))

        new_correct, total = eval_task(patterns, new_mask, new_theta, new_decay, new_C,
                                        input_w, output_w, H, ticks, use_c19)

        # Accept if improved OR neutral (allows exploration / network growth)
        if new_correct >= best_correct:
            mask = new_mask
            theta = new_theta
            decay = new_decay
            C_param = new_C
            if new_correct > best_correct:
                best_correct = new_correct
                stale = 0
            else:
                stale += 1
        else:
            stale += 1

        if best_correct == total:
            edges = int(np.count_nonzero(mask))
            c_stats = {'mean': round(float(C_param.mean()), 3),
                       'std': round(float(C_param.std()), 3),
                       'min': round(float(C_param.min()), 2),
                       'max': round(float(C_param.max()), 2)} if use_c19 else None
            return True, edges, attempt, c_stats

        if stale >= stale_limit:
            edges = int(np.count_nonzero(mask))
            c_stats = {'mean': round(float(C_param.mean()), 3),
                       'std': round(float(C_param.std()), 3),
                       'min': round(float(C_param.min()), 2),
                       'max': round(float(C_param.max()), 2)} if use_c19 else None
            return False, edges, attempt, c_stats

    edges = int(np.count_nonzero(mask))
    c_stats = {'mean': round(float(C_param.mean()), 3),
               'std': round(float(C_param.std()), 3),
               'min': round(float(C_param.min()), 2),
               'max': round(float(C_param.max()), 2)} if use_c19 else None
    return False, edges, max_attempts, c_stats


if __name__ == "__main__":
    H = 24
    SEEDS = [42, 123, 456, 789, 1337]
    TASKS = ['xor', 'parity3']
    MODES = [
        ('relu', 'ReLU'),
        ('c19_fixed', 'C19 fixed C=1.0'),
        ('c19_learnable', 'C19 learnable C'),
    ]

    print(f"C19 vs ReLU Nonlinearity Battery")
    print(f"hidden={H}, {len(SEEDS)} seeds, 8000 attempts, stale_limit=2500")
    print(f"{'='*70}")

    all_results = {}

    for task_name in TASKS:
        print(f"\n--- Task: {task_name.upper()} ---")
        for mode_key, mode_label in MODES:
            solves = 0
            edge_counts = []
            attempt_counts = []
            c_stats_list = []

            for seed in SEEDS:
                solved, edges, attempts, c_stats = run_evo_search(
                    task_name, H, mode_key, seed,
                    max_attempts=8000, stale_limit=2500, ticks=6
                )
                solves += int(solved)
                edge_counts.append(edges)
                attempt_counts.append(attempts)
                if c_stats:
                    c_stats_list.append(c_stats)

                status = "SOLVED" if solved else "FAILED"
                c_info = f" C={c_stats['mean']:.2f}+/-{c_stats['std']:.2f}" if c_stats else ""
                print(f"  {mode_label:20s} seed={seed:5d} {status} edges={edges:3d} attempts={attempts:5d}{c_info}")

            avg_edges = np.mean(edge_counts) if edge_counts else 0
            solve_rate = f"{solves}/{len(SEEDS)}"

            result_key = f"{task_name}_{mode_key}"
            all_results[result_key] = {
                'task': task_name, 'mode': mode_label,
                'solve_rate': solve_rate, 'solves': solves,
                'avg_edges': round(avg_edges, 1),
                'edge_counts': edge_counts,
                'c_stats': c_stats_list,
            }

            print(f"  >> {mode_label:20s} solve={solve_rate} avg_edges={avg_edges:.1f}")

    # Final summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Task':<10} {'Mode':<22} {'Solve':>8} {'Avg Edges':>10}")
    print(f"  {'-'*10} {'-'*22} {'-'*8} {'-'*10}")
    for key, r in all_results.items():
        print(f"  {r['task']:<10} {r['mode']:<22} {r['solve_rate']:>8} {r['avg_edges']:>10.1f}")

    # C parameter analysis for learnable runs
    print(f"\n  C19 Learnable C Analysis:")
    for key, r in all_results.items():
        if 'learnable' in key and r['c_stats']:
            means = [s['mean'] for s in r['c_stats']]
            stds = [s['std'] for s in r['c_stats']]
            print(f"    {r['task']:10s} C_mean={np.mean(means):.3f} C_std={np.mean(stds):.3f}"
                  f"  range=[{min(s['min'] for s in r['c_stats']):.2f}, {max(s['max'] for s in r['c_stats']):.2f}]")

    print(f"{'='*70}")
