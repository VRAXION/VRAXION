"""
Toy exhaustive topology test
=============================
H=4, V=4 — all 4096 possible topologies enumerated.
No projections: one-hot input directly into 4 neurons, argmax charge readout.
Topology IS the computation.

Two homogeneous tasks:
  Task A (increment): 0→1, 1→2, 2→3, 3→0
  Task B (reverse):   0→3, 1→2, 2→1, 3→0

For each topology, sweep theta to find best config.
Then analyze: do good topologies cluster? Do different tasks prefer different structures?
"""
import sys
import numpy as np
from collections import Counter
import time

H = 8
V = 4
TICKS = 4
DECAY = 0.16
MAX_CHARGE = 15.0
N_EDGES = H * H - H  # 56 (no self-connections)
# 2^56 is way too many — sample randomly
N_SAMPLES = 50000
SEED = 42

THETA_SWEEP = [1, 2, 3, 5, 8, 12]

TASKS = {
    'increment': {0: 1, 1: 2, 2: 3, 3: 0},
    'reverse':   {0: 3, 1: 2, 2: 1, 3: 0},
}

# ── Projections (fixed, random) ──────────────────────────────────────────────
proj_rng = np.random.RandomState(12345)
INPUT_PROJ = proj_rng.randn(V, H).astype(np.float32)
INPUT_PROJ /= np.linalg.norm(INPUT_PROJ, axis=1, keepdims=True)
OUTPUT_PROJ = proj_rng.randn(H, V).astype(np.float32)
OUTPUT_PROJ /= np.linalg.norm(OUTPUT_PROJ, axis=0, keepdims=True)

# ── Minimal forward pass (stripped to core) ──────────────────────────────────
def forward(injected, mask, theta, ticks=TICKS, decay=DECAY):
    """Minimal INSTNCT: charge accumulate → threshold → spike → reset."""
    charge = np.zeros(H, dtype=np.float32)
    state  = np.zeros(H, dtype=np.float32)

    for tick in range(ticks):
        # Decay
        charge = np.maximum(charge - decay, 0.0)

        # Input (first tick only)
        if tick == 0:
            charge += injected

        # Propagate (sparse add from active neurons)
        raw = state @ mask.astype(np.float32)
        charge += raw
        np.clip(charge, 0.0, MAX_CHARGE, out=charge)

        # Spike decision
        fired = charge >= theta
        state = fired.astype(np.float32)
        charge[fired] = 0.0

    return charge, state

def eval_task(mask, theta, task_map):
    """Run all V inputs through projections, return accuracy."""
    correct = 0
    for inp, target in task_map.items():
        one_hot = np.zeros(V, dtype=np.float32)
        one_hot[inp] = 1.0
        injected = one_hot @ INPUT_PROJ
        charge, state = forward(injected, mask, theta)
        logits = charge @ OUTPUT_PROJ
        pred = int(np.argmax(logits))
        if pred == target:
            correct += 1
    return correct / V

# ── Enumerate all topologies ─────────────────────────────────────────────────
def int_to_mask(n):
    """Convert integer 0..4095 to 4x4 bool mask (no self-connections)."""
    mask = np.zeros((H, H), dtype=np.bool_)
    bit = 0
    for i in range(H):
        for j in range(H):
            if i == j:
                continue
            if n & (1 << bit):
                mask[i, j] = True
            bit += 1
    return mask

def mask_to_int(mask):
    """Convert mask back to integer."""
    n = 0; bit = 0
    for i in range(H):
        for j in range(H):
            if i == j: continue
            if mask[i, j]: n |= (1 << bit)
            bit += 1
    return n

def hamming(a, b):
    """Hamming distance between two topology integers."""
    x = a ^ b
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Topology sampling: H={H}, V={V}')
    print(f'Samples: {N_SAMPLES}, Edges possible: {N_EDGES}')
    print(f'Theta sweep: {THETA_SWEEP}')
    print(f'Tasks: {list(TASKS.keys())}')
    print()

    # Random sampling of topologies with varying density
    rng = np.random.RandomState(SEED)
    results = {}
    masks = {}
    t0 = time.time()

    for i in range(N_SAMPLES):
        # Sample density uniformly from 5% to 50%
        density = rng.uniform(0.05, 0.50)
        mask = (rng.rand(H, H) < density).astype(np.bool_)
        np.fill_diagonal(mask, False)
        topo_id = i  # just use index as ID

        masks[topo_id] = mask
        results[topo_id] = {}

        for task_name, task_map in TASKS.items():
            best_score = 0.0
            best_theta = 0
            for theta in THETA_SWEEP:
                sc = eval_task(mask, float(theta), task_map)
                if sc > best_score:
                    best_score = sc
                    best_theta = theta
            results[topo_id][task_name] = (best_score, best_theta)

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{N_SAMPLES} sampled ({elapsed:.0f}s)')
            sys.stdout.flush()

    elapsed = time.time() - t0
    print(f'Done in {elapsed:.1f}s\n')

    # ── Analysis ──────────────────────────────────────────────────────────────
    def mask_hamming(m1, m2):
        return int(np.sum(m1 != m2))

    all_ids = list(results.keys())

    for task_name in TASKS:
        print(f'=== {task_name.upper()} ===')

        scores = [results[t][task_name][0] for t in all_ids]
        dist = Counter(scores)
        for sc in sorted(dist.keys()):
            pct = dist[sc] / N_SAMPLES * 100
            bar = '█' * int(pct / 2)
            print(f'  {sc*100:5.0f}% acc: {dist[sc]:5d} / {N_SAMPLES} ({pct:.1f}%) {bar}')

        # Good topologies (>= 75% acc)
        good = [t for t in all_ids if results[t][task_name][0] >= 0.75]
        perfect = [t for t in all_ids if results[t][task_name][0] == 1.0]
        print(f'\n  >= 75% acc: {len(good)} topologies')
        print(f'  Perfect (100%): {len(perfect)} topologies')

        best_group = perfect if perfect else good
        if best_group:
            theta_dist = Counter(results[t][task_name][1] for t in best_group)
            print(f'  Preferred theta: {dict(sorted(theta_dist.items()))}')

            edge_counts = [int(np.sum(masks[t])) for t in best_group]
            print(f'  Edge count: min={min(edge_counts)} max={max(edge_counts)} '
                  f'mean={sum(edge_counts)/len(edge_counts):.1f}')

            # Clustering: avg hamming between good topologies
            sample = best_group[:200]
            if len(sample) > 1:
                hd = []
                for i in range(len(sample)):
                    for j in range(i+1, min(len(sample), i+50)):
                        hd.append(mask_hamming(masks[sample[i]], masks[sample[j]]))
                avg_hd = sum(hd) / len(hd)
                # Expected random: depends on avg density
                avg_density = np.mean([np.sum(masks[t]) for t in sample]) / N_EDGES
                expected_hd = N_EDGES * 2 * avg_density * (1 - avg_density)
                print(f'  Avg Hamming between good: {avg_hd:.2f}')
                print(f'  Expected random at same density: {expected_hd:.2f}')
                if avg_hd < expected_hd * 0.85:
                    print(f'  → CLUSTERED')
                elif avg_hd > expected_hd * 1.15:
                    print(f'  → DISPERSED')
                else:
                    print(f'  → NEAR RANDOM')

            # Show samples
            print(f'\n  Sample best topologies (first 5):')
            for t in best_group[:5]:
                n_edges = int(np.sum(masks[t]))
                th = results[t][task_name][1]
                sc = results[t][task_name][0]
                print(f'    id={t:4d} score={sc*100:.0f}% theta={th} edges={n_edges}')

        print()

    # ── Cross-task analysis ───────────────────────────────────────────────────
    print('=== CROSS-TASK ANALYSIS ===')
    tasks = list(TASKS.keys())
    perfect_a = set(t for t in all_ids if results[t][tasks[0]][0] == 1.0)
    perfect_b = set(t for t in all_ids if results[t][tasks[1]][0] == 1.0)
    good_a = set(t for t in all_ids if results[t][tasks[0]][0] >= 0.75)
    good_b = set(t for t in all_ids if results[t][tasks[1]][0] >= 0.75)

    print(f'  Perfect at {tasks[0]} only: {len(perfect_a - perfect_b)}')
    print(f'  Perfect at {tasks[1]} only: {len(perfect_b - perfect_a)}')
    print(f'  Perfect at BOTH: {len(perfect_a & perfect_b)}')
    print(f'  Good(>=75%) at {tasks[0]} only: {len(good_a - good_b)}')
    print(f'  Good(>=75%) at {tasks[1]} only: {len(good_b - good_a)}')
    print(f'  Good(>=75%) at BOTH: {len(good_a & good_b)}')

    both_good = good_a & good_b
    if both_good:
        print(f'\n  "Universal" topologies (good at both), first 5:')
        for t in list(both_good)[:5]:
            n_edges = int(np.sum(masks[t]))
            th_a = results[t][tasks[0]][1]
            th_b = results[t][tasks[1]][1]
            sa = results[t][tasks[0]][0]
            sb = results[t][tasks[1]][0]
            print(f'    id={t} {tasks[0]}={sa*100:.0f}%(θ={th_a}) '
                  f'{tasks[1]}={sb*100:.0f}%(θ={th_b}) edges={n_edges}')

    # Score correlation between tasks
    sa = np.array([results[t][tasks[0]][0] for t in all_ids])
    sb = np.array([results[t][tasks[1]][0] for t in all_ids])
    corr = np.corrcoef(sa, sb)[0, 1]
    print(f'\n  Score correlation between tasks: {corr:.4f}')
    if abs(corr) > 0.3:
        print(f'  → Tasks share topological preference')
    elif abs(corr) < 0.05:
        print(f'  → Tasks are topologically independent')
    else:
        print(f'  → Weak relationship')
