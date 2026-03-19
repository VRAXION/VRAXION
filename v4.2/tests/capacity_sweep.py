"""Capacity sweep: does more neurons help sequential char prediction?
Test N from 81 to 960 neurons with diff-guided mutation."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.graph import SelfWiringGraph
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ALPHA = 2.0

def load_and_clean(fn):
    with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    return ' '.join(''.join(c if 'a' <= c <= 'z' or c == ' ' else ' ' for c in text).split())


def forward_seq(net, world, ticks=8, alpha=ALPHA):
    act = net.state.copy()
    for t in range(ticks):
        if t == 0:
            act[:net.V] = world
        net.charge += act @ net.mask
        total = np.abs(act).sum() + 1e-6
        act = np.maximum(net.charge - net.THRESHOLD, 0.0)
        act *= (1.0 / (1.0 + alpha * total))
        np.clip(net.charge, -1.0, 1.0, out=net.charge)
    net.state = act.copy()
    return net.charge[net.out_start:net.out_start + net.V], act


def eval_seq(net, text, char2idx, ticks=8):
    V = net.V
    net.reset()
    correct = 0
    error_map = np.zeros(V, dtype=np.float32)
    active_map = np.zeros((V, net.N), dtype=np.float32)

    for i in range(len(text) - 1):
        world = np.zeros(V, dtype=np.float32)
        world[char2idx[text[i]]] = 1.0
        output, act = forward_seq(net, world, ticks)
        pred = np.argmax(output)
        actual = char2idx[text[i + 1]]
        if pred == actual:
            correct += 1
        target = np.zeros(V, dtype=np.float32)
        target[actual] = 1.0
        e_out = np.exp(output - output.max())
        probs = e_out / e_out.sum()
        diff = target - probs
        error_map += diff
        for j in range(V):
            if abs(diff[j]) > 0.01:
                active_map[j] += act * diff[j]

    n = len(text) - 1
    return correct / n, error_map / n, active_map / n


def targeted_mutation(net, error_map, active_map, n_changes=5):
    V = net.V; N = net.N; out_start = net.out_start
    undo = []
    worst = np.argsort(-np.abs(error_map))[:n_changes]

    for j in worst:
        out_neuron = out_start + j
        direction = np.sign(error_map[j])
        source_scores = active_map[j].copy()
        source_scores[out_neuron] = 0

        if direction > 0:
            best_source = np.argmax(np.abs(source_scores))
            if net.mask[best_source, out_neuron] == 0:
                net.mask[best_source, out_neuron] = net.DRIVE * direction
                net.alive.append((best_source, out_neuron))
                net.alive_set.add((best_source, out_neuron))
                undo.append(('A', best_source, out_neuron))
            else:
                old = net.mask[best_source, out_neuron]
                net.mask[best_source, out_neuron] = np.clip(old + 0.3 * direction, -3, 3)
                undo.append(('S', best_source, out_neuron, old))
        else:
            conns = [(abs(net.mask[r, out_neuron]), r) for r, c in net.alive if c == out_neuron]
            if conns:
                _, ws = max(conns)
                old = net.mask[ws, out_neuron]
                new_val = old + 0.3 * direction
                if abs(new_val) < 0.1:
                    net.mask[ws, out_neuron] = 0
                    net.alive_set.discard((ws, out_neuron))
                    undo.append(('R', ws, out_neuron, old))
                else:
                    net.mask[ws, out_neuron] = np.clip(new_val, -3, 3)
                    undo.append(('S', ws, out_neuron, old))

    for _ in range(3):
        rundo = []
        net._rewire(rundo)
        undo.extend(rundo)
        rundo2 = []
        net._add(rundo2)
        undo.extend(rundo2)

    return undo


def replay_undo(net, undo):
    for entry in reversed(undo):
        op = entry[0]
        if op == 'A':
            net.mask[entry[1], entry[2]] = 0
            net.alive_set.discard((entry[1], entry[2]))
        elif op == 'R':
            net.mask[entry[1], entry[2]] = entry[3]
            net.alive_set.add((entry[1], entry[2]))
        elif op == 'S':
            net.mask[entry[1], entry[2]] = entry[3]
        elif op == 'W':
            _, r, c_old, c_new = entry
            sign = net.mask[r, c_new]
            net.mask[r, c_new] = 0
            net.mask[r, c_old] = sign
            net.alive_set.discard((r, c_new))
            net.alive_set.add((r, c_old))
    net.resync_alive()


# ============================================================
# Main
# ============================================================

train_text = load_and_clean('pride_prejudice.txt')
test_text = load_and_clean('alice.txt')
chars = sorted(set(train_text + test_text))
char2idx = {c: i for i, c in enumerate(chars)}
V = len(chars)

transitions = Counter()
for i in range(len(train_text) - 1):
    transitions[(train_text[i], train_text[i+1])] += 1
best_next = {}
for c in chars:
    nexts = {c2: cnt for (c1, c2), cnt in transitions.items() if c1 == c}
    if nexts: best_next[c] = max(nexts, key=nexts.get)
    else: best_next[c] = ' '
bigram_test = sum(1 for i in range(len(test_text)-1)
                  if best_next.get(test_text[i]) == test_text[i+1]) / (len(test_text)-1)

print(f"CAPACITY SWEEP | V={V}")
print(f"Bigram: {bigram_test*100:.1f}% | Frequency: ~18.5% | Random: {100/V:.1f}%")
print(f"{'='*70}")

# Sweep NV_RATIO: more compute neurons per vocab unit
CONFIGS = [
    # nv_ratio, N, compute neurons, budget
    (3,   81,  27, 300),
    (5,  135,  81, 300),
    (8,  216, 162, 200),
    (12, 324, 270, 150),
    (16, 432, 378, 100),
    (24, 648, 594,  80),
    (32, 864, 810,  60),
]

print(f"{'NV':>4s} {'N':>5s} {'compute':>8s} {'train':>7s} {'test':>7s} "
      f"{'vs_bi':>7s} {'conns':>6s} {'active':>7s} {'time':>5s}")
print("-" * 65)

for nv_ratio, expected_N, compute_n, n_cycles in CONFIGS:
    old_nv = SelfWiringGraph.NV_RATIO
    SelfWiringGraph.NV_RATIO = nv_ratio
    np.random.seed(42)
    net = SelfWiringGraph(V)
    SelfWiringGraph.NV_RATIO = old_nv
    N = net.N

    eval_text = train_text[:200 + 1]
    test_eval = test_text[:200 + 1]

    score, error_map, active_map = eval_seq(net, eval_text, char2idx)
    best = score

    t0 = time.time()
    improved = 0
    for cycle in range(n_cycles):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = targeted_mutation(net, error_map, active_map, n_changes=5)
        new_score, new_err, new_act = eval_seq(net, eval_text, char2idx)
        if new_score > score:
            score = new_score
            best = max(best, score)
            error_map = new_err
            active_map = new_act
            improved += 1
        else:
            replay_undo(net, undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)

    elapsed = time.time() - t0

    # Test
    test_score, _, _ = eval_seq(net, test_eval, char2idx)

    # Count active neurons on a sample
    net.reset()
    world = np.zeros(V); world[char2idx['t']] = 1.0
    _, act = forward_seq(net, world)
    n_active = (act > 0.01).sum()

    diff_bi = test_score - bigram_test
    marker = " <<<" if test_score > bigram_test else ""
    print(f"{nv_ratio:4d} {N:5d} {N-2*V:8d} {best*100:6.1f}% {test_score*100:6.1f}% "
          f"{diff_bi*100:+6.1f}pp {net.count_connections():6d} {n_active:3d}/{N:3d} "
          f"{elapsed:4.0f}s{marker}", flush=True)
