"""How does tick count affect loop formation?
Train same network with different tick counts, analyze loop structure."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from model.graph import SelfWiringGraph

V = 32
BUDGET = 8000
SEED = 42
TICK_COUNTS = [4, 8, 16, 32]


def train_with_ticks(net, targets, budget, ticks):
    """Standard train loop with custom tick count."""
    def evaluate():
        logits = net.forward_batch(ticks)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        V = net.V
        acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
        tp = probs[np.arange(V), targets[:V]].mean()
        return 0.5 * acc + 0.5 * tp

    score = evaluate()
    best = score
    stale = 0

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new_score = evaluate()

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

        if best >= 0.99 or stale >= 6000:
            break

    return best, att + 1


def analyze_loops(net):
    """Return loop stats dict."""
    N = net.N
    V = net.V
    mask = net.mask

    # Bidirectional pairs
    bidir = 0
    for r, c in net.alive:
        if mask[c, r] != 0:
            bidir += 1
    bidir //= 2

    # SCCs
    graph = csr_matrix((mask != 0).astype(np.float32))
    n_comp, labels = connected_components(graph, directed=True, connection='strong')
    scc_sizes = np.bincount(labels)
    biggest_scc = scc_sizes.max()

    # 3-cycles (sample)
    cycles_3 = 0
    for r, c in net.alive[:500]:
        for c2 in range(N):
            if mask[c, c2] != 0 and mask[c2, r] != 0:
                cycles_3 += 1

    # Feedback edges (output→input, output→compute)
    zones_input = set(range(V))
    zones_compute = set(range(V, 2*V))
    zones_output = set(range(2*V, 3*V))

    fb_out_comp = sum(1 for r, c in net.alive if r in zones_output and c in zones_compute)
    fb_out_in = sum(1 for r, c in net.alive if r in zones_output and c in zones_input)
    fb_comp_in = sum(1 for r, c in net.alive if r in zones_compute and c in zones_input)
    fwd_in_comp = sum(1 for r, c in net.alive if r in zones_input and c in zones_compute)
    fwd_comp_out = sum(1 for r, c in net.alive if r in zones_compute and c in zones_output)

    feedback_total = fb_out_comp + fb_out_in + fb_comp_in
    forward_total = fwd_in_comp + fwd_comp_out

    # Average cycle length potential: how many full cycles a loop of length L can do
    # in T ticks = T/L
    return {
        'conns': len(net.alive),
        'bidir': bidir,
        'cycles_3': cycles_3,
        'biggest_scc': biggest_scc,
        'scc_pct': biggest_scc / N * 100,
        'feedback': feedback_total,
        'forward': forward_total,
        'fb_ratio': feedback_total / max(forward_total, 1),
    }


print(f"LOOP vs TICKS | V={V} budget={BUDGET} seed={SEED}")
print(f"{'ticks':>5s} {'score':>7s} {'conns':>6s} {'bidir':>6s} {'3cyc':>6s} "
      f"{'SCC%':>6s} {'fwd':>5s} {'fb':>5s} {'fb/fwd':>7s}")
print("-" * 65)

for ticks in TICK_COUNTS:
    # Fresh network each time (same seed)
    np.random.seed(SEED)
    random.seed(SEED)
    net = SelfWiringGraph(V)
    targets = np.arange(V)
    np.random.shuffle(targets)

    random.seed(SEED * 1000 + 1)
    best, steps = train_with_ticks(net, targets, BUDGET, ticks)
    stats = analyze_loops(net)

    print(f"{ticks:5d} {best*100:6.1f}% {stats['conns']:6d} {stats['bidir']:6d} "
          f"{stats['cycles_3']:6d} {stats['scc_pct']:5.1f}% {stats['forward']:5d} "
          f"{stats['feedback']:5d} {stats['fb_ratio']:6.2f}x")

# Also show: signal survival per loop length at each tick count
print(f"\n{'='*60}")
print("SIGNAL SURVIVAL: how much signal remains after L loop iterations")
print("(retention=0.85, each loop cycle = L ticks)")
print(f"{'loop_len':>8s}", end="")
for ticks in TICK_COUNTS:
    print(f" {'t='+str(ticks):>8s}", end="")
print()
print("-" * (8 + 9 * len(TICK_COUNTS)))

for loop_len in [2, 3, 4, 5, 6, 8]:
    print(f"{loop_len:8d}", end="")
    for ticks in TICK_COUNTS:
        full_cycles = ticks // loop_len
        signal = 0.85 ** (full_cycles * loop_len)
        print(f" {signal:7.1%}", end="")
    print(f"   ({', '.join(str(t // loop_len) for t in TICK_COUNTS)} cycles)")
