"""Does divisive normalization enable more ticks + longer loops?
Compare: retain=0.85 vs divisive norm, at ticks 4/8/16/32.
Divisive norm: acts /= (1 + alpha * total_activity) — no decay, just normalization."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from model.graph import SelfWiringGraph

V = 32
BUDGET = 8000
SEED = 42
ALPHA = 2.0  # divisive norm strength (from mobile branch finding)


def forward_batch_divnorm(net, ticks=8, alpha=ALPHA):
    """Forward with divisive normalization instead of decay."""
    V, N = net.V, net.N
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        # NO decay (retain=1.0) — divisive norm handles stability
        # Divisive normalization per sample
        total = np.abs(acts).sum(axis=1, keepdims=True) + 1e-6
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        acts /= (1.0 + alpha * total)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def forward_batch_both(net, ticks=8, alpha=ALPHA):
    """Forward with BOTH retain + divisive norm — maybe best of both worlds."""
    V, N = net.V, net.N
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain  # decay
        total = np.abs(acts).sum(axis=1, keepdims=True) + 1e-6
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        acts /= (1.0 + alpha * total)  # + normalization
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def train_custom(net, targets, budget, forward_fn, ticks):
    def evaluate():
        logits = forward_fn(net, ticks)
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
    N = net.N
    mask = net.mask
    bidir = 0
    for r, c in net.alive:
        if mask[c, r] != 0:
            bidir += 1
    bidir //= 2

    graph = csr_matrix((mask != 0).astype(np.float32))
    _, labels = connected_components(graph, directed=True, connection='strong')
    scc_sizes = np.bincount(labels)

    cycles_3 = 0
    for r, c in net.alive[:500]:
        for c2 in range(N):
            if mask[c, c2] != 0 and mask[c2, r] != 0:
                cycles_3 += 1

    V = net.V
    fb = sum(1 for r, c in net.alive if r >= 2*V and c < 2*V)  # output→input/compute
    fwd = sum(1 for r, c in net.alive if r < 2*V and c >= 2*V)  # input/compute→output

    return {
        'conns': len(net.alive),
        'bidir': bidir,
        'cycles_3': cycles_3,
        'scc_max': scc_sizes.max(),
        'fb': fb,
        'fwd': fwd,
    }


# --- Main ---
MODES = {
    'retain':  lambda net, t: net.forward_batch(t),
    'divnorm': forward_batch_divnorm,
    'both':    forward_batch_both,
}
TICKS = [4, 8, 16, 32]

print(f"LOOP + DIVNORM vs TICKS | V={V} budget={BUDGET} seed={SEED} alpha={ALPHA}")
print(f"{'mode':<8s} {'ticks':>5s} {'score':>7s} {'time':>5s} {'conns':>6s} "
      f"{'bidir':>6s} {'3cyc':>6s} {'fb':>5s} {'fwd':>5s} {'fb/fw':>6s}")
print("-" * 72)

for mode_name, forward_fn in MODES.items():
    for ticks in TICKS:
        np.random.seed(SEED)
        random.seed(SEED)
        net = SelfWiringGraph(V)
        targets = np.arange(V)
        np.random.shuffle(targets)

        random.seed(SEED * 1000 + 1)
        t0 = time.time()
        best, steps = train_custom(net, targets, BUDGET, forward_fn, ticks)
        elapsed = time.time() - t0
        s = analyze_loops(net)

        fb_ratio = s['fb'] / max(s['fwd'], 1)
        print(f"{mode_name:<8s} {ticks:5d} {best*100:6.1f}% {elapsed:4.0f}s {s['conns']:6d} "
              f"{s['bidir']:6d} {s['cycles_3']:6d} {s['fb']:5d} {s['fwd']:5d} {fb_ratio:5.2f}x",
              flush=True)
    print()
