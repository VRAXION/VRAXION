"""Scale test: sparse forward, increasing V until it breaks.
Uses scipy sparse where beneficial, pair-seq mutation."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from scipy import sparse as sp
from model.graph import SelfWiringGraph


def forward_batch_sparse(net, ticks=8):
    """Sparse forward using scipy CSR."""
    V, N = net.V, net.N
    mask_csr = sp.csr_matrix(net.mask)
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        raw = acts @ mask_csr
        if sp.issparse(raw):
            raw = raw.toarray()
        else:
            raw = np.asarray(raw)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def evaluate(net, targets, use_sparse=False):
    if use_sparse:
        logits = forward_batch_sparse(net)
    else:
        logits = net.forward_batch(8)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    V = net.V
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def train_pair_seq(net, targets, budget, use_sparse=False):
    """Pair-seq with optional sparse forward."""
    score = evaluate(net, targets, use_sparse)
    best = score
    stale = 0
    add_n, rem_n = 1, 0

    for att in range(budget):
        old_loss, old_add, old_rem = int(net.loss_pct), add_n, rem_n
        any_ok = False

        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))
        if random.randint(1, 20) <= 7:
            add_n = max(0, min(15, add_n + random.choice([-1, 1])))
        if random.randint(1, 20) <= 7:
            rem_n = max(0, min(15, rem_n + random.choice([-1, 1])))

        # Phase A: adds
        undo_add = []
        for _ in range(add_n):
            net._add(undo_add)
        ns = evaluate(net, targets, use_sparse)
        if ns > score:
            score = ns; best = max(best, score); any_ok = True
        else:
            net.replay(undo_add); add_n = old_add

        # Phase B: removes
        undo_rem = []
        for _ in range(rem_n):
            net._remove(undo_rem)
        ns = evaluate(net, targets, use_sparse)
        if ns > score:
            score = ns; best = max(best, score); any_ok = True
        else:
            net.replay(undo_rem); rem_n = old_rem

        if any_ok:
            stale = 0
        else:
            net.loss_pct = np.int8(old_loss); stale += 1

        if best >= 0.99 or stale >= 6000:
            break

    return best, add_n, rem_n, net.count_connections(), att + 1


# --- Main ---
seed = 42
budget = 8000  # keep manageable

print(f"Sparse scale test | seed={seed} budget={budget}")
print(f"L1=32KB  L2=512KB  L3=4MB")
print(f"{'V':>5s} {'N':>5s}  {'wset':>6s}  {'score':>7s}  {'conns':>6s} {'dens':>5s}  "
      f"{'a/r':>6s}  {'time':>7s}  {'ms/att':>7s}  {'mode':>6s}")
print("-" * 85)

for V in [32, 64, 96, 128, 160, 192, 224, 256]:
    N = V * 3
    wset_kb = (3 * V * N * 4 + int(N * N * 0.04) * 12) / 1024

    # Decide sparse vs dense based on size
    # Sparse wins at V>=128 with low density
    use_sparse = V >= 128

    np.random.seed(seed)
    random.seed(seed)
    net = SelfWiringGraph(V)
    targets = np.arange(V)
    np.random.shuffle(targets)
    random.seed(seed * 1000 + 1)

    t0 = time.time()
    sc, a_n, r_n, conns, steps = train_pair_seq(net, targets, budget, use_sparse)
    elapsed = time.time() - t0

    density = conns / (N * N) * 100
    ms_att = elapsed / steps * 1000
    mode = "sparse" if use_sparse else "dense"

    if wset_kb < 512:
        cache = "L2"
    elif wset_kb < 4096:
        cache = "L3"
    else:
        cache = "RAM"

    print(f"{V:5d} {N:5d}  {wset_kb:5.0f}K  {sc*100:6.1f}%  {conns:6d} {density:4.1f}%  "
          f"+{a_n}/-{r_n}  {elapsed:6.1f}s  {ms_att:6.2f}ms  {mode:>6s}  [{cache}]")
    sys.stdout.flush()
