"""Cap sweep with 3-phase at V=128, seed 42, budget=48k.
Is the ~52-53% ceiling cap-limited? Exp 5 showed plateau at ratio=120 for 2-phase.
Test if 3-phase benefits from higher cap."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from scipy import sparse as sp
from model.graph import SelfWiringGraph


def prune_to_cap(net, cap):
    while len(net.alive) > cap:
        idx = random.randint(0, len(net.alive) - 1)
        r, c = net.alive[idx]
        net.mask[r, c] = 0
        net.alive[idx] = net.alive[-1]
        net.alive.pop()
        net.alive_set.discard((r, c))


def forward_batch_sparse(net, ticks=8):
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


def evaluate(net, targets):
    logits = forward_batch_sparse(net)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    V = net.V
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def train_3phase(net, targets, budget, cap):
    score = evaluate(net, targets)
    best = score
    stale = 0
    add_n, rem_n, rew_n = 1, 0, 0

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_add, old_rem, old_rew = add_n, rem_n, rew_n
        any_ok = False

        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))
        if random.randint(1, 20) <= 7:
            add_n = max(0, min(15, add_n + random.choice([-1, 1])))
        if random.randint(1, 20) <= 7:
            rem_n = max(0, min(15, rem_n + random.choice([-1, 1])))
        if random.randint(1, 20) <= 7:
            rew_n = max(0, min(15, rew_n + random.choice([-1, 1])))

        undo_add = []
        if net.count_connections() < cap:
            for _ in range(add_n):
                net._add(undo_add)
        ns = evaluate(net, targets)
        if ns > score:
            score = ns; best = max(best, score); any_ok = True
        else:
            net.replay(undo_add); add_n = old_add

        undo_rem = []
        for _ in range(rem_n):
            net._remove(undo_rem)
        ns = evaluate(net, targets)
        if ns > score:
            score = ns; best = max(best, score); any_ok = True
        else:
            net.replay(undo_rem); rem_n = old_rem

        undo_rew = []
        for _ in range(rew_n):
            net._rewire(undo_rew)
        ns = evaluate(net, targets)
        if ns > score:
            score = ns; best = max(best, score); any_ok = True
        else:
            net.replay(undo_rew); rew_n = old_rew

        if any_ok:
            stale = 0
        else:
            net.loss_pct = np.int8(old_loss); stale += 1

        if best >= 0.99 or stale >= 6000:
            break

    return best, net.count_connections(), att + 1, stale


V = 128
BUDGET = 48000
seed = 42
RATIOS = [80, 120, 160, 200]

print(f"3-phase cap sweep V={V}, seed={seed}, budget={BUDGET}")
print(f"2-phase cap sweep (exp 5, 12k): ratio=80→39.6%, 120→42.1%, 160→42.1%")
print(f"3-phase@cap80@48k: 50.6% (exp 13, seed 42)")
print(f"{'ratio':>6s}  {'cap':>6s}  {'score':>7s}  {'conns':>6s}  {'time':>7s}")
print("-" * 45)

for ratio in RATIOS:
    cap = V * ratio
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(V)
    targets = np.arange(V); np.random.shuffle(targets)
    prune_to_cap(net, cap)
    random.seed(seed * 1000 + 1)

    t0 = time.time()
    best, conns, steps, stale = train_3phase(net, targets, BUDGET, cap)
    elapsed = time.time() - t0

    print(f"{ratio:6d}  {cap:6d}  {best*100:6.1f}%  {conns:6d}  {elapsed:6.1f}s")
    sys.stdout.flush()
