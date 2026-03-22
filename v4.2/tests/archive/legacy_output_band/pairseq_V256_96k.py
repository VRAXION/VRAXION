"""Pair-seq V=256, cap=20480, seed 42, budget=96k.
Extends experiment 10 (48k→31.3%) to see if it approaches 40%."""
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


def train_pair_seq_capped(net, targets, budget, cap):
    score = evaluate(net, targets)
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

        # Phase A: adds (skip if at cap)
        undo_add = []
        if net.count_connections() < cap:
            for _ in range(add_n):
                net._add(undo_add)
        ns = evaluate(net, targets)
        if ns > score:
            score = ns; best = max(best, score); any_ok = True
        else:
            net.replay(undo_add); add_n = old_add

        # Phase B: removes
        undo_rem = []
        for _ in range(rem_n):
            net._remove(undo_rem)
        ns = evaluate(net, targets)
        if ns > score:
            score = ns; best = max(best, score); any_ok = True
        else:
            net.replay(undo_rem); rem_n = old_rem

        if any_ok:
            stale = 0
        else:
            net.loss_pct = np.int8(old_loss); stale += 1

        if (att + 1) % 8000 == 0:
            conns = net.count_connections()
            print(f"  [{att+1:6d}] best={best*100:5.1f}% conns={conns} +{add_n}/-{rem_n} stale={stale}")
            sys.stdout.flush()

        if best >= 0.99 or stale >= 6000:
            break

    return best, add_n, rem_n, net.count_connections(), att + 1, stale


V = 256
CAP = V * 80  # 20480
BUDGET = 96000
seed = 42

print(f"Pair-seq V={V}, cap={CAP}, seed={seed}, budget={BUDGET}")
print(f"Projected ~40% based on V=128 scaling curve")

np.random.seed(seed)
random.seed(seed)
net = SelfWiringGraph(V)
targets = np.arange(V)
np.random.shuffle(targets)

# Enforce cap at init
init_conns = net.count_connections()
prune_to_cap(net, CAP)
print(f"Init conns: {init_conns} → {net.count_connections()} (cap={CAP})")

random.seed(seed * 1000 + 1)
t0 = time.time()
best, a_n, r_n, conns, steps, stale = train_pair_seq_capped(net, targets, BUDGET, CAP)
elapsed = time.time() - t0

print(f"\nResult: {best*100:.1f}% in {steps} steps ({elapsed:.0f}s)")
print(f"Final: +{a_n}/-{r_n}, conns={conns}, stale={stale}")
