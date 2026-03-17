"""3-phase pair-seq: add → remove → rewire, each independently evaluated.
Compare vs 2-phase (add → remove) at V=128, cap=10240, 5 seeds, budget=24k.
Hypothesis: rewire phase lets network restructure without changing conn count."""
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


def train_2phase(net, targets, budget, cap):
    """Standard pair-seq: add → remove."""
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

        # Phase A: adds
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

        if best >= 0.99 or stale >= 6000:
            break

    return best, net.count_connections(), att + 1, stale


def train_3phase(net, targets, budget, cap):
    """3-phase pair-seq: add → remove → rewire."""
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

        # Phase A: adds
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

        # Phase C: rewires
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

    return best, net.count_connections(), att + 1, stale, add_n, rem_n, rew_n


V = 128
CAP = V * 80  # 10240
BUDGET = 24000
SEEDS = [0, 1, 2, 10, 42]

print(f"2-phase vs 3-phase pair-seq | V={V}, cap={CAP}, budget={BUDGET}")
print(f"3-phase adds rewire as independent 3rd phase")
print(f"Note: 3-phase does 3 evals/step vs 2 evals/step (50% more compute)")
print(f"{'seed':>6s}  {'2-phase':>8s}  {'3-phase':>8s}  {'delta':>7s}  {'2c':>5s}  {'3c':>5s}  {'3-final':>10s}  {'2-time':>7s}  {'3-time':>7s}")
print("-" * 85)

tot2 = tot3 = 0

for seed in SEEDS:
    # 2-phase
    np.random.seed(seed); random.seed(seed)
    net2 = SelfWiringGraph(V)
    tgt = np.arange(V); np.random.shuffle(tgt)
    prune_to_cap(net2, CAP)
    random.seed(seed * 1000 + 1)
    t0 = time.time()
    sc2, cn2, st2, sl2 = train_2phase(net2, tgt, BUDGET, CAP)
    t2 = time.time() - t0

    # 3-phase
    np.random.seed(seed); random.seed(seed)
    net3 = SelfWiringGraph(V)
    tgt3 = np.arange(V); np.random.shuffle(tgt3)
    prune_to_cap(net3, CAP)
    random.seed(seed * 1000 + 1)
    t0 = time.time()
    sc3, cn3, st3, sl3, a3, r3, w3 = train_3phase(net3, tgt3, BUDGET, CAP)
    t3 = time.time() - t0

    tot2 += sc2; tot3 += sc3
    delta = (sc3 - sc2) * 100
    print(f"{seed:6d}  {sc2*100:7.1f}%  {sc3*100:7.1f}%  {delta:+6.1f}%  {cn2:5d}  {cn3:5d}  +{a3}/-{r3}/w{w3}  {t2:6.1f}s  {t3:6.1f}s")
    sys.stdout.flush()

print("-" * 85)
n = len(SEEDS)
print(f"{'avg':>6s}  {tot2/n*100:7.1f}%  {tot3/n*100:7.1f}%  {(tot3-tot2)/n*100:+6.1f}%")
print(f"\nBaseline from exp 7: 2-phase avg at 24k = 42.6%")
