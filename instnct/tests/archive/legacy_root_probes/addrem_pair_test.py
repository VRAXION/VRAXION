"""A/B/C: drive-only vs add+rem pair (joint vs sequential eval).

Control:    single signed drive, +N=add, -N=remove
Pair-joint: add_n(0-15) + rem_n(0-15) executed together, 1 eval, full revert
Pair-seq:   add_n executed+eval+revert, then rem_n executed+eval+revert (2 evals/step)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph

SEEDS = [0, 1, 2, 10, 42]
VOCAB = 64
BUDGET = 16000


def evaluate(net, targets):
    logits = net.forward_batch(8)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    V = net.V
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def train_drive(net, targets, budget):
    """Single signed drive (control)."""
    score = evaluate(net, targets)
    best = score
    stale = 0
    drive = 1
    trace = [drive]

    for att in range(budget):
        old_loss, old_drive = int(net.loss_pct), drive
        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))
        if random.randint(1, 20) <= 7:
            drive = max(-15, min(15, drive + random.choice([-1, 1])))
        undo = []
        if drive > 0:
            for _ in range(drive): net._add(undo)
        elif drive < 0:
            for _ in range(-drive): net._remove(undo)
        ns = evaluate(net, targets)
        if ns > score:
            score = ns; best = max(best, score); stale = 0
        else:
            net.replay(undo); net.loss_pct = np.int8(old_loss); drive = old_drive; stale += 1
        if (att + 1) % 500 == 0: trace.append(drive)
        if best >= 0.99 or stale >= 6000: break
    return best, drive, net.count_connections(), trace


def train_pair_joint(net, targets, budget):
    """add_n + rem_n executed together, single eval, full revert."""
    score = evaluate(net, targets)
    best = score
    stale = 0
    add_n, rem_n = 1, 0
    trace = [(add_n, rem_n)]

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_add, old_rem = add_n, rem_n
        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))
        if random.randint(1, 20) <= 7:
            add_n = max(0, min(15, add_n + random.choice([-1, 1])))
        if random.randint(1, 20) <= 7:
            rem_n = max(0, min(15, rem_n + random.choice([-1, 1])))
        undo = []
        for _ in range(add_n): net._add(undo)
        for _ in range(rem_n): net._remove(undo)
        ns = evaluate(net, targets)
        if ns > score:
            score = ns; best = max(best, score); stale = 0
        else:
            net.replay(undo); net.loss_pct = np.int8(old_loss)
            add_n, rem_n = old_add, old_rem; stale += 1
        if (att + 1) % 500 == 0: trace.append((add_n, rem_n))
        if best >= 0.99 or stale >= 6000: break
    return best, add_n, rem_n, net.count_connections(), trace


def train_pair_seq(net, targets, budget):
    """add_n eval+revert, then rem_n eval+revert. 2 evals per step."""
    score = evaluate(net, targets)
    best = score
    stale = 0
    add_n, rem_n = 1, 0
    trace = [(add_n, rem_n)]

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_add, old_rem = add_n, rem_n
        any_ok = False

        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))
        if random.randint(1, 20) <= 7:
            add_n = max(0, min(15, add_n + random.choice([-1, 1])))
        if random.randint(1, 20) <= 7:
            rem_n = max(0, min(15, rem_n + random.choice([-1, 1])))

        # Phase A: adds
        undo_add = []
        for _ in range(add_n): net._add(undo_add)
        ns = evaluate(net, targets)
        if ns > score:
            score = ns; best = max(best, score); any_ok = True
        else:
            net.replay(undo_add); add_n = old_add

        # Phase B: removes
        undo_rem = []
        for _ in range(rem_n): net._remove(undo_rem)
        ns = evaluate(net, targets)
        if ns > score:
            score = ns; best = max(best, score); any_ok = True
        else:
            net.replay(undo_rem); rem_n = old_rem

        if any_ok: stale = 0
        else: net.loss_pct = np.int8(old_loss); stale += 1

        if (att + 1) % 500 == 0: trace.append((add_n, rem_n))
        if best >= 0.99 or stale >= 6000: break
    return best, add_n, rem_n, net.count_connections(), trace


print(f"A/B/C: drive vs pair-joint vs pair-seq | V={VOCAB} budget={BUDGET}")
print(f"  Note: pair-seq does 2 evals/step (2x compute)")
print(f"{'seed':>6s}  {'drive':>8s}  {'pJoint':>8s}  {'pSeq':>8s}  {'d-j':>7s} {'d-s':>7s}  {'jFinal':>8s} {'sFinal':>8s}  {'dC':>5s} {'jC':>5s} {'sC':>5s}")
print("-" * 105)

d_tot = j_tot = s_tot = 0

for seed in SEEDS:
    # Drive
    np.random.seed(seed); random.seed(seed)
    net1 = SelfWiringGraph(VOCAB)
    tgt = np.arange(VOCAB); np.random.shuffle(tgt)
    random.seed(seed * 1000 + 1)
    d_sc, d_drv, d_cn, d_tr = train_drive(net1, tgt, BUDGET)

    # Pair-joint
    np.random.seed(seed); random.seed(seed)
    net2 = SelfWiringGraph(VOCAB)
    tgt2 = np.arange(VOCAB); np.random.shuffle(tgt2)
    random.seed(seed * 1000 + 1)
    j_sc, j_a, j_r, j_cn, j_tr = train_pair_joint(net2, tgt2, BUDGET)

    # Pair-seq
    np.random.seed(seed); random.seed(seed)
    net3 = SelfWiringGraph(VOCAB)
    tgt3 = np.arange(VOCAB); np.random.shuffle(tgt3)
    random.seed(seed * 1000 + 1)
    s_sc, s_a, s_r, s_cn, s_tr = train_pair_seq(net3, tgt3, BUDGET)

    d_tot += d_sc; j_tot += j_sc; s_tot += s_sc
    print(f"{seed:6d}  {d_sc*100:7.1f}%  {j_sc*100:7.1f}%  {s_sc*100:7.1f}%  "
          f"{(j_sc-d_sc)*100:+6.1f}% {(s_sc-d_sc)*100:+6.1f}%  "
          f"+{j_a}/-{j_r}    +{s_a}/-{s_r}   {d_cn:5d} {j_cn:5d} {s_cn:5d}")

    # Compact trajectory (every 2000 = every 4th trace point)
    dt = " ".join(f"{d_tr[i]:+d}" for i in range(0, len(d_tr), 4))
    jt = " ".join(f"+{j_tr[i][0]}/-{j_tr[i][1]}" for i in range(0, len(j_tr), 4))
    st = " ".join(f"+{s_tr[i][0]}/-{s_tr[i][1]}" for i in range(0, len(s_tr), 4))
    print(f"       drv: [{dt}]")
    print(f"       jnt: [{jt}]")
    print(f"       seq: [{st}]")
    sys.stdout.flush()

print("-" * 105)
n = len(SEEDS)
print(f"{'avg':>6s}  {d_tot/n*100:7.1f}%  {j_tot/n*100:7.1f}%  {s_tot/n*100:7.1f}%  "
      f"{(j_tot-d_tot)/n*100:+6.1f}% {(s_tot-d_tot)/n*100:+6.1f}%")
