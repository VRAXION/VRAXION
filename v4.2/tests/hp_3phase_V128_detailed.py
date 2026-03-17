"""HP system DETAILED: log every death to see cross-life evolution.
Same config as hp_3phase_V128.py but with per-life stats."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from scipy import sparse as sp
from model.graph import SelfWiringGraph


HP_INIT = 100


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


def train_hp_3phase(net, targets, budget, cap, hp_init=HP_INIT):
    score = evaluate(net, targets)
    best = score
    hp = hp_init
    lives = 0
    add_n, rem_n, rew_n = 1, 0, 0

    best_state = net.save_state()

    # Per-life tracking
    life_start = 0
    life_successes = 0
    life_best_gain = 0.0  # how much this life improved best
    life_start_score = best
    hp_peak = hp  # max HP reached this life

    # Cross-life stats
    life_log = []  # (life_len, successes, improved_best, hp_peak, conns_at_death, add_n, rem_n, rew_n)

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_add, old_rem, old_rew = add_n, rem_n, rew_n
        any_ok = False

        # Drift params
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
            score = ns; any_ok = True
        else:
            net.replay(undo_add); add_n = old_add

        # Phase B: removes
        undo_rem = []
        for _ in range(rem_n):
            net._remove(undo_rem)
        ns = evaluate(net, targets)
        if ns > score:
            score = ns; any_ok = True
        else:
            net.replay(undo_rem); rem_n = old_rem

        # Phase C: rewires
        undo_rew = []
        for _ in range(rew_n):
            net._rewire(undo_rew)
        ns = evaluate(net, targets)
        if ns > score:
            score = ns; any_ok = True
        else:
            net.replay(undo_rew); rew_n = old_rew

        # HP system
        if any_ok:
            hp += 1
            life_successes += 1
        else:
            net.loss_pct = np.int8(old_loss)
            hp -= 1

        hp_peak = max(hp_peak, hp)

        # Update best checkpoint
        if score > best:
            best = score
            best_state = net.save_state()

        # Death → rebirth
        if hp <= 0:
            life_len = att - life_start
            improved = best - life_start_score
            conns = net.count_connections()
            life_log.append((life_len, life_successes, improved, hp_peak, conns, add_n, rem_n, rew_n))

            lives += 1
            net.restore_state(best_state)
            score = best
            hp = hp_init
            add_n, rem_n, rew_n = 1, 0, 0

            # Reset per-life tracking
            life_start = att + 1
            life_successes = 0
            life_start_score = best
            hp_peak = hp

        if best >= 0.99:
            break

    # Final life (didn't die)
    life_len = att - life_start + 1
    improved = best - life_start_score
    conns = net.count_connections()
    life_log.append((life_len, life_successes, improved, hp_peak, conns, add_n, rem_n, rew_n))

    return best, conns, att + 1, hp, lives, life_log


V = 128
CAP = V * 120
BUDGET = 48000
seed = 42

print(f"HP 3-phase DETAILED V={V}, cap={CAP}, seed={seed}, budget={BUDGET}")
print(f"HP_INIT={HP_INIT}")

np.random.seed(seed)
random.seed(seed)
net = SelfWiringGraph(V)
targets = np.arange(V)
np.random.shuffle(targets)

init_conns = net.count_connections()
prune_to_cap(net, CAP)
print(f"Init conns: {init_conns} → {net.count_connections()} (cap={CAP})")

random.seed(seed * 1000 + 1)
t0 = time.time()
best, conns, steps, hp, lives, life_log = train_hp_3phase(net, targets, BUDGET, CAP)
elapsed = time.time() - t0

print(f"\nResult: {best*100:.1f}% in {steps} steps ({elapsed:.0f}s)")
print(f"Total lives: {lives + 1} (deaths: {lives})")
print()

# Analyze life phases
print("=== LIFE LOG (every death) ===")
print(f"{'Life':>5} {'Len':>6} {'Succ':>5} {'Rate':>6} {'Improved':>9} {'HP peak':>8} {'Conns':>6} {'A/R/W':>8}")

# Show first 10, some middle, last 10
total = len(life_log)
show_indices = set()
show_indices.update(range(min(15, total)))  # first 15
show_indices.update(range(max(0, total - 10), total))  # last 10
# some evenly spaced in middle
if total > 30:
    for i in range(15, total - 10, max(1, (total - 25) // 10)):
        show_indices.add(i)

for i in sorted(show_indices):
    ln, succ, imp, hp_pk, cn, a, r, w = life_log[i]
    rate = succ / max(ln, 1) * 100
    if i > 0 and i - 1 not in show_indices:
        print("  ...")
    print(f"{i+1:5d} {ln:6d} {succ:5d} {rate:5.1f}% {imp*100:+8.2f}% {hp_pk:8d} {cn:6d} +{a}/-{r}/w{w}")

# Summary stats by era (first 25%, middle 50%, last 25%)
print("\n=== ERA ANALYSIS ===")
q1 = total // 4
q3 = 3 * total // 4
eras = [
    ("Early (1st 25%)", life_log[:q1]),
    ("Middle (25-75%)", life_log[q1:q3]),
    ("Late (last 25%)", life_log[q3:]),
]
for name, era in eras:
    if not era:
        continue
    lens = [e[0] for e in era]
    succs = [e[1] for e in era]
    imps = [e[2] for e in era]
    hps = [e[3] for e in era]
    avg_len = sum(lens) / len(lens)
    avg_succ_rate = sum(succs) / max(sum(lens), 1) * 100
    productive = sum(1 for i in imps if i > 0)
    avg_hp_peak = sum(hps) / len(hps)
    print(f"{name}: {len(era)} lives, avg_len={avg_len:.0f}, succ_rate={avg_succ_rate:.1f}%, "
          f"productive={productive}/{len(era)}, avg_hp_peak={avg_hp_peak:.0f}")
