"""HP system: 3-phase V=128 with cap=V*120, seed 42, budget=48k.
HP replaces stale counter. Success→+1 HP, failure→-1 HP, HP=0→death→restart from best.
Baseline: 3-phase V=128 cap120 48k = 52.7% (exp 15, stale limit)."""
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

    # Save best checkpoint
    best_state = net.save_state()

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
        else:
            net.loss_pct = np.int8(old_loss)
            hp -= 1

        # Update best checkpoint
        if score > best:
            best = score
            best_state = net.save_state()

        # Death → rebirth
        if hp <= 0:
            lives += 1
            net.restore_state(best_state)
            score = best
            hp = hp_init
            add_n, rem_n, rew_n = 1, 0, 0

        if (att + 1) % 8000 == 0:
            conns = net.count_connections()
            print(f"  [{att+1:6d}] best={best*100:5.1f}% hp={hp:4d} lives={lives} conns={conns} +{add_n}/-{rem_n}/w{rew_n}")
            sys.stdout.flush()

        if best >= 0.99:
            break

    return best, net.count_connections(), att + 1, hp, lives, add_n, rem_n, rew_n


V = 128
CAP = V * 120  # 15360
BUDGET = 48000
seed = 42

print(f"HP 3-phase pair-seq V={V}, cap={CAP} (ratio=120), seed={seed}, budget={BUDGET}")
print(f"HP_INIT={HP_INIT}, baseline=52.7% (3-phase stale limit)")
print(f"N={V*3}, matrix={V*3}x{V*3}={V*3*V*3} elements")

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
best, conns, steps, hp, lives, a_n, r_n, w_n = train_hp_3phase(net, targets, BUDGET, CAP)
elapsed = time.time() - t0

print(f"\nResult: {best*100:.1f}% in {steps} steps ({elapsed:.0f}s)")
print(f"Final: hp={hp}, lives={lives}, +{a_n}/-{r_n}/w{w_n}, conns={conns}")
