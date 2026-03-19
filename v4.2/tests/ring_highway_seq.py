"""Ring highway test: fixed circular backbone + mutation for sequential char LM.
The ring guarantees signal propagation — mutations learn what to do with it.
Tests: no ring (baseline), ring, double ring (fwd+bwd), ring + cross shortcuts."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from collections import Counter
from model.graph import SelfWiringGraph

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ALPHA = 2.0


def load_and_clean(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    clean = ''.join(c if (c >= 'a' and c <= 'z') or c == ' ' else ' ' for c in text)
    return ' '.join(clean.split())


def forward_divnorm_seq(net, world, ticks=8, alpha=ALPHA):
    act = net.state.copy()
    for t in range(ticks):
        if t == 0:
            act[:net.V] = world
        raw = act @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        net.charge += raw
        total = np.abs(act).sum() + 1e-6
        act = np.maximum(net.charge - net.THRESHOLD, 0.0)
        act /= (1.0 + alpha * total)
        net.charge = np.clip(net.charge, -1.0, 1.0)
    net.state = act.copy()
    return net.charge[net.out_start:net.out_start + net.V]


def evaluate_seq(net, text, char2idx, ticks=8, max_chars=200):
    net.reset()
    correct = 0
    V = net.V
    length = min(len(text) - 1, max_chars)
    for i in range(length):
        world = np.zeros(V, dtype=np.float32)
        world[char2idx[text[i]]] = 1.0
        output = forward_divnorm_seq(net, world, ticks)
        pred = np.argmax(output)
        if pred == char2idx[text[i + 1]]:
            correct += 1
    return correct / length


def add_ring(net, protected_set):
    """Add forward ring: i→i+1→...→N-1→0"""
    N = net.N
    for i in range(N):
        j = (i + 1) % N
        if net.mask[i, j] == 0:
            net.mask[i, j] = net.DRIVE
            net.alive.append((i, j))
            net.alive_set.add((i, j))
        protected_set.add((i, j))


def add_reverse_ring(net, protected_set):
    """Add backward ring: i→i-1→...→0→N-1"""
    N = net.N
    for i in range(N):
        j = (i - 1) % N
        if net.mask[i, j] == 0:
            net.mask[i, j] = net.DRIVE
            net.alive.append((i, j))
            net.alive_set.add((i, j))
        protected_set.add((i, j))


def add_cross_shortcuts(net, protected_set, n_shortcuts=20):
    """Add fixed shortcuts: input→output zone direct links"""
    V = net.V
    out_start = net.out_start
    np.random.seed(999)  # deterministic shortcuts
    for _ in range(n_shortcuts):
        i = np.random.randint(0, V)  # input neuron
        o = np.random.randint(out_start, out_start + V)  # output neuron
        if net.mask[i, o] == 0:
            net.mask[i, o] = net.DRIVE
            net.alive.append((i, o))
            net.alive_set.add((i, o))
        protected_set.add((i, o))


def train_seq_protected(net, train_text, char2idx, budget, ticks=8,
                        seq_len=100, stale_limit=8000, protected=None):
    """Train with protected connections (ring highway can't be mutated away)."""
    if protected is None:
        protected = set()

    eval_text = train_text[:seq_len + 1]

    def evaluate():
        return evaluate_seq(net, eval_text, char2idx, ticks, seq_len)

    # Custom mutate that respects protected connections
    def safe_mutate():
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        # Check if any protected connections were removed
        for entry in undo:
            if entry[0] == 'R':  # remove
                r, c = entry[1], entry[2]
                if (r, c) in protected:
                    # Restore this connection
                    net.mask[r, c] = entry[3]
                    net.alive.append((r, c))
                    net.alive_set.add((r, c))
        return undo, old_loss, old_drive

    score = evaluate()
    best = score
    stale = 0
    trajectory = [(0, float(best))]

    for att in range(budget):
        undo, old_loss, old_drive = safe_mutate()
        new_score = evaluate()

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            # Re-ensure protected connections after replay
            for r, c in protected:
                if net.mask[r, c] == 0:
                    net.mask[r, c] = net.DRIVE
                    if (r, c) not in net.alive_set:
                        net.alive.append((r, c))
                        net.alive_set.add((r, c))
            stale += 1

        if (att + 1) % 4000 == 0:
            trajectory.append((att + 1, float(best)))
        if best >= 0.99 or stale >= stale_limit:
            break

    if (att + 1) % 4000 != 0:
        trajectory.append((att + 1, float(best)))
    return float(best), att + 1, trajectory


# ============================================================
# Main
# ============================================================

train_text = load_and_clean('pride_prejudice.txt') + ' ' + load_and_clean('frankenstein.txt')
test_text = load_and_clean('alice.txt')
chars = sorted(set(train_text + test_text))
char2idx = {c: i for i, c in enumerate(chars)}
V = len(chars)

# Bigram baseline
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

print(f"V={V} N={V*3} | Bigram test baseline: {bigram_test*100:.1f}%")
print(f"{'='*70}")

CONFIGS = [
    ('no_ring',      lambda net, ps: None),
    ('ring_fwd',     lambda net, ps: add_ring(net, ps)),
    ('ring_bidir',   lambda net, ps: (add_ring(net, ps), add_reverse_ring(net, ps))),
    ('ring+short',   lambda net, ps: (add_ring(net, ps), add_cross_shortcuts(net, ps))),
]

BUDGET = 16000
SEQ_LEN = 50
TICKS = 4
SEED = 42

results = []

for name, setup_fn in CONFIGS:
    print(f"\n--- {name} ---")
    np.random.seed(SEED)
    random.seed(SEED)
    net = SelfWiringGraph(V)
    protected = set()
    setup_fn(net, protected)
    init_conns = net.count_connections()

    random.seed(SEED * 1000 + 1)
    t0 = time.time()
    best, steps, traj = train_seq_protected(
        net, train_text, char2idx, BUDGET, TICKS, SEQ_LEN,
        protected=protected)
    elapsed = time.time() - t0

    # Test on Alice
    test_accs = []
    for start in range(0, min(len(test_text)-SEQ_LEN-1, 3000), SEQ_LEN):
        acc = evaluate_seq(net, test_text[start:start+SEQ_LEN+1], char2idx, TICKS, SEQ_LEN)
        test_accs.append(acc)
    test_avg = np.mean(test_accs)

    traj_str = " -> ".join(f"{b*100:.1f}" for _, b in traj)
    print(f"Train: {best*100:.1f}% Test: {test_avg*100:.1f}% ({elapsed:.0f}s, {steps} steps)")
    print(f"Init conns: {init_conns} Final: {net.count_connections()} Protected: {len(protected)}")
    print(f"Trajectory: {traj_str}")
    results.append((name, best, test_avg, len(protected), net.count_connections(), elapsed))

print(f"\n{'='*70}")
print(f"SUMMARY | bigram_test={bigram_test*100:.1f}%")
print(f"{'name':<15s} {'train':>7s} {'test':>7s} {'vs_bi':>7s} {'prot':>5s} {'conns':>6s} {'time':>5s}")
print("-" * 55)
for name, train, test, prot, conns, elapsed in results:
    diff = test - bigram_test
    marker = " <<<" if test > bigram_test else ""
    print(f"{name:<15s} {train*100:6.1f}% {test*100:6.1f}% {diff*100:+6.1f}pp {prot:5d} {conns:6d} {elapsed:4.0f}s{marker}")
