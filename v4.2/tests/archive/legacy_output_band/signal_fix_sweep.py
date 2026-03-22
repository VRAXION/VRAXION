"""Signal fix sweep: why does the signal die? Test fixes.
The problem: 0-2 active neurons out of 81, signal doesn't reach output.
Test: more ticks, weaker divnorm, denser init, direct highways."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.graph import SelfWiringGraph
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_and_clean(fn):
    with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    return ' '.join(''.join(c if 'a' <= c <= 'z' or c == ' ' else ' ' for c in text).split())

train_text = load_and_clean('pride_prejudice.txt')
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


def forward_seq(net, world, ticks, alpha):
    act = net.state.copy()
    for t in range(ticks):
        if t == 0:
            act[:net.V] = world
        net.charge += act @ net.mask
        total = np.abs(act).sum() + 1e-6
        act = np.maximum(net.charge - net.THRESHOLD, 0.0)
        act *= (1.0 / (1.0 + alpha * total))
        np.clip(net.charge, -1.0, 1.0, out=net.charge)
    net.state = act.copy()
    return net.charge[net.out_start:net.out_start + net.V], act


def stream_and_learn(net, text, char2idx, lr, ticks, alpha,
                     max_chars=10000, log_every=5000):
    V = net.V; N = net.N; out_start = net.out_start
    net.reset()
    correct = 0; total = 0; recent_correct = 0; recent_total = 0

    for i in range(min(len(text) - 1, max_chars)):
        world = np.zeros(V, dtype=np.float32)
        world[char2idx[text[i]]] = 1.0

        # Phase 1: predict (full ticks)
        output, act = forward_seq(net, world, ticks, alpha)
        pred = np.argmax(output)
        actual = char2idx[text[i + 1]]
        is_correct = (pred == actual)
        if is_correct: correct += 1; recent_correct += 1
        total += 1; recent_total += 1

        # Delta update on output connections
        if lr > 0:
            target = np.zeros(V, dtype=np.float32)
            target[actual] = 1.0
            e_out = np.exp(output - output.max())
            probs = e_out / e_out.sum()
            diff = target - probs

            active_idx = np.where(act > 0.01)[0]
            if len(active_idx) > 0 and len(active_idx) < N // 2:
                act_a = act[active_idx]
                out_idx = np.arange(out_start, out_start + V)
                delta = np.outer(act_a, diff) * lr
                sub = net.mask[np.ix_(active_idx, out_idx)]
                wh = sub != 0
                if wh.any():
                    sub[wh] += delta[wh]
                    np.clip(sub, -3.0, 3.0, out=sub)
                    net.mask[np.ix_(active_idx, out_idx)] = sub

                # Exploration
                if not is_correct and np.random.random() < 0.05:
                    src = np.random.choice(active_idx)
                    dst = out_start + actual
                    if src != dst and net.mask[src, dst] == 0:
                        net.mask[src, dst] = net.DRIVE
                        net.alive.append((src, dst))
                        net.alive_set.add((src, dst))

        if (i + 1) % log_every == 0:
            r_acc = recent_correct / max(recent_total, 1)
            o_acc = correct / total
            active_count = (act > 0.01).sum()
            print(f"    [{i+1:6d}] overall={o_acc*100:.1f}% recent={r_acc*100:.1f}% "
                  f"active={active_count}/{N} conns={net.count_connections()}", flush=True)
            recent_correct = 0; recent_total = 0

    return correct / max(total, 1)


def add_highways(net):
    """Add direct input→output connections."""
    V = net.V; out_start = net.out_start
    added = 0
    for i in range(V):
        o = out_start + i
        if net.mask[i, o] == 0:
            net.mask[i, o] = net.DRIVE
            net.alive.append((i, o))
            net.alive_set.add((i, o))
            added += 1
    return added


def set_density(net, target_pct):
    """Reinitialize mask with higher density."""
    N = net.N
    net.mask[:] = 0
    d = target_pct / 100
    r = np.random.rand(N, N)
    net.mask[r < d / 2] = -net.DRIVE
    net.mask[r > 1 - d / 2] = net.DRIVE
    np.fill_diagonal(net.mask, 0)
    net.resync_alive()


# ============================================================
# Configs to test
# ============================================================

CONFIGS = [
    # name,          ticks, alpha, density%, highway, lr
    ('baseline_4t',      4,  2.0,    4,  False, 0.1),
    ('8ticks',           8,  2.0,    4,  False, 0.1),
    ('16ticks',         16,  2.0,    4,  False, 0.1),
    ('weak_divnorm',     8,  0.5,    4,  False, 0.1),
    ('no_divnorm',       8,  0.0,    4,  False, 0.1),
    ('dense_10pct',      8,  2.0,   10,  False, 0.1),
    ('dense_20pct',      8,  2.0,   20,  False, 0.1),
    ('highway',          8,  2.0,    4,   True, 0.1),
    ('highway+weak_dn',  8,  0.5,    4,   True, 0.1),
    ('dense10+highway',  8,  2.0,   10,   True, 0.1),
    ('dense10+weak_dn',  8,  0.5,   10,  False, 0.1),
    ('best_combo',       8,  0.5,   10,   True, 0.1),
]

print(f"SIGNAL FIX SWEEP | V={V} N={V*3}")
print(f"Bigram baseline: {bigram_test*100:.1f}% | Frequency: ~18.5% | Random: {100/V:.1f}%")
print(f"{'='*80}")
print(f"{'config':<20s} {'train':>7s} {'test':>7s} {'vs_bi':>7s} {'conns':>6s} {'time':>5s}")
print("-" * 55)

results = []
for name, ticks, alpha, density, highway, lr in CONFIGS:
    np.random.seed(42)
    net = SelfWiringGraph(V)

    if density != 4:
        set_density(net, density)
    if highway:
        add_highways(net)

    init_conns = net.count_connections()

    print(f"\n--- {name} (t={ticks} a={alpha} d={density}% hw={highway}) ---")
    print(f"  Init conns: {init_conns}")

    # Train
    t0 = time.time()
    train_acc = stream_and_learn(net, train_text, char2idx, lr, ticks, alpha,
                                  max_chars=20000, log_every=10000)
    # Test (no learning)
    test_acc = stream_and_learn(net, test_text, char2idx, 0, ticks, alpha,
                                 max_chars=5000, log_every=5000)
    elapsed = time.time() - t0

    diff_bi = test_acc - bigram_test
    marker = " <<<" if test_acc > bigram_test else ""
    print(f"  Result: train={train_acc*100:.1f}% test={test_acc*100:.1f}% "
          f"vs_bigram={diff_bi*100:+.1f}pp conns={net.count_connections()}{marker}")
    results.append((name, train_acc, test_acc, net.count_connections(), elapsed))

print(f"\n{'='*80}")
print(f"SUMMARY (sorted by test accuracy)")
print(f"{'config':<20s} {'train':>7s} {'test':>7s} {'vs_bi':>7s} {'conns':>6s} {'time':>5s}")
print("-" * 55)
for name, tr, te, conns, elapsed in sorted(results, key=lambda x: -x[2]):
    diff = te - bigram_test
    marker = " <<<" if te > bigram_test else ""
    print(f"{name:<20s} {tr*100:6.1f}% {te*100:6.1f}% {diff*100:+6.1f}pp {conns:6d} {elapsed:4.0f}s{marker}")
