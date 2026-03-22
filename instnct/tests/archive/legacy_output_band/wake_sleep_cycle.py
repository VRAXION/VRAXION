"""Wake/Sleep cycle training on word bigram V=128.
WAKE: sparse add + mostly rewire, 1 connection/step
SLEEP: only remove (crystallize), 1 connection/step, strict monotonic
Tests: random remove vs weakest remove in crystallize."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from collections import Counter
from model.graph import SelfWiringGraph

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ALPHA = 2.0


def load_and_clean(fn):
    with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    return ' '.join(''.join(c if 'a' <= c <= 'z' or c == ' ' else ' ' for c in text).split())


def build_word_bigrams(text, top_n):
    words = text.split()
    word_counts = Counter(words)
    vocab = [w for w, _ in word_counts.most_common(top_n)]
    word2idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    transitions = Counter()
    for i in range(len(words) - 1):
        if words[i] in word2idx and words[i+1] in word2idx:
            transitions[(words[i], words[i+1])] += 1
    targets = np.zeros(V, dtype=np.int32)
    for i, w in enumerate(vocab):
        nexts = {w2: cnt for (w1, w2), cnt in transitions.items() if w1 == w}
        if nexts:
            targets[i] = word2idx[max(nexts, key=nexts.get)]
    return vocab, targets


def forward_divnorm(net, ticks=8, alpha=ALPHA):
    V, N = net.V, net.N
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        raw = acts @ net.mask
        charges += raw
        total = np.abs(acts).sum(axis=1, keepdims=True) + 1e-6
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        acts /= (1.0 + alpha * total)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def evaluate(net, targets, ticks=8):
    logits = forward_divnorm(net, ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    V = net.V
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return float(acc), float(0.5 * acc + 0.5 * tp)


def burst_add(net, target_density_pct=4):
    """Quickly add connections to reach target density."""
    N = net.N
    target_conns = int(N * N * target_density_pct / 100)
    added = 0
    while net.count_connections() < target_conns:
        r, c = random.randint(0, N-1), random.randint(0, N-1)
        if r != c and net.mask[r, c] == 0:
            net.mask[r, c] = net.DRIVE if random.randint(0, 1) else -net.DRIVE
            net.alive.append((r, c))
            net.alive_set.add((r, c))
            added += 1
        if added > target_conns * 2:
            break
    return added


def wake_phase(net, targets, budget, ticks=8, add_prob=0.15):
    """WAKE: mostly rewire + sparse add, 1 connection per step."""
    acc, score = evaluate(net, targets, ticks)
    best_acc = acc
    best_score = score
    stale = 0
    improved = 0

    for step in range(budget):
        # Choose operation: 85% rewire, 15% add
        undo = []
        if random.random() < add_prob:
            net._add(undo)
        else:
            net._rewire(undo)

        new_acc, new_score = evaluate(net, targets, ticks)
        if new_score > score:
            score = new_score
            acc = new_acc
            best_acc = max(best_acc, acc)
            best_score = max(best_score, score)
            stale = 0
            improved += 1
        else:
            net.replay(undo)
            stale += 1

        if stale >= budget // 3:
            break

    return best_acc, best_score, step + 1, improved


def sleep_random(net, targets, ticks=8, max_attempts=5000):
    """SLEEP crystallize: remove RANDOM connections, keep if score doesn't drop."""
    acc, score = evaluate(net, targets, ticks)
    removed = 0
    stale = 0

    for step in range(max_attempts):
        if not net.alive:
            break

        # Pick random connection
        idx = random.randint(0, len(net.alive) - 1)
        r, c = net.alive[idx]
        old_val = net.mask[r, c]

        # Remove it
        net.mask[r, c] = 0
        net.alive[idx] = net.alive[-1]
        net.alive.pop()
        net.alive_set.discard((r, c))

        new_acc, new_score = evaluate(net, targets, ticks)
        if new_score >= score:
            # Keep removed
            score = new_score
            acc = new_acc
            removed += 1
            stale = 0
        else:
            # Restore
            net.mask[r, c] = old_val
            net.alive.append((r, c))
            net.alive_set.add((r, c))
            stale += 1

        if stale >= max_attempts // 3:
            break

    return removed, acc, score, step + 1


def sleep_weakest(net, targets, ticks=8, max_attempts=5000):
    """SLEEP crystallize: remove WEAKEST connections first."""
    acc, score = evaluate(net, targets, ticks)
    removed = 0
    stale = 0

    # Sort by absolute weight (weakest first)
    edges = sorted(net.alive, key=lambda rc: abs(net.mask[rc[0], rc[1]]))

    for r, c in edges:
        if stale >= max_attempts // 3:
            break
        if net.mask[r, c] == 0:
            continue

        old_val = net.mask[r, c]
        net.mask[r, c] = 0
        net.alive_set.discard((r, c))

        new_acc, new_score = evaluate(net, targets, ticks)
        if new_score >= score:
            score = new_score
            acc = new_acc
            removed += 1
            stale = 0
        else:
            net.mask[r, c] = old_val
            net.alive_set.add((r, c))
            stale += 1

    net.resync_alive()
    return removed, acc, score, len(edges)


# ============================================================
# Main
# ============================================================

text = load_and_clean('pride_prejudice.txt') + ' ' + load_and_clean('frankenstein.txt')
V = 128
vocab, targets = build_word_bigrams(text, V)

print(f"WAKE/SLEEP CYCLE | V={V} N={V*3}")
print(f"Task: word bigram prediction ({V} words)")
print(f"{'='*70}")

CONFIGS = [
    ('random_remove', sleep_random),
    ('weakest_remove', sleep_weakest),
]

N_CYCLES = 5
WAKE_BUDGET = 8000
SEED = 42

for config_name, sleep_fn in CONFIGS:
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*70}")

    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(V)

    # Burst add to ~4% density
    added = burst_add(net, 4)
    init_acc, init_score = evaluate(net, targets)
    print(f"Init: {init_acc*100:.1f}% acc, {net.count_connections()} conns (burst added {added})")

    total_time = 0
    for cycle in range(N_CYCLES):
        print(f"\n  --- Cycle {cycle+1}/{N_CYCLES} ---")

        # WAKE
        t0 = time.time()
        w_acc, w_score, w_steps, w_improved = wake_phase(
            net, targets, WAKE_BUDGET)
        w_time = time.time() - t0
        print(f"  WAKE:  acc={w_acc*100:.1f}% score={w_score*100:.1f}% "
              f"steps={w_steps} improved={w_improved} "
              f"conns={net.count_connections()} ({w_time:.0f}s)")

        # SLEEP
        t0 = time.time()
        s_removed, s_acc, s_score, s_steps = sleep_fn(net, targets)
        s_time = time.time() - t0
        print(f"  SLEEP: removed={s_removed} acc={s_acc*100:.1f}% "
              f"score={s_score*100:.1f}% conns={net.count_connections()} ({s_time:.0f}s)")

        total_time += w_time + s_time

    final_acc, final_score = evaluate(net, targets)
    print(f"\n  FINAL: acc={final_acc*100:.1f}% score={final_score*100:.1f}% "
          f"conns={net.count_connections()} total_time={total_time:.0f}s")

print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")
