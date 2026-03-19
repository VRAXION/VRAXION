"""Streaming byte-by-byte learning with three-factor update rule.
No mutation+selection cycle. No global eval. Just stream text,
predict next char, update connections based on right/wrong.
Δw = pre × post × error_signal (Hebbian if correct, anti-Hebbian if wrong)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.graph import SelfWiringGraph

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ALPHA = 2.0


def load_and_clean(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    clean = ''.join(c if (c >= 'a' and c <= 'z') or c == ' ' else ' ' for c in text)
    return ' '.join(clean.split())


def forward_divnorm_seq(net, world, ticks=8, alpha=ALPHA):
    """Stateful forward (optimized), returns (output, final_activations)."""
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
    output = net.charge[net.out_start:net.out_start + net.V]
    return output, act


def stream_learn(net, text, char2idx, chars, lr=0.01, ticks=8,
                 max_chars=None, log_every=1000):
    """Stream text byte-by-byte with diff-as-input learning.
    Two-phase per char:
      Phase 1: feed char → forward → predict
      Phase 2: feed diff (target-output) as input → forward → Hebbian update
    The network processes its own error through its own wiring."""
    V = net.V
    N = net.N
    net.reset()

    if max_chars is None:
        max_chars = len(text) - 1

    correct = 0
    total = 0
    recent_correct = 0
    recent_total = 0
    log = []
    out_start = net.out_start

    for i in range(min(len(text) - 1, max_chars)):
        # === PHASE 1: Normal input → prediction ===
        world = np.zeros(V, dtype=np.float32)
        world[char2idx[text[i]]] = 1.0
        output, act_phase1 = forward_divnorm_seq(net, world, ticks // 2)

        # Predict next char
        pred = np.argmax(output)
        actual = char2idx[text[i + 1]]
        is_correct = (pred == actual)

        if is_correct:
            correct += 1
            recent_correct += 1
        total += 1
        recent_total += 1

        # Compute diff: target - softmax(output)
        target = np.zeros(V, dtype=np.float32)
        target[actual] = 1.0
        e_out = np.exp(output - output.max())
        probs = e_out / e_out.sum()
        diff = target - probs  # (V,) — positive AND negative values

        # === PHASE 2: Feed diff as input → network processes error ===
        # diff goes on same input neurons — network sees negative values = "error signal"
        output2, act_phase2 = forward_divnorm_seq(net, diff, ticks // 2)

        # === LEARNING: Hebbian on phase 2 activations ===
        # Neurons active after error processing → their connections get updated
        active_idx = np.where(act_phase2 > 0.01)[0]

        if len(active_idx) > 0 and len(active_idx) < N // 2:
            act_a = act_phase2[active_idx]

            # Update connections TO output neurons using diff direction
            out_idx = np.arange(out_start, out_start + V)
            delta_out = np.outer(act_a, diff) * lr
            sub_out = net.mask[np.ix_(active_idx, out_idx)]
            wh_out = sub_out != 0
            if wh_out.any():
                sub_out[wh_out] += delta_out[wh_out]
                np.clip(sub_out, -3.0, 3.0, out=sub_out)
                net.mask[np.ix_(active_idx, out_idx)] = sub_out

            # Strengthen connections between error-active neurons (Hebbian)
            if len(active_idx) > 1:
                inner = np.outer(act_a, act_a) * (lr * 0.05)
                sub_in = net.mask[np.ix_(active_idx, active_idx)]
                wh_in = sub_in != 0
                if wh_in.any():
                    sub_in[wh_in] += inner[wh_in]
                    np.clip(sub_in, -3.0, 3.0, out=sub_in)
                    net.mask[np.ix_(active_idx, active_idx)] = sub_in

            # Exploration: add connection toward correct output
            if not is_correct and np.random.random() < 0.05:
                src = np.random.choice(active_idx)
                dst = out_start + actual
                if src != dst and net.mask[src, dst] == 0:
                    net.mask[src, dst] = net.DRIVE
                    net.alive.append((src, dst))
                    net.alive_set.add((src, dst))

        # Logging
        if (i + 1) % log_every == 0:
            recent_acc = recent_correct / max(recent_total, 1)
            overall_acc = correct / total
            conns = len(net.alive)
            log.append((i + 1, overall_acc, recent_acc, conns))
            print(f"  [{i+1:7d}] overall={overall_acc*100:.1f}% "
                  f"recent={recent_acc*100:.1f}% conns={conns}",
                  flush=True)
            recent_correct = 0
            recent_total = 0

    overall_acc = correct / max(total, 1)
    return overall_acc, log


# ============================================================
# Main
# ============================================================

train_text = load_and_clean('pride_prejudice.txt')
test_text = load_and_clean('alice.txt')
chars = sorted(set(train_text + test_text))
char2idx = {c: i for i, c in enumerate(chars)}
V = len(chars)

# Bigram baseline
from collections import Counter
transitions = Counter()
for i in range(len(train_text) - 1):
    transitions[(train_text[i], train_text[i+1])] += 1
best_next = {}
for c in chars:
    nexts = {c2: cnt for (c1, c2), cnt in transitions.items() if c1 == c}
    if nexts: best_next[c] = max(nexts, key=nexts.get)
    else: best_next[c] = ' '
bigram_train = sum(1 for i in range(len(train_text)-1)
                   if best_next.get(train_text[i]) == train_text[i+1]) / (len(train_text)-1)
bigram_test = sum(1 for i in range(len(test_text)-1)
                  if best_next.get(test_text[i]) == test_text[i+1]) / (len(test_text)-1)

print(f"STREAMING BYTE-BY-BYTE LEARNING")
print(f"V={V} N={V*3} | Train: {len(train_text)} chars | Test: {len(test_text)} chars")
print(f"Bigram baseline: train={bigram_train*100:.1f}% test={bigram_test*100:.1f}%")
print(f"Frequency baseline: ~18.5%")
print(f"Random baseline: {100/V:.1f}%")

# ============================================================
# Crystallize: remove weak connections, strict monotonic
# ============================================================

def crystallize(net, eval_text, char2idx, ticks=8, max_attempts=2000):
    """Remove connections one by one. Keep only if accuracy doesn't drop."""
    def quick_eval():
        net.reset()
        correct = 0
        V = net.V
        length = min(len(eval_text) - 1, 200)
        for i in range(length):
            world = np.zeros(V, dtype=np.float32)
            world[char2idx[eval_text[i]]] = 1.0
            output, _ = forward_divnorm_seq(net, world, ticks // 2)
            # Also do diff phase to be consistent
            target = np.zeros(V, dtype=np.float32)
            target[char2idx[eval_text[i+1]]] = 1.0
            e_out = np.exp(output - output.max())
            probs = e_out / e_out.sum()
            diff = target - probs
            output2, _ = forward_divnorm_seq(net, diff, ticks // 2)
            if np.argmax(output) == char2idx[eval_text[i+1]]:
                correct += 1
        return correct / length

    score = quick_eval()
    removed = 0
    stale = 0

    # Sort connections by absolute weight (weakest first)
    edges = [(abs(net.mask[r, c]), r, c) for r, c in net.alive]
    edges.sort()  # weakest first

    for strength, r, c in edges:
        if stale >= max_attempts:
            break
        # Try removing
        old_val = net.mask[r, c]
        net.mask[r, c] = 0
        net.alive_set.discard((r, c))

        new_score = quick_eval()
        if new_score >= score:
            # Keep removed — doesn't hurt
            removed += 1
            score = new_score
            stale = 0
        else:
            # Restore
            net.mask[r, c] = old_val
            net.alive_set.add((r, c))
            stale += 1

    # Rebuild alive list
    net.resync_alive()
    return removed, score


# ============================================================
# Rewire: random structural exploration
# ============================================================

def rewire_phase(net, eval_text, char2idx, ticks=8, attempts=500):
    """Random rewire attempts. Keep if accuracy improves."""
    def quick_eval():
        net.reset()
        correct = 0
        V = net.V
        length = min(len(eval_text) - 1, 200)
        for i in range(length):
            world = np.zeros(V, dtype=np.float32)
            world[char2idx[eval_text[i]]] = 1.0
            output, _ = forward_divnorm_seq(net, world, ticks // 2)
            target = np.zeros(V, dtype=np.float32)
            target[char2idx[eval_text[i+1]]] = 1.0
            e_out = np.exp(output - output.max())
            probs = e_out / e_out.sum()
            diff = target - probs
            output2, _ = forward_divnorm_seq(net, diff, ticks // 2)
            if np.argmax(output) == char2idx[eval_text[i+1]]:
                correct += 1
        return correct / length

    score = quick_eval()
    improved = 0
    for _ in range(attempts):
        undo = []
        net._rewire(undo)
        new_score = quick_eval()
        if new_score > score:
            score = new_score
            improved += 1
        else:
            net.replay(undo)
    return improved, score


# ============================================================
# Combined: stream → crystallize → rewire → repeat
# ============================================================

def train_combined(net, train_text, char2idx, chars, lr=0.1, ticks=8,
                   stream_chars=10000, n_cycles=5, log_every=5000):
    """Combined training: streaming Hebbian + crystallize + rewire."""
    V = net.V
    eval_text = train_text[:500]  # fixed eval segment for crystallize/rewire
    pos = 0  # position in text

    for cycle in range(n_cycles):
        print(f"\n  === CYCLE {cycle+1}/{n_cycles} ===")

        # Phase 1: Streaming learning
        end_pos = min(pos + stream_chars, len(train_text) - 1)
        chunk = train_text[pos:end_pos + 1]
        acc, log = stream_learn(net, chunk, char2idx, chars,
                                lr=lr, ticks=ticks, log_every=log_every)
        pos = end_pos
        if pos >= len(train_text) - 2:
            pos = 0  # wrap around
        print(f"  Stream: {acc*100:.1f}% conns={net.count_connections()}")

        # Phase 2: Crystallize (prune weak connections)
        removed, crystal_score = crystallize(net, eval_text, char2idx, ticks)
        print(f"  Crystallize: removed {removed}, score={crystal_score*100:.1f}%, "
              f"conns={net.count_connections()}")

        # Phase 3: Rewire (explore)
        improved, rewire_score = rewire_phase(net, eval_text, char2idx, ticks, attempts=200)
        print(f"  Rewire: {improved} improvements, score={rewire_score*100:.1f}%, "
              f"conns={net.count_connections()}")


print("=" * 60)
print("COMBINED: Streaming + Crystallize + Rewire")
print("=" * 60)

np.random.seed(42)
net = SelfWiringGraph(V)

# Run combined training: 5 cycles of stream→crystallize→rewire
train_combined(net, train_text, char2idx, chars,
               lr=0.1, ticks=8, stream_chars=10000, n_cycles=5)

# Final test on Alice (no learning)
print(f"\n--- FINAL TEST (Alice, 10k chars, no learning) ---")
test_acc, test_log = stream_learn(
    net, test_text, char2idx, chars,
    lr=0, max_chars=10000, log_every=5000)

print(f"\nFinal: test={test_acc*100:.1f}% (bigram: {bigram_test*100:.1f}%) "
      f"conns={net.count_connections()}")
