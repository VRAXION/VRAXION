"""Diff-guided mutation: diff scouts, mutation acts, eval confirms.
Phase 1: Stream N chars, accumulate error map (cheap)
Phase 2: Targeted mutation based on error map
Phase 3: Eval same N chars, keep/revert (expensive, 1x)
Repeat."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.graph import SelfWiringGraph
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ALPHA = 2.0

def load_and_clean(fn):
    with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    return ' '.join(''.join(c if 'a' <= c <= 'z' or c == ' ' else ' ' for c in text).split())


def forward_seq(net, world, ticks=8, alpha=ALPHA):
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


def eval_seq(net, text, char2idx, ticks=8, alpha=ALPHA):
    """Eval accuracy on a fixed text segment. Returns (accuracy, error_map).
    error_map[j] = average diff for output neuron j (+ means needs more, - means too much)."""
    V = net.V; N = net.N
    net.reset()
    correct = 0
    error_map = np.zeros(V, dtype=np.float32)
    # Track which neurons are active for each output neuron's errors
    active_near_output = np.zeros((V, N), dtype=np.float32)

    for i in range(len(text) - 1):
        world = np.zeros(V, dtype=np.float32)
        world[char2idx[text[i]]] = 1.0
        output, act = forward_seq(net, world, ticks, alpha)

        pred = np.argmax(output)
        actual = char2idx[text[i + 1]]
        if pred == actual:
            correct += 1

        # Accumulate error map
        target = np.zeros(V, dtype=np.float32)
        target[actual] = 1.0
        e_out = np.exp(output - output.max())
        probs = e_out / e_out.sum()
        diff = target - probs
        error_map += diff

        # Track which neurons are active when each output is wrong
        for j in range(V):
            if abs(diff[j]) > 0.01:
                active_near_output[j] += act * diff[j]

    n = len(text) - 1
    return correct / n, error_map / n, active_near_output / n


def targeted_mutation(net, error_map, active_map, n_changes=5):
    """Mutate based on error map: add/strengthen connections toward needy outputs."""
    V = net.V; N = net.N
    out_start = net.out_start
    undo = []

    # Find output neurons with biggest errors
    worst_outputs = np.argsort(-np.abs(error_map))[:n_changes]

    for j in worst_outputs:
        out_neuron = out_start + j
        direction = np.sign(error_map[j])  # +1 if needs more, -1 if too much

        # Find best source neuron: most active when this output was wrong
        source_scores = active_map[j].copy()
        source_scores[out_neuron] = 0  # no self-loop

        if direction > 0:
            # Need MORE activation at output j
            # Add/strengthen connection from most relevant neuron
            best_source = np.argmax(np.abs(source_scores))
            if net.mask[best_source, out_neuron] == 0:
                # Add new connection
                net.mask[best_source, out_neuron] = net.DRIVE * direction
                net.alive.append((best_source, out_neuron))
                net.alive_set.add((best_source, out_neuron))
                undo.append(('A', best_source, out_neuron))
            else:
                # Strengthen existing
                old = net.mask[best_source, out_neuron]
                net.mask[best_source, out_neuron] = np.clip(old + 0.3 * direction, -3.0, 3.0)
                undo.append(('S', best_source, out_neuron, old))
        else:
            # Need LESS activation at output j
            # Weaken or remove strongest connection to this output
            conns_to_j = [(abs(net.mask[r, out_neuron]), r)
                          for r, c in net.alive if c == out_neuron]
            if conns_to_j:
                _, worst_source = max(conns_to_j)
                old = net.mask[worst_source, out_neuron]
                new_val = old + 0.3 * direction  # weaken
                if abs(new_val) < 0.1:
                    # Remove
                    net.mask[worst_source, out_neuron] = 0
                    net.alive_set.discard((worst_source, out_neuron))
                    undo.append(('R', worst_source, out_neuron, old))
                else:
                    net.mask[worst_source, out_neuron] = np.clip(new_val, -3.0, 3.0)
                    undo.append(('S', worst_source, out_neuron, old))

    # Also: random rewire for exploration (2 attempts)
    for _ in range(2):
        rundo = []
        net._rewire(rundo)
        undo.extend(rundo)

    return undo


def replay_undo(net, undo):
    """Revert targeted mutations."""
    for entry in reversed(undo):
        op = entry[0]
        if op == 'A':
            r, c = entry[1], entry[2]
            net.mask[r, c] = 0
            net.alive_set.discard((r, c))
        elif op == 'R':
            r, c, old_val = entry[1], entry[2], entry[3]
            net.mask[r, c] = old_val
            net.alive_set.add((r, c))
        elif op == 'S':
            r, c, old_val = entry[1], entry[2], entry[3]
            net.mask[r, c] = old_val
        elif op == 'W':
            _, r, c_old, c_new = entry
            sign = net.mask[r, c_new]
            net.mask[r, c_new] = 0
            net.mask[r, c_old] = sign
            net.alive_set.discard((r, c_new))
            net.alive_set.add((r, c_old))
    net.resync_alive()


def train_diff_guided(net, train_text, char2idx, eval_len=200,
                      n_cycles=500, n_changes=5, ticks=8, log_every=50):
    """Main loop: diff scout → targeted mutation → eval confirm."""
    eval_text = train_text[:eval_len + 1]

    # Initial eval
    score, error_map, active_map = eval_seq(net, eval_text, char2idx, ticks)
    best = score

    improved = 0
    trajectory = [(0, float(best))]

    for cycle in range(n_cycles):
        # Save state for revert
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)

        # Targeted mutation based on error map
        undo = targeted_mutation(net, error_map, active_map, n_changes)

        # Eval on same text
        new_score, new_error_map, new_active_map = eval_seq(
            net, eval_text, char2idx, ticks)

        if new_score > score:
            score = new_score
            best = max(best, score)
            error_map = new_error_map
            active_map = new_active_map
            improved += 1
        else:
            replay_undo(net, undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)

        if (cycle + 1) % log_every == 0:
            trajectory.append((cycle + 1, float(best)))
            top_err = np.argsort(-np.abs(error_map))[:3]
            print(f"  [{cycle+1:4d}] best={best*100:.1f}% improved={improved} "
                  f"conns={net.count_connections()} "
                  f"top_err={[f'{error_map[j]:+.2f}' for j in top_err]}",
                  flush=True)

    return float(best), improved, trajectory


# ============================================================
# Main
# ============================================================

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

print(f"DIFF-GUIDED MUTATION | V={V} N={V*3}")
print(f"Bigram: {bigram_test*100:.1f}% | Frequency: ~18.5% | Random: {100/V:.1f}%")

# Test different eval lengths and mutation sizes
CONFIGS = [
    ('eval100_m3',  100,  3, 500),
    ('eval100_m5',  100,  5, 500),
    ('eval200_m5',  200,  5, 500),
    ('eval200_m10', 200, 10, 500),
    ('eval500_m5',  500,  5, 300),
]

print(f"\n{'config':<16s} {'train':>7s} {'test':>7s} {'vs_bi':>7s} {'impr':>5s} {'conns':>6s} {'time':>5s}")
print("-" * 60)

for name, eval_len, n_changes, n_cycles in CONFIGS:
    np.random.seed(42)
    net = SelfWiringGraph(V)

    print(f"\n--- {name} ---")
    t0 = time.time()
    best, improved, traj = train_diff_guided(
        net, train_text, char2idx, eval_len, n_cycles, n_changes)
    elapsed = time.time() - t0

    # Test on Alice
    test_score, _, _ = eval_seq(net, test_text[:500], char2idx)

    diff_bi = test_score - bigram_test
    marker = " <<<" if test_score > bigram_test else ""
    print(f"{name:<16s} {best*100:6.1f}% {test_score*100:6.1f}% {diff_bi*100:+6.1f}pp "
          f"{improved:5d} {net.count_connections():6d} {elapsed:4.0f}s{marker}")
