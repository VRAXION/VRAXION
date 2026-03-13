"""
VRAXION v22 — "Nem tudom" / Abstain Mechanism Test
====================================================
Tests whether a self-wiring graph network can learn WHEN to answer
vs when to abstain, using an extra output neuron.

Key insight from introspection research: the human brain actively
signals "I don't know" rather than guessing. The current system is
FORCED to answer every input, which produces bad predictions at low
confidence.

Architecture:
  - V+1 output neurons: V class neurons + 1 abstain neuron
  - Shared I/O: first V+1 neurons are input/output
  - Input is one-hot V (class neurons only, abstain gets no direct input)
  - Output is softmax over V+1 logits

Scoring:
  - Correct answer:   +1
  - Wrong answer:     -1 (punishment!)
  - Abstain:           0 (neutral)
  - Score = (correct - incorrect) / total

Tests:
  A) 16-class lookup: forced vs abstain vs strong penalty
  B) 32-class lookup: where abstain matters more
  C) English bigram: where confidence varies naturally
"""

import numpy as np
import random
import time
import json
from datetime import datetime


class SelfWiringGraph:
    """Self-wiring graph with configurable output size for abstain."""

    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30,
                 n_outputs=None):
        self.N = n_neurons
        self.V = vocab
        self.n_outputs = n_outputs if n_outputs is not None else vocab
        self.flip_rate = flip_rate
        self.last_score = 0.0

        # Ternary mask
        r = np.random.rand(n_neurons, n_neurons)
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        self.mask[r < density / 2] = -1
        self.mask[r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        # Binary weights
        self.W = np.where(
            np.random.rand(n_neurons, n_neurons) > 0.5,
            np.float32(0.5), np.float32(1.5)
        )

        # 4D addresses for self-wiring
        self.addr = np.random.randn(n_neurons, 4).astype(np.float32)
        self.addr[:self.n_outputs, 3] = 0.0
        self.addr[self.n_outputs:, 3] = 0.5
        self.target_W = np.random.randn(n_neurons, 4).astype(np.float32) * 0.1

        # Persistent state
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.decay = 0.5

    def reset(self):
        self.state *= 0

    def forward(self, world, ticks=6):
        """Forward pass. Input is V-sized one-hot, output is n_outputs-sized."""
        act = self.state.copy()
        Weff = self.W * self.mask

        for t in range(ticks):
            act = act * self.decay
            if t == 0:
                # Inject input into first V neurons (NOT abstain neuron)
                act[:self.V] = world
            raw = act @ Weff + act * 0.1
            act = np.where(raw > 0, raw, np.float32(0.01) * raw)
            np.clip(act, -10.0, 10.0, out=act)

        self.state = act.copy()
        return act[:self.n_outputs]

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

    def save_state(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(),
                self.addr.copy(), self.target_W.copy())

    def restore_state(self, s):
        self.W = s[0].copy()
        self.mask = s[1].copy()
        self.state = s[2].copy()
        self.addr = s[3].copy()
        self.target_W = s[4].copy()

    def mutate_structure(self, rate=0.05):
        r = random.random()
        if r < self.flip_rate:
            alive = np.argwhere(self.mask != 0)
            if len(alive) > 0:
                n = max(1, int(len(alive) * rate * 0.5))
                idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                for j in range(len(idx)):
                    r2, c = int(idx[j][0]), int(idx[j][1])
                    self.mask[r2, c] *= -1
        else:
            action = random.choice(['add_pos', 'add_neg', 'remove', 'rewire'])
            if action in ('add_pos', 'add_neg'):
                dead = np.argwhere(self.mask == 0)
                dead = dead[dead[:, 0] != dead[:, 1]]
                if len(dead) > 0:
                    n = max(1, int(len(dead) * rate))
                    idx = dead[np.random.choice(len(dead), min(n, len(dead)), replace=False)]
                    sign = 1.0 if action == 'add_pos' else -1.0
                    for j in range(len(idx)):
                        r2, c = int(idx[j][0]), int(idx[j][1])
                        self.mask[r2, c] = sign
                        self.W[r2, c] = random.choice([np.float32(0.5), np.float32(1.5)])
            elif action == 'remove':
                alive = np.argwhere(self.mask != 0)
                if len(alive) > 3:
                    n = max(1, int(len(alive) * rate))
                    idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                    for j in range(len(idx)):
                        self.mask[int(idx[j][0]), int(idx[j][1])] = 0
            else:
                alive = np.argwhere(self.mask != 0)
                if len(alive) > 0:
                    n = max(1, int(len(alive) * rate))
                    idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                    for j in range(len(idx)):
                        r2, c = int(idx[j][0]), int(idx[j][1])
                        old_sign = self.mask[r2, c]
                        old_w = self.W[r2, c]
                        self.mask[r2, c] = 0
                        nc = random.randint(0, self.N - 1)
                        while nc == r2:
                            nc = random.randint(0, self.N - 1)
                        self.mask[r2, nc] = old_sign
                        self.W[r2, nc] = old_w

    def mutate_weights(self):
        alive = np.argwhere(self.mask != 0)
        if len(alive) > 0:
            n = max(1, int(len(alive) * 0.05))
            idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
            for j in range(len(idx)):
                r2, c = int(idx[j][0]), int(idx[j][1])
                self.W[r2, c] = np.float32(1.5) if self.W[r2, c] < 1.0 else np.float32(0.5)

    def self_wire(self):
        if self.last_score < 0.3:
            top_k, max_new = 2, 1
        elif self.last_score < 0.7:
            top_k, max_new = 3, 2
        else:
            top_k, max_new = 5, 3
        act = self.state
        a2 = np.abs(act[self.n_outputs:])
        if a2.sum() < 0.01:
            return
        nc = min(top_k, len(a2))
        top = np.argpartition(a2, -nc)[-nc:] + self.n_outputs
        new = 0
        for ni in top:
            ni = int(ni)
            if np.abs(act[ni]) < 0.1:
                continue
            tgt = self.addr[ni] + np.abs(act[ni]) * self.target_W[ni]
            d = ((self.addr - tgt) ** 2).sum(axis=1)
            d[ni] = float('inf')
            near = int(np.argmin(d))
            if self.mask[ni, near] == 0:
                self.mask[ni, near] = random.choice([-1.0, 1.0])
                self.W[ni, near] = random.choice([np.float32(0.5), np.float32(1.5)])
                new += 1
            if new >= max_new:
                break


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ============================================================
# Evaluation functions
# ============================================================

def evaluate_forced(net, inputs, targets, vocab, ticks=6):
    """Standard forced-choice evaluation (no abstain)."""
    net.reset()
    correct = 0
    total = len(inputs)
    for p in range(2):  # 2 passes for state buildup
        for i in range(total):
            world = np.zeros(vocab, dtype=np.float32)
            world[inputs[i]] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits[:vocab])
            if p == 1 and np.argmax(probs) == targets[i]:
                correct += 1
    acc = correct / total
    return {
        'score': acc,  # for forced choice, score = accuracy
        'accuracy': acc,
        'precision': acc,
        'abstain_rate': 0.0,
        'correct': correct,
        'incorrect': total - correct,
        'abstained': 0,
    }


def evaluate_abstain(net, inputs, targets, vocab, ticks=6, penalty=-1):
    """Evaluation with abstain option. Output is V+1 (V classes + 1 abstain)."""
    net.reset()
    correct = 0
    incorrect = 0
    abstained = 0
    total = len(inputs)

    for p in range(2):  # 2 passes for state buildup
        if p == 0:
            # Warmup pass — just run forward, don't score
            for i in range(total):
                world = np.zeros(vocab, dtype=np.float32)
                world[inputs[i]] = 1.0
                net.forward(world, ticks)
            continue

        for i in range(total):
            world = np.zeros(vocab, dtype=np.float32)
            world[inputs[i]] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits)  # V+1 probabilities
            pred = np.argmax(probs)

            if pred == vocab:  # abstain neuron won
                abstained += 1
            elif pred == targets[i]:
                correct += 1
            else:
                incorrect += 1

    answered = total - abstained
    precision = correct / max(1, answered)
    score = (correct + penalty * incorrect) / total

    return {
        'score': score,
        'precision': precision,
        'abstain_rate': abstained / total,
        'correct': correct,
        'incorrect': incorrect,
        'abstained': abstained,
    }


# ============================================================
# Training loops
# ============================================================

def train_forced(net, inputs, targets, vocab, max_attempts=8000,
                 ticks=6, label=""):
    """Train with forced choice (standard accuracy)."""

    def evaluate():
        r = evaluate_forced(net, inputs, targets, vocab, ticks)
        net.last_score = r['score']
        return r['score'], r

    score, stats = evaluate()
    best = score
    best_stats = stats
    phase = "STRUCTURE"
    kept = 0
    stale = 0
    switched = False

    for att in range(max_attempts):
        state = net.save_state()
        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        new_score, new_stats = evaluate()
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            if score > best:
                best = score
                best_stats = new_stats
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        if (att + 1) % 2000 == 0:
            s = best_stats
            print(f"  [{att+1:5d}] {label:25s} | Score: {best:+6.3f} | "
                  f"Acc: {s['accuracy']*100:5.1f}% | "
                  f"Conns: {net.count_connections():4d} | Kept: {kept:3d}")

        if best >= 0.99:
            break
        if stale >= 6000:
            break

    return best, best_stats, kept


def train_abstain(net, inputs, targets, vocab, max_attempts=8000,
                  ticks=6, penalty=-1, label=""):
    """Train with abstain scoring."""

    def evaluate():
        r = evaluate_abstain(net, inputs, targets, vocab, ticks, penalty)
        net.last_score = max(0, r['score'])  # for self-wire arousal
        return r['score'], r

    score, stats = evaluate()
    best = score
    best_stats = stats
    phase = "STRUCTURE"
    kept = 0
    stale = 0
    switched = False

    for att in range(max_attempts):
        state = net.save_state()
        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        new_score, new_stats = evaluate()
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            if score > best:
                best = score
                best_stats = new_stats
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        if (att + 1) % 2000 == 0:
            s = best_stats
            ans = s['correct'] + s['incorrect']
            print(f"  [{att+1:5d}] {label:25s} | Score: {best:+6.3f} | "
                  f"Prec: {s['precision']*100:5.1f}% | "
                  f"Abstain: {s['abstain_rate']*100:4.1f}% "
                  f"({s['abstained']}/{s['correct']+s['incorrect']+s['abstained']}) | "
                  f"C/I: {s['correct']}/{s['incorrect']} | "
                  f"Conns: {net.count_connections():4d} | Kept: {kept:3d}")

        if best >= 0.99:
            break
        if stale >= 6000:
            break

    return best, best_stats, kept


# ============================================================
# Test A: 16-class lookup
# ============================================================

def test_16class(seed=42):
    V = 16
    N = 80
    MAX_ATTEMPTS = 16000

    print("\n" + "#" * 70)
    print("#  TEST A: 16-class lookup — forced vs abstain")
    print(f"#  {N} neurons, {MAX_ATTEMPTS} attempts, seed={seed}")
    print("#" * 70)

    results = {}

    # A1: Forced choice baseline
    print(f"\n{'='*60}")
    print(f"  A1: Forced choice (V={V} outputs, standard accuracy)")
    print(f"{'='*60}")
    np.random.seed(seed)
    random.seed(seed)
    perm = np.random.permutation(V)
    inputs = list(range(V))

    net = SelfWiringGraph(N, V, n_outputs=V)
    t0 = time.time()
    best, stats, kept = train_forced(net, inputs, perm, V,
                                     max_attempts=MAX_ATTEMPTS,
                                     label="forced_choice")
    elapsed = time.time() - t0
    results['forced'] = {**stats, 'time': round(elapsed, 1), 'kept': kept,
                         'connections': net.count_connections()}
    print(f"\n  FINAL forced: score={best:+.3f} acc={stats['accuracy']*100:.1f}% "
          f"time={elapsed:.1f}s")

    # A2: Abstain (penalty=-1)
    print(f"\n{'='*60}")
    print(f"  A2: Abstain (V+1={V+1} outputs, penalty=-1)")
    print(f"{'='*60}")
    np.random.seed(seed)
    random.seed(seed)
    perm = np.random.permutation(V)

    net = SelfWiringGraph(N + 1, V, n_outputs=V + 1)  # +1 neuron for abstain
    t0 = time.time()
    best, stats, kept = train_abstain(net, inputs, perm, V,
                                      max_attempts=MAX_ATTEMPTS,
                                      penalty=-1, label="abstain_p1")
    elapsed = time.time() - t0
    results['abstain_p1'] = {**stats, 'time': round(elapsed, 1), 'kept': kept,
                             'connections': net.count_connections()}
    print(f"\n  FINAL abstain_p1: score={best:+.3f} prec={stats['precision']*100:.1f}% "
          f"abstain={stats['abstain_rate']*100:.1f}% time={elapsed:.1f}s")

    # A3: Abstain (penalty=-2, stronger punishment)
    print(f"\n{'='*60}")
    print(f"  A3: Abstain (V+1={V+1} outputs, penalty=-2)")
    print(f"{'='*60}")
    np.random.seed(seed)
    random.seed(seed)
    perm = np.random.permutation(V)

    net = SelfWiringGraph(N + 1, V, n_outputs=V + 1)
    t0 = time.time()
    best, stats, kept = train_abstain(net, inputs, perm, V,
                                      max_attempts=MAX_ATTEMPTS,
                                      penalty=-2, label="abstain_p2")
    elapsed = time.time() - t0
    results['abstain_p2'] = {**stats, 'time': round(elapsed, 1), 'kept': kept,
                             'connections': net.count_connections()}
    print(f"\n  FINAL abstain_p2: score={best:+.3f} prec={stats['precision']*100:.1f}% "
          f"abstain={stats['abstain_rate']*100:.1f}% time={elapsed:.1f}s")

    return results


# ============================================================
# Test B: 32-class lookup
# ============================================================

def test_32class(seed=42):
    V = 32
    N = 160
    MAX_ATTEMPTS = 20000

    print("\n" + "#" * 70)
    print("#  TEST B: 32-class lookup — where abstain matters more")
    print(f"#  {N} neurons, {MAX_ATTEMPTS} attempts, seed={seed}")
    print("#" * 70)

    results = {}

    # B1: Forced choice
    print(f"\n{'='*60}")
    print(f"  B1: Forced choice (V={V})")
    print(f"{'='*60}")
    np.random.seed(seed)
    random.seed(seed)
    perm = np.random.permutation(V)
    inputs = list(range(V))

    net = SelfWiringGraph(N, V, n_outputs=V)
    t0 = time.time()
    best, stats, kept = train_forced(net, inputs, perm, V,
                                     max_attempts=MAX_ATTEMPTS,
                                     label="forced_32")
    elapsed = time.time() - t0
    results['forced'] = {**stats, 'time': round(elapsed, 1), 'kept': kept,
                         'connections': net.count_connections()}
    print(f"\n  FINAL forced: score={best:+.3f} acc={stats['accuracy']*100:.1f}% "
          f"time={elapsed:.1f}s")

    # B2: Abstain (penalty=-1)
    print(f"\n{'='*60}")
    print(f"  B2: Abstain (V+1={V+1}, penalty=-1)")
    print(f"{'='*60}")
    np.random.seed(seed)
    random.seed(seed)
    perm = np.random.permutation(V)

    net = SelfWiringGraph(N + 1, V, n_outputs=V + 1)
    t0 = time.time()
    best, stats, kept = train_abstain(net, inputs, perm, V,
                                      max_attempts=MAX_ATTEMPTS,
                                      penalty=-1, label="abstain_32_p1")
    elapsed = time.time() - t0
    results['abstain_p1'] = {**stats, 'time': round(elapsed, 1), 'kept': kept,
                             'connections': net.count_connections()}
    print(f"\n  FINAL abstain_p1: score={best:+.3f} prec={stats['precision']*100:.1f}% "
          f"abstain={stats['abstain_rate']*100:.1f}% time={elapsed:.1f}s")

    # B3: Abstain (penalty=-2)
    print(f"\n{'='*60}")
    print(f"  B3: Abstain (V+1={V+1}, penalty=-2)")
    print(f"{'='*60}")
    np.random.seed(seed)
    random.seed(seed)
    perm = np.random.permutation(V)

    net = SelfWiringGraph(N + 1, V, n_outputs=V + 1)
    t0 = time.time()
    best, stats, kept = train_abstain(net, inputs, perm, V,
                                      max_attempts=MAX_ATTEMPTS,
                                      penalty=-2, label="abstain_32_p2")
    elapsed = time.time() - t0
    results['abstain_p2'] = {**stats, 'time': round(elapsed, 1), 'kept': kept,
                             'connections': net.count_connections()}
    print(f"\n  FINAL abstain_p2: score={best:+.3f} prec={stats['precision']*100:.1f}% "
          f"abstain={stats['abstain_rate']*100:.1f}% time={elapsed:.1f}s")

    return results


# ============================================================
# Test C: English bigram prediction
# ============================================================

def test_bigram(seed=42):
    """English bigram prediction with abstain.
    The key test: 'q' -> 'u' (high confidence, should answer)
    vs 't' -> anything (low confidence, should abstain)."""

    # Build bigram data from a sample English text
    text = (
        "the quick brown fox jumps over the lazy dog "
        "she sells seashells by the seashore "
        "peter piper picked a peck of pickled peppers "
        "how much wood would a woodchuck chuck "
        "the cat sat on the mat "
        "to be or not to be that is the question "
        "all that glitters is not gold "
        "a stitch in time saves nine "
    ).lower()

    # Build char-to-index mapping (only lowercase + space)
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    V = len(chars)

    # Build bigram pairs
    inputs_raw = []
    targets_raw = []
    for i in range(len(text) - 1):
        c1, c2 = text[i], text[i + 1]
        inputs_raw.append(char2idx[c1])
        targets_raw.append(char2idx[c2])

    # Deduplicate: for each unique input, find the most common target
    from collections import Counter
    bigram_counts = {}
    for inp, tgt in zip(inputs_raw, targets_raw):
        if inp not in bigram_counts:
            bigram_counts[inp] = Counter()
        bigram_counts[inp][tgt] += 1

    # Create training set: unique input -> most common target
    inputs = []
    targets = []
    confidences = {}  # track how "clear" each bigram is
    for inp, counter in sorted(bigram_counts.items()):
        most_common_tgt, count = counter.most_common(1)[0]
        total = sum(counter.values())
        confidence = count / total
        inputs.append(inp)
        targets.append(most_common_tgt)
        confidences[inp] = {
            'target': most_common_tgt,
            'confidence': confidence,
            'total': total,
            'char': chars[inp],
            'target_char': chars[most_common_tgt],
        }

    N = max(120, V * 5)
    MAX_ATTEMPTS = 20000

    print("\n" + "#" * 70)
    print("#  TEST C: English bigram prediction — where abstain shines")
    print(f"#  V={V} chars, {N} neurons, {MAX_ATTEMPTS} attempts, seed={seed}")
    print(f"#  {len(inputs)} unique bigrams from sample text")
    print("#" * 70)

    # Show confidence distribution
    print(f"\n  Bigram confidence distribution:")
    high_conf = sum(1 for v in confidences.values() if v['confidence'] >= 0.8)
    med_conf = sum(1 for v in confidences.values() if 0.4 <= v['confidence'] < 0.8)
    low_conf = sum(1 for v in confidences.values() if v['confidence'] < 0.4)
    print(f"    High (>=80%): {high_conf} bigrams")
    print(f"    Med  (40-80%): {med_conf} bigrams")
    print(f"    Low  (<40%):  {low_conf} bigrams")
    print(f"\n  Examples:")
    sorted_conf = sorted(confidences.items(), key=lambda x: -x[1]['confidence'])
    for idx, (inp, info) in enumerate(sorted_conf[:5]):
        print(f"    '{info['char']}' -> '{info['target_char']}' "
              f"({info['confidence']*100:.0f}% conf, {info['total']} samples)")
    print(f"    ...")
    for idx, (inp, info) in enumerate(sorted_conf[-3:]):
        print(f"    '{info['char']}' -> '{info['target_char']}' "
              f"({info['confidence']*100:.0f}% conf, {info['total']} samples)")

    results = {}

    # C1: Forced choice
    print(f"\n{'='*60}")
    print(f"  C1: Forced choice (V={V})")
    print(f"{'='*60}")
    np.random.seed(seed)
    random.seed(seed)

    net = SelfWiringGraph(N, V, n_outputs=V)
    t0 = time.time()
    best, stats, kept = train_forced(net, inputs, targets, V,
                                     max_attempts=MAX_ATTEMPTS,
                                     label="bigram_forced")
    elapsed = time.time() - t0
    results['forced'] = {**stats, 'time': round(elapsed, 1), 'kept': kept,
                         'connections': net.count_connections()}
    print(f"\n  FINAL forced: score={best:+.3f} acc={stats['accuracy']*100:.1f}% "
          f"time={elapsed:.1f}s")

    # C2: Abstain (penalty=-1)
    print(f"\n{'='*60}")
    print(f"  C2: Abstain (V+1={V+1}, penalty=-1)")
    print(f"{'='*60}")
    np.random.seed(seed)
    random.seed(seed)

    net = SelfWiringGraph(N + 1, V, n_outputs=V + 1)
    t0 = time.time()
    best, stats, kept = train_abstain(net, inputs, targets, V,
                                      max_attempts=MAX_ATTEMPTS,
                                      penalty=-1, label="bigram_abstain_p1")
    elapsed = time.time() - t0
    results['abstain_p1'] = {**stats, 'time': round(elapsed, 1), 'kept': kept,
                             'connections': net.count_connections()}
    print(f"\n  FINAL abstain_p1: score={best:+.3f} prec={stats['precision']*100:.1f}% "
          f"abstain={stats['abstain_rate']*100:.1f}% time={elapsed:.1f}s")

    # C3: Abstain (penalty=-2)
    print(f"\n{'='*60}")
    print(f"  C3: Abstain (V+1={V+1}, penalty=-2)")
    print(f"{'='*60}")
    np.random.seed(seed)
    random.seed(seed)

    net = SelfWiringGraph(N + 1, V, n_outputs=V + 1)
    t0 = time.time()
    best, stats, kept = train_abstain(net, inputs, targets, V,
                                      max_attempts=MAX_ATTEMPTS,
                                      penalty=-2, label="bigram_abstain_p2")
    elapsed = time.time() - t0
    results['abstain_p2'] = {**stats, 'time': round(elapsed, 1), 'kept': kept,
                             'connections': net.count_connections()}
    print(f"\n  FINAL abstain_p2: score={best:+.3f} prec={stats['precision']*100:.1f}% "
          f"abstain={stats['abstain_rate']*100:.1f}% time={elapsed:.1f}s")

    # Analyze: which bigrams did the abstain model answer?
    print(f"\n  --- Per-bigram analysis (abstain_p1 final model) ---")
    net2 = net  # reuse last trained model
    net2.reset()
    # Warmup
    for i in range(len(inputs)):
        world = np.zeros(V, dtype=np.float32)
        world[inputs[i]] = 1.0
        net2.forward(world)
    # Score
    answered_confs = []
    abstained_confs = []
    for i in range(len(inputs)):
        world = np.zeros(V, dtype=np.float32)
        world[inputs[i]] = 1.0
        logits = net2.forward(world)
        probs = softmax(logits)
        pred = np.argmax(probs)
        conf = confidences[inputs[i]]['confidence']
        if pred == V:
            abstained_confs.append(conf)
        else:
            answered_confs.append(conf)

    if answered_confs:
        print(f"    Answered ({len(answered_confs)}): "
              f"avg confidence={np.mean(answered_confs)*100:.1f}%")
    if abstained_confs:
        print(f"    Abstained ({len(abstained_confs)}): "
              f"avg confidence={np.mean(abstained_confs)*100:.1f}%")
    if answered_confs and abstained_confs:
        print(f"    >>> Abstain targets lower-confidence bigrams: "
              f"{'YES' if np.mean(abstained_confs) < np.mean(answered_confs) else 'NO'}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    # Test A: 16-class
    all_results['16_class'] = test_16class()

    # Test B: 32-class
    all_results['32_class'] = test_32class()

    # Test C: English bigram
    all_results['bigram'] = test_bigram()

    # === Final Summary ===
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — Abstain Mechanism")
    print("=" * 70)

    for task_name, results in all_results.items():
        print(f"\n  {task_name}:")
        print(f"    {'Mode':<20s} {'Score':>7s} {'Prec':>7s} {'Abstain':>8s} "
              f"{'C/I/A':>12s} {'Time':>6s}")
        print(f"    {'-'*20} {'-'*7} {'-'*7} {'-'*8} {'-'*12} {'-'*6}")
        for mode, r in results.items():
            cia = f"{r['correct']}/{r['incorrect']}/{r['abstained']}"
            print(f"    {mode:<20s} {r['score']:+6.3f} "
                  f"{r['precision']*100:5.1f}% "
                  f"{r['abstain_rate']*100:6.1f}% "
                  f"{cia:>12s} {r['time']:5.1f}s")

    # Key insight check
    print(f"\n  KEY QUESTION: Does abstain improve net score?")
    for task_name, results in all_results.items():
        forced_score = results['forced']['score']
        abstain_score = results.get('abstain_p1', {}).get('score', 0)
        diff = abstain_score - forced_score
        print(f"    {task_name}: forced={forced_score:+.3f} "
              f"abstain={abstain_score:+.3f} "
              f"delta={diff:+.3f} "
              f"{'ABSTAIN WINS' if diff > 0 else 'FORCED WINS' if diff < 0 else 'TIE'}")

    # Save
    json_path = f"/home/user/VRAXION/v4/research/v22/tests/abstain_results_{timestamp}.json"
    json_safe = {}
    for task, results in all_results.items():
        json_safe[task] = {}
        for mode, r in results.items():
            json_safe[task][mode] = {k: v for k, v in r.items()
                                     if not isinstance(v, np.generic)}
    with open(json_path, 'w') as f:
        json.dump(json_safe, f, indent=2, default=str)
    print(f"\n  Results saved to: {json_path}")

    return all_results


if __name__ == "__main__":
    main()
