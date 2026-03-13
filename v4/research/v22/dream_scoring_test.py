"""
VRAXION v22 — Dream Output Scoring: MSE vs Accuracy
=====================================================
Tests whether continuous scoring functions (MSE, target_prob, log_prob,
margin, combined) accept more mutations than binary accuracy scoring.

Problem:
  Binary scoring (correct=+1, wrong=0) is blind to improvements that
  don't flip the argmax. A mutation that raises P(correct) from 0.3→0.5
  gets ZERO reward if another class sits at 0.6. Result: ~0.1% acceptance
  rate — most beneficial mutations are invisible.

Dream Output Scoring:
  The "dream output" is the perfect one-hot distribution where P(correct)=1.0.
  Score = -MSE(dream, actual) or similar continuous metric.
  This rewards ANY improvement toward the dream, not just argmax flips.

Setup:
  Ternary mask (-1/0/+1), binary weight (0.5/1.5), flip=30%, leaky_relu.

Tests:
  A) 16-class lookup (80 neurons, 8000 attempts) — all 6 scoring modes
  B) 32-class lookup (160 neurons, 12000 attempts) — top 2-3 from A
  C) 64-class lookup (320 neurons, 20000 attempts) — best from B
"""

import numpy as np
import random
import time
import json
from datetime import datetime


# ============================================================
# Network
# ============================================================

class SelfWiringGraph:
    """Self-wiring graph with leaky_relu activation."""

    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30):
        self.N = n_neurons
        self.V = vocab
        self.flip_rate = flip_rate
        self.last_score = 0.0

        # Ternary mask: -1 (inhibit), 0 (no connection), +1 (excite)
        r = np.random.rand(n_neurons, n_neurons)
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        self.mask[r < density / 2] = -1
        self.mask[r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        # Binary weights: 0.5 (weak) or 1.5 (strong), positive only
        self.W = np.where(
            np.random.rand(n_neurons, n_neurons) > 0.5,
            np.float32(0.5), np.float32(1.5)
        )

        # 4D addresses for self-wiring
        self.addr = np.random.randn(n_neurons, 4).astype(np.float32)
        self.addr[:vocab, 3] = 0.0
        self.addr[vocab:, 3] = 0.5
        self.target_W = np.random.randn(n_neurons, 4).astype(np.float32) * 0.1

        # Persistent state
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.decay = 0.5

    def reset(self):
        self.state *= 0

    def forward(self, world, ticks=6):
        act = self.state.copy()
        Weff = self.W * self.mask

        for t in range(ticks):
            act = act * self.decay
            if t == 0:
                act[:self.V] = world
            raw = act @ Weff + act * 0.1
            act = np.where(raw > 0, raw, np.float32(0.01) * raw)  # leaky_relu
            np.clip(act, -10.0, 10.0, out=act)

        self.state = act.copy()
        return act[:self.V]

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
            else:  # rewire
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


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ============================================================
# Scoring functions
# ============================================================

def score_accuracy(probs, target, vocab):
    """Binary: correct prediction = 1, wrong = 0."""
    return 1.0 if np.argmax(probs) == target else 0.0


def score_mse(probs, target, vocab):
    """Distance from dream output (one-hot). Higher = better (less negative)."""
    dream = np.zeros(vocab)
    dream[target] = 1.0
    return -np.mean((dream - probs) ** 2)


def score_target_prob(probs, target, vocab):
    """Simplest: probability of the correct class."""
    return float(probs[target])


def score_log_prob(probs, target, vocab):
    """Cross-entropy style: log probability of correct class."""
    return float(np.log(probs[target] + 1e-8))


def score_margin(probs, target, vocab):
    """Margin of correct class over runner-up (positive = winning)."""
    sorted_probs = np.sort(probs)[::-1]
    target_prob = probs[target]
    if np.argmax(probs) == target:
        return target_prob - sorted_probs[1]  # margin over 2nd best
    else:
        return target_prob - sorted_probs[0]  # negative margin


def score_combined(probs, target, vocab):
    """Accuracy + target_prob combined 50/50."""
    acc = 1.0 if np.argmax(probs) == target else 0.0
    return 0.5 * acc + 0.5 * float(probs[target])


SCORING_MODES = {
    'accuracy':    score_accuracy,
    'mse':         score_mse,
    'target_prob': score_target_prob,
    'log_prob':    score_log_prob,
    'margin':      score_margin,
    'combined':    score_combined,
}


# ============================================================
# Evaluation
# ============================================================

def evaluate(net, inputs, targets, vocab, scoring_fn, ticks=6):
    """Evaluate network. Returns (avg_score, accuracy).
    Two passes: warmup (p=0) builds state, scoring (p=1) measures."""
    net.reset()
    total_score = 0.0
    correct = 0
    n = len(inputs)

    for p in range(2):
        for i in range(n):
            world = np.zeros(vocab, dtype=np.float32)
            world[inputs[i]] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits[:vocab])

            if p == 1:
                total_score += scoring_fn(probs, targets[i], vocab)
                if np.argmax(probs) == targets[i]:
                    correct += 1

    return total_score / n, correct / n


# ============================================================
# Training loop
# ============================================================

def train(net, inputs, targets, vocab, scoring_fn, scoring_name,
          max_attempts=8000, ticks=6, label=""):
    """Train using mutation+selection. SCORE decides keep/revert, not accuracy."""

    def eval_net():
        sc, acc = evaluate(net, inputs, targets, vocab, scoring_fn, ticks)
        net.last_score = max(0.0, acc)
        return sc, acc

    score, accuracy = eval_net()
    best_score = score
    best_acc = accuracy
    phase = "STRUCTURE"
    kept = 0
    stale = 0
    switched = False

    for att in range(max_attempts):
        state = net.save_state()

        # Mutate
        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        # Evaluate
        new_score, new_acc = eval_net()

        # Keep or revert — SCORE decides, not accuracy
        if new_score > score:
            score = new_score
            accuracy = new_acc
            kept += 1
            stale = 0
            if score > best_score:
                best_score = score
                best_acc = accuracy
        else:
            net.restore_state(state)
            stale += 1

        # Phase transition
        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        # Logging
        if (att + 1) % 2000 == 0:
            accept_pct = kept / (att + 1) * 100
            pos, neg = net.pos_neg_ratio()
            print(f"  [{att+1:5d}] {label:25s} | Score: {best_score:+8.5f} | "
                  f"Acc: {best_acc*100:5.1f}% | "
                  f"Accept: {accept_pct:5.2f}% ({kept}/{att+1}) | "
                  f"Conns: {net.count_connections():4d} (+:{pos} -:{neg}) | "
                  f"Phase: {phase}")

        # Termination: accuracy reached 100%
        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    accept_rate = kept / max(1, min(att + 1, max_attempts)) * 100
    return {
        'scoring': scoring_name,
        'best_score': float(best_score),
        'accuracy': float(best_acc),
        'acceptance_rate': round(accept_rate, 3),
        'kept': kept,
        'total_attempts': min(att + 1, max_attempts) if 'att' in dir() else 0,
        'connections': net.count_connections(),
        'pos': int(net.pos_neg_ratio()[0]),
        'neg': int(net.pos_neg_ratio()[1]),
    }


# ============================================================
# Test A: 16-class lookup — all 6 scoring modes
# ============================================================

def test_16class(seed=42):
    V = 16
    N = 80
    MAX_ATTEMPTS = 8000

    print("\n" + "#" * 70)
    print("#  TEST A: 16-class lookup — all 6 scoring modes")
    print(f"#  {N} neurons, {MAX_ATTEMPTS} attempts, seed={seed}")
    print("#" * 70)

    results = {}

    for name, fn in SCORING_MODES.items():
        print(f"\n{'='*60}")
        print(f"  Scoring: {name}")
        print(f"{'='*60}")

        # Same seed for fair comparison
        np.random.seed(seed)
        random.seed(seed)
        perm = np.random.permutation(V)
        inputs = list(range(V))

        net = SelfWiringGraph(N, V)
        t0 = time.time()
        r = train(net, inputs, perm, V, fn, name,
                  max_attempts=MAX_ATTEMPTS, label=f"16c_{name}")
        elapsed = time.time() - t0
        r['time'] = round(elapsed, 1)
        results[name] = r

        print(f"\n  RESULT {name}: acc={r['accuracy']*100:.1f}% "
              f"accept={r['acceptance_rate']:.2f}% "
              f"kept={r['kept']} time={elapsed:.1f}s")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  TEST A SUMMARY — 16-class ({MAX_ATTEMPTS} attempts)")
    print(f"{'='*70}")
    print(f"  {'Scoring':<14s} {'Accuracy':>8s} {'Accept%':>9s} "
          f"{'Kept':>6s} {'Score':>10s} {'Time':>6s}")
    print(f"  {'-'*14} {'-'*8} {'-'*9} {'-'*6} {'-'*10} {'-'*6}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]['acceptance_rate']):
        print(f"  {name:<14s} {r['accuracy']*100:6.1f}% "
              f"{r['acceptance_rate']:8.3f}% "
              f"{r['kept']:5d} "
              f"{r['best_score']:+9.5f} "
              f"{r['time']:5.1f}s")

    return results


# ============================================================
# Test B: 32-class lookup — top 2-3 from A
# ============================================================

def test_32class(top_modes, seed=42):
    V = 32
    N = 160
    MAX_ATTEMPTS = 12000

    print("\n" + "#" * 70)
    print("#  TEST B: 32-class lookup — top scoring modes from A")
    print(f"#  {N} neurons, {MAX_ATTEMPTS} attempts, seed={seed}")
    print(f"#  Modes: {', '.join(top_modes)}")
    print("#" * 70)

    results = {}

    for name in top_modes:
        fn = SCORING_MODES[name]
        print(f"\n{'='*60}")
        print(f"  Scoring: {name}")
        print(f"{'='*60}")

        np.random.seed(seed)
        random.seed(seed)
        perm = np.random.permutation(V)
        inputs = list(range(V))

        net = SelfWiringGraph(N, V)
        t0 = time.time()
        r = train(net, inputs, perm, V, fn, name,
                  max_attempts=MAX_ATTEMPTS, label=f"32c_{name}")
        elapsed = time.time() - t0
        r['time'] = round(elapsed, 1)
        results[name] = r

        print(f"\n  RESULT {name}: acc={r['accuracy']*100:.1f}% "
              f"accept={r['acceptance_rate']:.2f}% "
              f"kept={r['kept']} time={elapsed:.1f}s")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  TEST B SUMMARY — 32-class ({MAX_ATTEMPTS} attempts)")
    print(f"{'='*70}")
    print(f"  {'Scoring':<14s} {'Accuracy':>8s} {'Accept%':>9s} "
          f"{'Kept':>6s} {'Score':>10s} {'Time':>6s}")
    print(f"  {'-'*14} {'-'*8} {'-'*9} {'-'*6} {'-'*10} {'-'*6}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]['acceptance_rate']):
        print(f"  {name:<14s} {r['accuracy']*100:6.1f}% "
              f"{r['acceptance_rate']:8.3f}% "
              f"{r['kept']:5d} "
              f"{r['best_score']:+9.5f} "
              f"{r['time']:5.1f}s")

    return results


# ============================================================
# Test C: 64-class lookup — single best from B
# ============================================================

def test_64class(best_mode, seed=42):
    V = 64
    N = 320
    MAX_ATTEMPTS = 20000

    print("\n" + "#" * 70)
    print("#  TEST C: 64-class lookup — best scoring mode from B")
    print(f"#  {N} neurons, {MAX_ATTEMPTS} attempts, seed={seed}")
    print(f"#  Mode: {best_mode}")
    print("#" * 70)

    fn = SCORING_MODES[best_mode]

    # Also run accuracy baseline for comparison
    results = {}
    for name in ['accuracy', best_mode] if best_mode != 'accuracy' else ['accuracy']:
        cur_fn = SCORING_MODES[name]
        print(f"\n{'='*60}")
        print(f"  Scoring: {name}")
        print(f"{'='*60}")

        np.random.seed(seed)
        random.seed(seed)
        perm = np.random.permutation(V)
        inputs = list(range(V))

        net = SelfWiringGraph(N, V)
        t0 = time.time()
        r = train(net, inputs, perm, V, cur_fn, name,
                  max_attempts=MAX_ATTEMPTS, label=f"64c_{name}")
        elapsed = time.time() - t0
        r['time'] = round(elapsed, 1)
        results[name] = r

        print(f"\n  RESULT {name}: acc={r['accuracy']*100:.1f}% "
              f"accept={r['acceptance_rate']:.2f}% "
              f"kept={r['kept']} time={elapsed:.1f}s")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  TEST C SUMMARY — 64-class ({MAX_ATTEMPTS} attempts)")
    print(f"{'='*70}")
    print(f"  {'Scoring':<14s} {'Accuracy':>8s} {'Accept%':>9s} "
          f"{'Kept':>6s} {'Score':>10s} {'Time':>6s}")
    print(f"  {'-'*14} {'-'*8} {'-'*9} {'-'*6} {'-'*10} {'-'*6}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]['acceptance_rate']):
        print(f"  {name:<14s} {r['accuracy']*100:6.1f}% "
              f"{r['acceptance_rate']:8.3f}% "
              f"{r['kept']:5d} "
              f"{r['best_score']:+9.5f} "
              f"{r['time']:5.1f}s")

    return results


# ============================================================
# Select top modes by acceptance rate
# ============================================================

def select_top_modes(results, n=3):
    """Pick top N scoring modes by acceptance_rate. Always include 'accuracy' as baseline."""
    ranked = sorted(results.items(), key=lambda x: -x[1]['acceptance_rate'])
    top = [name for name, _ in ranked[:n]]
    # Ensure accuracy baseline is always included
    if 'accuracy' not in top:
        top.append('accuracy')
    return top


# ============================================================
# Main
# ============================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    print("=" * 70)
    print("  VRAXION v22 — Dream Output Scoring Test")
    print("  Binary accuracy vs continuous scoring for mutation acceptance")
    print("=" * 70)
    print(f"  Setup: ternary mask, binary weights, flip=30%, leaky_relu")
    print(f"  Key metric: ACCEPTANCE RATE (kept / total attempts)")
    print()

    # === Test A: 16-class, all 6 modes ===
    results_a = test_16class()
    all_results['16_class'] = results_a

    # === Select top modes for B ===
    top_modes = select_top_modes(results_a, n=3)
    print(f"\n  >>> Top modes for Test B (by acceptance rate): {top_modes}")

    # === Test B: 32-class, top modes ===
    results_b = test_32class(top_modes)
    all_results['32_class'] = results_b

    # === Select best mode for C ===
    # From B results, pick the one with highest acceptance rate (excluding accuracy baseline)
    non_acc = {k: v for k, v in results_b.items() if k != 'accuracy'}
    if non_acc:
        best_mode = max(non_acc.items(), key=lambda x: x[1]['acceptance_rate'])[0]
    else:
        best_mode = 'accuracy'
    print(f"\n  >>> Best mode for Test C (by acceptance rate): {best_mode}")

    # === Test C: 64-class, best mode + accuracy baseline ===
    results_c = test_64class(best_mode)
    all_results['64_class'] = results_c

    # === Final Summary ===
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — Dream Output Scoring")
    print("=" * 70)

    for task_name, results in all_results.items():
        print(f"\n  {task_name}:")
        print(f"    {'Scoring':<14s} {'Accuracy':>8s} {'Accept%':>9s} "
              f"{'Kept':>6s} {'Score':>10s} {'Time':>6s}")
        print(f"    {'-'*14} {'-'*8} {'-'*9} {'-'*6} {'-'*10} {'-'*6}")
        for name, r in sorted(results.items(), key=lambda x: -x[1]['acceptance_rate']):
            print(f"    {name:<14s} {r['accuracy']*100:6.1f}% "
                  f"{r['acceptance_rate']:8.3f}% "
                  f"{r['kept']:5d} "
                  f"{r['best_score']:+9.5f} "
                  f"{r['time']:5.1f}s")

    # Key insight: acceptance rate comparison
    print(f"\n  KEY METRIC — Acceptance Rate Improvement over Accuracy Baseline:")
    for task_name, results in all_results.items():
        baseline = results.get('accuracy', {}).get('acceptance_rate', 0)
        print(f"    {task_name} (baseline accuracy: {baseline:.3f}%):")
        for name, r in sorted(results.items(), key=lambda x: -x[1]['acceptance_rate']):
            if name == 'accuracy':
                continue
            ratio = r['acceptance_rate'] / max(0.001, baseline)
            print(f"      {name:<14s} {r['acceptance_rate']:8.3f}% "
                  f"({ratio:.1f}x baseline) "
                  f"acc={r['accuracy']*100:.1f}%")

    # Save JSON
    json_path = f"/home/user/VRAXION/v4/research/v22/tests/dream_scoring_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {json_path}")

    return all_results


if __name__ == "__main__":
    main()
