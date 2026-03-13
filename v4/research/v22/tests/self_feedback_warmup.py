"""
VRAXION v22 — Self-Feedback Warmup / Schedule Test
====================================================
Self-feedback (sfb) = the signal a neuron feeds back to itself.
Currently fixed at 0.1. Evolution found 0.3 optimal — but 0.3 explodes
on fresh networks because topology can't handle it yet.

Hypothesis: sfb should START low and GROW as topology matures.
Like warmup in transformer training.

5 configurations tested:
  1. fix_01      — fixed sfb=0.1 (baseline)
  2. fix_03      — fixed sfb=0.3 (known to explode fresh)
  3. linear_wu   — linear warmup 0.1→0.3 over attempts
  4. acc_wu      — accuracy-gated warmup 0.1→0.3
  5. learn_sfb   — learnable sfb only (5% mutation budget)

Tasks: 16-class (80N, 8K attempts) and 64-class (320N, 20K attempts), 5 seeds each.
"""

import numpy as np
import random
import time
import json
from datetime import datetime


# ============================================================
# Schedule functions
# ============================================================

def sfb_linear(attempt, max_attempts, accuracy):
    """Linear warmup: 0.1 → 0.3 over training."""
    progress = attempt / max_attempts
    return 0.1 + 0.2 * progress


def sfb_accuracy(attempt, max_attempts, accuracy):
    """Accuracy-gated stepped warmup."""
    if accuracy < 0.3:
        return 0.1
    elif accuracy < 0.6:
        return 0.15
    elif accuracy < 0.8:
        return 0.2
    else:
        return 0.3


def sfb_fixed(value):
    """Returns a schedule function that always returns a fixed value."""
    def fn(attempt, max_attempts, accuracy):
        return value
    return fn


# ============================================================
# Network with scheduled/learnable self-feedback
# ============================================================

class SfbNet:
    """Capacitor neuron with configurable self-feedback handling.

    mode='fixed':    sfb is a constant (set at init)
    mode='schedule': sfb comes from a schedule function (linear/accuracy)
    mode='learnable': sfb evolves via mutation (5% budget)
    """

    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30,
                 mode='fixed', sfb_init=0.1, sfb_schedule_fn=None):
        self.N = n_neurons
        self.V = vocab
        self.flip_rate = flip_rate
        self.last_acc = 0.0
        self.mode = mode

        # Capacitor params (fixed except sfb)
        self.threshold = 0.5
        self.leak = 0.85
        self.acc_rate = 0.3
        self.self_fb = sfb_init
        self.sfb_schedule_fn = sfb_schedule_fn

        # For tracking current sfb in schedule mode
        self._current_sfb = sfb_init

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

        # 4D addresses
        self.addr = np.random.randn(n_neurons, 4).astype(np.float32)
        self.addr[:vocab, 3] = 0.0
        self.addr[vocab:, 3] = 0.5
        self.target_W = np.random.randn(n_neurons, 4).astype(np.float32) * 0.1

        # Persistent state
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.charge = np.zeros(n_neurons, dtype=np.float32)

    def get_sfb(self):
        """Return current effective self-feedback value."""
        if self.mode == 'schedule':
            return self._current_sfb
        return self.self_fb

    def update_schedule(self, attempt, max_attempts, accuracy):
        """Update sfb from schedule (called externally by training loop)."""
        if self.mode == 'schedule' and self.sfb_schedule_fn is not None:
            self._current_sfb = self.sfb_schedule_fn(attempt, max_attempts, accuracy)

    def reset(self):
        self.state *= 0
        self.charge *= 0

    def forward(self, world, ticks=8):
        act = self.state.copy()
        Weff = self.W * self.mask
        sfb = self.get_sfb()

        for t in range(ticks):
            if t == 0:
                act[:self.V] = world
            raw = act @ Weff + act * sfb
            self.charge += raw * self.acc_rate
            self.charge *= self.leak
            act = np.maximum(self.charge - self.threshold, 0.0)
            self.charge = np.clip(self.charge,
                                  -self.threshold * 2,
                                  self.threshold * 2)

        self.state = act.copy()
        return self.charge[:self.V]

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

    def save_state(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(),
                self.addr.copy(), self.target_W.copy(), self.charge.copy(),
                self.self_fb)

    def restore_state(self, s):
        self.W = s[0].copy()
        self.mask = s[1].copy()
        self.state = s[2].copy()
        self.addr = s[3].copy()
        self.target_W = s[4].copy()
        self.charge = s[5].copy()
        self.self_fb = s[6]

    def mutate(self):
        if self.mode == 'learnable' and random.random() < 0.05:
            self.self_fb += random.gauss(0, 0.02)
            self.self_fb = max(0.0, min(0.5, self.self_fb))
        else:
            self._mutate_topology()

    def _mutate_topology(self):
        phase_r = random.random()
        if phase_r < 0.3:
            # Weight mutation
            alive = np.argwhere(self.mask != 0)
            if len(alive) > 0:
                n = max(1, int(len(alive) * 0.05))
                idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                for j in range(len(idx)):
                    r2, c = int(idx[j][0]), int(idx[j][1])
                    self.W[r2, c] = np.float32(1.5) if self.W[r2, c] < 1.0 else np.float32(0.5)
        else:
            rate = 0.05
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


# ============================================================
# Softmax + scoring
# ============================================================

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def score_combined(probs, target, vocab):
    acc = 1.0 if np.argmax(probs) == target else 0.0
    return 0.5 * acc + 0.5 * float(probs[target])


# ============================================================
# Evaluate
# ============================================================

def evaluate(net, inputs, targets, vocab, ticks=8):
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
                total_score += score_combined(probs, targets[i], vocab)
                if np.argmax(probs) == targets[i]:
                    correct += 1

    return total_score / n, correct / n


# ============================================================
# Training loop with schedule support
# ============================================================

def train(net, inputs, targets, vocab, max_attempts=8000, ticks=8,
          log_interval=2000, label=""):
    """Train using mutation+selection with combined scoring.
    Supports scheduled sfb via net.update_schedule()."""

    score, accuracy = evaluate(net, inputs, targets, vocab, ticks)
    best_score = score
    best_acc = accuracy
    kept = 0
    stale = 0
    logs = []

    for att in range(max_attempts):
        # Update schedule before mutation (so sfb reflects current state)
        net.update_schedule(att, max_attempts, best_acc)

        state = net.save_state()
        net.mutate()

        new_score, new_acc = evaluate(net, inputs, targets, vocab, ticks)

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

        # Logging
        if (att + 1) % log_interval == 0:
            conns = net.count_connections()
            sfb_val = net.get_sfb()

            log_entry = {
                'step': att + 1,
                'accuracy': round(best_acc, 4),
                'sfb': round(sfb_val, 4),
                'connections': conns,
                'kept': kept,
            }
            logs.append(log_entry)

            print(f"  [{att+1:5d}] {label:20s} | Acc: {best_acc*100:5.1f}% | "
                  f"sfb={sfb_val:.3f} | Conns: {conns:5d} | Kept: {kept:3d}")

        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    return best_acc, kept, logs


# ============================================================
# Config definitions
# ============================================================

CONFIGS = {
    'fix_01': {
        'mode': 'fixed',
        'sfb_init': 0.1,
        'sfb_schedule_fn': None,
        'desc': 'Fixed sfb=0.1 (baseline)',
    },
    'fix_03': {
        'mode': 'fixed',
        'sfb_init': 0.3,
        'sfb_schedule_fn': None,
        'desc': 'Fixed sfb=0.3 (explodes fresh)',
    },
    'linear_wu': {
        'mode': 'schedule',
        'sfb_init': 0.1,
        'sfb_schedule_fn': sfb_linear,
        'desc': 'Linear warmup 0.1→0.3',
    },
    'acc_wu': {
        'mode': 'schedule',
        'sfb_init': 0.1,
        'sfb_schedule_fn': sfb_accuracy,
        'desc': 'Accuracy warmup 0.1→0.3',
    },
    'learn_sfb': {
        'mode': 'learnable',
        'sfb_init': 0.1,
        'sfb_schedule_fn': None,
        'desc': 'Learnable sfb (5% budget)',
    },
}

CONFIG_ORDER = ['fix_01', 'fix_03', 'linear_wu', 'acc_wu', 'learn_sfb']


# ============================================================
# Run experiment
# ============================================================

def run_experiment(task_classes, n_neurons, max_attempts, seeds, log_interval=2000):
    """Run all configs for a given task size."""

    perm = None  # Will be set per-seed
    results = {}

    for cfg_name in CONFIG_ORDER:
        cfg = CONFIGS[cfg_name]
        accs = []

        for seed in seeds:
            np.random.seed(seed)
            random.seed(seed)

            V = task_classes
            perm = np.random.permutation(V)
            inputs = list(range(V))

            label = f"{cfg_name}_{V}c_s{seed}"

            print(f"\n{'='*60}")
            print(f"  {cfg_name} | {V}-class | seed={seed}")
            print(f"  {cfg['desc']}")
            print(f"{'='*60}")

            net = SfbNet(
                n_neurons, V,
                mode=cfg['mode'],
                sfb_init=cfg['sfb_init'],
                sfb_schedule_fn=cfg['sfb_schedule_fn'],
            )

            t0 = time.time()
            best_acc, total_kept, logs = train(
                net, inputs, perm, V,
                max_attempts=max_attempts,
                ticks=8,
                log_interval=log_interval,
                label=label,
            )
            elapsed = time.time() - t0

            final_sfb = net.get_sfb()
            print(f"  DONE: {best_acc*100:.1f}% | final_sfb={final_sfb:.3f} | "
                  f"kept={total_kept} | {elapsed:.1f}s")

            accs.append(best_acc)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        best = max(accs)

        results[cfg_name] = {
            'mean_acc': round(float(mean_acc), 4),
            'std_acc': round(float(std_acc), 4),
            'best_acc': round(float(best), 4),
            'all_accs': [round(float(a), 4) for a in accs],
        }

    return results


# ============================================================
# Main
# ============================================================

def main():
    seeds = [42, 123, 456, 789, 1337]

    all_results = {}

    # --- 16-class ---
    print("\n" + "#" * 70)
    print("#  16-CLASS TEST (80 neurons, 8000 attempts, 5 seeds)")
    print("#" * 70)

    results_16 = run_experiment(
        task_classes=16, n_neurons=80,
        max_attempts=8000, seeds=seeds,
        log_interval=2000,
    )
    all_results['16_class'] = results_16

    print(f"\n{'='*70}")
    print(f"  16-CLASS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<20s} {'Mean Acc':>9s} {'Std':>7s} {'Best':>7s}")
    print(f"  {'-'*20} {'-'*9} {'-'*7} {'-'*7}")
    for cfg_name in CONFIG_ORDER:
        r = results_16[cfg_name]
        print(f"  {cfg_name:<20s} {r['mean_acc']*100:8.1f}% {r['std_acc']*100:6.1f}% {r['best_acc']*100:6.1f}%")

    # --- 64-class ---
    print("\n" + "#" * 70)
    print("#  64-CLASS TEST (320 neurons, 20000 attempts, 5 seeds)")
    print("#" * 70)

    results_64 = run_experiment(
        task_classes=64, n_neurons=320,
        max_attempts=20000, seeds=seeds,
        log_interval=4000,
    )
    all_results['64_class'] = results_64

    print(f"\n{'='*70}")
    print(f"  64-CLASS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<20s} {'Mean Acc':>9s} {'Std':>7s} {'Best':>7s}")
    print(f"  {'-'*20} {'-'*9} {'-'*7} {'-'*7}")
    for cfg_name in CONFIG_ORDER:
        r = results_64[cfg_name]
        print(f"  {cfg_name:<20s} {r['mean_acc']*100:8.1f}% {r['std_acc']*100:6.1f}% {r['best_acc']*100:6.1f}%")

    # --- Final comparison ---
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Config':<20s} {'16-class':>10s} {'64-class':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    for cfg_name in CONFIG_ORDER:
        r16 = all_results['16_class'][cfg_name]
        r64 = all_results['64_class'][cfg_name]
        print(f"  {cfg_name:<20s} {r16['mean_acc']*100:9.1f}% {r64['mean_acc']*100:9.1f}%")

    # Save JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'/home/user/VRAXION/v4/research/v22/tests/sfb_warmup_results_{timestamp}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == '__main__':
    main()
