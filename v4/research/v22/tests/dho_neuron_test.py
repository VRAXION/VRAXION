"""
DHO Neuron Benchmark — HORN Paper Inspired Oscillator Upgrade
==============================================================
The HORN paper (Effenberger, Singer et al. PNAS 2025) showed that damped
harmonic oscillator (DHO) neurons beat all non-oscillating architectures.

Our capacitor neuron is a PRIMITIVE DHO — it charges and decays but does
NOT oscillate. The missing ingredient: VELOCITY (momentum that sustains
oscillation).

Capacitor (current):
  charge += input * 0.3
  charge *= 0.85
  output = max(charge - threshold, 0)
  -> One bump then decays. Does NOT oscillate.

DHO (new):
  force = input * alpha - charge * omega^2 - velocity * gamma
  velocity += force
  charge += velocity
  output = charge
  -> OSCILLATES! Up-down with decaying amplitude.

Tests:
  Phase 1: 5-way A/B on 16-class (capacitor, dho_homo, dho_hetero, dho_evolve, leaky_relu)
  Phase 2: Winner + capacitor on 32/64-class
  Phase 3: Tick sweep with best DHO
  Phase 4: Extended 8K run on 64-class

All tests parallel, 5 seeds, combined scoring.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from v22_best_config import SelfWiringGraph, softmax


# ============================================================
#  DHONet — Damped Harmonic Oscillator neuron model
# ============================================================

class DHONet:
    """Self-Wiring Graph with DHO neuron dynamics.

    Each neuron is a damped harmonic oscillator with:
      omega: natural frequency (how fast it oscillates)
      gamma: damping (how fast oscillation decays)
      alpha: excitability (how strongly it responds to input)

    DHO update per tick:
      force = input * alpha - charge * omega^2 - velocity * gamma
      velocity += force
      charge += velocity
    """

    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30,
                 omega_range=(0.3, 1.5), gamma_range=(0.05, 0.3),
                 alpha_range=(0.2, 0.8), homogeneous=False):
        self.N = n_neurons
        self.V = vocab
        self.flip_rate = flip_rate
        self.last_acc = 0.0

        # Ternary mask (-1/0/+1)
        r = np.random.rand(n_neurons, n_neurons)
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        self.mask[r < density / 2] = -1
        self.mask[r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        # Binary weights (0.5/1.5)
        self.W = np.where(
            np.random.rand(n_neurons, n_neurons) > 0.5,
            np.float32(0.5), np.float32(1.5)
        )

        # 4D addresses for self-wiring
        self.addr = np.random.randn(n_neurons, 4).astype(np.float32)
        self.addr[:vocab, 3] = 0.0
        self.addr[vocab:, 3] = 0.5
        self.target_W = np.random.randn(n_neurons, 4).astype(np.float32) * 0.1

        # DHO parameters PER NEURON
        if homogeneous:
            self.omega = np.full(n_neurons, 1.0, dtype=np.float32)
            self.gamma = np.full(n_neurons, 0.15, dtype=np.float32)
            self.alpha = np.full(n_neurons, 0.5, dtype=np.float32)
        else:
            self.omega = np.random.uniform(
                omega_range[0], omega_range[1], n_neurons).astype(np.float32)
            self.gamma = np.random.uniform(
                gamma_range[0], gamma_range[1], n_neurons).astype(np.float32)
            self.alpha = np.random.uniform(
                alpha_range[0], alpha_range[1], n_neurons).astype(np.float32)

        # State: charge (position) + velocity
        self.charge = np.zeros(n_neurons, dtype=np.float32)
        self.velocity = np.zeros(n_neurons, dtype=np.float32)
        self.state = np.zeros(n_neurons, dtype=np.float32)

    def reset(self):
        self.charge *= 0
        self.velocity *= 0
        self.state *= 0

    def forward(self, world, ticks=8):
        """DHO forward pass.

        force = raw * alpha - charge * omega^2 - velocity * gamma
        velocity += force
        charge += velocity
        """
        act = self.state.copy()
        Weff = self.W * self.mask

        for t in range(ticks):
            if t == 0:
                act[:self.V] = world

            # Signal propagation
            raw = act @ Weff + act * 0.1

            # DHO dynamics
            force = (raw * self.alpha
                     - self.charge * (self.omega ** 2)
                     - self.velocity * self.gamma)
            self.velocity += force
            self.charge += self.velocity

            # Clamp
            self.charge = np.clip(self.charge, -3.0, 3.0)
            self.velocity = np.clip(self.velocity, -3.0, 3.0)

            # Output activation = charge (like capacitor reads charge)
            act = self.charge.copy()

        self.state = act.copy()
        return self.charge[:self.V]

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

    # === State management (10-element tuple) ===

    def save_state(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(),
                self.addr.copy(), self.target_W.copy(), self.charge.copy(),
                self.velocity.copy(), self.omega.copy(),
                self.gamma.copy(), self.alpha.copy())

    def restore_state(self, s):
        self.W = s[0].copy()
        self.mask = s[1].copy()
        self.state = s[2].copy()
        self.addr = s[3].copy()
        self.target_W = s[4].copy()
        self.charge = s[5].copy()
        self.velocity = s[6].copy()
        self.omega = s[7].copy()
        self.gamma = s[8].copy()
        self.alpha = s[9].copy()

    # === Mutation operators ===

    def mutate_structure(self, rate=0.05):
        """Structural mutation: flip / add / remove / rewire."""
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
        """Weight mutation: toggle 0.5 <-> 1.5."""
        alive = np.argwhere(self.mask != 0)
        if len(alive) > 0:
            n = max(1, int(len(alive) * 0.05))
            idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
            for j in range(len(idx)):
                r2, c = int(idx[j][0]), int(idx[j][1])
                self.W[r2, c] = np.float32(1.5) if self.W[r2, c] < 1.0 else np.float32(0.5)

    def mutate_dho_params(self, rate=0.05):
        """Mutate DHO parameters (omega, gamma, alpha) per neuron."""
        n_mutate = max(1, int(self.N * rate))
        neurons = np.random.choice(self.N, n_mutate, replace=False)
        for ni in neurons:
            param = random.choice(['omega', 'gamma', 'alpha'])
            if param == 'omega':
                self.omega[ni] += random.gauss(0, 0.1)
                self.omega[ni] = np.clip(self.omega[ni], 0.1, 3.0)
            elif param == 'gamma':
                self.gamma[ni] += random.gauss(0, 0.03)
                self.gamma[ni] = np.clip(self.gamma[ni], 0.01, 0.5)
            else:
                self.alpha[ni] += random.gauss(0, 0.05)
                self.alpha[ni] = np.clip(self.alpha[ni], 0.05, 1.0)

    def self_wire(self):
        """Active neurons propose new connections (inverse arousal)."""
        if self.last_acc < 0.3:
            top_k, max_new = 2, 1
        elif self.last_acc < 0.7:
            top_k, max_new = 3, 2
        else:
            top_k, max_new = 5, 3

        act = self.state
        a2 = np.abs(act[self.V:])
        if a2.sum() < 0.01:
            return

        nc = min(top_k, len(a2))
        top = np.argpartition(a2, -nc)[-nc:] + self.V
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


# ============================================================
#  Leaky ReLU forward (for baseline comparison)
# ============================================================

class LeakyReLUNet:
    """Original leaky_relu architecture for comparison."""

    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30):
        self.N = n_neurons
        self.V = vocab
        self.flip_rate = flip_rate
        self.last_acc = 0.0
        self.decay = 0.5

        r = np.random.rand(n_neurons, n_neurons)
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        self.mask[r < density / 2] = -1
        self.mask[r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        self.W = np.where(
            np.random.rand(n_neurons, n_neurons) > 0.5,
            np.float32(0.5), np.float32(1.5)
        )

        self.addr = np.random.randn(n_neurons, 4).astype(np.float32)
        self.addr[:vocab, 3] = 0.0
        self.addr[vocab:, 3] = 0.5
        self.target_W = np.random.randn(n_neurons, 4).astype(np.float32) * 0.1

        self.state = np.zeros(n_neurons, dtype=np.float32)

    def reset(self):
        self.state *= 0

    def forward(self, world, ticks=8):
        act = self.state.copy()
        Weff = self.W * self.mask
        for t in range(ticks):
            act = act * self.decay
            if t == 0:
                act[:self.V] = world
            raw = act @ Weff + act * 0.1
            act = np.where(raw > 0, raw, np.float32(0.01) * raw)
            np.clip(act, -10.0, 10.0, out=act)
        self.state = act.copy()
        return act[:self.V]

    def count_connections(self):
        return int((self.mask != 0).sum())

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
        if self.last_acc < 0.3:
            top_k, max_new = 2, 1
        elif self.last_acc < 0.7:
            top_k, max_new = 3, 2
        else:
            top_k, max_new = 5, 3
        act = self.state
        a2 = np.abs(act[self.V:])
        if a2.sum() < 0.01:
            return
        nc = min(top_k, len(a2))
        top = np.argpartition(a2, -nc)[-nc:] + self.V
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


# ============================================================
#  Training with configurable neuron type + combined scoring
# ============================================================

def train_config(label, n_classes, n_neurons, seed, max_attempts=4000,
                 neuron_type='capacitor', ticks=8, dho_evolve=False):
    """Train with a specific neuron type. Returns result dict."""
    np.random.seed(seed)
    random.seed(seed)

    V = n_classes

    # Create network based on type
    if neuron_type == 'capacitor':
        net = SelfWiringGraph(n_neurons, V)
    elif neuron_type == 'dho_homo':
        net = DHONet(n_neurons, V, homogeneous=True)
    elif neuron_type == 'dho_hetero':
        net = DHONet(n_neurons, V, homogeneous=False)
    elif neuron_type == 'dho_evolve':
        net = DHONet(n_neurons, V, homogeneous=False)
        dho_evolve = True
    elif neuron_type == 'leaky_relu':
        net = LeakyReLUNet(n_neurons, V)
    else:
        raise ValueError(f"Unknown neuron_type: {neuron_type}")

    is_dho = neuron_type.startswith('dho')

    perm = np.random.permutation(V)
    inputs = list(range(V))
    targets = perm.tolist()

    def evaluate():
        net.reset()
        correct = 0
        total_score = 0.0

        for p in range(2):
            for i in range(len(inputs)):
                world = np.zeros(V, dtype=np.float32)
                world[inputs[i]] = 1.0
                logits = net.forward(world, ticks=ticks)
                probs = softmax(logits[:V])

                if p == 1:
                    acc_i = 1.0 if np.argmax(probs) == targets[i] else 0.0
                    tp = float(probs[targets[i]])
                    total_score += 0.5 * acc_i + 0.5 * tp
                    if acc_i > 0:
                        correct += 1

        acc = correct / len(inputs)
        score = total_score / len(inputs)
        net.last_acc = acc
        return score, acc

    score, acc = evaluate()
    best_score = score
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    switched = False

    # DHO stats tracking
    dho_stats_log = []

    t0 = time.time()

    for att in range(max_attempts):
        state = net.save_state()

        # Mutation
        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            r = random.random()
            if is_dho and dho_evolve and r < 0.15:
                net.mutate_dho_params(0.05)
            elif r < 0.45:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        net.self_wire()
        new_score, new_acc = evaluate()

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_score = max(best_score, score)
            best_acc = max(best_acc, new_acc)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        # Log DHO stats every 2000 attempts
        if is_dho and (att + 1) % 2000 == 0:
            dho_stats_log.append({
                'att': att + 1,
                'omega_mean': float(np.mean(net.omega)),
                'omega_std': float(np.std(net.omega)),
                'omega_min': float(np.min(net.omega)),
                'omega_max': float(np.max(net.omega)),
                'gamma_mean': float(np.mean(net.gamma)),
                'gamma_std': float(np.std(net.gamma)),
                'alpha_mean': float(np.mean(net.alpha)),
                'alpha_std': float(np.std(net.alpha)),
                'energy': float(np.mean(net.charge ** 2)),
                'acc': best_acc,
            })

        if best_acc >= 0.99 or stale >= 3500:
            break

    elapsed = time.time() - t0
    accept_rate = (kept / max(att + 1, 1)) * 100

    result = {
        'label': label,
        'seed': seed,
        'n_classes': n_classes,
        'acc': best_acc,
        'score': best_score,
        'kept': kept,
        'accept_rate': accept_rate,
        'time': elapsed,
        'conns': net.count_connections(),
    }

    # Add DHO final stats
    if is_dho:
        result['omega_mean'] = float(np.mean(net.omega))
        result['omega_std'] = float(np.std(net.omega))
        result['gamma_mean'] = float(np.mean(net.gamma))
        result['alpha_mean'] = float(np.mean(net.alpha))
        result['energy'] = float(np.mean(net.charge ** 2))
        result['dho_stats_log'] = dho_stats_log

    return result


def worker(args):
    return train_config(**args)


def aggregate_results(results):
    groups = defaultdict(list)
    for r in results:
        groups[r['label']].append(r)

    summary = {}
    for label, runs in groups.items():
        accs = [r['acc'] for r in runs]
        rates = [r['accept_rate'] for r in runs]
        times = [r['time'] for r in runs]
        summary[label] = {
            'acc_mean': np.mean(accs),
            'acc_std': np.std(accs),
            'accept_mean': np.mean(rates),
            'time_mean': np.mean(times),
            'n': len(runs),
        }
    return summary


def print_summary(title, summary):
    print(f"\n  {'='*70}")
    print(f"  {title}")
    print(f"  {'='*70}")
    print(f"  {'Config':<35s} {'Acc':>7s} {'Std':>6s} {'Accept%':>8s} {'Time':>6s}")
    print(f"  {'-'*65}")
    for label in sorted(summary, key=lambda k: summary[k]['acc_mean'], reverse=True):
        s = summary[label]
        print(f"  {label:<35s} {s['acc_mean']*100:6.1f}% {s['acc_std']*100:5.1f}% "
              f"{s['accept_mean']:7.3f}% {s['time_mean']:5.0f}s")
    sys.stdout.flush()


def print_dho_stats(results):
    """Print DHO-specific statistics from evolved networks."""
    dho_results = [r for r in results if 'omega_mean' in r]
    if not dho_results:
        return

    print(f"\n  --- DHO Parameter Distribution (final) ---")
    groups = defaultdict(list)
    for r in dho_results:
        groups[r['label']].append(r)

    for label, runs in sorted(groups.items()):
        omegas = [r['omega_mean'] for r in runs]
        omega_stds = [r['omega_std'] for r in runs]
        gammas = [r['gamma_mean'] for r in runs]
        alphas = [r['alpha_mean'] for r in runs]
        energies = [r['energy'] for r in runs]
        print(f"  {label}:")
        print(f"    omega: mean={np.mean(omegas):.3f} (inner_std={np.mean(omega_stds):.3f})")
        print(f"    gamma: mean={np.mean(gammas):.3f}")
        print(f"    alpha: mean={np.mean(alphas):.3f}")
        print(f"    energy: mean={np.mean(energies):.4f}")
    sys.stdout.flush()


# ============================================================
#  Test configurations
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]


def build_phase1_configs(n_classes, n_neurons, max_attempts=4000, ticks=8):
    """Phase 1: 5-way A/B comparison."""
    configs = []
    modes = [
        ('capacitor', 'capacitor'),
        ('dho_homo', 'dho_homo'),
        ('dho_hetero', 'dho_hetero'),
        ('dho_evolve', 'dho_evolve'),
        ('leaky_relu', 'leaky_relu'),
    ]
    for label_suffix, ntype in modes:
        label = f'{n_classes}c {label_suffix}'
        for seed in SEEDS:
            configs.append(dict(
                label=label, seed=seed,
                n_classes=n_classes, n_neurons=n_neurons,
                max_attempts=max_attempts, neuron_type=ntype, ticks=ticks))
    return configs


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    n_workers = os.cpu_count() or 4
    print(f"  CPU cores: {n_workers}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Scoring: combined (0.5*acc + 0.5*target_prob)")

    # === PHASE 1: 5-way A/B on 16-class ===
    print(f"\n{'#'*70}")
    print(f"  PHASE 1: Neuron Model Comparison — 16-class, 256 neurons")
    print(f"{'#'*70}", flush=True)

    configs_p1 = build_phase1_configs(16, 256, max_attempts=4000, ticks=8)
    print(f"  Running {len(configs_p1)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_p1 = list(pool.map(worker, configs_p1))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    summary_p1 = aggregate_results(results_p1)
    print_summary("PHASE 1: 16-CLASS RESULTS", summary_p1)
    print_dho_stats(results_p1)

    # Find best and best DHO
    best_p1 = max(summary_p1, key=lambda k: summary_p1[k]['acc_mean'])
    dho_labels = [k for k in summary_p1 if 'dho' in k]
    best_dho = max(dho_labels, key=lambda k: summary_p1[k]['acc_mean']) if dho_labels else None

    print(f"\n  Best overall: {best_p1} ({summary_p1[best_p1]['acc_mean']*100:.1f}%)")
    if best_dho:
        print(f"  Best DHO: {best_dho} ({summary_p1[best_dho]['acc_mean']*100:.1f}%)")
    sys.stdout.flush()

    # === PHASE 2: Winner + capacitor on 32 and 64-class ===
    # Use best DHO (even if capacitor won, we want to see DHO on harder tasks)
    test_mode = best_dho.replace('16c ', '') if best_dho else 'dho_evolve'

    print(f"\n{'#'*70}")
    print(f"  PHASE 2: {test_mode} vs capacitor on 32-class and 64-class")
    print(f"{'#'*70}", flush=True)

    configs_p2 = []
    for nc in [32, 64]:
        for ntype in ['capacitor', test_mode]:
            label = f'{nc}c {ntype}'
            for seed in SEEDS:
                configs_p2.append(dict(
                    label=label, seed=seed,
                    n_classes=nc, n_neurons=256,
                    max_attempts=4000, neuron_type=ntype, ticks=8))

    print(f"  Running {len(configs_p2)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_p2 = list(pool.map(worker, configs_p2))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    results_32 = [r for r in results_p2 if r['n_classes'] == 32]
    results_64 = [r for r in results_p2 if r['n_classes'] == 64]
    summary_32 = aggregate_results(results_32)
    summary_64 = aggregate_results(results_64)
    print_summary("32-CLASS RESULTS", summary_32)
    print_summary("64-CLASS RESULTS", summary_64)
    print_dho_stats(results_p2)

    # === PHASE 3: Tick sweep with best DHO on 16-class ===
    print(f"\n{'#'*70}")
    print(f"  PHASE 3: Tick Sweep — {test_mode} on 16-class")
    print(f"{'#'*70}", flush=True)

    configs_p3 = []
    for nticks in [6, 8, 12, 16]:
        label = f'16c {test_mode} t={nticks}'
        for seed in SEEDS:
            configs_p3.append(dict(
                label=label, seed=seed,
                n_classes=16, n_neurons=256,
                max_attempts=4000, neuron_type=test_mode, ticks=nticks))

    print(f"  Running {len(configs_p3)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_p3 = list(pool.map(worker, configs_p3))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    summary_p3 = aggregate_results(results_p3)
    print_summary("TICK SWEEP RESULTS", summary_p3)

    best_tick_label = max(summary_p3, key=lambda k: summary_p3[k]['acc_mean'])
    best_ticks = int(best_tick_label.split('t=')[1])
    print(f"\n  Best tick count: {best_ticks}", flush=True)

    # === PHASE 4: Extended run on 64-class ===
    print(f"\n{'#'*70}")
    print(f"  PHASE 4: Extended 8K run — 64-class, {test_mode} t={best_ticks} vs capacitor")
    print(f"{'#'*70}", flush=True)

    configs_p4 = []
    for seed in SEEDS:
        configs_p4.append(dict(
            label=f'64c {test_mode} t={best_ticks} 8K',
            seed=seed, n_classes=64, n_neurons=256,
            max_attempts=8000, neuron_type=test_mode, ticks=best_ticks))
        configs_p4.append(dict(
            label='64c capacitor 8K',
            seed=seed, n_classes=64, n_neurons=256,
            max_attempts=8000, neuron_type='capacitor', ticks=8))

    print(f"  Running {len(configs_p4)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_p4 = list(pool.map(worker, configs_p4))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    summary_p4 = aggregate_results(results_p4)
    print_summary("64-CLASS EXTENDED (8K)", summary_p4)
    print_dho_stats(results_p4)

    # === FINAL ===
    print(f"\n{'#'*70}")
    print(f"  DONE")
    print(f"{'#'*70}", flush=True)
