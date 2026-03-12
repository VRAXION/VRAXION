"""
VRAXION v22 — Bio-Tuned Capacitor Sweep
=========================================
Tests capacitor neuron with biological features (AHP + refractory)
but gain/ticks scaled to our network size (~11 synapses/neuron, not 10,000).

Problem with pure bio params:
  - input_gain=0.033 was calibrated for 10,000 synapses/neuron
  - Our network has ~11 effective inputs/neuron → 99% sparsity, nothing fires

Solution: keep bio dynamics (AHP, refractory, leak), scale gain to network.

Sweep:
  - gain: 0.10, 0.15, 0.20, 0.30 (network-scaled)
  - ticks: 8, 10, 12 (practical range)
  - AHP depth: -0.2, -0.3 (below resting)
  - refractory: 1, 2 ticks
  - baseline: original capacitor (gain=0.3, no AHP, no refractory)
"""

import numpy as np
import random
import time
import json
import sys
from datetime import datetime


class SelfWiringGraph:
    """Self-wiring graph with bio-tuned capacitor activation."""

    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30,
                 activation='leaky_relu', threshold=0.5, leak=0.85,
                 input_gain=0.3, ahp_depth=0.0, refractory_ticks=0):
        self.N = n_neurons
        self.V = vocab
        self.flip_rate = flip_rate
        self.last_acc = 0.0
        self.activation = activation
        self.threshold = threshold
        self.leak = leak
        self.input_gain = input_gain
        self.ahp_depth = ahp_depth
        self.refractory_ticks = refractory_ticks

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
        self.addr[:vocab, 3] = 0.0
        self.addr[vocab:, 3] = 0.5
        self.target_W = np.random.randn(n_neurons, 4).astype(np.float32) * 0.1

        # Persistent state
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.decay = 0.5

        # Capacitor state
        self.charge = np.zeros(n_neurons, dtype=np.float32)
        self.refractory_counter = np.zeros(n_neurons, dtype=np.int32)

    def reset(self):
        self.state *= 0
        self.charge *= 0
        self.refractory_counter *= 0

    def forward(self, world, ticks=8):
        act = self.state.copy()
        Weff = self.W * self.mask

        for t in range(ticks):
            if t == 0:
                act[:self.V] = world

            raw = act @ Weff + act * 0.1

            if self.activation == 'leaky_relu':
                act = np.where(raw > 0, raw, np.float32(0.01) * raw)
                act = np.clip(act, -10, 10)

            elif self.activation == 'capacitor':
                # Original capacitor (no bio features)
                self.charge += raw * self.input_gain
                self.charge *= self.leak
                act = np.maximum(self.charge - self.threshold, 0.0)
                self.charge = np.clip(self.charge,
                                      -self.threshold * 2,
                                      self.threshold * 2)

            elif self.activation == 'bio_capacitor':
                # === Bio-tuned capacitor with AHP + refractory ===

                # 1. Refractory: neurons in refractory period get no input
                refractory_mask = (self.refractory_counter <= 0).astype(np.float32)
                self.refractory_counter = np.maximum(self.refractory_counter - 1, 0)

                # 2. Charge accumulation (only non-refractory neurons)
                self.charge += raw * self.input_gain * refractory_mask
                self.charge *= self.leak

                # 3. Detect firing (charge > threshold)
                fired = self.charge > self.threshold

                # 4. Output: activation for neurons above threshold
                act = np.maximum(self.charge - self.threshold, 0.0)

                # 5. AHP + refractory for neurons that fired
                if np.any(fired):
                    self.charge[fired] = self.ahp_depth  # reset below resting
                    self.refractory_counter[fired] = self.refractory_ticks

                # 6. Clamp charge
                self.charge = np.clip(self.charge,
                                      self.ahp_depth - 0.1,
                                      self.threshold * 2)

        self.state = act.copy()

        # Output: read charge for capacitor variants
        if self.activation in ('capacitor', 'bio_capacitor'):
            return self.charge[:self.V]
        return act[:self.V]

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

    def active_neuron_count(self):
        """Count neurons with non-negligible activation."""
        return int((np.abs(self.state) > 0.01).sum())

    def charge_stats(self):
        if self.activation in ('capacitor', 'bio_capacitor'):
            internal = self.charge[self.V:]
            active = int((np.abs(internal) > 0.01).sum())
            return {
                'mean': float(np.mean(internal)),
                'std': float(np.std(internal)),
                'active': active,
                'active_pct': float(active / len(internal) * 100),
            }
        return None

    def save_state(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(),
                self.addr.copy(), self.target_W.copy(), self.charge.copy(),
                self.refractory_counter.copy())

    def restore_state(self, s):
        self.W = s[0].copy()
        self.mask = s[1].copy()
        self.state = s[2].copy()
        self.addr = s[3].copy()
        self.target_W = s[4].copy()
        self.charge = s[5].copy()
        self.refractory_counter = s[6].copy()

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


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def train(net, inputs, targets, vocab, max_attempts=8000, ticks=8, label=""):

    def evaluate():
        net.reset()
        correct = 0
        for p in range(2):
            for i in range(len(inputs)):
                world = np.zeros(vocab, dtype=np.float32)
                world[inputs[i]] = 1.0
                logits = net.forward(world, ticks)
                probs = softmax(logits)
                if p == 1 and np.argmax(probs) == targets[i]:
                    correct += 1
        acc = correct / len(inputs)
        net.last_acc = acc
        return acc

    score = evaluate()
    best = score
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

        new_score = evaluate()

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best = max(best, score)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        if (att + 1) % 4000 == 0:
            cs = net.charge_stats()
            active_str = f" | Active: {cs['active']}/{net.N - net.V} ({cs['active_pct']:.0f}%)" if cs else ""
            print(f"  [{att+1:5d}] {label:30s} | Acc: {best*100:5.1f}%"
                  f" | Conns: {net.count_connections():4d}"
                  f" | Kept: {kept:3d}{active_str}")

        if best >= 0.99:
            break
        if stale >= 6000:
            break

    return best, kept


# === SWEEP CONFIGS ===

CONFIGS = {}

# Baseline: original capacitor (the current champion at 75%)
CONFIGS['cap_original'] = {
    'activation': 'capacitor',
    'threshold': 0.5, 'leak': 0.85,
    'input_gain': 0.3, 'ahp_depth': 0.0, 'refractory_ticks': 0,
    'ticks': 8,
}

# Baseline: leaky_relu
CONFIGS['leaky_relu'] = {
    'activation': 'leaky_relu',
    'threshold': 0.0, 'leak': 0.0,
    'input_gain': 0.0, 'ahp_depth': 0.0, 'refractory_ticks': 0,
    'ticks': 8,
}

# Bio-tuned sweep: AHP + refractory, gain scaled to network
# Key insight: ~11 inputs/neuron, so gain=0.15 means ~1.65 charge/tick
# With threshold=0.5 and leak=0.90, need ~4 ticks to fire
for gain in [0.10, 0.15, 0.20, 0.25]:
    for ahp in [-0.2, -0.3]:
        for refr in [1, 2]:
            name = f"bio_g{gain:.2f}_ahp{abs(ahp):.1f}_r{refr}"
            CONFIGS[name] = {
                'activation': 'bio_capacitor',
                'threshold': 0.5, 'leak': 0.90,
                'input_gain': gain, 'ahp_depth': ahp,
                'refractory_ticks': refr,
                'ticks': 10,
            }

# Also test bio with leak=0.85 (faster, like original)
for gain in [0.15, 0.20, 0.25]:
    name = f"bio_g{gain:.2f}_fast"
    CONFIGS[name] = {
        'activation': 'bio_capacitor',
        'threshold': 0.5, 'leak': 0.85,
        'input_gain': gain, 'ahp_depth': -0.2,
        'refractory_ticks': 1,
        'ticks': 8,
    }


def main():
    V = 16
    N = 80
    SEED = 42
    MAX_ATTEMPTS = 12000

    print("=" * 70)
    print("VRAXION v22 — Bio-Tuned Capacitor Sweep")
    print(f"  16-class lookup | {N} neurons | {MAX_ATTEMPTS} attempts | seed={SEED}")
    print(f"  {len(CONFIGS)} configurations to test")
    print("=" * 70)

    results = {}

    for name, cfg in CONFIGS.items():
        np.random.seed(SEED)
        random.seed(SEED)

        perm = np.random.permutation(V)
        inputs = list(range(V))
        ticks = cfg['ticks']

        net = SelfWiringGraph(
            N, V,
            activation=cfg['activation'],
            threshold=cfg['threshold'],
            leak=cfg['leak'],
            input_gain=cfg['input_gain'],
            ahp_depth=cfg['ahp_depth'],
            refractory_ticks=cfg['refractory_ticks'],
        )

        t0 = time.time()
        best_acc, total_kept = train(
            net, inputs, perm, V,
            max_attempts=MAX_ATTEMPTS, ticks=ticks, label=name,
        )
        elapsed = time.time() - t0

        cs = net.charge_stats()
        active_pct = cs['active_pct'] if cs else 'n/a'

        results[name] = {
            'accuracy': best_acc,
            'connections': net.count_connections(),
            'kept': total_kept,
            'time': round(elapsed, 1),
            'active_pct': active_pct,
            'config': cfg,
        }

        print(f"\n  >>> {name}: {best_acc*100:.1f}% | {elapsed:.1f}s | "
              f"active={active_pct}{'%' if isinstance(active_pct, float) else ''}\n")

    # === Summary ===
    print("\n" + "=" * 70)
    print("  SWEEP RESULTS — sorted by accuracy")
    print("=" * 70)
    print(f"  {'Config':<35s} {'Acc':>6s} {'Kept':>5s} {'Active':>7s} {'Time':>6s} {'Gain':>5s} {'AHP':>5s} {'Refr':>4s} {'Leak':>5s}")
    print(f"  {'-'*35} {'-'*6} {'-'*5} {'-'*7} {'-'*6} {'-'*5} {'-'*5} {'-'*4} {'-'*5}")

    sorted_r = sorted(results.items(), key=lambda x: (-x[1]['accuracy'], x[1]['time']))
    for name, r in sorted_r:
        cfg = r['config']
        act_str = f"{r['active_pct']:.0f}%" if isinstance(r['active_pct'], float) else r['active_pct']
        gain_str = f"{cfg['input_gain']:.2f}" if cfg['input_gain'] > 0 else "—"
        ahp_str = f"{cfg['ahp_depth']:.1f}" if cfg['ahp_depth'] != 0 else "—"
        refr_str = f"{cfg['refractory_ticks']}" if cfg['refractory_ticks'] > 0 else "—"
        leak_str = f"{cfg['leak']:.2f}" if cfg['leak'] > 0 else "—"
        print(f"  {name:<35s} {r['accuracy']*100:5.1f}% {r['kept']:5d} {act_str:>7s} "
              f"{r['time']:5.1f}s {gain_str:>5s} {ahp_str:>5s} {refr_str:>4s} {leak_str:>5s}")

    # Top 3
    print(f"\n  TOP 3:")
    for i, (name, r) in enumerate(sorted_r[:3]):
        print(f"    {i+1}. {name}: {r['accuracy']*100:.1f}%")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"/home/user/VRAXION/v4/research/v22/tests/bio_sweep_results_{timestamp}.json"
    json_results = {name: {k: v for k, v in r.items() if k != 'config'}
                    for name, r in results.items()}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to: {json_path}")


if __name__ == "__main__":
    main()
