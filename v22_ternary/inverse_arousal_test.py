"""
VRAXION v22 — Inverse Arousal Self-Wiring Test
===============================================
Inverse arousal: when accuracy is LOW, self-wire LESS (let mutation explore).
When accuracy is HIGH, self-wire MORE (targeted exploitation).

Low acc → top_k=2, max_new=1  (mostly mutation)
High acc → top_k=5, max_new=3  (targeted self-wiring)

This is the explore→exploit transition: mutation explores when you know nothing,
self-wiring exploits when activations are informative.

Offline 16-class result: inverse reached 100% @ 1000 attempts, 3555 connections
(half the baseline). Testing at 64-class / 320 neurons.

Author: Daniel (researcher) + Claude (advisor)
Date: 2026-03-12
"""

import numpy as np
import time
import random


class SelfWiringNet:
    """Self-wiring network with inverse arousal."""

    def __init__(self, n_neurons=160, n_in=32, n_out=32, n_ticks=8,
                 decay=0.5, activation='leaky_relu'):
        self.n_neurons = n_neurons
        self.n_in = n_in
        self.n_out = n_out
        self.n_ticks = n_ticks
        self.decay = decay
        self.activation = activation

        # Inverse arousal params (will be set based on accuracy)
        self.top_k_wire = 2  # start low (low accuracy assumed)
        self.max_new_per_wire = 1

        # 3D+1D addresses
        self.addresses = np.zeros((n_neurons, 4), dtype=np.float32)
        self.addresses[:, :3] = np.random.rand(n_neurons, 3).astype(np.float32)
        for i in range(n_neurons):
            if i < n_in:
                self.addresses[i, 3] = 0.0
            elif i >= n_neurons - n_out:
                self.addresses[i, 3] = 1.0
            else:
                self.addresses[i, 3] = 0.5

        self.target_W = np.random.randn(n_neurons, 4).astype(np.float32) * 0.2

        # Connections
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        self._init_connections()
        self.W = np.random.randn(n_neurons, n_neurons).astype(np.float32) * 0.3 + 0.2
        self.W *= self.mask

        self._effective_W = self.W * self.mask
        self._effective_W_dirty = False

        self.state = np.zeros(n_neurons, dtype=np.float32)
        self._last_activations = np.zeros(n_neurons, dtype=np.float32)

    def _init_connections(self):
        n = self.n_neurons
        n_in = self.n_in
        n_out = self.n_out
        i_start = n_in
        i_end = n - n_out
        n_int = i_end - i_start
        if n_int == 0:
            return

        for i in range(n_in):
            for j in random.sample(range(i_start, i_end), min(3, n_int)):
                self.mask[i, j] = 1.0

        for i in range(i_start, i_end):
            others = [x for x in range(i_start, i_end) if x != i]
            if others:
                for j in random.sample(others, min(2, len(others))):
                    self.mask[i, j] = 1.0
            for _ in range(2):
                self.mask[i, random.randint(n - n_out, n - 1)] = 1.0

        for _ in range(int(n * n * 0.02)):
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i != j:
                self.mask[i, j] = 1.0

    def _activate(self, raw):
        if self.activation == 'leaky_relu':
            return np.where(raw > 0, raw, 0.01 * raw)
        return raw

    def set_arousal(self, accuracy):
        """Inverse arousal: low acc → less wiring, high acc → more wiring."""
        if accuracy < 0.3:
            self.top_k_wire = 2
            self.max_new_per_wire = 1
        elif accuracy < 0.6:
            self.top_k_wire = 3
            self.max_new_per_wire = 2
        else:
            self.top_k_wire = 5
            self.max_new_per_wire = 3

    def forward_batch(self, input_batch):
        B = input_batch.shape[0]
        states = np.zeros((B, self.n_neurons), dtype=np.float32)

        if self._effective_W_dirty:
            self._effective_W = self.W * self.mask
            self._effective_W_dirty = False
        eW = self._effective_W

        for tick in range(self.n_ticks):
            np.maximum(states[:, :self.n_in], input_batch,
                       out=states[:, :self.n_in])
            activated = self._activate(states)
            incoming = activated @ eW
            states = states * self.decay + incoming

        return states[:, -self.n_out:]

    def forward_single(self, input_vec):
        self.state = np.zeros(self.n_neurons, dtype=np.float32)

        if self._effective_W_dirty:
            self._effective_W = self.W * self.mask
            self._effective_W_dirty = False
        eW = self._effective_W

        for tick in range(self.n_ticks):
            np.maximum(self.state[:self.n_in], input_vec,
                       out=self.state[:self.n_in])
            activated = self._activate(self.state)
            incoming = activated @ eW
            self.state = self.state * self.decay + incoming

        self._last_activations = self._activate(self.state)
        return self.state[-self.n_out:]

    def self_wire(self):
        """Self-wire with arousal-controlled intensity."""
        activated = self._last_activations
        active_mask = np.abs(activated) > 0.01
        n_active = active_mask.sum()
        if n_active == 0:
            return 0

        active_states = np.abs(self.state) * active_mask
        k = min(self.top_k_wire, int(n_active))
        if k == 0:
            return 0
        top_indices = np.argpartition(active_states, -k)[-k:]

        added = 0
        for i in top_indices:
            if added >= self.max_new_per_wire:
                break
            target_addr = self.addresses[i] + np.abs(activated[i]) * self.target_W[i]
            dists = np.linalg.norm(self.addresses - target_addr[np.newaxis, :], axis=1)
            dists[i] = np.inf

            for _ in range(5):
                j = dists.argmin()
                if self.mask[i, j] == 0:
                    self.mask[i, j] = 1.0
                    self.W[i, j] = np.random.randn() * 0.15
                    self._effective_W_dirty = True
                    added += 1
                    break
                dists[j] = np.inf

        return added

    def snapshot(self):
        return {
            'W': self.W.copy(),
            'mask': self.mask.copy(),
            'target_W': self.target_W.copy(),
        }

    def restore(self, snap):
        self.W = snap['W'].copy()
        self.mask = snap['mask'].copy()
        self.target_W = snap['target_W'].copy()
        self._effective_W_dirty = True


def make_task(n_classes):
    indices = list(range(n_classes))
    targets = indices.copy()
    random.shuffle(targets)
    inp = np.zeros((n_classes, n_classes), dtype=np.float32)
    for i in range(n_classes):
        inp[i, indices[i]] = 1.0
    return inp, targets


def evaluate(net, inp, targets):
    out = net.forward_batch(inp)
    preds = out.argmax(axis=1)
    return sum(1 for p, t in zip(preds, targets) if p == t) / len(targets)


def hill_climb(net, inp, targets, current_acc, n_mutations=20):
    n = net.n_neurons
    best_acc = current_acc

    active = np.argwhere(net.mask > 0)
    if len(active) == 0:
        return best_acc

    n_active = len(active)

    for trial in range(n_mutations):
        snap = net.snapshot()
        mut_type = trial % 4

        if mut_type == 0:
            k = max(1, n_active // 20)
            chosen = active[np.random.choice(n_active, min(k, n_active), replace=False)]
            for r, c in chosen:
                net.W[r, c] += np.random.randn() * 0.15
        elif mut_type == 1:
            k = max(1, n_active // 5)
            chosen = active[np.random.choice(n_active, min(k, n_active), replace=False)]
            for r, c in chosen:
                net.W[r, c] += np.random.randn() * 0.3
        elif mut_type == 2:
            idx = np.random.randint(n_active)
            r, c = active[idx]
            net.W[r, c] = np.random.randn() * 0.5
        else:
            active_W = np.abs(net.W) * net.mask
            active_W[net.mask == 0] = np.inf
            flat = active_W.flatten()
            worst = flat.argmin()
            if flat[worst] < np.inf:
                r, c = divmod(worst, n)
                net.mask[r, c] = 0.0
                net.W[r, c] = 0.0

            for _ in range(10):
                i, j = random.randint(0, n-1), random.randint(0, n-1)
                if i != j and net.mask[i, j] == 0:
                    net.mask[i, j] = 1.0
                    net.W[i, j] = np.random.randn() * 0.3
                    break

        np.clip(net.W, -2.0, 2.0, out=net.W)
        net.W *= net.mask
        net._effective_W_dirty = True

        acc = evaluate(net, inp, targets)
        if acc >= best_acc:
            best_acc = acc
            if mut_type == 3:
                active = np.argwhere(net.mask > 0)
                n_active = len(active)
        else:
            net.restore(snap)

    return best_acc


def run_test(n_classes, n_neurons, max_attempts, seed=42, arousal_mode='inverse'):
    np.random.seed(seed)
    random.seed(seed)

    label = f"{arousal_mode} arousal, {n_classes}-class, {n_neurons}n"
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    net = SelfWiringNet(
        n_neurons=n_neurons, n_in=n_classes, n_out=n_classes,
        n_ticks=8, decay=0.5, activation='leaky_relu',
    )

    inp, targets = make_task(n_classes)

    acc = evaluate(net, inp, targets)
    best_acc = acc
    best_step = 0
    n_conns = int(net.mask.sum())

    print(f"  Init: acc={acc*100:.1f}%, conns={n_conns}")

    start = time.time()
    step = 0
    stale = 0
    log_interval = 2000

    while step < max_attempts:
        # Set arousal based on current accuracy
        if arousal_mode == 'inverse':
            net.set_arousal(acc)
        elif arousal_mode == 'none':
            net.top_k_wire = 5
            net.max_new_per_wire = 3

        # Hill-climbing mutation
        new_acc = hill_climb(net, inp, targets, acc, n_mutations=20)
        step += 20

        if new_acc > best_acc:
            best_acc = new_acc
            best_step = step
            stale = 0
        else:
            stale += 20

        acc = new_acc

        # Self-wire (arousal-controlled intensity)
        if step % 100 == 0:
            n_wire_samples = min(8, n_classes)
            for i in range(n_wire_samples):
                iv = np.zeros(n_classes, dtype=np.float32)
                iv[i] = 1.0
                net.forward_single(iv)
                net.self_wire()

        # Log
        if step % log_interval == 0:
            elapsed = time.time() - start
            n_conns = int(net.mask.sum())
            print(f"  [{step:>6d}] acc={acc*100:>5.1f}%  "
                  f"best={best_acc*100:.1f}%(@{best_step})  "
                  f"conns={n_conns}  top_k={net.top_k_wire}  "
                  f"stale={stale}  t={elapsed:.1f}s")

        # Stale recovery
        if stale >= 8000:
            active = net.mask > 0
            noise = np.random.randn(*net.W.shape).astype(np.float32) * 0.2
            net.W += noise * active
            np.clip(net.W, -2.0, 2.0, out=net.W)
            net.W *= net.mask
            net._effective_W_dirty = True
            stale = 0

        if acc >= 1.0:
            elapsed = time.time() - start
            n_conns = int(net.mask.sum())
            print(f"  SUCCESS! 100% @ step {step}, conns={n_conns}, time={elapsed:.1f}s")
            return {
                'label': label, 'solved': True, 'step': step,
                'conns': n_conns, 'time': elapsed, 'best_acc': 1.0,
                'n_classes': n_classes, 'arousal': arousal_mode,
            }

    elapsed = time.time() - start
    n_conns = int(net.mask.sum())
    print(f"  MAX ATTEMPTS. Best: {best_acc*100:.1f}%(@{best_step}), "
          f"conns={n_conns}, time={elapsed:.1f}s")
    return {
        'label': label, 'solved': False, 'step': max_attempts,
        'conns': n_conns, 'time': elapsed, 'best_acc': best_acc,
        'n_classes': n_classes, 'arousal': arousal_mode,
    }


def main():
    print("VRAXION v22 — Inverse Arousal Self-Wiring Test")
    print("=" * 70)

    results = []

    # --- 32-class comparison: inverse vs no-arousal ---
    print("\n### 32-class comparison ###")
    r = run_test(32, 160, 30_000, seed=42, arousal_mode='inverse')
    results.append(r)
    r = run_test(32, 160, 30_000, seed=42, arousal_mode='none')
    results.append(r)

    # --- 64-class with inverse arousal ---
    print("\n### 64-class with inverse arousal ###")
    r = run_test(64, 320, 60_000, seed=42, arousal_mode='inverse')
    results.append(r)
    r = run_test(64, 320, 60_000, seed=42, arousal_mode='none')
    results.append(r)

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<45} {'Solved':>6} {'Best':>8} {'Conns':>6} {'Time':>7}")
    print("-" * 75)
    for r in results:
        print(f"{r['label']:<45} {'YES' if r['solved'] else 'NO':>6} "
              f"{r['best_acc']*100:>7.1f}% {r['conns']:>6} {r['time']:>6.1f}s")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    results = main()
