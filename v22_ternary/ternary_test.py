"""
VRAXION v22 — Ternary Activation Test
======================================
Tests ternary_soft activation vs leaky_relu baseline in v18 self-wiring framework.

Ternary_soft:
  Internal neurons: +1 if raw > thresh, -1 if raw < -thresh, else 0
  Input neurons: raw input value
  Output neurons: leaky_relu (softmax needs continuous)

Pure hill-climbing mutation (v18 framework). No reward learning.

Author: Daniel (researcher) + Claude (advisor)
Date: 2026-03-12
"""

import numpy as np
import time
import random


class SelfWiringNet:
    """Self-wiring network with configurable activation."""

    def __init__(self, n_neurons=160, n_in=32, n_out=32, n_ticks=8,
                 decay=0.5, top_k_wire=5,
                 activation='leaky_relu', ternary_thresh=0.3):
        self.n_neurons = n_neurons
        self.n_in = n_in
        self.n_out = n_out
        self.n_ticks = n_ticks
        self.decay = decay
        self.top_k_wire = top_k_wire
        self.activation = activation
        self.ternary_thresh = ternary_thresh

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

        # Ternary distribution tracking
        self._ternary_pos = 0
        self._ternary_neg = 0
        self._ternary_zero = 0
        self._ternary_samples = 0

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
        """Apply activation function."""
        result = raw.copy()
        i_start = self.n_in
        i_end = self.n_neurons - self.n_out

        if self.activation == 'leaky_relu':
            # All neurons: leaky_relu
            result = np.where(raw > 0, raw, 0.01 * raw)

        elif self.activation == 'ternary_soft':
            # Internal neurons: ternary
            int_raw = raw[i_start:i_end]
            ternary = np.zeros_like(int_raw)
            ternary[int_raw > self.ternary_thresh] = 1.0
            ternary[int_raw < -self.ternary_thresh] = -1.0
            result[i_start:i_end] = ternary

            # Track distribution
            self._ternary_pos += (ternary > 0).sum()
            self._ternary_neg += (ternary < 0).sum()
            self._ternary_zero += (ternary == 0).sum()
            self._ternary_samples += 1

            # Output neurons: leaky_relu
            out_raw = raw[-self.n_out:]
            result[-self.n_out:] = np.where(out_raw > 0, out_raw, 0.01 * out_raw)

        # Input neurons: keep raw (injected directly)
        return result

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

            # Activate per-sample
            activated = np.zeros_like(states)
            for b in range(B):
                activated[b] = self._activate(states[b])

            incoming = activated @ eW
            states = states * self.decay + incoming

        return states[:, -self.n_out:]

    def forward_single(self, input_vec):
        """Single sample forward for self-wiring."""
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
        activated = self._last_activations
        active_mask = np.abs(activated) > 0.01
        n_active = active_mask.sum()
        if n_active == 0:
            return

        active_states = np.abs(self.state) * active_mask
        k = min(self.top_k_wire, int(n_active))
        top_indices = np.argpartition(active_states, -k)[-k:]

        for i in top_indices:
            target_addr = self.addresses[i] + np.abs(activated[i]) * self.target_W[i]
            dists = np.linalg.norm(self.addresses - target_addr[np.newaxis, :], axis=1)
            dists[i] = np.inf

            for _ in range(5):
                j = dists.argmin()
                if self.mask[i, j] == 0:
                    self.mask[i, j] = 1.0
                    self.W[i, j] = np.random.randn() * 0.15
                    self._effective_W_dirty = True
                    break
                dists[j] = np.inf

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

    def get_ternary_dist(self):
        """Get ternary distribution stats and reset."""
        if self._ternary_samples == 0:
            return {'pos': 0, 'neg': 0, 'zero': 0}
        result = {
            'pos': self._ternary_pos / self._ternary_samples,
            'neg': self._ternary_neg / self._ternary_samples,
            'zero': self._ternary_zero / self._ternary_samples,
        }
        self._ternary_pos = 0
        self._ternary_neg = 0
        self._ternary_zero = 0
        self._ternary_samples = 0
        return result


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
            # Small perturbation
            k = max(1, n_active // 20)
            chosen = active[np.random.choice(n_active, min(k, n_active), replace=False)]
            for r, c in chosen:
                net.W[r, c] += np.random.randn() * 0.15
        elif mut_type == 1:
            # Medium perturbation
            k = max(1, n_active // 5)
            chosen = active[np.random.choice(n_active, min(k, n_active), replace=False)]
            for r, c in chosen:
                net.W[r, c] += np.random.randn() * 0.3
        elif mut_type == 2:
            # Single weight reset
            idx = np.random.randint(n_active)
            r, c = active[idx]
            net.W[r, c] = np.random.randn() * 0.5
        else:
            # Structure: swap weakest → new
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


def run_experiment(activation, ternary_thresh, n_classes=32, n_neurons=160,
                   max_attempts=30_000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    label = activation
    if activation == 'ternary_soft':
        label = f'ternary_soft t={ternary_thresh}'

    print(f"\n{'='*70}")
    print(f"  {label}  |  {n_classes}-class, {n_neurons} neurons")
    print(f"{'='*70}")

    net = SelfWiringNet(
        n_neurons=n_neurons, n_in=n_classes, n_out=n_classes,
        n_ticks=8, decay=0.5, top_k_wire=5,
        activation=activation, ternary_thresh=ternary_thresh,
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

    # Self-wire a few initial rounds
    for _ in range(10):
        for i in range(n_classes):
            iv = np.zeros(n_classes, dtype=np.float32)
            iv[i] = 1.0
            net.forward_single(iv)
            net.self_wire()

    acc = evaluate(net, inp, targets)
    n_conns = int(net.mask.sum())
    print(f"  After wiring: acc={acc*100:.1f}%, conns={n_conns}")

    while step < max_attempts:
        # Hill-climbing mutation batch
        new_acc = hill_climb(net, inp, targets, acc, n_mutations=20)
        step += 20  # each mutation = 1 attempt

        if new_acc > best_acc:
            best_acc = new_acc
            best_step = step
            stale = 0
        else:
            stale += 20

        acc = new_acc

        # Self-wire occasionally
        if step % 200 == 0:
            for i in range(min(8, n_classes)):
                iv = np.zeros(n_classes, dtype=np.float32)
                iv[i] = 1.0
                net.forward_single(iv)
                net.self_wire()

        # Log
        if step % log_interval == 0:
            elapsed = time.time() - start
            n_conns = int(net.mask.sum())
            tern = net.get_ternary_dist()

            tern_str = ""
            if activation == 'ternary_soft':
                tern_str = (f"  ternary[+:{tern['pos']:.1f} "
                           f"-:{tern['neg']:.1f} "
                           f"0:{tern['zero']:.1f}]")

            print(f"  [{step:>6d}] acc={acc*100:>5.1f}%  "
                  f"best={best_acc*100:.1f}%(@{best_step})  "
                  f"conns={n_conns}  stale={stale}{tern_str}  "
                  f"t={elapsed:.1f}s")

        # Stale recovery
        if stale >= 8000:
            # Bigger perturbation
            active = net.mask > 0
            noise = np.random.randn(*net.W.shape).astype(np.float32) * 0.2
            net.W += noise * active
            np.clip(net.W, -2.0, 2.0, out=net.W)
            net.W *= net.mask
            net._effective_W_dirty = True
            stale = 0

        # SUCCESS
        if acc >= 1.0:
            elapsed = time.time() - start
            n_conns = int(net.mask.sum())
            print(f"  SUCCESS! 100% @ step {step}, conns={n_conns}, time={elapsed:.1f}s")
            return {
                'label': label, 'solved': True, 'step': step,
                'conns': n_conns, 'time': elapsed, 'best_acc': 1.0,
                'n_classes': n_classes,
            }

    elapsed = time.time() - start
    n_conns = int(net.mask.sum())
    print(f"  MAX ATTEMPTS. Best: {best_acc*100:.1f}%(@{best_step}), "
          f"conns={n_conns}, time={elapsed:.1f}s")
    return {
        'label': label, 'solved': False, 'step': max_attempts,
        'conns': n_conns, 'time': elapsed, 'best_acc': best_acc,
        'n_classes': n_classes,
    }


def main():
    print("VRAXION v22 — Ternary Activation Test")
    print("32-class / 160 neurons / max 30k attempts / seed=42\n")

    configs = [
        ('leaky_relu', 0.0),
        ('ternary_soft', 0.1),
        ('ternary_soft', 0.3),
        ('ternary_soft', 0.5),
        ('ternary_soft', 1.0),
    ]

    results = []
    for act, thresh in configs:
        r = run_experiment(act, thresh, n_classes=32, n_neurons=160,
                           max_attempts=30_000, seed=42)
        results.append(r)

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY — 32-class / 160 neurons")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'Solved':>6} {'Best':>8} {'Step':>8} {'Conns':>6} {'Time':>6}")
    print("-" * 65)
    for r in results:
        print(f"{r['label']:<25} {'YES' if r['solved'] else 'NO':>6} "
              f"{r['best_acc']*100:>7.1f}% {r['step']:>8} "
              f"{r['conns']:>6} {r['time']:>5.1f}s")
    print(f"{'='*70}")

    # If any ternary_soft solved AND has fewer conns than leaky_relu, run 64-class
    lr_result = results[0]
    winners = [r for r in results[1:]
               if r['solved'] and r['conns'] < lr_result.get('conns', 99999)]

    if winners:
        print(f"\n\nTernary winners found! Running 64-class test...")
        for w in winners:
            act = 'ternary_soft'
            thresh = float(w['label'].split('=')[1])
            r64 = run_experiment(act, thresh, n_classes=64, n_neurons=320,
                                 max_attempts=60_000, seed=42)
            results.append(r64)

            # Also run leaky_relu 64-class baseline
            r64_lr = run_experiment('leaky_relu', 0.0, n_classes=64, n_neurons=320,
                                    max_attempts=60_000, seed=42)
            results.append(r64_lr)

            print(f"\n64-class comparison:")
            print(f"  {r64['label']}: solved={r64['solved']}, "
                  f"best={r64['best_acc']*100:.1f}%, conns={r64['conns']}")
            print(f"  leaky_relu:  solved={r64_lr['solved']}, "
                  f"best={r64_lr['best_acc']*100:.1f}%, conns={r64_lr['conns']}")

    return results


if __name__ == '__main__':
    results = main()
