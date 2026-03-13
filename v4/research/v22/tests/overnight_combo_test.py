"""
OVERNIGHT COMBO TEST — All winning ideas combined
===================================================
Phase 1: Baseline (shared I/O)
Phase 2: Split I/O
Phase 3: Split I/O + Highway
Phase 4: Chain Ensemble
Phase 5: Rebalanced Ensemble
Phase 6: Pop_2
Phase 7: Mega Combo

Crash protection: each test in try/except, results saved after each.
"""

import numpy as np
import math
import random
import time
import traceback
import json
import os
from datetime import datetime

# ============================================================
# Constants
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]
TICKS = 6
INTERNAL = 64

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'overnight_results.json')
PROGRESS_FILE = os.path.join(SCRIPT_DIR, 'overnight_progress.txt')

results = {}


# ============================================================
# Infrastructure
# ============================================================

def log_progress(msg):
    """Write to progress file immediately."""
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    with open(PROGRESS_FILE, 'a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def save_results():
    """Save all results to JSON."""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def safe_test(name, fn):
    """Crash-safe test wrapper."""
    log_progress(f"START: {name}")
    try:
        result = fn()
        results[name] = result
        save_results()
        log_progress(f"DONE: {name} -> acc={result.get('avg_acc', '?')}")
    except Exception as e:
        log_progress(f"CRASH: {name} -> {traceback.format_exc()}")
        results[name] = {'error': str(e), 'traceback': traceback.format_exc()}
        save_results()
    time.sleep(2)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ============================================================
# SharedIONet — baseline (first V neurons = input AND output)
# ============================================================

class SharedIONet:
    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30,
                 threshold=0.5, leak=0.85):
        self.N = n_neurons
        self.V = vocab
        self.flip_rate = flip_rate
        self.threshold = threshold
        self.leak = leak

        r = np.random.rand(n_neurons, n_neurons)
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        self.mask[r < density / 2] = -1
        self.mask[r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        self.W = np.where(
            np.random.rand(n_neurons, n_neurons) > 0.5,
            np.float32(0.5), np.float32(1.5))

        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.charge = np.zeros(n_neurons, dtype=np.float32)

    def reset(self):
        self.state *= 0
        self.charge *= 0

    def forward(self, world, ticks=6):
        act = self.state.copy()
        Weff = self.W * self.mask
        for t in range(ticks):
            if t == 0:
                act[:self.V] = world
            raw = act @ Weff + act * 0.1
            self.charge += raw * 0.3
            self.charge *= self.leak
            act = np.maximum(self.charge - self.threshold, 0.0)
            self.charge = np.clip(self.charge, -self.threshold * 2, self.threshold * 2)
        self.state = act.copy()
        return self.charge[:self.V]

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

    def save_state(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(), self.charge.copy())

    def restore_state(self, s):
        self.W, self.mask, self.state, self.charge = s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy()

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


# ============================================================
# SplitIONet — separate input/output + optional highway
# ============================================================

class SplitIONet:
    def __init__(self, vocab, internal=64, hw_size=0, density=0.06,
                 flip_rate=0.30, threshold=0.5, leak=0.85):
        self.V = vocab
        self.internal = internal
        self.hw_size = hw_size
        self.flip_rate = flip_rate
        self.threshold = threshold
        self.leak = leak

        # Layout: [input V][internal][hw_fw × hw_size][hw_bw × hw_size][output V]
        self.N = vocab + internal + hw_size * 2 + vocab
        self.out_start = self.N - vocab
        N = self.N

        r = np.random.rand(N, N)
        self.mask = np.zeros((N, N), dtype=np.float32)
        self.mask[r < density / 2] = -1
        self.mask[r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        self.W = np.where(
            np.random.rand(N, N) > 0.5,
            np.float32(0.5), np.float32(1.5))

        self.state = np.zeros(N, dtype=np.float32)
        self.charge = np.zeros(N, dtype=np.float32)

        # Highway indices to protect from mutation
        self._hw_connections = set()
        if hw_size > 0:
            self._add_highway()

    def _add_highway(self):
        """Add dedicated forward + backward highway channels."""
        V = self.V
        hw_fw_start = V + self.internal
        hw_bw_start = hw_fw_start + self.hw_size
        out_start = self.out_start

        # Forward chain: input → hw_fw[0] → ... → hw_fw[n-1] → output
        for h in range(self.hw_size - 1):
            self.mask[hw_fw_start + h, hw_fw_start + h + 1] = 1.0
            self.W[hw_fw_start + h, hw_fw_start + h + 1] = 1.5
            self._hw_connections.add((hw_fw_start + h, hw_fw_start + h + 1))
        # input → first hw_fw
        for i in range(V):
            self.mask[i, hw_fw_start] = 1.0
            self.W[i, hw_fw_start] = 1.5
            self._hw_connections.add((i, hw_fw_start))
        # last hw_fw → output
        for o in range(out_start, self.N):
            self.mask[hw_fw_start + self.hw_size - 1, o] = 1.0
            self.W[hw_fw_start + self.hw_size - 1, o] = 1.5
            self._hw_connections.add((hw_fw_start + self.hw_size - 1, o))

        # Backward chain: output → hw_bw[0] → ... → hw_bw[n-1] → input
        for h in range(self.hw_size - 1):
            self.mask[hw_bw_start + h, hw_bw_start + h + 1] = 1.0
            self.W[hw_bw_start + h, hw_bw_start + h + 1] = 1.5
            self._hw_connections.add((hw_bw_start + h, hw_bw_start + h + 1))
        for o in range(out_start, self.N):
            self.mask[o, hw_bw_start] = 1.0
            self.W[o, hw_bw_start] = 1.5
            self._hw_connections.add((o, hw_bw_start))
        for i in range(V):
            self.mask[hw_bw_start + self.hw_size - 1, i] = 1.0
            self.W[hw_bw_start + self.hw_size - 1, i] = 1.5
            self._hw_connections.add((hw_bw_start + self.hw_size - 1, i))

    def reset(self):
        self.state *= 0
        self.charge *= 0

    def forward(self, world, ticks=6):
        act = self.state.copy()
        Weff = self.W * self.mask
        for t in range(ticks):
            if t == 0:
                act[:self.V] = world
            raw = act @ Weff + act * 0.1
            self.charge += raw * 0.3
            self.charge *= self.leak
            act = np.maximum(self.charge - self.threshold, 0.0)
            self.charge = np.clip(self.charge, -self.threshold * 2, self.threshold * 2)
        self.state = act.copy()
        return self.charge[self.out_start:]

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

    def save_state(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(), self.charge.copy())

    def restore_state(self, s):
        self.W, self.mask, self.state, self.charge = s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy()

    def _is_highway(self, r, c):
        return (r, c) in self._hw_connections

    def mutate_structure(self, rate=0.05):
        r = random.random()
        if r < self.flip_rate:
            alive = np.argwhere(self.mask != 0)
            if len(alive) > 0:
                n = max(1, int(len(alive) * rate * 0.5))
                idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                for j in range(len(idx)):
                    r2, c = int(idx[j][0]), int(idx[j][1])
                    if not self._is_highway(r2, c):
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
                        r2, c = int(idx[j][0]), int(idx[j][1])
                        if not self._is_highway(r2, c):
                            self.mask[r2, c] = 0
            else:  # rewire
                alive = np.argwhere(self.mask != 0)
                if len(alive) > 0:
                    n = max(1, int(len(alive) * rate))
                    idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                    for j in range(len(idx)):
                        r2, c = int(idx[j][0]), int(idx[j][1])
                        if self._is_highway(r2, c):
                            continue
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
                if not self._is_highway(r2, c):
                    self.W[r2, c] = np.float32(1.5) if self.W[r2, c] < 1.0 else np.float32(0.5)


# ============================================================
# Scoring & Evaluation
# ============================================================

def combined_score(net, perm, V, ticks=TICKS):
    """2-pass eval with combined scoring: 0.5*acc + 0.5*target_prob."""
    net.reset()
    correct = 0
    target_prob = 0.0
    for p in range(2):
        for i in range(V):
            world = np.zeros(V, dtype=np.float32)
            world[i] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits[:V])
            if p == 1:
                if np.argmax(probs) == perm[i]:
                    correct += 1
                target_prob += probs[perm[i]]
    acc = correct / V
    tp = target_prob / V
    return 0.5 * acc + 0.5 * tp, acc


# ============================================================
# Specialist Mutations
# ============================================================

def mutate_refiner(net):
    """50% weight toggle, 50% single flip."""
    if random.random() < 0.5:
        net.mutate_weights()
    else:
        alive = np.argwhere(net.mask != 0)
        if len(alive) > 0:
            j = alive[np.random.choice(len(alive))]
            r, c = int(j[0]), int(j[1])
            hw = hasattr(net, '_hw_connections') and (r, c) in net._hw_connections
            if not hw:
                net.mask[r, c] *= -1


def mutate_refiner_aggr(net):
    """1-2 flips + weight toggle."""
    alive = np.argwhere(net.mask != 0)
    if len(alive) > 0:
        n_flips = random.randint(1, 2)
        idx = alive[np.random.choice(len(alive), min(n_flips, len(alive)), replace=False)]
        for j in range(len(idx)):
            r, c = int(idx[j][0]), int(idx[j][1])
            hw = hasattr(net, '_hw_connections') and (r, c) in net._hw_connections
            if not hw:
                net.mask[r, c] *= -1
    net.mutate_weights()


def mutate_rewirer(net):
    """2x connection rewire."""
    alive = np.argwhere(net.mask != 0)
    if len(alive) > 0:
        n = min(2, len(alive))
        idx = alive[np.random.choice(len(alive), n, replace=False)]
        for j in range(len(idx)):
            r, c = int(idx[j][0]), int(idx[j][1])
            hw = hasattr(net, '_hw_connections') and (r, c) in net._hw_connections
            if hw:
                continue
            old_sign = net.mask[r, c]
            old_w = net.W[r, c]
            net.mask[r, c] = 0
            nc = random.randint(0, net.N - 1)
            while nc == r:
                nc = random.randint(0, net.N - 1)
            net.mask[r, nc] = old_sign
            net.W[r, nc] = old_w


def mutate_scout(net):
    """3x random add/flip (exploration)."""
    for _ in range(3):
        if random.random() < 0.5:
            # random add
            dead = np.argwhere(net.mask == 0)
            dead = dead[dead[:, 0] != dead[:, 1]]
            if len(dead) > 0:
                j = dead[np.random.choice(len(dead))]
                r, c = int(j[0]), int(j[1])
                net.mask[r, c] = random.choice([-1.0, 1.0])
                net.W[r, c] = random.choice([np.float32(0.5), np.float32(1.5)])
        else:
            # random flip
            alive = np.argwhere(net.mask != 0)
            if len(alive) > 0:
                j = alive[np.random.choice(len(alive))]
                r, c = int(j[0]), int(j[1])
                hw = hasattr(net, '_hw_connections') and (r, c) in net._hw_connections
                if not hw:
                    net.mask[r, c] *= -1


SPECIALISTS = {
    'refiner': mutate_refiner,
    'refiner_aggr': mutate_refiner_aggr,
    'rewirer': mutate_rewirer,
    'scout': mutate_scout,
}


# ============================================================
# Training Functions
# ============================================================

def train_basic(net, perm, V, max_attempts=8000, ticks=TICKS):
    """Standard (1+1)-ES with combined scoring."""
    score, acc = combined_score(net, perm, V, ticks)
    best_acc = acc
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

        new_score, new_acc = combined_score(net, perm, V, ticks)
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0
        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    return best_acc, score, kept, net.count_connections()


def train_pop2(net, perm, V, max_attempts=16000, ticks=TICKS):
    """Pop_2: try 2 mutations, keep better one."""
    score, acc = combined_score(net, perm, V, ticks)
    best_acc = acc
    phase = "STRUCTURE"
    kept = 0
    stale = 0
    switched = False

    for att in range(max_attempts):
        state = net.save_state()

        # Candidate 1
        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()
        s1, a1 = combined_score(net, perm, V, ticks)
        state1 = net.save_state()

        # Candidate 2
        net.restore_state(state)
        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()
        s2, a2 = combined_score(net, perm, V, ticks)

        # Pick better
        if s1 >= s2:
            best_s, best_a = s1, a1
            if s1 > score:
                net.restore_state(state1)
            else:
                net.restore_state(state)
        else:
            best_s, best_a = s2, a2
            if s2 <= score:
                net.restore_state(state)

        if best_s > score:
            score = best_s
            kept += 1
            stale = 0
            best_acc = max(best_acc, max(a1, a2))
        else:
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0
        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    return best_acc, score, kept, net.count_connections()


def train_ensemble(net, perm, V, rounds=4, att_per_specialist=4000,
                   ticks=TICKS, rebalanced=False):
    """Chain ensemble: 4 specialists, jackpot broadcast."""
    score, acc = combined_score(net, perm, V, ticks)
    best_acc = acc
    best_state = net.save_state()

    spec_names = list(SPECIALISTS.keys())
    win_counts = {s: 0 for s in spec_names}
    round_history = []

    total_budget = att_per_specialist * len(spec_names)

    for rnd in range(rounds):
        # Allocate attempts per specialist
        if rebalanced and rnd > 0:
            total_wins = sum(win_counts.values())
            if total_wins > 0:
                alloc = {}
                for s in spec_names:
                    ratio = max(0.1, win_counts[s] / total_wins)
                    alloc[s] = ratio
                total_r = sum(alloc.values())
                alloc = {s: int(total_budget * alloc[s] / total_r) for s in spec_names}
            else:
                alloc = {s: att_per_specialist for s in spec_names}
        else:
            alloc = {s: att_per_specialist for s in spec_names}

        round_best_score = score
        round_best_state = best_state
        round_winner = None

        for spec_name in spec_names:
            spec_fn = SPECIALISTS[spec_name]
            n_att = alloc[spec_name]

            net.restore_state(best_state)
            s_score = score
            s_kept = 0

            for att in range(n_att):
                state = net.save_state()
                spec_fn(net)
                new_s, new_a = combined_score(net, perm, V, ticks)
                if new_s > s_score:
                    s_score = new_s
                    s_kept += 1
                    if new_a > best_acc:
                        best_acc = new_a
                else:
                    net.restore_state(state)

            if s_score > round_best_score:
                round_best_score = s_score
                round_best_state = net.save_state()
                round_winner = spec_name

            win_counts[spec_name] += s_kept

        # Jackpot broadcast: best specialist wins
        if round_winner:
            best_state = round_best_state
            score = round_best_score
            log_progress(f"  Round {rnd+1}: winner={round_winner} "
                         f"acc={best_acc*100:.1f}% score={score:.4f}")
        else:
            log_progress(f"  Round {rnd+1}: no improvement")

        round_history.append({
            'round': rnd + 1,
            'winner': round_winner,
            'best_acc': best_acc,
            'score': score,
            'win_counts': dict(win_counts),
        })

    net.restore_state(best_state)
    return best_acc, score, win_counts, round_history, net.count_connections()


def train_mega(net, perm, V, rounds=4, att_per_specialist=4000, ticks=TICKS):
    """Mega combo: rebalanced ensemble + pop_2 inside each specialist."""
    score, acc = combined_score(net, perm, V, ticks)
    best_acc = acc
    best_state = net.save_state()

    spec_names = list(SPECIALISTS.keys())
    win_counts = {s: 0 for s in spec_names}
    round_history = []
    total_budget = att_per_specialist * len(spec_names)

    for rnd in range(rounds):
        # Rebalanced allocation
        if rnd > 0:
            total_wins = sum(win_counts.values())
            if total_wins > 0:
                alloc = {}
                for s in spec_names:
                    ratio = max(0.1, win_counts[s] / total_wins)
                    alloc[s] = ratio
                total_r = sum(alloc.values())
                alloc = {s: int(total_budget * alloc[s] / total_r) for s in spec_names}
            else:
                alloc = {s: att_per_specialist for s in spec_names}
        else:
            alloc = {s: att_per_specialist for s in spec_names}

        round_best_score = score
        round_best_state = best_state
        round_winner = None

        for spec_name in spec_names:
            spec_fn = SPECIALISTS[spec_name]
            n_att = alloc[spec_name]

            net.restore_state(best_state)
            s_score = score
            s_kept = 0

            for att in range(n_att):
                state = net.save_state()

                # Pop_2: try two mutations, keep better
                spec_fn(net)
                s1, a1 = combined_score(net, perm, V, ticks)
                state1 = net.save_state()

                net.restore_state(state)
                spec_fn(net)
                s2, a2 = combined_score(net, perm, V, ticks)

                # Pick better candidate
                if s1 >= s2:
                    cand_s, cand_a = s1, a1
                    if s1 > s_score:
                        net.restore_state(state1)
                    else:
                        net.restore_state(state)
                else:
                    cand_s, cand_a = s2, a2
                    if s2 <= s_score:
                        net.restore_state(state)

                if cand_s > s_score:
                    s_score = cand_s
                    s_kept += 1
                    if max(a1, a2) > best_acc:
                        best_acc = max(a1, a2)

            if s_score > round_best_score:
                round_best_score = s_score
                round_best_state = net.save_state()
                round_winner = spec_name

            win_counts[spec_name] += s_kept

        if round_winner:
            best_state = round_best_state
            score = round_best_score
            log_progress(f"  Mega Round {rnd+1}: winner={round_winner} "
                         f"acc={best_acc*100:.1f}% score={score:.4f}")
        else:
            log_progress(f"  Mega Round {rnd+1}: no improvement")

        round_history.append({
            'round': rnd + 1,
            'winner': round_winner,
            'best_acc': best_acc,
            'score': score,
        })

    net.restore_state(best_state)
    return best_acc, score, win_counts, round_history, net.count_connections()


# ============================================================
# Multi-seed runner
# ============================================================

def run_seeds(label, make_net_fn, train_fn, V, seeds=SEEDS, **train_kwargs):
    """Run a config across multiple seeds, return result dict."""
    accs = []
    scores = []
    conns = []
    t0 = time.perf_counter()

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        perm = np.random.permutation(V)
        net = make_net_fn()

        result = train_fn(net, perm, V, **train_kwargs)
        # result is (best_acc, score, ..., connections) — last is always connections
        accs.append(result[0])
        scores.append(result[1])
        conns.append(result[-1])
        log_progress(f"  {label} seed={seed}: acc={result[0]*100:.1f}%")

    elapsed = time.perf_counter() - t0
    avg_acc = np.mean(accs)
    best_acc = max(accs)

    print(f"\n  {label}:")
    print(f"    Seeds: {[f'{a*100:.1f}%' for a in accs]}")
    print(f"    avg={avg_acc*100:.1f}%, best={best_acc*100:.1f}%")
    print(f"    Time: {elapsed:.1f}s | Conns: {int(np.mean(conns))}")

    return {
        'accs': [float(a) for a in accs],
        'avg_acc': float(avg_acc),
        'best_acc': float(best_acc),
        'avg_score': float(np.mean(scores)),
        'avg_conns': float(np.mean(conns)),
        'time': float(elapsed),
    }


def run_seeds_ensemble(label, make_net_fn, V, seeds=SEEDS, **ens_kwargs):
    """Run ensemble training across seeds."""
    accs = []
    all_win_counts = []
    conns = []
    t0 = time.perf_counter()

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        perm = np.random.permutation(V)
        net = make_net_fn()

        best_acc, score, win_counts, history, n_conns = train_ensemble(
            net, perm, V, **ens_kwargs)
        accs.append(best_acc)
        all_win_counts.append(win_counts)
        conns.append(n_conns)
        log_progress(f"  {label} seed={seed}: acc={best_acc*100:.1f}%")

    elapsed = time.perf_counter() - t0
    avg_acc = np.mean(accs)

    # Aggregate win counts
    agg_wins = {}
    for wc in all_win_counts:
        for k, v in wc.items():
            agg_wins[k] = agg_wins.get(k, 0) + v

    print(f"\n  {label}:")
    print(f"    Seeds: {[f'{a*100:.1f}%' for a in accs]}")
    print(f"    avg={avg_acc*100:.1f}%, best={max(accs)*100:.1f}%")
    print(f"    Win counts: {agg_wins}")
    print(f"    Time: {elapsed:.1f}s")

    return {
        'accs': [float(a) for a in accs],
        'avg_acc': float(avg_acc),
        'best_acc': float(max(accs)),
        'win_counts': agg_wins,
        'avg_conns': float(np.mean(conns)),
        'time': float(elapsed),
    }


def run_seeds_mega(label, make_net_fn, V, seeds=SEEDS, **mega_kwargs):
    """Run mega combo training across seeds."""
    accs = []
    all_win_counts = []
    conns = []
    t0 = time.perf_counter()

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        perm = np.random.permutation(V)
        net = make_net_fn()

        best_acc, score, win_counts, history, n_conns = train_mega(
            net, perm, V, **mega_kwargs)
        accs.append(best_acc)
        all_win_counts.append(win_counts)
        conns.append(n_conns)
        log_progress(f"  {label} seed={seed}: acc={best_acc*100:.1f}%")

    elapsed = time.perf_counter() - t0
    avg_acc = np.mean(accs)

    agg_wins = {}
    for wc in all_win_counts:
        for k, v in wc.items():
            agg_wins[k] = agg_wins.get(k, 0) + v

    print(f"\n  {label}:")
    print(f"    Seeds: {[f'{a*100:.1f}%' for a in accs]}")
    print(f"    avg={avg_acc*100:.1f}%, best={max(accs)*100:.1f}%")
    print(f"    Win counts: {agg_wins}")
    print(f"    Time: {elapsed:.1f}s")

    return {
        'accs': [float(a) for a in accs],
        'avg_acc': float(avg_acc),
        'best_acc': float(max(accs)),
        'win_counts': agg_wins,
        'avg_conns': float(np.mean(conns)),
        'time': float(elapsed),
    }


# ============================================================
# Helper: pick best config from results
# ============================================================

def pick_best_config(result_keys, default_hw=2):
    """Pick the config with highest avg_acc from results dict."""
    best_key = None
    best_avg = -1
    for k in result_keys:
        if k in results and 'avg_acc' in results[k]:
            if results[k]['avg_acc'] > best_avg:
                best_avg = results[k]['avg_acc']
                best_key = k
    return best_key


# ============================================================
# TEST PHASES
# ============================================================

def phase1():
    """Phase 1: Baseline (shared I/O)."""
    log_progress("=" * 60)
    log_progress("PHASE 1: BASELINE (Shared I/O)")
    log_progress("=" * 60)

    # 1a: V=32
    safe_test('baseline_V32', lambda: run_seeds(
        'baseline_V32',
        lambda: SharedIONet(32 + INTERNAL, 32),
        train_basic, V=32, max_attempts=8000))

    # 1b: V=64
    safe_test('baseline_V64', lambda: run_seeds(
        'baseline_V64',
        lambda: SharedIONet(64 + INTERNAL, 64),
        train_basic, V=64, max_attempts=8000))


def phase2():
    """Phase 2: Split I/O."""
    log_progress("=" * 60)
    log_progress("PHASE 2: SPLIT I/O")
    log_progress("=" * 60)

    # 2a: V=32
    safe_test('split_io_V32', lambda: run_seeds(
        'split_io_V32',
        lambda: SplitIONet(32, INTERNAL, hw_size=0),
        train_basic, V=32, max_attempts=8000))

    # 2b: V=64
    safe_test('split_io_V64', lambda: run_seeds(
        'split_io_V64',
        lambda: SplitIONet(64, INTERNAL, hw_size=0),
        train_basic, V=64, max_attempts=8000))


def phase3():
    """Phase 3: Split I/O + Highway."""
    log_progress("=" * 60)
    log_progress("PHASE 3: SPLIT I/O + HIGHWAY")
    log_progress("=" * 60)

    for hw in [1, 2, 4]:
        # V=32
        safe_test(f'split_hw{hw}_V32', lambda hw=hw: run_seeds(
            f'split_hw{hw}_V32',
            lambda: SplitIONet(32, INTERNAL, hw_size=hw),
            train_basic, V=32, max_attempts=8000))

        # V=64
        safe_test(f'split_hw{hw}_V64', lambda hw=hw: run_seeds(
            f'split_hw{hw}_V64',
            lambda: SplitIONet(64, INTERNAL, hw_size=hw),
            train_basic, V=64, max_attempts=8000))


def phase4():
    """Phase 4: Chain Ensemble on best config."""
    log_progress("=" * 60)
    log_progress("PHASE 4: CHAIN ENSEMBLE")
    log_progress("=" * 60)

    # Find best V=32 config
    v32_keys = ['split_io_V32', 'split_hw1_V32', 'split_hw2_V32', 'split_hw4_V32']
    best32 = pick_best_config(v32_keys)
    # Find best V=64 config
    v64_keys = ['split_io_V64', 'split_hw1_V64', 'split_hw2_V64', 'split_hw4_V64']
    best64 = pick_best_config(v64_keys)

    # Determine hw_size from key
    def hw_from_key(key):
        if key is None:
            return 2  # fallback
        if 'hw1' in key:
            return 1
        if 'hw2' in key:
            return 2
        if 'hw4' in key:
            return 4
        return 0

    hw32 = hw_from_key(best32)
    hw64 = hw_from_key(best64)
    log_progress(f"  Best V=32: {best32} (hw={hw32})")
    log_progress(f"  Best V=64: {best64} (hw={hw64})")

    # 4a: V=32
    safe_test('ensemble_V32', lambda: run_seeds_ensemble(
        'ensemble_V32',
        lambda: SplitIONet(32, INTERNAL, hw_size=hw32),
        V=32, rounds=4, att_per_specialist=4000))

    # 4b: V=64
    safe_test('ensemble_V64', lambda: run_seeds_ensemble(
        'ensemble_V64',
        lambda: SplitIONet(64, INTERNAL, hw_size=hw64),
        V=64, rounds=4, att_per_specialist=4000))


def phase5():
    """Phase 5: Rebalanced Ensemble (V=64 only, 8 rounds)."""
    log_progress("=" * 60)
    log_progress("PHASE 5: REBALANCED ENSEMBLE")
    log_progress("=" * 60)

    v64_keys = ['split_io_V64', 'split_hw1_V64', 'split_hw2_V64', 'split_hw4_V64']
    best64 = pick_best_config(v64_keys)
    hw64 = 2  # fallback
    if best64:
        if 'hw1' in best64:
            hw64 = 1
        elif 'hw4' in best64:
            hw64 = 4
        elif 'hw2' in best64:
            hw64 = 2
        elif 'split_io' in best64:
            hw64 = 0

    log_progress(f"  Using: hw={hw64}")

    safe_test('rebalanced_V64', lambda: run_seeds_ensemble(
        'rebalanced_V64',
        lambda: SplitIONet(64, INTERNAL, hw_size=hw64),
        V=64, rounds=8, att_per_specialist=4000, rebalanced=True))


def phase6():
    """Phase 6: Pop_2 on best architecture."""
    log_progress("=" * 60)
    log_progress("PHASE 6: POP_2")
    log_progress("=" * 60)

    v32_keys = ['split_io_V32', 'split_hw1_V32', 'split_hw2_V32', 'split_hw4_V32']
    v64_keys = ['split_io_V64', 'split_hw1_V64', 'split_hw2_V64', 'split_hw4_V64']
    best32 = pick_best_config(v32_keys)
    best64 = pick_best_config(v64_keys)

    def hw_from_key(key):
        if key is None:
            return 2
        if 'hw1' in key:
            return 1
        if 'hw2' in key:
            return 2
        if 'hw4' in key:
            return 4
        return 0

    hw32 = hw_from_key(best32)
    hw64 = hw_from_key(best64)

    safe_test('pop2_V32', lambda: run_seeds(
        'pop2_V32',
        lambda: SplitIONet(32, INTERNAL, hw_size=hw32),
        train_pop2, V=32, max_attempts=16000))

    safe_test('pop2_V64', lambda: run_seeds(
        'pop2_V64',
        lambda: SplitIONet(64, INTERNAL, hw_size=hw64),
        train_pop2, V=64, max_attempts=16000))


def phase7():
    """Phase 7: MEGA COMBO (everything together, V=64 only)."""
    log_progress("=" * 60)
    log_progress("PHASE 7: MEGA COMBO")
    log_progress("=" * 60)

    v64_keys = ['split_io_V64', 'split_hw1_V64', 'split_hw2_V64', 'split_hw4_V64']
    best64 = pick_best_config(v64_keys)
    hw64 = 2  # fallback
    if best64:
        if 'hw1' in best64:
            hw64 = 1
        elif 'hw4' in best64:
            hw64 = 4
        elif 'hw2' in best64:
            hw64 = 2
        elif 'split_io' in best64:
            hw64 = 0

    log_progress(f"  Mega config: V=64, hw={hw64}, rebalanced ensemble + pop_2")

    safe_test('mega_V64', lambda: run_seeds_mega(
        'mega_V64',
        lambda: SplitIONet(64, INTERNAL, hw_size=hw64),
        V=64, seeds=[42, 123, 777],  # 3 seeds (longest test)
        rounds=4, att_per_specialist=4000))


# ============================================================
# Final Summary
# ============================================================

def print_summary():
    """Print final ranking table."""
    log_progress("=" * 60)
    log_progress("FINAL SUMMARY")
    log_progress("=" * 60)

    # Collect all results
    entries = []
    for key, val in results.items():
        if isinstance(val, dict) and 'avg_acc' in val:
            v = 32 if 'V32' in key or '_V32' in key else 64
            entries.append((key, v, val['avg_acc'], val.get('best_acc', 0), val.get('time', 0)))

    # Sort by avg_acc descending
    entries.sort(key=lambda x: x[2], reverse=True)

    # Build summary table
    print(f"\n{'='*80}")
    print(f"  OVERNIGHT COMBO TEST — FINAL RANKING")
    print(f"{'='*80}")
    print(f"  {'Rank':>4} | {'Config':<30} | {'V':>3} | {'Avg':>6} | {'Best':>6} | {'Time':>6}")
    print(f"  {'-'*4}-+-{'-'*30}-+-{'-'*3}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    for i, (key, v, avg, best, t) in enumerate(entries):
        print(f"  {i+1:>4} | {key:<30} | {v:>3} | {avg*100:>5.1f}% | {best*100:>5.1f}% | {t:>5.0f}s")

    print(f"{'='*80}")

    # Save summary
    results['_summary'] = [
        {'rank': i+1, 'config': key, 'V': v, 'avg_acc': avg, 'best_acc': best, 'time': t}
        for i, (key, v, avg, best, t) in enumerate(entries)
    ]
    save_results()


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    log_progress("OVERNIGHT COMBO TEST STARTED")
    log_progress(f"Seeds: {SEEDS}")
    log_progress(f"Ticks: {TICKS}, Internal: {INTERNAL}")

    t_start = time.perf_counter()

    phase1()
    phase2()
    phase3()
    phase4()
    phase5()
    phase6()
    phase7()
    print_summary()

    total = time.perf_counter() - t_start
    log_progress(f"TOTAL TIME: {total:.0f}s ({total/60:.1f} min)")
    log_progress("OVERNIGHT COMBO TEST FINISHED")
