"""
Expressiveness Test: SWG vs MLP — Same Params, Unlimited Time
==============================================================
Question: Given the same parameter count and enough time,
what can SWG learn that MLP can/cannot, and vice versa?

We fix the network size, then throw increasingly hard tasks at both.
SWG gets massive budget (no stale limit). MLP gets enough epochs.

Tasks (increasing difficulty):
  1. Identity:       input i → output i
  2. Permutation:    input i → σ(i)
  3. Shift-by-k:     input i → (i+k) mod V
  4. XOR pairs:      input (i,j) → i XOR j  (binary encoding)
  5. Composition:    two stacked permutations σ₂(σ₁(i))
  6. Many-to-one:    random non-injective mapping (V→V with collisions)
  7. Density stress: random mapping with V/2 unique outputs (high collision)
  8. Capacity test:  how many random associations can it memorize?
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random as pyrandom
from model.graph import SelfWiringGraph

# ── Helpers ──

def set_seeds(seed):
    np.random.seed(seed)
    pyrandom.seed(seed)

def softmax_rows(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ══════════════════════════════════════════════════════════
#  SWG trainer (unlimited budget, generous stale limit)
# ══════════════════════════════════════════════════════════

def train_swg(V, targets, seed, max_att=200000, stale_limit=50000):
    set_seeds(seed)
    N = V * 3
    net = SelfWiringGraph(N, V)

    def evaluate():
        logits = net.forward_batch(ticks=8)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
        tp = probs[np.arange(V), targets[:V]].mean()
        return acc, 0.5 * acc + 0.5 * tp

    acc, score = evaluate()
    best_acc = acc
    best_score = score
    stale = 0

    for att in range(max_att):
        old_loss = int(net.loss_pct)
        undo = net.mutate()
        new_acc, new_score = evaluate()

        if new_score > score:
            score = new_score
            best_score = max(best_score, score)
            best_acc = max(best_acc, new_acc)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            stale += 1
            if pyrandom.randint(1, 20) <= 7:
                net.signal = np.int8(1 - int(net.signal))
            if pyrandom.randint(1, 20) <= 7:
                net.grow = np.int8(1 - int(net.grow))

        if best_acc >= 1.0 or stale >= stale_limit:
            break

    return best_acc, att + 1


# ══════════════════════════════════════════════════════════
#  MLP trainer (same param count, Adam optimizer)
# ══════════════════════════════════════════════════════════

class MLP:
    def __init__(self, V, hidden):
        self.W1 = np.random.randn(V, hidden).astype(np.float32) * np.sqrt(2.0 / V)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = np.random.randn(hidden, V).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(V, dtype=np.float32)
        # Adam state
        self.t = 0
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def forward(self, X):
        self.X = X
        self.h = X @ self.W1 + self.b1
        self.a = np.maximum(self.h, 0)
        self.logits = self.a @ self.W2 + self.b2
        return self.logits

    def backward_adam(self, probs, targets_onehot, lr=0.001):
        V = probs.shape[0]
        d_logits = (probs - targets_onehot) / V
        grads = [
            self.a.T @ d_logits,                          # dW2
            d_logits.sum(axis=0),                          # db2
            self.X.T @ ((d_logits @ self.W2.T) * (self.h > 0)),  # dW1
            ((d_logits @ self.W2.T) * (self.h > 0)).sum(axis=0), # db1
        ]
        # Reorder to match params: W1, b1, W2, b2
        grads = [grads[2], grads[3], grads[0], grads[1]]

        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * g ** 2
            mh = self.m[i] / (1 - beta1 ** self.t)
            vh = self.v[i] / (1 - beta2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def param_count(self):
        return sum(p.size for p in self.params)


def train_mlp(V, targets, seed, max_epochs=10000):
    set_seeds(seed)
    hidden = V  # same-ish param count as SWG
    mlp = MLP(V, hidden)
    X = np.eye(V, dtype=np.float32)
    targets_oh = np.zeros((V, V), dtype=np.float32)
    for i in range(V):
        targets_oh[i, targets[i]] = 1.0

    best_acc = 0.0
    stale = 0
    for epoch in range(max_epochs):
        logits = mlp.forward(X)
        probs = softmax_rows(logits)
        preds = np.argmax(probs, axis=1)
        acc = (preds == targets[:V]).mean()
        best_acc = max(best_acc, acc)

        if best_acc >= 1.0:
            return best_acc, epoch + 1

        mlp.backward_adam(probs, targets_oh, lr=0.003)
        if acc <= best_acc:
            stale += 1
        else:
            stale = 0
        if stale > 2000:
            break

    return best_acc, epoch + 1


# ══════════════════════════════════════════════════════════
#  TASK GENERATORS
# ══════════════════════════════════════════════════════════

def task_identity(V, seed):
    """i → i"""
    return np.arange(V, dtype=int)

def task_random_perm(V, seed):
    """i → σ(i)"""
    set_seeds(seed)
    return np.random.permutation(V).astype(int)

def task_shift(V, seed, k=None):
    """i → (i + k) mod V"""
    if k is None:
        k = V // 3
    return np.array([(i + k) % V for i in range(V)], dtype=int)

def task_reverse(V, seed):
    """i → V-1-i"""
    return np.arange(V-1, -1, -1, dtype=int)

def task_xor_pairs(V, seed):
    """For V that's a power of 2: i → i XOR mask"""
    set_seeds(seed)
    mask = pyrandom.randint(1, V - 1)
    return np.array([i ^ mask for i in range(V)], dtype=int)

def task_double_perm(V, seed):
    """Composition: σ₂(σ₁(i)) — two stacked permutations"""
    set_seeds(seed)
    p1 = np.random.permutation(V)
    p2 = np.random.permutation(V)
    return p2[p1].astype(int)

def task_many_to_one(V, seed):
    """Random non-injective: each input maps to random output (collisions!)"""
    set_seeds(seed)
    return np.random.randint(0, V, size=V).astype(int)

def task_half_outputs(V, seed):
    """Only V/2 unique outputs — high collision density"""
    set_seeds(seed)
    outputs = np.arange(V // 2)
    return np.random.choice(outputs, size=V).astype(int)

def task_modular(V, seed):
    """i → (a*i + b) mod V — affine modular arithmetic"""
    set_seeds(seed)
    # pick a coprime to V
    a = pyrandom.choice([k for k in range(2, V) if np.gcd(k, V) == 1])
    b = pyrandom.randint(0, V - 1)
    return np.array([(a * i + b) % V for i in range(V)], dtype=int)


# ══════════════════════════════════════════════════════════
#  BENCHMARK
# ══════════════════════════════════════════════════════════

TASKS = [
    ("Identity",        task_identity),
    ("Random Perm",     task_random_perm),
    ("Shift V/3",       task_shift),
    ("Reverse",         task_reverse),
    ("XOR mask",        task_xor_pairs),
    ("Affine mod",      task_modular),
    ("Double Perm",     task_double_perm),
    ("Many-to-one",     task_many_to_one),
    ("Half outputs",    task_half_outputs),
]


def run_expressiveness(V, n_seeds=5):
    print(f"\n{'='*78}")
    print(f"  EXPRESSIVENESS: V={V}  |  SWG params={(V*3)**2} ternary  |  MLP params~{2*V*V+2*V} float")
    print(f"  SWG info: {(V*3)**2 * 2:,} bits  |  MLP info: {(2*V*V+2*V) * 32:,} bits  |  ratio: {((2*V*V+2*V)*32) / ((V*3)**2*2):.1f}x")
    print(f"{'='*78}")

    # Determine budgets based on V
    if V <= 16:
        swg_budget, swg_stale = 50000, 30000
        mlp_epochs = 5000
    elif V <= 32:
        swg_budget, swg_stale = 100000, 50000
        mlp_epochs = 8000
    else:
        swg_budget, swg_stale = 200000, 80000
        mlp_epochs = 10000

    results = {}

    for task_name, task_fn in TASKS:
        swg_accs = []
        mlp_accs = []

        for si in range(n_seeds):
            seed = 300 + si
            targets = task_fn(V, seed)

            # SWG
            acc_s, att_s = train_swg(V, targets, seed, swg_budget, swg_stale)
            swg_accs.append(acc_s)

            # MLP
            acc_m, ep_m = train_mlp(V, targets, seed, mlp_epochs)
            mlp_accs.append(acc_m)

        swg_arr = np.array(swg_accs)
        mlp_arr = np.array(mlp_accs)

        results[task_name] = {
            'swg_mean': swg_arr.mean(), 'swg_std': swg_arr.std(),
            'mlp_mean': mlp_arr.mean(), 'mlp_std': mlp_arr.std(),
            'swg_all': swg_accs, 'mlp_all': mlp_accs,
        }

        # Who wins?
        if swg_arr.mean() >= 0.99 and mlp_arr.mean() >= 0.99:
            verdict = "BOTH 100%"
        elif swg_arr.mean() >= mlp_arr.mean() - 0.01:
            verdict = "SWG ≥ MLP" if swg_arr.mean() > mlp_arr.mean() + 0.01 else "TIE"
        else:
            verdict = f"MLP +{(mlp_arr.mean()-swg_arr.mean())*100:.0f}pp"

        print(f"  {task_name:<16}  SWG: {swg_arr.mean()*100:5.1f}% ± {swg_arr.std()*100:3.1f}%  "
              f"  MLP: {mlp_arr.mean()*100:5.1f}% ± {mlp_arr.std()*100:3.1f}%  "
              f"  → {verdict}")

    return results


if __name__ == '__main__':
    print("=" * 78)
    print("  EXPRESSIVENESS TEST: Same Params, Unlimited Time — What Can Each Learn?")
    print("=" * 78)

    all_results = {}
    for V in [16, 32, 64]:
        all_results[V] = run_expressiveness(V, n_seeds=5)

    # ── Final comparison table ──
    print(f"\n\n{'='*78}")
    print(f"  FINAL COMPARISON — Accuracy (%) by Task and Model")
    print(f"{'='*78}\n")

    print(f"  {'Task':<16}", end="")
    for V in [16, 32, 64]:
        print(f"  {'V='+str(V)+' SWG':>10} {'MLP':>6}", end="")
    print()
    print(f"  {'─'*72}")

    for task_name, _ in TASKS:
        print(f"  {task_name:<16}", end="")
        for V in [16, 32, 64]:
            r = all_results[V][task_name]
            s = f"{r['swg_mean']*100:5.1f}"
            m = f"{r['mlp_mean']*100:5.1f}"
            print(f"  {s:>10} {m:>6}", end="")
        print()

    # ── Bit efficiency ──
    print(f"\n\n{'='*78}")
    print(f"  BIT EFFICIENCY — Accuracy per kilobit of parameters")
    print(f"{'='*78}\n")

    for V in [16, 32, 64]:
        swg_bits = (V * 3) ** 2 * 2 / 1000
        mlp_bits = (2 * V * V + 2 * V) * 32 / 1000

        print(f"  V={V}:  SWG={swg_bits:.1f}Kb  MLP={mlp_bits:.1f}Kb  (MLP is {mlp_bits/swg_bits:.1f}x larger)")

        for task_name, _ in TASKS:
            r = all_results[V][task_name]
            swg_eff = r['swg_mean'] * 100 / swg_bits
            mlp_eff = r['mlp_mean'] * 100 / mlp_bits
            winner = "SWG" if swg_eff > mlp_eff else "MLP"
            ratio = max(swg_eff, mlp_eff) / max(min(swg_eff, mlp_eff), 0.001)
            print(f"    {task_name:<16}  SWG: {swg_eff:5.2f}%/Kb  MLP: {mlp_eff:5.2f}%/Kb  → {winner} {ratio:.1f}x more efficient")
        print()

    # ── Summary ──
    print(f"\n{'='*78}")
    print(f"  VERDICT")
    print(f"{'='*78}")

    # Count where SWG matches MLP (within 5pp)
    total = 0
    swg_matches = 0
    swg_wins_bits = 0
    for V in [16, 32, 64]:
        swg_bits = (V * 3) ** 2 * 2
        mlp_bits = (2 * V * V + 2 * V) * 32
        for task_name, _ in TASKS:
            r = all_results[V][task_name]
            total += 1
            if r['swg_mean'] >= r['mlp_mean'] - 0.05:
                swg_matches += 1
            # Bit efficiency
            swg_eff = r['swg_mean'] / swg_bits
            mlp_eff = r['mlp_mean'] / mlp_bits
            if swg_eff > mlp_eff:
                swg_wins_bits += 1

    print(f"\n  SWG matches MLP accuracy (within 5pp): {swg_matches}/{total} tasks")
    print(f"  SWG wins on bit-efficiency:             {swg_wins_bits}/{total} tasks")
    print(f"\n{'='*78}")
