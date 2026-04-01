"""
No-Backprop v12 — Single Network Self-Modification
=====================================================
ONE network. No population. No EA.
  1. Smart init (Xavier)
  2. Try random change → keep if better, revert if worse
  3. Phase 1: structure only (connections) until plateau
  4. Phase 2: structure + weights

Like the brain: one network, modifying itself based on feedback.
"""

import torch
import torch.nn.functional as F
import math
import random
import time


class SparseNet:
    def __init__(self, layer_sizes, density=0.5):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.W = []
        self.mask = []
        for i in range(self.n_layers):
            fi, fo = layer_sizes[i], layer_sizes[i + 1]
            # Xavier init
            s = math.sqrt(2.0 / (fi + fo))
            self.W.append(torch.randn(fi, fo) * s)
            self.mask.append((torch.rand(fi, fo) < density).float())
        self.score = 0.0

    def forward(self, x):
        h = x
        for i in range(self.n_layers):
            h = h @ (self.W[i] * self.mask[i])
            if i < self.n_layers - 1:
                h = torch.relu(h)
        return h

    def count_connections(self):
        return sum(int(m.sum().item()) for m in self.mask)

    def save_state(self):
        return ([w.clone() for w in self.W], [m.clone() for m in self.mask])

    def restore_state(self, state):
        W_saved, mask_saved = state
        self.W = [w.clone() for w in W_saved]
        self.mask = [m.clone() for m in mask_saved]

    def mutate_structure(self, rate=0.05):
        """Try a random structural change."""
        i = random.randint(0, self.n_layers - 1)
        fi, fo = self.layer_sizes[i], self.layer_sizes[i + 1]

        action = random.choice(["add", "remove", "rewire"])

        if action == "add":
            dead = (self.mask[i] == 0).nonzero(as_tuple=False)
            if len(dead) > 0:
                n = max(1, int(len(dead) * rate))
                idx = dead[torch.randperm(len(dead))[:n]]
                for j in range(len(idx)):
                    r, c = idx[j][0].item(), idx[j][1].item()
                    self.mask[i][r, c] = 1
                    self.W[i][r, c] = random.gauss(0, math.sqrt(2.0 / (fi + fo)))

        elif action == "remove":
            alive = (self.mask[i] == 1).nonzero(as_tuple=False)
            if len(alive) > 3:
                n = max(1, int(len(alive) * rate))
                idx = alive[torch.randperm(len(alive))[:n]]
                for j in range(len(idx)):
                    self.mask[i][idx[j][0], idx[j][1]] = 0

        elif action == "rewire":
            alive = (self.mask[i] == 1).nonzero(as_tuple=False)
            if len(alive) > 0:
                n = max(1, int(len(alive) * rate))
                idx = alive[torch.randperm(len(alive))[:n]]
                for j in range(len(idx)):
                    r, c = idx[j][0].item(), idx[j][1].item()
                    self.mask[i][r, c] = 0
                    new_c = random.randint(0, fo - 1)
                    self.mask[i][r, new_c] = 1
                    self.W[i][r, new_c] = self.W[i][r, c]

    def mutate_weights(self, scale=0.05):
        """Try a random weight perturbation."""
        i = random.randint(0, self.n_layers - 1)
        noise = torch.randn_like(self.W[i]) * scale
        self.W[i] += noise * self.mask[i]

    def mutate_both(self, struct_rate=0.05, weight_scale=0.05):
        """Both structure and weights."""
        if random.random() < 0.5:
            self.mutate_structure(struct_rate)
        else:
            self.mutate_weights(weight_scale)


def evaluate(net, X, targets):
    logits = net.forward(X)
    probs = torch.softmax(logits, dim=-1)
    correct = (probs.argmax(-1) == targets).float().mean().item()
    target_probs = probs[range(len(targets)), targets].mean().item()
    return 0.5 * correct + 0.5 * target_probs


def run_task(name, X, targets, layer_sizes, vocab,
             attempts=20000, patience=2000):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    random.seed(42); torch.manual_seed(42)
    net = SparseNet(layer_sizes, density=0.5)

    current_score = evaluate(net, X, targets)
    best_score = current_score
    phase = "STRUCTURE"
    kept = 0
    plateau_counter = 0
    phase_switched = False
    t0 = time.time()

    logits = net.forward(X)
    acc = (logits.argmax(-1) == targets).float().mean().item() * 100
    print(f"  Start: {acc:.0f}% | Score: {current_score:.3f} | "
          f"Conns: {net.count_connections()}")

    for attempt in range(attempts):
        # Save state
        state = net.save_state()

        # Mutate based on phase
        if phase == "STRUCTURE":
            net.mutate_structure(rate=0.05)
        else:
            net.mutate_both(struct_rate=0.02, weight_scale=0.05)

        # Evaluate
        new_score = evaluate(net, X, targets)

        if new_score > current_score:
            current_score = new_score
            kept += 1
            plateau_counter = 0
            if current_score > best_score:
                best_score = current_score
        else:
            # Revert
            net.restore_state(state)
            plateau_counter += 1

        # Phase switch: plateau in structure → add weights
        if phase == "STRUCTURE" and plateau_counter > patience and not phase_switched:
            phase = "BOTH"
            phase_switched = True
            elapsed = time.time() - t0
            logits = net.forward(X)
            acc = (logits.argmax(-1) == targets).float().mean().item() * 100
            print(f"  ── PLATEAU at attempt {attempt+1} ({elapsed:.1f}s) ──")
            print(f"  Switching to STRUCTURE + WEIGHTS")
            print(f"  Acc: {acc:.0f}% | Score: {current_score:.3f} | "
                  f"Conns: {net.count_connections()}")
            plateau_counter = 0

        # Logging
        if (attempt + 1) % 2000 == 0:
            logits = net.forward(X)
            acc = (logits.argmax(-1) == targets).float().mean().item() * 100
            elapsed = time.time() - t0
            print(f"  [{phase:9s}] Attempt {attempt+1:5d} | "
                  f"Acc: {acc:.0f}% | Score: {current_score:.3f} | "
                  f"Kept: {kept} | Conns: {net.count_connections()} | "
                  f"{elapsed:.1f}s")

        # Early stop
        if best_score > 0.99:
            elapsed = time.time() - t0
            print(f"  → Solved at attempt {attempt+1}! ({elapsed:.1f}s)")
            break

    elapsed = time.time() - t0
    logits = net.forward(X)
    acc = (logits.argmax(-1) == targets).float().mean().item() * 100
    conns = net.count_connections()
    max_c = sum(m.numel() for m in net.mask)
    print(f"\n  Final: {acc:.0f}% | Score: {best_score:.3f} | "
          f"Conns: {conns}/{max_c} ({conns/max_c*100:.0f}%) | "
          f"Kept: {kept}/{attempt+1} | {elapsed:.1f}s")

    return net, acc, best_score, elapsed


def run():
    results = {}

    # ═══════ 8-class ═══════
    torch.manual_seed(42)
    v = 8; perm = torch.randperm(v); X = torch.eye(v)
    print(f"Tasks: 8/16/32-class permutations")
    print(f"Method: single network, try→keep/revert, structure first then both")

    net, acc, score, t = run_task(
        f"8-class: {list(range(v))} → {perm.tolist()}",
        X, perm, [v, 16, 16, v], v, attempts=10000)
    results["8"] = (acc, score, t)

    # ═══════ 16-class ═══════
    torch.manual_seed(42)
    v = 16; perm = torch.randperm(v); X = torch.eye(v)
    net, acc, score, t = run_task(
        "16-class permutation",
        X, perm, [v, 32, 32, v], v, attempts=20000)
    results["16"] = (acc, score, t)

    # ═══════ 32-class ═══════
    torch.manual_seed(42)
    v = 32; perm = torch.randperm(v); X = torch.eye(v)
    net, acc, score, t = run_task(
        "32-class permutation",
        X, perm, [v, 64, 64, v], v, attempts=50000, patience=3000)
    results["32"] = (acc, score, t)

    # ═══════ Backprop comparison ═══════
    print(f"\n{'='*60}")
    print(f"  BACKPROP COMPARISONS")
    print(f"{'='*60}")
    import torch.nn as nn
    bp_results = {}
    for v in [8, 16, 32]:
        torch.manual_seed(42)
        perm = torch.randperm(v); X = torch.eye(v)
        bp = nn.Sequential(
            nn.Linear(v, v*2), nn.ReLU(),
            nn.Linear(v*2, v*2), nn.ReLU(),
            nn.Linear(v*2, v))
        opt = torch.optim.Adam(bp.parameters(), lr=1e-3)
        for epoch in range(3000):
            loss = F.cross_entropy(bp(X), perm)
            opt.zero_grad(); loss.backward(); opt.step()
        bp_acc = (bp(X).argmax(-1) == perm).float().mean().item() * 100
        bp_results[v] = bp_acc
        print(f"  {v}-class: {bp_acc:.0f}%")

    # ═══════ Summary ═══════
    print(f"\n{'='*60}")
    print(f"  SUMMARY: Single-Network Self-Modification")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Ours':>8} {'Backprop':>10} {'Time':>8}")
    print(f"  {'-'*38}")
    for v in [8, 16, 32]:
        acc, score, t = results[str(v)]
        print(f"  {v}-class   {acc:>7.0f}% {bp_results[v]:>9.0f}% {t:>7.1f}s")


if __name__ == "__main__":
    run()
