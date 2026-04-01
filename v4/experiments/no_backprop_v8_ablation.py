"""
No-Backprop v8 — ABLATION: What matters more, structure or weights?
=====================================================================
Test 1: Evolve ONLY structure (weights stay random init)
Test 2: Evolve ONLY weights (fully connected, no structure change)
Test 3: Evolve BOTH (the full v8)
"""

import torch
import torch.nn.functional as F
import math
import random
import time


class SparseNet:
    def __init__(self, layer_sizes, density=0.3):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.W = []
        self.mask = []
        for i in range(self.n_layers):
            fi, fo = layer_sizes[i], layer_sizes[i + 1]
            s = math.sqrt(2.0 / fi)
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

    def clone(self):
        new = SparseNet.__new__(SparseNet)
        new.layer_sizes = self.layer_sizes[:]
        new.n_layers = self.n_layers
        new.W = [w.clone() for w in self.W]
        new.mask = [m.clone() for m in self.mask]
        new.score = self.score
        return new

    def mutate_structure_only(self, rate=0.15):
        """Change ONLY which connections exist. Don't touch weight values."""
        for i in range(self.n_layers):
            fi, fo = self.layer_sizes[i], self.layer_sizes[i + 1]
            n_conns = int(self.mask[i].sum().item())

            # Add connections
            n_add = max(1, int(n_conns * rate * 0.3))
            dead = (self.mask[i] == 0).nonzero(as_tuple=False)
            if len(dead) > 0:
                idx = dead[torch.randperm(len(dead))[:n_add]]
                for j in range(len(idx)):
                    r, c = idx[j]
                    self.mask[i][r, c] = 1
                    # Weight stays whatever it was at init

            # Remove connections
            n_remove = max(1, int(n_conns * rate * 0.2))
            alive = (self.mask[i] == 1).nonzero(as_tuple=False)
            if len(alive) > n_remove + 2:
                idx = alive[torch.randperm(len(alive))[:n_remove]]
                for j in range(len(idx)):
                    r, c = idx[j]
                    self.mask[i][r, c] = 0

            # Rewire
            n_rewire = max(1, int(n_conns * rate * 0.1))
            alive = (self.mask[i] == 1).nonzero(as_tuple=False)
            if len(alive) > 0:
                idx = alive[torch.randperm(len(alive))[:n_rewire]]
                for j in range(len(idx)):
                    r, c = idx[j]
                    self.mask[i][r, c] = 0
                    new_c = random.randint(0, fo - 1)
                    self.mask[i][r, new_c] = 1

    def mutate_weights_only(self, rate=0.15):
        """Change ONLY weight values. Structure stays fixed."""
        for i in range(self.n_layers):
            noise = torch.randn_like(self.W[i]) * 0.1 * rate
            self.W[i] += noise * self.mask[i]

    def mutate_both(self, rate=0.15):
        """Change structure AND weights."""
        for i in range(self.n_layers):
            fi, fo = self.layer_sizes[i], self.layer_sizes[i + 1]
            n_conns = int(self.mask[i].sum().item())

            # Weights
            noise = torch.randn_like(self.W[i]) * 0.1 * rate
            self.W[i] += noise * self.mask[i]

            # Add
            n_add = max(1, int(n_conns * rate * 0.3))
            dead = (self.mask[i] == 0).nonzero(as_tuple=False)
            if len(dead) > 0:
                idx = dead[torch.randperm(len(dead))[:n_add]]
                for j in range(len(idx)):
                    r, c = idx[j]
                    self.mask[i][r, c] = 1
                    self.W[i][r, c] = random.gauss(0, math.sqrt(2.0 / fi))

            # Remove
            n_remove = max(1, int(n_conns * rate * 0.2))
            alive = (self.mask[i] == 1).nonzero(as_tuple=False)
            if len(alive) > n_remove + 2:
                idx = alive[torch.randperm(len(alive))[:n_remove]]
                for j in range(len(idx)):
                    r, c = idx[j]
                    self.mask[i][r, c] = 0

            # Rewire
            n_rewire = max(1, int(n_conns * rate * 0.1))
            alive = (self.mask[i] == 1).nonzero(as_tuple=False)
            if len(alive) > 0:
                idx = alive[torch.randperm(len(alive))[:n_rewire]]
                for j in range(len(idx)):
                    r, c = idx[j]
                    self.mask[i][r, c] = 0
                    new_c = random.randint(0, fo - 1)
                    self.mask[i][r, new_c] = 1
                    self.W[i][r, new_c] = self.W[i][r, c]


def evaluate(net, X, targets, vocab):
    logits = net.forward(X)
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    correct = (preds == targets).float().mean().item()
    target_probs = probs[range(vocab), targets].mean().item()
    return 0.5 * correct + 0.5 * target_probs


def run_evolution(name, mutate_fn, vocab, X, targets, density=0.3,
                  pop_size=200, generations=500, elite=20):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    population = [SparseNet([vocab, 16, 16, vocab], density=density)
                  for _ in range(pop_size)]

    # For weights-only: make all fully connected
    if "WEIGHTS" in name and "STRUCTURE" not in name:
        for net in population:
            for i in range(net.n_layers):
                net.mask[i] = torch.ones_like(net.mask[i])

    best_ever_score = 0
    best_ever_net = None
    t0 = time.time()

    for gen in range(generations):
        for net in population:
            net.score = evaluate(net, X, targets, vocab)

        population.sort(key=lambda n: n.score, reverse=True)

        if population[0].score > best_ever_score:
            best_ever_score = population[0].score
            best_ever_net = population[0].clone()

        if (gen + 1) % 100 == 0 or gen == 0:
            best = population[0]
            logits = best.forward(X)
            acc = (logits.argmax(-1) == targets).float().mean().item() * 100
            conns = best.count_connections()
            elapsed = time.time() - t0
            print(f"  Gen {gen+1:4d} | Score: {best.score:.3f} | "
                  f"Acc: {acc:.0f}% | Conns: {conns} | {elapsed:.1f}s")

        if best_ever_score > 0.99:
            print(f"  → Solved at gen {gen+1}!")
            break

        new_pop = []
        for i in range(elite):
            new_pop.append(population[i].clone())

        while len(new_pop) < pop_size:
            a, b = random.sample(population[:pop_size//3], 2)
            parent = a if a.score >= b.score else b
            child = parent.clone()
            mutate_fn(child)
            new_pop.append(child)

        population = new_pop

    elapsed = time.time() - t0

    # Final acc
    logits = best_ever_net.forward(X)
    acc = (logits.argmax(-1) == targets).float().mean().item() * 100
    conns = best_ever_net.count_connections()
    max_c = sum(m.numel() for m in best_ever_net.mask)
    print(f"  → Final: {acc:.0f}% | {conns}/{max_c} conns | {elapsed:.1f}s")

    return acc, elapsed


def run():
    vocab = 8
    random.seed(42)
    torch.manual_seed(42)
    perm = torch.randperm(vocab)
    targets = perm
    X = torch.eye(vocab)

    print(f"Task: {list(range(vocab))} → {perm.tolist()}\n")

    # Test 1: ONLY structure
    random.seed(42); torch.manual_seed(42)
    acc1, t1 = run_evolution(
        "Test 1: ONLY STRUCTURE (weights = random init, never change)",
        lambda net: net.mutate_structure_only(0.15),
        vocab, X, targets)

    # Test 2: ONLY weights
    random.seed(42); torch.manual_seed(42)
    acc2, t2 = run_evolution(
        "Test 2: ONLY WEIGHTS (fully connected, no structure change)",
        lambda net: net.mutate_weights_only(0.15),
        vocab, X, targets, density=1.0)

    # Test 3: BOTH
    random.seed(42); torch.manual_seed(42)
    acc3, t3 = run_evolution(
        "Test 3: BOTH (structure + weights)",
        lambda net: net.mutate_both(0.15),
        vocab, X, targets)

    print(f"\n{'='*55}")
    print(f"  ABLATION RESULTS")
    print(f"{'='*55}")
    print(f"  Structure only:  {acc1:.0f}%  ({t1:.1f}s)")
    print(f"  Weights only:    {acc2:.0f}%  ({t2:.1f}s)")
    print(f"  Both:            {acc3:.0f}%  ({t3:.1f}s)")
    print(f"{'='*55}")


if __name__ == "__main__":
    run()
