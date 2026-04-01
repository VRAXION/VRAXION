"""
No-Backprop v8 — Neuroevolution: Evolve STRUCTURE, not weights
================================================================
Sparse networks represented as MASKED weight matrices (fast torch ops).
- Population of random topologies (masks)
- Evaluate → Select → Mutate the STRUCTURE
- No backprop. No gradients. Pure evolution.
"""

import torch
import torch.nn.functional as F
import math
import random
import time


class SparseNet:
    """
    Network = weight matrix * binary mask.
    Mask defines STRUCTURE (what's connected).
    Weights define strength.
    Evolution changes BOTH.
    """

    def __init__(self, layer_sizes, density=0.3):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1

        self.W = []
        self.mask = []
        for i in range(self.n_layers):
            fi, fo = layer_sizes[i], layer_sizes[i + 1]
            s = math.sqrt(2.0 / fi)
            self.W.append(torch.randn(fi, fo) * s)
            # Random sparse mask
            self.mask.append((torch.rand(fi, fo) < density).float())

        self.score = 0.0

    def forward(self, x):
        h = x
        for i in range(self.n_layers):
            h = h @ (self.W[i] * self.mask[i])  # masked weights
            if i < self.n_layers - 1:
                h = torch.relu(h)
        return h

    def count_connections(self):
        return sum(int(m.sum().item()) for m in self.mask)

    def max_connections(self):
        return sum(m.numel() for m in self.mask)

    def clone(self):
        new = SparseNet.__new__(SparseNet)
        new.layer_sizes = self.layer_sizes[:]
        new.n_layers = self.n_layers
        new.W = [w.clone() for w in self.W]
        new.mask = [m.clone() for m in self.mask]
        new.score = self.score
        return new

    def mutate(self, rate=0.1):
        for i in range(self.n_layers):
            fi, fo = self.layer_sizes[i], self.layer_sizes[i + 1]
            n_conns = int(self.mask[i].sum().item())

            # 1. Perturb existing weights (only where mask=1)
            noise = torch.randn_like(self.W[i]) * 0.1 * rate
            self.W[i] += noise * self.mask[i]

            # 2. Add new connections (structural mutation)
            n_add = max(1, int(n_conns * rate * 0.3))
            dead = (self.mask[i] == 0).nonzero(as_tuple=False)
            if len(dead) > 0:
                idx = dead[torch.randperm(len(dead))[:n_add]]
                for j in range(len(idx)):
                    r, c = idx[j]
                    self.mask[i][r, c] = 1
                    self.W[i][r, c] = random.gauss(0, math.sqrt(2.0 / fi))

            # 3. Remove connections (structural mutation)
            n_remove = max(1, int(n_conns * rate * 0.2))
            alive = (self.mask[i] == 1).nonzero(as_tuple=False)
            if len(alive) > n_remove + 2:  # keep at least some
                idx = alive[torch.randperm(len(alive))[:n_remove]]
                for j in range(len(idx)):
                    r, c = idx[j]
                    self.mask[i][r, c] = 0

            # 4. Rewire: move a connection to a different target
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
    """Batch evaluate — all inputs at once."""
    logits = net.forward(X)  # [vocab, vocab]
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    correct = (preds == targets).float().mean().item()
    # Smooth score: accuracy + avg prob on correct
    target_probs = probs[range(vocab), targets].mean().item()
    return 0.5 * correct + 0.5 * target_probs


def run():
    vocab = 8
    random.seed(42)
    torch.manual_seed(42)
    perm = torch.randperm(vocab)
    targets = perm

    # Batch input: all 8 one-hot vectors
    X = torch.eye(vocab)

    print(f"Task: learn mapping {list(range(vocab))} → {perm.tolist()}")
    print(f"(1-to-1 permutation)\n")

    # ═══════ NEUROEVOLUTION ═══════
    print("=" * 55)
    print("  Neuroevolution — Evolve STRUCTURE")
    print("  No backprop. No gradients. No Hebbian.")
    print("=" * 55)

    POP_SIZE = 200
    GENERATIONS = 1000
    ELITE = 20
    MUTATE_RATE = 0.15
    DENSITY = 0.3

    population = [SparseNet([vocab, 16, 16, vocab], density=DENSITY)
                  for _ in range(POP_SIZE)]

    best_ever_score = 0
    best_ever_net = None
    t0 = time.time()

    for gen in range(GENERATIONS):
        # Evaluate all
        for net in population:
            net.score = evaluate(net, X, targets, vocab)

        population.sort(key=lambda n: n.score, reverse=True)

        if population[0].score > best_ever_score:
            best_ever_score = population[0].score
            best_ever_net = population[0].clone()

        if (gen + 1) % 100 == 0 or gen == 0:
            best = population[0]
            avg_score = sum(n.score for n in population) / POP_SIZE
            conns = best.count_connections()
            max_c = best.max_connections()

            logits = best.forward(X)
            acc = (logits.argmax(-1) == targets).float().mean().item() * 100

            elapsed = time.time() - t0
            print(f"  Gen {gen+1:4d} | Best: {best.score:.3f} | "
                  f"Avg: {avg_score:.3f} | Acc: {acc:.0f}% | "
                  f"Conns: {conns}/{max_c} | {elapsed:.1f}s")

        # Early stop
        if best_ever_score > 0.99:
            print(f"  → Solved at gen {gen+1}!")
            break

        # Selection + reproduction
        new_pop = []

        # Elite
        for i in range(ELITE):
            new_pop.append(population[i].clone())

        # Children
        while len(new_pop) < POP_SIZE:
            # Tournament
            a, b = random.sample(population[:POP_SIZE//3], 2)
            parent = a if a.score >= b.score else b
            child = parent.clone()
            child.mutate(rate=MUTATE_RATE)
            new_pop.append(child)

        population = new_pop

    elapsed = time.time() - t0

    # ═══════ FINAL ═══════
    net = best_ever_net
    print(f"\n  Final mapping (score={best_ever_score:.3f}, {elapsed:.1f}s):")
    correct = 0
    for val in range(vocab):
        x = torch.zeros(vocab)
        x[val] = 1.0
        logits = net.forward(x)
        pred_idx = logits.argmax().item()
        target = perm[val].item()
        mark = "✓" if pred_idx == target else "✗"
        probs = torch.softmax(logits, dim=-1)
        conf = probs[pred_idx].item()
        print(f"    {val} → {pred_idx} (exp:{target}) {mark}  conf: {conf:.3f}")
        if pred_idx == target:
            correct += 1

    evo_acc = correct / vocab * 100
    conns = net.count_connections()
    max_c = net.max_connections()
    print(f"\n  Connections: {conns}/{max_c} ({conns/max_c*100:.0f}% density)")

    # Show structure
    print(f"\n  Structure (mask sparsity per layer):")
    for i, m in enumerate(net.mask):
        density = m.mean().item() * 100
        print(f"    Layer {i}: {m.shape[0]}×{m.shape[1]} | "
              f"{int(m.sum())}/{m.numel()} connections ({density:.0f}%)")

    # ═══════ BACKPROP COMPARISON ═══════
    print(f"\n{'='*55}")
    print("  Backprop comparison (same hidden size)")
    print("=" * 55)

    import torch.nn as nn
    bp = nn.Sequential(nn.Linear(vocab, 16), nn.ReLU(),
                       nn.Linear(16, 16), nn.ReLU(),
                       nn.Linear(16, vocab))
    opt = torch.optim.Adam(bp.parameters(), lr=1e-3)

    for epoch in range(2000):
        for val in range(vocab):
            x = torch.zeros(vocab); x[val] = 1.0
            target = perm[val].item()
            loss = F.cross_entropy(bp(x).unsqueeze(0), torch.tensor([target]))
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 500 == 0:
            c = sum(1 for v in range(vocab)
                    if bp(F.one_hot(torch.tensor(v), vocab).float()).argmax().item()
                    == perm[v].item())
            print(f"  Epoch {epoch+1:4d} | Acc: {c/vocab*100:.0f}%")

    bp_acc = sum(1 for v in range(vocab)
                 if bp(F.one_hot(torch.tensor(v),
                 vocab).float()).argmax().item() == perm[v].item()) / vocab * 100

    print(f"\n{'='*55}")
    print(f"  Neuroevolution (structure): {evo_acc:.0f}%")
    print(f"  Backprop (Adam):            {bp_acc:.0f}%")
    print(f"{'='*55}")


if __name__ == "__main__":
    run()
