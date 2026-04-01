"""
No-Backprop v9 — Two-Phase Brain Development
===============================================
Phase 1 (infant): Evolve STRUCTURE (pruning, wiring)
  - Weights stay random
  - Only the connection pattern changes
  - Like 0-4 years old

Phase 2 (mature): Evolve WEIGHTS (fine-tuning)
  - Structure is FROZEN
  - Only weight values change
  - Like 4+ years old

No backprop. No gradients. Pure evolution in two phases.
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

    def mutate_structure(self, rate=0.15):
        """Phase 1: change ONLY structure."""
        for i in range(self.n_layers):
            fi, fo = self.layer_sizes[i], self.layer_sizes[i + 1]
            n_conns = int(self.mask[i].sum().item())

            # Add
            n_add = max(1, int(n_conns * rate * 0.3))
            dead = (self.mask[i] == 0).nonzero(as_tuple=False)
            if len(dead) > 0:
                idx = dead[torch.randperm(len(dead))[:n_add]]
                for j in range(len(idx)):
                    self.mask[i][idx[j][0], idx[j][1]] = 1

            # Remove
            n_remove = max(1, int(n_conns * rate * 0.2))
            alive = (self.mask[i] == 1).nonzero(as_tuple=False)
            if len(alive) > n_remove + 2:
                idx = alive[torch.randperm(len(alive))[:n_remove]]
                for j in range(len(idx)):
                    self.mask[i][idx[j][0], idx[j][1]] = 0

            # Rewire
            n_rewire = max(1, int(n_conns * rate * 0.1))
            alive = (self.mask[i] == 1).nonzero(as_tuple=False)
            if len(alive) > 0:
                idx = alive[torch.randperm(len(alive))[:n_rewire]]
                for j in range(len(idx)):
                    r, c = idx[j][0].item(), idx[j][1].item()
                    self.mask[i][r, c] = 0
                    new_c = random.randint(0, fo - 1)
                    self.mask[i][r, new_c] = 1

    def mutate_weights(self, rate=0.15):
        """Phase 2: change ONLY weights. Structure frozen."""
        for i in range(self.n_layers):
            noise = torch.randn_like(self.W[i]) * 0.1 * rate
            self.W[i] += noise * self.mask[i]


def evaluate(net, X, targets, vocab):
    logits = net.forward(X)
    probs = torch.softmax(logits, dim=-1)
    correct = (probs.argmax(-1) == targets).float().mean().item()
    target_probs = probs[range(vocab), targets].mean().item()
    return 0.5 * correct + 0.5 * target_probs


def evolve(population, mutate_fn, X, targets, vocab,
           generations, elite, label):
    best_ever_score = 0
    best_ever_net = None
    pop_size = len(population)
    t0 = time.time()

    for gen in range(generations):
        for net in population:
            net.score = evaluate(net, X, targets, vocab)
        population.sort(key=lambda n: n.score, reverse=True)

        if population[0].score > best_ever_score:
            best_ever_score = population[0].score
            best_ever_net = population[0].clone()

        if (gen + 1) % 50 == 0 or gen == 0:
            best = population[0]
            logits = best.forward(X)
            acc = (logits.argmax(-1) == targets).float().mean().item() * 100
            conns = best.count_connections()
            max_c = sum(m.numel() for m in best.mask)
            elapsed = time.time() - t0
            print(f"  {label} Gen {gen+1:4d} | Score: {best.score:.3f} | "
                  f"Acc: {acc:.0f}% | Conns: {conns}/{max_c} | {elapsed:.1f}s")

        if best_ever_score > 0.99:
            elapsed = time.time() - t0
            print(f"  → Solved at gen {gen+1}! ({elapsed:.1f}s)")
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
        population[:] = new_pop

    return best_ever_net, population


def run():
    vocab = 8
    random.seed(42)
    torch.manual_seed(42)
    perm = torch.randperm(vocab)
    targets = perm
    X = torch.eye(vocab)

    print(f"Task: {list(range(vocab))} → {perm.tolist()}\n")

    POP = 200
    ELITE = 20

    # ══════════════════════════════════════════
    #  TWO-PHASE (the brain way)
    # ══════════════════════════════════════════
    print("=" * 60)
    print("  TWO-PHASE BRAIN DEVELOPMENT")
    print("=" * 60)

    random.seed(42); torch.manual_seed(42)
    population = [SparseNet([vocab, 16, 16, vocab], density=0.5)
                  for _ in range(POP)]
    t_total = time.time()

    # ── Phase 1: INFANT — structure only ──
    print(f"\n  ── Phase 1: INFANT (structure evolves, weights random) ──")
    best_struct, population = evolve(
        population,
        lambda net: net.mutate_structure(0.15),
        X, targets, vocab,
        generations=300, elite=ELITE, label="[STRUCT]")

    struct_acc = (best_struct.forward(X).argmax(-1) == targets).float().mean().item() * 100
    struct_conns = best_struct.count_connections()
    print(f"\n  Structure found: {struct_acc:.0f}% accuracy, {struct_conns} connections")
    print(f"  Structure per layer:")
    for i, m in enumerate(best_struct.mask):
        print(f"    Layer {i}: {int(m.sum())}/{m.numel()} "
              f"({m.mean().item()*100:.0f}%)")

    # ── Phase 2: MATURE — freeze structure, evolve weights ──
    print(f"\n  ── Phase 2: MATURE (structure FROZEN, weights evolve) ──")

    # Seed Phase 2 population from best structure
    population = []
    for _ in range(POP):
        net = best_struct.clone()
        # Randomize weights but keep structure
        for i in range(net.n_layers):
            fi = net.layer_sizes[i]
            s = math.sqrt(2.0 / fi)
            net.W[i] = torch.randn_like(net.W[i]) * s
        population.append(net)

    best_final, _ = evolve(
        population,
        lambda net: net.mutate_weights(0.15),
        X, targets, vocab,
        generations=500, elite=ELITE, label="[WEIGHT]")

    phase2_time = time.time() - t_total

    # ══════════════════════════════════════════
    #  SINGLE-PHASE comparisons
    # ══════════════════════════════════════════

    # Single phase: weights only (fully connected)
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Weights only (fully connected)")
    print(f"{'='*60}")
    random.seed(42); torch.manual_seed(42)
    pop_w = [SparseNet([vocab, 16, 16, vocab], density=1.0) for _ in range(POP)]
    best_w, _ = evolve(
        pop_w,
        lambda net: net.mutate_weights(0.15),
        X, targets, vocab,
        generations=500, elite=ELITE, label="[W-ONLY]")

    # Single phase: both together
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Both together (no phases)")
    print(f"{'='*60}")

    def mutate_both(net):
        net.mutate_structure(0.15)
        net.mutate_weights(0.15)

    random.seed(42); torch.manual_seed(42)
    pop_b = [SparseNet([vocab, 16, 16, vocab], density=0.5) for _ in range(POP)]
    best_b, _ = evolve(
        pop_b, mutate_both,
        X, targets, vocab,
        generations=500, elite=ELITE, label="[BOTH ]")

    # ══════════════════════════════════════════
    #  FINAL
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")

    for name, net in [("Two-phase (brain)", best_final),
                      ("Weights only", best_w),
                      ("Both together", best_b)]:
        logits = net.forward(X)
        acc = (logits.argmax(-1) == targets).float().mean().item() * 100
        score = evaluate(net, X, targets, vocab)
        conns = net.count_connections()
        max_c = sum(m.numel() for m in net.mask)
        print(f"  {name:25s} | Acc: {acc:.0f}% | Score: {score:.3f} | "
              f"Conns: {conns}/{max_c}")

    # Show final mappings for two-phase
    print(f"\n  Two-phase final mapping:")
    for val in range(vocab):
        x = torch.zeros(vocab); x[val] = 1.0
        logits = best_final.forward(x)
        pred = logits.argmax().item()
        target = perm[val].item()
        conf = torch.softmax(logits, dim=-1)[pred].item()
        mark = "✓" if pred == target else "✗"
        print(f"    {val} → {pred} (exp:{target}) {mark}  conf: {conf:.3f}")


if __name__ == "__main__":
    run()
