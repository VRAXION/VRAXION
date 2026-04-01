"""
No-Backprop v11 — Continuous Annealing (Brain Development)
============================================================
No hard phases. One continuous process:
  - Start: high mutation rate (structure + weights) → rough solution
  - Gradually cool down → fine-tuning, associations
  - Structure never frozen, just slows down
  - Weights never reset, just refine
  - The rough solution IS the warm start for the fine solution

Like the brain: coarse wiring first, then refinement,
but it's a CONTINUUM not discrete phases.
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

    def mutate(self, struct_rate, weight_rate):
        """
        Both structure and weights mutate.
        Rates control how aggressively.
        As rates decrease → annealing → convergence.
        """
        for i in range(self.n_layers):
            fi, fo = self.layer_sizes[i], self.layer_sizes[i + 1]
            n_conns = max(1, int(self.mask[i].sum().item()))

            # ── Weight mutation (always active) ──
            noise = torch.randn_like(self.W[i]) * weight_rate
            self.W[i] += noise * self.mask[i]

            # ── Structure mutation (slows down over time) ──
            if struct_rate > 0.001:  # below this, don't bother
                # Add connections
                n_add = max(1, int(n_conns * struct_rate * 0.3))
                dead = (self.mask[i] == 0).nonzero(as_tuple=False)
                if len(dead) > 0:
                    idx = dead[torch.randperm(len(dead))[:n_add]]
                    for j in range(len(idx)):
                        r, c = idx[j][0].item(), idx[j][1].item()
                        self.mask[i][r, c] = 1
                        # New connection gets small weight (not disrupting)
                        self.W[i][r, c] = random.gauss(0, weight_rate)

                # Remove connections
                n_remove = max(1, int(n_conns * struct_rate * 0.2))
                alive = (self.mask[i] == 1).nonzero(as_tuple=False)
                if len(alive) > n_remove + 2:
                    idx = alive[torch.randperm(len(alive))[:n_remove]]
                    for j in range(len(idx)):
                        self.mask[i][idx[j][0], idx[j][1]] = 0

                # Rewire
                n_rewire = max(1, int(n_conns * struct_rate * 0.1))
                alive = (self.mask[i] == 1).nonzero(as_tuple=False)
                if len(alive) > 0:
                    idx = alive[torch.randperm(len(alive))[:n_rewire]]
                    for j in range(len(idx)):
                        r, c = idx[j][0].item(), idx[j][1].item()
                        self.mask[i][r, c] = 0
                        new_c = random.randint(0, fo - 1)
                        self.mask[i][r, new_c] = 1
                        self.W[i][r, new_c] = self.W[i][r, c]


def evaluate(net, X, targets):
    logits = net.forward(X)
    probs = torch.softmax(logits, dim=-1)
    correct = (probs.argmax(-1) == targets).float().mean().item()
    target_probs = probs[range(len(targets)), targets].mean().item()
    return 0.5 * correct + 0.5 * target_probs


def run_annealing(name, X, targets, layer_sizes, vocab,
                  pop_size=200, generations=800, elite=20,
                  struct_start=0.20, struct_end=0.01,
                  weight_start=0.15, weight_end=0.02):
    """
    Continuous annealing: rates decay smoothly from start to end.
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Struct rate: {struct_start} → {struct_end}")
    print(f"  Weight rate: {weight_start} → {weight_end}")
    print(f"{'='*60}")

    population = [SparseNet(layer_sizes, density=0.5) for _ in range(pop_size)]
    best_ever_score = 0
    best_ever_net = None
    t0 = time.time()

    for gen in range(generations):
        # ── Annealing schedule ──
        progress = gen / max(generations - 1, 1)  # 0 → 1
        struct_rate = struct_start * (struct_end / struct_start) ** progress
        weight_rate = weight_start * (weight_end / weight_start) ** progress

        # Evaluate
        for net in population:
            net.score = evaluate(net, X, targets)
        population.sort(key=lambda n: n.score, reverse=True)

        if population[0].score > best_ever_score:
            best_ever_score = population[0].score
            best_ever_net = population[0].clone()

        if (gen + 1) % 100 == 0 or gen == 0:
            best = population[0]
            logits = best.forward(X)
            acc = (logits.argmax(-1) == targets).float().mean().item() * 100
            conns = best.count_connections()
            max_c = sum(m.numel() for m in best.mask)
            elapsed = time.time() - t0
            print(f"  Gen {gen+1:4d} | Score: {best.score:.3f} | "
                  f"Acc: {acc:.0f}% | Conns: {conns}/{max_c} | "
                  f"s_rate={struct_rate:.4f} w_rate={weight_rate:.4f} | "
                  f"{elapsed:.1f}s")

        if best_ever_score > 0.99:
            elapsed = time.time() - t0
            print(f"  → Solved at gen {gen+1}! ({elapsed:.1f}s)")
            break

        # Selection + reproduction
        new_pop = []
        for i in range(elite):
            new_pop.append(population[i].clone())
        while len(new_pop) < pop_size:
            a, b = random.sample(population[:pop_size//3], 2)
            parent = a if a.score >= b.score else b
            child = parent.clone()
            child.mutate(struct_rate, weight_rate)
            new_pop.append(child)
        population[:] = new_pop

    elapsed = time.time() - t0

    # Final results
    logits = best_ever_net.forward(X)
    acc = (logits.argmax(-1) == targets).float().mean().item() * 100
    conns = best_ever_net.count_connections()
    max_c = sum(m.numel() for m in best_ever_net.mask)

    print(f"\n  Final: {acc:.0f}% | Score: {best_ever_score:.3f} | "
          f"Conns: {conns}/{max_c} ({conns/max_c*100:.0f}%) | {elapsed:.1f}s")

    return best_ever_net, best_ever_score, acc, elapsed


def run():
    # ═══════ TASK 1: 8-class ═══════
    torch.manual_seed(42)
    vocab1 = 8
    perm1 = torch.randperm(vocab1)
    X1 = torch.eye(vocab1)

    print(f"Task 1: {list(range(vocab1))} → {perm1.tolist()}")

    random.seed(42); torch.manual_seed(42)
    net1, score1, acc1, t1 = run_annealing(
        "8-class — Continuous Annealing",
        X1, perm1, [vocab1, 16, 16, vocab1], vocab1)

    # ═══════ TASK 2: 16-class ═══════
    torch.manual_seed(42)
    vocab2 = 16
    perm2 = torch.randperm(vocab2)
    X2 = torch.eye(vocab2)

    print(f"\nTask 2: 16-class permutation")

    random.seed(42); torch.manual_seed(42)
    net2, score2, acc2, t2 = run_annealing(
        "16-class — Continuous Annealing",
        X2, perm2, [vocab2, 32, 32, vocab2], vocab2,
        generations=1000)

    # ═══════ TASK 3: 32-class (even harder) ═══════
    torch.manual_seed(42)
    vocab3 = 32
    perm3 = torch.randperm(vocab3)
    X3 = torch.eye(vocab3)

    print(f"\nTask 3: 32-class permutation")

    random.seed(42); torch.manual_seed(42)
    net3, score3, acc3, t3 = run_annealing(
        "32-class — Continuous Annealing",
        X3, perm3, [vocab3, 64, 64, vocab3], vocab3,
        pop_size=300, generations=1500)

    # ═══════ BACKPROP COMPARISONS ═══════
    print(f"\n{'='*60}")
    print(f"  BACKPROP COMPARISONS")
    print(f"{'='*60}")

    import torch.nn as nn
    bp_results = {}
    for vname, vocab, perm in [("8-class", vocab1, perm1),
                                ("16-class", vocab2, perm2),
                                ("32-class", vocab3, perm3)]:
        X = torch.eye(vocab)
        bp = nn.Sequential(
            nn.Linear(vocab, vocab * 2), nn.ReLU(),
            nn.Linear(vocab * 2, vocab * 2), nn.ReLU(),
            nn.Linear(vocab * 2, vocab))
        opt = torch.optim.Adam(bp.parameters(), lr=1e-3)

        for epoch in range(2000):
            loss = F.cross_entropy(bp(X), perm)
            opt.zero_grad(); loss.backward(); opt.step()

        bp_acc = (bp(X).argmax(-1) == perm).float().mean().item() * 100
        bp_results[vname] = bp_acc
        print(f"  {vname}: {bp_acc:.0f}%")

    # ═══════ SUMMARY ═══════
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<15} {'Annealing':>10} {'Backprop':>10}")
    print(f"  {'-'*35}")
    print(f"  {'8-class':<15} {acc1:>9.0f}% {bp_results['8-class']:>9.0f}%")
    print(f"  {'16-class':<15} {acc2:>9.0f}% {bp_results['16-class']:>9.0f}%")
    print(f"  {'32-class':<15} {acc3:>9.0f}% {bp_results['32-class']:>9.0f}%")

    # Show structure of best 32-class net
    if net3:
        print(f"\n  32-class structure:")
        for i, m in enumerate(net3.mask):
            d = m.mean().item() * 100
            print(f"    Layer {i}: {m.shape[0]}×{m.shape[1]} → {int(m.sum())}/{m.numel()} ({d:.0f}%)")

    # Show some mappings
    print(f"\n  32-class mappings (first 16):")
    logits = net3.forward(X3)
    probs = torch.softmax(logits, dim=-1)
    for val in range(min(32, 16)):
        pred = logits[val].argmax().item()
        tgt = perm3[val].item()
        conf = probs[val][pred].item()
        mark = "✓" if pred == tgt else "✗"
        print(f"    {val:2d} → {pred:2d} (exp:{tgt:2d}) {mark}  conf: {conf:.3f}")


if __name__ == "__main__":
    run()
