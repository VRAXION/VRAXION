"""
No-Backprop v10 — Three-Phase Brain Development
==================================================
Phase 1 INFANT:  Aggressive structure search (weights random)
Phase 2 MATURE:  Weight fine-tuning (structure frozen)
Phase 3 ADULT:   Both, but structure mutation very slow (adaptation)

Also: harder task — 16-class permutation + XOR-like non-linearity
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
        for i in range(self.n_layers):
            fi, fo = self.layer_sizes[i], self.layer_sizes[i + 1]
            n_conns = int(self.mask[i].sum().item())

            n_add = max(1, int(n_conns * rate * 0.3))
            dead = (self.mask[i] == 0).nonzero(as_tuple=False)
            if len(dead) > 0:
                idx = dead[torch.randperm(len(dead))[:n_add]]
                for j in range(len(idx)):
                    self.mask[i][idx[j][0], idx[j][1]] = 1

            n_remove = max(1, int(n_conns * rate * 0.2))
            alive = (self.mask[i] == 1).nonzero(as_tuple=False)
            if len(alive) > n_remove + 2:
                idx = alive[torch.randperm(len(alive))[:n_remove]]
                for j in range(len(idx)):
                    self.mask[i][idx[j][0], idx[j][1]] = 0

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
        for i in range(self.n_layers):
            noise = torch.randn_like(self.W[i]) * 0.1 * rate
            self.W[i] += noise * self.mask[i]

    def mutate_adult(self, struct_rate=0.03, weight_rate=0.15):
        """Phase 3: both, but structure mutation very gentle."""
        self.mutate_weights(weight_rate)
        self.mutate_structure(struct_rate)


def evaluate(net, X, targets, vocab):
    logits = net.forward(X)
    probs = torch.softmax(logits, dim=-1)
    correct = (probs.argmax(-1) == targets).float().mean().item()
    target_probs = probs[range(len(targets)), targets].mean().item()
    return 0.5 * correct + 0.5 * target_probs


def evolve(population, mutate_fn, X, targets, vocab,
           generations, elite, label, print_every=50):
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

        if (gen + 1) % print_every == 0 or gen == 0:
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

    return best_ever_net, best_ever_score, population


def run_task(name, vocab, X, targets, layer_sizes, density=0.5,
             pop_size=200, elite=20):
    print(f"\n{'#'*65}")
    print(f"  TASK: {name}")
    print(f"  {vocab} classes, {len(targets)} mappings")
    print(f"{'#'*65}")

    # ══════════════════════════════════════
    #  THREE-PHASE
    # ══════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  THREE-PHASE BRAIN DEVELOPMENT")
    print(f"{'='*60}")

    t_total = time.time()

    # Phase 1: INFANT
    print(f"\n  ── Phase 1: INFANT (structure only) ──")
    random.seed(42); torch.manual_seed(42)
    pop = [SparseNet(layer_sizes, density=density) for _ in range(pop_size)]
    best_struct, score1, pop = evolve(
        pop, lambda n: n.mutate_structure(0.15),
        X, targets, vocab,
        generations=300, elite=elite, label="[INFANT]")

    struct_acc = (best_struct.forward(X).argmax(-1) == targets).float().mean().item() * 100
    print(f"\n  Phase 1 done: {struct_acc:.0f}% acc, score {score1:.3f}")

    # Phase 2: MATURE — freeze structure, new random weights
    print(f"\n  ── Phase 2: MATURE (weights only, structure frozen) ──")
    pop = []
    for _ in range(pop_size):
        net = best_struct.clone()
        for i in range(net.n_layers):
            fi = net.layer_sizes[i]
            s = math.sqrt(2.0 / fi)
            net.W[i] = torch.randn_like(net.W[i]) * s
        pop.append(net)

    best_mature, score2, pop = evolve(
        pop, lambda n: n.mutate_weights(0.15),
        X, targets, vocab,
        generations=500, elite=elite, label="[MATURE]")

    mature_acc = (best_mature.forward(X).argmax(-1) == targets).float().mean().item() * 100
    print(f"\n  Phase 2 done: {mature_acc:.0f}% acc, score {score2:.3f}")

    # Phase 3: ADULT — both, but structure very slow
    print(f"\n  ── Phase 3: ADULT (both, structure slow) ──")
    # Start from best mature weights
    pop = []
    for _ in range(pop_size):
        net = best_mature.clone()
        # Small perturbation to create diversity
        net.mutate_weights(0.05)
        pop.append(net)

    best_adult, score3, pop = evolve(
        pop, lambda n: n.mutate_adult(struct_rate=0.03, weight_rate=0.10),
        X, targets, vocab,
        generations=300, elite=elite, label="[ADULT ]")

    adult_acc = (best_adult.forward(X).argmax(-1) == targets).float().mean().item() * 100
    total_time = time.time() - t_total
    print(f"\n  Phase 3 done: {adult_acc:.0f}% acc, score {score3:.3f}")

    # ══════════════════════════════════════
    #  COMPARISON: weights only (no structure)
    # ══════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Weights only (fully connected)")
    print(f"{'='*60}")
    random.seed(42); torch.manual_seed(42)
    pop_w = [SparseNet(layer_sizes, density=1.0) for _ in range(pop_size)]
    best_w, score_w, _ = evolve(
        pop_w, lambda n: n.mutate_weights(0.15),
        X, targets, vocab,
        generations=1000, elite=elite, label="[W-ONLY]")

    # ══════════════════════════════════════
    #  COMPARISON: Backprop
    # ══════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Backprop (Adam)")
    print(f"{'='*60}")
    import torch.nn as nn
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes) - 2:
            layers.append(nn.ReLU())
    bp = nn.Sequential(*layers)
    opt = torch.optim.Adam(bp.parameters(), lr=1e-3)

    for epoch in range(2000):
        logits = bp(X)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 500 == 0:
            acc = (bp(X).argmax(-1) == targets).float().mean().item() * 100
            print(f"  Epoch {epoch+1:4d} | Acc: {acc:.0f}% | Loss: {loss.item():.4f}")

    bp_acc = (bp(X).argmax(-1) == targets).float().mean().item() * 100

    # ══════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  RESULTS: {name}")
    print(f"{'='*60}")
    print(f"  Three-phase (brain):  {adult_acc:.0f}%  score={score3:.3f}  "
          f"conns={best_adult.count_connections()}/{sum(m.numel() for m in best_adult.mask)}  "
          f"({total_time:.0f}s)")
    print(f"  Weights only:         {(best_w.forward(X).argmax(-1)==targets).float().mean().item()*100:.0f}%  "
          f"score={score_w:.3f}")
    print(f"  Backprop:             {bp_acc:.0f}%")

    # Show mappings
    print(f"\n  Three-phase mappings:")
    logits = best_adult.forward(X)
    probs = torch.softmax(logits, dim=-1)
    for val in range(min(vocab, 16)):
        pred = logits[val].argmax().item()
        tgt = targets[val].item()
        conf = probs[val][pred].item()
        mark = "✓" if pred == tgt else "✗"
        print(f"    {val:2d} → {pred:2d} (exp:{tgt:2d}) {mark}  conf: {conf:.3f}")

    return adult_acc, score3


def run():
    # ═══════ TASK 1: Easy (8-class permutation) ═══════
    torch.manual_seed(42)
    vocab1 = 8
    perm1 = torch.randperm(vocab1)
    X1 = torch.eye(vocab1)
    run_task("8-class permutation",
             vocab1, X1, perm1,
             layer_sizes=[vocab1, 16, 16, vocab1])

    # ═══════ TASK 2: Harder (16-class permutation) ═══════
    torch.manual_seed(42)
    vocab2 = 16
    perm2 = torch.randperm(vocab2)
    X2 = torch.eye(vocab2)
    run_task("16-class permutation",
             vocab2, X2, perm2,
             layer_sizes=[vocab2, 32, 32, vocab2])

    # ═══════ TASK 3: Non-linear (2-input XOR-like) ═══════
    # Input: 2 bits → output depends on XOR
    # [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0
    print(f"\n{'#'*65}")
    print(f"  TASK: XOR (non-linear, requires hidden layer)")
    print(f"{'#'*65}")
    X3 = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
    targets3 = torch.tensor([0, 1, 1, 0])
    vocab3 = 2
    run_task("XOR",
             vocab3, X3, targets3,
             layer_sizes=[2, 8, 2],
             density=0.5, pop_size=200)


if __name__ == "__main__":
    run()
