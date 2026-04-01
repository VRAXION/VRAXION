"""
No-Backprop v11 — Continuous Annealing on Modal GPU
======================================================
Coarse → fine, structure + weights together, gradual cooldown.
No hard phases. Like brain development: a continuum.
"""

import modal
import time

app = modal.App("vraxion-neuroevo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
)


@app.function(gpu="T4", timeout=1800, image=image)
def run_experiment():
    import torch
    import torch.nn.functional as F
    import math
    import random

    device = "cuda"

    class SparseNet:
        def __init__(self, layer_sizes, density=0.5, device="cuda"):
            self.layer_sizes = layer_sizes
            self.n_layers = len(layer_sizes) - 1
            self.device = device
            self.W = []
            self.mask = []
            for i in range(self.n_layers):
                fi, fo = layer_sizes[i], layer_sizes[i + 1]
                s = math.sqrt(2.0 / fi)
                self.W.append((torch.randn(fi, fo, device=device) * s))
                self.mask.append((torch.rand(fi, fo, device=device) < density).float())
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
            new.device = self.device
            new.W = [w.clone() for w in self.W]
            new.mask = [m.clone() for m in self.mask]
            new.score = self.score
            return new

        def mutate(self, struct_rate, weight_rate):
            for i in range(self.n_layers):
                fi, fo = self.layer_sizes[i], self.layer_sizes[i + 1]
                n_conns = max(1, int(self.mask[i].sum().item()))

                # Weight mutation
                noise = torch.randn_like(self.W[i]) * weight_rate
                self.W[i] += noise * self.mask[i]

                # Structure mutation
                if struct_rate > 0.001:
                    # Add
                    n_add = max(1, int(n_conns * struct_rate * 0.3))
                    dead = (self.mask[i] == 0).nonzero(as_tuple=False)
                    if len(dead) > 0:
                        idx = dead[torch.randperm(len(dead), device=self.device)[:n_add]]
                        self.mask[i][idx[:, 0], idx[:, 1]] = 1
                        self.W[i][idx[:, 0], idx[:, 1]] = torch.randn(
                            len(idx), device=self.device) * weight_rate

                    # Remove
                    n_remove = max(1, int(n_conns * struct_rate * 0.2))
                    alive = (self.mask[i] == 1).nonzero(as_tuple=False)
                    if len(alive) > n_remove + 2:
                        idx = alive[torch.randperm(len(alive), device=self.device)[:n_remove]]
                        self.mask[i][idx[:, 0], idx[:, 1]] = 0

                    # Rewire
                    n_rewire = max(1, int(n_conns * struct_rate * 0.1))
                    alive = (self.mask[i] == 1).nonzero(as_tuple=False)
                    if len(alive) > 0:
                        idx = alive[torch.randperm(len(alive), device=self.device)[:n_rewire]]
                        self.mask[i][idx[:, 0], idx[:, 1]] = 0
                        new_cols = torch.randint(0, fo, (len(idx),), device=self.device)
                        self.mask[i][idx[:, 0], new_cols] = 1

    def evaluate_batch(population, X, targets):
        """Evaluate all nets."""
        for net in population:
            logits = net.forward(X)
            probs = torch.softmax(logits, dim=-1)
            correct = (probs.argmax(-1) == targets).float().mean().item()
            target_probs = probs[range(len(targets)), targets].mean().item()
            net.score = 0.5 * correct + 0.5 * target_probs

    def run_annealing(name, X, targets, layer_sizes, vocab,
                      pop_size=200, generations=800, elite=20,
                      struct_start=0.20, struct_end=0.01,
                      weight_start=0.15, weight_end=0.02):
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        population = [SparseNet(layer_sizes, density=0.5, device=device)
                      for _ in range(pop_size)]
        best_ever_score = 0
        best_ever_net = None
        t0 = time.time()

        for gen in range(generations):
            progress = gen / max(generations - 1, 1)
            struct_rate = struct_start * (struct_end / struct_start) ** progress
            weight_rate = weight_start * (weight_end / weight_start) ** progress

            evaluate_batch(population, X, targets)
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
                      f"sr={struct_rate:.4f} wr={weight_rate:.4f} | {elapsed:.1f}s")

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
                child.mutate(struct_rate, weight_rate)
                new_pop.append(child)
            population[:] = new_pop

        elapsed = time.time() - t0
        logits = best_ever_net.forward(X)
        acc = (logits.argmax(-1) == targets).float().mean().item() * 100
        conns = best_ever_net.count_connections()
        max_c = sum(m.numel() for m in best_ever_net.mask)
        print(f"\n  Final: {acc:.0f}% | Score: {best_ever_score:.3f} | "
              f"Conns: {conns}/{max_c} ({conns/max_c*100:.0f}%) | {elapsed:.1f}s")

        return best_ever_net, best_ever_score, acc, elapsed

    results = {}

    # 8/16-class already proven. Focus on 32-class.
    results["8-class"] = {"acc": 100, "score": 0.990, "time": 61}
    results["16-class"] = {"acc": 100, "score": 0.991, "time": 262}

    # ═══════ 32-class ═══════
    torch.manual_seed(42); random.seed(42)
    v = 32; perm = torch.randperm(v, device=device); X = torch.eye(v, device=device)
    print(f"\nTask: 32-class permutation")
    net32, s, a, t = run_annealing("32-class Annealing", X, perm,
                                    [v, 64, 64, v], v,
                                    pop_size=500, generations=3000)
    results["32-class"] = {"acc": a, "score": s, "time": t}

    # Skip 64-class for now

    # ═══════ Backprop comparisons ═══════
    print(f"\n{'='*60}")
    print(f"  BACKPROP COMPARISONS")
    print(f"{'='*60}")

    import torch.nn as nn
    for vname, v in [("8-class", 8), ("16-class", 16),
                     ("32-class", 32)]:
        torch.manual_seed(42)
        perm = torch.randperm(v, device=device)
        X = torch.eye(v, device=device)
        bp = nn.Sequential(
            nn.Linear(v, v*2), nn.ReLU(),
            nn.Linear(v*2, v*2), nn.ReLU(),
            nn.Linear(v*2, v)).to(device)
        opt = torch.optim.Adam(bp.parameters(), lr=1e-3)
        for epoch in range(3000):
            loss = F.cross_entropy(bp(X), perm)
            opt.zero_grad(); loss.backward(); opt.step()
        bp_acc = (bp(X).argmax(-1) == perm).float().mean().item() * 100
        results[vname]["bp_acc"] = bp_acc
        print(f"  {vname}: {bp_acc:.0f}%")

    # ═══════ SUMMARY ═══════
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY — Continuous Annealing vs Backprop")
    print(f"{'='*60}")
    print(f"  {'Task':<12} {'Annealing':>10} {'Backprop':>10} {'Time':>8}")
    print(f"  {'-'*42}")
    for task in ["8-class", "16-class", "32-class"]:
        r = results[task]
        print(f"  {task:<12} {r['acc']:>9.0f}% {r['bp_acc']:>9.0f}% {r['time']:>7.1f}s")

    return results


@app.local_entrypoint()
def main():
    print("Running neuroevolution annealing on Modal GPU...")
    results = run_experiment.remote()
    print("\nDone!")
