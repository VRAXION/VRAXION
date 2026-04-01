"""
No-Backprop v16b — Graph Network, 32-class only
==================================================
Same as v16 but focused on 32-class with longer timeout.
Optimized: batched forward pass with matrix ops.
"""

import modal
import time

app = modal.App("vraxion-graph-32")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(cpu=4, memory=2048, timeout=3600, image=image)
def run_experiment():
    import torch
    import math
    import random

    class GraphNet:
        def __init__(self, n_neurons, n_in, n_out, density=0.1):
            self.N = n_neurons
            self.n_in = n_in
            self.n_out = n_out
            s = math.sqrt(2.0 / n_neurons)
            self.W = torch.randn(n_neurons, n_neurons) * s
            self.mask = (torch.rand(n_neurons, n_neurons) < density).float()
            self.mask.fill_diagonal_(0)
            # Pre-compute effective weights
            self.Weff = self.W * self.mask

        def _update_weff(self):
            self.Weff = self.W * self.mask

        def forward(self, world, diff, ticks=8):
            act = torch.zeros(self.N)
            inp = torch.cat([world, diff])
            act[:self.n_in] = inp
            Weff = self.Weff
            for t in range(ticks):
                act = torch.relu(act @ Weff)
                act[:self.n_in] = inp
            return act[-self.n_out:]

        def count_connections(self):
            return int(self.mask.sum().item())

        def save_state(self):
            return (self.W.clone(), self.mask.clone(), self.Weff.clone())

        def restore_state(self, state):
            self.W = state[0].clone()
            self.mask = state[1].clone()
            self.Weff = state[2].clone()

        def mutate_structure(self, rate=0.05):
            action = random.choice(["add", "remove", "rewire"])
            if action == "add":
                dead = (self.mask == 0).nonzero(as_tuple=False)
                dead = dead[dead[:, 0] != dead[:, 1]]
                if len(dead) > 0:
                    n = max(1, int(len(dead) * rate))
                    idx = dead[torch.randperm(len(dead))[:n]]
                    for j in range(len(idx)):
                        r, c = idx[j][0].item(), idx[j][1].item()
                        self.mask[r, c] = 1
                        self.W[r, c] = random.gauss(0, math.sqrt(2.0 / self.N))
            elif action == "remove":
                alive = (self.mask == 1).nonzero(as_tuple=False)
                if len(alive) > 3:
                    n = max(1, int(len(alive) * rate))
                    idx = alive[torch.randperm(len(alive))[:n]]
                    for j in range(len(idx)):
                        self.mask[idx[j][0], idx[j][1]] = 0
            elif action == "rewire":
                alive = (self.mask == 1).nonzero(as_tuple=False)
                if len(alive) > 0:
                    n = max(1, int(len(alive) * rate))
                    idx = alive[torch.randperm(len(alive))[:n]]
                    for j in range(len(idx)):
                        r, c = idx[j][0].item(), idx[j][1].item()
                        self.mask[r, c] = 0
                        new_c = random.randint(0, self.N - 1)
                        while new_c == r:
                            new_c = random.randint(0, self.N - 1)
                        self.mask[r, new_c] = 1
                        self.W[r, new_c] = self.W[r, c]
            self._update_weff()

        def mutate_weights(self, scale=0.05):
            noise = torch.randn_like(self.W) * scale
            self.W += noise * self.mask
            self._update_weff()

    def evaluate(net, inputs, targets, vocab, ticks=8):
        total_score = 0.0
        correct = 0
        prev_diff = torch.zeros(vocab)
        for idx in range(len(inputs)):
            world = torch.zeros(vocab)
            world[inputs[idx]] = 1.0
            logits = net.forward(world, prev_diff, ticks=ticks)
            probs = torch.softmax(logits, dim=-1)
            predicted = probs.argmax().item()
            target = targets[idx].item()
            total_score += probs[target].item()
            if predicted == target:
                correct += 1
            target_vec = torch.zeros(vocab)
            target_vec[target] = 1.0
            prev_diff = target_vec - probs.detach()
        n = len(inputs)
        acc = correct / n
        avg_prob = total_score / n
        return 0.5 * acc + 0.5 * avg_prob, acc

    print(f"32-class Graph Network — Extended Run")
    print(f"=" * 60)

    torch.manual_seed(42); random.seed(42)
    v = 32
    perm = torch.randperm(v)
    inputs = list(range(v))
    n_neurons = 192
    n_in = v * 2
    n_out = v
    ticks = 8
    density = 0.10
    stale_limit = 30000
    max_attempts = 500000

    net = GraphNet(n_neurons, n_in, n_out, density=density)

    print(f"  Neurons: {n_neurons} | Ticks: {ticks} | Density: {density}")
    print(f"  Stale limit: {stale_limit}")

    current_score, current_acc = evaluate(net, inputs, perm, v, ticks)
    best_score = current_score
    phase = "STRUCTURE"
    kept = 0
    stale = 0
    phase_switched = False
    t0 = time.time()

    print(f"  Start: Acc={current_acc*100:.0f}% | Score: {current_score:.3f} | "
          f"Conns: {net.count_connections()}")

    for attempt in range(max_attempts):
        state = net.save_state()

        if phase == "STRUCTURE":
            net.mutate_structure(rate=0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(rate=0.02)
            else:
                net.mutate_weights(scale=0.05)

        new_score, new_acc = evaluate(net, inputs, perm, v, ticks)

        if new_score > current_score:
            current_score = new_score
            kept += 1
            stale = 0
            if current_score > best_score:
                best_score = current_score
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 3000 and not phase_switched:
            phase = "BOTH"
            phase_switched = True
            _, acc = evaluate(net, inputs, perm, v, ticks)
            elapsed = time.time() - t0
            print(f"  -- PLATEAU at {attempt+1} ({elapsed:.1f}s) --")
            print(f"  -> BOTH | Acc: {acc*100:.0f}% | Score: {current_score:.3f}")
            stale = 0

        if (attempt + 1) % 5000 == 0:
            _, acc = evaluate(net, inputs, perm, v, ticks)
            elapsed = time.time() - t0
            print(f"  [{phase:9s}] {attempt+1:6d} | "
                  f"Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                  f"Kept: {kept} | Conns: {net.count_connections()} | "
                  f"Stale: {stale} | {elapsed:.1f}s")

        if best_score > 0.99:
            elapsed = time.time() - t0
            print(f"  -> Solved at {attempt+1}! ({elapsed:.1f}s)")
            break

        if stale >= stale_limit:
            elapsed = time.time() - t0
            _, acc = evaluate(net, inputs, perm, v, ticks)
            print(f"  -> STOPPED at {attempt+1}: no improvement for "
                  f"{stale_limit} attempts ({elapsed:.1f}s)")
            print(f"     Acc: {acc*100:.0f}% | Score: {current_score:.3f}")
            break

    elapsed = time.time() - t0
    _, final_acc = evaluate(net, inputs, perm, v, ticks)
    print(f"\n  Final: {final_acc*100:.0f}% | Score: {best_score:.3f} | "
          f"Conns: {net.count_connections()} | {elapsed:.1f}s")

    prev_diff = torch.zeros(v)
    for idx in range(32):
        world = torch.zeros(v)
        world[inputs[idx]] = 1.0
        logits = net.forward(world, prev_diff, ticks=ticks)
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax().item()
        tgt = perm[idx].item()
        conf = probs[pred].item()
        mark = "ok" if pred == tgt else "X "
        print(f"    {inputs[idx]:2d} -> {pred:2d} (exp:{tgt:2d}) {mark}  conf:{conf:.3f}")
        target_vec = torch.zeros(v)
        target_vec[tgt] = 1.0
        prev_diff = target_vec - probs.detach()

    print(f"\n  Compare: v12 (layered) 32-class = 91%")
    return {"acc": final_acc, "score": best_score, "time": elapsed}


@app.local_entrypoint()
def main():
    print("Running 32-class graph network on Modal CPU (extended)...")
    results = run_experiment.remote()
    print(f"\nDone! Final acc: {results['acc']*100:.0f}%")
