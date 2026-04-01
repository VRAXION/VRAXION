"""
No-Backprop v16 — Graph Network (No Layers)
=============================================
N neurons in a pool. No layers. Random connections.
Signal enters, bounces around for T ticks, output read.
Diff fed back as input. Try→keep/revert.

Back to basics: just neurons and connections.
"""

import modal
import time

app = modal.App("vraxion-graph")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(cpu=2, memory=1024, timeout=1800, image=image)
def run_experiment():
    import torch
    import math
    import random

    class GraphNet:
        """
        N neurons. No layers.
        First `n_in` neurons = input (world + diff).
        Last `n_out` neurons = output.
        Everything else = internal.
        Sparse NxN connection matrix, signal bounces T ticks.
        """

        def __init__(self, n_neurons, n_in, n_out, density=0.1):
            self.N = n_neurons
            self.n_in = n_in
            self.n_out = n_out

            # Connection matrix: who sends to whom
            s = math.sqrt(2.0 / n_neurons)
            self.W = torch.randn(n_neurons, n_neurons) * s
            self.mask = (torch.rand(n_neurons, n_neurons) < density).float()

            # No self-connections
            self.mask.fill_diagonal_(0)

        def forward(self, world, diff, ticks=5):
            """Inject input, let it bounce, read output."""
            act = torch.zeros(self.N)

            # Inject: first n_in neurons get [world | diff]
            inp = torch.cat([world, diff])
            act[:self.n_in] = inp

            for t in range(ticks):
                # All neurons update simultaneously
                new_act = act @ (self.W * self.mask)
                new_act = torch.relu(new_act)

                # Re-inject input every tick (persistent stimulus)
                new_act[:self.n_in] = inp

                act = new_act

            # Read output from last n_out neurons
            return act[-self.n_out:]

        def count_connections(self):
            return int(self.mask.sum().item())

        def save_state(self):
            return (self.W.clone(), self.mask.clone())

        def restore_state(self, state):
            self.W = state[0].clone()
            self.mask = state[1].clone()

        def mutate_structure(self, rate=0.05):
            action = random.choice(["add", "remove", "rewire"])
            if action == "add":
                dead = (self.mask == 0).nonzero(as_tuple=False)
                # Exclude diagonal
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

        def mutate_weights(self, scale=0.05):
            noise = torch.randn_like(self.W) * scale
            self.W += noise * self.mask

    def evaluate(net, inputs, targets, vocab, ticks=5):
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

            # Diff for next step
            target_vec = torch.zeros(vocab)
            target_vec[target] = 1.0
            prev_diff = target_vec - probs.detach()

        n = len(inputs)
        acc = correct / n
        avg_prob = total_score / n
        return 0.5 * acc + 0.5 * avg_prob, acc

    def run_task(name, inputs, targets, vocab, n_neurons, ticks=5,
                 density=0.1, max_attempts=200000, stale_limit=10000):
        """Run until no improvement for stale_limit attempts."""
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  Neurons: {n_neurons} | Input: {vocab*2} | Output: {vocab}")
        print(f"  Ticks: {ticks} | Density: {density}")
        print(f"  Stop after {stale_limit} attempts with no improvement")
        print(f"{'='*60}")

        random.seed(42); torch.manual_seed(42)
        n_in = vocab * 2  # world + diff
        n_out = vocab
        net = GraphNet(n_neurons, n_in, n_out, density=density)

        current_score, current_acc = evaluate(net, inputs, targets, vocab, ticks)
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

            new_score, new_acc = evaluate(net, inputs, targets, vocab, ticks)

            if new_score > current_score:
                current_score = new_score
                kept += 1
                stale = 0
                if current_score > best_score:
                    best_score = current_score
            else:
                net.restore_state(state)
                stale += 1

            # Phase switch on plateau
            if phase == "STRUCTURE" and stale > 3000 and not phase_switched:
                phase = "BOTH"
                phase_switched = True
                _, acc = evaluate(net, inputs, targets, vocab, ticks)
                elapsed = time.time() - t0
                print(f"  -- PLATEAU at {attempt+1} ({elapsed:.1f}s) --")
                print(f"  -> BOTH | Acc: {acc*100:.0f}% | Score: {current_score:.3f}")
                stale = 0

            if (attempt + 1) % 5000 == 0:
                _, acc = evaluate(net, inputs, targets, vocab, ticks)
                elapsed = time.time() - t0
                print(f"  [{phase:9s}] {attempt+1:6d} | "
                      f"Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                      f"Kept: {kept} | Conns: {net.count_connections()} | "
                      f"Stale: {stale} | {elapsed:.1f}s")

            if best_score > 0.99:
                elapsed = time.time() - t0
                print(f"  -> Solved at {attempt+1}! ({elapsed:.1f}s)")
                break

            # STOP if nothing moves
            if stale >= stale_limit:
                elapsed = time.time() - t0
                _, acc = evaluate(net, inputs, targets, vocab, ticks)
                print(f"  -> STOPPED at {attempt+1}: no improvement for "
                      f"{stale_limit} attempts ({elapsed:.1f}s)")
                print(f"     Acc: {acc*100:.0f}% | Score: {current_score:.3f}")
                break

        elapsed = time.time() - t0
        _, final_acc = evaluate(net, inputs, targets, vocab, ticks)

        print(f"\n  Final: {final_acc*100:.0f}% | Score: {best_score:.3f} | "
              f"Conns: {net.count_connections()} | Kept: {kept} | {elapsed:.1f}s")

        # Show per-input results
        prev_diff = torch.zeros(vocab)
        for idx in range(min(len(inputs), 16)):
            world = torch.zeros(vocab)
            world[inputs[idx]] = 1.0
            logits = net.forward(world, prev_diff, ticks=ticks)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax().item()
            tgt = targets[idx].item()
            conf = probs[pred].item()
            mark = "ok" if pred == tgt else "X "
            print(f"    {inputs[idx]:2d} -> {pred:2d} (exp:{tgt:2d}) {mark}  "
                  f"conf:{conf:.3f}")
            target_vec = torch.zeros(vocab)
            target_vec[tgt] = 1.0
            prev_diff = target_vec - probs.detach()

        return final_acc, best_score, elapsed

    results = {}

    # 8-class
    torch.manual_seed(42)
    v = 8; perm = torch.randperm(v)
    a, s, t = run_task(
        f"8-class: {perm.tolist()}",
        list(range(v)), perm, v,
        n_neurons=48, ticks=5, density=0.15,
        max_attempts=100000, stale_limit=15000)
    results["8"] = (a, s, t)

    # 16-class
    torch.manual_seed(42)
    v = 16; perm = torch.randperm(v)
    a, s, t = run_task(
        "16-class permutation",
        list(range(v)), perm, v,
        n_neurons=96, ticks=6, density=0.12,
        max_attempts=200000, stale_limit=20000)
    results["16"] = (a, s, t)

    # 32-class
    torch.manual_seed(42)
    v = 32; perm = torch.randperm(v)
    a, s, t = run_task(
        "32-class permutation",
        list(range(v)), perm, v,
        n_neurons=192, ticks=8, density=0.08,
        max_attempts=300000, stale_limit=25000)
    results["32"] = (a, s, t)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY -- Graph Network (No Layers)")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Acc':>8} {'Score':>8} {'Time':>8}")
    print(f"  {'-'*36}")
    prev = {"8": 100, "16": 100, "32": 91}
    for v in ["8", "16", "32"]:
        a, s, t = results[v]
        print(f"  {v}-class   {a*100:>6.0f}%  {s:>6.3f}  {t:>6.1f}s")
    print(f"\n  Compare v12 (layered): 8->100%, 16->100%, 32->91%")

    return results


@app.local_entrypoint()
def main():
    print("Running graph network (no layers) on Modal CPU...")
    results = run_experiment.remote()
    print("\nDone!")
