"""
No-Backprop v17 — Graph + Recurrent (v15 + v16)
==================================================
N neurons in a pool. No layers. Random connections.
Each neuron has persistent STATE that survives across ticks.
Diff fed as input. Loops can form naturally.

The signal doesn't just pass through — neurons REMEMBER.
"""

import modal
import time

app = modal.App("vraxion-graph-recurrent")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(cpu=2, memory=1024, timeout=3600, image=image)
def run_experiment():
    import torch
    import math
    import random

    class GraphRecurrentNet:
        """
        N neurons, no layers.
        Each neuron has a persistent activation that carries over between ticks.
        Input neurons get [world | diff] injected.
        Output read from designated output neurons.
        Neurons remember — loops naturally form feedback circuits.
        """

        def __init__(self, n_neurons, n_in, n_out, density=0.1):
            self.N = n_neurons
            self.n_in = n_in
            self.n_out = n_out

            # Connection matrix
            s = math.sqrt(2.0 / n_neurons)
            self.W = torch.randn(n_neurons, n_neurons) * s
            self.mask = (torch.rand(n_neurons, n_neurons) < density).float()
            self.mask.fill_diagonal_(0)

            # Persistent neuron state
            self.state = torch.zeros(n_neurons)

            # Decay factor — neurons don't hold activation forever
            # This is learned too (part of W essentially)
            self.decay = 0.5

        def reset_state(self):
            self.state = torch.zeros(self.N)

        def forward(self, world, diff, ticks=5):
            """
            Inject input, let signal bounce with persistent state.
            Neurons keep their activation between ticks (with decay).
            """
            inp = torch.cat([world, diff])
            act = self.state.clone()

            Weff = self.W * self.mask
            for t in range(ticks):
                # Decay existing activation
                act = act * self.decay

                # Inject input
                act[:self.n_in] = inp

                # All neurons update: new = relu(connections @ current)
                act = torch.relu(act @ Weff + act * 0.1)  # small self-residual

                # Re-inject input (persistent stimulus)
                act[:self.n_in] = inp

            # Save state for next call
            self.state = act.detach()

            # Read output
            return act[-self.n_out:]

        def count_connections(self):
            return int(self.mask.sum().item())

        def save_state(self):
            return (self.W.clone(), self.mask.clone(), self.state.clone())

        def restore_state(self, saved):
            self.W = saved[0].clone()
            self.mask = saved[1].clone()
            self.state = saved[2].clone()

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

        def mutate_weights(self, scale=0.05):
            noise = torch.randn_like(self.W) * scale
            self.W += noise * self.mask

    def evaluate(net, inputs, targets, vocab, ticks=5, n_passes=2):
        """
        Multi-pass: first pass cold, subsequent passes get feedback.
        Score only on last pass.
        """
        total_score = 0.0
        correct = 0
        n_items = len(inputs)

        net.reset_state()
        prev_diff = torch.zeros(vocab)

        for pass_num in range(n_passes):
            for idx in range(n_items):
                world = torch.zeros(vocab)
                world[inputs[idx]] = 1.0

                logits = net.forward(world, prev_diff, ticks=ticks)
                probs = torch.softmax(logits, dim=-1)
                predicted = probs.argmax().item()
                target = targets[idx].item()

                if pass_num == n_passes - 1:
                    total_score += probs[target].item()
                    if predicted == target:
                        correct += 1

                target_vec = torch.zeros(vocab)
                target_vec[target] = 1.0
                prev_diff = target_vec - probs.detach()

        acc = correct / n_items
        avg_prob = total_score / n_items
        return 0.5 * acc + 0.5 * avg_prob, acc

    def evaluate_cold_warm(net, inputs, targets, vocab, ticks=5):
        """Compare cold (no feedback) vs warm (after feedback)."""
        # Cold pass
        net.reset_state()
        prev_diff = torch.zeros(vocab)
        cold_correct = 0
        for idx in range(len(inputs)):
            world = torch.zeros(vocab)
            world[inputs[idx]] = 1.0
            logits = net.forward(world, prev_diff, ticks=ticks)
            probs = torch.softmax(logits, dim=-1)
            if probs.argmax().item() == targets[idx].item():
                cold_correct += 1
            target_vec = torch.zeros(vocab)
            target_vec[targets[idx].item()] = 1.0
            prev_diff = target_vec - probs.detach()

        # Warm pass (state carries over from cold)
        warm_correct = 0
        for idx in range(len(inputs)):
            world = torch.zeros(vocab)
            world[inputs[idx]] = 1.0
            logits = net.forward(world, prev_diff, ticks=ticks)
            probs = torch.softmax(logits, dim=-1)
            if probs.argmax().item() == targets[idx].item():
                warm_correct += 1
            target_vec = torch.zeros(vocab)
            target_vec[targets[idx].item()] = 1.0
            prev_diff = target_vec - probs.detach()

        cold_acc = cold_correct / len(inputs) * 100
        warm_acc = warm_correct / len(inputs) * 100
        return cold_acc, warm_acc

    def run_task(name, inputs, targets, vocab, n_neurons, ticks=5,
                 density=0.08, n_passes=2, max_attempts=200000,
                 stale_limit=15000):
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  Neurons: {n_neurons} | Ticks: {ticks} | Passes: {n_passes}")
        print(f"  Density: {density} | Stale limit: {stale_limit}")
        print(f"{'='*60}")

        random.seed(42); torch.manual_seed(42)
        n_in = vocab * 2
        n_out = vocab
        net = GraphRecurrentNet(n_neurons, n_in, n_out, density=density)

        current_score, current_acc = evaluate(
            net, inputs, targets, vocab, ticks, n_passes)
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

            new_score, new_acc = evaluate(
                net, inputs, targets, vocab, ticks, n_passes)

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
                _, acc = evaluate(net, inputs, targets, vocab, ticks, n_passes)
                elapsed = time.time() - t0
                print(f"  -- PLATEAU at {attempt+1} ({elapsed:.1f}s) --")
                print(f"  -> BOTH | Acc: {acc*100:.0f}% | Score: {current_score:.3f}")
                stale = 0

            if (attempt + 1) % 5000 == 0:
                _, acc = evaluate(net, inputs, targets, vocab, ticks, n_passes)
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
                _, acc = evaluate(net, inputs, targets, vocab, ticks, n_passes)
                print(f"  -> STOPPED at {attempt+1}: stale {stale_limit} ({elapsed:.1f}s)")
                print(f"     Acc: {acc*100:.0f}% | Score: {current_score:.3f}")
                break

        elapsed = time.time() - t0
        _, final_acc = evaluate(net, inputs, targets, vocab, ticks, n_passes)

        print(f"\n  Final: {final_acc*100:.0f}% | Score: {best_score:.3f} | "
              f"Conns: {net.count_connections()} | {elapsed:.1f}s")

        # Cold vs Warm comparison
        cold_acc, warm_acc = evaluate_cold_warm(net, inputs, targets, vocab, ticks)
        print(f"  Cold: {cold_acc:.0f}% | Warm: {warm_acc:.0f}%", end="")
        if warm_acc > cold_acc:
            print(f" | LEARNED from feedback! (+{warm_acc-cold_acc:.0f}%)")
        else:
            print(f" | No feedback benefit")

        # Per-input details (warm pass)
        net.reset_state()
        prev_diff = torch.zeros(vocab)
        # Cold pass (skip output)
        for idx in range(len(inputs)):
            world = torch.zeros(vocab)
            world[inputs[idx]] = 1.0
            logits = net.forward(world, prev_diff, ticks=ticks)
            probs = torch.softmax(logits, dim=-1)
            target_vec = torch.zeros(vocab)
            target_vec[targets[idx].item()] = 1.0
            prev_diff = target_vec - probs.detach()
        # Warm pass (show output)
        print(f"\n  Warm pass details:")
        for idx in range(min(len(inputs), 16)):
            world = torch.zeros(vocab)
            world[inputs[idx]] = 1.0
            logits = net.forward(world, prev_diff, ticks=ticks)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax().item()
            tgt = targets[idx].item()
            conf = probs[pred].item()
            state_mag = net.state.abs().mean().item()
            mark = "ok" if pred == tgt else "X "
            print(f"    {inputs[idx]:2d} -> {pred:2d} (exp:{tgt:2d}) {mark}  "
                  f"conf:{conf:.3f}  state:{state_mag:.3f}")
            target_vec = torch.zeros(vocab)
            target_vec[tgt] = 1.0
            prev_diff = target_vec - probs.detach()

        return final_acc, best_score, elapsed, cold_acc, warm_acc

    results = {}

    # 8-class
    torch.manual_seed(42)
    v = 8; perm = torch.randperm(v)
    a, s, t, cold, warm = run_task(
        f"8-class: {perm.tolist()}",
        list(range(v)), perm, v,
        n_neurons=48, ticks=5, density=0.08,
        n_passes=2, max_attempts=50000, stale_limit=15000)
    results["8"] = {"acc": a, "cold": cold, "warm": warm, "time": t}

    # 16-class
    torch.manual_seed(42)
    v = 16; perm = torch.randperm(v)
    a, s, t, cold, warm = run_task(
        "16-class permutation",
        list(range(v)), perm, v,
        n_neurons=80, ticks=6, density=0.08,
        n_passes=2, max_attempts=100000, stale_limit=20000)
    results["16"] = {"acc": a, "cold": cold, "warm": warm, "time": t}

    # 32-class
    torch.manual_seed(42)
    v = 32; perm = torch.randperm(v)
    a, s, t, cold, warm = run_task(
        "32-class permutation",
        list(range(v)), perm, v,
        n_neurons=160, ticks=8, density=0.06,
        n_passes=2, max_attempts=200000, stale_limit=25000)
    results["32"] = {"acc": a, "cold": cold, "warm": warm, "time": t}

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY -- v17 Graph + Recurrent")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Final':>8} {'Cold':>8} {'Warm':>8} {'Learned?':>10}")
    print(f"  {'-'*46}")
    for v in ["8", "16", "32"]:
        r = results[v]
        learned = "YES" if r["warm"] > r["cold"] else "no"
        print(f"  {v}-class   {r['acc']*100:>6.0f}%  {r['cold']:>6.0f}%  "
              f"{r['warm']:>6.0f}%  {learned:>10}")

    print(f"\n  Compare:")
    print(f"  v12 (layered):        8->100%, 16->100%, 32->91%")
    print(f"  v15 (recurrent):      8->88%,  16->88%,  32->88%")
    print(f"  v16 (graph):          8->100%, 16->100%, 32->81%")

    return results


@app.local_entrypoint()
def main():
    print("Running graph + recurrent network on Modal CPU...")
    results = run_experiment.remote()
    print("\nDone!")
