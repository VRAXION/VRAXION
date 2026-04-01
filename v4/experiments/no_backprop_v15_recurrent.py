"""
No-Backprop v15 — Recurrent Network with Embodied Feedback
=============================================================
The network has INTERNAL STATE that persists across steps.
The diff (pain/dopamine) goes in as input and the network's
internal state changes — it doesn't just see the diff, it
BECOMES different because of it.

Like the brain: the pain doesn't just inform you, it CHANGES you.

Single network, try→keep/revert, structure→both phases.
"""

import modal
import time

app = modal.App("vraxion-recurrent")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(cpu=2, memory=1024, timeout=900, image=image)
def run_experiment():
    import torch
    import torch.nn.functional as F
    import math
    import random

    class RecurrentBrainNet:
        """
        Recurrent sparse network with internal state.
        Input: [world | prev_diff | internal_state]
        Output: [response | new_internal_state]

        The network reads its own state and writes a new one.
        The diff modifies the state, not just the output.
        """

        def __init__(self, vocab, hidden, state_size, density=0.5):
            self.vocab = vocab
            self.hidden = hidden
            self.state_size = state_size

            # Input: world(vocab) + diff(vocab) + state(state_size)
            # Output: response(vocab) + new_state(state_size)
            input_size = vocab + vocab + state_size
            output_size = vocab + state_size

            self.layer_sizes = [input_size, hidden, hidden, output_size]
            self.n_layers = 3

            self.W = []
            self.mask = []
            for i in range(self.n_layers):
                fi, fo = self.layer_sizes[i], self.layer_sizes[i + 1]
                s = math.sqrt(2.0 / (fi + fo))
                self.W.append(torch.randn(fi, fo) * s)
                self.mask.append((torch.rand(fi, fo) < density).float())

            # Internal state
            self.state = torch.zeros(state_size)

        def reset_state(self):
            self.state = torch.zeros(self.state_size)

        def forward(self, world_input, diff_input):
            """
            One step:
            1. Concat [world | diff | state] as input
            2. Forward through sparse layers
            3. Split output into [response | new_state]
            4. Update internal state
            """
            full_input = torch.cat([world_input, diff_input, self.state])

            h = full_input
            for i in range(self.n_layers):
                h = h @ (self.W[i] * self.mask[i])
                if i < self.n_layers - 1:
                    h = torch.relu(h)

            # Split output
            response = h[:self.vocab]
            new_state = torch.tanh(h[self.vocab:])  # bounded state

            # Update internal state
            self.state = new_state.detach()

            return response

        def count_connections(self):
            return sum(int(m.sum().item()) for m in self.mask)

        def save_full_state(self):
            return (
                [w.clone() for w in self.W],
                [m.clone() for m in self.mask],
                self.state.clone()
            )

        def restore_full_state(self, saved):
            self.W = [w.clone() for w in saved[0]]
            self.mask = [m.clone() for m in saved[1]]
            self.state = saved[2].clone()

        def mutate_structure(self, rate=0.05):
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
            i = random.randint(0, self.n_layers - 1)
            noise = torch.randn_like(self.W[i]) * scale
            self.W[i] += noise * self.mask[i]

    def evaluate_recurrent(net, inputs, targets, vocab, n_passes=2):
        """
        Run through all inputs MULTIPLE times.
        First pass: no diff (cold start).
        Second pass: diff from first pass feeds back.
        The network should get BETTER on the second pass
        because it has seen the feedback.
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

                logits = net.forward(world, prev_diff)
                probs = torch.softmax(logits, dim=-1)
                predicted = probs.argmax().item()
                target = targets[idx].item()

                # Only score the LAST pass (after seeing feedback)
                if pass_num == n_passes - 1:
                    total_score += probs[target].item()
                    if predicted == target:
                        correct += 1

                # Compute diff for next step
                target_vec = torch.zeros(vocab)
                target_vec[target] = 1.0
                prev_diff = target_vec - probs.detach()

        acc = correct / n_items
        avg_prob = total_score / n_items
        return 0.5 * acc + 0.5 * avg_prob, acc

    def run_task(name, inputs, targets, vocab, hidden, state_size,
                 attempts=30000, patience=2000, n_passes=2):
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  Input: {vocab}(world) + {vocab}(diff) + {state_size}(state)")
        print(f"  Output: {vocab}(response) + {state_size}(new_state)")
        print(f"  Passes per eval: {n_passes}")
        print(f"{'='*60}")

        random.seed(42); torch.manual_seed(42)
        net = RecurrentBrainNet(vocab, hidden, state_size, density=0.5)

        current_score, current_acc = evaluate_recurrent(
            net, inputs, targets, vocab, n_passes)
        best_score = current_score
        phase = "STRUCTURE"
        kept = 0
        plateau_counter = 0
        phase_switched = False
        t0 = time.time()

        print(f"  Start: Acc={current_acc*100:.0f}% | Score: {current_score:.3f} | "
              f"Conns: {net.count_connections()}")

        for attempt in range(attempts):
            state = net.save_full_state()

            if phase == "STRUCTURE":
                net.mutate_structure(rate=0.05)
            else:
                if random.random() < 0.3:
                    net.mutate_structure(rate=0.02)
                else:
                    net.mutate_weights(scale=0.05)

            new_score, new_acc = evaluate_recurrent(
                net, inputs, targets, vocab, n_passes)

            if new_score > current_score:
                current_score = new_score
                kept += 1
                plateau_counter = 0
                if current_score > best_score:
                    best_score = current_score
            else:
                net.restore_full_state(state)
                plateau_counter += 1

            if phase == "STRUCTURE" and plateau_counter > patience and not phase_switched:
                phase = "BOTH"
                phase_switched = True
                _, acc = evaluate_recurrent(net, inputs, targets, vocab, n_passes)
                elapsed = time.time() - t0
                print(f"  ── PLATEAU at {attempt+1} ({elapsed:.1f}s) ──")
                print(f"  → BOTH | Acc: {acc*100:.0f}% | Score: {current_score:.3f}")
                plateau_counter = 0

            if (attempt + 1) % 2000 == 0:
                _, acc = evaluate_recurrent(net, inputs, targets, vocab, n_passes)
                elapsed = time.time() - t0
                print(f"  [{phase:9s}] {attempt+1:5d} | "
                      f"Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                      f"Kept: {kept} | Conns: {net.count_connections()} | "
                      f"{elapsed:.1f}s")

            if best_score > 0.99:
                elapsed = time.time() - t0
                print(f"  → Solved at {attempt+1}! ({elapsed:.1f}s)")
                break

        elapsed = time.time() - t0
        _, final_acc = evaluate_recurrent(net, inputs, targets, vocab, n_passes)

        print(f"\n  Final: {final_acc*100:.0f}% | Score: {best_score:.3f} | "
              f"Conns: {net.count_connections()} | {elapsed:.1f}s")

        # Show per-input results (last pass)
        net.reset_state()
        prev_diff = torch.zeros(vocab)
        # First pass (cold)
        for idx in range(len(inputs)):
            world = torch.zeros(vocab)
            world[inputs[idx]] = 1.0
            logits = net.forward(world, prev_diff)
            probs = torch.softmax(logits, dim=-1)
            target_vec = torch.zeros(vocab)
            target_vec[targets[idx].item()] = 1.0
            prev_diff = target_vec - probs.detach()

        # Second pass (warm — has seen feedback)
        print(f"\n  Pass 2 (after feedback):")
        for idx in range(min(len(inputs), 16)):
            world = torch.zeros(vocab)
            world[inputs[idx]] = 1.0
            logits = net.forward(world, prev_diff)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax().item()
            tgt = targets[idx].item()
            conf = probs[pred].item()
            state_mag = net.state.abs().mean().item()
            mark = "✓" if pred == tgt else "✗"
            print(f"    {inputs[idx]:2d} → {pred:2d} (exp:{tgt:2d}) {mark}  "
                  f"conf:{conf:.3f}  state_mag:{state_mag:.3f}")
            target_vec = torch.zeros(vocab)
            target_vec[tgt] = 1.0
            prev_diff = target_vec - probs.detach()

        # Compare: pass 1 (cold) vs pass 2 (warm)
        net.reset_state()
        prev_diff = torch.zeros(vocab)
        cold_correct = 0
        for idx in range(len(inputs)):
            world = torch.zeros(vocab)
            world[inputs[idx]] = 1.0
            logits = net.forward(world, prev_diff)
            probs = torch.softmax(logits, dim=-1)
            if probs.argmax().item() == targets[idx].item():
                cold_correct += 1
            target_vec = torch.zeros(vocab)
            target_vec[targets[idx].item()] = 1.0
            prev_diff = target_vec - probs.detach()

        warm_correct = 0
        for idx in range(len(inputs)):
            world = torch.zeros(vocab)
            world[inputs[idx]] = 1.0
            logits = net.forward(world, prev_diff)
            probs = torch.softmax(logits, dim=-1)
            if probs.argmax().item() == targets[idx].item():
                warm_correct += 1
            target_vec = torch.zeros(vocab)
            target_vec[targets[idx].item()] = 1.0
            prev_diff = target_vec - probs.detach()

        cold_acc = cold_correct / len(inputs) * 100
        warm_acc = warm_correct / len(inputs) * 100
        print(f"\n  Cold (no feedback): {cold_acc:.0f}%")
        print(f"  Warm (after feedback): {warm_acc:.0f}%")
        if warm_acc > cold_acc:
            print(f"  → Network LEARNED from feedback! (+{warm_acc-cold_acc:.0f}%)")
        else:
            print(f"  → Feedback didn't help (or not needed)")

        return final_acc, best_score, elapsed, cold_acc, warm_acc

    results = {}

    # ═══════ 8-class ═══════
    torch.manual_seed(42)
    v = 8; perm = torch.randperm(v)
    a, s, t, cold, warm = run_task(
        f"8-class: {perm.tolist()}",
        list(range(v)), perm, v,
        hidden=32, state_size=16,
        attempts=15000, n_passes=2)
    results["8"] = {"acc": a, "cold": cold, "warm": warm, "time": t}

    # ═══════ 16-class ═══════
    torch.manual_seed(42)
    v = 16; perm = torch.randperm(v)
    a, s, t, cold, warm = run_task(
        "16-class permutation",
        list(range(v)), perm, v,
        hidden=64, state_size=32,
        attempts=30000, n_passes=2)
    results["16"] = {"acc": a, "cold": cold, "warm": warm, "time": t}

    # ═══════ 32-class ═══════
    torch.manual_seed(42)
    v = 32; perm = torch.randperm(v)
    a, s, t, cold, warm = run_task(
        "32-class permutation",
        list(range(v)), perm, v,
        hidden=128, state_size=64,
        attempts=60000, patience=3000, n_passes=3)
    results["32"] = {"acc": a, "cold": cold, "warm": warm, "time": t}

    # ═══════ Summary ═══════
    print(f"\n{'='*60}")
    print(f"  SUMMARY — Recurrent + Embodied Feedback")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Final':>8} {'Cold':>8} {'Warm':>8} {'Learned?':>10}")
    print(f"  {'-'*46}")
    for v in ["8", "16", "32"]:
        r = results[v]
        learned = "YES ✓" if r["warm"] > r["cold"] else "no"
        print(f"  {v}-class   {r['acc']*100:>6.0f}%  {r['cold']:>6.0f}%  "
              f"{r['warm']:>6.0f}%  {learned:>10}")

    return results


@app.local_entrypoint()
def main():
    print("Running recurrent embodied learning on Modal CPU...")
    results = run_experiment.remote()
    print("\nDone!")
