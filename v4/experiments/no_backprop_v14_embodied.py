"""
No-Backprop v14 — Embodied Learning
======================================
The diff (pain/dopamine) is just another INPUT.
The network doesn't know what's "world" and what's "feedback".
It has to LEARN to use the feedback signal.

Input at t: [world_input | prev_diff]
  - world_input: one-hot of current value
  - prev_diff: (target - output) from previous step, or zeros if first step

Single network, try→keep/revert self-modification.
Structure first, then structure+weights.
"""

import modal
import time

app = modal.App("vraxion-embodied")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(cpu=2, memory=1024, timeout=600, image=image)
def run_experiment():
    import torch
    import torch.nn.functional as F
    import math
    import random

    class BrainNet:
        def __init__(self, layer_sizes, density=0.5):
            self.layer_sizes = layer_sizes
            self.n_layers = len(layer_sizes) - 1
            self.W = []
            self.mask = []
            for i in range(self.n_layers):
                fi, fo = layer_sizes[i], layer_sizes[i + 1]
                s = math.sqrt(2.0 / (fi + fo))
                self.W.append(torch.randn(fi, fo) * s)
                self.mask.append((torch.rand(fi, fo) < density).float())

        def forward(self, x):
            h = x
            for i in range(self.n_layers):
                h = h @ (self.W[i] * self.mask[i])
                if i < self.n_layers - 1:
                    h = torch.relu(h)
            return h

        def count_connections(self):
            return sum(int(m.sum().item()) for m in self.mask)

        def save_state(self):
            return ([w.clone() for w in self.W], [m.clone() for m in self.mask])

        def restore_state(self, state):
            self.W = [w.clone() for w in state[0]]
            self.mask = [m.clone() for m in state[1]]

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

    def evaluate_embodied(net, inputs, targets, vocab):
        """
        Evaluate with embodied feedback loop.
        Each input gets [world | prev_diff].
        The network sees the diff from its PREVIOUS mistake.
        """
        total_score = 0.0
        correct = 0
        prev_diff = torch.zeros(vocab)  # no feedback on first step

        for idx in range(len(inputs)):
            # Build input: [world | prev_diff]
            world = torch.zeros(vocab)
            world[inputs[idx]] = 1.0
            full_input = torch.cat([world, prev_diff])

            # Forward
            logits = net.forward(full_input)
            probs = torch.softmax(logits, dim=-1)
            predicted = probs.argmax().item()
            target = targets[idx].item()

            # Score
            total_score += probs[target].item()
            if predicted == target:
                correct += 1

            # Compute diff for NEXT step
            target_vec = torch.zeros(vocab)
            target_vec[target] = 1.0
            prev_diff = target_vec - probs.detach()

        n = len(inputs)
        acc = correct / n
        avg_prob = total_score / n
        return 0.5 * acc + 0.5 * avg_prob, acc

    def run_task(name, inputs, targets, vocab, layer_sizes,
                 attempts=30000, patience=2000):
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  Input size: {vocab}(world) + {vocab}(diff) = {vocab*2}")
        print(f"{'='*60}")

        random.seed(42); torch.manual_seed(42)
        net = BrainNet(layer_sizes, density=0.5)

        current_score, current_acc = evaluate_embodied(
            net, inputs, targets, vocab)
        best_score = current_score
        phase = "STRUCTURE"
        kept = 0
        plateau_counter = 0
        phase_switched = False
        t0 = time.time()

        print(f"  Start: Acc={current_acc*100:.0f}% | Score: {current_score:.3f} | "
              f"Conns: {net.count_connections()}")

        for attempt in range(attempts):
            state = net.save_state()

            if phase == "STRUCTURE":
                net.mutate_structure(rate=0.05)
            else:
                if random.random() < 0.3:
                    net.mutate_structure(rate=0.02)
                else:
                    net.mutate_weights(scale=0.05)

            new_score, new_acc = evaluate_embodied(
                net, inputs, targets, vocab)

            if new_score > current_score:
                current_score = new_score
                kept += 1
                plateau_counter = 0
                if current_score > best_score:
                    best_score = current_score
            else:
                net.restore_state(state)
                plateau_counter += 1

            if phase == "STRUCTURE" and plateau_counter > patience and not phase_switched:
                phase = "BOTH"
                phase_switched = True
                _, acc = evaluate_embodied(net, inputs, targets, vocab)
                elapsed = time.time() - t0
                print(f"  ── PLATEAU at attempt {attempt+1} ({elapsed:.1f}s) ──")
                print(f"  → STRUCTURE + WEIGHTS | Acc: {acc*100:.0f}% | "
                      f"Score: {current_score:.3f}")
                plateau_counter = 0

            if (attempt + 1) % 2000 == 0:
                _, acc = evaluate_embodied(net, inputs, targets, vocab)
                elapsed = time.time() - t0
                print(f"  [{phase:9s}] {attempt+1:5d} | "
                      f"Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                      f"Kept: {kept} | Conns: {net.count_connections()} | "
                      f"{elapsed:.1f}s")

            if best_score > 0.99:
                elapsed = time.time() - t0
                print(f"  → Solved at attempt {attempt+1}! ({elapsed:.1f}s)")
                break

        elapsed = time.time() - t0
        _, final_acc = evaluate_embodied(net, inputs, targets, vocab)

        print(f"\n  Final: {final_acc*100:.0f}% | Score: {best_score:.3f} | "
              f"Conns: {net.count_connections()} | {elapsed:.1f}s")

        # Show per-input results
        prev_diff = torch.zeros(vocab)
        for idx in range(min(len(inputs), 16)):
            world = torch.zeros(vocab)
            world[inputs[idx]] = 1.0
            full_input = torch.cat([world, prev_diff])
            logits = net.forward(full_input)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax().item()
            tgt = targets[idx].item()
            conf = probs[pred].item()
            diff_mag = prev_diff.abs().sum().item()
            mark = "✓" if pred == tgt else "✗"
            print(f"    {inputs[idx]:2d} → {pred:2d} (exp:{tgt:2d}) {mark}  "
                  f"conf:{conf:.3f}  diff_mag:{diff_mag:.2f}")
            target_vec = torch.zeros(vocab)
            target_vec[tgt] = 1.0
            prev_diff = target_vec - probs.detach()

        return final_acc, best_score, elapsed

    results = {}

    # ═══════ 8-class ═══════
    torch.manual_seed(42)
    v = 8
    perm = torch.randperm(v)
    inputs = list(range(v))
    # Input = vocab*2 (world + diff), hidden, output = vocab
    a, s, t = run_task(
        f"8-class: {perm.tolist()}",
        inputs, perm, v,
        [v*2, 32, 32, v],
        attempts=15000)
    results["8"] = (a, s, t)

    # ═══════ 16-class ═══════
    torch.manual_seed(42)
    v = 16
    perm = torch.randperm(v)
    inputs = list(range(v))
    a, s, t = run_task(
        "16-class permutation",
        inputs, perm, v,
        [v*2, 64, 64, v],
        attempts=30000)
    results["16"] = (a, s, t)

    # ═══════ 32-class ═══════
    torch.manual_seed(42)
    v = 32
    perm = torch.randperm(v)
    inputs = list(range(v))
    a, s, t = run_task(
        "32-class permutation",
        inputs, perm, v,
        [v*2, 128, 128, v],
        attempts=60000, patience=3000)
    results["32"] = (a, s, t)

    # ═══════ Compare: same task WITHOUT diff (blind) ═══════
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Same network WITHOUT diff input")
    print(f"{'='*60}")

    def evaluate_blind(net, inputs, targets, vocab):
        total_score = 0.0
        correct = 0
        for idx in range(len(inputs)):
            world = torch.zeros(vocab)
            world[inputs[idx]] = 1.0
            # Pad with zeros where diff would be
            full_input = torch.cat([world, torch.zeros(vocab)])
            logits = net.forward(full_input)
            probs = torch.softmax(logits, dim=-1)
            total_score += probs[targets[idx].item()].item()
            if probs.argmax().item() == targets[idx].item():
                correct += 1
        n = len(inputs)
        return 0.5 * correct/n + 0.5 * total_score/n, correct/n

    # Quick 8-class blind test
    torch.manual_seed(42); random.seed(42)
    v = 8; perm = torch.randperm(v)
    net_blind = BrainNet([v*2, 32, 32, v], density=0.5)
    score, acc = evaluate_blind(net_blind, list(range(v)), perm, v)
    best = score
    for attempt in range(15000):
        state = net_blind.save_state()
        if random.random() < 0.3:
            net_blind.mutate_structure(0.05)
        else:
            net_blind.mutate_weights(0.05)
        new_score, _ = evaluate_blind(net_blind, list(range(v)), perm, v)
        if new_score > best:
            best = new_score
        else:
            net_blind.restore_state(state)
    _, blind_acc = evaluate_blind(net_blind, list(range(v)), perm, v)
    print(f"  8-class WITHOUT diff: {blind_acc*100:.0f}%")

    # ═══════ Summary ═══════
    print(f"\n{'='*60}")
    print(f"  SUMMARY — Embodied Learning (diff as input)")
    print(f"{'='*60}")
    print(f"  {'Task':<12} {'Embodied':>10} {'v12 blind':>10}")
    print(f"  {'-'*34}")
    v12 = {"8": 100, "16": 100, "32": 91}
    for v in ["8", "16", "32"]:
        a, s, t = results[v]
        print(f"  {v}-class    {a*100:>8.0f}%  {v12[v]:>8}%")

    return results


@app.local_entrypoint()
def main():
    print("Running embodied learning on Modal CPU...")
    results = run_experiment.remote()
    print("\nDone!")
