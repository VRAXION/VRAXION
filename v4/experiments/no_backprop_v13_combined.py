"""
No-Backprop v13 — Single Network + Dopamine/Pain guided mutation
==================================================================
Combine v7 (dopamine/pain) + v12 (single-network self-modification):

Instead of blind "is score better?", the TWO signals GUIDE what to change:
  - Dopamine (good): keep change + reinforce nearby connections
  - Pain (bad): revert + but specifically attack the DOMINANT wrong output
    (targeted structural pruning, not random)

One network. No population. Guided self-modification.
"""

import torch
import torch.nn.functional as F
import math
import random
import time


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
        W_saved, mask_saved = state
        self.W = [w.clone() for w in W_saved]
        self.mask = [m.clone() for m in mask_saved]

    def random_structural_change(self, rate=0.05):
        """Random structural mutation."""
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

    def random_weight_change(self, scale=0.05):
        """Random weight perturbation."""
        i = random.randint(0, self.n_layers - 1)
        noise = torch.randn_like(self.W[i]) * scale
        self.W[i] += noise * self.mask[i]

    def pain_response(self, predicted, target):
        """
        PAIN: targeted response — attack the dominant WRONG output.
        Don't just randomly mutate. Specifically weaken the wrong answer.
        """
        # Weaken connections TO the wrong output in the last layer
        i = self.n_layers - 1
        fi = self.layer_sizes[i]

        # Remove some connections going to the wrong output
        col = predicted
        alive_to_wrong = (self.mask[i][:, col] == 1).nonzero(as_tuple=False)
        if len(alive_to_wrong) > 2:
            n_cut = max(1, len(alive_to_wrong) // 5)
            idx = alive_to_wrong[torch.randperm(len(alive_to_wrong))[:n_cut]]
            self.mask[i][idx.squeeze(-1), col] = 0

        # Weaken weights to wrong output
        self.W[i][:, col] *= 0.9

        # Add some connections TO the correct output
        dead_to_correct = (self.mask[i][:, target] == 0).nonzero(as_tuple=False)
        if len(dead_to_correct) > 0:
            n_add = max(1, len(dead_to_correct) // 5)
            idx = dead_to_correct[torch.randperm(len(dead_to_correct))[:n_add]]
            for j in range(len(idx)):
                r = idx[j][0].item()
                self.mask[i][r, target] = 1
                self.W[i][r, target] = abs(random.gauss(0, 0.1))

    def dopamine_response(self, predicted):
        """
        DOPAMINE: reinforce connections TO the correct output.
        Slightly strengthen what's working.
        """
        i = self.n_layers - 1
        # Strengthen weights to correct output
        self.W[i][:, predicted] *= 1.05
        # Clamp to prevent explosion
        self.W[i] = self.W[i].clamp(-5, 5)


def evaluate(net, X, targets):
    logits = net.forward(X)
    probs = torch.softmax(logits, dim=-1)
    correct = (probs.argmax(-1) == targets).float().mean().item()
    target_probs = probs[range(len(targets)), targets].mean().item()
    return 0.5 * correct + 0.5 * target_probs


def evaluate_detailed(net, X, targets, vocab):
    """Returns per-input details for targeted pain/dopamine."""
    logits = net.forward(X)
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    details = []
    for i in range(vocab):
        details.append({
            "input": i,
            "predicted": preds[i].item(),
            "target": targets[i].item(),
            "correct": preds[i].item() == targets[i].item(),
            "confidence": probs[i, preds[i]].item(),
            "target_prob": probs[i, targets[i]].item(),
        })
    return details


def run_task(name, X, targets, layer_sizes, vocab,
             attempts=30000, patience=2000):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    random.seed(42); torch.manual_seed(42)
    net = BrainNet(layer_sizes, density=0.5)

    current_score = evaluate(net, X, targets)
    best_score = current_score
    phase = "STRUCTURE"
    kept = 0
    plateau_counter = 0
    phase_switched = False
    t0 = time.time()

    logits = net.forward(X)
    acc = (logits.argmax(-1) == targets).float().mean().item() * 100
    print(f"  Start: {acc:.0f}% | Score: {current_score:.3f} | "
          f"Conns: {net.count_connections()}")

    for attempt in range(attempts):
        state = net.save_state()

        # ── Random mutation (structure or both) ──
        if phase == "STRUCTURE":
            net.random_structural_change(rate=0.05)
        else:
            if random.random() < 0.3:
                net.random_structural_change(rate=0.02)
            else:
                net.random_weight_change(scale=0.05)

        new_score = evaluate(net, X, targets)

        if new_score > current_score:
            # ── DOPAMINE: it worked! ──
            current_score = new_score
            kept += 1
            plateau_counter = 0

            # Reinforce: find which inputs are now correct
            details = evaluate_detailed(net, X, targets, vocab)
            for d in details:
                if d["correct"]:
                    net.dopamine_response(d["predicted"])

            # Re-evaluate after dopamine (might have changed score)
            current_score = evaluate(net, X, targets)

            if current_score > best_score:
                best_score = current_score

        else:
            # ── PAIN: revert, but also attack worst mistakes ──
            net.restore_state(state)
            plateau_counter += 1

            # Targeted pain: find the WORST wrong predictions
            details = evaluate_detailed(net, X, targets, vocab)
            wrong = [d for d in details if not d["correct"]]
            if wrong and random.random() < 0.3:
                # Attack the most confident wrong prediction
                worst = max(wrong, key=lambda d: d["confidence"])
                net.pain_response(worst["predicted"], worst["target"])
                # Re-evaluate
                current_score = evaluate(net, X, targets)

        # Phase switch
        if phase == "STRUCTURE" and plateau_counter > patience and not phase_switched:
            phase = "BOTH"
            phase_switched = True
            elapsed = time.time() - t0
            logits = net.forward(X)
            acc = (logits.argmax(-1) == targets).float().mean().item() * 100
            print(f"  ── PLATEAU at attempt {attempt+1} ({elapsed:.1f}s) ──")
            print(f"  Switching to STRUCTURE + WEIGHTS")
            print(f"  Acc: {acc:.0f}% | Score: {current_score:.3f}")
            plateau_counter = 0

        if (attempt + 1) % 2000 == 0:
            logits = net.forward(X)
            acc = (logits.argmax(-1) == targets).float().mean().item() * 100
            elapsed = time.time() - t0
            print(f"  [{phase:9s}] Attempt {attempt+1:5d} | "
                  f"Acc: {acc:.0f}% | Score: {current_score:.3f} | "
                  f"Kept: {kept} | Conns: {net.count_connections()} | "
                  f"{elapsed:.1f}s")

        if best_score > 0.99:
            elapsed = time.time() - t0
            print(f"  → Solved at attempt {attempt+1}! ({elapsed:.1f}s)")
            break

    elapsed = time.time() - t0
    logits = net.forward(X)
    acc = (logits.argmax(-1) == targets).float().mean().item() * 100
    probs = torch.softmax(logits, dim=-1)
    conns = net.count_connections()
    max_c = sum(m.numel() for m in net.mask)

    print(f"\n  Final: {acc:.0f}% | Score: {best_score:.3f} | "
          f"Conns: {conns}/{max_c} ({conns/max_c*100:.0f}%) | {elapsed:.1f}s")

    # Show mappings
    for val in range(min(vocab, 16)):
        pred = logits[val].argmax().item()
        tgt = targets[val].item()
        conf = probs[val, pred].item()
        mark = "✓" if pred == tgt else "✗"
        print(f"    {val:2d} → {pred:2d} (exp:{tgt:2d}) {mark}  conf: {conf:.3f}")

    return acc, best_score, elapsed


def run():
    results = {}

    # 8-class
    torch.manual_seed(42)
    v = 8; perm = torch.randperm(v); X = torch.eye(v)
    print(f"Method: single network + dopamine/pain guided mutation\n")
    a, s, t = run_task(f"8-class: {perm.tolist()}", X, perm,
                       [v, 16, 16, v], v, attempts=15000)
    results["8"] = (a, s, t)

    # 16-class
    torch.manual_seed(42)
    v = 16; perm = torch.randperm(v); X = torch.eye(v)
    a, s, t = run_task("16-class permutation", X, perm,
                       [v, 32, 32, v], v, attempts=25000)
    results["16"] = (a, s, t)

    # 32-class
    torch.manual_seed(42)
    v = 32; perm = torch.randperm(v); X = torch.eye(v)
    a, s, t = run_task("32-class permutation", X, perm,
                       [v, 64, 64, v], v, attempts=60000, patience=3000)
    results["32"] = (a, s, t)

    # Comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'v13 (d+p)':>12} {'v12 (blind)':>12} {'Backprop':>10}")
    print(f"  {'-'*46}")
    v12 = {"8": 100, "16": 100, "32": 91}
    for v in ["8", "16", "32"]:
        a, s, t = results[v]
        print(f"  {v}-class   {a:>10.0f}%  {v12[v]:>10}%     100%")


if __name__ == "__main__":
    run()
