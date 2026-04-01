"""
No-Backprop v22 — Accuracy-Modulated Self-Wiring (Arousal)
===========================================================
v20e baseline + accuracy controls self-wiring intensity.

Insight: the diff already DIRECTS self-wiring (high diff → strong
activation → enters top-k → proposes connections). But the INTENSITY
is fixed. Like noradrenaline in the brain:
  - Low accuracy → high arousal → aggressive self-wiring (top_k=8, max_new=6)
  - High accuracy → low arousal → stabilize (top_k=2, max_new=1)

The accuracy is passed into forward() so self-wiring adapts.
All within keep/revert — safe, no v13-style destabilization.

64-class test.
"""

import modal
import time
import sys

app = modal.App("vraxion-v22-arousal")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(cpu=2, memory=1024, timeout=3600, image=image)
def run_experiment():
    import torch
    import torch.nn.functional as F
    import math
    import random

    class ArousalSelfWiringNet:
        """v20e + accuracy-modulated self-wiring intensity."""

        def __init__(self, n_neurons, n_in, n_out, addr_dim=4, density=0.06):
            self.N = n_neurons
            self.n_in = n_in
            self.n_out = n_out
            self.addr_dim = addr_dim

            s = math.sqrt(2.0 / n_neurons)
            self.W = torch.randn(n_neurons, n_neurons) * s
            self.mask = (torch.rand(n_neurons, n_neurons) < density).float()
            self.mask.fill_diagonal_(0)

            self.addresses = torch.randn(n_neurons, addr_dim)
            self.target_W = torch.randn(n_neurons, addr_dim) * 0.1

            self.state = torch.zeros(n_neurons)
            self.decay = 0.5

        def reset_state(self):
            self.state = torch.zeros(self.N)

        def forward(self, world, diff, ticks=8, accuracy=0.0):
            inp = torch.cat([world, diff])
            act = self.state.clone()
            Weff = self.W * self.mask

            for t in range(ticks):
                act = act * self.decay
                act[:self.n_in] = inp
                act = F.leaky_relu(act @ Weff + act * 0.1)
                act[:self.n_in] = inp

            self.state = act.detach()
            self._self_wire(act, accuracy)
            return act[-self.n_out:]

        def _self_wire(self, activations, accuracy):
            """Arousal-modulated self-wiring: worse performance = more aggressive."""
            if accuracy > 0.9:
                top_k, max_new = 2, 1      # jól megy, stabilizálj
            elif accuracy > 0.5:
                top_k, max_new = 4, 3      # közepes, keress aktívan
            else:
                top_k, max_new = 8, 6      # rossz, keress agresszívan

            internal_start = self.n_in
            act = activations[internal_start:].detach()

            if act.sum() < 0.01:
                return

            n_candidates = min(top_k, len(act))
            _, top_idx = act.topk(n_candidates)
            top_idx = top_idx + internal_start

            new_connections = 0
            for neuron_idx in top_idx:
                ni = neuron_idx.item()
                if activations[ni] < 0.1:
                    continue

                target = self.addresses[ni] + activations[ni] * self.target_W[ni]
                dists = ((self.addresses - target.unsqueeze(0)) ** 2).sum(dim=1)
                dists[ni] = float('inf')
                nearest = dists.argmin().item()

                if self.mask[ni, nearest] == 0:
                    self.mask[ni, nearest] = 1
                    self.W[ni, nearest] = random.gauss(0, math.sqrt(2.0 / self.N))
                    new_connections += 1

                if new_connections >= max_new:
                    break

        def count_connections(self):
            return int(self.mask.sum().item())

        def save_state(self):
            return (
                self.W.clone(), self.mask.clone(),
                self.state.clone(),
                self.addresses.clone(), self.target_W.clone()
            )

        def restore_state(self, saved):
            self.W = saved[0].clone()
            self.mask = saved[1].clone()
            self.state = saved[2].clone()
            self.addresses = saved[3].clone()
            self.target_W = saved[4].clone()

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
            self.target_W += torch.randn_like(self.target_W) * scale * 0.5
            self.addresses += torch.randn_like(self.addresses) * scale * 0.2

    def evaluate(net, inputs, targets, vocab, ticks=8, n_passes=2, accuracy_hint=0.0):
        """accuracy_hint is passed to forward() for arousal modulation."""
        total_score = 0.0
        correct = 0
        n_items = len(inputs)
        net.reset_state()
        prev_diff = torch.zeros(vocab)

        for pass_num in range(n_passes):
            for idx in range(n_items):
                world = torch.zeros(vocab)
                world[inputs[idx]] = 1.0
                logits = net.forward(world, prev_diff, ticks=ticks,
                                     accuracy=accuracy_hint)
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

    print(f"v22 — 64-CLASS, Accuracy-Modulated Self-Wiring (Arousal)")
    print(f"=" * 60)
    sys.stdout.flush()

    torch.manual_seed(42); random.seed(42)
    v = 64
    perm = torch.randperm(v)
    inputs = list(range(v))
    n_neurons = 320
    n_in = v * 2
    n_out = v
    ticks = 8

    net = ArousalSelfWiringNet(n_neurons, n_in, n_out, addr_dim=4, density=0.06)

    print(f"  Neurons: {n_neurons} (internal: {n_neurons - n_in - n_out})")
    print(f"  Ticks: {ticks}")
    print(f"  Arousal: acc<0.5 → top_k=8,max=6 | acc<0.9 → 4,3 | acc>0.9 → 2,1")
    sys.stdout.flush()

    current_score, current_acc = evaluate(net, inputs, perm, v, ticks,
                                          accuracy_hint=0.0)
    best_score = current_score
    best_acc = current_acc
    phase = "STRUCTURE"
    kept = 0
    stale = 0
    phase_switched = False
    stale_limit = 20000
    max_attempts = 300000
    t0 = time.time()

    print(f"  Start: Acc={current_acc*100:.0f}% | Score: {current_score:.3f} | "
          f"Conns: {net.count_connections()}")
    sys.stdout.flush()

    for attempt in range(max_attempts):
        state = net.save_state()

        if phase == "STRUCTURE":
            net.mutate_structure(rate=0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(rate=0.02)
            else:
                net.mutate_weights(scale=0.05)

        # Pass current accuracy as arousal hint
        new_score, new_acc = evaluate(net, inputs, perm, v, ticks,
                                      accuracy_hint=current_acc)

        if new_score > current_score:
            current_score = new_score
            current_acc = new_acc
            kept += 1
            stale = 0
            if current_score > best_score:
                best_score = current_score
            if new_acc > best_acc:
                best_acc = new_acc
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 3000 and not phase_switched:
            phase = "BOTH"
            phase_switched = True
            _, acc = evaluate(net, inputs, perm, v, ticks,
                              accuracy_hint=current_acc)
            elapsed = time.time() - t0
            print(f"  -- PLATEAU at {attempt+1} ({elapsed:.1f}s) --")
            print(f"  -> BOTH | Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                  f"Conns: {net.count_connections()}")
            sys.stdout.flush()
            stale = 0

        if (attempt + 1) % 2000 == 0:
            _, acc = evaluate(net, inputs, perm, v, ticks,
                              accuracy_hint=current_acc)
            elapsed = time.time() - t0
            print(f"  [{phase:9s}] {attempt+1:6d} | "
                  f"Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                  f"Kept: {kept} | Conns: {net.count_connections()} | "
                  f"Stale: {stale} | {elapsed:.1f}s")
            sys.stdout.flush()

        if new_acc >= 1.0:
            elapsed = time.time() - t0
            print(f"  -> Solved at {attempt+1}! ({elapsed:.1f}s)")
            sys.stdout.flush()
            break

        if stale >= stale_limit:
            elapsed = time.time() - t0
            _, acc = evaluate(net, inputs, perm, v, ticks,
                              accuracy_hint=current_acc)
            print(f"  -> STOPPED at {attempt+1}: stale {stale_limit} ({elapsed:.1f}s)")
            print(f"     Acc: {acc*100:.0f}% | Score: {current_score:.3f}")
            sys.stdout.flush()
            break

    elapsed = time.time() - t0
    _, final_acc = evaluate(net, inputs, perm, v, ticks,
                            accuracy_hint=current_acc)
    print(f"\n  Final: {final_acc*100:.0f}% | Best: {best_acc*100:.0f}% | "
          f"Score: {best_score:.3f} | Conns: {net.count_connections()} | {elapsed:.1f}s")
    sys.stdout.flush()

    return {"acc": final_acc, "best_acc": best_acc, "score": best_score,
            "time": elapsed, "conns": net.count_connections()}


@app.local_entrypoint()
def main():
    print("Running v22 arousal-modulated self-wiring...")
    results = run_experiment.remote()
    print(f"\nDone! {results['acc']*100:.0f}% | Best: {results['best_acc']*100:.0f}% | "
          f"Conns: {results['conns']}")
