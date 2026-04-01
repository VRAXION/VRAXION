"""
No-Backprop v18 — Self-Wiring Graph Network
=============================================
Each neuron has an ADDRESS (small vector).
When a neuron fires, it outputs a TARGET ADDRESS.
Closest neuron to that target gets a new connection.

The network decides its own wiring. Consciously.
32-class only, quick test.
"""

import modal
import time

app = modal.App("vraxion-selfwire")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(cpu=2, memory=1024, timeout=3600, image=image)
def run_experiment():
    import torch
    import math
    import random

    class SelfWiringNet:
        """
        N neurons, each with:
        - address: a small vector (identity in "brain space")
        - target_W: weights that compute "where I want to connect"

        After each forward pass, active neurons propose new connections
        by outputting a target address. Nearest neuron gets wired.
        """

        def __init__(self, n_neurons, n_in, n_out, addr_dim=4, density=0.06):
            self.N = n_neurons
            self.n_in = n_in
            self.n_out = n_out
            self.addr_dim = addr_dim

            # Connection matrix
            s = math.sqrt(2.0 / n_neurons)
            self.W = torch.randn(n_neurons, n_neurons) * s
            self.mask = (torch.rand(n_neurons, n_neurons) < density).float()
            self.mask.fill_diagonal_(0)

            # Neuron addresses — fixed positions in "brain space"
            self.addresses = torch.randn(n_neurons, addr_dim)

            # Target projection: each neuron's activation → target address
            # This is what lets the neuron "choose" where to connect
            self.target_W = torch.randn(n_neurons, addr_dim) * 0.1

            # Persistent state
            self.state = torch.zeros(n_neurons)
            self.decay = 0.5

        def reset_state(self):
            self.state = torch.zeros(self.N)

        def forward(self, world, diff, ticks=8):
            inp = torch.cat([world, diff])
            act = self.state.clone()
            Weff = self.W * self.mask

            for t in range(ticks):
                act = act * self.decay
                act[:self.n_in] = inp
                act = torch.relu(act @ Weff + act * 0.1)
                act[:self.n_in] = inp

            self.state = act.detach()
            self._self_wire(act)
            return act[-self.n_out:]

        def _self_wire(self, activations, top_k=3, max_new=2):
            """
            Active neurons propose connections.
            Each active neuron computes a target address,
            finds the nearest neuron, and connects to it.
            """
            # Find most active neurons (excluding input/output)
            internal_start = self.n_in
            internal_end = self.N - (self.N - self.n_in - (self.N - self.n_in - self.n_out) - self.n_out)
            act = activations[internal_start:].detach()

            if act.sum() < 0.01:
                return

            # Top-k most active neurons
            n_candidates = min(top_k, len(act))
            _, top_idx = act.topk(n_candidates)
            top_idx = top_idx + internal_start  # offset back to global

            new_connections = 0
            for neuron_idx in top_idx:
                ni = neuron_idx.item()
                if activations[ni] < 0.1:
                    continue

                # Neuron computes target address:
                # target = address + activation * target_W
                target = self.addresses[ni] + activations[ni] * self.target_W[ni]

                # Find nearest neuron by address distance
                dists = ((self.addresses - target.unsqueeze(0)) ** 2).sum(dim=1)
                dists[ni] = float('inf')  # not self

                nearest = dists.argmin().item()

                # Add connection if doesn't exist
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
            # Also mutate target weights and addresses
            self.target_W += torch.randn_like(self.target_W) * scale * 0.5
            self.addresses += torch.randn_like(self.addresses) * scale * 0.2

    def evaluate(net, inputs, targets, vocab, ticks=8, n_passes=2):
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

    # === 32-class only ===
    print(f"v18 Self-Wiring Graph Network — 32-class")
    print(f"=" * 60)

    torch.manual_seed(42); random.seed(42)
    v = 32
    perm = torch.randperm(v)
    inputs = list(range(v))
    n_neurons = 160
    n_in = v * 2
    n_out = v
    ticks = 8

    net = SelfWiringNet(n_neurons, n_in, n_out, addr_dim=4, density=0.06)

    print(f"  Neurons: {n_neurons} | Ticks: {ticks} | Addr dim: 4")
    print(f"  Self-wiring: top-3 active neurons propose connections each forward pass")

    current_score, current_acc = evaluate(net, inputs, perm, v, ticks)
    best_score = current_score
    phase = "STRUCTURE"
    kept = 0
    stale = 0
    phase_switched = False
    stale_limit = 20000
    max_attempts = 200000
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
            print(f"  -> BOTH | Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                  f"Conns: {net.count_connections()}")
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
            print(f"  -> STOPPED at {attempt+1}: stale {stale_limit} ({elapsed:.1f}s)")
            print(f"     Acc: {acc*100:.0f}% | Score: {current_score:.3f}")
            break

    elapsed = time.time() - t0
    _, final_acc = evaluate(net, inputs, perm, v, ticks)
    print(f"\n  Final: {final_acc*100:.0f}% | Score: {best_score:.3f} | "
          f"Conns: {net.count_connections()} | {elapsed:.1f}s")

    # Per-input details
    net.reset_state()
    prev_diff = torch.zeros(v)
    # Cold pass
    for idx in range(len(inputs)):
        world = torch.zeros(v)
        world[inputs[idx]] = 1.0
        logits = net.forward(world, prev_diff, ticks=ticks)
        probs = torch.softmax(logits, dim=-1)
        target_vec = torch.zeros(v)
        target_vec[perm[idx].item()] = 1.0
        prev_diff = target_vec - probs.detach()
    # Warm pass
    print(f"\n  Per-input (warm pass):")
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

    print(f"\n  Compare 32-class:")
    print(f"  v12 (layered):          91%")
    print(f"  v16 (graph):            81%")
    print(f"  v17 (graph+recurrent):  88% (timeout)")
    print(f"  v18 (self-wiring):      {final_acc*100:.0f}%")

    return {"acc": final_acc, "score": best_score, "time": elapsed,
            "conns": net.count_connections()}


@app.local_entrypoint()
def main():
    print("Running self-wiring network on Modal CPU...")
    results = run_experiment.remote()
    print(f"\nDone! 32-class: {results['acc']*100:.0f}%")
