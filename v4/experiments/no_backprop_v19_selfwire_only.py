"""
No-Backprop v19 — Self-Wiring ONLY (no random mutation)
=========================================================
The network builds itself ONLY through self-wiring.
No random structural mutation. No random weight mutation.
Only: self-wire proposes connections, try→keep/revert evaluates.

target_W and addresses are perturbed (evolved) so the network
can learn WHERE to wire. But the wiring itself comes from
the self-wiring mechanism only.

32-class, quick test.
"""

import modal
import time

app = modal.App("vraxion-v19-selfwire-only")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(cpu=2, memory=1024, timeout=3600, image=image)
def run_experiment():
    import torch
    import math
    import random

    class SelfWireOnlyNet:
        """
        No random mutation of structure.
        Structure changes ONLY through self-wiring.
        We evolve: target_W, addresses, W (weights of existing connections).
        We do NOT randomly add/remove/rewire connections.
        """

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

        def _self_wire(self, activations, top_k=5, max_new=3):
            """More aggressive self-wiring since it's the only builder."""
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

            # Also: prune weak connections from active neurons
            # Active neurons can also REMOVE connections
            for neuron_idx in top_idx:
                ni = neuron_idx.item()
                alive = (self.mask[ni] == 1).nonzero(as_tuple=False)
                if len(alive) > 3:
                    # Remove weakest connection
                    weights = self.W[ni, alive.squeeze(-1)].abs()
                    weakest = weights.argmin()
                    col = alive[weakest].item()
                    self.mask[ni, col] = 0

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

        def mutate_wiring_params(self, scale=0.05):
            """Evolve ONLY the parameters that control self-wiring + weights."""
            # Perturb target_W — this changes WHERE neurons want to wire
            self.target_W += torch.randn_like(self.target_W) * scale

            # Perturb addresses — this changes neuron positions in brain space
            self.addresses += torch.randn_like(self.addresses) * scale * 0.3

            # Perturb existing weights
            noise = torch.randn_like(self.W) * scale
            self.W += noise * self.mask

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

    print(f"v19 Self-Wiring ONLY — No Random Mutation")
    print(f"=" * 60)

    torch.manual_seed(42); random.seed(42)
    v = 32
    perm = torch.randperm(v)
    inputs = list(range(v))
    n_neurons = 160
    n_in = v * 2
    n_out = v
    ticks = 8

    net = SelfWireOnlyNet(n_neurons, n_in, n_out, addr_dim=4, density=0.06)

    print(f"  Neurons: {n_neurons} | Ticks: {ticks}")
    print(f"  NO random structural mutation")
    print(f"  Self-wiring builds the graph (top-5 active, max 3 new + prune weak)")
    print(f"  Evolving: target_W, addresses, weights")

    current_score, current_acc = evaluate(net, inputs, perm, v, ticks)
    best_score = current_score
    kept = 0
    stale = 0
    stale_limit = 25000
    max_attempts = 200000
    t0 = time.time()

    print(f"  Start: Acc={current_acc*100:.0f}% | Score: {current_score:.3f} | "
          f"Conns: {net.count_connections()}")

    for attempt in range(max_attempts):
        state = net.save_state()

        # ONLY evolve wiring parameters — no random structure change
        net.mutate_wiring_params(scale=0.05)

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

        if (attempt + 1) % 2000 == 0:
            _, acc = evaluate(net, inputs, perm, v, ticks)
            elapsed = time.time() - t0
            print(f"  {attempt+1:6d} | "
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

    print(f"\n  v18 (self-wire + mutation): 100% at ~10k")
    print(f"  v19 (self-wire ONLY):      {final_acc*100:.0f}%")

    return {"acc": final_acc, "score": best_score, "time": elapsed}


@app.local_entrypoint()
def main():
    print("Running v19 self-wire only...")
    results = run_experiment.remote()
    print(f"\nDone! {results['acc']*100:.0f}%")
