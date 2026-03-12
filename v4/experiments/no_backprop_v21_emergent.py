"""
No-Backprop v21 — Emergent Topology from Coordinates
======================================================
NO mask matrix. Connectivity = function of distance in 4D space.
Two neurons are connected proportional to their proximity:
  Weff[i,j] = W[i,j] * exp(-dist²/σ²)

The topology IS the addresses. Moving a neuron in 4D space
changes ALL its connections at once. The "kapcsolási rajz"
emerges from neuron positions — nothing external.

Learnable params: W (weights), addresses (positions), σ (reach).
No mutate_structure(). No mask. No external topology randomizer.

32-class first (quick validation), then 64.
"""

import modal
import time

app = modal.App("vraxion-v21-emergent")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(cpu=2, memory=1024, timeout=1800, image=image)
def run_experiment():
    import torch
    import torch.nn.functional as F
    import math
    import random

    class EmergentNet:
        """
        No mask. Connectivity emerges from neuron positions.
        Weff[i,j] = W[i,j] * exp(-dist(addr_i, addr_j)² / σ²)
        """

        def __init__(self, n_neurons, n_in, n_out, addr_dim=4):
            self.N = n_neurons
            self.n_in = n_in
            self.n_out = n_out
            self.addr_dim = addr_dim

            s = math.sqrt(2.0 / n_neurons)
            self.W = torch.randn(n_neurons, n_neurons) * s
            # No mask!

            # Neuron positions in 4D space
            self.addresses = torch.randn(n_neurons, addr_dim)

            # Per-neuron reach (sigma) — how far each neuron's connections extend
            # Start small = sparse. Let evolution grow it if needed.
            self.sigma = torch.full((n_neurons,), 0.3)

            # Self-wiring direction
            self.target_W = torch.randn(n_neurons, addr_dim) * 0.1

            self.state = torch.zeros(n_neurons)
            self.decay = 0.5

        def reset_state(self):
            self.state = torch.zeros(self.N)

        def _compute_kernel(self):
            """Distance-based connectivity kernel. Precomputed per forward call."""
            # Pairwise squared distances: [N, N]
            diff = self.addresses.unsqueeze(0) - self.addresses.unsqueeze(1)  # [N, N, D]
            dist_sq = (diff ** 2).sum(dim=2)  # [N, N]

            # Per-neuron sigma: use sender's sigma
            sigma_sq = (self.sigma ** 2).unsqueeze(1)  # [N, 1]

            kernel = torch.exp(-dist_sq / (sigma_sq + 1e-6))
            kernel.fill_diagonal_(0)  # no self-connection through kernel
            return kernel

        def forward(self, world, diff, ticks=8):
            inp = torch.cat([world, diff])
            act = self.state.clone()

            # Compute effective weights from positions
            kernel = self._compute_kernel()
            Weff = self.W * kernel

            for t in range(ticks):
                act = act * self.decay
                act[:self.n_in] = inp
                act = F.leaky_relu(act @ Weff + act * 0.1)
                act[:self.n_in] = inp

            self.state = act.detach()
            self._self_wire(act)
            return act[-self.n_out:]

        def _self_wire(self, activations, top_k=3, step_size=0.05):
            """
            Self-wiring = active neurons MOVE toward their target.
            No mask manipulation — moving in address space changes connectivity.
            """
            internal_start = self.n_in
            act = activations[internal_start:].detach()

            if act.sum() < 0.01:
                return

            n_candidates = min(top_k, len(act))
            _, top_idx = act.topk(n_candidates)
            top_idx = top_idx + internal_start

            for neuron_idx in top_idx:
                ni = neuron_idx.item()
                if activations[ni] < 0.1:
                    continue

                # Neuron computes where it WANTS to be
                target_pos = self.addresses[ni] + activations[ni] * self.target_W[ni]

                # Move slightly toward target position
                direction = target_pos - self.addresses[ni]
                self.addresses[ni] += direction * step_size

        def count_effective_connections(self, threshold=0.1):
            """Count connections with kernel > threshold."""
            kernel = self._compute_kernel()
            return int((kernel > threshold).sum().item())

        def save_state(self):
            return (
                self.W.clone(),
                self.state.clone(),
                self.addresses.clone(),
                self.target_W.clone(),
                self.sigma.clone()
            )

        def restore_state(self, saved):
            self.W = saved[0].clone()
            self.state = saved[1].clone()
            self.addresses = saved[2].clone()
            self.target_W = saved[3].clone()
            self.sigma = saved[4].clone()

        def mutate(self, scale=0.05):
            """Mutate everything — but NO structure mutation. Topology emerges."""
            # Weights
            self.W += torch.randn_like(self.W) * scale

            # Addresses — moving neurons changes connectivity!
            self.addresses += torch.randn_like(self.addresses) * scale * 1.0

            # Target direction
            self.target_W += torch.randn_like(self.target_W) * scale * 0.3

            # Sigma — neuron's reach
            self.sigma += torch.randn_like(self.sigma) * scale * 0.5
            self.sigma.clamp_(0.05, 5.0)

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

    # === 32-class quick test ===
    print(f"v21 Emergent Topology — No Mask, Connectivity from Positions")
    print(f"=" * 60)

    torch.manual_seed(42); random.seed(42)
    v = 32
    perm = torch.randperm(v)
    inputs = list(range(v))
    n_neurons = 160
    n_in = v * 2
    n_out = v
    ticks = 8

    net = EmergentNet(n_neurons, n_in, n_out, addr_dim=4)

    print(f"  Neurons: {n_neurons} | Ticks: {ticks}")
    print(f"  NO MASK — Weff = W * exp(-dist²/σ²)")
    print(f"  Topology emerges from neuron positions in 4D space")
    print(f"  Per-neuron sigma (reach) — evolvable")

    current_score, current_acc = evaluate(net, inputs, perm, v, ticks)
    best_score = current_score
    best_acc = current_acc
    kept = 0
    stale = 0
    stale_limit = 20000
    max_attempts = 100000
    t0 = time.time()

    print(f"  Start: Acc={current_acc*100:.0f}% | Score: {current_score:.3f} | "
          f"Eff conns: {net.count_effective_connections()}")

    for attempt in range(max_attempts):
        state = net.save_state()
        net.mutate(scale=0.05)

        new_score, new_acc = evaluate(net, inputs, perm, v, ticks)

        if new_score > current_score:
            current_score = new_score
            kept += 1
            stale = 0
            if current_score > best_score:
                best_score = current_score
            if new_acc > best_acc:
                best_acc = new_acc
        else:
            net.restore_state(state)
            stale += 1

        if (attempt + 1) % 1000 == 0:
            _, acc = evaluate(net, inputs, perm, v, ticks)
            elapsed = time.time() - t0
            sigma_mean = net.sigma.mean().item()
            sigma_std = net.sigma.std().item()
            eff_conns = net.count_effective_connections()
            print(f"  {attempt+1:6d} | "
                  f"Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                  f"Kept: {kept} | Eff conns: {eff_conns} | "
                  f"σ: {sigma_mean:.2f}±{sigma_std:.2f} | "
                  f"Stale: {stale} | {elapsed:.1f}s")

        if new_acc >= 1.0:
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
    sigma_final = net.sigma.mean().item()

    print(f"\n  Final: {final_acc*100:.0f}% | Best: {best_acc*100:.0f}% | "
          f"Score: {best_score:.3f} | {elapsed:.1f}s")
    print(f"  σ: {sigma_final:.3f} (controls connectivity radius)")
    print(f"  Eff connections (>0.1): {net.count_effective_connections()}")

    print(f"\n  Compare 32-class:")
    print(f"  v18  (mask + relu):         100% at ~10k")
    print(f"  v20e (mask + leaky):        100% at ~8k")
    print(f"  v21  (emergent, no mask):   {final_acc*100:.0f}%")

    return {"acc": final_acc, "best_acc": best_acc, "score": best_score,
            "time": elapsed, "sigma": sigma_final,
            "eff_conns": net.count_effective_connections()}


@app.local_entrypoint()
def main():
    print("Running v21 emergent topology...")
    results = run_experiment.remote()
    print(f"\nDone! {results['acc']*100:.0f}% | "
          f"σ={results['sigma']:.2f} | eff_conns={results['eff_conns']}")
