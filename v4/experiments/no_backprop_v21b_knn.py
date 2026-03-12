"""
No-Backprop v21b — KNN Emergent Topology
==========================================
Topology from positions but DISCRETE: each neuron connects
to its K nearest neighbors. Moving a neuron flips connections
on/off — localized, discrete changes like the mask approach
but the "mask" emerges from positions.

No smooth kernel. No stored mask. K-nearest = the topology.

32-class first.
"""

import modal
import time

app = modal.App("vraxion-v21b-knn")

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

    class KNNEmergentNet:
        """
        Each neuron connects to its K nearest neighbors in 4D space.
        No mask, no kernel. K is per-neuron and evolvable.
        """

        def __init__(self, n_neurons, n_in, n_out, addr_dim=4, k=5):
            self.N = n_neurons
            self.n_in = n_in
            self.n_out = n_out
            self.addr_dim = addr_dim
            self.default_k = k

            s = math.sqrt(2.0 / n_neurons)
            self.W = torch.randn(n_neurons, n_neurons) * s

            self.addresses = torch.randn(n_neurons, addr_dim)
            self.target_W = torch.randn(n_neurons, addr_dim) * 0.1

            # Per-neuron K (how many neighbors to connect to)
            self.K = torch.full((n_neurons,), float(k))

            self.state = torch.zeros(n_neurons)
            self.decay = 0.5

        def reset_state(self):
            self.state = torch.zeros(self.N)

        def _compute_knn_mask(self):
            """Compute mask from K-nearest neighbors."""
            diff = self.addresses.unsqueeze(0) - self.addresses.unsqueeze(1)
            dist_sq = (diff ** 2).sum(dim=2)
            dist_sq.fill_diagonal_(float('inf'))

            mask = torch.zeros(self.N, self.N)
            for i in range(self.N):
                ki = max(1, min(int(self.K[i].item()), self.N - 1))
                _, indices = dist_sq[i].topk(ki, largest=False)
                mask[i, indices] = 1.0
            return mask

        def forward(self, world, diff, ticks=8):
            inp = torch.cat([world, diff])
            act = self.state.clone()

            mask = self._compute_knn_mask()
            Weff = self.W * mask

            for t in range(ticks):
                act = act * self.decay
                act[:self.n_in] = inp
                act = F.leaky_relu(act @ Weff + act * 0.1)
                act[:self.n_in] = inp

            self.state = act.detach()
            self._self_wire(act)
            return act[-self.n_out:]

        def _self_wire(self, activations, top_k=3, step_size=0.05):
            """Active neurons move toward their target."""
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
                target_pos = self.addresses[ni] + activations[ni] * self.target_W[ni]
                direction = target_pos - self.addresses[ni]
                self.addresses[ni] += direction * step_size

        def count_connections(self):
            mask = self._compute_knn_mask()
            return int(mask.sum().item())

        def save_state(self):
            return (
                self.W.clone(), self.state.clone(),
                self.addresses.clone(), self.target_W.clone(),
                self.K.clone()
            )

        def restore_state(self, saved):
            self.W = saved[0].clone()
            self.state = saved[1].clone()
            self.addresses = saved[2].clone()
            self.target_W = saved[3].clone()
            self.K = saved[4].clone()

        def mutate(self, scale=0.05):
            # Weights
            self.W += torch.randn_like(self.W) * scale

            # Addresses — moving neurons changes KNN!
            self.addresses += torch.randn_like(self.addresses) * scale

            # Target direction
            self.target_W += torch.randn_like(self.target_W) * scale * 0.3

            # K — per-neuron connectivity degree
            self.K += torch.randn_like(self.K) * scale * 2.0
            self.K.clamp_(1.0, 30.0)

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

    print(f"v21b KNN Emergent Topology")
    print(f"=" * 60)

    torch.manual_seed(42); random.seed(42)
    v = 32
    perm = torch.randperm(v)
    inputs = list(range(v))
    n_neurons = 160
    n_in = v * 2
    n_out = v
    ticks = 8

    net = KNNEmergentNet(n_neurons, n_in, n_out, addr_dim=4, k=5)

    print(f"  Neurons: {n_neurons} | Ticks: {ticks}")
    print(f"  NO MASK — K nearest neighbors = connections")
    print(f"  Per-neuron K (init=5, evolvable)")
    print(f"  Self-wiring = neurons MOVE in address space")

    current_score, current_acc = evaluate(net, inputs, perm, v, ticks)
    best_score = current_score
    best_acc = current_acc
    kept = 0
    stale = 0
    stale_limit = 15000
    max_attempts = 100000
    t0 = time.time()

    print(f"  Start: Acc={current_acc*100:.0f}% | Score: {current_score:.3f} | "
          f"Conns: {net.count_connections()}")

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

        if (attempt + 1) % 500 == 0:
            _, acc = evaluate(net, inputs, perm, v, ticks)
            elapsed = time.time() - t0
            k_mean = net.K.mean().item()
            k_std = net.K.std().item()
            print(f"  {attempt+1:6d} | "
                  f"Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                  f"Kept: {kept} | K: {k_mean:.1f}±{k_std:.1f} | "
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
    k_final = net.K.mean().item()

    print(f"\n  Final: {final_acc*100:.0f}% | Best: {best_acc*100:.0f}% | "
          f"Score: {best_score:.3f} | {elapsed:.1f}s")
    print(f"  K: {k_final:.1f} | Conns: {net.count_connections()}")

    print(f"\n  Compare 32-class:")
    print(f"  v18  (mask + relu):         100% at ~10k")
    print(f"  v21  (Gauss kernel):        19% (failed)")
    print(f"  v21b (KNN):                 {final_acc*100:.0f}%")

    return {"acc": final_acc, "best_acc": best_acc, "score": best_score,
            "time": elapsed, "k_mean": k_final,
            "conns": net.count_connections()}


@app.local_entrypoint()
def main():
    print("Running v21b KNN emergent topology...")
    results = run_experiment.remote()
    print(f"\nDone! {results['acc']*100:.0f}% | K={results['k_mean']:.1f}")
