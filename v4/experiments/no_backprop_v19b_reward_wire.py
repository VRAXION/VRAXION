"""
No-Backprop v19b — Reward-Modulated Self-Wiring
==================================================
The reward signal (keep/revert) directly updates target_W.
No random mutation of structure OR wiring params.

When a self-wiring attempt leads to improvement:
  -> reinforce: push target_W toward that direction
When it leads to worse performance:
  -> anti-reinforce: push target_W away

The system learns WHERE to wire from the reward signal.
If this works, the two levels (mutation + self-wiring) merge into one.

32-class, key test.
"""

import modal
import time

app = modal.App("vraxion-v19b-reward-wire")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(cpu=2, memory=1024, timeout=3600, image=image)
def run_experiment():
    import torch
    import math
    import random

    class RewardWiringNet:
        """
        Self-wiring + reward-modulated learning of wiring params.
        No random mutation at all.

        Each step:
        1. Forward pass (self-wiring proposes new connections)
        2. Evaluate (score)
        3. If better: reinforce target_W direction (the neurons "learned" where to wire)
        4. If worse: revert AND anti-reinforce target_W
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

            # Track which neurons fired and their targets for reward learning
            self.last_wire_actions = []  # [(neuron_idx, target_addr, accepted)]

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
            """Self-wire and record actions for reward learning."""
            self.last_wire_actions = []
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

                accepted = False
                if self.mask[ni, nearest] == 0:
                    self.mask[ni, nearest] = 1
                    self.W[ni, nearest] = random.gauss(0, math.sqrt(2.0 / self.N))
                    new_connections += 1
                    accepted = True

                # Record the action for reward learning
                self.last_wire_actions.append((ni, target.clone(), accepted))

                if new_connections >= max_new:
                    break

            # Prune weak connections from active neurons
            for neuron_idx in top_idx:
                ni = neuron_idx.item()
                alive = (self.mask[ni] == 1).nonzero(as_tuple=False)
                if len(alive) > 3:
                    weights = self.W[ni, alive.squeeze(-1)].abs()
                    weakest = weights.argmin()
                    col = alive[weakest].item()
                    self.mask[ni, col] = 0

        def reward_update(self, reward, lr=0.01):
            """
            Update target_W based on reward signal.
            reward > 0: reinforce (push target_W toward successful targets)
            reward < 0: anti-reinforce (push target_W away)
            """
            for ni, target_addr, accepted in self.last_wire_actions:
                if not accepted:
                    continue

                # Direction the neuron "aimed" at
                current_addr = self.addresses[ni]
                direction = target_addr - current_addr

                # Modulate: reward * direction * lr
                # Positive reward -> push target_W to aim here again
                # Negative reward -> push target_W to aim elsewhere
                self.target_W[ni] += reward * direction * lr

            # Also slightly adjust weights based on reward
            # Positive reward: slightly strengthen active connections
            # Negative reward: slightly weaken
            if abs(reward) > 0.001:
                active_mask = (self.state.abs() > 0.1).float()
                weight_update = reward * 0.005
                self.W += weight_update * self.mask * active_mask.unsqueeze(1)

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

        def small_perturbation(self, scale=0.01):
            """Tiny perturbation to escape local minima. Not random mutation."""
            self.target_W += torch.randn_like(self.target_W) * scale
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

    print(f"v19b Reward-Modulated Self-Wiring")
    print(f"=" * 60)

    torch.manual_seed(42); random.seed(42)
    v = 32
    perm = torch.randperm(v)
    inputs = list(range(v))
    n_neurons = 160
    n_in = v * 2
    n_out = v
    ticks = 8

    net = RewardWiringNet(n_neurons, n_in, n_out, addr_dim=4, density=0.06)

    print(f"  Neurons: {n_neurons} | Ticks: {ticks}")
    print(f"  NO random mutation")
    print(f"  Reward signal modulates target_W (learning WHERE to wire)")
    print(f"  Small perturbation for exploration (scale=0.01)")

    current_score, current_acc = evaluate(net, inputs, perm, v, ticks)
    best_score = current_score
    kept = 0
    stale = 0
    stale_limit = 30000
    max_attempts = 200000
    t0 = time.time()

    # Track target_W magnitude over time
    tw_history = []

    print(f"  Start: Acc={current_acc*100:.0f}% | Score: {current_score:.3f} | "
          f"Conns: {net.count_connections()}")
    print(f"  target_W magnitude: {net.target_W.norm(dim=1).mean().item():.4f}")

    for attempt in range(max_attempts):
        state = net.save_state()

        # Small perturbation for exploration
        net.small_perturbation(scale=0.01)

        new_score, new_acc = evaluate(net, inputs, perm, v, ticks)
        delta = new_score - current_score

        if new_score > current_score:
            current_score = new_score
            kept += 1
            stale = 0
            if current_score > best_score:
                best_score = current_score

            # REWARD: reinforce self-wiring direction
            net.reward_update(reward=delta * 10, lr=0.02)

        else:
            # ANTI-REWARD: push target_W away from bad direction
            net.reward_update(reward=delta * 5, lr=0.01)

            net.restore_state(state)
            stale += 1

        if (attempt + 1) % 2000 == 0:
            _, acc = evaluate(net, inputs, perm, v, ticks)
            elapsed = time.time() - t0
            tw_mag = net.target_W.norm(dim=1).mean().item()
            tw_history.append(tw_mag)
            print(f"  {attempt+1:6d} | "
                  f"Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                  f"Kept: {kept} | Conns: {net.count_connections()} | "
                  f"target_W: {tw_mag:.4f} | "
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
    tw_final = net.target_W.norm(dim=1).mean().item()

    print(f"\n  Final: {final_acc*100:.0f}% | Score: {best_score:.3f} | "
          f"Conns: {net.count_connections()} | {elapsed:.1f}s")
    print(f"  target_W magnitude: start=0.178 -> final={tw_final:.4f}")

    if tw_history:
        print(f"  target_W trajectory: {' -> '.join(f'{x:.4f}' for x in tw_history[:10])}")

    if tw_final > 0.25:
        print(f"  -> target_W GREW — neurons learned to aim further!")
    elif tw_final < 0.12:
        print(f"  -> target_W SHRANK — neurons learned to aim closer")
    else:
        print(f"  -> target_W stable — reward signal not strong enough?")

    print(f"\n  Compare:")
    print(f"  v18 (mutation + self-wire): 100%")
    print(f"  v19 (self-wire only):      22%")
    print(f"  v19b (reward self-wire):   {final_acc*100:.0f}%")

    return {"acc": final_acc, "score": best_score, "time": elapsed,
            "tw_final": tw_final}


@app.local_entrypoint()
def main():
    print("Running v19b reward-modulated self-wiring...")
    results = run_experiment.remote()
    print(f"\nDone! {results['acc']*100:.0f}% | target_W: {results['tw_final']:.4f}")
