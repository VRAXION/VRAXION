"""
No-Backprop v4 — Oversize + Pruning (Brain-like)
===================================================
Instead of searching WHERE to add signal in a huge space,
start OVERSIZED and PRUNE what doesn't help.

The brain way:
  1. Start with WAY too many connections (6x overproduced)
  2. Global signal: good/bad
  3. Active + good → survives
  4. Active + bad → weakened
  5. Inactive → dies (pruning)
  6. What remains = learned structure

No backprop. No gradients. Just: oversize → prune → structure.
"""

import modal
import time

app = modal.App("vraxion-prune")

vraxion_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pyyaml")
    .add_local_dir("/home/deck/work/vraxion/v4/model", remote_path="/root/vraxion/model")
    .add_local_dir("/home/deck/work/vraxion/v4/training", remote_path="/root/vraxion/training")
    .add_local_dir("/home/deck/work/vraxion/v4/config", remote_path="/root/vraxion/config")
    .add_local_dir("/home/deck/work/vraxion/v4/training_data", remote_path="/root/vraxion/training_data")
)


@app.function(gpu="T4", timeout=900, image=vraxion_image)
def run_experiment():
    import sys
    sys.path.insert(0, "/root/vraxion/training")
    sys.path.insert(0, "/root/vraxion/model")

    import torch
    import torch.nn.functional as F
    import numpy as np
    from pathlib import Path
    from model_factory import load_model_config, build_model_spec, build_model_from_spec
    from train import ByteDataset, func_discover_dat

    device = "cuda"
    v4_root = Path("/root/vraxion")

    # ═══════════════════════════════════════════
    #  Brain-like Pruning Network
    # ═══════════════════════════════════════════

    class PruningRing(torch.nn.Module):
        """
        Ring memory network that learns by PRUNING, not by gradient descent.

        Each weight has:
          - value (the weight itself)
          - activity (how often it fires)
          - survival score (accumulated reward when active)
          - alive mask (0 = pruned, 1 = alive)
        """

        def __init__(self, ring_slots=16, hidden=128, vocab=256, oversize=4):
            super().__init__()
            # OVERSIZED — start with `oversize` x more capacity than needed
            H = hidden * oversize  # 4x oversized hidden dim

            self.M = ring_slots
            self.H = H
            self.V = vocab
            self.target_H = hidden  # what we'll prune DOWN to

            # Weights
            self.embed = torch.nn.Parameter(torch.randn(vocab, H) * 0.02)
            self.W_rh = torch.nn.Parameter(torch.randn(H, H) * 0.02)
            self.W_hw = torch.nn.Parameter(torch.randn(H, H) * 0.02)
            self.W_ho = torch.nn.Parameter(torch.randn(H, vocab) * 0.02)

            # Per-neuron tracking (NOT learned, just bookkeeping)
            self.register_buffer('activity', torch.zeros(H))
            self.register_buffer('survival_score', torch.zeros(H))
            self.register_buffer('alive', torch.ones(H))
            self.register_buffer('activation_count', torch.zeros(H))

            # Ring state
            self.register_buffer('ring', torch.zeros(ring_slots, H))
            self.register_buffer('ptr', torch.tensor(0))

            # Track current activations for reward modulation
            self.current_hidden = None

        def reset_state(self):
            self.ring.zero_()
            self.ptr.zero_()

        def forward(self, x_byte):
            """Forward one byte. Returns logits for next byte."""
            # Embed
            x = self.embed[x_byte] * self.alive  # dead neurons contribute nothing

            # Read ring
            r = self.ptr.item()
            slots = [(r - 1) % self.M, r, (r + 1) % self.M]
            ring_read = self.ring[slots].mean(0)

            # Hidden (masked by alive)
            hidden = torch.tanh(x + ring_read @ self.W_rh) * self.alive

            # Track which neurons were active
            self.current_hidden = hidden.detach().clone()
            activation = hidden.abs()

            # Update activity tracking (exponential moving average)
            self.activity = 0.99 * self.activity + 0.01 * activation

            # Track how many times each neuron was significantly active
            active_mask = (activation > 0.1).float()
            self.activation_count += active_mask

            # Write to ring
            write_val = torch.tanh(hidden @ self.W_hw) * self.alive
            self.ring[self.ptr.item()] = write_val
            self.ptr = (self.ptr + 1) % self.M

            # Output
            logits = hidden @ self.W_ho
            return logits

        def receive_global_signal(self, reward):
            """
            THE LEARNING MECHANISM.
            One global signal. The network decides what to do with it.

            reward > 0: neurons that were active get survival boost
            reward < 0: neurons that were active get survival penalty
            reward ≈ 0: no change

            Then: strengthen surviving neurons, weaken dying ones.
            """
            if self.current_hidden is None:
                return

            # Which neurons were active?
            activation = self.current_hidden.abs()
            active = (activation > 0.1).float()

            # Update survival scores
            # Active + positive reward = survive
            # Active + negative reward = die faster
            self.survival_score += reward * active * activation

            # Modulate weights based on survival
            # Neurons with high survival → their connections strengthen slightly
            # Neurons with low survival → their connections weaken
            survival_norm = torch.tanh(self.survival_score * 0.01)  # [-1, 1]

            # Weight modulation (very subtle — like synaptic scaling)
            scale = 1.0 + 0.001 * survival_norm * self.alive
            with torch.no_grad():
                # Scale rows of weight matrices by neuron health
                self.W_rh.data *= scale.unsqueeze(1)
                self.W_hw.data *= scale.unsqueeze(1)
                # Scale embed outputs
                self.embed.data *= scale.unsqueeze(0)
                # Scale output contributions
                self.W_ho.data *= scale.unsqueeze(1)

        def prune(self, prune_ratio=0.05):
            """
            Kill the weakest neurons.
            Like synaptic pruning in infant brain development.

            Returns number of neurons pruned.
            """
            alive_indices = (self.alive > 0).nonzero(as_tuple=True)[0]
            n_alive = alive_indices.numel()
            if n_alive <= self.target_H:
                return 0  # Already at target size

            # How many to prune this round
            n_prune = max(1, int(n_alive * prune_ratio))
            # Don't prune below target
            n_prune = min(n_prune, n_alive - self.target_H)
            if n_prune <= 0:
                return 0

            # Score = survival + activity (both matter)
            score = self.survival_score + self.activity * 0.1

            # Among alive neurons, find the weakest
            alive_scores = score[alive_indices]
            _, weakest_idx = alive_scores.topk(n_prune, largest=False)
            prune_neurons = alive_indices[weakest_idx]

            # Kill them
            with torch.no_grad():
                self.alive[prune_neurons] = 0
                # Zero out their weights (they're dead)
                self.W_rh.data[prune_neurons, :] = 0
                self.W_rh.data[:, prune_neurons] = 0
                self.W_hw.data[prune_neurons, :] = 0
                self.W_hw.data[:, prune_neurons] = 0
                self.W_ho.data[prune_neurons, :] = 0
                self.embed.data[:, prune_neurons] = 0
                self.ring[:, prune_neurons] = 0

            return n_prune

    # ═══════════════════════════════════════════
    #  Run experiment
    # ═══════════════════════════════════════════

    model_config = load_model_config(v4_root)
    files = func_discover_dat(str(v4_root / "training_data"))

    BATCH = 8
    SEQ = 64
    NUM_STEPS = 500
    PRUNE_EVERY = 50
    PRUNE_RATIO = 0.1

    print(f"{'='*60}")
    print(f"  Brain-Like Pruning — No Backprop")
    print(f"  Start 4x oversized → prune to target → structure emerges")
    print(f"{'='*60}")

    torch.manual_seed(1337)
    net = PruningRing(ring_slots=16, hidden=64, vocab=256, oversize=4).to(device)

    total_params = sum(p.numel() for p in net.parameters())
    alive_start = net.alive.sum().item()
    print(f"  Total params: {total_params:,}")
    print(f"  Hidden dim: {net.H} (oversized) → target: {net.target_H}")
    print(f"  Alive neurons: {alive_start:.0f}")

    dataset = ByteDataset(files, seq_len=SEQ, embed_mode=True, seed=42)
    t0 = time.time()

    for step in range(NUM_STEPS):
        x, y, mask = dataset.sample_batch(BATCH, device)

        total_reward = 0.0
        correct = 0
        total = 0

        net.reset_state()

        for t_step in range(SEQ - 1):
            # Forward one byte at a time
            logits = net(x[0, t_step].item())
            pred = torch.softmax(logits, dim=0)

            target = y[0, t_step].item()
            m = mask[0, t_step].item()

            if m > 0:
                # Global signal: how surprised were we?
                prob_correct = pred[target].item()
                reward = prob_correct - (1.0 / 256)  # relative to random baseline
                reward = max(-1.0, min(1.0, reward * 10))  # scale and clamp

                net.receive_global_signal(reward)
                total_reward += reward

                if pred.argmax().item() == target:
                    correct += 1
                total += 1

        # Periodic pruning
        if (step + 1) % PRUNE_EVERY == 0 and step < NUM_STEPS - 50:
            n_pruned = net.prune(prune_ratio=PRUNE_RATIO)
            alive = net.alive.sum().item()
            acc = correct / max(total, 1) * 100
            avg_reward = total_reward / max(total, 1)
            elapsed = time.time() - t0
            print(f"  Step {step+1:3d} | Acc: {acc:6.2f}% | "
                  f"Reward: {avg_reward:+.4f} | "
                  f"Alive: {alive:.0f}/{net.H} | "
                  f"Pruned: {n_pruned} | {elapsed:.1f}s")
        elif (step + 1) % 100 == 0:
            alive = net.alive.sum().item()
            acc = correct / max(total, 1) * 100
            avg_reward = total_reward / max(total, 1)
            elapsed = time.time() - t0
            print(f"  Step {step+1:3d} | Acc: {acc:6.2f}% | "
                  f"Reward: {avg_reward:+.4f} | "
                  f"Alive: {alive:.0f}/{net.H} | {elapsed:.1f}s")

    # Final stats
    alive_final = net.alive.sum().item()
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Neurons: {alive_start:.0f} → {alive_final:.0f} "
          f"({alive_final/alive_start*100:.1f}% survived)")
    print(f"  Pruned: {alive_start - alive_final:.0f} neurons")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Method: ZERO backprop, global reward + pruning only")

    # Compare: how does same-size backprop do?
    print(f"\n  --- Backprop comparison (same architecture, target size) ---")

    torch.manual_seed(1337)
    bp_net = PruningRing(ring_slots=16, hidden=64, vocab=256, oversize=1).to(device)
    bp_net.train()
    optimizer = torch.optim.Adam(bp_net.parameters(), lr=1e-3)

    for step in range(NUM_STEPS):
        x, y, mask = dataset.sample_batch(BATCH, device)

        bp_net.reset_state()
        all_logits = []
        for t_step in range(SEQ - 1):
            logits = bp_net(x[0, t_step].item())
            all_logits.append(logits)

        all_logits = torch.stack(all_logits)
        targets = y[0, :SEQ-1]
        loss = F.cross_entropy(all_logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bp_net.parameters(), 10.0)
        optimizer.step()

        if (step + 1) % 100 == 0:
            with torch.no_grad():
                preds = all_logits.argmax(-1)
                acc = (preds == targets).float().mean().item() * 100
            print(f"  Step {step+1:3d} | Acc: {acc:6.2f}% | Loss: {loss.item():.4f}")

    return {"alive_start": alive_start, "alive_final": alive_final, "time": elapsed}


@app.local_entrypoint()
def main():
    print("Running brain-like pruning experiment...")
    results = run_experiment.remote()
    print("\nDone!")
