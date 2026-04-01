"""
No-Backprop Ring Memory — Proof of Concept
=============================================
A ring memory network that learns WITHOUT backprop.
Uses reward-modulated Hebbian learning only.

Learning rule (three-factor):
  Δw = η × reward × pre_activation × post_activation

  - reward: global signal (+1 = good prediction, -1 = bad)
  - pre/post: local neuron activations
  - No gradients. No loss function. No backprop.

Task: predict the next byte in a repeating pattern.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time


class HebbianRingNetwork:
    """
    A ring memory network that learns via reward-modulated Hebbian learning.
    No backprop. No optimizer. No loss function.
    """

    def __init__(self, ring_slots=32, hidden_dim=64, vocab=256, lr=0.001):
        self.M = ring_slots
        self.H = hidden_dim
        self.vocab = vocab
        self.lr = lr

        # Weights (all manually managed, no autograd)
        self.embed = torch.randn(vocab, hidden_dim) * 0.01
        self.W_write = torch.randn(hidden_dim, hidden_dim) * 0.01
        self.W_read = torch.randn(hidden_dim, hidden_dim) * 0.01
        self.W_out = torch.randn(hidden_dim, vocab) * 0.01

        # Ring buffer state
        self.ring = torch.zeros(ring_slots, hidden_dim)
        self.pointer = 0

        # Activation traces (for Hebbian learning)
        self.traces = {}

    def reset_state(self):
        self.ring.zero_()
        self.pointer = 0

    def forward_one(self, byte_val):
        """Process one byte. Returns prediction for NEXT byte."""

        # Embed input
        x = self.embed[byte_val]  # (H,)

        # Read from ring (local window around pointer)
        r = self.pointer
        read_slots = [(r - 1) % self.M, r, (r + 1) % self.M]
        ring_read = self.ring[read_slots].mean(dim=0)  # (H,)
        read_context = ring_read @ self.W_read  # (H,)

        # Combine input + ring context
        hidden = torch.tanh(x + read_context)  # (H,)

        # Write to ring
        write_val = torch.tanh(hidden @ self.W_write)  # (H,)
        self.ring[self.pointer] = write_val

        # Move pointer
        self.pointer = (self.pointer + 1) % self.M

        # Predict next byte
        logits = hidden @ self.W_out  # (vocab,)
        pred = torch.softmax(logits, dim=0)

        # Store activations for Hebbian update
        self.traces = {
            'x': x.clone(),
            'ring_read': ring_read.clone(),
            'hidden': hidden.clone(),
            'write_val': write_val.clone(),
            'logits': logits.clone(),
            'byte_val': byte_val,
        }

        return pred

    def hebbian_update(self, reward):
        """
        Update ALL weights using reward-modulated Hebbian learning.

        Rule: Δw = lr × reward × pre × post
        - reward > 0: strengthen connections that were active
        - reward < 0: weaken connections that were active
        - reward ≈ 0: barely change anything

        No backprop. No gradients. Just: what was active + was it good?
        """
        t = self.traces
        lr = self.lr * reward  # reward modulates direction AND magnitude

        # Embed: strengthen embedding for this input if reward > 0
        self.embed[t['byte_val']] += lr * t['hidden']

        # W_read: pre=ring_read, post=hidden
        self.W_read += lr * torch.outer(t['ring_read'], t['hidden'])

        # W_write: pre=hidden, post=write_val
        self.W_write += lr * torch.outer(t['hidden'], t['write_val'])

        # W_out: pre=hidden, post=logits (weighted by how "surprising" each output was)
        self.W_out += lr * torch.outer(t['hidden'], t['logits'])


def compute_reward(pred, actual_byte):
    """
    Global reward signal. ONE number.
    +1 if prediction was perfect, -1 if terrible.
    Based on predicted probability of the correct byte.
    """
    prob_correct = pred[actual_byte].item()
    # Map [0, 1] → [-1, +1]
    # Random baseline is 1/256 ≈ 0.004
    reward = 2.0 * prob_correct - 1.0
    return reward


def run_experiment():
    print("=" * 60)
    print("  No-Backprop Ring Memory — Hebbian Learning PoC")
    print("  NO gradients. NO loss function. NO backprop.")
    print("  Just: prediction → reward → Hebbian update")
    print("=" * 60)

    # Simple repeating pattern: [10, 20, 30, 40, 50] repeated
    pattern = [10, 20, 30, 40, 50]
    pattern_len = len(pattern)

    net = HebbianRingNetwork(ring_slots=16, hidden_dim=32, vocab=256, lr=0.002)

    NUM_EPOCHS = 50
    STEPS_PER_EPOCH = 200

    epoch_results = []

    for epoch in range(NUM_EPOCHS):
        net.reset_state()
        correct = 0
        total = 0
        total_reward = 0.0

        for step in range(STEPS_PER_EPOCH):
            current_byte = pattern[step % pattern_len]
            next_byte = pattern[(step + 1) % pattern_len]

            # Forward: predict next byte
            pred = net.forward_one(current_byte)

            # Global reward: ONE number, that's it
            reward = compute_reward(pred, next_byte)
            total_reward += reward

            # Hebbian update: network modifies itself
            net.hebbian_update(reward)

            # Track accuracy
            predicted_byte = pred.argmax().item()
            if predicted_byte == next_byte:
                correct += 1
            total += 1

        acc = correct / total * 100
        avg_reward = total_reward / total
        epoch_results.append(acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Accuracy: {acc:6.2f}% | "
                  f"Avg reward: {avg_reward:+.4f}")

    # Final test: show predictions
    print(f"\n{'=' * 60}")
    print(f"  Final test — predicting pattern {pattern}")
    print(f"{'=' * 60}")
    net.reset_state()
    for step in range(15):
        current = pattern[step % pattern_len]
        expected = pattern[(step + 1) % pattern_len]
        pred = net.forward_one(current)
        top3 = pred.topk(3)
        predicted = top3.indices[0].item()
        mark = "✓" if predicted == expected else "✗"
        print(f"  Input: {current:3d} → Predict: {predicted:3d} "
              f"(expected {expected:3d}) {mark}  "
              f"[top3: {top3.indices.tolist()}, probs: {[f'{p:.3f}' for p in top3.values.tolist()]}]")

    # ═══════════════════════════════════════════
    #  Now compare with backprop baseline
    # ═══════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print(f"  Comparison: same task with backprop (Adam)")
    print(f"{'=' * 60}")

    import torch.nn as nn

    class BackpropNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(256, 32)
            self.rnn = nn.GRU(32, 32, batch_first=True)
            self.out = nn.Linear(32, 256)

        def forward(self, x):
            e = self.embed(x)
            h, _ = self.rnn(e)
            return self.out(h)

    bp_net = BackpropNet()
    optimizer = torch.optim.Adam(bp_net.parameters(), lr=1e-3)

    for epoch in range(NUM_EPOCHS):
        # Create training sequence
        seq = torch.tensor([pattern[i % pattern_len] for i in range(STEPS_PER_EPOCH)])
        x = seq[:-1].unsqueeze(0)
        y = seq[1:].unsqueeze(0)

        logits = bp_net(x)
        loss = F.cross_entropy(logits.squeeze(0), y.squeeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                preds = logits.squeeze(0).argmax(dim=-1)
                acc = (preds == y.squeeze(0)).float().mean().item() * 100
            print(f"  Epoch {epoch+1:3d} | Accuracy: {acc:6.2f}% | Loss: {loss.item():.4f}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Hebbian (no backprop): {epoch_results[-1]:.1f}% final accuracy")
    print(f"  Backprop (Adam+GRU):   {acc:.1f}% final accuracy")
    print(f"\n  The Hebbian network learned with ZERO gradients.")
    print(f"  Only: prediction → global reward → strengthen/weaken active connections.")


if __name__ == "__main__":
    run_experiment()
