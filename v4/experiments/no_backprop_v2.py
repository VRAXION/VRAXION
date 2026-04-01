"""
No-Backprop Ring Memory v2 — Surprise-based learning
======================================================
Key fix: reward is not "right/wrong" but "MORE or LESS surprised than before?"
This gives useful signal even when predictions are bad.

Also: use simpler task (4 values, not 256) and
eligibility traces for temporal credit assignment.
"""

import torch
import torch.nn.functional as F
import time


class HebbianRingV2:
    def __init__(self, ring_slots=16, hidden_dim=64, vocab=8, lr=0.01):
        self.M = ring_slots
        self.H = hidden_dim
        self.V = vocab
        self.lr = lr

        # Weights
        self.embed = torch.randn(vocab, hidden_dim) * 0.02
        self.W_read = torch.randn(hidden_dim, hidden_dim) * 0.02
        self.W_write = torch.randn(hidden_dim, hidden_dim) * 0.02
        self.W_out = torch.randn(hidden_dim, vocab) * 0.02

        # State
        self.ring = torch.zeros(ring_slots, hidden_dim)
        self.pointer = 0

        # Eligibility traces — decaying memory of recent activity
        self.trace_embed_idx = []
        self.trace_hidden = []
        self.trace_ring_read = []
        self.trace_write_val = []
        self.trace_decay = 0.9  # how fast old traces fade

        # Surprise tracking
        self.prev_surprise = 1.0  # start high

    def reset(self):
        self.ring.zero_()
        self.pointer = 0
        self.trace_embed_idx = []
        self.trace_hidden = []
        self.trace_ring_read = []
        self.trace_write_val = []
        self.prev_surprise = 1.0

    def forward_one(self, byte_val):
        """One step. Returns prediction probs."""
        x = self.embed[byte_val]

        # Read from ring
        r = self.pointer
        slots = [(r - 1) % self.M, r, (r + 1) % self.M]
        ring_read = self.ring[slots].mean(dim=0)
        read_ctx = torch.tanh(ring_read @ self.W_read)

        # Combine
        hidden = torch.tanh(x + read_ctx)

        # Write
        write_val = torch.tanh(hidden @ self.W_write)
        self.ring[self.pointer] = write_val
        self.pointer = (self.pointer + 1) % self.M

        # Predict
        logits = hidden @ self.W_out
        pred = torch.softmax(logits, dim=0)

        # Store trace (decayed)
        self.trace_embed_idx.append(byte_val)
        self.trace_hidden.append(hidden.clone())
        self.trace_ring_read.append(ring_read.clone())
        self.trace_write_val.append(write_val.clone())

        # Keep only last N traces
        max_traces = 10
        if len(self.trace_hidden) > max_traces:
            self.trace_embed_idx = self.trace_embed_idx[-max_traces:]
            self.trace_hidden = self.trace_hidden[-max_traces:]
            self.trace_ring_read = self.trace_ring_read[-max_traces:]
            self.trace_write_val = self.trace_write_val[-max_traces:]

        return pred

    def learn(self, reward):
        """
        Reward-modulated Hebbian update with eligibility traces.
        Recent activations get stronger updates, older ones decay.
        """
        n = len(self.trace_hidden)
        if n == 0:
            return

        for i in range(n):
            # Exponential decay: most recent = strongest
            age = n - 1 - i
            weight = self.lr * reward * (self.trace_decay ** age)

            h = self.trace_hidden[i]
            r = self.trace_ring_read[i]
            w = self.trace_write_val[i]
            idx = self.trace_embed_idx[i]

            # Update weights
            self.embed[idx] += weight * h
            self.W_read += weight * torch.outer(r, h)
            self.W_write += weight * torch.outer(h, w)
            self.W_out += weight * torch.outer(h, torch.zeros(self.V))

        # Normalize weights to prevent explosion
        for W in [self.W_read, self.W_write, self.W_out]:
            norm = W.norm()
            if norm > 10.0:
                W.mul_(10.0 / norm)
        for i in range(self.V):
            norm = self.embed[i].norm()
            if norm > 5.0:
                self.embed[i].mul_(5.0 / norm)


def run():
    print("=" * 60)
    print("  No-Backprop v2 — Surprise-Based Hebbian Learning")
    print("=" * 60)

    # Simple pattern with small vocab
    pattern = [0, 1, 2, 3, 2, 1]
    vocab = max(pattern) + 1
    plen = len(pattern)

    net = HebbianRingV2(ring_slots=16, hidden_dim=32, vocab=vocab, lr=0.005)

    NUM_EPOCHS = 100
    STEPS = 200

    for epoch in range(NUM_EPOCHS):
        net.reset()
        correct = 0
        total = 0
        surprises = []

        for step in range(STEPS):
            curr = pattern[step % plen]
            next_val = pattern[(step + 1) % plen]

            pred = net.forward_one(curr)

            # SURPRISE = how unexpected was the actual next value?
            surprise = 1.0 - pred[next_val].item()  # 0=expected, 1=total surprise

            # REWARD = did surprise DECREASE compared to before?
            # This is the key: not "right/wrong" but "improving/worsening"
            reward = net.prev_surprise - surprise  # positive = less surprised = improving
            net.prev_surprise = surprise * 0.9 + net.prev_surprise * 0.1  # smooth

            net.learn(reward)

            if pred.argmax().item() == next_val:
                correct += 1
            total += 1
            surprises.append(surprise)

        acc = correct / total * 100
        avg_surprise = sum(surprises) / len(surprises)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Acc: {acc:6.2f}% | "
                  f"Avg surprise: {avg_surprise:.4f}")

    # Final test
    print(f"\n  Final test — pattern: {pattern}")
    net.reset()
    # Warm up ring
    for step in range(20):
        net.forward_one(pattern[step % plen])

    for step in range(12):
        curr = pattern[(step + 20) % plen]
        next_val = pattern[(step + 21) % plen]
        pred = net.forward_one(curr)
        top = pred.argmax().item()
        mark = "✓" if top == next_val else "✗"
        probs = [f"{pred[i].item():.3f}" for i in range(vocab)]
        print(f"  {curr} → pred:{top} (exp:{next_val}) {mark}  probs:{probs}")

    # Compare: backprop
    print(f"\n  --- Backprop comparison (same task) ---")
    import torch.nn as nn

    class TinyRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, 32)
            self.rnn = nn.GRU(32, 32, batch_first=True)
            self.out = nn.Linear(32, vocab)
        def forward(self, x):
            return self.out(self.rnn(self.embed(x))[0])

    bp = TinyRNN()
    opt = torch.optim.Adam(bp.parameters(), lr=1e-3)
    for epoch in range(NUM_EPOCHS):
        seq = torch.tensor([pattern[i % plen] for i in range(STEPS)])
        x, y = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)
        loss = F.cross_entropy(bp(x).squeeze(0), y.squeeze(0))
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                acc = (bp(x).squeeze(0).argmax(-1) == y.squeeze(0)).float().mean().item() * 100
            print(f"  Epoch {epoch+1:3d} | Acc: {acc:6.2f}% | Loss: {loss.item():.4f}")


if __name__ == "__main__":
    run()
