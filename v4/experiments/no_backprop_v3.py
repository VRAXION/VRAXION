"""
No-Backprop v3 — Global Reward + Self-Modification
=====================================================
The network modifies itself. We only provide ONE global signal.

Method: Weight perturbation
  1. Try a random nudge to weights
  2. Did the global signal improve? Keep it. Didn't? Revert.
  3. No backprop. No gradients. No loss function.

This is biologically plausible — the brain does something similar:
  Try → Reward → Reinforce (or not)
"""

import torch
import torch.nn.functional as F
import time
import copy


class SelfModifyingRing:
    """A ring network that modifies its own weights based on global reward."""

    def __init__(self, ring_slots=16, hidden=32, vocab=8):
        self.M = ring_slots
        self.H = hidden
        self.V = vocab

        # All weights in a single flat vector for easy perturbation
        self.weights = {
            'embed': torch.randn(vocab, hidden) * 0.1,
            'W_rh': torch.randn(hidden, hidden) * 0.1,   # ring → hidden
            'W_hw': torch.randn(hidden, hidden) * 0.1,   # hidden → write
            'W_ho': torch.randn(hidden, vocab) * 0.1,    # hidden → output
        }

        self.ring = torch.zeros(ring_slots, hidden)
        self.ptr = 0

    def reset_state(self):
        self.ring.zero_()
        self.ptr = 0

    def forward(self, seq):
        """Forward pass over a sequence. Returns predictions."""
        preds = []
        for byte_val in seq:
            # Read ring
            slots = [(self.ptr - 1) % self.M, self.ptr, (self.ptr + 1) % self.M]
            ring_ctx = self.ring[slots].mean(0)

            # Hidden
            x = self.weights['embed'][byte_val]
            h = torch.tanh(x + ring_ctx @ self.weights['W_rh'])

            # Write
            self.ring[self.ptr] = torch.tanh(h @ self.weights['W_hw'])
            self.ptr = (self.ptr + 1) % self.M

            # Predict
            logits = h @ self.weights['W_ho']
            preds.append(logits)

        return torch.stack(preds)

    def get_global_signal(self, preds, targets):
        """ONE number. How good was the overall prediction?"""
        probs = torch.softmax(preds, dim=-1)
        # Average probability assigned to correct answers
        correct_probs = probs[range(len(targets)), targets]
        return correct_probs.mean().item()  # 0 to 1

    def try_modification(self, seq, targets, noise_scale=0.01):
        """
        THE CORE LEARNING MECHANISM:
        1. Measure current performance (global signal)
        2. Try random perturbation
        3. Measure again
        4. Keep if better, revert if worse
        """
        # Current performance
        self.reset_state()
        current_preds = self.forward(seq)
        current_signal = self.get_global_signal(current_preds, targets)

        # Generate random perturbation
        noise = {}
        for key, w in self.weights.items():
            noise[key] = torch.randn_like(w) * noise_scale

        # Apply perturbation
        for key in self.weights:
            self.weights[key] += noise[key]

        # Measure with perturbation
        self.reset_state()
        new_preds = self.forward(seq)
        new_signal = self.get_global_signal(new_preds, targets)

        if new_signal > current_signal:
            # Better! Keep the change.
            return new_signal, True
        else:
            # Worse. Revert.
            for key in self.weights:
                self.weights[key] -= noise[key]
            return current_signal, False

    def try_modification_guided(self, seq, targets, noise_scale=0.01):
        """
        Smarter version: try BOTH directions of perturbation.
        Keep whichever is better (or revert if both are worse).
        """
        # Current
        self.reset_state()
        current_preds = self.forward(seq)
        current_signal = self.get_global_signal(current_preds, targets)

        noise = {}
        for key, w in self.weights.items():
            noise[key] = torch.randn_like(w) * noise_scale

        # Try positive direction
        for key in self.weights:
            self.weights[key] += noise[key]
        self.reset_state()
        pos_signal = self.get_global_signal(self.forward(seq), targets)

        # Try negative direction
        for key in self.weights:
            self.weights[key] -= 2 * noise[key]  # undo pos, apply neg
        self.reset_state()
        neg_signal = self.get_global_signal(self.forward(seq), targets)

        # Pick best
        if pos_signal >= neg_signal and pos_signal > current_signal:
            # Positive was best
            for key in self.weights:
                self.weights[key] += 2 * noise[key]
            return pos_signal, "+"
        elif neg_signal > current_signal:
            # Negative was best (already applied)
            return neg_signal, "-"
        else:
            # Both worse, revert to original
            for key in self.weights:
                self.weights[key] += noise[key]
            return current_signal, "="


def run():
    print("=" * 60)
    print("  No-Backprop v3 — Self-Modifying Ring Network")
    print("  ZERO gradients. ZERO loss. ONE global signal.")
    print("  Network tries random changes, keeps what works.")
    print("=" * 60)

    pattern = [0, 1, 2, 3, 2, 1]
    vocab = max(pattern) + 1
    plen = len(pattern)

    # Training sequence
    seq_len = 60
    seq = [pattern[i % plen] for i in range(seq_len)]
    inputs = torch.tensor(seq[:-1])
    targets = torch.tensor(seq[1:])

    net = SelfModifyingRing(ring_slots=16, hidden=32, vocab=vocab)

    # Measure starting performance
    net.reset_state()
    start_preds = net.forward(inputs)
    start_acc = (start_preds.argmax(-1) == targets).float().mean().item() * 100

    print(f"\n  Starting accuracy: {start_acc:.1f}%")
    print(f"  Pattern: {pattern}")
    print(f"  Learning method: try random change → keep if global signal improves")
    print()

    NUM_ATTEMPTS = 5000
    NOISE_SCALE = 0.03

    best_signal = 0
    kept = 0
    t_start = time.time()

    for attempt in range(NUM_ATTEMPTS):
        signal, direction = net.try_modification_guided(
            inputs, targets, noise_scale=NOISE_SCALE
        )

        if direction != "=":
            kept += 1
        best_signal = max(best_signal, signal)

        if (attempt + 1) % 500 == 0:
            net.reset_state()
            preds = net.forward(inputs)
            acc = (preds.argmax(-1) == targets).float().mean().item() * 100
            elapsed = time.time() - t_start
            print(f"  Attempt {attempt+1:5d} | Acc: {acc:6.2f}% | "
                  f"Signal: {signal:.4f} | Kept: {kept} | "
                  f"Time: {elapsed:.1f}s")

    # Final
    net.reset_state()
    final_preds = net.forward(inputs)
    final_acc = (final_preds.argmax(-1) == targets).float().mean().item() * 100

    print(f"\n  Final accuracy: {final_acc:.1f}%")
    print(f"  Modifications kept: {kept}/{NUM_ATTEMPTS} ({kept/NUM_ATTEMPTS*100:.1f}%)")

    # Show predictions
    print(f"\n  Predictions on pattern:")
    net.reset_state()
    for i in range(12):
        curr = pattern[i % plen]
        expected = pattern[(i + 1) % plen]
        logits = net.forward(torch.tensor([curr]))
        pred = logits[0].argmax().item()
        probs = torch.softmax(logits[0], dim=0)
        prob_str = [f"{probs[v].item():.3f}" for v in range(vocab)]
        mark = "✓" if pred == expected else "✗"
        print(f"    {curr}→{pred} (exp:{expected}) {mark}  probs:{prob_str}")

    # Backprop comparison
    print(f"\n  --- Backprop comparison ---")
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
    for epoch in range(100):
        x = inputs.unsqueeze(0)
        y = targets.unsqueeze(0)
        loss = F.cross_entropy(bp(x).squeeze(0), y.squeeze(0))
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        bp_acc = (bp(inputs.unsqueeze(0)).squeeze(0).argmax(-1) == targets).float().mean().item() * 100
    print(f"  Backprop (100 epochs): {bp_acc:.1f}%")

    print(f"\n  {'='*60}")
    print(f"  Self-modifying: {final_acc:.1f}%  vs  Backprop: {bp_acc:.1f}%")
    print(f"  {'='*60}")


if __name__ == "__main__":
    run()
