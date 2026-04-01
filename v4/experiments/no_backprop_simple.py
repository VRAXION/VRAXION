"""
No-Backprop v5 — Simple feedforward neural net
=================================================
Strip away everything fancy. Just a standard neural network.
Test ONLY the learning mechanism: global reward + build/prune/strengthen/weaken.

3-layer feedforward: Input → Hidden1 → Hidden2 → Output
Task: learn a simple pattern (XOR, or byte sequence)
"""

import torch
import torch.nn.functional as F
import time
import math


class BrainNet:
    """
    Standard feedforward neural network.
    NO backprop. Learns via:
      - Global reward signal
      - Active + good → strengthen
      - Active + bad → weaken
      - Inactive → prune
      - Co-active + good → new connection (synaptogenesis)
    """

    def __init__(self, layer_sizes, oversize=4):
        """
        layer_sizes: e.g. [8, 64, 64, 4] = input→h1→h2→output
        oversize: multiply hidden layers by this factor
        """
        self.n_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes.copy()
        self.target_sizes = layer_sizes.copy()

        # Oversize hidden layers (not input/output)
        for i in range(1, len(self.layer_sizes) - 1):
            self.layer_sizes[i] *= oversize

        # Weight matrices
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # Xavier-ish init
            scale = math.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(torch.randn(fan_in, fan_out) * scale)
            self.biases.append(torch.zeros(fan_out))

        # Per-neuron tracking for each hidden layer
        self.activity = []
        self.survival = []
        self.alive = []
        for i in range(1, len(self.layer_sizes) - 1):  # hidden layers only
            size = self.layer_sizes[i]
            self.activity.append(torch.zeros(size))
            self.survival.append(torch.zeros(size))
            self.alive.append(torch.ones(size))

        # Activation traces for learning
        self.activations = []

    def forward(self, x):
        """Standard forward pass. Track activations."""
        self.activations = [x.clone()]

        h = x
        for i in range(self.n_layers):
            h = h @ self.weights[i] + self.biases[i]

            if i < self.n_layers - 1:
                # Hidden layer: ReLU + alive mask
                h = torch.relu(h)
                h = h * self.alive[i]  # dead neurons output 0

                # Track activity
                act = h.abs().mean(dim=0) if h.dim() > 1 else h.abs()
                self.activity[i] = 0.95 * self.activity[i] + 0.05 * act.detach()

            self.activations.append(h.clone())

        return h  # logits

    def learn(self, reward, lr=0.1):
        """
        THE CORE: global reward → local weight changes.

        For each weight w[i,j] connecting neuron i to neuron j:
          Δw = lr × reward × activation_i × activation_j

        Plus survival tracking for pruning.
        """
        for layer in range(self.n_layers):
            pre = self.activations[layer]    # activations before this layer
            post = self.activations[layer + 1]  # activations after this layer

            # Average over batch if batched
            if pre.dim() > 1:
                pre_mean = pre.mean(dim=0)
                post_mean = post.mean(dim=0)
            else:
                pre_mean = pre
                post_mean = post

            # Reward-modulated Hebbian update
            # Δw = lr × reward × pre × post
            # But we need to be smarter about the output layer
            if layer == self.n_layers - 1:
                # Output layer: use softmax error as modulation
                # post = logits, we want to strengthen correct, weaken wrong
                delta = torch.outer(pre_mean, post_mean * reward)
            else:
                # Hidden layers: pure Hebbian with reward modulation
                delta = torch.outer(pre_mean, post_mean) * reward

            self.weights[layer] += lr * delta
            self.biases[layer] += lr * post_mean * reward * 0.1

            # Update survival scores for hidden layers
            if layer < self.n_layers - 1:
                active = (post_mean.abs() > 0.01).float()
                self.survival[layer] += reward * active

        # Weight normalization to prevent explosion
        for i in range(len(self.weights)):
            norm = self.weights[i].norm()
            max_norm = 5.0 * math.sqrt(self.weights[i].numel())
            if norm > max_norm:
                self.weights[i] *= max_norm / norm

    def prune_and_grow(self, prune_ratio=0.05, grow_ratio=0.02):
        """
        1. PRUNE: kill neurons with lowest survival
        2. GROW: create new random connections where co-activity is high
        """
        total_pruned = 0
        total_grown = 0

        for layer_idx in range(len(self.alive)):
            alive_mask = self.alive[layer_idx]
            n_alive = alive_mask.sum().item()
            target = self.target_sizes[layer_idx + 1]

            # --- PRUNE ---
            if n_alive > target:
                n_prune = max(1, int(n_alive * prune_ratio))
                n_prune = min(n_prune, int(n_alive - target))

                if n_prune > 0:
                    alive_indices = (alive_mask > 0).nonzero(as_tuple=True)[0]
                    scores = self.survival[layer_idx][alive_indices]
                    _, weakest = scores.topk(n_prune, largest=False)
                    kill = alive_indices[weakest]

                    self.alive[layer_idx][kill] = 0
                    # Zero weights to/from dead neurons
                    self.weights[layer_idx][:, kill] = 0
                    if layer_idx + 1 < len(self.weights):
                        self.weights[layer_idx + 1][kill, :] = 0
                    self.biases[layer_idx][kill] = 0

                    total_pruned += n_prune

            # --- GROW ---
            # Reinitialize dead neurons with small random weights
            # (synaptogenesis — new connections where needed)
            dead_indices = (alive_mask == 0).nonzero(as_tuple=True)[0]
            if len(dead_indices) > 0 and n_alive < self.layer_sizes[layer_idx + 1]:
                n_grow = max(1, int(len(dead_indices) * grow_ratio))
                grow = dead_indices[torch.randperm(len(dead_indices))[:n_grow]]

                scale = 0.01
                self.alive[layer_idx][grow] = 1
                fan_in = self.weights[layer_idx].shape[0]
                self.weights[layer_idx][:, grow] = torch.randn(fan_in, len(grow)) * scale
                if layer_idx + 1 < len(self.weights):
                    fan_out = self.weights[layer_idx + 1].shape[1]
                    self.weights[layer_idx + 1][grow, :] = torch.randn(len(grow), fan_out) * scale
                self.biases[layer_idx][grow] = 0
                self.survival[layer_idx][grow] = 0
                self.activity[layer_idx][grow] = 0

                total_grown += n_grow

        return total_pruned, total_grown

    def alive_count(self):
        return [int(a.sum().item()) for a in self.alive]

    def total_alive(self):
        return sum(self.alive_count())


def run():
    print("=" * 60)
    print("  No-Backprop v5 — Simple Feedforward + Brain Learning")
    print("  Build + Prune + Strengthen + Weaken")
    print("  Global reward only. Zero backprop.")
    print("=" * 60)

    # ── Task: learn a repeating pattern ──
    pattern = [0, 1, 2, 3, 2, 1]
    vocab = max(pattern) + 1  # 4

    # Network: 4 input (one-hot) → 64 hidden → 64 hidden → 4 output
    net = BrainNet([vocab, 32, 32, vocab], oversize=4)

    total_h = sum(net.layer_sizes[1:-1])
    print(f"  Architecture: {net.layer_sizes}")
    print(f"  Target: {net.target_sizes}")
    print(f"  Total hidden neurons: {total_h} (will prune to {sum(net.target_sizes[1:-1])})")
    print(f"  Task: predict next in pattern {pattern}")
    print()

    NUM_EPOCHS = 500
    SEQ_LEN = 60
    LR = 0.05

    for epoch in range(NUM_EPOCHS):
        correct = 0
        total = 0
        epoch_reward = 0.0

        for step in range(SEQ_LEN):
            curr = pattern[step % len(pattern)]
            target = pattern[(step + 1) % len(pattern)]

            # One-hot input
            x = torch.zeros(vocab)
            x[curr] = 1.0

            # Forward
            logits = net.forward(x)
            pred = torch.softmax(logits, dim=-1)
            predicted = pred.argmax().item()

            # Global reward: ONE number
            prob_correct = pred[target].item()
            reward = (prob_correct - 0.25) * 4  # scale: -1 to +3 range

            # Learn from global reward
            net.learn(reward, lr=LR)
            epoch_reward += reward

            if predicted == target:
                correct += 1
            total += 1

        # Periodic pruning + growing
        if (epoch + 1) % 20 == 0:
            pruned, grown = net.prune_and_grow(prune_ratio=0.08, grow_ratio=0.02)

        acc = correct / total * 100
        avg_reward = epoch_reward / total

        if (epoch + 1) % 50 == 0 or epoch == 0:
            alive = net.alive_count()
            print(f"  Epoch {epoch+1:4d} | Acc: {acc:6.2f}% | "
                  f"Reward: {avg_reward:+.3f} | "
                  f"Alive: {alive}")

    # ── Final test ──
    print(f"\n  Final predictions:")
    for step in range(12):
        curr = pattern[step % len(pattern)]
        target = pattern[(step + 1) % len(pattern)]
        x = torch.zeros(vocab)
        x[curr] = 1.0
        logits = net.forward(x)
        pred = torch.softmax(logits, dim=-1)
        top = pred.argmax().item()
        probs = [f"{pred[i].item():.3f}" for i in range(vocab)]
        mark = "✓" if top == target else "✗"
        print(f"    {curr}→{top} (exp:{target}) {mark}  {probs}")

    # ── Backprop comparison ──
    print(f"\n  --- Backprop comparison ---")
    import torch.nn as nn

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(vocab, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, vocab)
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    bp_net = SimpleNet()
    opt = torch.optim.Adam(bp_net.parameters(), lr=1e-3)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for step in range(SEQ_LEN):
            curr = pattern[step % len(pattern)]
            target_val = pattern[(step + 1) % len(pattern)]
            x = torch.zeros(vocab)
            x[curr] = 1.0
            logits = bp_net(x)
            loss = F.cross_entropy(logits.unsqueeze(0),
                                   torch.tensor([target_val]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            # Test
            c = 0
            for step in range(SEQ_LEN):
                curr = pattern[step % len(pattern)]
                target_val = pattern[(step + 1) % len(pattern)]
                x = torch.zeros(vocab)
                x[curr] = 1.0
                with torch.no_grad():
                    p = bp_net(x).argmax().item()
                if p == target_val:
                    c += 1
            bp_acc = c / SEQ_LEN * 100
            print(f"  Epoch {epoch+1:4d} | Acc: {bp_acc:6.2f}% | "
                  f"Loss: {total_loss/SEQ_LEN:.4f}")

    print(f"\n  {'='*60}")
    alive = net.alive_count()
    print(f"  Brain-like: {acc:.1f}% | Alive: {alive} (from {net.layer_sizes[1:-1]})")
    print(f"  Backprop:   {bp_acc:.1f}%")
    print(f"  {'='*60}")


if __name__ == "__main__":
    run()
