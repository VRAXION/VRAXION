"""
No-Backprop v6 — Cleaned up, stable, simple
=============================================
Simple feedforward. Simple task (1-to-1 mapping, no context needed).
Focus on: does the learning mechanism WORK?
"""

import torch
import torch.nn.functional as F
import math


class BrainNet:
    def __init__(self, sizes, oversize=4):
        self.sizes = sizes.copy()
        self.target = sizes.copy()
        for i in range(1, len(self.sizes) - 1):
            self.sizes[i] *= oversize

        self.W = []
        self.b = []
        for i in range(len(self.sizes) - 1):
            s = math.sqrt(2.0 / self.sizes[i])
            self.W.append(torch.randn(self.sizes[i], self.sizes[i+1]) * s)
            self.b.append(torch.zeros(self.sizes[i+1]))

        self.alive = [torch.ones(self.sizes[i]) for i in range(1, len(self.sizes)-1)]
        self.survival = [torch.zeros(self.sizes[i]) for i in range(1, len(self.sizes)-1)]
        self.acts = []

    def forward(self, x):
        self.acts = [x]
        h = x
        for i in range(len(self.W)):
            h = h @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                h = torch.relu(h)
                h = h * self.alive[i]
            self.acts.append(h)
        return h

    def learn(self, reward, lr=0.02):
        """Reward-modulated Hebbian with proper normalization."""
        for i in range(len(self.W)):
            pre = self.acts[i].detach()
            post = self.acts[i+1].detach()

            if pre.dim() > 1:
                pre = pre.mean(0)
                post = post.mean(0)

            # Normalize activations to prevent explosion
            pre_norm = pre / (pre.norm() + 1e-8)
            post_norm = post / (post.norm() + 1e-8)

            # Hebbian update: Δw = lr * reward * pre * post
            delta = lr * reward * torch.outer(pre_norm, post_norm)

            # Clamp update magnitude
            delta = delta.clamp(-0.1, 0.1)

            self.W[i] += delta
            self.b[i] += (lr * 0.1 * reward * post_norm).clamp(-0.01, 0.01)

            # Track survival for hidden layers
            if i < len(self.alive):
                active = (post.abs() > 0.01).float()
                self.survival[i] += reward * active * 0.1

    def prune(self, ratio=0.05):
        pruned = 0
        for idx in range(len(self.alive)):
            alive_idx = (self.alive[idx] > 0).nonzero(as_tuple=True)[0]
            n_alive = len(alive_idx)
            target = self.target[idx + 1]
            n_kill = min(max(1, int(n_alive * ratio)), n_alive - target)
            if n_kill <= 0:
                continue

            scores = self.survival[idx][alive_idx]
            _, weakest = scores.topk(n_kill, largest=False)
            kill = alive_idx[weakest]

            self.alive[idx][kill] = 0
            self.W[idx][:, kill] = 0
            if idx + 1 < len(self.W):
                self.W[idx+1][kill, :] = 0
            self.b[idx][kill] = 0
            pruned += n_kill
        return pruned

    def grow(self, ratio=0.01):
        grown = 0
        for idx in range(len(self.alive)):
            dead = (self.alive[idx] == 0).nonzero(as_tuple=True)[0]
            if len(dead) == 0:
                continue
            n_grow = max(1, int(len(dead) * ratio))
            revive = dead[torch.randperm(len(dead))[:n_grow]]

            self.alive[idx][revive] = 1
            fi = self.W[idx].shape[0]
            self.W[idx][:, revive] = torch.randn(fi, len(revive)) * 0.01
            if idx + 1 < len(self.W):
                fo = self.W[idx+1].shape[1]
                self.W[idx+1][revive, :] = torch.randn(len(revive), fo) * 0.01
            self.b[idx][revive] = 0
            self.survival[idx][revive] = 0
            grown += len(revive)
        return grown


def run():
    # ── Task: simple 1-to-1 mapping ──
    # Input: one-hot of 8 values
    # Output: a fixed permutation (e.g., 0→3, 1→7, 2→1, etc.)
    # This is solvable by a feedforward net without context

    vocab = 8
    # Random but fixed permutation
    torch.manual_seed(42)
    perm = torch.randperm(vocab)
    print(f"Task: learn mapping {list(range(vocab))} → {perm.tolist()}")
    print(f"(feedforward solvable, no context needed)\n")

    # ═══════ BRAIN-LIKE ═══════
    print("=" * 50)
    print("  Brain-like (no backprop)")
    print("=" * 50)

    net = BrainNet([vocab, 16, 16, vocab], oversize=4)
    print(f"  Arch: {net.sizes}, target: {net.target}")

    for epoch in range(2000):
        correct = 0
        for val in range(vocab):
            x = torch.zeros(vocab)
            x[val] = 1.0
            target = perm[val].item()

            logits = net.forward(x)
            pred = torch.softmax(logits, dim=-1)
            predicted = pred.argmax().item()

            # Global reward
            reward = pred[target].item() - 1.0/vocab
            reward = reward * 8  # scale

            net.learn(reward, lr=0.02)

            if predicted == target:
                correct += 1

        # Prune + grow periodically
        if (epoch + 1) % 50 == 0:
            net.prune(ratio=0.08)
            net.grow(ratio=0.02)

        acc = correct / vocab * 100
        if (epoch + 1) % 200 == 0 or epoch == 0:
            alive = [int(a.sum()) for a in net.alive]
            print(f"  Epoch {epoch+1:5d} | Acc: {acc:6.1f}% | Alive: {alive}")

    # Final
    print(f"\n  Final mapping:")
    for val in range(vocab):
        x = torch.zeros(vocab)
        x[val] = 1.0
        logits = net.forward(x)
        pred = logits.argmax().item()
        target = perm[val].item()
        mark = "✓" if pred == target else "✗"
        probs = torch.softmax(logits, dim=-1)
        print(f"    {val} → {pred} (exp:{target}) {mark}  "
              f"conf: {probs[pred].item():.3f}")

    brain_acc = sum(1 for v in range(vocab)
                    if net.forward(F.one_hot(torch.tensor(v),
                    vocab).float()).argmax().item() == perm[v].item()) / vocab * 100

    # ═══════ BACKPROP ═══════
    print(f"\n{'='*50}")
    print("  Backprop comparison")
    print("=" * 50)

    import torch.nn as nn
    bp = nn.Sequential(nn.Linear(vocab, 16), nn.ReLU(),
                       nn.Linear(16, 16), nn.ReLU(),
                       nn.Linear(16, vocab))
    opt = torch.optim.Adam(bp.parameters(), lr=1e-3)

    for epoch in range(2000):
        total_loss = 0
        correct = 0
        for val in range(vocab):
            x = torch.zeros(vocab); x[val] = 1.0
            target = perm[val].item()
            logits = bp(x)
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target]))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            if logits.argmax().item() == target:
                correct += 1

        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:5d} | Acc: {correct/vocab*100:6.1f}% | "
                  f"Loss: {total_loss/vocab:.4f}")

    bp_acc = sum(1 for v in range(vocab)
                 if bp(F.one_hot(torch.tensor(v),
                 vocab).float()).argmax().item() == perm[v].item()) / vocab * 100

    print(f"\n{'='*50}")
    print(f"  Brain-like (no backprop): {brain_acc:.1f}%")
    print(f"  Backprop (Adam):          {bp_acc:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    run()
