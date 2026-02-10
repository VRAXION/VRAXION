"""
Test learned pointer dynamics (jump/inertia/deadzone).

We know Gaussian attention works (100% accuracy).
The ONLY remaining feature is learned pointer movement.
"""

import torch
import torch.nn as nn

print("="*70)
print("POINTER DYNAMICS TEST")
print("="*70)
print()

# Data
torch.manual_seed(42)
x = torch.randint(0, 10, (100, 16, 1)).float()
y = x[:, -1, 0].long()


def train_and_test(model, name, steps=500):
    """Train model and return best accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_acc = 0.0
    for step in range(steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == y).float().mean().item()
        if acc > max_acc:
            max_acc = acc

        if step % 100 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}, acc={acc*100:5.1f}%")

    result = "PASS" if max_acc > 0.90 else "FAIL"
    print(f"  Best: {max_acc*100:.1f}% [{result}]")
    return max_acc


# ==============================================================================
# BASELINE: Gaussian attention + simple walk
# ==============================================================================
print("[BASELINE] Gaussian attention + simple walk (we know this works)")
print("-"*70)

class GaussianWalk(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 256)
        self.update = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256)
        self.output = nn.Linear(256, 10)
        self.context_strength = nn.Parameter(torch.tensor(0.2))

    def _gaussian_attention(self, pointer, num_positions):
        B = pointer.size(0)
        K = 2
        tau = 8.0
        offsets = torch.arange(-K, K + 1, device=pointer.device)
        base = torch.floor(pointer).long().clamp(0, num_positions - 1)
        indices = (base.unsqueeze(1) + offsets.unsqueeze(0)) % num_positions
        indices_float = indices.float()
        delta = indices_float - pointer.unsqueeze(1)
        logits = -(delta ** 2) / tau
        weights = torch.softmax(logits, dim=1)
        return indices, weights

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, 256)
        memory = torch.zeros(B, 64, 256)
        pointer = torch.zeros(B)

        for t in range(T):
            inp = torch.tanh(self.embed(x[:, t, :]))

            # Gaussian READ
            indices, weights = self._gaussian_attention(pointer, 64)
            indices_exp = indices.unsqueeze(-1).expand(-1, -1, 256)
            neighborhood = memory.gather(1, indices_exp)
            context = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)

            context_scale = torch.sigmoid(self.context_strength)
            combined = inp + context_scale * context

            h = torch.tanh(self.update(combined + h))
            h = self.norm(h)

            # Gaussian WRITE
            update_broadcast = h.unsqueeze(1).expand(-1, weights.size(1), -1)
            contribution = weights.unsqueeze(-1) * update_broadcast
            memory = memory.scatter_add(1, indices_exp, contribution)

            # Simple walk
            pointer = (pointer + 1.0) % 64

        return self.output(h)

torch.manual_seed(42)
train_and_test(GaussianWalk(), "GaussianWalk")
print()


# ==============================================================================
# TEST 1: Add jump targets (but always walk)
# ==============================================================================
print("[TEST 1] Add jump targets (jump_prob=0, always walk)")
print("-"*70)

class WithJumpTargets(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 256)
        self.update = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256)
        self.output = nn.Linear(256, 10)
        self.context_strength = nn.Parameter(torch.tensor(0.2))

        # Jump targets (not used yet)
        self.jump_targets = nn.Parameter(torch.rand(64) * 64)

    def _gaussian_attention(self, pointer, num_positions):
        B = pointer.size(0)
        K = 2
        tau = 8.0
        offsets = torch.arange(-K, K + 1, device=pointer.device)
        base = torch.floor(pointer).long().clamp(0, num_positions - 1)
        indices = (base.unsqueeze(1) + offsets.unsqueeze(0)) % num_positions
        indices_float = indices.float()
        delta = indices_float - pointer.unsqueeze(1)
        logits = -(delta ** 2) / tau
        weights = torch.softmax(logits, dim=1)
        return indices, weights

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, 256)
        memory = torch.zeros(B, 64, 256)
        pointer = torch.zeros(B)

        for t in range(T):
            inp = torch.tanh(self.embed(x[:, t, :]))

            indices, weights = self._gaussian_attention(pointer, 64)
            indices_exp = indices.unsqueeze(-1).expand(-1, -1, 256)
            neighborhood = memory.gather(1, indices_exp)
            context = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)

            context_scale = torch.sigmoid(self.context_strength)
            combined = inp + context_scale * context

            h = torch.tanh(self.update(combined + h))
            h = self.norm(h)

            update_broadcast = h.unsqueeze(1).expand(-1, weights.size(1), -1)
            contribution = weights.unsqueeze(-1) * update_broadcast
            memory = memory.scatter_add(1, indices_exp, contribution)

            # Still just walk (jump targets exist but unused)
            pointer = (pointer + 1.0) % 64

        return self.output(h)

torch.manual_seed(42)
train_and_test(WithJumpTargets(), "WithJumpTargets")
print()


# ==============================================================================
# TEST 2: LEARNED jump probability
# ==============================================================================
print("[TEST 2] LEARNED jump/walk probability")
print("-"*70)

class LearnedJump(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 256)
        self.update = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256)
        self.output = nn.Linear(256, 10)
        self.context_strength = nn.Parameter(torch.tensor(0.2))

        # Pointer control
        self.jump_targets = nn.Parameter(torch.rand(64) * 64)
        self.jump_score_head = nn.Linear(256, 1)  # Learn when to jump

    def _gaussian_attention(self, pointer, num_positions):
        B = pointer.size(0)
        K = 2
        tau = 8.0
        offsets = torch.arange(-K, K + 1, device=pointer.device)
        base = torch.floor(pointer).long().clamp(0, num_positions - 1)
        indices = (base.unsqueeze(1) + offsets.unsqueeze(0)) % num_positions
        indices_float = indices.float()
        delta = indices_float - pointer.unsqueeze(1)
        logits = -(delta ** 2) / tau
        weights = torch.softmax(logits, dim=1)
        return indices, weights

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, 256)
        memory = torch.zeros(B, 64, 256)
        pointer = torch.empty(B).uniform_(0, 64)  # Random init

        for t in range(T):
            inp = torch.tanh(self.embed(x[:, t, :]))

            indices, weights = self._gaussian_attention(pointer, 64)
            indices_exp = indices.unsqueeze(-1).expand(-1, -1, 256)
            neighborhood = memory.gather(1, indices_exp)
            context = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)

            context_scale = torch.sigmoid(self.context_strength)
            combined = inp + context_scale * context

            h = torch.tanh(self.update(combined + h))
            h = self.norm(h)

            update_broadcast = h.unsqueeze(1).expand(-1, weights.size(1), -1)
            contribution = weights.unsqueeze(-1) * update_broadcast
            memory = memory.scatter_add(1, indices_exp, contribution)

            # LEARNED pointer movement
            ptr_int = pointer.long().clamp(0, 63)
            jump_target = self.jump_targets[ptr_int]
            jump_score = self.jump_score_head(h).squeeze(-1)
            jump_prob = torch.sigmoid(jump_score)

            walk_position = (pointer + 1.0) % 64
            new_position = (1 - jump_prob) * walk_position + jump_prob * jump_target

            pointer = new_position % 64

        return self.output(h)

torch.manual_seed(42)
train_and_test(LearnedJump(), "LearnedJump")
print()


# ==============================================================================
# TEST 3: Add INERTIA (momentum)
# ==============================================================================
print("[TEST 3] Add INERTIA (momentum=0.6)")
print("-"*70)

class WithInertia(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 256)
        self.update = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256)
        self.output = nn.Linear(256, 10)
        self.context_strength = nn.Parameter(torch.tensor(0.2))

        self.jump_targets = nn.Parameter(torch.rand(64) * 64)
        self.jump_score_head = nn.Linear(256, 1)
        self.pointer_momentum = 0.6  # Inertia

    def _gaussian_attention(self, pointer, num_positions):
        B = pointer.size(0)
        K = 2
        tau = 8.0
        offsets = torch.arange(-K, K + 1, device=pointer.device)
        base = torch.floor(pointer).long().clamp(0, num_positions - 1)
        indices = (base.unsqueeze(1) + offsets.unsqueeze(0)) % num_positions
        indices_float = indices.float()
        delta = indices_float - pointer.unsqueeze(1)
        logits = -(delta ** 2) / tau
        weights = torch.softmax(logits, dim=1)
        return indices, weights

    def _circular_distance(self, a, b, num_positions):
        half = num_positions / 2.0
        return torch.remainder(b - a + half, num_positions) - half

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, 256)
        memory = torch.zeros(B, 64, 256)
        pointer = torch.empty(B).uniform_(0, 64)

        for t in range(T):
            inp = torch.tanh(self.embed(x[:, t, :]))

            indices, weights = self._gaussian_attention(pointer, 64)
            indices_exp = indices.unsqueeze(-1).expand(-1, -1, 256)
            neighborhood = memory.gather(1, indices_exp)
            context = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)

            context_scale = torch.sigmoid(self.context_strength)
            combined = inp + context_scale * context

            h = torch.tanh(self.update(combined + h))
            h = self.norm(h)

            update_broadcast = h.unsqueeze(1).expand(-1, weights.size(1), -1)
            contribution = weights.unsqueeze(-1) * update_broadcast
            memory = memory.scatter_add(1, indices_exp, contribution)

            # Pointer with jump
            ptr_int = pointer.long().clamp(0, 63)
            jump_target = self.jump_targets[ptr_int]
            jump_score = self.jump_score_head(h).squeeze(-1)
            jump_prob = torch.sigmoid(jump_score)

            walk_position = (pointer + 1.0) % 64
            new_position = (1 - jump_prob) * walk_position + jump_prob * jump_target

            # Apply INERTIA
            new_position = (
                (1 - self.pointer_momentum) * new_position +
                self.pointer_momentum * pointer
            )

            pointer = new_position % 64

        return self.output(h)

torch.manual_seed(42)
train_and_test(WithInertia(), "WithInertia")
print()


print("="*70)
print("SUMMARY")
print("="*70)
print("Which pointer feature breaks learning?")
print("="*70)
