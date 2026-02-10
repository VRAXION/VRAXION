"""
Test Gaussian attention and scatter-add writes.

We know pointer walking works (100% accuracy).
Now test if Gaussian attention breaks it.
"""

import torch
import torch.nn as nn

print("="*70)
print("GAUSSIAN ATTENTION TEST")
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
# BASELINE: Walking pointer (we know this works)
# ==============================================================================
print("[BASELINE] Walking pointer (single position read/write)")
print("-"*70)

class WalkingPointer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 256)
        self.update = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256)
        self.output = nn.Linear(256, 10)
        self.context_strength = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, 256)
        memory = torch.zeros(B, 64, 256)
        pointer = torch.zeros(B)

        for t in range(T):
            inp = torch.tanh(self.embed(x[:, t, :]))

            # Read single position
            ptr_int = pointer.long()
            context = memory[torch.arange(B), ptr_int, :]
            context_scale = torch.sigmoid(self.context_strength)
            combined = inp + context_scale * context

            h = torch.tanh(self.update(combined + h))
            h = self.norm(h)

            # Write single position
            memory = memory.clone()
            memory[torch.arange(B), ptr_int, :] = h

            # Walk
            pointer = (pointer + 1.0) % 64

        return self.output(h)

torch.manual_seed(42)
train_and_test(WalkingPointer(), "WalkingPointer")
print()


# ==============================================================================
# TEST 1: Gaussian READ (soft gather from neighborhood)
# ==============================================================================
print("[TEST 1] Gaussian READ (scatter write still single position)")
print("-"*70)

class GaussianRead(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 256)
        self.update = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256)
        self.output = nn.Linear(256, 10)
        self.context_strength = nn.Parameter(torch.tensor(0.2))
        self.attention_radius = 2
        self.attention_temperature = 8.0

    def _gaussian_attention(self, pointer, num_positions):
        """Compute Gaussian attention weights."""
        B = pointer.size(0)
        K = self.attention_radius

        # Offsets: [-K, ..., 0, ..., +K]
        offsets = torch.arange(-K, K + 1, device=pointer.device)

        # Indices
        base = torch.floor(pointer).long().clamp(0, num_positions - 1)
        indices = (base.unsqueeze(1) + offsets.unsqueeze(0)) % num_positions
        indices_float = indices.float()

        # Distance (simple, not circular)
        delta = indices_float - pointer.unsqueeze(1)

        # Gaussian weights
        logits = -(delta ** 2) / self.attention_temperature
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

            # Single position WRITE (for now)
            memory = memory.clone()
            ptr_int = pointer.long()
            memory[torch.arange(B), ptr_int, :] = h

            # Walk
            pointer = (pointer + 1.0) % 64

        return self.output(h)

torch.manual_seed(42)
train_and_test(GaussianRead(), "GaussianRead")
print()


# ==============================================================================
# TEST 2: Gaussian WRITE (scatter-add to neighborhood)
# ==============================================================================
print("[TEST 2] Gaussian WRITE (scatter-add to neighborhood)")
print("-"*70)

class GaussianWrite(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 256)
        self.update = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256)
        self.output = nn.Linear(256, 10)
        self.context_strength = nn.Parameter(torch.tensor(0.2))
        self.attention_radius = 2
        self.attention_temperature = 8.0

    def _gaussian_attention(self, pointer, num_positions):
        B = pointer.size(0)
        K = self.attention_radius
        offsets = torch.arange(-K, K + 1, device=pointer.device)
        base = torch.floor(pointer).long().clamp(0, num_positions - 1)
        indices = (base.unsqueeze(1) + offsets.unsqueeze(0)) % num_positions
        indices_float = indices.float()
        delta = indices_float - pointer.unsqueeze(1)
        logits = -(delta ** 2) / self.attention_temperature
        weights = torch.softmax(logits, dim=1)
        return indices, weights

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, 256)
        memory = torch.zeros(B, 64, 256)
        pointer = torch.zeros(B)

        for t in range(T):
            inp = torch.tanh(self.embed(x[:, t, :]))

            # Single position READ (for now)
            ptr_int = pointer.long()
            context = memory[torch.arange(B), ptr_int, :]

            context_scale = torch.sigmoid(self.context_strength)
            combined = inp + context_scale * context

            h = torch.tanh(self.update(combined + h))
            h = self.norm(h)

            # Gaussian WRITE (scatter-add)
            indices, weights = self._gaussian_attention(pointer, 64)
            indices_exp = indices.unsqueeze(-1).expand(-1, -1, 256)
            update_broadcast = h.unsqueeze(1).expand(-1, weights.size(1), -1)
            contribution = weights.unsqueeze(-1) * update_broadcast
            memory = memory.scatter_add(1, indices_exp, contribution)

            # Walk
            pointer = (pointer + 1.0) % 64

        return self.output(h)

torch.manual_seed(42)
train_and_test(GaussianWrite(), "GaussianWrite")
print()


# ==============================================================================
# TEST 3: BOTH Gaussian read AND write
# ==============================================================================
print("[TEST 3] Gaussian READ + WRITE (full attention)")
print("-"*70)

class GaussianBoth(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 256)
        self.update = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256)
        self.output = nn.Linear(256, 10)
        self.context_strength = nn.Parameter(torch.tensor(0.2))
        self.attention_radius = 2
        self.attention_temperature = 8.0

    def _gaussian_attention(self, pointer, num_positions):
        B = pointer.size(0)
        K = self.attention_radius
        offsets = torch.arange(-K, K + 1, device=pointer.device)
        base = torch.floor(pointer).long().clamp(0, num_positions - 1)
        indices = (base.unsqueeze(1) + offsets.unsqueeze(0)) % num_positions
        indices_float = indices.float()
        delta = indices_float - pointer.unsqueeze(1)
        logits = -(delta ** 2) / self.attention_temperature
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

            # Walk
            pointer = (pointer + 1.0) % 64

        return self.output(h)

torch.manual_seed(42)
train_and_test(GaussianBoth(), "GaussianBoth")
print()


print("="*70)
print("SUMMARY")
print("="*70)
print("Does Gaussian attention break learning?")
print("="*70)
