"""
Test if using pre-norm for writes vs post-norm for output matters.

Hypothesis: RingMemoryModel writes the PRE-norm state_update to memory,
but outputs from the POST-norm hidden_state. Maybe this breaks learning?
"""

import torch
import torch.nn as nn

print("="*70)
print("PRE-NORM vs POST-NORM TEST")
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
# BASELINE: Use same variable for write and output
# ==============================================================================
print("[BASELINE] Use POST-norm for both write and output")
print("-"*70)

class PostNormBoth(nn.Module):
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

            indices, weights = self._gaussian_attention(pointer, 64)
            indices_exp = indices.unsqueeze(-1).expand(-1, -1, 256)
            neighborhood = memory.gather(1, indices_exp)
            context = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)

            context_scale = torch.sigmoid(self.context_strength)
            combined = inp + context_scale * context

            # Update and normalize
            h = torch.tanh(self.update(combined + h))
            h = self.norm(h)  # POST-norm

            # Write POST-norm to memory
            update_broadcast = h.unsqueeze(1).expand(-1, weights.size(1), -1)
            contribution = weights.unsqueeze(-1) * update_broadcast
            memory = memory.scatter_add(1, indices_exp, contribution)

            pointer = (pointer + 1.0) % 64

        # Output from POST-norm
        return self.output(h)

torch.manual_seed(42)
train_and_test(PostNormBoth(), "PostNormBoth")
print()


# ==============================================================================
# TEST: Use PRE-norm for write, POST-norm for output (like RingMemoryModel)
# ==============================================================================
print("[TEST] Use PRE-norm for write, POST-norm for output (RingMemoryModel style)")
print("-"*70)

class PreNormWrite(nn.Module):
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
        hidden_state = torch.zeros(B, 256)
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

            # Compute state_update (PRE-norm)
            state_update = torch.tanh(self.update(combined + hidden_state))

            # Apply norm to hidden_state (POST-norm)
            hidden_state = state_update  # Assignment before norm
            hidden_state = self.norm(hidden_state)

            # Write PRE-norm (state_update) to memory
            update_broadcast = state_update.unsqueeze(1).expand(-1, weights.size(1), -1)
            contribution = weights.unsqueeze(-1) * update_broadcast
            memory = memory.scatter_add(1, indices_exp, contribution)

            pointer = (pointer + 1.0) % 64

        # Output from POST-norm (hidden_state)
        return self.output(hidden_state)

torch.manual_seed(42)
train_and_test(PreNormWrite(), "PreNormWrite")
print()


print("="*70)
print("SUMMARY")
print("="*70)
print("Does using PRE-norm for writes break learning?")
print("="*70)
