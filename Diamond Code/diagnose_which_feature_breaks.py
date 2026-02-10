"""
Progressive feature addition test.

Start with a WORKING recurrent model (100% accuracy),
then add Ring Memory features ONE AT A TIME to find which breaks learning.
"""

import torch
import torch.nn as nn

print("="*70)
print("PROGRESSIVE FEATURE ADDITION")
print("="*70)
print("Start with working recurrent, add features until it breaks")
print("="*70)
print()

# Data
torch.manual_seed(42)
x = torch.randint(0, 10, (100, 16, 1)).float()
y = x[:, -1, 0].long()


def train_and_test(model, name, steps=300):
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
# BASELINE: Working recurrent model
# ==============================================================================
print("[BASELINE] Simple Recurrent (we know this works)")
print("-"*70)

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 256)
        self.update = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, 256)
        for t in range(T):
            inp = torch.tanh(self.embed(x[:, t, :]))
            h = torch.tanh(self.update(inp + h))
            h = self.norm(h)
        return self.output(h)

torch.manual_seed(42)
train_and_test(Baseline(), "Baseline")
print()


# ==============================================================================
# STEP 1: Add external memory buffer (but don't use it)
# ==============================================================================
print("[STEP 1] Add memory buffer (not used)")
print("-"*70)

class WithMemoryBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 256)
        self.update = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, 256)
        memory = torch.zeros(B, 64, 256)  # Buffer exists but unused

        for t in range(T):
            inp = torch.tanh(self.embed(x[:, t, :]))
            h = torch.tanh(self.update(inp + h))
            h = self.norm(h)
        return self.output(h)

torch.manual_seed(42)
train_and_test(WithMemoryBuffer(), "WithMemoryBuffer")
print()


# ==============================================================================
# STEP 2: Read from memory (fixed position 0)
# ==============================================================================
print("[STEP 2] Read from memory position 0")
print("-"*70)

class WithMemoryRead(nn.Module):
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

        for t in range(T):
            inp = torch.tanh(self.embed(x[:, t, :]))

            # Read from position 0
            context = memory[:, 0, :]
            context_scale = torch.sigmoid(self.context_strength)
            combined = inp + context_scale * context

            h = torch.tanh(self.update(combined + h))
            h = self.norm(h)
        return self.output(h)

torch.manual_seed(42)
train_and_test(WithMemoryRead(), "WithMemoryRead")
print()


# ==============================================================================
# STEP 3: Write to memory (fixed position 0)
# ==============================================================================
print("[STEP 3] Write to memory position 0")
print("-"*70)

class WithMemoryWrite(nn.Module):
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

        for t in range(T):
            inp = torch.tanh(self.embed(x[:, t, :]))

            # Read
            context = memory[:, 0, :]
            context_scale = torch.sigmoid(self.context_strength)
            combined = inp + context_scale * context

            h = torch.tanh(self.update(combined + h))
            h = self.norm(h)

            # Write (CLONE to avoid inplace)
            memory = memory.clone()
            memory[:, 0, :] = h

        return self.output(h)

torch.manual_seed(42)
train_and_test(WithMemoryWrite(), "WithMemoryWrite")
print()


# ==============================================================================
# STEP 4: Add pointer (but keep it fixed at 0)
# ==============================================================================
print("[STEP 4] Add pointer variable (fixed at 0)")
print("-"*70)

class WithPointer(nn.Module):
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
        pointer = torch.zeros(B)  # Fixed at 0

        for t in range(T):
            inp = torch.tanh(self.embed(x[:, t, :]))

            # Read at pointer position (always 0)
            ptr_int = pointer.long()
            context = memory[torch.arange(B), ptr_int, :]
            context_scale = torch.sigmoid(self.context_strength)
            combined = inp + context_scale * context

            h = torch.tanh(self.update(combined + h))
            h = self.norm(h)

            # Write at pointer position
            memory = memory.clone()
            memory[torch.arange(B), ptr_int, :] = h

        return self.output(h)

torch.manual_seed(42)
train_and_test(WithPointer(), "WithPointer")
print()


# ==============================================================================
# STEP 5: Let pointer walk (+1 each step)
# ==============================================================================
print("[STEP 5] Pointer walks (+1 each step)")
print("-"*70)

class WithPointerWalk(nn.Module):
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

            # Read at pointer
            ptr_int = pointer.long()
            context = memory[torch.arange(B), ptr_int, :]
            context_scale = torch.sigmoid(self.context_strength)
            combined = inp + context_scale * context

            h = torch.tanh(self.update(combined + h))
            h = self.norm(h)

            # Write at pointer
            memory = memory.clone()
            memory[torch.arange(B), ptr_int, :] = h

            # Move pointer (+1 with wrap)
            pointer = (pointer + 1.0) % 64

        return self.output(h)

torch.manual_seed(42)
train_and_test(WithPointerWalk(), "WithPointerWalk", steps=500)
print()


print("="*70)
print("SUMMARY")
print("="*70)
print("At which step does learning break?")
print("="*70)
