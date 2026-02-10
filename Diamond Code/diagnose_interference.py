"""
Test if processing the full sequence is causing interference.

Hypothesis: Ring memory accumulates info from ALL timesteps,
which confuses the model when trying to predict the last token.
"""

import torch
import sys
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Diamond Code")

print("="*70)
print("INTERFERENCE DIAGNOSIS")
print("="*70)
print()

# Generate data
torch.manual_seed(42)
x = torch.randint(0, 10, (100, 16, 1)).float()
y = x[:, -1, 0].long()

print(f"Task: Predict last token from sequence")
print(f"Data: {x.shape}, Labels: {y.shape}")
print()


# ==============================================================================
# TEST 1: Simple model - only process last token
# ==============================================================================
print("[TEST 1] Feedforward: ONLY process last token")
print("-"*70)

class LastTokenOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Linear(1, 256)
        self.hidden = torch.nn.Linear(256, 256)
        self.output = torch.nn.Linear(256, 10)

    def forward(self, x):
        # ONLY use last token
        h = torch.tanh(self.embed(x[:, -1, :]))
        h = torch.tanh(self.hidden(h))
        return self.output(h)

torch.manual_seed(42)
model = LastTokenOnly()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

max_acc = 0.0
for step in range(500):
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

print(f"  Best: {max_acc*100:.1f}%")
print()


# ==============================================================================
# TEST 2: Recurrent - process ALL tokens
# ==============================================================================
print("[TEST 2] Recurrent: Process ALL tokens sequentially")
print("-"*70)

class RecurrentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Linear(1, 256)
        self.hidden_update = torch.nn.Linear(256, 256)
        self.output = torch.nn.Linear(256, 10)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, 256)

        # Process sequence
        for t in range(T):
            inp = torch.tanh(self.embed(x[:, t, :]))
            h = torch.tanh(self.hidden_update(inp + h))

        return self.output(h)

torch.manual_seed(42)
model = RecurrentModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

max_acc = 0.0
for step in range(500):
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

print(f"  Best: {max_acc*100:.1f}%")
print()


# ==============================================================================
# TEST 3: Recurrent with LayerNorm
# ==============================================================================
print("[TEST 3] Recurrent + LayerNorm")
print("-"*70)

class RecurrentNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Linear(1, 256)
        self.hidden_update = torch.nn.Linear(256, 256)
        self.norm = torch.nn.LayerNorm(256)
        self.output = torch.nn.Linear(256, 10)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, 256)

        for t in range(T):
            inp = torch.tanh(self.embed(x[:, t, :]))
            h = torch.tanh(self.hidden_update(inp + h))
            h = self.norm(h)  # Normalize

        return self.output(h)

torch.manual_seed(42)
model = RecurrentNorm()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

max_acc = 0.0
for step in range(500):
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

print(f"  Best: {max_acc*100:.1f}%")
print()


# ==============================================================================
# TEST 4: What if we process sequence in REVERSE?
# ==============================================================================
print("[TEST 4] Recurrent but REVERSE order (start from last token)")
print("-"*70)

class RecurrentReverse(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Linear(1, 256)
        self.hidden_update = torch.nn.Linear(256, 256)
        self.norm = torch.nn.LayerNorm(256)
        self.output = torch.nn.Linear(256, 10)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, 256)

        # Process in REVERSE (T-1, T-2, ..., 0)
        for t in range(T-1, -1, -1):
            inp = torch.tanh(self.embed(x[:, t, :]))
            h = torch.tanh(self.hidden_update(inp + h))
            h = self.norm(h)

        return self.output(h)

torch.manual_seed(42)
model = RecurrentReverse()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

max_acc = 0.0
for step in range(500):
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

print(f"  Best: {max_acc*100:.1f}%")
print()

print("="*70)
print("CONCLUSION")
print("="*70)
print("Does processing the full sequence cause interference?")
print("="*70)
