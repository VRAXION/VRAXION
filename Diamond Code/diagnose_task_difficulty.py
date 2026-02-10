"""
Test if the COPY task is actually learnable.

Maybe the issue is that the task itself is too hard for this model size?
Let's test progressively simpler tasks.
"""

import torch

print("="*70)
print("TASK DIFFICULTY ANALYSIS")
print("="*70)
print()

# ==============================================================================
# TASK 1: Binary classification (easiest)
# ==============================================================================
print("[TASK 1] Binary: All 0s vs All 1s")
print("-"*70)

class BinaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 2)

    def forward(self, x):
        return self.fc(x[:, -1, :])  # Just last token

# Data: sequence of all 0s → class 0, sequence of all 1s → class 1
torch.manual_seed(42)
x_binary = torch.zeros(100, 16, 1)
x_binary[50:] = 1.0  # Half are all-1s
y_binary = torch.zeros(100, dtype=torch.long)
y_binary[50:] = 1  # Half are class 1

model = BinaryModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(50):
    optimizer.zero_grad()
    logits = model(x_binary)
    loss = torch.nn.functional.cross_entropy(logits, y_binary)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    acc = (logits.argmax(dim=1) == y_binary).float().mean().item()

print(f"Final accuracy: {acc*100:.1f}%")
print(f"Result: {'PASS' if acc > 0.90 else 'FAIL'}")
print()


# ==============================================================================
# TASK 2: 3-class (medium)
# ==============================================================================
print("[TASK 2] 3-Class: Constant sequence -> class")
print("-"*70)

class ThreeClassModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 3)

    def forward(self, x):
        return self.fc(x[:, -1, :])

# Data: all 0s → 0, all 1s → 1, all 2s → 2
torch.manual_seed(42)
x_three = torch.zeros(90, 16, 1)
x_three[30:60] = 1.0
x_three[60:] = 2.0
y_three = torch.zeros(90, dtype=torch.long)
y_three[30:60] = 1
y_three[60:] = 2

model = ThreeClassModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

max_acc = 0.0
for step in range(100):
    optimizer.zero_grad()
    logits = model(x_three)
    loss = torch.nn.functional.cross_entropy(logits, y_three)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == y_three).float().mean().item()
    if acc > max_acc:
        max_acc = acc

print(f"Best accuracy: {max_acc*100:.1f}%")
print(f"Result: {'PASS' if max_acc > 0.90 else 'FAIL'}")
print()


# ==============================================================================
# TASK 3: 10-class with embedding (original COPY task)
# ==============================================================================
print("[TASK 3] 10-Class with Embedding: Original COPY")
print("-"*70)

class CopyModel(torch.nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.input_projection = torch.nn.Linear(1, embedding_dim)
        self.output_head = torch.nn.Linear(embedding_dim, 10)

    def forward(self, x):
        embed = torch.tanh(self.input_projection(x[:, -1, :]))
        return self.output_head(embed)

torch.manual_seed(42)
x = torch.randint(0, 10, (100, 16, 1)).float()
y = x[:, -1, 0].long()

model = CopyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

max_acc = 0.0
for step in range(200):
    optimizer.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    if acc > max_acc:
        max_acc = acc

    if step % 50 == 0:
        print(f"  Step {step:3d}: loss={loss.item():.4f}, acc={acc*100:5.1f}%")

print(f"Best accuracy: {max_acc*100:.1f}%")
print(f"Result: {'PASS' if max_acc > 0.90 else 'FAIL'}")
print()


# ==============================================================================
# TASK 4: Same as Task 3 but with MORE training steps
# ==============================================================================
print("[TASK 4] 10-Class with MORE steps (500 instead of 200)")
print("-"*70)

torch.manual_seed(42)
x = torch.randint(0, 10, (100, 16, 1)).float()
y = x[:, -1, 0].long()

model = CopyModel()
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

print(f"Best accuracy: {max_acc*100:.1f}%")
print(f"Result: {'PASS' if max_acc > 0.90 else 'FAIL'}")
print()


# ==============================================================================
# TASK 5: Same data, but LARGER model
# ==============================================================================
print("[TASK 5] 10-Class with BIGGER model (256-dim instead of 64)")
print("-"*70)

class BigCopyModel(torch.nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.input_projection = torch.nn.Linear(1, embedding_dim)
        self.hidden = torch.nn.Linear(embedding_dim, embedding_dim)
        self.output_head = torch.nn.Linear(embedding_dim, 10)

    def forward(self, x):
        embed = torch.tanh(self.input_projection(x[:, -1, :]))
        hidden = torch.tanh(self.hidden(embed))
        return self.output_head(hidden)

torch.manual_seed(42)
x = torch.randint(0, 10, (100, 16, 1)).float()
y = x[:, -1, 0].long()

model = BigCopyModel()
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

print(f"Best accuracy: {max_acc*100:.1f}%")
print(f"Result: {'PASS' if max_acc > 0.90 else 'FAIL'}")
print()

print("="*70)
print("CONCLUSION")
print("="*70)
print("Which tasks are actually learnable with this architecture?")
print("="*70)
