"""
Diagnostic Probes - Isolate which component is broken.

Test each component independently:
1. Baseline: Can model learn WITHOUT any ring memory?
2. Fixed pointer: Can it learn if pointer doesn't move?
3. No context: Can it learn without reading from memory?
4. No writes: Can it learn without writing to memory?
5. Pointer movement: Is pointer actually moving?
6. Memory updates: Is memory being written to?
7. Gradient flow: Which gradients are dead?
"""

import torch
import sys
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Diamond Code")
from ring_memory_model import RingMemoryModel

# Common setup
torch.manual_seed(42)
x = torch.randint(0, 10, (100, 16, 1)).float()
y = x[:, -1, 0].long()  # Predict last token

print("="*70)
print("DIAGNOSTIC PROBES - Component Isolation")
print("="*70)
print(f"Task: COPY (predict last token)")
print(f"Data: {x.shape}, Labels: {y.shape}, Classes: 10")
print(f"Random baseline: ~10%")
print("="*70)
print()


# ==============================================================================
# PROBE 1: Baseline - No ring memory at all
# ==============================================================================
print("[PROBE 1] Baseline: Input -> Output (no ring, no memory)")
print("-"*70)

class BaselineModel(torch.nn.Module):
    """Just input projection → output. No memory, no recurrence."""
    def __init__(self, input_size, num_outputs, embedding_dim=64):
        super().__init__()
        self.input_projection = torch.nn.Linear(input_size, embedding_dim)
        self.output_head = torch.nn.Linear(embedding_dim, num_outputs)

    def forward(self, x):
        # Just use last timestep
        input_embed = torch.tanh(self.input_projection(x[:, -1, :]))
        logits = self.output_head(input_embed)
        return logits

torch.manual_seed(42)
baseline = BaselineModel(input_size=1, num_outputs=10)
optimizer = torch.optim.Adam(baseline.parameters(), lr=0.001)

max_acc = 0.0
for step in range(100):
    optimizer.zero_grad()
    logits = baseline(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == y).float().mean().item()

    if acc > max_acc:
        max_acc = acc

    if step % 20 == 0:
        print(f"  Step {step:3d}: loss={loss.item():.4f}, acc={acc*100:5.1f}%")

print(f"  Best: {max_acc*100:.1f}%")
print(f"  Result: {'PASS' if max_acc > 0.90 else 'FAIL'} (target >90%)")
print()


# ==============================================================================
# PROBE 2: Fixed pointer - No pointer movement
# ==============================================================================
print("[PROBE 2] Fixed Pointer: Ring memory with pointer stuck at 0")
print("-"*70)

class FixedPointerModel(torch.nn.Module):
    """Ring memory but pointer is always at position 0."""
    def __init__(self, input_size, num_outputs, num_positions=64, embedding_dim=64):
        super().__init__()
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim

        self.input_projection = torch.nn.Linear(input_size, embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        self.output_head = torch.nn.Linear(embedding_dim, num_outputs)
        self.context_strength = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        B, T, _ = x.shape

        memory_ring = torch.zeros(B, self.num_positions, self.embedding_dim)
        hidden_state = torch.zeros(B, self.embedding_dim)

        for t in range(T):
            # Input
            input_embed = torch.tanh(self.input_projection(x[:, t, :]))

            # Read from position 0 only
            context_read = memory_ring[:, 0, :]

            # Mix
            context_scale = torch.sigmoid(self.context_strength)
            combined = input_embed + context_scale * context_read

            # Update
            state_update = torch.tanh(combined + hidden_state)
            hidden_state = self.layer_norm(state_update)

            # Write back to position 0 (non-inplace)
            memory_ring = memory_ring.clone()
            memory_ring[:, 0, :] = state_update

        logits = self.output_head(hidden_state)
        return logits

torch.manual_seed(42)
fixed_model = FixedPointerModel(input_size=1, num_outputs=10)
optimizer = torch.optim.Adam(fixed_model.parameters(), lr=0.001)

max_acc = 0.0
for step in range(100):
    optimizer.zero_grad()
    logits = fixed_model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == y).float().mean().item()

    if acc > max_acc:
        max_acc = acc

    if step % 20 == 0:
        print(f"  Step {step:3d}: loss={loss.item():.4f}, acc={acc*100:5.1f}%")

print(f"  Best: {max_acc*100:.1f}%")
print(f"  Result: {'PASS' if max_acc > 0.90 else 'FAIL'} (target >90%)")
print()


# ==============================================================================
# PROBE 3: No context reads - Only writes to memory
# ==============================================================================
print("[PROBE 3] No Context: Model doesn't read from memory")
print("-"*70)

class NoReadModel(torch.nn.Module):
    """Ring memory but context_read is always zero."""
    def __init__(self, input_size, num_outputs, num_positions=64, embedding_dim=64):
        super().__init__()
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim

        self.input_projection = torch.nn.Linear(input_size, embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        self.output_head = torch.nn.Linear(embedding_dim, num_outputs)

    def forward(self, x):
        B, T, _ = x.shape

        memory_ring = torch.zeros(B, self.num_positions, self.embedding_dim)
        hidden_state = torch.zeros(B, self.embedding_dim)

        for t in range(T):
            # Input only (no context read)
            input_embed = torch.tanh(self.input_projection(x[:, t, :]))

            # Update
            state_update = torch.tanh(input_embed + hidden_state)
            hidden_state = self.layer_norm(state_update)

            # Write to memory (doesn't matter where, not used)
            memory_ring[:, 0, :] = state_update

        logits = self.output_head(hidden_state)
        return logits

torch.manual_seed(42)
noread_model = NoReadModel(input_size=1, num_outputs=10)
optimizer = torch.optim.Adam(noread_model.parameters(), lr=0.001)

max_acc = 0.0
for step in range(100):
    optimizer.zero_grad()
    logits = noread_model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == y).float().mean().item()

    if acc > max_acc:
        max_acc = acc

    if step % 20 == 0:
        print(f"  Step {step:3d}: loss={loss.item():.4f}, acc={acc*100:5.1f}%")

print(f"  Best: {max_acc*100:.1f}%")
print(f"  Result: {'PASS' if max_acc > 0.90 else 'FAIL'} (target >90%)")
print()


# ==============================================================================
# PROBE 4: Check if pointer moves in full model
# ==============================================================================
print("[PROBE 4] Pointer Movement: Does pointer position change?")
print("-"*70)

torch.manual_seed(42)
model = RingMemoryModel(input_size=1, num_outputs=10, num_memory_positions=64)

# Single forward pass
with torch.no_grad():
    logits, aux_loss, debug = model(x[:4], return_debug=True)

    trajectory = [p.mean().item() for p in debug["pointer_trajectory"]]

    print(f"  Pointer trajectory (mean across batch):")
    print(f"    Start:  {trajectory[0]:.2f}")
    print(f"    Step 5: {trajectory[5]:.2f}")
    print(f"    Step 10: {trajectory[10]:.2f}")
    print(f"    End:    {trajectory[-1]:.2f}")

    total_movement = sum(abs(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1))
    print(f"  Total movement: {total_movement:.2f}")

    if total_movement < 0.1:
        print(f"  Result: FAIL - Pointer barely moves (< 0.1)")
    else:
        print(f"  Result: PASS - Pointer is moving")

print()


# ==============================================================================
# PROBE 5: Check if memory is being updated
# ==============================================================================
print("[PROBE 5] Memory Updates: Is memory ring being written to?")
print("-"*70)

torch.manual_seed(42)
model = RingMemoryModel(input_size=1, num_outputs=10, num_memory_positions=64)

# Need to trace through forward pass to capture memory state
# Use hooks to capture memory_ring

memory_states = []

def trace_forward(model, x):
    """Manually trace forward to capture memory states."""
    B, T, _ = x.shape

    memory_ring = torch.zeros(B, model.num_memory_positions, model.embedding_dim)
    hidden_state = torch.zeros(B, model.embedding_dim)
    pointer_position = torch.empty(B).uniform_(0, model.num_memory_positions)

    memory_norms = []

    for t in range(T):
        # Abbreviated forward (just enough to update memory)
        input_embed = model._activate(model.input_projection(x[:, t, :]))

        indices, weights = model._gaussian_attention_weights(
            pointer_position, model.num_memory_positions
        )

        indices_exp = indices.unsqueeze(-1).expand(-1, -1, model.embedding_dim)
        neighborhood = memory_ring.gather(1, indices_exp)
        context_read = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)

        context_scale = torch.sigmoid(model.context_strength)
        combined_input = input_embed + context_scale * context_read

        state_update = model._activate(combined_input + hidden_state)
        hidden_state = state_update

        if model.layer_norm is not None:
            hidden_state = model.layer_norm(hidden_state)

        # Write
        update_broadcast = state_update.unsqueeze(1).expand(-1, weights.size(1), -1)
        contribution = weights.unsqueeze(-1) * update_broadcast
        memory_ring = memory_ring.scatter_add(1, indices_exp, contribution)

        # Track memory magnitude
        memory_norms.append(memory_ring.norm().item())

        # Pointer update (simplified)
        pointer_position = (pointer_position + 1.0) % model.num_memory_positions

    return memory_norms

with torch.no_grad():
    norms = trace_forward(model, x[:4])

    print(f"  Memory ring norm over time:")
    print(f"    Step 0:  {norms[0]:.4f}")
    print(f"    Step 5:  {norms[5]:.4f}")
    print(f"    Step 10: {norms[10]:.4f}")
    print(f"    End:     {norms[-1]:.4f}")

    if norms[-1] < 0.01:
        print(f"  Result: FAIL - Memory is essentially empty")
    else:
        print(f"  Result: PASS - Memory is being updated")

print()
print("="*70)
print("SUMMARY")
print("="*70)
print("Which components can learn the COPY task?")
print("This will tell us where the problem is.")
print("="*70)
