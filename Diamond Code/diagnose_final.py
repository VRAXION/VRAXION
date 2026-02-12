"""
Final diagnostic: Compare EXACT same architecture,
only difference is simple walk vs learned jump.
"""

import torch
import sys
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Diamond Code")
from ring_memory_model import RingMemoryModel

print("="*70)
print("FINAL DIAGNOSTIC: Simple Walk vs Learned Jump")
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
        logits, aux_loss, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y) + aux_loss
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
# Create a MODIFIED RingMemoryModel that forces simple walking
# ==============================================================================
class ForceWalkRingMemory(RingMemoryModel):
    """Same as RingMemoryModel but force simple +1 walk (no learned jumps)."""

    def forward(self, x, return_debug=False):
        B, T, _ = x.shape

        # Initialize state (same as RingMemoryModel)
        memory_ring = torch.zeros(
            B, self.num_memory_positions, self.embedding_dim,
            device=x.device, dtype=x.dtype
        )
        hidden_state = torch.zeros(B, self.embedding_dim, device=x.device, dtype=x.dtype)
        pointer_position = torch.empty(B, device=x.device).uniform_(0, self.num_memory_positions)

        debug_info = {"pointer_trajectory": [], "attention_entropy": []} if return_debug else None

        # Process sequence (same as RingMemoryModel)
        for t in range(T):
            # 1. Project input
            input_embed = self.input_projection(x[:, t, :])
            input_embed = self._activate(input_embed)

            # 2. Read from ring (Gaussian attention)
            indices, weights = self._gaussian_attention_weights(
                pointer_position,
                self.num_memory_positions
            )

            # Gather neighborhood
            indices_exp = indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
            neighborhood = memory_ring.gather(1, indices_exp)

            # Weighted sum
            context_read = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)

            # 3. Context injection
            context_scale = torch.sigmoid(self.context_strength)
            combined_input = input_embed + context_scale * context_read

            # 4. State update
            state_update = self._activate(combined_input + hidden_state)
            hidden_state = state_update

            # Apply normalization if available
            if self.layer_norm is not None:
                hidden_state = self.layer_norm(hidden_state)

            # 5. Write to ring (scatter-add)
            update_broadcast = state_update.unsqueeze(1).expand(-1, weights.size(1), -1)
            contribution = weights.unsqueeze(-1) * update_broadcast
            memory_ring = memory_ring.scatter_add(1, indices_exp, contribution)

            # 6. Pointer update - FORCE SIMPLE WALK
            pointer_position = (pointer_position + 1.0) % self.num_memory_positions

            # Debug recording
            if return_debug:
                debug_info["pointer_trajectory"].append(pointer_position.detach().cpu())
                entropy = -(weights * torch.log(weights + 1e-9)).sum(dim=1).mean()
                debug_info["attention_entropy"].append(entropy.item())

        # 7. Output
        logits = self.output_head(hidden_state)

        # Auxiliary loss
        aux_loss = 0.0

        if return_debug:
            return logits, aux_loss, debug_info
        return logits, aux_loss, None


print("[TEST 1] RingMemoryModel with LEARNED jump/walk (original)")
print("-"*70)

torch.manual_seed(42)
model1 = RingMemoryModel(
    input_size=1,
    num_outputs=10,
    num_memory_positions=64,
    embedding_dim=256,
    pointer_momentum=0.0,  # No inertia for fair comparison
    movement_threshold=0.0,  # No deadzone for fair comparison
)

train_and_test(model1, "Learned-Jump")
print()


print("[TEST 2] RingMemoryModel with FORCED simple walk")
print("-"*70)

torch.manual_seed(42)
model2 = ForceWalkRingMemory(
    input_size=1,
    num_outputs=10,
    num_memory_positions=64,
    embedding_dim=256,
    pointer_momentum=0.0,
    movement_threshold=0.0,
)

train_and_test(model2, "Forced-Walk")
print()


print("="*70)
print("CONCLUSION")
print("="*70)
print("Is the LEARNED jump/walk the problem?")
print("="*70)
