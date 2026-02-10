"""Test if simple walk reaches >90% with more training steps."""

import torch
import sys
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Diamond Code")
from ring_memory_model import RingMemoryModel

print("="*70)
print("SIMPLE WALK - EXTENDED TRAINING")
print("="*70)
print()

# Data
torch.manual_seed(42)
x = torch.randint(0, 10, (100, 16, 1)).float()
y = x[:, -1, 0].long()

# Force simple walk version
class ForceWalkRingMemory(RingMemoryModel):
    def forward(self, x, return_debug=False):
        B, T, _ = x.shape

        memory_ring = torch.zeros(
            B, self.num_memory_positions, self.embedding_dim,
            device=x.device, dtype=x.dtype
        )
        hidden_state = torch.zeros(B, self.embedding_dim, device=x.device, dtype=x.dtype)
        pointer_position = torch.empty(B, device=x.device).uniform_(0, self.num_memory_positions)

        debug_info = {"pointer_trajectory": [], "attention_entropy": []} if return_debug else None

        for t in range(T):
            input_embed = self.input_projection(x[:, t, :])
            input_embed = self._activate(input_embed)

            indices, weights = self._gaussian_attention_weights(
                pointer_position,
                self.num_memory_positions
            )

            indices_exp = indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
            neighborhood = memory_ring.gather(1, indices_exp)
            context_read = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)

            context_scale = torch.sigmoid(self.context_strength)
            combined_input = input_embed + context_scale * context_read

            state_update = self._activate(combined_input + hidden_state)
            hidden_state = state_update

            if self.layer_norm is not None:
                hidden_state = self.layer_norm(hidden_state)

            update_broadcast = state_update.unsqueeze(1).expand(-1, weights.size(1), -1)
            contribution = weights.unsqueeze(-1) * update_broadcast
            memory_ring = memory_ring.scatter_add(1, indices_exp, contribution)

            # FORCE SIMPLE WALK
            pointer_position = (pointer_position + 1.0) % self.num_memory_positions

            if return_debug:
                debug_info["pointer_trajectory"].append(pointer_position.detach().cpu())
                entropy = -(weights * torch.log(weights + 1e-9)).sum(dim=1).mean()
                debug_info["attention_entropy"].append(entropy.item())

        logits = self.output_head(hidden_state)
        aux_loss = 0.0

        if return_debug:
            return logits, aux_loss, debug_info
        return logits, aux_loss, None


torch.manual_seed(42)
model = ForceWalkRingMemory(
    input_size=1,
    num_outputs=10,
    num_memory_positions=64,
    embedding_dim=256,
    pointer_momentum=0.0,
    movement_threshold=0.0,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

max_acc = 0.0
max_acc_step = 0

print("Training for 1000 steps...")
print()

for step in range(1000):
    optimizer.zero_grad()
    logits, aux_loss, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y) + aux_loss
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == y).float().mean().item()

    if acc > max_acc:
        max_acc = acc
        max_acc_step = step

    if step % 100 == 0:
        print(f"Step {step:4d}: loss={loss.item():.4f}, acc={acc*100:5.1f}%")

print()
print("="*70)
print(f"Best accuracy: {max_acc*100:.1f}% at step {max_acc_step}")

if max_acc > 0.90:
    print("[PASS] Simple walk reaches >90%!")
else:
    print(f"[FAIL] Only reached {max_acc*100:.1f}%, target was >90%")

print("="*70)
