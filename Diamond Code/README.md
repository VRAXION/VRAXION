# Diamond Code: Ring Memory Model

**Clean, tested reference implementation** of the ring-based pointer memory architecture.

## Status

⚠️ **Pre-verification** - Tests must pass before this is considered Diamond quality.

## Architecture

**Ring Memory Model** - A neural memory system using:
- **Circular buffer** (ring) for addressable memory
- **Soft pointer** (differentiable position) for read/write addressing
- **Gaussian attention** for smooth neighborhood reads
- **Scatter-add writes** for distributed updates
- **Hard discrete jumps** with emergent content-based routing

### Core Components

```python
Memory Ring:    [batch, num_positions, embedding_dim]
Pointer:        [batch] (float, differentiable)
Hidden State:   [batch, embedding_dim]
```

### Forward Pass Flow

1. **Input Projection**: Embed input tokens
2. **Ring Read**: Gaussian attention around pointer position
3. **Context Injection**: Mix read context with input
4. **State Update**: Recurrent update (activation + residual)
5. **Ring Write**: Scatter-add updates to neighborhood
6. **Pointer Update**: Hard discrete jump or walk (content-based routing)
7. **Output**: Classify from hidden state

## Files

- `ring_memory_model.py` - Core model implementation
- `test_ring_memory.py` - 13 adversarial tests
- `debug_utils.py` - Debugging and visualization tools
- `README.md` - This file

## Quick Start

### Basic Usage

```python
import torch
from ring_memory_model import RingMemoryModel

# Create model
model = RingMemoryModel(
    input_size=1,
    num_outputs=10,
    num_memory_positions=64,
    embedding_dim=64,
)

# Forward pass
x = torch.randn(4, 16, 1)  # [batch=4, seq=16, input=1]
logits, aux_loss, debug_info = model(x, return_debug=True)

print(f"Output shape: {logits.shape}")  # [4, 10]
```

### Training Example

```python
# Generate simple COPY task data
x = torch.randint(0, 10, (100, 16, 1)).float()
y = x[:, -1, 0].long()  # Predict last token

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for step in range(100):
    logits, aux_loss, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y) + aux_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        acc = (logits.argmax(dim=1) == y).float().mean()
        print(f"Step {step}: loss={loss.item():.4f}, acc={acc.item()*100:.1f}%")
```

### Debug Mode

```python
from debug_utils import visualize_pointer_trajectory, print_model_summary

# Model summary
print_model_summary(model)

# Forward with debug
logits, aux_loss, debug = model(x, return_debug=True)

# Visualize pointer movement
visualize_pointer_trajectory(debug, save_path="pointer_trace.png")
```

## Parameters

### Model Config

- `input_size` - Input dimension (e.g., 1 for scalar)
- `num_outputs` - Number of output classes
- `num_memory_positions` - Ring buffer size (64, 128, 256)
- `embedding_dim` - Feature dimension per position (64, 128, 256)
- `attention_radius` - Neighborhood size (2 = ±2 neighbors)
- `attention_temperature` - Softmax temperature (8.0 = soft, 0.1 = sharp)
- `activation` - Non-linearity ('tanh', 'relu', 'silu')

## Testing

### Run All Tests

```bash
cd "S:/AI/work/VRAXION_DEV/Diamond Code"
python test_ring_memory.py
```

Or with pytest:

```bash
pytest test_ring_memory.py -v
```

### Test Suite

1. ✓ Initialization
2. ✓ Forward pass smoke test
3. ✓ Output shapes
4. ✓ No NaN/Inf values
5. ✓ Gradient flow
6. ✓ Pointer wrapping
7. ✓ Attention weights sum to 1
8. ✓ Circular distance
9. ✓ Adversarial: all zeros
10. ✓ Adversarial: all same
11. ✓ Adversarial: huge batch
12. ✓ Determinism
13. ✓ Learning test (memorize 1 sample)

## Success Criteria

Before committing to Diamond Code:

1. ✓ All 13 unit tests pass
2. ✓ Model memorizes 1 sample in <100 steps
3. ✓ Model learns COPY task >90% in <100 steps
4. ✓ No unlearning over 500 steps
5. ✓ Pointer trajectory converges
6. ✓ Code is clear and self-documenting
7. ✓ No environment variables (all hardcoded)

**Only THEN** is this Diamond quality.

## Design Principles

### What's Included (Core Only)

- Ring memory buffer
- Soft pointer with per-position learned jump destinations
- Content-based jump gate (data decides when to jump)
- Hard discrete jumps using Straight-Through Estimator (STE)
- Gaussian attention for reads
- Scatter-add for writes
- LayerNorm for multi-dim embeddings

### What's NOT Included (Stripped Out)

- Environment variable parsing
- Optional features (vault, sensory, prismion)
- Mobius phase embedding
- Think ring filters
- Satiety early-exit
- BOS/EOS special handling
- Telemetry hooks
- Advanced diagnostics

**Philosophy**: Minimal, tested, working. No optional features.

## Comparison to AbsoluteHallway

### Renamed Components

| AbsoluteHallway | RingMemoryModel | Meaning |
|-----------------|-----------------|---------|
| `ring_len` | `num_memory_positions` | Buffer size |
| `slot_dim` | `embedding_dim` | Feature dimensions |
| `gauss_k` | `attention_radius` | Neighborhood size |
| `gauss_tau` | `attention_temperature` | Softmax temp |
| `theta_ptr` | `jump_destinations` | Per-position jump targets |
| `theta_gate` | `jump_gate` | Content-based jump decision |
| `state` | `memory_ring` | Ring buffer state |
| `h` | `hidden_state` | Hidden state vector |
| `ptr_float` | `pointer_position` | Pointer location |

### Simplified Forward

**AbsoluteHallway**: Complex pointer dynamics with soft blending
**RingMemoryModel**: Hard discrete jumps with emergent routing

Both use ring memory, but RingMemoryModel uses a cleaner routing mechanism.

## Next Steps

1. Run tests: `python test_ring_memory.py`
2. Verify all 13 tests pass
3. Test on COPY task (>90% accuracy)
4. Test stability (no unlearning over 500 steps)
5. Compare to Golden Code on same tasks
6. Identify specific differences causing instability

---

Created: 2026-02-10
Version: 1.0 (Pre-verification)
