# Gradient Triangulation for Expert Write Weighting

## The Setup

We have a ring-buffer pointer network (INSTNCT) where N experts share a circular memory buffer. Each timestep, every expert:
1. **Reads** from the ring at its pointer position
2. **Thinks** (updates its hidden state)
3. **Writes** back to the ring (additive, via `scatter_add`)
4. **Moves** its pointer (phi-jump or +1 walk)

Currently, all expert writes are simply summed into the ring with equal weight. Experts have no awareness of each other — their gradients carry no information about what other experts wrote.

## The Problem

During the forward pass, expert writes are committed immediately. The loss (and therefore the optimal write value) is only known after the forward pass completes. Each expert writes "blind" — it doesn't know if another expert is about to write something complementary or contradictory to the same ring slot.

With simple additive writes:
- If Expert₀ writes +5 and Expert₁ writes -5, the net effect is 0 — as if nothing happened, even though both experts had strong (opposing) opinions.
- The gradient for both experts is identical (=1), regardless of what the other wrote. They learn independently, blind to each other.

## The Idea: Gradient Triangulation with 1-Frame Delay

**Core insight:** After backprop, we know the gradient at each expert's write point. The gradient tells us the direction and magnitude of error — how far each expert was from the optimal write value. With 3+ experts writing to overlapping ring positions, we can **triangulate** the optimal value (the point where gradient = 0).

### How it works:

**Step T (forward pass):**
- Each expert computes its write value and writes to the ring normally
- Each expert also **saves** its write value to a side buffer (detached, negligible memory)

**Step T (backward pass):**
- Gradients flow back through the computation graph
- We now have, for each expert: (write_value, gradient_at_write)
- Small |gradient| → expert was close to optimal
- Large |gradient| → expert was far from optimal

**Triangulation (after backward):**
Using the secant method / linear interpolation across expert positions:

For experts at positions x₀, x₁, x₂ with gradients g₀, g₁, g₂, the zero-crossing (optimal point D) can be estimated. In 1D with 2 points:

```
D ≈ x₀ - g₀ × (x₁ - x₀) / (g₁ - g₀)
```

With 3+ points, we can fit a better estimate (least-squares on the gradient field). Since each ring slot is a vector (slot_dim=32), this is 32 independent 1D triangulations.

**Step T+1 (next forward pass):**
- We now have a **triangulated estimate of D** from the previous step
- Expert writes are weighted by inverse gradient magnitude:
  - Expert with small gradient (was close to D) → high weight
  - Expert with large gradient (was far from D) → low weight
- Alternatively: use the triangulated D directly as a correction target

### The 1-frame delay is acceptable because:
- Consecutive training steps have similar data distributions
- The loss landscape changes slowly relative to individual steps
- An exponential moving average (EMA) of the triangulated D smooths out noise

## Why This Could Work

1. **Zero learnable parameters added.** The weighting comes from observed gradient magnitudes — pure measurement, not learned. No risk of overfitting the weighting mechanism.

2. **Chicken-and-egg problem solved.** We don't need to know the optimal write during forward pass. We use the *previous* step's gradient information, which is a good proxy.

3. **More experts = better triangulation.** With N=2, we get a rough interpolation. With N=3, proper triangulation. With N=4+, overdetermined system → robust least-squares estimate. This provides a principled reason to increase N beyond "more capacity."

4. **Experts become aware of each other.** Currently, Expert₀'s gradient carries zero information about Expert₁. With triangulation-based weighting, the confidence assigned to each expert depends on the *relative* performance of all experts. This creates implicit coordination without explicit communication.

5. **Negligible overhead.** Saving N vectors + computing a softmax over gradient magnitudes is effectively free compared to the T×N forward loop.

6. **Compatible with existing architecture.** No changes to model structure, loss function, or state_dict. The triangulation is a training-time optimization that can be disabled at inference (equal weights).

## Honest Assessment

**What's genuinely novel:** The idea of using multi-expert gradient information to triangulate the optimal write target in a ring-buffer architecture, then feeding that back with 1-frame delay. This specific combination — ring buffer + multiple experts + gradient triangulation — hasn't been explored to my knowledge.

**What's related in existing literature:**
- **Mixture of Experts (MoE)** routing uses load-balancing losses and learned routers — but the routing is learned, not derived from gradient geometry.
- **Secant method / Newton's method** for root-finding is classical optimization — we're applying it to find gradient-zero in the write-value space.
- **Exponential moving average of gradient statistics** is used in Adam, RMSprop, etc. — but applied to parameter updates, not to expert weighting.
- **Multi-agent credit assignment** in RL — similar problem of attributing outcomes to individual agents, but different solution mechanisms.

**Risks and unknowns:**
- The 1-frame delay could be too stale in early training when the loss landscape changes rapidly. Mitigation: use EMA with adaptive decay.
- The linear interpolation (secant method) assumes locally linear gradients (quadratic loss surface). If the loss is highly non-convex locally, the triangulation could be inaccurate. Mitigation: use it for weighting (soft influence) rather than hard targeting.
- "Rich get richer" dynamics: the best-performing expert gets the most weight, gets the cleanest gradient signal, improves fastest. The weakest expert could get permanently silenced. Mitigation: temperature-controlled softmax ensuring minimum weight for all experts, or periodic weight resets.

## Proposed Experiment

1. Start with N=3 or N=4 experts (currently N=2)
2. Implement gradient magnitude tracking with EMA (decay=0.95)
3. Weight expert writes by `softmax(-ema_gradient_magnitudes / temperature)`
4. Compare against baseline (equal weights) on same training data
5. Measure: convergence speed, final loss, expert utilization (are all experts contributing?)

## Implementation Complexity

Minimal — estimated ~30 lines of code change:
- Add `self._write_grad_ema` buffer (N floats)
- Register backward hook or use `retain_grad()` on write tensors
- Compute weighted writes using softmax of inverse gradient magnitudes
- No changes to model architecture, checkpointing, or inference
