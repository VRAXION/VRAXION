# Diamond Code Swarm Combiner — Research Brief

## For: Deep research / second opinion
## Date: 2026-02-12
## Context: VRAXION Diamond Code — swarm byte-prediction model

---

## 1. Architecture Overview

We have a **SwarmByteRingModel** — N independent "beings" (small neural nets) that share a ring memory buffer. Each being:
- Sees a **subset** of the output bits (K out of num_bits) via a binary receptive mask
- Reads/writes to shared ring memory via gaussian-attention pointers
- Outputs logits for its K visible bits, placed into a full-width vector (zeros elsewhere)

A **combiner** aggregates N being outputs into one final prediction.

### Current Configuration
- N = 16 beings
- num_bits = 64
- K = 40 bits per being (each being covers 40/64 = 62.5% of bits)
- Average coverage per bit: ~10 beings
- Embedding dim: 64
- Depth: 2 (ring read/write cycles per timestep)
- Ring memory: 16 positions × 64 dims
- Sequence length: 16 timesteps
- Device: CUDA (RTX 4070 Ti SUPER)

### Mask Generation
Masks are generated via `generate_combinatorial_masks()` — a greedy algorithm that:
1. Assigns each being K random bits
2. Ensures every bit is covered by at least `min_coverage` beings
3. Maximizes diversity (Jaccard distance between masks)

Result: each bit covered by 5-11 beings (min_cov=5, max_cov=11).

---

## 2. The Combiner (Current Implementation)

```python
def _masked_combine(self, being_stack, training):
    # being_stack: [N, B, num_bits] — raw logits from all beings
    # Non-covered bits are 0.0 (logit) by construction

    masks = self.receptive_masks          # [N, num_bits] binary
    probs = torch.sigmoid(being_stack)    # [N, B, num_bits]
    confidence = (probs - 0.5).abs() * 2.0  # [N, B, num_bits] range [0,1]
    mask_exp = masks.unsqueeze(1)         # [N, 1, num_bits]

    # Learned gate (added recently, didn't help)
    gate = torch.sigmoid(self.combiner_gate).unsqueeze(1)  # [N, 1, num_bits]

    if training:
        weights = confidence * mask_exp * gate
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
        p_combined = (weights * probs).sum(dim=0)   # [B, num_bits]
    else:
        # Hard: pick most confident covering being
        conf_masked = confidence * gate
        conf_masked[mask_exp.expand_as(conf_masked) < 0.5] = -1.0
        best_idx = conf_masked.argmax(dim=0)
        p_combined = probs.gather(0, best_idx.unsqueeze(0)).squeeze(0)

    p_combined = p_combined.clamp(1e-6, 1 - 1e-6)
    return torch.log(p_combined / (1 - p_combined))  # back to logit space
```

---

## 3. The Task (Data Format)

`generate_multitask_batch()` produces mixed arithmetic on bit vectors:

```
Input  x[0] = operand A (random integer as 64 bits)
Input  x[1] = operand B (random integer as 64 bits)
Input  x[2] = op_code  (one-hot: add=1, and=2, or=4, xor=8)
Input  x[3..15] = zeros

Target y[0] = A          (copy of input — trivial)
Target y[1] = B          (copy of input — trivial)
Target y[2] = A op B     (the actual computation)
Target y[3..15] = zeros  (trivial)
```

Loss: `BCEWithLogitsLoss(output, target)` averaged over ALL positions and ALL bits.

---

## 4. Observed Behavior (700 steps)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Train loss | 0.125 | Dropping steadily |
| bit_acc (position 2) | 0.50 | **Chance level** — ensemble can't predict math result |
| bit_oracle | 1.0 | Meaningless — 16 random guesses guarantee each bit covered |
| being_accs (all 16) | 0.0000 | No being solves full byte at position 2 |
| per-bit accs (pos 2) | 0.28–0.72 | Random noise around 0.50 |
| oracle (full byte) | 0.0000 | No being solves any complete byte |
| EVAL loss | 0.345 | Much higher than train (eval uses hard top-1 selection) |
| hamming distance | ~31/64 | Chance = 32, so marginally better |

### Loss Breakdown (estimated)
- Positions 0-1 (copy task, 12.5% of loss): nearly solved → ~0
- Position 2 (math task, 6.25% of loss): chance → 0.693
- Positions 3-15 (zero task, 81.25% of loss): nearly solved → ~0
- **Expected total: 0.693 × (1/16) ≈ 0.043** (floor if trivial positions fully solved)
- **Observed: 0.125** → trivial positions not fully solved either

---

## 5. Root Cause Analysis (Our Finding)

### The Confidence Dead Zone

The combiner weight formula:
```
weight_n = |sigmoid(logit_n) - 0.5| × 2.0 × mask_n × gate_n
```

When a being is **uncertain** about a bit (logit ≈ 0, prob ≈ 0.5):
- confidence ≈ 0
- weight ≈ 0
- gradient through this being ≈ 0
- **Being can never learn because it gets no gradient signal**

This creates a **chicken-and-egg problem**: beings need gradient to learn, but they need confidence to get gradient.

**Why it doesn't affect positions 0-1**: The input signal is directly available at those timesteps. Beings quickly develop nonzero confidence on input-echo tasks, breaking the symmetry early.

**Why position 2 is stuck**: Computing A op B requires multi-step reasoning (read A from memory, read B from memory, apply operation). All beings start equally uncertain → all get zero gradient → none can learn.

### Secondary Issue: Loss Dilution
Position 2 is only 1/16 = 6.25% of the total loss. The gradient signal for the math task is 16× weaker than if position 2 were the only target.

### Tertiary Issue: Ring Memory Chaos
16 beings write simultaneously to shared ring memory. At timestep 2, each being reads a ring state that contains overlapping, uncoordinated writes from all 16 beings at timesteps 0 and 1. This makes the read content essentially noise for any individual being.

---

## 6. Proposed Fix (Not Yet Implemented)

Remove the confidence term from training weights. Use only mask and gate:
```python
if training:
    weights = mask_exp * gate  # drop confidence
    weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
    p_combined = (weights * probs).sum(dim=0)
```

This ensures all covering beings get equal gradient regardless of certainty level. The gate (learnable) can still specialize later once beings develop differentiated predictions.

---

## 7. Research Questions

### Q1: Confidence-Weighted Ensembles and Gradient Flow
In mixture-of-experts and ensemble learning literature, is confidence-based weighting known to create gradient dead zones? Specifically: when ensemble member weights are proportional to `|output - 0.5|` (certainty), does this create a bootstrapping failure where uncertain members get zero gradient?

**What we want**: Literature references, known failure modes, and recommended alternatives. Are there confidence-weighted aggregation schemes that maintain gradient flow to uncertain members? For example:
- Temperature-scaled softmax over confidences
- Additive confidence floor (confidence + epsilon)
- Gumbel-softmax for top-k selection
- Straight-through estimators for hard assignment

### Q2: Loss Weighting in Multi-Position Sequence Tasks
Our loss averages over 16 timesteps where 15/16 are trivially solvable and 1/16 is the actual task. This means 93.75% of gradient goes to positions the model can trivially solve.

**What we want**: What is the best practice for handling this? Options we've considered:
- Position-weighted loss (upweight position 2)
- Curriculum: only train on position 2 once positions 0-1 converge
- Teacher forcing vs autoregressive (currently autoregressive-style)
- Masking trivial positions from loss entirely

Are there principled approaches from the transformer/sequence prediction literature?

### Q3: Shared Memory with Simultaneous Writers
Our architecture has N agents writing to the same memory bank simultaneously (scatter_add). This was a deliberate choice for vectorization (old code wrote sequentially).

**What we want**: In multi-agent memory architectures (Neural Turing Machines, DNC, MERLIN, etc.), how is write contention handled? Specifically:
- Does simultaneous writing to shared memory provably degrade learning vs sequential writing?
- Are there known mechanisms for coordinating multi-writer memory? (write gates, attention-based arbitration, slot-based allocation)
- Should each being have its own memory, with a shared read-only view?

### Q4: Mixture of Experts vs Our "Masked Beings" Architecture
Our architecture is similar to Mixture of Experts (MoE) but differs in key ways:
- Each "expert" (being) covers a SUBSET of output bits (not the full output)
- Multiple experts cover each bit (5-11 coverage)
- Routing is FIXED (binary mask) not learned
- Aggregation is confidence-weighted average, not top-k gating

**What we want**:
- In MoE literature, what happens when the gating/routing mechanism prevents gradient flow to underperforming experts? (Expert collapse, dead experts problem)
- Is our fixed-mask approach fundamentally flawed? Should routing be learned (like Switch Transformer)?
- With K=40/64 overlap, each bit has ~10 covering experts. Is this too many? What's the optimal coverage for aggregation?
- Would a sparse MoE approach (top-1 or top-2 per bit) be strictly better?

### Q5: Can This Architecture Learn Multi-Step Arithmetic?
The task requires: read A (step 0), read B (step 1), compute A op B (step 2). Each being has 64-dim hidden state, 64-dim embedding, depth-2 ring interaction.

**What we want**:
- Can a network with 64-dim hidden state learn 64-bit addition/XOR/AND/OR? What's the minimum capacity?
- Does the ring memory mechanism (gaussian attention pointer, read via attention, write via scatter) provide sufficient "scratch space" for multi-step computation?
- Are there known architectures for learning bit-level arithmetic that we should draw from? (Neural ALU, NALU, Neural GPUs, etc.)

### Q6: The Synchronous vs Sequential Write Question
We changed from sequential writes (being 0 writes, then being 1 reads updated memory and writes, etc.) to synchronous writes (all beings read same state, write simultaneously via scatter_add).

**What we want**:
- Is there theoretical or empirical evidence that sequential-dependency in multi-agent memory systems aids learning? (i.e., does the ordering create useful inductive bias?)
- Our vectorized path is ~10× faster but may lose an important sequential inductive bias. Is there a middle ground? (e.g., 2-phase: first half writes, second half reads updated state)

### Q7: Diagnostic Experiments We Should Run
Given all the above, what experiments would most efficiently identify the primary bottleneck?

Candidates:
1. **Single-being baseline**: 1 being, K=64 (full view), same architecture. Can it learn position 2 at all?
2. **Position-2-only loss**: Mask loss to only position 2. Does bit_acc improve?
3. **Uniform weights (no confidence)**: Remove confidence from combiner. Does gradient flow to position 2?
4. **Reduce K to 8**: Minimal overlap (2 beings per bit). Does the averaging problem disappear?
5. **XOR-only task**: Simplest operation (bitwise, no carry propagation). Can the model learn this?
6. **Separate rings**: Each being gets its own ring memory. Does this help?

**What we want**: Which 2-3 experiments should we run FIRST to most efficiently narrow down the bottleneck? And in what order?

---

## 8. Code Pointers

All code is in: `S:/AI/work/VRAXION_DEV/Diamond Code/`

| File | Purpose |
|------|---------|
| `swarm_model.py` | Main model (SwarmByteRingModel), combiner at line 782-825 |
| `test_swarm_config.py` | Training loop, data generation, metrics |
| `diamond_dashboard.py` | Streamlit dashboard for real-time monitoring |
| `byte_data.py` | Data generators (copy, logic, addition tasks) |
| `traindat_loader.py` | Raw byte file loader |

---

## 9. Summary

**What works**: Vectorized forward pass (16 beings in parallel on GPU), beings learn trivial positions, masks and coverage are correct.

**What's broken**: The confidence-weighted combiner creates a gradient dead zone that prevents beings from learning the hard task (position 2 = arithmetic). Combined with loss dilution (position 2 is 6.25% of total loss) and ring memory write contention, the model appears to learn (loss drops) while actually only solving trivial sub-tasks.

**What we need**: A principled fix to the combiner that maintains gradient flow to uncertain beings, plus guidance on whether this architecture can learn multi-step bit arithmetic at all.
