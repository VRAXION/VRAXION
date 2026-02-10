# Möbius Helix Implementation Results

## Summary

Testing three Möbius implementations on 2-pair associative recall task (harder than 1-pair baseline).

**Key Finding:** The mathematically correct TRUE Möbius helix with Z₂ holonomy tracking **underperforms** the topologically incorrect 2-layer implementation.

---

## Results Comparison

| Implementation | Best Accuracy | Jump Gate | Params | vs Baseline |
|----------------|---------------|-----------|--------|-------------|
| **No Möbius** (baseline) | 75.2% | 0% | 516 | - |
| **2-Layer "Möbius"** (incorrect) | **78.4%** | 7% | 644 | +3.2% ✅ |
| **TRUE Möbius** (fixed +1 init) | 72.4% | 0-1% | 644 | -2.8% ❌ |
| **TRUE Möbius** (random ±1 init) | 74.6% | 0% | 644 | -0.6% ❌ |

---

## Implementation Details

### 2-Layer "Möbius" (Incorrect)

**What it does:**
- Uses θ = (position / num_positions) × π (only half rotation, 0° → 180°)
- Creates two discrete phase layers at θ=0 and θ=π
- No holonomy state tracking
- Topologically: 2 stacked circles, NOT a Möbius strip

**Code:**
```python
theta = (pointer_position / float(self.num_memory_positions)) * math.pi
phase_cos = torch.cos(theta).unsqueeze(1)
phase_sin = torch.sin(theta).unsqueeze(1)
context_read = context_read + phase_cos * self.phase_embed[0] + phase_sin * self.phase_embed[1]
```

**Why it's wrong:**
- Discontinuous: at wrap (63 → 0), phase jumps from θ=180° back to θ=0°
- Not a continuous double-cover
- Missing the "twist" that defines a Möbius strip

**Why it works:**
- Simple phase modulation provides 2× distinct contexts for same position
- No complex holonomy dynamics to interfere with learning
- Stable gradients (no sign flips)

---

### TRUE Möbius (Correct Topology)

**What it does:**
- Uses θ = (position / num_positions) × 2π (full rotation, 0° → 360°)
- Tracks holonomy state (±1) that flips on pointer wrap
- Multiplies phase modulation by holonomy sign
- Topologically: continuous seam-free double-cover (true Möbius strip)

**Code:**
```python
# Initialize holonomy randomly per sample
holonomy_state = torch.randint(0, 2, (B,), device=x.device) * 2.0 - 1.0  # Random ±1

# In forward loop:
theta = (pointer_position / float(self.num_memory_positions)) * (2.0 * math.pi)
phase_cos = torch.cos(theta).unsqueeze(1)
phase_sin = torch.sin(theta).unsqueeze(1)
context_read = context_read + holonomy_state.unsqueeze(1) * (
    phase_cos * self.phase_embed[0] + phase_sin * self.phase_embed[1]
)

# Detect wrap and flip holonomy
old_pointer_position = pointer_position.clone()
# ... [pointer update] ...
wrapped = (pointer_position < 1.0) & (old_pointer_position >= self.num_memory_positions - 1.0)
holonomy_state = torch.where(wrapped, -holonomy_state, holonomy_state)
```

**Why it's correct:**
- Continuous: phase varies smoothly from 0° → 360°
- Holonomy flips on wrap (63 → 0), creating true double-cover
- Same physical position has TWO distinct representations depending on holonomy state
- Mathematically faithful to Z₂ Möbius topology

**Why it underperforms:**
- **Hypothesis 1 - Training difficulty**: Holonomy flips create sudden sign changes in phase modulation, potentially disrupting gradient flow
- **Hypothesis 2 - Wrap ambiguity**: With JUMPS (not just walks), "completing a lap" is ill-defined; wrap detection might trigger incorrectly
- **Hypothesis 3 - Insufficient exposure**: With 64 positions and 32 timesteps, pointer wraps ≤1 time per sequence; model sees mostly one holonomy state
- **Hypothesis 4 - Random init noise**: Random ±1 initialization adds variance that makes learning harder

---

## Experimental Setup

**Task:** 2-pair associative recall (assoc_clean)
- 2 keys (values 0 or 1)
- 2 pairs per sequence (key appears twice, same value both times)
- Sequence length: 32 timesteps
- Ring size: 64 positions
- Model: 64×64D (516 params baseline, 644 params with Möbius phase embeddings)

**Training:**
- Streaming data (fresh samples every step)
- Adam optimizer, lr=0.001
- 2500-2700 steps per run

**Evaluation:**
- Fixed eval set (500 samples, seed=9999)
- Evaluated every 50 steps

---

## Detailed Findings

### Finding 1: Holonomy Flips Work Correctly

Test with `test_mobius_holonomy.py` (8 positions, 100 timesteps):
- ✅ Wraps detected when pointer crosses 7 → 0 boundary
- ✅ Holonomy flips at exactly the same timesteps as wraps
- ✅ Both holonomy states (+1 and -1) used during sequence
- ✅ No NaN or Inf values

**Conclusion:** Implementation is correct at the code level.

---

### Finding 2: Limited Holonomy Exposure in Training

Debug with `debug_training_holonomy.py` (64 positions, 32 timesteps):
```
Holonomy trajectory: [ 1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. ... -1.]
Pointer trajectory:  [63.83  0.83  1.83  2.83  3.83  4.83  5.83 ... 30.83]
```

- Holonomy starts at +1
- Flips to -1 at step 1 (wrap from 63.83 → 0.83)
- **Stays at -1 for remaining 31 timesteps** (no second wrap)

**Implication:** Model trains on sequences where holonomy is constant for ~97% of timesteps. Double-cover property underutilized.

**Fix Attempt:** Random ±1 initialization per sample
- Improvement: 72.4% → 74.6% (+2.2%)
- Still below 2-layer: 78.4% (-3.8%)

---

### Finding 3: Jump Gate Not Activating

All TRUE Möbius runs: jump_gate ≈ 0% throughout training
- Model learns to suppress jumping entirely
- Reverts to pure sequential walking
- Routing mechanism collapses

Compare to 2-layer Möbius: jump_gate ≈ 7%
- Jump gate activates and provides routing diversity
- Emergent routing patterns develop

**Implication:** TRUE Möbius may be interfering with the jump mechanism, possibly due to holonomy sign flips making routing inconsistent.

---

## Hypotheses for Poor Performance

### Hypothesis 1: Gradient Disruption from Sign Flips

When holonomy flips (±1 → ∓1), phase modulation sign reverses:
```python
phase = holonomy_state * (cos(θ) * embed[0] + sin(θ) * embed[1])
```

If holonomy flips mid-sequence:
- Same position (e.g., position 0) produces opposite-sign contributions
- Gradient flow may be disrupted by these discontinuities
- Model struggles to learn stable representations

**Test:** Train with holonomy frozen (no flips) - does performance improve?

---

### Hypothesis 2: Wrap Detection Ambiguity with Jumps

Current wrap detection:
```python
wrapped = (pointer_position < 1.0) & (old_pointer_position >= num_positions - 1.0)
```

Triggers on:
- Walk wrap: 63.5 → 0.5 ✅ (natural wrap, should flip)
- Jump wrap: 63 → 0 ❓ (discrete jump across boundary - is this a "lap"?)
- Jump non-wrap: 40 → 5 ❌ (doesn't cross boundary, correctly no flip)

**Problem:** With jumps, "completing a lap around the ring" is ill-defined. Holonomy should flip after traversing 64 positions of distance, but:
- Jumps skip intermediate positions
- Hard to define "accumulated distance" with arbitrary jumps

**Potential solutions:**
1. Only flip on walks (position_new = position_old + 1 mod num_positions)
2. Track accumulated distance and flip every 64 units
3. Disable jumps when using TRUE Möbius (force walk-only)

---

### Hypothesis 3: Insufficient Sequence Length

With 64 positions and 32 timesteps:
- Pointer wraps at most 1 time per sequence (if walking)
- Most sequences experience only one holonomy state
- Model doesn't learn to use double-cover effectively

**Test:** Increase seq_len to 128 or 256, forcing multiple wraps per sequence

---

### Hypothesis 4: Inductive Bias Mismatch

The 2-layer version, while topologically incorrect, may provide a **better inductive bias** for this task:
- Two discrete contexts (θ=0 and θ=π) create clear separation
- No dynamic sign flips to confuse gradients
- Simple, stable phase modulation

TRUE Möbius:
- Continuous phase rotation (0° → 360°) may be too smooth
- Dynamic holonomy flips add complexity
- More degrees of freedom = harder to optimize

**Analogy:** Like how ReLU (simple, discontinuous) often outperforms smooth activations (sigmoid, tanh) despite being "mathematically uglier"

---

## Next Steps

### Option 1: Accept 2-Layer as "Good Enough"
- Topologically incorrect but empirically superior
- Document as "Möbius-inspired phase embedding" (not true Möbius)
- Move forward with 2-layer implementation

### Option 2: Debug TRUE Möbius
Investigate further:
1. **Test with longer sequences** (seq_len=128) - do multiple wraps help?
2. **Test with smaller ring** (num_positions=32) - more frequent wraps?
3. **Test wrap detection variants** - only flip on walks, not jumps?
4. **Test frozen holonomy** - disable flips, use random ±1 but keep constant?
5. **Visualize holonomy trajectories** - are flips happening too often/rarely?

### Option 3: Hybrid Approach
- Use 2-layer for training (better performance)
- Implement TRUE Möbius as research experiment
- Document both in paper/blog post

---

## Conclusion

**Empirical result:** 2-layer "Möbius" (78.4%) outperforms TRUE Möbius (74.6%) on 2-pair task.

**Theoretical insight:** Mathematical correctness ≠ empirical performance. The topologically incorrect 2-layer implementation provides a simpler, more stable learning signal that's easier to optimize.

**Recommendation:** Further investigation needed before committing to TRUE Möbius. The 2-layer version is a viable alternative for practical use.

---

## Code Commits

- **8610119**: Implement TRUE Möbius helix with Z₂ holonomy tracking
- **[pending]**: Document Möbius implementation comparison results

---

## Session Date

2026-02-10
