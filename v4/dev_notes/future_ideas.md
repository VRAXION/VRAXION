# INSTNCT — Future Ideas (v5+)

Ideas that are NOT for v4 (minimal baseline) but worth revisiting later.

---

## Gravitational Pointer Dynamics

**Source:** YouTube video discussion (2026-02-26), inspired by g = −∇φ

**Core idea:** Ring slots accumulate "mass" (importance/usage weight). Pointer movement is biased toward high-mass slots — instead of blind φ-jumps, the pointer "falls" toward useful memory.

**How it could work:**
- Each slot gets a scalar mass `m[j]` — updated when written to (additive) or decayed over time
- Pointer "feels" a force: `F = Σ m[j] / dist(ptr, j)²` summed over nearby slots
- Movement becomes: `ptr_next = blend(φ_jump, walk, gravity_pull)`
- The gravity_pull term biases toward the nearest high-mass slot

**Relation to v2:** Thermodynamic governor controlled pointer *speed* via entropy. This controls pointer *direction* via content-dependent attraction. One step further.

**Relation to helix:** Helix gives the pointer a *history* (where it's been). Gravity gives it *intent* (where it should go). Complementary.

**Why not v4:** v4 is the clean mathematical baseline. Adding content-dependent pointer dynamics would make it impossible to isolate what each component contributes. First prove helix works, then layer gravity on top.

**Risk:** O(M) force computation per step (sum over all slots). Could use a kernel window (only nearby slots) to keep it O(R).

---

## Poisson-Guided Pointer Movement

**Source:** Same YouTube video (2026-02-26), Poisson's equation ∇²φ = 4πGρ

**Core idea:** Instead of ad-hoc gravity forces, solve Poisson's equation on the ring to compute a proper potential field from slot "mass density". The pointer follows the gradient of this field.

**How it could work on the ring (1D circle):**
```
1. Mass density:  ρ[j] = ||ring[j]||₂           (L2 norm of slot content)
2. Solve Poisson: ∇²φ = ρ  →  FFT on circle, O(M log M)
3. Force:         g[j] = -(φ[j+1] - φ[j-1]) / 2  (discrete gradient)
4. Movement:      ptr += α · g[ptr]               (drift toward potential wells)
```

**Three viable implementations (ranked by v5 readiness):**

| Approach | Cost | How |
|----------|------|-----|
| **Lokális gradiens** | O(R) | Only look at window neighbors, not whole ring |
| **Content-based pull** | O(R) | `grad = cosine_sim(hidden, ring[j±1])` — relevance, not mass |
| **Full Poisson (FFT)** | O(M log M) | Proper physics, but expensive for small models |

**Best candidate for v5:** Content-based pull — the pointer drifts toward ring slots whose content is *similar* to the current hidden state. Like a mini-attention for pointer direction. O(R) cost because we already have the window.

**Relation to gravitational idea:** This IS the gravitational idea, but with a proper mathematical framework (Poisson) instead of ad-hoc inverse-square forces.

---

## Learned Jump Gate (NEXT PRIORITY — after I/O layer)

**Source:** Adversarial audit (2026-02-27), deep research + codebase archeology

**Core idea:** Replace fixed `prob=0.5` in pointer movement with `gate = sigmoid(linear(hidden))` per expert. Each expert decides jump vs walk based on its current hidden state.

**Evidence:**
- Pointer is input-independent (diagnostic: std=0.00 across batch) — confirmed bottleneck
- Aligned echo 83% vs random 63% — model uses position cue, not pattern recognition
- Skip RNN (ICLR 2018) uses identical mechanism for binary state update decisions
- NTM interpolation gate `g` is the same pattern
- Old VRAXION stride gate existed but used fixed per-role probs, not input-dependent

**Implementation:** ~10 lines, ~4K params. See `pointer_research_2026-02-27.md` for full details.

**Init trick:** gate bias=0 -> sigmoid(0)=0.5 -> starts at current 50-50 behavior.

---
