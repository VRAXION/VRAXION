# CRITICAL RESEARCH NOTICE: Logic Mismatch in Training Recipes

## Status: Identified & Verified
**Target:** `instnct/recipes/train_english_1024n_18w.py` (and potentially other active recipes)
**Architecture Version:** v5.0 (Musical Axonal Brain)
**Issue:** Parameter Stagnation (Freq, Phase, Rho, Delays not learning)

---

### 1. Problem Diagnosis
The training recipe uses a **hardcoded forward pass** inside `_eval_bigram` and `eval_accuracy`. This code was written for the v4.2 era and does NOT incorporate the new v5.0 dynamics.

**The hardcoded loop currently looks like this (simplified):**
```python
for t in range(8):
    charge += raw
    charge *= ret # Old percentage-based decay
    act = np.maximum(charge - theta, 0.0)
    charge = np.maximum(charge, 0.0)
```

### 2. Why Evolution Failed to Tune C19 (Freq/Rho)
The mutation engine in `mutate()` correctly changes `self.freq`, `self.phase`, and `self.rho`. However, when the `worker_eval` runs, it uses the hardcoded loop above. 

**Result:**
- The effect of frequency and phase modulation is **mathematically invisible** to the scoring function.
- `Delta Score` is always `0` for any rhythm mutation.
- Evolution rejects all `FR`, `PH`, and `RH` mutations, leaving the parameters at their random initial state.

### 3. Missing v5.0 Features in Recipes
Currently, the English training loop is "blind" to:
1. **C19 Soft-Wave:** No `effective_theta` calculation using `sin(t * freq + phase)`.
2. **Int4 Scaling:** No `MAX_CHARGE = 15.0` clamping.
3. **Axonal Delay:** No ring-buffer lookup for `history[d-1]`.
4. **Fix-Amount Leak:** Still using old multiplicative `charge *= ret`.

### 4. Required Resolution
The recipes must be refactored to use the canonical **`SelfWiringGraph.rollout_token_batch`** instead of local manual loops. This ensures that any future architectural change in `graph.py` is automatically reflected in all training sessions.

---
*Note: This mismatch explains why the English test hit a plateau at ~18%. The network was being mutated for a 3D temporal world but judged by a 2D static evaluator.*
