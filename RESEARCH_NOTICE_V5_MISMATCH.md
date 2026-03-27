# RESEARCH NOTICE: Logic Mismatch in Training Recipes

## Status: RESOLVED (2026-03-27)
**Target:** `instnct/recipes/train_english_1024n_18w.py`
**Architecture Version:** v5.0 (Musical Axonal Brain)
**Original Issue:** Parameter Stagnation (Freq, Phase, Rho not learning)

---

### Resolution

The English recipe (`train_english_1024n_18w.py`) has been refactored:
- `_eval_bigram()` and `eval_accuracy()` now delegate to `SelfWiringGraph.rollout_token()` instead of using a hardcoded forward loop.
- This eliminates **three known divergences** between the old recipe and canonical graph.py:
  1. **Decay:** old used multiplicative `charge *= (1-decay)`, canonical uses subtractive `charge = max(charge - decay, 0)`
  2. **C19 Soft-Wave:** old used additive `max(0, theta + rho*wave)`, canonical uses multiplicative `clip(theta * (1 + rho*wave), 1, 15)`
  3. **Hard Reset:** old had no fired-neuron reset, canonical does `charge[fired] = 0.0`

**Verified by:** `instnct/tests/test_recipe_canonical_ab.py` — A/B smoke test confirming old/new diverge and new/graph are bit-identical.

Any future change to `graph.py` forward dynamics is now automatically reflected in training.

---

### Original Problem (archived for context)

The training recipe used a hardcoded v4.2-era forward pass inside `_eval_bigram` and `eval_accuracy`. C19 Soft-Wave parameters (freq, phase, rho) were visible to the mutation engine but invisible to the scoring function, causing evolution to reject all rhythm mutations. This explained the ~18% English plateau.
