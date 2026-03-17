# Surprise-Modulated Local Learning — Design Notes

## Status: DESIGN PHASE (not yet implemented)

## Core Insight
The human brain does NOT use backprop. But it also does NOT use blind mutation+selection.
It uses **local learning rules modulated by a global surprise signal**.

## What the brain has that VRAXION v4.2 lacks

| Brain mechanism | VRAXION v4.2 equivalent | Gap |
|---|---|---|
| Hebbian plasticity (fire together → wire together) | None — mutations are random | LOCAL learning rule |
| Dopamine prediction error (surprise) | score_surprise() in eval sweep | Used only for eval, not for learning |
| Eligibility traces (recently-active synapses) | None | Need to track active edges during forward |
| Three-factor rule: pre × post × surprise | None | The key missing piece |
| Homeostatic plasticity | loss_pct drift (partial) | Per-neuron firing rate regulation |

## Three-Factor Learning Rule

The biological learning rule for each synapse is:

```
Δw_ij = η × pre_i × post_j × surprise
```

Where:
- `pre_i` = activation of source neuron i
- `post_j` = activation of target neuron j
- `surprise` = global scalar: how unexpected was the outcome?
  - positive surprise (better than expected) → strengthen active paths
  - negative surprise (worse than expected) → weaken active paths
- `η` = learning rate

### VRAXION translation

During forward_batch, for each tick:
```python
# acts[i] @ mask[i,j] → raw[j]
# eligibility[i,j] += acts[i] * acts[j]   (outer product, sparse)
```

After forward, compute surprise:
```python
probs = softmax(logits)
surprise = -log(prob_of_correct_target)  # per-sample
mean_surprise = surprise.mean()
baseline_surprise = running_average(surprise)  # EMA
delta = baseline_surprise - mean_surprise  # positive = better than expected
```

Apply local update:
```python
# Only modify edges where eligibility > threshold
mask += learning_rate * eligibility * delta
# Clip back to ternary {-DRIVE, 0, +DRIVE}
```

## Key advantages over mutation+selection

1. **Directed search** — modifies exactly the edges that were active (not random)
2. **Credit assignment** — eligibility traces solve "which edge helped?"
3. **No undo needed** — the update IS the learning, no accept/reject cycle
4. **Scales better** — O(active_edges) per update instead of O(budget × intensity)

## Surprise scoring function (BACKUP from predictive_eval_sweep.py)

```python
def score_surprise(net, targets):
    """Accuracy + normalized log-likelihood (entropy-based scoring)."""
    logits = net.forward_batch(TICKS)
    probs = safe_softmax(logits)
    V = net.V
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = np.clip(probs[np.arange(V), targets[:V]], 1e-10, 1.0)
    mean_surp = (-np.log(tp)).mean()           # mean surprise (cross-entropy)
    norm_surp = 1.0 - min(mean_surp / np.log(V), 1.0)  # 0=max surprise, 1=no surprise
    return 0.5 * acc + 0.5 * norm_surp, acc
```

## Implementation plan (when ready)

### Phase 1: Eligibility-tracked forward
- Modify `forward_batch()` to accumulate per-edge eligibility
- Eligibility = sum over ticks of `|acts_pre| * |acts_post|` for each edge
- Store as sparse dict or mask-shaped array

### Phase 2: Surprise signal
- After forward: compute prediction error (cross-entropy vs running baseline)
- `delta = baseline - current_surprise`  (positive = improvement)
- Maintain EMA baseline with α=0.1

### Phase 3: Local update rule
- `Δmask = lr * eligibility * sign(delta)`
- For ternary mask: if |accumulated Δ| crosses threshold → flip sign or add/remove edge
- Alternative: keep float "tendency" per edge, snap to ternary periodically

### Phase 4: Hybrid
- Use surprise-modulated local updates as PRIMARY learning
- Keep mutation+selection as EXPLORATION mechanism (for discovering new topologies)
- Strategy bits (signal, grow, intensity) could modulate the learning rate

## Open questions

1. How to maintain ternary constraint with continuous updates?
   - Option A: accumulate float deltas, snap periodically
   - Option B: probabilistic: Δ sets probability of flip/add/remove
2. How to handle the baseline EMA when training on different targets?
3. Should eligibility traces decay within a forward pass? (biological: yes, τ ~seconds)
4. Can this coexist with mutation+selection in a hybrid approach?

## References (conceptual)
- Schultz et al. (1997): Dopamine prediction error
- Frémaux & Gerstner (2016): Three-factor learning rules
- Miconi et al. (2018): Differentiable plasticity (related but uses backprop to LEARN the rule)
