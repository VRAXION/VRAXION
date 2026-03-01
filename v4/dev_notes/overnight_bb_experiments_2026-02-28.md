# Overnight BB Experiment Log — 2026-02-28/03-01

## Goal

A/B test the Bulletin Board (BB) temporal cache against a clean baseline.
Run 24 champion: 41.4% masked_acc @ 1000 steps (S=0.3, vshape, no BB).
Run 32: 37.4% with BB (gate bias=-2.2, no L2-norm, no telemetry) — BB was dead.

Since then, three fixes applied:

1. L2-normalized cosine attention (prevents query norm explosion)
2. Temperature scaling tau=4 (prevents entropy collapse)
3. Output scale=0.1 (prevents magnitude imbalance vs ring)

Gate bias changed from -2.2 to 0.0 (beta=0.50 starting).

## Experiments

### Run 33A: Baseline (bb_enabled=false)

- **Config**: bb_enabled=false, S=0.3, vshape, 1000 steps, batch=128, seq=64
- **Purpose**: Clean baseline — same config as Run 24
- **Status**: DONE
- **Result**: Peak 38.9% @ step 950, final 37.9% @ step 1000
- **Duration**: 928s (0.9s/step), 5905 MB VRAM, 711K params
- **Trajectory**: 100:19.0% -> 200:26.2% -> 300:30.8% -> 500:34.3% -> 700:35.9% -> 1000:37.9%
- **Last 200 avg**: 37.5%

### Run 33B: BB + learned gate

- **Config**: bb_enabled=true, gate_bias=0.0, scale=0.1, tau=4.0, taps=[1,5,10]
- **Purpose**: Test BB with all stability fixes applied
- **Status**: DONE
- **Result**: Peak 39.2% @ step 900, final 37.3% @ step 1000
- **Duration**: 1524s (1.5s/step), 6561 MB VRAM, 1.1M params
- **Trajectory**: 100:19.3% -> 200:26.8% -> 300:31.5% -> 500:35.7% -> 700:36.5% -> 1000:37.3%
- **Last 200 avg**: 37.9%
- **BB Telemetry**: Expert 0 gate saturated (0.999) then crashed to 0.000 at step 440.
  Expert 1 gate oscillated 0.002-0.55. Model learned to prune BB.

### Run 33C: BB + fixed scale (no gate)

- **Config**: bb_enabled=true, bb_gate_mode=fixed, bb_scale=0.05, tau=4.0
- **Purpose**: Force BB content into model (remove gate instability)
- **Status**: DONE
- **Result**: Peak 38.0% @ step 860, final 38.0% @ step 1000
- **Duration**: 1455s (1.5s/step), 6436 MB VRAM, 1.1M params
- **Trajectory**: 100:19.4% -> 200:26.0% -> 300:29.9% -> 500:34.3% -> 700:37.1% -> 1000:38.0%
- **Last 200 avg**: 37.4%
- **BB Telemetry**: Entropy stable 0.93-0.98 (healthy). ctx_vs_ring stable ~0.03.
  BB signal applied consistently but at very low ratio vs ring.

## Summary Table

```text
Run              Peak    Final   Last200   Time/step  Params
33A (no BB)     38.9%    37.9%    37.5%     0.9s/step   711K
33B (gate)      39.2%    37.3%    37.9%     1.5s/step  1108K
33C (no gate)   38.0%    38.0%    37.4%     1.5s/step  1105K
```

All three runs land in the 37-39% band. BB makes no measurable difference.

## Conclusion

**BB temporal cache is a dead end at this architecture/scale.**

- With learned gate (Run B): model learns to prune BB (gate crashes to 0)
- With fixed scale (Run C): BB signal too small to matter (3% of ring), accuracy identical
- Both BB variants add 56% params and 67% training time for zero benefit
- The ring buffer (M=1024) + hidden state already provide sufficient temporal memory
- BB's raw input cache doesn't add information the model can't already access

**Recommendation**: Disable BB (bb_enabled=false) and focus on other improvements.

## Run 33D: Extended Baseline (2000 steps) — NEW ALL-TIME RECORD

- **Config**: bb_enabled=false, S=0.3, vshape, 2000 steps, batch=128, seq=64
- **Purpose**: Check if 1000 steps is too short
- **Status**: DONE
- **Result**: Peak 44.9% @ step 1620, final 44.5% @ step 2000
- **Duration**: 1983s (1.0s/step), 5905 MB VRAM, 711K params
- **Trajectory**:

```text
Step  | Acc    | Phase gain
  500 | 35.9%  | +35.9% (from 0)
 1000 | 40.5%  | +4.6%
 1500 | 43.0%  | +2.5%
 2000 | 44.5%  | +1.5%
```

- **Last 200 avg**: 43.5%
- **Trend**: +0.12% per 100 steps in last 500 (still positive, diminishing)
- **LR at end**: 0.000000 (cosine decay exhausted)
- **Note**: Beats Run 24 (41.4%) by +3.5%. The 38-39% "plateau" at 1000 steps was
  NOT a model capacity limit — it was just the midpoint of the learning curve.

## Key Insight

**The architecture is NOT the bottleneck. Training budget is.**

All BB A/B comparisons at 1000 steps were premature. The model continues gaining
~2-4% accuracy per 500-step block beyond step 1000. The cosine LR schedule decay
to 0 at step 2000 means the model is essentially frozen at that point — a longer
schedule might push further.

## Run 33E: Extended Baseline (3000 steps)

- **Config**: bb_enabled=false, S=0.3, vshape, 3000 steps
- **Purpose**: Does extending cosine LR over 3000 steps push accuracy past 45%?
  At 2000 steps, LR was already 0 for the last ~200 steps. A 3000-step schedule
  gives more time at meaningful LR.
- **Status**: DONE
- **Result**: Peak 47.8% @ step 2960, final 46.9% @ step 3000
- **Duration**: 2923s (1.0s/step), 5905 MB VRAM, 711K params
- **Trajectory**:

```text
Step  | Acc    | Phase gain
  500 | 35.2%  | +35.2% (from 0)
 1000 | 41.0%  | +5.8%
 1500 | 42.6%  | +1.6%
 2000 | 44.4%  | +1.8%
 2500 | 47.3%  | +2.9%
 3000 | 46.9%  | -0.4% (LR exhausted)
```

- **Last 200 avg**: 46.6%
- **LR died**: < 1e-5 at step 2830. Last 170 steps are frozen (no learning).
- **Trend**: accuracy plateaued ~46-47% once LR went below 1e-5
- **Note**: Beats Run 33D (44.9%) by +2.9%. The 3000-step cosine schedule gives
  ~800 more steps of meaningful LR vs the 2000-step schedule. However, the last
  ~200 steps (LR≈0) are wasted — the model is frozen.

## Updated Summary Table

```text
Run              Peak    Final   Last200   Time/step  Params  Steps
33A (no BB)     38.9%    37.9%    37.5%     0.9s       711K   1000
33B (BB gate)   39.2%    37.3%    37.9%     1.5s      1108K   1000
33C (BB fixed)  38.0%    38.0%    37.4%     1.5s      1105K   1000
33D (no BB)     44.9%    44.5%    43.5%     1.0s       711K   2000
33E (no BB)     47.8%    46.9%    46.6%     1.0s       711K   3000  ← NEW RECORD
```

## Scaling Analysis

```text
Steps  | Peak   | Gain vs prev | Marginal gain per 1000 steps
1000   | 38.9%  |     —        |  +38.9%
2000   | 44.9%  |  +6.0%       |  +6.0%
3000   | 47.8%  |  +2.9%       |  +2.9%
```

Diminishing returns: 1000→2000 gives +6.0%, but 2000→3000 gives only +2.9%.
The model is learning, but each additional 1000 steps gives roughly half the
gain of the previous block. Extrapolating: 4000 steps might yield ~49%, 5000
steps ~49.5-50%. But 10,000 steps probably won't reach 55%.

## Final Conclusions (Overnight Session)

1. **BB temporal cache is a dead end**: zero benefit at any configuration tested.
   Disable and move on.

2. **Training duration was the real bottleneck**: the "38-39% plateau" at 1000
   steps was just the midpoint of the learning curve, not a capacity limit.

3. **Cosine LR decay wastes the tail**: last ~6% of steps have LR < 1e-5 and
   contribute nothing. A warmup-constant-cooldown schedule might be more efficient.

4. **Diminishing returns set in hard**: gains per 1000 steps approximately halve
   each time (6.0% → 2.9% → ~1.5%?). The current architecture at 711K params
   is approaching its capacity ceiling around 48-50%.

5. **Next experiments to try** (priority order):
   a. Longer run (5000 steps) to find the true ceiling
   b. Larger model (hidden_dim=4096, more params) to raise the ceiling
   c. Alternative LR schedules (constant+cooldown, cyclic)
   d. Curriculum learning (start with easy tasks, progressively harder)
