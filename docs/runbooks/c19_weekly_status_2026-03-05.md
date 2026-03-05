# C19 Weekly Status — 2026-03-05

This note records the current state of the standalone C19 investigation that ran alongside the main nightly line this week.

Important:
- This is a research status note, not a merged runtime change.
- Most of the work happened in standalone A/B and sweep scripts outside the current nightly codepath.
- The current goal is to decide which C19 asymmetry pattern deserves promotion into the next integration pass.

## Scope

This week focused on four questions:

1. Is symmetric C19 too weak or too ambiguous in recurrent use?
2. Does negative-side asymmetry help?
3. Is full dual-phi better than simple `neg*phi only`?
4. Are `C` regularization and tail-limit changes worth carrying forward?

## What Was Tested

- C19 cost / profiling:
  - activation bottleneck profiling
  - cheaper alternatives and inference timing
- Standalone asymmetry sweeps:
  - `original`
  - `asym neg=1.5`
  - `asym neg=phi`
  - `asym neg=2.0`
  - `asym neg=phi^2`
  - `asym neg=3.0`
- Standalone dual-family tests:
  - `neg*phi only`
  - `dual-phi`
  - `dual-phi-inv`
  - `dual-sqrt-phi`
  - `dual-phi-same`
- Learnable-`C` checks:
  - causal swap test
  - regularization A/B
  - flat vs phi-structured regularization
  - tail-limit sweeps
- Real-data GPU A/B:
  - baseline vs `dual-phi`
  - `neg-phi` vs `dual-phi`
- Real-data GPU telemetry pilot:
  - `dual-phi` vs `dual-phi-envelope(alpha)`
  - `alpha = 0.02, 0.05, 0.10`
  - tail-hit and `|x|/C` quantile logging

## What Looks Confirmed

### 1) Plain symmetric C19 is not the best learner

Short local CPU A/B on the echo task showed the symmetric baseline lagging the asymmetric variants by a large margin over 100 steps.

Observed quick-run pattern:
- `original (1.0)`: 31.5% best acc
- `asym neg=phi`: 56.8% best acc
- `asym neg=2.0`: 63.0% best acc
- `asym neg=phi^2`: 69.0% best acc

Interpretation:
- breaking symmetry helps;
- the model appears to benefit from "knowing which side it is on";
- stronger asymmetry increases gradient volatility.

### 2) The sign matters: stronger negative side is useful, stronger positive side is dangerous

Local dual-family A/B on the echo task showed:

- `neg*phi only`: 56.8% best acc, stable enough
- `dual-phi`: 30.7% best acc, extremely stable but too damped in this short regime
- `dual-phi-inv` (`neg*1/phi`, `pos*phi`): catastrophic instability
- `dual-phi-same` (`neg*phi`, `pos*phi`): catastrophic instability

Interpretation:
- it is not enough to say "some phi asymmetry";
- the useful direction is "negative side gets the stronger gain";
- making the positive side stronger is currently treated as a bad direction.

### 3) Dual-phi is the current standalone winner in longer real-data runs

The decisive 500-step WikiText GPU A/B was reproduced locally this week using the same real-data script family as the devlog note in [v4/dev_notes/ab_c19_dualphi_results_2026-03-05.md](../../v4/dev_notes/ab_c19_dualphi_results_2026-03-05.md).

Observed head-to-head result:

| Metric | Neg-Phi | Dual-Phi | Delta |
|---|---:|---:|---:|
| Final Acc | 51.7% | 53.4% | +1.65% |
| Best Acc | 54.0% | 55.9% | +1.9% |
| Final Loss | 1.6979 | 1.6264 | -4.2% |
| BPC | 2.450 | 2.346 | -0.104 |
| Wall Time | 1175s | 1128s | -47s |
| Max GradNorm | 7.4 | 3.5 | 2x lower |
| Grad Spikes | 0 | 0 | both stable |

Key read:
- `dual-phi` beat the old symmetric baseline in the prior A/B;
- `dual-phi` also beat `neg-phi only` in the direct same-config comparison;
- `dual-phi` took the lead around step 14 and kept it;
- the shorter CPU echo regime understated `dual-phi`, but the longer real-data run paid back its extra damping.

### 4) Learnable `C` looks real

The causal `C` swap test passed in prior checking, which is evidence that `C` carries task-specific signal rather than acting as pure noise.

Current read:
- learnable `C` is not yet fully production-qualified;
- but it is no longer reasonable to treat it as decorative.

### 5) Light outer-loop damping is inert in the current regime

A local 200-step WikiText telemetry pilot compared plain `dual-phi` against `dual-phi-envelope(alpha)` with `alpha = 0.02, 0.05, 0.10`.

Observed pattern:
- all four variants finished at essentially the same place (`45.5%` final acc);
- `tail_hit = 0.0000%` for every variant;
- `p99 |x|/C = 1.05`;
- `max |x|/C ~= 2.20-2.21`, still far below the `6C` tail boundary.

Interpretation:
- the model is operating deep inside the periodic core;
- light damping of farther arches does not meaningfully change the active regime;
- this makes "more loops before tail" and "small soft envelope before tail" low-priority ideas for the current task.

## What Is Still Open

### 1) The dual-phi verdict is strong, but still based on a narrow validation slice

What is already true:
- `dual-phi > symmetric baseline`;
- `dual-phi > neg-phi only`;
- `dual-phi` does this while staying at least as stable.

What is still missing before a mainline promotion:
- more than one seed;
- at least one longer or sequential run;
- confirmation that the same gain holds after integration into the active model path.

### 2) Learnable `C` regularization is still under research

The current best read is that simple flat regularization around `lambda ~= 1e-4` is the strongest practical candidate.

Still open:
- whether phi-structured regularization has any real edge;
- how much regularization survives after moving from toy tests to production training.

### 3) Tail-limit changes are not yet compelling

The current read is:
- there is no strong evidence yet that the current `6C` tail boundary needs to change;
- new telemetry says the active distribution sits far below the tail (`p99 |x|/C ~= 1.05`, `max ~= 2.21`);
- light outer-loop damping also failed to change behavior in this regime;
- tail-limit work remains lower priority than asymmetry and `C` regularization.

## Current Best Read

If we had to summarize the week in one sentence:

> C19 wants asymmetry, and the best asymmetry found so far is dual-phi: stronger negative arches, weaker positive arches.

Practical version:
- symmetric baseline is too weak;
- `neg*phi only` helped expose the direction of the effect;
- `dual-phi` is now the current lead standalone variant, not just the prettier hypothesis;
- the sign of the asymmetry matters more than the raw amount of scaling.

## Planned Next Tests

The next tests should be about confidence, not rediscovery:

- repeat the `neg-phi` vs `dual-phi` WikiText A/B across more seeds;
- run at least one longer or sequential validation;
- carry the winning activation into the active model path and confirm the gain survives integration;
- continue `C` regularization work only after the activation verdict is stable;
- if tail work is revisited, do it with a forced-tail stress task or much stronger envelope, not with more light damping.

## Promotion Guidance

Safe to say now:
- asymmetry is real;
- negative-side amplification is meaningful;
- positive-side amplification is dangerous;
- `dual-phi` is the current standalone winner and belongs in nightly-level experiment notes.
- light outer-loop damping does not buy anything in the current WikiText regime.

Not safe to say yet:
- that `dual-phi` should already replace the active mainline C19 in production;
- that tail-limit changes should be merged;
- that phi-structured regularization has beaten the simpler flat regularization.
