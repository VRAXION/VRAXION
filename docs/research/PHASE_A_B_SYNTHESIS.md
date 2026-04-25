# Phase A + Phase B Synthesis

## Core Finding

Neuron count `H` is not a standalone architectural verdict. The observed scaling profile depends on the training recipe and fixture:

- `mutual_inhibition` shows an inverted-U under the fixed 20k-step budget: `H=128 -> 3.76%`, `H=256 -> 5.28%`, `H=384 -> 3.52%`.
- `bytepair_proj` does not show the same inverted-U: `H=128 -> 5.24%`, `H=256 -> 3.62%`, `H=384 -> 3.16%`, with high variance and near-collapse at `H=384`.
- Phase B confirms that the `mutual_inhibition` H=384 drop is directionally a training-horizon confound: `B1`/40k steps recovers to `5.50%` mean peak, near the Phase A `H=256` reference (`5.28%`).

## What Is Proven

- Phase A is a complete 30-cell baseline: two fixtures x three H values x five seeds.
- Phase B B0 exactly reproduces Phase A `mutual_inhibition, H=384` on checked metrics and seeds.
- For `mutual_inhibition`, the fixed-budget inverted-U should not be described as intrinsic H=384 failure.
- `bytepair_proj` H=384 has a different failure mode: very low accept rate (`0.61%`), low alive fraction (`0.05`), and high seed sensitivity (`0.00%..6.00%` peak range).

## What Is Not Proven

- Phase B does not prove that `bytepair_proj` would be fixed by longer horizon.
- Phase B does not prove a resonance, chaos, or edge-of-chaos mechanism for 12 ticks.
- The `C_K` decomposition is useful as a diagnostic panel, but is not yet a validated scalar objective.
- Operator schedule retuning is a strong Phase C hypothesis, not a measured improvement yet.

## Interpretation

The safe paper-level claim is recipe-dependence:

> Search dimensionality interacts with training horizon, mutation schedule, and fixture-specific pruning policy. A fixed `H` sweep alone can create misleading architectural conclusions.

This is stronger and safer than the original `L = Psi * sigma_mu / D` reading. The data says `D` is not simply bad; larger `D` needs the right budget and policy. If those do not scale, the system can look intrinsically worse even when the limitation is procedural.

## Next Experiments

1. **Phase B.1: MI horizon and tie-policy ablation**
   - Scope: `mutual_inhibition`, `H=384`.
   - Matrix: `accept_ties=true/false x steps=20k/40k/80k x 5 seeds`.
   - Purpose: confirm the horizon effect, test whether 80k makes H=384 exceed the H=256 reference, and determine whether neutral accepts matter.

2. **Phase C: operator schedule retuning**
   - Scope: `mutual_inhibition`, likely H=384.
   - Hypothesis: increase draw share for `theta` and `channel`, remove or reduce `projection_weight`.
   - Requirement: pre-register as a predictive test; do not claim boost before measuring it.

3. **Bytepair-proj collapse/prune ablation**
   - Scope: `bytepair_proj`, `H=384`.
   - Hypothesis: collapse is driven by grow-prune/crystallize policy, not just H.
   - Candidate knobs: prune aggressiveness, keep-best restore, minimum alive constraint, accept gate around prune cycles.

## Reporting Boundary

Phase B.1 should stay on `mutual_inhibition`. Bytepair-proj should not be folded into the same experiment, because it adds a grow-prune mechanism and would confound the clean horizon question.
