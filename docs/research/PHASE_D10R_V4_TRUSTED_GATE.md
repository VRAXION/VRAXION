# Phase D10r-v4 Trusted Gate

Date: 2026-04-30

## Purpose

D10r-v4 changes the evaluator trust gate from raw score reading to
control-adjusted scoring:

```text
trusted_mo = real_mo - max(control_mo)
```

This blocks candidates that look good only because a projection/readout control
can produce an equal or stronger score.

## Code Changes

`tools/_scratch/d10r_hardened_eval.py` now includes:

- `trusted_mo_mean`, `trusted_mo_ci_low`, `trusted_mo_ci_high`
- `trusted_mo_pass`
- `random_projection_null`
- default fair `state_shuffle_shared` instead of the old independent
  `state_shuffle`
- entropy and max-probability diagnostics per row

## Smoke Run

Run root:

```text
output/phase_d10r_v4_trusted_gate_20260430/smoke
```

Setup:

- `eval_len=256`
- eval seeds `970001..970004`
- controls: `random_label`, `random_projection_null`,
  `state_shuffle_shared`, `no_network_random_state`
- `control_repeats=2`

## Result

Verdict:

```text
D10R_CONTROL_LEAK_FAIL
```

Beta.8:

```text
real_mo_delta_mean  +0.006776
trusted_mo_mean     -0.006871
trusted_mo_ci_low   -0.015173
trusted_mo_pass     false
```

Negative/checkpoint controls:

```text
seed_42    trusted_mo_pass true
seed_1042  trusted_mo_pass true
```

## Interpretation

The new gate is doing the right kind of blocking. Raw beta.8 signal is still
positive, but it does not beat the worst adversarial control with confidence in
this smoke. Also, two alternate H=384 D7 baseline seeds pass the trusted score
against seed2042 baseline, so the current negative-control set is not clean
enough to unlock release-ready claims.

This does not invalidate beta.8 as a research finding. It says the current
evaluator/readout protocol is still too permissive for release promotion.

## Release Impact

D10s, H512, H8192, and release-candidate network promotion remain blocked until:

1. beta.8 or a successor passes `trusted_mo`,
2. no negative checkpoint passes `trusted_mo`,
3. controls are run at longer eval length and fresh seeds,
4. the negative-control set is cleaned up so it represents genuinely bad/null
   networks rather than alternate baselines with their own signal.

## Next Step

D10r-v5 should split the control problem into two lanes:

- `artifact_nulls`: random projection, no-network random state, shuffled labels,
  and shared state shuffle.
- `alternate_baseline_controls`: other D7 seeds, reported separately and not
  mixed with null controls unless their expected behavior is defined.

Only after artifact nulls are clean should D10s wiring-prior smoke run again.

