# Phase D10r-v7 State Shuffle Diagnostic

Date: 2026-04-30

## Purpose

D10r-v6 fixed the `random_projection_null` pairing bug and narrowed the
remaining release blocker to state shuffling. D10r-v7 adds two diagnostics:

- artifact family bounds, so repeated controls are judged as a family
- per-metric margins, so a blocker can be traced to smooth/accuracy/echo/unigram

This is still an evaluator trust gate. It does not search for a new checkpoint.

## Implementation

`d10r_hardened_eval.py` now writes:

- `d10r_control_family_bounds.csv`
- `d10r_metric_margins.csv`

The report also shows:

- family-level artifact bounds
- state-shuffle metric margins
- `blocking_control_families`

The new verdict is:

```text
D10R_V7_STATE_SHUFFLE_BOUND_BLOCKED
```

## Run

Bounded main:

```text
output/phase_d10r_v7_state_shuffle_diagnostic_20260430/bounded_main
```

Setup:

- `eval_len=1000`
- eval seeds `970001..970008`
- `control_repeats=4`
- positive only: baseline vs beta.8
- controls:
  - `random_projection_null`
  - `state_shuffle_shared`
  - `state_shuffle_projection_consistent`
  - `no_network_random_state`

## Evidence

Beta.8 still has positive raw signal:

```text
real_mo_delta_mean   +0.014526
real_mo_delta_ci_low +0.014479
```

Family bounds:

```text
random_projection_null   bound_ci_low +0.014470  PASS
no_network_random_state  bound_ci_low +0.008416  PASS
state_shuffle_shared     bound_ci_low -0.030587  FAIL
```

The exact blocker is:

```text
state_shuffle_shared_02
```

Worst seed:

```text
970004
```

The metric breakdown shows the failure is not a single harmless metric wobble.
For `state_shuffle_shared_02`, the worst seed produced:

```text
smooth   -0.018381
accuracy -0.007000
unigram  -0.034870
mo       -0.073935
```

Here the numbers are real-minus-control margins. Negative means the shuffled
state control beat the real beta.8 run on that metric/seed.

Projection-consistent state shuffle remained exactly zero-margin, as expected:

```text
state_shuffle_projection_consistent_* margins = 0
```

That sanity check means the shuffle transform itself is mechanically correct.

## Verdict

`D10R_V7_STATE_SHUFFLE_BOUND_BLOCKED`

D10r-v7 rules out broad random/null readout luck as the current blocker. The
remaining issue is narrower: a shared hidden-state permutation can still produce
beta.8-like or stronger signal on at least one repeat/seed.

## Release Impact

D10s, H512, and H8192 remain blocked.

The blocker is now specific enough to attack directly:

```text
old: projection/readout artifact
now: state permutation invariance / state identity weakness
```

## Next Step

D10r-v8 should test whether the state-shuffle blocker is caused by projection
symmetry or by the hidden state distribution itself:

- add projection-row uniqueness diagnostics
- add per-class score-drift diagnostics for `state_shuffle_shared_02`
- add state-zone masks: input-like rows, middle rows, output-like rows
- test a stricter readout that penalizes state-shuffle invariance
- only unlock D10s if beta.8 beats the state-shuffle bound
