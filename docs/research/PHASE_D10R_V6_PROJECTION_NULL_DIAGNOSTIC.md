# Phase D10r-v6 Projection Null Diagnostic

Date: 2026-04-30

## Purpose

D10r-v5 identified `random_projection_null_03` as the strongest artifact
blocker. D10r-v6 tests whether that was a real artifact problem or another
paired-control fairness bug.

## Finding

The v5 `random_projection_null` had the same fairness issue as the old
independent `state_shuffle`: each checkpoint received a different random
projection. That made the baseline-vs-candidate comparison vulnerable to random
readout luck.

D10r-v6 changes `random_projection_null` to use one shared random projection
across all checkpoints in the paired comparison. The old behavior remains
available as `random_projection_null_independent`, but is marked as diagnostic
only and does not affect `trusted_mo`.

## Evidence

Diagnostic smoke:

```text
output/phase_d10r_v6_projection_null_20260430/smoke
```

With shared random projection, the projection-null margins became positive:

```text
random_projection_null_00 margin_mean +0.021003
random_projection_null_01 margin_mean +0.010820
```

The old independent diagnostic reproduced readout-luck behavior:

```text
random_projection_null_independent_01 margin_mean -0.021816
```

Bounded main:

```text
output/phase_d10r_v6_projection_null_20260430/bounded_main
```

Setup:

- `eval_len=1000`
- eval seeds `970001..970008`
- `control_repeats=4`
- artifact controls:
  - `random_label`
  - `random_projection_null`
  - `state_shuffle_shared`
  - `no_network_random_state`

Beta.8:

```text
real_mo_delta_mean +0.014526
trusted_mo_mean    -0.011094
trusted_mo_ci_low  -0.032010
```

The shared random projection null no longer blocks:

```text
random_projection_null_00 margin_ci_low +0.031891
random_projection_null_01 margin_ci_low +0.018555
random_projection_null_02 margin_ci_low +0.020567
random_projection_null_03 margin_ci_low +0.014471
```

Remaining blocker:

```text
state_shuffle_shared_02 margin_mean   -0.000826
state_shuffle_shared_02 margin_ci_low -0.024172
```

## Verdict

`D10R_V6_STATE_SHUFFLE_BLOCKED`

The random projection null issue is resolved as a paired-control bug. The
release gate remains blocked because a shared state shuffle can still match or
beat the beta.8 signal on some eval seeds, mainly through smooth/unigram spikes.

## Release Impact

D10s, H512, and H8192 remain blocked. The blocker is now narrower:

```text
old blocker: projection/readout artifact in random_projection_null
new blocker: state/readout invariance under state_shuffle_shared
```

## Next Step

D10r-v7 should focus only on `state_shuffle_shared`:

- add state-shuffle class/zone diagnostics to identify which output-state
  permutation creates the spike
- compare raw state shuffle against projection-consistent shuffle
- add a state-shuffle ensemble bound, analogous to the random projection null
  bound
- only unlock D10s if beta.8 beats the shared state-shuffle bound
