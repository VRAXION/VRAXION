# Phase D10r-v8 State Identity Gate

Date: 2026-04-30

## Purpose

D10r-v8 closes the remaining beta.8 release-readiness blocker: whether the
validated H=384 `seed2042_improved_generalist_v1` signal survives artifact-null
controls that preserve or disrupt state identity.

Earlier D10r revisions separated two control classes:

- Artifact nulls decide whether the evaluator/readout is trustworthy.
- Alternate D7 seed checkpoints are report-only baselines, not null controls.

D10r-v8 focuses on the last artifact family that still blocked beta.8:
`state_shuffle_shared`.

## Implementation

`tools/_scratch/d10r_hardened_eval.py` now emits D10r-v8 diagnostics:

- `projection_row_uniqueness.csv`
- `state_shuffle_zone_bounds.csv`
- `state_shuffle_score_drift.csv`

The new diagnostics split state shuffles into:

- full shared state shuffle
- projection-consistent shuffle sanity control
- active-row shuffle
- high-norm row shuffle
- low-norm row shuffle
- duplicate projection row shuffle
- near-similar projection row shuffle

The projection-consistent control should stay near zero. It is a guard that the
shuffle implementation itself is not inventing a difference.

## Run

Output root:

```text
output/phase_d10r_v8_state_identity_gate_20260430/main
```

Run shape:

```text
eval_len: 1000
eval_seeds: 970001..970016
control_repeats: 8
device: cuda
elapsed: 5680.44s
```

Primary artifact controls:

- `random_projection_null`
- `state_shuffle_shared`
- `state_shuffle_projection_consistent`
- `no_network_random_state`

## Results

Beta.8 still has a real raw signal:

```text
real_mo_delta_mean    +0.0144964868
real_mo_delta_ci_low  +0.0144065531
```

Artifact controls:

| Control family | Bound mean | CI low | Pass |
|---|---:|---:|---|
| random_projection_null | +0.0102982134 | +0.0102142170 | yes |
| no_network_random_state | +0.0088318603 | +0.0076125350 | yes |
| state_shuffle_shared | -0.0527821628 | -0.1112784165 | no |

The trusted beta.8 margin against artifact controls fails:

```text
trusted_mo_mean    -0.0537089232
trusted_mo_ci_low  -0.1117082303
trusted_mo_ci_high -0.0141610696
```

State-zone diagnostics:

| Zone | Rows | Bound mean | CI low | Pass |
|---|---:|---:|---:|---|
| active_rows | 237 | -0.0556154784 | -0.0963420603 | no |
| high_norm | 60 | -0.0667727456 | -0.1290671080 | no |
| low_norm | 60 | -0.1528460831 | -0.2419417072 | no |
| duplicate_projection | 0 | 0.0000000000 | 0.0000000000 | n/a |
| similar_projection | 0 | 0.0000000000 | 0.0000000000 | n/a |
| projection_consistent | 237 | 0.0000000000 | 0.0000000000 | sanity pass |

Projection-row diagnostics found no duplicate or near-duplicate projection-row
explanation for the failure:

```text
active_fraction     1.0
duplicate_fraction  0.0
similar_fraction    0.0
```

## Verdict

```text
D10R_V8_WEAK_STATE_IDENTITY_FAIL
```

Interpretation:

- The beta.8 checkpoint has a stable raw MO improvement.
- The random projection null passes.
- The no-network random-state null passes.
- The projection-consistent shuffle stays at zero, so the shuffle diagnostic is
  internally sane.
- The arbitrary shared state shuffle beats beta.8 strongly.
- Active, high-norm, and low-norm row shuffles also fail.
- The failure is not explained by duplicate or near-duplicate projection rows.

Therefore beta.8 is a real research finding, but it is not a release-candidate
network under the hardened D10r-v8 artifact gate.

## Release Impact

D10s, H512, and H8192 release-path work remain blocked for the beta.8 branch.
The next useful work is not another beta.8 promotion test. It should be a pivot:

- state-identity-aware candidate search,
- projection/readout redesign,
- wiring/training redesign with state identity in the acceptance gate,
- or a new candidate family that passes D10r before scaling.

## Progress Map

```text
GLOBAL RELEASE-READY AI MAP

[1] beta.8 H384 improvement
    DONE

[2] random projection artifact
    PASS

[3] no-network random artifact
    PASS

[4] state identity gate
    FAIL: weak state identity

[5] D10s wiring-prior sweep
    BLOCKED for beta.8 release path

[6] H512/H8192
    BLOCKED for beta.8 release path

[7] next
    pivot to state-anchored search/redesign
```
