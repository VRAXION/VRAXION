# Phase D10u State-Anchored Wiring Prior Search

Date: 2026-04-30

## Purpose

D10r-v8 showed that beta.8 is a real H=384 research finding but not a
release-candidate checkpoint, because `state_shuffle_shared` can beat the real
checkpoint. D10u pivots from "promote beta.8" to "search for candidates that
are good and state-anchored from the start."

The search objective is now:

```text
candidate real MO improvement > artifact-null controls,
including state_shuffle_shared.
```

This is not a release run. It is a short proof that the wiring-prior search can
use D10r-v8 controls inside ranking instead of only after the fact.

## Implementation

`tools/_scratch/d10s_wiring_prior_sweep.py` now supports state-anchored ranking:

- default controls are D10r-v8 artifact controls:
  - `random_projection_null`
  - `state_shuffle_shared`
  - `state_shuffle_projection_consistent`
  - `no_network_random_state`
- diagnostic controls are excluded from worst-control selectivity ranking
- `state_shuffle_projection_consistent` remains a sanity diagnostic
- each candidate records:
  - `hardened_selectivity`
  - `hardened_selectivity_ci_low`
  - `worst_artifact_control`
  - `worst_artifact_margin`
  - `state_anchor_pass`
  - `candidate_class`

Candidate classes:

- `STRICT_TRUSTED`: all strict gates pass and selectivity gate passes
- `NEAR_TRUSTED`: one near miss and selectivity gate passes
- `WEAK_STATE_ANCHORED`: real MO is positive and selectivity gate passes, but
  task gates are still too weak
- `WEAK_SIGNAL`: mean selectivity is positive, but CI gate does not pass
- `REJECT`: no usable state-anchored signal

## Short Scout

Output root:

```text
output/phase_d10u_state_anchored_wiring_20260430/short_scout
```

Run shape:

```text
eval_len: 128
eval_seeds: 970001,970002,970003
control_repeats: 1
selectivity_gate: ci
checkpoints: seed_2042, seed_42, seed_4042
arms: random_sparse_baseline, edge_threshold_coadapted, motif_biased
proposals_per_arm: 6
elapsed: 99.64s
```

Verdict:

```text
D10S_NO_TRUSTED_SIGNAL
```

Top signals:

| seed | arm | class | smooth | accuracy | unigram | selectivity CI low | worst control |
|---|---|---|---:|---:|---:|---:|---|
| seed_2042 | edge_threshold_coadapted | WEAK_STATE_ANCHORED | +0.013509 | +0.000000 | +0.006381 | +0.012567 | no_network_random_state |
| seed_4042 | edge_threshold_coadapted | WEAK_STATE_ANCHORED | +0.004118 | +0.000000 | +0.007596 | +0.007998 | no_network_random_state |
| seed_2042 | edge_threshold_coadapted | WEAK_STATE_ANCHORED | +0.006248 | +0.000000 | +0.004811 | +0.000514 | no_network_random_state |
| seed_4042 | random_sparse_baseline | WEAK_STATE_ANCHORED | +0.000461 | +0.000000 | +0.002415 | +0.003995 | state_shuffle_shared |

## Interpretation

The important result is not the top seed2042 row. The useful signal is that
seed4042 produced weak state-anchored candidates under the D10r-v8 artifact
controls.

This does not unlock H512 or a release path. The candidates still miss the
strict task gates, especially accuracy and smooth magnitude. It does show that
the state-anchor gate is not impossible to satisfy; the next search should
optimize toward strict gates while keeping the same D10r-v8 controls in the
acceptance loop.

## Release Impact

Current status:

```text
No release candidate.
No H512 unlock.
State-anchored search path remains alive.
```

The next useful run is a focused D10u ladder:

- seed targets: seed2042 and seed4042
- arms: edge_threshold_coadapted first, random_sparse_baseline as control
- acceptance: state-anchor pass is mandatory
- objective: raise smooth and accuracy without losing unigram/echo safety
- confirm: any near/strict candidate must pass D10r-v8 with longer eval

## Progress Map

```text
GLOBAL RELEASE-READY AI MAP

[1] beta.8 H384 improvement
    DONE

[2] beta.8 artifact hardening
    DONE
    result: state identity fail

[3] state-anchored search machinery
    DONE
    result: D10u smoke/scout works

[4] weak non-seed signal
    CURRENT
    seed4042 has weak state-anchored signal

[5] focused D10u ladder
    NEXT
    goal: convert weak state-anchored signal into near/strict trusted candidate

[6] D10r-v8 confirm
    BLOCKED until near/strict candidate exists

[7] H512/H8192
    BLOCKED until D10r-v8 confirm passes
```
