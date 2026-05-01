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

## Focused Ladder Update

Output root:

```text
output/phase_d10u_focused_ladder_20260430/bounded
```

Run shape:

```text
mode: ladder
eval_len: 128
eval_seeds: 970001,970002,970003
control_repeats: 1
checkpoints: seed_2042, seed_4042
arms: edge_threshold_coadapted, random_sparse_baseline
ladder_rounds: 3
proposals_per_round: 6
elapsed: 129.70s
```

Verdict:

```text
D10U_SEED2042_NEAR_OR_STRICT_SIGNAL
```

Best exported scout candidate:

| seed | arm | class | smooth | accuracy | echo | unigram | selectivity CI low |
|---|---|---|---:|---:|---:|---:|---:|
| seed_2042 | edge_threshold_coadapted | STRICT_TRUSTED | +0.025639 | +0.007813 | +0.000000 | +0.081762 | +0.143935 |

Additional ladder signal:

| seed | arm | class | smooth | accuracy | unigram | note |
|---|---|---|---:|---:|---:|---|
| seed_4042 | edge_threshold_coadapted | WEAK_STATE_ANCHORED | +0.020501 | +0.015625 | +0.035454 | non-seed state-anchor signal, still not strict |

Export sanity:

```text
RELOAD_OK 8
```

Interpretation:

- D10u ladder machinery works and exports reloadable checkpoints.
- A seed2042 `STRICT_TRUSTED` scout candidate exists under short eval.
- This is not release evidence because `eval_len=128` is still scout-level.
- The next gate is a longer D10r-v8 confirm on the exported top candidate.

## Bounded D10r-v8 Confirm

Output root:

```text
output/phase_d10u_top01_d10r_confirm_20260430/bounded_1000
```

Run shape:

```text
target: top_01_seed_2042_edge_threshold_coadapted.ckpt
baseline: seed_2042 D7 H=384 baseline
eval_len: 1000
eval_seeds: 970021,970022,970023,970024
control_repeats: 2
controls: random_projection_null, state_shuffle_shared,
          state_shuffle_projection_consistent, no_network_random_state
elapsed: 426.61s
```

Verdict:

```text
D10R_V8_STATE_IDENTITY_PASS
```

Confirm metrics:

| metric | value |
|---|---:|
| real MO delta mean | +0.212136 |
| real MO delta CI low | +0.184054 |
| trusted MO mean | +0.196925 |
| trusted MO CI low | +0.170111 |
| median selectivity CI low | +0.185850 |
| state_shuffle_shared bound CI low | +0.184446 |
| random_projection_null bound CI low | +0.170111 |
| no_network_random_state bound CI low | +0.181960 |

State-identity diagnostics:

| diagnostic | result |
|---|---|
| projection-consistent shuffle | exactly zero drift |
| duplicate projection rows | 0 rows |
| similar projection rows | 0 rows |
| active-row shuffle bound | pass |
| high-norm row shuffle bound | pass |
| low-norm row shuffle bound | pass |

Interpretation:

- The top_01 candidate passes the same D10r-v8 state-identity gate that blocked
  beta.8.
- The result is not explained by random projection, no-network random state, or
  projection-consistent shuffle artifacts.
- This reopens the release-candidate path, but does not finish it. The confirm
  is still bounded: 4 eval seeds at `eval_len=1000`.

Next gates:

- run final promotion-grade confirmation at `eval_len=16000` with 30 fresh
  seeds
- only after those pass should this become a release-candidate checkpoint

## Longer 4k D10r-v8 Confirm

Output root:

```text
output/phase_d10u_top01_d10r_confirm_20260430/confirm_4000_4seed
```

Run shape:

```text
target: top_01_seed_2042_edge_threshold_coadapted.ckpt
baseline: seed_2042 D7 H=384 baseline
eval_len: 4000
eval_seeds: 970031,970032,970033,970034
control_repeats: 2
elapsed: 2456.36s
```

Verdict:

```text
D10R_V8_STATE_IDENTITY_PASS
```

Confirm metrics:

| metric | value |
|---|---:|
| real MO delta mean | +0.185918 |
| real MO delta CI low | +0.185742 |
| trusted MO mean | +0.159960 |
| trusted MO CI low | +0.131453 |
| median selectivity CI low | +0.186294 |
| state_shuffle_shared bound CI low | +0.134342 |
| random_projection_null bound CI low | +0.174151 |
| no_network_random_state bound CI low | +0.185484 |

State-identity diagnostics:

| diagnostic | result |
|---|---|
| projection-consistent shuffle | exactly zero drift |
| duplicate projection rows | 0 rows |
| similar projection rows | 0 rows |
| active-row shuffle bound | pass |
| high-norm row shuffle bound | pass |
| low-norm row shuffle bound | diagnostic weak / non-blocking |

Interpretation:

- The top_01 candidate survived the longer `eval_len=4000` gate with a large
  positive artifact-adjusted margin.
- The main beta.8 blocker, `state_shuffle_shared`, is no longer blocking for
  this candidate.
- This is the strongest D10u result so far, but it still has only 4 fresh eval
  seeds. The release-candidate gate remains a 30-fresh-seed, long-eval confirm.

## Promotion-Grade 16k / 30-Seed Confirm

Output root:

```text
output/phase_d10u_top01_d10r_confirm_20260430/confirm_16000_30seed_sharded_v2
```

Run shape:

```text
target: top_01_seed_2042_edge_threshold_coadapted.ckpt
baseline: seed_2042 D7 H=384 baseline
eval_len: 16000
eval_seeds: 970101..970130
control_repeats: 2
controls: random_projection_null, state_shuffle_shared,
          state_shuffle_projection_consistent, no_network_random_state
runner: one shard / one eval seed at a time, fail-stop
```

Verdict:

```text
D10U_TOP01_16K_SHARDED_PASS
```

Aggregate result:

| Metric | Value |
|---|---:|
| completed shards | 30 / 30 |
| failed shards | 0 / 30 |
| per-shard verdict | 30 x `D10R_V8_STATE_IDENTITY_PASS` |
| blocking control families | none |
| minimum trusted MO CI low | +0.084493 |
| minimum real MO delta CI low | +0.178087 |

Weakest shard:

```text
shard_24 / seed 970125
trusted_mo_ci_low: +0.084493
real_mo_delta_ci_low: +0.178160
blocking_control_families: []
```

Decision:

```text
D10U_TOP01_RELEASE_CANDIDATE_RESEARCH_CHECKPOINT
```

Interpretation:

- `top_01` passed the promotion-grade D10r-v8 artifact gate.
- The result survived long evaluation (`eval_len=16000`) and 30 fresh seeds.
- The state-identity blocker that rejected beta.8 does not reject this
  candidate.
- This promotes `top_01` to a release-candidate research checkpoint.
- This does not make it the public mainline grower replacement and does not
  by itself prove H512/H8192 universality.

Next packaging gate:

```text
artifact copy + checksum + reload smoke + compact release-candidate docs
```

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

[4] state-anchored strict scout candidate
    DONE
    result: seed2042 top_01 STRICT_TRUSTED at eval_len=128

[5] bounded D10r-v8 confirm
    DONE
    result: top_01 passes state identity at eval_len=1000

[6] longer D10r-v8 confirm
    DONE
    result: top_01 passes state identity at eval_len=4000

[7] promotion-grade confirm
    DONE
    result: top_01 passes eval_len=16000, 30 fresh seeds

[8] release-candidate package
    NEXT
    goal: stable artifact path, checksum, reload smoke, public docs

[9] H512/H8192
    UNBLOCKED FOR PILOT PLANNING
    still blocked for release claims until scaling evidence exists
```
