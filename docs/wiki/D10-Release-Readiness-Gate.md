# D10 Release Readiness Gate

Status date: 2026-04-30

This page records the current D10 gate between the H384 beta.8 research
checkpoint and any future release-ready / high-H AI claim.

## Current Verdict

```text
D10U_TOP01_BOUNDED_STATE_IDENTITY_PASS
```

Meaning:

- `seed2042_improved_generalist_v1` remains a real H384 research finding.
- It is not a release-candidate checkpoint.
- The beta.8 branch was blocked by state identity.
- D10u produced a new top_01 state-anchored candidate that passes the same
  D10r-v8 state-identity gate at bounded confirm budget.
- This reopens the release-candidate path, but does not finish it. The current
  pass is `eval_len=1000` with 4 fresh eval seeds, not a promotion-grade
  16k/30-seed confirm.

## Evidence Snapshot: beta.8 Blocker

D10r-v8 main:

```text
eval_len:        1000
eval_seeds:      970001..970016
control_repeats: 8
device:          cuda
```

Beta.8 real signal:

```text
real_mo_delta_mean    +0.0144964868
real_mo_delta_ci_low  +0.0144065531
```

Artifact control result:

| Control family | CI low | Status |
|---|---:|---|
| random_projection_null | +0.0102142170 | pass |
| no_network_random_state | +0.0076125350 | pass |
| state_shuffle_shared | -0.1112784165 | fail |

State-zone diagnostics:

| Zone | Rows | CI low | Status |
|---|---:|---:|---|
| active_rows | 237 | -0.0963420603 | fail |
| high_norm | 60 | -0.1290671080 | fail |
| low_norm | 60 | -0.2419417072 | fail |
| duplicate_projection | 0 | 0.0000000000 | no explanation |
| similar_projection | 0 | 0.0000000000 | no explanation |
| projection_consistent | 237 | 0.0000000000 | sanity pass |

## Evidence Snapshot: D10u top_01 Bounded Confirm

D10u focused ladder exported:

```text
top_01_seed_2042_edge_threshold_coadapted.ckpt
```

D10r-v8 bounded confirm:

```text
eval_len:        1000
eval_seeds:      970021..970024
control_repeats: 2
device:          cuda
verdict:         D10R_V8_STATE_IDENTITY_PASS
```

Top_01 margins:

| Metric | Value |
|---|---:|
| real MO delta mean | +0.212136 |
| real MO delta CI low | +0.184054 |
| trusted MO mean | +0.196925 |
| trusted MO CI low | +0.170111 |
| state_shuffle_shared bound CI low | +0.184446 |
| random_projection_null bound CI low | +0.170111 |
| no_network_random_state bound CI low | +0.181960 |

Sanity diagnostics:

```text
projection-consistent shuffle: zero drift
duplicate projection rows:     0
similar projection rows:       0
active/high/low row shuffles:  pass
```

## What Is Blocked

Blocked:

- beta.8 release-candidate claim
- H512/H8192 release-path scaling until top_01 survives promotion-grade confirm
- any "release-ready AI" claim before longer fresh-seed validation

Not blocked:

- top_01 promotion-grade confirm
- state-anchored candidate search as a fallback
- wiring/training redesign with artifact controls inside the acceptance loop

## Current Next Step

```text
D10u top_01 promotion-grade confirm
```

Goal:

```text
prove that top_01 keeps the state-anchored advantage at longer eval budgets
and more fresh eval seeds.
```

Minimum next gates:

```text
eval_len=4000 confirm with more fresh seeds
then eval_len=16000 / 30-seed confirm if 4000 passes
```

## Global Plan Map

```text
[1] beta.8 H384 improvement
    DONE

[2] causal mechanism
    DONE: edge + threshold co-adaptation

[3] beta.8 artifact hardening
    DONE: state identity fail

[4] D10u state-anchored search
    DONE: top_01 strict scout candidate

[5] D10r-v8 confirm
    DONE: top_01 bounded state-identity pass

[6] promotion-grade confirm
    CURRENT: eval_len=4000/16000, more fresh seeds

[7] H512/H8192
    BLOCKED until promotion-grade confirm passes

[8] release-ready AI candidate
    BLOCKED until promotion-grade confirm passes
```

## Canonical Repo References

- `docs/research/PHASE_D10R_V8_STATE_IDENTITY_GATE.md`
- `docs/research/PHASE_D10U_STATE_ANCHORED_WIRING_PRIOR.md`
- `tools/_scratch/d10r_hardened_eval.py`
- `tools/_scratch/d10s_wiring_prior_sweep.py`
