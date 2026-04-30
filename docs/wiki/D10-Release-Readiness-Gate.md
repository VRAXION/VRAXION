# D10 Release Readiness Gate

Status date: 2026-04-30

This page records the current D10 gate between the H384 beta.8 research
checkpoint and any future release-ready / high-H AI claim.

## Current Verdict

```text
D10R_V8_WEAK_STATE_IDENTITY_FAIL
```

Meaning:

- `seed2042_improved_generalist_v1` remains a real H384 research finding.
- It is not a release-candidate checkpoint.
- The remaining blocker is not random projection or no-network random state.
- The blocker is state identity: `state_shuffle_shared` can beat the real
  beta.8 signal.

## Evidence Snapshot

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

## What Is Blocked

Blocked:

- beta.8 release-candidate claim
- H512/H8192 release-path scaling from beta.8
- any "release-ready AI" claim based on beta.8

Not blocked:

- state-anchored candidate search
- projection/readout redesign
- wiring/training redesign with artifact controls inside the acceptance loop
- D10u focused ladder work

## Current Next Step

```text
D10u focused state-anchored ladder
```

Goal:

```text
convert weak seed2042/seed4042 state-anchored signals
into a near/strict trusted candidate.
```

The candidate must improve the real task and beat D10r-v8 artifact controls,
including `state_shuffle_shared`, before H512/H8192 work is unlocked.

## Global Plan Map

```text
[1] beta.8 H384 improvement
    DONE

[2] causal mechanism
    DONE: edge + threshold co-adaptation

[3] beta.8 artifact hardening
    DONE: state identity fail

[4] D10u state-anchored search
    CURRENT: weak seed4042 signal

[5] D10r-v8 confirm
    BLOCKED until near/strict candidate exists

[6] H512/H8192
    BLOCKED until D10r-v8 confirm passes

[7] release-ready AI candidate
    BLOCKED
```

## Canonical Repo References

- `docs/research/PHASE_D10R_V8_STATE_IDENTITY_GATE.md`
- `docs/research/PHASE_D10U_STATE_ANCHORED_WIRING_PRIOR.md`
- `tools/_scratch/d10r_hardened_eval.py`
- `tools/_scratch/d10s_wiring_prior_sweep.py`
