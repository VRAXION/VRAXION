# Phase D16B Context Climb Smoke

Date: 2026-05-02

## Summary

D16B adds `context-climb` to the direct landscape runner. The mode searches
around the trusted H384 `top_01` checkpoint with edge/threshold mutations and an
explicit context-carrying objective.

The smoke run completed and produced all required artifacts:

```text
output/phase_d16b_context_climb_smoke_20260502/
```

## Run Shape

```text
start checkpoint = output/releases/v5.0.0-beta.10/seed2042_top01_h384_research.ckpt
H                = 384
mode             = context-climb
mutation scope   = edge,threshold
radii            = 4,8,16,32
climbers         = 2
steps            = 10
eval_len         = 512
eval seeds       = 974001,974002
export top       = 2
```

## Result

```text
D16B_CONTEXT_TRADEOFF
```

Smoke facts:

| Metric | Value |
|---|---:|
| proposals | 20 |
| accepted | 0 |
| context signal candidates | 1 |
| tradeoff candidates | 8 |
| artifact candidates | 10 |

The best safe-looking candidate did not beat the current target-gain threshold.
The strongest context candidate had safety regressions and was correctly
classified as a tradeoff.

## Reload Sanity

The exported checkpoints reload through `chain_diagnosis`.

| Candidate | D16 gate | Context-dependent predictions | Note |
|---|---|---:|---|
| `top_01.ckpt` | `D16_CONTEXT_BLOCKED` | 0 / 4 | safe-looking but no fixed-probe context gain |
| `top_02.ckpt` | `D16_PARTIAL_CONTEXT_SIGNAL` | 3 / 4 | context appears, but safety tradeoff blocks promotion |

## Interpretation

This is a useful smoke result: context behavior is reachable by local
edge/threshold mutation, but the first reachable context behavior trades off
existing safety metrics. The next run should keep the same mode and increase the
search budget, not switch back to high-H brute force.

## Next Step

Run the bounded D16B main:

```text
mo_climbers = 12
mo_steps    = 80
eval_len    = 1000
eval seeds  = 974001..974008
```

Promotion remains blocked until a candidate passes:

- D16 context gate,
- D10r-v8 artifact/state gate,
- and a longer confirm.
