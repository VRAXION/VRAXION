# Phase D14 H384 State-Anchored Basin Atlas

Date: 2026-05-01

## Summary

D14 tested whether the H384 landscape contains more than the already packaged
D13 `top_01` research checkpoint. It re-scanned the known hot D9.3 tile parents
from the D13 checkpoint and filtered the top candidates with the same D10r-v8
artifact/state-identity controls that made `top_01` trustworthy.

Bounded run root:

```text
output/phase_d14_h384_state_anchored_basin_atlas_20260501/bounded_fast_confirm
```

Verdict:

```text
D14_MULTI_BASIN_SIGNAL
```

This is a positive H384 atlas signal, not yet a promotion-grade release claim.
It says there are at least two state-anchored child-basin candidates in the
`9_26` parent region, and the top control tile also remains very strong. That
means the result is useful, but still mixed: it is not a clean "only child
tiles win" map.

## Run Shape

Quadtree rescan:

```text
start checkpoint = output/releases/v5.0.0-beta.10/seed2042_top01_h384_research.ckpt
baseline         = output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt
parents          = 9_26, 12_29, 11_16
radii            = 4, 8, 16, 32
mutation types   = edge, threshold
samples/tile     = 2
eval_len         = 1000
eval seeds       = 971001..971004
rows             = 384
runtime          = 716.4 s
```

State-anchor filter:

```text
top candidates   = 6
eval_len         = 256
eval seeds       = 971011..971014
control repeats  = 1
controls         = random_projection_null,
                   state_shuffle_shared,
                   state_shuffle_projection_consistent,
                   no_network_random_state
```

Focused confirm was rescoped from `eval_len=4000, 8 seeds, 2 repeats` because
the observed ETA was closer to 2.5-3 hours. The bounded confirm used:

```text
selected ranks   = 2, 4, 6
eval_len         = 1000
eval seeds       = 971021..971024
control repeats  = 1
```

## Results

Filter results:

| rank | parent | child | control | verdict | trusted CI low | real CI low |
|---:|---|---|---|---|---:|---:|
| 1 | 9_26 | 19_53 | false | D10R_V8_STATE_IDENTITY_PASS | +0.093604 | +0.132722 |
| 2 | CONTROL | 18_54 | true | D10R_V8_STATE_IDENTITY_PASS | +0.325604 | +0.350703 |
| 3 | 9_26 | 18_53 | false | D10R_UNDERPOWERED_NEEDS_LONGER_EVAL | -0.004580 | +0.035227 |
| 4 | 9_26 | 18_53 | false | D10R_V8_STATE_IDENTITY_PASS | +0.132148 | +0.151388 |
| 5 | 9_26 | 19_53 | false | D10R_V8_STATE_IDENTITY_PASS | +0.064542 | +0.092342 |
| 6 | 9_26 | 19_53 | false | D10R_V8_STATE_IDENTITY_PASS | +0.152836 | +0.181325 |

Focused confirm:

| rank | parent | child | control | verdict | trusted CI low | real CI low |
|---:|---|---|---|---|---:|---:|
| 2 | CONTROL | 18_54 | true | D10R_V8_STATE_IDENTITY_PASS | +0.348034 | +0.379728 |
| 4 | 9_26 | 18_53 | false | D10R_V8_STATE_IDENTITY_PASS | +0.085532 | +0.117314 |
| 6 | 9_26 | 19_53 | false | D10R_V8_STATE_IDENTITY_PASS | +0.163545 | +0.183577 |

## Interpretation

What improved:

- D13 had one packaged H384 checkpoint.
- D14 found two additional state-anchored child candidates under the same
  artifact-control family.
- Both confirmed non-control child candidates are in parent `9_26`, with child
  regions `18_53` and `19_53`.

What remains unresolved:

- The best control tile `18_54` is stronger than the non-control child rows.
- Therefore the map still has a wide/projection-shadow component.
- This is not yet a clean multi-parent basin atlas.
- This is not promotion-grade until a new candidate passes `eval_len=16000`,
  30 fresh seeds.

## Release-Ready Progress

```text
Long-horizon release-ready AI:
[========__] 80%

[1] One proven H384 checkpoint
    DONE: D13 top_01 packaged

[2] More H384 basin signal
    DONE: D14 found two state-anchored non-control child regions

[3] Main caution
    OPEN: strong control tile also passes

[4] Next gate
    run 16k/30-seed confirm for rank 6 or rank 4,
    plus keep the rank 2 control as an adversarial comparator
```

## Next Step

Do not jump to H512 from this result alone. The next clean gate is:

```text
D14b 16k / 30 fresh-seed confirm
target: rank 6 non-control child 9_26 / 19_53
adversarial comparator: rank 2 control child 18_54
```

If rank 6 survives and the control comparator stays interpretable, H384 has a
real basin-family signal. If rank 6 fails or the control dominates, D14 should
be treated as a useful atlas signal but not a release-ready capability step.
