# E64 Field Size And Agency View Sweep Result

Status: completed and checker validated.

## Decision

```text
decision = e64_near_28f_32g_20x80_default_confirmed
checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = cf0250981306c06e
row_count = 67584
gradient_descent_used = false
optimizer_used = false
backprop_used = false
```

## Locked Sizing Recommendation

```text
default Flow Field   = 28x28 cells
default Ground Field = 32x32 cells
Proposal Field       = 20 slots x 80 bits
Agency View          = 896 mechanical summary bits

extended mode        = 32x32 Flow/Ground, Agency-selected only
research ceiling     = 48x48, not default
avoid default        = 64x64 overcapacity
```

The first clean default was not the largest matrix. The clean frontier was:

```text
near_28f_32g_20x80_default:
  success = 1.000000
  false_commit_rate = 0.000000
  net_utility = 0.791282
  cost = 2.58

wide_32x32_20x80:
  success = 1.000000
  false_commit_rate = 0.000000
  net_utility = 0.709405
  cost = 3.18
```

So the lock is:

```text
Use 28x28 Flow plus 32x32 Ground as the default body.
Keep Proposal Field at 20x80-bit slots.
Keep Agency View at 896 bits.
Use 32x32 Flow/Ground only as an Agency-selected extended mode.
```

## Key Controls

| system | success | false commit | missed commit | net utility | interpretation |
|---|---:|---:|---:|---:|---|
| asymmetric_24f_32g_20x80_control | 0.984701 | 0.000000 | 0.015299 | 0.794920 | Almost enough, but 24x24 Flow still misses some rows. |
| near_28f_32g_20x80_default | 1.000000 | 0.000000 | 0.000000 | 0.791282 | First fully clean default. |
| wide_32x32_20x80 | 1.000000 | 0.000000 | 0.000000 | 0.709405 | Clean, but lower net utility due cost/search load. |
| proposal_width_64_control | 0.714355 | 0.159017 | 0.126628 | 0.280916 | 64-bit proposal records are too narrow. |
| proposal_starved_32x32 | 0.070312 | 0.500000 | 0.429688 | -0.954879 | Large fields do not fix too few/narrow proposal slots. |
| agency_starved_32x32 | 0.251953 | 0.498047 | 0.250000 | -0.729339 | Large fields are unsafe if Agency cannot see enough. |
| oversized_64x64_32x80 | 0.932292 | 0.067708 | 0.000000 | 0.212120 | Overcapacity creates decoy/false-commit pressure. |

## Interpretation

E64 found three distinct size boundaries:

```text
1. 24x24 Flow is slightly too small for the current stress mix.
2. 64-bit proposal slots are too narrow; 80-bit slots were needed.
3. Agency view must grow with proposal pressure; starving Agency creates false commits.
```

The result argues against simply maxing every matrix. The 64x64 overcapacity
control had more raw room, but worse false commits and much worse net utility.

## Boundary

E64 is a deterministic symbolic/numeric sizing probe for the VRAXION
Flow/Ground/Proposal/Agency interfaces. It does not test raw language
reasoning, deployed model behavior, consciousness, AGI, or model scale.
