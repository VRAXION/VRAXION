# E56B Text Field Max Capacity Search Falloff Sweep Result

```text
decision = e56b_no_clean_capacity_within_3x_gate
checker_failure_count = 0
sample_only_checker_passed = true
run_id = cc93384b31ff59ca
gradient_descent_used = false
optimizer_used = false
backprop_used = false
```

## Purpose

E56A showed that the Text Field works and that overlap is required for boundary
robustness. E56B measured the capacity/search-cost curve before locking a
maximum trainable Text Field size.

The user-provided operating constraint was:

```text
acceptable slowdown = 1x to 3x
```

The probe therefore distinguishes:

```text
largest useful configuration inside the 3x budget
first clean configuration overall
search-space falloff beyond useful capacity
```

## Result

No tested Text Field configuration reached the clean success threshold inside
the 1x-3x slowdown budget.

The largest useful configuration inside the budget was:

```text
selected_max_trainable = gate_edge_5x256_o64
unique_coverage_bytes = 1024
work_bytes = 1280
slowdown = 2.750
success = 0.857143
false_commit_rate = 0.000000
```

The first clean configuration overall was:

```text
first_clean_overall = wide_4x512_o128
unique_coverage_bytes = 1664
work_bytes = 2048
slowdown = 4.500
success = 1.000000
false_commit_rate = 0.000000
```

## Capacity Curve

| configuration | unique coverage bytes | work bytes | slowdown | success | attempts to 95 | false commit |
|---|---:|---:|---:|---:|---:|---:|
| fast_default_4x128_o32 | 416 | 512 | 1.000 | 0.417389 | 999999 | 0.000000 |
| normal_4x256_o64 | 832 | 1024 | 2.000 | 0.814668 | 999999 | 0.000000 |
| gate_edge_5x256_o64 | 1024 | 1280 | 2.750 | 0.857143 | 999999 | 0.000000 |
| max_v1_8x256_o64 | 1600 | 2048 | 5.600 | 1.000000 | 4032 | 0.000000 |
| wide_4x512_o128 | 1664 | 2048 | 4.500 | 1.000000 | 3240 | 0.000000 |
| wide_8x512_o128 | 3200 | 4096 | 21.000 | 0.915944 | 999999 | 0.084056 |
| oversize_8x1024_o256 | 6400 | 8192 | 92.400 | 0.753954 | 999999 | 0.246046 |

## Interpretation

The bottleneck was search-space falloff before a hard hardware bottleneck.

Within the accepted 1x-3x slowdown window, increasing frame capacity improved
coverage and success but did not reach clean behavior. The first clean option
required accepting approximately 4.5x slowdown. Larger oversize fields degraded:
they became slower, less stable, and started producing false commits.

This means the practical lock should not be "make the Text Field as large as
possible." Capacity needs a gate.

## Recommendation

```text
runtime default:
  E56A 4x128 overlap32

strict 3x max:
  5x256 overlap64
  use only as capped max / longer-input mode
  not clean enough to be the only long-context strategy

clean max v1 if >3x is allowed:
  4x512 overlap128
  first tested clean configuration
  slowdown = 4.5x
```

If the final runtime must stay within the 3x slowdown limit, longer text should
be handled by multiple cycles, retrieval/ASK behavior, or active evidence
selection rather than by expanding the single Text Field beyond the gate.

If a 4.5x mode is acceptable, `wide_4x512_o128` is the first clean max-capacity
candidate.

## Boundary

E56B is a deterministic Text Field capacity/search-cost sweep. It does not
claim raw language reasoning, AGI, consciousness, deployment quality, or
model-scale behavior.
