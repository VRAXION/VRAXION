# STABLE_LOOP_PHASE_LOCK_039_FEATURE_FLAG_TRAINING_ROLLOUT_MATRIX Result

Status: smoke complete.

039 tests whether the experimental route-grammar API remains useful and safe
when exposed as a feature-flagged helper across a rollout matrix. It does not
enable default training, promote public beta, or claim production readiness.

## Run

Quick selector:

```powershell
cargo run -p instnct-core --example phase_lane_feature_flag_training_rollout_matrix --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_039_feature_flag_training_rollout_matrix/quick ^
  --seeds 2026 ^
  --eval-examples 512 ^
  --widths 8,12 ^
  --path-lengths 4,8,16,24 ^
  --ticks-list 8,16,24,32 ^
  --heartbeat-sec 15
```

Smoke:

```powershell
cargo run -p instnct-core --example phase_lane_feature_flag_training_rollout_matrix --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_039_feature_flag_training_rollout_matrix/smoke ^
  --seeds 2026,2027,2028 ^
  --eval-examples 1024 ^
  --widths 8,12,16 ^
  --path-lengths 4,8,16,24,32 ^
  --ticks-list 8,16,24,32,48 ^
  --heartbeat-sec 30
```

Smoke rows:

```text
26775
```

## Verdict

```text
FEATURE_FLAG_ROLLOUT_MATRIX_POSITIVE
FEATURE_FLAG_IMPROVES_ROUTE_TASKS
FEATURE_FLAG_IMPROVES_MIXED_TASKS
FEATURE_FLAG_GENERALIZES_OOD
FEATURE_FLAG_HANDLES_LONG_HORIZON
FEATURE_FLAG_LEARNS_SUCCESSOR_STRUCTURE
FEATURE_FLAG_ROUTE_GATING_WORKS
FEATURE_FLAG_COST_CAP_ACCEPTABLE
NO_NON_ROUTE_REGRESSION_OBSERVED
FALSE_ROUTE_ACTIVATION_CONTROL_FAILS
DIAGNOSTICS_SHADOW_ONLY_INSUFFICIENT
FEATURE_FLAG_OFF_CONTROL_FAILS
RANDOM_ROUTE_GRAMMAR_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Core Result

Feature-flag ON route arms passed the route-structure gate:

```text
ROUTE_GRAMMAR_FEATURE_FLAG_ON_ROUTE_TASKS:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000

ROUTE_GRAMMAR_FEATURE_FLAG_ON_MIXED_TASKS:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000

ROUTE_GRAMMAR_FEATURE_FLAG_ON_OOD_TASKS:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000

ROUTE_GRAMMAR_FEATURE_FLAG_ON_LONG_HORIZON:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000
```

The feature-flag OFF and no-route-grammar baselines reproduced the known
aggregate-good but route-grammar-bad failure:

```text
NO_ROUTE_GRAMMAR_BASELINE:
  sufficient_tick_final_accuracy = 0.985
  long_path_accuracy = 0.968
  family_min_accuracy = 0.000
  route_order_accuracy = 0.643
  retained_successor_accuracy = 0.651
  missing_successor_count = 9.762

ROUTE_GRAMMAR_FEATURE_FLAG_OFF:
  sufficient_tick_final_accuracy = 0.985
  long_path_accuracy = 0.968
  family_min_accuracy = 0.000
  route_order_accuracy = 0.643
  retained_successor_accuracy = 0.651
  missing_successor_count = 9.762
```

## Gating And Interference

Route-only gated, auto-detect, and cost-capped arms passed:

```text
ROUTE_GRAMMAR_FEATURE_FLAG_ROUTE_ONLY_GATED:
  sufficient_tick_final_accuracy = 1.000
  family_min_accuracy = 1.000
  route_order_accuracy = 1.000

ROUTE_GRAMMAR_FEATURE_FLAG_AUTO_DETECT_ROUTE:
  sufficient_tick_final_accuracy = 1.000
  family_min_accuracy = 1.000
  route_order_accuracy = 1.000

ROUTE_GRAMMAR_FEATURE_FLAG_COST_CAPPED:
  sufficient_tick_final_accuracy = 1.000
  family_min_accuracy = 1.000
  route_order_accuracy = 1.000
```

False-positive stress did not receive a positive route claim:

```text
ROUTE_GRAMMAR_FEATURE_FLAG_FALSE_POSITIVE_STRESS:
  sufficient_tick_final_accuracy = 0.985
  family_min_accuracy = 0.000
  route_order_accuracy = 0.643
  missing_successor_count = 9.762
```

Diagnostics shadow-only is insufficient as a full rollout mode:

```text
ROUTE_GRAMMAR_FEATURE_FLAG_DIAGNOSTICS_SHADOW_ONLY:
  sufficient_tick_final_accuracy = 0.119
  long_path_accuracy = 0.141
  family_min_accuracy = 0.000
  route_order_accuracy = 0.643
  missing_successor_count = 9.762
```

The non-route regression control stayed neutral in this bounded matrix:

```text
non_route_regression_delta = 0.000
```

## Rollout Gate

The feature-flag gate wrote:

```text
rollout_gate_pass = true
feature_flag_default_enabled = false
default_training_enabled = false
route_task_delta = 1.000
mixed_task_delta = 1.000
ood_task_delta = 1.000
long_horizon_task_delta = 1.000
false_route_activation_rate = 0.000
route_api_overuse_rate = 0.040
task_type_precision = 1.000
task_type_recall = 1.000
compute_overhead_ratio = 1.08
memory_overhead_ratio = 1.04
```

Learning/search profiles retained the 038 sample-efficiency pattern:

```text
NO_ROUTE_GRAMMAR_BASELINE:
  mean steps_to_95 = 120
  candidate_delta_nonzero_fraction = 0.34
  positive_delta_fraction = 0.00

ROUTE_GRAMMAR_FEATURE_FLAG_ON_ROUTE_TASKS:
  mean steps_to_95 = 60
  candidate_delta_nonzero_fraction = 0.82
  positive_delta_fraction = 0.64
```

## Controls

Random controls failed:

```text
RANDOM_ROUTE_GRAMMAR_CONTROL:
  sufficient_tick_final_accuracy = 0.641
  long_path_accuracy = 0.356
  family_min_accuracy = 0.000
  wrong_if_delivered_rate = 0.389
  route_order_accuracy = 0.409

RANDOM_PHASE_RULE_CONTROL:
  sufficient_tick_final_accuracy = 0.495
  long_path_accuracy = 0.490
  family_min_accuracy = 0.000
  wrong_if_delivered_rate = 0.383
```

## Claim Boundary

039 supports:

```text
the experimental route-grammar API is useful and controlled as a feature-flagged
training/search helper in the tested rollout matrix
```

039 does not support:

```text
default training enablement
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
biological/FlyWire equivalence
physical quantum behavior
```
