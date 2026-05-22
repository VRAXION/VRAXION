# STABLE_LOOP_PHASE_LOCK_040_ROUTE_GRAMMAR_CANARY_TRAINING_ROLLOUT Result

Status: smoke complete.

040 tests whether the experimental route-grammar API remains safe under a
feature-flagged canary rollout. It does not enable default training, promote
public beta, or claim production readiness.

## Run

Quick selector:

```powershell
cargo run -p instnct-core --example phase_lane_route_grammar_canary_training_rollout --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_040_route_grammar_canary_training_rollout/quick ^
  --seeds 2026 ^
  --eval-examples 512 ^
  --widths 8,12 ^
  --path-lengths 4,8,16,24 ^
  --ticks-list 8,16,24,32 ^
  --heartbeat-sec 15
```

Smoke:

```powershell
cargo run -p instnct-core --example phase_lane_route_grammar_canary_training_rollout --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_040_route_grammar_canary_training_rollout/smoke ^
  --seeds 2026,2027,2028 ^
  --eval-examples 1024 ^
  --widths 8,12,16 ^
  --path-lengths 4,8,16,24,32 ^
  --ticks-list 8,16,24,32,48 ^
  --heartbeat-sec 30
```

Smoke rows:

```text
34650
```

## Verdict

```text
CANARY_TRAINING_ROLLOUT_POSITIVE
CANARY_5_PERCENT_HAS_SIGNAL
CANARY_25_PERCENT_HAS_SIGNAL
CANARY_50_PERCENT_HAS_SIGNAL
CANARY_LONG_HORIZON_STABLE
CANARY_MIXED_TASKS_STABLE
CANARY_LEARNS_SUCCESSOR_STRUCTURE
CANARY_ROUTE_GATING_WORKS
CANARY_ROLLBACK_GATE_WORKS
CANARY_REGRESSION_CORPUS_PASSES
CANARY_OVERHEAD_ACCEPTABLE
CANARY_SHADOW_MODE_SAFE_NOT_SUFFICIENT
CANARY_NO_NON_ROUTE_REGRESSION
CANARY_0_PERCENT_CONTROL_FAILS
RANDOM_ROUTE_GRAMMAR_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Exposure Matrix

The 5%, 25%, and 50% canary exposure arms passed the route-structure gate:

```text
CANARY_ROUTE_GRAMMAR_EXPOSURE_5:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000

CANARY_ROUTE_GRAMMAR_EXPOSURE_25:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000

CANARY_ROUTE_GRAMMAR_EXPOSURE_50:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000
```

The 0% canary control reproduced the known no-grammar failure:

```text
CANARY_ROUTE_GRAMMAR_EXPOSURE_0:
  sufficient_tick_final_accuracy = 0.985
  long_path_accuracy = 0.968
  family_min_accuracy = 0.000
  route_order_accuracy = 0.643
  retained_successor_accuracy = 0.651
  missing_successor_count = 9.762
```

## Long, Mixed, And Non-Route Canary

Long-horizon and mixed-task canaries passed:

```text
CANARY_LONG_HORIZON_EXPOSURE_25:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000

CANARY_MIXED_TASK_EXPOSURE_25:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
```

Non-route canary arms did not show regression in this bounded matrix:

```text
CANARY_NON_ROUTE_EXPOSURE_25:
  sufficient_tick_final_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000

non_route_regression_delta = 0.000
```

## Rollback And Regression Corpus

Rollback and regression corpus arms passed:

```text
CANARY_ROLLBACK_ON_REGRESSION:
  sufficient_tick_final_accuracy = 1.000
  family_min_accuracy = 1.000
  route_order_accuracy = 1.000

CANARY_ROLLBACK_ON_OVERHEAD:
  sufficient_tick_final_accuracy = 1.000
  family_min_accuracy = 1.000
  route_order_accuracy = 1.000

CANARY_REGRESSION_CORPUS:
  sufficient_tick_final_accuracy = 1.000
  family_min_accuracy = 1.000
  route_order_accuracy = 1.000
```

Shadow mode stayed safe but insufficient as a solved transport path:

```text
CANARY_SHADOW_MODE:
  sufficient_tick_final_accuracy = 0.119
  long_path_accuracy = 0.141
  family_min_accuracy = 0.000
  route_order_accuracy = 0.643
  missing_successor_count = 9.762
```

## Canary Gate

The canary gate wrote:

```text
canary_gate_pass = true
canary_default_enabled = false
default_training_enabled = false
canary_route_task_delta = 1.000
canary_mixed_task_delta = 1.000
canary_ood_task_delta = 1.000
canary_long_horizon_task_delta = 1.000
false_route_activation_rate = 0.000
route_api_overuse_rate = 0.040
task_type_precision = 1.000
task_type_recall = 1.000
compute_overhead_ratio = 1.08
memory_overhead_ratio = 1.04
```

The bounded learning/search profile preserved the 038/039 efficiency pattern:

```text
NO_ROUTE_GRAMMAR_BASELINE:
  mean steps_to_95 = 120
  candidate_delta_nonzero_fraction = 0.34
  positive_delta_fraction = 0.00

CANARY_ROUTE_GRAMMAR_EXPOSURE_5:
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

040 supports:

```text
route grammar is canary-rollout ready in the tested bounded training matrix
```

040 does not support:

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
