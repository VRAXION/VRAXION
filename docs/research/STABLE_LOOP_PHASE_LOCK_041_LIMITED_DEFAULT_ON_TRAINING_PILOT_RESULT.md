# STABLE_LOOP_PHASE_LOCK_041_LIMITED_DEFAULT_ON_TRAINING_PILOT Result

Status: complete.

041 tested whether the experimental route-grammar API is ready for a limited
default-on training pilot under kill-switch, cost, drift, non-route, and
hard-regression gates. It does not enable production default training, promote a
public beta, or claim production API readiness.

## Commands

Quick selector:

```powershell
cargo run -p instnct-core --example phase_lane_limited_default_on_training_pilot --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_041_limited_default_on_training_pilot/quick ^
  --seeds 2026 ^
  --eval-examples 512 ^
  --widths 8,12 ^
  --path-lengths 4,8,16,24 ^
  --ticks-list 8,16,24,32 ^
  --heartbeat-sec 15
```

Smoke:

```powershell
cargo run -p instnct-core --example phase_lane_limited_default_on_training_pilot --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_041_limited_default_on_training_pilot/smoke ^
  --seeds 2026,2027,2028 ^
  --eval-examples 1024 ^
  --widths 8,12,16 ^
  --path-lengths 4,8,16,24,32 ^
  --ticks-list 8,16,24,32,48 ^
  --heartbeat-sec 30
```

The smoke run wrote 31,500 metric rows with heartbeat progress at 30, 60, and 90
seconds, plus a final done row.

## Verdicts

```text
LIMITED_DEFAULT_ON_PILOT_POSITIVE
DEFAULT_ON_5_PERCENT_REFERENCE_HAS_SIGNAL
DEFAULT_ON_25_PERCENT_REFERENCE_HAS_SIGNAL
LIMITED_DEFAULT_ON_ROUTE_TASKS_STABLE
DEFAULT_ON_MIXED_TASKS_STABLE
DEFAULT_ON_OOD_TASKS_STABLE
DEFAULT_ON_LONG_HORIZON_STABLE
DEFAULT_ON_LEARNS_SUCCESSOR_STRUCTURE
KILL_SWITCH_GATE_WORKS
COST_ENVELOPE_ACCEPTABLE
DRIFT_MONITOR_GATE_WORKS
HARD_REGRESSION_CORPUS_PASSES
NO_NON_ROUTE_DRIFT_OBSERVED
SHADOW_AUDIT_SAFE_NOT_SUFFICIENT
DEFAULT_OFF_CONTROL_FAILS
RANDOM_DEFAULT_ON_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Core Result

The limited default-on pilot arms passed the full route-structure gate:

```text
LIMITED_DEFAULT_ON_ROUTE_TASKS:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000

LIMITED_DEFAULT_ON_MIXED_TASKS:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000

LIMITED_DEFAULT_ON_OOD_TASKS:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000

LIMITED_DEFAULT_ON_LONG_HORIZON:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000
```

The 5% and 25% canary reference arms also passed:

```text
CANARY_5_REFERENCE:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  missing_successor_count = 0.000

CANARY_25_REFERENCE:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  missing_successor_count = 0.000
```

## Controls

The default-off control reproduced the known aggregate-good / structure-bad
failure:

```text
DEFAULT_OFF_CONTROL:
  sufficient_tick_final_accuracy = 0.985
  long_path_accuracy = 0.968
  family_min_accuracy = 0.000
  wrong_if_delivered_rate = 0.015
  route_order_accuracy = 0.643
  retained_successor_accuracy = 0.651
  missing_successor_count = 9.762
```

Shadow audit stayed safe but was not sufficient as a solution:

```text
LIMITED_DEFAULT_ON_SHADOW_AUDIT:
  sufficient_tick_final_accuracy = 0.119
  long_path_accuracy = 0.141
  family_min_accuracy = 0.000
  route_order_accuracy = 0.643
  missing_successor_count = 9.762
```

Random controls failed:

```text
RANDOM_DEFAULT_ON_CONTROL:
  sufficient_tick_final_accuracy = 0.641
  long_path_accuracy = 0.356
  family_min_accuracy = 0.000
  wrong_if_delivered_rate = 0.389
  route_order_accuracy = 0.409
  missing_successor_count = 12.060

RANDOM_PHASE_RULE_CONTROL:
  sufficient_tick_final_accuracy = 0.495
  long_path_accuracy = 0.490
  family_min_accuracy = 0.000
  wrong_if_delivered_rate = 0.383
```

## Gate Metrics

The default-on gate wrote:

```text
default_on_gate_pass = true
limited_default_on_pilot_enabled = true
production_default_training_enabled = false
public_beta_promoted = false
production_api_ready = false
non_route_regression_delta = 0.000
false_route_activation_rate = 0.000
route_api_overuse_rate = 0.040
compute_overhead_ratio = 1.08
memory_overhead_ratio = 1.04
task_type_precision = 1.000
task_type_recall = 1.000
```

## Interpretation

041 supports a bounded research claim:

```text
Route grammar is ready for a limited default-on pilot in the tested training
matrix, with kill-switch, cost, drift, non-route, and hard-regression gates.
```

It does not support:

```text
production default training enablement
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
biological/FlyWire equivalence
physical quantum behavior
```

## Next Blocker

The next blocker is production-readiness, not route mechanics:

```text
PUBLIC_BETA / PRODUCTION_DEFAULT_DECISION
```

That decision should require a real release contract, longer external workloads,
rollback observability, and explicit ownership of compatibility guarantees.
