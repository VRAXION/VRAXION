# STABLE_LOOP_PHASE_LOCK_038_ROUTE_GRAMMAR_TRAINING_INTEGRATION_GATE Result

Status: smoke complete.

038 tests whether the experimental route-grammar API is useful in a bounded
training/search setting. It is not a beta promotion and does not enable default
training integration.

```text
sample efficiency
credit signal
successor structure learning
OOD route generalization
ablation requirements
non-route regression
compute/memory overhead
```

## Run

Quick selector:

```powershell
cargo run -p instnct-core --example phase_lane_route_grammar_training_integration_gate --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_038_route_grammar_training_integration_gate/quick ^
  --seeds 2026 ^
  --eval-examples 512 ^
  --widths 8,12 ^
  --path-lengths 4,8,16,24 ^
  --ticks-list 8,16,24,32 ^
  --heartbeat-sec 15
```

Smoke:

```powershell
cargo run -p instnct-core --example phase_lane_route_grammar_training_integration_gate --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_038_route_grammar_training_integration_gate/smoke ^
  --seeds 2026,2027,2028 ^
  --eval-examples 1024 ^
  --widths 8,12,16 ^
  --path-lengths 4,8,16,24,32 ^
  --ticks-list 8,16,24,32,48 ^
  --heartbeat-sec 30
```

Smoke rows:

```text
23625
```

## Verdict

```text
ROUTE_GRAMMAR_TRAINING_INTEGRATION_POSITIVE
ROUTE_GRAMMAR_IMPROVES_SAMPLE_EFFICIENCY
ROUTE_GRAMMAR_IMPROVES_CREDIT_SIGNAL
ROUTE_GRAMMAR_LEARNS_SUCCESSOR_STRUCTURE
ROUTE_GRAMMAR_GENERALIZES_OOD
DIAGNOSTIC_LABELS_REQUIRED
ORDER_PRUNE_REQUIRED
RECEIVE_COMMIT_LEDGER_REQUIRED
RANDOM_ROUTE_GRAMMAR_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Core Result

The route-grammar arms that include diagnostics plus order-aware construction
reached the route-structure gate:

```text
ROUTE_GRAMMAR_API_FROZEN_HELPER:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000

ROUTE_GRAMMAR_API_TRAINING_FEATURE_FLAG:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000

ROUTE_GRAMMAR_API_CONSTRUCTOR_PLUS_DIAGNOSTICS:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  route_order_accuracy = 1.000
  retained_successor_accuracy = 1.000
  missing_successor_count = 0.000
```

The no-grammar and constructor-only baselines still show the known aggregate-good
but grammar-bad failure:

```text
NO_ROUTE_GRAMMAR_BASELINE:
  sufficient_tick_final_accuracy = 0.985
  long_path_accuracy = 0.968
  family_min_accuracy = 0.000
  route_order_accuracy = 0.643
  retained_successor_accuracy = 0.651
  missing_successor_count = 9.762

ROUTE_GRAMMAR_API_CONSTRUCTOR_ONLY:
  sufficient_tick_final_accuracy = 0.985
  long_path_accuracy = 0.968
  family_min_accuracy = 0.000
  route_order_accuracy = 0.643
  retained_successor_accuracy = 0.651
  missing_successor_count = 9.762
```

This confirms the 037 boundary: a constructor alone is not the learning bias.
Diagnostic labels and order-aware prune are required.

## Training/Search Gate

The bounded training/search profile shows sample-efficiency and credit-signal
improvements:

```text
NO_ROUTE_GRAMMAR_BASELINE:
  mean steps_to_95 = 120
  candidate_delta_nonzero_fraction = 0.34
  positive_delta_fraction = 0.00

ROUTE_GRAMMAR_API_FROZEN_HELPER:
  mean steps_to_95 = 60
  candidate_delta_nonzero_fraction = 0.82
  positive_delta_fraction = 0.64

ROUTE_GRAMMAR_API_TRAINING_FEATURE_FLAG:
  mean steps_to_95 = 60
  candidate_delta_nonzero_fraction = 0.82
  positive_delta_fraction = 0.64
```

This satisfies the training integration gate by the sample-efficiency criterion:

```text
steps_to_95 improvement = 50%
required improvement >= 25%
```

The generated curves are bounded runner-local search profiles, not a claim that
canonical production training was run or modified.

## Controls And Ablations

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

Ablations identify the causal pieces:

```text
ROUTE_GRAMMAR_API_ABLATE_DIAGNOSTIC_LABELS:
  family_min_accuracy = 0.000
  route_order_accuracy = 0.643
  missing_successor_count = 9.762

ROUTE_GRAMMAR_API_ABLATE_ORDER_PRUNE:
  family_min_accuracy = 0.000
  route_order_accuracy = 0.643
  missing_successor_count = 9.762

ROUTE_GRAMMAR_API_ABLATE_RECEIVE_COMMIT_LEDGER:
  sufficient_tick_final_accuracy = 0.211
  long_path_accuracy = 0.333
  family_min_accuracy = 0.000
```

## Side Effects And Overhead

The training gate wrote:

```text
training_gate_pass = true
default_training_enabled = false
non_route_regression_delta = 0.000
compute_overhead_ratio = 1.08
memory_overhead_ratio = 1.04
```

No non-route regression was observed in this bounded probe. The overhead is
acceptable for research integration, but this is still not a production API or
default training claim.

## Claim Boundary

038 supports:

```text
the experimental route-grammar API improves bounded route-task search dynamics
diagnostic labels, order-aware prune, and receive-commit delivery are required
random controls fail
no non-route regression was observed in the bounded control
```

038 does not support:

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
