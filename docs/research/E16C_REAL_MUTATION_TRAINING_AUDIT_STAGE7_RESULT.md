# E16C Real Mutation Training Audit Stage 7 Result

Status: completed.

## Decision

```text
decision = e16c_real_mutation_training_stage7_confirmed
next = E16C_STAGE8_REAL_MUTATION_REPAIR_CONFIRM
primary_system = REAL_MUTATION_TRAINED_PRUNED_MEMORY_POLICY_PRIMARY
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e16c_real_mutation_training_audit_stage7/
```

## Real Training Audit

```text
source_fixture_audit_passed = true
aggregate_recomputed_from_episode_logs = true
train_episode_count = 210
validation_episode_count = 100
heldout_episode_count = 180
population_size = 34
generations = 9
candidate_count_evaluated = 1656
best_generation = 5
```

The static fixture reference from the prior Stage 7 repair is included only as
an invalid reference row. It is not eligible as primary.

## Heldout Metrics

```text
best_baseline_system = MAJORITY_MEMORY_NO_ABSTAIN

primary_multi_sentence_binding_accuracy = 1.000
primary_long_horizon_recall = 1.000
delta_vs_best_baseline_binding_accuracy = +0.222222
delta_vs_best_baseline_long_horizon_recall = +0.200000

ambiguous_abstain_accuracy = 1.000
nested_depth2_accuracy = 1.000
nested_depth3_accuracy = 1.000
capacity_pressure_accuracy = 1.000
stale_update_rejection_rate = 1.000
corrupt_then_repair_success_rate = 1.000
distractor_gap_survival = 1.000
trace_validity = 1.000
wrong_writeback_rate = 0.000
destructive_overwrite_rate = 0.000
```

## Pruned Policy

```text
policy_id = pol_129e1ed503de
memory_slot_count = 6
eviction_strategy = score
key_addressing_mode = key
confidence_update_rule = confidence
nested_resolution_depth = 3
repair_weight = 0.000
ambiguity_abstain_threshold = 0.080
program_len = 12
```

## Capacity Sweep

| slots | binding | recall | depth2 | depth3 | capacity | pass |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.574 | 0.233 | 0.000 | 0.000 | 0.000 | false |
| 2 | 0.741 | 0.533 | 1.000 | 0.000 | 0.333 | false |
| 3 | 0.907 | 0.833 | 1.000 | 1.000 | 0.500 | false |
| 4 | 0.963 | 0.933 | 1.000 | 1.000 | 0.667 | false |
| 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | true |
| 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | true |
| 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | true |

The first passing capacity is `6` memory slots.

## Ablations

The ablation report is recomputed from real heldout policy execution. It verifies
the expected failures for no memory slots, low capacity, no stale rejection, no
repair evidence, no ambiguity abstain, and no nested resolution.

## Boundary

This confirms real deterministic mutation/search training over Stage 7 memory policies in a controlled synthetic text-flow proxy. It does not prove general natural-language AI or production training readiness.

## Verification

```text
python3 -m py_compile scripts/probes/run_e16c_real_mutation_training_audit_stage7.py scripts/probes/run_e16c_real_mutation_training_audit_stage7_check.py
python3 scripts/probes/run_e16c_real_mutation_training_audit_stage7.py --out target/pilot_wave/e16c_real_mutation_training_audit_stage7
python3 scripts/probes/run_e16c_real_mutation_training_audit_stage7_check.py --out target/pilot_wave/e16c_real_mutation_training_audit_stage7 --write-summary
```
