# E16C Stage 7 Memory Binding Capacity Repair Result

Status: completed.

## Decision

```text
decision = e16c_stage7_memory_binding_capacity_repair_confirmed
next = E16C_STAGE8_NOISY_MULTI_SENTENCE_REPAIR_CONFIRM
primary_system = MUTATION_TRAINED_PRUNED_MEMORY_POLICY_PRIMARY
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e16c_stage7_memory_binding_capacity_repair/
```

## Repair Result

```text
baseline_stage7_binding_accuracy = 0.622
repaired_stage7_binding_accuracy = 0.872
delta_binding_accuracy = +0.250

baseline_long_horizon_recall = 0.611
repaired_long_horizon_recall = 0.856
delta_long_horizon_recall = +0.245

best_memory_slot_count = 8
first_passing_memory_slot_count = 6
discovered_policy_count = 128
pruned_policy_count = 9
```

## Primary Gate Metrics

```text
nested_depth2_accuracy = 0.840
nested_depth3_accuracy = 0.718
capacity_pressure_accuracy = 0.806
stale_update_rejection_rate = 0.914
corrupt_then_repair_success_rate = 0.866
ambiguous_abstain_accuracy = 0.888
distractor_gap_survival = 0.890
mixed_memory_template_accuracy = 0.858
trace_validity = 0.976
wrong_writeback_rate = 0.008
destructive_overwrite_rate = 0.003
branch_contamination_rate = 0.000
```

## Capacity Sweep

| slots | binding | recall | depth2 | depth3 | capacity | pass |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.600 | 0.574 | 0.440 | 0.280 | 0.308 | false |
| 2 | 0.686 | 0.652 | 0.608 | 0.454 | 0.424 | false |
| 3 | 0.732 | 0.706 | 0.696 | 0.612 | 0.552 | false |
| 4 | 0.756 | 0.746 | 0.744 | 0.640 | 0.642 | false |
| 6 | 0.872 | 0.856 | 0.840 | 0.718 | 0.806 | true |
| 8 | 0.892 | 0.874 | 0.858 | 0.748 | 0.834 | true |
| 12 | 0.894 | 0.876 | 0.858 | 0.750 | 0.836 | true |

The first slot count that clears the Stage 7 repair gate is `6`. The highest
quality/cost sweep point is `8`, while the pruned primary uses the lower-cost
passing policy.

## Stage 8 Stretch

```text
stage8_repair_success_rate = 0.713
stage8_noise_rejection_rate = 0.801
stage8_canonical_decoder_exact_accuracy = 0.782
stage8_trace_validity = 0.914
```

Stage 8 improved enough to justify the next targeted milestone, but it remains
a downstream stretch measurement in this run.

## Ablations

```text
no_memory_slots_binding_accuracy = 0.180
low_memory_capacity_capacity_pressure_accuracy = 0.286
no_stale_rejection_rate = 0.322
no_repair_success_rate = 0.346
no_ambiguity_abstain_accuracy = 0.214
no_nested_depth2_accuracy = 0.226
no_nested_depth3_accuracy = 0.118
```

The ablations are coherent: memory slots, sufficient capacity, stale rejection,
repair evidence, ambiguity abstain, and nested resolution are all required for
the repaired Stage 7 behavior.

## Boundary

This is a deterministic synthetic controlled text-flow Stage 7 memory binding repair probe. It tests targeted mutation/search over memory policies. It does not prove general natural-language AI.

## Verification

```text
python3 -m py_compile scripts/probes/run_e16c_stage7_memory_binding_capacity_repair.py scripts/probes/run_e16c_stage7_memory_binding_capacity_repair_check.py
python3 scripts/probes/run_e16c_stage7_memory_binding_capacity_repair.py --out target/pilot_wave/e16c_stage7_memory_binding_capacity_repair
python3 scripts/probes/run_e16c_stage7_memory_binding_capacity_repair_check.py --out target/pilot_wave/e16c_stage7_memory_binding_capacity_repair --write-summary
```
