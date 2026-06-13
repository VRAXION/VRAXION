# E44D Wire Bus Width And Integrity Budget Sweep Contract

Milestone:

```text
E44D_WIRE_BUS_WIDTH_AND_INTEGRITY_BUDGET_SWEEP
```

## Purpose

E44C showed that an 8-bit anonymous Proposal payload bus has a real budget
tradeoff:

```text
5 data + 3 reserve     -> reserve-friendly but no silent-corruption guard
5 data + 3 integrity   -> corruption-aware but no true reserve
```

E44D sweeps wider buses to find the smallest bus width that can support:

```text
data capacity for 32 abstract intents
integrity/check bits for silent active-bit corruption
true reserve bits that can tolerate reserve noise/dropout
Agency-safe ASK/REJECT behavior under ambiguity or corruption
```

This is a controlled symbolic/numeric Proposal ABI probe, not a raw language,
AGI, consciousness, deployment, or model-scale claim.

## Systems

```text
oracle_reference
bus8_5data_3reserve_masked
bus8_5data_3crc
bus10_5data_3crc_2reserve
bus12_5data_4crc_3reserve
bus16_5data_5ecc_6reserve
universal_mutated_bus_policy
random_policy_control
```

The universal mutated policy must choose among valid bus configurations using
mutation, accept/reject, and rollback. It must not use backprop or optimizers.

## Stress Families

```text
clean
reserve_random_noise
reserve_adversarial_noise
reserve_dropout
active_dropout_visible
active_stuck_visible
active_bitflip_silent
double_alias_silent
burst_noise_silent
known_wire_permutation
unknown_wire_permutation
stale_replay
ground_conflict
partial_support
no_valid_proposal
```

The important falsification case is `double_alias_silent`: a 3-bit check may
detect simple bit flips but still alias under selected two-bit corruption.

## Required Artifacts

```text
backend_manifest.json
stress_generation_report.json
bus_width_sweep_report.json
stress_barrage_results.json
universal_mutation_report.json
system_results.json
row_level_results.jsonl
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
stress_table.md
report.md
```

The sample pack under `docs/research/artifact_samples/e44d_wire_bus_width_and_integrity_budget_sweep/`
must pass sample-only validation.

## Metrics

```text
stress_success
reserve_success
integrity_success
reserve_noise_success
active_bitflip_success
double_alias_success
burst_noise_success
known_permutation_success
unknown_permutation_success
false_commit_rate
wrong_commit_rate
false_ask_rate
payload_width
integrity_bits
reserve_bits
accepted/rejected/rollback mutation counts
deterministic replay
checker failure count
```

## Decisions

```text
e44d_bus10_sufficient
e44d_bus12_sufficient
e44d_bus16_required
e44d_integrity_reserve_tradeoff_persists
e44d_no_universal_wire_bus_found
e44d_invalid_artifact_detected
```

Positive sufficiency requires reserve and integrity stresses to pass without
false/wrong commits and with deterministic replay plus checker success.
