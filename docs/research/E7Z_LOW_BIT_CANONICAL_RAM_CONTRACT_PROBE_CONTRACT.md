# E7Z Low-Bit Canonical RAM Contract Probe Contract

## Purpose

`E7Z_LOW_BIT_CANONICAL_RAM_CONTRACT_PROBE` follows E7W/E7X/E7Y.

Core question:

```text
Does numeric pocket composition improve if the shared producer -> RAM ->
consumer boundary is forced into a shared binary/ternary/int4 canonical code
from the start?
```

Secondary question:

```text
Is external RAM low-bit communication enough, or must pocket internals also be
low-bit?
```

E7Z does not add semantic labels, dense graph routing, image tasks, language
tasks, or a new router architecture. It only audits the low-bit communication
contract at the Flow/RAM boundary.

## Systems

```text
continuous_direct_write_baseline
oracle_write_continuous_reference
oracle_write_binary_projected
oracle_write_ternary_projected
oracle_write_int4_projected
learned_binary_ram_boundary
learned_ternary_ram_boundary
learned_int4_ram_boundary
learned_binary_ram_boundary_plus_mutation_repair
learned_ternary_ram_boundary_plus_mutation_repair
learned_int4_ram_boundary_plus_mutation_repair
pure_binary_pocket_and_ram
pure_ternary_pocket_and_ram
int4_pocket_and_ram
mixed_precision_pocket_float_ram_lowbit
dense_graph_danger_control
```

Oracle/projection systems are diagnostic references only. Primary learned
low-bit systems train producer and consumer pockets on the same low-bit RAM
code from the start.

## Low-Bit Codes

```text
primary result cell: 0/1
binary support cells: {-1, +1}
ternary support cells: {-1, 0, +1}
int4 support cells: 16 uniform levels over [-1, +1]
continuous support cells: float baseline
```

Support cells are anonymous mechanical channels. No semantic lane names are
model inputs.

## Mutation Repair

Mutation repair runs after low-bit boundary training and mutates only small
boundary parameters:

```text
primary threshold
support thresholds
ternary deadzones
int4 scales
```

It uses accept/reject/rollback on row-level validation score and never calls
optimizers or backprop.

## Required Metrics

```text
composition usefulness
answer accuracy
route accuracy
heldout/OOD/counterfactual/adversarial usefulness
projected-oracle performance
canonical-state validity
next-pocket compatibility error
support-channel sign mismatch rate
support-channel silence rate
bundle MAE
multi-cell pattern correlation
write spread
changed cell count
bit budget
compression ratio
mutation repair gain
accepted/rejected/rollback counts
deterministic replay
checker failure_count
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
pocket_training_report.json
projected_oracle_report.json
low_bit_boundary_report.json
progressive_freeze_report.json
mutation_repair_report.json
bit_budget_report.json
system_results.json
aggregate_metrics.json
decision.json
summary.json
report.md
deterministic_replay.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
```

## Decision Labels

```text
e7z_binary_canonical_ram_contract_positive
e7z_ternary_canonical_ram_contract_positive
e7z_int4_canonical_ram_contract_positive
e7z_low_bit_ram_contract_partially_positive
e7z_low_bit_training_or_commit_learning_bottleneck
e7z_full_low_bit_pocket_ram_preferred
e7z_external_low_bit_ram_boundary_sufficient
e7z_low_bit_mutation_repair_positive
e7z_low_bit_canonical_ram_contract_not_sufficient
e7z_graph_soup_regression_detected
```

## Guardrails

```text
real row-level eval
no semantic labels
no new router
no hardcoded oracle leakage in primary learned systems
oracle/projection variants marked reference/diagnostic
mutation repair uses no backprop
deterministic replay required
checker failure_count must be 0
```

## Boundary

E7Z only tests low-bit canonical RAM communication in a controlled numeric
pocket-router proxy. It does not prove raw-language learning, AGI,
consciousness, deployed-model behavior, or model-scale behavior.
