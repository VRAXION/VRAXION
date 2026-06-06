# E7P Numeric Pocket Adapter Joint Training Audit Contract

## Purpose

`E7P_NUMERIC_POCKET_ADAPTER_JOINT_TRAINING_AUDIT` follows E7O. E7O showed that real learned numeric pockets can be strong standalone modules while composition through `Flow[D]` remains weak.

Core question:

```text
Can E7O's composition failure be repaired by training each pocket's input adapter, core, and output adapter inside the shared Flow[D] contract while the router and all other pockets stay frozen?
```

## Systems

```text
standalone_pocket_then_fixed_adapter
adapter_only_training
pocket_core_only_training
joint_adapter_plus_pocket_training
joint_adapter_plus_pocket_with_slot_contract
full_end_to_end_training_control
oracle_intermediate_state_reference
```

The local-training systems freeze the router and all non-target pockets. `full_end_to_end_training_control` is diagnostic only and must not be treated as the primary architecture.

## Training Scope

```text
adapter_only_training:
  train input_adapter/output_adapter
  freeze pocket core

pocket_core_only_training:
  train core/carry
  freeze adapters

joint_adapter_plus_pocket_training:
  train input_adapter + core + output_adapter
  freeze router and other pockets

joint_adapter_plus_pocket_with_slot_contract:
  same as joint
  add stronger preservation, read/write mask, and result-slot hygiene losses
```

## Metrics

```text
standalone pocket accuracy
composition usefulness
answer accuracy
route accuracy
heldout/OOD/counterfactual/adversarial usefulness
state preservation error
result-slot corruption rate
output calibration error
next-pocket input compatibility error
teacher-forcing recovery
adapter/core/router/composition error attribution
parameter count
bit budget
deterministic replay hash match
checker failure_count
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
baseline_pocket_training_report.json
adapter_training_report.json
flow_contract_report.json
composition_report.json
error_attribution_report.json
system_results.json
leakage_report.json
deterministic_replay.json
aggregate_metrics.json
decision.json
summary.json
report.md
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
```

Long runs must write progress and heartbeat artifacts continuously. No run may depend only on a final write.

## Decision Labels

```text
e7p_joint_adapter_pocket_training_positive
e7p_adapter_contract_bottleneck_confirmed
e7p_pocket_core_training_bottleneck_confirmed
e7p_typed_slot_contract_required
e7p_local_pocket_training_insufficient
e7p_numeric_pocket_composition_not_yet_viable
```

## Checker Gates

The checker fails on missing artifacts, missing systems, missing row-level samples, missing flow-contract metrics, missing training rows, failed deterministic replay, replay hash mismatch, unfrozen-router leakage flags, optimizer/backprop outside allowed local training, or claims outside this controlled numeric proxy.

## Boundary

E7P is a controlled numeric pocket-flow interface probe. It does not make raw-language, deployed-model, AGI, consciousness, or model-scale claims.
