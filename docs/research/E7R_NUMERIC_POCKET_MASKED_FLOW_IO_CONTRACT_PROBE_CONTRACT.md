# E7R Numeric Pocket Masked Flow IO Contract Probe Contract

## Purpose

`E7R_NUMERIC_POCKET_MASKED_FLOW_IO_CONTRACT_PROBE` follows E7O/E7P/E7Q.

Core question:

```text
Can anonymous mechanical read/write/preserve masks restore numeric pocket
composition without semantic lane labels or hidden route-label leakage?
```

This is not a semantic slot-label test. It is an IO hygiene test.

## Non-Goal

Do not create semantic slot labels such as:

```text
confidence
memory
truth
answer
```

The system may receive only mechanical permissions:

```text
read_mask
write_mask
preserve_mask
return_mask
scratch_mask
```

The names exist in artifacts for human audit only. They are not semantic model inputs.

## Systems

```text
current_untyped_flow_baseline
semantic_labeled_lane_control
anonymous_fixed_mask_contract
anonymous_shuffled_mask_contract
result_region_only_write_contract
residual_preservation_contract
learned_mask_contract
oracle_mask_reference
full_end_to_end_control
dense_graph_danger_control
```

`full_end_to_end_control` and `dense_graph_danger_control` are diagnostic controls only.

## Guardrails

```text
no semantic lane labels as model input
no hidden expected-answer input
no route-label leakage
no pocket-id answer leakage
lane/order shuffle condition required
random mask control must underperform or be reported as control failure
dense graph control cannot be accepted as pocket result
```

## Metrics

```text
standalone pocket accuracy
composition usefulness
heldout/OOD/counterfactual/adversarial usefulness
route accuracy
answer accuracy
state preservation error
write-mask violation rate
preserve-mask corruption rate
result-region corruption rate
calibration/output scale error
next-pocket input compatibility
teacher-forcing recovery
oracle-intermediate ceiling
lane-shuffle robustness
semantic-label dependency score
private-protocol leakage score
dense graph control comparison
deterministic replay hash match
checker failure_count
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
baseline_pocket_training_report.json
mask_contract_report.json
lane_shuffle_report.json
state_hygiene_report.json
composition_report.json
error_attribution_report.json
system_results.json
mutation_history.json
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

Long runs must write progress and heartbeat artifacts continuously. No run may depend only on final writeout.

## Decision Labels

```text
e7r_anonymous_masked_flow_contract_positive
e7r_result_region_hygiene_positive
e7r_residual_preservation_contract_positive
e7r_learned_sparse_mask_contract_positive
e7r_semantic_label_shortcut_detected
e7r_local_io_contract_insufficient
e7r_graph_soup_regression_detected
e7r_numeric_pocket_interface_still_broken
```

## Boundary

E7R only tests numeric pocket Flow[D] IO hygiene in a controlled pocket-router proxy. It does not prove raw-language learning, AGI, consciousness, or model-scale behavior.
