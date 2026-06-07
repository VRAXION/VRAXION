# E8D Substrate-First RAM Language Pretraining Probe Contract

## Summary

`E8D_SUBSTRATE_FIRST_RAM_LANGUAGE_PRETRAINING_PROBE` tests whether the E8
numeric pocket/RAM composition bottleneck is better addressed by learning a
shared Flow/RAM substrate before pocket composition, instead of adding another
producer-side target/loss trick.

Core question:

```text
Can a stable shared RAM language be learned first, so pockets become
valid Flow_before -> valid Flow_after operators over that substrate?
```

## Systems

```text
no_substrate_baseline
bridge_only_baseline
substrate_autoencoder
substrate_transition_model
low_bit_substrate_codebook
frozen_substrate_then_producer
frozen_substrate_then_consumer
frozen_substrate_then_pocket_composition
jointly_mutable_substrate_and_pockets
oracle_substrate_reference
dense_graph_danger_control
```

## Guardrails

The substrate must not learn a `RAM -> final answer` objective. It may learn
valid intermediate state geometry and mechanical route-step transitions only.

Required negative claims:

```text
new_router = false
semantic_lane_labels_as_model_input = false
substrate_final_answer_objective = false
oracle_write_at_inference_for_learned_systems = false
```

Oracle substrate is diagnostic/reference only and does not count as a learned
success.

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
substrate_pretraining_report.json
ram_validity_report.json
producer_dynamics_report.json
consumer_read_report.json
compatibility_report.json
mutation_repair_report.json
gradient_diagnostics_report.json
system_results.json
row_level_samples.json
aggregate_metrics.json
decision.json
summary.json
report.md
deterministic_replay.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
```

## Metrics

```text
substrate validity score
reconstruction / transition MAE
producer write compatibility
consumer read compatibility
composition usefulness
heldout/OOD/counterfactual/adversarial usefulness
state drift per step
code entropy / collapse
low-bit utilization
route accuracy
answer accuracy
gradient diagnostics
mutation accepted/rejected/rollback counts
deterministic replay hash match
```

## Decisions

```text
e8d_substrate_first_positive
e8d_bridge_adapter_sufficient
e8d_pocket_to_substrate_write_bottleneck
e8d_substrate_consumer_read_bottleneck
e8d_frozen_substrate_too_rigid
e8d_graph_soup_regression_detected
e8d_substrate_language_not_helpful
```

Positive substrate-first evidence requires:

```text
best substrate-first composition usefulness beats both no-substrate and bridge
by >= 0.03
producer write compatibility does not regress
consumer read compatibility does not regress
OOD/counterfactual/adversarial do not collapse
checker failure_count = 0
deterministic replay passes
```

## Boundary

E8D is a controlled symbolic/numeric Flow/RAM substrate probe. It does not test
raw language, images, AGI, consciousness, deployed model behavior, or model
scale.
