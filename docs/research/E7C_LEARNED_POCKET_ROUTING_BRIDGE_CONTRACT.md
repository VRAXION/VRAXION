# E7C Learned Pocket Routing Bridge Contract

## Purpose

E7C tests whether the E7B pocket-routing result survives when pocket outputs are no longer hand-coded symbolic outputs. Pocket models are trained separately, frozen, and then router systems are trained or mutated over those learned pocket outputs.

Core question:

```text
Can mutation/rollback learn a router over learned frozen pockets?
```

## Runner And Checker

- Runner: `scripts/probes/run_e7c_learned_pocket_routing_bridge.py`
- Checker: `scripts/probes/run_e7c_learned_pocket_routing_bridge_check.py`
- Default artifact root: `target/pilot_wave/e7c_learned_pocket_routing_bridge/`

## Systems

```text
monolithic_backprop_model
monolithic_mutation_model
learned_pockets_gradient_router
learned_pockets_mutation_router
learned_binary_pockets_mutation_router
router_plus_limited_pocket_repair
random_router_control
oracle_learned_pocket_router_reference
oracle_symbolic_reference
```

## Pocket Bridge

Each seed trains a frozen learned pocket model:

```text
input:
  a, b, key, threshold, a-b, split one-hot

outputs:
  candidate answer for each route pocket
  branch flag
```

The router receives learned pocket predictions/probabilities, not symbolic pocket outputs. A binary-pocket mode also rounds the learned pocket outputs before routing.

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
learned_pocket_training_report.json
pocket_library_report.json
system_results.json
mutation_history.json
training_history.json
composition_report.json
leakage_report.json
deterministic_replay.json
aggregate_metrics.json
decision.json
summary.json
report.md
checker_report.json
progress.jsonl
hardware_heartbeat.jsonl
partial_status/*.json
mutation_history_snapshots/*.json
partial_aggregate_snapshot.json
```

## Metrics

Required system metrics:

```text
answer_accuracy
route_accuracy
composition_accuracy
shortcut_rate
usefulness_score
heldout_usefulness
ood_usefulness
counterfactual_usefulness
adversarial_usefulness
generalization_gap
parameter_count
```

Required pocket metrics:

```text
candidate_answer_accuracy
branch_accuracy
oracle_learned_route_answer_ceiling
```

## Decisions

Allowed decisions:

```text
e7c_learned_pocket_mutation_router_viable
e7c_binary_learned_pocket_router_viable
e7c_gradient_router_only_viable
e7c_symbolic_only_scaffold_detected
e7c_learned_pocket_quality_bottleneck
e7c_leak_or_artifact_detected
```

## Checker Gates

The checker fails on missing artifacts, missing systems/seeds, missing learned pocket training report, missing GPU pocket/gradient history, mutation-only backprop/optimizer use, no rejected mutations, rollback mismatch, deterministic replay mismatch, failed random control without leak decision, missing row-level samples, missing hardware heartbeat, or broad claims in the report.

## Boundary

This is a controlled learned-pocket routing bridge. It tests routing over separately learned frozen pocket outputs; it does not claim open-ended language reasoning or model-scale behavior.
