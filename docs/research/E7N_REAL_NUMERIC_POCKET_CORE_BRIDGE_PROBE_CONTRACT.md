# E7N Real Numeric Pocket Core Bridge Probe Contract

## Purpose

`E7N_REAL_NUMERIC_POCKET_CORE_BRIDGE_PROBE` replaces the earlier symbolic pocket proxy with a real learned numeric pocket core.

Core question:

```text
Can a small real matrix pocket be trained by backprop, quantized, mutation-repaired, pruned/crystallized, and registered as a callable router pocket?
```

This is intentionally not an image/MNIST/addition task yet. E7N first tests the single-pocket numeric bridge on clean vector micro-skills.

## Pocket Contract

```text
CALL(pocket_id, Flow[D]) -> Flow[D]
```

The internal pocket has real matrices:

```text
input_adapter: D -> K
core_matrix:   K -> K
output_adapter: K -> D
```

Backprop is allowed only in the float pocket pretraining path. Quantization, mutation repair, and pruning/crystallization must not use optimizers or backprop.

## Systems

```text
symbolic_proxy_pocket_reference
float_numeric_pocket_backprop
quantized_numeric_pocket_int8
quantized_numeric_pocket_int4
quantized_numeric_pocket_ternary
quantized_numeric_pocket_binary
quantized_pocket_plus_mutation_repair
quantized_pocket_plus_prune_crystallize
quantized_pocket_plus_repair_plus_prune
random_pocket_control
```

## Micro-Skills

```text
xor/parity
compare
small modular-add decision
threshold
route-check decision
counterfactual flip
```

The task uses deterministic numeric `Flow[D]` rows with train, validation, heldout, OOD, counterfactual, and adversarial splits.

## Required Metrics

```text
train/heldout/OOD/counterfactual/adversarial accuracy
router-call usefulness
quantization drop
mutation repair gain
prune compression ratio
post-prune quality delta
final parameter count / active parameter count
bit budget
latency/cost estimate
stability under repeated calls
accepted/rejected/rollback mutation counts
deterministic replay hash match
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
numeric_pocket_training_report.json
quantization_report.json
mutation_repair_report.json
pruning_crystallization_report.json
pocket_registry_report.json
router_call_report.json
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

Long runs must write progress, heartbeat, mutation snapshots, training snapshots, and partial aggregate state before final artifacts.

## Decision Labels

```text
e7n_real_numeric_pocket_viable
e7n_quantized_numeric_pocket_viable
e7n_mutation_repair_numeric_pocket_positive
e7n_numeric_pocket_crystallization_positive
e7n_binary_numeric_pocket_viable
e7n_float_only_numeric_pocket
e7n_numeric_pocket_bridge_failed
```

## Checker Gates

The checker fails on missing artifacts, missing systems, missing row-level samples, missing real matrix state hashes, missing quantization precision rows, mutation backprop/optimizer usage, missing accepted/rejected/rollback counts for mutation systems, rollback mismatch, missing parameter diff/hash, random-control leakage, deterministic replay mismatch, missing progress/heartbeat artifacts, or broad claims outside this controlled proxy.

## Boundary

E7N is a controlled numeric pocket-core bridge probe. It is not a raw-language, deployed-model, AGI, consciousness, or model-scale result.
