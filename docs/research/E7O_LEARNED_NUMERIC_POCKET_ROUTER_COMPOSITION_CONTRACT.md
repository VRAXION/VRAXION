# E7O Learned Numeric Pocket Router Composition Contract

## Purpose

`E7O_LEARNED_NUMERIC_POCKET_ROUTER_COMPOSITION` tests whether several real learned numeric pockets can be composed by a router.

Core question:

```text
Can the pocket-router architecture still work when pockets are real learned numeric modules, not symbolic segment proxies?
```

E7O intentionally stays on controlled numeric composition rows. It does not use MNIST, image digit addition, raw language, or deployed-model workloads.

## Pocket Interface

```text
CALL(pocket_id, Flow[D]) -> Flow[D]
```

Each pocket is a real numeric matrix core inherited from E7N:

```text
input_adapter:  D -> K
core_matrix:    K -> K
output_adapter: K -> D
```

Backprop is allowed only for standalone numeric pocket pretraining and monolithic gradient baselines. Mutation-router systems must use mutation, accept/reject, rollback, and deterministic replay without optimizers or backprop.

## Pocket Skills

```text
compare
mod_add
parity
threshold
counterfactual_flip
verify
```

Each pocket is trained as a float numeric pocket, quantized to int8/int4/ternary/binary variants, pruned for the int4-pruned branch when viable, and registered as a callable pocket with an ID and contract.

## Composition Families

```text
compare -> threshold
mod_add -> parity
counterfactual_flip -> recompute
verify -> route correction
mixed chain length 4
adversarial misleading branch
```

Rows include train, validation, heldout, OOD, counterfactual, and adversarial splits.

## Systems

```text
symbolic_proxy_pocket_router_reference
float_numeric_pocket_library_router
int8_numeric_pocket_library_router
int4_pruned_numeric_pocket_library_router
ternary_binary_numeric_pocket_router
mixed_precision_numeric_pocket_router
monolithic_backprop_model
monolithic_mutation_model
dense_graph_danger_control
oracle_router_over_numeric_pockets
```

The symbolic and oracle systems are references. The primary evidence systems are mutation routers over frozen learned numeric pocket libraries.

## Required Metrics

```text
per-pocket standalone accuracy
pocket quality gate pass/fail
route accuracy
answer accuracy
composition usefulness
heldout/OOD/counterfactual/adversarial usefulness
pocket error rate
router error rate
composition error rate
quantization/pruning quality loss
mean route steps
pocket reuse count
parameter count
bit budget
accepted/rejected/rollback mutation counts
deterministic replay hash match
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
numeric_pocket_training_report.json
pocket_library_report.json
router_training_report.json
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

Long runs must write heartbeat, progress rows, pocket training snapshots, router mutation snapshots, and partial aggregate snapshots before final artifacts.

## Decision Labels

```text
e7o_int4_numeric_pocket_router_composition_positive
e7o_float_only_numeric_pocket_composition
e7o_router_over_numeric_pockets_failure
e7o_numeric_pocket_quality_bottleneck
e7o_monolithic_model_preferred_for_numeric_composition
e7o_numeric_pocket_router_collapses_to_graph_soup
e7o_mixed_precision_numeric_pocket_router_preferred
```

## Checker Gates

The checker fails on missing artifacts, missing system variants, missing pocket variants, missing row-level samples, mutation-router optimizer/backprop usage, missing accepted/rejected/rollback counts, rollback mismatch, missing parameter diff/hash, missing pocket standalone quality, symbolic proxy used as primary, deterministic replay mismatch, missing progress/heartbeat artifacts, or broad claims outside this controlled numeric proxy.

## Boundary

E7O is a controlled numeric pocket-router composition probe. It does not make raw-language, deployed-model, AGI, consciousness, or model-scale claims.
