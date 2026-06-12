# E33B Gradientless Flow Breakpoint Audit Contract

Status: implemented probe contract.

## Purpose

E33 showed that a separate gradient saturation harness did not reproduce the
previous clean Flow/Pocket regime. E33B removes that gradient envelope and
audits the actual checked E24/E25/E26/E27 Flow/Pocket primary path.

Core question:

```text
Without gradient descent, where is the last clean controlled Flow/Pocket
milestone and where is the first real break?
```

## Scope

E33B imports the existing checked helpers:

```text
E24 unscaffolded online ruleshift discovery
E25 naturalized ruleshift text stream discovery
E26 hard-skip text reasoning failure map
E27 unresolved Flow-state information-seeking repair
```

It evaluates only Flow/Pocket primary systems and Flow/Pocket/control
ablations. It does not train neural baselines and does not call a gradient
optimizer.

## Required Negative Constraint

```text
gradient_descent_used = false
optimizer_used = false
backprop_used = false
```

The checker fails if the E33B runner contains optimizer/backprop tokens.

## Metrics

Primary metrics:

```text
composition_success
resolution_success where E27 uses information-seeking actions
answer_correct
trace_exact
evidence_span_valid
split-level success
first failed split
deterministic replay
```

## Decision Labels

```text
e33b_gradientless_breakpoint_localized
e33b_gradientless_all_controlled_clean
e33b_gradientless_breakpoint_not_reproduced
e33b_gradientless_artifact_invalid
```

Positive localization requires:

```text
E24 primary min split success >= 0.98
E25 primary min split success >= 0.98
E26 primary stages 1-4 min success >= 0.98
E26 stage5_missing_evidence_ambiguous <= 0.05
E27 primary min split success >= 0.98
checker failure_count = 0
deterministic replay passes
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
breakpoint_ladder_report.json
system_results.json
row_level_results.jsonl
aggregate_metrics.json
decision.json
summary.json
deterministic_replay.json
resource_usage_report.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
report.md
```

## Boundary

E33B is a controlled symbolic/naturalized-text breakpoint audit. It does not
claim raw language reasoning, AGI, consciousness, deployed-model behavior, or
model-scale behavior.
