# E87 Dense Potential Sparse Active Set Selector

```text
decision = e87_dense_potential_sparse_selector_confirmed
checker_failure_count = 0
seeds = 16
workers = 16
```

## Purpose

Test whether a single selector can see the whole Pocket Library as dense
potential connections, then converge to a sparse active set of useful favorites.

This tests the user's idea:

```text
one candidate gets the whole skill set as potential dense connections
it slowly filters which pockets it likes
parallel seeds reveal whether a stable top emerges
```

Boundary:

```text
scoped visible calculation-trace routing / validation only
not open-domain model training
not natural-language reasoning
not Core / True Golden promotion
```

## Result

```text
dense_potential_size = 16
final_active_set_size_mean = 7.000
active_set_reduction_mean = 0.5625

validation_action_min = 1.000000
adversarial_action_min = 1.000000
validation_false_call_max = 0.000000
adversarial_false_call_max = 0.000000
validation_false_commit_max = 0.000000
adversarial_false_commit_max = 0.000000

unsafe_final_selection_count = 0
redundant_final_selection_count = 0
top_k_jaccard_mean = 1.000000
```

Mutation evidence:

```text
accepted_mutations_total = 144
rejected_mutations_total = 4864
rollback_count_total = 304
plateau_rounds_mean = 19
```

## Stable Top

Every seed converged to the same active set:

```text
calc_scribe_native_seed
square_trace_adapter
arrow_trace_adapter
standalone_plain_trace_adapter
unicode_operator_normalizer
invalid_trace_rejector
long_text_scope_guard
```

Rejected / dropped from the final active set:

```text
native_seed_clone
square_adapter_clone
arrow_adapter_clone
numeric_alias_overreach
full_library_scan_overreach
invalid_direct_commit
long_text_plain_overreach
noop_trace_observer
expensive_debug_probe
```

## Counterfactual Contribution

Removing the stable pockets caused measurable loss:

```text
calc_scribe_native_seed        action_loss = 0.396461
invalid_trace_rejector         action_loss = 0.209373
square_trace_adapter           action_loss = 0.201973
standalone_plain_trace_adapter action_loss = 0.194691
arrow_trace_adapter            action_loss = 0.096344
unicode_operator_normalizer    action_loss = 0.072441
long_text_scope_guard          action_loss = 0.000130
```

The long-text scope guard has low average action loss because its trigger is
rare, but its ablation introduced nonzero false-call risk. It is therefore kept
as a safety/scope guard rather than a high-frequency utility pocket.

## Interpretation

E87 confirms the dense-potential / sparse-execution pattern:

```text
start:
  all 16 PocketTokens visible as potential connections

training:
  unsafe, redundant, and no-op pockets are pruned by score and guard checks

final:
  7-pocket active set
  identical across all 16 seeds
  zero unsafe final selections
  zero redundant final selections
```

This is closer to a governed evolutionary active-set selector than a full
library scan. The candidate does not run every pocket forever. It learns a
stable favorite set under deterministic validation/adversarial scoring.

## Artifacts

```text
target/pilot_wave/e87_dense_potential_sparse_active_set_selector/
  run_manifest.json
  library_manifest.json
  task_generation_report.json
  progress.jsonl
  seed_progress/
  partial_aggregate_snapshot.json
  seed_results.json
  aggregate_metrics.json
  selection_frequency_report.json
  selector_evolution_report.json
  counterfactual_report.json
  mutation_summary.json
  deterministic_replay.json
  decision.json
  checker_summary.json
  report.md
  selector_history.jsonl
  row_level_samples.jsonl
```

## Decision

```text
E87 is positive.

The Pocket Library can be exposed as dense potential connections while runtime
execution remains sparse and governed.

Next useful step:
test whether this selector can manage multiple capability families, not only
CALC-SCRIBE trace validation.
```
