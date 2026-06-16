# E136H Existing Operator Refinement Mutation/Prune Night Cycle Contract

## Purpose

E136H checks the refinement path for already-promoted E132 math-text and E136A
assistant-text operators:

```text
existing operator library
-> broad seed replay
-> label/kernel alignment audit
-> conservative mutation/prune variant selection
-> no destructive committed-library removal
```

The job is intentionally not a first-pass new-operator search. It tests whether
existing operator titles, trigger kernels, and selected variants can be improved
or held back under wider replay evidence.

## Variant Classes

Each existing operator must be assigned one selected refinement class:

```text
semantic_verified_pruned
semantic_tightened_trigger
abstract_kernel_shadow
hold_for_more_evidence
```

E136H separates two questions:

```text
label_alignment_score = does the human-facing title match the activations?
kernel_value_score    = does the trigger kernel still carry useful signal?
```

Low label alignment is not automatically a drop. A useful but hard-to-name
operator can remain as an abstract kernel shadow instead of being destructively
pruned.

## Gates

E136H may confirm only if:

```text
cycles_completed = cycles_requested
operator_count = 34
current_activation_total > 0
selected_activation_total > 0
pruned_activation_total > 0
hard_negative_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0
verified_label_count + tentative_label_tighten_count + abstract_but_useful_count >= 24
mutation_attempt_total > 0
```

## Required Artifacts

The run must emit:

```text
run_manifest.json
progress.jsonl
cycle_metrics.jsonl
mutation_events.jsonl
operator_refinement_results.json
selected_variants.json
label_alignment_report.json
abstract_kernel_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
checker_summary.json
```

## Boundary

This confirms operator governance and conservative refinement evidence only. It
does not claim new neural weights, open-domain assistant readiness, production
runtime pruning, autonomous hidden thought, or destructive removal of existing
library entries.
