# STABLE_LOOP_PHASE_LOCK_031_MISSING_SUCCESSOR_LABEL_ACQUISITION Contract

## Summary

030 showed that targeted missing-successor / order-completion teacher labels close the route-grammar gap:

```text
missing_successor_count = 0
route_order_accuracy = 1.000
retained_successor_accuracy = 1.000
family_min_accuracy = 1.000
```

031 tests whether those labels can be generated from public/diagnostic graph signals instead of being externally supplied.

No public `instnct-core` API changes.

## Required Arms

```text
HAND_TARGETED_TEACHER_REFERENCE
MISSING_SUCCESSOR_TARGETED_TEACHER_030_BASELINE

REACHABILITY_GAP_LABELS
DEAD_END_BACKTRACE_LABELS
DELIVERY_FAILURE_ATTRIBUTION_LABELS
FRONTIER_EXPANSION_TRACE_LABELS
PRUNE_RESIDUAL_MISSING_LINK_LABELS
GRAPH_INVARIANT_SUCCESSOR_LABELS
GRAPH_INVARIANT_CONTINUITY_LABELS
MIXED_AUTONOMOUS_LABELS

RANDOM_LABEL_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

## Metrics

```text
sufficient_tick_final_accuracy
long_path_accuracy
family_min_accuracy
wrong_if_delivered_rate
route_order_accuracy
retained_successor_accuracy
missing_successor_count
duplicate_successor_count
branch_count
cycle_count
source_to_target_reachability
successor_coverage
order_completion_rate
label_precision
label_recall
label_false_positive_rate
label_false_negative_rate
missing_successor_detection_rate
order_gap_detection_rate
dead_end_detection_rate
reachability_repair_rate
label_noise_rate
label_coverage
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_control_accuracy
```

## Verdicts

```text
MISSING_SUCCESSOR_LABEL_ACQUISITION_POSITIVE
REACHABILITY_GAP_LABELS_WORK
DEAD_END_BACKTRACE_LABELS_WORK
DELIVERY_FAILURE_ATTRIBUTION_WORKS
FRONTIER_TRACE_LABELS_WORK
PRUNE_RESIDUAL_LABELS_WORK
GRAPH_INVARIANT_LABELS_WORK
MIXED_AUTONOMOUS_LABELS_WORK
AUTONOMOUS_LABELS_PARTIAL_SIGNAL
EXTERNAL_GRAMMAR_TEACHER_STILL_REQUIRED
RANDOM_LABEL_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Decision Gate

```text
MISSING_SUCCESSOR_LABEL_ACQUISITION_POSITIVE if any autonomous/public diagnostic label source reaches:

sufficient_tick_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
wrong_if_delivered_rate <= 0.10
route_order_accuracy >= 0.90
retained_successor_accuracy >= 0.90
missing_successor_count <= 0.05
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
random controls fail
```

If labels detect some missing successors but family-min remains weak:

```text
AUTONOMOUS_LABELS_PARTIAL_SIGNAL
```

If only hand/reference targeted teacher works:

```text
EXTERNAL_GRAMMAR_TEACHER_STILL_REQUIRED
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
targeted_teacher_metrics.jsonl
family_min_teacher_metrics.jsonl
order_completion_metrics.jsonl
teacher_trace_metrics.jsonl
label_acquisition_metrics.jsonl
grammar_metrics.jsonl
delivery_metrics.jsonl
routing_metrics.jsonl
family_metrics.jsonl
counterfactual_metrics.jsonl
control_metrics.jsonl
locality_audit.jsonl
mechanism_ranking.json
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

## Claim Boundary

031 can support that missing-successor / order-completion labels are acquirable from graph diagnostics in the toy phase-lane substrate. It does not prove production routing, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.
