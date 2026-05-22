# STABLE_LOOP_PHASE_LOCK_030_FAMILY_MIN_ORDER_COMPLETE_TEACHER Contract

## Summary

029 found that generic hard negatives preserve the 028 partial signal but do not recover full route grammar:

```text
sufficient_tick_final_accuracy ~= 0.984
long_path_accuracy ~= 0.957
family_min_accuracy = 0.000
route_order_accuracy ~= 0.769
retained_successor_accuracy ~= 0.779
missing_successor_count ~= 5.3
```

030 tests the exact remaining failure mode:

```text
family-min collapse
missing successor links
incomplete ordered route recovery
high-aggregate-but-not-complete traps
```

No public `instnct-core` API changes.

## Required Arms

```text
HAND_GRAMMAR_SUPERVISION_REFERENCE
COUNTERFACTUAL_CORRUPTION_029_BASELINE
HARD_NEGATIVE_MIXED_029_BASELINE

MISSING_SUCCESSOR_TARGETED_TEACHER
FAMILY_MIN_TARGETED_TEACHER
ORDER_COMPLETION_TEACHER
WORST_FAMILY_REPLAY_TEACHER
HIGH_AGGREGATE_LOW_FAMILY_REPLAY
SUCCESSOR_COVERAGE_TEACHER
ORDER_COMPLETION_PLUS_FAMILY_MIN_TEACHER
MIXED_TARGETED_TEACHER

RANDOM_TARGETED_TEACHER_CONTROL
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
family_min_repair_rate
teacher_label_coverage
teacher_label_noise_rate
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_control_accuracy
```

## Verdicts

```text
FAMILY_MIN_ORDER_TEACHER_POSITIVE
MISSING_SUCCESSOR_TEACHER_WORKS
FAMILY_MIN_TEACHER_WORKS
ORDER_COMPLETION_TEACHER_WORKS
WORST_FAMILY_REPLAY_WORKS
MIXED_TARGETED_TEACHER_WORKS
GENERIC_HARD_NEGATIVES_STILL_INSUFFICIENT
EXTERNAL_GRAMMAR_TEACHER_STILL_REQUIRED
RANDOM_TARGETED_TEACHER_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Decision Gate

```text
FAMILY_MIN_ORDER_TEACHER_POSITIVE if a non-hand targeted teacher reaches:

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

If targeted teacher improves aggregate delivery but family-min remains 0:

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
targeted_teacher_detail_metrics.jsonl
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

030 can support that targeted family-min/order-complete teacher labels close the weak-label grammar gap in the toy phase-lane substrate. It does not prove autonomous teacher discovery, production routing, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.
