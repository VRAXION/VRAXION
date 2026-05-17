# STABLE_LOOP_PHASE_LOCK_028_GRAMMAR_LABEL_DISTILLATION Contract

## Summary

027 showed that explicit route-grammar supervision works, while delivery-only/self-supervised grammar remains insufficient.

028 tests whether the 027 grammar labels can be generated or distilled from weaker sources:

```text
solved short routes
dense-prune teacher traces
counterfactual route corruptions
synthetic grammar violations
short->long curriculum
teacher-student self-training
```

This is a teacher-source / label-source probe, not a production routing claim.

## Required Arms

```text
HAND_GRAMMAR_SUPERVISION_REFERENCE
SELF_SUPERVISED_DELIVERY_GRAMMAR_027_BASELINE
DENSE_PRUNE_TEACHER_TRACE_DISTILLATION
SHORT_ROUTE_TEACHER_DISTILLATION
COUNTERFACTUAL_CORRUPTION_LABELS
SYNTHETIC_BRANCH_CYCLE_LABELS
SYNTHETIC_SUCCESSOR_VALIDITY_LABELS
SYNTHETIC_CONTINUITY_LABELS
SHORT_TO_LONG_CURRICULUM
TEACHER_STUDENT_SELF_TRAINING
MIXED_WEAK_LABEL_DISTILLATION
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
branch_count
cycle_count
duplicate_successor_count
missing_successor_count
route_continuity_score
source_to_target_reachability
grammar_precision
grammar_recall
grammar_false_positive_rate
grammar_false_negative_rate
teacher_label_coverage
teacher_label_noise_rate
pseudo_label_accept_rate
pseudo_label_error_rate
short_to_long_transfer_accuracy
prune_success_rate
transfer_success_rate
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_control_accuracy
```

## Verdicts

```text
GRAMMAR_LABEL_DISTILLATION_POSITIVE
DENSE_PRUNE_TEACHER_TRACES_WORK
SYNTHETIC_CORRUPTIONS_WORK
SHORT_ROUTE_CURRICULUM_TRANSFERS
TEACHER_STUDENT_SELF_TRAINING_WORKS
MIXED_WEAK_LABELS_WORK
SELF_SUPERVISED_DELIVERY_STILL_INSUFFICIENT
LABEL_NOISE_TOO_HIGH
SHORT_TO_LONG_TRANSFER_FAILS
RANDOM_LABEL_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
EXTERNAL_GRAMMAR_TEACHER_STILL_REQUIRED
PRODUCTION_API_NOT_READY
```

## Decision Gate

Report `GRAMMAR_LABEL_DISTILLATION_POSITIVE` only if a non-hand, non-random weak-label arm reaches:

```text
sufficient_tick_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
wrong_if_delivered_rate <= 0.10
route_order_accuracy >= 0.90
retained_successor_accuracy >= 0.90
branch_count <= 0.05
cycle_count <= 0.05
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
random controls fail
wall/private/nonlocal/direct leaks = 0
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
distillation_metrics.jsonl
teacher_trace_metrics.jsonl
synthetic_corruption_metrics.jsonl
curriculum_metrics.jsonl
self_training_metrics.jsonl
grammar_metrics.jsonl
delivery_metrics.jsonl
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

No black-box runs: append heartbeat progress and refresh `summary.json` / `report.md` during long runs.

## Claim Boundary

028 can support grammar-label distillation in toy phase-lane tasks. It cannot claim production routing, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.
