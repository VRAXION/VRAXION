# STABLE_LOOP_PHASE_LOCK_029_HARD_NEGATIVE_GRAMMAR_DISTILLATION Contract

## Summary

028 found strong partial weak-label signal but no full pass:

```text
best weak labels:
  sufficient_tick_final_accuracy ~= 0.984
  long_path_accuracy ~= 0.957
  family_min_accuracy = 0.000
  route_order_accuracy ~= 0.769
  retained_successor_accuracy ~= 0.779
```

029 tests whether targeted hard negatives can close that gap. The hard negatives are route candidates that look good under aggregate delivery but violate ordered route grammar.

No public `instnct-core` API changes.

## Required Arms

```text
HAND_GRAMMAR_SUPERVISION_REFERENCE
COUNTERFACTUAL_CORRUPTION_028_BASELINE
MIXED_WEAK_LABEL_028_BASELINE
SHORTCUT_DELIVERS_WRONG_ORDER_NEGATIVES
BRANCH_REACHES_TARGET_NEGATIVES
CYCLE_REACHES_TARGET_NEGATIVES
MISSING_SUCCESSOR_ALT_PATH_NEGATIVES
DUPLICATE_DELIVERY_NEGATIVES
STALE_DELIVERY_NEGATIVES
FAMILY_MIN_ADVERSARIAL_NEGATIVES
HIGH_AGGREGATE_LOW_FAMILY_MIN_NEGATIVES
HARD_NEGATIVE_MIXED_DISTILLATION
HARD_NEGATIVE_CURRICULUM_SHORT_TO_LONG
HARD_NEGATIVE_TEACHER_STUDENT
RANDOM_HARD_NEGATIVE_CONTROL
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
hard_negative_precision
hard_negative_recall
hard_negative_false_positive_rate
hard_negative_false_negative_rate
hard_negative_family_coverage
high_aggregate_trap_detection_rate
shortcut_detection_rate
cycle_detection_rate
branch_detection_rate
grammar_precision
grammar_recall
prune_success_rate
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_control_accuracy
```

## Verdicts

```text
HARD_NEGATIVE_GRAMMAR_DISTILLATION_POSITIVE
SHORTCUT_NEGATIVES_REQUIRED
BRANCH_CYCLE_NEGATIVES_REQUIRED
FAMILY_MIN_NEGATIVES_REQUIRED
HIGH_AGGREGATE_TRAP_NEGATIVES_REQUIRED
COUNTERFACTUAL_CORRUPTION_SIGNAL_CONFIRMED
WEAK_LABELS_STILL_INSUFFICIENT
EXTERNAL_GRAMMAR_TEACHER_STILL_REQUIRED
RANDOM_HARD_NEGATIVE_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Decision Gate

Report `HARD_NEGATIVE_GRAMMAR_DISTILLATION_POSITIVE` only if a non-hand hard-negative arm reaches:

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

If aggregate improves but family-min remains zero, report `WEAK_LABELS_STILL_INSUFFICIENT`.

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
hard_negative_metrics.jsonl
distillation_metrics.jsonl
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

029 can support hard-negative grammar-label distillation in toy phase-lane tasks. It cannot claim production routing, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.
