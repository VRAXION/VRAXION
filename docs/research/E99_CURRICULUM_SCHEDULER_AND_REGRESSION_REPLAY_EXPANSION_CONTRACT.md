# E99 Curriculum Scheduler And Regression Replay Expansion Contract

## Purpose

E99 expands the Operator Library with training-governance skills. These
Operators choose a next lesson from observed capability gaps while preserving
regression replay, adversarial coverage, difficulty ramping, compute budget, and
promotion prechecks.

This is not autonomous open-ended training. It is a controlled curriculum
scheduler proxy for Operator-library growth.

## Required Boundary

```text
controlled curriculum scheduler proxy
not autonomous open-ended training
not direct promotion from train score
not no-replay curriculum
not model-scale claim
```

## Stable Candidate Targets

```text
capability_gap_detector_lens
lesson_candidate_ranker_scribe
regression_replay_set_guard
adversarial_family_sampler_lens
difficulty_ramp_t_stab
compute_budget_allocator_guard
promotion_gate_precheck_guard
next_mutation_queue_scribe
```

## Controls

```text
random_lesson_selector
novelty_only_curriculum
easy_only_sampler
no_replay_curriculum
train_metric_only_promoter
budgetless_curriculum_expander
stale_gap_repeater
gap_detector_echo_clone
```

## Required Artifacts

```text
run_manifest.json
operator_library_manifest.json
task_generation_report.json
progress.jsonl
partial_aggregate_snapshot.json
seed_results.json
aggregate_metrics.json
selection_frequency_report.json
counterfactual_report.json
operator_lifecycle_report.json
mutation_summary.json
deterministic_replay.json
decision.json
summary.json
report.md
row_level_samples.jsonl
operator_evolution_history.jsonl
```

Sample pack:

```text
docs/research/artifact_samples/e99_curriculum_scheduler_and_regression_replay_expansion/
```

## Metrics

```text
curriculum_success
gap_target_accuracy
replay_coverage
adversarial_coverage
budget_validity
promotion_safety
stale_lesson_rate
forgetting_risk_rate
unsafe_promotion_rate
over_budget_rate
counterfactual curriculum success loss
accepted/rejected/rollback mutation counts
deterministic replay hash match
checker failure count
```

## Pass Requirements

```text
validation_curriculum_success_min = 1.000000
adversarial_curriculum_success_min = 1.000000
validation_gap_target_accuracy_min = 1.000000
validation_replay_coverage_min = 1.000000
validation_adversarial_coverage_min = 1.000000
validation_budget_validity_min = 1.000000
validation_promotion_safety_min = 1.000000
adversarial_forgetting_risk_rate_max = 0.000000
adversarial_unsafe_promotion_rate_max = 0.000000
adversarial_over_budget_rate_max = 0.000000
adversarial_stale_lesson_rate_max = 0.000000
checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic replay passes
```

## Decisions

```text
e99_curriculum_scheduler_regression_replay_expansion_confirmed
e99_curriculum_scheduler_incomplete
e99_replay_coverage_failure
e99_adversarial_coverage_failure
e99_budget_guard_failure
e99_promotion_precheck_failure
e99_artifact_or_replay_failure
```

## Interpretation Rule

A positive result means the library gained scoped training-governance skills:
gap detection, lesson ranking, replay preservation, adversarial sampling,
difficulty ramping, compute budgeting, promotion prechecking, and concrete
mutation queue writing.

It does not mean the system can autonomously train itself without supervision.
