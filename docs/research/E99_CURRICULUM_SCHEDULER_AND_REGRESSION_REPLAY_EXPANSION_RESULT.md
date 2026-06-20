# E99 Curriculum Scheduler And Regression Replay Expansion Result

```text
decision = e99_curriculum_scheduler_regression_replay_expansion_confirmed
checker_failure_count = 0
sample_only_checker_failure_count = 0
```

Boundary:

```text
controlled curriculum scheduler proxy
not autonomous open-ended training
not direct promotion from train score
```

## Key Metrics

```text
seeds = 16
validation_curriculum_success_min = 1.000000
validation_curriculum_success_mean = 1.000000
adversarial_curriculum_success_min = 1.000000
adversarial_curriculum_success_mean = 1.000000
validation_gap_target_accuracy_min = 1.000000
validation_replay_coverage_min = 1.000000
validation_adversarial_coverage_min = 1.000000
validation_budget_validity_min = 1.000000
validation_promotion_safety_min = 1.000000
adversarial_forgetting_risk_rate_max = 0.000000
adversarial_unsafe_promotion_rate_max = 0.000000
adversarial_over_budget_rate_max = 0.000000
adversarial_stale_lesson_rate_max = 0.000000
accepted_mutations_total = 128
rejected_mutations_total = 512
rollback_count_total = 512
```

## Stable Operator Candidates

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

## Rejected Controls

```text
random_lesson_selector          -> Quarantine
novelty_only_curriculum         -> Quarantine
easy_only_sampler               -> Quarantine
no_replay_curriculum            -> Quarantine
train_metric_only_promoter      -> Quarantine
budgetless_curriculum_expander  -> Quarantine
stale_gap_repeater              -> Deprecated
gap_detector_echo_clone         -> Redundant
```

## Interpretation

E99 adds scoped curriculum scheduling and regression replay Operators. The
skills detect capability gaps, rank next lessons, preserve replay for prior
stable skills, sample adversarial families, ramp difficulty, enforce compute
budget, precheck promotion gates, and write the next concrete mutation queue.

This is not autonomous open-ended training. It is a controlled training
governance proxy for Operator-library growth.

## Artifacts

```text
target/pilot_wave/e99_curriculum_scheduler_and_regression_replay_expansion/
archived_public_artifact_sample_removed
```
