# E49 Pocket Manager Credit Assignment And Lifecycle Probe Result

## Decision

```text
decision = e49_pocket_manager_credit_lifecycle_positive
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = fc00a3c66145d780
```

E49 tested whether call-level `PocketEvaluationEvent` records, delayed
outcome, counterfactual ablation, harm/trace feedback, and reuse/cost signals
can govern pocket lifecycle better than final-answer-only, immediate-only, or
call-count popularity scoring.

## Result Table

```text
| system | lifecycle_accuracy | weighted_lifecycle_credit | promote_correct_core | quarantine_dangerous_specialist | avoid_credit_hijack | delayed_harm_detection | cost_adjusted_utility | OOD_survival | adversarial_survival | wrong_commit_delta | prune_false_positive |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no_manager_random_reuse | 0.182 | 0.450 | 0.000 | 0.500 | 1.000 | 1.000 | -0.118 | 0.577 | 0.518 | 0.214 | 0.400 |
| final_answer_only_score | 0.182 | 0.309 | 1.000 | 0.000 | 0.000 | 0.000 | -0.159 | 0.572 | 0.506 | 0.239 | 0.200 |
| immediate_score_only | 0.182 | 0.323 | 1.000 | 0.250 | 0.000 | 0.000 | -0.198 | 0.566 | 0.496 | 0.269 | 0.400 |
| call_count_popularity_score | 0.091 | 0.155 | 0.500 | 0.000 | 0.000 | 0.000 | -0.351 | 0.486 | 0.416 | 0.363 | 0.600 |
| full_event_credit_manager | 0.909 | 0.986 | 1.000 | 1.000 | 1.000 | 1.000 | 0.298 | 0.724 | 0.696 | 0.000 | 0.000 |
| oracle_lifecycle_reference | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.298 | 0.724 | 0.696 | 0.000 | 0.000 |
```

## Full Manager Policy

The mutation/rollback feature selection loop tried 648 feature-set mutations:

```text
accepted = 3
rejected = 645
rollback_count = 645
attempts_to_95 = 15
```

The final enabled signal groups were:

```json
[
  "adversarial",
  "counterfactual",
  "immediate",
  "reuse"
]
```

The full manager promoted the general evidence lens and unresolved defer guard
to `core`, preserved safe specialists as `specialist` or `active`, deprecated
the credit hijacker/redundant clone, and quarantined or banned delayed poison,
stale trace, spam, and overfit shortcut pockets.

## Interpretation

Final-answer-only, immediate-only, and call-count popularity controls failed
the important governance cases: credit hijacking, delayed poison, cheap spam,
overfit shortcuts, and prune false positives. The full event manager was not
perfectly oracle-exact, but it reached the positive threshold with:

```text
weighted_lifecycle_credit = 0.986
avoid_credit_hijack = 1.000
delayed_harm_detection = 1.000
quarantine_dangerous_specialist = 1.000
wrong_commit_delta = 0.000
prune_false_positive = 0.000
```

This supports locking the Pocket Manager as a lifecycle layer: every pocket call
emits a structured event, accepted proposals receive delayed/counterfactual
credit, dangerous pockets are quarantined or banned, and useful pockets become
core/active/specialist without relying on final-answer correlation alone.

## Boundary

This is a controlled symbolic/numeric Pocket Manager lifecycle probe. It does
not prove raw language reasoning, deployed assistant behavior, model-scale
behavior, AGI, or consciousness.
