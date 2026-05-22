# ITERATIVE_REFINEMENT_DYNAMICS_001 Result

`ITERATIVE_REFINEMENT_DYNAMICS_001` tests the output-refeed idea directly:

```text
state_t -> learned transition model -> state_{t+1}
state_{t+1} is fed back as the next input
```

The goal is to distinguish three cases:

```text
teacher-forced transition works, but free-run drifts
teacher-forced transition and free-run both work
final target is reached by shortcut, but per-step dynamics are wrong
```

This is a deterministic toy integer transition probe. It is not a language-model gate, not GPT-like assistant readiness, not production readiness, and not a general reasoning claim.

## Runner

```text
scripts/probes/run_iterative_refinement_dynamics_001.py
```

Default output:

```text
target/pilot_wave/iterative_refinement_dynamics_001/smoke
```

## Task

The transition rule is:

```text
current, target, op -> next_state
```

Example:

```text
current = 10
target = -65
op = -1

10 -> 9 -> 8 -> ... -> -65
```

Training uses shorter trajectories than the longest evaluation cases. Evaluation includes longer heldout free-run trajectories where the model's own output is repeatedly re-fed. Wrong-direction commands are scored as safe hold cases, not target-reaching cases.

## Arms

```text
learned_delta_classifier
checker_guarded_learned_delta_classifier
oracle_transition
direct_to_target_baseline
```

The direct-to-target baseline is a shortcut control. It may reach the target, but it should not match the expected per-step transition path.

## Main Metrics

```text
teacher_forced_transition_accuracy
free_run_convergence_rate
final_target_accuracy
free_run_transition_accuracy
wrong_direction_rate
overshoot_rate
drift_rate
cycle_rate
teacher_forced_vs_free_run_gap
max_stable_horizon
```

## Positive Criteria

```text
checkpoint changes
teacher_forced_transition_accuracy >= 0.995
free_run_convergence_rate >= 0.95
final_target_accuracy >= 0.95
wrong_direction_rate <= 0.01
cycle_rate <= 0.01
teacher_forced_vs_free_run_gap <= 0.05
direct_to_target shortcut rejected by low per-step transition accuracy
```

## Result

Status: `positive`

Default run:

```text
output:                                target/pilot_wave/iterative_refinement_dynamics_001/smoke
max_train_steps:                       60
train_example_count:                   478998
teacher_forced_eval_count:             2992
trajectory_eval_count:                 784
train_loss_initial:                    2.390363
train_loss_final:                      0.000000736
eval_loss_initial:                     2.373175
eval_loss_final:                       0.000005334
teacher_forced_transition_accuracy:    1.000
free_run_convergence_rate:             1.000
final_target_or_safe_hold_accuracy:    1.000
free_run_transition_accuracy:          1.000
teacher_forced_vs_free_run_gap:        0.000
max_stable_horizon:                    150
wrong_direction_rate:                  0.000
overshoot_rate:                        0.000
drift_rate:                            0.000
cycle_rate:                            0.000
checker_repair_rate:                   0.000
wall_clock_sec:                        36.454
```

Verdicts:

```text
ITERATIVE_REFINEMENT_DYNAMICS_POSITIVE
LEARNED_TRANSITION_RULE_STABLE_IN_FREE_RUN
LONG_HORIZON_REFEED_CONVERGES
TEACHER_FORCED_FREE_RUN_GAP_ACCEPTABLE
CHECKPOINT_CHANGED
ORACLE_CONTROL_PASSES
DIRECT_TO_TARGET_SHORTCUT_REJECTED
GPT_LIKE_READINESS_NOT_CLAIMED
PRODUCTION_READINESS_NOT_CLAIMED
```

Arm summary:

```text
learned_delta_classifier:
  convergence:              1.000
  transition accuracy:       1.000
  wrong direction:           0.000
  cycle:                     0.000
  max stable horizon:        150

checker_guarded_learned_delta_classifier:
  convergence:              1.000
  transition accuracy:       1.000
  checker repair rate:       0.000

oracle_transition:
  convergence:              1.000
  transition accuracy:       1.000

direct_to_target_baseline:
  target/safe-hold success:  0.939
  transition accuracy:       0.155
```

The direct-to-target baseline is rejected as an iterative mechanism: it can move toward the final target, but it does not reproduce the required step-by-step transition path.

## No-Stop Control

The original positive gate uses an external loop stop:

```text
stop when target/safe-hold is reached
```

A no-stop control was added to test what happens if that external stop is removed and the learned model is forced to keep running for `20` extra ticks after the stop state should have been reached.

Without explicit target-hold training:

```text
output:                                  target/pilot_wave/iterative_refinement_dynamics_001/no_stop_control
verdict:                                 NO_STOP_EXTERNAL_STOP_REQUIRED
teacher_forced_transition_accuracy:      1.000
free_run_convergence_rate:               1.000
free_run_transition_accuracy:            1.000
no_stop_reached_stop_state_rate:         1.000
no_stop_final_at_stop_state_rate:        0.061
no_stop_exit_after_stop_state_rate:      0.939
no_stop_post_stop_zero_delta_rate:       0.578
no_stop_mean_abs_error_after_extra:      1.903
no_stop_runaway_rate:                    0.000
```

Interpretation:

```text
The model reaches the stop state, but without an external stop it usually leaves it again.
It does not run away catastrophically, but it does not reliably self-disable the loop.
```

With explicit target-hold examples added to training:

```text
output:                                  target/pilot_wave/iterative_refinement_dynamics_001/no_stop_target_hold_aug
verdict:                                 NO_STOP_INTERNAL_HOLD_EMERGED
teacher_forced_transition_accuracy:      1.000
free_run_convergence_rate:               1.000
free_run_transition_accuracy:            1.000
no_stop_final_at_stop_state_rate:        1.000
no_stop_exit_after_stop_state_rate:      0.000
no_stop_post_stop_zero_delta_rate:       1.000
no_stop_mean_abs_error_after_extra:      0.000
no_stop_runaway_rate:                    0.000
```

Interpretation:

```text
Internal loop shutdown is learnable when the target-reached hold state is part of the transition objective.
It does not appear automatically from the ordinary move-toward-target transition rule.
```

## Strict-Short Control

A stricter `max_train_steps=20` control was also run:

```text
output:                                target/pilot_wave/iterative_refinement_dynamics_001/strict_short20
train_example_count:                   80628
teacher_forced_transition_accuracy:    0.993984
free_run_convergence_rate:             1.000
final_target_or_safe_hold_accuracy:    1.000
free_run_transition_accuracy:          0.999322
wrong_direction_rate:                  0.000
cycle_rate:                            0.000
max_stable_horizon:                    150
```

This near-miss failed only the strict teacher-forced threshold. It shows that the iterative loop itself can remain stable even with much less coverage, but exact one-step generalization benefits from broader transition coverage.

## Interpretation

The output-refeed mechanism is viable in this toy setting when the model is trained as a transition function rather than as a one-shot answer function:

```text
current_state -> learned delta -> next_state -> refeed
```

The important result is not simply that the target was reached. The learned arm also matched the expected per-step transition path, avoided wrong-direction updates, avoided overshoot, avoided cycles, and generalized to longer heldout trajectories than the training horizon.

The result does not imply that arbitrary language-model outputs improve by being repeatedly re-fed. The loop needs an explicit state representation, an update rule objective, a stop condition, and ideally a checker.

The no-stop control makes the stop condition boundary explicit:

```text
without target-hold training:
  external stop/checker is required

with target-hold training:
  the model can learn the internal zero-delta hold transition
```

## Claim Boundary

This probe only tests toy transition dynamics and output-refeed stability. It does not prove open-domain reasoning, natural language understanding, GPT-like assistant capability, production readiness, or safety alignment.
