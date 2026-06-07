# E8E Temporal Thought-Frame Divergence Audit Result

## Decision

```text
primary_decision = e8e_answer_shortcut_trace_invalid
gpu_confirm_decision = e8e_recoverable_state_drift
primary_checker_failure_count = 0
gpu_checker_failure_count = 0
primary_deterministic_replay_passed = true
gpu_deterministic_replay_passed = true
```

The primary CPU lane found that the dense graph danger control can score higher than the clean learned trace while having much worse trace validity. The GPU confirm did not reproduce the dense-over-clean score ordering, but it did reproduce the main trace-localization signal: repeated oracle resets recover performance, while one early reset does not.

## Primary CPU Evidence

```text
consumer_distill_trace_reference   useful=0.719238 trace=1.000000 frame_mae=0.000000 read_mae=0.000000
oracle_trace_reference             useful=0.719238 trace=1.000000 frame_mae=0.000000 read_mae=0.000000
current_best_learned_trace         useful=0.498261 trace=0.899807 frame_mae=0.100193 read_mae=0.075159 first_div=0.160 slope=0.028131
substrate_first_trace              useful=0.497463 trace=0.899905 frame_mae=0.100095 read_mae=0.075079 first_div=0.152 slope=0.028124
mutation_only_trace                useful=0.497892 trace=0.880261 frame_mae=0.119739 read_mae=0.090020 first_div=0.466 slope=0.032520
dense_graph_danger_trace           useful=0.529742 trace=0.646637 frame_mae=0.353363 read_mae=0.178143 first_div=1.001 slope=0.051148
```

Primary intervention signal for `current_best_learned_trace`:

```text
oracle_reset_after_each_step       0.719238
learned_step_1_oracle_rest         0.717795
consumer_sensitive_replacement     0.517776
oracle_reset_after_step_1          0.506187
baseline learned usefulness        0.498261
```

## GPU Confirm

```text
consumer_distill_trace_reference   useful=0.707031 trace=1.000000 frame_mae=0.000000 read_mae=0.000000
oracle_trace_reference             useful=0.707031 trace=1.000000 frame_mae=0.000000 read_mae=0.000000
current_best_learned_trace         useful=0.569883 trace=0.893249 frame_mae=0.106751 read_mae=0.078879 first_div=0.224 slope=0.032140
substrate_first_trace              useful=0.568918 trace=0.893271 frame_mae=0.106729 read_mae=0.078804 first_div=0.229 slope=0.032271
mutation_only_trace                useful=0.561256 trace=0.880789 frame_mae=0.119211 read_mae=0.088253 first_div=0.479 slope=0.035258
dense_graph_danger_trace           useful=0.526740 trace=0.623518 frame_mae=0.376482 read_mae=0.183360 first_div=1.000 slope=0.045383
```

GPU intervention signal for `current_best_learned_trace`:

```text
oracle_reset_after_each_step       0.707031
learned_step_1_oracle_rest         0.705500
consumer_sensitive_replacement     0.585773
oracle_reset_after_step_1          0.570739
baseline learned usefulness        0.569883
```

## Interpretation

The learned trace does not simply fail because a single first write is unrecoverable. A one-time reset after step 1 gives little improvement, while resetting at each route step recovers almost to the oracle/reference ceiling. That points to repeated route-boundary state drift or commit-format drift.

Consumer-sensitive replacement helps only modestly, so the mismatch is not isolated to the next consumer read-mask cells. The drift is distributed enough that a consumer-facing cell patch is not sufficient.

The worst drift is usually route step 3. Primary CPU worst skill is `verify`; GPU confirm worst skill is `counterfactual_flip`. The stable shared pattern is therefore step-depth accumulation, not one uniquely bad skill.

Substrate-first trace is nearly identical to the current learned trace. This reinforces E8D: snapshot/substrate pretraining did not create a meaningfully better process trajectory.

## Recommended Next Fix

Run a narrow route-return/commit-boundary probe:

```text
E8F_FLOW_RETURN_COMMIT_AND_STEPWISE_RENORMALIZATION_PROBE
```

Test only minimal fixes at the router-return boundary:

```text
learned pocket write -> mechanical commit/renormalize -> next pocket
```

Do not add a new router, semantic lanes, or a dense graph. The target is the repeated state transition boundary, because E8E says stepwise reset works but one-time reset and consumer-cell replacement are not enough.
