# STABLE_LOOP_PHASE_LOCK_111R_RETENTION_OR_LM_REGRESSION_ANALYSIS Contract

111R is an analysis-only failure analysis for the failed 111 standard distillation run.

It consumes the failed 111 artifacts plus positive 110, 109, and 108A artifacts. It performs no training, no repair, no checkpoint mutation, and no runtime/service/deploy/API/product change.

## Required Upstream State

111R requires the 111 root to be failed with `RAW_OOD_ACCURACY_NOT_IMPROVED`, `runtime_profile = standard`, `decision.next = 111R_RETENTION_OR_LM_REGRESSION_ANALYSIS`, positive training counters, a changed target 111 checkpoint, and unchanged source/release/package hashes.

111R also requires:

- `INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_POSITIVE`
- `DECODER_POLICY_INTEGRATION_POSITIVE`
- `RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_POSITIVE`

## Analysis Requirements

111R must explain why the 111 final raw eval failed despite reduced training loss:

- compare 110 raw eval path with 111 pre/post raw paths
- explain `pre_111_raw_ood_accuracy = 0.0` against prior raw OOD around `0.53`
- quantify `911...` final prompts generating `711...` teacher/train namespace outputs
- measure teacher-forcing loss versus autoregressive rollout behavior
- distinguish retention model collapse from scorer/format mismatch
- record actual sampled train mix proportions
- classify root cause and write a machine-readable next plan

The run must not classify the failure as "more steps needed" unless eval mismatch, namespace leakage, rollout collapse, retention mismatch, and target checkpoint collapse are all disproven.

## Outputs

Generated outputs are restricted to:

```text
target/pilot_wave/stable_loop_phase_lock_111r_retention_or_lm_regression_analysis/
```

Required reports include eval path compatibility, namespace leakage, rollout gap, retention regression, data balance, output collapse, root cause classification, recommended next plan, and human-readable failure samples.

## Boundary

111R does not claim GPT-like assistant readiness, open-domain assistant readiness, production chat, public API, deployment readiness, safety alignment, or that the failed 111 target checkpoint is a release candidate.
