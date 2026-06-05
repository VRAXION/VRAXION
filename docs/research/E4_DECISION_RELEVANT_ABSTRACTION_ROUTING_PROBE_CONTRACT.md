# E4_DECISION_RELEVANT_ABSTRACTION_ROUTING_PROBE Contract

## Purpose

E4 tests whether a small mutable real-backend system can choose the useful answer level before opening unnecessary details.

The controlled routing order is:

```text
verdict/status -> causal family -> mechanism -> evidence/detail
```

This is a symbolic proxy for answer-level selection. It is not a proof of natural-language reasoning.

## Systems

- `flat_detail_scanner`: scores detail/evidence nodes directly and is expected to over-open details.
- `bottom_up_evidence_scanner`: aggregates low-level evidence upward before deciding.
- `top_down_hierarchical_router`: routes verdict -> cause -> mechanism -> evidence.
- `dynamic_state_medium_router`: recurrent/state-medium router over the same symbolic tree features.
- `oracle_reference_only`: sanity reference only, never a learned comparison candidate.

## Required Artifacts

- `e4_backend_manifest.json`
- `e4_task_generation_report.json`
- `e4_routing_report.json`
- `e4_control_baseline_report.json`
- `e4_leakage_sentinel_report.json`
- `e4_no_synthetic_metric_audit.json`
- `e4_deterministic_replay_report.json`
- `e4_accept_reject_rollback_report.json`
- `e4_generation_metrics.json`
- `e4_row_level_eval_sample_heldout.json`
- `e4_row_level_eval_sample_ood.json`
- `e4_row_level_eval_sample_counterfactual.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`
- `progress.jsonl`

For every mutable system:

- `e4_candidate_<system>_initial.json`
- `e4_candidate_<system>_final.json`
- `e4_parameter_diff_<system>.json`
- `e4_mutation_history_<system>.json`

## Required Metrics

- verdict accuracy
- decision-relevant level accuracy
- over-detail rate
- under-detail rate
- irrelevant branch expansion rate
- causal path accuracy
- stopping-depth accuracy
- descend decision accuracy
- top-down path consistency
- answer usefulness score
- detail efficiency score
- heldout, OOD, and counterfactual performance
- accepted and rejected mutations
- rollback count
- before/after parameter diff
- deterministic replay hash comparison

## Decision Rules

- `e4_decision_relevant_abstraction_routing_confirmed`: top-down routing beats flat and bottom-up on verdict, level, branch economy, causal path, usefulness, and generalization.
- `e4_flat_detail_scanning_sufficient`: detail scanning matches or beats routing.
- `e4_answer_level_selection_failure`: verdict is mostly correct but the selected answer level is wrong.
- `e4_overbranching_failure`: answer is often correct but irrelevant branches are expanded too often.
- `e4_leak_or_task_artifact_detected`: route/name/index/correct-label leakage or task artifact controls fail.

## Checker Gates

The checker fails the run on:

- missing artifacts
- missing system variant
- no accepted or rejected mutations
- rollback mismatch
- missing parameter diff
- synthetic/static metrics
- hardcoded improvement flags
- leakage controls failing without leak decision
- deterministic replay failure
- missing progress writeouts
- forbidden optimizer/backprop imports or calls

## Boundary

E4 supersedes the previously suggested nonlinear projection isolation step. It moves the E-series from state-medium mechanics toward decision-relevant abstraction routing.
