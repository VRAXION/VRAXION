# E8F Proposal Memory And Router Commit Probe Contract

## Purpose

`E8F_PROPOSAL_MEMORY_AND_ROUTER_COMMIT_PROBE` tests the hypothesis that pocket output should be treated as a proposal/data packet, not as automatic Stable Flow truth.

E8E showed repeated route-return drift:

```text
oracle reset after each step ~= oracle ceiling
one reset after step 1 barely helps
consumer-sensitive cell replacement only partially helps
```

E8F therefore tests proposal memory and commit control mechanics over the same controlled symbolic/numeric pocket-router proxy.

## Core Model

```text
Stable Flow / Committed State
  -> Router selects Pocket
  -> Pocket produces proposal
  -> Proposal Memory stores proposal
  -> Router/Commit Controller reads Stable Flow + Proposal Memory
  -> committed Stable Flow'
```

Pocket output is not directly accepted as truth unless the system variant explicitly represents the old direct-overwrite baseline.

## Systems

```text
direct_overwrite_baseline
output_feedback_only
proposal_memory_no_commit
proposal_memory_plus_simple_commit
proposal_memory_plus_router_commit_gate
proposal_memory_plus_learned_commit
proposal_memory_plus_per_skill_commit
proposal_memory_ring_buffer
proposal_memory_plus_verifier_pocket
proposal_memory_plus_stepwise_renormalization
oracle_stepwise_commit_reference
dense_graph_danger_control
```

## Required Metrics

```text
composition usefulness
answer accuracy
trace validity
frame_mae to oracle trace
delta_mae to oracle transition
read_mae on next-pocket read cells
drift slope across route steps
first divergence step
worst route step
proposal acceptance/rejection rate
commit correction magnitude
overcommit/undercommit rate
proposal memory utilization
OOD/counterfactual/adversarial usefulness
dense graph trace-vs-answer gap
deterministic replay
checker failure_count
```

## Decision Labels

```text
e8f_proposal_memory_commit_positive
e8f_output_feedback_sufficient
e8f_commit_controller_required
e8f_shared_commit_controller_positive
e8f_per_skill_commit_required
e8f_proposal_trace_memory_positive
e8f_verifier_commit_required
e8f_stepwise_renormalization_positive
e8f_proposal_memory_not_sufficient
e8f_answer_shortcut_trace_invalid
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
proposal_memory_report.json
commit_controller_report.json
temporal_trace_report.json
dense_graph_control_report.json
producer_dynamics_report.json
system_results.json
row_level_samples.json
aggregate_metrics.json
decision.json
summary.json
report.md
deterministic_replay.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
```

## Hard Requirements

- Real row-level eval.
- No semantic labels.
- No oracle write at learned inference except explicit oracle reference arms.
- Dense graph answer improvement is not primary success if trace validity collapses.
- Deterministic replay must hash-match required artifacts.
- Checker must pass with `failure_count = 0`.
- No raw-language, image, AGI, consciousness, deployed-model, or model-scale claim.
