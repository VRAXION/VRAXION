# E43 Proposal Field Shared Thought Matrix Adversarial Probe Contract

## Summary

E43 stress-tests the Proposal Field / Thought Field idea after E41 and E42.

The architectural hypothesis is:

```text
Pocket Operators do not write directly to Flow Field.
Pocket Operators write temporary proposals into a Proposal Field.
Agency Field reads Flow + Ground + Proposal + Trace + Cost views.
Agency Field decides COMMIT / REJECT / DEFER / ASK / CALL / ANSWER.
Stable Flow changes only after Agency COMMIT.
```

This probe is adversarial. It asks whether a shared Proposal Field remains
reliable under collisions, toxic high-confidence proposals, proposal flooding,
stale cycle replay, Ground conflicts, Trace mismatch, location/scale poison,
partial truths, missing valid proposals, and colluding wrong pockets.

## Systems

```text
direct_flow_write_baseline
explicit_single_proposal_packet
shared_proposal_field
per_pocket_proposal_planes
shared_proposal_field_plus_agency
per_pocket_planes_plus_agency
oracle_commit_reference
toxic_pocket_control
proposal_flood_control
stale_proposal_control
```

`oracle_commit_reference` is an invalid ceiling control only.

## Adversarial Families

```text
collision_same_target
toxic_high_confidence
proposal_flood
stale_cycle_replay
ground_conflict
trace_mismatch
location_poison
scale_poison
partial_truth_decoy
no_valid_proposal
colluding_wrong_pockets
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
proposal_field_frames.jsonl
agency_decision_trace.jsonl
collision_map.json
toxic_pocket_report.json
stale_proposal_report.json
shared_vs_per_pocket_plane_summary.json
system_results.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
report.md
```

Sample pack:

```text
docs/research/artifact_samples/e43_proposal_field_shared_thought_matrix_adversarial_probe/
```

## Metrics

```text
agency_decision_success
action_accuracy
trace_exact_rate
false_commit_rate
missed_commit_rate
stale_commit_rate
toxic_commit_rate
collision_resolution_accuracy
ground_conflict_rejection
trace_mismatch_rejection
proposal_flood_resistance
no_valid_proposal_defer_accuracy
write_spread
illegal_direct_flow_write_rate
proposal_field_clear_rate
per_pocket_plane_gain
shared_field_collision_rate
deterministic_replay_match_rate
```

## Decision Labels

```text
e43_shared_proposal_field_adversarial_confirmed
e43_per_pocket_proposal_planes_required
e43_proposal_field_partial_collision_bottleneck
e43_direct_write_baseline_failed_as_expected
e43_invalid_oracle_or_artifact_detected
```

Positive for shared Proposal Field requires:

```text
shared_proposal_field_plus_agency action_accuracy >= 0.95
shared_proposal_field_plus_agency agency_decision_success >= 0.95
false_commit_rate <= 0.01
toxic_commit_rate <= 0.01
stale_commit_rate <= 0.01
collision_resolution_accuracy >= 0.95
no_valid_proposal_defer_accuracy >= 0.95
illegal_direct_flow_write_rate = 0
deterministic replay passes
sample-only checker passes
checker failure_count = 0
```

If per-pocket planes beat shared field under adversarial pressure, the decision
must be `e43_per_pocket_proposal_planes_required`.

## Boundary

E43 is a controlled symbolic/numeric Proposal Field and Agency Field proxy. It
does not prove raw language reasoning, AGI, consciousness, deployed-model
behavior, or model-scale behavior.
