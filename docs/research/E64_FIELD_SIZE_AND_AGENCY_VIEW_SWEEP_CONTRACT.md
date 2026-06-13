# E64 Field Size And Agency View Sweep Contract

## Purpose

E64 decides the first deterministic sizing recommendation for the VRAXION
runtime body before final curriculum/pocket generation starts.

Core question:

```text
What Flow Field, Ground Field, Proposal Field, and Agency View sizes are large
enough for the current architecture without causing unnecessary mutation/search
falloff?
```

## Boundary

E64 is a controlled symbolic/numeric sizing probe. It is not raw language
reasoning, deployed model behavior, consciousness, AGI, or model-scale behavior.

## Candidate Set

Systems compared:

```text
tiny_12x12_all
compact_16x16_all
balanced_24x24_all
proposal_width_64_control
asymmetric_24f_32g_20x80_control
near_28f_32g_20x80_default
wide_32x32_20x80
large_48x48_24x80
oversized_64x64_32x80
proposal_starved_32x32
agency_starved_32x32
```

The intended winning family is not assumed. The asymmetric candidate exists
because working state and stable anchors may have different natural capacities.

## Stages

```text
F0_local_short_commit
F1_active_evidence_trace
F2_proposal_collision_commit
F3_ground_contradiction_check
F4_text_digest_to_flow
F5_adversarial_proposal_flood
F6_multi_cycle_repair_memory
F7_overcapacity_decoy_pressure
```

The adversarial stages test collision, stale/ground conflict, flood, and
overcapacity decoy pressure.

## Metrics

Required per-system and aggregate metrics:

```text
success
trace_exact
false_commit_rate
missed_commit_rate
wrong_confident_rate
overpay_rate
net_utility
flow_capacity_pass
ground_capacity_pass
proposal_slot_pass
proposal_width_pass
agency_view_pass
latency_units
attempts_to_95
deterministic_replay_match_rate
```

## Decision Labels

```text
e64_near_28f_32g_20x80_default_confirmed
e64_wide_32x32_default_required
e64_compact_16x16_sufficient
e64_no_clean_size_within_cost_gate
e64_proposal_or_agency_view_bottleneck
e64_invalid_artifact_detected
```

## Positive Lock

The recommended default must satisfy:

```text
success >= 0.985
false_commit_rate == 0
wide_32x32_20x80 does not beat it by > 0.02 net utility
proposal and Agency bottleneck controls fail in the expected way
deterministic replay passes
checker failure_count == 0
```

## Required Artifacts

```text
backend_manifest.json
field_size_manifest.json
row_level_results.jsonl
system_results.json
stage_metrics.json
capacity_frontier_report.json
agency_view_report.json
proposal_capacity_report.json
recommendation.json
aggregate_metrics.json
decision.json
summary.json
deterministic_replay.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
report.md
```

Sample pack:

```text
docs/research/artifact_samples/e64_field_size_and_agency_view_sweep/
```

## Validation

```text
python -m py_compile scripts/probes/run_e64_field_size_and_agency_view_sweep.py scripts/probes/run_e64_field_size_and_agency_view_sweep_check.py
python scripts/probes/run_e64_field_size_and_agency_view_sweep.py
python scripts/probes/run_e64_field_size_and_agency_view_sweep_check.py --write-summary
python scripts/probes/run_e64_field_size_and_agency_view_sweep_check.py --sample-only docs/research/artifact_samples/e64_field_size_and_agency_view_sweep --write-summary
```

The runner must write partial progress and hardware heartbeat artifacts during
the run, not only at final completion.
