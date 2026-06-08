# E08 Flow Matrix Writeback Schema Confirm Result

Status: completed.

## Decision

```text
decision = e08_flow_matrix_writeback_schema_confirmed
next = E09_UNIVERSAL_POCKET_TRANSFORM_BLOCK_CONFIRM
best_schema_arm = REGION_OPERATOR_SCHEMA
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e08_flow_matrix_writeback_schema_confirm/
```

## Search Result

No equivalent E08 Flow Matrix writeback schema confirmation was found after
fetching current remote refs and searching all fetched local/remote branches.

Closest related files:

```text
docs/research/E7S_FLOW_GRID_VISUAL_DEBUG_AUDIT_CONTRACT.md
docs/research/E8F_PROPOSAL_MEMORY_AND_ROUTER_COMMIT_PROBE_CONTRACT.md
docs/research/E8H4_REGION_OPERATOR_COMPOSITION_SCALE_PROBE_CONTRACT.md
```

Coverage summary:

```text
E7S: Flow-grid visualization/debug audit.
E8F: proposal memory and commit control in a numeric proxy.
E8H4: region-operator composition/scale probe.
```

Those files are related but not equivalent to a dedicated binary Flow Matrix
writeback-schema safety test with direct overwrite, private dialect, shared
schema, gate/commit, rollback, stale-write, branch-boundary, and region-operator
arms.

## Key Metrics

| arm | state accuracy | useful recall | wrong write | destructive overwrite | branch contam | trace valid | cost/tick |
|---|---:|---:|---:|---:|---:|---:|---:|
| DIRECT_OVERWRITE_BASELINE | 0.501 | 0.267 | 0.843 | 0.288 | 0.177 | 0.018 | 5.500 |
| LOCAL_DIALECT_BASELINE | 0.500 | 0.000 | 1.000 | 0.354 | 0.367 | 0.213 | 2.000 |
| COMMON_SCHEMA_NO_GATE | 0.507 | 0.275 | 0.812 | 0.319 | 0.188 | 0.018 | 4.000 |
| COMMON_SCHEMA_GATED | 0.731 | 0.771 | 0.539 | 0.212 | 0.000 | 0.231 | 4.400 |
| COMMON_SCHEMA_GATED_WITH_ROLLBACK | 0.764 | 0.893 | 0.492 | 0.187 | 0.000 | 0.319 | 4.800 |
| REGION_OPERATOR_SCHEMA | 0.850 | 0.933 | 0.000 | 0.037 | 0.000 | 0.528 | 2.300 |

Positive-gate deltas for `REGION_OPERATOR_SCHEMA` versus
`DIRECT_OVERWRITE_BASELINE`:

```text
final_state_accuracy_delta = +0.349074
wrong_writeback_rate_reduction = 1.000000
destructive_overwrite_rate_reduction = 0.871629
branch_contamination_rate_reduction = 1.000000
trace_validity_delta = +0.510494
temporal_drift_rate_reduction = 0.699767
```

## Confirmed Findings

The region-operator schema arm preserved useful writeback recall at `0.933038`,
kept schema violations at `0.0`, eliminated branch contamination, rejected stale
write attempts at rate `1.0`, and reduced temporal drift from `0.498843` to
`0.149769`.

The common gated arms improved over direct overwrite, but still allowed more
wrong writebacks than the compact region-operator schema. The no-gate shared
schema arm showed that schema alone is not enough; gate/commit and trace checks
are required to protect Flow.

All required stress case families were present, including overlapping writes,
stale proposals, branch mismatch, locked targets, false high-confidence
proposals, reversed pocket order, noisy/partially corrupted Flow, and
conflicting transform operations.

## Verification

```text
python3 scripts/probes/run_e08_flow_matrix_writeback_schema_confirm.py
python3 scripts/probes/run_e08_flow_matrix_writeback_schema_confirm_check.py --out target/pilot_wave/e08_flow_matrix_writeback_schema_confirm --write-summary
```

The checker passed with `failure_count = 0`.

Boundary: E08 is a deterministic synthetic binary Flow Matrix writeback probe
only. It does not make language, image, deployment, model-scale, or broad
capability claims.
