# E08 Flow Matrix Writeback Schema Confirm Contract

## Purpose

`E08_FLOW_MATRIX_WRITEBACK_SCHEMA_CONFIRM` follows
`E07_BINARY_FLOW_MATRIX_POCKET_SCHEDULING_CONFIRM`.

E07 confirmed that cheap always-on detector layers, triggered complex pockets,
temporal rollout selection, and gated writeback can stabilize a synthetic
binary Flow Matrix runtime. E08 narrows the question to writeback itself:

```text
Can multiple pocket/pipeline blocks write back into one Flow Matrix through a
shared schema while preserving useful updates and rejecting destructive writes?
```

## Search Context

Before this milestone was added, current remote refs were fetched and the repo
was searched across local and remote branches in:

```text
docs/research/
scripts/probes/
docs/wiki/
CHANGELOG.md
```

Search terms included:

```text
E08
FLOW_MATRIX_WRITEBACK
Flow Matrix writeback
common matrix language
shared writeback schema
pocket writeback
branch contamination
destructive overwrite
stale proposal
region operator schema
detector wiring
gate commit
rollback writeback
Flow-grid operation schema
```

Closest related files:

```text
docs/research/E7S_FLOW_GRID_VISUAL_DEBUG_AUDIT_CONTRACT.md
docs/research/E8F_PROPOSAL_MEMORY_AND_ROUTER_COMMIT_PROBE_CONTRACT.md
docs/research/E8H4_REGION_OPERATOR_COMPOSITION_SCALE_PROBE_CONTRACT.md
```

Those files are related but not equivalent. E7S is a visual/debug audit. E8F
tests proposal memory and commit control in a numeric pocket-router proxy. E8H4
tests region-operator composition and scale. None is a dedicated stdlib-only
binary Flow Matrix writeback-schema test with direct overwrite, private dialect,
shared schema, gate/commit, rollback, and region-operator arms.

## Runner And Checker

- Runner: `scripts/probes/run_e08_flow_matrix_writeback_schema_confirm.py`
- Checker: `scripts/probes/run_e08_flow_matrix_writeback_schema_confirm_check.py`
- Default artifact root: `target/pilot_wave/e08_flow_matrix_writeback_schema_confirm/`

## Model

The runtime is a synthetic binary Flow Matrix:

```text
signal bits
distractor bits
branch bits
lock bits
trace/version bits
stale-state markers
target/write regions
```

The Main Matrix is a stable schema for writeback operations:

```text
read_region
detector_id
condition
transform_op
write_region
branch_id
lock_mask
trace_before
trace_after
confidence
cost
reason_code
```

Pocket blocks are deterministic detector/wiring/operation packets. They have no
neural weights, no gradient training, and no external datasets.

## Detector Families

Allowed detector examples:

```text
ANY(region)
COUNT(region) >= k
XOR/parity
EDGE(region)
GAP(region)
MATCH(template)
CHANGED(region)
BRANCH_ACTIVE(branch)
TARGET_UNLOCKED(slot)
CONFLICT(region_a, region_b)
STALE_TRACE(proposal, flow)
```

## Transform Operations

Allowed transform operations:

```text
SET
CLEAR
FLIP
COPY
MOVE
XOR_INTO
INHIBIT
LOCK
UNLOCK
SHIFT
FILL_GAP
DELETE_ISOLATED
```

## Arms

```text
DIRECT_OVERWRITE_BASELINE
LOCAL_DIALECT_BASELINE
COMMON_SCHEMA_NO_GATE
COMMON_SCHEMA_GATED
COMMON_SCHEMA_GATED_WITH_ROLLBACK
REGION_OPERATOR_SCHEMA
```

## Stress Cases

The runner must include controlled cases for:

```text
two pockets writing the same region
overlapping write regions
stale proposal based on older Flow state
branch mismatch
target locked
wrong detector with high confidence
correct detector with wrong target
noisy Flow state
partial Flow corruption
long temporal rollout
reversed pocket execution order
conflicting SET/CLEAR, FLIP/LOCK, COPY/INHIBIT operations
branch contamination attempt
destructive overwrite attempt
false high-confidence proposal
```

## Metrics

Required metrics:

```text
flow_integrity
final_state_accuracy
useful_writeback_recall
wrong_writeback_rate
destructive_overwrite_rate
branch_contamination_rate
schema_violation_rate
stale_write_rejection_rate
conflict_resolution_success
rollback_success_rate
trace_validity
temporal_drift_rate
oscillation_rate
attractor_collapse_rate
gate_false_accept_rate
gate_false_reject_rate
cost_per_tick
deterministic_replay_passed
no_neural_dependency_detected
no_overclaim_boundary_preserved
```

## Positive Gate

A pass requires `COMMON_SCHEMA_GATED_WITH_ROLLBACK` or
`REGION_OPERATOR_SCHEMA` to beat `DIRECT_OVERWRITE_BASELINE`:

```text
final_state_accuracy improved or preserved
useful_writeback_recall >= 0.75
destructive_overwrite_rate reduced by at least 70%
branch_contamination_rate reduced by at least 90%
wrong_writeback_rate reduced by at least 50%
schema_violation_rate == 0 for shared-schema arms
trace_validity higher than direct overwrite
stale writes rejected or safely rolled back
temporal_drift_rate lower than direct overwrite
deterministic_replay_passed = true
no neural libraries imported
no broad deployment or model-scale claim
```

## Decisions

Allowed decisions:

```text
e08_flow_matrix_writeback_schema_confirmed
e08_direct_overwrite_remains_best
e08_common_schema_not_sufficient
e08_gate_too_conservative
e08_branch_contamination_not_fixed
e08_stale_write_rollback_failure
e08_trace_validity_failure
e08_invalid_or_incomplete_run
```

## Boundary

E08 is a controlled synthetic binary-matrix writeback probe. It does not make
language, image, deployment, model-scale, or broad capability claims.
