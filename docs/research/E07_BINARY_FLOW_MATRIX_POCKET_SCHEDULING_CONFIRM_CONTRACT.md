# E07 Binary Flow Matrix Pocket Scheduling Confirm Contract

## Purpose

`E07_BINARY_FLOW_MATRIX_POCKET_SCHEDULING_CONFIRM` tests a non-neural binary
matrix runtime where pocket outputs must pass through a common proposal schema
and a gate/commit boundary before mutating the active flow state.

The specific question is:

```text
Do cheap always-on detector transforms plus triggered complex pockets beat
always-running complex pockets when both are constrained by a common binary
matrix proposal language?
```

## Search Context

Before this file was added, the repo was searched across fetched local and
remote branches for:

```text
E07
FLOW_MATRIX / Flow Matrix / Binary Flow Matrix
MAIN_MATRIX / Main Matrix
Pocket Pipeline / pocket pipeline / pocket transformation
common matrix language / flow matrix language
proposal commit / proposal gate commit
detector wiring
temporal rollout selection
always-on transform / triggered pocket
```

The closest existing E-line files were:

```text
docs/research/E7B_POCKET_ROUTING_COMPOSITION_PROBE_CONTRACT.md
docs/research/E7E_FLOW_PIPE_DRIFT_AND_ROUTER_REPAIR_PROBE_CONTRACT.md
docs/research/E8F_PROPOSAL_MEMORY_AND_ROUTER_COMMIT_PROBE_CONTRACT.md
```

Those are related but incomplete for this contract. E7B tests symbolic routing
over frozen pockets. E7E tests route-around and local repair under flow-pipe
drift. E8F tests proposal memory and commit mechanics in a numeric probe with
non-stdlib dependencies. None of those files test the full binary Flow/Main
matrix, cheap-vs-triggered scheduling, common matrix-language proposal schema,
snapshot-vs-rollout pocket selection, and branch contamination boundary in one
stdlib-only probe.

## Runner And Checker

- Runner: `scripts/probes/run_e07_binary_flow_matrix_pocket_scheduling_confirm.py`
- Checker: `scripts/probes/run_e07_binary_flow_matrix_pocket_scheduling_confirm_check.py`
- Default artifact root: `target/pilot_wave/e07_binary_flow_matrix_pocket_scheduling_confirm/`

## Model

The synthetic runtime uses binary state only:

```text
Flow Matrix:
  active signals
  distractors
  branch bits
  lock bits
  target/output slots
  temporal trace ages

Main Matrix:
  stable proposal schema
  valid signal types
  valid source regions
  valid operations
  valid reason codes
  branch/slot ownership rules
```

Pocket blocks are deterministic detector/operation packets. They have no
gradient training and no learned weights.

## Common Proposal Schema

Every gated mutation must be expressed as:

```text
signal_type
source_region
target_slot
operation
guard
branch_id
confidence
reason_code
```

Direct local pocket dialects are negative-control inputs. They may be counted
as attempted direct mutations in the no-gate baseline, but a gated arm must
reject them before commit.

## Detector Families

The probe includes deterministic detector examples:

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
```

## Operation Families

The runtime supports these operation labels in the common matrix language:

```text
SET
CLEAR
FLIP
COPY
XOR_INTO
INHIBIT
LOCK
UNLOCK
```

The synthetic target slots exercise `SET`, `CLEAR`, and `FLIP`; the remaining
labels are schema-valid vocabulary for contract checking and future extension.

## Arms

```text
SNAPSHOT_SELECTED_POCKET
ROLLOUT_SELECTED_POCKET
ALL_COMPLEX_ALWAYS_NO_GATE
ALL_COMPLEX_ALWAYS_GATED
DEFAULT_ALWAYS_ON_TRIGGERED_COMPLEX_GATED
DEFAULT_ADAPTIVE_TRIGGERED_COMPLEX_GATED
```

`SNAPSHOT_SELECTED_POCKET` selects a candidate using one-step local score.
`ROLLOUT_SELECTED_POCKET` selects with a deterministic multi-step rollout
score. The default arms run cheap detector transforms every tick and invoke
complex pockets only when salience, uncertainty, conflict, or recent trace
activity crosses the trigger rule.

## Metrics

Required metrics:

```text
final_state_accuracy
event_recall
false_positive_rate
useful_update_recall
wrong_commit_rate
destructive_overwrite_rate
branch_contamination_rate
stale_commit_rate
rejected_good_update_rate
gate_false_accept_rate
gate_false_reject_rate
temporal_drift_rate
oscillation_rate
attractor_collapse_rate
complex_calls_per_tick
avg_cost_per_tick
retry_or_trigger_cost
deterministic_replay_passed
```

## Positive Gate

The best default-triggered gated arm must beat the all-complex always no-gate
baseline by:

```text
final_state_accuracy improved or preserved
false_positive_rate lower
wrong_commit_rate reduced by at least 50%
destructive_overwrite_rate reduced by at least 70%
branch_contamination_rate reduced by at least 90%
avg_cost_per_tick reduced by at least 30%
useful_update_recall >= 0.75
deterministic replay passes
no oracle/truth leakage in selection or gate logic
all required reports written consistently
```

The reports must also show:

```text
snapshot-only selection can choose locally attractive pockets that hurt rollout
rollout-selected pockets are more stable over time
direct pocket dialect output is rejected by gated arms
common matrix-language proposal schema is required before commit
```

## Decisions

Allowed decisions:

```text
e07_binary_flow_matrix_pocket_scheduling_confirmed
e07_snapshot_selection_temporal_failure_detected
e07_trigger_policy_too_conservative
e07_branch_contamination_not_fixed
e07_common_matrix_language_contract_failure
e07_invalid_or_incomplete_run
```

## Checker Gates

The checker fails on missing artifacts, invalid decision labels, inconsistent
positive-gate math, failed deterministic replay, non-stdlib imports in the E07
runner, missing common-schema enforcement, direct dialect mutation in gated
arms, branch contamination not reduced by the required margin, or broad claims
outside this controlled synthetic runtime.

## Boundary

E07 is a deterministic synthetic binary-matrix scheduling probe. It does not
claim real-world deployment readiness, model-scale capability, or broad
reasoning behavior.
