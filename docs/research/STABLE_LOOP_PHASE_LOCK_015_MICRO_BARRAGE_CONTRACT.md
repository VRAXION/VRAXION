# STABLE_LOOP_PHASE_LOCK_015_MICRO_BARRAGE Contract

## Summary

015 is a small mechanism-selection barrage, not a full transport confirm.

Current blocker:

```text
local phase rule works
one-edge transport works
target receives signal
public stabilizers reduce wrong phase but do not solve long-chain transport
```

Goal:

```text
identify which new transport principle has signal against wrong-phase echo
```

This probe does not search, mutate, prune, change public `instnct-core` APIs, or
claim stable transport solved.

## Files

```text
docs/research/STABLE_LOOP_PHASE_LOCK_015_MICRO_BARRAGE_CONTRACT.md
instnct-core/examples/phase_lane_micro_barrage.rs
docs/research/STABLE_LOOP_PHASE_LOCK_015_MICRO_BARRAGE_RESULT.md
```

## Arms

```text
BASELINE_FULL16_014
BEST_PUBLIC_COMBO_014
SIGNED_PHASE_CANCELLATION
DUAL_LAYER_EB_FIELD
MOMENTUM_LANES
EMIT_ONCE_CONSUME
REFRACTORY_CELL
NO_REENTRY_MOMENTUM
PHASE_COMPETITION_PER_CELL
ARRIVAL_WINDOW_READOUT_DIAGNOSTIC
MOMENTUM_PLUS_CONSUME
SIGNED_PLUS_CELL_NORMALIZATION
DUAL_LAYER_PLUS_DAMPING
RANDOM_CONTROL_BASE
RANDOM_CONTROL_WITH_EACH_MECHANISM
```

Momentum is public only: local incoming direction, previous direction, and
source-target geometry are allowed; `true_path` and oracle next-cell direction
are forbidden.

Arrival-window readout is diagnostic-only. If it helps while final readout
fails, report `READOUT_WINDOW_ONLY_NOT_TRANSPORT`.

## Metrics And Outputs

Core ranking metrics:

```text
phase_final_accuracy
long_path_accuracy
family_min_accuracy
target_arrival_rate
wrong_if_arrived_rate
wrong_phase_growth_rate
correct_phase_power
wrong_phase_power
final_minus_best_gap
gate_shuffle_collapse
same_target_counterfactual_accuracy
echo_power
backflow_power
stale_phase_rate
reentry_count
random_control_accuracy
delta_vs_baseline_accuracy
delta_vs_baseline_wrong_if_arrived
```

Micro-signal gate:

```text
long_path_accuracy improves by >= +0.10
family_min_accuracy improves by >= +0.20
wrong_if_arrived_rate drops by >= 0.10
final_minus_best_gap does not increase by > 0.05
random control remains weak
```

Required outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
mechanism_metrics.jsonl
random_control_metrics.jsonl
family_metrics.jsonl
counterfactual_metrics.jsonl
mechanism_ranking.json
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

The runner appends progress and metrics continuously. There is no black-box run.

## Verdicts

```text
SIGNED_CANCELLATION_HAS_SIGNAL
SIGNED_CANCELLATION_KILLS_CORRECT_SIGNAL
DUAL_LAYER_HAS_SIGNAL
DUAL_LAYER_NO_SIGNAL
MOMENTUM_HAS_SIGNAL
MOMENTUM_ONLY_ORACLE_WORKS
CONSUME_REDUCES_ECHO
REFRACTORY_REDUCES_STALE_PHASE
NO_REENTRY_REDUCES_BACKFLOW
PHASE_COMPETITION_HAS_SIGNAL
READOUT_WINDOW_ONLY_NOT_TRANSPORT
MECHANISM_OVERPOWERS_RULE_CONTROL
NO_MICRO_MECHANIC_RESCUES
PRODUCTION_API_NOT_READY
```

## Claim Boundary

015 micro-barrage can support choosing a next mechanism for a full follow-up.
It cannot prove stable transport, production architecture, full VRAXION,
consciousness, language grounding, Prismion uniqueness, or physical quantum
behavior.
