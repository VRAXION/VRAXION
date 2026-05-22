# STABLE_LOOP_WAVEFRONT_001 Contract

## Purpose

Test whether a tied recurrent local loop can build global reachability through repeated settling.

This is not a parser, factuality, language, or full VRAXION test. It is a deterministic mechanism probe:

```text
2D grid: wall/start/target
  -> local recurrent loop for S steps
  -> target-cell reach readout
  -> reachable / unreachable
```

The prior `STABLE_LOOP_ATTRACTOR_SWEEP_001` was confounded because `SUMMARY_DIRECT_HEAD` nearly solved the task. This probe makes the main signal spatial and local so that information must propagate one cell per settling step.

## Dataset

Use 2D grids with channels:

```text
wall
start
target
```

Training uses mostly `8x8` grids. Evaluation includes `8x8`, `12x12`, and `16x16`.

Balance reachable and unreachable examples by shortest-path distance bucket:

```text
1-2
3-4
5-8
9-16
17-24
```

Include adversarial negatives:

```text
wall-blocked target
corridor blocked one cell early
disconnected target room
reachable distractor island while target remains unreachable
same target-local-neighborhood pairs with different global connectivity
```

## Oracles

Report two oracle notions:

```text
ORACLE_FULL_BFS
  full grid reachability

ORACLE_TRUNCATED_BFS_S
  target reached within exactly S local propagation steps
```

Per example, store:

```text
distance_to_target
target_reached_by_S
reachable_map
frontier_by_step
```

The BFS maps and frontiers are audit/probe targets only. They are not training labels.

## Locality Rules

Loop arms must be strictly local:

Allowed:

```text
3x3 local convolution
1x1 channel mixing
per-cell channel operations
target-cell-only readout
wall-like outside-grid boundary
```

Forbidden:

```text
global pooling
flattening
attention
spatial layernorm over HxW
batchnorm over spatial maps
direct readout from non-target cells
wraparound padding
```

Report `locality_audit = pass/fail`. If a local S-step loop solves `distance > S` cases too well, report `LOCALITY_LEAK_WARNING`.

## Arms

```text
ORACLE_FULL_BFS
ORACLE_TRUNCATED_BFS_S
SUMMARY_DIRECT_HEAD
TARGET_MARKER_ONLY
MLP_FLATTENED
GRU_FLATTENED
LSTM_FLATTENED
UNTIED_CNN_MATCHED_COMPUTE
LOCAL_MESSAGE_PASSING_GNN
TIED_LOCAL_CA_LOOP
ABC_TIED_LOOP_HARD_WALL
ABC_TIED_LOOP_LEARNED_WALL
HIGHWAY_SIDEPOCKET_WAVE_LOOP
PRISMION_PHASE_WAVE_LOOP
```

Loop steps:

```text
S = 1, 2, 4, 8, 16, 24, 32
```

Training modes:

```text
FIXED_S
VARIABLE_S
```

`VARIABLE_S` trains with random `S in {2,4,8}` and evaluates the full S curve.

## Metrics

Report:

```text
reachable_accuracy
heldout_larger_grid_accuracy
long_path_accuracy
distance_bucket_accuracy
reachable_balance_by_bucket
unreachable_near_miss_accuracy
same_target_neighborhood_pair_accuracy
settling_gain
propagation_curve_score
s_matches_distance_score
overrun_stability
convergence_rate
final_state_delta
noise_recovery_accuracy
wall_leak_rate
boundary_leak_rate
false_reach_rate
false_block_rate
target_marker_only_baseline
target_cell_reach_channel_accuracy
summary_direct_gap
untied_compute_gap
fixed_s_accuracy
variable_s_accuracy
s_curve_accuracy
locality_audit
state_norm_by_step
saturation_rate
```

Post-hoc probes after freezing:

```text
linear_probe_reachable_map_accuracy
linear_probe_frontier_accuracy
MLP_probe_reachable_map_accuracy_separate
frontier_iou_by_step
```

## Verdicts

```text
STABLE_LOOP_WAVEFRONT_POSITIVE
SUMMARY_SOLVES_TASK
UNTIED_COMPUTE_SOLVES_TASK
TARGET_MARKER_SHORTCUT
LOCALITY_LEAK_WARNING
WALL_GATE_FAILURE
LOOP_UNSTABLE
NO_SETTLING_GAIN
STANDARD_RNN_SUFFICIENT
TASK_TOO_EASY
TASK_TOO_HARD
PRISMION_WAVEFRONT_POSITIVE
```

Positive requires:

```text
ORACLE_FULL_BFS >= 0.99
ORACLE_TRUNCATED_BFS_S correct
locality_audit == pass
for distance <= S: high target reach accuracy
for distance > S: no magical local-loop reach unless flagged as leak
accuracy improves as S approaches shortest path length
overrun does not degrade after S exceeds needed path length
best tied/local loop beats SUMMARY_DIRECT_HEAD by >= 0.15 on long-path/larger-grid eval
best tied/local loop beats UNTIED_CNN_MATCHED_COMPUTE on larger-grid heldout
same_target_neighborhood_pair_accuracy favors loop over summary
wall_leak_rate and boundary_leak_rate stay low
```

## Run Hygiene

Write useful partial results continuously:

```text
queue.json
progress.jsonl
metrics.jsonl
summary.json
report.md
survivor_configs.json
wavefront_cases.jsonl
distance_bucket_metrics.jsonl
convergence_curves.jsonl
probe_results.jsonl
examples_sample.jsonl
contract_snapshot.md
job_progress/*.jsonl
```

Default:

```text
--device cpu
--jobs auto50
torch threads per worker = 1
```

Do not commit raw `target/` outputs.
