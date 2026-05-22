# STABLE_LOOP_WAVEFRONT_003_LONG_S_CURVE Contract

## Purpose

Test the hard-wall wavefront loop with the correct S-dependent target.

The critical fix from v2:

```text
Do not score a local loop at S=8 against full reachability for d=16.
```

This run separates two labels:

```text
TRUNCATED_REACHABLE_S
  target is reachable within the current S local propagation steps

FULL_REACHABLE
  target is reachable by unlimited/full BFS
```

S-curve and propagation metrics use `TRUNCATED_REACHABLE_S`. Full reachability is reported separately for large S / covered path buckets.

## Task

Input is a deterministic 2D grid with channels:

```text
wall
start
target
```

The model performs local recurrent propagation for:

```text
S = 4, 8, 16, 24, 32
```

Distance buckets:

```text
1-4
5-8
9-16
17-24
unreachable near-miss
```

## Hard-Wall State

Hard-wall local arms reuse the v2 reached/frontier design:

```text
wall
free_space
target_marker
reached
frontier
learned latent channels
```

After every update:

```text
reached = reached * free_space
frontier = frontier * free_space
```

Main hard-wall verdicts use fixed readout:

```text
reachable = reached[target_cell] > 0.5
```

No learned answer head is used for the main hard-wall loop verdict.

## Arms

```text
ORACLE_FULL_BFS
ORACLE_TRUNCATED_BFS_S
SUMMARY_DIRECT_HEAD
TARGET_MARKER_ONLY
LOCAL_MESSAGE_PASSING_GNN
HARD_WALL_ABC_LOOP
HARD_WALL_PRISMION_PHASE_LOOP
UNTIED_LOCAL_CNN_TARGET_READOUT
UNTIED_GLOBAL_CNN
```

`UNTIED_LOCAL_CNN_TARGET_READOUT` is the fair compute baseline:

```text
3x3 local layers
no global pooling
no flattening
target-cell readout only
```

`UNTIED_GLOBAL_CNN` is a shortcut/control baseline. If it wins, report `GLOBAL_COMPUTE_SOLVES_TASK`, not a fair local compute win.

## Training Modes

```text
FIXED_S
  train and evaluate at the same S

SAME_WEIGHTS_S_CURVE
  train one model with random S from {4,8,16,24}
  evaluate the same weights across S={4,8,16,24,32}
```

Local propagation arms and `UNTIED_LOCAL_CNN_TARGET_READOUT` train against `TRUNCATED_REACHABLE_S`. Global controls may train against full reachability, but they are not used for fair local-loop verdicts.

## Metrics

Required:

```text
truncated_accuracy_by_S
full_reachability_accuracy_at_large_S
acc_d_le_S
acc_d_gt_S
false_reach_d_gt_S
missed_reach_d_le_S
unreachable_false_reach_all_S
same_weights_s_curve_accuracy
monotonic_s_curve_score
propagation_curve_score
s_matches_distance_score
s_over_path_overrun_accuracy
overrun_stability_truncated_oracle
post_mask_frontier_wall_leak_rate
post_mask_reached_wall_leak_rate
pre_mask_frontier_wall_write_norm
pre_mask_reached_wall_write_norm
summary_direct_gap
untied_local_compute_gap
global_compute_gap
```

Overrun stability uses the truncated oracle at the larger S. A target changing from false at S=8 to true at S=16 is correct behavior, not instability.

## Verdicts

```text
PROPAGATION_CURVE_POSITIVE
FULL_REACHABILITY_POSITIVE
FULL_LABEL_USED_TOO_EARLY
SAME_WEIGHTS_S_CURVE_POSITIVE
SUMMARY_SOLVES_TASK
TARGET_MARKER_SHORTCUT
UNTIED_COMPUTE_SOLVES_FAIR_TASK
GLOBAL_COMPUTE_SOLVES_TASK
LOCALITY_LEAK_WARNING
WALL_GATE_FAILURE
FRONTIER_STUCK_WARNING
NO_SETTLING_GAIN
PRISMION_LONG_S_POSITIVE
TASK_TOO_EASY
TASK_TOO_HARD
```

Positive gate:

```text
ORACLE_FULL_BFS >= 0.99
ORACLE_TRUNCATED_BFS_S behaves correctly for each S
SUMMARY_DIRECT_HEAD near chance on long paths
TARGET_MARKER_ONLY near chance on long paths
TRUNCATED_REACHABLE_S is used for S-curve scoring
FULL_REACHABLE is only used for large-S/full-task reporting
acc_d_le_S high
missed_reach_d_le_S low
false_reach_d_gt_S low
unreachable_false_reach_all_S low
same model weights improve or remain consistent across S_eval
post-mask wall leak <= 0.02
pre-mask wall pressure remains low
UNTIED_LOCAL_CNN_TARGET_READOUT is the only fair compute baseline
UNTIED_GLOBAL_CNN is reported only as shortcut/control
```

Prismion is positive only if it beats both `HARD_WALL_ABC_LOOP` and `LOCAL_MESSAGE_PASSING_GNN` on truncated S/bucket metrics with equal or lower leak.

## Run Hygiene

Required outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
summary.json
report.md
survivor_configs.json
wavefront_cases.jsonl
distance_bucket_metrics.jsonl
s_curve_metrics.jsonl
convergence_curves.jsonl
probe_results.jsonl
examples_sample.jsonl
contract_snapshot.md
job_progress/*.jsonl
```

Raw `target/` outputs are not committed.

## Claim Boundary

This is not language, factuality, parser work, consciousness, or full VRAXION. It only tests whether repeated tied local updates can build the correct wavefront when evaluated against the physically valid S-truncated oracle.
