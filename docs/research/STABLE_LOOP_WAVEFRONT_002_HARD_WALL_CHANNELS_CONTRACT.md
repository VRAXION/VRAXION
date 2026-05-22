# STABLE_LOOP_WAVEFRONT_002_HARD_WALL_CHANNELS Contract

## Purpose

Test whether learned tied local loops become clean wavefront mechanisms when wall writes are structurally impossible and `reached` / `frontier` are explicit state channels.

This is a targeted patch after `STABLE_LOOP_WAVEFRONT_001`:

```text
summary shortcut fixed
target-marker shortcut fixed
canonical local wavefront works
learned loops improve with S but leak through walls / bad reach dynamics
```

The question is not whether the model learns walls from scratch. The first claim is whether hard physical masking enables a clean repeated local update.

## State And Readout

Learned hard-wall arms maintain explicit channels:

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

The main hard-wall answer is fixed:

```text
reachable = reached[target_cell] > 0.5
```

No learned answer head is allowed for the main verdict.

## Wall Pressure Audit

Hard masking makes post-mask leak easy to zero, so report both pre-mask and post-mask behavior:

```text
pre_mask_frontier_wall_write_norm
pre_mask_reached_wall_write_norm
post_mask_frontier_wall_leak_rate
post_mask_reached_wall_leak_rate
latent_wall_write_norm
```

Use separate verdicts:

```text
HARD_WALL_WAVEFRONT_POSITIVE
  hard physical masking enables clean propagation

LEARNED_WALL_GATE_POSITIVE
  pre-mask wall write pressure is also low

MASK_DOES_THE_WORK_WARNING
  post-mask leak is low but pre-mask wall pressure is high
```

## Arms

```text
ORACLE_FULL_BFS
ORACLE_TRUNCATED_BFS_S
SUMMARY_DIRECT_HEAD
TARGET_MARKER_ONLY
UNTIED_CNN_MATCHED_COMPUTE
LOCAL_MESSAGE_PASSING_GNN
HARD_WALL_REACHED_FRONTIER_LOOP
HARD_WALL_ABC_LOOP
HARD_WALL_HIGHWAY_SIDEPOCKET_LOOP
HARD_WALL_PRISMION_PHASE_LOOP
```

Training modes:

```text
FIXED_S
VARIABLE_S_TRAINING
```

`VARIABLE_S_TRAINING` samples `S in {2,4,8,16}` during training and evaluates `1,2,4,8,16,24,32`.

## Dataset

Reuse the v1 balanced wavefront generator and keep:

```text
same target-neighborhood contrasts
near-miss blocked targets
reachable distractor islands
open rooms
long corridors
one-cell bottlenecks
spiral corridors
disconnected chambers
Euclidean-close / path-far targets
```

Report `maze_family_accuracy` and `DATASET_FAMILY_BIAS_WARNING` if one family dominates.

## Metrics

Report:

```text
reachable_accuracy
heldout_larger_grid_accuracy
long_path_accuracy
distance_bucket_accuracy
acc_d_le_S
acc_d_gt_S
false_reach_d_gt_S
propagation_curve_score
same_target_neighborhood_pair_accuracy
settling_gain
s_generalization_gap
fixed_s_accuracy
variable_s_accuracy
overrun_stability
noise_recovery_accuracy
reached_monotonicity_violation_rate
frontier_decay_rate
frontier_stuck_rate
pre_mask_frontier_wall_write_norm
pre_mask_reached_wall_write_norm
post_mask_frontier_wall_leak_rate
post_mask_reached_wall_leak_rate
latent_wall_write_norm
summary_direct_gap
untied_compute_gap
locality_audit
maze_family_accuracy
```

## Verdicts

```text
HARD_WALL_WAVEFRONT_POSITIVE
LEARNED_WALL_GATE_POSITIVE
MASK_DOES_THE_WORK_WARNING
SUMMARY_SOLVES_TASK
UNTIED_COMPUTE_SOLVES_TASK
TARGET_MARKER_SHORTCUT
LOCALITY_LEAK_WARNING
WALL_GATE_FAILURE
FRONTIER_STUCK_WARNING
NO_SETTLING_GAIN
PRISMION_HARD_WALL_POSITIVE
DATASET_FAMILY_BIAS_WARNING
TASK_TOO_EASY
TASK_TOO_HARD
```

Positive gate:

```text
ORACLE_FULL_BFS >= 0.99
TARGET_MARKER_ONLY long path near chance
SUMMARY_DIRECT_HEAD long path near chance
locality_audit == pass
post-mask wall leak <= 0.02
positive settling_gain
acc_d_le_S high
false_reach_d_gt_S low
same_target_neighborhood_pair_accuracy strong
reached monotonicity violations low
frontier does not stay active forever
best hard-wall loop beats SUMMARY_DIRECT_HEAD by >= 0.15 on long/larger eval
best hard-wall loop is competitive with LOCAL_MESSAGE_PASSING_GNN
Prismion positive only if matched non-Prismion is beaten with equal/lower leak and better stability
```

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
