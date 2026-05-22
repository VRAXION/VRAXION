# STABLE_LOOP_PHASE_INTERFERENCE_003_PHASE_LOCK Contract

## Goal

The dynamic-cancel task proved that first-arrival BFS was no longer the shortcut, but it still used an additive vector-sum rule that canonical message passing could implement directly.

This probe changes the primitive being tested:

```text
phase must be transported through local phase gates
each gate rotates the passing wave
target phase depends on the ordered product of local phase rotations
```

The point is not broad architecture search. It is a narrow discriminator:

```text
Does an explicit phase/interference primitive help when the local update is phase multiplication / rotation, not additive reachability?
```

## Task

Grid channels:

```text
wall
target
source_real
source_imag
gate_real
gate_imag
```

A source emits a phase vector. On each local propagation step, a wave entering a cell is multiplied by that cell's gate phase.

Readout:

```text
target_vector = phase at target cell
norm < threshold => NONE
otherwise nearest phase bucket
```

Training signal:

```text
final target phase only
no path labels
no intermediate phase-map supervision
```

## Arms

```text
ORACLE_PHASE_LOCK_S
SUMMARY_DIRECT_HEAD
TARGET_MARKER_ONLY
LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK
HARD_WALL_ABC_PHASE_LOCK_LOOP
HARD_WALL_PRISMION_PHASE_LOCK_LOOP
UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE_LOCK
```

Fair claim:

```text
Prismion positive only if it beats ABC/GNN/untied-local on matched seed/S/path buckets,
while summary and target controls stay weak.
```

## Metrics

```text
phase_lock_accuracy
target_phase_accuracy
phase_bucket_accuracy
none_vs_phase_accuracy
long_path_accuracy
same_target_neighborhood_pair_accuracy
summary_direct_accuracy
target_marker_accuracy
wall_leak
prismion_minus_abc_by_seed
prismion_minus_gnn_by_seed
prismion_minus_untied_by_seed
```

## Claim Boundary

This does not prove consciousness or full VRAXION. It only tests whether explicit phase multiplication is useful in a local tied-loop phase-lock task.
