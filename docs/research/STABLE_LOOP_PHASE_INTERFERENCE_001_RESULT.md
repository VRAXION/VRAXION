# STABLE_LOOP_PHASE_INTERFERENCE_001 Result

## Status

Implemented and ran GPU diagnostics for K2 and K4 phase-interference wavefronts.

Important implementation note: an early Prismion run produced NaNs from the fixed phase readout at exact zero vectors. The runner now uses an epsilon-stabilized magnitude in the fixed readout, and the tables below use only the `nanfix` runs.

## Runs

```text
K2 smoke:
  out=target/pilot_wave/stable_loop_phase_interference_001/k2_smoke_nanfix
  phase_classes=2
  seeds=2026,2027
  train_examples=4096
  eval_examples=4096
  epochs=10
  width=32
  jobs=4
  device=cuda
  completed_jobs=14

K4 diagnostic:
  out=target/pilot_wave/stable_loop_phase_interference_001/k4_3seed_nanfix
  phase_classes=4
  seeds=2026,2027,2028
  train_examples=4096
  eval_examples=4096
  epochs=20
  width=32
  jobs=4
  device=cuda
  completed_jobs=21
```

## Runner Verdicts

K2:

```text
ABC_CEILING_TASK_TOO_EASY
CANONICAL_MESSAGE_PASSING_SUFFICIENT
PHASE_INTERFERENCE_TASK_VALID
SUMMARY_OR_TARGET_SHORTCUT_RETURNS
```

K4:

```text
ABC_CEILING_TASK_TOO_EASY
CANONICAL_MESSAGE_PASSING_SUFFICIENT
PHASE_INTERFERENCE_TASK_VALID
SUMMARY_OR_TARGET_SHORTCUT_RETURNS
```

## K2 Summary

| Arm | PhaseAcc | Cancel | Reinforce | Opposite | Pair | Decoy | FalseNone | FalsePhase | WallPre | WallPost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ORACLE_PHASE_WAVEFRONT_S` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.0000` | `0.0000` |
| `LOCAL_MESSAGE_PASSING_GNN_PHASE` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.0000` | `0.0000` |
| `HARD_WALL_ABC_PHASE_LOOP` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.0000` | `0.0000` |
| `HARD_WALL_PRISMION_PHASE_LOOP` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.0000` | `0.0000` |
| `SUMMARY_DIRECT_HEAD` | `0.910` | `0.935` | `0.952` | `0.890` | `0.824` | `1.000` | `0.000` | `0.090` | `0.0000` | `0.0000` |
| `TARGET_MARKER_ONLY` | `0.340` | `0.083` | `0.457` | `0.128` | `0.000` | `0.482` | `0.000` | `0.310` | `0.0000` | `0.0000` |
| `UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE` | `0.758` | `0.844` | `0.758` | `0.798` | `0.664` | `0.800` | `0.188` | `0.023` | `0.0000` | `0.0000` |

K2 is task-valid, but not useful for Prismion separation. ABC, GNN, and Prismion all hit ceiling, and the global summary control reaches `0.910`, far above the K2 majority baseline.

## K4 Summary

| Arm | PhaseAcc | Cancel | Reinforce | Opposite | Pair | Decoy | FalseNone | FalsePhase | WallPre | WallPost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ORACLE_PHASE_WAVEFRONT_S` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.0000` | `0.0000` |
| `LOCAL_MESSAGE_PASSING_GNN_PHASE` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.0000` | `0.0000` |
| `HARD_WALL_ABC_PHASE_LOOP` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.0000` | `0.0000` |
| `HARD_WALL_PRISMION_PHASE_LOOP` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.0000` | `0.0000` |
| `SUMMARY_DIRECT_HEAD` | `0.902` | `0.929` | `0.908` | `0.881` | `0.822` | `1.000` | `0.000` | `0.098` | `0.0000` | `0.0000` |
| `TARGET_MARKER_ONLY` | `0.285` | `0.820` | `0.092` | `0.727` | `0.091` | `0.000` | `0.715` | `0.000` | `0.0000` | `0.0000` |
| `UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE` | `0.733` | `0.855` | `0.719` | `0.817` | `0.571` | `0.785` | `0.196` | `0.020` | `0.0000` | `0.0000` |

K4 increases label space, but still does not separate Prismion from canonical local propagation. ABC, GNN, and Prismion all solve the current deterministic first-arrival phase task exactly.

## Matched Deltas

K2:

```text
Prismion - ABC PhaseAcc:          mean  0.0000, positive 0/2
Prismion - GNN PhaseAcc:          mean  0.0000, positive 0/2
Prismion - UntiedLocal PhaseAcc:  mean +0.2417, positive 2/2
Prismion - ABC FalsePhaseRate:    mean  0.0000
Prismion - ABC WallPressure:      mean  0.0000
```

K4:

```text
Prismion - ABC PhaseAcc:          mean  0.0000, positive 0/3
Prismion - GNN PhaseAcc:          mean  0.0000, positive 0/3
Prismion - UntiedLocal PhaseAcc:  mean +0.2670, positive 3/3
Prismion - ABC FalsePhaseRate:    mean  0.0000
Prismion - ABC WallPressure:      mean  0.0000
```

## Interpretation

The phase-interference task is valid as a deterministic local-loop diagnostic: oracle, GNN, ABC, and Prismion obey walls, propagate phase, cancel opposite waves, preserve same-target-neighborhood contrasts, and solve K2/K4 under truncated S evaluation.

It is not yet a useful Prismion discriminator. Canonical message passing and the ABC hard-wall loop match Prismion exactly on the current task. The correct verdict is:

```text
PHASE_INTERFERENCE_TASK_VALID
CANONICAL_MESSAGE_PASSING_SUFFICIENT
ABC_CEILING_TASK_TOO_EASY
SUMMARY_OR_TARGET_SHORTCUT_RETURNS
```

`PRISMION_PHASE_INTERFERENCE_POSITIVE` is not supported by this run.

## Main Blocker

The current first-arrival phase oracle is too clean and too structurally aligned with ordinary local message passing. Once hard-wall propagation and fixed phase readout are provided, ABC/GNN do not need a special phase primitive.

The global `SUMMARY_DIRECT_HEAD` also remains high (`0.910` on K2, `0.902` on K4), which means the generated corridors/families still expose shortcutable global structure.

## Next Experiment

Do not run more first-arrival K2/K4. The next task should make canonical propagation insufficient:

```text
STABLE_LOOP_PHASE_INTERFERENCE_002_DYNAMIC_CANCEL
```

Required hardening:

```text
later waves can cancel or overwrite earlier reached cells
phase state must update after first arrival
multiple collision times per cell
balanced random maze geometry to reduce summary-family shortcut
explicit summary/target-local controls
matched ABC/GNN/Prismion seed/S/bucket deltas
```

This directly tests whether a phase/interference primitive helps when the state is not a monotonic first-arrival BFS map.

## What This Does Not Prove

```text
does not prove consciousness
does not prove full VRAXION
does not prove natural language grounding
does not prove Prismion is uniquely required
does not invalidate stable-loop wavefront positive
only tests the local tied-loop phase-interference mechanism implemented here
```
