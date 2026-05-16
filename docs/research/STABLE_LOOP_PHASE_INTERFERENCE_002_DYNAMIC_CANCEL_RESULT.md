# STABLE_LOOP_PHASE_INTERFERENCE_002_DYNAMIC_CANCEL Result

## Status

Implemented and ran the dynamic-cancel phase-interference probe.

This run fixes the main flaw in `PHASE_INTERFERENCE_001`: the main oracle is no longer first-arrival monotonic BFS. Later waves can modify previous phase state through:

```text
state_next = decay * state + incoming_phase_sum
```

The run also clamps per-cell phase magnitude to prevent corridor feedback from turning the task into an activation-explosion test.

## Runs

```text
K2 smoke:
  out=target/pilot_wave/stable_loop_phase_interference_002_dynamic_cancel/k2_smoke
  phase_classes=2
  seeds=2026,2027
  train_examples=4096
  eval_examples=4096
  epochs=10
  width=32
  jobs=4
  device=cuda
  completed_jobs=18

K4 diagnostic:
  out=target/pilot_wave/stable_loop_phase_interference_002_dynamic_cancel/k4_3seed
  phase_classes=4
  seeds=2026,2027,2028
  train_examples=4096
  eval_examples=4096
  epochs=20
  width=32
  jobs=4
  device=cuda
  completed_jobs=27
```

## Runner Verdicts

K2:

```text
CANONICAL_MESSAGE_PASSING_SUFFICIENT
DYNAMIC_CANCEL_TASK_VALID
DYNAMIC_STABLE_LOOP_POSITIVE
SUMMARY_OR_TARGET_SHORTCUT_RETURNS
```

K4:

```text
CANONICAL_MESSAGE_PASSING_SUFFICIENT
DYNAMIC_CANCEL_TASK_VALID
DYNAMIC_STABLE_LOOP_POSITIVE
SUMMARY_OR_TARGET_SHORTCUT_RETURNS
```

## K2 Summary

| Arm | DynAcc | Cancel | Delay | Late | Reinforce | Gate | Pair | FalseNone | FalsePhase | WallPre | WallPost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ORACLE_DYNAMIC_PHASE_S` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.0000` | `0.0000` |
| `ORACLE_FIRST_ARRIVAL_BASELINE` | `0.467` | `0.349` | `0.132` | `0.166` | `0.136` | `0.134` | `0.131` | `0.000` | `0.316` | `0.4665` | `0.0000` |
| `LOCAL_MESSAGE_PASSING_GNN_DYNAMIC` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.0000` | `0.0000` |
| `HARD_WALL_ABC_DYNAMIC_LOOP` | `0.966` | `0.950` | `0.933` | `0.967` | `0.930` | `0.930` | `0.935` | `0.034` | `0.000` | `0.0000` | `0.0000` |
| `HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP` | `0.966` | `0.950` | `0.933` | `0.967` | `0.930` | `0.930` | `0.935` | `0.034` | `0.000` | `0.0000` | `0.0000` |
| `SUMMARY_DIRECT_HEAD` | `0.727` | `0.697` | `0.603` | `0.630` | `0.660` | `0.562` | `0.513` | `0.094` | `0.153` | `0.0000` | `0.0000` |
| `TARGET_MARKER_ONLY` | `0.521` | `0.727` | `0.667` | `0.502` | `0.660` | `0.660` | `0.087` | `0.479` | `0.000` | `0.0000` | `0.0000` |
| `UNTIED_LOCAL_CNN_TARGET_READOUT_DYNAMIC` | `0.724` | `0.813` | `0.801` | `0.685` | `0.794` | `0.796` | `0.510` | `0.193` | `0.065` | `0.0000` | `0.0000` |

K2 confirms the task change: the first-arrival baseline collapses to `0.467`. This is no longer first-arrival phase-BFS.

But Prismion does not separate from ABC:

```text
Prismion - ABC:  mean  0.0000, positive 0/2
Prismion - GNN:  mean -0.0338, positive 0/2
Prismion - UntiedLocal: mean +0.2424, positive 2/2
```

## K4 Summary

| Arm | DynAcc | Cancel | Delay | Late | Reinforce | Gate | Pair | FalseNone | FalsePhase | WallPre | WallPost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ORACLE_DYNAMIC_PHASE_S` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.0000` | `0.0000` |
| `ORACLE_FIRST_ARRIVAL_BASELINE` | `0.423` | `0.351` | `0.134` | `0.167` | `0.132` | `0.134` | `0.133` | `0.103` | `0.282` | `0.4226` | `0.0000` |
| `LOCAL_MESSAGE_PASSING_GNN_DYNAMIC` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.0000` | `0.0000` |
| `HARD_WALL_ABC_DYNAMIC_LOOP` | `0.971` | `0.951` | `0.934` | `0.966` | `0.935` | `0.933` | `0.937` | `0.029` | `0.000` | `0.0000` | `0.0000` |
| `HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP` | `0.971` | `0.951` | `0.934` | `0.966` | `0.935` | `0.933` | `0.937` | `0.029` | `0.000` | `0.0000` | `0.0000` |
| `SUMMARY_DIRECT_HEAD` | `0.766` | `0.731` | `0.652` | `0.647` | `0.653` | `0.651` | `0.550` | `0.112` | `0.108` | `0.0000` | `0.0000` |
| `TARGET_MARKER_ONLY` | `0.472` | `0.729` | `0.668` | `0.499` | `0.670` | `0.667` | `0.092` | `0.528` | `0.000` | `0.0000` | `0.0000` |
| `UNTIED_LOCAL_CNN_TARGET_READOUT_DYNAMIC` | `0.724` | `0.803` | `0.777` | `0.688` | `0.801` | `0.749` | `0.530` | `0.201` | `0.056` | `0.0000` | `0.0000` |

K4 keeps the same conclusion:

```text
Prismion - ABC:  mean  0.0000, positive 0/3
Prismion - GNN:  mean -0.0292, positive 0/3
Prismion - UntiedLocal: mean +0.2470, positive 3/3
```

## Interpretation

Dynamic cancellation worked as a task hardening patch:

```text
first-arrival baseline:
  K2 = 0.467
  K4 = 0.423
```

So the new task is genuinely non-monotonic relative to v1.

However, it still does not prove Prismion. The canonical dynamic message-passing baseline solves the oracle exactly, and ABC matches Prismion on the learned hard-wall loop:

```text
GNN_DYNAMIC = 1.000
ABC_DYNAMIC = Prismion_DYNAMIC
```

The current correct claim is:

```text
DYNAMIC_CANCEL_TASK_VALID
DYNAMIC_STABLE_LOOP_POSITIVE
CANONICAL_MESSAGE_PASSING_SUFFICIENT
SUMMARY_OR_TARGET_SHORTCUT_RETURNS
```

`PRISMION_DYNAMIC_CANCEL_POSITIVE` is not supported.

## Main Blockers

1. The deterministic dynamic oracle is still directly expressible by canonical local message passing.
2. Prismion and ABC currently implement the same effective vector update on this task.
3. The summary head remains too high:

```text
SUMMARY_DIRECT_HEAD:
  K2 = 0.727
  K4 = 0.766
```

This is much lower than phase v1, but still above a clean adversarial threshold.

## Next Experiment

Do not run more deterministic dynamic cancel with the same oracle. It has answered the question:

```text
dynamic cancel is valid,
but canonical message passing is sufficient.
```

The next useful discriminator should add a feature that is hard for simple additive local message passing:

```text
STABLE_LOOP_PHASE_INTERFERENCE_003_NONLINEAR_PHASE_LOCK
```

Candidate hardening:

```text
phase-locking threshold / hysteresis
local resonance bands
state-dependent gate that changes future propagation
non-commutative update order
dynamic source timing randomized per example
less family-regular geometry to reduce summary shortcut
learned GNN baseline instead of exact hand-coded dynamic oracle as the fair comparison
```

## What This Does Not Prove

```text
does not prove consciousness
does not prove full VRAXION
does not prove language grounding
does not prove Prismion is uniquely required
does not invalidate stable-loop wavefront positive
only tests the local tied-loop dynamic phase-cancel mechanism implemented here
```
