# STABLE_LOOP_WAVEFRONT_001 Result

## Run

```text
stage=smoke
seeds=2026,2027
train_examples=1024
eval_examples=1024
epochs=20
jobs=12
device=cpu
completed_jobs=42
```

## Verdict

```json
[
  "LOCALITY_LEAK_WARNING",
  "UNTIED_COMPUTE_SOLVES_TASK",
  "WALL_GATE_FAILURE"
]
```

## Control Gaps

```json
{
  "summary_direct_gap": 0.261352,
  "untied_compute_gap": -0.080593
}
```

## Interpretation

The wavefront dataset now blocks the previous summary shortcut:

```text
SUMMARY_DIRECT_HEAD long_path_accuracy = 0.500
TARGET_MARKER_ONLY long_path_accuracy  = 0.500
summary_direct_gap                     = +0.261
```

The canonical local propagation baseline behaves as expected:

```text
LOCAL_MESSAGE_PASSING_GNN S=1 reachable_accuracy = 0.537
LOCAL_MESSAGE_PASSING_GNN S=4 reachable_accuracy = 0.683
LOCAL_MESSAGE_PASSING_GNN S=8 reachable_accuracy = 0.758
wall_leak_rate = 0.000
```

This confirms the dataset/oracle path is forcing a stepwise wavefront.

The learned loop arms are not clean positives. They improve with more settling, but the mechanism is not clean:

```text
TIED_LOCAL_CA_LOOP S=8 reachable_accuracy = 0.839
TIED_LOCAL_CA_LOOP S=8 wall_leak_rate     = 0.387

PRISMION_PHASE_WAVE_LOOP S=8 reachable_accuracy = 0.877
PRISMION_PHASE_WAVE_LOOP S=8 wall_leak_rate     = 0.451

ABC_TIED_LOOP_HARD_WALL S=8 long_path_accuracy  = 0.534
```

The untied compute baseline is still stronger on long-path heldout:

```text
UNTIED_CNN_MATCHED_COMPUTE S=8 long_path_accuracy = 0.874
best learned local loop long_path_accuracy         = 0.743
untied_compute_gap                                 = -0.081
```

The smoke therefore gives a useful negative:

```text
summary shortcut fixed
deterministic local wavefront works
learned tied loops still leak or use bad wall/reach dynamics
untied compute remains a confound
```

The full valid grid was not launched because the smoke already failed positive-gate prerequisites. The runner supports it, but it would only amplify the current negative without an architecture patch.

## Decision

```text
Do not claim STABLE_LOOP_WAVEFRONT_POSITIVE.
Do not claim PRISMION_WAVEFRONT_POSITIVE.
Next useful patch: force a cleaner learned reached channel / frontier update, or add a local auxiliary probe-only penalty after training; then rerun smoke before valid_slice.
```

## Summary Table

| Arm | W | S | Mode | Reach | Larger | Long | SettleGain | WallLeak | Overrun |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `ABC_TIED_LOOP_HARD_WALL` | `16` | `1` | `FIXED_S` | `0.581` | `0.576` | `0.501` | `0.000` | `0.481` | `0.840` |
| `ABC_TIED_LOOP_HARD_WALL` | `16` | `4` | `FIXED_S` | `0.787` | `0.782` | `0.522` | `0.247` | `0.000` | `0.947` |
| `ABC_TIED_LOOP_HARD_WALL` | `16` | `8` | `FIXED_S` | `0.805` | `0.805` | `0.534` | `0.265` | `0.000` | `0.971` |
| `LOCAL_MESSAGE_PASSING_GNN` | `0` | `1` | `FIXED_S` | `0.537` | `0.532` | `0.500` | `0.000` | `0.000` | `0.689` |
| `LOCAL_MESSAGE_PASSING_GNN` | `0` | `4` | `FIXED_S` | `0.683` | `0.684` | `0.500` | `0.146` | `0.000` | `0.835` |
| `LOCAL_MESSAGE_PASSING_GNN` | `0` | `8` | `FIXED_S` | `0.758` | `0.757` | `0.515` | `0.222` | `0.000` | `0.804` |
| `ORACLE_FULL_BFS` | `0` | `1` | `FIXED_S` | `1.000` | `1.000` | `1.000` | `nan` | `nan` | `nan` |
| `ORACLE_TRUNCATED_BFS_S` | `0` | `1` | `FIXED_S` | `0.533` | `0.529` | `0.500` | `nan` | `nan` | `nan` |
| `ORACLE_TRUNCATED_BFS_S` | `0` | `4` | `FIXED_S` | `0.701` | `0.701` | `0.500` | `nan` | `nan` | `nan` |
| `ORACLE_TRUNCATED_BFS_S` | `0` | `8` | `FIXED_S` | `0.801` | `0.801` | `0.500` | `nan` | `nan` | `nan` |
| `PRISMION_PHASE_WAVE_LOOP` | `16` | `1` | `FIXED_S` | `0.591` | `0.579` | `0.504` | `0.000` | `0.042` | `0.377` |
| `PRISMION_PHASE_WAVE_LOOP` | `16` | `4` | `FIXED_S` | `0.758` | `0.748` | `0.518` | `0.198` | `0.468` | `0.781` |
| `PRISMION_PHASE_WAVE_LOOP` | `16` | `8` | `FIXED_S` | `0.877` | `0.858` | `0.743` | `0.356` | `0.451` | `0.603` |
| `SUMMARY_DIRECT_HEAD` | `16` | `1` | `FIXED_S` | `0.576` | `0.578` | `0.500` | `nan` | `nan` | `nan` |
| `TARGET_MARKER_ONLY` | `0` | `1` | `FIXED_S` | `0.529` | `0.534` | `0.500` | `nan` | `nan` | `nan` |
| `TIED_LOCAL_CA_LOOP` | `16` | `1` | `FIXED_S` | `0.584` | `0.575` | `0.504` | `0.000` | `0.002` | `0.375` |
| `TIED_LOCAL_CA_LOOP` | `16` | `4` | `FIXED_S` | `0.732` | `0.723` | `0.517` | `0.181` | `0.478` | `0.522` |
| `TIED_LOCAL_CA_LOOP` | `16` | `8` | `FIXED_S` | `0.839` | `0.824` | `0.669` | `0.307` | `0.387` | `0.517` |
| `UNTIED_CNN_MATCHED_COMPUTE` | `16` | `1` | `FIXED_S` | `0.679` | `0.669` | `0.506` | `nan` | `0.481` | `nan` |
| `UNTIED_CNN_MATCHED_COMPUTE` | `16` | `4` | `FIXED_S` | `0.832` | `0.823` | `0.627` | `nan` | `0.482` | `nan` |
| `UNTIED_CNN_MATCHED_COMPUTE` | `16` | `8` | `FIXED_S` | `0.923` | `0.888` | `0.874` | `nan` | `0.622` | `nan` |

## Claim Boundary

This is a deterministic 2D wavefront/reachability probe. It is not a parser, factuality system, language benchmark, consciousness claim, or full VRAXION architecture test.
