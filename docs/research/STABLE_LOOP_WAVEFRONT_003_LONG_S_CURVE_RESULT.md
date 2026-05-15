# STABLE_LOOP_WAVEFRONT_003_LONG_S_CURVE Result

## Run

```text
stage=smoke
seeds=2026,2027
train_examples=1024
eval_examples=1024
epochs=20
jobs=12
device=cpu
completed_jobs=64
```

## Verdict

```json
[
  "FULL_REACHABILITY_POSITIVE",
  "PROPAGATION_CURVE_POSITIVE",
  "SAME_WEIGHTS_S_CURVE_POSITIVE"
]
```

## Control Gaps

```json
{
  "summary_direct_gap": 0.403846,
  "untied_local_compute_gap": 0.058488,
  "global_compute_gap": 0.495133
}
```

## Summary Table

| Arm | W | S | Mode | TruncAcc | FullLarge | d<=S | d>S | FalseReach>d | SettleGain | PostWall | PreWall | OverrunTrunc |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HARD_WALL_ABC_LOOP` | `16` | `4` | `FIXED_S` | `0.981` | `0.975` | `0.947` | `0.992` | `0.008` | `0.000` | `0.000` | `0.009` | `0.812` |
| `HARD_WALL_ABC_LOOP` | `16` | `8` | `FIXED_S` | `0.948` | `0.946` | `1.000` | `0.953` | `0.047` | `-0.000` | `0.000` | `0.009` | `0.910` |
| `HARD_WALL_ABC_LOOP` | `16` | `16` | `FIXED_S` | `0.875` | `0.872` | `1.000` | `0.912` | `0.088` | `-0.073` | `0.000` | `0.006` | `0.829` |
| `HARD_WALL_ABC_LOOP` | `16` | `24` | `FIXED_S` | `0.829` | `0.829` | `0.908` | `nan` | `nan` | `-0.119` | `0.000` | `0.004` | `0.829` |
| `HARD_WALL_ABC_LOOP` | `16` | `24` | `SAME_WEIGHTS_S_CURVE` | `0.866` | `0.866` | `0.906` | `nan` | `nan` | `-0.082` | `0.000` | `0.004` | `0.866` |
| `HARD_WALL_ABC_LOOP` | `16` | `32` | `FIXED_S` | `0.829` | `0.829` | `0.908` | `nan` | `nan` | `-0.119` | `0.000` | `0.003` | `0.829` |
| `HARD_WALL_PRISMION_PHASE_LOOP` | `16` | `4` | `FIXED_S` | `0.979` | `0.983` | `0.950` | `0.974` | `0.026` | `0.000` | `0.000` | `0.007` | `0.879` |
| `HARD_WALL_PRISMION_PHASE_LOOP` | `16` | `8` | `FIXED_S` | `0.981` | `0.990` | `0.996` | `0.953` | `0.047` | `0.011` | `0.000` | `0.008` | `0.985` |
| `HARD_WALL_PRISMION_PHASE_LOOP` | `16` | `16` | `FIXED_S` | `0.985` | `0.992` | `1.000` | `0.912` | `0.088` | `0.017` | `0.000` | `0.006` | `0.993` |
| `HARD_WALL_PRISMION_PHASE_LOOP` | `16` | `24` | `FIXED_S` | `0.993` | `0.993` | `1.000` | `nan` | `nan` | `0.024` | `0.000` | `0.004` | `0.993` |
| `HARD_WALL_PRISMION_PHASE_LOOP` | `16` | `24` | `SAME_WEIGHTS_S_CURVE` | `0.991` | `0.991` | `0.998` | `nan` | `nan` | `0.023` | `0.000` | `0.004` | `0.991` |
| `HARD_WALL_PRISMION_PHASE_LOOP` | `16` | `32` | `FIXED_S` | `0.993` | `0.993` | `1.000` | `nan` | `nan` | `0.024` | `0.000` | `0.003` | `0.993` |
| `LOCAL_MESSAGE_PASSING_GNN` | `0` | `4` | `FIXED_S` | `0.948` | `0.955` | `1.000` | `0.936` | `0.064` | `0.000` | `nan` | `nan` | `0.927` |
| `LOCAL_MESSAGE_PASSING_GNN` | `0` | `8` | `FIXED_S` | `0.947` | `0.945` | `1.000` | `0.953` | `0.047` | `-0.001` | `nan` | `nan` | `0.875` |
| `LOCAL_MESSAGE_PASSING_GNN` | `0` | `16` | `FIXED_S` | `0.875` | `0.872` | `1.000` | `0.912` | `0.088` | `-0.073` | `nan` | `nan` | `0.853` |
| `LOCAL_MESSAGE_PASSING_GNN` | `0` | `24` | `FIXED_S` | `0.853` | `0.853` | `1.000` | `nan` | `nan` | `-0.095` | `nan` | `nan` | `0.853` |
| `LOCAL_MESSAGE_PASSING_GNN` | `0` | `32` | `FIXED_S` | `0.853` | `0.853` | `1.000` | `nan` | `nan` | `-0.095` | `nan` | `nan` | `0.853` |
| `ORACLE_FULL_BFS` | `0` | `1` | `FIXED_S` | `0.521` | `1.000` | `1.000` | `0.000` | `1.000` | `nan` | `nan` | `nan` | `nan` |
| `ORACLE_TRUNCATED_BFS_S` | `0` | `4` | `FIXED_S` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `nan` | `nan` | `nan` | `nan` |
| `ORACLE_TRUNCATED_BFS_S` | `0` | `8` | `FIXED_S` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `nan` | `nan` | `nan` | `nan` |
| `ORACLE_TRUNCATED_BFS_S` | `0` | `16` | `FIXED_S` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `nan` | `nan` | `nan` | `nan` |
| `ORACLE_TRUNCATED_BFS_S` | `0` | `24` | `FIXED_S` | `1.000` | `1.000` | `1.000` | `nan` | `nan` | `nan` | `nan` | `nan` | `nan` |
| `ORACLE_TRUNCATED_BFS_S` | `0` | `32` | `FIXED_S` | `1.000` | `1.000` | `1.000` | `nan` | `nan` | `nan` | `nan` | `nan` | `nan` |
| `SUMMARY_DIRECT_HEAD` | `16` | `1` | `FIXED_S` | `0.524` | `0.546` | `0.908` | `0.500` | `0.500` | `nan` | `nan` | `nan` | `nan` |
| `TARGET_MARKER_ONLY` | `0` | `1` | `FIXED_S` | `0.042` | `0.082` | `1.000` | `0.000` | `1.000` | `nan` | `nan` | `nan` | `nan` |
| `UNTIED_GLOBAL_CNN` | `16` | `1` | `FIXED_S` | `0.521` | `0.997` | `0.969` | `0.005` | `0.995` | `nan` | `nan` | `nan` | `nan` |
| `UNTIED_LOCAL_CNN_TARGET_READOUT` | `16` | `4` | `FIXED_S` | `0.942` | `0.915` | `0.846` | `0.988` | `0.012` | `0.000` | `nan` | `nan` | `0.659` |
| `UNTIED_LOCAL_CNN_TARGET_READOUT` | `16` | `8` | `FIXED_S` | `0.701` | `0.681` | `0.692` | `0.760` | `0.240` | `0.211` | `nan` | `nan` | `0.553` |
| `UNTIED_LOCAL_CNN_TARGET_READOUT` | `16` | `16` | `FIXED_S` | `0.613` | `0.643` | `0.810` | `0.288` | `0.712` | `-0.231` | `nan` | `nan` | `0.470` |
| `UNTIED_LOCAL_CNN_TARGET_READOUT` | `16` | `24` | `FIXED_S` | `0.660` | `0.660` | `0.773` | `nan` | `nan` | `0.160` | `nan` | `nan` | `0.479` |
| `UNTIED_LOCAL_CNN_TARGET_READOUT` | `16` | `24` | `SAME_WEIGHTS_S_CURVE` | `0.520` | `0.520` | `0.445` | `nan` | `nan` | `-0.332` | `nan` | `nan` | `0.478` |
| `UNTIED_LOCAL_CNN_TARGET_READOUT` | `16` | `32` | `FIXED_S` | `0.661` | `0.661` | `0.702` | `nan` | `nan` | `-0.192` | `nan` | `nan` | `0.661` |

## Interpretation

Smoke passes the main v3 label-split sanity check.

`ORACLE_TRUNCATED_BFS_S` is exact at every tested S, while `ORACLE_FULL_BFS` is intentionally wrong under the truncated S metric for `d>S` cases. This confirms the probe is no longer punishing local loops for not teleporting.

Shortcut controls do not solve the truncated propagation task:

```text
SUMMARY_DIRECT_HEAD TruncAcc = 0.524
TARGET_MARKER_ONLY  TruncAcc = 0.042
```

The strongest learned local loop is `HARD_WALL_PRISMION_PHASE_LOOP`:

```text
S=4  TruncAcc=0.979  FalseReach>d=0.026
S=8  TruncAcc=0.981  FalseReach>d=0.047
S=16 TruncAcc=0.985  FalseReach>d=0.088
S=24 TruncAcc=0.993
S=32 TruncAcc=0.993
```

The same-weights S-curve is also strong:

```text
HARD_WALL_PRISMION_PHASE_LOOP SAME_WEIGHTS_S_CURVE
  truncated S-curve average = 0.983
  S=4  = 0.968
  S=8  = 0.979
  S=16 = 0.984
  S=24 = 0.991
  S=32 = 0.991
```

Wall behavior remains clean:

```text
post_mask wall leak = 0.000
pre_mask wall pressure ~= 0.003-0.008
```

The fair untied local compute baseline does not solve the S-curve as well as the hard-wall loop. The global untied CNN solves the full shortcut/control task, but it is deliberately not a fair local propagation baseline.

## Claim Boundary

This is a deterministic 2D long-S hard-wall wavefront probe. S-curve metrics use the truncated reachability oracle; full reachability is reported separately for S values that cover the relevant path bucket. It is not a parser, factuality system, language benchmark, consciousness claim, or full VRAXION architecture test.
