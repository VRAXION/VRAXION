# STABLE_LOOP_WAVEFRONT_002_HARD_WALL_CHANNELS Result

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
  "LEARNED_WALL_GATE_POSITIVE",
  "UNTIED_COMPUTE_SOLVES_TASK"
]
```

## Control Gaps

```json
{
  "summary_direct_gap": 0.118308,
  "untied_compute_gap": -0.223638
}
```

## Interpretation

The v2 hard-wall patch fixes the v1 wall leak failure.

The learned hard-wall loops now have low post-mask leak and low pre-mask wall pressure:

```text
HARD_WALL_REACHED_FRONTIER_LOOP S=8 post_wall = 0.000
HARD_WALL_REACHED_FRONTIER_LOOP S=8 pre_wall  = 0.008

HARD_WALL_PRISMION_PHASE_LOOP S=8 post_wall = 0.000
HARD_WALL_PRISMION_PHASE_LOOP S=8 pre_wall  = 0.007
```

This supports the narrow `LEARNED_WALL_GATE_POSITIVE` verdict: under explicit reached/frontier channels, the model is not merely being rescued by the hard mask; pre-mask wall pressure is also low.

The hard-wall loops also reproduce the expected S-dependent wavefront curve:

```text
LOCAL_MESSAGE_PASSING_GNN S=1 reach = 0.537
LOCAL_MESSAGE_PASSING_GNN S=4 reach = 0.683
LOCAL_MESSAGE_PASSING_GNN S=8 reach = 0.758

HARD_WALL_ABC_LOOP S=1 reach = 0.537
HARD_WALL_ABC_LOOP S=4 reach = 0.683
HARD_WALL_ABC_LOOP S=8 reach = 0.758

HARD_WALL_PRISMION_PHASE_LOOP S=1 reach = 0.550
HARD_WALL_PRISMION_PHASE_LOOP S=4 reach = 0.713
HARD_WALL_PRISMION_PHASE_LOOP S=8 reach = 0.796
```

The run is still not a full stable-loop positive because the matched untied compute control remains much stronger on long paths:

```text
UNTIED_CNN_MATCHED_COMPUTE S=8 long_path_accuracy = 0.874
HARD_WALL_PRISMION_PHASE_LOOP S=8 long_path_accuracy = 0.520
HARD_WALL_ABC_LOOP S=8 long_path_accuracy = 0.515
untied_compute_gap = -0.224
```

So the current decision is:

```text
wall/channel blocker fixed
summary and target-marker shortcuts remain controlled
canonical and learned hard-wall loops show clean S-dependent propagation
long-path generalization is still weak versus untied compute
```

The valid slice was not launched because smoke failed the positive-gate prerequisite `UNTIED_CNN_MATCHED_COMPUTE does not solve the long-path task better than the learned hard-wall loop`.

## Decision

```text
Do not claim HARD_WALL_WAVEFRONT_POSITIVE yet.
Do keep the hard-wall reached/frontier design.
Next useful patch: train/evaluate longer S values in smoke, especially S=16/24, because S=8 cannot solve the 9-24 long bucket by construction.
```

## Summary Table

| Arm | W | S | Mode | Reach | Larger | Long | SettleGain | PostWall | PreWall | FrontierStuck | Overrun |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `HARD_WALL_ABC_LOOP` | `16` | `1` | `FIXED_S` | `0.537` | `0.532` | `0.500` | `0.000` | `0.000` | `0.013` | `0.005` | `0.844` |
| `HARD_WALL_ABC_LOOP` | `16` | `4` | `FIXED_S` | `0.683` | `0.684` | `0.500` | `0.146` | `0.000` | `0.011` | `0.002` | `0.837` |
| `HARD_WALL_ABC_LOOP` | `16` | `8` | `FIXED_S` | `0.758` | `0.757` | `0.515` | `0.222` | `0.000` | `0.008` | `0.001` | `0.804` |
| `HARD_WALL_PRISMION_PHASE_LOOP` | `16` | `1` | `FIXED_S` | `0.550` | `0.546` | `0.500` | `0.000` | `0.000` | `0.011` | `0.002` | `0.987` |
| `HARD_WALL_PRISMION_PHASE_LOOP` | `16` | `4` | `FIXED_S` | `0.713` | `0.713` | `0.500` | `0.167` | `0.000` | `0.010` | `0.002` | `0.859` |
| `HARD_WALL_PRISMION_PHASE_LOOP` | `16` | `8` | `FIXED_S` | `0.796` | `0.795` | `0.520` | `0.250` | `0.000` | `0.007` | `0.001` | `0.859` |
| `HARD_WALL_REACHED_FRONTIER_LOOP` | `16` | `1` | `FIXED_S` | `0.547` | `0.542` | `0.500` | `0.000` | `0.000` | `0.009` | `0.005` | `0.861` |
| `HARD_WALL_REACHED_FRONTIER_LOOP` | `16` | `4` | `FIXED_S` | `0.700` | `0.698` | `0.500` | `0.158` | `0.000` | `0.004` | `0.002` | `0.858` |
| `HARD_WALL_REACHED_FRONTIER_LOOP` | `16` | `8` | `FIXED_S` | `0.758` | `0.757` | `0.515` | `0.222` | `0.000` | `0.008` | `0.001` | `0.804` |
| `LOCAL_MESSAGE_PASSING_GNN` | `0` | `1` | `FIXED_S` | `0.537` | `0.532` | `0.500` | `0.000` | `nan` | `nan` | `nan` | `0.689` |
| `LOCAL_MESSAGE_PASSING_GNN` | `0` | `4` | `FIXED_S` | `0.683` | `0.684` | `0.500` | `0.146` | `nan` | `nan` | `nan` | `0.835` |
| `LOCAL_MESSAGE_PASSING_GNN` | `0` | `8` | `FIXED_S` | `0.758` | `0.757` | `0.515` | `0.222` | `nan` | `nan` | `nan` | `0.804` |
| `ORACLE_FULL_BFS` | `0` | `1` | `FIXED_S` | `1.000` | `1.000` | `1.000` | `nan` | `nan` | `nan` | `nan` | `nan` |
| `ORACLE_TRUNCATED_BFS_S` | `0` | `1` | `FIXED_S` | `0.533` | `0.529` | `0.500` | `nan` | `nan` | `nan` | `nan` | `nan` |
| `ORACLE_TRUNCATED_BFS_S` | `0` | `4` | `FIXED_S` | `0.701` | `0.701` | `0.500` | `nan` | `nan` | `nan` | `nan` | `nan` |
| `ORACLE_TRUNCATED_BFS_S` | `0` | `8` | `FIXED_S` | `0.801` | `0.801` | `0.500` | `nan` | `nan` | `nan` | `nan` | `nan` |
| `SUMMARY_DIRECT_HEAD` | `16` | `1` | `FIXED_S` | `0.576` | `0.578` | `0.500` | `nan` | `nan` | `nan` | `nan` | `nan` |
| `TARGET_MARKER_ONLY` | `0` | `1` | `FIXED_S` | `0.529` | `0.534` | `0.500` | `nan` | `nan` | `nan` | `nan` | `nan` |
| `UNTIED_CNN_MATCHED_COMPUTE` | `16` | `1` | `FIXED_S` | `0.679` | `0.669` | `0.506` | `nan` | `nan` | `nan` | `nan` | `nan` |
| `UNTIED_CNN_MATCHED_COMPUTE` | `16` | `4` | `FIXED_S` | `0.832` | `0.823` | `0.627` | `nan` | `nan` | `nan` | `nan` | `nan` |
| `UNTIED_CNN_MATCHED_COMPUTE` | `16` | `8` | `FIXED_S` | `0.923` | `0.888` | `0.874` | `nan` | `nan` | `nan` | `nan` | `nan` |

## Claim Boundary

This is a deterministic 2D hard-wall wavefront/reachability probe. It is not a parser, factuality system, language benchmark, consciousness claim, or full VRAXION architecture test.
