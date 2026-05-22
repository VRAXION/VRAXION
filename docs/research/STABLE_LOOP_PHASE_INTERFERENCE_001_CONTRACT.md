# STABLE_LOOP_PHASE_INTERFERENCE_001 Contract

## Goal

Test whether phase/interference dynamics help when local wavefronts carry signed or multi-phase signals that can reinforce, cancel, or compete.

This is not a parser, factuality benchmark, language task, consciousness claim, or full VRAXION architecture test. It isolates one mechanism:

```text
local tied recurrent loop
  + hard wall constraint
  + phase-carrying wavefronts
  + final target phase supervision only
```

The current wavefront checkpoint supports the conservative claim:

```text
stable-loop wavefront positive
Prismion useful-looking, but not uniquely required
ABC/local message passing often reaches ceiling on plain reachability
```

So this run does not repeat plain reachability. It asks whether phase/interference behavior creates a harder separation.

## Task

Grid channels:

```text
wall
target
source_real
source_imag
```

The signal propagates locally through free cells. Each source emits a phase vector.

For `K=2`:

```text
PHASE_0 = +1 + 0i
PHASE_1 = -1 + 0i
```

For `K=4`:

```text
PHASE_0 = +1 + 0i
PHASE_1 =  0 + 1i
PHASE_2 = -1 + 0i
PHASE_3 =  0 - 1i
```

When multiple wavefronts arrive at the same cell on the same step, their vectors are summed. If the magnitude is below threshold, the cell is canceled and emits no new frontier. Otherwise the surviving phase bucket is the nearest phase direction.

Main label:

```text
target phase class:
  NONE
  PHASE_0
  PHASE_1
  PHASE_2
  PHASE_3
```

Training signal:

```text
final target phase only
no intermediate phase map supervision
no frontier supervision
no cancellation map supervision
```

Oracle/probe artifacts may include phase maps, frontier maps, collision cells, and target phase by S.

## Task Families

Required deterministic families:

```text
single_source_reach
opposite_phase_collision
same_phase_reinforcement
branch_cancellation
decoy_phase
same_target_neighborhood_phase_contrast
K4_phase_competition
```

Optional diagnostic:

```text
dynamic_cancel
```

The main verdict uses first-arrival collision, not dynamic cancel.

## Oracles

```text
ORACLE_PHASE_WAVEFRONT_S
ORACLE_FULL_PHASE_WAVEFRONT
```

Main oracle mode:

```text
FIRST_ARRIVAL_COLLISION
```

For `S < collision/path distance`, score against the truncated target phase at S. Do not punish a local loop for not knowing future collisions.

## Arms

Required:

```text
ORACLE_PHASE_WAVEFRONT_S
SUMMARY_DIRECT_HEAD
TARGET_MARKER_ONLY
LOCAL_MESSAGE_PASSING_GNN_PHASE
HARD_WALL_ABC_PHASE_LOOP
HARD_WALL_PRISMION_PHASE_LOOP
UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE
```

No broad model zoo. Global/flattened controls are not fair baselines for the local-loop verdict.

## Locality Rules

Fair local loop arms:

```text
no global pooling
no flattening
no attention
no spatial layernorm over HxW
no learned answer head over full grid
target-cell readout only
hard wall mask enforced
```

Allowed:

```text
3x3 local convolution
1x1 per-cell channel mixing
hard target-cell phase readout
hard wall mask using input free_space
```

The main hard-wall arms use a fixed target readout:

```text
target vector = reached_real/reached_imag at target cell
magnitude below threshold => NONE
otherwise nearest phase bucket
```

## Metrics

Core:

```text
target_phase_accuracy
none_vs_phase_accuracy
phase_bucket_accuracy
cancellation_case_accuracy
reinforcement_case_accuracy
opposite_collision_accuracy
same_target_neighborhood_pair_accuracy
decoy_phase_resistance
truncated_phase_accuracy_by_S
full_phase_accuracy_at_large_S
```

S-curve:

```text
same_weights_s_curve_accuracy
propagation_curve_score
s_matches_phase_collision_distance_score
false_future_collision_rate
```

Errors:

```text
false_none_rate
false_phase_rate
false_positive_phase_after_cancel
wrong_phase_rate
unreachable_false_reach_all_S
```

Stability:

```text
overrun_stability
noise_recovery_accuracy
final_state_delta
phase_vector_norm_by_step
phase_saturation_rate
```

Wall:

```text
post_mask_wall_leak
pre_mask_wall_pressure
```

Comparisons:

```text
prismion_minus_abc_by_seed
prismion_minus_gnn_by_seed
prismion_minus_untied_local_by_seed
mean/std/min/max deltas
```

## Verdicts

```text
PHASE_INTERFERENCE_TASK_VALID
CANONICAL_MESSAGE_PASSING_SUFFICIENT
PRISMION_PHASE_INTERFERENCE_POSITIVE
PRISMION_K2_ONLY_WEAK
ABC_CEILING_TASK_TOO_EASY
PHASE_LOOP_UNSTABLE
SUMMARY_OR_TARGET_SHORTCUT_RETURNS
UNTIED_LOCAL_SUFFICIENT
TASK_TOO_HARD
```

Prismion is positive only if it beats ABC and GNN on matched seed/S/bucket metrics, especially collision/cancellation/K4 tasks, with equal or lower wall pressure and no worse false-none/false-phase rates.

## Run Plan

Use GPU jobs=4 with no CPU sweep in parallel.

Smoke:

```powershell
python scripts/probes/run_stable_loop_phase_interference_probe.py ^
  --out target/pilot_wave/stable_loop_phase_interference_001/k2_smoke ^
  --phase-classes 2 ^
  --seeds 2026,2027 ^
  --train-examples 4096 ^
  --eval-examples 4096 ^
  --epochs 10 ^
  --width 32 ^
  --jobs 4 ^
  --device cuda ^
  --heartbeat-sec 15
```

K2 confirmation:

```powershell
python scripts/probes/run_stable_loop_phase_interference_probe.py ^
  --out target/pilot_wave/stable_loop_phase_interference_001/k2_5seed ^
  --phase-classes 2 ^
  --seeds 2026-2030 ^
  --train-examples 4096 ^
  --eval-examples 4096 ^
  --epochs 10 ^
  --width 32 ^
  --jobs 4 ^
  --device cuda ^
  --heartbeat-sec 30
```

K4 diagnostic:

```powershell
python scripts/probes/run_stable_loop_phase_interference_probe.py ^
  --out target/pilot_wave/stable_loop_phase_interference_001/k4_3seed ^
  --phase-classes 4 ^
  --seeds 2026,2027,2028 ^
  --train-examples 4096 ^
  --eval-examples 4096 ^
  --epochs 20 ^
  --width 32 ^
  --jobs 4 ^
  --device cuda ^
  --heartbeat-sec 30
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
summary.json
report.md
phase_cases.jsonl
phase_bucket_metrics.jsonl
collision_metrics.jsonl
convergence_curves.jsonl
examples_sample.jsonl
contract_snapshot.md
job_progress/*.jsonl
```

Raw `target/` outputs are not committed.
