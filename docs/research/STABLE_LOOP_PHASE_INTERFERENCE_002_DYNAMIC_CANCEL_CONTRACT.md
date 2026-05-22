# STABLE_LOOP_PHASE_INTERFERENCE_002_DYNAMIC_CANCEL Contract

## Goal

`STABLE_LOOP_PHASE_INTERFERENCE_001` was valid but not discriminative: K2/K4 first-arrival phase-BFS was solved by ABC, GNN, and Prismion, while the summary head remained high.

This run removes the monotonic first-arrival rule. Later waves can dynamically cancel or overwrite previous phase state.

Core question:

```text
Do Prismion-style phase/interference updates help when phase state is non-monotonic?
```

This is still a local tied-loop mechanism probe. It is not language, parser grounding, factuality, consciousness, or full VRAXION.

## Dynamic Task

Grid channels:

```text
wall
target_marker
source_real
source_imag
delayed_source_real
delayed_source_imag
```

The delayed source activates at a fixed local loop step. The model can see the delayed source location/phase, but the wave is injected only when the internal step reaches the delay.

State channels:

```text
phase_real
phase_imag
phase_frontier_real
phase_frontier_imag
phase_memory_real
phase_memory_imag
learned latent channels
```

Dynamic oracle update:

```text
incoming = neighbor_sum(frontier)
incoming += delayed_injection_at_delay_step
state_next = decay * state + incoming
frontier_next = incoming
```

The state is read at the target cell:

```text
norm < threshold => NONE
otherwise nearest phase bucket
```

Unlike v1, reached cells are not frozen. Later opposite waves can change a cell's state.

## Required Families

```text
single_phase_sanity
simultaneous_opposite_collision
delayed_opposite_cancel
late_wrong_phase_after_target
reinforced_phase_vs_cancel
gate_open_close
same_target_neighborhood_dynamic_pair
K4_phase_equilibrium
```

## Required Arms

```text
ORACLE_DYNAMIC_PHASE_S
ORACLE_DYNAMIC_PHASE_FULL
ORACLE_FIRST_ARRIVAL_BASELINE
SUMMARY_DIRECT_HEAD
TARGET_MARKER_ONLY
LOCAL_MESSAGE_PASSING_GNN_DYNAMIC
HARD_WALL_ABC_DYNAMIC_LOOP
HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP
UNTIED_LOCAL_CNN_TARGET_READOUT_DYNAMIC
```

The fair local-loop verdict must not use global pooling, flattening, attention, global readout, or learned target MLP readout.

## Metrics

```text
dynamic_phase_accuracy
target_phase_accuracy
none_vs_phase_accuracy
phase_bucket_accuracy
cancellation_accuracy
delayed_cancel_accuracy
late_overwrite_accuracy
reinforcement_accuracy
gate_dynamic_accuracy
same_target_neighborhood_dynamic_pair_accuracy
summary_direct_accuracy
target_marker_accuracy
first_arrival_baseline_accuracy
truncated_dynamic_accuracy_by_S
full_dynamic_accuracy_at_large_S
same_weights_s_curve_accuracy
dynamic_s_matches_oracle_score
overrun_matches_dynamic_oracle
false_none_rate
false_phase_rate
wrong_phase_rate
false_survival_after_cancel
false_cancel_without_antiphase
wrong_late_overwrite_rate
phase_norm_by_step
phase_saturation_rate
final_state_delta
convergence_to_dynamic_oracle
noise_recovery_accuracy
post_mask_wall_leak
pre_mask_wall_pressure
prismion_minus_abc_by_seed
prismion_minus_gnn_by_seed
prismion_minus_untied_by_seed
```

## Verdicts

```text
DYNAMIC_CANCEL_TASK_VALID
DYNAMIC_STABLE_LOOP_POSITIVE
PRISMION_DYNAMIC_CANCEL_POSITIVE
CANONICAL_MESSAGE_PASSING_SUFFICIENT
FIRST_ARRIVAL_TASK_TOO_EASY
SUMMARY_OR_TARGET_SHORTCUT_RETURNS
ABC_CEILING_TASK_TOO_EASY
PHASE_LOOP_UNSTABLE
TASK_TOO_HARD
```

Prismion is positive only if it beats ABC/GNN on matched dynamic-cancel families, especially delayed cancel, late overwrite, and K4 equilibrium, with equal/lower wall pressure and no worse false-none/false-phase rates.

## Run Plan

Smoke:

```powershell
python scripts/probes/run_stable_loop_phase_interference_dynamic_cancel_probe.py ^
  --out target/pilot_wave/stable_loop_phase_interference_002_dynamic_cancel/k2_smoke ^
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
python scripts/probes/run_stable_loop_phase_interference_dynamic_cancel_probe.py ^
  --out target/pilot_wave/stable_loop_phase_interference_002_dynamic_cancel/k2_5seed ^
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
python scripts/probes/run_stable_loop_phase_interference_dynamic_cancel_probe.py ^
  --out target/pilot_wave/stable_loop_phase_interference_002_dynamic_cancel/k4_3seed ^
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
dynamic_phase_cases.jsonl
collision_metrics.jsonl
s_curve_metrics.jsonl
paired_seed_deltas.jsonl
examples_sample.jsonl
contract_snapshot.md
job_progress/*.jsonl
```
