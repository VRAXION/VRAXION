# ITERATIVE_REFINEMENT_LONG_BARRAGE_001 Result

`ITERATIVE_REFINEMENT_LONG_BARRAGE_001` is a long local stress harness for the toy iterative refinement loop.

It repeatedly runs:

```text
scripts/probes/run_iterative_refinement_dynamics_001.py
```

with varied:

```text
seed
max_train_steps
hidden size
target-hold augmentation on/off
no-stop extra ticks
```

The aim is to map:

```text
when the transition loop remains stable
when external stop/checker is required
when internal target-hold emerges
which settings fail teacher-forced or free-run dynamics
```

## Runner

```text
scripts/probes/run_iterative_refinement_long_barrage_001.py
```

Suggested detached run:

```bash
nohup .venv/bin/python scripts/probes/run_iterative_refinement_long_barrage_001.py \
  --fresh \
  --time-budget-hours 8.5 \
  --out target/pilot_wave/iterative_refinement_long_barrage_001/day_run \
  > target/pilot_wave/iterative_refinement_long_barrage_001/day_run/nohup.log 2>&1 &
```

## Outputs

```text
target/pilot_wave/iterative_refinement_long_barrage_001/day_run/queue.json
target/pilot_wave/iterative_refinement_long_barrage_001/day_run/progress.jsonl
target/pilot_wave/iterative_refinement_long_barrage_001/day_run/run_metrics.jsonl
target/pilot_wave/iterative_refinement_long_barrage_001/day_run/metrics.csv
target/pilot_wave/iterative_refinement_long_barrage_001/day_run/summary.json
target/pilot_wave/iterative_refinement_long_barrage_001/day_run/report.md
target/pilot_wave/iterative_refinement_long_barrage_001/day_run/runs/
```

## Result

Status: `completed`

The 8.5 hour run completed at `2026-05-20T13:05:30Z` and executed `711` configurations.

Aggregate:

```text
completed_count:                    711
positive_count:                     261
failed_count:                       450
internal_hold_count:                172
external_stop_required_count:       539
best_no_stop_final_at_stop_state:   1.000
worst_positive_no_stop_final:       0.061
```

Phase breakdown:

```text
coverage_matrix:
  runs:                     10
  positive:                 4
  internal_hold:            3
  external_stop_required:   7

no_stop_extra_tick_stress:
  runs:                     8
  positive:                 8
  internal_hold:            4
  external_stop_required:   4

capacity_matrix:
  runs:                     8
  positive:                 8
  internal_hold:            4
  external_stop_required:   4

adaptive_random:
  runs:                     685
  positive:                 241
  internal_hold:            161
  external_stop_required:   524
```

## Main Findings

The transition loop has a clear coverage threshold:

```text
max_train_steps=5:
  positive: 0 / 79
  avg teacher-forced accuracy: 0.611
  avg free-run convergence:    0.455

max_train_steps=10:
  positive: 0 / 99
  avg teacher-forced accuracy: 0.790
  avg free-run convergence:    0.683

max_train_steps=20:
  positive: 9 / 76
  avg teacher-forced accuracy: 0.982
  avg free-run convergence:    0.971

max_train_steps=30:
  positive: 67 / 93
  avg teacher-forced accuracy: 0.997
  avg free-run convergence:    0.997

max_train_steps=40:
  positive: 86 / 92
  avg teacher-forced accuracy: 0.999
  avg free-run convergence:    1.000

max_train_steps=60:
  positive: 99 / 99
  avg teacher-forced accuracy: 1.000
  avg free-run convergence:    1.000
```

Target-hold augmentation cleanly separates internal stop from external stop:

```text
without target-hold examples:
  runs:                         382
  positive:                     143
  internal_hold_count:          0
  avg free-run convergence:     0.796
  avg no-stop final-at-stop:    0.078
  avg exit-after-stop:          0.730

with target-hold examples:
  runs:                         329
  positive:                     118
  internal_hold_count:          172
  avg free-run convergence:     0.850
  avg no-stop final-at-stop:    0.850
  avg exit-after-stop:          0.000
```

Representative clean positives:

```text
hold_mts40_s4108:
  target_hold:                  true
  max_train_steps:              40
  hidden:                       192
  no_stop_extra_ticks:          20
  free_run_convergence:         1.000
  no_stop_final_at_stop:        1.000
  exit_after_stop:              0.000

hold_extra500_s4117:
  target_hold:                  true
  max_train_steps:              60
  hidden:                       192
  no_stop_extra_ticks:          500
  free_run_convergence:         1.000
  no_stop_final_at_stop:        1.000
  exit_after_stop:              0.000

hold_hidden64_s4119:
  target_hold:                  true
  max_train_steps:              60
  hidden:                       64
  no_stop_extra_ticks:          100
  free_run_convergence:         1.000
  no_stop_final_at_stop:        1.000
  exit_after_stop:              0.000
```

Representative external-stop positives:

```text
nohold_mts40_s4103:
  target_hold:                  false
  max_train_steps:              40
  free_run_convergence:         1.000
  no_stop_final_at_stop:        0.061
  exit_after_stop:              0.939

nohold_extra500_s4116:
  target_hold:                  false
  max_train_steps:              60
  no_stop_extra_ticks:          500
  free_run_convergence:         1.000
  no_stop_final_at_stop:        0.061
  exit_after_stop:              0.939
```

## Interpretation

The long run supports three bounded conclusions:

```text
1. The learned transition loop can be stable in free-run when coverage is sufficient.
2. Without explicit target-hold training, successful transition models still usually need an external stop/checker.
3. With target-hold training, internal zero-delta hold can emerge and remain stable for long no-stop extra-tick stress.
```

The negative side is also clear:

```text
low coverage, especially max_train_steps <= 10, often causes cycles and weak teacher-forced transition accuracy.
target-hold examples do not rescue an undertrained transition rule by themselves.
```

## Claim Boundary

This barrage maps a toy state-transition loop. It does not prove language understanding, GPT-like readiness, production readiness, safety alignment, consciousness, or open-domain reasoning.
