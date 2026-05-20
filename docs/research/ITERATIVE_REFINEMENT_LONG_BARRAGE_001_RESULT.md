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

## Claim Boundary

This barrage maps a toy state-transition loop. It does not prove language understanding, GPT-like readiness, production readiness, safety alignment, consciousness, or open-domain reasoning.
