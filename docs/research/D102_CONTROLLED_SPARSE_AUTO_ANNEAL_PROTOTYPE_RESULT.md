# D102 Controlled Sparse Auto-Anneal Prototype Result

## Status

Implemented in `scripts/probes/run_d102_controlled_sparse_auto_anneal_prototype.py` with validation in `scripts/probes/run_d102_controlled_sparse_auto_anneal_prototype_check.py`.

## Run command

```bash
python scripts/probes/run_d102_controlled_sparse_auto_anneal_prototype.py \
  --out target/pilot_wave/d102_controlled_sparse_auto_anneal_prototype \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 23001,23002,23003,23004,23005,23006,23007,23008,23009,23010 \
  --train-rows-per-seed 560 \
  --test-rows-per-seed 560 \
  --ood-rows-per-seed 560 \
  --stress-seeds 23101,23102,23103,23104,23105,23106 \
  --stress-rows-per-seed 720
```

## Validation command

```bash
python scripts/probes/run_d102_controlled_sparse_auto_anneal_prototype_check.py \
  --out target/pilot_wave/d102_controlled_sparse_auto_anneal_prototype
```

## Expected healthy decision

- `decision=d102_controlled_sparse_auto_anneal_prototype_confirmed`
- `next=D103_SPARSE_RECURRENT_MICROCIRCUIT_SCALE_CONFIRM`
- `final_sparse_stage=stage4_8pct`
- `final_sparse_pct=8`
- `final_anneal_pressure=light`

## Boundary

D102 is only a controlled sparse auto-anneal prototype for controlled symbolic ECF/IPF joint formula discovery. It creates a checkpointed sparse candidate copy only, preserves the dense baseline, and does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
