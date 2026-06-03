# D103 Sparse Recurrent Microcircuit Scale Confirm Result

## Status

Implemented in `scripts/probes/run_d103_sparse_recurrent_microcircuit_scale_confirm.py` with validation in `scripts/probes/run_d103_sparse_recurrent_microcircuit_scale_confirm_check.py`.

## Run command

```bash
python scripts/probes/run_d103_sparse_recurrent_microcircuit_scale_confirm.py \
  --out target/pilot_wave/d103_sparse_recurrent_microcircuit_scale_confirm \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 24001,24002,24003,24004,24005,24006,24007,24008,24009,24010,24011,24012 \
  --train-rows-per-seed 640 \
  --test-rows-per-seed 640 \
  --ood-rows-per-seed 640 \
  --stress-seeds 24101,24102,24103,24104,24105,24106,24107,24108 \
  --stress-rows-per-seed 820
```

## Validation command

```bash
python scripts/probes/run_d103_sparse_recurrent_microcircuit_scale_confirm_check.py \
  --out target/pilot_wave/d103_sparse_recurrent_microcircuit_scale_confirm
```

## Expected healthy decision

- `decision=d103_sparse_recurrent_microcircuit_scale_confirmed`
- `next=D104_SPARSE_RECURRENT_GENERALIZATION_AND_COMPRESSION_FRONTIER_MAP`
- `final_sparse_candidate_name=D102_SPARSE_AUTO_ANNEAL_8PCT_LIGHT_PROTECTED_FAIR`
- `final_sparse_pct=8`
- `final_anneal_pressure=light`
- `d104_ready=true`

## Boundary

D103 is only a sparse recurrent microcircuit scale-confirmation run for controlled symbolic ECF/IPF joint formula discovery. It validates the D102 8% light-pressure protected sparse candidate under scale. It does not increase sparsity, does not perform production pruning, and does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
