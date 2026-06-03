# D101 Auto Anneal and Sparse Stabilization Prep Result

## Status

Implemented in `scripts/probes/run_d101_auto_anneal_and_sparse_stabilization_prep.py` with validation in `scripts/probes/run_d101_auto_anneal_and_sparse_stabilization_prep_check.py`.

## Run command

```bash
python scripts/probes/run_d101_auto_anneal_and_sparse_stabilization_prep.py \
  --out target/pilot_wave/d101_auto_anneal_and_sparse_stabilization_prep \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 22001,22002,22003,22004,22005,22006,22007,22008,22009,22010 \
  --train-rows-per-seed 560 \
  --test-rows-per-seed 560 \
  --ood-rows-per-seed 560 \
  --stress-seeds 22101,22102,22103,22104,22105,22106 \
  --stress-rows-per-seed 720
```

## Validation command

```bash
python scripts/probes/run_d101_auto_anneal_and_sparse_stabilization_prep_check.py \
  --out target/pilot_wave/d101_auto_anneal_and_sparse_stabilization_prep
```

## Expected healthy decision

- `decision=d101_auto_anneal_sparse_stabilization_prep_ready`
- `next=D102_CONTROLLED_SPARSE_AUTO_ANNEAL_PROTOTYPE`
- `recommended_safe_shadow_sparsity_pct=8`
- `recommended_safe_anneal_pressure=light`

## Boundary

D101 is only an auto-anneal and sparse-stabilization preparation run for controlled symbolic ECF/IPF joint formula discovery. It uses non-destructive shadow masks and pressure probes only. It does not perform irreversible pruning and does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
