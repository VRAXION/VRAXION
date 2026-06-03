# D104 Sparse Recurrent Generalization and Compression Frontier Map Result

## Status

Implemented in `scripts/probes/run_d104_sparse_recurrent_generalization_and_compression_frontier_map.py` with validation in `scripts/probes/run_d104_sparse_recurrent_generalization_and_compression_frontier_map_check.py`.

## Run command

```bash
python scripts/probes/run_d104_sparse_recurrent_generalization_and_compression_frontier_map.py \
  --out target/pilot_wave/d104_sparse_recurrent_generalization_and_compression_frontier_map \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 25001,25002,25003,25004,25005,25006,25007,25008,25009,25010 \
  --train-rows-per-seed 560 \
  --test-rows-per-seed 560 \
  --ood-rows-per-seed 560 \
  --family-seeds 25101,25102,25103,25104,25105,25106,25107,25108 \
  --family-rows-per-seed 520 \
  --stress-seeds 25201,25202,25203,25204,25205,25206 \
  --stress-rows-per-seed 720
```

## Validation command

```bash
python scripts/probes/run_d104_sparse_recurrent_generalization_and_compression_frontier_map_check.py \
  --out target/pilot_wave/d104_sparse_recurrent_generalization_and_compression_frontier_map
```

## Expected healthy decision

- `decision=d104_sparse_recurrent_generalization_frontier_mapped`
- `next=D105_CROSS_FAMILY_TRAIN_LOOP_INTEGRATION_PLAN`
- `final_sparse_candidate_name=D102_SPARSE_AUTO_ANNEAL_8PCT_LIGHT_PROTECTED_FAIR`
- `final_sparse_pct=8`
- `final_anneal_pressure=light`
- `d105_ready=true`

## Boundary

D104 is only a sparse recurrent generalization and compression frontier map for controlled symbolic formula-discovery tasks. It validates transfer across additional controlled symbolic families only. It does not increase sparsity, does not perform production pruning, does not use raw visual Raven or natural-language pretraining, and does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
