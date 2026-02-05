# GPU VRAM Budget Breakdown v1 (OD1 / ChanWidth=1)

Contract ID: `vram_breakdown_v1`
Path (relative to `Golden Draft/`): `docs/gpu/vram_breakdown_v1.md`
Repo path: `Golden Draft/docs/gpu/vram_breakdown_v1.md`

This document defines a **VRAM accounting model** for Chapter #1 (OD1 / ChanWidth=1) and validates it against
the probe harness (VRA-32).

- Workload definition (VRA-31): `docs/gpu/workload_schema_v1.md`
- Objective/stability contract (VRA-30): `docs/gpu/objective_contract_v1.md`
- Probe harness (VRA-32): `Golden Draft/tools/gpu_capacity_probe.py`

---

## 1) Purpose & Scope

VRA-34 exists to make VRAM budgeting **explicit and predictable** so we can:

- choose safe starting batches (feeds VRA-35 capacity model),
- understand which knobs dominate VRAM (feeds VRA-28 GPU limiter),
- and avoid collecting data with an implicit/incorrect VRAM mental model.

**Scope (locked for Chapter #1):**
- OD1 presets (`od1_canon_small|real|stress`)
- `out_dim = 1` (ChanWidth=1)
- Measurements on Windows WDDM (RTX 4070 Ti SUPER 16GB) are supported but must account for paging behavior.

---

## 2) Definitions: allocated vs reserved (what we measure)

Probe harness metrics:
- `peak_vram_allocated_bytes` = `torch.cuda.max_memory_allocated()`
  - live tensor allocations peak (cleaner signal for fitting slopes).
- `peak_vram_reserved_bytes` = `torch.cuda.max_memory_reserved()`
  - allocator-reserved/cached peak (guardrail metric; `reserved >= allocated`).

Define:
```
overhead_bytes = peak_vram_reserved_bytes - peak_vram_allocated_bytes
```

Overhead includes the caching allocator, fragmentation, and safety headroom.

**Guardrail metric:** we budget and guard on **reserved**, because it is the quantity that hits the VRAM ceiling
first on real systems.

---

## 3) Measurement method (probe harness)

Harness: `Golden Draft/tools/gpu_capacity_probe.py`

For the datapoints in this doc we used (OD1 / out_dim=1):
- `precision = fp16`
- `amp = 1`
- `warmup_steps = 2`
- `measure_steps = 3`

**Why such short windows?** Under Windows WDDM, larger batches can enter an overcommit/paging regime that causes
very long stalls. VRAM peaks tend to occur early; short windows are sufficient to capture peak VRAM while keeping
the probe responsive.

The harness resets CUDA peak stats after warmup (so peaks correspond to the measure window).

Artifacts for these runs are stored under the gitignored directory:
`bench_vault/_tmp/vra34_vram_breakdown_v1/`

---

## 4) VRAM accounting model (analytic terms + explicit formula)

### 4.1 Dominant dynamic term: ring buffer bytes

For OD1/out_dim=1, the dominant empirical term is:

```
ring_buf_bytes = B * synth_len * ring_len * slot_dim * bytes_per_elem(precision)
```

Where:
- `B` = batch size
- `synth_len` = synthetic sequence length (from ColonySpec / `VRX_SYNTH_LEN`)
- `ring_len`, `slot_dim` = AntSpec footprint knobs
- `bytes_per_elem(fp16) = 2`, `bytes_per_elem(bf16) = 2`, `bytes_per_elem(fp32) = 4`

Intuition: this term corresponds to “per-token-per-slot state” that grows with both the sensory ring length and
slot width.

### 4.2 Static + overhead terms

We separate “static model state” and allocator overhead:

- `base_alloc_bytes` — static allocations (weights + grads + optimizer + misc persistent buffers).
- `overhead_bytes` — reserved-allocated delta from the caching allocator.

### 4.3 Prediction equations (per Ant tier)

We model:
```
pred_peak_allocated_bytes ≈ base_alloc_bytes + k_ring * ring_buf_bytes
pred_peak_reserved_bytes  ≈ pred_peak_allocated_bytes + overhead_bytes
```

We fit `k_ring` on **allocated** deltas (cleaner), and use a median overhead per tier for reserved prediction.

Fitted constants (fp16/amp=1, out_dim=1, this machine):

| Ant tier | ring_len×slot_dim | k_ring (allocated slope) | base_alloc_bytes (approx) | overhead_bytes (median) |
|---|---:|---:|---:|---:|
| small | 2048×256 | ~2.0398 | ~30 MiB | ~27 MiB |
| real | 8192×576 | ~2.0159 | ~60 MiB | ~42 MiB |
| stress | 16384×768 | (1 point) | (1 point) | ~47 MiB |

Notes:
- `stress` currently has only one valid datapoint; treat its prediction as “anchored” rather than validated.
- `overhead_bytes` can jump under WDDM paging/overcommit; see §7.

---

## 5) Predicted vs measured (peak_reserved) — OD1 / out_dim=1

Datapoints use these presets (probe harness accepts presets directly):
- Ant: `od1_canon_small`, `od1_canon_real`, `od1_canon_stress`
- Colony: `od1_canon_real` (`L=256`), `od1_canon_small` (`L=128`)

All rows are `precision=fp16`, `amp=1`, `out_dim=1`.

| Case | Ant (ring×slot) | Colony (synth_len) | B | Pred peak_reserved (GiB) | Meas peak_reserved (GiB) | Err | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| DP1 small×real | 2048×256 | 256 | 16 | 8.215 | 8.215 | +0.00% | baseline |
| DP9 small×real | 2048×256 | 256 | 24 | 12.294 | 12.296 | -0.00% | midpoint |
| DP2 small×real | 2048×256 | 256 | 32 | 16.374 | 16.369 | +0.03% | stability FAIL: `vram_guard` (boundary row) |
| DP8 real×real | 8192×576 | 256 | 1 | 4.635 | 4.610 | +0.55% | low-batch anchor (overhead dominates) |
| DP3 real×real | 8192×576 | 256 | 2 | 9.171 | 9.162 | +0.09% | baseline |
| DP4 real×real | 8192×576 | 256 | 3 | 13.706 | 13.715 | -0.06% | — |
| DP5 real×small | 8192×576 | 128 | 4 | 9.171 | 9.229 | -0.63% | same `B*synth_len` as DP3; overhead is higher |
| DP6 stress×real | 16384×768 | 256 | 1 | 12.201 | 12.201 | +0.00% | single-point anchored (stress tier) |

The prediction error stays within ~±0.7% on this dataset. The dominant failures at larger batches were **WDDM stalls**
and “paging regime” behavior, not OOM.

---

## 6) Dominant terms (what knobs matter most)

Ranked by impact on peak VRAM in the OD1/out_dim=1 regime:

1) **`batch_size` (`B`)** — linear.
   - DP1→DP9→DP2 shows near-linear scaling in the small tier.
2) **`synth_len`** — linear (enters the dominant term).
   - DP3 (B2,L256) vs DP5 (B4,L128) holds `B*synth_len` constant; peak_reserved stays in the same ballpark.
3) **`ring_len`** — linear.
   - increasing ring length grows the dominant term directly.
4) **`slot_dim`** — linear.
   - increases per-token-per-slot state footprint.
5) **`precision` bytes/elem** — linear via `bytes_per_elem(precision)`.
6) **Allocator overhead (`reserved - allocated`)** — typically tens of MiB here, but can jump under paging.

**Out of scope for Chapter #1:** `out_dim` scaling. Keep `out_dim=1` for the main table; treat any out_dim spotchecks
as appendix-only until the capacity model (VRA-35) formalizes it.

---

## 7) Windows / WDDM paging regime (important caveat)

Under Windows WDDM, the driver can overcommit and page GPU memory. In this regime:
- runs may stall for long periods,
- `peak_vram_reserved_bytes` can behave unexpectedly (including exceeding physical VRAM in extreme cases),
- and datapoints become unsuitable for fitting a stable model.

Example (excluded from fits / table):
- `bench_vault/_tmp/vra34_vram_breakdown_v1/dp03_realxreal_B04_fp16_amp1_E1_w02_m03`

Recommendation:
- For larger batches or long experiments, prefer Linux; on WDDM keep probes conservative and document stalls explicitly.

---

## 8) Reproduce (exact commands)

Run from repo root (outputs are gitignored):

```bash
python "Golden Draft/tools/gpu_capacity_probe.py" --ant od1_canon_small --colony od1_canon_real --out-dim 1 --batch 16 --warmup-steps 2 --measure-steps 3 --precision fp16 --amp 1 --output-dir bench_vault/_tmp/vra34_vram_breakdown_v1/dp01_smallxreal_B16_fp16_amp1_E1_w02_m03
python "Golden Draft/tools/gpu_capacity_probe.py" --ant od1_canon_small --colony od1_canon_real --out-dim 1 --batch 24 --warmup-steps 2 --measure-steps 3 --precision fp16 --amp 1 --output-dir bench_vault/_tmp/vra34_vram_breakdown_v1/dp09_smallxreal_B24_fp16_amp1_E1_w02_m03
python "Golden Draft/tools/gpu_capacity_probe.py" --ant od1_canon_small --colony od1_canon_real --out-dim 1 --batch 32 --warmup-steps 2 --measure-steps 3 --precision fp16 --amp 1 --output-dir bench_vault/_tmp/vra34_vram_breakdown_v1/dp02_smallxreal_B32_fp16_amp1_E1_w02_m03

python "Golden Draft/tools/gpu_capacity_probe.py" --ant od1_canon_real --colony od1_canon_real --out-dim 1 --batch 1 --warmup-steps 2 --measure-steps 3 --precision fp16 --amp 1 --output-dir bench_vault/_tmp/vra34_vram_breakdown_v1/dp08_realxreal_B01_fp16_amp1_E1_w02_m03
python "Golden Draft/tools/gpu_capacity_probe.py" --ant od1_canon_real --colony od1_canon_real --out-dim 1 --batch 2 --warmup-steps 2 --measure-steps 3 --precision fp16 --amp 1 --output-dir bench_vault/_tmp/vra34_vram_breakdown_v1/dp04_realxreal_B02_fp16_amp1_E1_w02_m03
python "Golden Draft/tools/gpu_capacity_probe.py" --ant od1_canon_real --colony od1_canon_real --out-dim 1 --batch 3 --warmup-steps 2 --measure-steps 3 --precision fp16 --amp 1 --output-dir bench_vault/_tmp/vra34_vram_breakdown_v1/dp04b_realxreal_B03_fp16_amp1_E1_w02_m03
python "Golden Draft/tools/gpu_capacity_probe.py" --ant od1_canon_real --colony od1_canon_small --out-dim 1 --batch 4 --warmup-steps 2 --measure-steps 3 --precision fp16 --amp 1 --output-dir bench_vault/_tmp/vra34_vram_breakdown_v1/dp07_realxsmall_B04_fp16_amp1_E1_w02_m03

python "Golden Draft/tools/gpu_capacity_probe.py" --ant od1_canon_stress --colony od1_canon_real --out-dim 1 --batch 1 --warmup-steps 2 --measure-steps 3 --precision fp16 --amp 1 --output-dir bench_vault/_tmp/vra34_vram_breakdown_v1/dp05_stressxreal_B01_fp16_amp1_E1_w02_m03
```

---

## Appendix: out_dim spotcheck (out of scope; do not use for OD1 budgeting)

One out-of-scope spotcheck exists:
- `bench_vault/_tmp/vra34_vram_breakdown_v1/dp01_small_B16_fp16_amp1_E16`

It suggests `out_dim` is not a dominant VRAM driver in the current probe harness, but this is **not** validated
across tiers and must not be used for capacity modeling until VRA-35 formalizes it.

