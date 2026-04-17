# Quantization research diagnostics (2026-04-17/18)

These scripts ran the 2026-04-17/18 quantization championship research. See
`VALIDATED_FINDINGS.md` and `docs/playground/quant_final_verdict.html` for
results.

All scripts share the same Beukers-gate char-LM task (mask-center, 27-class
alphabet, FineWeb + code corpora) so results are directly comparable.

## Quick summary

| Script | Category | Status | Headline |
| --- | --- | --- | --- |
| `diag_qat_ste.py` | Core sweep | Winner | QAT int8 = absolute winner, 86.40% FineWeb |
| `diag_quant_sweep_gpu.py` | Core sweep | Baseline | Main 4-mode staged INQ reference |
| `diag_quant_sweep_gpu_mid.py` | Core sweep | Baseline | int8 matches float; int5/fp16 redundant |
| `diag_float_extended_control.py` | Control | Proved artifact | Long float beats all "quant wins" |
| `diag_progressive_quant.py` | Alternative | Failed | -14.85pp vs batch |
| `diag_generational_growth.py` | Alternative | Failed | -5.2pp vs single-shot |
| `diag_random_rotation.py` | Alternative | Low-value | Dominated by QAT |
| `diag_cluster_stacking.py` | Alternative | Limited | K=2 boosting, 31.70% at 200 clusters |
| `diag_beukers_cluster_stacking.py` | Alternative | Underperforms | Joint-Beukers clusters stall |
| `diag_sparse_exhaustive.py` | Alternative | Valid (small K) | K=3/4/5 per-class exhaustive |
| `diag_sparse_exhaustive_v2.py` | Alternative | Valid (small K) | K=4 sweet spot, K=5 overfits |
| `diag_true_exhaustive.py` | Alternative | Optimum (small D) | 3^16 = 43M, mathematical optimum |
| `diag_exhaustive_cluster_stack.py` | Alternative | Dominated | Float+PTQ wins every metric |

## Core sweeps (baselines + main results)

| Script | Research question | One-sentence finding | Status |
| --- | --- | --- | --- |
| `diag_quant_sweep_gpu.py` | How do float / int4 / ternary / binary compare at nf=1024 under staged INQ? | Reference 4-mode sweep; staged INQ pushed int4 slightly above float baseline, but see the control. | Baseline |
| `diag_quant_sweep_gpu_mid.py` | Do int5 / int8 / fp16 fill the precision gap between int4 and float? | int8 matches float; int5 and fp16 are redundant at this model size. | Baseline |
| `diag_qat_ste.py` | Does Quantization-Aware Training with Straight-Through Estimator beat staged INQ? | QAT int8 wins the championship at 86.40% FineWeb; QAT int4 close second. | Winner |

## Control experiments (protocol revisions)

| Script | Research question | One-sentence finding | Status |
| --- | --- | --- | --- |
| `diag_float_extended_control.py` | Was the "+1.4pp int4 win" a real quant effect or just extra training epochs? | Float with 400 epochs or staged-matched training beats every quant variant — the "win" was a protocol artifact. | Negative result (revised prior finding) |

## Alternative approaches tested (negative results + limited sub-approaches)

| Script | Research question | One-sentence finding | Status |
| --- | --- | --- | --- |
| `diag_progressive_quant.py` | Can you grow one neuron at a time and int4-freeze each as you go? | Progressive growth lost -14.85pp FineWeb vs batch training. | Failed |
| `diag_generational_growth.py` | Does stacking Gen1+Gen2+Gen3 on LUT-frozen previous generations beat single-shot? | Generational stacking lost -5.2pp vs single-shot at matched nf. | Failed |
| `diag_random_rotation.py` | Does rotating a 1-50% hot float buffer on an int4 backbone converge? | Converges but is strictly dominated by QAT; not worth the complexity. | Low-value |
| `diag_cluster_stacking.py` | Does K=2 sparse per-class exhaustive cluster boosting approach dense float? | 200 clusters reach 31.70% (vs 34.2% dense float) — promising but limited. | Limited |
| `diag_beukers_cluster_stacking.py` | Do joint 2-projection Beukers exhaustive clusters beat simple sparse ones? | Stalls early (same best cluster rediscovered), underperforms the simpler K=2 scheme. | Underperforms |
| `diag_sparse_exhaustive.py` | Can exhaustive K-sparse binary search beat gradient-trained sparse? | Exhaustive beats gradient sparse at small K; validates the sub-approach. | Valid (small K) |
| `diag_sparse_exhaustive_v2.py` | Which K is the sweet spot (memory-managed K=3/4/5 sweep)? | K=4 is the sweet spot; K=5 overfits the 5k training set. | Valid (small K) |
| `diag_true_exhaustive.py` | What's the mathematical optimum when topology + sign are searched jointly (3^16)? | True ternary optimum 21.25% vs float 30.25% at D=16 — shows the ceiling of ternary-only. | Optimum (small D) |
| `diag_exhaustive_cluster_stack.py` | Can stacked true-ternary clusters beat float + PTQ on accuracy/storage/time? | Float + int4 PTQ dominates every metric — stacked exhaustive clusters are not competitive. | Dominated |

## How to run

Most scripts share the same argv pattern:

```
python tools/<script>.py <fineweb_path> <code_path>
```

Defaults (if args omitted):

- FineWeb: `S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt`
- Code: `instnct-core/tests/fixtures/code_corpus.txt`

Cluster-stacking and true-exhaustive scripts (`diag_cluster_stacking.py`,
`diag_beukers_cluster_stacking.py`, `diag_sparse_exhaustive*.py`,
`diag_true_exhaustive.py`, `diag_exhaustive_cluster_stack.py`) only take the
FineWeb path — they do not train on the code corpus.

## Hardware requirements

- **Recommended:** CUDA GPU. Developed on RTX 4070 Ti Super (16 GB VRAM);
  nf=1024 with batch 4096 fits easily.
- **CPU fallback:** all scripts detect `cuda` / `cpu` automatically, but the
  nf=1024 sweeps are impractically slow on CPU. For CPU work, use the Rust
  equivalents under `instnct-core/examples/` (e.g. the `diag_*` examples),
  which run the same architecture via rayon.
- **Memory note:** `diag_true_exhaustive.py` enumerates 3^16 = 43M configs;
  `diag_exhaustive_cluster_stack.py` does 3^14 per cluster. Keep chunk sizes
  at their defaults unless you have >12 GB free VRAM.
