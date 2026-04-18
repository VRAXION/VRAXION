# Quantization research diagnostics (2026-04-17/18)

These scripts ran the 2026-04-17/18 quantization championship research. See
`VALIDATED_FINDINGS.md` and `docs/playground/quant_final_verdict.html` for
results.

All scripts share the same Beukers-gate char-LM task (mask-center, 27-class
alphabet, FineWeb + code corpora) so results are directly comparable.

**Note (2026-04-18 cleanup):** the full set of alternatives tested during the
championship (progressive growing, generational stacking, cluster stacking,
sparse / true exhaustive) lived here as `tools/diag_*.py` scripts. They were
removed from this directory during mainline cleanup; each experiment is
preserved as a blueprint entry on the [Timeline-Archive wiki
page](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive#archived-scripts-2026-04-18)
under "Archived scripts (2026-04-18)". Git history retains the code itself.

## Quick summary

| Script | Category | Status | Headline |
| --- | --- | --- | --- |
| `diag_qat_ste.py` | Core sweep | Winner | QAT int8 = absolute winner, 86.40% FineWeb |
| `diag_quant_sweep_gpu.py` | Core sweep | Baseline | Main 4-mode staged INQ reference |
| `diag_quant_sweep_gpu_mid.py` | Core sweep | Baseline | int8 matches float; int5/fp16 redundant |
| `diag_float_extended_control.py` | Control | Proved artifact | Long float beats all "quant wins" |

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

## How to run

Most scripts share the same argv pattern:

```
python tools/<script>.py <fineweb_path> <code_path>
```

Defaults (if args omitted):

- FineWeb: `S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt`
- Code: `instnct-core/tests/fixtures/code_corpus.txt`

## Hardware requirements

- **Recommended:** CUDA GPU. Developed on RTX 4070 Ti Super (16 GB VRAM);
  nf=1024 with batch 4096 fits easily.
- **CPU fallback:** all scripts detect `cuda` / `cpu` automatically, but the
  nf=1024 sweeps are impractically slow on CPU. For CPU work, use the Rust
  equivalents under `instnct-core/examples/` (e.g. the `diag_*` examples),
  which run the same architecture via rayon.
