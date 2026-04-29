# Phase D10g GPU Evaluator Feasibility

Date: 2026-04-29

Verdict: `D10G_GPU_EVALUATOR_READY`

Scope: batched scout/prototype evaluator only. This is not a release checkpoint gate and must not promote candidates without CPU or CPU-vs-GPU confirmation.

## Summary

D10g tested whether a PyTorch CUDA evaluator is worth building before H512/H1024 scaling. The result is positive for batched scouting:

- Toy CPU reference vs Torch CPU/CUDA state match passed with `max_abs_diff=0`.
- H=384 baseline vs beta.8 CPU/CUDA metric outputs matched to tiny numerical tolerance.
- CUDA was slower at tiny batch sizes, but much faster once candidate batch size was high.
- Batch 64 H=384 throughput reached `56.79 candidates/s` on CUDA vs `8.43 candidates/s` on Torch CPU, about `6.7x`.
- Negative control did not become a false generalist.

The evaluator is useful only as a batched scout path. Mutation generation, accept/reject, checkpoint saving, and final confirmation stay CPU-owned.

## Key Results

Hardware/runtime:

```text
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
VRAM: 16GB
PyTorch: 2.5.1+cu121
CUDA available: yes
nvcc: unavailable
D10b CPU main: running independently, GPU unused
```

Toy correctness:

| gate | device | passed | max_abs_diff |
|---|---|---:|---:|
| `CPU_GPU_STATE_MATCH_TOY` | CPU | yes | 0 |
| `CPU_GPU_STATE_MATCH_TOY` | CUDA | yes | 0 |

H=384 smoke, `eval_len=128`, 2 eval seeds:

| device | smooth delta | accuracy delta | echo delta | unigram delta | elapsed |
|---|---:|---:|---:|---:|---:|
| Torch CPU | +0.00929697 | +0.00781250 | 0.0 | +0.00239436 | 0.79s |
| CUDA | +0.00929697 | +0.00781250 | 0.0 | +0.00239433 | 1.74s |

H=384 metric confirm, `eval_len=1000`, 4 eval seeds:

| device | smooth delta | accuracy delta | echo delta | unigram delta | elapsed |
|---|---:|---:|---:|---:|---:|
| Torch CPU | +0.00706110 | +0.00100000 | 0.0 | -0.00422765 | 11.18s |
| CUDA | +0.00706109 | +0.00100000 | 0.0 | -0.00422767 | 18.63s |

The 1000-token unigram sign differs from the D9.4b CPU verdict because this scratch probe uses a simple deterministic offset schedule, not Rust `StdRng`. This is not a CPU-vs-GPU drift; CPU and CUDA agree. Therefore this probe is valid for GPU feasibility, not for checkpoint promotion.

Throughput probe, `eval_len=128`, 2 eval seeds:

| batch | Torch CPU candidates/s | CUDA candidates/s | CUDA/CPU |
|---:|---:|---:|---:|
| 1 | 1.76 | 0.68 | 0.39x |
| 4 | 4.58 | 3.65 | 0.80x |
| 8 | 5.86 | 7.07 | 1.21x |
| 16 | 7.12 | 14.43 | 2.03x |
| 32 | 8.40 | 28.97 | 3.45x |
| 64 | 8.43 | 56.79 | 6.73x |

Negative control:

- `seed_4042` was not a false `FULL_GENERALIST`.
- It showed smooth positive but unigram negative and no accuracy improvement.

## Implementation Notes

Implemented scratch script:

```text
tools/_scratch/d10g_gpu_eval_probe.py
```

The script includes:

- Minimal Python parser for current Rust `bincode` checkpoint layout.
- Minimal VCBP packed-file reader.
- Dense batched Torch propagation preserving the current tick semantics.
- Toy CPU reference comparison.
- Real H=384 metric smoke and throughput modes.

The dense adjacency path is acceptable for the intended H512/H1024 scout scale. It is not intended to replace the CPU sparse evaluator for tiny batches.

## Next Gate

Use this GPU path only when there are many candidates to evaluate at once:

```text
candidate batch >= 16: useful
candidate batch >= 64: clearly useful
single candidate / small confirm: keep CPU
```

Recommended next implementation, after D10b finishes:

- If D10b finds strict or near-strict candidates, feed the exported candidates into the D10g GPU evaluator as a batched scout.
- Confirm any promoted candidate with the existing Rust CPU evaluator.
- If H512 work begins, use GPU evaluator for wide scout only, not final verdict.

## Progress Map

```text
GLOBAL AI PLAN MAP

[1] Real H=384 improvement
    DONE: beta.8 seed2042 generalist

[2] Causal mechanism
    DONE: D9.4b edge + threshold co-adaptation

[3] H=384 seed replication
    RUNNING: D10b CPU main

[3.5] GPU evaluator feasibility
    DONE: D10g PyTorch CUDA proof
    result: useful for batched scout, not promotion
        |
        |-- if D10b positive
        |      use GPU evaluator for batched H512/H1024 scout
        |
        '-- if D10b negative
               keep beta.8 as local finding; GPU remains infra-ready

[4] H512/GPU scaling
    BLOCKED until D10b is positive
```

