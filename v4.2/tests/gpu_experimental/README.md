# GPU Experimental

Experimental GPU-side search and evaluation work for the current PassiveIO mainline.

## Branch Metadata

- Branch: `codex/gpu-passiveio-swarm`
- Base branch: `main`
- Base commit: `1095c8a`
- Status: experimental, isolated from the canonical CPU training loop
- Scope: add reproducible GPU probes and batch-search harnesses without changing [`graph.py`](S:/AI/work/VRAXION_DEV/v4.2/model/graph.py)

## Why This Exists

The current `main` model is a good fit for GPU batch evaluation:

- PassiveIO architecture with dense hidden-only mask
- `H = 3V`
- float32 dense forward
- candidate quality is limited by proposal hit-rate, not by raw VRAM

CPU findings that motivate this folder:

- local-neighborhood quality is strong early and decays late
- late-stage accepted-move rate can fall to low single digits
- `add_only` growth is strongest at `V >= 64`
- end-of-run crystal pruning helps, and deeper crystal passes can remove a large redundant fraction of edges without score loss

This suggests a clean GPU strategy:

- batch many candidate neighbors in parallel
- prefer simple proposals (`add_only`, `remove_only`, pass-based crystal)
- keep the learned intelligence in selection, not in smart proposal heuristics

## Current Mainline Assumptions

From [`graph.py`](S:/AI/work/VRAXION_DEV/v4.2/model/graph.py):

- `V` = task / vocab size
- `H = 3V`
- learnable matrix: `mask [H, H]`
- fixed projections:
  - `W_in [V, H]`
  - `W_out [H, V]`
- default forward ticks: `6`
- current CPU training is still canonical; GPU work here is additive research

## Local GPU Envelope

Measured on:

- GPU: `NVIDIA GeForce RTX 4070 Ti SUPER`
- VRAM: `16 GiB`

Conservative dense candidate-stack probe:

| V | H | K | Used VRAM | Batch time | Candidate/s |
|---|---:|---:|---:|---:|---:|
| 64 | 192 | 64 | 22 MiB | 1.28 ms | 49.8k |
| 128 | 384 | 64 | 74 MiB | 3.16 ms | 20.3k |
| 256 | 768 | 32 | 166 MiB | 11.21 ms | 2.85k |
| 512 | 1536 | 8 | 166 MiB | 16.56 ms | 483 |

Interpretation:

- VRAM is not the first limit for `V <= 512`
- the real limit is dense compute, which scales roughly with `K * V * H^2`
- with `H = 3V`, this is effectively cubic in `V`

## Folder Intent

This folder is for GPU research artifacts that are mergeable before a full GPU backend is canonical:

- standalone probes
- A/B harnesses
- VRAM sizing scripts
- batch candidate search prototypes
- short READMEs / notes that explain what was tested and why

It is acceptable to merge well-isolated experimental GPU tooling before the final GPU training path is complete.

## Planned First Probes

1. `gpu_crystal_pass_ab.py`
   - compare random-remove crystal against shuffled-pass crystal
   - first on frozen winner graphs
   - then under batched GPU evaluation

2. `gpu_swarm_v1.py`
   - `K=1` vs `K=32` vs `K=64`
   - `add_only` candidate swarm
   - best-of-batch master update

3. `gpu_specialist_mix_ab.py`
   - specialist proposal workers:
     - add-only
     - remove-only
     - rewire-light
   - phase- and size-dependent mixes

## Merge Philosophy

Merge to `main` when the GPU artifacts are:

- isolated
- runnable
- documented
- not touching canonical CPU behavior
- backed by logs or measurable output

Do not wait for a final "100% complete GPU rewrite" if the research harness is already useful and safe.
