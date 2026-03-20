# GPU Experimental

Experimental GPU-side search and evaluation work for the current PassiveIO mainline.

## Branch Metadata

- Branch: `codex/gpu-passiveio-swarm`
- Base branch: `main`
- Original base commit: `1095c8a`
- Current branch also includes the newer `main` crystal update commit `f5ece0b` (`pass-based crystallize()` merge)
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
- end-of-run crystal pruning helps
- deeper crystal passes can remove a large redundant fraction of edges without score loss
- after each accepted remove, the graph is a genuinely new system; crystal is not a one-shot cleanup, it is an iterative compression process over successive reduced systems

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
   - compare retry/random crystal against shuffled-pass crystal
   - crystal definition:
     - shuffle current alive-edge list
     - test each edge once for safe remove
     - if at least one edge is removed, rebuild and start a fresh pass
     - stop only when a full pass removes zero edges
   - first on frozen winner graphs
   - then under batched GPU evaluation
   - expected win condition:
     - equal score
     - larger removed-edge fraction
     - lower wall time than retry/coupon-counter crystal

2. `gpu_swarm_v1.py`
   - `K=1` vs `K=32` vs `K=64`
   - `add_only` candidate swarm
   - best-of-batch master update
   - expected win condition:
     - more promoted updates per wall time
     - better score trajectory at equal budget

3. `gpu_specialist_mix_ab.py`
   - specialist proposal workers:
     - add-only
     - remove-only
     - rewire-light
   - phase- and size-dependent mixes

## Current Measured Status

### Stage A: Crystal A/B

Current branch harness:

- [`gpu_crystal_pass_ab.py`](S:/AI/work/VRAXION_DEV/v4.2/tests/gpu_experimental/gpu_crystal_pass_ab.py)

Key implementation note:

- the GPU crystal harness uses a **fixed score floor** (`score_before - eps`) for remove acceptance
- this prevents long chains of individually "almost equal" accepts from drifting score downward
- this is an intentional experimental guardrail for Stage A verification

Measured V64 matrix (`seeds=42,77,123,321`, realistic add-only grown start graphs):

- retry/random crystal:
  - removed median: `61.83%`
  - attempts median: `22,796`
  - wall median: `59.57s`
- pass-based crystal:
  - removed median: `62.36%`
  - attempts median: `13,001`
  - wall median: `35.66s`
- median score delta (pass - retry): `-2.78e-05`

Interpretation:

- pass-based crystal is **effectively score-preserving**
- prune is slightly better
- remove attempts and wall time are materially better
- repeated runs are deterministic

Working verdict:

- `V=64` Stage A is a practical **PASS**
- pass-based crystal is the preferred GPU crystal primitive

### Stage B: Swarm Width Smoke

Current branch harness:

- [`gpu_swarm_v1.py`](S:/AI/work/VRAXION_DEV/v4.2/tests/gpu_experimental/gpu_swarm_v1.py)

Initial smoke (`V=64`, `seed=42`, `total_evals=256`, empty-start, add-only, crystal-end):

- `K=1`
  - score: `8.62%`
  - wall: `642ms`
- `K=32`
  - score: `7.03%`
  - wall: `53ms`
- `K=64`
  - score: `4.69%`
  - wall: `69ms`

Interpretation:

- naive best-of-batch master promotion is **much faster**
- but at equal total eval budget it is also **promotion-starved**
- this means Stage B is currently a **throughput demo, not yet a quality win**

Working verdict:

- keep the harness
- do not bake naive swarm width into any recommendation yet
- if revisited, the next likely direction is a less greedy scheduler:
  - fixed wall-time comparison
  - micro-rollouts / island workers
  - or multiple promotions per batch

### Stage B.5: Periodic Crystal Schedule Smoke

Current branch harness:

- [`gpu_crystal_schedule_ab.py`](S:/AI/work/VRAXION_DEV/v4.2/tests/gpu_experimental/gpu_crystal_schedule_ab.py)

What it tests:

- `segments=1`: continuous add-only growth, then final crystal
- `segments=2`: grow half -> crystal -> grow half -> final crystal
- `segments=4`: four short grow chunks with mid-crystal between them, then final crystal

Short V64 smoke (`seed=42`, `total_evals=512`, `K=1`):

- `seg=1`: `14.12%`, `78` edges
- `seg=2`: `13.34%`, `50` edges
- `seg=4`: `14.09%`, `31` edges

Interpretation:

- early budget: more frequent crystal strongly compresses the graph
- quality does not clearly improve at this stage
- `seg=4` nearly matched baseline score while using far fewer edges

Longer V64 single-seed probe (`seed=42`, `total_evals=2048`, `K=1`):

- `seg=1`: `22.76%`, `100` edges
- `seg=2`: `25.20%`, `183` edges
- `seg=4`: `22.77%`, `114` edges

Interpretation:

- at a longer budget, one mid-crystal can produce a real score gain on at least some runs
- the gain is not compression-driven; the winner here is denser, not sparser

V64 three-seed probe (`seeds=42,77,123`, `total_evals=2048`, `K=1`):

- `seg=1` median: `26.70%`, `122` edges
- `seg=2` median: `26.69%`, `160` edges

Working verdict:

- periodic mid-crystal is **not yet a clean GPU win**
- it can help on some seeds / budgets, but the current fixed-segment scheduler is not robust enough to bake
- likely next improvement:
  - trigger crystal on state (stale / edge load / plateau), not on fixed equal segments

### Stage B.6: Tick-Budget Plateau Probe

Current branch harness:

- [`gpu_tick_plateau_probe.py`](S:/AI/work/VRAXION_DEV/v4.2/tests/gpu_experimental/gpu_tick_plateau_probe.py)

What it tests:

- same seeded add-only grow budget
- same pass-based crystal logic
- only `ticks` changes

Quick V64 probe (`seeds=42,77`, `grow_attempts=2048`):

- `ticks=6`
  - grow edges mean: `2254.5`
  - crystal edges mean: `618.0`
  - removed mean: `72.7%`
  - score after mean: `26.50%`
- `ticks=10`
  - grow edges mean: `2159.5`
  - crystal edges mean: `628.0`
  - removed mean: `70.9%`
  - score after mean: `26.18%`
- `ticks=15`
  - grow edges mean: `2026.0`
  - crystal edges mean: `500.5`
  - removed mean: `75.3%`
  - score after mean: `20.75%`

Interpretation:

- the crystal plateau is **not invariant** to tick budget
- changing `ticks` materially changes both:
  - how many edges growth keeps
  - where crystal lands
- the effect is not yet monotonic on this small probe
- higher `ticks` can produce a smaller plateau, but the current grow budget may then be under-training the longer-range regime

Working verdict:

- the "plateau edge count is architecture-level and tick-sensitive" hypothesis is plausible
- but the current quick probe is still too small to claim a simple law like "more ticks always means fewer edges"

## Current Hypothesis

For the current `main` model:

- growth should be simple, mostly `add_only`
- crystal should be deep, iterative, and pass-based
- larger models likely want:
  - growth phase: mostly add
  - finalization phase: repeated prune passes until zero-remove fixed point

This means the first useful GPU backend is not "full sparse evolutionary magic".
It is:

- fast batched candidate growth
- fast batched crystal passes
- simple master/candidate promotion logic

## Planned Run Series

Ordered from highest-signal / lowest-risk to more speculative:

1. **Crystal A/B**
   - `retry crystal` vs `pass crystal`
   - sizes: `V=64`, `V=128`
   - output:
     - score before/after
     - edges before/after
     - removed fraction
     - time

2. **Swarm Width A/B**
   - `K=1`, `K=32`, `K=64`
   - proposal: `add_only`
   - sizes: `V=64` first, then `V=128`
   - output:
     - score trajectory
     - best score at fixed wall time
     - candidate/s
     - promotions per minute

3. **Specialist Mix A/B**
   - all add
   - add + prune specialists
   - add + prune + light rewire specialists
   - sizes: `V=64`, `V=128`

4. **Scale Gate**
   - validate whether `V=256` remains practical on the local 4070 Ti SUPER
   - if not, keep `V=256` as a research-only size and focus on `V=64/128`

## Merge Philosophy

Merge to `main` when the GPU artifacts are:

- isolated
- runnable
- documented
- not touching canonical CPU behavior
- backed by logs or measurable output

Do not wait for a final "100% complete GPU rewrite" if the research harness is already useful and safe.
