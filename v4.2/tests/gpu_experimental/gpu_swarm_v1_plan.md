# GPU Swarm V1 Plan

## Goal

Build a minimal GPU-side candidate swarm for the current PassiveIO mainline:

- many local candidate graphs in parallel
- same accepted master graph as starting point
- batch evaluate all candidates on GPU
- promote the best improving candidate back to the master

This is a search-speed experiment, not a canonical model rewrite.

## Core Motivation

The useful quantity is not raw state-space size. It is:

`p_next_better = proposal-distribution mass on improving local neighbors`

If a single proposal has improve probability `p`, then a batch of `K` proposals has:

`P(hit) = 1 - (1 - p)^K`

This is why GPU matters here: not mainly because one proposal is faster, but because many local proposals can be tested together.

## Current CPU Lessons To Carry Forward

1. `add_only` growth wins at `V >= 64`
2. `rewire` is size-dependent and tends to hurt on larger networks
3. end-of-run crystal pruning helps
4. deeper crystal passes reveal large redundancy
5. proposal simplicity is often better than clever but expensive heuristics
6. after each accepted remove, the graph is a new system; crystal therefore needs iterative re-evaluation, not one-shot cleanup logic

## V1 Non-Goals

- no GPU rewrite of the full canonical CPU trainer
- no sparse GPU backend yet
- no learned worker ecology yet
- no change to the CPU source of truth in [`graph.py`](S:/AI/work/VRAXION_DEV/v4.2/model/graph.py)

## Proposed Tensor Model

Shared:

- `accepted_mask [H, H]`
- `W_in [V, H]`
- `W_out [H, V]`
- targets / row index helpers

Per candidate in a dense v1:

- `candidate_mask [H, H]`
- `acts [V, H]`
- `charges [V, H]`
- `raw [V, H]`
- `logits [V]`

Batched:

- `candidate_masks [K, H, H]`
- `acts [K, V, H]`
- `charges [K, V, H]`
- `raw [K, V, H]`
- `logits [K, V, V]` or reduced score-only path

## V1 Scheduler

### Master

Stores:

- best mask
- best score
- best edge count
- version counter

### Workers

Each candidate starts from the current master and applies one proposal recipe.

V1 recipes:

- `grower`: add-only
- `pruner`: remove-only
- `rewirer`: light rewire

Initial recommendation:

- `V < 64`: allow some `rewirer`
- `V >= 64`: mostly `grower`, optional `pruner`

## V1 Acceptance Rule

For a candidate batch:

1. materialize `K` candidate masks
2. evaluate all `K` on GPU
3. choose highest-score candidate
4. promote only if:
   - `score_new > score_best + eps`
   - or `abs(score_new - score_best) <= eps` and edge count is lower

Suggested:

- `eps = 1e-6`

## Crystal V1

The CPU discussion strongly suggests pass-based crystal is better than retry-based random remove.

Preferred crystal design:

- take current alive-edge list
- shuffle once
- test each edge exactly once for remove acceptance
- if at least one edge was removed, rebuild and start next pass
- if a full pass removes zero edges, stop

This reaches a clean 1-edge prune optimum without coupon-collector waste.
It also matches the real system dynamics:

- after each accepted remove, the system is different
- other edges may become newly removable
- therefore crystal is an iterative fixed-point search over successive reduced systems

Stop condition:

- stop when one full shuffled pass removes zero edges

This replaces retry-based stop logic such as:

- fixed patience
- coupon-collector attempt caps

Those can still exist as safety guards, but they should not be the primary crystal definition.

## First A/B Matrix

### A/B 1: Crystal

- retry/random crystal
- shuffled-pass crystal

Metrics:

- removed edge fraction
- score delta
- wall time
- passes needed
- attempted removes

### A/B 2: Swarm Width

- `K=1`
- `K=32`
- `K=64`

Metrics:

- best score trajectory
- promoted update count
- candidate/s
- wall time to target score

### A/B 3: Specialist Mix

- all add
- add + prune
- add + prune + light rewire

Metrics:

- final score
- final edge count
- crystal remove fraction

## Concrete Execution Order

### Phase 1: Crystal First

Reason:

- highest confidence CPU signal
- simplest GPU-side engineering
- directly tests the new fixed-point pruning insight

Run set:

- `V=64`: 5-10 frozen winners
- `V=128`: 5-10 frozen winners
- compare retry crystal vs pass crystal

Promote if:

- score is preserved
- removed fraction is higher or equal
- wall time is lower or acceptable

Current status:

- `V=64` practical pass reached on the GPU branch
- pass-based crystal beats retry/random on remove-attempt count and wall time
- the remaining score delta is negligible (`-2.78e-05` median on the measured V64 matrix)
- branch direction: keep pass-based crystal as the canonical GPU crystal primitive

### Phase 2: Add-Only Swarm

Reason:

- current CPU results suggest large models prefer simple growth

Run set:

- `V=64`: `K=1`, `K=32`, `K=64`
- `V=128`: `K=1`, `K=32`, `K=64`

Promote if:

- better score trajectory at equal wall time
- better best-of-run score at equal attempt budget

Current status:

- first naive `best-of-batch` smoke is **negative on quality** at equal eval budget
- `K=32/64` is dramatically faster, but `K=1` keeps more promotions and wins score in the short run
- branch interpretation:
  - this v1 scheduler is a valid throughput probe
  - it is not yet a bake-ready training improvement
  - if revisited, the likely next scheduler should reduce promotion starvation

### Phase 2b: Periodic Crystal Scheduler

Current branch probe:

- [`gpu_crystal_schedule_ab.py`](S:/AI/work/VRAXION_DEV/v4.2/tests/gpu_experimental/gpu_crystal_schedule_ab.py)

What it tests:

- split the same total add budget into multiple grow chunks
- run pass-based crystal between chunks
- compare against continuous growth with only end crystal

Current status:

- short-budget V64: frequent crystal compresses hard, but does not clearly improve score
- longer V64 single-seed: one mid-crystal can improve score materially
- longer V64 three-seed median: score is roughly tied, but the segmented schedule is denser

Interpretation:

- the CPU-side "grow -> crystal -> grow" win does not yet transfer cleanly to the current GPU empty-start harness
- fixed equal segments are probably too blunt
- if revisited, the next scheduler should crystalize on state:
  - stale threshold
  - edge-load threshold
  - or score plateau

### Phase 2c: State-Triggered Crystal Scheduler

Current branch probe:

- [`gpu_state_crystal_ab.py`](S:/AI/work/VRAXION_DEV/v4.2/tests/gpu_experimental/gpu_state_crystal_ab.py)

What it tests:

- `continuous_end`
- `stale_once`
- `stale_repeat2`

with:

- `K=1`
- identical add-only proposal stream across policies
- pass-based mid-crystal only when the stale threshold is hit
- final pass-based crystal in every policy

Important correction:

- the first state-triggered smoke was proposal-stream-confounded
- the corrected harness now keeps growth RNG invariant across policies so scheduler comparisons are fair

Current status on corrected V64 matrix (`seeds=42,77,123`, `total_evals=2048`, `ticks=6`):

- no policy passed the positive gate
- aggressive triggers (`64`) harmed score
- medium triggers (`128`) produced baseline-tie behavior, not enough edge savings
- conservative triggers (`256`) usually never fired and collapsed to baseline
- all runs were deterministic

Interpretation:

- state-triggered mid-crystal is **not** a bake-ready GPU win in this scheduler family
- do not escalate this exact probe to `V=128`
- keep the harness as a documented negative result
- current branch baseline remains:
  - continuous add-only growth
  - one deep pass-based final crystal

### Phase 3: Specialist Mix

Only after Phase 2 is stable.

Run set:

- add-only
- add + prune specialists
- add + prune + light rewire specialists

The expected result is that larger `V` stays close to add-first behavior, with specialists helping primarily in the finalization stage rather than in core growth.

## Practical Starting Sizes

- `V=64, H=192`: best first target
- `V=128, H=384`: second target
- `V=256, H=768`: only after `V=64/128` shape is clear

Recommended initial batch widths:

- `V=64`: `K=64`
- `V=128`: `K=32-64`
- `V=256`: `K=16-32`

## Merge Criteria

This branch work is ready for merge when it provides:

- runnable scripts
- a short README
- reproducible logs or summary tables
- zero changes to canonical CPU behavior

The first merge target is an experimental GPU toolkit, not the final GPU backend.
