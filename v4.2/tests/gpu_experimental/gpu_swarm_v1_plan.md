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

## First A/B Matrix

### A/B 1: Crystal

- random crystal
- shuffled-pass crystal

Metrics:

- removed edge fraction
- score delta
- wall time
- attempts needed

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
