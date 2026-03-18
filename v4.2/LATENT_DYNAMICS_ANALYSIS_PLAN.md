# Latent Dynamics Analysis Plan

This note defines how to test whether the current self-wiring graph has a
meaningful internal representation space, or only recurrent activity with no
useful latent structure.

## Why This Matters

The recent loop findings are topological evidence:

- loops form
- feedback edges increase during training
- strongly connected recurrent pockets emerge

That is important, but it does not yet prove that the model develops a useful
internal representation.

The real question is:

> Do the recurrent loops create a structured internal state-space that carries
> task-relevant information over ticks?

If yes, that is an emergent latent-like dynamic space.
If not, then the loops may still be real, but they are mostly recurrent
plumbing rather than representation.

## Terminology

For this repository, use these terms precisely:

- `explicit latent space`: a dedicated encoder-produced latent variable such as
  `z` in an encoder-decoder or VAE. The current model does not have this.
- `implicit latent-like dynamics`: a useful internal state trajectory formed by
  `charge` and `state` over recurrent ticks. This may exist in the current
  graph.
- `topological loops`: cycles in the connectivity graph. These enable internal
  dynamics but are not themselves the latent space.

## Current State

What we already know:

- `loop_analysis.py` shows that loops and feedback edges emerge under training.
- `signal_trace.py` shows that signal survives only for a limited effective
  horizon under the current retention and threshold settings.
- `loop_vs_ticks.py` and related scripts already probe recurrence at the
  topological and coarse learning-curve level.

What is still missing:

- state-trace capture across ticks
- class-separability analysis in internal state
- evidence that feedback loops change representation quality rather than only
  loop count
- tests that distinguish structured internal memory from recurrent noise

## Main Hypotheses

### H1: Emergent latent-like dynamics

Training causes the graph to use recurrent loops to build a structured internal
state-space.

Expected signatures:

- trained nets show higher state separability than random nets
- internal states predict class identity better than raw topology alone
- feedback ablation damages this separability and/or accuracy
- useful information survives across multiple ticks before readout

### H0: Recurrent plumbing only

Training increases loops, but the internal dynamics do not organize into a
useful representational space.

Expected signatures:

- low class separability in internal state
- little difference between trained and random state geometry
- feedback ablation barely changes state quality
- most predictive signal appears only very near the output slice

## Decision-Critical Measurements

We do not need a large benchmark matrix first. We need a small set of sharp
measurements that decide whether the internal dynamics are meaningful.

### 1. State separability over ticks

For each input class and each tick, capture:

- full `charge` vector
- full `act` vector
- output logits

Measure:

- within-class distance
- between-class distance
- nearest-centroid or linear-probe accuracy from internal state

Decision signal:

- if separability rises over ticks in trained nets and not in random nets, the
  model is building internal class structure

### 2. Effective dimensionality

For the per-tick state matrices, measure:

- PCA variance spectrum
- participation ratio / effective rank
- optional simple 2D projection for visualization

Decision signal:

- if trained nets use more than a trivial one- or two-mode attractor while
  preserving separability, that supports a useful latent-like space

### 3. Temporal persistence

Test how much task-relevant information survives across ticks.

Measure:

- probe accuracy from `charge_t` and `act_t`
- decay curve of separability across ticks
- delayed readout quality

Decision signal:

- if useful state exists only at the final tick, there is weak internal memory
- if useful information appears earlier and remains available across ticks,
  recurrence is doing more than immediate feed-through

### 4. Functional feedback ablation

Compare each trained graph against ablated variants:

- remove output->compute edges
- remove output->input edges
- remove all explicit feedback-zone edges

Then re-run forward only, without retraining.

Measure:

- score drop
- accuracy drop
- internal separability drop

Decision signal:

- if loop removal hurts both performance and internal state quality, the loops
  are functional, not cosmetic

### 5. Tick-local perturbation propagation

Inject a small perturbation into one neuron or one zone at tick `t`, then track
how much it changes later internal states and final output.

Measure:

- output sensitivity by source zone and source tick
- perturbation survival curve
- whether the effect remains local or reverberates

Decision signal:

- if perturbations propagate through recurrent pockets in a structured way, that
  is stronger evidence for real internal computation

## Minimal Experimental Matrix

Keep the first pass small and decision-oriented.

### Models

1. random init
2. trained baseline
3. trained baseline with feedback ablation

### Sizes

1. `V=32`
2. `V=64`

### Tick budgets

1. `ticks=4`
2. `ticks=8`
3. `ticks=16`

### Seeds

Start with `3` seeds.
If the signal is strong and consistent, scale up after that.

## First Implementation Pass

### Phase A: Trace primitive

Add a test-only helper:

- `forward_batch_trace(net, ticks=8)`

Record per tick:

- `acts`
- `charges`
- `raw`
- `active_mask`
- optional zone summaries
- final logits

This is the minimum shared primitive that all later analyses need.

### Phase B: Latent diagnostics script

Create a focused script, for example:

- `tests/latent_dynamics_probe.py`

Responsibilities:

- load or train a graph
- capture trace
- compute separability metrics
- compute effective-rank metrics
- dump a compact text table and JSON summary

### Phase C: Feedback ablation script

Create a paired script, for example:

- `tests/feedback_ablation_probe.py`

Responsibilities:

- copy the trained mask
- remove selected feedback zones
- rerun trace and compare metrics against original

## Output Format

Each run should save:

- score / accuracy
- tick-wise separability table
- tick-wise effective-rank table
- perturbation or ablation deltas
- compact JSON result for later comparison

The first pass should optimize for text summaries, not plotting infrastructure.

## Interpretation Rules

We should call the result a meaningful latent-like internal space only if most
of the following hold:

1. trained > random on internal-state separability
2. separability emerges before final readout, not only at the last tick
3. effective dimension is nontrivial but not pure noise
4. feedback ablation measurably damages both score and state quality
5. perturbations show structured recurrent influence rather than instant decay

If only loop count rises, but these do not hold, then the correct conclusion is:

> the graph is recurrent, but we do not yet have evidence of a useful internal
> latent-like state-space

## Fastest Path To First Answer

Do this first, in order:

1. implement `forward_batch_trace()`
2. run `random init` vs `trained baseline` at `V=32`, `ticks=8`
3. compute per-tick class separability from full `charge`
4. ablate feedback edges and rerun the same probe

That is the shortest path to an actual answer.

## Why This Is Worth Doing

If the result is positive, it sharpens the project claim substantially:

- not only does the graph evolve recurrent loops
- it also builds an internal dynamic representation space through those loops

If the result is negative, that is still useful:

- it tells us loops alone are not enough
- it tells us where to focus next: retention, normalization, credit routing, or
  explicit representation pressure

Either way, the experiment is decision-complete.
