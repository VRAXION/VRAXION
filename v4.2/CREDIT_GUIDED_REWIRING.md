# Credit-Guided Rewiring

Status: active design note for the `v4.2` self-wiring graph line.

## Why this note exists

The current self-wiring graph already shows two important properties:

- recurrence emerges naturally in the learned topology
- direct "pain" injection is not yet a reliable training signal

That combination matters. It means the next step should not be a stronger forward perturbation, but a backward-style credit mechanism that uses the current graph itself as the blame-routing scaffold.

## Current graph mechanics

The active reference core is [model/graph.py](./model/graph.py).

The batch forward path is:

```python
raw_t = a_{t-1} @ M
c_t = clip((c_{t-1} + raw_t) * retention, -1, 1)
a_t = relu(c_t - threshold)
```

Current defaults:

- `DRIVE = 0.6`
- `THRESHOLD = 0.5`
- `retention ~= 0.85` at `loss_pct = 15`
- standard eval horizon: `ticks = 8`

Operationally this means:

- one tick is roughly one graph hop
- thin single-path signals attenuate quickly
- short dense recurrent pockets are more plausible than long clean reverberating loops

## Fresh evidence

### 1. Loops are real

Quick run:

```bash
cd v4.2
/home/deck/.local/share/modal-env/bin/python tests/loop_analysis.py
```

Observed on March 18, 2026:

- random init already had a largest SCC of `93 / 96` neurons
- after `8000` training steps, the largest SCC was `96 / 96`
- bidirectional pairs increased from `5` to `572`
- sampled 3-cycles increased from `33` to `5654`
- feedback edges such as `output -> compute` and `output -> input` increased sharply after training

Interpretation:

- the learned graph is strongly recurrent
- feedback is not a hand-authored feature here; it emerges under mutation + selection
- this is topological evidence, not yet proof of long-horizon functional reuse

### 2. Direct pain routing is still weak

Quick smoke on clean `origin/main` semantics, bugfixed target routing, `V=32`, `NV_RATIO=4`, `budget=4000`, `ticks=8`, `seeds=[0,42,123]`:

- `baseline`: `45.9%`
- `dedicated pain band`: `43.6%` (`-2.3pp` vs baseline)
- `bar / thermometer pain`: `43.7%` (`-2.2pp`)
- `everyN`: `45.4%` (`-0.5pp`)

Interpretation:

- corrected direct pain probes are not winners on the clean mainline
- the more structured variants are not catastrophic in this small smoke, but they still fail to beat the baseline
- direct injection remains a probe, not a main training doctrine

## Why the simple pain idea stalls

The pain direction is attractive because it feels biologically plausible: extra channels, no explicit backprop, no hardcoded routing.

The mechanical problem is different:

- the injected signal does not follow the learned graph topology
- the same current that is supposed to teach also disturbs the forward computation being judged
- a strong topological loop count does not automatically mean the graph can safely absorb arbitrary injected error current

Even when the routing shape is improved, the signal is still being inserted from outside the graph's own credit structure.

## Proposed direction

Use the current accepted graph as the routing scaffold for blame.

Not:

- dense full-mask gradient descent
- stronger forward pain injection
- hardcoded feedforward teachers

Instead:

1. run the normal forward pass
2. trace active neurons and active edges over ticks
3. compute output diff at the end
4. propagate blame backward through the same graph
5. accumulate edge credit
6. use that credit to bias `add`, `remove`, `flip`, and `rewire`

## Backward sketch

For small `V`, exact unrolled backward over `ticks=8` is feasible.
For larger graphs, the same idea should become sparse and activity-gated.

Minimal form:

```python
delta_out = softmax(logits) - onehot(target)

for t in reversed(range(T)):
    delta_preact_t = delta_a_t * gate_thresh_t
    delta_preclip_t = delta_c_t * gate_clip_t
    delta_raw_t = retention * delta_preclip_t
    delta_c_{t-1} += retention * delta_preclip_t
    delta_a_{t-1} += delta_raw_t @ M.T
    edge_credit += a_{t-1}.T @ delta_raw_t
```

The useful object is the edge credit matrix, not a reinjected pain field.

Interpretation:

- missing edge with strong positive credit: add candidate
- existing edge with strong negative credit: remove candidate
- existing edge with wrong sign: flip candidate
- low-value edge in a saturated region: prune candidate

## Scaling stance

Exact dense BPTT is not the intended end state.

The current dense `act @ mask` path is already the expensive part at scale, so the practical large-graph version should be:

- sparse over alive edges
- limited to active neurons and active edges
- optionally limited to a short blame horizon
- proposal-based for nonexistent edges

For nonexistent edges, there is no direct edge gradient. The practical add path is:

- rank top sender neurons by recent activity
- rank top receiver neurons by recent blame
- only evaluate a small cross-product of add proposals

## Current research position

What is now supported by code and smoke evidence:

- recurrence is real in the learned topology
- direct pain injection is not yet a robust improvement path
- the next clean experiment is credit-guided rewiring, not stronger pain injection

What is still not proven:

- whether long loops are functionally reused across many effective turns of computation
- whether exact backward is necessary, or whether an active-subgraph approximation is enough
- which edge-credit heuristic is best for `add/remove/flip/rewire`

## Next implementation step

Prototype the credit path outside the core first.

Recommended sequence:

1. `forward_batch_trace()`
2. `backward_credit()`
3. `rank_edge_actions()`
4. compare baseline random mutate vs credit-guided mutate

A concrete sketch lives in [CREDIT_GUIDED_REWIRING_SKETCH.md](./CREDIT_GUIDED_REWIRING_SKETCH.md).
