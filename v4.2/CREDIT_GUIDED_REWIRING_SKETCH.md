# Credit-Guided Rewiring Sketch

Status: implementation sketch for the next `v4.2` experiment.

This is deliberately not wired into [model/graph.py](./model/graph.py) yet.
The goal is to prototype blame routing and edge ranking in an external test file first.

## Target

Build a test-only path that:

1. traces the standard forward pass
2. computes an output loss
3. propagates blame backward on the same graph
4. converts blame into edge action proposals
5. keeps the existing mutation+selection doctrine intact

## Forward trace contract

A minimal trace function should record, for each tick:

- `acts_t`
- `charges_t`
- `raw_t`
- `preclip_t`
- active neuron mask
- active edge list or active edge contribution summary

Suggested signature:

```python
def forward_batch_trace(net, ticks=8):
    ...
    return {
        "logits": logits,
        "acts": acts_by_tick,
        "charges": charges_by_tick,
        "raw": raw_by_tick,
        "preclip": preclip_by_tick,
        "active_nodes": active_nodes_by_tick,
        "active_edges": active_edges_by_tick,
    }
```

## Loss

Use the same eval target as the current graph:

```python
probs = softmax(logits)
delta_out = probs - onehot(target)
```

This keeps the credit prototype aligned with the current score.

## Backward credit

### Exact small-graph version

For small graphs and short horizons, use an exact unrolled backward pass.

```python
def backward_credit(net, trace, targets):
    delta_a = zeros_like(trace["acts"])
    delta_c = zeros_like(trace["charges"])
    edge_credit = zeros_like(net.mask)

    delta_c[-1][:, out_slice] += softmax(trace["logits"]) - onehot(targets)

    for t in reversed(range(T)):
        gate_thresh = (trace["charges"][t] > net.THRESHOLD).astype(np.float32)
        gate_clip = (np.abs(trace["preclip"][t]) < 1.0).astype(np.float32)

        delta_preact = delta_a[t] * gate_thresh
        delta_preclip = (delta_c[t] + delta_preact) * gate_clip
        delta_raw = net.retention * delta_preclip

        if t > 0:
            delta_c[t - 1] += net.retention * delta_preclip
            delta_a[t - 1] += delta_raw @ net.mask.T
            edge_credit += trace["acts"][t - 1].T @ delta_raw

    return edge_credit
```

This does not need to be numerically perfect to be useful. It only needs to rank edge actions better than random.

### Sparse large-graph version

For larger graphs, restrict blame propagation to:

- active neurons only
- active edges only
- optionally last `H` ticks only

That keeps the method compatible with scaling.

## Edge action ranking

Convert edge credit into structural actions.

### Remove / flip candidates

For existing edges `(i, j)`:

```python
signed_credit = edge_credit[i, j] * sign(mask[i, j])
```

Interpretation:

- strongly negative `signed_credit`: remove or flip candidate
- near-zero `signed_credit`: prune candidate if the region is crowded
- strongly positive `signed_credit`: keep

### Add candidates

For nonexistent edges, do not score all `N^2` pairs.

Instead:

1. rank senders by recent activity
2. rank receivers by recent blame
3. form a small sender x receiver proposal set
4. score only those pairs

Suggested heuristic:

```python
sender_score[i] = mean_recent_activity[i]
receiver_score[j] = mean_recent_abs_blame[j]
proposal_score[i, j] = sender_score[i] * receiver_score[j]
```

Then filter out self-connections and already-alive edges.

## Mutation integration options

### Option A: bias the existing mutator

Keep `net.mutate()` as the main path, but bias random draws toward high-credit proposals.

Pros:

- smallest disruption
- keeps current doctrine intact

Cons:

- harder to interpret cleanly

### Option B: explicit guided action phase

Run a short guided phase before or after normal mutation:

- guided `add/remove/flip/rewire`
- evaluate
- keep/revert

Pros:

- easier to benchmark directly
- clearer ablation against baseline

Cons:

- more invasive experimental loop

For the first prototype, Option B is cleaner.

## Minimal experiment matrix

Run at `V=64` first.

Conditions:

- baseline random mutate
- direct pain injection baseline
- credit-guided remove/flip only
- credit-guided add/remove/flip
- credit-guided add/remove/flip/rewire

Report:

- mean score across seeds
- variance across seeds
- accepted mutation rate
- edge action mix
- score trajectory

## Acceptance bar

Treat the prototype as promising only if it shows at least one of these:

- better mean score than baseline
- lower seed variance than pain probes
- clearer structural convergence than random mutation

If it only adds complexity without improving any of those, it is not ready for the main doctrine.
