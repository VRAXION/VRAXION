<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# INSTNCT / SWG v4.2

> **Status:** Active architecture line
>
> **Canonical code path:** [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py)

This page describes the current architecture line behind VRAXION while keeping the public state truthful.

## Status Taxonomy

- **Current mainline**: what is actually shipped in code on `main`
- **Validated finding**: experiment-backed result not yet promoted into the canonical code path
- **Experimental branch**: active target or prototype direction, not yet a live default

If a setting is not present in `v4.2/model/graph.py`, it should not be described here as the live default.

## What The Architecture Is

INSTNCT / SWG v4.2 is a gradient-free self-wiring architecture.

- `W_in` and `W_out` are fixed random passive I/O projections.
- The learnable object is a hidden-to-hidden ternary graph.
- Neurons keep persistent charge/state across ticks.
- Training is done by mutation + selection, not backpropagation through the graph.

That architecture is the reason VRAXION is interesting. The product claim is not “one more benchmark result,” but a different learnable object and training loop.

## Current Mainline

The current mainline is whatever is actually present in [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py).

As of the current `main` branch:

- `THRESHOLD = 0.5`
- `INJ_SCALE = 3.0`
- `DRIVE = 0.6`

Those are the settings that count as **Current mainline** until promotion happens in code.

## Validated Findings Not Yet Promoted

The strongest current validated findings are summarized canonically in [`VALIDATED_FINDINGS.md`](https://github.com/VRAXION/VRAXION/blob/main/VALIDATED_FINDINGS.md).

### Flip mutation

- **Status:** Validated finding
- **Evidence:** [issue #112](https://github.com/VRAXION/VRAXION/issues/112)
- **Setup:** English next-byte, 1024 neurons, 18 workers, 200 steps
- **Outcome:** `flip` beat float weight perturbation by `+1.89%` eval
- **Mainline status:** not yet promoted into `graph.py` defaults

### Low theta + `INJ_SCALE=1.0`

- **Status:** Validated finding
- **Evidence:** [issue #113](https://github.com/VRAXION/VRAXION/issues/113)
- **Setup:** empty-start English sweep, 18 workers, 100 steps
- **Outcome:** `scale=1.0 + theta=0.03` beat `scale=3.0 + theta=0.1` (`12.91%` vs `11.01%`)
- **Mainline status:** not yet promoted into `graph.py` defaults

## Experimental Branch

- **Current next target:** [issue #114](https://github.com/VRAXION/VRAXION/issues/114)
- **Label:** Experimental branch
- **Meaning:** mixed 18-worker swarm is an active implementation target, not a live recipe on `main`

## How To Verify Quickly

```bash
python -m compileall v4.2 tools
python v4.2/tests/test_model.py
python tools/check_public_surface.py
```

## Related Pages

- [[Home]]
- [`README.md`](https://github.com/VRAXION/VRAXION/blob/main/README.md)
- [`VALIDATED_FINDINGS.md`](https://github.com/VRAXION/VRAXION/blob/main/VALIDATED_FINDINGS.md)
