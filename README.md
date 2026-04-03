# INSTNCT Core

`instnct-core` is the low-level recurrent spiking substrate behind VRAXION v5.

The public beta surface is intentionally small:

- `ConnectionGraph` manages sparse directed topology.
- `PropagationWorkspace` owns reusable hot-path buffers.
- `PropagationParameters`, `PropagationState`, and `PropagationConfig` define one forward pass.
- `propagate_token` runs the checked public propagation entrypoint.

## Quickstart

```rust
use instnct_core::{
    propagate_token, ConnectionGraph, PropagationConfig, PropagationParameters,
    PropagationState, PropagationWorkspace,
};

let mut graph = ConnectionGraph::new(2);
assert!(graph.add_edge(0, 1));

let input = [4, 0];
let threshold = [1, 1];
let channel = [1, 1];
let polarity = [1, 1];
let mut activation = [0, 0];
let mut charge = [0, 0];
let mut workspace = PropagationWorkspace::new(2);

propagate_token(
    &input,
    &graph,
    &PropagationParameters {
        threshold: &threshold,
        channel: &channel,
        polarity: &polarity,
    },
    &mut PropagationState {
        activation: &mut activation,
        charge: &mut charge,
    },
    &PropagationConfig::default(),
    &mut workspace,
)?;

# Ok::<(), instnct_core::PropagationError>(())
```

## Stability Notes

- The crate-root exports are the supported public beta API.
- Internal modules and benchmark-only hooks are intentionally not part of the compatibility promise.
- The checked public entrypoint rejects malformed input instead of panicking or silently truncating state.
