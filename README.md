# INSTNCT Core

> **VRAXION** /vræk.ʃən/ — **INSTNCT** /ˈɪnstɪŋkt/

`instnct-core` is the spiking network engine behind VRAXION v5. Integer-only forward pass, gradient-free evolution, zero `unsafe`.

## Quickstart

```rust
use instnct_core::{Network, PropagationConfig};

// Create a 256-neuron network and wire some edges
let mut net = Network::new(256);
net.graph_mut().add_edge(10, 42);
net.graph_mut().add_edge(42, 10);

// Run one token through the spiking forward pass
let input = vec![0i32; 256];
net.propagate(&input, &PropagationConfig::default())?;

// Snapshot for evolution rollback
let snapshot = net.save_state();
net.graph_mut().add_edge(5, 99);  // mutate
// ... evaluate ...
net.restore_state(&snapshot);     // rollback if worse

# Ok::<(), instnct_core::NetworkError>(())
```

## Public API

| Type | Purpose |
|------|---------|
| `Network` | Owned spiking network (topology + params + state) |
| `NetworkSnapshot` | Frozen state for evolution rollback |
| `ConnectionGraph` | Sparse directed graph (edge list + HashSet) |
| `propagate_token` | Low-level checked forward pass (for custom callers) |
| `PropagationConfig` | Timing: ticks per token, input duration, decay interval |
| `NetworkError` | Wraps `PropagationError` for validation failures |

## Stability Notes

- The crate-root exports are the supported public beta API.
- Internal modules and benchmark-only hooks are not part of the compatibility promise.
- The checked entrypoint rejects malformed input instead of panicking or silently truncating state.
- `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]` enforced.
