# Vraxion `instnct-core`

`instnct-core` is the Rust implementation surface for the INSTNCT v5 beta lane:
an integer-only spiking network engine with gradient-free evolution, sparse graph
topology, rollback snapshots, checkpoint persistence, and zero `unsafe`.

This repo is moving toward `v5.0.0 Public Beta` as a Rust-first public surface.
The current standard is reproducibility and implementation maturity, not a claimed
breakthrough beyond the tested `17-18%` language band.

## Public beta posture

- Rust is the main user-facing beta lane.
- The crate-root re-exports are the supported public beta API.
- The canonical beta runner is `instnct-core/examples/evolve_language.rs`.
- Other Rust examples are retained as experimental research surfaces and are not
  the compatibility promise.
- The broader project still keeps a Python reference line for developers, but the
  stable beta contract in this repo is Rust.

## Canonical beta run

```powershell
cargo run --release --example evolve_language -- <corpus-path> `
  --steps 15000 `
  --seed-count 12 `
  --seed-base 42 `
  --full-len 2000 `
  --report-dir target/beta-report
```

The canonical runner writes a minimum evidence bundle into `--report-dir`:

- `run_cmd.txt`
- `env.json`
- `metrics.json`
- `summary.md`

See [BETA.md](BETA.md) for the exact contract.

## Public API

| Type / function | Purpose |
|---|---|
| `Network` | Owned spiking network (topology + params + state) |
| `NetworkSnapshot` | Frozen runtime state for rollback |
| `ConnectionGraph` | Sparse directed graph surface |
| `InitConfig`, `build_network` | Proven init defaults and canonical network construction |
| `evolution_step`, `EvolutionConfig`, `StepOutcome` | Paired evaluation and mutation loop |
| `Int8Projection` | Learnable integer readout surface |
| `SdrTable` | Sparse token input table |
| `save_checkpoint`, `load_checkpoint`, `CheckpointMeta` | Atomic persistence bundle |
| `propagate_token` and propagation types | Checked low-level forward pass |

## Quickstart

```rust,no_run
use instnct_core::{build_network, InitConfig, PropagationConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;

let init = InitConfig::phi(256);
let mut rng = StdRng::seed_from_u64(42);
let mut net = build_network(&init, &mut rng);

let input = vec![0i32; init.neuron_count];
net.propagate(&input, &PropagationConfig::default())?;

# Ok::<(), instnct_core::NetworkError>(())
```

## Stability notes

- The crate-root exports are the supported public beta API.
- Internal modules and benchmark-only hooks are not part of the compatibility promise.
- The canonical beta runner is a public workflow surface, but not a stable library API.
- Experimental examples are kept for research continuity and may change without notice.
- `#![forbid(unsafe_code)]` and `#![deny(missing_docs, unreachable_pub)]` remain enforced.
