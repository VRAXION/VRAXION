# VRAXION Runtime

Minimal locked Rust runtime kernel for the current VRAXION probe stack.

This crate is deliberately small. It is not a training system and not a general
assistant. It contains only the currently locked runtime mechanics that are
ready to present as a homogeneous core:

```text
binary ingress reassembly
text field mode selection
locked body v1 sizing
PocketToken registry governance
Pocket Manager promotion policy
Next Mutation lifecycle gate
proposal boundary
Agency commit/reject/defer
trace-backed egress rendering
```

The research probes remain in `scripts/probes/`. This crate is the deterministic
runtime kernel those probes can converge toward.

## Module map

```text
bit_codec      shared deterministic bit helpers
binary_ingress frame encode/reassembly + integrity/requested-feature guards
text_field     Agency-selected text frame modes
body           Flow/Ground/Proposal/Agency locked body v1
pocket         PocketToken, registry, digest/ABI/lifecycle load guard
manager        vector score + challenger promotion policy
next_mutation  one-slot candidate -> mutation/rollback -> Golden Disc lifecycle
proposal       temporary Pocket proposal ABI
agency         commit/reject/defer/answer boundary
egress         rendering from committed state only
```

## Locked body v1

```text
Flow Field      = 28x28 cells
Ground Field    = 32x32 cells
Proposal Field  = 20 slots x 80 bits
Agency View     = 896 mechanical summary bits
```

Extended mode is `32x32` Flow/Ground and remains Agency-selected only. The
runtime keeps `48x48` as a research ceiling and treats `64x64` as an
overcapacity control, not a default.

The public API is re-exported from `lib.rs` so older probes can keep importing
`vraxion_runtime::*` while the implementation stays split into audit-friendly
modules.

## Run

```powershell
cargo test -p vraxion-runtime
cargo run -p vraxion-runtime --bin adversarial_probe --release -- 10000
cargo run -p vraxion-runtime --bin locked_body_preflight --release -- 10000
cargo run -p vraxion-runtime --bin pocket_governance_preflight --release -- 10000
cargo run -p vraxion-runtime --bin pocket_manager_preflight --release -- 10000
cargo run -p vraxion-runtime --bin next_mutation_preflight --release -- 10000
```

## Boundary

This crate does not claim raw language reasoning, AGI, consciousness, production
deployment, or model-scale behavior.
