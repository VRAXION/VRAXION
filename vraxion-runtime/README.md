# VRAXION Runtime

Minimal locked Rust runtime kernel for the current VRAXION mainline.

This crate is deliberately small. It is not a general assistant. It contains
the currently locked runtime mechanics and the first final curriculum runner
that are ready to present as a homogeneous core:

```text
binary ingress reassembly
text field mode selection
locked body v1 sizing
PocketToken registry governance
Pocket Manager promotion policy
Next Mutation lifecycle gate
Persistent Pocket Library store
Curriculum runner preflight glue
Curriculum queue preflight glue
Curriculum resume/checkpoint preflight glue
Final bake preflight entrypoint
Final curriculum pocket-generation runner
proposal boundary
Agency commit/reject/defer
trace-backed egress rendering
```

Older research probes were removed from active `main` during the E74 public
surface cleanup and remain available from the archive tags. This crate is now
the only active runtime surface.

## Module map

```text
bit_codec      shared deterministic bit helpers
binary_ingress frame encode/reassembly + integrity/requested-feature guards
text_field     Agency-selected text frame modes
body           Flow/Ground/Proposal/Agency locked body v1
pocket         PocketToken, registry, digest/ABI/lifecycle load guard
manager        vector score + challenger promotion policy
next_mutation  one-slot candidate -> mutation/rollback -> Golden Disc lifecycle
library        persistent registry/tokens/artifacts/ledgers store model
curriculum     active-set -> guarded-load -> body commit -> promotion row loop
final_bake     unified body/text/registry/manager/library/curriculum bake gate
final_training final curriculum runner with checkpoints/progress/library growth
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

The public API is re-exported from `lib.rs` while the implementation stays split
into audit-friendly modules.

## Run

```powershell
cargo test -p vraxion-runtime
cargo run -p vraxion-runtime --bin adversarial_probe --release -- 10000
cargo run -p vraxion-runtime --bin locked_body_preflight --release -- 10000
cargo run -p vraxion-runtime --bin pocket_governance_preflight --release -- 10000
cargo run -p vraxion-runtime --bin pocket_manager_preflight --release -- 10000
cargo run -p vraxion-runtime --bin next_mutation_preflight --release -- 10000
cargo run -p vraxion-runtime --bin pocket_library_preflight --release -- 10000
cargo run -p vraxion-runtime --bin curriculum_runner_preflight --release -- 10000
cargo run -p vraxion-runtime --bin curriculum_queue_preflight --release -- 10000
cargo run -p vraxion-runtime --bin curriculum_resume_preflight --release -- 10000
cargo run -p vraxion-runtime --bin final_bake_preflight --release -- 10000
cargo run -p vraxion-runtime --bin final_training_runner --release -- 10000
```

## Boundary

This crate does not claim raw language reasoning, AGI, consciousness, production
deployment, or model-scale behavior.
