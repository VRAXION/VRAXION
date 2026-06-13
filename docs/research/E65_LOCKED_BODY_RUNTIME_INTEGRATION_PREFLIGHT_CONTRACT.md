# E65 Locked Body Runtime Integration Preflight Contract

## Purpose

E65 consolidates the locked VRAXION runtime body into the Rust kernel.

It is not a new capability probe. It checks that the currently locked pieces
run together in one deterministic Rust path:

```text
binary/text ingress
-> proposal-only Pocket boundary
-> Proposal Field capacity check
-> Agency commit/reject boundary
-> Flow/Ground update only after commit
-> Egress rendering from committed state
```

## Locked Body V1

```text
Flow Field      = 28x28 cells
Ground Field    = 32x32 cells
Proposal Field  = 20 slots x 80 bits
Agency View     = 896 mechanical summary bits
```

Extended mode:

```text
32x32 Flow/Ground
Agency-selected only
```

Research ceiling:

```text
48x48
```

Avoid default:

```text
64x64
```

## Required Rust Surface

```text
vraxion-runtime::body
DEFAULT_BODY
EXTENDED_BODY
RESEARCH_CEILING_BODY
OVERCAPACITY_AVOID_DEFAULT
LockedBodyRuntime
ProposalField
FieldMatrix
locked_body_preflight binary
```

## Invariants

```text
64-bit Proposal width control must reject full evidence records.
80-bit Proposal slot must accept full evidence records.
Proposal Field must reject slot overflow.
Invalid ingress must not mutate Flow/Ground.
Valid ingress must commit through Agency before Flow/Ground update.
Egress must render only from committed state.
```

## Validation

```text
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test -p vraxion-runtime
cargo run --release -p vraxion-runtime --bin locked_body_preflight -- 1000000 target/pilot_wave/e65_locked_body_runtime_integration_preflight
cargo run --release -p vraxion-runtime --bin adversarial_probe -- 1000000
```

Boundary: E65 is a deterministic runtime integration preflight. It is not a raw
language reasoning, AGI, consciousness, deployment-quality, or model-scale
claim.
