# E65 Locked Body Runtime Integration Preflight Result

Status: completed for the Rust runtime kernel.

## Decision

```text
decision = e65_locked_body_runtime_preflight_passed
```

## Locked Rust Body

The Rust crate now exposes the E64 size lock directly:

```text
DEFAULT_BODY.name        = near_28f_32g_20x80_default
Flow Field               = 28x28 cells
Ground Field             = 32x32 cells
Proposal Field           = 20 slots x 80 bits
Agency View              = 896 mechanical summary bits
```

It also preserves:

```text
EXTENDED_BODY            = wide_32x32_20x80
RESEARCH_CEILING_BODY    = large_48x48_24x80
OVERCAPACITY_AVOID_DEFAULT = oversized_64x64_32x80
PROPOSAL_WIDTH_64_CONTROL = regression control
```

## Rust Preflight

```text
cargo run --release -p vraxion-runtime --bin locked_body_preflight -- 1000000 target/pilot_wave/e65_locked_body_runtime_integration_preflight

passed = true
rounds = 1000000
cases = 3000002
success = 3000002
false_commit = 0
missed_commit = 0
rows_per_sec = 361059.425
```

The preflight writes:

```text
target/pilot_wave/e65_locked_body_runtime_integration_preflight/runtime_config.json
target/pilot_wave/e65_locked_body_runtime_integration_preflight/preflight_results.json
target/pilot_wave/e65_locked_body_runtime_integration_preflight/progress.jsonl
target/pilot_wave/e65_locked_body_runtime_integration_preflight/report.md
```

## Regression Probe

The prior Rust adversarial probe still passes after the body integration:

```text
cargo run --release -p vraxion-runtime --bin adversarial_probe -- 1000000

passed = true
cases = 10000007
success = 10000007
false_commit = 0
false_frame = 0
wrong_feature = 0
rows_per_sec = 638260.927
```

## Interpretation

E65 moves the current locked AI body from probe-only documentation into the
Rust runtime kernel. The crate now has a deterministic one-body path:

```text
binary ingress
-> proposal field
-> Agency boundary
-> Flow/Ground commit
-> egress rendering
```

The result does not mean main curriculum training is complete. It means the
standard body v1 is now available as a consolidated Rust runtime target for the
next training and pocket-generation steps.

## Boundary

E65 is a deterministic runtime integration preflight. It is not a raw language
reasoning, AGI, consciousness, deployment-quality, or model-scale claim.
