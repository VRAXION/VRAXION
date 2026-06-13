# VRAXION Runtime

Minimal locked Rust runtime kernel for the current VRAXION probe stack.

This crate is deliberately small. It is not a training system and not a general
assistant. It contains only the currently locked runtime mechanics that are
ready to present as a homogeneous core:

```text
binary ingress reassembly
text field mode selection
proposal boundary
Agency commit/reject/defer
trace-backed egress rendering
```

The research probes remain in `scripts/probes/`. This crate is the deterministic
runtime kernel those probes can converge toward.

## Run

```powershell
cargo test -p vraxion-runtime
cargo run -p vraxion-runtime --bin adversarial_probe --release -- 10000
```

## Boundary

This crate does not claim raw language reasoning, AGI, consciousness, production
deployment, or model-scale behavior.
