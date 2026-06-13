# E75 Rust Final Curriculum Pocket Generation Runner Contract

## Summary

E75 adds the first final-training-facing Rust runner on top of the locked
runtime stack.

Core question:

```text
Can the locked Rust runtime run a deterministic curriculum / pocket-generation
loop with preflight gating, progress writeouts, checkpoints, resume behavior,
Pocket Library growth, and zero unsafe commits/promotions?
```

This is not a new architecture probe. It wires the already locked pieces into a
training-runner shape.

## Runtime Surface

Library module:

```text
vraxion-runtime/src/final_training.rs
```

Public API:

```text
vraxion_runtime::run_final_curriculum_pocket_generation(config)
vraxion_runtime::FinalTrainingConfig
vraxion_runtime::FinalTrainingSummary
```

Thin CLI:

```text
vraxion-runtime/src/bin/final_training_runner.rs
```

Example:

```powershell
cargo run --release -p vraxion-runtime --bin final_training_runner -- 10000 target\pilot_wave\e75_rust_final_curriculum_pocket_generation_runner --preflight-rounds 1000 --checkpoint-interval 1000
```

Resume example:

```powershell
cargo run --release -p vraxion-runtime --bin final_training_runner -- 12000 target\pilot_wave\e75_rust_final_curriculum_pocket_generation_runner --resume --preflight-rounds 1000 --checkpoint-interval 1000
```

## Required Behavior

The runner must:

```text
run the final-bake preflight gate before training
run deterministic curriculum queues
promote safe candidate Pocket artifacts through the Pocket Library store
write progress.jsonl
write checkpoint_latest.json
write partial_summary.json
write final_training_results.json
write library_summary.json
write report.md
support deterministic resume/replay from checkpoint position
```

Pass conditions:

```text
passed = true
preflight_gate_passed = true
completed_queues = requested rounds
failed_count = 0
bad_commit_rate = 0
unsafe_promotion_rate = 0
generated_pocket_count > 0
checkpoint_written = true
```

## Artifact Root

```text
target/pilot_wave/e75_rust_final_curriculum_pocket_generation_runner/
```

Required files:

```text
final_training_results.json
progress.jsonl
checkpoint_latest.json
partial_summary.json
library_summary.json
report.md
preflight_gate/
```

Sample pack:

```text
docs/research/artifact_samples/e75_rust_final_curriculum_pocket_generation_runner/
```

## Boundary

E75 is a deterministic Rust final-training runner bootstrap. It does not prove
open-ended learning, raw text reasoning, AGI, consciousness, production
deployment, or model-scale behavior.
