# E74 Rust Final Bake API Extraction Contract

## Summary

E74 turns the E73 final-bake preflight from a large binary-local implementation
into a reusable Rust runtime API with a thin CLI wrapper.

Core question:

```text
Can the final-bake orchestration live inside vraxion-runtime as a public API,
while preserving the E73 one-command evidence behavior?
```

This moves the project closer to a final consolidated Rust program: downstream
callers can invoke the bake gate directly from Rust instead of shelling out to a
script-like binary.

## Runtime Surface

Library module:

```text
vraxion-runtime/src/final_bake.rs
```

Public API:

```text
vraxion_runtime::run_final_bake_preflight(rounds, out) -> FinalBakeSummary
```

Thin CLI:

```text
vraxion-runtime/src/bin/final_bake_preflight.rs
```

The CLI may only parse arguments, call the library API, and print the returned
summary.

## Required Behavior

The extracted API must preserve the E73 gate behavior:

```text
body gate
text gate
registry gate
manager + mutation gate
library gate
curriculum checkpoint/resume gate
artifact writeout
progress writeout
```

Pass conditions:

```text
passed = true
resume_passed = true
final_checksum_match = true
bad_commit_rate = 0
unsafe_promotion_rate = 0
```

## Artifact Root

```text
target/pilot_wave/e74_rust_final_bake_api_extraction/
```

Required files:

```text
final_bake_results.json
progress.jsonl
report.md
```

Sample pack:

```text
archived_public_artifact_sample_removed
```

## Boundary

E74 is a Rust runtime consolidation milestone. It does not prove open-ended
learning, raw text reasoning, AGI, consciousness, production deployment, or
model-scale behavior.
