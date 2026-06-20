# E73 Rust Final Bake Preflight Contract

## Summary

E73 adds the first unified Rust final-bake entrypoint for the currently locked
runtime stack.

Core question:

```text
Can the locked Rust runtime be validated from one consolidated binary instead
of separate component preflight commands?
```

This is still a deterministic preflight. It is not open-ended training and not a
raw language/model-scale claim.

## Runtime Surface

Library API and binary:

```text
vraxion-runtime/src/final_bake.rs
vraxion-runtime/src/bin/final_bake_preflight.rs
```

Cargo command:

```powershell
cargo run --release -p vraxion-runtime --bin final_bake_preflight -- 1000000 target\pilot_wave\e73_rust_final_bake_preflight
```

The binary directly calls `vraxion_runtime::*` APIs. It must not shell out to
the older preflight binaries as an orchestration shortcut.

The public API is:

```text
vraxion_runtime::run_final_bake_preflight(rounds, out)
```

The CLI is intentionally thin: it parses arguments, calls the library API, and
prints the returned summary.

## Gates

The final bake preflight must cover:

```text
binary ingress + locked body commit/reject behavior
Text Field mode selection
PocketToken registry/load governance
Pocket Manager promotion policy
Next Mutation lifecycle gate
Persistent Pocket Library store behavior
Curriculum queue checkpoint/resume audit
```

Required pass conditions:

```text
body_passed = body_cases
text_passed = text_cases
registry_passed = registry_cases
manager_mutation_passed = manager_mutation_cases
library_passed = library_cases
resume_passed = true
final_checksum_match = true
bad_commit_rate = 0
unsafe_promotion_rate = 0
```

The runner must write progress artifacts during execution and not rely on
end-only output.

## Artifact Root

```text
target/pilot_wave/e73_rust_final_bake_preflight/
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

E73 does not prove open-ended learning, raw text reasoning, AGI,
consciousness, production deployment, or model-scale behavior. It validates a
single Rust final-bake preflight entrypoint over the current locked mechanics.
