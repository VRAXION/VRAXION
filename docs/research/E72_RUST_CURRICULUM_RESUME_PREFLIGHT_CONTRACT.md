# E72 Rust Curriculum Resume Preflight Contract

## Summary

E72 checks the first resume-safe Rust curriculum layer after the E70/E71
runtime consolidation.

Core question:

```text
Can a long curriculum queue run be checkpointed and resumed without changing
the final deterministic state?
```

This is a runtime safety preflight, not a new learning claim. It exists to
prevent black-box long runs: final training must have resumable checkpoints,
progress writeouts, and deterministic replay/audit.

## Runtime Surface

Files:

```text
vraxion-runtime/src/curriculum.rs
vraxion-runtime/src/bin/curriculum_resume_preflight.rs
```

The preflight uses the locked Rust curriculum runner:

```text
active Pocket set
-> guarded Pocket load
-> binary ingress
-> Proposal Field
-> Agency commit boundary
-> Flow/Ground sync
-> trace-backed egress
-> next mutation lifecycle gate
-> Pocket Library promotion/store
```

E72 adds:

```text
CurriculumCheckpoint
CurriculumResumeAudit
audit_resume(reference, resumed)
```

## Required Behavior

The runner must execute:

```text
1. uninterrupted reference run: 0..rounds
2. checkpointed first half: 0..split
3. resumed second half: split..rounds
4. final audit: reference final state == resumed final state
```

Required pass conditions:

```text
checkpoint_resume_compatible = true
final_checksum_match = true
final_queue_match = true
final_lesson_match = true
final_promotion_match = true
bad_commit_rate = 0
unsafe_promotion_rate = 0
```

The runner must write progress artifacts during execution and must not rely on
end-only output.

## Artifact Root

```text
target/pilot_wave/e72_rust_curriculum_resume_preflight/
```

Required files:

```text
resume_config.json
checkpoint_mid.json
checkpoint_final.json
preflight_results.json
progress.jsonl
report.md
```

Sample pack:

```text
docs/research/artifact_samples/e72_rust_curriculum_resume_preflight/
```

## Command

```powershell
cargo run --release -p vraxion-runtime --bin curriculum_resume_preflight -- 1000000 target\pilot_wave\e72_rust_curriculum_resume_preflight
```

## Boundary

E72 does not prove open-ended training, raw language reasoning, AGI,
consciousness, production service readiness, or model-scale behavior. It only
validates resume-safe deterministic curriculum mechanics inside the locked Rust
preflight stack.
