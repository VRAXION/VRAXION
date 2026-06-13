# E72 Rust Curriculum Resume Preflight Result

## Decision

```text
decision = e72_rust_curriculum_resume_checkpoint_confirmed
```

E72 confirms that the Rust curriculum queue can be checkpointed and resumed
without changing the final deterministic state.

## Evidence Run

Command:

```powershell
cargo run --release -p vraxion-runtime --bin curriculum_resume_preflight -- 1000000 target\pilot_wave\e72_rust_curriculum_resume_preflight
```

Primary result:

```text
passed = true
rounds = 1000000
split = 500000
reference_queues = 1000000
resumed_queues = 1000000
reference_lessons = 4000000
resumed_lessons = 4000000
checkpoint_resume_compatible = true
final_checksum_match = true
final_queue_match = true
final_lesson_match = true
final_promotion_match = true
bad_commit_rate = 0.000000
unsafe_promotion_rate = 0.000000
seconds = 29.228364300
queues_per_sec = 68426.682
lessons_per_sec = 273706.729
```

Final checkpoint:

```text
completed_queues = 1000000
completed_lessons = 4000000
promoted_count = 4000000
failed_count = 0
bad_commit_count = 0
unsafe_promotion_count = 0
checksum = 7802516916729865757
```

## What This Means

The E72 preflight runs the same curriculum sequence in two ways:

```text
uninterrupted reference run
checkpoint at halfway -> resume -> final state
```

The final checksum, queue count, lesson count, and promotion count match. This
means the current Rust curriculum preflight can be resumed deterministically
from checkpoint state.

## Important Runtime Fix

During E72, the preflight exposed a useful lifecycle issue: repeated promotion
of the same `pocket_uid` must update the canonical Pocket Library record rather
than append duplicate registry/token/artifact rows forever.

The Rust store now treats `insert_pocket` as canonical `pocket_uid` update:

```text
same pocket_uid -> replace canonical record
new pocket_uid  -> append new record
```

This prevents duplicate UID library growth during long curriculum loops while
preserving generation and ledger accounting.

## Artifact Samples

Committed sample pack:

```text
docs/research/artifact_samples/e72_rust_curriculum_resume_preflight/
```

Files:

```text
resume_config_sample.json
checkpoint_mid_sample.json
checkpoint_final_sample.json
preflight_results_sample.json
progress_sample.jsonl
```

## Boundary

E72 is a controlled deterministic runtime preflight. It does not claim raw
language reasoning, open-ended training success, AGI, consciousness, production
deployment, or model-scale behavior.
