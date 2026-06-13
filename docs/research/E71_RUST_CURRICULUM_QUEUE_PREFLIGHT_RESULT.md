# E71 Rust Curriculum Queue Preflight Result

Status: completed for the Rust runtime kernel.

## Decision

```text
decision = e71_rust_curriculum_queue_preflight_passed
```

## Locked Rust Queue Surface

The Rust crate now exposes a bounded curriculum queue layer:

```text
CurriculumQueueLesson
CurriculumQueueReport
RustCurriculumRunner::run_queue
```

The queue path is:

```text
lesson family
-> active Pocket Set
-> guarded load
-> Proposal Field
-> Agency commit
-> Flow/Ground sync
-> trace-backed egress
-> lifecycle + promotion gates
-> store writeback
-> reload snapshot
```

## Rust Preflight

```text
cargo run --release -p vraxion-runtime --bin curriculum_queue_preflight -- 1000000 target/pilot_wave/e71_rust_curriculum_queue_preflight

passed = true
rounds = 1000000
queues = 1000000
lessons = 4000000
queue_success_rate = 1.000000
lesson_success_rate = 1.000000
promotion_success_rate = 1.000000
flow_ground_sync_rate = 1.000000
proposal_boundary_success_rate = 1.000000
reload_match_rate = 1.000000
quality_delta_positive_rate = 1.000000
adversarial_block_rate = 1.000000
no_active_queue_block_rate = 1.000000
unsafe_queue_block_rate = 1.000000
bad_stream_block_rate = 1.000000
stale_token_block_rate = 1.000000
stale_write_block_rate = 1.000000
bad_commit_rate = 0.000000
unsafe_promotion_rate = 0.000000
queues_per_sec = 37314.853
lessons_per_sec = 149259.412
```

The preflight writes:

```text
target/pilot_wave/e71_rust_curriculum_queue_preflight/curriculum_queue_config.json
target/pilot_wave/e71_rust_curriculum_queue_preflight/queue_row_samples.json
target/pilot_wave/e71_rust_curriculum_queue_preflight/preflight_results.json
target/pilot_wave/e71_rust_curriculum_queue_preflight/progress.jsonl
target/pilot_wave/e71_rust_curriculum_queue_preflight/report.md
```

The progress artifact includes a timed mid-run writeout:

```text
event = progress
round = 687116
queues = 687116
lessons = 2748464
adversarial_blocks = 3435580
bad_commit = 0
unsafe_promotion = 0
```

## Interpretation

E71 moves the Rust consolidation one step closer to the final training runtime:

```text
E70 proved one deterministic row can traverse all locked gates.
E71 proves a bounded multi-lesson queue can do the same with multiple candidate writebacks.
```

The adversarial controls show the queue is still gated:

```text
no active Pocket -> block
unsafe direct-write candidate -> block
wrong-feature stream -> no commit / no promotion
stale token -> block
concurrent stale store write -> block
```

## Boundary

E71 is a deterministic queue preflight. It does not run open-ended curriculum
training, learn new real skills, promote production artifacts, or claim raw
language reasoning, AGI, consciousness, deployment-quality, or model-scale
behavior.
