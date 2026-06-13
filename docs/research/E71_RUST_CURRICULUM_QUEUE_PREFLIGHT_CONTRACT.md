# E71 Rust Curriculum Queue Preflight Contract

## Purpose

E71 extends E70 from one deterministic curriculum row to a bounded curriculum
queue.

It is not final open-ended training. It verifies that multiple lesson families
and multiple promotion candidates can move through the consolidated Rust runtime
without direct Flow writes, unsafe promotion, stale load, or queue-level bad
commit.

## Required Rust Surface

```text
vraxion-runtime::curriculum
CurriculumQueueLesson
CurriculumQueueReport
RustCurriculumRunner::run_queue
curriculum_queue_preflight binary
```

## Required Queue Path

```text
queue lesson family
-> active Pocket Set
-> guarded Pocket load
-> binary evidence ingress
-> Proposal Field
-> Agency commit
-> Flow/Ground sync
-> trace-backed egress
-> Next Mutation lifecycle evidence
-> Pocket Manager promotion evidence
-> Pocket Library store promotion
-> queue reload snapshot
```

## Required Lesson Families

```text
frame_integrity
requested_feature_match
trace_commit
reload_writeback
```

The family names are mechanical stress labels, not semantic model lanes.

## Required Blocks

```text
quarantined active set must block.
unsafe direct-write candidate evidence must block promotion.
wrong-feature stream must not commit or promote.
stale token must block.
concurrent stale store write must block.
bad commit rate must remain zero.
unsafe promotion rate must remain zero.
```

## Validation

```text
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test -p vraxion-runtime
cargo run --release -p vraxion-runtime --bin curriculum_queue_preflight -- 1000000 target/pilot_wave/e71_rust_curriculum_queue_preflight
```

Boundary: E71 is a deterministic queue preflight. It does not run open-ended
curriculum training, learn new real skills, promote production artifacts, or
make raw language reasoning, AGI, consciousness, deployment-quality, or
model-scale claims.
