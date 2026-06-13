# E70 Rust Curriculum Runner Preflight Result

Status: completed for the Rust runtime kernel.

## Decision

```text
decision = e70_rust_curriculum_runner_preflight_passed
```

## Locked Rust Runner Surface

The Rust crate now exposes the first deterministic curriculum runner glue:

```text
CurriculumLesson
CurriculumVerdict
CurriculumBlockReason
RustCurriculumRunner
```

The runner path is:

```text
PocketToken active set
-> guarded load
-> LockedBodyRuntime binary evidence commit
-> Proposal Field boundary
-> Agency commit
-> Flow/Ground sync
-> trace-backed egress
-> Next Mutation lifecycle gate
-> Pocket Manager promotion gate
-> Pocket Library store promotion
-> reload snapshot check
```

## Rust Preflight

```text
cargo run --release -p vraxion-runtime --bin curriculum_runner_preflight -- 1000000 target/pilot_wave/e70_rust_curriculum_runner_preflight

passed = true
rounds = 1000000
curriculum_success_rate = 1.000000
active_set_success_rate = 1.000000
commit_success_rate = 1.000000
flow_ground_sync_rate = 1.000000
trace_render_success_rate = 1.000000
proposal_boundary_success_rate = 1.000000
promotion_success_rate = 1.000000
reload_success_rate = 1.000000
quality_delta_positive_rate = 1.000000
adversarial_block_rate = 1.000000
no_active_block_rate = 1.000000
bad_frame_block_rate = 1.000000
stale_token_block_rate = 1.000000
unsafe_candidate_block_rate = 1.000000
stale_write_block_rate = 1.000000
bad_commit_rate = 0.000000
unsafe_promotion_rate = 0.000000
rows_per_sec = 77962.484
```

The preflight writes:

```text
target/pilot_wave/e70_rust_curriculum_runner_preflight/curriculum_runner_config.json
target/pilot_wave/e70_rust_curriculum_runner_preflight/curriculum_trace_sample.json
target/pilot_wave/e70_rust_curriculum_runner_preflight/preflight_results.json
target/pilot_wave/e70_rust_curriculum_runner_preflight/progress.jsonl
target/pilot_wave/e70_rust_curriculum_runner_preflight/report.md
```

## Interpretation

E70 is the first Rust row-loop over the consolidated body:

```text
load a safe Pocket from the library
use it to process binary evidence through the Proposal Field
commit only through Agency
sync Flow/Ground
render trace-backed output
promote a candidate only after lifecycle and manager gates
verify persistent reload shape
```

This is the missing bridge between the earlier Rust preflights:

```text
E65 locked the runtime body.
E66 locked PocketToken registry governance.
E67 locked promotion policy.
E68 locked Next Mutation lifecycle.
E69 locked persistent Pocket Library store.
E70 runs one deterministic curriculum row through all of those gates together.
```

## Boundary

E70 is a deterministic row-loop preflight. It does not run open-ended
curriculum training, create new real Pocket skills, promote production
artifacts, or claim raw language reasoning, AGI, consciousness,
deployment-quality, or model-scale behavior.
