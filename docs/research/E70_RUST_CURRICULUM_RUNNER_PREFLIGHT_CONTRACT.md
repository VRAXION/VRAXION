# E70 Rust Curriculum Runner Preflight Contract

## Purpose

E70 consolidates the first deterministic Rust curriculum runner path over the
locked runtime kernel.

It is not final training. It verifies that a row can move through the complete
currently locked runtime route:

```text
Pocket Library active set
-> guarded Pocket load
-> binary evidence ingress
-> Proposal Field
-> Agency commit boundary
-> Flow/Ground update
-> trace-backed egress
-> Next Mutation lifecycle evidence
-> Pocket Manager promotion evidence
-> persistent store promotion
-> reload snapshot check
```

## Required Rust Surface

```text
vraxion-runtime::curriculum
CurriculumLesson
CurriculumVerdict
CurriculumBlockReason
RustCurriculumRunner
curriculum_runner_preflight binary
```

## Required Rules

```text
active Pocket Set must be selected from Registry + PocketToken state.
quarantined pockets must not run.
stale tokens must block.
Pocket load must pass digest/token/ABI/lifecycle guards.
Pocket output must enter the Proposal Field, not Flow directly.
Agency must be the only Flow/Ground commit boundary.
Flow and Ground committed cells must agree after successful evidence commit.
Trace-backed egress must render only from committed Agency state.
Candidate promotion must pass Next Mutation lifecycle gates.
Candidate promotion must pass Pocket Manager promotion gates.
Unsafe/direct-write candidate evidence must not promote.
Store reload snapshot must match after promotion.
```

## Required Blocks

```text
no active pocket must block before runtime mutation.
wrong-feature binary frame must not commit or promote.
stale token must block.
unsafe direct-write candidate must block promotion.
concurrent stale store write must block.
bad commit rate must remain zero.
unsafe promotion rate must remain zero.
```

## Validation

```text
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test -p vraxion-runtime
cargo run --release -p vraxion-runtime --bin curriculum_runner_preflight -- 1000000 target/pilot_wave/e70_rust_curriculum_runner_preflight
```

Boundary: E70 is a deterministic row-loop preflight. It does not run open-ended
curriculum training, create new real Pocket skills, promote production
artifacts, or make raw language reasoning, AGI, consciousness,
deployment-quality, or model-scale claims.
