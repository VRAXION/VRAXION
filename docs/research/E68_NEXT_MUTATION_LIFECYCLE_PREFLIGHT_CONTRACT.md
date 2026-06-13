# E68 Next Mutation Lifecycle Preflight Contract

## Purpose

E68 consolidates the E51 Next Mutation slot lifecycle into the Rust runtime
kernel.

It is not a new training probe. It checks that a candidate cannot become a
Golden Disc unless it passes the full one-slot lifecycle:

```text
NEXT_MUTATION
-> LIGHT_PROBE_PASS
-> ACTIVE_REFINEMENT
-> STABLE
-> S_RANK
-> GOLDEN_DISC
```

## Required Rust Surface

```text
vraxion-runtime::next_mutation
MutationLifecycleStage
MutationBlockReason
MutationStats
NextMutationEvidence
NextMutationVerdict
GoldenDiscRecord
evaluate_next_mutation_lifecycle
next_mutation_preflight binary
```

## Required Rules

```text
exactly one active next-mutation lane
sandbox-only candidate execution
candidate writes only to Proposal Field
light probe required before refinement
mutation/rollback evidence required before promotion
rollback_count must equal rejected mutations
refinement must improve candidate quality
prune/crystallize must pass before S-rank
unique value by counterfactual ablation required
challenger sweep must not find a better mutation
trace/replay/wrong-commit gates required
direct Flow write is disallowed
Golden Disc requires frozen identity: uid + digest + PocketToken metadata
```

## Required Blocks

```text
parallel candidate spam must discard.
light-probe-only must not promote.
refinement without uniqueness must not promote.
rollback mismatch must not promote.
challenger loss must not promote.
direct Flow write must discard before quality is considered.
missing frozen identity must not save Golden Disc.
```

## Validation

```text
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test -p vraxion-runtime
cargo run --release -p vraxion-runtime --bin next_mutation_preflight -- 1000000 target/pilot_wave/e68_next_mutation_lifecycle_preflight
```

Boundary: E68 is a deterministic lifecycle preflight. It does not create new
Pocket skills, run curriculum training, promote any real production artifact,
or make raw language reasoning, AGI, consciousness, deployment-quality, or
model-scale claims.
