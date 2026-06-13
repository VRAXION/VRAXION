# Getting Started with VRAXION

_Last updated: 2026-06-13_

## Start Here

```text
branch = main
current_release = v5.0.0-e75.0
release_head = 41fc0af81d1aec27220a653fdfc8666f748a228f
runtime_slice = 3f519732949b73d5b55ae90a740381ca81143948
runtime_subject = Add Rust final curriculum runner
base_runtime_slice = 0879a2c004cf6a002bd5639d9cb7a759709a41aa Extract Rust final bake API
```

The current mainline is:

```text
E69 Pocket Library store
E70 curriculum runner
E71 curriculum queue
E72 curriculum resume
E73 final bake
E74 final-bake API extraction
E75 final curriculum pocket-generation runner
```

## Repository Layout

| Path | Purpose |
|---|---|
| `vraxion-runtime/` | Active Rust runtime and preflight binaries |
| `docs/research/E75_RUST_FINAL_CURRICULUM_POCKET_GENERATION_RUNNER_*.md` | Current final-training runner contract/result evidence |
| `docs/research/artifact_samples/e75_rust_final_curriculum_pocket_generation_runner/` | Current sample progress/checkpoint/result/report artifacts |
| `docs/` | Current public docs |
| `target/` | Generated local evidence artifacts; ignored by Git |

## Quick Checks

```powershell
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test -p vraxion-runtime
cargo test --workspace
```

Current preflight entrypoints:

```powershell
cargo run --release -p vraxion-runtime --bin pocket_library_preflight -- --help
cargo run --release -p vraxion-runtime --bin curriculum_runner_preflight -- --help
cargo run --release -p vraxion-runtime --bin curriculum_queue_preflight -- --help
cargo run --release -p vraxion-runtime --bin curriculum_resume_preflight -- --help
cargo run --release -p vraxion-runtime --bin final_bake_preflight -- --help
cargo run --release -p vraxion-runtime --bin final_training_runner -- 1000 target/ci/e75_final_training_smoke --preflight-rounds 100 --checkpoint-interval 100
```

## Operating Rule

Long runs must continuously write partial outcomes and resumable progress. A run that only writes a result at the end is not acceptable.

## Reading Order

| Step | Document | Why |
|---|---|---|
| 1 | [`README.md`](../README.md) | Current front-door summary |
| 2 | [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md) | Current status and claim boundary |
| 3 | [`VALIDATED_FINDINGS.md`](../VALIDATED_FINDINGS.md) | Current validated E69-E75 chain |
| 4 | [Timeline Archive wiki](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive) | Chronology and cleanup record |
| 5 | [Consolidation Archive wiki](https://github.com/VRAXION/VRAXION/wiki/Consolidation-Archive-2026-06-13) | Branch/wiki deletion manifest |

## Archive

The active branch surface is `main` only. Historical branch heads are preserved under:

```text
archive/branches/2026-06-13/*
```

The pre-cleanup wiki is preserved under:

```text
archive/wiki/pre-consolidation-2026-06-13
```

The pre-E74 public-surface cleanup repo state is preserved under:

```text
archive/repo/pre-e74-public-surface-cleanup-2026-06-13
```
