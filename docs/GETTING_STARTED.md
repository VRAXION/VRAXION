# Getting Started with VRAXION

_Last updated: 2026-06-13_

## Start Here

```text
branch = main
runtime_slice = 0879a2c004cf6a002bd5639d9cb7a759709a41aa
runtime_subject = Extract Rust final bake API
base_runtime_slice = 51cd82a11d8f1d2b98ee3e49538c7c26afdb767b Add Rust final bake preflight
```

The current mainline is:

```text
E69 Pocket Library store
E70 curriculum runner
E71 curriculum queue
E72 curriculum resume
E73 final bake
E74 final-bake API extraction
```

## Repository Layout

| Path | Purpose |
|---|---|
| `vraxion-runtime/` | Active Rust runtime and preflight binaries |
| `docs/research/E74_RUST_FINAL_BAKE_API_EXTRACTION_*.md` | Current final-bake API contract/result evidence |
| `docs/research/artifact_samples/e74_rust_final_bake_api_extraction/` | Current sample progress/result/report artifacts |
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
```

## Operating Rule

Long runs must continuously write partial outcomes and resumable progress. A run that only writes a result at the end is not acceptable.

## Reading Order

| Step | Document | Why |
|---|---|---|
| 1 | [`README.md`](../README.md) | Current front-door summary |
| 2 | [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md) | Current status and claim boundary |
| 3 | [`VALIDATED_FINDINGS.md`](../VALIDATED_FINDINGS.md) | Current validated E69-E74 chain |
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
