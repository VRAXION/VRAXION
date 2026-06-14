# Getting Started with VRAXION

_Last updated: 2026-06-14_

## Start Here

```text
branch = main
current_release = v5.0.0-e79.0
current_main_head = 56a9cf0305c1bfddd0e9b763b5e0d80fc9ec3bca
current_main_subject = Add E85 calc scribe mixed stream integration
latest_released_runtime_slice = a908a838a1119540ed88bc91e10cfcb0bdae92a8
latest_released_runtime_subject = Add training data curriculum readiness gate
base_runtime_slice = 0879a2c004cf6a002bd5639d9cb7a759709a41aa Extract Rust final bake API
```

The current `main` branch is:

```text
E69 Pocket Library store
E70 curriculum runner
E71 curriculum queue
E72 curriculum resume
E73 final bake
E74 final-bake API extraction
E75 final curriculum pocket-generation runner
E76 multi-lane final-training supervisor
E77 global Pocket Library merge supervisor
E78 canonical final_train campaign entrypoint
E79 training data/curriculum readiness gate
E80 dataset-backed scoring evidence
E81 CALC-SCRIBE v002 multiseed training
E82 CALC-SCRIBE v003 floor-division confirm
E83 CALC-SCRIBE v003 LocalGolden reload
E84 CALC-SCRIBE transfer and negative-scope probe
E85 CALC-SCRIBE mixed-stream inference integration
```

`v5.0.0-e79.0` is still the latest GitHub release. E80-E85 are tracked post-release evidence on `main`.

## Repository Layout

| Path | Purpose |
|---|---|
| `vraxion-runtime/` | Active Rust runtime and preflight/final-training binaries |
| `docs/research/E79_TRAINING_DATA_CURRICULUM_READINESS_*.md` | Latest released training-data/curriculum readiness gate contract/result evidence |
| `docs/research/E80_*` through `docs/research/E85_*` | Current post-release evidence for dataset-backed scoring and CALC-SCRIBE scoped validation |
| `scripts/probes/run_e80_*` through `scripts/probes/run_e85_*` | Repro/check scripts for the post-release evidence layer |
| `docs/research/artifact_samples/e79_training_data_curriculum_readiness/` | Latest release sample readiness manifest/progress/result/curriculum artifacts |
| `docs/research/E78_FINAL_TRAIN_CAMPAIGN_ENTRYPOINT_*.md` | Canonical final-training entrypoint contract/result evidence below the E79 gate |
| `docs/research/E76_*`, `docs/research/E77_*` | Current supervisor evidence below the canonical E78 entrypoint |
| `docs/` | Current public docs and GitHub Pages front door |
| `target/` | Generated local evidence artifacts; ignored by Git |

## Quick Checks

```powershell
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test -p vraxion-runtime
cargo test --workspace
```

Current preflight and final-training entrypoints:

```powershell
cargo run --release -p vraxion-runtime --bin pocket_library_preflight -- --help
cargo run --release -p vraxion-runtime --bin curriculum_runner_preflight -- --help
cargo run --release -p vraxion-runtime --bin curriculum_queue_preflight -- --help
cargo run --release -p vraxion-runtime --bin curriculum_resume_preflight -- --help
cargo run --release -p vraxion-runtime --bin final_bake_preflight -- --help
cargo run --release -p vraxion-runtime --bin training_data_readiness -- 3 8 target/ci/e79_training_data_readiness_smoke
cargo run --release -p vraxion-runtime --bin final_train -- 3 8 target/ci/e79_final_train_smoke --preflight-rounds 4 --checkpoint-interval 4
```

Current post-release evidence entrypoint:

```powershell
python scripts/probes/run_e85_calc_scribe_mixed_stream_inference_integration_probe.py --help
python scripts/probes/run_e85_calc_scribe_mixed_stream_inference_integration_probe_check.py --help
```

## Operating Rule

Long runs must continuously write partial outcomes and resumable progress. A run that only writes a result at the end is not acceptable.

## Reading Order

| Step | Document | Why |
|---|---|---|
| 1 | [`README.md`](../README.md) | Current front-door summary |
| 2 | [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md) | Current status and claim boundary |
| 3 | [`VALIDATED_FINDINGS.md`](../VALIDATED_FINDINGS.md) | Current validated E69-E85 chain |
| 4 | [`docs/research/E85_CALC_SCRIBE_MIXED_STREAM_INFERENCE_INTEGRATION_RESULT.md`](research/E85_CALC_SCRIBE_MIXED_STREAM_INFERENCE_INTEGRATION_RESULT.md) | Current mixed-stream CALC-SCRIBE evidence |
| 5 | [`docs/research/E79_TRAINING_DATA_CURRICULUM_READINESS_RESULT.md`](research/E79_TRAINING_DATA_CURRICULUM_READINESS_RESULT.md) | Latest released training-data/curriculum readiness gate evidence |
| 6 | [Timeline Archive wiki](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive) | Chronology and cleanup record |
| 7 | [Consolidation Archive wiki](https://github.com/VRAXION/VRAXION/wiki/Consolidation-Archive-2026-06-13) | Branch/wiki deletion manifest |

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
