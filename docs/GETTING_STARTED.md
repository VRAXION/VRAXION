# Getting Started with VRAXION

_Last updated: 2026-06-14_

## Start Here

```text
branch = main
current_release = v5.0.0-e79.0
current_evidence_anchor = 05415f5b06a43440742715ea93a5e2ec97632f21
current_evidence_subject = Add E113 FineWeb light stress recycle probe
latest_released_runtime_slice = a908a838a1119540ed88bc91e10cfcb0bdae92a8
latest_released_runtime_subject = Add training data curriculum readiness gate
```

`v5.0.0-e79.0` is still the latest GitHub release. E80-E113 are tracked post-release evidence on `main`.

## Repository Layout

| Path | Purpose |
|---|---|
| `vraxion-runtime/` | Active Rust runtime and preflight/final-training binaries |
| `docs/research/E79_TRAINING_DATA_CURRICULUM_READINESS_*.md` | Latest released training-data/curriculum readiness gate contract/result evidence |
| `docs/research/E80_*` through `docs/research/E113_*` | Current post-release evidence layer |
| `scripts/probes/run_e80_*` through `scripts/probes/run_e113_*` | Repro/check scripts for the post-release evidence layer |
| `docs/research/artifact_samples/` | Tracked sample artifacts used by evidence CI |
| `scripts/tools/generate_operator_rank_dashboard.py` | Self-contained Operator rank dashboard generator |
| `docs/` | Current public docs and GitHub Pages front door |
| `target/` | Generated local evidence artifacts; ignored by Git |

## Quick Checks

```powershell
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test --workspace
python -m compileall -q scripts
```

Current released preflight and final-training entrypoints:

```powershell
cargo run --release -p vraxion-runtime --bin pocket_library_preflight -- --help
cargo run --release -p vraxion-runtime --bin curriculum_runner_preflight -- --help
cargo run --release -p vraxion-runtime --bin curriculum_queue_preflight -- --help
cargo run --release -p vraxion-runtime --bin curriculum_resume_preflight -- --help
cargo run --release -p vraxion-runtime --bin final_bake_preflight -- --help
cargo run --release -p vraxion-runtime --bin training_data_readiness -- 3 8 target/ci/e79_training_data_readiness_smoke
cargo run --release -p vraxion-runtime --bin final_train -- 3 8 target/ci/e79_final_train_smoke --preflight-rounds 4 --checkpoint-interval 4
```

Current post-release evidence checks:

```powershell
python scripts/probes/run_e89_operator_naming_canonicalization_check.py
python scripts/probes/run_e107_operator_library_e90_e106_survival_role_and_regression_gauntlet_check.py --sample-only docs/research/artifact_samples/e107_operator_library_e90_e106_survival_role_and_regression_gauntlet
python scripts/probes/run_e108_external_dataset_operator_transfer_and_negative_scope_gauntlet_check.py --sample-only docs/research/artifact_samples/e108_external_dataset_operator_transfer_and_negative_scope_gauntlet
python scripts/probes/run_e109_operator_rank_ladder_and_golden_watch_probation_mode_check.py --sample-only docs/research/artifact_samples/e109_operator_rank_ladder_and_golden_watch_probation_mode
python scripts/probes/run_e110_promote_or_drop_operator_grind_wave1_check.py --sample-only docs/research/artifact_samples/e110_promote_or_drop_operator_grind_wave1
python scripts/probes/run_e111_bronze_mutation_prune_promote_or_drop_wave_check.py --sample-only docs/research/artifact_samples/e111_bronze_mutation_prune_promote_or_drop_wave
python scripts/probes/run_e112_gold_to_core_prune_heavy_probation_wave_check.py --sample-only docs/research/artifact_samples/e112_gold_to_core_prune_heavy_probation_wave
```

E113's checker validates a full local FineWeb stress artifact, not a tracked sample pack:

```powershell
python scripts/probes/run_e113_fineweb_light_stress_hard_mutation_recycle_check.py --out target/pilot_wave/e113_fineweb_light_stress_hard_mutation_recycle
```

Dashboard smoke:

```powershell
python scripts/tools/generate_operator_rank_dashboard.py --out target/ci/operator_rank_dashboard/index.html
```

## Reading Order

| Step | Document | Why |
|---|---|---|
| 1 | [`README.md`](../README.md) | Current front-door summary |
| 2 | [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md) | Current status and claim boundary |
| 3 | [`VALIDATED_FINDINGS.md`](../VALIDATED_FINDINGS.md) | Current validated chain |
| 4 | [`docs/research/E113_FINEWEB_LIGHT_STRESS_HARD_MUTATION_RECYCLE_RESULT.md`](research/E113_FINEWEB_LIGHT_STRESS_HARD_MUTATION_RECYCLE_RESULT.md) | Current post-release evidence head |
| 5 | [`docs/research/OPERATOR_LIBRARY_CARDS.md`](research/OPERATOR_LIBRARY_CARDS.md) | Operator rank/card surface |
| 6 | [`docs/research/E79_TRAINING_DATA_CURRICULUM_READINESS_RESULT.md`](research/E79_TRAINING_DATA_CURRICULUM_READINESS_RESULT.md) | Latest released training-data/curriculum readiness gate |
| 7 | [Timeline Archive wiki](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive) | Chronology and cleanup record |

## Operating Rule

Long runs must continuously write partial outcomes and resumable progress. A run that only writes a result at the end is not acceptable.

## Archive

The active branch surface is `main` only. Historical branch heads are preserved under:

```text
archive/branches/2026-06-13/*
```

The pre-cleanup wiki is preserved under:

```text
archive/wiki/pre-consolidation-2026-06-13
```
