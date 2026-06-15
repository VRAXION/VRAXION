# Getting Started with VRAXION

_Last updated: 2026-06-15_

## Start Here

```text
branch = main
current_release = v6.1.7
current_evidence_anchor = E130B arithmetic text-IO transfer and word-problem no-call gauntlet
current_evidence_subject = E129 arithmetic trace operators transferred to visible-expression text IO with hidden word-problem no-call
latest_released_runtime_slice = a908a838a1119540ed88bc91e10cfcb0bdae92a8
latest_released_runtime_subject = Add training data curriculum readiness gate
```

`v6.1.7` is the latest GitHub release. It anchors the E127 cycle-40 governed text-operator library checkpoint.

## Repository Layout

| Path | Purpose |
|---|---|
| `vraxion-runtime/` | Active Rust runtime and preflight/final-training binaries |
| `docs/research/E79_TRAINING_DATA_CURRICULUM_READINESS_*.md` | Latest released training-data/curriculum readiness gate contract/result evidence |
| `CODEX_HANDOVER.md` | First-read handover for fresh Codex sessions |
| `docs/research/E80_*` through `docs/research/E130B_*` | Current evidence layer |
| `scripts/probes/run_e80_*` through `scripts/probes/run_e130b_*` | Repro/check scripts for the current evidence layer |
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

E127 sample artifact smoke:

```powershell
python -m py_compile scripts/probes/run_e127_overnight_text_skill_farm_orange_cycle.py scripts/tools/generate_operator_rank_dashboard.py
python scripts/tools/generate_operator_rank_dashboard.py --e127 docs/research/artifact_samples/e127_overnight_text_skill_farm_orange_cycle --out target/ci/operator_rank_dashboard/index.html
```

E128 assistant text-IO lightweight render-training smoke:

```powershell
python -m py_compile scripts/probes/run_e128_assistant_text_io_lightweight_render_training.py
python scripts/probes/run_e128_assistant_text_io_lightweight_render_training.py --out target/ci/e128_assistant_text_io_lightweight_render_training
```

E129 arithmetic trace Orange/Legendary probation:

```powershell
python -m py_compile scripts/probes/run_e129_arithmetic_trace_orange_legendary_probation.py
python scripts/probes/run_e129_arithmetic_trace_orange_legendary_probation.py --out target/ci/e129_arithmetic_trace_orange_legendary_probation
```

E130A CoreMemoryCandidate-to-Orange backfill gauntlet:

```powershell
python -m py_compile scripts/probes/run_e130a_corememory_to_orange_backfill_gauntlet.py
python scripts/probes/run_e130a_corememory_to_orange_backfill_gauntlet.py --out target/ci/e130a_corememory_to_orange_backfill_gauntlet
```

E130B arithmetic text-IO transfer and word-problem no-call gauntlet:

```powershell
python -m py_compile scripts/probes/run_e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet.py
python scripts/probes/run_e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet.py --out target/ci/e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet --visible-cases-per-operator 120 --word-problem-cases-per-operator 60
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
| 4 | [`CODEX_HANDOVER.md`](../CODEX_HANDOVER.md) | Fresh-session handover and next steps |
| 5 | [`docs/research/E130B_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET_RESULT.md`](research/E130B_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET_RESULT.md) | Current arithmetic text-IO transfer evidence |
| 6 | [`docs/research/E130A_COREMEMORY_TO_ORANGE_BACKFILL_GAUNTLET_RESULT.md`](research/E130A_COREMEMORY_TO_ORANGE_BACKFILL_GAUNTLET_RESULT.md) | Orange backfill evidence |
| 7 | [`docs/research/E129_ARITHMETIC_TRACE_ORANGE_LEGENDARY_PROBATION_RESULT.md`](research/E129_ARITHMETIC_TRACE_ORANGE_LEGENDARY_PROBATION_RESULT.md) | Arithmetic trace evidence |
| 8 | [`docs/research/E128_ASSISTANT_TEXT_IO_LIGHTWEIGHT_RENDER_TRAINING_RESULT.md`](research/E128_ASSISTANT_TEXT_IO_LIGHTWEIGHT_RENDER_TRAINING_RESULT.md) | Text-IO bridge evidence |
| 9 | [`docs/research/E127_OVERNIGHT_TEXT_SKILL_FARM_ORANGE_CYCLE_RESULT.md`](research/E127_OVERNIGHT_TEXT_SKILL_FARM_ORANGE_CYCLE_RESULT.md) | E127 operator-library checkpoint |
| 10 | [`docs/research/OPERATOR_LIBRARY_CARDS.md`](research/OPERATOR_LIBRARY_CARDS.md) | Operator rank/card surface |
| 11 | [`docs/research/E79_TRAINING_DATA_CURRICULUM_READINESS_RESULT.md`](research/E79_TRAINING_DATA_CURRICULUM_READINESS_RESULT.md) | Latest released training-data/curriculum readiness gate |
| 12 | [Timeline Archive wiki](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive) | Chronology and cleanup record |

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
