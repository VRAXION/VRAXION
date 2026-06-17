# Getting Started with VRAXION

_Last updated: 2026-06-17_

## Start Here

```text
branch = main
current_release = v6.1.7
current_evidence_anchor = E136N2 Agency Matrix arbitration smoke
current_evidence_subject = trained Agency Matrix arbitration over the E136N primary/secondary proposal surface
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
| `docs/research/E80_*` through `docs/research/E136N2_*` | Current evidence layer |
| `scripts/probes/run_e80_*` through `scripts/probes/run_e136n2_*` | Repro/check scripts for the current evidence layer |
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

E131 visible equation extraction and assistant arithmetic render gauntlet:

```powershell
python -m py_compile scripts/probes/run_e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet.py
python scripts/probes/run_e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet.py --out target/ci/e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet --sample-out "" --visible-cases-per-operator 160 --word-problem-cases-per-operator 80 --dataset-row-limit 12000 --allow-builtin-dataset
```

E132 external math-text skill farm mutation/prune Orange cycle smoke:

```powershell
python -m py_compile scripts/probes/run_e132_external_math_text_skill_farm_mutation_prune_orange_cycle.py
python scripts/probes/run_e132_external_math_text_skill_farm_mutation_prune_orange_cycle.py --dataset target/datasets/missing_e132_smoke.jsonl --allow-builtin-dataset --out target/ci/e132_external_math_text_skill_farm_mutation_prune_orange_cycle --sample-out "" --min-dataset-rows 1 --min-external-support 1 --min-orange 16 --dataset-row-limit 80
```

E133 math-text route composition and no-solve assistant confirm smoke:

```powershell
python -m py_compile scripts/probes/run_e133_math_text_route_composition_and_no_solve_assistant_confirm.py
python scripts/probes/run_e133_math_text_route_composition_and_no_solve_assistant_confirm.py --dataset target/datasets/missing_e133_smoke.jsonl --allow-builtin-dataset --out target/ci/e133_math_text_route_composition_and_no_solve_assistant_confirm --sample-out "" --route-cases-per-operator 40 --hidden-cases-per-operator 20 --control-cases-per-operator 10 --dataset-row-limit 300
```

E134 external math-text OOD route stress and counterexample gauntlet smoke:

```powershell
python -m py_compile scripts/probes/run_e134_external_math_text_ood_route_stress_and_counterexample_gauntlet.py
python scripts/probes/run_e134_external_math_text_ood_route_stress_and_counterexample_gauntlet.py --dataset target/datasets/missing_e134_smoke.jsonl --allow-builtin-dataset --out target/ci/e134_external_math_text_ood_route_stress_and_counterexample_gauntlet --sample-out "" --ood-cases-per-operator 40 --counterexample-cases-per-operator 20 --hidden-cases-per-operator 15 --control-cases-per-operator 10 --dataset-row-limit 300
```

E135 math-text multi-route assistant dialogue-state gauntlet smoke:

```powershell
python -m py_compile scripts/probes/run_e135_math_text_multi_route_assistant_dialogue_state_gauntlet.py
python scripts/probes/run_e135_math_text_multi_route_assistant_dialogue_state_gauntlet.py --dataset target/datasets/missing_e135_smoke.jsonl --allow-builtin-dataset --out target/ci/e135_math_text_multi_route_assistant_dialogue_state_gauntlet --sample-out "" --dialogue-cases-per-operator 30 --counterexample-dialogue-cases-per-operator 12 --control-cases-per-operator 8 --dataset-row-limit 300
```

E136A assistant-text skill farm mutation/prune Orange cycle smoke:

```powershell
python -m py_compile scripts/probes/run_e136a_assistant_text_skill_farm_mutation_prune_orange_cycle.py
python scripts/probes/run_e136a_assistant_text_skill_farm_mutation_prune_orange_cycle.py --dataset target/datasets/missing_e136a_smoke.jsonl --dataset-manifest target/datasets/missing_e136a_manifest.json --download-manifest target/datasets/missing_e136a_download.json --allow-builtin-dataset --dataset-row-limit 120 --min-assistant-support 1 --min-dataset-rows 1 --out target/ci/e136a_assistant_text_skill_farm_mutation_prune_orange_cycle --sample-out ""
```

E136B assistant-text route composition and boundary confirm smoke:

```powershell
python -m py_compile scripts/probes/run_e136b_assistant_text_route_composition_and_boundary_confirm.py
python scripts/probes/run_e136b_assistant_text_route_composition_and_boundary_confirm.py --dataset target/datasets/missing_e136b_smoke.jsonl --dataset-manifest target/datasets/missing_e136b_manifest.json --download-manifest target/datasets/missing_e136b_download.json --allow-builtin-dataset --dataset-row-limit 120 --min-dataset-rows 1 --route-cases-per-operator 40 --boundary-cases-per-operator 12 --control-cases-per-operator 8 --out target/ci/e136b_assistant_text_route_composition_and_boundary_confirm --sample-out ""
```

E136C assistant-text polished render quick test:

```powershell
python -m py_compile scripts/probes/run_e136c_assistant_text_polished_render_quick_test.py
python scripts/probes/run_e136c_assistant_text_polished_render_quick_test.py --out target/ci/e136c_assistant_text_polished_render_quick_test --sample-out ""
```

E136D OutputTextField binary matrix smoke:

```powershell
python -m py_compile scripts/probes/run_e136d_output_text_field_binary_matrix_smoke.py
python scripts/probes/run_e136d_output_text_field_binary_matrix_smoke.py --out target/ci/e136d_output_text_field_binary_matrix_smoke --sample-out ""
cargo test -p vraxion-runtime output_text_field
```

E136E idle think-tick proposal refinement smoke:

```powershell
python -m py_compile scripts/probes/run_e136e_idle_think_tick_proposal_refinement_smoke.py
python scripts/probes/run_e136e_idle_think_tick_proposal_refinement_smoke.py --out target/ci/e136e_idle_think_tick_proposal_refinement_smoke --sample-out ""
```

E136F idle think-tick heldout series confirm:

```powershell
python -m py_compile scripts/probes/run_e136f_idle_think_tick_heldout_series_confirm.py
python scripts/probes/run_e136f_idle_think_tick_heldout_series_confirm.py --out target/ci/e136f_idle_think_tick_heldout_series_confirm --sample-out ""
```

E136G adaptive idle tick budget confirm:

```powershell
python -m py_compile scripts/probes/run_e136g_adaptive_idle_tick_budget_confirm.py
python scripts/probes/run_e136g_adaptive_idle_tick_budget_confirm.py --out target/ci/e136g_adaptive_idle_tick_budget_confirm --sample-out ""
```

E136H existing operator refinement mutation/prune night cycle:

```powershell
python -m py_compile scripts/probes/run_e136h_existing_operator_refinement_mutation_prune_night_cycle.py
python scripts/probes/run_e136h_existing_operator_refinement_mutation_prune_night_cycle.py --cycles 2 --e132-rows-per-cycle 200 --e136a-rows-per-cycle 400 --out target/ci/e136h_existing_operator_refinement_mutation_prune_night_cycle --sample-out ""
```

E136I operator supersession and output ledger planning:

```powershell
python -m py_compile scripts/probes/run_e136i_operator_supersession_and_output_ledger_planning.py
python scripts/probes/run_e136i_operator_supersession_and_output_ledger_planning.py --out target/ci/e136i_operator_supersession_and_output_ledger_planning --sample-out ""
```

E136J shadow variant apply and residual prune confirm:

```powershell
python -m py_compile scripts/probes/run_e136j_shadow_variant_apply_and_residual_prune_confirm.py
python scripts/probes/run_e136j_shadow_variant_apply_and_residual_prune_confirm.py --max-cycles 2 --e132-batch-rows 128 --e136a-batch-rows 128 --out target/ci/e136j_shadow_variant_apply_and_residual_prune_confirm --sample-out ""
```

E136K operator replacement apply plan:

```powershell
python -m py_compile scripts/probes/run_e136k_operator_replacement_apply_plan_or_flow_scale_transfer.py
python scripts/probes/run_e136k_operator_replacement_apply_plan_or_flow_scale_transfer.py --out target/ci/e136k_operator_replacement_apply_plan_or_flow_scale_transfer --sample-out ""
```

E136L runtime replacement canary:

```powershell
python -m py_compile scripts/probes/run_e136l_runtime_replacement_canary_and_tightened_challenger_confirm.py
python scripts/probes/run_e136l_runtime_replacement_canary_and_tightened_challenger_confirm.py --e132-rows 512 --e136a-rows 512 --out target/ci/e136l_runtime_replacement_canary_and_tightened_challenger_confirm --sample-out ""
```

E136M runtime replacement overlay:

```powershell
python -m py_compile scripts/probes/run_e136m_runtime_replacement_apply_or_abstract_lineage_split.py
python scripts/probes/run_e136m_runtime_replacement_apply_or_abstract_lineage_split.py --out target/ci/e136m_runtime_replacement_apply_or_abstract_lineage_split --sample-out ""
```

E136N primary/secondary variant governance:

```powershell
python -m py_compile scripts/probes/run_e136n_primary_secondary_variant_governance.py
python scripts/probes/run_e136n_primary_secondary_variant_governance.py --out target/ci/e136n_primary_secondary_variant_governance --sample-out ""
```

E136N2 Agency Matrix arbitration smoke:

```powershell
python -m py_compile scripts/probes/run_e136n2_agency_matrix_arbitration_smoke.py
python scripts/probes/run_e136n2_agency_matrix_arbitration_smoke.py --out target/ci/e136n2_agency_matrix_arbitration_smoke --sample-out ""
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
| 5 | [`docs/research/E136N2_AGENCY_MATRIX_ARBITRATION_SMOKE_RESULT.md`](research/E136N2_AGENCY_MATRIX_ARBITRATION_SMOKE_RESULT.md) | Current Agency Matrix arbitration evidence |
| 6 | [`docs/research/E136N_PRIMARY_SECONDARY_VARIANT_GOVERNANCE_RESULT.md`](research/E136N_PRIMARY_SECONDARY_VARIANT_GOVERNANCE_RESULT.md) | Primary/secondary variant governance evidence |
| 7 | [`docs/research/E136M_RUNTIME_REPLACEMENT_APPLY_OR_ABSTRACT_LINEAGE_SPLIT_RESULT.md`](research/E136M_RUNTIME_REPLACEMENT_APPLY_OR_ABSTRACT_LINEAGE_SPLIT_RESULT.md) | Runtime replacement overlay evidence |
| 8 | [`docs/research/E136L_RUNTIME_REPLACEMENT_CANARY_AND_TIGHTENED_CHALLENGER_CONFIRM_RESULT.md`](research/E136L_RUNTIME_REPLACEMENT_CANARY_AND_TIGHTENED_CHALLENGER_CONFIRM_RESULT.md) | Runtime replacement canary evidence |
| 9 | [`docs/research/E136K_OPERATOR_REPLACEMENT_APPLY_PLAN_OR_FLOW_SCALE_TRANSFER_RESULT.md`](research/E136K_OPERATOR_REPLACEMENT_APPLY_PLAN_OR_FLOW_SCALE_TRANSFER_RESULT.md) | Operator replacement apply-plan evidence |
| 10 | [`docs/research/E136J_SHADOW_VARIANT_APPLY_AND_RESIDUAL_PRUNE_CONFIRM_RESULT.md`](research/E136J_SHADOW_VARIANT_APPLY_AND_RESIDUAL_PRUNE_CONFIRM_RESULT.md) | Shadow-apply/residual-prune evidence |
| 11 | [`docs/research/E136I_OPERATOR_SUPERSESSION_AND_OUTPUT_LEDGER_PLANNING_RESULT.md`](research/E136I_OPERATOR_SUPERSESSION_AND_OUTPUT_LEDGER_PLANNING_RESULT.md) | Supersession/output ledger evidence |
| 12 | [`docs/research/E136H_EXISTING_OPERATOR_REFINEMENT_MUTATION_PRUNE_NIGHT_CYCLE_RESULT.md`](research/E136H_EXISTING_OPERATOR_REFINEMENT_MUTATION_PRUNE_NIGHT_CYCLE_RESULT.md) | Existing-operator refinement evidence |
| 13 | [`docs/research/E136G_ADAPTIVE_IDLE_TICK_BUDGET_CONFIRM_RESULT.md`](research/E136G_ADAPTIVE_IDLE_TICK_BUDGET_CONFIRM_RESULT.md) | Adaptive idle tick-budget evidence |
| 14 | [`docs/research/E136F_IDLE_THINK_TICK_HELDOUT_SERIES_CONFIRM_RESULT.md`](research/E136F_IDLE_THINK_TICK_HELDOUT_SERIES_CONFIRM_RESULT.md) | Idle think-tick heldout-series evidence |
| 15 | [`docs/research/E136E_IDLE_THINK_TICK_PROPOSAL_REFINEMENT_SMOKE_RESULT.md`](research/E136E_IDLE_THINK_TICK_PROPOSAL_REFINEMENT_SMOKE_RESULT.md) | Idle think-tick proposal-refinement smoke |
| 16 | [`docs/research/E136D_OUTPUT_TEXT_FIELD_BINARY_MATRIX_SMOKE_RESULT.md`](research/E136D_OUTPUT_TEXT_FIELD_BINARY_MATRIX_SMOKE_RESULT.md) | OutputTextField binary matrix evidence |
| 17 | [`docs/research/E136C_ASSISTANT_TEXT_POLISHED_RENDER_QUICK_TEST_RESULT.md`](research/E136C_ASSISTANT_TEXT_POLISHED_RENDER_QUICK_TEST_RESULT.md) | Polished assistant-text render quick evidence |
| 18 | [`docs/research/E136B_ASSISTANT_TEXT_ROUTE_COMPOSITION_AND_BOUNDARY_CONFIRM_RESULT.md`](research/E136B_ASSISTANT_TEXT_ROUTE_COMPOSITION_AND_BOUNDARY_CONFIRM_RESULT.md) | Assistant-text route-composition/boundary evidence |
| 19 | [`docs/research/E136A_ASSISTANT_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md`](research/E136A_ASSISTANT_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md) | Assistant-text skill-farm evidence |
| 20 | [`docs/research/E135_MATH_TEXT_MULTI_ROUTE_ASSISTANT_DIALOGUE_STATE_GAUNTLET_RESULT.md`](research/E135_MATH_TEXT_MULTI_ROUTE_ASSISTANT_DIALOGUE_STATE_GAUNTLET_RESULT.md) | Controlled dialogue-state evidence |
| 21 | [`docs/research/E134_EXTERNAL_MATH_TEXT_OOD_ROUTE_STRESS_AND_COUNTEREXAMPLE_GAUNTLET_RESULT.md`](research/E134_EXTERNAL_MATH_TEXT_OOD_ROUTE_STRESS_AND_COUNTEREXAMPLE_GAUNTLET_RESULT.md) | OOD route-stress/counterexample evidence |
| 22 | [`docs/research/E133_MATH_TEXT_ROUTE_COMPOSITION_AND_NO_SOLVE_ASSISTANT_CONFIRM_RESULT.md`](research/E133_MATH_TEXT_ROUTE_COMPOSITION_AND_NO_SOLVE_ASSISTANT_CONFIRM_RESULT.md) | Math-text route-composition/no-solve evidence |
| 23 | [`docs/research/E132_EXTERNAL_MATH_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md`](research/E132_EXTERNAL_MATH_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md) | External math-text skill-farm evidence |
| 24 | [`docs/research/E131_VISIBLE_EQUATION_EXTRACTION_AND_ASSISTANT_ARITHMETIC_RENDER_GAUNTLET_RESULT.md`](research/E131_VISIBLE_EQUATION_EXTRACTION_AND_ASSISTANT_ARITHMETIC_RENDER_GAUNTLET_RESULT.md) | Visible equation assistant-render evidence |
| 25 | [`docs/research/E130B_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET_RESULT.md`](research/E130B_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET_RESULT.md) | Arithmetic text-IO transfer evidence |
| 26 | [`docs/research/E130A_COREMEMORY_TO_ORANGE_BACKFILL_GAUNTLET_RESULT.md`](research/E130A_COREMEMORY_TO_ORANGE_BACKFILL_GAUNTLET_RESULT.md) | Orange backfill evidence |
| 27 | [`docs/research/E129_ARITHMETIC_TRACE_ORANGE_LEGENDARY_PROBATION_RESULT.md`](research/E129_ARITHMETIC_TRACE_ORANGE_LEGENDARY_PROBATION_RESULT.md) | Arithmetic trace evidence |
| 28 | [`docs/research/E128_ASSISTANT_TEXT_IO_LIGHTWEIGHT_RENDER_TRAINING_RESULT.md`](research/E128_ASSISTANT_TEXT_IO_LIGHTWEIGHT_RENDER_TRAINING_RESULT.md) | Text-IO bridge evidence |
| 29 | [`docs/research/E127_OVERNIGHT_TEXT_SKILL_FARM_ORANGE_CYCLE_RESULT.md`](research/E127_OVERNIGHT_TEXT_SKILL_FARM_ORANGE_CYCLE_RESULT.md) | E127 operator-library checkpoint |
| 30 | [`docs/research/OPERATOR_LIBRARY_CARDS.md`](research/OPERATOR_LIBRARY_CARDS.md) | Operator rank/card surface |
| 31 | [`docs/research/E79_TRAINING_DATA_CURRICULUM_READINESS_RESULT.md`](research/E79_TRAINING_DATA_CURRICULUM_READINESS_RESULT.md) | Latest released training-data/curriculum readiness gate |
| 32 | [Timeline Archive wiki](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive) | Chronology and cleanup record |

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
