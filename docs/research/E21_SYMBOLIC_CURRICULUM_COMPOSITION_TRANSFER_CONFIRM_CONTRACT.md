# E21_SYMBOLIC_CURRICULUM_COMPOSITION_TRANSFER_CONFIRM Contract

## Purpose

E21 tests whether staged curriculum training can produce reusable Flow/Pocket symbolic reasoning primitives that transfer to locked hard composite symbolic tasks.

## Boundary

This is a controlled symbolic curriculum-composition transfer audit for a Flow/Pocket policy. It tests whether reusable primitive pockets learned through staged curriculum improve locked hard symbolic reasoning tasks. It does not prove general mathematics, theorem proving, GPT-like generation, AGI, consciousness, or production readiness.

## Required protocol

- Generate a locked hard pretest before curriculum.
- Run pre-curriculum primary and baselines on the locked hard pretest.
- Train staged primitive curriculum.
- Freeze/export reusable primitive pockets and train a composition controller.
- Run locked hard posttest and new heldout transfer tasks.
- Compare against monolithic equal-budget training, no-freeze curriculum, no-reusable-pocket ablation, static/regex baselines, and invalid oracle controls.
- Recompute aggregate metrics from per-episode logs.
- Commit only a compact artifact sample pack, not the full target directory.

## Curriculum stages

- Stage 0: digit and symbol boundary recovery.
- Stage 1: single-digit addition/subtraction.
- Stage 2: multi-digit addition/subtraction with carry/borrow.
- Stage 3: multiplication/division.
- Stage 4: signed arithmetic.
- Stage 5: fractions and simplification.
- Stage 6: parentheses and precedence.
- Stage 7: linear equations.
- Stage 8: powers and roots.
- Stage 9: radical simplification.
- Stage 10: multi-step composite expressions.
- Stage 11: heldout composition transfer.

## Full-confirm minimums

- generations_completed >= 100.
- population_size >= 160.
- candidate_count_evaluated >= 16000.
- heldout_episode_count >= 2400.
- stress_episode_count >= 2400.
- locked_hard_pretest_episode_count >= 1000.
- locked_hard_posttest_episode_count >= 1000.
- curriculum_stage_count >= 10.
- checkpoint_count >= 100.
- committed_sample_episode_count >= 500.

## Pass gates

- locked_hard_pretest_accuracy <= 0.55.
- locked_hard_posttest_accuracy >= 0.80.
- improvement_vs_pretest >= 0.25.
- delta_vs_monolithic_equal_budget >= 0.10.
- delta_vs_no_reusable_pocket_transfer_ablation >= 0.15.
- heldout_composition_transfer_accuracy >= 0.75.
- earlier_stage_regression_average >= 0.90.
- undefined/ambiguous handling accuracy >= 0.90.
- canonical_answer_accuracy >= 0.85.
- trace_validity >= 0.90.
- renderer_faithfulness >= 0.98.
- oracle/eval/sympy/hand-solver controls rejected as primary.
- aggregate metrics recomputed from per-episode logs.
- committed sample pack and sample-only checker pass.
- checker_failure_count = 0.

## Run command

python3 scripts/probes/run_e21_symbolic_curriculum_composition_transfer_confirm.py --out target/pilot_wave/e21_symbolic_curriculum_composition_transfer_confirm --artifact-sample-dir docs/research/artifact_samples/e21_symbolic_curriculum_composition_transfer --strict-budget --no-downshift --generations 140 --population 192 --train-episodes 9000 --validation-episodes 2200 --heldout-episodes 3000 --stress-episodes 3000 --locked-hard-pretest-episodes 1200 --locked-hard-posttest-episodes 1200 --checkpoint-every 1 --max-runtime-minutes 360 --resume

## Checker commands

python3 scripts/probes/run_e21_symbolic_curriculum_composition_transfer_confirm_check.py --out target/pilot_wave/e21_symbolic_curriculum_composition_transfer_confirm --artifact-sample-dir docs/research/artifact_samples/e21_symbolic_curriculum_composition_transfer --write-summary

python3 scripts/probes/run_e21_symbolic_curriculum_composition_transfer_confirm_check.py --sample-only docs/research/artifact_samples/e21_symbolic_curriculum_composition_transfer --write-summary
