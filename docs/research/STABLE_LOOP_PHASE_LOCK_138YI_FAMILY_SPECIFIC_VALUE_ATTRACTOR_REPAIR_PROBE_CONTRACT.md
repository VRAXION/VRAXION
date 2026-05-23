# STABLE_LOOP_PHASE_LOCK_138YI_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PROBE Contract

## Scope

138YI is a deterministic targeted repair/probe after 138YF. It may train only a new target checkpoint under `target/pilot_wave/stable_loop_phase_lock_138yi_family_specific_value_attractor_repair_probe/`, then final-evaluate only through `scripts/probes/shared_raw_generation_helper.py`.

The bottleneck being tested is `family_specific_mode_collapse_missing_intra_family_contrast`: the prior probe path learned the `ANSWER=E` wrapper and avoided stale chat, train namespace leak, parrot-copy shortcut, and global train-value replay, but it still collapsed within a family to wrong family-specific value attractors.

## Hard Boundaries

138YI must not mutate source checkpoints, modify `shared_raw_generation_helper.py`, import old phase runners, delete or consolidate files, start services, deploy, modify runtime/service/deploy/product/release surfaces, modify `docs/product`, modify `docs/releases`, modify SDK exports, change root `LICENSE`, or claim broad assistant readiness.

Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.

## Upstream Requirements

138YI requires:

- 138YF decision `family_specific_value_attractor_repair_plan_complete` with next `138YI_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PROBE`.
- 138YF primary bottleneck `family_specific_mode_collapse_missing_intra_family_contrast`.
- 138U decision `wrong_value_attractor_analysis_complete` with root cause `family_specific_train_value_attractor`.
- 138U strict train-row membership rate `0.09895833333333333`; this prevents a global memorized lookup claim.
- 138W helper/canary/AST/leakage/controls/determinism integrity, source checkpoint unchanged, target checkpoint changed, generated text before scoring, no expected/scorer metadata in helper requests, `parrot_trap_detected = false`, `stale_chat_fragment_rate = 0.0`, and `train_namespace_leak_rate = 0.0`.

Scout-First Laziness is a design hypothesis only. Missing Intra-Family Variance is the artifact-level proxy. 138YI must not claim measured scout behavior, measured grower early stopping, or measured hidden-state carrier without explicit instrumentation artifacts.

## Dataset And Contrast Groups

138YI builds fresh train/eval rows with disjoint row hashes, prompt hashes, expected outputs, value namespaces, and no near duplicate prompt at token Jaccard `>= 0.90`.

The core eval uses same-family contrast groups. Each group has the same wrapper and similar prompt shape but different prompt-specific facts, different expected values, different distractors, and different rule/table bindings. A group passes only if all rows emit `ANSWER=E`, all rows emit distinct correct values, no two different expected values collapse to the same generated value, no family default value is emitted, no high-frequency train value is replayed, no prompt-copy-only shortcut is used, no stale chat appears, and no train namespace appears.

If the model learns only the family, it fails. If it emits one family-default value for multiple same-family prompts, it fails. If it copies prompt values but fails derived/OOD values, it fails.

## Required Gates

Final eval must use only `scripts/probes/shared_raw_generation_helper.py`. Helper requests may contain only `prompt`, `checkpoint_path`, `checkpoint_hash`, `seed`, `max_new_tokens`, and `generation_config`. Generated text must exist before scoring, and expected output may be used only after generation.

138YI must rerun forbidden-input rejection, expected-output canary, AST shortcut scan over helper/runner/checker, helper provenance verification, checkpoint hash verification, leakage audit, scorer controls, and deterministic replay.

Positive requires infrastructure gates plus metric gates, including `answer_value_accuracy >= 0.25`, `exact_answer_accuracy >= 0.20`, `value_after_prefix_accuracy >= 0.25`, `intra_family_contrastive_accuracy >= 0.30`, `intra_family_unique_correct_value_rate >= 0.25`, `intra_family_mode_collapse_rate <= 0.60`, `family_default_attractor_rate <= 0.50`, `family_dominant_wrong_value_rate <= 0.50`, derived/OOD family value metrics above their declared thresholds, stale/train namespace below gates, `parrot_trap_detected = false`, `family_default_shortcut_detected = false`, `high_frequency_train_value_replay_detected = false`, and deterministic replay pass.

Controls are scorer-only and must fail: `STATIC_OUTPUT_CONTROL`, `COPY_PROMPT_CONTROL`, `RANDOM_ANSWER_CONTROL`, `DISTRACTOR_COPY_CONTROL`, `STALE_CHAT_FRAGMENT_CONTROL`, `TRAIN_NAMESPACE_REPLAY_CONTROL`, `PREFIX_ONLY_CONTROL`, `GENERIC_VALUE_CONTROL`, `FAMILY_DEFAULT_VALUE_CONTROL`, `SAME_VALUE_FOR_ALL_ROWS_CONTROL`, `PARROT_COPY_CONTROL`, and `HIGH_FREQUENCY_TRAIN_VALUE_CONTROL`.

## Decision Routes

Positive route: `FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_POSITIVE -> family_specific_value_attractor_repair_positive -> 139YI_FAMILY_SPECIFIC_VALUE_ATTRACTOR_SCALE_CONFIRM`.

Clean negative routes include `no_intra_family_value_improvement -> 138YI_FAILURE_ANALYSIS`, `family_mode_collapse_persists -> 138YM_FAMILY_MODE_COLLAPSE_FAILURE_ANALYSIS`, `parrot_trap_copy_shortcut_detected -> 138P_PARROT_TRAP_VALUE_COPY_ANALYSIS`, `high_frequency_train_value_replay_detected -> 138YH_HIGH_FREQUENCY_VALUE_REPLAY_ANALYSIS`, `family_default_shortcut_detected -> 138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS`, `stale_chat_rollout_failure -> 138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS`, `namespace_rollout_failure -> 138S_NAMESPACE_ROLLOUT_FAILURE_ANALYSIS`, `nondeterministic_family_contrastive_probe -> 138N_DETERMINISM_FAILURE_ANALYSIS`, `family_contrastive_eval_leakage -> 138L_FAMILY_CONTRASTIVE_EVAL_LEAKAGE_REDESIGN`, `raw_helper_integrity_failure -> 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL`, and `family_contrastive_training_path_missing -> 138YIA_FAMILY_CONTRASTIVE_TRAINING_HELPER_INTEGRATION_PLAN`.

Only `reasoning_subtrack_real_raw_evidence_partially_restored` may become true on a fully gated positive. All broad capability and readiness flags must remain false.
