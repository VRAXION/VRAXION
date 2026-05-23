# STABLE_LOOP_PHASE_LOCK_138YH_HIGH_FREQUENCY_VALUE_REPLAY_ANALYSIS Contract

138YH is artifact-only analysis after the 138YI clean negative. It investigates why helper-only rollout emits `ANSWER=E` followed by `TR...` train-value-namespace candidates instead of prompt-specific `EV...` values.

This phase does not train, repair, run new inference, call `shared_raw_generation_helper.py`, run torch forward passes, mutate checkpoints, modify helper/backend code, import old runners, delete or consolidate files, start services, deploy, modify runtime/service/deploy/product/release surfaces, modify `docs/product`, modify `docs/releases`, modify SDK exports, or change root `LICENSE`.

## Required Upstream

138YH requires 138YI:

- `verdict = FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_FAILS`
- `decision = high_frequency_train_value_replay_detected`
- `next = 138YH_HIGH_FREQUENCY_VALUE_REPLAY_ANALYSIS`
- `answer_value_accuracy = 0.0`
- `exact_answer_accuracy = 0.0`
- `intra_family_contrastive_accuracy = 0.0`
- `intra_family_mode_collapse_rate >= 0.90`
- `family_default_attractor_rate >= 0.75`
- `family_default_shortcut_detected = true`
- `high_frequency_train_value_replay_detected = true`
- `parrot_trap_detected = false`
- `stale_chat_fragment_rate = 0.0`
- `train_namespace_leak_rate = 0.0`
- `determinism_replay_passed = true`

Important interpretation guardrail: `train_namespace_leak_rate = 0.0` means no `ANSWER=T` wrapper leak. `TR...` after `ANSWER=E` is value replay, not `ANSWER=T` namespace leakage.

## Analysis Contract

138YH must compute strict train membership and frequency ranks from artifacts. It must not assume global memorization or output-head/logit/internal prior without explicit logits, activation, or output-head artifacts.

The required reports are replay extraction, train value frequency, replay ranks, family replay shape, contrast group replay, objective reward artifact review, scorer/dataset artifact review, root cause, next repair recommendation, diagnostic gaps, risk register, decision, summary, and report.

Root cause must be exactly one of:

- `global_high_frequency_train_value_replay`
- `family_local_high_frequency_value_replay`
- `family_default_shortcut_replay`
- `same_value_for_all_rows_collapse`
- `objective_missing_frequency_penalty`
- `dataset_low_value_diversity_artifact`
- `mixed_high_frequency_replay`
- `high_frequency_replay_ambiguous`

All capability flags remain false: reasoning restored, reasoning subtrack partial evidence, raw assistant, structured/tool, GPT-like, open-domain, production chat, public API, deployment, and safety alignment.
