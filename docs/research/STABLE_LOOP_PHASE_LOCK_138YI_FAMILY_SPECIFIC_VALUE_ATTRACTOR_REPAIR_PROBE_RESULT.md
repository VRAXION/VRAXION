# STABLE_LOOP_PHASE_LOCK_138YI_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PROBE Result

## Status

This document records the 138YI repair/probe result contract and the expected generated artifact root:

`target/pilot_wave/stable_loop_phase_lock_138yi_family_specific_value_attractor_repair_probe/smoke`

The executable source of truth is `scripts/probes/run_stable_loop_phase_lock_138yi_family_specific_value_attractor_repair_probe.py`; validation is enforced by `scripts/probes/run_stable_loop_phase_lock_138yi_family_specific_value_attractor_repair_probe_check.py`.

## Interpretation Rules

138YI is not a broad capability milestone. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. The result is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.

The result is positive only if helper-only free rollout improves value grounding and same-family contrastive behavior under all gates. The result is a clean negative if the target checkpoint changes but the helper-only rollout still fails value improvement, same-family distinction, family-default rejection, high-frequency train-value replay rejection, parrot rejection, stale suppression, namespace suppression, deterministic replay, leakage, controls, or helper integrity.

## Required Evidence

The generated artifacts must include upstream manifests for 138YF, 138U, and 138W; deterministic train/eval configs; source and target checkpoint integrity manifests; helper provenance; forbidden-input rejection; expected-output canary; AST shortcut scan; train/eval dataset manifests and rows; contrast group manifest; OOD family/value manifest; leakage audit; training metrics; helper-only raw generation traces/results; scoring results; contrast group results; intra-family contrastive metrics; family default attractor report; high-frequency value replay report; value grounding metrics; parrot trap report; scorer-only controls; generated-before-scoring proof; per-family, per-seed, and aggregate metrics; deterministic replay report; failure samples; evidence rebuild status; decision; summary; and report.

Key decision metrics are `answer_value_accuracy`, `exact_answer_accuracy`, `value_after_prefix_accuracy`, `intra_family_contrastive_accuracy`, `intra_family_unique_correct_value_rate`, `intra_family_mode_collapse_rate`, `family_default_attractor_rate`, `family_dominant_wrong_value_rate`, rule/table/composition/OOD value accuracies, `train_namespace_leak_rate`, `stale_chat_fragment_rate`, `parrot_trap_detected`, `family_default_shortcut_detected`, and `high_frequency_train_value_replay_detected`.

## Anti-Shortcut Notes

If the model learns only the family, it fails. If it emits one family-default value for multiple same-family prompts, it fails. If it copies prompt values but fails derived/OOD values, it fails. Prefix-only success, namespace-only success, target-checkpoint-changed-only success, train-loss-only success, teacher-forcing-only success, expected-output construction, old runner imports, oracle/rerank/verifier/LLM judge, constrained decoding, JSON mode, regex fixer, post-generation repair, retry loop, best-of-n, and threshold weakening are rejected.

Scout-First Laziness remains a design hypothesis. Missing Intra-Family Variance is measured only through artifact-level output proxies such as same-family contrast groups and mode-collapse metrics.

## Capability Flags

`reasoning_restored = false`

`raw_assistant_capability_restored = false`

`structured_tool_capability_restored = false`

`gpt_like_readiness_claimed = false`

`open_domain_assistant_readiness_claimed = false`

`production_chat_claimed = false`

`public_api_claimed = false`

`deployment_readiness_claimed = false`

`safety_alignment_claimed = false`
