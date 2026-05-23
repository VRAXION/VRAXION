# STABLE_LOOP_PHASE_LOCK_138U_WRONG_VALUE_ATTRACTOR_ANALYSIS Contract

138U is an artifact-only analysis milestone after 138WV. It does not train, repair, run new inference, call `shared_raw_generation_helper.py`, run torch forward passes, mutate checkpoints, modify helper/backend code, import old runners, delete or consolidate files, start services, deploy, modify runtime/service/deploy/product/release surfaces, modify SDK exports, modify docs/product or docs/releases, or change root `LICENSE`.

## Upstream Contract

138U requires 138WV to have completed:

- `decision = wrapper_value_decoupling_failure_analysis_complete`
- `root_cause = wrong_specific_value_attractor_dominant`
- `next = 138U_WRONG_VALUE_ATTRACTOR_ANALYSIS`
- `wrong_specific_value_rate = 1.0`
- `train_seen_value_rate = 1.0`
- `expected_value_candidate_rate = 0.0`
- `immediate_termination_proxy_rate = 0.0`
- `default_neutral_attractor_rate = 0.0`
- `structural_format_echo_rate = 0.0`
- `unknown_post_wrapper_behavior_rate = 0.0`
- `literal_eos_claimed = false`
- `topological_inhibition_claim_status = hypothesis_only_diagnostic_gap_without_instrumentation`

138U also requires 138W helper integrity: canary, AST, leakage, controls, and determinism passed; source checkpoint unchanged; target checkpoint changed; generated text existed before scoring; no expected/scorer metadata reached helper requests; `parrot_trap_detected = false`; `stale_chat_fragment_rate = 0.0`; `train_namespace_leak_rate = 0.0`.

## Analysis Contract

138U identifies why `ANSWER=E` is followed by a wrong specific train-seen value instead of the expected eval value. It writes:

- `wrong_value_distribution_report.json`
- `train_value_attractor_report.json`
- `eval_value_miss_report.json`
- `wrong_value_vs_prompt_report.json`
- `value_source_family_failure_report.json`
- `attractor_root_cause.json`
- `next_repair_recommendation.json`

Allowed root causes include `global_train_value_prior_attractor`, `high_frequency_train_value_attractor`, `family_specific_train_value_attractor`, `distractor_value_attractor`, `wrong_table_entry_attractor`, `prompt_copy_wrong_value_attractor`, `output_head_value_prior`, `mixed_wrong_value_attractors`, and `wrong_value_attractor_ambiguous`.

`output_head_value_prior` is not a measured hidden-state or logits fact unless a future artifact instruments it. Without logits/hidden states it is recorded through `diagnostic_gap`.

## Decision Contract

If the analysis completes, 138U writes:

- `decision = wrong_value_attractor_analysis_complete`
- `next = 138Y_VALUE_PRIOR_SUPPRESSION_AND_GROUNDING_OBJECTIVE_PLAN` for a global or high-frequency train value prior

If the attractor remains ambiguous:

- `decision = wrong_value_attractor_ambiguous`
- `next = 138UB_WRONG_VALUE_ATTRACTOR_MANUAL_REVIEW_PACKET`

If helper integrity regresses:

- `decision = raw_helper_integrity_failure`
- `next = 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL`

## Boundary

Reasoning is not restored. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.
