# STABLE_LOOP_PHASE_LOCK_138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN Contract

138YF is an artifact-only planning milestone after 138U. It does not train, repair, run new inference, call `shared_raw_generation_helper.py`, run torch forward passes, mutate checkpoints, modify helper/backend code, import old runners, delete or consolidate files, start services, deploy, modify runtime/service/deploy/product/release surfaces, modify SDK exports, modify docs/product or docs/releases, or change root `LICENSE`.

## Upstream Contract

138YF requires 138U to have completed:

- `decision = wrong_value_attractor_analysis_complete`
- `root_cause = family_specific_train_value_attractor`
- `next = 138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN`
- `wrong_specific_value_rate = 1.0`
- `expected_value_candidate_rate = 0.0`
- `generated_values_seen_in_train_rate = 0.09895833333333333`

138YF must preserve the 138U adversarial distinction:

- The 138WV `train_seen_value` candidate label is not the same as strict train-row membership.
- Strict 138U train-row membership is low enough that a global memorized lookup claim is not supported.
- The route is not `global_train_value_prior_attractor`, not `high_frequency_train_value_attractor`, not `prompt_copy_wrong_value_attractor`, not `distractor_value_attractor`, and not `wrong_table_entry_attractor`.

138YF also requires 138W helper integrity: canary, AST, leakage, controls, and determinism passed; source checkpoint unchanged; target checkpoint changed; generated text before scoring; no expected/scorer metadata reached helper requests; `parrot_trap_detected = false`; `stale_chat_fragment_rate = 0.0`; and `train_namespace_leak_rate = 0.0`.

## Planning Contract

138YF translates Qwen's concepts into artifact/proxy language only:

- `Scout-First Laziness` is a design hypothesis, not measured scout/grower behavior.
- `Missing Intra-Family Variance` is the artifact-level failure proxy.
- The repair concept is an `intra_family_contrastive_objective`.

The plan must define:

- family-specific attractor summary
- train-membership reconciliation
- intra-family mode collapse metrics
- intra-family contrastive objective requirements
- deep scout forcing hypothesis as `diagnostic_gap`
- carrier proxy requirements
- anti-shortcut requirements
- `138YI_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PROBE`

Required 138YI metrics include `intra_family_contrastive_accuracy`, `intra_family_unique_correct_value_rate`, `intra_family_mode_collapse_rate`, `family_default_attractor_rate`, per-family answer-value accuracy, per-family exact-answer accuracy, per-family rule/table/OOD value accuracy, and deterministic replay.

## Decision Contract

If the plan is complete:

- `decision = family_specific_value_attractor_repair_plan_complete`
- `next = 138YI_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PROBE`

If upstream evidence is missing:

- `decision = upstream_138u_artifact_missing`
- `next = 138YF_UPSTREAM_138U_ARTIFACT_MISSING`

If family-specific evidence no longer holds:

- `decision = family_specific_attractor_evidence_recheck`
- `next = 138YF_FAMILY_ATTRACTOR_EVIDENCE_RECHECK`

If helper integrity regresses:

- `decision = raw_helper_integrity_failure`
- `next = 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL`

## Boundary

Reasoning is not restored. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.
