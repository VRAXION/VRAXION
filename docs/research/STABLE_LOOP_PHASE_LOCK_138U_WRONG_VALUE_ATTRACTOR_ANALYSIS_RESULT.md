# STABLE_LOOP_PHASE_LOCK_138U_WRONG_VALUE_ATTRACTOR_ANALYSIS Result

138U is the artifact-only follow-up to 138WV. It analyzes the post-wrapper wrong-value attractor without training, inference, helper calls, torch forward passes, checkpoint mutation, helper/backend edits, old runner imports, deletion, deployment, or runtime/release/product surface changes.

## Expected Evidence Shape

138WV established that the failure is not silence, neutral default emission, or format echo. The expected 138U artifact profile is:

- `ANSWER=E` wrapper is emitted cleanly.
- The post-wrapper candidate is a wrong specific train-seen value.
- The generated candidate is not the expected eval value.
- `wrong_specific_value_rate = 1.0`
- `train_seen_value_rate = 1.0`
- `expected_value_candidate_rate = 0.0`

138U separates these possible explanations:

- `global_train_value_prior_attractor`
- `high_frequency_train_value_attractor`
- `family_specific_train_value_attractor`
- `distractor_value_attractor`
- `wrong_table_entry_attractor`
- `prompt_copy_wrong_value_attractor`
- `output_head_value_prior`
- `mixed_wrong_value_attractors`
- `wrong_value_attractor_ambiguous`

Any output-head or hidden-state explanation remains `diagnostic_gap` unless future instrumentation provides logits, activations, or comparable evidence.

## Machine Route

If current smoke artifacts match a global or high-frequency train-value prior, the route is:

- `decision = wrong_value_attractor_analysis_complete`
- `next = 138Y_VALUE_PRIOR_SUPPRESSION_AND_GROUNDING_OBJECTIVE_PLAN`

In the current smoke evidence, strict artifact analysis instead found a family-specific wrong-value attractor shape. The observed route is:

- `decision = wrong_value_attractor_analysis_complete`
- `root_cause = family_specific_train_value_attractor`
- `next = 138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN`

The important adversarial finding is that 138WV's `train_seen_value` label identifies a `TR...` candidate class, while strict train-row membership is lower than the upstream label implied. 138U therefore separates the upstream label from computed train-row membership and does not force a global prior route. It is still a clean negative capability state, not a capability restoration.

## Boundary

Reasoning is not restored. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.
