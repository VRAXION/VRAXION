# STABLE_LOOP_PHASE_LOCK_138WV_WRAPPER_VALUE_DECOUPLING_FAILURE_ANALYSIS Contract

## Purpose

138WV is artifact-only analysis after 138W. It reads the existing 138W raw generation, scoring, trace, control, leakage, canary, AST, checkpoint, and replay artifacts. It does not train, repair, run new inference, call `shared_raw_generation_helper.py`, run torch forward passes, mutate checkpoints, import old runners, delete/consolidate files, start services, deploy, or modify runtime/release/product surfaces.

138W routed here because:

```text
verdict = ANSWER_VALUE_GROUNDING_REPAIR_FAILS
decision = wrapper_success_without_value_grounding_persists
next = 138WV_WRAPPER_VALUE_DECOUPLING_FAILURE_ANALYSIS
```

The key profile is:

```text
answer_prefix_accuracy = 1.0
eval_namespace_emission_accuracy = 1.0
answer_value_accuracy = 0.0
exact_answer_accuracy = 0.0
value_after_prefix_accuracy = 0.0
parrot_trap_detected = false
post_wrapper_garbage_token_rate = 0.0
stale_chat_fragment_rate = 0.0
train_namespace_leak_rate = 0.0
```

## EOS Guardrail

`immediate_termination_proxy` is a text-output proxy, not literal EOS evidence.

The current helper records `stop_reason = max_new_tokens`, so 138WV must not claim literal EOS behavior. Any topological inhibition explanation is a hypothesis and remains `diagnostic_gap` without explicit instrumentation.

## Required Analysis

138WV must parse every generated row after the first `ANSWER=E` marker and write:

- `post_wrapper_value_anatomy_report.json`
- `silence_taxonomy_report.json`
- `attractor_distribution_report.json`
- `value_candidate_report.json`
- `parrot_and_derivation_recheck.json`
- `wrapper_value_decoupling_root_cause.json`
- `next_repair_recommendation.json`

Every row must get exactly one primary class:

- `immediate_termination_proxy`
- `empty_or_whitespace_after_wrapper`
- `default_neutral_attractor`
- `structural_format_echo`
- `generic_wrong_value`
- `repeated_symbol_or_punctuation`
- `wrong_specific_value`
- `delayed_correct_value_wrong_position`
- `garbled_after_wrapper`
- `unknown_post_wrapper_behavior`

If `wrong_specific_value_attractor_dominant` is selected, the required next route is `138U_WRONG_VALUE_ATTRACTOR_ANALYSIS`.

## Boundary

Reasoning is not restored. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. not GPT-like readiness. not open-domain assistant readiness. not production chat. not public API. not deployment readiness. not safety alignment.
