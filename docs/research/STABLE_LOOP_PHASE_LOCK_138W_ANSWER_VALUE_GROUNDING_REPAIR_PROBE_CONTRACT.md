# STABLE_LOOP_PHASE_LOCK_138W_ANSWER_VALUE_GROUNDING_REPAIR_PROBE Contract

## Purpose

138W is a targeted answer-value grounding repair/probe after 138V. It may train only a new target checkpoint under `target/`, then final-evaluate only through `scripts/probes/shared_raw_generation_helper.py`.

The probe asks whether the model can emit the correct prompt-derived or rule-derived value after the `ANSWER=E` wrapper under helper-only free rollout.

This is not a broad assistant capability milestone.

## Upstream Basis

138V established:

- `decision = answer_value_grounding_objective_redesign_plan_complete`
- `next = 138W_ANSWER_VALUE_GROUNDING_REPAIR_PROBE`
- `primary_bottleneck = wrapper_success_without_value_grounding`
- `hidden_state_residual_signal_measurement = diagnostic_gap`

138S/138I established:

- `answer_prefix_accuracy = 1.0`
- `eval_namespace_emission_accuracy = 1.0`
- `answer_value_accuracy = 0.0`
- `exact_answer_accuracy = 0.0`
- `P(wrong_value | stale_chat) = 1.0`
- `P(wrong_value | no_stale_chat) = 1.0`

## Required Guardrails

Final eval must use `shared_raw_generation_helper.py` only. Scoring must happen only after `generated_text` exists.

138W must rerun:

- forbidden-input rejection
- expected-output canary
- AST shortcut scan
- helper provenance verification
- checkpoint hash verification
- leakage audit
- scorer controls
- deterministic replay

Forbidden:

- old runner imports
- direct model generation outside the helper for final eval
- expected/scorer metadata in helper requests
- LLM judge
- verifier/rerank/oracle
- constrained decoding
- JSON mode
- regex fixer as post-generation repair
- retry loop
- best-of-n
- threshold weakening to force positive

## Parrot Trap

Parrot Trap is a hard guardrail. Direct prompt-value copy is diagnostic only and cannot be enough for a positive.

Required metrics:

- `prompt_value_copy_accuracy`
- `rule_derived_value_accuracy`
- `table_derived_value_accuracy`
- `composition_derived_value_accuracy`
- `ood_symbol_value_accuracy`
- `copy_only_success_rate`
- `parrot_trap_detected`

If it only copies prompt values, it fails. If it cannot derive held-out values, it fails.

## Post-Wrapper Proxies

`Wrapper-Induced Amnesia` is tested with output-level proxies only. `Residual Signal Carrier` remains a design concept; hidden-state evidence is `diagnostic_gap` unless explicitly instrumented.

Required proxy metrics:

- `value_after_prefix_accuracy`
- `value_position_error_rate`
- `empty_value_after_prefix_rate`
- `generic_value_after_prefix_rate`
- `post_wrapper_garbage_token_rate`
- `value_emission_latency_mean`
- `value_emission_latency_p95`
- `repeated_token_after_prefix_rate`
- `prefix_success_value_failure_rate`
- `eval_namespace_success_value_failure_rate`
- `no_stale_wrong_value_rate`

If the model only learns `ANSWER=E` prefix, it fails.

## Boundary

138W may partially restore only reasoning subtrack real-raw evidence if fully positive. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. not GPT-like readiness. not open-domain assistant readiness. not production chat. not public API. not deployment readiness. not safety alignment.
