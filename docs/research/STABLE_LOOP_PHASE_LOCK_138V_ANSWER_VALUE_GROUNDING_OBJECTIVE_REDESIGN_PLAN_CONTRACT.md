# STABLE_LOOP_PHASE_LOCK_138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN Contract

## Purpose

138V is a planning-only milestone after 138S. It designs the next answer-value grounding objective without training, inference, helper calls, torch forward passes, checkpoint mutation, helper/backend edits, old runner imports, deletion/consolidation, services, deployment, runtime/release/product changes, or root LICENSE changes.

The upstream diagnosis is artifact-derived:

- `answer_prefix_accuracy = 1.0`
- `eval_namespace_emission_accuracy = 1.0`
- `answer_value_accuracy = 0.0`
- `P(wrong_value | stale_chat) = 1.0`
- `P(wrong_value | no_stale_chat) = 1.0`
- `wrapper_success_without_value_grounding`

## Layer Separation

138V must explicitly separate three layers.

1. Wrapper Reflex
   The model already emits `ANSWER=E` and the `ANSWER=` prefix reliably. This is not grounding and must not count as success.

2. Value Carrier
   The next objective must preserve the prompt-derived value through wrapper generation. `Residual Signal Carrier` is a design concept, not a measured hidden-state fact. Hidden-state or internal residual-carrier claims are `diagnostic_gap` unless a future probe explicitly instruments them.

3. Value Grounding
   The generated value after `ANSWER=E` must match the prompt-provided or rule-derived value on held-out/OOD rows.

Required 138W output-level proxy metrics:

- `value_after_prefix_accuracy`
- `answer_value_accuracy`
- `exact_answer_accuracy`
- `prefix_success_value_failure_rate`
- `eval_namespace_success_value_failure_rate`
- `no_stale_wrong_value_rate`
- `value_position_error_rate`
- `empty_value_after_prefix_rate`
- `generic_value_after_prefix_rate`
- `prompt_value_copy_accuracy`
- `rule_derived_value_accuracy`
- `table_derived_value_accuracy`

## Required 138W Route

The next milestone is `138W_ANSWER_VALUE_GROUNDING_REPAIR_PROBE`.

138W must require:

- `shared_raw_generation_helper.py only`
- generated text before scoring
- expected-output canary
- AST shortcut scan
- deterministic replay
- controls fail
- leakage rejected
- source checkpoint unchanged
- target checkpoint under `target/` only
- no helper/backend modification
- no old runner imports
- no expected/scorer metadata in helper requests

Positive gates must include:

- `answer_value_accuracy improves from 0.0`
- exact answer accuracy improves from 0.0
- `prefix_success_value_failure_rate decreases`
- value-after-prefix accuracy improves from 0.0
- no-stale wrong-value rate decreases from 1.0
- stale-chat rate remains tracked and below gate
- train-namespace leak remains below gate
- deterministic replay passes

Clean negative remains valid.

## Explicit Rejects

138V and 138W must reject:

- more teacher-forcing as sufficient evidence
- more loss weighting as sufficient evidence
- train-loss-only success
- prefix-only success
- namespace-only success
- expected-output construction
- old runner imports
- oracle/rerank/verifier/LLM judge
- constrained decoding
- JSON mode
- regex fixer
- post-generation repair
- retry loop
- best-of-n
- threshold weakening to force positive

## Boundary

Reasoning is not restored. The reasoning subtrack real-raw evidence is not partially restored. raw assistant capability remains quarantined. structured/tool capability remains invalidated. Not GPT-like readiness. Not open-domain assistant readiness. Not production chat. Not public API. Not deployment readiness. Not safety alignment.
