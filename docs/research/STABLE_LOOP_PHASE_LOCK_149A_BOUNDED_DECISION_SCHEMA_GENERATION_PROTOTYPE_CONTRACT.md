# STABLE_LOOP_PHASE_LOCK_149A_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE Contract

149A is the executable bounded decision schema generation prototype after accepted 148Z.

Expected route:

```text
decision = bounded_decision_schema_generation_prototype_positive
verdict = INSTNCT_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE_POSITIVE
next = 149H_BOUNDED_DECISION_SCHEMA_GENERATION_SCALE_CONFIRM
```

Boundary: constrained model-facing distillation evidence only, canonical structured prompts only, bounded two-line decision schema generation only; not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority.

## Primitive

149A extends the 148A/148H runner-local byte-level full-line generation setup from one bounded line to two bounded lines:

```text
SELECTED=<label>
REASON_CODE=<bounded_code>
```

`REASON_CODE` is a closed audit tag, not a natural-language explanation. Final value remains deterministic copy from the generated `SELECTED=<label>` line. Direct opaque value-token generation remains out of scope.

Valid labels:

```text
A
B
C
fallback
```

Valid reason codes:

```text
priority_quorum
priority_recency
priority_validity
fallback_invalid_high_priority
structural_invalid_fallback
```

## Required Guardrails

The generation input must end at:

```text
<OUTPUT>
```

The runner must not prepend `SELECTED=`, must not prepend `REASON_CODE=`, must not deterministically wrap a schema, and must not repair malformed output. Schema validity must be scored from raw generated text. Selected label and reason code must not be extracted from longer invalid text.

Training must use the full bounded target:

```text
SELECTED=<label>
REASON_CODE=<bounded_code>
```

It must not use selected-line-only training, reason-only training, constrained label-only decoding, or constrained reason-only decoding.

## Acceptance

Positive 149A requires selected-line generation, reason-code generation, full bounded schema exact match, final value copy, OOD behavior, shuffled-target control failure, leakage checks, raw schema scoring, deterministic replay, and boundary flags to pass.

A positive 149A proves only bounded two-line decision schema generation on canonical structured prompts. It does not prove natural-language reasoning, open-ended arbitration, GPT-like/Gemma-like assistant capability, production readiness, or architecture superiority.
