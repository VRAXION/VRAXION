# STABLE_LOOP_PHASE_LOCK_147A_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE Contract

147A is the executable LM-style canonical structured text distillation prototype after positive 146Z.

Boundary: 147A is constrained model-facing distillation evidence only with canonical structured prompts only; not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority.

## Expected Route

```text
decision = lm_style_canonical_structured_text_distillation_prototype_positive
verdict = INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE_POSITIVE
next = 147H_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRM
```

## Prototype Primitive

```text
canonical structured prompt
-> runner-local PyTorch byte-level selected next-byte model
-> generated SELECTED=<A|B|C|fallback>
-> deterministic candidate-value copy
-> ANSWER=<value> in reports only
```

The learned output is only the selected label. Direct opaque value-token generation is out of scope.

## Strict Schema

Valid generated output is exactly one line:

```text
SELECTED=A
SELECTED=B
SELECTED=C
SELECTED=fallback
```

Invalid outputs include `SELECTED=pocket_a`, `ANSWER=...`, `winner=pocket_*`, `selected_pocket_id=...`, malformed labels, multiple `SELECTED` lines, and any extra text before or after the selected line.

## Required Audits

147A must write heartbeat progress and must include generation input leakage checks, generated schema checks, label distribution checks, OOD family checks, anti-memorization checks, model artifact checks, model input checks, feature path checks, OOD split definition checks, baseline margin checks, shortcut scanning, leakage audits, and value-token leakage audits.

Teacher traces and labels may exist only in curriculum/scoring artifacts. They are forbidden in model-facing input and feature extraction input.
