# STABLE_LOOP_PHASE_LOCK_147H_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRM Contract

147H is the scale confirm after positive 147A.

Boundary: 147H is constrained model-facing distillation evidence only with canonical structured prompts only; not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority.

## Expected Route

```text
decision = lm_style_canonical_structured_text_distillation_scale_confirmed
verdict = INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRMED
next = 147Z_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_NEXT_DECISION_PLAN
```

## Scale Claim

147H must preserve the 147A model family. The model predicts the selected label byte/token after a fixed `SELECTED=` prefix. The runner deterministically wraps that prediction into one strict selected-label line and copies the final value from the candidate line.

147H does not claim free-form full-line generation. Direct opaque value-token generation remains out of scope.

Valid wrapped schema:

```text
SELECTED=A
SELECTED=B
SELECTED=C
SELECTED=fallback
```

## Required Guardrails

The scale run must verify upstream 147A, keep `shared_raw_generation_helper.py` unchanged, avoid `raw_generate`, avoid external APIs or pretrained models, and write heartbeat progress throughout curriculum generation, local training, evaluation, replay, and final decision.

Required audits include label-byte generation, same-model-family, per-seed gates, deterministic replay, strict schema, generation input leakage, OOD family row counts, anti-memorization, shortcut scanning, value-token leakage, and baseline margin reports.
