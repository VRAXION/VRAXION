# STABLE_LOOP_PHASE_LOCK_148Z_FULL_SELECTED_LINE_GENERATION_NEXT_DECISION_PLAN Contract

148Z is a planning-only, artifact-only next-decision milestone after positive 148H. It does not train a model, call `raw_generate`, import `shared_raw_generation_helper.py`, run torch forward passes, mutate checkpoints, or modify helper/runtime/product surfaces.

Expected route:

```text
decision = bounded_decision_schema_generation_prototype_plan_recommended
selected_option = bounded_decision_schema_generation_prototype
next = 149A_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE
```

Boundary: constrained model-facing distillation evidence only, canonical structured prompts only, bounded decision schema planning only; not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority.

## Upstream Requirement

148Z requires 148H to be accepted as:

```text
decision = full_selected_line_generation_scale_confirmed
verdict = INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRMED
next = 148Z_FULL_SELECTED_LINE_GENERATION_NEXT_DECISION_PLAN
```

The required 148H evidence includes strong bounded full `SELECTED=<label>` line generation at scale, no hidden `SELECTED=` prefix, no runner wrapper, no post-generation repair, no substring extraction, no first-byte-only training, no constrained label-only decoding, and deterministic replay.

## Target 149A

149A should test bounded multi-line decision schema generation:

```text
SELECTED=<label>
REASON_CODE=<bounded_code>
```

This is not natural-language reasoning. `REASON_CODE` is a bounded audit tag. Final value remains deterministic copy from the generated selected line, and direct opaque value-token generation remains out of scope.

Required 149A guardrails:

```text
canonical structured prompts only
raw generated text stored
schema scored from raw generated text
no deterministic schema wrapper
no selected-line prefix injection
no reason-code prefix injection
no post-generation repair
no free-text reason generation
```
