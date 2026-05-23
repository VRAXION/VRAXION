# STABLE_LOOP_PHASE_LOCK_135A_STRUCTURED_TOOL_RAW_GENERATION_AUDIT Result

135A audits the 134/135 structured-output/tool-API-like evidence path for expected-output or oracle shortcuts.

Expected current outcome:

```text
STRUCTURED_TOOL_ORACLE_SHORTCUT_DETECTED
STRUCTURED_OUTPUT_SCALE_CONFIRM_INVALID_AS_MODEL_EVIDENCE
```

because both structured/tool runners contain a positive-arm path equivalent to:

```text
if arm == MAIN_ARM:
    return row["expected_output"]
```

This means 134/135 can still be useful harness tests, but they must not be treated as proof that the model itself generated the structured/tool-like outputs through raw autoregressive generation.

## Decision

If the shortcut is detected:

```text
decision = structured_tool_oracle_shortcut_detected
next = 135B_STRUCTURED_TOOL_REAL_RAW_EVAL_REBUILD
```

If no shortcut is detected:

```text
decision = structured_tool_raw_generation_audit_positive
next = 136_POST_STRUCTURED_TOOL_REPAIR_CEILING_AND_GAP_REMAP
```

## Boundary

135A is audit-only. It does not train, repair, run model inference, mutate checkpoints, start services, deploy, add public APIs, or modify runtime/service/product/release surfaces.

It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.
