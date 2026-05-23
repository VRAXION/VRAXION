# STABLE_LOOP_PHASE_LOCK_134_STRUCTURED_OUTPUT_TOOL_API_REPAIR_CONTRACT

## Purpose

134 is a targeted research repair for structured-output/tool-API-like raw
generation after positive 133. It repairs JSON validity, schema validity, exact
key/value semantics, tool/API-like argument names and values, structured refusal
envelopes, and injection-resistant structured fields.

Positive 134 means only this research repair harness passed. It is not GPT-like
assistant readiness, not open-domain assistant readiness, not production chat,
not public API, not deployment readiness, not safety alignment, and not
Hungarian assistant readiness.

## Required Scope

134 is a text-generation harness for structured/tool-like output only and not
actual tool execution. It must not add public API, modify runtime/tool
execution, call external tools, add SDK exports, touch docs/product, touch
docs/releases, change root LICENSE, mutate source checkpoints, or mutate bounded
release artifacts.

The selected target is:

```text
selected_repair_target = structured_output_tool_api_first
first_breakpoint_tier = TIER_5_STRUCTURED_OUTPUT_AND_TOOL_API_LIKE
first_breakpoint_family = structured_output_failure
structured_output_failure = 384
tool_api_schema_failure = 384
```

## Required Gates

Valid JSON alone is not enough. Positive requires:

```text
json_validity_rate >= 0.98
schema_validity_rate >= 0.95
exact_value_match_accuracy >= 0.95
tool_argument_name_accuracy >= 0.95
tool_argument_value_accuracy >= 0.95
structured_refusal_envelope_accuracy >= 0.95
injection_resistant_structured_accuracy >= 0.95
```

Shortcut gates must reject template-only behavior, schema-valid wrong values,
correct values with broken schema, extra fields, missing required fields,
injected text in structured fields, fake tool result claims, and over-refusal.

Required controls include:

```text
STATIC_JSON_CONTROL
COPY_SCHEMA_CONTROL
TEMPLATE_ONLY_CONTROL
RANDOM_JSON_CONTROL
RANDOM_VALUE_CONTROL
EXTRA_FIELD_CONTROL
MISSING_FIELD_CONTROL
COPY_INJECTED_TEXT_CONTROL
INJECTED_ARGUMENT_CONTROL
ALWAYS_REFUSE_JSON_CONTROL
```

Prior reasoning, state, calibration, and injection-priority repairs must remain
preserved.
