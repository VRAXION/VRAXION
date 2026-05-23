# STABLE_LOOP_PHASE_LOCK_135_STRUCTURED_OUTPUT_TOOL_API_REPAIR_SCALE_CONFIRM Contract

135 is an eval-only scale confirmation after `STRUCTURED_OUTPUT_TOOL_API_REPAIR_POSITIVE` in 134. It confirms whether the 134 repaired raw checkpoint generalizes on fresh, larger, multi-seed structured-output and tool/API-like text rows.

This milestone is a text-generation harness for structured/tool-like output only. It is not actual tool execution, not runtime/tool execution integration, not a public API, not production chat, not deployment readiness, not safety alignment, not GPT-like assistant readiness, not open-domain assistant readiness, and not Hungarian assistant readiness.

## Required Configuration

Positive requires the full configured run:

```text
seeds = 2251,2252,2253,2254,2255
eval_rows_per_family = 96
json_schema_variants = 24
tool_api_variants = 24
nested_structure_variants = 16
array_variants = 14
format_conversion_variants = 16
regex_transform_variants = 12
table_rows = 128
multi_doc_count = 12
long_context_chars = 32768
noise_blocks = 32
injection_variants = 24
state_carry_variants = 12
schema_mutation_variants = 12
heartbeat_sec = 20
```

The runner must load the 134 repaired checkpoint read-only and record `repaired_checkpoint_path`, `checkpoint_hash_before`, `checkpoint_hash_after`, `checkpoint_hash_unchanged = true`, and `target_134_checkpoint_read_only = true`.

## Gates

The positive arm is `POST_134_STRUCTURED_OUTPUT_TOOL_API_REPAIRED_RAW_SCALE_CONFIRM`. Final eval must be raw text generation only: no integrated policy, decoder reference, oracle rerank, expected-answer metadata, teacher forcing, verifier rerank, LLM judge, actual tool execution, or runtime tool call.

Every seed must pass independently. The gates cover JSON validity, schema validity, exact key/value match, tool argument name/value correctness, no extra/missing fields, nested structure, array order, format conversion, regex transform, table-to-JSON, multi-doc structured output, structured refusal envelopes, long-context schema rows, state-carry structured rows, and injection-resistant structured fields.

Structured shortcut failures must be rejected: template-only output, static JSON patterns, schema-valid wrong values, correct values with wrong schema, extra fields, missing required fields, wrong argument names, wrong argument values, injected text inside structured fields, copied injected text, fake tool-result claims, and over-refusal structured rows.

Required controls must fail:

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
RANDOM_SCHEMA_CONTROL
```

Prior reasoning, multi-turn state, hallucination/refusal calibration, prompt-injection/instruction-priority, retention, namespace, collapse, leakage, checkpoint integrity, bounded release integrity, and boundary gates remain hard gates.

## Decision

Success writes:

```text
STRUCTURED_OUTPUT_TOOL_API_REPAIR_SCALE_CONFIRM_POSITIVE
UPSTREAM_134_STRUCTURED_TOOL_REPAIR_VERIFIED
STRUCTURED_OUTPUT_REPAIR_GENERALIZES
JSON_SCHEMA_SEMANTICS_GENERALIZE
TOOL_ARGUMENT_REPAIR_GENERALIZES
STRUCTURED_REFUSAL_CONFIRMED
TEMPLATE_ONLY_SHORTCUT_REJECTED
FAKE_TOOL_USE_REJECTED
INJECTION_IN_STRUCTURED_FIELDS_REJECTED
```

and:

```text
decision = structured_output_tool_api_repair_scale_confirmed
next = 136_POST_STRUCTURED_TOOL_REPAIR_CEILING_AND_GAP_REMAP
```

Failure routes include `135A_STRUCTURED_TOOL_TARGET_REVALIDATION`, `135B_STRUCTURED_TOOL_SCALE_FAILURE_ANALYSIS`, `135D_TEMPLATE_ONLY_DEGENERATION_ANALYSIS`, `135S_SCHEMA_SEMANTICS_SCALE_FAILURE_ANALYSIS`, `135F_FAKE_TOOL_RESULT_CLAIM_ANALYSIS`, `135I_INJECTED_STRUCTURED_FIELD_ANALYSIS`, `135R_PRIOR_REPAIR_REGRESSION_ANALYSIS`, `135T_RETENTION_OR_COLLAPSE_REGRESSION_ANALYSIS`, `135L_STRUCTURED_TOOL_EVAL_LEAKAGE_REDESIGN`, `135E_STRUCTURED_TOOL_SCORER_OR_TASK_WEAKNESS_ANALYSIS`, `135M_CHECKPOINT_INTEGRITY_FAILURE_ANALYSIS`, and `135C_BOUNDARY_FAILURE_ANALYSIS`.
