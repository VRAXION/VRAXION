# STABLE_LOOP_PHASE_LOCK_135A_STRUCTURED_TOOL_RAW_GENERATION_AUDIT Contract

135A is an audit-only milestone after 135. It exists because the 134/135 structured-output runners may return `expected_output` directly for the positive arm, which would invalidate the structured/tool results as model/raw-generation evidence.

135A performs no training, no repair, no model inference, no checkpoint mutation, no service startup, no deployment, no public API work, and no runtime/product/release mutation.

## Required Audit

The audit must inspect the 134 and 135 structured-output runners for:

```text
direct expected_output return in the positive arm
expected_payload usage in the generation path
oracle metadata used during final eval
deterministic answer construction instead of model output
```

It must read 134/135 artifacts:

```text
raw_generation_results.jsonl
structured_tool_repair_dataset.jsonl
structured_tool_scale_dataset.jsonl
summary.json
decision.json
```

For the positive arm, it must determine whether `generated_text` exists independently and whether it was produced by a raw model generation function. If `MAIN_ARM` returns `expected_output` directly, the audit must emit:

```text
STRUCTURED_TOOL_ORACLE_SHORTCUT_DETECTED
STRUCTURED_OUTPUT_SCALE_CONFIRM_INVALID_AS_MODEL_EVIDENCE
```

and route to:

```text
135B_STRUCTURED_TOOL_REAL_RAW_EVAL_REBUILD
```

If no shortcut is found, emit:

```text
STRUCTURED_TOOL_RAW_GENERATION_AUDIT_POSITIVE
```

and route to:

```text
136_POST_STRUCTURED_TOOL_REPAIR_CEILING_AND_GAP_REMAP
```

## Required Artifacts

The runner must write:

```text
queue.json
progress.jsonl
audit_config.json
source_code_audit.json
positive_arm_generation_path_report.json
artifact_trace_report.json
oracle_shortcut_report.json
evidence_reclassification.json
decision.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` must be written during the audit, not only at the end.

## Boundary

135A is audit-only. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.
