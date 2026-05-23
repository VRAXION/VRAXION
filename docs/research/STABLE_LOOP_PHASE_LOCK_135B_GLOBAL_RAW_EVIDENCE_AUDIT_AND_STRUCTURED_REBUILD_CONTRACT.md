# STABLE_LOOP_PHASE_LOCK_135B_GLOBAL_RAW_EVIDENCE_AUDIT_AND_STRUCTURED_REBUILD Contract

135B is an audit/rebuild milestone after 135A. It is not a narrow structured-output retry. It first audits the global raw assistant evidence chain from phases 100 through 135, then attempts structured/tool real raw eval rebuild only if the global audit is clean and a safe repo-local raw generation helper exists.

135B does not train, repair, mutate checkpoints, start services, deploy, add public APIs, modify runtime/tool execution, touch product/release docs, change SDK exports, or change root `LICENSE`.

## Stage A

Stage A scans non-checker phase runners from 100 through 135 and classifies every phase as one of:

```text
REAL_RAW_GENERATION_EVIDENCE
DETERMINISTIC_HARNESS_ONLY
ORACLE_SHORTCUT_DETECTED
NEEDS_MANUAL_REVIEW
NOT_RAW_EVIDENCE_PHASE
```

The scan must use AST-based detection for positive-arm expected-output construction, generated text assigned from expected material, `expected_payload` use in generation paths, scorer metadata used during generation, oracle/rerank/verifier/LLM judge paths, and raw-only flags that conflict with source code.

Hard rule: a phase with positive arm output constructed from `expected_output`, `expected_payload`, answer metadata, or deterministic row labels cannot remain classified as raw model evidence.

## Stage B

Stage B may run only if Stage A finds no broader shortcut affecting the raw evidence chain. If broader shortcuts are found, Stage B must record:

```text
not_attempted_due_to_global_shortcut_audit
```

If Stage B runs, generation input may include prompt, checkpoint path/hash, seed, max_new_tokens, and generation settings only. It must not include expected output, expected payload, required keys, schema answer object, scorer metadata, or oracle labels.

No fake helper, simulated model output, constrained decoding, JSON mode, grammar decoder, regex fixer, retry loop, best-of-n, rerank, verifier, LLM judge, teacher forcing, post-generation repair, actual tool execution, or runtime tool call is allowed.

## Decisions

If broader shortcuts are found:

```text
decision = raw_evidence_chain_partially_invalidated
next = 135D_GLOBAL_RAW_EVIDENCE_REBUILD_PLAN
```

If global audit is clean and structured real raw eval passes:

```text
decision = structured_tool_real_raw_eval_rebuild_positive
next = 136_POST_STRUCTURED_TOOL_REPAIR_CEILING_AND_GAP_REMAP
```

If no safe raw generation helper exists:

```text
decision = structured_tool_real_raw_eval_blocked
next = 135C_STRUCTURED_TOOL_RAW_GENERATION_HELPER_INTEGRATION_PLAN
```

## Boundary

135B is audit/rebuild only. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.
