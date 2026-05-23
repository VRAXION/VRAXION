# STABLE_LOOP_PHASE_LOCK_135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE Contract

135E establishes raw generation infrastructure only. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated as model evidence. No capability is restored.

135E does not train, repair, mutate checkpoints, start services, deploy, delete files, consolidate old runners, modify runtime/release/product surfaces, change public APIs, change SDK exports, touch product/release docs, touch old runners, or change the root LICENSE.

## Required upstream

135E requires positive 135D:

- `decision = global_raw_evidence_rebuild_plan_complete`
- `next = 135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE`
- raw assistant capability remains quarantined
- structured/tool capability remains invalidated as model evidence

## Shared helper

135E adds `scripts/probes/shared_raw_generation_helper.py`. It is standalone and must not import old phase runners.

Allowed request keys are exactly:

- `prompt`
- `checkpoint_path`
- `checkpoint_hash`
- `seed`
- `max_new_tokens`
- `generation_config`

Any unknown or forbidden key fails with `RAW_GENERATION_FORBIDDEN_INPUT_DETECTED`. Forbidden examples include `expected_output`, `expected_payload`, `expected_answer`, `required_keys`, `required_keywords`, `forbidden_outputs`, `schema_answer_object`, `scorer_metadata`, `labels`, `oracle_data`, `target_json`, `row_answer`, `gold_output`, `eval_family`, and `answer`.

The helper response must include `generated_text`, `token_count`, `stop_reason`, `generation_trace_hash`, `model_checkpoint_hash`, `generation_config_hash`, `helper_backend`, and `helper_version`.

## Backend and canary

135E must use a real repo-local checkpoint-backed byte-LM backend. If torch is unavailable, no checkpoint exists, checkpoint hash mismatches, checkpoint shape cannot be safely loaded, generation fails, or the loader must guess architecture unsafely, it must fail closed with `RAW_GENERATION_BACKEND_MISSING`.

The expected-output canary duplicates a row, corrupts `expected_output` and `expected_payload` outside the helper request, keeps the prompt and helper request identical, and requires `generated_text`, `generation_trace_hash`, `token_count`, and `stop_reason` to remain identical. If any generation-side field changes, emit `ORACLE_SHORTCUT_DETECTED`.

AST scan must reject old runner imports, expected material in generation paths, deterministic positive-arm construction, generated text assigned from expected material, oracle/rerank/verifier/LLM judge paths, constrained decoding, JSON mode, grammar decoder, regex fixer, post-generation repair, retry loop, best-of-n, actual tool execution, and runtime tool calls.

## Positive result

Positive 135E emits `SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE_POSITIVE` with:

- `decision = shared_raw_generation_helper_and_canary_ready`
- `next = 136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD`

Positive means helper plus canary infrastructure is ready. It does not mean raw assistant capability restored, structured/tool capability restored, GPT-like readiness, open-domain assistant readiness, production chat, public API, deployment readiness, or safety alignment.
