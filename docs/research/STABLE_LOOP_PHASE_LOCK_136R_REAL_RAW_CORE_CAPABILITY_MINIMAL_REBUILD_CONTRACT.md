# STABLE_LOOP_PHASE_LOCK_136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD Contract

136R is the first rebuild step after the 135E shared raw-generation helper and canary gate. It is eval-only and records minimal core real-raw evidence through `scripts/probes/shared_raw_generation_helper.py`.

Positive 136R means only `REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD_POSITIVE`: generated text was produced by the shared helper before scoring, forbidden expected/scorer metadata was rejected, expected-output canary passed, AST shortcut scan passed, and provenance was written. It does not restore raw assistant capability.

## Required Upstream

135E must be positive:

- `decision = shared_raw_generation_helper_and_canary_ready`
- `next = 136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD`
- `raw assistant capability remains quarantined`
- `structured/tool capability remains invalidated`

## Raw Generation Requirements

All generated text must come through `shared_raw_generation_helper.py`. Helper requests may contain only `prompt`, `checkpoint_path`, `checkpoint_hash`, `seed`, `max_new_tokens`, and `generation_config`.

The runner must run forbidden-input tests, expected-output canary, AST shortcut scan, and a small core real-raw row set. Generated text must exist and be produced before any scoring. Semantic accuracy is diagnostic only and is not a gate in 136R.

Final eval flags must show no integrated policy, decoder reference, oracle rerank, expected answer usage during eval, teacher forcing, verifier rerank, LLM judge, actual tool execution, runtime tool call, constrained decoding, JSON mode, grammar decoder, post-generation repair, retry loop, or best-of-n.

## Boundary

136R records minimal core real-raw evidence only. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated as model evidence. It is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.

## Decision

If complete:

- `decision = real_raw_core_capability_minimal_rebuild_recorded`
- `next = 137R_REAL_RAW_REASONING_REBUILD`

If helper, canary, AST, provenance, or metadata gates fail, 136R must fail closed and route back to the corresponding raw helper or rebuild analysis path.
