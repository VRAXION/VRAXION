# STABLE_LOOP_PHASE_LOCK_137R_REAL_RAW_REASONING_REBUILD Contract

137R is the first real-raw reasoning rebuild after the 135E/136R trust-root reset. It is eval-only and must use `scripts/probes/shared_raw_generation_helper.py` for every generation call.

Clean negative is valid. If real raw generation works but reasoning gates fail, the correct result is `REAL_RAW_REASONING_REBUILD_FAILS`, `decision = real_raw_reasoning_not_restored`, and `next = 137B_REAL_RAW_REASONING_REPAIR_PLAN`.

## Required Behavior

137R requires positive 136R, 135E, and 135D upstream artifacts. The runner must rerun forbidden-input rejection, expected-output canary, AST shortcut scan, helper provenance verification, leakage audit, and checkpoint hash checks.

Every helper request may contain only `prompt`, `checkpoint_path`, `checkpoint_hash`, `seed`, `max_new_tokens`, and `generation_config`. Expected outputs, expected payloads, labels, scorer metadata, oracle data, target JSON, gold output, row answers, and expected values must never reach generation.

Generated text must be produced before scoring. `generated_before_scoring_report.json` must prove that generation completed first, scoring consumed immutable generated text, and scoring did not feed back into generation.

## Eval

The configured run uses seeds `2271,2272,2273,2274,2275`, `eval_rows_per_family = 96`, reasoning depths `1,2,3,4`, `table_rows = 32`, `multi_doc_count = 4`, and `max_new_tokens = 96`.

Families cover provided-fact QA, single-step reasoning, two-step reasoning, rule chaining, table/rule reasoning, small arithmetic from supplied values, contradiction resolution, multi-doc priority reasoning, hallucination-trap reasoning, and short-explanation diagnostics.

Scoring is deterministic only. No LLM judge, verifier, rerank, constrained decoding, retry loop, JSON mode, grammar decoder, regex fixer, post-generation repair, actual tool execution, runtime tool call, or best-of-n is allowed.

Controls are scorer-only controls and must fail: `STATIC_OUTPUT_CONTROL`, `COPY_PROMPT_CONTROL`, `RANDOM_ANSWER_CONTROL`, and `DISTRACTOR_COPY_CONTROL`.

## Positive Gates

Every seed must independently pass the configured reasoning thresholds. Aggregate `mean_real_raw_reasoning_accuracy` must be at least `0.75`. Controls must fail, leakage must be rejected, canary and AST scan must pass, helper provenance must be written, generated text must be before scoring, and checkpoint hash must be unchanged.

On positive only, `reasoning_subtrack_real_raw_evidence_restored = true`. Even then, raw assistant capability remains quarantined and structured/tool capability remains invalidated.

## Boundary

137R rebuilds only the reasoning subtrack if positive. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated as model evidence. It is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.
