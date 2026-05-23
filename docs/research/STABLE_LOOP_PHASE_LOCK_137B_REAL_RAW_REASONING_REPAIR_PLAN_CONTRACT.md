# STABLE_LOOP_PHASE_LOCK_137B_REAL_RAW_REASONING_REPAIR_PLAN Contract

137B is a planning-only forensic diagnosis after the 137R clean negative. It reads existing 137R artifacts, diagnoses why real-raw reasoning scored zero, and writes the next repair/probe plan.

137B must not train, repair, run new inference, call `shared_raw_generation_helper.py` for new generations, mutate checkpoints, alter helper/backend code, delete files, consolidate old runners, start services, deploy, modify runtime/release/product surfaces, or change root LICENSE.

## Required Upstream

137B requires 137R clean negative:

- `verdict = REAL_RAW_REASONING_REBUILD_FAILS`
- `decision = real_raw_reasoning_not_restored`
- `next = 137B_REAL_RAW_REASONING_REPAIR_PLAN`
- `mean_real_raw_reasoning_accuracy = 0.0`

It also requires that 137R canary passed, AST scan passed, controls failed, leakage was rejected, checkpoint hash was unchanged, and no expected/scorer metadata reached generation. Positive 136R, 135E, and 135D are also required.

## Required Analysis

The runner must produce artifact-derived diagnosis reports for helper integrity, scorer/task weakness, leakage, checkpoint/model capability gap, prompt-distribution mismatch, decoding/config mismatch, byte-level unreadable output, and exact-match-too-strict cases.

Generation quality metrics must come only from existing 137R `raw_generation_results.jsonl` and `raw_generation_trace.jsonl`. Scoring mismatch must prove whether exact scoring is strict but valid by checking expected-token inclusion, near-match rate, controls, and leakage.

## Decision

Expected decision:

- `decision = real_raw_reasoning_repair_plan_complete`
- `next = 138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE`

The decision must keep `reasoning_restored = false` and `raw_assistant_capability_restored = false`.

## Boundary

137B is planning only. Reasoning is not restored. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated as model evidence. It is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.
