# STABLE_LOOP_PHASE_LOCK_138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE Contract

## Purpose

138R is a deterministic targeted real-raw reasoning repair/probe after 137B.
It tests whether a new target checkpoint trained under `target/` can improve
real-raw reasoning when final evaluation uses only
`scripts/probes/shared_raw_generation_helper.py`.

137B diagnosed the 137R clean negative as a checkpoint/model capability gap
with prompt-distribution mismatch. 138R is the first bounded repair/probe for
that diagnosis.

## Boundary

138R may train only a new target checkpoint under:

`target/pilot_wave/stable_loop_phase_lock_138r_real_raw_reasoning_repair_training_plan_or_probe/smoke/checkpoints/target_138r_reasoning_repair/model.pt`

It must not mutate source checkpoints, modify `shared_raw_generation_helper.py`,
import old phase runners, start services, deploy, delete files, consolidate old
runners, modify runtime/service/deploy/product/release surfaces, modify SDK
exports, touch docs/product or docs/releases, or change root `LICENSE`.

138R may partially restore only reasoning subtrack real-raw evidence if every
gate passes. Raw assistant capability remains quarantined. Structured/tool
capability remains invalidated as model evidence. It is not GPT-like readiness,
not open-domain assistant readiness, not production chat, not public API, not
deployment readiness, and not safety alignment.

## Required Inputs

138R requires:

- 137B `decision = real_raw_reasoning_repair_plan_complete`
- 137R clean negative with `decision = real_raw_reasoning_not_restored`
- 136R `REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD_POSITIVE`
- 135E `SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE_POSITIVE`

It reads 137B diagnosis artifacts, 137R raw/scoring/helper artifacts, and
136R/135E helper provenance. The immutable source checkpoint is the 102
byte-GRU checkpoint selected by the helper unless a provenance-safe selection
proves otherwise.

## Required Behavior

The target checkpoint must remain helper-compatible: strict `model_state_dict`,
`seq_len`, `vocab_size`, exact byte-GRU state keys, no extra keys, and no
missing keys.

Final evaluation must use only `scripts/probes/shared_raw_generation_helper.py`.
Helper requests may contain only `prompt`, `checkpoint_path`, `checkpoint_hash`,
`seed`, `max_new_tokens`, and `generation_config`. Expected outputs, expected
payloads, scorer metadata, labels, oracle data, schema answers, gold outputs,
row answers, and eval-family metadata must never enter helper requests.

Scoring happens only after `generated_text` exists. Scoring is deterministic
only: exact/required answer token, numeric exact match, regex, forbidden
distractor token, and forbidden stale `User:` / `Assistant:` fragments. There is
no LLM judge, verifier, rerank, constrained decoding, retry loop, JSON mode,
grammar decoder, regex fixer, post-generation repair, tool execution, or
best-of-n.

138R must rerun forbidden-input rejection, expected-output canary, AST shortcut
scan over helper/runner/checker, helper provenance verification, checkpoint
hash verification, leakage audit, scorer controls, and generated-before-scoring
proof.

## Determinism

138R records Python, numpy if used, torch if used, device, CUDA availability,
checkpoint hashes, dataset hash, train/eval config hashes, helper source hash,
and deterministic algorithm settings. Row generation and JSON writes must be
stable. Wall-clock time and random UUIDs must not influence dataset, training,
evaluation, decision, or score.

After final eval, 138R reruns final eval with the same target checkpoint, rows,
seeds, helper request hashes, and config. Generated text hashes, generation
trace hashes, per-row pass/fail, per-family metrics, aggregate metrics, and
decision-critical metrics must be exactly identical.

## Decision Routes

Positive:

`REAL_RAW_REASONING_REPAIR_PROBE_POSITIVE -> real_raw_reasoning_repair_probe_positive -> 139R_REAL_RAW_REASONING_REPAIR_SCALE_CONFIRM`

Clean negative:

`REAL_RAW_REASONING_REPAIR_PROBE_FAILS -> real_raw_reasoning_repair_probe_failed -> 138B_REAL_RAW_REASONING_REPAIR_FAILURE_ANALYSIS`

No safe helper-compatible training path:

`REAL_RAW_REASONING_TRAINING_HELPER_MISSING -> real_raw_reasoning_training_helper_missing -> 138A_REAL_RAW_REASONING_TRAINING_HELPER_INTEGRATION_PLAN`

Determinism mismatch:

`DETERMINISM_REPLAY_MISMATCH -> nondeterministic_repair_probe -> 138N_DETERMINISM_FAILURE_ANALYSIS`

Other fail-closed routes include helper integrity failure, scorer/task weakness,
repair eval leakage, stale chat rollout failure, and teacher-forcing/objective
failure.

