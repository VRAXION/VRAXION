# STABLE_LOOP_PHASE_LOCK_138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE Contract

## Purpose

138I is a deterministic targeted real-raw reasoning repair/probe after 138H.
It tests whether the confirmed `ANSWER=T...` train-namespace rollout failure
can be reduced while helper-only free rollout emits `ANSWER=E...` eval answers
with the correct value.

This is not a broad assistant capability milestone. A clean negative is valid.

## Boundary

138I may train only a new target checkpoint under:

`target/pilot_wave/stable_loop_phase_lock_138i_real_raw_reasoning_rollout_aligned_repair_probe/smoke/checkpoints/target_138i_rollout_aligned_reasoning/model.pt`

It must not mutate source checkpoints, modify `shared_raw_generation_helper.py`,
import old phase runners, start services, deploy, delete files, consolidate old
runners, modify runtime/service/deploy/product/release surfaces, modify SDK
exports, touch docs/product or docs/releases, or change root `LICENSE`.

138I may partially restore only reasoning subtrack real-raw evidence if every
gate passes. Raw assistant capability remains quarantined. Structured/tool
capability remains invalidated as model evidence. It is not GPT-like readiness,
not open-domain assistant readiness, not production chat, not public API, not
deployment readiness, and not safety alignment.

## Required Inputs

138I requires:

- 138H `decision = rollout_aligned_objective_redesign_plan_complete`
- 138H `next = 138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE`
- 138H `primary_bottleneck = train_namespace_rollout_alignment_failure`
- 138GA `decision = objective_failure_disambiguated`
- 138GA all 38 near-match rows classified as `train_namespace_overlap`
- 138R clean negative with helper/canary/AST/leakage/controls/determinism passed
- 138R source checkpoint unchanged, target checkpoint changed, and rollout accuracy `0.0`

## Required Behavior

The target checkpoint must remain helper-compatible: strict `model_state_dict`,
`seq_len`, `vocab_size`, exact byte-GRU state keys, no extra keys, and no
missing keys.

Training must target rollout alignment, not just train loss. The train/eval
design must keep namespaces, prompts, row hashes, and expected outputs disjoint;
it must penalize `ANSWER=T...` replay on eval-style prompts, reward
`ANSWER=E...` with the correct value, and penalize stale `User:` / `Assistant:`
continuation.

Final evaluation must use only `scripts/probes/shared_raw_generation_helper.py`.
Helper requests may contain only `prompt`, `checkpoint_path`, `checkpoint_hash`,
`seed`, `max_new_tokens`, and `generation_config`. Expected outputs, expected
payloads, scorer metadata, labels, oracle data, schema answers, gold outputs,
row answers, and eval-family metadata must never enter helper requests.

Scoring happens only after `generated_text` exists. Scoring separates:

- `answer_prefix_accuracy`
- `namespace_accuracy`
- `answer_value_accuracy`
- `exact_answer_accuracy`
- `train_namespace_leak_rate`
- `stale_user_assistant_fragment_rate`
- `off_prompt_output_rate`

`ANSWER=` prefix alone is not success. Checkpoint change alone is not evidence.
Train loss improvement alone is not evidence.

## Required Gates

Positive requires all of:

- source checkpoint unchanged and target checkpoint changed
- helper-only rollout accuracy improved versus the 138R baseline
- `post_train_namespace_leak_rate <= 0.05`
- `post_eval_namespace_emission_accuracy >= 0.45`
- `post_answer_prefix_accuracy >= 0.60`
- `post_answer_value_accuracy >= 0.35`
- stale fragment rate `<= 0.10`
- off-prompt output rate `<= 0.30`
- forbidden-input rejection passes
- expected-output canary passes
- AST shortcut scan passes
- leakage rejected
- scorer controls fail, including `TRAIN_NAMESPACE_REPLAY_CONTROL`
- provenance written
- generated-before-scoring proof passes
- deterministic replay passes
- no expected/scorer metadata reaches helper requests

## Decision Routes

Positive:

`REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_POSITIVE -> real_raw_reasoning_rollout_aligned_repair_positive -> 139R_REAL_RAW_REASONING_REPAIR_SCALE_CONFIRM`

Clean negative without rollout improvement:

`REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_FAILS -> no_rollout_improvement -> 138I_FAILURE_ANALYSIS`

Namespace leak persists:

`REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_FAILS -> namespace_rollout_failure -> 138S_NAMESPACE_ROLLOUT_FAILURE_ANALYSIS`

No safe rollout-aligned training path:

`ROLLOUT_ALIGNED_TRAINING_PATH_MISSING -> rollout_aligned_training_path_missing -> 138IA_ROLLOUT_ALIGNED_TRAINING_HELPER_INTEGRATION_PLAN`

Determinism mismatch:

`DETERMINISM_REPLAY_MISMATCH -> nondeterministic_repair_probe -> 138N_DETERMINISM_FAILURE_ANALYSIS`

Other fail-closed routes include helper integrity failure, scorer/task weakness,
repair eval leakage, stale chat rollout failure, and teacher-forcing/objective
failure. Threshold weakening, helper modification, old runner imports,
oracle/rerank/verifier/LLM judge paths, post-generation repair, retry loops,
best-of-n, JSON mode, constrained decoding, and regex fixing are rejected.
