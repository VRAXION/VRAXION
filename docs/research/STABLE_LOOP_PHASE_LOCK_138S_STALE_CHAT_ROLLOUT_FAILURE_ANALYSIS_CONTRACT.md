# STABLE_LOOP_PHASE_LOCK_138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS Contract

## Purpose

138S is an artifact-only analysis milestone after the 138I clean negative.
138I suppressed `ANSWER=T...` train namespace leakage and emitted
`ANSWER=E...` on every eval row, but answer value accuracy stayed at `0.0` and
stale `User:` / `Assistant:` fragments remained above gate.

138S asks whether stale chat rollout explains the value failure, or whether the
model learned the wrapper without grounding the answer value.

## Boundary

138S must not train, repair, run new inference, call
`shared_raw_generation_helper.py`, run torch forward passes, mutate checkpoints,
modify helper/backend code, import old runners, start services, deploy, delete
or consolidate files, modify runtime/service/deploy/product/release surfaces,
modify docs/product, modify docs/releases, or change root `LICENSE`.

138S does not restore reasoning, reasoning subtrack evidence, raw assistant
capability, structured/tool capability, GPT-like readiness, open-domain
assistant readiness, production chat, public API, deployment readiness, or
safety alignment.

## Required Upstream

138S requires 138I:

- `verdict = REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_FAILS`
- `decision = stale_chat_rollout_failure`
- `next = 138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS`
- canary, AST, controls, leakage, deterministic replay, provenance, and
  generated-before-scoring gates passed
- source checkpoint unchanged
- target checkpoint changed
- no expected/scorer metadata reached helper requests
- `post_train_namespace_leak_rate = 0.0`
- `post_eval_namespace_emission_accuracy = 1.0`
- `post_answer_prefix_accuracy = 1.0`
- `post_answer_value_accuracy = 0.0`
- stale chat fragment rate above `0.10`

## Required Analysis

138S writes deterministic artifact-derived reports for:

- stale chat distribution by family and seed
- answer value grounding failure
- prefix/value decoupling
- source-prior versus training-objective evidence, with missing fields recorded
  as `diagnostic_gap`
- one primary failure taxonomy label for every failed eval row
- stale chat/value coupling probabilities
- a machine-readable next repair recommendation

The next recommendation must choose from:

- `138T_STALE_CHAT_SUPPRESSION_AND_VALUE_GROUNDING_REPAIR_PLAN`
- `138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN`
- `138P_PROMPT_OUTPUT_FORMAT_REBALANCE_PLAN`
- `138Q_STALE_SOURCE_PRIOR_CHECKPOINT_REVIEW`
- `138E_SCORER_OR_TASK_WEAKNESS_ANALYSIS`

## Decision Rules

If analysis completes:

`stale_chat_rollout_failure_analysis_complete -> next_repair_recommendation.recommended_next`

If 138I artifacts are missing or inconsistent:

`upstream_138i_artifact_missing -> 138S_UPSTREAM_138I_ARTIFACT_MISSING`

If helper integrity regressed:

`raw_helper_integrity_failure -> 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL`

If stale/value failure cannot be distinguished:

`stale_value_failure_ambiguous -> 138SB_STALE_VALUE_MANUAL_REVIEW_PACKET`

All capability and readiness flags remain false in `decision.json`,
`summary.json`, and `report.md`.
