# STABLE_LOOP_PHASE_LOCK_138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN Contract

## Purpose

138H is planning-only objective redesign after 138GA. 138GA resolved the
nonzero near-match ambiguity by classifying every near-match row as
`train_namespace_overlap`. The bottleneck is therefore
`train_namespace_rollout_alignment_failure`, not restored reasoning and not a
scorer/eval false negative.

138H designs the next 138I repair/probe. It does not train or repair.

## Boundary

138H must not train, run inference, call `shared_raw_generation_helper.py`, run
torch forward passes, mutate checkpoints, modify helper/backend code, import old
runners, start services, deploy, delete or consolidate files, modify
runtime/release/product surfaces, or change root `LICENSE`.

All capability claims remain false. Reasoning is not restored. The reasoning
subtrack real-raw evidence is not partially restored. Raw assistant capability
remains quarantined. Structured/tool capability remains invalidated. It is not
GPT-like readiness, not open-domain assistant readiness, not production chat,
not public API, not deployment readiness, and not safety alignment.

## Required Upstream

138H requires:

- 138GA `decision = objective_failure_disambiguated`
- 138GA `next = 138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN`
- 138GA `near_match_row_count = 38`
- 138GA `total_scored_row_count = 960`
- 138GA `primary_label_counts = {"train_namespace_overlap": 38}`
- 138GA `meaningful_near_match_rate = 0.0`
- 138G `decision = objective_failure_ambiguous`
- 138G teacher-forced loss fields recorded as `diagnostic_gap`
- 138R final rollout accuracy `0.0`
- 138R expected-token inclusion `0.0`
- 138R helper, canary, AST, leakage, controls, determinism replay, checkpoint
  integrity, and generated-before-scoring gates passed

If these are absent or inconsistent, 138H must fail closed to the configured
upstream or evidence recheck route.

## 138I Design Requirements

The 138I plan must separately gate three concerns:

1. Output namespace alignment: eval rows must not emit `ANSWER=T...`; where
   applicable, eval rows require `ANSWER=E...`.
2. Free-rollout alignment: helper-only autoregressive `generated_text` must
   improve before scoring. Train loss or teacher-forced loss alone is not a
   success condition.
3. Scoring/format discipline: no threshold weakening, expected-output
   construction, constrained decoding, JSON mode, regex fixer, retry loop,
   best-of-n, verifier, rerank, LLM judge, or post-generation repair.

138I must require `shared_raw_generation_helper.py` only, generated text before
scoring, no expected/scorer metadata in helper requests, expected-output canary,
AST shortcut scan, deterministic replay, failed controls, rejected leakage,
immutable source checkpoint, and target checkpoint under `target/` only.

## Decision Routes

If the artifact-derived plan is complete:

`rollout_aligned_objective_redesign_plan_complete -> 138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE`

If 138GA artifacts are missing or inconsistent:

`upstream_138ga_artifact_missing -> 138H_UPSTREAM_138GA_ARTIFACT_MISSING`

If the evidence no longer supports the objective disambiguation:

`rollout_objective_redesign_blocked -> 138HA_ROLLOUT_OBJECTIVE_EVIDENCE_RECHECK`

If helper integrity regression appears:

`raw_helper_integrity_failure -> 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL`
