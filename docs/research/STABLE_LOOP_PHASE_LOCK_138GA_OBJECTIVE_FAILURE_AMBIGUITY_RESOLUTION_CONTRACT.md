# STABLE_LOOP_PHASE_LOCK_138GA_OBJECTIVE_FAILURE_AMBIGUITY_RESOLUTION Contract

## Purpose

138GA is artifact-only near-match ambiguity resolution after 138G. 138G routed
to `objective_failure_ambiguous -> 138GA_OBJECTIVE_FAILURE_AMBIGUITY_RESOLUTION`
because 138R had `near_match_rate > 0.0` while exact expected-token inclusion
remained `0.0`.

138GA determines whether those near-match rows are meaningful partial reasoning,
formatting/scorer contribution, train-namespace rollout, stale/prompt/distractor
overlap, weak scorer digit-substring overlap, or still unknown.

## Boundary

138GA must not train, run new inference, call `shared_raw_generation_helper.py`,
run torch forward passes, mutate checkpoints, modify helper/backend code, import
old runners, start services, deploy, delete or consolidate files, modify
runtime/release/product surfaces, or change root `LICENSE`.

All capability claims remain false. Reasoning is not restored. The reasoning
subtrack real-raw evidence is not partially restored. Raw assistant capability
remains quarantined. Structured/tool capability remains invalidated. It is not
GPT-like readiness, not open-domain assistant readiness, not production chat,
not public API, not deployment readiness, and not safety alignment.

## Required Upstream

138GA requires 138G to show:

- `decision = objective_failure_ambiguous`
- `next = 138GA_OBJECTIVE_FAILURE_AMBIGUITY_RESOLUTION`
- `near_match_rate > 0.0`
- `expected_token_inclusion_rate = 0.0`
- teacher-forced loss fields recorded as `diagnostic_gap`
- `artifact_only_analysis = true`
- all capability flags false

138GA also requires the 138R scoring, raw generation, eval row, trace, aggregate,
helper/canary/AST, leakage, controls, and generated-before-scoring artifacts to
remain present and internally consistent.

## Classification Policy

Near-match rows are extracted from existing artifacts only. The row count and
rate are computed from `scoring_results.jsonl`; they are not hard-coded. If the
computed rate disagrees with 138G or 138R, the decision is
`near_match_artifact_inconsistency -> 138GA_NEAR_MATCH_ARTIFACT_INCONSISTENCY_ANALYSIS`.

Each near-match row must have exactly one primary label:

- `meaningful_partial_answer`
- `numeric_partial_match`
- `formatting_or_wrapper_mismatch`
- `train_namespace_overlap`
- `stale_chat_overlap`
- `prompt_copy_overlap`
- `distractor_overlap`
- `common_token_overlap`
- `scorer_false_near_match`
- `unknown_near_match`

`meaningful_partial_answer` is strict. It cannot be assigned for generic token
overlap, an `ANSWER=` prefix alone, train namespace tokens, prompt-copy overlap,
or stale chat text. If generated text emits `ANSWER=T...` while expected output
requires `ANSWER=E...`, the primary class is `train_namespace_overlap`.

## Decision Routes

If meaningful near-match rate is at least `0.02`, or formatting/wrapper mismatch
dominates the nontrivial near matches:

`scorer_or_eval_design_contributes -> 138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS`

If near matches are mostly train-namespace, common-token, stale, prompt-copy,
distractor, or scorer-false overlap:

`objective_failure_disambiguated -> 138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN`

If the unknown bucket is material:

`ambiguity_requires_manual_sample_review -> 138GB_NEAR_MATCH_MANUAL_REVIEW_PACKET`

If helper integrity regression appears:

`raw_helper_integrity_failure -> 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL`

