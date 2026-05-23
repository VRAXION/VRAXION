# STABLE_LOOP_PHASE_LOCK_138G_REAL_RAW_REASONING_OBJECTIVE_FAILURE_ANALYSIS Contract

## Purpose

138G is artifact-only analysis after the 138R clean negative. It explains why
the 138R target checkpoint changed and helper/canary/AST/leakage/controls and
deterministic replay passed, while helper-only final rollout still produced
`mean_real_raw_reasoning_accuracy = 0.0`.

This phase does not repair the model. It identifies whether the 138R failure is
objective/rollout mismatch, scoring ambiguity, missing diagnostic evidence, or
helper integrity regression.

## Boundary

138G must not train, run new inference, call `shared_raw_generation_helper.py`
for new generations, run torch forward passes, mutate checkpoints, modify
helper/backend code, import old runners, start services, deploy, delete or
consolidate files, modify runtime/release/product surfaces, or change root
`LICENSE`.

All claims remain bounded. Reasoning is not restored. Raw assistant capability
remains quarantined. Structured/tool capability remains invalidated. It is not
GPT-like readiness, not open-domain assistant readiness, not production chat,
not public API, not deployment readiness, and not safety alignment.

## Required Upstream

138G requires 138R to show:

- `verdict = REAL_RAW_REASONING_REPAIR_PROBE_FAILS`
- `decision = teacher_forcing_or_training_objective_failure`
- `next = 138G_REAL_RAW_REASONING_OBJECTIVE_FAILURE_ANALYSIS`
- `determinism_replay_passed = true`
- `mean_real_raw_reasoning_accuracy = 0.0`
- `expected_token_inclusion_rate = 0.0`
- helper, canary, AST, leakage, and controls passed
- source checkpoint unchanged
- target checkpoint changed

If these artifacts are missing or inconsistent, 138G must route to
`138G_UPSTREAM_138R_ARTIFACT_MISSING`.

## Evidence Policy

Every analysis report must tag report fields as one of:

- `artifact_observed`
- `computed_from_artifact`
- `diagnostic_gap`
- `inference`

Missing evidence is a `diagnostic_gap`, not an inferred fact. In particular,
138G must not claim teacher-forced loss improved unless exact 138R artifact
paths and fields contain teacher-forced initial and final loss values. The
available 138R training metrics may support train-loss movement, but not a
dedicated teacher-forced loss claim unless such fields exist.

## Required Analysis

138G writes deterministic, artifact-derived reports for:

- train-loss and teacher-forced-loss evidence versus final rollout metrics
- raw rollout output patterns
- `ANSWER=T...` versus `ANSWER=E...` namespace behavior
- prompt-to-answer alignment
- first expected-token mismatch and answer-prefix mismatch
- stop/EOS behavior
- scoring strictness and near-match ambiguity
- checkpoint/objective gap
- diagnostic gaps
- conservative root-cause selection

If near-match evidence is nonzero, 138G must route to
`objective_failure_ambiguous` rather than overclaiming an objective root cause.

## Decision Routes

If analysis supports objective/rollout mismatch:

`objective_failure_analysis_complete -> 138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN`

If objective failure cannot be distinguished from scorer/eval weakness:

`objective_failure_ambiguous -> 138GA_OBJECTIVE_FAILURE_AMBIGUITY_RESOLUTION`

If upstream artifacts are missing:

`upstream_138r_artifact_missing -> 138G_UPSTREAM_138R_ARTIFACT_MISSING`

If helper integrity regression appears:

`raw_helper_integrity_failure -> 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL`

## Next Redesign Guardrails

The 138H redesign requirements must include helper-only final eval, generated
text before scoring, expected-output canary, AST shortcut scan, deterministic
replay, failed controls, rejected leakage, and clean negative accepted.

It must reject teacher-forcing-only success, loss-only success, expected-output
construction, old runner imports, oracle/rerank/verifier/LLM judge,
post-generation repair, and threshold weakening to force positive.

