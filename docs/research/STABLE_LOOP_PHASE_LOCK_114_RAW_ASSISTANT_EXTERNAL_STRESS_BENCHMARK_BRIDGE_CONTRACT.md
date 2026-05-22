# STABLE_LOOP_PHASE_LOCK_114_RAW_ASSISTANT_EXTERNAL_STRESS_BENCHMARK_BRIDGE_CONTRACT

114 is an eval-only external-style stress bridge after positive 113. It tests
whether the 112 scale-confirmed raw current-chassis path holds on benchmark-like
prompts with deterministic rubric-bounded scoring.

114 does not train, repair, deploy, start services, mutate checkpoints, modify
runtime/product/release surfaces, or make readiness claims.

## Required Upstreams

- 113 `RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_POSITIVE`
- 112 `CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE`
- 111X `CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_POSITIVE`
- 111R `RETENTION_OR_LM_REGRESSION_ANALYSIS_POSITIVE`
- 110 `INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_POSITIVE`
- 099 `BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE`

## Benchmark Rules

The only positive arm is `POST_112_RAW_CURRENT_CHASSIS_EXTERNAL_STYLE`.
`INTEGRATED_DECODER_POLICY_REFERENCE_DIAGNOSTIC` is diagnostic only. Raw/helper
metrics must not be merged.

All scored facts must be provided in the prompt, synthetic local facts, or stable
local facts. Do not score current-world internet facts. Scoring must use exact
values, required keywords, forbidden outputs, regex checks, JSON schema checks,
case/slot correctness, refusal markers, and collapse metrics. No LLM judge and
no subjective scoring are allowed.

Controls `STATIC_OUTPUT_CONTROL`, `COPY_PROMPT_CONTROL`, and
`RANDOM_SLOT_CONTROL` must fail. Leakage checks must compare against prior
108A/109/110/111/111R/111X/112/113 samples and eval artifacts.

## Positive Verdicts

- `RAW_ASSISTANT_EXTERNAL_STRESS_BENCHMARK_BRIDGE_POSITIVE`
- `UPSTREAM_113_PACKAGE_VERIFIED`
- `EXTERNAL_STYLE_BENCHMARK_EVALUATED`
- `RAW_CURRENT_CHASSIS_EXTERNAL_STYLE_PASSES`
- `HELPER_PATH_SEPARATED`
- `BENCHMARK_LEAKAGE_REJECTED`
- `CONTROLS_REJECTED`
- `RETENTION_PASSES`
- `COLLAPSE_REJECTED`
- `OVERCLAIM_REJECTED`
- `NO_TRAINING_PERFORMED`
- `CHECKPOINTS_UNCHANGED`
- `PRODUCTION_CHAT_NOT_CLAIMED`
- `GPT_LIKE_READINESS_NOT_CLAIMED`

## Boundary

114 is an external-style stress bridge only. It is deterministic
rubric-bounded scoring. It is not GPT-like assistant readiness, not open-domain
assistant readiness, not production chat, not public API, not deployment
readiness, and not safety alignment.
