# STABLE_LOOP_PHASE_LOCK_117_TARGETED_CAPABILITY_REPAIR_OR_SCALE_PLAN_CONTRACT

117 is an analysis/planning only milestone after positive 116. It reads the 116
ceiling and gap map, ranks capability failures, selects the next targeted
repair or scale milestone, and writes a concrete 118 plan.

117 does not train, repair, run model inference, start services, deploy, mutate
checkpoints, modify runtime/service/release/product surfaces, or make readiness
claims.

## Required Upstreams

- 116 `RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP_POSITIVE`
- 115 `EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_POSITIVE`
- 114 `RAW_ASSISTANT_EXTERNAL_STRESS_BENCHMARK_BRIDGE_POSITIVE`
- 113 `RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_POSITIVE`
- 112 `CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE`
- 099 `BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE`

## Required Decision

117 must select `118_REASONING_FIRST_RAW_ASSISTANT_REPAIR` unless artifact
evidence contradicts the 116 result. The evidence to cite is:

- `first_breakpoint_tier = TIER_4_MULTI_STEP_REASONING`
- `reasoning_failure_count = 161`
- reasoning is the largest failure class
- later long-context and combined failures include reasoning as a compounded
  factor

## Positive Verdicts

- `TARGETED_CAPABILITY_REPAIR_OR_SCALE_PLAN_POSITIVE`
- `UPSTREAM_116_CEILING_MAP_VERIFIED`
- `BREAKPOINT_ANALYSIS_WRITTEN`
- `FAILURE_PRIORITY_MAP_WRITTEN`
- `ROOT_VS_SYMPTOM_ANALYSIS_WRITTEN`
- `REPAIR_TARGET_SELECTED`
- `EVAL_GATE_PROPOSAL_WRITTEN`
- `NEXT_MILESTONE_PLAN_WRITTEN`
- `NO_TRAINING_PERFORMED`
- `BOUNDED_RELEASE_UNCHANGED`
- `PRODUCTION_CHAT_NOT_CLAIMED`
- `GPT_LIKE_READINESS_NOT_CLAIMED`

## Boundary

117 is planning only. It is not GPT-like assistant readiness, not open-domain
assistant readiness, not production chat, not public API, not deployment
readiness, and not safety alignment.
