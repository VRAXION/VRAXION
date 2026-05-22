# STABLE_LOOP_PHASE_LOCK_115_EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_CONTRACT

115 is an eval-only external-style raw assistant stress confirmation after
positive 114. It scales the 114 bridge to five seeds, larger row counts, longer
contexts, more noise, more table/doc complexity, and stricter control gates.

115 does not train, repair, deploy, start services, mutate checkpoints, modify
runtime/service/release/product surfaces, or make readiness claims.

## Required Upstreams

- 114 `RAW_ASSISTANT_EXTERNAL_STRESS_BENCHMARK_BRIDGE_POSITIVE`
- 113 `RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_POSITIVE`
- 112 `CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE`
- 111X `CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_POSITIVE`
- 099 `BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE`

## Benchmark Rules

The only positive-scored arm is
`POST_112_RAW_CURRENT_CHASSIS_EXTERNAL_STRESS`. Helper/reference metrics remain
diagnostic and must not be merged into the raw score.

The full configured run is required: five seeds, 96 rows per family, 8192
long-context characters, 16 noise blocks, 8 format variants, 32 table rows, and
5 documents for multi-doc tasks.

Scoring is deterministic rubric-bounded: exact values, required keywords,
forbidden outputs, regex checks, JSON/JSONL parse validity, slot/case
correctness, refusal markers, and collapse metrics. No LLM judge, subjective
scoring, internet facts, or current-world facts may be used.

## Positive Verdicts

- `EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_POSITIVE`
- `UPSTREAM_114_BRIDGE_VERIFIED`
- `RAW_EXTERNAL_STRESS_CONFIRMED`
- `CONTROLS_FAILED`
- `LEAKAGE_REJECTED`
- `NAMESPACE_MEMORIZATION_REJECTED`
- `RETENTION_PASSES`
- `COLLAPSE_REJECTED`
- `BOUNDED_RELEASE_UNCHANGED`
- `PRODUCTION_CHAT_NOT_CLAIMED`
- `GPT_LIKE_READINESS_NOT_CLAIMED`

## Boundary

115 is eval-only with deterministic rubric-bounded scoring. It is not GPT-like
assistant readiness, not open-domain assistant readiness, not production chat,
not public API, not deployment readiness, and not safety alignment.
