# STABLE_LOOP_PHASE_LOCK_098_PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR Contract

098 is a packaging-only private evaluation RC refresh that binds the 094B/095/096/097 generation-repair evidence into the existing 089 private evaluation package lineage.

It is not clean deploy proof, not production deployment, not a public API, not hosted SaaS, not GPT-like assistant readiness, not open-domain chat, not production chat, and not safety alignment.

## Required Upstreams

```text
089 PRIVATE_EVALUATION_RC_PACKAGE_POSITIVE
094B CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_POSITIVE
095 CHAT_DECODER_GENERATION_REPAIR_POSITIVE
096 FRESH_CHAT_GENERATION_EVAL_POSITIVE
097 CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM_POSITIVE
```

## Hard Wall

098 must not train, run inference, start service, run deployment smoke, mutate checkpoints, mutate packages, or modify runtime/service/deploy code.

Required:

```text
packaging_only = true
train_step_count = 0
inference_run_count = 0
service_started = false
deployment_smoke_run = false
```

## Required Outputs

```text
queue.json
progress.jsonl
refresh_config.json
upstream_refresh_manifest.json
generation_repair_evidence_manifest.json
artifact_hash_manifest.json
refreshed_claim_boundary.json
operator_generation_repair_delta.md
acceptance_delta_checklist.md
rollback_pointer.json
rc_refresh_index.json
private_evaluation_rc_generation_repair_refresh.zip
summary.json
report.md
```

## Positive Gates

Positive requires:

```text
all upstreams positive
original 089 package hash matches 089 summary
generation repair evidence bound
operator delta written
rollback pointer written
refresh zip written
all overclaim flags false
```

## Positive Verdicts

```text
PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_POSITIVE
UPSTREAM_089_PACKAGE_VERIFIED
GENERATION_REPAIR_PROVENANCE_INCLUDED
FRESH_GENERATION_EVAL_PROVENANCE_INCLUDED
MULTI_SEED_OOD_RETENTION_PROVENANCE_INCLUDED
ARTIFACT_HASH_VERIFIED
OPERATOR_DELTA_WRITTEN
ROLLBACK_POINTER_WRITTEN
RC_REFRESH_ZIP_WRITTEN
NO_TRAINING_PERFORMED
NO_INFERENCE_PERFORMED
PRODUCTION_CHAT_NOT_CLAIMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

## Failure Verdicts

```text
PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_FAILS
UPSTREAM_089_ARTIFACT_MISSING
UPSTREAM_STACK_NOT_POSITIVE
UPSTREAM_GENERATION_REPAIR_ARTIFACT_MISSING
UPSTREAM_GENERATION_REPAIR_NOT_POSITIVE
ARTIFACT_HASH_MISMATCH
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
PUBLIC_API_CLAIM_DETECTED
HOSTED_SAAS_CLAIM_DETECTED
```

If 098 passes, continue to `099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE`.
