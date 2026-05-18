# STABLE_LOOP_PHASE_LOCK_099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE Contract

099 is the clean local/private bounded deploy-readiness gate.

It runs a fresh local/private deployment harness smoke into a target-only 099 output directory, then binds the latest private evaluation package refresh, packaged-winner proof, and long-run stability evidence.

099 may claim local/private release-readiness only if the gate is positive. It must not claim production deployment, public API, hosted SaaS, GPT-like assistant readiness, open-domain chat, production chat, or safety alignment.

## Required Upstreams

```text
098 PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_POSITIVE
089B PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_POSITIVE
088 BOUNDED_CHAT_LONG_RUN_CONCURRENCY_RESOURCE_STABILITY_POSITIVE
```

## Fresh Harness Smoke

099 must generate a target-only local/private deploy config from `tools/instnct_deploy/config/example.local.json` and run:

```text
python tools/instnct_deploy/instnct_deploy.py smoke --config <generated_config> --out <099_harness_smoke>
```

The generated config must keep local/private mode and redirect both harness output and bounded-chat service smoke output under the 099 target root.

## Positive Gates

Positive requires:

```text
fresh_harness_smoke_exit_code = 0
deployment_harness_gate_pass = true
sdk_smoke_still_passes = true
bounded_chat_service_smoke_pass = true
artifact_hash_verified = true
checkpoint_hash_unchanged = true
rollback_pointer_written = true
train_step_count = 0
upstream_098_positive = true
upstream_089b_positive = true
upstream_088_positive = true
```

## Required Artifacts

```text
queue.json
progress.jsonl
generated_clean_local_private_deploy_config.json
clean_deploy_config_manifest.json
upstream_release_manifest.json
fresh_harness_child_manifest.json
fresh_harness_validation.json
release_readiness_evidence_chain.json
claim_boundary.json
fresh_harness_stdout.txt
fresh_harness_stderr.txt
summary.json
report.md
deployment_harness_smoke/
deployment_harness_service_smoke/
```

Progress, summary, and report must be written from start and refreshed by heartbeat while the fresh harness child is running.

## Positive Verdicts

```text
BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE
UPSTREAM_RELEASE_EVIDENCE_VERIFIED
CLEAN_LOCAL_PRIVATE_CONFIG_GENERATED
FRESH_DEPLOYMENT_HARNESS_SMOKE_PASSES
SDK_SMOKE_STILL_PASSES
BOUNDED_CHAT_SERVICE_SMOKE_PASSES
ARTIFACT_HASH_VERIFIED
CHECKPOINT_UNCHANGED
ROLLBACK_POINTER_WRITTEN
PRIVATE_EVAL_RC_REFRESH_VERIFIED
PACKAGED_WINNER_REPRO_VERIFIED
LONG_RUN_STABILITY_VERIFIED
LOCAL_PRIVATE_RELEASE_READY
PRODUCTION_DEPLOYMENT_NOT_CLAIMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

## Failure Verdicts

```text
BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_FAILS
UPSTREAM_098_ARTIFACT_MISSING
UPSTREAM_098_NOT_POSITIVE
UPSTREAM_089B_ARTIFACT_MISSING
UPSTREAM_089B_NOT_POSITIVE
UPSTREAM_088_ARTIFACT_MISSING
UPSTREAM_088_NOT_POSITIVE
DEPLOY_CONFIG_MISSING
FRESH_HARNESS_SMOKE_TIMEOUT
FRESH_HARNESS_SMOKE_FAILS
FRESH_HARNESS_ARTIFACT_MISSING
FRESH_HARNESS_GATE_FAILS
TRAINING_SIDE_EFFECT_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
PUBLIC_API_CLAIM_DETECTED
PRODUCTION_DEPLOYMENT_CLAIM_DETECTED
```
