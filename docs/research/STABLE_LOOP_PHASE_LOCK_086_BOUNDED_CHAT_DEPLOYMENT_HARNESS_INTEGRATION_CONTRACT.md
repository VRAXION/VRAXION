# STABLE_LOOP_PHASE_LOCK_086_BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION Contract

## Summary

086 is the local/private deployment harness integration for the 085 bounded chat service API alpha.

The deployment harness must validate:

```text
config -> healthcheck -> existing SDK smoke -> 085 service smoke -> artifact provenance -> audit -> rollback pointer
```

This is local/private deployment harness integration only. It is not production deployment, not hosted SaaS, not public beta, not public API, not SDK release, not GPT-like assistant, not open-domain chat, not production chat, and not safety alignment.

## Allowed Changes

Allowed files:

```text
tools/instnct_deploy/instnct_deploy.py
tools/instnct_deploy/config/example.local.json
tools/instnct_deploy/README.md
scripts/probes/run_stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_086_BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_086_BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_RESULT.md
```

086 must not modify `instnct-core/`, the 085 service implementation, service API behavior beyond orchestration from the deployment harness, SDK/public exports, release docs, root `LICENSE`, or checkpoint artifacts.

## Harness Behavior

`tools/instnct_deploy/instnct_deploy.py` must preserve the existing SDK smoke and require:

```text
sdk_smoke_still_passes = true
```

It must add bounded chat service orchestration by running only the 085 child command:

```text
python tools/instnct_service_alpha/instnct_service_alpha.py smoke --config tools/instnct_service_alpha/config/example.local.json --out target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/service_smoke
```

The harness must not implement bounded-chat service logic, must not implement inference logic, and must not call 084 directly except through the 085 smoke path.

## Config

`tools/instnct_deploy/config/example.local.json` must include:

```text
bounded_chat_service_alpha_enabled = true
bounded_chat_service_config_path = tools/instnct_service_alpha/config/example.local.json
bounded_chat_service_smoke_out = target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/service_smoke
bounded_chat_require_085_positive = true
```

Production/public flags remain false.

## Required Artifacts

086 smoke writes under:

```text
target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/
```

Required aggregate artifacts:

```text
queue.json
progress.jsonl
resolved_config.json
healthcheck.json
sdk_smoke_manifest.json
bounded_chat_service_manifest.json
bounded_chat_service_metrics.json
bounded_chat_request_response.json
artifact_validation.json
rollback_pointer.json
audit_log.jsonl
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` must be written from the start and refreshed by phase.

## Gates

Positive requires:

```text
deployment_harness_gate_pass = true
config_schema_valid = true
local_private_policy_allowed = true
healthcheck_pass = true
sdk_smoke_still_passes = true
bounded_chat_service_smoke_started = true
bounded_chat_service_smoke_completed = true
bounded_chat_service_smoke_exit_code = 0
bounded_chat_service_summary_newer_than_086_start = true
bounded_chat_service_report_newer_than_086_start = true
child_command recorded exactly
bounded_chat_service_smoke_pass = true
BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE present
bounded_chat_route_registered = true
bounded_chat_child_084_positive = true
artifact_hash_verified = true
checkpoint_hash_unchanged = true
train_step_count = 0
auth_required = true
auth_rejection_has_no_child_side_effect = true
policy_rejection_has_no_child_side_effect = true
rate_limit_metadata_present = true
bad_input_handled = true
unsupported_input_handled = true
audit_log_written = true
child_runtime_artifacts_preserved = true
rollback_pointer_written = true
production_deployment_claimed = false
public_api_claimed = false
hosted_saas_claimed = false
gpt_like_assistant_readiness_claimed = false
```

`artifact_validation.json` must include:

```text
083_artifact_root
083_artifact_package_zip_hash
084_child_checkpoint_hash
085_service_child_job_path
086_harness_smoke_path
checkpoint_hash_unchanged = true
```

`audit_log.jsonl` must include the harness run id, timestamp, config hash, SDK smoke status, bounded chat service smoke status, child service smoke path, checkpoint hash when available, and final verdict.

`rollback_pointer.json` must include the current config path/hash, previous local/private harness config path/hash if available, 085 service smoke output path, 083 artifact root, instruction to disable bounded_chat_service_alpha_enabled, and no automatic production rollback claim.

## Failure Guards

Policy rejection must happen before child side effects. Public, production, hosted, or regulated configs are rejected during `validate-config` or preflight. The harness must not start the 085 service smoke child and must not create the bounded chat service smoke output for rejected configs.

Failure verdicts:

```text
BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_FAILS
CONFIG_SCHEMA_INVALID
POLICY_GUARD_REJECTS_REGULATED_DEPLOYMENT
HEALTHCHECK_FAILS
SDK_SMOKE_THROUGH_HARNESS_FAILS
BOUNDED_CHAT_SERVICE_SMOKE_THROUGH_HARNESS_FAILS
UPSTREAM_085_ARTIFACT_MISSING
STALE_SERVICE_SMOKE_ARTIFACT_USED
ARTIFACT_HASH_MISMATCH
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
AUTH_POLICY_RATE_LIMIT_REGRESSION_DETECTED
BAD_INPUT_REGRESSION_DETECTED
UNSUPPORTED_INPUT_REGRESSION_DETECTED
AUDIT_LOG_MISSING
ROLLBACK_POINTER_MISSING
PUBLIC_BIND_DETECTED
PRODUCTION_DEPLOYMENT_CLAIM_DETECTED
PUBLIC_API_CLAIM_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
ROOT_LICENSE_CHANGED
```

Positive verdicts:

```text
BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_POSITIVE
DEPLOYMENT_HARNESS_CONFIG_VALID
DEPLOYMENT_HEALTHCHECK_PASSES
SDK_SMOKE_THROUGH_HARNESS_STILL_PASSES
BOUNDED_CHAT_SERVICE_SMOKE_THROUGH_HARNESS_PASSES
BOUNDED_CHAT_ARTIFACT_PROVENANCE_VERIFIED
CHECKPOINT_UNCHANGED_THROUGH_HARNESS
AUTH_POLICY_RATE_LIMIT_THROUGH_HARNESS_PASSES
ROLLBACK_POINTER_WRITTEN
AUDIT_LOGGING_POSITIVE
PRODUCTION_DEPLOYMENT_NOT_CLAIMED
```

## Validation

```powershell
python -m py_compile tools/instnct_deploy/instnct_deploy.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration_check.py
python tools/instnct_deploy/instnct_deploy.py validate-config --config tools/instnct_deploy/config/example.local.json
python tools/instnct_deploy/instnct_deploy.py healthcheck --config tools/instnct_deploy/config/example.local.json
python tools/instnct_deploy/instnct_deploy.py smoke --config tools/instnct_deploy/config/example.local.json --out target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/smoke
python scripts/probes/run_stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_085_bounded_chat_service_api_alpha_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_084_bounded_chat_inference_runtime_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package_check.py --check-only
git diff --check
```

If 086 passes, next milestone is `087_BOUNDED_CHAT_OOD_RED_TEAM_EVAL`.

If 086 fails, next milestone is `086B_BOUNDED_CHAT_DEPLOYMENT_HARNESS_FAILURE_ANALYSIS`.
