# STABLE_LOOP_PHASE_LOCK_085_BOUNDED_CHAT_SERVICE_API_ALPHA Result

## Status

085 adds the localhost/private bounded chat service API alpha route:

```text
POST /v1/bounded-chat/infer
```

This is service API alpha only and localhost/private only. It is not deploy-ready service, not public API, not SDK surface, not GPT-like assistant, not open-domain chat, not production chat, not safety alignment, and not public beta / GA / hosted SaaS.

The implementation keeps the service bound to `127.0.0.1` and rejects public bind attempts.

## Runtime Path

The Python service does not implement bounded-chat inference directly. It calls:

```text
cargo run -p instnct-core --example phase_lane_bounded_chat_inference_runtime
```

The service parses and preserves:

```text
single_inference.json
runtime_metrics.json
summary.json
report.md
audit_log.jsonl
```

The child runtime must report:

```text
artifact_hash_verified = true
checkpoint_hash_unchanged = true
train_step_count = 0
```

The bounded chat service config uses:

```text
bounded_chat_artifact_root
bounded_chat_runtime_out_root
bounded_chat_max_input_chars
bounded_chat_max_response_tokens
bounded_chat_timeout_ms
target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke
target/pilot_wave/stable_loop_phase_lock_085_bounded_chat_service_api_alpha/
```

## Service Envelope And Audit

Every response includes:

```text
ok
value
error
request_id
idempotency_key
route
rate_limit
artifact_hash
child_job_path
```

For success:

```text
value.inference
```

is the parsed 084 inference row.

The service-level `audit_log.jsonl` records:

```text
request_id
timestamp
route
auth_result
policy_result
status
prompt_sha256
child_job_path
checkpoint_sha256
artifact_package_zip_sha256
```

## Artifacts

085 smoke writes:

```text
queue.json
progress.jsonl
service_config_resolved.json
route_manifest.json
bounded_chat_request_response.json
bad_input_results.jsonl
unsupported_input_results.jsonl
auth_policy_results.jsonl
rate_limit_report.json
child_runtime_manifest.json
audit_log.jsonl
service_metrics.json
summary.json
report.md
```

## Positive Metrics

The positive service metrics are:

```text
localhost_bind_only = true
public_bind_rejected = true
auth_required = true
auth_rejection_has_no_child_side_effect = true
policy_rejection_has_no_child_side_effect = true
rate_limit_metadata_present = true
bounded_chat_route_registered = true
bounded_chat_single_prompt_pass = true
bounded_chat_json_envelope_pass = true
bounded_chat_child_084_positive = true
artifact_hash_verified = true
checkpoint_hash_unchanged = true
train_step_count = 0
unsupported_input_handled = true
bad_input_handled = true
timeout_guard_pass = true
idempotency_reuse_pass = true
idempotency_conflict_pass = true
audit_log_written = true
child_runtime_artifacts_preserved = true
existing_062_routes_preserved = true
service_api_alpha_only = true
production_chat_claimed = false
gpt_like_assistant_readiness_claimed = false
sdk_surface_exposed = false
deployment_harness_mutated = false
```

Bad input and unsupported coverage:

```text
missing prompt
non-string prompt
empty prompt
whitespace prompt
oversized prompt
malformed JSON
invalid max_response_tokens
unsupported topic
status = unsupported
```

## Verdicts

Positive verdicts:

```text
BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE
LOCALHOST_BIND_RESTRICTED
AUTH_GUARD_PASSES
POLICY_GUARD_PASSES
RATE_LIMIT_METADATA_PASSES
BOUNDED_CHAT_ROUTE_REGISTERED
BOUNDED_CHAT_INFERENCE_CHILD_RUNTIME_PASSES
ARTIFACT_PACKAGE_VERIFIED_BY_CHILD
CHECKPOINT_UNCHANGED
JSON_RESPONSE_ENVELOPE_PASSES
BAD_INPUT_HANDLED
UNSUPPORTED_INPUT_HANDLED
AUDIT_LOG_WRITTEN
NO_TRAINING_PERFORMED
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure verdicts covered:

```text
BOUNDED_CHAT_SERVICE_API_ALPHA_FAILS
UPSTREAM_083_ARTIFACT_MISSING
UPSTREAM_084_RUNTIME_MISSING
PUBLIC_BIND_DETECTED
SERVICE_API_PUBLIC_EXPOSURE_DETECTED
AUTH_GUARD_MISSING
AUTHZ_SIDE_EFFECT_LEAK
POLICY_REJECTION_SIDE_EFFECT_LEAK
RATE_LIMIT_BOUNDARY_MISSING
BOUNDED_CHAT_ROUTE_MISSING
BOUNDED_CHAT_INFERENCE_CHILD_RUNTIME_FAILS
TIMEOUT_GUARD_FAILS
ARTIFACT_HASH_MISMATCH
CHECKPOINT_MUTATION_DETECTED
JSON_RESPONSE_ENVELOPE_MISSING
BAD_INPUT_NOT_HANDLED
UNSUPPORTED_INPUT_NOT_HANDLED
AUDIT_LOG_MISSING
TRAINING_SIDE_EFFECT_DETECTED
ORACLE_SHORTCUT_DETECTED
SDK_PUBLIC_EXPORT_MUTATION_DETECTED
DEPLOYMENT_HARNESS_MUTATION_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
ROOT_LICENSE_CHANGED
```

## Validation Commands

```powershell
python -m py_compile tools/instnct_service_alpha/instnct_service_alpha.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_085_bounded_chat_service_api_alpha_check.py
python tools/instnct_service_alpha/instnct_service_alpha.py healthcheck --config tools/instnct_service_alpha/config/example.local.json
python tools/instnct_service_alpha/instnct_service_alpha.py smoke --config tools/instnct_service_alpha/config/example.local.json --out target/pilot_wave/stable_loop_phase_lock_085_bounded_chat_service_api_alpha/smoke
python scripts/probes/run_stable_loop_phase_lock_085_bounded_chat_service_api_alpha_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_084_bounded_chat_inference_runtime_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_062_service_api_alpha_check.py --check-only
git diff --check
```

If 085 passes, next milestone is `086_BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION`. If 085 fails, next milestone is `085B_BOUNDED_CHAT_SERVICE_API_ALPHA_FAILURE_ANALYSIS`.
