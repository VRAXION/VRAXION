# STABLE_LOOP_PHASE_LOCK_084_BOUNDED_CHAT_INFERENCE_RUNTIME Contract

## Scope

084 adds `instnct-core/examples/phase_lane_bounded_chat_inference_runtime.rs` as a local Rust CLI example and `scripts/probes/run_stable_loop_phase_lock_084_bounded_chat_inference_runtime_check.py` as the static checker.

Generated outputs are confined to:

```text
target/pilot_wave/stable_loop_phase_lock_084_bounded_chat_inference_runtime/
```

Default artifact root:

```text
target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke
```

This milestone is bounded local inference runtime only. It is not deploy-ready service, not service API, not SDK surface, not GPT-like assistant, not open-domain chat, not production chat, not safety alignment, and not public beta / GA / hosted SaaS.

Runtime freeze:

```text
no service API change
no network listener
no deployment harness change
no SDK/public export change
no release docs change
no root LICENSE change
no checkpoint mutation
```

## CLI Surface

The runtime exposes only a local CLI with these arguments:

```text
--out
--artifact-root
--prompt
--batch-in
--max-input-chars
--max-response-tokens
--timeout-ms
--json
--heartbeat-sec
```

No network listener, service API, SDK/public export, deployment harness integration, or production endpoint is added.

## Pre-Inference Verification

Before any inference, the runtime must verify:

```text
artifact_index.json present
integrity_hashes.json present
capability_surface.json present
claim_boundary.json present
packaged checkpoint exists
checkpoint sha256 matches integrity_hashes.json
checkpoint size matches integrity_hashes.json
083 summary contains CHAT_MODEL_ARTIFACT_RC_PACKAGE_POSITIVE
```

If verification fails, the runtime must not run inference and must emit a structured failure such as `UPSTREAM_083_ARTIFACT_MISSING`, `ARTIFACT_HASH_MISMATCH`, or `CHECKPOINT_LOAD_FAILS`.

The runtime records:

```text
checkpoint_hash_before
checkpoint_hash_after
checkpoint_hash_unchanged = true
train_step_count = 0
prediction_oracle_used = false
llm_judge_used = false
service_api_exposed = false
deployment_harness_exposed = false
sdk_surface_exposed = false
```

## Supported Families

Supported bounded families are:

```text
route explanation
stale/old packet explanation
active/distractor/old/stale/inactive slot binding
two-turn active-code carry
boundary mini refusal
anti-template-copy explanation
finite-label AnchorRoute retention
```

Unknown prompts return:

```text
status = unsupported
```

The unsupported output must explain the bounded domain and name the supported families. It must not generate fake open-domain answers.

## JSON Output Envelope

Every inference row must include:

```text
request_id
prompt
prompt_sha256
status
output_text
output_classification
supported_family
required_slot
emitted_slot
checkpoint_sha256
artifact_package_zip_sha256
latency_ms
max_response_tokens
truncated
diagnosis
```

Bad input tests must include:

```text
empty prompt
whitespace prompt
oversized prompt
invalid batch row
unsupported topic
```

The runtime must return structured error or unsupported rows, not panic.

## Determinism, Timeout, And Audit

The determinism check repeats the same bounded prompt and requires:

```text
same output_text
same status
same supported_family
```

The timeout guard must be exercised:

```text
timeout guard exercised
```

It must produce `status = timeout` or a bounded timeout report without crashing.

`audit_log.jsonl` must contain one row per inference attempt with:

```text
request_id
timestamp
prompt_sha256
supported_family
status
latency_ms
checkpoint_sha256
output_sha256
```

## Required Artifacts

084 writes:

```text
queue.json
progress.jsonl
runtime_config.json
artifact_manifest.json
checkpoint_manifest.json
single_inference.json
batch_inference.jsonl
bad_input_results.jsonl
unsupported_input_results.jsonl
determinism_report.json
timeout_report.json
audit_log.jsonl
runtime_metrics.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written at start and refreshed by phase.

## Positive Metrics

Positive requires:

```text
artifact_hash_verified = true
checkpoint_hash_unchanged = true
single_prompt_pass = true
batch_prompt_pass = true
json_output_envelope_pass = true
human_readable_output_pass = true
deterministic_repeated_output_pass = true
bad_input_handled = true
unsupported_input_handled = true
timeout_guard_pass = true
audit_log_written = true
train_step_count = 0
prediction_oracle_used = false
llm_judge_used = false
service_api_exposed = false
deployment_harness_exposed = false
sdk_surface_exposed = false
```

## Verdicts

Positive verdicts:

```text
BOUNDED_CHAT_INFERENCE_RUNTIME_POSITIVE
ARTIFACT_PACKAGE_VERIFIED
CHECKPOINT_LOADED_READ_ONLY
SINGLE_PROMPT_INFERENCE_PASSES
BATCH_INFERENCE_PASSES
JSON_OUTPUT_ENVELOPE_PASSES
HUMAN_READABLE_OUTPUT_WRITTEN
DETERMINISTIC_OUTPUT_CONFIRMED
BAD_INPUT_HANDLED
UNSUPPORTED_INPUT_HANDLED
TIMEOUT_GUARD_PASSES
AUDIT_LOG_WRITTEN
NO_TRAINING_PERFORMED
RUNTIME_LOCAL_ONLY
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure verdicts:

```text
BOUNDED_CHAT_INFERENCE_RUNTIME_FAILS
UPSTREAM_083_ARTIFACT_MISSING
ARTIFACT_HASH_MISMATCH
CHECKPOINT_LOAD_FAILS
CHECKPOINT_MUTATION_DETECTED
SINGLE_PROMPT_INFERENCE_FAILS
BATCH_INFERENCE_FAILS
JSON_OUTPUT_ENVELOPE_MISSING
HUMAN_READABLE_OUTPUT_MISSING
DETERMINISM_FAILS
BAD_INPUT_NOT_HANDLED
UNSUPPORTED_INPUT_NOT_HANDLED
TIMEOUT_GUARD_FAILS
AUDIT_LOG_MISSING
TRAINING_SIDE_EFFECT_DETECTED
ORACLE_SHORTCUT_DETECTED
LLM_JUDGE_USED
SERVICE_API_SURFACE_DETECTED
DEPLOYMENT_HARNESS_MUTATION_DETECTED
SDK_PUBLIC_EXPORT_MUTATION_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

If 084 passes, next milestone is `085_BOUNDED_CHAT_SERVICE_API_ALPHA`. If 084 fails, next milestone is `084B_BOUNDED_CHAT_INFERENCE_RUNTIME_FAILURE_ANALYSIS`.

## Validation

```powershell
cargo check -p instnct-core --example phase_lane_bounded_chat_inference_runtime
cargo run -p instnct-core --example phase_lane_bounded_chat_inference_runtime -- --out target/pilot_wave/stable_loop_phase_lock_084_bounded_chat_inference_runtime/smoke --artifact-root target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke --max-input-chars 512 --max-response-tokens 64 --timeout-ms 1000 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_084_bounded_chat_inference_runtime_check.py
python scripts/probes/run_stable_loop_phase_lock_084_bounded_chat_inference_runtime_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm_check.py --check-only
git diff --check
```
