# STABLE_LOOP_PHASE_LOCK_087_BOUNDED_CHAT_OOD_RED_TEAM_EVAL Result

## Status

087 implements an eval-only service/harness-level OOD and red-team check for the local/private bounded chat stack.

This is not training, not checkpoint repair, not checkpoint mutation, not a new model, not a public API, not production deployment, not GPT-like assistant readiness, not open-domain chat, and not safety alignment.

## Evaluated Stack

087 evaluates the full local/private path:

```text
083 model artifact RC
-> 084 local inference runtime
-> 085 localhost/private bounded chat API alpha
-> 086 deployment harness smoke
```

The evaluator starts the existing service alpha in `serve` mode with `port = 0`, keeps `127.0.0.1`, and attacks:

```text
POST /v1/bounded-chat/infer
```

It does not call the model runner directly and records `direct_model_runner_used = false`.

## Required Evidence

The smoke output under `target/pilot_wave/stable_loop_phase_lock_087_bounded_chat_ood_red_team_eval/smoke` contains:

```text
queue.json
progress.jsonl
eval_config.json
upstream_086_manifest.json
service_child_manifest.json
red_team_dataset.jsonl
red_team_results.jsonl
valid_control_results.jsonl
unsupported_results.jsonl
injection_results.jsonl
malformed_input_results.jsonl
policy_rejection_results.jsonl
rate_limit_report.json
side_effect_audit.json
json_envelope_validation.json
audit_log_validation.json
artifact_integrity_validation.json
checkpoint_integrity_validation.json
summary.json
report.md
```

The required positive status is:

```text
BOUNDED_CHAT_OOD_RED_TEAM_EVAL_POSITIVE
UPSTREAM_086_STACK_VERIFIED
VALID_BOUNDED_CONTROLS_PASS
OPEN_DOMAIN_UNSUPPORTED_HANDLED
PROMPT_INJECTION_REJECTED
POLICY_SENSITIVE_REQUESTS_REJECTED
MALFORMED_INPUTS_HANDLED
BAD_INPUT_SIDE_EFFECTS_REJECTED
AUTH_POLICY_SIDE_EFFECTS_REJECTED
JSON_ENVELOPE_VALIDATED
AUDIT_LOGGING_VALIDATED
ARTIFACT_HASH_VERIFIED
CHECKPOINT_UNCHANGED
RATE_LIMIT_METADATA_PASSES
PRODUCTION_CHAT_NOT_CLAIMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

## Interpretation

Passing 087 means the current localhost/private bounded stack rejects or safely bounds the tested OOD, malformed, injection-like, policy-sensitive, auth, and rate-limit cases through the service API surface.

Passing 087 does not mean production deployment, public API, open-domain chat, GPT-like assistant readiness, or safety alignment.

Next milestone on pass: `088_BOUNDED_CHAT_LONG_RUN_CONCURRENCY_RESOURCE_STABILITY`.

Next milestone on fail: `087B_BOUNDED_CHAT_OOD_RED_TEAM_FAILURE_ANALYSIS`.
