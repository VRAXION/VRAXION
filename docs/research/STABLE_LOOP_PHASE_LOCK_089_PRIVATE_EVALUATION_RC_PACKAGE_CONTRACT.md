# STABLE_LOOP_PHASE_LOCK_089_PRIVATE_EVALUATION_RC_PACKAGE Contract

## Summary

089 is a packaging-only Private Evaluation RC bundle for the validated local/private bounded chat stack:

```text
083 artifact RC -> 084 local runtime -> 085 localhost API alpha -> 086 harness -> 087 OOD/red-team -> 088 long-run stability
```

It parses upstream evidence, binds hashes, writes operator materials, and creates a target-only RC zip. It does not train, run inference, start service, run deployment smoke, mutate checkpoints, mutate the 083 model artifact, or modify runtime/service/harness code.

This is not clean deploy proof, not production deployment, not a public API, not hosted SaaS, not GPT-like assistant readiness, not open-domain chat, not production chat, and not safety alignment.

## Implementation

Add only:

```text
scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package.py
scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_089_PRIVATE_EVALUATION_RC_PACKAGE_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_089_PRIVATE_EVALUATION_RC_PACKAGE_RESULT.md
```

Generated outputs go only under:

```text
target/pilot_wave/stable_loop_phase_lock_089_private_evaluation_rc_package/
```

Do not modify `instnct-core/`, `tools/instnct_service_alpha/`, `tools/instnct_deploy/`, `docs/product/`, `docs/releases/`, SDK/public exports, root `LICENSE`, checkpoints, or the model artifact package.

## Required Upstreams

Require positive upstreams:

```text
083 CHAT_MODEL_ARTIFACT_RC_PACKAGE_POSITIVE
084 BOUNDED_CHAT_INFERENCE_RUNTIME_POSITIVE
085 BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE
086 BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_POSITIVE
087 BOUNDED_CHAT_OOD_RED_TEAM_EVAL_POSITIVE
088 BOUNDED_CHAT_LONG_RUN_CONCURRENCY_RESOURCE_STABILITY_POSITIVE
```

Also require 088 metrics:

```text
total_requests = 240
completed_requests = 240
audit_log_coverage_rate = 1.0
child_job_orphan_count = 0
checkpoint_hash_unchanged = true
direct_model_runner_used = false
train_step_count = 0
```

Failures:

```text
UPSTREAM_083_ARTIFACT_MISSING
UPSTREAM_084_ARTIFACT_MISSING
UPSTREAM_085_ARTIFACT_MISSING
UPSTREAM_086_ARTIFACT_MISSING
UPSTREAM_087_ARTIFACT_MISSING
UPSTREAM_088_ARTIFACT_MISSING
UPSTREAM_STACK_NOT_POSITIVE
```

## Required Artifacts

Write:

```text
queue.json
progress.jsonl
package_config.json
upstream_stack_manifest.json
artifact_hash_manifest.json
private_eval_capability_surface.json
private_eval_known_limitations.json
claim_boundary.json
operator_quickstart.md
operator_runbook.md
one_command_smoke.ps1
sample_prompts_expected_outputs.jsonl
audit_and_log_locations.json
rollback_pointer.json
troubleshooting.md
acceptance_checklist.md
rc_package_index.json
private_evaluation_rc_package.zip
summary.json
report.md
```

Record:

```text
source_083_artifact_zip_sha256
packaged_083_artifact_zip_sha256
private_evaluation_rc_package_zip_sha256
packaged_083_artifact_zip_sha256 == source_083_artifact_zip_sha256
```

`progress.jsonl`, `summary.json`, and `report.md` must be written at start and refreshed after upstream verification, manifest generation, operator material generation, zip creation, integrity verification, and final verdict.

## Operator Materials

`operator_quickstart.md` must include prerequisites, expected directory layout, one-command smoke, service start command or pointer, sample prompt, expected bounded output, unsupported prompt example, audit/log locations, and rollback pointer.

`operator_runbook.md` must include validate config, healthcheck, smoke, start service, test bounded prompt, test unsupported prompt, inspect audit logs, stop service, rollback, and troubleshooting path.

`one_command_smoke.ps1` must validate local/private config, run healthcheck, run or point to the exact deployment harness smoke command, avoid production/public config, avoid hosted service startup, and fail nonzero on errors.

`private_eval_known_limitations.json` must include bounded domain only, local/private only, no open-domain chat, no GPT-like assistant readiness, no Hungarian chat proof, no long multi-turn proof, no production safety alignment, no public API, no hosted SaaS, no clinical/high-stakes use, and current latency is not production-throughput evidence.

`claim_boundary.json` must include false flags for production deployment, public API, hosted SaaS, GPT-like assistant readiness, open-domain chat, safety alignment, clinical/high-stakes use, and deploy-ready-by-itself.

`sample_prompts_expected_outputs.jsonl` must include bounded active-slot, context carry, stale/distractor suppression, boundary mini, unsupported open-domain, and bad-input/unsupported behavior note rows with `prompt`, `expected_status`, `expected_behavior`, `example_output_or_pattern`, `source_upstream`, and `claim_boundary_note`.

`acceptance_checklist.md` must state what 089 proves, what 089 does not prove, and what `090_BOUNDED_LOCAL_PRIVATE_DEPLOY_READY_GATE` must still verify in a clean/fresh environment.

## Gates And Verdicts

Positive requires packaging only, `train_step_count = 0`, `inference_run_count = 0`, `service_started = false`, `deployment_smoke_run = false`, `checkpoint_hash_unchanged = true`, `artifact_hash_verified = true`, all upstreams positive, operator materials present, no generated artifact staged, and all overclaim flags false.

Positive verdicts:

```text
PRIVATE_EVALUATION_RC_PACKAGE_POSITIVE
UPSTREAM_STACK_PROVENANCE_VERIFIED
MODEL_ARTIFACT_HASH_BOUND
LOCAL_RUNTIME_PROVENANCE_INCLUDED
SERVICE_API_ALPHA_PROVENANCE_INCLUDED
HARNESS_PROVENANCE_INCLUDED
OOD_RED_TEAM_PROVENANCE_INCLUDED
LONG_RUN_STABILITY_PROVENANCE_INCLUDED
OPERATOR_RUNBOOK_WRITTEN
ONE_COMMAND_SMOKE_WRITTEN
ROLLBACK_POINTER_WRITTEN
RC_ZIP_WRITTEN
NO_TRAINING_PERFORMED
NO_INFERENCE_PERFORMED
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure verdicts:

```text
PRIVATE_EVALUATION_RC_PACKAGE_FAILS
UPSTREAM_STACK_NOT_POSITIVE
ARTIFACT_HASH_MISMATCH
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
INFERENCE_SIDE_EFFECT_DETECTED
SERVICE_STARTED_UNEXPECTEDLY
OPERATOR_RUNBOOK_MISSING
ONE_COMMAND_SMOKE_MISSING
ROLLBACK_POINTER_MISSING
KNOWN_LIMITATIONS_MISSING
CLAIM_BOUNDARY_MISSING
RC_ZIP_MISSING
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
PUBLIC_API_CLAIM_DETECTED
HOSTED_SAAS_CLAIM_DETECTED
```

## Validation

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package_check.py
python scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package.py --out target/pilot_wave/stable_loop_phase_lock_089_private_evaluation_rc_package/smoke --upstream-083-root target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke --upstream-084-root target/pilot_wave/stable_loop_phase_lock_084_bounded_chat_inference_runtime/smoke --upstream-085-root target/pilot_wave/stable_loop_phase_lock_085_bounded_chat_service_api_alpha/smoke --upstream-086-root target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/smoke --upstream-087-root target/pilot_wave/stable_loop_phase_lock_087_bounded_chat_ood_red_team_eval/smoke --upstream-088-root target/pilot_wave/stable_loop_phase_lock_088_bounded_chat_long_run_concurrency_resource_stability/smoke --heartbeat-sec 20
python scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_088_bounded_chat_long_run_concurrency_resource_stability_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_087_bounded_chat_ood_red_team_eval_check.py --check-only
git diff --check
```

If 089 passes, next milestone is `090_BOUNDED_LOCAL_PRIVATE_DEPLOY_READY_GATE`.
