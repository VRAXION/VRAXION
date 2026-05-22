# STABLE_LOOP_PHASE_LOCK_083_CHAT_MODEL_ARTIFACT_RC_PACKAGE Contract

083 is a packaging-only Model Artifact RC gate for the 080
`TOKEN_COMPOSITION_DIVERSITY_REPAIR` checkpoint, using 082 as the multi-seed
proof.

083 does not train, run inference, repair checkpoint, mutate checkpoint, add
runtime, add service API, add SDK/public export, touch release docs, or change
root LICENSE.

Allowed work:

```text
copy existing 080 checkpoint into package dir
hash files
collect eval provenance
write package manifests
write artifact zip under target/
```

This is private bounded model artifact RC only.

not deploy-ready by itself
not inference runtime
not service/API integration
not GPT-like assistant
not production chat
not full English LM
not language grounding
not safety alignment
not public beta / GA / hosted SaaS

no instnct-core runtime change
no service API change
no deployment harness change
no SDK/public export change
no release docs change
no root LICENSE change
no upstream checkpoint mutation

## Files

```text
scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package.py
scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_083_CHAT_MODEL_ARTIFACT_RC_PACKAGE_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_083_CHAT_MODEL_ARTIFACT_RC_PACKAGE_RESULT.md
```

Generated outputs are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/
```

## Inputs

Default source checkpoint:

```text
target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke/checkpoints/chat_composition_diversity_repair/model_checkpoint.json
```

Required upstreams:

```text
target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/smoke
target/pilot_wave/stable_loop_phase_lock_081_chat_diversity_fresh_confirm/smoke
target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke
target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke
```

Missing upstream failures:

```text
UPSTREAM_082_ARTIFACT_MISSING
UPSTREAM_080_ARTIFACT_MISSING
```

## Builder Behavior

Write `progress.jsonl`, `summary.json`, and `report.md` from start and refresh
by phase.

Required upstream 082 verification:

```text
CHAT_DIVERSITY_MULTI_SEED_CONFIRM_POSITIVE
all_seed_pass = true
all child_exit_code = 0
all child_recheck_pass = true
all checkpoint_hash_unchanged = true
all train_step_count = 0
```

Byte-level checkpoint integrity:

```text
source_checkpoint_sha256
packaged_checkpoint_sha256
source_checkpoint_size_bytes
packaged_checkpoint_size_bytes
packaged_checkpoint_sha256 == source_checkpoint_sha256
packaged_checkpoint_size_bytes == source_checkpoint_size_bytes
artifact_package_zip_sha256
```

The package may record upstream-reported checkpoint hashes separately because
prior milestones may use different semantic hash fields. 083's hard package
gate is byte-level source-to-copy equality.

## Required Artifacts

```text
queue.json
progress.jsonl
package_config.json
source_checkpoint_manifest.json
packaged_checkpoint_manifest.json
integrity_hashes.json
upstream_082_manifest.json
eval_provenance_manifest.json
capability_surface.json
known_limitations.json
claim_boundary.json
sample_prompts_outputs.jsonl
repro_commands.ps1
rollback_pointer.json
artifact_index.json
artifact_package.zip
summary.json
report.md
```

`artifact_package.zip` must include:

```text
packaged checkpoint
artifact_index.json
integrity_hashes.json
capability_surface.json
known_limitations.json
claim_boundary.json
eval_provenance_manifest.json
sample_prompts_outputs.jsonl
repro_commands.ps1
rollback_pointer.json
```

## Capability Surface

`capability_surface.json` must mark supported:

```text
bounded_domain_chat_composition = true
finite_label_anchorroute_retention = true
context_slot_binding = true
multi_seed_chat_diversity_confirmed = true
```

And unsupported or not claimed:

```text
open_domain_chat_supported = false
gpt_like_assistant_readiness_claimed = false
full_English_LM_supported = false
language_grounding_claimed = false
production_chat_claimed = false
safety_alignment_claimed = false
public_beta_claimed = false
GA_claimed = false
hosted_SaaS_claimed = false
deploy_ready_by_itself = false
```

`known_limitations.json` must include:

```text
bounded English domain only
no open-domain chat
no Hungarian chat proof
no long multi-turn proof
no production safety alignment
no service/API runtime
no deployment harness integration
no public beta / GA
no hosted SaaS
no clinical/high-stakes use
```

## Samples, Repro, Rollback

`sample_prompts_outputs.jsonl` must use real 082 child outputs from:

```text
fresh instruction
short explanation
context slot
two-turn carry
boundary mini
anti-template-copy
finite-label retention
```

Each row records:

```text
source_seed
eval_family
prompt
model_output
expected_behavior
pass_fail
novelty_flag
template_copy_flag
skeleton_reuse_flag
slot_binding_diagnosis
```

`repro_commands.ps1` must include:

```text
082 checker
081 checker
080 checker
package hash verification
no training command as required step
```

Optional eval reproduction commands may be listed only when labeled optional.

`rollback_pointer.json` must include:

```text
previous checkpoint path/hash if available
source 080 checkpoint path/hash
artifact package path
rollback instruction
no automatic production rollback claim
```

## Verdicts

Positive verdicts:

```text
CHAT_MODEL_ARTIFACT_RC_PACKAGE_POSITIVE
UPSTREAM_082_MULTI_SEED_PROOF_VERIFIED
SOURCE_CHECKPOINT_VERIFIED
PACKAGED_CHECKPOINT_HASH_MATCHES_SOURCE
EVAL_PROVENANCE_MANIFEST_WRITTEN
CAPABILITY_SURFACE_RECORDED
KNOWN_LIMITATIONS_RECORDED
SAMPLE_PROMPTS_OUTPUTS_WRITTEN
ROLLBACK_POINTER_WRITTEN
REPRO_COMMANDS_WRITTEN
NO_TRAINING_PERFORMED
RUNTIME_SURFACE_UNCHANGED
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure verdicts:

```text
CHAT_MODEL_ARTIFACT_RC_PACKAGE_FAILS
UPSTREAM_082_ARTIFACT_MISSING
UPSTREAM_080_ARTIFACT_MISSING
UPSTREAM_082_NOT_POSITIVE
SOURCE_CHECKPOINT_MISSING
CHECKPOINT_COPY_HASH_MISMATCH
CHECKPOINT_MUTATION_DETECTED
EVAL_PROVENANCE_INCOMPLETE
CAPABILITY_SURFACE_MISSING
KNOWN_LIMITATIONS_MISSING
SAMPLE_PROMPTS_OUTPUTS_MISSING
ROLLBACK_POINTER_MISSING
REPRO_COMMANDS_MISSING
TRAINING_SIDE_EFFECT_DETECTED
ORACLE_SHORTCUT_DETECTED
LLM_JUDGE_USED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

If 083 passes, next milestone:

```text
084_BOUNDED_CHAT_INFERENCE_RUNTIME
```

If 083 fails, next milestone:

```text
083B_CHAT_MODEL_ARTIFACT_PACKAGE_FAILURE_ANALYSIS
```

## Validation

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package_check.py

python scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package.py --out target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke --upstream-082-root target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/smoke --upstream-081-root target/pilot_wave/stable_loop_phase_lock_081_chat_diversity_fresh_confirm/smoke --upstream-080-root target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --heartbeat-sec 20

python scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py --check-only
git diff --check
```
