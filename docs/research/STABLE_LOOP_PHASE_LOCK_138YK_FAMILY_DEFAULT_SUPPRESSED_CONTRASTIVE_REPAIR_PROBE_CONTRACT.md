# STABLE_LOOP_PHASE_LOCK_138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE Contract

138YK is a targeted repair/probe after 138YJ. It may train only one new target
checkpoint under `target/pilot_wave/stable_loop_phase_lock_138yk_family_default_suppressed_contrastive_repair_probe/`.
The immutable source checkpoint is the 138YI target checkpoint:

`target/pilot_wave/stable_loop_phase_lock_138yi_family_specific_value_attractor_repair_probe/smoke/checkpoints/target_138yi_family_contrastive_value/model.pt`

Final evaluation must use only `scripts/probes/shared_raw_generation_helper.py`.
Generated text must exist before scoring, and helper requests may contain only
`prompt`, `checkpoint_path`, `checkpoint_hash`, `seed`, `max_new_tokens`, and
`generation_config`.

## Required Design

The probe builds `family_default_value_bank.json` from 138YJ/138YD/138YI
failure artifacts and writes `hard_negative_default_rows.jsonl`. Every eval row
must carry `forbidden_family_default_values`, `family_default_hard_negative =
true`, and `pass_requires_no_family_default = true`.

A row fails if it emits a forbidden family default. A same-family contrast group
fails if distinct expected values collapse to one generated value. Controls must
include `HARD_NEGATIVE_DEFAULT_CONTROL` and
`PEER_EXPECTED_VALUE_CONFUSION_CONTROL`, and all controls must fail.

If family default values are still used, it fails. If hard negative default rows
are violated, it fails. If same-family prompts still collapse to one generated
value, it fails. If generated_text is not produced by the shared helper before
scoring, the result is invalid.

## Capability Boundary

138YK is not a broad assistant milestone. It does not claim reasoning restored
except that `reasoning_subtrack_real_raw_evidence_partially_restored` may become
true only on a fully gated positive. Raw assistant capability remains
quarantined. Structured/tool capability remains invalidated. It is not GPT-like
readiness, not open-domain assistant readiness, not production chat, not public
API, not deployment readiness, and not safety alignment.

