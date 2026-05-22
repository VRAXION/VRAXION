# STABLE_LOOP_PHASE_LOCK_095_CHAT_DECODER_GENERATION_REPAIR Contract

## Summary

095 is a target-only decoder generation repair PoC after 094B.

094B diagnosed the 094 gap primarily as `STOP_CONDITION_MISMATCH`, with finite-label weakness and unproven warm-start advantage. 095 repairs generation at the decoder/prompt-constraint layer only. It does not train, mutate checkpoints, modify runtime/service/deploy code, deploy anything, or prove GPT-like assistant readiness.

## Key Changes

Add only:

```text
scripts/probes/run_stable_loop_phase_lock_095_chat_decoder_generation_repair.py
scripts/probes/run_stable_loop_phase_lock_095_chat_decoder_generation_repair_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_095_CHAT_DECODER_GENERATION_REPAIR_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_095_CHAT_DECODER_GENERATION_REPAIR_RESULT.md
```

Outputs stay under:

```text
target/pilot_wave/stable_loop_phase_lock_095_chat_decoder_generation_repair/
```

Do not modify checkpoints, 093/094/094B artifacts, runtime/service/deploy code, SDK/public exports, product/release docs, root LICENSE, or 083/089 packages.

## Behavior

Default smoke:

```powershell
python scripts/probes/run_stable_loop_phase_lock_095_chat_decoder_generation_repair.py --out target/pilot_wave/stable_loop_phase_lock_095_chat_decoder_generation_repair/smoke --upstream-094b-root target/pilot_wave/stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis/smoke --seed 2028 --heartbeat-sec 20
```

Require positive 094B and `recommended_next_milestone = 095_CHAT_DECODER_GENERATION_REPAIR`.

The repair is target-only and prompt-derived:

```text
stop_after_first_complete_clause
prompt_family_scaffold
finite_label_copy_constraint
unsupported_refusal_guard
candidate_rerank_without_expected_response
```

It must record:

```text
expected_response_used_for_generation = false
response_table_used = false
no_training_performed = true
optimizer_step_count = 0
checkpoint_unchanged = true
```

## Gates

Positive requires:

```text
repaired_generated_accuracy >= 0.90
generation_accuracy_delta >= 0.40
bounded_slot_accuracy >= 0.90
finite_label_accuracy >= 0.90
unsupported_refusal_accuracy >= 0.90
max_new_bytes_stop_rate = 0.0
checkpoint_unchanged = true
no_training_performed = true
optimizer_step_count = 0
```

Positive verdicts:

```text
CHAT_DECODER_GENERATION_REPAIR_POSITIVE
UPSTREAM_094B_GAP_ANALYSIS_VERIFIED
TARGET_ONLY_DECODER_REPAIR_WRITTEN
GENERATION_ACCURACY_REPAIRED
STOP_CONDITION_REPAIRED
FINITE_LABEL_OUTPUT_REPAIRED
CHECKPOINTS_UNCHANGED
NO_TRAINING_PERFORMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

Failures include:

```text
CHAT_DECODER_GENERATION_REPAIR_FAILS
UPSTREAM_094B_ARTIFACT_MISSING
UPSTREAM_094B_NOT_POSITIVE
DECODER_REPAIR_INSUFFICIENT
DECODER_REPAIR_FAMILY_REGRESSION
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
PUBLIC_RELEASE_CLAIM_DETECTED
```

## Next

If 095 passes, proceed to `096_FRESH_CHAT_GENERATION_EVAL`.
