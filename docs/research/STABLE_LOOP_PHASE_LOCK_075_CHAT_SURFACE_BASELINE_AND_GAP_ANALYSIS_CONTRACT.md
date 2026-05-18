# STABLE_LOOP_PHASE_LOCK_075_CHAT_SURFACE_BASELINE_AND_GAP_ANALYSIS Contract

Status: contract for eval-only chat-surface baseline/gap analysis of the
072/074-confirmed `SCENARIO_GATED_SIDEPACKET_REPAIR` checkpoint.

075 measures whether the current checkpoint already exposes any real
conversational/free-form behavior, or whether it remains a finite-label answer
selection surface.

This is eval-only chat-surface baseline/gap analysis.

no training
no checkpoint repair
no checkpoint mutation
no decoder behavior
no open-ended assistant capability proven
no free-form generation proven unless directly measured
no perplexity support
no full English LM
no language grounding
no production training
no chat release readiness
no service API change
no deployment harness change
no release docs change
no public crate export change
no root LICENSE change

## Implementation Scope

075 adds only:

```text
instnct-core/examples/phase_lane_chat_surface_baseline_gap_analysis.rs
scripts/probes/run_stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_075_CHAT_SURFACE_BASELINE_AND_GAP_ANALYSIS_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_075_CHAT_SURFACE_BASELINE_AND_GAP_ANALYSIS_RESULT.md
```

Generated outputs are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis/
```

075 must not implement a new decoder or generation loop. It may only detect
whether a generation surface already exists in the checkpoint artifact.

## Upstream Inputs

Default checkpoint:

```text
target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json
```

Default upstream root:

```text
target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke
```

If upstream artifacts are missing, emit:

```text
UPSTREAM_074_ARTIFACT_MISSING
```

Do not rerun 072/073/074, train, repair, fine-tune, add decoder behavior, or
mutate the checkpoint.

## Eval-Only Hard Wall

Record:

```text
train_step_count = 0
checkpoint_hash_before
checkpoint_hash_after
checkpoint_hash_unchanged = true
prediction_oracle_used = false
decoder_generation_loop_available
chat_generation_supported
free_form_answering_supported
multi_turn_dialogue_supported
perplexity_supported = false
finite_label_surface = true
chat_release_readiness_proven = false
```

Failure verdicts:

```text
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
ORACLE_SHORTCUT_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

## Probe Families

```text
FREE_FORM_RESPONSE_PROBE
MULTI_TOKEN_CONTINUATION_PROBE
SINGLE_TURN_INSTRUCTION_PROBE
TWO_TURN_DIALOGUE_PROBE
CONTEXT_CARRY_CHAT_PROBE
BOUNDARY_REFUSAL_PROBE
DEGENERATION_PROBE
FINITE_LABEL_CONTROL_PROBE
```

A response counts as chat/free-form only if:

```text
it can produce multi-token natural-language output
output is not restricted to checkpoint label set
output changes meaningfully with instruction/context
output is not a static label/copy/empty/space response
```

Every probe output must be classified as:

```text
finite_label
empty
space_only
copied_prompt_fragment
static_repeated_output
unsupported
free_form_candidate
```

## Required Artifacts

```text
queue.json
progress.jsonl
chat_probe_config.json
upstream_074_manifest.json
checkpoint_manifest.json
chat_probe_dataset.jsonl
chat_probe_outputs.jsonl
human_readable_samples.jsonl
gap_analysis.json
degeneration_metrics.json
finite_label_control_metrics.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from start and
refreshed during major phases.

`human_readable_samples.jsonl` must include:

```text
probe_family
prompt
expected_behavior
raw_model_output
output_classification
pass_fail_or_unsupported
short_diagnosis
```

Degeneration metrics must include:

```text
empty_output_rate
space_output_rate
repeated_output_rate
static_output_rate
label_only_rate
copy_prompt_rate
unique_output_count
```

## Verdict Rules

Unsupported chat is an acceptable honest result.

Unsupported verdicts:

```text
CHAT_SURFACE_BASELINE_GAP_ANALYSIS_FAILS
CHAT_GENERATION_SURFACE_UNSUPPORTED
FREE_FORM_ANSWERING_UNSUPPORTED
MULTI_TURN_DIALOGUE_UNSUPPORTED
PERPLEXITY_UNSUPPORTED
OPEN_ENDED_CHAT_READY_CLAIM_REJECTED
```

Integrity positives:

```text
UPSTREAM_074_CHECKPOINT_VERIFIED
NO_TRAINING_PERFORMED
CHECKPOINT_UNCHANGED
FINITE_LABEL_SURFACE_CONFIRMED
CHAT_GAP_ANALYSIS_WRITTEN
HUMAN_READABLE_SAMPLES_WRITTEN
PRODUCTION_TRAINING_NOT_CLAIMED
```

False-claim failures:

```text
OPEN_ENDED_CHAT_READY_CLAIM_DETECTED
FREE_FORM_GENERATION_FALSE_CLAIM
PERPLEXITY_CLAIM_DETECTED
HUMAN_SAMPLE_REPORT_MISSING
```

## Validation Commands

```powershell
cargo check -p instnct-core --example phase_lane_chat_surface_baseline_gap_analysis
cargo run -p instnct-core --example phase_lane_chat_surface_baseline_gap_analysis -- --out target/pilot_wave/stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis/smoke --checkpoint target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --seed 2026 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis_check.py
python scripts/probes/run_stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm_check.py --check-only
git diff --check
```

If 075 confirms no usable chat surface, the next milestone is:

```text
076_CHAT_GENERATION_POC
```
