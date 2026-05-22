# STABLE_LOOP_PHASE_LOCK_128_POST_CALIBRATION_REPAIR_CEILING_AND_GAP_REMAP_CONTRACT

## Purpose

128 is an eval-only post-calibration ceiling/gap remap. It runs after the
reasoning, multi-turn state, and hallucination/refusal calibration repairs have
all been scale-confirmed.

Positive 128 means the post-calibration ceiling/gap map is complete. It does
not mean all tiers are solved.

## Required Run

The full configured run is mandatory:

```text
seeds = 2201,2202,2203,2204
rows_per_family_per_tier = 48
max_context_chars = 81920
noise_blocks = 80
format_variants = 24
table_rows = 160
multi_doc_count = 14
multi_turn_depth = 10
prompt_injection_variants = 12
ambiguity_variants = 12
```

The positive arm is:

```text
POST_127_REASONING_STATE_CALIBRATION_REPAIRED_CEILING_MAP
```

The expected positive next milestone is:

```text
129_TARGETED_POST_CALIBRATION_REPAIR_OR_SCALE_PLAN
```

## Hard Gates

Positive requires:

- `POST_CALIBRATION_REPAIR_CEILING_AND_GAP_REMAP_POSITIVE`
- `POST_CALIBRATION_CEILING_MAP_COMPLETE`
- `UPSTREAM_127_CALIBRATION_SCALE_CONFIRM_VERIFIED`
- `REASONING_REPAIR_PRESERVED`
- `STATE_REPAIR_PRESERVED`
- `CALIBRATION_REPAIR_PRESERVED`
- `RETENTION_PRESERVED`
- `COLLAPSE_REJECTED`
- `NAMESPACE_MEMORIZATION_REJECTED`
- `CONTROLS_FAILED`
- `LEAKAGE_REJECTED`
- `BOUNDED_RELEASE_UNCHANGED`
- `PRODUCTION_CHAT_NOT_CLAIMED`
- `GPT_LIKE_READINESS_NOT_CLAIMED`

Prior repairs are hard gates:

```text
tier4_reasoning_accuracy >= 0.97
tier8_reasoning_combo_accuracy >= 0.90
reasoning_failure_rate <= 0.05

multi_turn_state_accuracy >= 0.95
depth_8_state_accuracy >= 0.90
tier4_multi_turn_breakpoint_accuracy >= 0.95
stale_state_copy_rate <= 0.05
stale_decoy_leak_rate <= 0.05

answerable_fact_response_accuracy >= 0.95
insufficient_fact_refusal_accuracy >= 0.95
hallucination_trap_pass_rate >= 0.95
always_refuse_rate <= 0.05
over_refusal_rate <= 0.08
under_refusal_rate <= 0.08
insufficient_fact_hallucination_rate <= 0.05
```

Controls must fail:

```text
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
RANDOM_FACT_CONTROL
RANDOM_SLOT_CONTROL
ALWAYS_REFUSE_CONTROL
RANDOM_REFUSAL_CONTROL
```

Checkpoint provenance is read through the 127/126 manifests and must record:

```text
repaired_checkpoint_path
checkpoint_hash_before
checkpoint_hash_after
checkpoint_hash_unchanged = true
target_126_checkpoint_read_only = true
```

`decision.json` must include the machine-readable first breakpoint or
`ceiling_not_reached_within_config`, the top failure families, the primary next
repair target, prior-repair preservation booleans, and:

```text
next = 129_TARGETED_POST_CALIBRATION_REPAIR_OR_SCALE_PLAN
```

## Boundary

128 is an eval-only post-calibration ceiling/gap remap. It is not GPT-like
assistant readiness, not open-domain assistant readiness, not production chat,
not public API, not deployment readiness, not safety alignment, and not
Hungarian assistant readiness.
