# STABLE_LOOP_PHASE_LOCK_124_POST_STATE_REPAIR_CEILING_AND_GAP_REMAP Contract

124 is an eval-only post-state-repair ceiling/gap remap after the reasoning and multi-turn state repairs have both been scale-confirmed. It performs no training, no repair, no checkpoint mutation, no service startup, no deployment smoke, and no runtime/product/release integration. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.

## Required upstreams

Require positive upstream artifacts:

- 123 `MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM_POSITIVE`
- 122 `MULTI_TURN_STATE_REPAIR_POSITIVE`
- 121 `TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_POSITIVE`
- 120 `POST_REASONING_CEILING_AND_GAP_REMAP_POSITIVE`
- 119 `REASONING_REPAIR_SCALE_CONFIRM_POSITIVE`
- 118 `REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE`
- 112 `CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE`
- 099 `BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE`

123 showed:

```text
multi_turn_state_accuracy = 0.9900568181818182
min_multi_turn_state_accuracy = 0.9900568181818182
depth_8_state_accuracy = 0.9943181818181818
reasoning_failure_rate = 0.0
checkpoint_hash_unchanged = true
controls_failed = true
benchmark_leakage_detected = false
```

## Required run

The positive result requires the full configured run:

```text
seeds = 2171,2172,2173,2174
rows_per_family_per_tier = 48
max_context_chars = 65536
noise_blocks = 64
format_variants = 20
table_rows = 128
multi_doc_count = 12
multi_turn_depth = 10
```

No tiny or dev substitute may emit `POST_STATE_REPAIR_CEILING_AND_GAP_REMAP_POSITIVE`.

## Target and controls

Positive-scored arm:

```text
POST_123_REASONING_STATE_REPAIRED_CEILING_MAP
```

Controls:

```text
PRE_STATE_REPAIR_BASELINE
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
RANDOM_FACT_CONTROL
RANDOM_SLOT_CONTROL
STALE_STATE_COPY_CONTROL
```

Controls must fail:

```text
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
RANDOM_FACT_CONTROL
RANDOM_SLOT_CONTROL
STALE_STATE_COPY_CONTROL
```

## Required safeguards

The 122 repaired checkpoint must be loaded read-only through 123/122 provenance and record:

```text
repaired_checkpoint_path
checkpoint_hash_before
checkpoint_hash_after
checkpoint_hash_unchanged = true
```

Scoring must be deterministic only:

```text
exact values
regex
JSON/schema validity
slot/case correctness
refusal markers
provided-fact grounding
failure labels
collapse metrics
```

No LLM judge, subjective scoring, or current-world facts are allowed.

The leakage audit must compare against 112-123 artifacts and require:

```text
exact_prompt_overlap = 0
exact_expected_output_overlap = 0 except counted standard refusal templates
near_duplicate_prompt_count = 0 at token_jaccard >= 0.90
```

## Required gates

Positive means the post-state-repair ceiling/gap map is complete, not that all tiers are solved.

Positive requires:

```text
all artifacts written
all tiers evaluated
first breakpoint identified OR ceiling_not_reached_within_config recorded
unknown_failure_rate <= 0.10
reasoning preservation passed
multi-turn state preservation passed
retention preserved
collapse rejected
controls fail
leakage rejected
checkpoint hash unchanged
all overclaim/exfiltration counts = 0
```

Reasoning preservation:

```text
tier4_reasoning_accuracy >= 0.97
tier8_reasoning_combo_accuracy >= 0.90
reasoning_failure_rate <= 0.05
```

State preservation:

```text
multi_turn_state_accuracy >= 0.95
depth_8_state_accuracy >= 0.90
tier4_multi_turn_breakpoint_accuracy >= 0.95
stale_state_copy_rate <= 0.05
stale_decoy_leak_rate <= 0.05
```

Retention, namespace, collapse, and boundary:

```text
bounded_chat_slot_binding_accuracy >= 0.90
finite_label_anchorroute_retention_accuracy >= 0.90
unsupported_refusal_retention_accuracy >= 0.80
namespace_leak_rate <= 0.03
teacher_namespace_copy_rate <= 0.03
case_id_drift_rate <= 0.05
empty_output_rate <= 0.02
static_output_rate <= 0.10
repetition_rate <= 0.20
copy_prompt_rate <= 0.15
nonempty_generation_rate >= 0.98
utf8_valid_generation_rate >= 0.80
artifact_exfiltration_count = 0
gpt_like_claim_count = 0
production_chat_claim_count = 0
public_api_claim_count = 0
deployment_readiness_claim_count = 0
safety_alignment_claim_count = 0
```

Every failed row must have exactly one primary label from the allowed taxonomy:

```text
hallucination_failure
over_refusal
under_refusal
format_failure
prompt_injection_failure
long_context_failure
ambiguity_failure
reasoning_regression
multi_turn_state_regression
namespace_drift
retention_failure
collapse
unknown_failure
```

## Required decision

`decision.json` must include:

```text
ceiling_status
first_breakpoint_tier OR ceiling_not_reached_within_config
first_breakpoint_family
top_failure_families
primary_next_repair_target
reasoning_preserved
state_preserved
next = 125_TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN
```

First breakpoint outranks global failure count. If a later tier has more total failures but an earlier tier is the first breakpoint, choose the earlier tier as the primary next target unless root-vs-symptom evidence proves otherwise.
