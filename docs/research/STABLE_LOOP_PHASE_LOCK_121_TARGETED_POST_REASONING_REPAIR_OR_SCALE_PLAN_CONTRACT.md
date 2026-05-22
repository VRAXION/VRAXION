# STABLE_LOOP_PHASE_LOCK_121_TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN Contract

121 is planning only. It reads existing artifacts and writes a targeted post-reasoning repair plan. It performs no training, no repair, no model inference, no checkpoint mutation, no service startup, no deployment smoke, and no runtime/product/release integration.

121 is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.

## Required upstreams

The runner requires positive upstream artifacts from:

- 120 `POST_REASONING_CEILING_AND_GAP_REMAP_POSITIVE`
- 119 `REASONING_REPAIR_SCALE_CONFIRM_POSITIVE`
- 118 `REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE`
- 116 `RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP_POSITIVE`
- 115 `EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_POSITIVE`
- 112 `CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE`
- 099 `BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE`

The 120 evidence that must be preserved in `decision.json` is:

```text
first_breakpoint_tier = TIER_4_MULTI_TURN_STATE_UPDATE
primary_next_repair_target = multi_turn_state_failure
reasoning_regression_rejected = true
reasoning_failure_rate = 0.0
```

## Required decision

The expected positive decision is:

```text
selected_next_milestone = 122_MULTI_TURN_STATE_REPAIR
selected_repair_target = multi_turn_state_first
```

The selection rule is:

```text
first breakpoint outranks global failure count
```

Long-context, format, injection, or hallucination/refusal failures must not be selected first only because their global count is high. A later-tier target may be selected first only if `root_vs_symptom_analysis.json` proves that it is upstream of the Tier 4 multi-turn state failure.

## Required 122 plan content

`next_milestone_plan.json` must draft `122_MULTI_TURN_STATE_REPAIR`, not a generic SFT milestone.

Required multi-turn data design:

- multi-turn corrections
- active vs stale state tracking
- override chains
- slot updates across turns
- table/doc facts plus state updates
- bounded refusal with state carry
- stale-state decoys

Required anti-111 safeguards:

- train/eval namespace disjointness
- anti-memorization rows
- leakage audit against 112-121 artifacts
- scheduled sampling or rollout-style objective if training is used
- raw-only final eval
- no teacher-forcing-only success
- no oracle rerank
- no expected-answer metadata
- no decoder reference
- no integrated policy during final eval

Required 122 eval gates:

- multi_turn_state_accuracy
- state_tracking_accuracy
- multi_turn_correction_accuracy
- stale_state_rejection_accuracy
- override_chain_accuracy
- active_slot_after_update_accuracy
- retention metrics
- collapse metrics
- namespace drift metrics
- leakage metrics

Reasoning preservation gates are mandatory:

```text
tier4_reasoning_accuracy >= 0.97
tier8_reasoning_combo_accuracy >= 0.90
reasoning_failure_rate <= 0.05
```

If these are missing, the checker must fail with `REASONING_PRESERVATION_GATE_MISSING`.

## Required artifacts

The runner must write:

```text
queue.json
progress.jsonl
analysis_config.json
upstream_120_manifest.json
upstream_119_manifest.json
upstream_118_manifest.json
upstream_116_manifest.json
upstream_115_manifest.json
upstream_112_manifest.json
upstream_099_manifest.json
post_reasoning_failure_priority_map.json
breakpoint_analysis.json
root_vs_symptom_analysis.json
repair_target_selection.json
training_design_options.json
eval_gate_proposal.json
risk_register.json
111_failure_prevention_map.json
next_milestone_plan.json
decision.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` must be refreshed at startup, upstream verification, artifact loading, failure prioritization, root/symptom analysis, repair target selection, eval gate proposal, decision writing, and final verdict.

## Hard gates

Positive 121 requires:

```text
upstream_120_positive = true
first_breakpoint_tier = TIER_4_MULTI_TURN_STATE_UPDATE
primary_next_repair_target = multi_turn_state_failure
reasoning_regression_rejected = true
reasoning_failure_rate = 0.0
failure_priority_map_written = true
breakpoint_analysis_written = true
root_vs_symptom_analysis_written = true
repair_target_selection_written = true
next_milestone_plan_written = true
decision_written = true
```

No-side-effect gates:

```text
train_step_count = 0
optimizer_step_count = 0
inference_run_count = 0
service_started = false
deployment_smoke_run = false
checkpoint_mutated = false
```

Boundary flags must remain false for GPT-like readiness, open-domain assistant readiness, production chat, public API, deployment readiness, safety alignment, and Hungarian assistant readiness.

