# STABLE_LOOP_PHASE_LOCK_125_TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN Contract

125 is planning only. It reads existing artifacts and writes a targeted post-state repair plan. It performs no training, no repair, no model inference, no checkpoint mutation, no service startup, no deployment smoke, and no runtime/product/release integration.

125 is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.

## Required upstreams

The runner requires positive upstream artifacts from:

- 124 `POST_STATE_REPAIR_CEILING_AND_GAP_REMAP_POSITIVE`
- 123 `MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM_POSITIVE`
- 122 `MULTI_TURN_STATE_REPAIR_POSITIVE`
- 121 `TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_POSITIVE`
- 120 `POST_REASONING_CEILING_AND_GAP_REMAP_POSITIVE`
- 119 `REASONING_REPAIR_SCALE_CONFIRM_POSITIVE`
- 118 `REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE`
- 112 `CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE`
- 099 `BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE`

The 124 evidence that must be preserved in `decision.json` is:

```text
first_breakpoint_tier = TIER_4_HALLUCINATION_REFUSAL_BALANCE
first_breakpoint_family = hallucination_failure
primary_next_repair_target = hallucination_failure
reasoning_preserved = true
state_preserved = true
unknown_failure_rate = 0.0
```

## Required decision

The expected positive decision is:

```text
selected_next_milestone = 126_HALLUCINATION_REFUSAL_BALANCE_REPAIR
selected_repair_target = hallucination_refusal_balance_first
```

The selection rule is:

```text
first breakpoint outranks global failure count
```

Format, prompt-injection, and long-context targets must not be selected first only because their global counts are higher. A later-tier target may be selected first only if `root_vs_symptom_analysis.json` proves that it is upstream of the Tier 4 hallucination/refusal balance failure.

## Required 126 plan content

`next_milestone_plan.json` must draft `126_HALLUCINATION_REFUSAL_BALANCE_REPAIR`, not generic SFT and not refusal-only training.

Required calibration data categories:

- provided-fact answerable rows
- insufficient-fact refusal rows
- ambiguity without priority rows
- ambiguity with explicit priority rows
- hallucination traps
- over-refusal traps
- under-refusal traps
- multi-doc evidence sufficiency
- table evidence sufficiency
- state-carry with insufficient facts
- long-context distractor plus missing fact

Required anti-111 safeguards:

- train/eval namespace disjointness
- anti-memorization rows
- leakage audit against 112-125 artifacts
- scheduled sampling or rollout-style objective if training is used
- raw-only final eval
- no teacher-forcing-only success
- no oracle rerank
- no expected-answer metadata
- no decoder reference
- no integrated policy during final eval
- no verifier rerank
- no LLM judge

Required always-refuse degeneration gates:

- answerable_fact_response_accuracy
- insufficient_fact_refusal_accuracy
- over_refusal_rate
- under_refusal_rate
- ambiguity_refusal_accuracy
- explicit_priority_answer_accuracy
- evidence_sufficiency_classification_accuracy

The 126 plan must explicitly reject a solution where hallucination improves only because the model refuses everything.

Prior repair preservation gates are mandatory:

```text
tier4_reasoning_accuracy >= 0.97
tier8_reasoning_combo_accuracy >= 0.90
reasoning_failure_rate <= 0.05

multi_turn_state_accuracy >= 0.95
depth_8_state_accuracy >= 0.90
tier4_multi_turn_breakpoint_accuracy >= 0.95
stale_state_copy_rate <= 0.05
stale_decoy_leak_rate <= 0.05
```

## Required artifacts

The runner must write:

```text
queue.json
progress.jsonl
analysis_config.json
upstream_124_manifest.json
upstream_123_manifest.json
upstream_122_manifest.json
upstream_121_manifest.json
upstream_120_manifest.json
upstream_119_manifest.json
upstream_118_manifest.json
upstream_112_manifest.json
upstream_099_manifest.json
post_state_failure_priority_map.json
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

Positive 125 requires:

```text
upstream_124_positive = true
first_breakpoint_tier = TIER_4_HALLUCINATION_REFUSAL_BALANCE
first_breakpoint_family = hallucination_failure
primary_next_repair_target = hallucination_failure
reasoning_preserved = true
state_preserved = true
unknown_failure_rate = 0.0
failure_priority_map_written = true
breakpoint_analysis_written = true
root_vs_symptom_analysis_written = true
repair_target_selection_written = true
eval_gate_proposal_written = true
risk_register_written = true
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
bounded_release_artifact_unchanged = true
```
