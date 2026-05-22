# STABLE_LOOP_PHASE_LOCK_125_TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN Result

125 is planning only. It reads existing artifacts and writes a targeted post-state repair plan. It performs no training, no repair, no model inference, no checkpoint mutation, no service startup, no deployment smoke, and no runtime/product/release integration.

125 is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.

## Expected positive result

Expected positive verdicts:

```text
TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN_POSITIVE
UPSTREAM_124_CEILING_MAP_VERIFIED
POST_STATE_BREAKPOINT_ANALYSIS_WRITTEN
FAILURE_PRIORITY_MAP_WRITTEN
ROOT_VS_SYMPTOM_ANALYSIS_WRITTEN
HALLUCINATION_REFUSAL_TARGET_SELECTED
EVAL_GATE_PROPOSAL_WRITTEN
NEXT_MILESTONE_PLAN_WRITTEN
NO_TRAINING_PERFORMED
BOUNDED_RELEASE_UNCHANGED
PRODUCTION_CHAT_NOT_CLAIMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

Expected decision:

```text
selected_next_milestone = 126_HALLUCINATION_REFUSAL_BALANCE_REPAIR
selected_repair_target = hallucination_refusal_balance_first
```

## Required evidence

The result must cite:

```text
first_breakpoint_tier = TIER_4_HALLUCINATION_REFUSAL_BALANCE
first_breakpoint_family = hallucination_failure
primary_next_repair_target = hallucination_failure
reasoning_preserved = true
state_preserved = true
unknown_failure_rate = 0.0
```

It must cite Tier 4 first-breakpoint evidence:

```text
hallucination_failure = 48
over_refusal = 48
ambiguity_failure = 48
```

It must cite later/global evidence without letting it override the first breakpoint:

```text
format_failure = 352
prompt_injection_failure = 224
long_context_failure = 160
ambiguity_failure = 112
hallucination_failure = 112
under_refusal = 64
over_refusal = 48
```

## Required 126 plan

`next_milestone_plan.json` must draft `126_HALLUCINATION_REFUSAL_BALANCE_REPAIR`.

The draft must be calibration-focused, not generic SFT and not refusal-only training. It must prevent always-refuse degeneration with hard gates for:

```text
answerable_fact_response_accuracy
insufficient_fact_refusal_accuracy
over_refusal_rate
under_refusal_rate
ambiguity_refusal_accuracy
explicit_priority_answer_accuracy
evidence_sufficiency_classification_accuracy
```

Required 126 data categories:

```text
provided-fact answerable rows
insufficient-fact refusal rows
ambiguity without priority rows
ambiguity with explicit priority rows
hallucination traps
over-refusal traps
under-refusal traps
multi-doc evidence sufficiency
table evidence sufficiency
state-carry with insufficient facts
long-context distractor plus missing fact
```

Required anti-111 safeguards:

```text
train/eval namespace disjointness
anti-memorization rows
leakage audit against 112-125 artifacts
scheduled sampling or rollout-style objective if training is used
raw-only final eval
no teacher-forcing-only success
no oracle rerank
no expected-answer metadata
no decoder reference
no integrated policy during final eval
no verifier rerank
no LLM judge
```

Prior repair preservation gates:

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

## Failure routes

Failure verdicts include:

```text
TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN_FAILS
UPSTREAM_124_ARTIFACT_MISSING
UPSTREAM_124_NOT_POSITIVE
BREAKPOINT_ANALYSIS_MISSING
FAILURE_PRIORITY_MAP_MISSING
REPAIR_TARGET_SELECTION_MISSING
NEXT_MILESTONE_PLAN_MISSING
REASONING_PRESERVATION_GATE_MISSING
STATE_PRESERVATION_GATE_MISSING
ALWAYS_REFUSE_DEGENERATION_GATE_MISSING
TRAINING_SIDE_EFFECT_DETECTED
INFERENCE_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
BOUNDED_RELEASE_MUTATION_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
ROOT_LICENSE_CHANGED
```

## Validation

Run:

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_125_targeted_post_state_repair_or_scale_plan.py
python scripts/probes/run_stable_loop_phase_lock_125_targeted_post_state_repair_or_scale_plan.py --out target/pilot_wave/stable_loop_phase_lock_125_targeted_post_state_repair_or_scale_plan/smoke --upstream-124-root target/pilot_wave/stable_loop_phase_lock_124_post_state_repair_ceiling_and_gap_remap/smoke --upstream-123-root target/pilot_wave/stable_loop_phase_lock_123_multi_turn_state_repair_scale_confirm/smoke --upstream-122-root target/pilot_wave/stable_loop_phase_lock_122_multi_turn_state_repair/smoke --upstream-121-root target/pilot_wave/stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan/smoke --upstream-120-root target/pilot_wave/stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap/smoke --upstream-119-root target/pilot_wave/stable_loop_phase_lock_119_reasoning_repair_scale_confirm/smoke --upstream-118-root target/pilot_wave/stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair/smoke --upstream-112-root target/pilot_wave/stable_loop_phase_lock_112_current_chassis_raw_generation_scale_confirm/smoke --upstream-099-root target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_125_targeted_post_state_repair_or_scale_plan_check.py
python scripts/probes/run_stable_loop_phase_lock_125_targeted_post_state_repair_or_scale_plan_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_124_post_state_repair_ceiling_and_gap_remap_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_123_multi_turn_state_repair_scale_confirm_check.py --check-only
git diff --check
```
