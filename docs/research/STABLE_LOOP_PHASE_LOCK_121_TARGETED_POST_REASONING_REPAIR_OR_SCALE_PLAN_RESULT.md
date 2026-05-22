# STABLE_LOOP_PHASE_LOCK_121_TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN Result

121 is planning only. It reads existing artifacts and writes a targeted post-reasoning repair plan. It performs no training, no repair, no model inference, no checkpoint mutation, no service startup, no deployment smoke, and no runtime/product/release integration.

121 is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.

## Expected positive result

The expected positive verdict is:

```text
TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_POSITIVE
UPSTREAM_120_POST_REASONING_MAP_VERIFIED
POST_REASONING_BREAKPOINT_ANALYSIS_WRITTEN
FAILURE_PRIORITY_MAP_WRITTEN
ROOT_VS_SYMPTOM_ANALYSIS_WRITTEN
MULTI_TURN_STATE_REPAIR_TARGET_SELECTED
EVAL_GATE_PROPOSAL_WRITTEN
NEXT_MILESTONE_PLAN_WRITTEN
NO_TRAINING_PERFORMED
BOUNDED_RELEASE_UNCHANGED
PRODUCTION_CHAT_NOT_CLAIMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

The expected decision is:

```text
selected_next_milestone = 122_MULTI_TURN_STATE_REPAIR
selected_repair_target = multi_turn_state_first
```

## Evidence required from 120

The runner and checker must preserve this 120 evidence in `decision.json`:

```text
first_breakpoint_tier = TIER_4_MULTI_TURN_STATE_UPDATE
primary_next_repair_target = multi_turn_state_failure
reasoning_regression_rejected = true
reasoning_failure_rate = 0.0
```

The decision must explicitly explain:

- why not hallucination/refusal first
- why not format/injection first
- why not long-context first
- why not more general training
- why not deploy polish
- why not architecture pivot

The governing rule is that the first breakpoint outranks global failure count. Later-tier long-context, format, injection, or hallucination/refusal failures are treated as compounded symptoms unless `root_vs_symptom_analysis.json` proves otherwise.

## Required 122 draft

`next_milestone_plan.json` must draft `122_MULTI_TURN_STATE_REPAIR` with specific multi-turn repair content:

- multi-turn corrections
- active vs stale state tracking
- override chains
- slot updates across turns
- table/doc facts plus state updates
- bounded refusal with state carry
- stale-state decoys

It must prevent 111-style failure through:

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

It must include concrete eval gates:

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

It must preserve reasoning repair with:

```text
tier4_reasoning_accuracy >= 0.97
tier8_reasoning_combo_accuracy >= 0.90
reasoning_failure_rate <= 0.05
```

Missing these gates is `REASONING_PRESERVATION_GATE_MISSING`.

## Validation

Run:

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan.py
python scripts/probes/run_stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan.py --out target/pilot_wave/stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan/smoke --upstream-120-root target/pilot_wave/stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap/smoke --upstream-119-root target/pilot_wave/stable_loop_phase_lock_119_reasoning_repair_scale_confirm/smoke --upstream-118-root target/pilot_wave/stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair/smoke --upstream-116-root target/pilot_wave/stable_loop_phase_lock_116_raw_assistant_capability_ceiling_and_gap_map/smoke --upstream-115-root target/pilot_wave/stable_loop_phase_lock_115_external_style_raw_assistant_stress_confirm/smoke --upstream-112-root target/pilot_wave/stable_loop_phase_lock_112_current_chassis_raw_generation_scale_confirm/smoke --upstream-099-root target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan_check.py
python scripts/probes/run_stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_119_reasoning_repair_scale_confirm_check.py --check-only
git diff --check
```

