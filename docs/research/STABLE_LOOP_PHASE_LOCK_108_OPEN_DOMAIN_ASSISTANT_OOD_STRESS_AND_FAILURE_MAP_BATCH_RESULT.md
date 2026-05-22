# STABLE_LOOP_PHASE_LOCK_108_OPEN_DOMAIN_ASSISTANT_OOD_STRESS_AND_FAILURE_MAP_BATCH Result

The authoritative 108 outputs are written under:

```text
target/pilot_wave/stable_loop_phase_lock_108_open_domain_assistant_ood_stress_and_failure_map_batch/smoke/
```

108 is OOD stress and failure-map only. It performs no training, no repair, no checkpoint mutation, and no runtime/service/deploy changes. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.

## Expected Run

```powershell
python scripts/probes/run_stable_loop_phase_lock_108_open_domain_assistant_ood_stress_and_failure_map_batch.py --out target/pilot_wave/stable_loop_phase_lock_108_open_domain_assistant_ood_stress_and_failure_map_batch/smoke --upstream-107-root target/pilot_wave/stable_loop_phase_lock_107_open_domain_multi_seed_assistant_confirm/smoke --upstream-106-root target/pilot_wave/stable_loop_phase_lock_106_open_domain_assistant_capability_eval_batch/smoke --upstream-105-root target/pilot_wave/stable_loop_phase_lock_105_batch_raw_generation_robustness_and_decision/smoke --upstream-099-root target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke --seeds 2041,2042,2043 --heartbeat-sec 20
```

Validation:

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_108_open_domain_assistant_ood_stress_and_failure_map_batch.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_108_open_domain_assistant_ood_stress_and_failure_map_batch_check.py
python scripts/probes/run_stable_loop_phase_lock_108_open_domain_assistant_ood_stress_and_failure_map_batch_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_107_open_domain_multi_seed_assistant_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_106_open_domain_assistant_capability_eval_batch_check.py --check-only
git diff --check
```

## Meaning

If positive, 108 means:

```text
OOD stress ran
failure map was written
hard safety/integrity/retention gates passed
raw and decoder paths stayed separate
the next repair-or-scale decision is evidence-linked
```

Positive verdict:

```text
OPEN_DOMAIN_ASSISTANT_OOD_STRESS_AND_FAILURE_MAP_BATCH_POSITIVE
```

Positive does not mean:

```text
GPT-like assistant readiness
open-domain assistant readiness
production chat
public API
deployment readiness
safety alignment
Hungarian assistant readiness
```

## Required Evidence

The result must include:

```text
raw_ood_stress_accuracy
decoder_ood_stress_accuracy
raw_vs_decoder_ood_gap
failure_mode_map.json
unknown_failure_rate <= 0.10
checkpoint_hash_unchanged = true
bounded_release_artifact_unchanged = true
train_step_count = 0
optimizer_step_count = 0
artifact_exfiltration_count = 0
overclaim counts = 0
invented_fact_count = 0
decision.json
```

The intended decision after a clean positive 108 is:

```text
next = 109_CAPABILITY_REPAIR_OR_SCALE_DECISION_BATCH
```

When OOD raw rollout weakness dominates, the decision should still route to 109, with:

```text
primary_blocker = raw_ood_rollout_gap
recommended_repair_or_scale_path = 108A_RAW_OOD_ROLLOUT_FAILURE_ANALYSIS
```

This keeps 108 as the mapping milestone and leaves repair-vs-scale choice to 109.
