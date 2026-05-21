# STABLE_LOOP_PHASE_LOCK_106_OPEN_DOMAIN_ASSISTANT_CAPABILITY_EVAL_BATCH Result

The authoritative 106 outputs are written under:

```text
target/pilot_wave/stable_loop_phase_lock_106_open_domain_assistant_capability_eval_batch/smoke/
```

106 is capability eval only. It performs no training, no repair, no checkpoint mutation, and no runtime/service/deploy changes. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.

## Expected Run

```powershell
python scripts/probes/run_stable_loop_phase_lock_106_open_domain_assistant_capability_eval_batch.py --out target/pilot_wave/stable_loop_phase_lock_106_open_domain_assistant_capability_eval_batch/smoke --upstream-105-root target/pilot_wave/stable_loop_phase_lock_105_batch_raw_generation_robustness_and_decision/smoke --upstream-099-root target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke --seed 2034 --heartbeat-sec 20
```

Validation:

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_106_open_domain_assistant_capability_eval_batch.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_106_open_domain_assistant_capability_eval_batch_check.py
python scripts/probes/run_stable_loop_phase_lock_106_open_domain_assistant_capability_eval_batch_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_105_batch_raw_generation_robustness_and_decision_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_100_open_vocab_assistant_capability_scale_check.py --check-only
git diff --check
```

## Meaning

If positive, 106 means:

```text
raw generation has minimum viability
decoder-repaired path is strong
open-domain-style eval is recorded
multi-turn/context/refusal/injection behavior is measured
retention survives
no overclaim or exfiltration occurred
```

Positive verdict:

```text
OPEN_DOMAIN_ASSISTANT_CAPABILITY_EVAL_BATCH_POSITIVE
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

## Failure Meaning

If raw is weak but decoder passes, 106 must not emit the positive verdict. It emits:

```text
RAW_GENERATION_TOO_WEAK
```

and writes:

```text
next = 107_RAW_TO_DECODER_BRIDGE_REPAIR
diagnosis = useful assistant behavior exists mostly behind decoder-repaired path
```

Other failure routes:

```text
DECODER_REPAIRED_GENERATION_FAILS -> 106B_OPEN_DOMAIN_ASSISTANT_FAILURE_ANALYSIS
RETENTION_REGRESSION_DETECTED -> 106R_RETENTION_REGRESSION_ANALYSIS
```

## Required Evidence

The result must include:

```text
upstream_105_checkpoint_source = 102_repair_checkpoint
checkpoint_hash_unchanged = true
bounded_release_artifact_unchanged = true
raw_generated_prompt_response_accuracy
decoder_generated_prompt_response_accuracy
raw_vs_decoder_gap
eval_row_hashes_match = true
llm_judge_used = false
prediction_oracle_used = false
```

Open-domain style QA is rubric-bounded and uses provided facts only. It is not a current-world factual knowledge proof.
