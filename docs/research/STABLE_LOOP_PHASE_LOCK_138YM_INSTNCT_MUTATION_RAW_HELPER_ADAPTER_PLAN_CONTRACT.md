# STABLE_LOOP_PHASE_LOCK_138YM_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PLAN Contract

## Purpose

138YM is artifact-only planning after 138YL.

138YL established:

```text
decision = instnct_mutation_helper_integration_analysis_complete
next = 138YM_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PLAN
instnct_helper_adapter_required = true
real_raw_value_grounding_comparison_ready = false
```

138YM designs the strict adapter contract required before an INSTNCT mutation/grower generator can be compared with the current byte-GRU `shared_raw_generation_helper.py` path.

It does not train, infer, call the helper, run torch forward passes, mutate checkpoints, modify helper/backend/runtime/service/product/release surfaces, import old phase runners, delete files, deploy, or change root `LICENSE`.

## Required Plan Artifacts

Generated outputs must stay under:

```text
target/pilot_wave/stable_loop_phase_lock_138ym_instnct_mutation_raw_helper_adapter_plan/
```

Required artifacts:

```text
queue.json
progress.jsonl
upstream_138yl_manifest.json
analysis_config.json
adapter_contract.json
helper_surface_change_plan.json
instnct_checkpoint_contract.json
prompt_encoder_contract.json
iterative_propagation_schedule.json
output_decoder_contract.json
forbidden_metadata_policy.json
canary_and_ast_gate_plan.json
determinism_plan.json
comparison_eval_plan.json
target_138yn_milestone_plan.json
diagnostic_gap_register.json
risk_register.json
decision.json
summary.json
report.md
```

## Required Content

The adapter contract must preserve the same helper request key policy:

```text
prompt
checkpoint_path
checkpoint_hash
seed
max_new_tokens
generation_config
```

It must forbid expected/scorer/oracle material, including:

```text
expected_output
expected_answer
labels
oracle_data
scorer_metadata
gold_output
eval_family
answer
expected_values
```

The future adapter must define:

```text
backend_name = repo_local_instnct_mutation_graph
prompt_encoder_contract
iterative_propagation_schedule
output_decoder_contract
instnct_checkpoint_contract
deterministic replay
expected-output canary
AST shortcut scan
generated_text before scoring
```

The plan must explicitly preserve that 138YM itself does not modify the helper. Future helper backend modification is allowed only in the next checked adapter probe.

## Decision

If complete:

```text
decision = instnct_mutation_raw_helper_adapter_plan_complete
next = 138YN_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PROBE
```

The later value-grounding comparison is not ready until the adapter probe passes.

## Boundary

Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.
