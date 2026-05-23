# STABLE_LOOP_PHASE_LOCK_138YO_INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_PROBE

138YO is the helper-only comparison probe after 138YN.

It compares:

- `byte_gru_138yk_target_existing_helper_rollout`
- `instnct_mutation_adapter_138yn_same_prompt`
- `instnct_mutation_adapter_138yn_pocket_gate_ablation`

The comparison uses the same 138YK eval rows. The byte-GRU arm is read from the
existing 138YK raw helper rollout artifacts. The INSTNCT arm is generated fresh
through `scripts/probes/shared_raw_generation_helper.py` using the
`repo_local_instnct_mutation_graph` backend introduced by 138YN.

This phase does not train, mutate checkpoints, modify helper/backend code,
import old phase runners, start services, deploy, change public API surfaces, or
change root `LICENSE`.

Required outputs:

- `queue.json`
- `progress.jsonl`
- `upstream_138yn_manifest.json`
- `upstream_138yk_manifest.json`
- `analysis_config.json`
- `eval_dataset_manifest.json`
- `eval_rows.jsonl`
- `ast_shortcut_scan_report.json`
- `expected_output_canary_report.json`
- `forbidden_input_rejection_report.json`
- `instnct_checkpoint_manifest.json`
- `instnct_ablation_checkpoint_manifest.json`
- `byte_gru_raw_generation_results.jsonl`
- `instnct_raw_generation_results.jsonl`
- `instnct_ablation_raw_generation_results.jsonl`
- `byte_gru_scoring_results.jsonl`
- `instnct_scoring_results.jsonl`
- `instnct_ablation_scoring_results.jsonl`
- `byte_gru_value_grounding_metrics.json`
- `instnct_value_grounding_metrics.json`
- `instnct_ablation_value_grounding_metrics.json`
- `contrast_group_results.jsonl`
- `arm_comparison.json`
- `pocket_ablation_report.json`
- `control_results.jsonl`
- `control_arm_report.json`
- `generated_before_scoring_report.json`
- `determinism_replay_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`

Decision routes:

- `instnct_adapter_prompt_bound_value_grounding_improves -> 138YP_INSTNCT_MUTATION_POCKET_GATED_VALUE_GROUNDING_PLAN`
- `instnct_mutation_value_grounding_comparison_positive -> 139YO_INSTNCT_MUTATION_VALUE_GROUNDING_SCALE_CONFIRM`
- `instnct_mutation_value_grounding_not_better_than_byte_gru -> 138YOB_INSTNCT_MUTATION_COMPARISON_FAILURE_ANALYSIS`
- `nondeterministic_instnct_mutation_comparison -> 138N_DETERMINISM_FAILURE_ANALYSIS`
- `scorer_or_task_weakness -> 138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS`

Guardrails:

- Improvement in this phase is adapter-path evidence only.
- Architecture superiority is not claimed.
- Pocket/highway mechanism responsibility is not claimed unless pocket writeback
  is nonzero and pocket ablation changes decision-critical metrics.
- Broad assistant capability, structured/tool capability, GPT-like readiness,
  open-domain readiness, production readiness, public API readiness, deployment
  readiness, and safety alignment remain false.

This is explicitly not GPT-like readiness.
