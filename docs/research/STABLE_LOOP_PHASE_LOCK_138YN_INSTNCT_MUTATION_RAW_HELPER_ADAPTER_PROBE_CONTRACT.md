# STABLE_LOOP_PHASE_LOCK_138YN_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PROBE Contract

## Purpose

138YN is the targeted adapter/probe after 138YM. It tests whether `shared_raw_generation_helper.py` can safely dispatch to a strict repo-local INSTNCT mutation graph backend:

```text
repo_local_instnct_mutation_graph
```

This is not a value-grounding comparison yet. It proves or rejects helper-compatible raw generation plumbing for a minimal INSTNCT mutation graph manifest.

## Boundaries

138YN may modify only:

```text
scripts/probes/shared_raw_generation_helper.py backend dispatch
```

It may add only the 138YN runner/checker/docs besides generated outputs. It must not train, mutate source checkpoints, import old phase runners, start services, deploy, modify runtime/service/product/release surfaces, modify SDK exports, or change root `LICENSE`.

## Required Gates

The probe must write:

```text
queue.json
progress.jsonl
upstream_138ym_manifest.json
adapter_contract.json
helper_provenance_verification.json
forbidden_input_rejection_report.json
expected_output_canary_report.json
ast_shortcut_scan_report.json
instnct_checkpoint_manifest.json
prompt_encoder_trace.jsonl
iterative_propagation_trace.jsonl
raw_generation_trace.jsonl
raw_generation_results.jsonl
generated_before_scoring_report.json
determinism_replay_report.json
decision.json
summary.json
report.md
```

All helper requests must use only:

```text
prompt
checkpoint_path
checkpoint_hash
seed
max_new_tokens
generation_config
```

Expected/scorer/oracle material must be rejected before generation.

## Decision

If complete:

```text
decision = instnct_mutation_raw_helper_adapter_probe_complete
next = 138YO_INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_PROBE
```

If adapter generation fails:

```text
decision = adapter_generation_missing
next = 138YNA_INSTNCT_ADAPTER_GENERATION_FAILURE_ANALYSIS
```

## Boundary Claims

Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.
