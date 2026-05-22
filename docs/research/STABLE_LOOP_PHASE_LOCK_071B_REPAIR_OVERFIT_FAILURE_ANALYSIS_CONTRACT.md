# STABLE_LOOP_PHASE_LOCK_071B_REPAIR_OVERFIT_FAILURE_ANALYSIS Contract

Status: contract for analysis-only attribution of the 071 repaired-checkpoint
fresh confirmation failure.

071B explains why the 071 fresh counterfactual, context extraction, and pocket
suppression rows failed. It parses existing 071 JSON and JSONL artifacts and
writes source-attribution outputs plus a concrete 072 curriculum patch.

This is analysis-only.

no training
no inference
no checkpoint repair
no checkpoint mutation
no 069/070/071 rerun
no model capability improved
no production training
no open-ended assistant
no free-form generation
no language grounding
no GA
no public beta
no hosted SaaS
no service API change
no deployment harness change
no release docs change
no public crate export change
no root LICENSE change

## Runner And Checker

Analysis runner:

```text
scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis.py
```

Static checker:

```text
scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis_check.py
```

071B may add only the analysis runner, static checker, and 071B research docs.

## Required Upstream Artifacts

071B requires these 071 artifacts:

```text
summary.json
per_family_metrics.json
human_readable_samples.jsonl
failure_case_samples.jsonl
baseline_metrics.json
no_route_feature_control_metrics.json
```

If any required 071 artifact is missing, fail with
`UPSTREAM_071_ARTIFACT_MISSING`. Do not silently skip missing files.

## Analysis Targets

```text
FRESH_COUNTERFACTUAL_BINDING
FRESH_CONTEXT_ENTITY_EXTRACTION
FRESH_IRRELEVANT_POCKET_SUPPRESSION
```

Required wrong-answer source labels:

```text
old_scenario_value
distractor_scenario_value
first_ledger_value
side_note_value
inactive_pocket_value
stale_pocket_value
copy_first_match_value
no_route_control_value
unknown_label
```

Hard gate:

```text
unknown_label <= 20%
```

If unknown attribution exceeds that limit, fail with
`WRONG_ANSWER_SOURCE_UNCLASSIFIED_TOO_HIGH`.

## Required Attribution Reports

Counterfactual report fields:

```text
active_scenario_miss_rate
old_scenario_selection_rate
distractor_scenario_selection_rate
first_ledger_value_selection_rate
inactive_pocket_selection_rate
stale_pocket_selection_rate
```

Failure verdict if missing:

```text
COUNTERFACTUAL_ANALYSIS_INCOMPLETE
```

Context extraction report fields:

```text
exact_anchor_success_rate
key_collision_rate
side_note_value_selection_rate
copy_first_match_agreement_rate
no_route_agreement_rate
```

Failure verdict if missing:

```text
CONTEXT_ANALYSIS_INCOMPLETE
```

Pocket suppression report fields:

```text
irrelevant_pocket_selection_rate
stale_pocket_selection_rate
side_note_value_selection_rate
no_route_agreement_rate
```

Failure verdict if missing:

```text
POCKET_ANALYSIS_INCOMPLETE
```

## Required Generated Artifacts

Generated artifacts are written only under `target/`:

```text
queue.json
progress.jsonl
analysis_config.json
upstream_071_manifest.json
failure_cluster_report.json
counterfactual_source_attribution.json
context_extraction_source_attribution.json
pocket_suppression_source_attribution.json
template_failure_matrix.json
wrong_answer_source_matrix.json
active_scenario_miss_report.json
distractor_scenario_selection_report.json
stale_pocket_selection_report.json
key_collision_report.json
no_route_control_comparison.json
recommended_curriculum_patch.json
human_failure_digest.jsonl
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from the start so
071B is not a black-box run.

`human_failure_digest.jsonl` must include:

```text
task_family
input
expected
model_output
classified_wrong_source
no_route_output
copy_first_match
short_diagnosis
```

Failure verdict if digest input is missing:

```text
FAILURE_CASE_INPUT_MISSING
```

## Required Curriculum Patch

`recommended_curriculum_patch.json` must include:

```text
active scenario marker strengthening
same key / different scenario training
stale scenario suppression
inactive pocket negative examples
scenario:active
scenario:old
scenario:distractor
answer-only plus trace-mixed variants
no-route and copy-first controls retained
no FineWeb scale-up as immediate fix
```

Failure verdict if missing:

```text
CURRICULUM_PATCH_MISSING
```

## Static Checker

The checker prints compact JSON:

```text
check_pass
missing_docs
placeholder_hits
missing_commands
missing_boundary_tokens
forbidden_claim_hits
generated_artifact_staged
root_license_changed
runtime_surface_mutation_detected
missing_required_terms
verdicts
```

Failure verdicts for side effects and surface changes:

```text
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

## Validation Commands

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis.py
python scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis.py --out target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke --upstream-071-root target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/smoke --upstream-070-root target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke --benchmark-069-root target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis_check.py
python scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_070_distractor_resistant_anchorroute_check.py --check-only
git diff --check
```

## Verdicts

Positive verdicts:

```text
REPAIR_OVERFIT_FAILURE_ANALYSIS_POSITIVE
UPSTREAM_071_FAILURE_PROFILE_LOADED
COUNTERFACTUAL_FAILURE_CLUSTERS_WRITTEN
CONTEXT_EXTRACTION_FAILURE_CLUSTERS_WRITTEN
POCKET_SUPPRESSION_FAILURE_CLUSTERS_WRITTEN
WRONG_ANSWER_SOURCE_ATTRIBUTION_WRITTEN
ACTIVE_SCENARIO_MISS_RATE_RECORDED
DISTRACTOR_SCENARIO_SELECTION_RECORDED
KEY_COLLISION_REPORT_WRITTEN
CURRICULUM_PATCH_RECOMMENDED
NO_TRAINING_PERFORMED
PRODUCTION_TRAINING_NOT_CLAIMED
```

Failure verdicts:

```text
REPAIR_OVERFIT_FAILURE_ANALYSIS_FAILS
UPSTREAM_071_ARTIFACT_MISSING
FAILURE_CASE_INPUT_MISSING
WRONG_ANSWER_SOURCE_UNCLASSIFIED_TOO_HIGH
COUNTERFACTUAL_ANALYSIS_INCOMPLETE
CONTEXT_ANALYSIS_INCOMPLETE
POCKET_ANALYSIS_INCOMPLETE
CURRICULUM_PATCH_MISSING
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

## Scope Meaning

071B success means the 071 failure is source-attributed well enough to design
`072_COUNTERFACTUAL_SCENARIO_BINDING_REPAIR`.

071B does not mean model capability improved or a checkpoint was repaired. no capability gate passed.
