# STABLE_LOOP_PHASE_LOCK_071B_REPAIR_OVERFIT_FAILURE_ANALYSIS Result

Status: implementation result document for analysis-only attribution of the 071
repaired-checkpoint fresh confirmation failure.

071B parses existing 071 artifacts and attributes the remaining
counterfactual, context extraction, and pocket suppression failures to concrete
wrong-answer sources. It writes a specific 072 curriculum patch.

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

## Implementation Summary

Analysis runner:

```text
scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis.py
```

Static checker:

```text
scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis_check.py
```

The runner validates the required 071 artifacts:

```text
summary.json
per_family_metrics.json
human_readable_samples.jsonl
failure_case_samples.jsonl
baseline_metrics.json
no_route_feature_control_metrics.json
```

If any artifact is missing, the runner emits
`UPSTREAM_071_ARTIFACT_MISSING` and stops without trying to recreate upstream
state.

## Required Analysis Targets

```text
FRESH_COUNTERFACTUAL_BINDING
FRESH_CONTEXT_ENTITY_EXTRACTION
FRESH_IRRELEVANT_POCKET_SUPPRESSION
```

The analysis classifies failed supported rows into:

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

Failure verdict if exceeded:

```text
WRONG_ANSWER_SOURCE_UNCLASSIFIED_TOO_HIGH
```

## Required Reports

Counterfactual report fields:

```text
active_scenario_miss_rate
old_scenario_selection_rate
distractor_scenario_selection_rate
first_ledger_value_selection_rate
inactive_pocket_selection_rate
stale_pocket_selection_rate
```

Context extraction report fields:

```text
exact_anchor_success_rate
key_collision_rate
side_note_value_selection_rate
copy_first_match_agreement_rate
no_route_agreement_rate
```

Pocket suppression report fields:

```text
irrelevant_pocket_selection_rate
stale_pocket_selection_rate
side_note_value_selection_rate
no_route_agreement_rate
```

Failure verdicts if incomplete:

```text
COUNTERFACTUAL_ANALYSIS_INCOMPLETE
CONTEXT_ANALYSIS_INCOMPLETE
POCKET_ANALYSIS_INCOMPLETE
```

## Required Artifacts

Generated artifacts are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/
```

Artifact names:

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

`progress.jsonl`, `summary.json`, and `report.md` are written from the start, so
071B is not a black-box run.

`human_failure_digest.jsonl` includes:

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

Failure verdict if missing:

```text
FAILURE_CASE_INPUT_MISSING
```

## Curriculum Patch

`recommended_curriculum_patch.json` includes:

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

## Observed Smoke Result

The 071B analysis smoke completed with:

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

Observed attribution:

```text
unknown_label_rate = 0.0

FRESH_COUNTERFACTUAL_BINDING:
  failure_rows = 28 / 30
  active_scenario_miss_rate = 0.9333333333333333
  distractor_scenario_selection_rate = 0.8214285714285714
  old_scenario_selection_rate = 0.17857142857142858
  no_route_agreement_rate = 1.0

FRESH_CONTEXT_ENTITY_EXTRACTION:
  failure_rows = 9 / 30
  first_ledger_value_selection_rate = 0.5555555555555556
  side_note_value_selection_rate = 0.4444444444444444
  key_collision_rate = 0.4
  no_route_agreement_rate = 0.8888888888888888

FRESH_IRRELEVANT_POCKET_SUPPRESSION:
  failure_rows = 10 / 30
  irrelevant_pocket_selection_rate = 1.0
  inactive_pocket_selection_rate = 0.9
  stale_pocket_selection_rate = 0.1
  no_route_agreement_rate = 1.0
```

Interpretation:

```text
dominant counterfactual failure:
  distractor_scenario_value

dominant context extraction failures:
  first_ledger_value
  side_note_value

dominant pocket suppression failure:
  inactive_pocket_value
```

This supports `072_COUNTERFACTUAL_SCENARIO_BINDING_REPAIR` as a scenario-state
binding and stale/inactive pocket suppression repair, not FineWeb scale-up as
the immediate fix.

## Required Commands

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

## Static Checker Output

The checker prints compact JSON with:

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

071B success means the 071 failure is explained well enough to design
`072_COUNTERFACTUAL_SCENARIO_BINDING_REPAIR`.

071B does not mean model capability improved or a checkpoint was repaired. no capability gate passed.
