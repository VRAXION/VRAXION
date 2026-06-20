# E115 Alpha-Weave Pressure Cell Schema Validation Contract

## Purpose

E115 locks the first canonical alpha-Weave pressure-cell schema before targeted
curriculum generation starts.

Core question:

```text
Can we define a stable pressure-cell unit that pressures rare Operator routes
without leaking checker-only answers, route labels, or task labels into the
candidate-visible input?
```

## Schema Boundary

Every private cell is split into candidate-visible input and checker/trainer
only metadata. The public repository intentionally does not publish the full
private field layout.

```text
candidate_visible_input
  visible to runtime/candidate

checker_only_expectations
  checker/trainer only; private

private_training_metadata
  route pressure and budget metadata
  hidden from candidate

private_adversarial_variants
  counterfactual pack for shortcut attacks
```

The candidate-visible path is the only candidate/runtime input path.

## Required Public Fields

```text
context
current_cycle
observations[]
query
```

Each observation must contain:

```text
obs_id
cycle
order
source_id
source_trust
text
span
```

## Required Variant Pack

```text
answerable_base
missing_evidence
weak_source
unresolved_conflict
stale_replay
source_trust_inversion
order_swap
quote_or_inactive_scope
negative_scope_story_text
citation_id_shortcut_trap
citation_span_trap
overbudget_fullscan_trap
```

## Required Controls

These controls must be invalidated by the pack. A weak control may pass a
happy-path row, but it must fail on at least one adversarial trap and therefore
must not be accepted as a general policy:

```text
label_leak_control
latest_only_control
source_name_shortcut_control
citation_id_shortcut_control
full_scan_control
answer_without_trace_control
negative_scope_overcall_control
```

## Required Gates

```text
schema_validity = true
checker_metadata_leak_rate = 0
route_label_leak_rate = 0
primary_success_rate = 1
false_commit_rate = 0
wrong_scope_call_rate = 0
unsupported_answer_rate = 0
over_budget_rate = 0
controls_all_invalid_as_general_policy = true
deterministic replay passes
checker failure_count = 0
```

## Required Artifacts

```text
run_manifest.json
alpha_weave_pressure_cell_schema_v1.json
sample_cell_pack.json
public_input_samples.json
machine_solve_view.json
schema_validation_report.json
adversarial_validation_report.json
control_results.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
partial_aggregate_snapshot.json
progress.jsonl
report.md
checker_summary.json
```

## Boundary

This is a schema/curriculum-unit validation probe. It is not final training,
not PermaCore/TrueGolden promotion, and not an open-domain reasoning claim.
