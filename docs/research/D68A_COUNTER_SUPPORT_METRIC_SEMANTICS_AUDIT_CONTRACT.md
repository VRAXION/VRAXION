# D68A Counter-Support Metric Semantics Audit Contract

## Goal

Audit the D68 counter-support metrics before attempting another triage repair.

D68 reduced reported unnecessary counter-support, but failed hard accuracy gates.
The current metric definitions are too coarse:

```text
unnecessary = selected counter/external while DECIDE was already correct
internal_needed = DECIDE wrong and any internal counter fixes
counter TP = selected any internal counter while any internal counter would fix
```

This does not prove that the selected concrete counter action fixed the row, or
that a reported unnecessary request consumed extra support.

## Required Files

```text
scripts/probes/run_d68a_counter_support_metric_semantics_audit.py
scripts/probes/run_d68a_counter_support_metric_semantics_audit_check.py
docs/research/D68A_COUNTER_SUPPORT_METRIC_SEMANTICS_AUDIT_CONTRACT.md
docs/research/D68A_COUNTER_SUPPORT_METRIC_SEMANTICS_AUDIT_RESULT.md
```

Generated artifacts live only under:

```text
target/pilot_wave/d68a_counter_support_metric_semantics_audit/
```

## Required Audits

```text
counter_metric_definition_report.json
concrete_counter_action_report.json
causal_counter_removal_report.json
cheapest_correct_support_report.json
d68_harm_classification_report.json
diagnostic_margin_stability_report.json
support_accounting_report.json
regime_blind_audit_report.json
artifact_completeness_and_rebuild_parity_report.json
```

## Required Metrics

```text
reported_unnecessary_counter
causal_unnecessary_counter
no_cost_unnecessary_counter_request
harmful_unnecessary_counter
reported_missed_counter
concrete_selected_counter_missed
selected_concrete_counter_fixes
wrong_counter_type_rate
wrong_concrete_counter_rate
weak_top1_top2_path_failure_rate
lost_joint_counter_path_failure_rate
lost_external_path_failure_rate
support_over_cheapest_correct
d68_loss_rows_vs_d67
false_confidence
abstain
fallback_rows
failed_jobs
```

## Decisions

```text
counter_support_metrics_confirmed
counter_metrics_valid_but_need_rename
counter_support_metric_pipeline_not_confirmed
d68a_artifact_insufficient_for_metric_audit
```

## Hard Gates

```text
no broad claims
no label echo as fair oracle
truth hidden from fair arms
oracle arms reference-only
no Python hash
no fake accuracies
support accounting explicit
selected concrete counter correctness measured
failed jobs visible
no black-box run: queue/progress/partial reports written regularly
```

## Boundary

D68A only audits counter-support metric semantics for controlled symbolic joint
formula discovery. It does not prove full VRAXION brain, raw visual Raven
reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture
superiority, or production readiness.
