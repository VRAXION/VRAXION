# D65R Aggregation Causal Isolation And Cost Frontier Repair Contract

## Question

D65 showed that Rust sparse set aggregation runs, but did not prove it is
necessary. D65R asks whether Rust sparse set aggregation is:

```text
A. causally necessary for accuracy under harder aggregation-dependent conditions
B. mainly useful for support-cost efficiency
C. redundant because the current controller solves through other paths
```

## Tracks

```text
D65_REPLAY
AGGREGATION_REQUIRED_FEATURE_STARVATION
SUPPORT_BUDGET_CAPPED
HIGH_AMBIGUITY_SUPPORT_SET
CORRELATED_ADVERSARIAL_AGGREGATION_STRESS
COST_FRONTIER_TRACK
```

The feature-starvation track removes parallel non-aggregate symbolic diagnostics.
The support-budget cap prevents ablation from silently compensating by always
requesting expensive counter-support.

## Arms

```text
SYMBOLIC_SET_AGGREGATION_REFERENCE
RUST_SPARSE_SET_AGGREGATION
RUST_SPARSE_SCORE_SHAPE_AGGREGATION
RUST_SPARSE_SUPPORT_COHERENCE_AGGREGATION
RUST_SPARSE_COUNTERFACTUAL_DELTA_AGGREGATION
HYBRID_SYMBOLIC_RUST_AGGREGATION
AGGREGATE_ONLY_CONTROLLER
NON_AGGREGATE_DIAGNOSTIC_ONLY_CONTROLLER
RANDOM_SCORE_AGGREGATION_CONTROL
AGGREGATION_ABLATION_CONTROL
SUPPORT_CONTENT_CORRUPTION_CONTROL
ALWAYS_COUNTER_COMPENSATION_CONTROL
COST_CAPPED_ABLATION_CONTROL
TRUTH_LEAK_SENTINEL_REFERENCE_ONLY
```

## Required Reports

```text
d65_upstream_manifest.json
causal_isolation_report.json
feature_starvation_report.json
support_budget_cap_report.json
cost_frontier_report.json
compensation_path_report.json
aggregation_quality_report.json
content_corruption_report.json
truth_leak_audit_report.json
rust_invocation_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

## Decisions

```text
rust_sparse_set_aggregation_causally_confirmed
  next = D66_RUST_SPARSE_SUPPORT_SCORING_MIGRATION_PLAN

rust_sparse_set_aggregation_efficiency_confirmed
  next = D66_RUST_SPARSE_SUPPORT_SCORING_WITH_AGGREGATION_COST_CONTROL

set_aggregation_redundant_under_current_controller
  next = D65R_REDEFINE_OR_SKIP_AGGREGATION_LAYER

d65r_aggregation_repair_failed
  next = D65_REPAIR
```

## Boundary

D65R only tests the causal and support-cost role of Rust sparse set aggregation
in controlled symbolic joint formula discovery. It does not prove full VRAXION
brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome
success, architecture superiority, or production readiness.
