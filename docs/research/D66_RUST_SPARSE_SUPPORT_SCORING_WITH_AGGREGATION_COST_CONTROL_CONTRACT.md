# D66 Rust Sparse Support Scoring With Aggregation Cost Control Contract

## Question

D65R confirmed Rust sparse set aggregation as an efficiency layer, not an
accuracy-necessary truth layer. D66 asks:

```text
Can Rust sparse aggregation help score/triage support and counter-support so
the controller keeps accuracy and safety while spending less support?
```

## Scope

Controlled symbolic joint formula discovery:

```text
formula = operator(cell_a, cell_b) mod 9
```

The formula solver remains the symbolic controlled stack. D66 only tests the
Rust sparse aggregation-backed support-scoring/control layer.

## Arms

```text
D65R_RUST_SET_AGG_REFERENCE
SUPPORT_SCORING_WITH_RUST_AGGREGATION
COUNTER_SUPPORT_TRIAGE_WITH_RUST_AGGREGATION
SUPPORT_BUDGET_CAPPED_RUST_AGGREGATION
AGGREGATION_ABLATION_CONTROL
RANDOM_SCORE_CONTROL
COST_MATCHED_RANDOM_SUPPORT_CONTROL
ALWAYS_COUNTER_CONTROL
NON_AGGREGATE_DIAGNOSTIC_ONLY
SUPPORT_CONTENT_CORRUPTION_CONTROL
TRUTH_LEAK_SENTINEL_REFERENCE_ONLY
```

## Metrics

```text
exact_joint_accuracy
correlated/adversarial/external/indistinguishable behavior
false_confidence_rate
abstain_rate
average_total_support_used
average_counter_support_used
cost_adjusted_accuracy
support_saved_vs_ablation
support_saved_vs_always_counter
unnecessary_counter_support_rate
missed_counter_support_rate
support_over_cheapest_correct_mean
rust_path_invoked
fallback_rows
failed_jobs
```

## Required Reports

```text
d65r_upstream_manifest.json
support_scoring_report.json
support_triage_report.json
counter_support_triage_report.json
support_cost_frontier_report.json
support_budget_report.json
ablation_compensation_report.json
content_corruption_report.json
truth_leak_audit_report.json
rust_invocation_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

## Decision Logic

```text
rust_sparse_support_scoring_cost_control_confirmed
  next = D67_RUST_SPARSE_SUPPORT_SCORING_SCALE_CONFIRM

support_scoring_cost_control_not_confirmed
  next = D66_REPAIR
```

Positive requires:

```text
best Rust aggregation support-scoring arm exact >= D65R reference exact - 0.003
support saved >= 1.0 vs aggregation ablation
support saved >= 1.0 vs always-counter
false_confidence <= 0.01
indistinguishable abstain >= 0.99
random/content controls worse
fallback_rows = 0
failed_jobs = []
```

## Hard Gates

```text
no broad claims
no label echo as fair oracle
no Python hash
no fake fixed accuracies
truth hidden from fair arms
Rust path invoked for Rust arms
fallback rows reported and zero
support-cost frontier required
failed jobs visible
```

## Boundary

D66 only tests Rust sparse aggregation-backed support scoring and counter-support
cost control in controlled symbolic joint formula discovery. It does not prove
full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness,
DNA/genome success, architecture superiority, or production readiness.
