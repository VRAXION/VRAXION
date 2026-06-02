# D67 Rust Sparse Support Scoring Scale Confirm Contract

## Question

D66 confirmed Rust sparse aggregation-backed support scoring as a
support-cost control layer. D67 asks:

```text
Does the D66 support-cost improvement hold at larger scale, and where does
the system still spend unnecessary or miss necessary counter-support?
```

## Scope

Controlled symbolic joint formula discovery:

```text
formula = operator(cell_a, cell_b) mod 9
```

The formula solver remains the controlled symbolic stack. D67 only tests the
Rust sparse aggregation-backed support-scoring/control layer.

## Tracks

```text
REPLAY_COST_FRONTIER
SUPPORT_BUDGET_CAPPED
FIXED_BUDGET_SWEEP
COUNTER_SUPPORT_TRIAGE
HIGH_AMBIGUITY_TOP1_TOP2
CORRELATED_ADVERSARIAL_SUPPORT
CLEAN_UNNECESSARY_COUNTER_AUDIT
MIXED_UNNECESSARY_COUNTER_AUDIT
OOD_COST_FRONTIER
```

## Arms

```text
D66_BEST_REPLAY
RUST_SPARSE_SUPPORT_SCORING
RUST_SPARSE_COUNTER_SUPPORT_TRIAGE
RUST_SPARSE_SUPPORT_SCORING_CAP_7
RUST_SPARSE_SUPPORT_SCORING_CAP_9
AGGREGATION_ABLATION_CONTROL
COST_MATCHED_RANDOM_SUPPORT_CONTROL
ALWAYS_COUNTER_CONTROL
NON_AGGREGATE_DIAGNOSTIC_ONLY
SUPPORT_CONTENT_CORRUPTION_CONTROL
TRUTH_LEAK_SENTINEL_REFERENCE_ONLY
```

## Metrics

```text
exact_joint_accuracy
correlated_echo_accuracy
adversarial_distractor_accuracy
external_test_required_accuracy
indistinguishable_abstain_rate
false_confidence_rate
average_total_support_used
average_counter_support_used
cost_adjusted_accuracy
support_saved_vs_ablation
support_saved_vs_always_counter
unnecessary_counter_support_rate
missed_counter_support_rate
accuracy_at_fixed_budget
budget_frontier_auc
rust_path_invoked
fallback_rows
failed_jobs
```

## Required Reports

```text
d66_upstream_manifest.json
scale_summary_report.json
support_scoring_report.json
support_triage_report.json
counter_support_triage_report.json
support_cost_frontier_report.json
fixed_budget_sweep_report.json
clean_unnecessary_counter_audit_report.json
mixed_unnecessary_counter_audit_report.json
unnecessary_counter_support_report.json
missed_counter_support_report.json
regime_breakdown_report.json
ood_cost_frontier_report.json
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
rust_sparse_support_scoring_scale_confirmed
  next = D68_COUNTER_SUPPORT_MINIMIZATION_AND_TRIAGE_OPTIMIZATION

support_scoring_scale_confirmed_counter_triage_gap
  next = D68_COUNTER_SUPPORT_TRIAGE_REPAIR

support_scoring_scale_not_confirmed
  next = D67_REPAIR
```

Positive scale confirmation requires:

```text
best Rust support-scoring exact >= D66 replay exact - 0.003
correlated_echo >= 0.995
adversarial_distractor >= 0.995
external_test_required >= 0.990
support_saved_vs_ablation >= 2.0
support_saved_vs_always >= 2.0
false_confidence <= 0.01
indistinguishable_abstain >= 0.99
random/content/non-aggregate controls worse
fallback_rows = 0
failed_jobs = []
```

If those pass but unnecessary counter-support remains high, D67 still
scale-confirms the D66 mechanism but routes to D68 counter-support triage
repair.

## Run Plan

Default healthy scale:

```text
seeds = 12601,12602,12603,12604,12605
rows = 240 per seed/regime/split
workers = auto
cpu_target = 50-75
heartbeat_sec = 20
```

If stable and runtime is acceptable, an optional larger run may use 400 rows.
The actual scale mode must be reported.

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
unnecessary/missed counter-support frontier required
truth leak sentinel required
failed jobs visible
no black-box run: queue/progress/partial artifacts required
```

## Boundary

D67 only scale-confirms Rust sparse aggregation-backed support scoring and
counter-support cost control in controlled symbolic joint formula discovery. It
does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved,
AGI, consciousness, DNA/genome success, architecture superiority, or production
readiness.
