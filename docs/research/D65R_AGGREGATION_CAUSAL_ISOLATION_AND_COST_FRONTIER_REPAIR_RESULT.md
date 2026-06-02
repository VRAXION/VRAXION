# D65R Aggregation Causal Isolation And Cost Frontier Repair Result

Status: scale-lite smoke passed locally.

Artifact root:

```text
target/pilot_wave/d65r_aggregation_causal_isolation_and_cost_frontier_repair/smoke
```

Run:

```text
python scripts/probes/run_d65r_aggregation_causal_isolation_and_cost_frontier_repair.py --out target/pilot_wave/d65r_aggregation_causal_isolation_and_cost_frontier_repair/smoke --seeds 12401,12402,12403,12404,12405 --train-rows-per-seed 10 --test-rows-per-seed 12 --ood-rows-per-seed 12 --workers auto --cpu-target 50-75 --heartbeat-sec 20 --scale-mode scale-lite
python scripts/probes/run_d65r_aggregation_causal_isolation_and_cost_frontier_repair_check.py --check-only --out target/pilot_wave/d65r_aggregation_causal_isolation_and_cost_frontier_repair/smoke
```

Decision:

```text
decision = rust_sparse_set_aggregation_efficiency_confirmed
verdict  = D65R_RUST_SPARSE_SET_AGGREGATION_EFFICIENCY_CONFIRMED
next     = D66_RUST_SPARSE_SUPPORT_SCORING_WITH_AGGREGATION_COST_CONTROL
```

Key result:

```text
best_track = D65_REPLAY
best_arm   = RUST_SPARSE_SET_AGGREGATION
rust_exact = 1.0000
ablation_exact = 1.0000
causal_gap = 0.0000
support_delta_vs_ablation = 2.1400
rust_false_confidence = 0.0000
failed_jobs = []
```

Interpretation:

```text
D65R did not prove an accuracy-only causal necessity gap for aggregation.
It did confirm an efficiency/cost role: the Rust sparse set aggregation path
matched the ablation exact score while using materially less support.
```

Cost frontier highlights:

```text
D65_REPLAY:
  RUST_SPARSE_SET_AGGREGATION exact=1.0000 support=8.86 cost_adjusted=0.99035
  AGGREGATION_ABLATION_CONTROL exact=1.0000 support=11.00 cost_adjusted=0.98500
  RANDOM_SCORE_AGGREGATION_CONTROL exact=0.9933 support=8.00 cost_adjusted=0.98583
  NON_AGGREGATE_DIAGNOSTIC_ONLY_CONTROLLER exact=0.5600 support=5.00 cost_adjusted=0.56000
  SUPPORT_CONTENT_CORRUPTION_CONTROL exact=0.9333 support=7.61 cost_adjusted=0.92681

SUPPORT_BUDGET_CAPPED:
  RUST_SPARSE_SET_AGGREGATION exact=0.9933 support=6.46 cost_adjusted=0.98968
  AGGREGATION_ABLATION_CONTROL exact=0.9933 support=8.00 cost_adjusted=0.98583
  RANDOM_SCORE_AGGREGATION_CONTROL exact=0.7367 support=6.24 cost_adjusted=0.73357
```

Causal isolation track:

```text
AGGREGATION_REQUIRED_FEATURE_STARVATION:
  RUST_SPARSE_SET_AGGREGATION exact=1.0000 support=8.86
  AGGREGATION_ABLATION_CONTROL exact=1.0000 support=11.00
  NON_AGGREGATE_DIAGNOSTIC_ONLY_CONTROLLER exact=0.5600 support=5.00
```

This says aggregation-side features are useful, but the decisive D65R finding is
still cost/efficiency rather than an exact-accuracy-only gap over every ablation.

Rust invocation:

```text
rust_path_invoked = true
fallback_rows = 0
python_precomputed_final_aggregate_label_rows = 0
rust_aggregation_rows = 69120
rust_controller_rows = 80640
```

Truth-leak audit:

```text
fair_arms_using_truth_label = []
python_precomputed_final_aggregate_label_used_by_fair_arms = false
reference_only_arms = TRUTH_LEAK_SENTINEL_REFERENCE_ONLY
```

Boundary: D65R only tests the causal and support-cost role of Rust sparse set
aggregation in controlled symbolic joint formula discovery. It does not prove
full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI,
consciousness, DNA/genome success, architecture superiority, or production
readiness.
