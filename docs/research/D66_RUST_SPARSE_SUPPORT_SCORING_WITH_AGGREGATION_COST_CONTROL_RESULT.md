# D66 Rust Sparse Support Scoring With Aggregation Cost Control Result

Status: healthy milestone smoke passed locally.

Artifact root:

```text
target/pilot_wave/d66_rust_sparse_support_scoring_with_aggregation_cost_control/smoke
```

Run:

```text
python scripts/probes/run_d66_rust_sparse_support_scoring_with_aggregation_cost_control.py --out target/pilot_wave/d66_rust_sparse_support_scoring_with_aggregation_cost_control/smoke --seeds 12501,12502,12503,12504,12505 --train-rows-per-seed 120 --test-rows-per-seed 120 --ood-rows-per-seed 120 --workers auto --cpu-target 50-75 --heartbeat-sec 20 --scale-mode healthy-120
python scripts/probes/run_d66_rust_sparse_support_scoring_with_aggregation_cost_control_check.py --check-only --out target/pilot_wave/d66_rust_sparse_support_scoring_with_aggregation_cost_control/smoke
```

The requested 800 row setting was audited first. In this runner the row count is
per seed and per support regime, so 800 would expand to 32,000 rows per split
and 352,000 controller packs per split. The run was intentionally re-scoped to
120 rows per seed/regime, yielding 4,800 rows per split and 52,800 controller
packs per split. This is a healthy milestone run, not a microprobe.

Decision:

```text
decision = rust_sparse_support_scoring_cost_control_confirmed
verdict  = D66_RUST_SPARSE_SUPPORT_SCORING_COST_CONTROL_CONFIRMED
next     = D67_RUST_SPARSE_SUPPORT_SCORING_SCALE_CONFIRM
best_arm = SUPPORT_SCORING_WITH_RUST_AGGREGATION
```

Key metrics:

```text
reference_exact = 0.9990
best_exact = 0.9990
best_support = 7.6720
ablation_support = 11.0000
always_counter_support = 11.0000
support_saved_vs_ablation = 3.3280
support_saved_vs_always = 3.3280
false_confidence = 0.0000
indistinguishable_abstain = 1.0000
failed_jobs = []
```

Support-cost frontier:

```text
SUPPORT_SCORING_WITH_RUST_AGGREGATION:
  exact = 0.9990
  support = 7.672
  counter_support = 2.672
  cost_adjusted = 0.992320

D65R_RUST_SET_AGG_REFERENCE:
  exact = 0.9990
  support = 8.839
  counter_support = 3.839
  cost_adjusted = 0.989403

SUPPORT_BUDGET_CAPPED_RUST_AGGREGATION:
  exact = 0.9957
  support = 6.472
  counter_support = 1.472
  cost_adjusted = 0.991987

AGGREGATION_ABLATION_CONTROL:
  exact = 0.9990
  support = 11.000
  counter_support = 6.000
  cost_adjusted = 0.984000

ALWAYS_COUNTER_CONTROL:
  exact = 0.9990
  support = 11.000
  counter_support = 6.000
  cost_adjusted = 0.984000

COST_MATCHED_RANDOM_SUPPORT_CONTROL:
  exact = 0.7530
  support = 6.256
  counter_support = 1.256
  cost_adjusted = 0.749860

SUPPORT_CONTENT_CORRUPTION_CONTROL:
  exact = 0.9537
  support = 7.635
  counter_support = 2.635
  cost_adjusted = 0.947079
```

Best-arm regime behavior:

```text
CLEAN_INDEPENDENT_SUPPORT exact = 1.0000 support = 5.115
CORRELATED_ECHO_SUPPORT exact = 0.9967 support = 11.000
ADVERSARIAL_DISTRACTOR_SUPPORT exact = 0.9983 support = 11.000
MIXED_CLEAN_AND_CORRELATED exact = 1.0000 support = 5.290
MIXED_CLEAN_AND_ADVERSARIAL exact = 1.0000 support = 5.955
DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT exact = 0.9967 support = 11.000
EXTERNAL_TEST_REQUIRED_SUPPORT exact = 0.9950 support = 9.000
INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT abstain = 1.0000 false_confidence = 0.0000
```

Rust invocation:

```text
rust_path_invoked = true
fallback_rows = 0
python_precomputed_final_aggregate_label_rows = 0
rust_aggregation_rows = 115200
rust_controller_rows = 105600
```

Important caveat:

```text
The best arm saves support materially, but unnecessary counter-support remains
high in clean and mixed regimes. D66 confirms cost-control over ablation and
always-counter compensation; it does not finish counter-support minimization.
D67 should scale-confirm the result and keep the unnecessary-counter frontier
visible.
```

Boundary: D66 only tests Rust sparse aggregation-backed support scoring and
counter-support cost control in controlled symbolic joint formula discovery. It
does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved,
AGI, consciousness, DNA/genome success, architecture superiority, or production
readiness.
