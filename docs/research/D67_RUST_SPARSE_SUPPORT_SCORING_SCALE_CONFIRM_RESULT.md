# D67 Rust Sparse Support Scoring Scale Confirm Result

Status: healthy-240 scale-confirm run passed locally.

Artifact root:

```text
target/pilot_wave/d67_rust_sparse_support_scoring_scale_confirm/smoke
```

Run:

```text
python scripts/probes/run_d67_rust_sparse_support_scoring_scale_confirm.py --out target/pilot_wave/d67_rust_sparse_support_scoring_scale_confirm/smoke --seeds 12601,12602,12603,12604,12605 --train-rows-per-seed 240 --test-rows-per-seed 240 --ood-rows-per-seed 240 --workers auto --cpu-target 50-75 --heartbeat-sec 20 --scale-mode healthy-240
python scripts/probes/run_d67_rust_sparse_support_scoring_scale_confirm_check.py --check-only --out target/pilot_wave/d67_rust_sparse_support_scoring_scale_confirm/smoke
```

The first healthy-240 attempt exposed a no-black-box instrumentation gap in the
blocking Rust aggregation bridge. The runner was patched before the final run
so bridge waits emit `rust_aggregation_bridge_wait_heartbeat` and
`rust_policy_bridge_wait_heartbeat` progress records every heartbeat.

Scale:

```text
rows_per_seed_per_regime_per_split = 240
rows_per_split = 9,600
rust_aggregation_rows = 230,400
rust_controller_rows = 211,200
elapsed_sec = 7,340
```

Decision:

```text
decision = support_scoring_scale_confirmed_counter_triage_gap
verdict  = D67_SUPPORT_SCORING_SCALE_CONFIRMED_COUNTER_TRIAGE_GAP
next     = D68_COUNTER_SUPPORT_TRIAGE_REPAIR
best_arm = D66_BEST_REPLAY
```

The replay and `RUST_SPARSE_SUPPORT_SCORING` arms tied on the core metrics. The
decision therefore scale-confirms the D66 mechanism rather than claiming a new
accuracy mechanism.

Decision reason:

```text
best_exact = 0.999333
reference_exact = 0.999333
correlated_echo = 0.997500
adversarial_distractor = 0.999167
external_test_required = 0.991667
false_confidence = 0.000000
indistinguishable_abstain = 1.000000
best_support = 7.6515
ablation_support = 11.0000
always_counter_support = 11.0000
support_saved_vs_ablation = 3.3485
support_saved_vs_always = 3.3485
unnecessary_counter_support_rate = 0.570333
missed_counter_support_rate = 0.000000
```

Support-cost frontier:

| arm | exact | support | counter | cost-adjusted | unnecessary | missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D66_BEST_REPLAY | 0.999333 | 7.6515 | 2.6515 | 0.992705 | 0.570333 | 0.000000 |
| RUST_SPARSE_SUPPORT_SCORING | 0.999333 | 7.6515 | 2.6515 | 0.992705 | 0.570333 | 0.000000 |
| RUST_SPARSE_COUNTER_SUPPORT_TRIAGE | 0.995500 | 7.0125 | 2.0125 | 0.990469 | 0.570333 | 0.000000 |
| RUST_SPARSE_SUPPORT_SCORING_CAP_7 | 0.570333 | 5.0000 | 0.0000 | 0.570333 | 0.570333 | 0.000000 |
| RUST_SPARSE_SUPPORT_SCORING_CAP_9 | 0.607500 | 5.2515 | 0.2515 | 0.606871 | 0.570333 | 0.000000 |
| AGGREGATION_ABLATION_CONTROL | 0.999333 | 11.0000 | 6.0000 | 0.984333 | 0.570333 | 0.000000 |
| ALWAYS_COUNTER_CONTROL | 0.999333 | 11.0000 | 6.0000 | 0.984333 | 0.570333 | 0.000000 |
| COST_MATCHED_RANDOM_SUPPORT_CONTROL | 0.570333 | 5.0000 | 0.0000 | 0.570333 | 0.240000 | 0.000000 |
| NON_AGGREGATE_DIAGNOSTIC_ONLY | 0.570333 | 5.0000 | 0.0000 | 0.570333 | 0.000000 | 0.429333 |
| SUPPORT_CONTENT_CORRUPTION_CONTROL | 0.951833 | 7.6355 | 2.6355 | 0.945245 | 0.570333 | 0.000000 |

Regime behavior for D66 replay / Rust scoring:

```text
CLEAN_INDEPENDENT_SUPPORT:
  exact = 1.000000
  support = 5.0733
  unnecessary_counter = 0.993333

CORRELATED_ECHO_SUPPORT:
  exact = 0.997500
  support = 11.0000
  unnecessary_counter = 0.016667

ADVERSARIAL_DISTRACTOR_SUPPORT:
  exact = 0.999167
  support = 11.0000
  unnecessary_counter = 0.015000

MIXED_CLEAN_AND_CORRELATED:
  exact = 1.000000
  support = 5.2515
  unnecessary_counter = 0.964167

MIXED_CLEAN_AND_ADVERSARIAL:
  exact = 1.000000
  support = 5.9415
  unnecessary_counter = 0.856667

EXTERNAL_TEST_REQUIRED_SUPPORT:
  exact = 0.991667
  support = 9.0000
  unnecessary_counter = 0.008333

INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT:
  abstain = 1.000000
  false_confidence = 0.000000
```

Interpretation:

```text
D67 scale-confirms D66 cost control:
  exact stays high
  correlated/adversarial/external gates pass
  support saving vs ablation/always-counter remains > 3.0
  random/content/non-aggregate controls remain worse
  Rust path is invoked with zero fallback

D67 also confirms the counter-triage gap:
  clean and mixed regimes still spend too much unnecessary counter-support
  hard correlated/adversarial regimes still require high support
  simple CAP_7/CAP_9 budget caps collapse in hard regimes
```

Rust invocation:

```text
rust_path_invoked = true
fallback_rows = 0
python_precomputed_final_aggregate_label_rows = 0
rust_aggregation_rows = 230400
rust_controller_rows = 211200
failed_jobs = []
```

Truth leak audit:

```text
fair_arms_using_truth_label = []
python_precomputed_final_aggregate_label_used_by_fair_arms = false
```

Boundary: D67 only scale-confirms Rust sparse aggregation-backed support
scoring and counter-support cost control in controlled symbolic joint formula
discovery. It does not prove full VRAXION brain, raw visual Raven reasoning,
Raven solved, AGI, consciousness, DNA/genome success, architecture superiority,
or production readiness.
