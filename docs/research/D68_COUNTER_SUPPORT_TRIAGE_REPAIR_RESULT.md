# D68 Counter-Support Triage Repair Result

Status: completed local healthy run.

This result page is populated from:

```text
target/pilot_wave/d68_counter_support_triage_repair/smoke/
```

Run mode:

```text
scale_mode = healthy-240
seeds = 12701,12702,12703,12704,12705
train_rows_per_seed = 240
test_rows_per_seed = 240
ood_rows_per_seed = 240
cpu_target = 50-75
heartbeat_sec = 20
```

Expected decision set:

```text
counter_support_triage_repair_confirmed
counter_triage_high_recall_high_cost
counter_triage_recall_failure
counter_support_triage_repair_not_confirmed
```

Observed decision:

```text
decision = counter_support_triage_repair_not_confirmed
verdict = D68_COUNTER_SUPPORT_TRIAGE_REPAIR_NOT_CONFIRMED
next = D68_REPAIR
best_arm = TRAINED_THRESHOLD_TRIAGE_GATE
```

Summary:

```text
D68 successfully reduced unnecessary counter-support, but did not preserve the
D67 accuracy floor under the hard correlated/adversarial regimes. This is a
useful negative result: the triage signal can remove waste, but the current
repair policy swaps too much of D67's stronger joint/external behavior for a
cheaper top1/top2 counter path.
```

Key metrics:

| arm | exact | correlated | adversarial | external | support | counter | unnecessary | missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D67_BEST_REPLAY | 0.999333 | 0.999167 | 0.997500 | 0.995000 | 7.6795 | 2.6795 | 0.565667 | 0.000000 |
| TRAINED_THRESHOLD_TRIAGE_GATE | 0.993833 | 0.980833 | 0.988333 | 0.995000 | 6.4795 | 1.4795 | 0.058833 | 0.000000 |
| COUNTER_TRIAGE_MULTI_SIGNAL_GATE | 0.993833 | 0.980833 | 0.988333 | 0.995000 | 6.8730 | 1.8730 | 0.190000 | 0.000000 |
| ALWAYS_COUNTER_CONTROL | 0.999333 | 0.999167 | 0.997500 | 0.995000 | 11.0000 | 6.0000 | 0.565667 | 0.000000 |
| RANDOM_COUNTER_CONTROL | 0.783667 | n/a | n/a | n/a | 6.8530 | 1.8530 | 0.299000 | 0.212833 |
| SHUFFLED_TRIAGE_SIGNAL_CONTROL | 0.565667 | n/a | n/a | n/a | 5.5445 | 0.5445 | 0.181500 | 0.434000 |
| BAD_TRIAGE_SIGNAL_CONTROL | 0.565667 | n/a | n/a | n/a | 5.0000 | 0.0000 | 0.000000 | 0.434000 |

Decision reasons:

```text
reference_exact = 0.999333
best_exact = 0.993833
required_exact_floor = reference_exact - 0.003 = 0.996333

correlated_echo = 0.980833
adversarial_distractor = 0.988333
required_correlated_adversarial_floor = 0.995

support_saved_vs_same_run_d67 = 1.2000
support_saved_vs_always = 4.5205
unnecessary_counter_support_rate = 0.058833
clean_unnecessary_counter_support_rate = 0.023333
mixed_clean_correlated_unnecessary_counter_support_rate = 0.054167
mixed_clean_adversarial_unnecessary_counter_support_rate = 0.189167

missed_counter_support_rate = 0.000000
external_test_missed_rate = 0.000000
false_confidence = 0.000000
indistinguishable_abstain = 1.000000
```

Rust invocation and guardrails:

```text
rust_path_invoked = true
rust_aggregation_rows = 345600
rust_controller_rows = 403200
fallback_rows = 0
python_precomputed_final_aggregate_label_rows = 0
failed_jobs = []
fair_arms_using_forbidden_metadata = []
fair_arms_using_truth_label = []
```

Interpretation:

```text
The triage classifier did what it was asked to do on cost: it removed most of
the obviously unnecessary counter-support. The failure is not black-box noise;
the artifact reports show the tradeoff clearly. D67 often used stronger
joint/external actions, while D68's best triage path used cheaper top1/top2
counter-support. That cheaper action family was not strong enough on pure
correlated and adversarial regimes.
```

Recommended next:

```text
D68_REPAIR should keep the useful cost triage, but add a hard-regime escalation
route that preserves D67-style joint/external behavior when the signal indicates
pure correlated/adversarial pressure.
```

Boundary:

```text
D68 only tests counter-support triage for Rust sparse aggregation-backed support
scoring in controlled symbolic joint formula discovery. The formula solver
remains symbolic. It does not prove full VRAXION brain, raw visual Raven
reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture
superiority, or production readiness.
```
