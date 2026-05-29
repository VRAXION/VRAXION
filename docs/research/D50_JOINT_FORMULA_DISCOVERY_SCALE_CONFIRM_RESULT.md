# D50 Joint Formula Discovery Scale Confirm Result

Status:

```text
completed
scale_mode = scale_lite
```

Artifact root:

```text
target/pilot_wave/d50_joint_formula_discovery_scale_confirm/smoke/
```

Decision:

```text
decision = joint_formula_discovery_scale_confirmed
verdict = D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRMED
next = D51_MUTABLE_ECF_CONTROLLER_PROTOTYPE
```

Run shape:

```text
seeds = 10401,10402,10403,10404,10405
train_rows_per_seed = 800
test_rows_per_seed = 800
ood_rows_per_seed = 800
workers = auto
```

Scale-lite was used because the full requested 8 seed x 1200 row run would be materially larger than the prior D49B run. The runner supports full mode, but this local validation used the allowed scale-lite fallback.

Key `FULL_REPAIRED_ECF_CONTROLLER` metrics:

```text
exact_joint_accuracy = 0.99920
cell_pair_equivalence_accuracy = 1.00000
cell_hit_top2_accuracy = 1.00000
operator_exact_accuracy = 0.99920
operator_equivalence_accuracy = 0.99920
average_total_support_used = 8.73920
average_counter_support_used = 3.73920
failed_jobs = []
```

Regime metrics:

```text
clean = 1.00000
correlated_echo = 0.99875
adversarial_distractor = 0.99725
mixed_clean_correlated = 1.00000
mixed_clean_adversarial = 1.00000
distinguishable_correlated_false = 0.99875
external_test_required = 0.99825
indistinguishable_abstain_rate = 1.00000
indistinguishable_false_confidence_rate = 0.00000
```

Minimum seed metrics:

```text
min_seed_exact_joint = 0.99800
min_seed_correlated_echo = 0.99625
min_seed_adversarial_distractor = 0.99000
```

Support-cost frontier:

```text
FULL_REPAIRED_ECF_CAP_7 exact_joint = 0.97435, support = 6.24640
FULL_REPAIRED_ECF_CAP_9 exact_joint = 0.99785, support = 7.49280
FULL_REPAIRED_ECF_CONTROLLER exact_joint = 0.99920, support = 8.73920
```

Component ablation:

```text
JOINT_INTERACTION_COUNTERFACTUAL exact_joint = 0.99430
MULTI_STAGE_COUNTERFACTUAL_REPAIR exact_joint = 0.99900
NO_CELL_COUNTERFACTUAL exact_joint = 0.99880
NO_OPERATOR_COUNTERFACTUAL exact_joint = 0.99840
NO_JOINT_INTERACTION_COUNTERFACTUAL exact_joint = 0.99830
```

Controls:

```text
RANDOM_EXTRA_SUPPORT_CONTROL exact_joint = 0.26885
SHUFFLED_COUNTER_SUPPORT_CONTROL exact_joint = 0.39065
NO_COUNTERFACTUAL_CONTROL exact_joint = 0.56445
```

The run writes `decision.json`, `aggregate_metrics.json`, `scale_summary.json`, `component_ablation_report.json`, `support_cost_frontier_report.json`, `indistinguishability_report.json`, `external_test_required_report.json`, and `report.md` under the artifact root.

Boundary:

```text
D50 only scale-confirms controlled symbolic joint formula discovery with robust ECF support.
It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority.
```
