# D51 Mutable ECF Controller Prototype Result

Status:

```text
completed
scale_mode = scale_lite
```

Artifact root:

```text
target/pilot_wave/d51_mutable_ecf_controller_prototype/smoke/
```

Decision:

```text
decision = mutable_ecf_controller_prototype_positive
verdict = D51_MUTABLE_ECF_CONTROLLER_PROTOTYPE_POSITIVE
next = D52_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM
```

Run shape:

```text
seeds = 10501,10502,10503,10504,10505
train_rows_per_seed = 800
test_rows_per_seed = 800
ood_rows_per_seed = 800
generations = 150
population = 64
workers = auto
```

Scale-lite was used as allowed by the plan. The scale-lite reduction changed only the mutable search size:

```text
full requested search = generations 300, population 96
scale-lite search = generations 150, population 64
```

Best mutable controller:

```text
best_mutable_arm = MUTABLE_RULE_TABLE_CONTROLLER
exact_joint_accuracy = 0.99945
correlated_echo_accuracy = 0.99875
adversarial_distractor_accuracy = 0.99850
external_test_required_accuracy = 0.99500
indistinguishable_abstain_rate = 1.00000
indistinguishable_false_confidence_rate = 0.00000
false_confidence_rate = 0.00000
average_support_used = 7.67180
average_counter_support_used = 2.67180
failed_jobs = []
```

Reference comparison:

```text
HANDCODED_CAP_7_REFERENCE exact = 0.97585, support = 6.24950
HANDCODED_CAP_9_REFERENCE exact = 0.99770, support = 7.49900
HANDCODED_D50_FULL_REPAIRED_ECF_REFERENCE exact = 0.99945, support = 8.74850
MUTABLE_RULE_TABLE_CONTROLLER exact = 0.99945, support = 7.67180
```

Controls:

```text
RANDOM_POLICY_CONTROL exact = 0.55925, support = 6.83975
GREEDY_DECIDE_CONTROL exact = 0.56415, support = 5.00000
ALWAYS_COUNTER_SUPPORT_CONTROL exact = 0.99945, support = 11.00000
```

Best controller action distribution:

```text
DECIDE = 10188
REQUEST_COUNTER_TOP1_TOP2 = 1812
REQUEST_JOINT_COUNTER = 12000
REQUEST_EXTERNAL_TEST = 4000
ABSTAIN = 4000
REQUEST_SUPPORT = 0
```

Interpretation:

```text
The mutable rule-table controller matched the D50 full reference exact score while using less support.
It learned the useful controller split: decide on easy cases, use joint counter-support on stress cases, request external tests where the external channel is available, and abstain on indistinguishable cases.
```

The run writes controller feature, mutation, action-distribution, support-cost, false-confidence, regime, comparison, best-policy, aggregate, decision, summary, and markdown reports under the artifact root.

Boundary:

```text
D51 only tests mutable control policy for controlled symbolic joint formula discovery.
It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority.
```
