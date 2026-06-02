# D55 Sparse Firing ECF Controller Prototype Result

## Status

```text
completed
scale_mode = prototype
```

## Artifact Root

```text
target/pilot_wave/d55_sparse_firing_ecf_controller_prototype/smoke
```

## Decision

```text
decision = sparse_firing_ecf_controller_prototype_strong_positive
verdict = D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE_STRONG_POSITIVE
next = D56_SPARSE_FIRING_ECF_CONTROLLER_SCALE_CONFIRM
```

## Main Metrics

`REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION`:

```text
exact_joint_accuracy = 0.99925
correlated_echo = 0.99875
adversarial_distractor = 0.99750
external_test_required = 0.99500
indistinguishable_abstain = 1.00000
indistinguishable_false_confidence = 0.00000
support = 7.66460
counter_support = 2.66460
sparse_firing_used = true
spike_update_executed_count = 2688000
```

Controls:

```text
RANDOM_POLICY_CONTROL exact = 0.55695
GREEDY_DECIDE_CONTROL exact = 0.56580
SPIKE_SHUFFLE_CONTROL exact = 0.51660
FIRING_THRESHOLD_ABLATION exact = 0.56580
CONNECTION_REWIRE_ABLATION exact = 0.56100
ALWAYS_COUNTER_CONTROL support = 11.00000
```

Mutation path:

```text
mutation_counts:
  gate_action = 1216
  gate_channel = 1178
  gate_priority = 1361
  gate_threshold = 2504
  gate_weight = 2061

accepted_mutation_counts:
  gate_action = 90
  gate_channel = 271
  gate_priority = 518
  gate_threshold = 488
  gate_weight = 540
```

## Interpretation

D55 confirms that the D54 controller policy can be executed through a controller-local sparse firing path using charge, threshold, channel/phase, sparse gate firing, and spike-derived action readout. The formula solver remains the controlled symbolic D50-D54 task; the sparse controller only chooses the ECF action.

The ablations matter:

```text
spike shuffle breaks the action readout
threshold ablation removes the counter-support gates
connection rewire breaks the gate-to-action mapping
random/greedy controls remain much worse
always-counter is accurate but too expensive and fails abstain/external behavior
```

## Boundary

```text
D55 only tests a controller-local sparse firing ECF action policy for controlled symbolic joint formula discovery. It does not prove full VRAXION sparse firing brain learning, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
```
