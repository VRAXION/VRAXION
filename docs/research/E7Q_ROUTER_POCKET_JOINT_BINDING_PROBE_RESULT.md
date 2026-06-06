# E7Q Router-Pocket Joint Binding Probe Result

Run root:

```text
target/pilot_wave/e7q_router_pocket_joint_binding_probe
```

## Status

```text
decision = e7q_joint_binding_not_yet_viable
best_non_reference_system = trained_router_trained_pocket
deterministic_replay_passed = True
checker_failure_count = 0
```

## Evidence Run

```text
seeds = 99601,99602,99603,99604,99605,99606
train_rows_per_seed = 720
validation_rows_per_seed = 300
heldout_rows_per_seed = 300
ood_rows_per_seed = 300
counterfactual_rows_per_seed = 300
adversarial_rows_per_seed = 300
device = cuda
```

## Mean Scores

```text
frozen_router_trained_pocket              useful=0.669722 acc=0.769722 route=1.000000 reuse=0.669722 risk=0.000000
trained_router_frozen_pocket              useful=0.471389 acc=0.601389 route=0.997500 reuse=0.471389 risk=0.000000
trained_router_trained_pocket             useful=0.680972 acc=0.810972 route=0.999583 reuse=0.711389 risk=0.209583
trained_router_trained_pocket_slot_guard  useful=0.652917 acc=0.782917 route=0.999861 reuse=0.682917 risk=0.181528
full_end_to_end_training_control          useful=0.536111 acc=0.676111 route=0.000000 reuse=0.000000 risk=1.000000
random_router_control                     useful=0.453611 acc=0.583611 route=0.111111 reuse=0.453611 risk=0.000000
oracle_route_reference                    useful=0.992917 acc=1.000000 route=1.000000 reuse=0.992917 risk=0.000000
```

## Flow Contract Metrics

```text
frozen_router_trained_pocket              preserve=0.023989 slot_corrupt=0.406262 compat=0.122139
trained_router_frozen_pocket              preserve=0.043612 slot_corrupt=0.422581 compat=0.173506
trained_router_trained_pocket             preserve=0.117097 slot_corrupt=0.804977 compat=0.259067
trained_router_trained_pocket_slot_guard  preserve=0.036614 slot_corrupt=0.477789 compat=0.172959
```

## Interpretation

E7Q did not confirm that jointly training the router/control layer and pocket library solves numeric pocket composition.

The route head itself learned the route almost perfectly:

```text
trained_router_frozen_pocket route_accuracy = 0.997500
```

But route discovery alone did not fix composition:

```text
trained_router_frozen_pocket usefulness = 0.471389
```

Joint binding gave a small score gain over the E7P-style frozen-router trained-pocket reference:

```text
frozen_router_trained_pocket  = 0.669722
trained_router_trained_pocket = 0.680972
gain                          = 0.011250
```

That gain is too small to call a solved binding mechanism, especially because unguarded joint training damaged the flow contract:

```text
unguided slot_corrupt = 0.804977
unguided risk         = 0.209583
```

The slot-guard variant improved flow hygiene but did not win on usefulness:

```text
slot_guard_usefulness = 0.652917
slot_guard_risk       = 0.181528
```

The diagnostic full end-to-end control also did not win:

```text
full_end_to_end_training_control = 0.536111
```

## What This Means

E7Q separates three things:

```text
router route learning:
  works

joint router+pocket binding:
  gives a small gain

reusable clean pocket interface:
  still not solved
```

The main bottleneck remains the typed Flow[D] pocket interface and result-slot hygiene, not route selection.

## Next Step

E7R should not simply make the router bigger. The next clean test should make the pocket interface more explicit:

```text
typed lane contract
dedicated read/write masks
result-slot ownership
scratch-slot reset or merge policy
composition-time slot hygiene loss
reuse-after-binding gate retained
```

The goal is to improve reusable pocket composition without allowing the router and pocket to form a private protocol.

## Runtime Artifacts

```text
target/pilot_wave/e7q_router_pocket_joint_binding_probe/report.md
target/pilot_wave/e7q_router_pocket_joint_binding_probe/deterministic_replay.json
target/pilot_wave/e7q_router_pocket_joint_binding_probe/reuse_after_binding_report.json
target/pilot_wave/e7q_router_pocket_joint_binding_probe/checker_summary.json
```

## Boundary

E7Q is a controlled numeric Flow[D] router-pocket binding probe. It does not make raw-language, deployed-model, AGI, consciousness, or model-scale claims.
