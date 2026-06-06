# E7P Numeric Pocket Adapter Joint Training Audit Result

Run root:

```text
target/pilot_wave/e7p_numeric_pocket_adapter_joint_training_audit
```

## Decision

```text
decision = e7p_numeric_pocket_composition_not_yet_viable
best_non_reference_system = joint_adapter_plus_pocket_with_slot_contract
deterministic_replay_passed = True
checker_failure_count = 0
```

## Evidence Run

```text
seeds = 99501,99502,99503,99504,99505,99506
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
standalone_pocket_then_fixed_adapter             useful=0.529861 acc=0.629861 preserve=0.044666 slot_corrupt=0.424320
adapter_only_training                            useful=0.636632 acc=0.736632 preserve=0.026936 slot_corrupt=0.474653
pocket_core_only_training                        useful=0.515278 acc=0.615278 preserve=0.051903 slot_corrupt=0.454673
joint_adapter_plus_pocket_training               useful=0.648438 acc=0.748437 preserve=0.026652 slot_corrupt=0.426678
joint_adapter_plus_pocket_with_slot_contract     useful=0.715972 acc=0.815972 preserve=0.018069 slot_corrupt=0.304803
full_end_to_end_training_control                 useful=0.653125 acc=0.753125 preserve=0.030371 slot_corrupt=0.531742
oracle_intermediate_state_reference              useful=0.992917 acc=1.000000 preserve=0.000000 slot_corrupt=0.000000
```

## Interpretation

E7P improved the E7O composition path but did not solve it. The best local system was joint adapter-plus-pocket training with an explicit slot contract. It beat the standalone fixed-adapter baseline by `+0.186111` usefulness and reduced state preservation error from `0.044666` to `0.018069`.

The slot contract mattered more than naive joint training:

```text
joint_adapter_plus_pocket_training           = 0.648438 usefulness
joint_adapter_plus_pocket_with_slot_contract = 0.715972 usefulness
```

The diagnostic full end-to-end control did not beat the slot-contract local system:

```text
full_end_to_end_training_control             = 0.653125 usefulness
slot_contract_local_system                   = 0.715972 usefulness
```

The remaining gap to oracle is still large:

```text
slot_contract_local_system = 0.715972
oracle_reference           = 0.992917
gap                        = 0.276945
```

This points to a real flow-interface/contract problem rather than a router problem. Route accuracy stayed `1.0`, router error was `0.0`, and composition error was `0.0`; failures concentrated in adapter/pocket behavior and result-slot corruption.

## Next Step

E7Q should focus on an explicit flow-interface contract, not on larger routing or dense end-to-end training. The likely next falsification is a slot-preserving pocket interface with typed flow lanes, stronger input/output adapter contract losses, and a check that composition survives when pockets are frozen and reused across unseen routes.

## Runtime Artifacts

```text
target/pilot_wave/e7p_numeric_pocket_adapter_joint_training_audit/report.md
target/pilot_wave/e7p_numeric_pocket_adapter_joint_training_audit/deterministic_replay.json
target/pilot_wave/e7p_numeric_pocket_adapter_joint_training_audit/checker_summary.json
```

## Boundary

E7P is a controlled numeric pocket-flow interface probe. It does not make raw-language, deployed-model, AGI, consciousness, or model-scale claims.
