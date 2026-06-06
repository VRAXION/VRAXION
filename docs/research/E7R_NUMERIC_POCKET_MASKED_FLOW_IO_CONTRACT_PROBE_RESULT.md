# E7R Numeric Pocket Masked Flow IO Contract Probe Result

Run root:

```text
target/pilot_wave/e7r_numeric_pocket_masked_flow_io_contract_probe
```

## Status

```text
status = complete
decision = e7r_learned_sparse_mask_contract_positive
best_non_reference_system = learned_mask_contract
checker_failure_count = 0
deterministic_replay_passed = true
```

## Evidence Configuration

```text
seeds = 99701,99702,99703,99704,99705,99706
run_root = target/pilot_wave/e7r_numeric_pocket_masked_flow_io_contract_probe
artifact_contract = pass
deterministic_replay_hash_match = pass
```

The run wrote `progress.jsonl`, `hardware_heartbeat.jsonl`, replay artifacts,
mutation history, row-level system results, and aggregate reports.

## Mean Scores

```text
system                                      usefulness  answer_acc  next_compat  shuffle_robustness
current_untyped_flow_baseline              0.650139    0.750139    0.134566     0.000000
semantic_labeled_lane_control              0.786250    0.886250    0.000000     0.000000
anonymous_fixed_mask_contract              0.770972    0.870972    0.000000     0.000000
anonymous_shuffled_mask_contract           0.768194    0.868194    0.020935     0.996346
result_region_only_write_contract          0.759722    0.859722    0.000000     0.000000
residual_preservation_contract             0.767639    0.867639    0.000000     0.000000
learned_mask_contract                      0.812083    0.912083    0.003818     0.000000
oracle_mask_reference                      0.992917    1.000000    0.000000     0.000000
full_end_to_end_control                    0.527361    0.667361    0.000000     0.000000
dense_graph_danger_control                 0.539444    0.679444    0.000000     0.000000
```

All masked contract systems had:

```text
write_mask_violation_rate = 0.0
preserve_mask_corruption_rate = 0.0
result_region_corruption_rate = 0.0
```

## Interpretation

E7R found that mechanical Flow[D] IO hygiene helps numeric pocket
composition. The untyped baseline remained weak, while anonymous masked
contracts improved usefulness without semantic slot labels. The strongest
non-reference result was the learned sparse mask contract:

```text
baseline_usefulness = 0.650138888889
anonymous_fixed_usefulness = 0.770972222222
anonymous_shuffled_usefulness = 0.768194444444
semantic_labeled_usefulness = 0.78625
learned_mask_usefulness = 0.812083333333
learned_mask_sparsity = 0.047222222222
```

The lane-shuffle condition stayed robust:

```text
anonymous_shuffled_mask_contract / anonymous_fixed_mask_contract = 0.996346410211
```

That supports the intended reading: the gain came from anonymous mechanical
read/write/preserve discipline, not from a fixed semantic lane name shortcut.

The semantic labeled lane control performed well, but it did not beat the
learned sparse mask contract. The dense graph and full end-to-end controls
underperformed, so this run did not collapse back into a dense graph solution.

## Scientific Meaning

E7R is a positive IO-contract result, not a final pocket-composition solution.
The masks reduced state corruption and improved composition usefulness, but
the best learned mask still remained well below the oracle reference. The
remaining gap is likely a numeric pocket interface and trainability issue, not
just a router issue.

Recommended next step:

```text
E7S_FLOW_IO_CONTRACT_PLUS_ROUTER_POCKET_CO_TRAINING_FALSIFICATION
```

## Boundary

E7R only tests numeric pocket Flow[D] IO hygiene in a controlled pocket-router proxy. It does not prove raw-language learning, AGI, consciousness, or model-scale behavior.
