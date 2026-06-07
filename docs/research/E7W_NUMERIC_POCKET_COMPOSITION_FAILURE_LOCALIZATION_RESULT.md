# E7W Numeric Pocket Composition Failure Localization Result

Status: complete.

```text
decision = e7w_output_write_contract_bottleneck
best_system = oracle_intermediate_state_after_each_pocket
baseline_usefulness = 0.599023
oracle_intermediate_usefulness = 0.900000
oracle_read_usefulness = 0.684375
oracle_write_usefulness = 0.900000
calibration_bridge_usefulness = 0.621289
residual_integration_usefulness = 0.602148
specific_pocket_bottleneck = verify weakest, but not sole cause
deterministic_replay_passed = true
checker_failure_count = 0
```

## Mean Scores

```text
baseline_best_current                            useful=0.599023 acc=0.699023 ram=0.007726 out=0.180710 next=0.006629
oracle_route_only                                useful=0.599023 acc=0.699023 ram=0.007726 out=0.180710 next=0.006629
oracle_intermediate_state_after_each_pocket      useful=0.900000 acc=1.000000 ram=0.000000 out=0.170524 next=0.000000
one_real_pocket_at_a_time                        useful=0.864062 acc=0.964063 ram=0.000711 out=0.028421 next=0.000742
oracle_read_map_real_write                       useful=0.684375 acc=0.784375 ram=0.004263 out=0.170524 next=0.004453
real_read_map_oracle_write                       useful=0.900000 acc=1.000000 ram=0.000000 out=0.000000 next=0.000000
output_calibration_bridge                        useful=0.621289 acc=0.721289 ram=0.006172 out=0.154102 next=0.004793
residual_delta_integration                       useful=0.602148 acc=0.702148 ram=0.013671 out=0.301973 next=0.012650
broad_read_tiny_write_reference                  useful=0.595508 acc=0.695508 ram=0.007513 out=0.177730 next=0.004757
pruned_read_tiny_write_reference                 useful=0.599023 acc=0.699023 ram=0.007726 out=0.180710 next=0.006629
```

## One-Real-Pocket Attribution

```text
compare               0.900000
mod_add               0.900000
threshold             0.893164
parity                0.881250
counterfactual_flip   0.816406
verify                0.793555
```

`verify` and `counterfactual_flip` are the weakest isolated real pockets, but
the main localization is not a single pocket. The decisive comparison is:

```text
oracle_read_map_real_write = 0.684375
real_read_map_oracle_write = 0.900000
```

So read context helps, but oracle write fully restores composition. The likely
root is the value/format contract of what a pocket writes into Flow/RAM. The
intermediate state drift is a downstream symptom of bad writes.

## Recommended Next Step

```text
E7X_OUTPUT_WRITE_CONTRACT_AND_VALUE_FORMAT_PROBE
```

Test learned/calibrated write bridges, canonical value formats, probability vs
binary write, confidence-preserving write, and per-pocket output normalization.
Do not change router architecture yet.

Boundary: this is a controlled numeric pocket composition localization probe,
not a raw-language, AGI, consciousness, or model-scale claim.
