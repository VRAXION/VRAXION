# E7N Real Numeric Pocket Core Bridge Probe Result

Run root:

```text
target/pilot_wave/e7n_real_numeric_pocket_core_bridge_probe
```

## Decision

```text
decision = e7n_mutation_repair_numeric_pocket_positive
best_non_control_system = float_numeric_pocket_backprop
deterministic_replay_passed = true
checker_failure_count = 0
```

## Mean Scores

```text
symbolic_proxy_pocket_reference                acc=1.000000 useful=1.000000 bits=0.0 active=0.0
float_numeric_pocket_backprop                  acc=0.888333 useful=0.798333 bits=861440.0 active=26920.0
quantized_numeric_pocket_int8                  acc=0.888125 useful=0.798125 bits=215552.0 active=26920.0
quantized_numeric_pocket_int4                  acc=0.863611 useful=0.773611 bits=107872.0 active=26920.0
quantized_numeric_pocket_ternary               acc=0.627361 useful=0.561529 bits=54032.0 active=26920.0
quantized_numeric_pocket_binary                acc=0.612500 useful=0.570896 bits=27112.0 active=26920.0
quantized_pocket_plus_mutation_repair          acc=0.628958 useful=0.587354 bits=27112.0 active=26920.0
quantized_pocket_plus_prune_crystallize        acc=0.863611 useful=0.773611 bits=89742.7 active=22387.7
quantized_pocket_plus_repair_plus_prune        acc=0.629861 useful=0.588312 bits=27051.0 active=26859.0
random_pocket_control                          acc=0.112431 useful=0.112431 bits=0.0 active=0.0
```

## Interpretation

E7N confirms the first real numeric-pocket bridge on this controlled proxy:

```text
real matrix pocket pretraining works
int8 preserves the float pocket almost exactly
int4 remains viable with moderate quality loss
ternary/binary are not yet high-quality direct replacements
binary mutation repair gives a small but real positive gain
int4 pruning/crystallization compresses without additional quality loss
```

Key bridge details:

```text
float_accuracy = 0.888333333333
int8_accuracy = 0.888125
int4_accuracy = 0.863611111111
ternary_accuracy = 0.627361111111
binary_accuracy = 0.6125
repair_gain = 0.016458333333
prune_compression_ratio = 0.168363051015
random_accuracy = 0.112430555556
```

The strongest practical path from this run is:

```text
float numeric pocket birth
-> int8 or int4 quantization
-> int4 prune/crystallize
-> register as callable router pocket
```

Binary remains a research branch, not the main practical bridge for E7N. Mutation repair helped binary, but not enough to close the int4 gap.

## Boundary

E7N is a controlled numeric pocket-core bridge probe. It does not make raw-language, deployed-model, AGI, consciousness, or model-scale claims.
