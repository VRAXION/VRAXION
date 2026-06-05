# E7A7 Low-Bit Repair Operator Audit Result

## Decision

```text
decision = e7a7_qat_preferred_over_post_repair
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e7a7_low_bit_repair_operator_audit
```

## What Ran

E7A7 reused the E7A6 plain matrix-core and E7A3 task generation. It audited:

```text
levels = int3, ternary, binary
widths = 16, 32
blocks = input_projection, recurrent_state, carry_gate, state_bias, output_head
```

For each low-bit level it ran:

```text
full low-bit no repair
single-block low-bit damage
single-block int8 restore
full mutation repair
targeted block mutation repair
top-sensitive-pair mutation repair
fake-quant QAT then quantize
```

## Key Result Table

| level | width | low-bit eval | top restored block | restore gain | full repair | best targeted | best pair | QAT |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `int3` | 16 | 0.939167 | `recurrent_state` | 0.006667 | 0.925833 | 0.948333 | 0.934167 | |
| `int3` | 32 | 0.934167 | `input_projection` | 0.013333 | 0.937500 | 0.950000 | 0.934167 | |
| `ternary` | 16 | 0.875833 | `input_projection` | 0.045000 | 0.866667 | 0.916667 | 0.875000 | 0.923333 |
| `ternary` | 32 | 0.894167 | `input_projection` | 0.056667 | 0.930833 | 0.906667 | 0.921667 | 0.946667 |
| `binary` | 16 | 0.842500 | `input_projection` | 0.066667 | 0.832500 | 0.860000 | 0.842500 | 0.895000 |
| `binary` | 32 | 0.845833 | `input_projection` | 0.089167 | 0.900833 | 0.875833 | 0.913333 | 0.931667 |

Best float32 matrix-core:

```text
width = 32
eval_accuracy = 0.951667
```

## Interpretation

The main low-bit damage point was usually `input_projection`, not the recurrent state or output head. Restoring only `input_projection` to int8 gave the largest recovery for ternary/binary width 32:

```text
ternary restore gain = +0.056667
binary  restore gain = +0.089167
```

Mutation repair helped, but it did not match QAT:

```text
ternary width32:
  low-bit = 0.894167
  best repair = 0.930833
  QAT = 0.946667

binary width32:
  low-bit = 0.845833
  best repair = 0.913333
  QAT = 0.931667
```

This suggests the matrix-core is not inherently unable to operate at ternary/binary. The stronger result is that post-quantization mutation repair is currently weaker than training with quantization pressure already present.

## What Did Not Get Proven

This does not prove a broad reasoning result. It only says that on this controlled symbolic/numeric matrix-core proxy:

```text
low-bit breakage is most visible around the input projection
QAT recovers more cleanly than post-quantization mutation repair
mutation repair remains useful but is not yet the best low-bit recovery path
```

## Next Recommended Experiment

The next narrow experiment should improve mutation repair rather than add new architecture:

```text
E7A8_INPUT_PROJECTION_AWARE_LOW_BIT_REPAIR
```

Core test:

```text
compare generic repair vs input-projection-aware repair
add scale mutation / blockwise scale mutation
test ternary and binary only
include QAT as reference, not as the target system
```
