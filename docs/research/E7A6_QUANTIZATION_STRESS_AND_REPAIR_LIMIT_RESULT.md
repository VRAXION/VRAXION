# E7A6 Quantization Stress And Repair Limit Result

## Decision

```text
decision = e7a6_mutation_repair_partial_low_bit_recovery
deterministic_replay_passed = true
checker_failure_count = 0
final_e7_verdict = intentionally deferred
```

Run root:

```text
target/pilot_wave/e7a6_quantization_stress_and_repair_limit/
```

## Frontier

Best float32 matrix-core:

```text
width = 32
eval_accuracy = 0.965833
heldout = 0.953333
OOD = 0.966667
counterfactual = 0.976667
adversarial = 0.966667
```

| level | no-repair eval | no-repair drop | stable no-repair | repair eval | repair drop | stable repair |
|---|---:|---:|---|---:|---:|---|
| `int8` | 0.965833 | 0.000000 | true | 0.965000 | 0.000833 | true |
| `int4` | 0.960000 | 0.005833 | true | 0.958333 | 0.007500 | true |
| `int3` | 0.938333 | 0.027500 | false | 0.934167 | 0.031667 | false |
| `ternary` | 0.883333 | 0.082500 | false | 0.925833 | 0.040000 | false |
| `binary` | 0.840833 | 0.125000 | false | 0.893333 | 0.072500 | false |

Random control best eval was `0.265833` and did not pass.

## Interpretation

The plain matrix-core is robust through `int4`. `int8` is lossless at this scale, and `int4` stays comfortably inside the `0.02` stability-drop threshold.

`int3` is the first real boundary: it still passes the solve thresholds, but its eval drop is `0.0275`, so it fails the stricter stability gate.

`ternary` and `binary` break the core. Mutation repair is useful there, but only partially:

```text
ternary repair gain = +0.042500
binary repair gain  = +0.052500
```

That repair gain is real, but it does not recover the low-bit cores to stable float-near performance. The correct interpretation is partial low-bit recovery, not full recovery.
