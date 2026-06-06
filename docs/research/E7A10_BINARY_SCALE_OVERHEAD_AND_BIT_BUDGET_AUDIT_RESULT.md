# E7A10 Binary Scale Overhead And Bit-Budget Audit Result

## Decision

```text
decision = e7a10_binary_same_budget_preferred
deterministic_replay_passed = true
checker_failure_count = 0
```

Artifact root:

```text
target/pilot_wave/e7a10_binary_scale_overhead_and_bit_budget_audit/
```

## Main Result

The int4 width-32 reference used for the same-budget comparison was:

```text
int4_direct width32
eval_accuracy = 0.942592592593
total_bit_cost = 6352
```

The best binary system that stayed inside that bit budget was:

```text
binary_minimal_scale_qat width64
eval_accuracy = 0.950925925926
total_bit_cost = 5124
scale_bit_cost = 0
compression_vs_float32 = 32.0x
```

The best binary system without the bit-budget restriction was:

```text
binary_channel_scale_qat_paramwise_freeze width64
eval_accuracy = 0.952777777778
total_bit_cost = 9444
scale_bit_cost = 4320
compression_vs_float32 = 17.362x
```

## Interpretation

Binary was not only rescued by per-channel scale overhead in this run. The strongest same-budget result came from a fixed minimal-scale binary QAT policy at width64, which beat the int4 width32 reference while using fewer measured bits.

Direct binary remained weak, so the result does not support naive post-training binary quantization. Binary needed QAT, and the best unrestricted binary still benefited from channel-scale QAT plus paramwise mutation/freeze repair.

The practical read is:

```text
quality-first simple path: int4 remains easy and stable
same-budget compact path: binary width scaling is viable here
highest binary score: channel-scale QAT + mutation/freeze repair, but with scale overhead
```

## Guardrails

This remains a controlled symbolic/numeric matrix-core compression audit. It does not prove anything about raw natural-language reasoning, AGI, consciousness, or model-scale behavior.

The next sanity check should retest the surprising minimal-scale binary result across a shifted task family and a wider bit-budget sweep, because fixed-scale binary winning under budget is the important new claim to try to break.
