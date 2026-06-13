# E44C Reserve Wire Mask And Noise Stress Result

## Decision

```text
decision = e44c_eight_bit_not_universal_under_silent_noise
payload_bits = 8
data_bits = 5
reserve_bits = 3
universal_selected_mode = crc3_integrity
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = 51637219aff9d55f
```

E44C stress-tested the proposed 8-bit anonymous Proposal payload bus before
locking it as a universal default.

The result is not a clean "8-bit reserve is universally enough." The stress
barrage found a real tradeoff:

```text
masked reserve bits:
  good at ignoring reserve noise
  unsafe under silent active-bit corruption

CRC/integrity reserve bits:
  good at detecting active bitflip / burst noise
  sensitive to reserve-bit noise
```

## Primary Stress Table

```text
| system | stress_success | reserve_noise_success | active_dropout_success | active_bitflip_success | burst_noise_success | known_permutation_success | unknown_permutation_success | false_commit_rate | wrong_commit_rate | false_ask_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle_integrity_reference | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| unmasked8_full_payload_decoder | 0.384 | 0.000 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | 0.286 | 0.529 | 0.071 |
| active5_ignore_reserve_mask | 0.680 | 1.000 | 1.000 | 0.000 | 0.000 | 0.525 | 0.000 | 0.286 | 0.236 | 0.000 |
| active5_visible_dropout_guard | 0.786 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.214 | 0.179 | 0.000 |
| crc3_integrity_guard | 0.754 | 0.025 | 1.000 | 1.000 | 1.000 | 1.000 | 0.525 | 0.034 | 0.018 | 0.212 |
| universal_mutated_wire_setup | 0.754 | 0.025 | 1.000 | 1.000 | 1.000 | 1.000 | 0.525 | 0.034 | 0.018 | 0.212 |
| random_policy_control | 0.436 | 0.000 | 0.400 | 0.650 | 0.350 | 0.000 | 0.350 | 0.159 | 0.325 | 0.086 |
```

## Confirm Seeds

```text
seed 44602: e44c_eight_bit_not_universal_under_silent_noise, selected crc3_integrity
seed 44603: e44c_eight_bit_not_universal_under_silent_noise, selected crc3_integrity
```

## Interpretation

The universal mutation selector consistently chose `crc3_integrity`. That means
when false/wrong commit is heavily penalized, the system prefers to spend the
3 reserve bits on integrity checks instead of leaving them as passive reserve.

But CRC3 over an 8-bit bus is still not a full deployment-grade answer:

```text
active single-bit corruption: handled
visible active dropout: handled
known wire permutation: handled
reserve random noise: mostly false-ask
unknown permutation: partial
some silent multi-bit cases: residual false/wrong commit
```

So the clean conclusion is:

```text
8-bit bus = good minimal/default bus for clean + reserve-noise-masked use
8-bit bus != universal deployment-stress bus
```

## Recommended Lock

Do not lock "8-bit with 3 free reserve bits" as the universal deployment ABI.

Lock this narrower rule:

```text
minimum payload capacity = 5 bits
simple default payload = 8 anonymous bits
reserve bits must be protected by active_mask if inactive
if silent corruption matters, reserve bits should become integrity bits
```

For a stronger deployment ABI, the next test should expand the bus:

```text
E44D_WIRE_BUS_WIDTH_AND_INTEGRITY_BUDGET_SWEEP

compare:
  8-bit  = 5 data + 3 reserve/integrity
  10-bit = 5 data + 3 integrity + 2 reserve
  12-bit = 5 data + stronger integrity + reserve
  16-bit = data + integrity + spare capacity
```

The open question is whether 10/12/16 bits can provide both:

```text
reserve noise tolerance
and
silent corruption detection
```

without falling into false asks or wrong commits.

## Boundary

This is a controlled symbolic/numeric Proposal ABI stress probe. It does not
prove raw language reasoning, deployed AI assistant behavior, model-scale
behavior, AGI, or consciousness.
