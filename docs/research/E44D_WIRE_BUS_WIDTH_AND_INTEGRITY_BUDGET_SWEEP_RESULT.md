# E44D Wire Bus Width And Integrity Budget Sweep Result

## Decision

```text
decision = e44d_bus12_sufficient
minimal_passing_config = bus12_5data_4crc_3reserve
universal_selected_config = bus12_5data_4crc_3reserve
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = c4e7b262ba2516f5
```

E44D tested the open E44C question: can the Proposal payload bus carry data,
integrity, and true reserve capacity at the same time?

## Stress Table

```text
| system | stress_success | reserve_success | integrity_success | reserve_noise_success | active_bitflip_success | double_alias_success | burst_noise_success | known_permutation_success | unknown_permutation_success | false_commit_rate | wrong_commit_rate | false_ask_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle_reference | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| bus8_5data_3reserve_masked | 0.733 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.267 | 0.250 | 0.000 |
| bus8_5data_3crc | 0.702 | 0.000 | 0.667 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.525 | 0.098 | 0.082 | 0.200 |
| bus10_5data_3crc_2reserve | 0.902 | 1.000 | 0.667 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.525 | 0.098 | 0.082 | 0.000 |
| bus12_5data_4crc_3reserve | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| bus16_5data_5ecc_6reserve | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| universal_mutated_bus_policy | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| random_policy_control | 0.215 | 0.008 | 0.375 | 0.000 | 0.475 | 0.350 | 0.300 | 0.025 | 0.300 | 0.222 | 0.313 | 0.125 |
```

## Interpretation

The 8-bit options split the budget:

```text
8-bit reserve mode = 5 data + 3 reserve, but no silent-corruption guard
8-bit CRC mode     = 5 data + 3 integrity, but no true reserve
```

The 10-bit option separated CRC and reserve, but the 3-bit checksum still had
an adversarial alias under the `double_alias_silent` stress. The 12-bit option
was the smallest passing configuration in this sweep:

```text
12-bit = 5 data + 4 integrity + 3 reserve
```

The universal mutation selector also selected:

```text
bus12_5data_4crc_3reserve
```

## Bus Width Sweep

```json
{
  "configs": {
    "bus10_5data_3crc_2reserve": {
      "integrity": "crc3",
      "integrity_bits": 3,
      "known_remap": true,
      "payload_bits": 10,
      "reserve_bits": 2,
      "unknown_wire_guard": false
    },
    "bus12_5data_4crc_3reserve": {
      "integrity": "crc4",
      "integrity_bits": 4,
      "known_remap": true,
      "payload_bits": 12,
      "reserve_bits": 3,
      "unknown_wire_guard": true
    },
    "bus16_5data_5ecc_6reserve": {
      "integrity": "crc5",
      "integrity_bits": 5,
      "known_remap": true,
      "payload_bits": 16,
      "reserve_bits": 6,
      "unknown_wire_guard": true
    },
    "bus8_5data_3crc": {
      "integrity": "crc3",
      "integrity_bits": 3,
      "known_remap": true,
      "payload_bits": 8,
      "reserve_bits": 0,
      "unknown_wire_guard": false
    },
    "bus8_5data_3reserve_masked": {
      "integrity": "none",
      "integrity_bits": 0,
      "known_remap": true,
      "payload_bits": 8,
      "reserve_bits": 3,
      "unknown_wire_guard": false
    }
  },
  "minimal_passing_config": "bus12_5data_4crc_3reserve",
  "pass_gate": {
    "bus10_5data_3crc_2reserve": false,
    "bus12_5data_4crc_3reserve": true,
    "bus16_5data_5ecc_6reserve": true,
    "bus8_5data_3crc": false,
    "bus8_5data_3reserve_masked": false
  },
  "tradeoff": {
    "bus10_crc_reserve": "separates reserve and CRC3 but fails double-alias corruption",
    "bus12_crc_reserve": "smallest passing bus in this sweep",
    "bus8_crc": "integrity-capable but spends the extra three bits on checks, not true reserve",
    "bus8_reserve": "reserve-capable but unprotected against silent active-bit corruption"
  }
}
```

## Boundary

This is a controlled symbolic/numeric Proposal ABI stress probe. It does not
prove raw language reasoning, deployed AI assistant behavior, model-scale
behavior, AGI, or consciousness.
