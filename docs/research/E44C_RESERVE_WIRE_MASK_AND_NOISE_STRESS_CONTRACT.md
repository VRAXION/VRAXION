# E44C Reserve Wire Mask And Noise Stress Contract

## Milestone

`E44C_RESERVE_WIRE_MASK_AND_NOISE_STRESS`

## Purpose

E44C stress-tests the E44/E44B Proposal payload bus before locking it as a
default system part.

The current candidate default is:

```text
8 anonymous payload bits
5 minimum active data bits
3 reserve bits
fixed mechanical Proposal header
Agency validation before commit
```

The core question is:

```text
Should reserve bits stay masked/inactive fallback capacity,
or should they be recruited as integrity bits under stress?
```

This is a controlled symbolic/numeric Proposal ABI probe. It is not a raw
language, AGI, consciousness, deployed-model, or model-scale claim.

## Stress Families

```text
clean
reserve_random_noise
reserve_adversarial_noise
reserve_dropout
active_dropout_visible
active_stuck_visible
active_bitflip_silent
burst_noise_silent
known_wire_permutation
unknown_wire_permutation
stale_replay
ground_conflict
partial_support
no_valid_proposal
```

Silent active-bit corruption should not be blindly committed. If the system
cannot safely decode the payload, the correct behavior is `ASK`, not wrong
commit.

## Compared Systems

```text
oracle_integrity_reference
unmasked8_full_payload_decoder
active5_ignore_reserve_mask
active5_visible_dropout_guard
crc3_integrity_guard
universal_mutated_wire_setup
random_policy_control
```

The universal mutated system runs a multi-generation policy selection over the
same stress set. It may select masked reserve behavior, visible-dropout guard,
full 8-bit decoding, or CRC/integrity reserve behavior.

## Metrics

```text
stress_success
action_accuracy
false_commit_rate
wrong_commit_rate
false_ask_rate
reserve_noise_success
reserve_adversarial_success
active_dropout_success
active_bitflip_success
burst_noise_success
known_permutation_success
unknown_permutation_success
accepted/rejected/rollback mutation counts
deterministic replay hash match
```

## Decision Labels

```text
e44c_masked_reserve_default_positive
e44c_integrity_reserve_needed_for_universal_stress
e44c_eight_bit_not_universal_under_silent_noise
e44c_universal_wire_setup_selected
e44c_invalid_artifact_detected
```

## Required Artifacts

```text
backend_manifest.json
stress_generation_report.json
stress_barrage_results.json
universal_mutation_report.json
system_results.json
row_level_results.jsonl
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
stress_table.md
report.md
```
