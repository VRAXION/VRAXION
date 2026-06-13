# E59 Bit-Slip Tolerant Reassembly Lock

Status: completed and checker validated.

## Decision

```text
decision = e59_bitslip_tolerant_reassembly_locked
checker_failure_count = 0
sample_only_checker_passed = true
run_id = 613b5e9f6cfcdee2
gradient_descent_used = false
optimizer_used = false
backprop_used = false
```

## Systems

| system | closed loop | bit slip | false frame | wrong feature | false commit | net utility |
|---|---:|---:|---:|---:|---:|---:|
| strict_single_offset_full_guard | 0.600000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 0.345000 |
| end_marker_only_decoder | 0.105773 | 0.003762 | 0.900000 | 0.866840 | 0.200000 | -1.891042 |
| loose_start_only_decoder | 0.424826 | 0.756800 | 0.502214 | 0.351910 | 0.300000 | -0.847263 |
| multi_offset_crc_no_feature_guard | 0.700000 | 1.000000 | 0.200000 | 0.200000 | 0.200000 | 0.024469 |
| multi_offset_crc_requested_no_ambiguity_guard | 0.900000 | 1.000000 | 0.000000 | 0.000000 | 0.100000 | 0.749469 |
| bitslip_tolerant_reassembly_lock | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.984469 |
| oracle_frame_reference | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| random_control | 0.142231 | 0.008825 | 0.547700 | 0.530469 | 0.164323 | -1.281600 |

## What Was Locked

```text
bitstream
-> multi-offset frame hypotheses
-> START/LENGTH/PAYLOAD/CRC/END validation
-> decoded_feature == requested_feature
-> conflict/ambiguity rejection
-> Agency commit only after guards pass
```

This is the concrete E58 bitslip fix. It does not trust one nominal packet
boundary and it does not treat EOF/END as enough.

## Why The Extra Guards Matter

Failure examples from the checked evidence run:

```text
strict_single_offset_full_guard:
  bit insert before frame -> false defer
  reason: no resync after insertion/drop before frame

multi_offset_crc_no_feature_guard:
  valid CRC decoy -> wrong feature commit
  reason: CRC proves frame integrity, not that it answers the requested feature

multi_offset_crc_requested_no_ambiguity_guard:
  two valid requested-feature frames disagree -> false commit
  reason: requested_feature alone is unsafe without conflict/ambiguity handling

loose_start_only_decoder:
  false START-like decoy -> false frame commit
  reason: START without CRC/END/request guard is unsafe
```

## Concrete Examples

- `insert_before_frame_recovered`: offset scan finds the real frame after one inserted bit and commits only after requested-feature validation.
- `wrong_feature_decoy_rejected`: a valid CRC frame for the wrong feature is rejected or skipped.
- `conflicting_duplicate_deferred`: conflicting valid frames for the requested feature cause DEFER instead of first-frame commit.
- `truncated_packet_deferred`: START/LENGTH without full CRC/END cannot commit.

## Interpretation

E59 converts the E58 candidate into a locked binary ingress component. The
remaining safe rule is:

```text
No binary evidence enters Flow/Ground from raw bitstream unless the reassembly
candidate passes structure, integrity, requested-feature, and ambiguity guards.
```

EOF helps, but is not enough. CRC helps, but is not enough. The requested-feature
guard and ambiguity guard are mandatory.

## Boundary

E59 locks the binary ingress bit-slip tolerant reassembly layer for the
controlled VRAXION symbolic/numeric IO stack. It is not a raw language, AGI,
consciousness, deployment, or model-scale claim.
