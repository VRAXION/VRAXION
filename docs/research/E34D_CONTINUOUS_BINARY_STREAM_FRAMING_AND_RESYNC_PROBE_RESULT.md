# E34D Continuous Binary Stream Framing And Resync Probe Result

Status: complete.

Decision:

```text
e34d_crc_guard_positive_but_resync_brittle
```

## Summary

E34D tested whether the E34C continuous binary ingress failures were mainly a
packet-boundary problem. The result is a clean partial confirmation:

- `START + LENGTH + CRC + END + requested_feature_guard` removed the normal
  continuous-stream and adversarial-sync-decoy failures.
- It reduced wrong feature commits to zero in the primary run.
- It did not solve bit insertion/deletion. Strict packet framing is brittle
  when the stream slips by one or more bits.

This means EOF/END is useful, but it is not sufficient. The next bottleneck is
bit-slip-tolerant resynchronization/reassembly, not ordinary packet validation.

## Primary Run

Run root:

```text
target/pilot_wave/e34d_continuous_binary_stream_framing_and_resync_probe
```

Primary decision:

```text
e34d_crc_guard_positive_but_resync_brittle
```

Primary key metrics:

```text
multi_frame_resync_guard closed_loop_success = 0.781111
continuous_stream_success                   = 1.000000
adversarial_sync_decoy_success              = 1.000000
packet_clean_success                        = 1.000000
packet_noise_10_success                     = 0.993333
continuous_bit_insert_success               = 0.363333
continuous_bit_drop_success                 = 0.330000
false_frame_commit_rate                     = 0.004227
wrong_feature_write_rate                    = 0.000000
deterministic_replay_match_rate             = 1.000000
accepted_mutations                          = 17
rejected_mutations                          = 1903
rollback_count                              = 1903
```

For comparison, the `start_only_baseline` kept higher overall success but was
unsafe:

```text
start_only_baseline closed_loop_success = 0.962222
start_only false_frame_commit_rate      = 0.028840
start_only wrong_feature_write_rate     = 0.011765
```

So the loose decoder is tolerant but dirty. The strict decoder is clean but
brittle under bit slips.

## CPU Confirm

Run root:

```text
target/pilot_wave/e34d_continuous_binary_stream_framing_and_resync_probe_cpu_confirm
```

CPU confirm decision:

```text
e34d_crc_guard_positive_but_resync_brittle
```

CPU confirm key metrics:

```text
multi_frame_resync_guard closed_loop_success = 0.793939
continuous_stream_success                   = 1.000000
adversarial_sync_decoy_success              = 1.000000
continuous_bit_insert_success               = 0.431818
continuous_bit_drop_success                 = 0.345455
false_frame_commit_rate                     = 0.004950
wrong_feature_write_rate                    = 0.000000
accepted_mutations                          = 16
rejected_mutations                          = 1520
```

The confirm reproduced the same pattern: clean normal framing and adversarial
decoy rejection, but weak bit-slip recovery.

## Extra Stress Seeds

Extra non-canonical stress lanes were run under `target/pilot_wave/` only to
falsify the diagnosis. They are not part of the sample pack.

```text
seed34403: decision=e34d_framing_still_bottleneck, multi=0.786364, insert=0.400000, drop=0.327273, wrong_feature=0.000000
seed34404: decision=e34d_crc_guard_positive_but_resync_brittle, multi=0.796212, insert=0.422727, drop=0.359091, wrong_feature=0.000000
seed34405: decision=e34d_crc_guard_positive_but_resync_brittle, multi=0.792708, insert=0.437500, drop=0.343750, wrong_feature=0.000000
seed34406: decision=e34d_framing_still_bottleneck, multi=0.796875, insert=0.387500, drop=0.393750, wrong_feature=0.000000
seed34407: decision=e34d_framing_still_bottleneck, multi=0.780208, insert=0.356250, drop=0.331250, wrong_feature=0.000000
```

These stress seeds support the same interpretation: requested-feature guarded
CRC framing prevents wrong feature commits, while bit insertion/deletion remains
the hard failure family.

## Checker

Primary target checker:

```text
passed = true
failure_count = 0
```

Primary sample-only checker:

```text
passed = true
failure_count = 0
```

CPU confirm target checker:

```text
passed = true
failure_count = 0
```

CPU confirm sample-only checker:

```text
passed = true
failure_count = 0
```

## Interpretation

E34D answers the EOF question directly:

```text
EOF/END helps, but it does not solve the real continuous-stream problem alone.
```

The useful protocol is closer to:

```text
START_SYNC + LENGTH + CRC + END_SYNC + requested_feature_guard
```

This protocol solves:

```text
normal packet stream
normal continuous stream
adversarial sync decoys
wrong requested-feature commits
```

It does not solve:

```text
bit insertion
bit deletion
loss of frame phase
stream reassembly after slip
```

The remaining failure is not mainly Flow Field reasoning. It is an Ingress Codec
framing/resynchronization problem.

## Recommended Next Step

```text
E34E_BITSLIP_TOLERANT_STREAM_REASSEMBLY_PROBE
```

Test:

```text
multi-offset frame hypotheses
sliding-window CRC voting
soft frame confidence
packet repetition with majority decode
resync after insert/drop
commit only after requested_feature and frame confidence both pass
```

Positive target:

```text
keep wrong_feature_write_rate near 0
raise continuous_bit_insert/drop success above 0.90
preserve adversarial_sync_decoy success at 1.0
```

Boundary: E34D is a controlled binary ingress probe. It does not make language,
AGI, consciousness, or deployed-model claims.
