# E34C Active Evidence World Binary Ingress Smoke Result

Decision:

```text
decision = e34c_binary_packet_confirmed_framing_bottleneck
target_checker_failure_count = 0
sample_only_checker_passed = true
gradient_descent_used = false
optimizer_used = false
backprop_used = false
```

## What Ran

E34C reuses the E34 active-evidence world, but replaces text/tuple observations with binary ingress:

```text
binary packet / stream
-> Ingress Codec
-> trusted evidence write into Flow Field
-> active INSPECT_BITS(feature)
-> ANSWER only after the state is resolved
```

Systems:

```text
learned_binary_ingress_policy
forced_initial_binary_answer
random_binary_action_control
ask_all_binary_until_unique
first_sync_shortcut_control
oracle_tuple_reference
```

Splits:

```text
packet_clean
packet_noise_02
packet_noise_05
packet_noise_10
source_aware_rumor
continuous_stream
adversarial_decoy
```

The run used mutation/rollback only. No gradient descent, optimizer, or backprop path is used.

## Primary Result

Run root:

```text
target/pilot_wave/e34c_active_evidence_world_binary_ingress_smoke
```

Primary learned policy:

```text
closed_loop_success              = 0.989286
answer_correct                   = 0.993254
trace_exact                      = 0.989286
wrong_confident_answer           = 0.000397
binary_ingress_accuracy          = 0.992130
accepted_flow_write_accuracy     = 0.996570
frame_sync_accuracy              = 0.996759
avg_steps                        = 4.033333
packet_min_success               = 0.994444
continuous_stream_success        = 0.958333
accepted_mutations               = 7
rejected_mutations               = 2153
rollback_count                   = 2153
deterministic_replay_match_rate  = 1.000000
```

Primary split success:

```text
packet_clean        = 1.000000
packet_noise_02     = 1.000000
packet_noise_05     = 1.000000
packet_noise_10     = 0.994444
source_aware_rumor  = 1.000000
continuous_stream   = 0.958333
adversarial_decoy   = 0.972222
```

Controls:

```text
forced_initial_binary_answer.closed_loop_success = 0.000000
forced_initial_binary_answer.wrong_confident     = 1.000000
random_binary_action_control.closed_loop_success = 0.459524
ask_all_binary_until_unique.closed_loop_success  = 0.962698
ask_all_binary_until_unique.avg_steps            = 4.400000
first_sync_shortcut.adversarial_success          = 0.000000
oracle_tuple_reference.closed_loop_success       = 1.000000
```

## CPU Confirm

Run root:

```text
target/pilot_wave/e34c_active_evidence_world_binary_ingress_smoke_cpu_confirm
```

Independent seed confirm:

```text
closed_loop_success              = 0.988312
binary_ingress_accuracy          = 0.991374
accepted_flow_write_accuracy     = 0.996158
frame_sync_accuracy              = 0.996158
packet_min_success               = 1.000000
continuous_stream_success        = 0.954545
accepted_mutations               = 2
rejected_mutations               = 1726
rollback_count                   = 1726
deterministic_replay_match_rate  = 1.000000
```

The confirm reproduces the primary interpretation: binary packets are clean, but continuous stream framing remains below packet performance.

## Important Implementation Audit

During smoke validation, two harness issues were found and fixed before the evidence run:

```text
1. REPEAT=5 still used the old REPEAT=3 majority threshold.
   Fix: majority_decode now uses a true majority threshold.

2. Corrupt/parity-fail packets were being accepted into the Flow Field.
   Fix: learned ingress accepts a Flow write only when trust, parity, and valid feature ID pass.
   Corrupt packets can be rejected and re-requested deterministically.
```

This matters scientifically: the positive result is not "all raw packets decode perfectly." The stronger statement is:

```text
The system can use binary packets as evidence, reject corrupt packets before stable Flow commit,
and continue active evidence seeking until the state is resolved.
```

## Interpretation

E34C confirms the binary ingress path on this controlled active-evidence proxy:

```text
clean packets          -> solved
2% noisy packets       -> solved
5% noisy packets       -> solved
10% noisy packets      -> near-clean with parity/retry hygiene
source-aware rumor     -> solved
adversarial decoy      -> high but not perfect
continuous bitstream   -> still lower than packet mode
```

So the current bottleneck is not "binary input cannot be used." The remaining weakness is frame/synchronization under continuous stream and decoy framing conditions.

Boundary:

```text
This is not raw language reasoning, AGI, consciousness, deployed-model behavior, or model-scale evidence.
It is a controlled binary-ingress active-evidence harness.
```

## Checker Status

```text
primary target checker failure_count = 0
primary sample-only checker failure_count = 0
cpu confirm target checker failure_count = 0
cpu confirm sample-only checker failure_count = 0
```

## Recommended Next Step

```text
E34D_CONTINUOUS_BINARY_STREAM_FRAMING_AND_RESYNC_PROBE
```

Focus:

```text
sync acquisition
resync after bit loss/insertions
multiple candidate frame hypotheses
adversarial sync decoys
frame confidence before Flow commit
packet vs continuous stream cost/latency tradeoff
```
