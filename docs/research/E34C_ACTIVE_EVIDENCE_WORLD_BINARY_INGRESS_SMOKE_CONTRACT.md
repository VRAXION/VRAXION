# E34C Active Evidence World Binary Ingress Smoke Contract

## Purpose

`E34C_ACTIVE_EVIDENCE_WORLD_BINARY_INGRESS_SMOKE` tests whether the E34A/E34B
active evidence loop survives when observations arrive as binary packets or
continuous bitstreams.

The core question:

```text
If text ingress is removed, can the system still build evidence from raw-ish
binary packets, and where does continuous bitstream framing break?
```

## Boundary

E34C is a deterministic symbolic/binary active-evidence probe. It is not a raw
language understanding proof, chatbot benchmark, AGI claim, consciousness claim,
deployed-model claim, or model-scale claim.

## Binary Packet

Each observation encodes:

```text
sync pattern
feature_id bits
value bit
trust/source bit
temporal bit
parity/check bit
filler bits
```

The primary system sees binary packet/stream bits and may only update the Flow
Field after binary ingress decoding.

## Splits

```text
packet_clean
packet_noise_02
packet_noise_05
packet_noise_10
source_aware_rumor
continuous_stream
adversarial_decoy
```

## Systems

```text
learned_binary_ingress_policy
forced_initial_binary_answer
random_binary_action_control
ask_all_binary_until_unique
first_sync_shortcut_control
oracle_tuple_reference
```

`oracle_tuple_reference` is an invalid ceiling/control. `first_sync_shortcut_control`
tests whether simply taking the first sync-looking frame is enough.

## Metrics

```text
closed_loop_success
answer_correct
trace_exact
binary_ingress_accuracy
frame_sync_accuracy
wrong_confident_answer
false_ask
redundant_actions
average_steps_to_answer
first_useful_evidence_action
accepted/rejected/rollback mutation counts
parameter diff/hash
deterministic replay
checker failure count
```

## Decisions

```text
e34c_binary_packet_confirmed_framing_bottleneck
e34c_binary_ingress_confirmed
e34c_binary_ingress_codec_bottleneck
e34c_binary_ingress_failed
e34c_artifact_invalid
```

Positive requires:

```text
learned closed-loop success >= 0.95
binary ingress accuracy >= 0.96
wrong confident answer <= 0.03
learned average steps < ask-all baseline
random control substantially worse
forced initial answer fails as expected
first-sync shortcut fails on adversarial decoys
accepted and rejected mutations both present
deterministic replay passes
checker failure count = 0
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
binary_ingress_report.json
policy_initial_state.json
policy_final_state.json
parameter_diff.json
mutation_history.jsonl
row_level_results.jsonl
system_results.json
aggregate_metrics.json
deterministic_replay.json
resource_usage_report.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
report.md
```

The run must write progress and heartbeat artifacts during execution. No
black-box long run is allowed.
