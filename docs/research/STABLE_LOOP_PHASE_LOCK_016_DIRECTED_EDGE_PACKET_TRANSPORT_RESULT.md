# STABLE_LOOP_PHASE_LOCK_016_DIRECTED_EDGE_PACKET_TRANSPORT Result

Status: implemented, static validation complete, Phase A quick selector complete.

## Question

```text
Can directed edge-state packet transport prevent node-broadcast wrong-phase
echo and restore stable long-chain phase transport?
```

## Planned Runs

Static:

```powershell
cargo check -p instnct-core --example phase_lane_directed_edge_packet_transport
cargo test -p instnct-core jackpot_traced_emits_candidate_rows_and_accept_invariants
git diff --check
```

Phase A quick selector:

```powershell
cargo run -p instnct-core --example phase_lane_directed_edge_packet_transport --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_016_directed_edge_packet_transport/quick ^
  --seeds 2026 ^
  --eval-examples 512 ^
  --widths 8,12 ^
  --path-lengths 4,8,16,24 ^
  --ticks-list 8,16,24,32 ^
  --heartbeat-sec 15
```

Phase B smoke only if Phase A shows signal:

```powershell
cargo run -p instnct-core --example phase_lane_directed_edge_packet_transport --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_016_directed_edge_packet_transport/smoke ^
  --seeds 2026,2027,2028 ^
  --eval-examples 1024 ^
  --widths 8,12,16 ^
  --path-lengths 4,8,16,24,32 ^
  --ticks-list 8,16,24,32,48 ^
  --heartbeat-sec 30
```

## Verdict

```text
EDGE_PACKET_KILLS_TRANSPORT
EDGE_PACKET_RANDOM_RULE_FAILS
EDGE_PACKET_REDUCES_WRONG_PHASE
NO_EDGE_PACKET_SIGNAL
PRODUCTION_API_NOT_READY
TTL_TOO_LONG_ALLOWS_ECHO
TTL_TOO_SHORT_KILLS_TRANSPORT
```

Interpretation:

```text
the tested public directed-edge packet variants do not pass the signal gate

directed edge packets reduce some wrong-phase/readout effects, but not enough
to beat node broadcast on long-path/family-min/final stability

target-settled readout is the best diagnostic row, but it is explicitly not a
transport-solved claim

Phase B smoke is skipped because Phase A found no public non-diagnostic signal
```

## Phase A Quick Summary

Quick root:

```text
target/pilot_wave/stable_loop_phase_lock_016_directed_edge_packet_transport/quick
```

Ranking:

| Arm | Acc | Long | Family min | Wrong-if-arrived | Random rule | Random route | Fanout | Signal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EDGE_PACKET_PLUS_TARGET_SETTLED_READOUT | 0.904 | 0.872 | 0.719 | 0.065 | 0.335 | 0.275 | 2.343 | false |
| BEST_PUBLIC_COMBO_014 | 0.788 | 0.717 | 0.625 | 0.181 | 0.335 | 0.275 | 0.000 | false |
| NODE_BROADCAST_BASELINE_014 | 0.734 | 0.771 | 0.469 | 0.234 | 0.335 | 0.275 | 0.000 | false |
| MOMENTUM_LANES_015_BASELINE | 0.734 | 0.771 | 0.469 | 0.234 | 0.335 | 0.275 | 0.000 | false |
| EDGE_PACKET_FLOOD | 0.685 | 0.693 | 0.422 | 0.283 | 0.335 | 0.275 | 2.347 | false |
| EDGE_PACKET_CONSUME_PHASE_ONLY | 0.674 | 0.735 | 0.469 | 0.295 | 0.335 | 0.275 | 2.343 | false |
| EDGE_PACKET_CONSUME_FULL_EDGE | 0.674 | 0.735 | 0.469 | 0.295 | 0.335 | 0.275 | 2.343 | false |
| EDGE_PACKET_PLUS_CELL_LOCAL_NORMALIZATION | 0.672 | 0.711 | 0.328 | 0.297 | 0.335 | 0.275 | 2.343 | false |
| RANDOM_ROUTE_EDGE_PACKET_CONTROL | 0.275 | 0.259 | 0.250 | 0.306 | 0.335 | 0.275 | 1.000 | false |
| EDGE_PACKET_TTL_2X_PATH | 0.569 | 0.595 | 0.422 | 0.400 | 0.335 | 0.275 | 2.345 | false |
| EDGE_PACKET_ORACLE_ROUTE_CORRECT_PHASE_DIAGNOSTIC | 0.408 | 0.438 | 0.391 | 0.451 | 0.335 | 0.275 | 1.000 | false |
| EDGE_PACKET_NO_REENTRY | 0.462 | 0.509 | 0.391 | 0.507 | 0.335 | 0.275 | 1.451 | false |
| EDGE_PACKET_TTL_PATH_PLUS_2 | 0.444 | 0.429 | 0.344 | 0.525 | 0.335 | 0.275 | 2.339 | false |
| EDGE_PACKET_ORACLE_ROUTE_RANDOM_PHASE_DIAGNOSTIC | 0.304 | 0.298 | 0.250 | 0.556 | 0.335 | 0.275 | 1.000 | false |
| EDGE_PACKET_TTL_PATH | 0.402 | 0.429 | 0.297 | 0.567 | 0.335 | 0.275 | 2.334 | false |
| RANDOM_RULE_EDGE_PACKET_CONTROL | 0.335 | 0.345 | 0.141 | 0.634 | 0.335 | 0.275 | 2.334 | false |
| EDGE_PACKET_PUBLIC_GRADIENT | 0.330 | 0.333 | 0.312 | 0.638 | 0.335 | 0.275 | 1.059 | false |
| EDGE_PACKET_PLUS_PUBLIC_NO_BACKFLOW | 0.330 | 0.333 | 0.312 | 0.638 | 0.335 | 0.275 | 1.059 | false |

## Output Artifacts

The quick run wrote:

```text
queue.json
progress.jsonl
metrics.jsonl
edge_packet_metrics.jsonl
routing_metrics.jsonl
ttl_metrics.jsonl
consume_metrics.jsonl
reentry_metrics.jsonl
family_metrics.jsonl
counterfactual_metrics.jsonl
random_control_metrics.jsonl
locality_audit.jsonl
mechanism_ranking.json
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

No raw `target/` outputs are committed.

## Claim Boundary

016 is a toy transport-carrier probe only. It cannot claim production
architecture, full VRAXION, language grounding, consciousness, Prismion
uniqueness, or physical quantum behavior.
