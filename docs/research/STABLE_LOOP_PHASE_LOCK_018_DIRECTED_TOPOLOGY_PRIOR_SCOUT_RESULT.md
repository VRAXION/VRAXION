# STABLE_LOOP_PHASE_LOCK_018_DIRECTED_TOPOLOGY_PRIOR_SCOUT Result

Status: implemented, static validation complete, quick selector complete.

## Question

```text
If the phase signal lives in a directed mutation graph instead of a bidirectional
spatial broadcast grid, does the echo blocker disappear?
```

## Data Access Note

Codex app download currently requires sign-in. The FlyWire connectivity dump is
available without auth through Zenodo DOI `10.5281/zenodo.10676866`.

Relevant no-auth file for a later real FlyWire topology ingest:

```text
proofread_connections_783.feather
size: 852,022,274 bytes
columns include: pre_pt_root_id, post_pt_root_id, neuropil, syn_count
```

018 does not depend on downloading this file. It first tests the directed-route
mechanism itself.

## Planned Static Checks

Passed:

```powershell
cargo check -p instnct-core --example phase_lane_directed_topology_prior_scout
cargo test -p instnct-core jackpot_traced_emits_candidate_rows_and_accept_invariants
git diff --check
```

## Planned Quick Selector

Run:

```powershell
cargo run -p instnct-core --example phase_lane_directed_topology_prior_scout --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_018_directed_topology_prior_scout/quick_v3 ^
  --seeds 2026 ^
  --eval-examples 512 ^
  --widths 8,12 ^
  --path-lengths 4,8,16,24 ^
  --ticks-list 8,16,24,32 ^
  --heartbeat-sec 15
```

Rows written: `2464`.

## Key Rows

| Arm | Final Acc | Best Tick | Sufficient Tick Best | Wrong-if-arrived | Reciprocal | Backflow |
|---|---:|---:|---:|---:|---:|---:|
| `TRUE_PATH_DIRECTED_ROUTE_DIAGNOSTIC` | `0.188` | `0.812` | `1.000` | `0.000` | `0.000` | `0.000` |
| `TRUE_PATH_PLUS_REVERSE_ABLATION` | `0.607` | `0.812` | `1.000` | `0.205` | `1.000` | `0.500` |
| `BIDIRECTIONAL_GRID_BASELINE` | `0.547` | `0.783` | `0.794` | `0.453` | `1.000` | `0.335` |
| `PUBLIC_GRADIENT_DAG` | `0.031` | `0.634` | `0.662` | `0.000` | `0.000` | `0.128` |
| `PUBLIC_MONOTONE_XY_ROUTE` | `0.031` | `0.634` | `0.662` | `0.000` | `0.000` | `0.128` |
| `RANDOM_PHASE_RULE_CONTROL` | `0.040` | `0.185` | `0.228` | `0.147` | `0.000` | `0.000` |
| `RANDOM_SAME_COUNT_DIRECTED` | `0.000` | `0.009` | `0.008` | `0.000` | `0.307` | `0.322` |

## Interpretation

The user hypothesis is mechanically correct in the diagnostic setting:

```text
If the graph contains only the correct directed route edges,
the phase pulse arrives cleanly.
```

Evidence:

```text
TRUE_PATH_DIRECTED_ROUTE_DIAGNOSTIC:
  sufficient_tick_best_accuracy = 1.000
  wrong_if_arrived_rate = 0.000
  reciprocal_edge_fraction = 0.000
  backflow_edge_fraction = 0.000
```

But it is not stable as a final-tick transport mechanism:

```text
phase_final_accuracy = 0.188
best_tick_accuracy = 0.812
```

This means the directed route carries a clean pulse, but without target memory /
settled delivery / exact readout timing, the final tick often misses it.

Reverse/reciprocal edges reintroduce echo:

```text
TRUE_PATH_PLUS_REVERSE_ABLATION:
  wrong_if_arrived_rate = 0.205
  reciprocal_edge_fraction = 1.000
  backflow_edge_fraction = 0.500
```

The public routing/topology priors did not solve this:

```text
PUBLIC_GRADIENT_DAG and PUBLIC_MONOTONE_XY_ROUTE:
  final accuracy = 0.031
  sufficient_tick_best_accuracy = 0.662
```

Controls remain weak:

```text
RANDOM_PHASE_RULE_CONTROL sufficient_tick_best_accuracy = 0.228
RANDOM_SAME_COUNT_DIRECTED sufficient_tick_best_accuracy = 0.008
```

## Current Read

```text
Directed edges are the right carrier shape.
Correct forward-only path edges eliminate the wrong-phase echo diagnostically.
Reverse edges reintroduce the problem.

But public construction/routing and stable delivery/readout are still unsolved.
```

## Verdict

```text
DIRECTED_ROUTE_HAS_CLEAN_ARRIVAL_DIAGNOSTIC
FINAL_READOUT_TIMING_LIMIT
FLYWIRE_EXACT_WIRING_NOT_REQUIRED
NO_DIRECTED_TOPOLOGY_SIGNAL
PRODUCTION_API_NOT_READY
RANDOM_DIRECTED_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
REVERSE_EDGES_REINTRODUCE_ECHO
```

## Claim Boundary

018 is a directed topology scout only. It cannot claim production architecture,
full VRAXION, language grounding, consciousness, FlyWire validation, Prismion
uniqueness, biological equivalence, or physical quantum behavior.
