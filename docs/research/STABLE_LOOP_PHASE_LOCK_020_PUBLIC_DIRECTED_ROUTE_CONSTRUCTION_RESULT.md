# STABLE_LOOP_PHASE_LOCK_020_PUBLIC_DIRECTED_ROUTE_CONSTRUCTION Result

Status: complete.

## Question

```text
Can public wall/source/target information construct a clean forward-only route
now that receive-commit delivery works?
```

## Answer

```text
NO for the tested public priors.
```

The 019 true-path upper bound still works, confirming the delivery substrate.
However, the public route constructors tested here do not recover the correct
directed route. Public shortest-path and distance-field priors often select
shortcuts/branches that are valid geometrically but wrong for the phase-label
path semantics.

## Quick Selector

Run:

```powershell
cargo run -p instnct-core --example phase_lane_public_directed_route_construction --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_020_public_directed_route_construction/quick_sufficient_gatefix ^
  --seeds 2026 ^
  --eval-examples 512 ^
  --widths 8,12 ^
  --path-lengths 4,8,16,24 ^
  --ticks-list 8,16,24,32 ^
  --heartbeat-sec 15
```

Key rows:

```text
TRUE_PATH_RECEIVE_COMMIT_LEDGER_DIAGNOSTIC:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  directed_edge_count = 13.0

PUBLIC_DISTANCE_FIELD_SINGLE_SUCCESSOR_RECEIVE_COMMIT_LEDGER:
  sufficient_tick_final_accuracy = 0.816
  long_path_accuracy = 0.636
  wrong_if_delivered_rate = 0.214
  directed_edge_count = 15.0

PUBLIC_BFS_SHORTEST_ROUTE_RECEIVE_COMMIT_LEDGER:
  sufficient_tick_final_accuracy = 0.654
  long_path_accuracy = 0.414
  wrong_if_delivered_rate = 0.375
  directed_edge_count = 5.0

PUBLIC_GRADIENT_RECEIVE_COMMIT_LEDGER:
  sufficient_tick_final_accuracy = 0.662
  long_path_accuracy = 0.436
  wrong_if_delivered_rate = 0.274

PUBLIC_MONOTONE_RECEIVE_COMMIT_LEDGER:
  sufficient_tick_final_accuracy = 0.662
  long_path_accuracy = 0.436
  wrong_if_delivered_rate = 0.274

RANDOM_SAME_COUNT_RECEIVE_COMMIT_LEDGER:
  phase_final_accuracy = 0.000

DIRECTION_SHUFFLE_RECEIVE_COMMIT_LEDGER:
  phase_final_accuracy = 0.000

RANDOM_PHASE_RULE_RECEIVE_COMMIT_LEDGER:
  sufficient_tick_final_accuracy = 0.231
```

## Smoke Gate

Smoke was skipped because no public non-diagnostic route arm passed the quick
delivery gate.

## Verdict

```text
TRUE_PATH_UPPER_BOUND_CONFIRMED
PUBLIC_ROUTE_CONSTRUCTION_FAILS
ROUTING_POLICY_STILL_BLOCKED
PUBLIC_GRADIENT_STILL_FAILS
PUBLIC_MONOTONE_STILL_FAILS
PUBLIC_WALL_FOLLOW_FAILS
REVERSE_EDGES_BREAK_SETTLED_DELIVERY
RANDOM_DIRECTED_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Interpretation

```text
019 solved delivery if the correct directed route is already supplied.

020 shows that simple public route priors do not construct that route.
```

The public shortest-path route is actively misleading in this setup: it finds a
short geometric route through public free cells, but that route is not the
phase-label path and therefore produces wrong deliveries. Distance-field and
acyclic no-reciprocal priors reduce reciprocal echo but still fail because they
do not encode the intended corridor identity.

This narrows the next blocker:

```text
public route construction needs route identity / breadcrumb / learned frontier
state, not just public shortest path or distance-to-target.
```

## Static Checks

```text
cargo check -p instnct-core --example phase_lane_public_directed_route_construction
  PASS

cargo test -p instnct-core jackpot_traced_emits_candidate_rows_and_accept_invariants
  PASS

git diff --check
  PASS
```

## Claim Boundary

020 is a public route-construction probe only. It does not claim production
architecture, FlyWire validation, full VRAXION, language grounding,
consciousness, Prismion uniqueness, biological equivalence, or physical quantum
behavior.
