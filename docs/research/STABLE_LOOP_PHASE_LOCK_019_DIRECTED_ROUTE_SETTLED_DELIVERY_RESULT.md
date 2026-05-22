# STABLE_LOOP_PHASE_LOCK_019_DIRECTED_ROUTE_SETTLED_DELIVERY Result

Status: complete.

## Question

```text
Can receive-committed target delivery preserve the clean directed-route pulse as
stable final output?
```

## Answer

```text
YES, but only in the diagnostic true-path route setting.
```

019 shows that a correct directed route plus receive-committed target delivery
turns the clean pulse arrival from 018 into stable settled final output when the
tick budget is physically sufficient for the path length.

It does not solve public routing. Public gradient/monotone routes still fail the
delivery gate.

## Smoke Result

Run:

```powershell
cargo run -p instnct-core --example phase_lane_directed_route_settled_delivery --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_019_directed_route_settled_delivery/smoke ^
  --seeds 2026,2027,2028 ^
  --eval-examples 1024 ^
  --widths 8,12,16 ^
  --path-lengths 4,8,16,24,32 ^
  --ticks-list 8,16,24,32,48 ^
  --heartbeat-sec 30
```

Key rows:

```text
TRUE_PATH_DIRECTED_ROUTE_RECEIVE_COMMIT_LEDGER_SUM:
  phase_final_accuracy = 0.760
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  duplicate_delivery_rate = 0.000
  stale_delivery_rate = 0.000

TRUE_PATH_DIRECTED_ROUTE_FINAL_TICK:
  phase_final_accuracy = 0.160
  sufficient_tick_final_accuracy = 0.211

TRUE_PATH_DIRECTED_ROUTE_TARGET_LATCH_1TICK:
  phase_final_accuracy = 0.160
  sufficient_tick_final_accuracy = 0.211

TRUE_PATH_PLUS_REVERSE_RECEIVE_COMMIT_LEDGER:
  sufficient_tick_final_accuracy = 0.899
  long_path_accuracy = 0.788
  wrong_if_delivered_rate = 0.366
  duplicate_delivery_rate = 0.509
  stale_delivery_rate = 0.502

PUBLIC_GRADIENT_RECEIVE_COMMIT_LEDGER:
  sufficient_tick_final_accuracy = 0.698
  long_path_accuracy = 0.485
  wrong_if_delivered_rate = 0.295

RANDOM_PHASE_RULE_RECEIVE_COMMIT_LEDGER:
  phase_final_accuracy = 0.355
  sufficient_tick_final_accuracy = 0.467

RANDOM_SAME_COUNT_RECEIVE_COMMIT_LEDGER:
  phase_final_accuracy = 0.006

DIRECTION_SHUFFLE_RECEIVE_COMMIT_LEDGER:
  phase_final_accuracy = 0.000
```

The all-bucket `phase_final_accuracy` for true-path ledger arms is below 0.95
because the sweep intentionally includes `ticks < path_length` buckets where
delivery cannot occur yet. The delivery claim therefore uses
`sufficient_tick_final_accuracy`, long-path sufficient buckets, and wrong
delivery metrics.

## Verdicts

```text
BEST_TICK_ONLY_NOT_STABLE
DIRECTED_ROUTE_DELIVERY_SOLVES_DIAGNOSTIC
FINAL_READOUT_TIMING_LIMIT_CONFIRMED
RECEIVE_COMMIT_LEDGER_MAX_WORKS
RECEIVE_COMMIT_LEDGER_STABILIZES_FINAL_READOUT
RECEIVE_COMMIT_LEDGER_SUM_WORKS
REVERSE_EDGES_BREAK_SETTLED_DELIVERY
TARGET_LATCH_INSUFFICIENT
TRUE_PATH_DELIVERY_WORKS_PUBLIC_ROUTING_OPEN
PUBLIC_ROUTE_DELIVERY_FAILS
ROUTING_POLICY_STILL_BLOCKED
RANDOM_DIRECTED_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Interpretation

```text
018:
  a clean directed true-path route can carry the pulse, but final tick misses it.

019:
  receive-committed target delivery preserves that pulse as stable settled output.

Still open:
  public routing / route construction.
```

Reverse edges still degrade settled delivery, producing wrong, duplicate, and
stale deliveries. A one-tick target latch is insufficient; a receive-commit
ledger is the clean delivery primitive in this diagnostic setup.

Public gradient/monotone routing improves over bidirectional baseline but does
not pass the route-delivery gate. This keeps the next blocker sharply scoped:
the transport/delivery pieces work under the correct directed route, but the
system still needs a public way to construct or choose that route.

## Static Checks

```text
cargo check -p instnct-core --example phase_lane_directed_route_settled_delivery
  PASS

cargo test -p instnct-core jackpot_traced_emits_candidate_rows_and_accept_invariants
  PASS

git diff --check
  PASS
```

## Claim Boundary

019 supports only:

```text
correct directed route + receive-committed settled delivery works diagnostically
final-tick readout was a timing/delivery problem
reverse edges break settled delivery
public routing remains unsolved
```

019 does not claim production architecture, FlyWire validation, public route
search, full VRAXION, language grounding, consciousness, Prismion uniqueness,
biological equivalence, or physical quantum behavior.
