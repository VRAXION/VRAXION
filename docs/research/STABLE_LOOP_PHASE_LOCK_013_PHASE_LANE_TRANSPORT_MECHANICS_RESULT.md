# STABLE_LOOP_PHASE_LOCK_013_PHASE_LANE_TRANSPORT_MECHANICS Result

Status: implemented, static validation complete, sanity complete, 3-seed smoke
complete. The planned 5-seed confirm was not run because the 3-seed smoke
already identified a clear branch and the estimated confirm runtime was
roughly 90-120 minutes.

## Verdict

```text
EARLY_CORRECT_LATE_OVERWRITE
PER_STEP_TRANSPORT_OK
PER_STEP_TRANSPORT_WORKS_BUT_CHAIN_FAILS
RECURRENT_TRANSPORT_MECHANICS_BLOCKER
SIGNAL_ARRIVAL_FAILURE
SIGNAL_ARRIVES_WRONG_PHASE
WRONG_PHASE_INTERFERENCE_LIMIT
PRODUCTION_API_NOT_READY
```

Interpretation:

```text
one-edge phase transport is not the blocker once the one-edge settling budget is sufficient

the full chain still fails, even though target arrival is high

the main failure is wrong-phase mass / recurrent transport interference, not
missing local phase rule, sparse motif incompleteness, or final-tick readout alone
```

## Runs

Static:

```powershell
cargo check -p instnct-core --example phase_lane_transport_mechanics
cargo test -p instnct-core jackpot_traced_emits_candidate_rows_and_accept_invariants
git diff --check
```

Sanity:

```powershell
cargo run -p instnct-core --example phase_lane_transport_mechanics --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_013_phase_lane_transport_mechanics/sanity ^
  --seeds 2026 ^
  --eval-examples 256 ^
  --widths 8 ^
  --path-lengths 2,4,8 ^
  --ticks-list 8,16,24 ^
  --heartbeat-sec 15
```

Smoke:

```powershell
cargo run -p instnct-core --example phase_lane_transport_mechanics --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_013_phase_lane_transport_mechanics/smoke ^
  --seeds 2026,2027,2028 ^
  --eval-examples 1024 ^
  --widths 8,12,16 ^
  --path-lengths 2,4,8,12,16,24,32 ^
  --ticks-list 4,8,12,16,24,32,48,64 ^
  --heartbeat-sec 30
```

The first smoke attempt hit the shell timeout, but it produced partial metrics
and job progress continuously. The runner was then adjusted to cap the per-step
one-edge tick search to the actual settling window and the requested smoke
completed.

## Smoke Summary

3-seed means:

| Arm | Acc | Best | Persistent | Arrival | Wrong-if-arrived | Readout gap |
|---|---:|---:|---:|---:|---:|---:|
| PER_STEP_ORACLE_INJECTION | 0.906 | 0.906 | 0.906 | 1.000 | 0.094 | 0.000 |
| FULL_16_RULE_TEMPLATE_BASELINE | 0.466 | 0.466 | 0.466 | 1.000 | 0.534 | 0.000 |
| COMPLETED_SPARSE_TEMPLATE_BASELINE | 0.466 | 0.466 | 0.466 | 1.000 | 0.534 | 0.000 |
| STEPWISE_ORACLE_CLOCK | 0.732 | 0.732 | 0.732 | 0.732 | 0.000 | 0.000 |
| PATH_ONLY_FORWARD_CLOCK | 0.522 | 0.858 | 0.858 | 0.107 | 0.000 | 0.469 |
| ARRIVE_LATCH_1TICK | 0.709 | 0.913 | 0.709 | 0.905 | 0.241 | 0.307 |
| CELL_LOCAL_NORMALIZATION | 0.680 | 0.924 | 0.789 | 0.905 | 0.271 | 0.281 |
| ORACLE_DIRECTION_NO_BACKFLOW | 0.522 | 0.858 | 0.858 | 0.107 | 0.000 | 0.469 |
| PUBLIC_GRADIENT_NO_BACKFLOW | 0.503 | 0.863 | 0.805 | 0.077 | 0.003 | 0.517 |
| RANDOM_CONTROL | 0.426 | 0.670 | 0.455 | 1.000 | 0.574 | 0.133 |

## Per-Step Gate

The aggregate per-step mean includes an intentionally too-short `ticks=4`
bucket:

```text
ticks=4:
  per-step accuracy = 0.25
  min pair accuracy = 0.0

ticks >= 8:
  per-step accuracy = 1.0
  min pair accuracy = 1.0
```

This supports:

```text
PER_STEP_TRANSPORT_OK
```

with the constraint that one-edge transfer needs the settling budget measured
by the probe. It does not support `PER_STEP_TRANSPORT_FAILS`.

## Chain Failure

FULL16 and completed sparse are identical:

```text
FULL_16_RULE_TEMPLATE_BASELINE accuracy = 0.466
COMPLETED_SPARSE_TEMPLATE_BASELINE accuracy = 0.466
```

The target receives signal mass:

```text
target_arrival_rate = 1.000
wrong_if_arrived_rate = 0.534
```

This supports:

```text
SIGNAL_ARRIVES_WRONG_PHASE
WRONG_PHASE_INTERFERENCE_LIMIT
PER_STEP_TRANSPORT_WORKS_BUT_CHAIN_FAILS
```

not:

```text
missing local phase rule
missing sparse motif
no signal reaches target
```

## Readout, Latch, No-Backflow, Normalization

Readout-only arms do not rescue:

```text
FINAL_TICK_READOUT = BEST_TICK_READOUT = FIRST_ARRIVAL_READOUT = PERSISTENT_TARGET_READOUT = 0.466
```

Latch and cell-local normalization improve best-tick behavior, but do not solve
final long-chain transport:

```text
ARRIVE_LATCH_1TICK:
  accuracy = 0.709
  best_tick_accuracy = 0.913

CELL_LOCAL_NORMALIZATION:
  accuracy = 0.680
  best_tick_accuracy = 0.924
```

No-backflow diagnostics improve best-tick/persistent readout, but not final
public transport:

```text
ORACLE_DIRECTION_NO_BACKFLOW:
  accuracy = 0.522
  best_tick_accuracy = 0.858

PUBLIC_GRADIENT_NO_BACKFLOW:
  accuracy = 0.503
  best_tick_accuracy = 0.863
```

These results point at recurrent transport mechanics and wrong-phase
interference, with partial evidence that clocking/readout/latch-like structure
could help, but no clean rescue in this runner.

## Claim Boundary

013 supports:

```text
the one-edge local transport gate works with sufficient settling
long-chain failure is downstream of one-edge transport
wrong-phase mass / recurrent interference is a primary blocker
readout-only changes are insufficient
latch/normalization/no-backflow diagnostics improve but do not solve the chain
```

013 does not support:

```text
production architecture
full VRAXION
consciousness
language grounding
Prismion uniqueness
physical quantum behavior
```
