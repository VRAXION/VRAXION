# STABLE_LOOP_PHASE_LOCK_012_PHASE_LANE_HORIZON_SCALING Result

Status: implemented, static validation complete, sanity complete, 3-seed smoke
complete, 5-seed confirm complete.

## Verdict

```text
FULL16_REFERENCE_BREAKS_ON_LONG_PATHS
HORIZON_LIMIT_IDENTIFIED
RANDOM_CONTROL_FAILS
RULE_TEMPLATE_STABLE_BUT_SETTLING_LIMITED
SPARSE_EQUALS_FULL16_HORIZON
PRODUCTION_API_NOT_READY
```

Interpretation:

```text
the completed 16-pair local phase rule is not the blocker

FULL_16_RULE_TEMPLATE, COMMON_CORE_15_PLUS_MISSING_1_2_3, and
DENSE_009_REFERENCE have the same horizon curve

the current recurrent substrate carries very short signals, but fails the
long-path horizon sweep under this settling/readout scheme

source persistence increases target power but does not rescue the horizon
```

## Runs

Static:

```powershell
cargo check -p instnct-core --example phase_lane_horizon_scaling
cargo test -p instnct-core jackpot_traced_emits_candidate_rows_and_accept_invariants
git diff --check
```

Sanity:

```powershell
cargo run -p instnct-core --example phase_lane_horizon_scaling --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_012_phase_lane_horizon_scaling/sanity ^
  --seeds 2026 ^
  --eval-examples 256 ^
  --widths 8 ^
  --path-lengths 2,4,8 ^
  --ticks-list 8,16,24 ^
  --heartbeat-sec 15
```

Smoke:

```powershell
cargo run -p instnct-core --example phase_lane_horizon_scaling --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_012_phase_lane_horizon_scaling/smoke ^
  --seeds 2026,2027,2028 ^
  --eval-examples 1024 ^
  --widths 8,12,16 ^
  --path-lengths 2,4,8,12,16,24,32,48 ^
  --ticks-list 4,8,12,16,24,32,48,64 ^
  --heartbeat-sec 30
```

Confirm:

```powershell
cargo run -p instnct-core --example phase_lane_horizon_scaling --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_012_phase_lane_horizon_scaling/confirm ^
  --seeds 2026-2030 ^
  --eval-examples 2048 ^
  --widths 8,12,16 ^
  --path-lengths 2,4,8,12,16,24,32,48 ^
  --ticks-list 4,8,12,16,24,32,48,64 ^
  --heartbeat-sec 30
```

## Confirm Arm Summary

5-seed means:

| Arm | Acc | Prob | Best tick prob | Final-best | Wrong | Power |
|---|---:|---:|---:|---:|---:|---:|
| FIXED_PHASE_LANE_REFERENCE | 1.000 | 0.970 | 0.970 | 0.000 | 0.000 | 1.000 |
| FULL_16_RULE_TEMPLATE | 0.309 | 0.332 | 0.368 | -0.035 | 0.264 | 0.677 |
| COMMON_CORE_15_PLUS_MISSING_1_2_3 | 0.309 | 0.332 | 0.368 | -0.035 | 0.264 | 0.677 |
| DENSE_009_REFERENCE | 0.309 | 0.332 | 0.368 | -0.035 | 0.264 | 0.677 |
| RANDOM_MATCHED_16_MOTIF_CONTROL | 0.230 | 0.252 | 0.272 | -0.021 | 0.286 | 0.647 |
| CANONICAL_JACKPOT_007_BASELINE | 0.255 | 0.277 | 0.389 | -0.112 | 0.254 | 0.062 |
| SOURCE_PERSIST_1_TICK | 0.308 | 0.329 | 0.365 | -0.036 | 0.266 | 0.724 |
| SOURCE_PERSIST_2_TICKS | 0.267 | 0.287 | 0.352 | -0.065 | 0.282 | 1.372 |
| SOURCE_PERSIST_ALL_TICKS | 0.242 | 0.264 | 0.348 | -0.085 | 0.252 | 2.615 |

## FULL16 Horizon Table

Each cell is `accuracy/probability/target_power`.

| Path length | t4 | t8 | t12 | t16 | t24 | t32 | t48 | t64 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.27/0.25/0.00 | 0.27/0.25/0.00 | 1.00/1.00/1.00 | 1.00/1.00/4.00 | 1.00/1.00/4.00 | 1.00/1.00/4.00 | 1.00/1.00/4.00 | 1.00/1.00/4.00 |
| 4 | 0.24/0.25/0.00 | 0.24/0.25/0.00 | 0.24/0.25/0.00 | 0.24/0.25/0.00 | 0.24/0.25/0.00 | 0.24/0.25/0.00 | 0.24/0.25/0.00 | 0.24/0.25/0.00 |
| 8 | 0.25/0.25/0.00 | 0.25/0.25/0.00 | 0.25/0.25/0.00 | 0.25/0.25/0.00 | 0.25/0.25/0.00 | 0.25/0.25/0.00 | 0.25/0.25/0.00 | 0.25/0.25/0.00 |
| 16 | 0.24/0.25/0.00 | 0.24/0.25/0.00 | 0.33/0.31/0.33 | 0.33/0.31/1.33 | 0.33/0.31/1.33 | 0.33/0.31/1.33 | 0.33/0.31/1.33 | 0.33/0.31/1.33 |
| 24 | 0.26/0.25/0.00 | 0.26/0.25/0.00 | 0.26/0.28/0.33 | 0.26/0.28/1.33 | 0.26/0.28/1.33 | 0.26/0.28/1.33 | 0.26/0.28/1.33 | 0.26/0.28/1.33 |
| 32 | 0.26/0.25/0.00 | 0.26/0.25/0.00 | 0.29/0.28/0.33 | 0.29/0.28/1.33 | 0.29/0.28/1.33 | 0.29/0.28/1.33 | 0.29/0.28/1.33 | 0.33/0.33/2.67 |
| 48 | 0.09/0.25/0.00 | 0.09/0.25/0.00 | 0.09/0.25/0.00 | 0.09/0.25/0.00 | 0.09/0.25/0.00 | 0.09/0.25/0.00 | 0.09/0.25/0.00 | 0.09/0.25/0.00 |

Minimum tick summary:

```text
path_length=2:
  minimum_ticks_for_95_accuracy = 12
  minimum_ticks_for_90_probability = 12

path_length in {4,8,12,16,24,32,48}:
  no tick bucket reached 95% accuracy or 90% correct probability
```

Long-path aggregate:

```text
FULL_16_RULE_TEMPLATE long-path accuracy = 0.210
COMMON_CORE_15_PLUS_MISSING_1_2_3 long-path accuracy = 0.210
DENSE_009_REFERENCE long-path accuracy = 0.210
```

## Diagnosis

The important negative is not motif failure:

```text
FULL_16_RULE_TEMPLATE == COMMON_CORE_15_PLUS_MISSING_1_2_3 == DENSE_009_REFERENCE
```

That supports:

```text
SPARSE_EQUALS_FULL16_HORIZON
RULE_TEMPLATE_STABLE_BUT_SETTLING_LIMITED
```

The failure is also not rescued by simply keeping the source on:

```text
SOURCE_PERSIST_ALL_TICKS:
  accuracy = 0.242
  probability = 0.264
  target_power_total = 2.615
```

This increases power but mostly adds unresolved/incorrect phase mass. The next
blocker is therefore substrate timing/readout/phase persistence, not local rule
completion.

## Audits

Main phase-rule arms:

```text
forbidden_private_field_leak = 0
nonlocal_edge_count = 0
direct_output_leak_rate = 0
wall_leak_rate = 0
```

The only observed wall-leak maxima came from the weak
`CANONICAL_JACKPOT_007_BASELINE` diagnostic arm; the completed rule, dense
reference, fixed reference, and persistence arms stayed at zero wall leak.

## Claim Boundary

012 supports:

```text
the completed local rule matches FULL16
the dense 009 reference has no horizon advantage in this runner
the current recurrent phase-lane substrate has a measured horizon/timing limit
```

012 does not support:

```text
production architecture
full VRAXION
consciousness
language grounding
Prismion uniqueness
physical quantum behavior
```
