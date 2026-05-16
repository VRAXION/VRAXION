# STABLE_LOOP_PHASE_LOCK_009_COINCIDENCE_OPERATOR_SPATIAL_REINSERTION Result

Status: implemented, static validation complete, sanity complete, 3-seed smoke complete.

## Verdict

```text
HAND_BUILT_SPATIAL_MOTIF_WORKS
CANONICAL_JACKPOT_STILL_INSUFFICIENT
COINCIDENCE_OPERATOR_RESCUES_SPATIAL_PHASE_DENSE
```

Interpretation:

```text
the 008 local motif works when hand-inserted spatially
canonical 007-style mutation remains far below the solved references
the audited local coincidence operator makes the spatial task reachable
the successful arm is dense / motif-rich, not evidence of a minimal efficient circuit
```

## Grounding

007 showed the spatial phase-lane task is valid but canonical mutation did not
grow phase transport.

008 isolated the missing local construct:

```text
phase_i + gate_g -> phase_(i+g)
```

and showed that it is representable, but not reliably grown by canonical
mutation alone. 009 reinserts the audited local `add_coincidence_gate` operator
into the spatial substrate.

## Runs

Static:

```powershell
cargo check -p instnct-core --example phase_lane_spatial_reinsertion
cargo test -p instnct-core jackpot_traced_emits_candidate_rows_and_accept_invariants
git diff --check
```

All passed.

Sanity:

```powershell
cargo run -p instnct-core --example phase_lane_spatial_reinsertion --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_009_coincidence_operator_spatial_reinsertion/sanity ^
  --seeds 2026 ^
  --steps 100 ^
  --eval-examples 256 ^
  --width 6 ^
  --ticks 8 ^
  --jackpot 6 ^
  --heartbeat-sec 15
```

Result: 13/13 jobs, 6,600 candidate rows, hand-built spatial motif passed.

Smoke:

```powershell
cargo run -p instnct-core --example phase_lane_spatial_reinsertion --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_009_coincidence_operator_spatial_reinsertion/smoke_fast ^
  --seeds 2026,2027,2028 ^
  --steps 400 ^
  --eval-examples 512 ^
  --width 8 ^
  --ticks 12 ^
  --jackpot 9 ^
  --heartbeat-sec 30
```

Result: 39/39 jobs, 118,800 candidate rows, 108,000 motif placement audit rows,
completed in 229.4 seconds after release build.

## Smoke Summary

Mean over seeds 2026, 2027, 2028:

| Arm | Accuracy | Correct prob | Counterfactual | Gate collapse | Motif drop | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| HAND_BUILT_SPATIAL_COINCIDENCE_REFERENCE | 1.000 | 1.000 | 1.000 | 1.000 | 0.761 | 1.000 | 1.000 |
| CANONICAL_JACKPOT_007_BASELINE | 0.327 | 0.317 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| FULL_SPATIAL_PLUS_COINCIDENCE_OPERATOR | 0.992 | 0.994 | 0.990 | 0.992 | 0.753 | 1.000 | 0.990 |
| COINCIDENCE_OPERATOR_STRICT | 0.992 | 0.994 | 0.990 | 0.992 | 0.753 | 1.000 | 0.990 |
| COINCIDENCE_OPERATOR_TIES | 0.961 | 0.941 | 0.917 | 0.943 | 0.722 | 0.487 | 1.000 |
| COINCIDENCE_OPERATOR_ZEROP | 0.979 | 0.974 | 0.990 | 0.961 | 0.740 | 0.580 | 1.000 |
| POLARITY_ONLY | 0.370 | 0.382 | 0.094 | 0.173 | 0.131 | 0.250 | 0.250 |
| CHANNEL_ONLY | 0.404 | 0.416 | 0.141 | 0.177 | 0.165 | 0.250 | 0.250 |

The strict coincidence arm and full-spatial coincidence arm both solve the
public-corridor spatial task across all three seeds. Canonical jackpot remains
near the unsolved 007 behavior.

The `*_TIES` and `*_ZEROP` arms also solve, but with lower motif precision,
which is why the runner reports the dense verdict rather than an efficient
minimal-motif claim.

## Audits

```text
forbidden_private_field_leak = 0
nonlocal_edge_count = 0 on successful arms
direct_output_leak_rate = 0 on successful arms
gate shuffle collapses solved-arm accuracy
same-target counterfactuals pass on solved arms
motif ablation drops solved-arm performance by ~0.75
```

The full spatial operator samples only public free cells from the wall/free mask.
The oracle-routing diagnostic samples private path cells and is reported only as
a placement diagnostic, not as the main spatial claim.

## Runtime Note

An initial smoke attempt recomputed full audit metrics inside candidate fitness
and ran too slowly. The final runner uses a lightweight candidate fitness based
on `correct_target_lane_probability_mean`; expensive locality, wall, motif, and
ablation audits are kept for checkpoints and final rows. This preserves the
no-black-box rule while keeping the smoke reproducible in minutes.

## Claim Boundary

This supports:

```text
the audited local coincidence mutation lane makes spatial phase construction reachable
the 007 blocker was consistent with missing local coincidence motif constructability
canonical jackpot alone remains insufficient in this setup
```

This does not support:

```text
efficient/minimal phase circuit discovery
production-ready sidepocket API
full VRAXION validity
consciousness
language grounding
Prismion uniqueness
```
