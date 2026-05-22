# STABLE_LOOP_PHASE_LOCK_010_COINCIDENCE_CRYSTALLIZE_PRUNE Result

Status: implemented, static validation complete, sanity complete, 3-seed smoke complete.

## Verdict

```text
DENSE_SOLUTION_REPRODUCED
SNAP_REVEALS_CANONICAL_MOTIF
CRYSTALLIZE_PRESERVES_BEHAVIOR
PRUNE_FINDS_SPARSE_CAUSAL_CORE
DENSE_OVERBUILT_BUT_PRUNABLE
CAUSAL_MOTIF_CORE_IDENTIFIED
SPARSE_CORE_STABLE_ACROSS_SEEDS
EXPERIMENTAL_MUTATION_LANE_SUPPORTED
PRODUCTION_API_NOT_READY
```

Interpretation:

```text
the dense coincidence forest is heavily overbuilt
snap/crystallize preserves behavior
greedy prune exposes a much smaller causal phase-transport core
the retained motif types are stable across seeds
```

## Grounding

009 found that the audited local coincidence operator can rescue spatial phase
transport, but the result was reported as dense / motif-rich rather than
efficient.

010 dissects that dense motif forest with:

```text
DENSE_RAW -> SNAP_ONLY -> SNAP_THEN_PRUNE
```

using separate train, guard, and final holdout case splits.

## Runs

Static:

```powershell
cargo check -p instnct-core --example phase_lane_coincidence_crystallize_prune
cargo test -p instnct-core jackpot_traced_emits_candidate_rows_and_accept_invariants
git diff --check
```

All passed.

Sanity:

```powershell
cargo run -p instnct-core --example phase_lane_coincidence_crystallize_prune --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_010_coincidence_crystallize_prune/sanity ^
  --seeds 2026 ^
  --steps 100 ^
  --eval-examples 256 ^
  --width 6 ^
  --ticks 8 ^
  --jackpot 6 ^
  --heartbeat-sec 15
```

Result: dense 576 motifs pruned to 13 motifs, with train/guard/holdout accuracy
at 100.0%.

Smoke:

```powershell
cargo run -p instnct-core --example phase_lane_coincidence_crystallize_prune --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_010_coincidence_crystallize_prune/smoke ^
  --seeds 2026,2027,2028 ^
  --steps 400 ^
  --eval-examples 512 ^
  --width 8 ^
  --ticks 12 ^
  --jackpot 9 ^
  --heartbeat-sec 30
```

Result: 3/3 jobs completed in 197.1 seconds after release build.

Produced:

```text
3072 prune_curve rows
3072 single-motif ablation rows
75 retained motif rows
```

## Smoke Summary

| Metric | Mean | Min | Max |
|---|---:|---:|---:|
| train_accuracy | 0.962 | 0.945 | 0.973 |
| guard_accuracy | 0.987 | 0.984 | 0.992 |
| final_holdout_accuracy | 0.969 | 0.961 | 0.977 |
| motif_count_initial | 1024.0 | 1024 | 1024 |
| motif_count_final | 25.0 | 24 | 26 |
| motif_prune_fraction | 0.976 | 0.975 | 0.977 |
| guard_counterfactual_accuracy | 1.000 | 1.000 | 1.000 |
| gate_shuffle_collapse | 0.932 | 0.891 | 0.992 |
| single_gate_ablation_drop_mean | 0.001 | 0.001 | 0.001 |
| single_gate_ablation_drop_max | 0.080 | 0.076 | 0.082 |
| cell_ablation_drop_mean | 0.018 | 0.018 | 0.018 |
| cell_ablation_drop_max | 0.750 | 0.750 | 0.750 |

Per seed:

| Seed | Initial motifs | Final motifs | Pruned | Train | Guard | Holdout | Counterfactual | Gate collapse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 1024 | 26 | 97.5% | 96.9% | 99.2% | 96.9% | 100.0% | 99.2% |
| 2027 | 1024 | 24 | 97.7% | 94.5% | 98.4% | 96.1% | 100.0% | 89.1% |
| 2028 | 1024 | 25 | 97.6% | 97.3% | 98.4% | 97.7% | 100.0% | 91.4% |

## Sparse Core Stability

Retained motif-type overlap:

```text
seed_overlap_jaccard = 0.9375
```

Common retained motif types:

```text
0_0_0
0_1_1
0_2_2
0_3_3
1_0_1
1_1_2
1_3_0
2_0_2
2_1_3
2_2_0
2_3_1
3_0_3
3_1_0
3_2_1
3_3_2
```

This supports:

```text
SPARSE_CORE_STABLE_ACROSS_SEEDS
```

## Audits

```text
forbidden_private_field_leak = 0
nonlocal_edge_count = 0
direct_output_leak_rate = 0
guard counterfactual stays >= 1.000
final holdout remains >= 0.961
gate shuffle still collapses
```

The result does not depend on raw 009 target outputs. Dense forests are
regenerated deterministically inside the 010 runner.

## Claim Boundary

This supports:

```text
the dense coincidence forest contains a sparse causal core
crystallize/snap preserves the canonical coincidence mechanism
most dense motifs are decorative for this toy phase-lane setup
the coincidence mutation lane is experimentally useful but overcomplete
```

This does not support:

```text
full VRAXION validity
consciousness
language grounding
production architecture
Prismion uniqueness
```
