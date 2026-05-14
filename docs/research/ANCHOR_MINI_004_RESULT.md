# ANCHOR-MINI-004 Result

## Verdict

```text
ANCHOR_MINI_004_VRAXION_SPARSE_STRONG_POSITIVE
```

ANCHOR-MINI-004 transfers the MINI-003 shortcut-flip stress test to an
audit-sized sparse mutation-selection carrier under `instnct-core/examples`.

The result supports the narrow carrier claim:

```text
Decomposed AnchorCell-style process signal survives in a sparse
mutation-selection carrier when the final decision is routed through process
state.
```

It does not prove natural-language AnchorCells, Qwen behavior, full INSTNCT
recurrent behavior, or symbol grounding at scale.

## Run

```bash
cargo build --release -p instnct-core --example evolve_anchor_mini003

python tools/anchorweave/run_anchor_mini003_vraxion_sparse_sweep.py ^
  --out target/anchorweave/anchor_mini004_vraxion_sparse/full_2026_05_10_metricfix ^
  --seeds 2026-2125 ^
  --jobs 16 ^
  --budget-hours 8 ^
  --skip-build
```

Hardware note: the sweep used parallel independent jobs. During the run,
Windows `_Total` CPU was sampled at about `95%`, confirming that the run was
not sequentially underutilized.

## Summary

```text
completed_jobs: 500 / 500
valid_jobs: 500
valid_seed_count: 100
budget_reached: false
blocked_jobs: 0
```

| carrier | jobs | valid | OOD accuracy | shortcut trap rate | process bit accuracy |
|---|---:|---:|---:|---:|---:|
| `SPARSE_DIRECT` | 100 | 100 | 0.099 | 1.000 | 0.000 |
| `SPARSE_AUX_DIRECT` | 100 | 100 | 0.250 | 0.252 | 1.000 |
| `SPARSE_ROUTED` | 100 | 100 | 1.000 | 0.000 | 1.000 |
| `SPARSE_HYBRID` | 100 | 100 | 1.000 | 0.000 | 1.000 |
| `SPARSE_SHUFFLED_ROUTED` | 100 | 100 | 0.250 | 0.249 | 0.995 |

## Conditions

All primary gates passed:

```text
valid_seeds_at_least_50: true
routed_beats_direct_by_0p25: true
routed_beats_shuffled_by_0p25: true
routed_trap_rate_le_0p25: true
hybrid_directionally_positive: true
aux_direct_not_equal_routed: true
no_blocked_jobs: true
```

Determinism check:

```text
same frozen dataset + same seed + same carrier produced identical report.json
SHA256 hash: 65564142C7FDB4F8E9DC44AF80CA59C83A8825DF7F1CD18A3CFF968801D6FDCC
```

## Interpretation

`SPARSE_DIRECT` learned the surface shortcut and failed OOD. `SPARSE_AUX_DIRECT`
learned process bits, but its final answer was not routed through them, so it
did not recover OOD performance. `SPARSE_ROUTED` and `SPARSE_HYBRID` solved the
OOD shortcut flip because the final decision used the process/match state.

The negative shuffled routed control shows that routed structure alone is not
enough; the process signal must be aligned with the task.

## Claim Boundary

This is a toy carrier result. It validates a narrow engineering hypothesis:

```text
AnchorCell process supervision needs a decision carrier that uses the process
state. Merely adding auxiliary process labels to an unconstrained direct answer
path is not enough.
```

Next required step: replace the audit-sized sparse feature carrier with a
closer INSTNCT recurrent/grower carrier while preserving the same dataset,
controls, and shortcut-flip gates.
