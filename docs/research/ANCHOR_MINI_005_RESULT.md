# ANCHOR-MINI-005 Result

## Verdict

```text
ANCHOR_MINI_005_INTERNALIZATION_STRONG_POSITIVE
```

ANCHOR-MINI-005 tests whether the sparse carrier can learn the process route
from ordinary situation features, rather than using visible oracle `match_bits`
at eval.

Narrow result:

```text
AnchorCell-style process supervision was internalized in the audit sparse
carrier on the shortcut-flip toy task.
```

This does not prove natural-language AnchorCells, full INSTNCT recurrent
behavior, Qwen behavior, or symbol grounding at scale.

## Run

```bash
cargo build --release -p instnct-core --example evolve_anchor_mini005

python tools/anchorweave/run_anchor_mini005_internalization.py ^
  --out target/anchorweave/anchor_mini005_internalization/full_2026_05_10 ^
  --seeds 2026-2125 ^
  --jobs 16 ^
  --budget-hours 8 ^
  --skip-build
```

## Summary

```text
completed_jobs: 500 / 500
valid_jobs: 500
valid_seed_count: 100
blocked_jobs: 0
budget_reached: false
```

| carrier | jobs | valid | OOD accuracy | shortcut trap rate | true process bit accuracy |
|---|---:|---:|---:|---:|---:|
| `SPARSE_DIRECT` | 100 | 100 | 0.100 | 1.000 | 0.000 |
| `SPARSE_ORACLE_ROUTED` | 100 | 100 | 1.000 | 0.000 | 1.000 |
| `SPARSE_LEARNED_PROCESS` | 100 | 100 | 1.000 | 0.000 | 1.000 |
| `SPARSE_LEARNED_HYBRID` | 100 | 100 | 1.000 | 0.000 | 1.000 |
| `SPARSE_SHUFFLED_PROCESS` | 100 | 100 | 0.331 | 0.222 | 0.572 |

## Conditions

All primary gates passed:

```text
valid_seeds_at_least_80: true
oracle_upper_bound_good: true
learned_beats_direct_by_0p25: true
learned_beats_shuffled_by_0p25: true
learned_trap_rate_le_0p25: true
learned_process_true_accuracy_high: true
hybrid_directionally_positive: true
shuffled_not_reproducing: true
no_blocked_jobs: true
```

Gaps:

```text
internalization_gap: +0.900
shuffled_control_gap: +0.669
oracle_gap: 0.000
```

## Interpretation

`SPARSE_DIRECT` learned the train-time surface shortcut and failed OOD.
`SPARSE_ORACLE_ROUTED` is the upper bound with visible process bits.

`SPARSE_LEARNED_PROCESS` matched the oracle upper bound while receiving only
raw goal/effect category indicators and surface priors at eval. It did not read
precomputed `match_bits` as inputs. This is the key internalization signal.

`SPARSE_LEARNED_HYBRID` also remained positive despite a direct surface bypass,
which suggests the learned process route can dominate the shortcut path in this
toy carrier.

`SPARSE_SHUFFLED_PROCESS` learned a semantically wrong process target and did
not reproduce the improvement. This guards against the result being caused only
by extra supervision volume or routed form.

## Claim Boundary

This is still a toy carrier result. It supports the narrow engineering claim:

```text
Process supervision can be learned from ordinary inputs and used at eval
without visible oracle process bits, if the carrier routes decisions through
the learned process state.
```

Next required step: test the same masking principle on a closer VRAXION/INSTNCT
carrier or a small causal-LM carrier while preserving the same controls.
