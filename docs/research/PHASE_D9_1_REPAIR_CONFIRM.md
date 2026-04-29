# Phase D9.1b Repair Confirm

Verdict: `D9_1_WEAK_REPAIR_CONFIRMED`

## Result

D9.1b confirmed the best representative D9.1a repair candidates on fresh seeds and longer eval lengths.

Scope:

```text
candidates = top_01, top_07, top_08
metrics = smooth, accuracy, echo, unigram
eval_lens = 4000,16000
eval_seeds = 950001..950030
```

Candidate summary:

| endpoint | verdict | smooth | accuracy | echo | unigram | unigram lower95 |
|---|---|---:|---:|---:|---:|---:|
| top_01 | `D9_1_WEAK_REPAIR_CONFIRMED` | +0.017157385 | +0.004551042 | -0.000079167 | -0.008735006 | -0.009210271 |
| top_07 | `D9_1_WEAK_REPAIR_CONFIRMED` | +0.017149898 | +0.004551042 | -0.000079167 | -0.008764920 | -0.009238769 |
| top_08 | `D9_1_WEAK_REPAIR_CONFIRMED` | +0.017149981 | +0.004551042 | -0.000079167 | -0.008761113 | -0.009235027 |

## Interpretation

The weak repair direction is real but very small.

Compared with the D9.0z unigram regression:

```text
D9.0z seed2042_improved_v1 unigram = -0.008823296
D9.1b best repair candidate unigram = -0.008735006
```

This is a stable improvement, but it repairs only a small fraction of the unigram regression. It does not meet:

```text
PARTIAL_REPAIR: unigram >= -0.0044
FULL_REPAIR:    unigram lower95 >= 0.0
```

## Decision

- Keep `seed2042_improved_v1` as the main smooth/accuracy specialist.
- Keep the D9.1 candidates as proof that local unigram repair is possible but weak.
- Do not spend more short-range local search budget on the same repair objective without changing the objective or mutation policy.

## Next Options

1. Treat the checkpoint as a specialist and proceed to beta.7 documentation.
2. Design D9.2 as a true multi-objective search, where unigram is part of acceptance from the beginning.
3. Try a broader mutation policy only if there is a clear reason to revisit channel/projection mutations.

## Artifacts

- Full report:
  `output/phase_d9_1b_repair_confirm_20260429/D9_1B_REPAIR_CONFIRM_REPORT.md`
- Summary:
  `output/phase_d9_1b_repair_confirm_20260429/d9_1b_repair_confirm_summary.json`
