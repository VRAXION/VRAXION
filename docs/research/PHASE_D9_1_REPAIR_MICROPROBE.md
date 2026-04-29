# Phase D9.1a Repair Microprobe

Verdict: `D9_1_WEAK_REPAIR_CANDIDATE_FOUND`

## Result

D9.1a tested whether local `edge`/`threshold` mutations around `seed2042_improved_v1` can preserve the D9 smooth/accuracy gain while reducing the unigram regression.

Reduced run:

```text
radii = 4,8,16
mutation_types = edge,threshold
samples_per_bucket = 10
proposals = 60
eval_len = 4000
repair_eval_seeds = 940001..940008
```

Class counts:

| class | count |
|---|---:|
| `WEAK_REPAIR` | 12 |
| `NO_REPAIR` | 1 |
| `FAIL_RETAIN` | 47 |

Best observed candidate:

| proposal | radius | type | class | smooth | accuracy | echo | unigram |
|---:|---:|---|---|---:|---:|---:|---:|
| 1 | 4 | edge | `WEAK_REPAIR` | +0.016783217 | +0.004843750 | 0.000000000 | -0.008260758 |

## Interpretation

The local repair signal exists, but it is weak.

Compared with the D9.0z unigram regression of `-0.008823296`, the best D9.1a candidate improved unigram to `-0.008260758` while retaining smooth and accuracy. This is a small repair, not a full fix.

No candidate reached:

```text
STRONG_REPAIR: unigram >= -0.0044
FULL_REPAIR:   unigram >=  0.0000
```

## Decision

- Keep the D9.1a candidates as evidence of a weak local repair direction.
- Do not claim the unigram regression is solved.
- Next step is D9.1b confirm on the top exported candidates, but it should be scheduled as a separate long run because full confirm is too expensive for the D9.1a implementation turn.

## Artifacts

- Report:
  `output/phase_d9_1a_repair_microprobe_20260429/D9_1A_REPAIR_MICROPROBE_REPORT.md`
- Samples:
  `output/phase_d9_1a_repair_microprobe_20260429/repair_samples.csv`
- Exported candidates:
  `output/phase_d9_1a_repair_microprobe_20260429/candidates/`
