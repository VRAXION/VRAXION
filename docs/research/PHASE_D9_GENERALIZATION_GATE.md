# Phase D9 Generalization Gate

Verdict: `D9_SMOOTH_ONLY_WIN`

Candidate: `seed2042_improved_v1`

## Result

The candidate is robustly better on the D9 smooth metric and also improves direct accuracy, but it does not improve all available evaluation views.

| metric | n | mean_delta | lower95 | upper95 | positive_rate |
|---|---:|---:|---:|---:|---:|
| smooth | 60 | +0.017327386 | +0.017139222 | +0.017515551 | 100.0% |
| accuracy | 60 | +0.004552083 | +0.004051453 | +0.005052714 | 95.0% |
| echo | 60 | -0.000085417 | -0.000261328 | +0.000090495 | 1.7% |
| unigram | 60 | -0.008823296 | -0.009311472 | -0.008335120 | 0.0% |

## Interpretation

`seed2042_improved_v1` should not be described as a globally better checkpoint yet.

The correct claim is narrower:

```text
seed2042_improved_v1 is a robust H=384 smooth-metric improvement
that also improves paired accuracy, but regresses unigram behavior.
```

This makes it a valuable specialist candidate and a useful search proof, not a finished general-purpose production checkpoint.

## Decision

- Keep `seed2042_improved_v1` as the current best smooth/accuracy candidate.
- Do not promote it as a broad generalization checkpoint.
- Next gate should decide whether to preserve the smooth/accuracy gain while repairing the unigram regression, or whether unigram is an undesirable auxiliary metric for this phase.

## Artifacts

- Full D9.0z report:
  `output/phase_d9_0z_generalization_gate_20260429/D9_0Z_GENERALIZATION_GATE_REPORT.md`
- Summary JSON:
  `output/phase_d9_0z_generalization_gate_20260429/d9_0z_generalization_summary.json`
- Candidate checkpoint:
  `output/phase_d9_0y_seed2042_improved_v1_candidate_20260429/seed2042_improved_v1.ckpt`
