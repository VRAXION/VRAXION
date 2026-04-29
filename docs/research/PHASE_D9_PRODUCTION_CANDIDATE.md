# Phase D9 Production Candidate

Verdict: `D9_PRODUCTION_CANDIDATE_CONFIRMED`

## Candidate

- Name: `seed2042_improved_v1`
- H: `384`
- Source tile: `11_16`
- Source endpoint: `D9.0r endpoint_01`
- Candidate checkpoint:
  `output/phase_d9_0y_seed2042_improved_v1_candidate_20260429/seed2042_improved_v1.ckpt`
- Baseline checkpoint:
  `output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt`

## Result

The candidate remains positive against the paired seed2042 baseline on fresh evaluation seeds independent of D9.0x.

| eval_len | n | mean_delta | lower95 | upper95 | min | max | positive_rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 30 | 0.017467888 | 0.017112529 | 0.017823248 | 0.015855210 | 0.019678299 | 100.0% |
| 16000 | 30 | 0.017186885 | 0.017074839 | 0.017298931 | 0.016745515 | 0.018096462 | 100.0% |

Overall:

```text
n = 60
mean_delta = +0.017327386
lower95 = +0.017139222
positive_rate = 100.0%
```

## Interpretation

`seed2042_improved_v1` is the current best H=384 smooth-metric production candidate. It is no longer just a landscape peak: it is an exported checkpoint that re-evaluates robustly above the seed2042 baseline.

The surrounding terrain remains narrow rather than plateau-like. Prior D9.0w overlap testing found three independent local islands, not a connected ridge or broad basin. This candidate is the highest and most stable of those islands.

## Caveat

This verdict is scoped to the smooth metric. A separate multi-task/generalization gate is required before broader deployment claims.

## Artifacts

- Full D9.0y report:
  `output/phase_d9_0y_seed2042_improved_v1_candidate_20260429/D9_0Y_PRODUCTION_CANDIDATE_REPORT.md`
- Manifest:
  `output/phase_d9_0y_seed2042_improved_v1_candidate_20260429/candidate_manifest.json`
- Fresh eval CSV:
  `output/phase_d9_0y_seed2042_improved_v1_candidate_20260429/fresh_eval/robustness_samples.csv`
- Prior D9.0x robustness report:
  `output/phase_d9_0x_endpoint_robustness_20260429/D9_0X_ENDPOINT_ROBUSTNESS_REPORT.md`
