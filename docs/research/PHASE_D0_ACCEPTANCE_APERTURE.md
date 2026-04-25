# Phase D0 Acceptance Aperture

## Verdict

- Status: `PASS`
- Candidate rows: `12,600,000` across `30` runs.
- Epsilon informative under current best-of-K selector: `False`.
- Recommended D1 mode: `zero-p-only`.
- Reason: Current best-of-K selector is zero-dominated; epsilon would usually select the same zero-delta moves as ties.

## Arm Summary

| horizon_steps | accept_ties | arm | n | best_negative_rate_mean | best_negative_rate_std | best_exact_zero_rate_mean | best_exact_zero_rate_std | best_near_zero_1e12_rate_mean | best_near_zero_1e12_rate_std | best_positive_rate_mean | best_positive_rate_std | positive_rate_mean | positive_rate_std | negative_rate_mean | negative_rate_std | exact_zero_rate_mean | exact_zero_rate_std | near_zero_1e12_rate_mean | near_zero_1e12_rate_std | gaussian_best_ks_stat_mean | gaussian_best_ks_stat_std | gaussian_delta_ks_stat_mean | gaussian_delta_ks_stat_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20000 | False | B1_S20_STRICT | 5 | 0 | 0 | 0.82804 | 0.0681041 | 0.83179 | 0.068177 | 0.17196 | 0.0681041 | 0.0244467 | 0.0115105 | 0.112674 | 0.0137744 | 0.862879 | 0.0236751 | 0.864479 | 0.0235807 | 0.454454 | 0.0218252 | 0.491951 | 0.010612 |
| 20000 | True | B1_S20_TIES | 5 | 0.00042 | 8.3666e-05 | 0.91547 | 0.0242621 | 0.91727 | 0.023495 | 0.08401 | 0.0243197 | 0.0123267 | 0.00385652 | 0.0902189 | 0.0110501 | 0.897454 | 0.0122862 | 0.898488 | 0.0118244 | 0.478921 | 0.00485477 | 0.505777 | 0.00510129 |
| 40000 | False | B1_S40_STRICT | 5 | 0.00017 | 0.000207214 | 0.82128 | 0.027536 | 0.8257 | 0.026749 | 0.178535 | 0.0273714 | 0.0253028 | 0.00493202 | 0.133776 | 0.0108429 | 0.840921 | 0.0145788 | 0.842884 | 0.0143131 | 0.443987 | 0.0115366 | 0.491138 | 0.0100409 |
| 40000 | True | B1_S40_TIES | 5 | 0.00068 | 0.000240052 | 0.873955 | 0.0270162 | 0.87706 | 0.0271998 | 0.125265 | 0.0272096 | 0.0200889 | 0.00502759 | 0.114912 | 0.00832117 | 0.864999 | 0.00556474 | 0.866737 | 0.00510735 | 0.464969 | 0.00427552 | 0.504027 | 0.00454856 |
| 80000 | False | B1_S80_STRICT | 5 | 0.0007 | 0.000227932 | 0.828013 | 0.0287932 | 0.832095 | 0.0284806 | 0.171222 | 0.0286298 | 0.026585 | 0.00573566 | 0.142176 | 0.0077435 | 0.831239 | 0.0110596 | 0.83335 | 0.0108929 | 0.444304 | 0.0120989 | 0.491138 | 0.0100409 |
| 80000 | True | B1_S80_TIES | 5 | 0.0008275 | 0.000119373 | 0.843862 | 0.0238572 | 0.84728 | 0.024478 | 0.155183 | 0.0237564 | 0.0256692 | 0.00490961 | 0.123427 | 0.00338085 | 0.850904 | 0.00761417 | 0.852692 | 0.00796885 | 0.452872 | 0.00949574 | 0.504027 | 0.00454856 |

## Operator Summary

| operator_id | n | positive_rate_mean | positive_rate_std | negative_rate_mean | negative_rate_std | exact_zero_rate_mean | exact_zero_rate_std | near_zero_1e12_rate_mean | near_zero_1e12_rate_std | gaussian_delta_ks_stat_mean | gaussian_delta_ks_stat_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| projection_weight | 30 | 0.00014369 | 0.000106528 | 0.00104755 | 0.000448897 | 0.998809 | 0.000469441 | 0.998828 | 0.000462631 | 0.507509 | 0.00238452 |
| enhance | 30 | 0.00751096 | 0.00465165 | 0.0343058 | 0.0182644 | 0.958183 | 0.0224482 | 0.95903 | 0.0221062 | 0.506844 | 0.00502293 |
| mirror | 30 | 0.0130727 | 0.00512333 | 0.0679852 | 0.0132621 | 0.918942 | 0.0176022 | 0.920448 | 0.017149 | 0.503666 | 0.00594536 |
| remove_edge | 30 | 0.0172256 | 0.00686557 | 0.0667159 | 0.0157613 | 0.916059 | 0.0213221 | 0.917787 | 0.0208614 | 0.502014 | 0.00624123 |
| add_edge | 30 | 0.013993 | 0.00540532 | 0.0708669 | 0.0128569 | 0.91514 | 0.0171466 | 0.916693 | 0.0167798 | 0.50332 | 0.00625933 |
| reverse | 30 | 0.025649 | 0.00952562 | 0.116214 | 0.0252529 | 0.858137 | 0.0329217 | 0.860595 | 0.0323555 | 0.490947 | 0.0119761 |
| rewire | 30 | 0.0268193 | 0.00995131 | 0.123243 | 0.0239935 | 0.849938 | 0.0320651 | 0.852474 | 0.0314883 | 0.48898 | 0.0118946 |
| loop2 | 30 | 0.0237068 | 0.00902121 | 0.128468 | 0.0217638 | 0.847825 | 0.0289388 | 0.850242 | 0.028305 | 0.489948 | 0.0116455 |
| loop3 | 30 | 0.0308771 | 0.0118259 | 0.168891 | 0.0264108 | 0.800232 | 0.0356732 | 0.803187 | 0.0350257 | 0.480084 | 0.0164382 |
| channel | 30 | 0.0413964 | 0.0134505 | 0.282522 | 0.0475662 | 0.676082 | 0.0549235 | 0.677085 | 0.0548321 | 0.44672 | 0.0284726 |
| theta | 30 | 0.0632202 | 0.0157534 | 0.389093 | 0.0472895 | 0.547687 | 0.0535068 | 0.549121 | 0.0535684 | 0.410467 | 0.040977 |

## Interpretation

- `C_K` remains the empirical progress metric; `A_pi(epsilon)` is only a Gaussian/isotropic null model.
- If `best_negative_rate` is near zero, nonzero epsilon does not open new selected moves because zero-delta candidates dominate the best-of-K aperture.
- In that case D1 should test probabilistic Zero-Drive (`neutral_p`) before any larger epsilon sweep.

## Artifacts

- Root: `output\phase_b1_horizon_ties_20260425`
- Summary JSON: `output\phase_b1_horizon_ties_20260425\analysis\acceptance_aperture_summary.json`
- Figures: `output\phase_b1_horizon_ties_20260425\analysis\figures`
