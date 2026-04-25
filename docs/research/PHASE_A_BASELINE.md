# Phase A Baseline Audit

## Verdict

Phase A is a valid 30-cell baseline artifact: two fixtures, three H values, five seeds each.
Mutual inhibition shows the inverted-U profile (`H=256` peak mean `5.28%`, `H=384` `3.52%`).
Bytepair projection does not show the same profile; it declines with H and becomes high-variance at `H=384` (peak mean `3.16%`, std `2.33pp`).
Phase B B0 replication matches Phase A mutual_inhibition H=384 on all checked metrics.

## Artifact Integrity

- Rows: `30`
- Fixtures: `bytepair_proj, mutual_inhibition`
- H values: `[128, 256, 384]`
- Seeds: `[42, 1042, 2042, 3042, 4042]`
- Driver log: `output\dimensionality_sweep\20260424_091217\driver.log`

## Fixture x H Summary

| fixture | H | n | peak_acc_mean | peak_acc_std | final_acc_mean | final_acc_std | accept_rate_pct_mean | accept_rate_pct_std | alive_frac_mean_mean | alive_frac_mean_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bytepair_proj | 128 | 5 | 5.2400 | 1.0738 | 4.1400 | 1.3520 | 25.9030 | 6.2759 | 0.6620 | 0.0894 |
| bytepair_proj | 256 | 5 | 3.6200 | 0.8106 | 2.3600 | 0.8503 | 7.6890 | 3.1161 | 0.1894 | 0.0308 |
| bytepair_proj | 384 | 5 | 3.1600 | 2.3287 | 2.3400 | 1.9920 | 0.6130 | 0.3901 | 0.0495 | 0.0305 |
| mutual_inhibition | 128 | 5 | 3.7600 | 0.9127 | 2.6800 | 0.9985 | 77.9260 | 1.7274 | 0.7211 | 0.0394 |
| mutual_inhibition | 256 | 5 | 5.2800 | 1.7894 | 4.1400 | 1.8555 | 41.2330 | 12.8363 | 0.4366 | 0.0959 |
| mutual_inhibition | 384 | 5 | 3.5200 | 1.1432 | 3.0400 | 1.4029 | 17.1970 | 6.8110 | 0.4475 | 0.1049 |

## Fixture Comparison

| H | bytepair_proj_mean_peak | mutual_inhibition_mean_peak | delta_bytepair_minus_mi | welch_t | welch_p |
| --- | --- | --- | --- | --- | --- |
| 128 | 5.24 | 3.76 | 1.48 | 2.34832 | 0.0476002 |
| 256 | 3.62 | 5.28 | -1.66 | -1.88954 | 0.111429 |
| 384 | 3.16 | 3.52 | -0.36 | -0.310299 | 0.76714 |

## Peak-Final Gap

| fixture | H | mean_gap_pp | std_gap_pp | max_gap_pp |
| --- | --- | --- | --- | --- |
| bytepair_proj | 128 | 1.1000 | 0.9000 | 2.6000 |
| bytepair_proj | 256 | 1.2600 | 0.6656 | 2.3000 |
| bytepair_proj | 384 | 0.8200 | 1.1145 | 2.4000 |
| mutual_inhibition | 128 | 1.0800 | 0.9094 | 2.6000 |
| mutual_inhibition | 256 | 1.1400 | 0.3209 | 1.6000 |
| mutual_inhibition | 384 | 0.4800 | 0.3421 | 1.0000 |

## Phase B Replication Check

- Available: `True`
- Seeds match: `True`
- Checked metrics match: `True`

## Interpretation Boundary

- Phase B's training-horizon confound claim is validated for the mutual_inhibition fixture, not automatically for bytepair_proj.
- Bytepair_proj H=384 appears dominated by collapse/prune/low-accept dynamics; it needs a separate prune-policy ablation rather than being folded into Phase B.1.
- The safe unified claim is recipe-dependence: H interacts with training horizon and fixture policy, so H is not a standalone architectural verdict.
