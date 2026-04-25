# Phase B Verdict

## Verdict

Phase B supports a training-horizon confound: B1 (2x steps) has the highest mean peak accuracy (5.50% vs B0 3.52%) and recovers to the H=256 reference band (5.28 +/- 1.79%). The Welch p-value is 0.0469, directional at raw alpha=0.05 but not Bonferroni-significant at alpha=0.0125.

Do not claim formal preregistered significance. Claim a strong directional finding and require an n>=10 replication for the strict Bonferroni gate.

## Artifact Integrity

- Runs: `25`
- Candidate rows: `6,300,000`
- Checkpoints: `25`
- Panel timeseries files: `25`
- Panel rows: `300`

## Arm Statistics

| arm | n | peak_acc_mean | peak_acc_std | peak_acc_median | peak_acc_min | peak_acc_max | final_acc_mean | accept_rate_pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 5 | 3.5200 | 1.1432 | 3.8000 | 2.0000 | 4.7000 | 3.0400 | 17.1970 |
| B1 | 5 | 5.5000 | 1.4748 | 6.1000 | 3.1000 | 6.6000 | 3.6800 | 17.8540 |
| B2 | 5 | 3.2600 | 2.9331 | 2.0000 | 0.0000 | 7.1000 | 2.6000 | 19.5880 |
| B3 | 5 | 3.1600 | 2.0707 | 3.1000 | 0.0000 | 5.3000 | 2.5200 | 8.2280 |
| B4 | 5 | 2.0000 | 1.7103 | 1.8000 | 0.0000 | 4.7000 | 1.6200 | 1.2960 |

## Pre-Registered Tests

| arm | baseline | mean_peak | baseline_mean_peak | delta_mean_peak | bootstrap_ci95_low | bootstrap_ci95_high | welch_t | welch_p | bonferroni_alpha | cohen_d | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | B0 | 5.5 | 3.52 | 1.98 | 0.48 | 3.38 | 2.37266 | 0.0469436 | 0.0125 | 1.5006 | DIRECTIONAL_UNDERPOWERED_AFTER_BONFERRONI |
| B2 | B0 | 3.26 | 3.52 | -0.26 | -2.62 | 2.24 | -0.184681 | 0.860483 | 0.0125 | -0.116802 | FAIL_OR_INCONCLUSIVE |
| B3 | B0 | 3.16 | 3.52 | -0.36 | -2.26 | 1.4 | -0.34032 | 0.744797 | 0.0125 | -0.215237 | FAIL_OR_INCONCLUSIVE |
| B4 | B0 | 2 | 3.52 | -1.52 | -3.04 | 0.14 | -1.65217 | 0.142604 | 0.0125 | -1.04493 | FAIL_OR_INCONCLUSIVE |

## Constructability

| arm | n | C_K_window_ratio_mean | V_raw_mean | V_sel_mean | M_pos_mean | R_neg_mean | cost_eval_ms_mean | accepted_nonpositive_steps_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 5 | 2.85572e-06 | 0.0142244 | 0.09905 | 0.00377716 | 0.0196212 | 11.7605 | 1458.4 |
| B1 | 5 | 2.9177e-06 | 0.0153089 | 0.10783 | 0.00363558 | 0.0283107 | 12.564 | 2828.4 |
| B2 | 5 | 1.94911e-06 | 0.00927278 | 0.11951 | 0.00738714 | 0.0528639 | 11.8746 | 1527.4 |
| B3 | 5 | 5.97008e-07 | 0.00597111 | 0.04258 | 0.00615419 | 0.0558314 | 21.3682 | 794 |
| B4 | 5 | 2.78633e-07 | 0.000322222 | 0.00258 | 0.0241783 | 0.0604391 | 11.2036 | 207.6 |

C_K agrees only weakly with peak accuracy: B1 is slightly above B0, while B3/B4 collapse clearly. Treat C_K as a diagnostic panel for now, not as a frozen scalar objective.

## Operator Findings

| arm | operator_id | candidate_rows | V_raw_mean | M_pos_mean | R_neg_mean | usefulness_proxy |
| --- | --- | --- | --- | --- | --- | --- |
| B2 | add_edge | 396258 | 0.00627515 | 0.0312305 | 0.0462432 | 0.000195976 |
| B1 | theta | 89951 | 0.0369862 | 0.00416689 | 0.0326719 | 0.000154117 |
| B0 | theta | 44695 | 0.0339138 | 0.00418736 | 0.0227949 | 0.000142009 |
| B1 | channel | 179823 | 0.0263986 | 0.00397887 | 0.0332987 | 0.000105037 |
| B3 | reverse | 117075 | 0.00771881 | 0.0131299 | 0.0550943 | 0.000101347 |
| B0 | channel | 89894 | 0.0234507 | 0.00432113 | 0.0227261 | 0.000101333 |
| B0 | loop3 | 44836 | 0.0220567 | 0.00408618 | 0.0194989 | 9.01279e-05 |
| B1 | loop3 | 89765 | 0.0230123 | 0.0038077 | 0.0264764 | 8.76239e-05 |
| B2 | theta | 89845 | 0.0222326 | 0.00313248 | 0.0601718 | 6.96433e-05 |
| B1 | rewire | 162333 | 0.0184083 | 0.00347028 | 0.0252675 | 6.38821e-05 |
| B3 | theta | 44725 | 0.0148304 | 0.00430608 | 0.0598918 | 6.38607e-05 |
| B1 | reverse | 233770 | 0.0180687 | 0.00347249 | 0.0252194 | 6.27436e-05 |

### Operator Interpretation

| operator_id | schedule_share | productivity_share | V_raw_mean | M_pos_mean | R_neg_mean | usefulness_proxy |
| --- | --- | --- | --- | --- | --- | --- |
| theta | 4.97% | 22.26% | 0.0339138 | 0.00418736 | 0.0227949 | 0.000142009 |
| channel | 9.99% | 15.88% | 0.0234507 | 0.00432113 | 0.0227261 | 0.000101333 |
| loop3 | 4.98% | 14.13% | 0.0220567 | 0.00408618 | 0.0194989 | 9.01279e-05 |
| reverse | 13.00% | 9.75% | 0.0176617 | 0.00352278 | 0.0177283 | 6.22184e-05 |
| rewire | 9.00% | 9.51% | 0.0170402 | 0.0035618 | 0.0179231 | 6.06936e-05 |

- `projection_weight` is effectively inert in B0: it used 4.98% of candidate draws but only reached `V_raw=8.94368e-05` and `usefulness_proxy=6.28798e-08`.
- `theta` is the strongest B0 operator by `V_raw*M_pos`: current draw share 4.97%, productivity share 22.26%, `V_raw=0.0339138`.
- `channel` is the second strongest B0 operator by the same proxy: current draw share 9.99%, productivity share 15.88%.
- The current mutation schedule is plausibly misallocated: high-draw operators like `add_edge` are not the most productive, while `theta` is under-sampled. A theta/channel-heavy retuned schedule is a Phase C hypothesis, not a Phase B result.
- B2 shows heavy-tail behavior rather than reliable improvement: peak accuracy mean `3.26%`, median `2.00%`, std `2.93%`, range `0.00%..7.10%`.
- B3 increases destructive risk: arm-level `R_neg` is `2.68x` B0 on average across operators (range `1.46x..3.25x`). The worst seed-level B3 case has nonzero `R_neg` mean `0.153` and max `0.159`.
- Do not label B3 as resonance or chaos from this analysis alone. The data supports higher destructive mutation sensitivity at 12 ticks; the mechanism needs a separate perturbation/Derrida-style test.

## Neutral Accept Policy

- Pearson correlation between peak accuracy and neutral-accept fraction: `0.143989846063325`
- Neutral accepted mutations are valid under `accept_ties=true`; this is policy behavior, not run corruption.
- Follow-up should test `accept_ties=true` vs `accept_ties=false` on the winning B1 horizon condition.

## Next Step

Run Phase B.1: H=384 with horizon scaling and tie-policy ablation. Minimum matrix: `accept_ties=true/false` x `20k/40k/80k` on the same seeds, with the same candidate/panel logging.
