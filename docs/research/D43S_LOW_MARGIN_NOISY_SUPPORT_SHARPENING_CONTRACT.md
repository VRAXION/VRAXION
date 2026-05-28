# D43S Low-Margin Noisy Support Sharpening Contract

## Goal

D43S is a targeted tail-diagnosis and hardening run for the D43 low-margin noisy-support failure pattern.

It focuses on controlled symbolic rows where support evidence is still unambiguous but the intended family wins by a small noisy-majority gap.

## Boundary

D43S only tests low-margin noisy-support tail hardening for D43 on a controlled symbolic task with fixed formula primitive candidates.

It does not prove formula primitive discovery, raw visual Raven reasoning, Raven solved, DNA/genome success, consciousness, AGI, general intelligence, or architecture superiority.

## Dataset Focus

The D43S dataset uses D43-style raw symbolic support boards, fixed formula primitive candidates, query boards, and unique target pockets.

The main distribution uses support counts `3` and `5`, with at least 70% of rows in `noisy_majority_margin_low`. Required strata are:

- `noisy_majority_margin_low`
- `noisy_majority_margin_high`
- `clean_unanimous`
- `pair_low_margin`
- `diag_low_margin`
- `row_low_margin`
- `support_count_3_low_margin`
- `support_count_5_low_margin`

## Arms

- `D43_BASELINE_REPLAY`
- `MARGIN_PRESERVING_OBJECTIVE`
- `TEMPERATURE_SHARPENED_EVIDENCE`
- `FAMILY_BALANCED_EDGE_OVERSAMPLING`
- `COMBINED_SHARPENED_MODEL`
- `HARD_VOTE_ORACLE_UPPER_BOUND`
- `SHUFFLED_CENTER_CONTROL`
- `SHUFFLED_FORMULA_CANDIDATE_CONTROL`
- `NO_CENTER_CONTROL`
- `SAME_QUERY_DIFFERENT_RAW_SUPPORT_COUNTERFACTUAL`
- `WRONG_SUPPORT_CONTROL`

## Hard Gates

Learned arms must not receive a precomputed support-evidence vector, boolean equality input, true family input, expected-selected feature, sample index lookup, or target oracle.

Positive D43S requires the combined sharpened model to improve noisy-majority low-margin accuracy over baseline by at least `0.001`, reach at least `0.999` on that stratum, keep overall test/OOD at least `0.999`, avoid easy-case regression above `0.0005`, and preserve collapsing controls.

