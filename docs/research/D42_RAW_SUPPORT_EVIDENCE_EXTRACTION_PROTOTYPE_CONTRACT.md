# D42 Raw Support Evidence Extraction Prototype Contract

D42 tests whether a mutable learned extractor can derive support-evidence-like rule scores from raw symbolic support boards, then feed the D41 support-rule-router stack.

## Scope

D42 removes the D41 shortcut where learned arms received a precomputed `support_evidence` vector. The main learned arm sees raw symbolic support boards with completed centers and fixed formula-candidate primitive values computed from those boards. It must learn the center-symbol to formula-candidate equality kernel and channel-to-family gate.

Pipeline:

raw symbolic support boards -> learned support-evidence extractor -> rule-family scores -> D41 rule-family/router path -> query target symbol -> selected pocket.

D42 is controlled symbolic reasoning. It is not raw visual Raven reasoning, formula primitive discovery, full hidden-rule Raven solving, DNA/genome v2, natural-language reasoning, Gemma-like assistant capability, architecture superiority, consciousness, or general intelligence.

## Dataset

Each row contains:

- raw support/context boards with completed center cells;
- one separate query board with target hidden;
- nine candidate pockets A-I, indexed 0-8 internally;
- exactly one pocket containing the query target symbol;
- `expected_selected` pointing to that unique target-containing pocket;
- intended family in `row`, `col`, `pair`, `mirror`, `diag`.

The oracle support-evidence vector is computed only for audits and upper-bound arms:

`support_evidence[j] = count(center == formula_j(board)) / support_count`.

The learned raw extractor must not receive that aggregate vector, the true family label, `expected_selected`, or the query target.

## Family Formulas

- `row`: `(b[1][0] + b[1][2]) % 9`
- `col`: `(b[0][1] + b[2][1]) % 9`
- `pair`: `(b[0][0] + b[2][2]) % 9`
- `mirror`: `(b[2][0] + b[0][2]) % 9`
- `diag`: `(b[0][0] + b[1][2] + b[2][1]) % 9`

## Scale Axes

- support counts: 1, 2, 3, 5
- support strata: `clean_unanimous`, `noisy_majority`
- margin strata: `margin_low`, `margin_high`
- held-out OOD support/query templates, pocket permutations, support-count mixtures, and symbol distributions without changing the answer rule

Rows are rejected unless intended-family evidence is uniquely highest. Positive rows must have:

- duplicate_target_pocket_rate = 0.0
- missing_target_pocket_rate = 0.0
- expected_selected_points_to_target_rate = 1.0
- ambiguous_support_rate = 0.0
- multi_family_support_tie_rate = 0.0
- intended_family_unique_evidence_rate = 1.0
- counterfactual_target_collision_rate = 0.0
- wrong_support_query_mismatch_rate = 1.0

## Arms

- `RANDOM_BASELINE`
- `QUERY_ONLY_BASELINE`
- `PRECOMPUTED_SUPPORT_EVIDENCE_UPPER_BOUND`
- `ORACLE_RAW_SUPPORT_EVIDENCE_EXTRACTOR`
- `MUTABLE_LEARNED_RAW_SUPPORT_EVIDENCE_EXTRACTOR`
- `SHUFFLED_CENTER_CONTROL`
- `SHUFFLED_FORMULA_CANDIDATE_CONTROL`
- `NO_CENTER_CONTROL`
- `NO_FORMULA_CANDIDATE_CONTROL`
- `WRONG_SUPPORT_CONTROL`
- `SAME_QUERY_DIFFERENT_RAW_SUPPORT_COUNTERFACTUAL`
- `RAW_SUPPORT_PLUS_LEARNED_ROUTER_COMPOSITION`

## Learned Raw Extractor

Tier A is required and implemented:

- fixed formula-candidate primitive values are computed from raw support boards;
- the learned extractor receives center symbol, formula candidate symbol, and formula channel;
- the learned extractor does not receive the boolean equality value or aggregated support evidence.

Tier B is not implemented:

- formula primitive discovery is unavailable and not claimed.

The mutable learned extractor uses:

- equality matrix `E` with shape `[9 center symbols, 9 formula-candidate symbols]`;
- channel gate matrix `C` with shape `[5 formula channels, 5 family outputs]`;
- optional family bias;
- evidence score per formula channel accumulates `E[center, candidate_value_j]` across support boards;
- `rule_scores = evidence_scores @ C + bias`;
- `inferred_family = argmax(rule_scores)`.

Supported mutations:

- `equality_weight_delta`
- `equality_row_delta`
- `equality_column_delta`
- `channel_weight_delta`
- `channel_row_delta`
- `channel_column_delta`
- `rule_bias_delta`
- `equality_row_swap`
- `equality_column_swap`
- `channel_row_swap`
- `channel_column_swap`
- `prune_small_weights`

## Positive Decision

D42 is prototype-positive only when:

- dataset invariants hold exactly;
- support evidence oracle test and OOD accuracy are 1.0;
- OOD rule invariance holds;
- precomputed and oracle raw support upper-bound arms are at least 0.99;
- learned raw extractor selected-pocket test and OOD accuracy are at least 0.95;
- learned raw extractor rule-family test and OOD accuracy are at least 0.95;
- learned raw extractor min seed test and OOD accuracy are at least 0.90;
- minimum support-count accuracy is at least 0.90;
- minimum margin-strata accuracy is at least 0.90;
- learned beats query-only test by at least 0.60;
- learned beats shuffled-center test by at least 0.70;
- learned beats no-center test by at least 0.60;
- same-query-different-raw-support accuracy is at least 0.95;
- wrong-support follow rate is at least 0.95;
- wrong-support selected-pocket accuracy is at most 0.05.

## Boundary

A positive D42 proves only that a learned raw symbolic support-evidence extractor can feed the D41 support-rule-router stack on a controlled symbolic task, with fixed formula primitive candidates available. It does not prove raw visual Raven reasoning, formula primitive discovery, full hidden-rule Raven solving, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.
