# D41 Rule Family Inference Router Scale Confirm Contract

D41 scale-confirms the D40 support-evidence rule-family inference router prototype.

## Scope

D41 tests whether support-evidence-based learned rule-family inference remains stable over more seeds, larger datasets, support-count variation, and clean/noisy-majority symbolic support evidence while feeding the known-rule router path.

Pipeline:

support/context evidence -> learned rule-family inference -> rule family state -> confirmed router path -> formula channel -> query target symbol -> selected pocket.

D41 is not raw visual Raven reasoning, full hidden-rule Raven solving, DNA/genome v2, natural-language reasoning, Gemma-like assistant capability, architecture superiority, consciousness, or general intelligence.

## Dataset

Each row contains:

- support/context boards with completed center cells;
- one query board with target hidden;
- nine candidate pockets A-I, indexed 0-8 internally;
- exactly one pocket containing the query target symbol;
- `expected_selected` pointing to that unique target-containing pocket;
- intended family in `row`, `col`, `pair`, `mirror`, `diag`.

Support evidence is a symbolic intermediate representation:

`support_evidence[j] = count(center == formula_j(board)) / support_count`.

The true family label must not be given to learned support-inference arms.

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
- held-out OOD support/query templates, pocket permutations, and symbol distributions without changing the answer rule

Rows are rejected unless intended-family evidence is uniquely highest. Positive rows must have:

- ambiguous_support_rate = 0.0
- multi_family_support_tie_rate = 0.0
- intended_family_unique_evidence_rate = 1.0
- counterfactual_target_collision_rate = 0.0
- wrong_support_query_mismatch_rate = 1.0

## Arms

- `RANDOM_BASELINE`
- `QUERY_ONLY_MONOLITHIC_BASELINE`
- `SUPPORT_EVIDENCE_ORACLE_RULE_SELECTOR`
- `TRUE_FAMILY_ORACLE_UPPER_BOUND`
- `MUTABLE_LEARNED_RULE_FAMILY_INFERENCE`
- `MUTABLE_LEARNED_RULE_INFERENCE_PLUS_LEARNED_ROUTER`
- `SHUFFLED_SUPPORT_EVIDENCE_CONTROL`
- `NO_SUPPORT_EVIDENCE_CONTROL`
- `WRONG_SUPPORT_CONTROL`
- `SAME_QUERY_DIFFERENT_SUPPORT_COUNTERFACTUAL`
- `SUPPORT_COUNT_GENERALIZATION_REPORT_ONLY`
- `SUPPORT_MARGIN_STRATA_REPORT_ONLY`

## Learned Rule Inference

The mutable learned rule-family inference arm uses:

- support evidence vector length 5;
- mutable rule matrix `R` with shape `[5 evidence channels, 5 family outputs]`;
- optional family bias;
- `rule_scores = evidence_vector @ R + bias`;
- `inferred_family = argmax(rule_scores)`;
- inferred family routed through the known formula candidate router.

Supported mutations:

- `rule_weight_delta`
- `rule_row_delta`
- `rule_column_delta`
- `rule_bias_delta`
- `rule_row_swap`
- `rule_column_swap`
- `prune_small_weights`

## Positive Decision

D41 is scale-confirmed only when:

- dataset invariants hold exactly;
- support evidence oracle test and OOD accuracy are at least 0.99;
- OOD rule invariance holds;
- learned selected-pocket test and OOD accuracy are at least 0.98;
- learned rule-family test and OOD accuracy are at least 0.98;
- learned min seed test and OOD accuracy are at least 0.95;
- minimum support-count accuracy is at least 0.95;
- minimum margin-strata accuracy is at least 0.95;
- oracle arms are at least 0.99;
- learned beats query-only test by at least 0.60;
- learned beats shuffled-support test by at least 0.80;
- learned beats no-support test by at least 0.60;
- same-query-different-support accuracy is at least 0.98;
- wrong-support follow rate is at least 0.98;
- wrong-support selected-pocket accuracy is at most 0.05.

## Boundary

A positive D41 proves only that support-evidence-based rule-family inference scale-confirms while feeding the known-rule router path on a controlled symbolic pocket task. It does not prove raw visual Raven reasoning, full hidden-rule Raven solving, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.
