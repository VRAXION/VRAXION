# D40 Rule Family Inference Router Prototype Contract

D40 tests whether support/context evidence can identify the intended rule family and feed that inferred family into the already-confirmed known-formula router path.

## Scope

D40 is a controlled symbolic support-context rule-family inference task:

support/context boards -> infer rule family scores -> route through rule/formula gate -> compute query target symbol -> select the unique target-containing pocket.

It is not full Raven solved, raw visual Raven reasoning, DNA/genome v2, natural-language reasoning, Gemma-like assistant capability, architecture superiority, consciousness, or general intelligence.

## Dataset

Each row contains:

- support/context boards with completed center cells;
- one query board with the center/target hidden;
- nine candidate pockets A-I, indexed 0-8 internally;
- exactly one pocket containing the query target symbol;
- `expected_selected` pointing to that unique target-containing pocket;
- intended family in `row`, `col`, `pair`, `mirror`, `diag`.

The intended family must be inferable from support evidence. The query board alone must not reveal the family, and learned hidden-family arms must not receive the true family label as an input.

## Family Formulas

- `row`: `(b[1][0] + b[1][2]) % 9`
- `col`: `(b[0][1] + b[2][1]) % 9`
- `pair`: `(b[0][0] + b[2][2]) % 9`
- `mirror`: `(b[2][0] + b[0][2]) % 9`
- `diag`: `(b[0][0] + b[1][2] + b[2][1]) % 9`

## Support Evidence

For each support board, the center cell is set to the intended family formula value. The support evidence vector has length five:

`e[j] = count(center == formula_j(board)) / support_count`.

Rows are rejected unless:

- intended family evidence is uniquely highest;
- support evidence count gap is at least 1.0 or normalized margin is at least 0.34;
- ambiguous support rate is 0.0;
- multi-family support tie rate is 0.0.

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

## Learned Rule Inference

The mutable learned rule-family inference arm uses:

- support evidence vector length 5;
- mutable rule matrix `R` with shape `[5 evidence channels, 5 family outputs]`;
- optional family bias vector length 5;
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

D40 is positive only when:

- dataset invariants hold exactly;
- support evidence oracle test and OOD accuracy are at least 0.99;
- OOD rule invariance holds;
- learned selected-pocket test and OOD accuracy are at least 0.95;
- learned rule-family test and OOD accuracy are at least 0.95;
- learned min seed test and OOD accuracy are at least 0.90;
- oracle arms are at least 0.99;
- learned beats query-only test by at least 0.40;
- learned beats shuffled-support test by at least 0.70;
- learned beats no-support test by at least 0.40;
- same-query-different-support accuracy is at least 0.95.

## Boundary

A positive D40 proves only that support-evidence-based rule-family inference can feed the known-rule router path on a controlled symbolic pocket task. It does not prove raw visual Raven reasoning, full hidden-rule Raven solving, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.
