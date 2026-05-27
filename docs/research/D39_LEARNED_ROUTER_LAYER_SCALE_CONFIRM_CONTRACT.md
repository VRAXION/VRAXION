# D39 Learned Router Layer Scale Confirm Contract

D39 scale-confirms the D38 learned conditioning/router field on the same controlled known-rule symbolic pocket task.

## Questions

1. Does the mutable learned router gate remain strong over more seeds and data?
2. Does it stay strong on OOD rows where the board symbols are transformed and targets are recomputed from the same formula rule?
3. Do shuffled-gate and no-family controls still collapse?
4. Is the learned gate identity stable across seeds?
5. Is the result still bounded to known-rule formula binding only?

## Dataset

- 3x3 symbolic board with symbols 0-8.
- Known family in `row`, `col`, `pair`, `mirror`, `diag`.
- Nine pockets A-I, indexed 0-8 internally.
- Exactly one pocket contains the target symbol.
- `expected_selected` points to that unique target-containing pocket.
- OOD does not apply an arbitrary label shift; targets are recomputed from transformed boards by the same family formula.

## Family Formulas

- `row`: `(b[1][0] + b[1][2]) % 9`
- `col`: `(b[0][1] + b[2][1]) % 9`
- `pair`: `(b[0][0] + b[2][2]) % 9`
- `mirror`: `(b[2][0] + b[0][2]) % 9`
- `diag`: `(b[0][0] + b[1][2] + b[2][1]) % 9`

## Arms

- `MONOLITHIC_FORMULA_BASELINE`
- `ORACLE_GATED_RULE_FORMULA_UPPER_BOUND`
- `MUTABLE_LEARNED_ROUTER_GATE`
- `SHUFFLED_GATE_CONTROL`
- `NO_FAMILY_INPUT_CONTROL`
- `EXPLICIT_TARGET_STATE_UPPER_BOUND`

## Learned Router

The mutable learned router gate uses:

- family one-hot vector length 5;
- formula-match vector length 5 for each pocket;
- mutable gate matrix `G` with shape `[5 family rows, 5 formula columns]`;
- optional gate bias and pocket bias;
- `score_i = sum_j G[family, j] * formula_match_i[j] + biases`.

The gate starts from random initialization and is optimized by mutation/search using the supported mutation families:

- `gate_weight_delta`
- `gate_row_delta`
- `gate_column_delta`
- `gate_bias_delta`
- `pocket_bias_delta`
- `gate_row_swap`
- `gate_column_swap`
- `prune_small_weights`

## Default Run

```bash
python scripts/probes/run_d39_learned_router_layer_scale_confirm.py --out target/pilot_wave/d39_learned_router_layer_scale_confirm/smoke --seeds 8501,8502,8503,8504,8505,8506,8507,8508 --train-rows-per-seed 800 --test-rows-per-seed 800 --ood-rows-per-seed 800 --generations 500 --population 128 --workers auto --cpu-target saturate --heartbeat-sec 20
```

Scale-lite is allowed only if local runtime is too high:

```bash
python scripts/probes/run_d39_learned_router_layer_scale_confirm.py --out target/pilot_wave/d39_learned_router_layer_scale_confirm/scale_lite --seeds 8501,8502,8503,8504,8505 --train-rows-per-seed 500 --test-rows-per-seed 500 --ood-rows-per-seed 500 --generations 300 --population 96 --workers auto --cpu-target saturate --heartbeat-sec 20
```

## Positive Decision

D39 is scale-confirmed only when:

- dataset invariants hold exactly;
- known-rule oracle test and OOD accuracy are 1.0;
- learned mean test and OOD accuracy are at least 0.95;
- learned min seed test and OOD accuracy are at least 0.90;
- oracle and explicit upper bounds are at least 0.99;
- learned test accuracy beats monolithic by at least 0.45;
- learned test accuracy beats shuffled gate by at least 0.70;
- learned test accuracy beats no-family by at least 0.45;
- shuffled gate test accuracy is at most 0.25;
- no-family test accuracy is at most 0.50.

## Hard Boundaries

D39 is not hidden-rule Raven solving, DNA/genome v2, natural-language reasoning, Gemma-like assistant capability, architecture superiority, general intelligence, or Raven solved.

A positive D39 proves only that the learned conditioning/router field scale-confirms on a controlled known-rule symbolic pocket task. It does not prove hidden-rule Raven reasoning, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.
