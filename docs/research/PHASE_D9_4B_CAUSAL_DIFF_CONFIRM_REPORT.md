# Phase D9.4b Causal Diff Confirm

Date: 2026-04-29

Verdict: `D9_4B_4000_CONFIRM_PASS_16000_PENDING`

## Summary

D9.4b re-ran the beta.8 causal-diff gate with a longer independent evaluation than the original D9.4a smoke.

The `eval_len=4000`, 30 fresh-seed confirm preserves the D9.4a conclusion: the beta.8 H=384 generalist improvement is explained by an edge-plus-threshold co-adapted package, not by projection/channel/polarity changes.

The `eval_len=16000`, 30 fresh-seed layer was not launched in this pass. The runtime probe estimated it at roughly 136 minutes, so it remains a separate long confirm before claiming the strongest D9.4b verdict.

## Inputs

Baseline:

```text
output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt
```

Target:

```text
output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt
```

Confirm output:

```text
output/phase_d9_4b_causal_diff_confirm_20260429/eval_len_4000/
```

## Runtime

- Probe: `eval_len=1000`, 2 seeds, 33.9s.
- Confirm: `eval_len=4000`, 30 seeds, 1965.5s.
- Estimated deferred layer: `eval_len=16000`, 30 seeds, about 136 minutes.

## Structural Diff

The structural diff is unchanged from D9.4a:

- Edges: 9718 -> 10140
- Added edges: 457
- Removed edges: 35
- Net edge delta: +422
- Threshold changes: 143
- Channel changes: 0
- Polarity changes: 0
- Projection bytes equal: true

## Target Multi-Objective Delta

At `eval_len=4000`, 30 fresh seeds:

| metric | delta |
|---|---:|
| smooth | +0.016988459485 |
| accuracy | +0.005175000000 |
| echo | +0.000000000000 |
| unigram | +0.005413779200 |
| mo_score | +0.027696628285 |
| mo_class | `FULL_GENERALIST` |

This passes the target-retention gate: smooth, accuracy, and unigram remain positive, echo remains neutral, and the candidate remains `FULL_GENERALIST`.

## Causal Loss Fractions

| ablation | loss_fraction_vs_target |
|---|---:|
| ablate_added_edges | 5.729 |
| ablate_all_edges | 5.608 |
| ablate_thresholds | 5.707 |
| ablate_all_edges_thresholds | 1.000 |

Both all-edge reversion and threshold reversion remove far more than 50% of the target multi-objective score. The all-edge-plus-threshold revert returns to the baseline reference. This supports `EDGE_THRESHOLD_COADAPTATION` at `eval_len=4000`.

Forward graft sanity also supports the same interpretation: edge-only and threshold-only grafts fail retention, while `graft_all_edges_thresholds` reproduces the full target score.

## Gate Result

`eval_len=4000`: `PASS`

Reason:

- verdict is `EDGE_THRESHOLD_COADAPTATION`
- target is `FULL_GENERALIST`
- smooth, accuracy, and unigram deltas are positive
- edge-all loss >= 0.5
- threshold loss >= 0.5
- projection/channel/polarity are unchanged

`eval_len=16000`: `PENDING`

Reason:

- estimated runtime is about 136 minutes from the measured probe
- should be run as a separate long confirm before upgrading this to `D9_4B_CAUSAL_CONFIRM_PASS`

## Implication

The D10 basin-universality pipeline is not blocked by the 4000-seed causal gate. The correct next choice is either:

1. Run the deferred `eval_len=16000` D9.4b confirm for the strongest causal lock.
2. Start D10 seed/task/H scout work with the explicit caveat that D9.4b currently has a 4000-confirm pass and a 16000-confirm pending.

The conservative default is to run the deferred 16000 layer before spending GPU or H=512+ budget.
