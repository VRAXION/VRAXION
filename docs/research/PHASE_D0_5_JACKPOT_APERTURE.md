# Phase D0.5 Jackpot Aperture

## Verdict

D0.5 resampled the completed Phase B.1 `K=9` candidate logs as prefix jackpots
`K in {1,2,3,5,9}` without launching new runs. It confirms that jackpot size is
an upstream search-aperture parameter: it changes the distribution presented to
the acceptance policy before `strict`, `zero_p`, or `ties` is applied.

The result does not justify an immediate epsilon sweep. It shows that `ties`
acceptance is already near-saturated by small K, while strict positive discovery
keeps increasing with K.

## 40k Strict Reference

| K | positive best | zero best | negative best | ties accept | strict accept | strict independent prediction | C_K |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.024240 | 0.816590 | 0.130870 | 0.840830 | 0.024240 | 0.025303 | 3.499e-6 |
| 2 | 0.047670 | 0.915825 | 0.025955 | 0.963495 | 0.047670 | 0.049946 | 3.413e-6 |
| 3 | 0.069765 | 0.918300 | 0.007735 | 0.988065 | 0.069765 | 0.073947 | 3.364e-6 |
| 5 | 0.110015 | 0.887570 | 0.001685 | 0.997585 | 0.110015 | 0.120091 | 3.190e-6 |
| 9 | 0.178535 | 0.821280 | 0.000170 | 0.999815 | 0.178535 | 0.205400 | 2.974e-6 |

## Interpretation

- `K` is the sampling funnel; `zero_p` is the downstream valve.
- `ties` saturates quickly: by `K=2`, ties acceptance is already `96.3%`; by `K=3`, it is `98.8%`.
- Strict discovery still benefits from larger K: positive-best rate rises from `2.4%` at `K=1` to `17.9%` at `K=9`.
- Per-cost `C_K` declines mildly as K grows, so the science question is not simply "largest K wins"; it is a tradeoff between discovery rate, neutral drift volume, and evaluation cost.
- The independent-binomial prediction roughly tracks strict accept, so the K effect is largely explained by order statistics of candidate sampling.

## D1 Implication

A fixed `K=9` zero-p sweep is useful but incomplete. A better D1 should include
a small K axis:

```text
K in {1,3,9}
policy in {strict, zero_p=0.3, zero_p=1.0}
seeds = 5
horizon = 40k
```

This is `45` runs and directly tests whether the best result comes from:

- high discovery with saturated neutral aperture (`K=9`),
- lower-cost partial aperture (`K=3`), or
- minimal pooling with high selection pressure (`K=1`).

If runtime must stay closer to the original 30-run D1, use:

```text
K in {3,9}
policy in {strict, zero_p=0.1, zero_p=0.3, zero_p=1.0}
seeds = 5
```

This is `40` runs and preserves the key comparison.
