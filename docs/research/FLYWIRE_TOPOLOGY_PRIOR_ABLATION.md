# FlyWire Topology Prior Ablation

Source summary:

- `target/context-cancellation-probe/flywire-topology-prior/topology_prior_summary.json`

Source graph:

- `/home/deck/work/flywire/mushroom_body.graphml`

## 1. Goal

Test whether a local mushroom-body / FlyWire-like recurrent topology improves the existing latent-refraction / authority-switch toy task compared with random sparse topology.

This is only a topology-prior sanity test. The local GraphML is used as a recurrent mask source, not as a biological simulation.

## 2. Setup

Benchmark:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py \
  --experiment latent_refraction \
  --input-mode entangled \
  --hidden 64 \
  --update-rate 0.3 \
  --steps 5 \
  --epochs 200 \
  --seeds 5 \
  --no-random-label-control
```

Topology modes:

- `random_sparse`: current random sparse baseline.
- `ring_sparse`: local ring plus skip edges.
- `reciprocal_motif`: random sparse mask with high bidirectional-pair fraction.
- `hub_rich`: heavy-tailed hub topology.
- `flywire_sampled`: weighted sampled subgraph from `mushroom_body.graphml`.
- `flywire_class_sampled`: class-balanced FlyWire sample using `KC`, `MBON`, `MBIN`, and `ORN/PN`-like classes.

Fairness rules used:

- same hidden size,
- same recurrent edge budget,
- same training config,
- same seeds,
- same recurrent weight initialization scale,
- same update depth/rate.

All modes averaged the same actual edge count:

```text
actual_edge_count: 545
density: 0.133057
```

## 3. Main Results

| Topology | Accuracy | Acc Std | Zero Recurrent | Randomized Recurrent | Recurrence Gain | Refraction Final | Authority Switch | Reciprocal Fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `random_sparse` | `0.939348` | `0.026778` | `0.592982` | `0.526566` | `0.346366` | `0.412281` | `0.397243` | `0.100348` |
| `ring_sparse` | `0.914286` | `0.031767` | `0.574436` | `0.518797` | `0.339850` | `0.370677` | `0.341353` | `0.935244` |
| `reciprocal_motif` | `0.926817` | `0.028889` | `0.570677` | `0.539348` | `0.356140` | `0.382456` | `0.362406` | `0.998759` |
| `hub_rich` | `0.971429` | `0.015356` | `0.580702` | `0.514035` | `0.390727` | `0.456642` | `0.444612` | `0.374419` |
| `flywire_sampled` | `0.932832` | `0.015319` | `0.567669` | `0.532581` | `0.365163` | `0.405764` | `0.382707` | `0.275222` |
| `flywire_class_sampled` | `0.929323` | `0.020767` | `0.588471` | `0.503008` | `0.340852` | `0.391729` | `0.370426` | `0.303018` |

Degree shape:

| Topology | In Std | In Max | Out Std | Out Max |
|---|---:|---:|---:|---:|
| `random_sparse` | `2.606772` | `14.6` | `2.489901` | `13.2` |
| `ring_sparse` | `0.483242` | `8.0` | `0.483242` | `8.0` |
| `reciprocal_motif` | `2.684198` | `14.4` | `2.680930` | `14.4` |
| `hub_rich` | `5.847085` | `39.8` | `10.967095` | `59.4` |
| `flywire_sampled` | `4.505165` | `19.6` | `4.958223` | `18.6` |
| `flywire_class_sampled` | `4.858099` | `18.8` | `5.788496` | `19.8` |

## 4. Delta vs Random Sparse

| Topology | Accuracy Delta | Gain Delta | Refraction Delta | Authority Delta | Acc Std Delta |
|---|---:|---:|---:|---:|---:|
| `ring_sparse` | `-0.025062` | `-0.006516` | `-0.041604` | `-0.055890` | `+0.004989` |
| `reciprocal_motif` | `-0.012531` | `+0.009774` | `-0.029825` | `-0.034837` | `+0.002110` |
| `hub_rich` | `+0.032081` | `+0.044361` | `+0.044361` | `+0.047369` | `-0.011422` |
| `flywire_sampled` | `-0.006516` | `+0.018797` | `-0.006517` | `-0.014536` | `-0.011459` |
| `flywire_class_sampled` | `-0.010025` | `-0.005514` | `-0.020552` | `-0.026817` | `-0.006011` |

## 5. Interpretation

The clear positive topology in this grid is `hub_rich`:

- highest accuracy,
- highest recurrence gain,
- highest final refraction index,
- highest authority-switch score,
- lower accuracy seed variance than `random_sparse`,
- randomized recurrent control collapses near baseline/chance-like behavior.

The local FlyWire-derived modes are not positive on the main authority/refraction metrics:

- `flywire_sampled` has slightly lower accuracy than `random_sparse`,
- lower refraction final,
- lower authority-switch score,
- but lower seed variance and higher recurrence gain.

The class-balanced FlyWire sample is weaker than the non-balanced FlyWire sample in this run.

The high-reciprocity controls are useful negative controls:

- `ring_sparse` and `reciprocal_motif` both have much higher reciprocal fraction than `random_sparse`,
- neither improves authority/refraction,
- so "more reciprocal pairs" alone is not sufficient.

## 6. Verdict

```json
{
  "flywire_topology_prior": "negative_for_metric_gain_unclear_for_stability",
  "hub_rich_topology_prior": "positive_in_this_toy_grid",
  "reason": "hub_rich improves accuracy, recurrence gain, refraction_index_final, authority_switch_score, and seed variance vs random_sparse at matched edge count. flywire_sampled does not beat random_sparse on authority/refraction, but reduces seed variance."
}
```

## 7. Safe Claim

In this controlled toy setting, a heavy-tailed hub-rich recurrent mask improves frame-conditioned latent refraction metrics at matched edge budget.

The local mushroom-body / FlyWire-sampled mask does not improve the main authority/refraction metrics in this first pass, though it may provide a modest stability prior.

## 8. Claim Boundary

Do not claim:

- biology,
- full FlyWire behavior,
- FlyWire proves VRAXION,
- consciousness,
- biological equivalence,
- production validation.

This is only a topology-prior ablation.

## 9. Proposed Next Step

Do not expand into full biology. If this branch continues, the clean next test is:

```text
hub_rich vs flywire_sampled, matched not only on edge count but also degree distribution.
```

That would separate:

```text
hub/heavy-tail benefit
```

from:

```text
specific mushroom-body wiring benefit
```
