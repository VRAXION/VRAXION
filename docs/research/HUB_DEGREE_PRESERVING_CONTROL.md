# Hub Degree-Preserving Control

Source summary:

- `target/context-cancellation-probe/hub-degree-preserving-control/hub_degree_preserving_summary.json`

Related prior:

- `docs/research/HUB_RICH_TOPOLOGY_PRIOR_VALIDATION.md`

## 1. Goal

Separate two explanations for the earlier `hub_rich` topology result:

```text
A) hub-rich works mostly because of the in/out degree distribution.
B) hub-rich works because of the specific sampled hub wiring / motif pattern.
```

The important control is `hub_degree_preserving_random`: create a `hub_rich` mask, degree-preserving shuffle its edges before training, then train from scratch.

This is different from the earlier post-training shuffle. Post-training shuffle only proved the trained model relies on its learned wiring. This control tests whether a randomized graph with the same hub degree sequence can learn the task from scratch.

## 2. Setup

Experiments:

- `latent_refraction`
- `multi_aspect_token_refraction`

Topology modes:

- `random_sparse`
- `hub_rich`
- `hub_degree_preserving_random`
- `flywire_sampled`
- `flywire_degree_preserving_random`

Configs:

| Experiment | Hidden | Update Rates | Steps | Epochs | Seeds |
|---|---:|---:|---:|---:|---:|
| `latent_refraction` | `64` | `0.2`, `0.3` | `5` | `200` | `5` |
| `multi_aspect_token_refraction` | `128` | `0.2`, `0.3` | `5` | `220` | `5` |

Random-label control was skipped for this compact topology grid.

All modes use matched edge budget.

## 3. Main Hub Control Results

### Latent Refraction

| Update | Topology | Accuracy | Acc Std | Recurrence Gain | Refraction Final | Authority Switch |
|---:|---|---:|---:|---:|---:|---:|
| `0.2` | `random_sparse` | `0.935589` | `0.018170` | `0.354637` | `0.410025` | `0.390978` |
| `0.2` | `hub_rich` | `0.945363` | `0.024474` | `0.367669` | `0.424561` | `0.398998` |
| `0.2` | `hub_degree_preserving_random` | `0.951378` | `0.025906` | `0.389975` | `0.436842` | `0.420802` |
| `0.3` | `random_sparse` | `0.939348` | `0.026778` | `0.346366` | `0.412281` | `0.397243` |
| `0.3` | `hub_rich` | `0.971429` | `0.015356` | `0.390727` | `0.456642` | `0.444611` |
| `0.3` | `hub_degree_preserving_random` | `0.949373` | `0.011798` | `0.382456` | `0.423559` | `0.397494` |

Latent-refraction read:

- At update `0.2`, `hub_degree_preserving_random` beats both `random_sparse` and `hub_rich` on the main metrics.
- At update `0.3`, the original `hub_rich` mask remains strongest on accuracy, final refraction, and authority switching.
- `hub_degree_preserving_random` still beats `random_sparse` on accuracy, recurrence gain, and final refraction at update `0.3`, but it no longer matches the full `hub_rich` authority-switch result.

This supports a mixed explanation:

```text
hub degree distribution explains a real part of the latent-refraction benefit,
but the specific hub wiring can still matter under stronger/faster recurrent dynamics.
```

### Multi-Aspect Token Refraction

| Update | Topology | Accuracy | Acc Std | Recurrence Gain | Refraction Final | Authority Switch |
|---:|---|---:|---:|---:|---:|---:|
| `0.2` | `random_sparse` | `0.852250` | `0.022338` | `0.333500` | `0.089750` | `0.103000` |
| `0.2` | `hub_rich` | `0.832250` | `0.059474` | `0.302250` | `0.071000` | `0.089750` |
| `0.2` | `hub_degree_preserving_random` | `0.856500` | `0.033358` | `0.327250` | `0.090750` | `0.109500` |
| `0.3` | `random_sparse` | `0.863750` | `0.028174` | `0.331500` | `0.081750` | `0.091500` |
| `0.3` | `hub_rich` | `0.840500` | `0.052085` | `0.318500` | `0.072000` | `0.099500` |
| `0.3` | `hub_degree_preserving_random` | `0.864500` | `0.046000` | `0.345750` | `0.077250` | `0.109500` |

Multi-aspect read:

- The original `hub_rich` mask again does not beat `random_sparse`.
- `hub_degree_preserving_random` is better than `hub_rich` in both multi-aspect configs.
- `hub_degree_preserving_random` roughly matches `random_sparse` on accuracy and improves authority-switch score, but it does not consistently improve final refraction index.
- Accuracy variance is not consistently reduced; the multi-aspect `hub_degree_preserving_random` rows have higher `acc_std` than `random_sparse`.

This does not support a universal hub prior. It suggests the specific sampled `hub_rich` wiring can be actively unhelpful for the stricter same-token multi-aspect task, while the hub degree distribution is less damaging and sometimes useful.

## 4. Hub Degree-Preserving Deltas

`hub_degree_preserving_random` versus `random_sparse`:

| Experiment | Update | Accuracy Delta | Gain Delta | Refraction Delta | Authority Delta | Acc Std Delta |
|---|---:|---:|---:|---:|---:|---:|
| `latent_refraction` | `0.2` | `+0.015789` | `+0.035338` | `+0.026817` | `+0.029824` | `+0.007736` |
| `latent_refraction` | `0.3` | `+0.010025` | `+0.036090` | `+0.011278` | `+0.000251` | `-0.014980` |
| `multi_aspect_token_refraction` | `0.2` | `+0.004250` | `-0.006250` | `+0.001000` | `+0.006500` | `+0.011020` |
| `multi_aspect_token_refraction` | `0.3` | `+0.000750` | `+0.014250` | `-0.004500` | `+0.018000` | `+0.017826` |

`hub_degree_preserving_random` versus `hub_rich`:

| Experiment | Update | Accuracy Delta | Gain Delta | Refraction Delta | Authority Delta | Acc Std Delta |
|---|---:|---:|---:|---:|---:|---:|
| `latent_refraction` | `0.2` | `+0.006015` | `+0.022306` | `+0.012281` | `+0.021804` | `+0.001432` |
| `latent_refraction` | `0.3` | `-0.022056` | `-0.008271` | `-0.033083` | `-0.047117` | `-0.003558` |
| `multi_aspect_token_refraction` | `0.2` | `+0.024250` | `+0.025000` | `+0.019750` | `+0.019750` | `-0.026116` |
| `multi_aspect_token_refraction` | `0.3` | `+0.024000` | `+0.027250` | `+0.005250` | `+0.010000` | `-0.006085` |

## 5. FlyWire Degree-Preserving Side Control

| Experiment | Update | Topology | Accuracy | Recurrence Gain | Refraction Final | Authority Switch |
|---|---:|---|---:|---:|---:|---:|
| `latent_refraction` | `0.2` | `flywire_sampled` | `0.905765` | `0.318296` | `0.352882` | `0.316541` |
| `latent_refraction` | `0.2` | `flywire_degree_preserving_random` | `0.945113` | `0.368171` | `0.422557` | `0.404010` |
| `latent_refraction` | `0.3` | `flywire_sampled` | `0.932832` | `0.365163` | `0.405764` | `0.382707` |
| `latent_refraction` | `0.3` | `flywire_degree_preserving_random` | `0.955890` | `0.398747` | `0.425564` | `0.407018` |
| `multi_aspect_token_refraction` | `0.2` | `flywire_sampled` | `0.837000` | `0.294250` | `0.050250` | `0.099750` |
| `multi_aspect_token_refraction` | `0.2` | `flywire_degree_preserving_random` | `0.843500` | `0.301750` | `0.079750` | `0.102250` |
| `multi_aspect_token_refraction` | `0.3` | `flywire_sampled` | `0.834250` | `0.307500` | `0.084750` | `0.099750` |
| `multi_aspect_token_refraction` | `0.3` | `flywire_degree_preserving_random` | `0.838000` | `0.290750` | `0.083000` | `0.108000` |

FlyWire side read:

- `flywire_degree_preserving_random` is usually stronger than the raw `flywire_sampled` mask.
- This further weakens the claim that the exact local FlyWire sampled wiring is the useful ingredient in this toy setup.
- The useful signal is still topology statistics / degree structure, not a biology claim.

## 6. Verdict

```json
{
  "degree_distribution_sufficient": "partly_supported",
  "specific_hub_wiring_matters": "partly_supported",
  "hub_prior_universal": false,
  "flywire_exact_wiring_supported": false,
  "reason": "Training degree-preserving hub masks from scratch recovers or improves much of the hub benefit, especially at latent_refraction update=0.2 and in multi_aspect relative to the sampled hub_rich mask. However, hub_rich remains clearly strongest on latent_refraction update=0.3 authority/refraction, so specific wiring can still matter under that setting. The multi_aspect task still does not support a universal hub-rich prior."
}
```

## 7. Interpretation

The cleanest current topology read is:

```text
Hub-heavy degree structure is a real useful prior for some refraction tasks.
The exact sampled hub wiring is not universally useful.
Specific hub wiring can matter in the strongest latent-refraction setting.
```

This narrows the previous result:

```text
old:
  hub_rich topology may help authority switching

updated:
  hub-like degree concentration is often the useful part,
  but specific hub wiring can add benefit in some recurrent regimes,
  and neither is a universal solution for same-token multi-aspect refraction.
```

## 8. Claim Boundary

Do not claim:

- biology,
- FlyWire validation,
- full VRAXION behavior,
- consciousness,
- biological equivalence,
- production validation.

Safe claim:

> In controlled toy settings, degree-concentrated recurrent topology can recover much of the hub-rich benefit for latent refraction and can avoid some failures of a particular sampled hub wiring. The result supports topology-prior sensitivity, not a universal hub rule or a FlyWire-specific claim.

## 9. Next Clean Test

The next topology test should not add semantics. It should vary hub concentration directly:

```text
few strong hubs
medium hubs
many weak hubs
matched edge count
same latent_refraction and multi_aspect grids
```

The key question:

```text
Is there an optimal hub concentration regime for authority switching?
```
