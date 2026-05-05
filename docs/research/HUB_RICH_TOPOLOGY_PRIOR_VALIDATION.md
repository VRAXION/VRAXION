# Hub-Rich Topology Prior Validation

Source summary:

- `target/context-cancellation-probe/hub-rich-validation/hub_rich_validation_summary.json`

Related prior:

- `docs/research/FLYWIRE_TOPOLOGY_PRIOR_ABLATION.md`

## 1. Goal

Validate whether the earlier `hub_rich` win is a real topology-prior signal or a one-grid artifact.

This run does not add a new mechanism. It keeps the existing latent-refraction and multi-aspect token-refraction tasks, changes only the recurrent mask topology, and matches edge budget across topology modes.

## 2. Setup

Experiments:

- `latent_refraction`
- `multi_aspect_token_refraction`

Topology modes:

- `random_sparse`
- `hub_rich`
- `flywire_sampled`
- `flywire_class_sampled`

Update rates:

- `0.2`
- `0.3`

Config:

| Experiment | Hidden | Steps | Epochs | Seeds |
|---|---:|---:|---:|---:|
| `latent_refraction` | `64` | `5` | `200` | `5` |
| `multi_aspect_token_refraction` | `128` | `5` | `220` | `5` |

Random-label control was skipped for this topology validation run to keep the grid compact.

## 3. Main Results

### Latent Refraction

| Update | Topology | Accuracy | Acc Std | Recurrence Gain | Refraction Final | Authority Switch | Randomized Recurrent |
|---:|---|---:|---:|---:|---:|---:|---:|
| `0.2` | `random_sparse` | `0.935589` | `0.018170` | `0.354637` | `0.410025` | `0.390977` | `0.545113` |
| `0.2` | `hub_rich` | `0.945363` | `0.024474` | `0.367669` | `0.424561` | `0.398997` | `0.557143` |
| `0.2` | `flywire_sampled` | `0.905764` | `0.026611` | `0.318296` | `0.352882` | `0.316541` | `0.540602` |
| `0.2` | `flywire_class_sampled` | `0.937594` | `0.016557` | `0.355138` | `0.407519` | `0.385965` | `0.537093` |
| `0.3` | `random_sparse` | `0.939348` | `0.026778` | `0.346366` | `0.412281` | `0.397243` | `0.526566` |
| `0.3` | `hub_rich` | `0.971429` | `0.015356` | `0.390727` | `0.456642` | `0.444612` | `0.514035` |
| `0.3` | `flywire_sampled` | `0.932832` | `0.015319` | `0.365163` | `0.405764` | `0.382707` | `0.532581` |
| `0.3` | `flywire_class_sampled` | `0.929323` | `0.020767` | `0.340852` | `0.391729` | `0.370426` | `0.503008` |

Latent-refraction read:

- `hub_rich` beats `random_sparse` at both update rates.
- The stronger result is at update `0.3`.
- FlyWire-sampled modes do not beat `random_sparse` on the main authority/refraction metrics.

### Multi-Aspect Token Refraction

| Update | Topology | Accuracy | Acc Std | Recurrence Gain | Refraction Final | Authority Switch | Randomized Recurrent |
|---:|---|---:|---:|---:|---:|---:|---:|
| `0.2` | `random_sparse` | `0.852250` | `0.022338` | `0.333500` | `0.089750` | `0.103000` | `0.508000` |
| `0.2` | `hub_rich` | `0.832250` | `0.059474` | `0.302250` | `0.071000` | `0.089750` | `0.516500` |
| `0.2` | `flywire_sampled` | `0.837000` | `0.013266` | `0.294250` | `0.050250` | `0.099750` | `0.512250` |
| `0.2` | `flywire_class_sampled` | `0.829750` | `0.026237` | `0.297000` | `0.078250` | `0.109000` | `0.508250` |
| `0.3` | `random_sparse` | `0.863750` | `0.028174` | `0.331500` | `0.081750` | `0.091500` | `0.513750` |
| `0.3` | `hub_rich` | `0.840500` | `0.052085` | `0.318500` | `0.072000` | `0.099500` | `0.510000` |
| `0.3` | `flywire_sampled` | `0.834250` | `0.017882` | `0.307500` | `0.084750` | `0.099750` | `0.514500` |
| `0.3` | `flywire_class_sampled` | `0.836500` | `0.028443` | `0.304750` | `0.060500` | `0.099750` | `0.518500` |

Multi-aspect read:

- `hub_rich` does not beat `random_sparse` on the stricter multi-aspect token task.
- At update `0.3`, `hub_rich` slightly improves authority-switch score, but loses accuracy, recurrence gain, and final refraction index.
- The earlier hub-rich win is therefore task-specific, not universal.

## 4. Hub vs Random Sparse Deltas

| Experiment | Update | Accuracy Delta | Gain Delta | Refraction Delta | Authority Delta | Acc Std Delta |
|---|---:|---:|---:|---:|---:|---:|
| `latent_refraction` | `0.2` | `+0.009774` | `+0.013032` | `+0.014536` | `+0.008020` | `+0.006305` |
| `latent_refraction` | `0.3` | `+0.032081` | `+0.044361` | `+0.044361` | `+0.047369` | `-0.011422` |
| `multi_aspect_token_refraction` | `0.2` | `-0.020000` | `-0.031250` | `-0.018750` | `-0.013250` | `+0.037135` |
| `multi_aspect_token_refraction` | `0.3` | `-0.023250` | `-0.013000` | `-0.009750` | `+0.008000` | `+0.023912` |

## 5. Degree-Preserving Shuffle

This diagnostic takes the trained `hub_rich` model and shuffles its recurrent mask while preserving the in/out degree distribution.

Important caveat:

This is a post-training shuffle. It tests whether the trained model relies on its specific hub wiring. It does not fully test whether a freshly trained degree-preserving random hub mask would recover the same gain.

| Experiment | Update | Accuracy Drop | Gain Drop | Refraction Drop | Authority Drop |
|---|---:|---:|---:|---:|---:|
| `latent_refraction` | `0.2` | `0.295990` | `0.295990` | `0.358145` | `0.333584` |
| `latent_refraction` | `0.3` | `0.322556` | `0.322556` | `0.392481` | `0.386717` |
| `multi_aspect_token_refraction` | `0.2` | `0.187000` | `0.187000` | `0.072000` | `0.066500` |
| `multi_aspect_token_refraction` | `0.3` | `0.240750` | `0.240750` | `0.081250` | `0.091000` |

Read:

- The trained hub-rich models depend strongly on their learned hub wiring.
- Shuffling edges after training destroys much of the recurrent advantage.

## 6. Hub Ablation

Top 10% hub-node ablation:

| Experiment | Update | Hub Acc Drop | Random Acc Drop | Hub Authority Drop | Random Authority Drop | Hub-Minus-Random Acc Drop | Hub-Minus-Random Authority Drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| `latent_refraction` | `0.2` | `0.374687` | `0.179699` | `0.401754` | `0.234586` | `0.194987` | `0.167168` |
| `latent_refraction` | `0.3` | `0.385213` | `0.187719` | `0.471178` | `0.267920` | `0.197494` | `0.203258` |
| `multi_aspect_token_refraction` | `0.2` | `0.289500` | `0.089750` | `0.144250` | `0.027250` | `0.199750` | `0.117000` |
| `multi_aspect_token_refraction` | `0.3` | `0.324250` | `0.094000` | `0.169750` | `0.023250` | `0.230250` | `0.146500` |

Read:

- Hub nodes are load-bearing inside trained `hub_rich` models.
- Top-hub ablation hurts much more than same-count random node ablation in both tasks.
- This is true even in multi-aspect runs where `hub_rich` does not beat `random_sparse` overall.

## 7. Verdict

```json
{
  "hub_rich_topology_prior": "mixed_task_specific_positive_for_latent_refraction",
  "hub_nodes_load_bearing": true,
  "flywire_topology_prior": "not_supported_as_metric_winner",
  "reason": "hub_rich wins both latent_refraction update rates on the main metrics, but does not beat random_sparse on multi_aspect_token_refraction. Hub node ablation hurts much more than random node ablation inside hub_rich models, so hubs are load-bearing once used, but the prior is not universally better across tasks."
}
```

## 8. Interpretation

The earlier hub-rich result partially validates:

```text
hub-rich topology can help latent-refraction authority switching
```

but does not validate the stronger claim:

```text
hub-rich topology is generally better for all prism / authority-switch tasks
```

The current best read is:

```text
Frame pointer + recurrent attractor + hub-rich integration topology
can be beneficial for group-level latent refraction,
but same-token multi-aspect refraction remains better served by random sparse masks in this grid.
```

FlyWire sampled topology remains negative as a metric winner in this validation run.

## 9. Claim Boundary

Do not claim:

- biology,
- FlyWire validation,
- full VRAXION behavior,
- consciousness,
- biological equivalence,
- production validation.

Safe claim:

> In controlled toy settings, hub-rich recurrent topology is load-bearing and can improve latent-refraction authority switching, but the advantage is task-specific and does not yet generalize to the stricter multi-aspect token-refraction task.

## 10. Next Clean Test

The best next topology test would be:

```text
train degree-preserving shuffled hub masks from scratch
```

This would separate:

```text
degree distribution as prior
```

from:

```text
specific sampled hub wiring pattern
```
