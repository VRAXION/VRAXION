# Authority Circuit Dissection

## Goal

Locate load-bearing recurrent nodes, edge groups, and motif candidates responsible for decision-authority switching in existing toy probes.

This report uses budgeted ablations. It does not add semantics, architecture, FlyWire biology, or wave/pointer mechanisms.

## Multi-Seed Validation Summary

| Experiment | Topology | Runs | Baseline Acc | Authority | Refraction | Top Node Drop | Top 5% Saliency Drop | Random 5% Drop | Hub 10% Drop | Random 10% Drop | Top 10% Retention | Random 10% Retention |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `latent_refraction` | `random_sparse` | `3` | `0.950710` | `0.428989` | `0.446533` | `0.216792` | `0.259816` | `0.134224` | `0.268588` | `0.187970` | `-0.084658` | `-0.065671` |
| `latent_refraction` | `hub_rich` | `3` | `0.971596` | `0.461988` | `0.470760` | `0.365497` | `0.427736` | `0.120440` | `0.480785` | `0.201754` | `-0.040789` | `-0.068980` |
| `latent_refraction` | `hub_degree_preserving_random` | `3` | `0.987886` | `0.502089` | `0.505848` | `0.506266` | `0.497076` | `0.146895` | `0.534670` | `0.235728` | `-0.045604` | `-0.027325` |

## Run Configuration

```json
{
  "experiments": [
    "latent_refraction"
  ],
  "seeds": 3,
  "latent_hidden": 64,
  "multi_hidden": 128,
  "steps": 5,
  "latent_epochs": 200,
  "multi_epochs": 220,
  "train_size": 1600,
  "test_size": 800,
  "topology_modes": [
    "random_sparse",
    "hub_rich",
    "hub_degree_preserving_random"
  ],
  "max_node_ablation": 24,
  "random_node_controls": 3,
  "edge_group_fraction": 0.05,
  "minimal_fractions": [
    0.05,
    0.1,
    0.2,
    0.4
  ],
  "smoke": false
}
```

## Verdict

```json
{
  "supports_localized_authority_circuit": "true",
  "supports_hub_load_bearing_authority": "true",
  "supports_degree_distribution_as_circuit_prior": "true",
  "supports_specific_hub_wiring": "unclear",
  "supports_reciprocal_motif_importance": "unclear",
  "supports_minimal_circuit_extractability": "unclear",
  "supports_frame_specific_circuit_candidates": "true",
  "notes_on_task_specificity": "This is a budgeted dissection. Treat positives as candidate circuit localization, not proof of a unique circuit, until rerun with larger node/edge budgets."
}
```

## Interpretation

- Node and saliency-group ablations indicate that authority switching is not fully homogeneous; small recurrent subsets can be disproportionately load-bearing.
- Hub ablations hurt authority/refraction more than same-count random node ablations in the validated configs.
- The degree-preserving hub-random control performs at least as well as hub-rich here, so the useful prior currently looks more like hub/degree concentration than a specific hand-built hub wiring pattern.
- Specific hub wiring remains unproven; current evidence does not require the original hub-rich edge pattern.
- Minimal circuit extraction remains unclear: keeping only top-K nodes/edges does not yet preserve authority/refraction cleanly enough to call this a compact extracted circuit.
- Reciprocal motifs are not yet load-bearing beyond random edge controls.

## Hub-Rich vs Degree-Preserving Hub

| Experiment | Random Authority | Hub-Rich Authority | Degree-Preserving Authority | Random Refraction | Hub-Rich Refraction | Degree-Preserving Refraction |
|---|---:|---:|---:|---:|---:|---:|
| `latent_refraction` | `0.428989` | `0.461988` | `0.502089` | `0.446533` | `0.470760` | `0.505848` |

## Node Ablation Top 10

### `latent_refraction` seed `0` topology `random_sparse`

| Node | Accuracy Drop | Authority Drop | Refraction Drop | Active Influence Drop |
|---:|---:|---:|---:|---:|
| `24` | `0.102757` | `0.209273` | `0.211779` | `0.120301` |
| `35` | `0.087719` | `0.199248` | `0.213033` | `0.105263` |
| `1` | `0.046366` | `0.126566` | `0.131579` | `0.085213` |
| `39` | `0.051378` | `0.121554` | `0.127820` | `0.085213` |
| `15` | `0.055138` | `0.114035` | `0.109023` | `0.061404` |
| `4` | `0.052632` | `0.101504` | `0.085213` | `0.051378` |
| `50` | `0.037594` | `0.100251` | `0.101504` | `0.063910` |
| `58` | `0.036341` | `0.096491` | `0.081454` | `0.053885` |
| `36` | `0.026316` | `0.073935` | `0.067669` | `0.047619` |
| `14` | `0.010025` | `0.071429` | `0.056391` | `0.040100` |

### `latent_refraction` seed `1` topology `random_sparse`

| Node | Accuracy Drop | Authority Drop | Refraction Drop | Active Influence Drop |
|---:|---:|---:|---:|---:|
| `21` | `0.110276` | `0.180451` | `0.174185` | `0.093985` |
| `20` | `0.130326` | `0.159148` | `0.176692` | `0.082707` |
| `18` | `0.070175` | `0.154135` | `0.137845` | `0.078947` |
| `51` | `0.082707` | `0.117794` | `0.111529` | `0.063910` |
| `63` | `0.061404` | `0.110276` | `0.132832` | `0.066416` |
| `37` | `0.026316` | `0.097744` | `0.095238` | `0.071429` |
| `44` | `0.026316` | `0.095238` | `0.083960` | `0.061404` |
| `15` | `0.037594` | `0.092732` | `0.083960` | `0.037594` |
| `2` | `0.037594` | `0.091479` | `0.088972` | `0.035088` |
| `7` | `0.022556` | `0.085213` | `0.081454` | `0.063910` |

### `latent_refraction` seed `2` topology `random_sparse`

| Node | Accuracy Drop | Authority Drop | Refraction Drop | Active Influence Drop |
|---:|---:|---:|---:|---:|
| `35` | `0.162907` | `0.260652` | `0.218045` | `0.107769` |
| `56` | `0.066416` | `0.191729` | `0.164160` | `0.062657` |
| `55` | `0.066416` | `0.154135` | `0.145363` | `0.038847` |
| `26` | `0.100251` | `0.139098` | `0.126566` | `0.051378` |
| `5` | `0.031328` | `0.112782` | `0.105263` | `0.065163` |
| `48` | `0.035088` | `0.110276` | `0.104010` | `0.051378` |
| `27` | `0.057644` | `0.088972` | `0.102757` | `0.047619` |
| `34` | `0.035088` | `0.086466` | `0.085213` | `0.028822` |
| `36` | `0.038847` | `0.085213` | `0.076441` | `0.012531` |
| `30` | `0.041353` | `0.080201` | `0.061404` | `0.027569` |

### `latent_refraction` seed `0` topology `hub_rich`

| Node | Accuracy Drop | Authority Drop | Refraction Drop | Active Influence Drop |
|---:|---:|---:|---:|---:|
| `17` | `0.265664` | `0.395990` | `0.486216` | `0.245614` |
| `33` | `0.213033` | `0.334586` | `0.310777` | `0.246867` |
| `24` | `0.136591` | `0.248120` | `0.206767` | `0.120301` |
| `20` | `0.098997` | `0.206767` | `0.197995` | `0.114035` |
| `43` | `0.114035` | `0.206767` | `0.186717` | `0.098997` |
| `4` | `0.196742` | `0.155388` | `0.161654` | `0.129073` |
| `47` | `0.057644` | `0.145363` | `0.130326` | `0.066416` |
| `5` | `0.062657` | `0.122807` | `0.117794` | `0.067669` |
| `6` | `0.085213` | `0.115288` | `0.101504` | `0.048872` |
| `55` | `0.025063` | `0.100251` | `0.100251` | `0.056391` |

### `latent_refraction` seed `1` topology `hub_rich`

| Node | Accuracy Drop | Authority Drop | Refraction Drop | Active Influence Drop |
|---:|---:|---:|---:|---:|
| `27` | `0.298246` | `0.394737` | `0.383459` | `0.236842` |
| `11` | `0.194236` | `0.335840` | `0.357143` | `0.203008` |
| `38` | `0.256892` | `0.334586` | `0.309524` | `0.169173` |
| `44` | `0.068922` | `0.177945` | `0.144110` | `0.051378` |
| `32` | `0.145363` | `0.171679` | `0.167920` | `0.076441` |
| `6` | `0.038847` | `0.109023` | `0.093985` | `0.043860` |
| `53` | `0.077694` | `0.109023` | `0.119048` | `0.031328` |
| `8` | `0.043860` | `0.101504` | `0.088972` | `0.037594` |
| `45` | `0.055138` | `0.092732` | `0.071429` | `0.020050` |
| `1` | `0.035088` | `0.087719` | `0.083960` | `0.040100` |

### `latent_refraction` seed `2` topology `hub_rich`

| Node | Accuracy Drop | Authority Drop | Refraction Drop | Active Influence Drop |
|---:|---:|---:|---:|---:|
| `16` | `0.320802` | `0.305764` | `0.317043` | `0.197995` |
| `46` | `0.200501` | `0.289474` | `0.334586` | `0.139098` |
| `32` | `0.181704` | `0.248120` | `0.253133` | `0.144110` |
| `42` | `0.182957` | `0.142857` | `0.144110` | `0.091479` |
| `37` | `0.078947` | `0.117794` | `0.111529` | `0.057644` |
| `26` | `0.073935` | `0.112782` | `0.102757` | `0.051378` |
| `62` | `0.043860` | `0.112782` | `0.105263` | `0.052632` |
| `43` | `0.061404` | `0.081454` | `0.078947` | `0.045113` |
| `20` | `0.051378` | `0.072682` | `0.055138` | `0.017544` |
| `6` | `0.008772` | `0.047619` | `0.050125` | `0.037594` |

### `latent_refraction` seed `0` topology `hub_degree_preserving_random`

| Node | Accuracy Drop | Authority Drop | Refraction Drop | Active Influence Drop |
|---:|---:|---:|---:|---:|
| `17` | `0.273183` | `0.458647` | `0.523810` | `0.256892` |
| `25` | `0.307018` | `0.419799` | `0.403509` | `0.240602` |
| `24` | `0.327068` | `0.407268` | `0.383459` | `0.208020` |
| `32` | `0.155388` | `0.269424` | `0.253133` | `0.136591` |
| `33` | `0.115288` | `0.251880` | `0.228070` | `0.131579` |
| `9` | `0.093985` | `0.220551` | `0.199248` | `0.156642` |
| `11` | `0.055138` | `0.192982` | `0.160401` | `0.097744` |
| `20` | `0.066416` | `0.144110` | `0.136591` | `0.068922` |
| `55` | `0.052632` | `0.139098` | `0.119048` | `0.057644` |
| `5` | `0.042607` | `0.119048` | `0.109023` | `0.067669` |

### `latent_refraction` seed `1` topology `hub_degree_preserving_random`

| Node | Accuracy Drop | Authority Drop | Refraction Drop | Active Influence Drop |
|---:|---:|---:|---:|---:|
| `11` | `0.409774` | `0.644110` | `0.558897` | `0.388471` |
| `8` | `0.219298` | `0.263158` | `0.278195` | `0.127820` |
| `27` | `0.129073` | `0.258145` | `0.240602` | `0.139098` |
| `53` | `0.228070` | `0.248120` | `0.258145` | `0.082707` |
| `32` | `0.081454` | `0.179198` | `0.170426` | `0.048872` |
| `1` | `0.047619` | `0.141604` | `0.112782` | `0.053885` |
| `6` | `0.048872` | `0.129073` | `0.101504` | `0.050125` |
| `37` | `0.035088` | `0.126566` | `0.107769` | `0.062657` |
| `44` | `0.038847` | `0.110276` | `0.101504` | `0.050125` |
| `19` | `0.052632` | `0.104010` | `0.096491` | `0.031328` |

### `latent_refraction` seed `2` topology `hub_degree_preserving_random`

| Node | Accuracy Drop | Authority Drop | Refraction Drop | Active Influence Drop |
|---:|---:|---:|---:|---:|
| `16` | `0.354637` | `0.416040` | `0.394737` | `0.235589` |
| `46` | `0.249373` | `0.353383` | `0.398496` | `0.187970` |
| `37` | `0.184211` | `0.279449` | `0.281955` | `0.121554` |
| `42` | `0.127820` | `0.157895` | `0.122807` | `0.062657` |
| `33` | `0.047619` | `0.130326` | `0.100251` | `0.050125` |
| `36` | `0.055138` | `0.109023` | `0.092732` | `0.033835` |
| `51` | `0.056391` | `0.083960` | `0.088972` | `0.023810` |
| `20` | `0.025063` | `0.077694` | `0.070175` | `0.037594` |
| `26` | `0.013784` | `0.077694` | `0.067669` | `0.041353` |
| `32` | `0.051378` | `0.076441` | `0.068922` | `0.008772` |

## Saliency Group Ablation

### `latent_refraction` seed `0` topology `random_sparse`

| Node Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.276942` | `0.170008` | `0.307018` | `0.176274` |
| `top_10pct` | `6` | `0.398496` | `0.210109` | `0.412281` | `0.182540` |
| `top_20pct` | `13` | `0.443609` | `0.334586` | `0.447368` | `0.332080` |

### `latent_refraction` seed `1` topology `random_sparse`

| Node Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.213033` | `0.120718` | `0.248120` | `0.124896` |
| `top_10pct` | `6` | `0.404762` | `0.193400` | `0.413534` | `0.232665` |
| `top_20pct` | `13` | `0.464912` | `0.319131` | `0.487469` | `0.340017` |

### `latent_refraction` seed `2` topology `random_sparse`

| Node Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.289474` | `0.111947` | `0.288221` | `0.100668` |
| `top_10pct` | `6` | `0.318296` | `0.247703` | `0.313283` | `0.251880` |
| `top_20pct` | `13` | `0.333333` | `0.321220` | `0.359649` | `0.329574` |

### `latent_refraction` seed `0` topology `hub_rich`

| Node Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.399749` | `0.159566` | `0.424812` | `0.169591` |
| `top_10pct` | `6` | `0.413534` | `0.271930` | `0.402256` | `0.266917` |
| `top_20pct` | `13` | `0.426065` | `0.322055` | `0.433584` | `0.337928` |

### `latent_refraction` seed `1` topology `hub_rich`

| Node Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.508772` | `0.122807` | `0.510025` | `0.118630` |
| `top_10pct` | `6` | `0.532581` | `0.301170` | `0.505013` | `0.289056` |
| `top_20pct` | `13` | `0.463659` | `0.345865` | `0.454887` | `0.356307` |

### `latent_refraction` seed `2` topology `hub_rich`

| Node Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.374687` | `0.078947` | `0.427318` | `0.065998` |
| `top_10pct` | `6` | `0.483709` | `0.160401` | `0.457393` | `0.144946` |
| `top_20pct` | `13` | `0.526316` | `0.274854` | `0.488722` | `0.278195` |

### `latent_refraction` seed `0` topology `hub_degree_preserving_random`

| Node Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.511278` | `0.180033` | `0.522556` | `0.177109` |
| `top_10pct` | `6` | `0.518797` | `0.268588` | `0.538847` | `0.276107` |
| `top_20pct` | `13` | `0.528822` | `0.475355` | `0.523810` | `0.491646` |

### `latent_refraction` seed `1` topology `hub_degree_preserving_random`

| Node Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.547619` | `0.159983` | `0.503759` | `0.147452` |
| `top_10pct` | `6` | `0.550125` | `0.242272` | `0.580201` | `0.241437` |
| `top_20pct` | `13` | `0.598997` | `0.458229` | `0.561404` | `0.419382` |

### `latent_refraction` seed `2` topology `hub_degree_preserving_random`

| Node Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.432331` | `0.100668` | `0.453634` | `0.083542` |
| `top_10pct` | `6` | `0.507519` | `0.152464` | `0.483709` | `0.152464` |
| `top_20pct` | `13` | `0.484962` | `0.385129` | `0.505013` | `0.393901` |

## Hub Ablation Vs Random Nodes

### `latent_refraction` seed `0` topology `random_sparse`

| Hub Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.180451` | `0.182957` | `0.185464` | `0.200501` |
| `top_10pct` | `6` | `0.322055` | `0.220969` | `0.340852` | `0.227652` |
| `top_20pct` | `13` | `0.327068` | `0.348789` | `0.374687` | `0.372598` |

### `latent_refraction` seed `1` topology `random_sparse`

| Hub Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.185464` | `0.096909` | `0.218045` | `0.097744` |
| `top_10pct` | `6` | `0.240602` | `0.222640` | `0.294486` | `0.242272` |
| `top_20pct` | `13` | `0.428571` | `0.317878` | `0.437343` | `0.353383` |

### `latent_refraction` seed `2` topology `random_sparse`

| Hub Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.194236` | `0.154135` | `0.164160` | `0.153718` |
| `top_10pct` | `6` | `0.243108` | `0.120301` | `0.224311` | `0.123225` |
| `top_20pct` | `13` | `0.402256` | `0.311195` | `0.384712` | `0.326232` |

### `latent_refraction` seed `0` topology `hub_rich`

| Hub Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.368421` | `0.160819` | `0.404762` | `0.147870` |
| `top_10pct` | `6` | `0.452381` | `0.213868` | `0.466165` | `0.197577` |
| `top_20pct` | `13` | `0.508772` | `0.371763` | `0.486216` | `0.405180` |

### `latent_refraction` seed `1` topology `hub_rich`

| Hub Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.489975` | `0.197995` | `0.469925` | `0.191729` |
| `top_10pct` | `6` | `0.515038` | `0.215121` | `0.508772` | `0.211779` |
| `top_20pct` | `13` | `0.526316` | `0.412698` | `0.536341` | `0.417711` |

### `latent_refraction` seed `2` topology `hub_rich`

| Hub Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.402256` | `0.070175` | `0.404762` | `0.079365` |
| `top_10pct` | `6` | `0.474937` | `0.176274` | `0.439850` | `0.170844` |
| `top_20pct` | `13` | `0.479950` | `0.416040` | `0.500000` | `0.402256` |

### `latent_refraction` seed `0` topology `hub_degree_preserving_random`

| Hub Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.555138` | `0.291980` | `0.612782` | `0.269006` |
| `top_10pct` | `6` | `0.508772` | `0.237678` | `0.531328` | `0.220134` |
| `top_20pct` | `13` | `0.528822` | `0.435255` | `0.487469` | `0.436508` |

### `latent_refraction` seed `1` topology `hub_degree_preserving_random`

| Hub Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.627820` | `0.152047` | `0.592732` | `0.134921` |
| `top_10pct` | `6` | `0.602757` | `0.236842` | `0.590226` | `0.227235` |
| `top_20pct` | `13` | `0.601504` | `0.490810` | `0.585213` | `0.488304` |

### `latent_refraction` seed `2` topology `hub_degree_preserving_random`

| Hub Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |
|---|---:|---:|---:|---:|---:|
| `top_5pct` | `3` | `0.508772` | `0.103175` | `0.512531` | `0.109858` |
| `top_10pct` | `6` | `0.492481` | `0.232665` | `0.473684` | `0.218881` |
| `top_20pct` | `13` | `0.575188` | `0.370927` | `0.570175` | `0.407686` |

## Edge Group Ablation

### `latent_refraction` seed `0` topology `random_sparse`

| Edge Group | Count | Accuracy Drop | Authority Drop | Refraction Drop |
|---|---:|---:|---:|---:|
| `top_proxy_edges` | `25` | `0.259398` | `0.339599` | `0.333333` |
| `hub_incoming_edges` | `25` | `0.026316` | `0.077694` | `0.072682` |
| `hub_outgoing_edges` | `25` | `0.030075` | `0.088972` | `0.088972` |
| `reciprocal_pair_edges` | `25` | `0.085213` | `0.170426` | `0.196742` |
| `random_same_count_edges` | `25` | `0.058897` | `0.129073` | `0.110276` |

### `latent_refraction` seed `1` topology `random_sparse`

| Edge Group | Count | Accuracy Drop | Authority Drop | Refraction Drop |
|---|---:|---:|---:|---:|
| `top_proxy_edges` | `24` | `0.246867` | `0.335840` | `0.350877` |
| `hub_incoming_edges` | `24` | `0.037594` | `0.110276` | `0.100251` |
| `hub_outgoing_edges` | `24` | `0.086466` | `0.126566` | `0.130326` |
| `reciprocal_pair_edges` | `24` | `0.085213` | `0.162907` | `0.160401` |
| `random_same_count_edges` | `24` | `0.026316` | `0.083960` | `0.085213` |

### `latent_refraction` seed `2` topology `random_sparse`

| Edge Group | Count | Accuracy Drop | Authority Drop | Refraction Drop |
|---|---:|---:|---:|---:|
| `top_proxy_edges` | `26` | `0.334586` | `0.309524` | `0.358396` |
| `hub_incoming_edges` | `26` | `0.022556` | `0.067669` | `0.063910` |
| `hub_outgoing_edges` | `26` | `0.090226` | `0.150376` | `0.129073` |
| `reciprocal_pair_edges` | `26` | `0.028822` | `0.078947` | `0.075188` |
| `random_same_count_edges` | `26` | `0.047619` | `0.087719` | `0.093985` |

### `latent_refraction` seed `0` topology `hub_rich`

| Edge Group | Count | Accuracy Drop | Authority Drop | Refraction Drop |
|---|---:|---:|---:|---:|
| `top_proxy_edges` | `25` | `0.159148` | `0.271930` | `0.246867` |
| `hub_incoming_edges` | `25` | `0.136591` | `0.191729` | `0.180451` |
| `hub_outgoing_edges` | `25` | `0.045113` | `0.131579` | `0.100251` |
| `reciprocal_pair_edges` | `25` | `0.052632` | `0.125313` | `0.093985` |
| `random_same_count_edges` | `25` | `0.126566` | `0.220551` | `0.214286` |

### `latent_refraction` seed `1` topology `hub_rich`

| Edge Group | Count | Accuracy Drop | Authority Drop | Refraction Drop |
|---|---:|---:|---:|---:|
| `top_proxy_edges` | `24` | `0.334586` | `0.456140` | `0.473684` |
| `hub_incoming_edges` | `24` | `0.055138` | `0.151629` | `0.131579` |
| `hub_outgoing_edges` | `24` | `0.119048` | `0.246867` | `0.231830` |
| `reciprocal_pair_edges` | `24` | `0.134085` | `0.201754` | `0.191729` |
| `random_same_count_edges` | `24` | `0.035088` | `0.104010` | `0.101504` |

### `latent_refraction` seed `2` topology `hub_rich`

| Edge Group | Count | Accuracy Drop | Authority Drop | Refraction Drop |
|---|---:|---:|---:|---:|
| `top_proxy_edges` | `26` | `0.199248` | `0.258145` | `0.284461` |
| `hub_incoming_edges` | `26` | `0.208020` | `0.271930` | `0.244361` |
| `hub_outgoing_edges` | `26` | `0.028822` | `0.086466` | `0.058897` |
| `reciprocal_pair_edges` | `26` | `0.030075` | `0.077694` | `0.060150` |
| `random_same_count_edges` | `26` | `0.026316` | `0.058897` | `0.051378` |

### `latent_refraction` seed `0` topology `hub_degree_preserving_random`

| Edge Group | Count | Accuracy Drop | Authority Drop | Refraction Drop |
|---|---:|---:|---:|---:|
| `top_proxy_edges` | `25` | `0.345865` | `0.448622` | `0.444862` |
| `hub_incoming_edges` | `25` | `0.045113` | `0.152882` | `0.127820` |
| `hub_outgoing_edges` | `25` | `0.030075` | `0.097744` | `0.085213` |
| `reciprocal_pair_edges` | `25` | `0.022556` | `0.107769` | `0.101504` |
| `random_same_count_edges` | `25` | `0.107769` | `0.273183` | `0.230576` |

### `latent_refraction` seed `1` topology `hub_degree_preserving_random`

| Edge Group | Count | Accuracy Drop | Authority Drop | Refraction Drop |
|---|---:|---:|---:|---:|
| `top_proxy_edges` | `24` | `0.260652` | `0.324561` | `0.299499` |
| `hub_incoming_edges` | `24` | `0.129073` | `0.141604` | `0.135338` |
| `hub_outgoing_edges` | `24` | `0.137845` | `0.180451` | `0.170426` |
| `reciprocal_pair_edges` | `24` | `0.105263` | `0.179198` | `0.162907` |
| `random_same_count_edges` | `24` | `0.017544` | `0.096491` | `0.092732` |

### `latent_refraction` seed `2` topology `hub_degree_preserving_random`

| Edge Group | Count | Accuracy Drop | Authority Drop | Refraction Drop |
|---|---:|---:|---:|---:|
| `top_proxy_edges` | `26` | `0.226817` | `0.335840` | `0.332080` |
| `hub_incoming_edges` | `26` | `0.280702` | `0.387218` | `0.398496` |
| `hub_outgoing_edges` | `26` | `0.051378` | `0.110276` | `0.107769` |
| `reciprocal_pair_edges` | `26` | `0.047619` | `0.096491` | `0.104010` |
| `random_same_count_edges` | `26` | `0.028822` | `0.107769` | `0.100251` |

## Minimal Circuit Survival

### `latent_refraction` seed `0` topology `random_sparse`

| Retained Budget | Top Nodes Authority Retention | Random Nodes Authority Retention | Top Edges Authority Retention | Random Edges Authority Retention |
|---|---:|---:|---:|---:|
| `5pct` | `-0.038690` | `-0.077381` | `0.056548` | `-0.020833` |
| `10pct` | `-0.154762` | `-0.032738` | `-0.095238` | `-0.101190` |
| `20pct` | `-0.130952` | `0.000000` | `0.110119` | `-0.059524` |
| `40pct` | `0.002976` | `0.008929` | `0.556548` | `0.089286` |

### `latent_refraction` seed `1` topology `random_sparse`

| Retained Budget | Top Nodes Authority Retention | Random Nodes Authority Retention | Top Edges Authority Retention | Random Edges Authority Retention |
|---|---:|---:|---:|---:|
| `5pct` | `-0.102719` | `-0.145015` | `-0.120846` | `0.090634` |
| `10pct` | `-0.093656` | `-0.039275` | `-0.033233` | `0.063444` |
| `20pct` | `-0.069486` | `0.009063` | `0.160121` | `0.060423` |
| `40pct` | `-0.253776` | `0.021148` | `0.537764` | `0.009063` |

### `latent_refraction` seed `2` topology `random_sparse`

| Retained Budget | Top Nodes Authority Retention | Random Nodes Authority Retention | Top Edges Authority Retention | Random Edges Authority Retention |
|---|---:|---:|---:|---:|
| `5pct` | `-0.066667` | `-0.002778` | `0.061111` | `-0.108333` |
| `10pct` | `-0.005556` | `-0.125000` | `0.022222` | `-0.013889` |
| `20pct` | `0.052778` | `-0.050000` | `0.169444` | `-0.008333` |
| `40pct` | `0.025000` | `-0.030556` | `0.705556` | `-0.011111` |

### `latent_refraction` seed `0` topology `hub_rich`

| Retained Budget | Top Nodes Authority Retention | Random Nodes Authority Retention | Top Edges Authority Retention | Random Edges Authority Retention |
|---|---:|---:|---:|---:|
| `5pct` | `-0.019608` | `-0.095238` | `-0.098039` | `-0.100840` |
| `10pct` | `-0.067227` | `0.036415` | `0.044818` | `-0.137255` |
| `20pct` | `-0.067227` | `-0.086835` | `0.126050` | `-0.033613` |
| `40pct` | `0.100840` | `-0.142857` | `0.492997` | `0.019608` |

### `latent_refraction` seed `1` topology `hub_rich`

| Retained Budget | Top Nodes Authority Retention | Random Nodes Authority Retention | Top Edges Authority Retention | Random Edges Authority Retention |
|---|---:|---:|---:|---:|
| `5pct` | `-0.135897` | `-0.033333` | `-0.241026` | `0.000000` |
| `10pct` | `0.025641` | `-0.020513` | `-0.138462` | `-0.069231` |
| `20pct` | `-0.030769` | `-0.053846` | `0.048718` | `0.048718` |
| `40pct` | `0.035897` | `-0.087179` | `0.415385` | `-0.076923` |

### `latent_refraction` seed `2` topology `hub_rich`

| Retained Budget | Top Nodes Authority Retention | Random Nodes Authority Retention | Top Edges Authority Retention | Random Edges Authority Retention |
|---|---:|---:|---:|---:|
| `5pct` | `-0.091922` | `-0.119777` | `-0.167131` | `-0.317549` |
| `10pct` | `-0.080780` | `-0.222841` | `0.044568` | `-0.041783` |
| `20pct` | `-0.058496` | `-0.267409` | `0.114206` | `0.025070` |
| `40pct` | `0.002786` | `-0.114206` | `0.629526` | `0.025070` |

### `latent_refraction` seed `0` topology `hub_degree_preserving_random`

| Retained Budget | Top Nodes Authority Retention | Random Nodes Authority Retention | Top Edges Authority Retention | Random Edges Authority Retention |
|---|---:|---:|---:|---:|
| `5pct` | `0.051095` | `-0.051095` | `0.004866` | `-0.036496` |
| `10pct` | `-0.019465` | `0.029197` | `-0.029197` | `0.136253` |
| `20pct` | `-0.029197` | `-0.138686` | `0.119221` | `-0.055961` |
| `40pct` | `-0.075426` | `-0.119221` | `0.360097` | `0.063260` |

### `latent_refraction` seed `1` topology `hub_degree_preserving_random`

| Retained Budget | Top Nodes Authority Retention | Random Nodes Authority Retention | Top Edges Authority Retention | Random Edges Authority Retention |
|---|---:|---:|---:|---:|
| `5pct` | `-0.025063` | `-0.040100` | `-0.070175` | `-0.107769` |
| `10pct` | `0.000000` | `-0.060150` | `-0.060150` | `-0.127820` |
| `20pct` | `-0.055138` | `-0.070175` | `-0.015038` | `-0.177945` |
| `40pct` | `0.072682` | `-0.258145` | `0.253133` | `0.025063` |

### `latent_refraction` seed `2` topology `hub_degree_preserving_random`

| Retained Budget | Top Nodes Authority Retention | Random Nodes Authority Retention | Top Edges Authority Retention | Random Edges Authority Retention |
|---|---:|---:|---:|---:|
| `5pct` | `-0.043367` | `-0.183673` | `0.107143` | `-0.068878` |
| `10pct` | `-0.117347` | `-0.051020` | `0.196429` | `-0.142857` |
| `20pct` | `-0.165816` | `0.005102` | `0.165816` | `0.020408` |
| `40pct` | `0.109694` | `-0.140306` | `0.591837` | `-0.020408` |

## Frame Specificity

### `latent_refraction` seed `0` topology `random_sparse`

```json
{
  "by_frame": {
    "danger_frame": {
      "top_node": 24,
      "target_refraction_drop": 0.25188,
      "mean_non_target_refraction_drop": 0.191729,
      "frame_specificity_score": 0.06015
    },
    "environment_frame": {
      "top_node": 35,
      "target_refraction_drop": 0.233083,
      "mean_non_target_refraction_drop": 0.203008,
      "frame_specificity_score": 0.030075
    },
    "visibility_frame": {
      "top_node": 24,
      "target_refraction_drop": 0.203008,
      "mean_non_target_refraction_drop": 0.216165,
      "frame_specificity_score": -0.013158
    }
  },
  "mean_frame_specificity_score": 0.025689
}
```

### `latent_refraction` seed `1` topology `random_sparse`

```json
{
  "by_frame": {
    "danger_frame": {
      "top_node": 21,
      "target_refraction_drop": 0.255639,
      "mean_non_target_refraction_drop": 0.133459,
      "frame_specificity_score": 0.12218
    },
    "environment_frame": {
      "top_node": 18,
      "target_refraction_drop": 0.289474,
      "mean_non_target_refraction_drop": 0.06203,
      "frame_specificity_score": 0.227444
    },
    "visibility_frame": {
      "top_node": 20,
      "target_refraction_drop": 0.180451,
      "mean_non_target_refraction_drop": 0.174812,
      "frame_specificity_score": 0.005639
    }
  },
  "mean_frame_specificity_score": 0.118421
}
```

### `latent_refraction` seed `2` topology `random_sparse`

```json
{
  "by_frame": {
    "danger_frame": {
      "top_node": 35,
      "target_refraction_drop": 0.281955,
      "mean_non_target_refraction_drop": 0.18609,
      "frame_specificity_score": 0.095865
    },
    "environment_frame": {
      "top_node": 35,
      "target_refraction_drop": 0.308271,
      "mean_non_target_refraction_drop": 0.172932,
      "frame_specificity_score": 0.135338
    },
    "visibility_frame": {
      "top_node": 56,
      "target_refraction_drop": 0.180451,
      "mean_non_target_refraction_drop": 0.156015,
      "frame_specificity_score": 0.024436
    }
  },
  "mean_frame_specificity_score": 0.085213
}
```

### `latent_refraction` seed `0` topology `hub_rich`

```json
{
  "by_frame": {
    "danger_frame": {
      "top_node": 17,
      "target_refraction_drop": 0.657895,
      "mean_non_target_refraction_drop": 0.400376,
      "frame_specificity_score": 0.257519
    },
    "environment_frame": {
      "top_node": 17,
      "target_refraction_drop": 0.56015,
      "mean_non_target_refraction_drop": 0.449248,
      "frame_specificity_score": 0.110902
    },
    "visibility_frame": {
      "top_node": 17,
      "target_refraction_drop": 0.240602,
      "mean_non_target_refraction_drop": 0.609023,
      "frame_specificity_score": -0.368421
    }
  },
  "mean_frame_specificity_score": 0.0
}
```

### `latent_refraction` seed `1` topology `hub_rich`

```json
{
  "by_frame": {
    "danger_frame": {
      "top_node": 11,
      "target_refraction_drop": 0.503759,
      "mean_non_target_refraction_drop": 0.283835,
      "frame_specificity_score": 0.219925
    },
    "environment_frame": {
      "top_node": 27,
      "target_refraction_drop": 0.503759,
      "mean_non_target_refraction_drop": 0.323308,
      "frame_specificity_score": 0.180451
    },
    "visibility_frame": {
      "top_node": 11,
      "target_refraction_drop": 0.233083,
      "mean_non_target_refraction_drop": 0.419173,
      "frame_specificity_score": -0.18609
    }
  },
  "mean_frame_specificity_score": 0.071429
}
```

### `latent_refraction` seed `2` topology `hub_rich`

```json
{
  "by_frame": {
    "danger_frame": {
      "top_node": 16,
      "target_refraction_drop": 0.398496,
      "mean_non_target_refraction_drop": 0.276316,
      "frame_specificity_score": 0.12218
    },
    "environment_frame": {
      "top_node": 46,
      "target_refraction_drop": 0.345865,
      "mean_non_target_refraction_drop": 0.328947,
      "frame_specificity_score": 0.016917
    },
    "visibility_frame": {
      "top_node": 46,
      "target_refraction_drop": 0.353383,
      "mean_non_target_refraction_drop": 0.325188,
      "frame_specificity_score": 0.028195
    }
  },
  "mean_frame_specificity_score": 0.055764
}
```

### `latent_refraction` seed `0` topology `hub_degree_preserving_random`

```json
{
  "by_frame": {
    "danger_frame": {
      "top_node": 17,
      "target_refraction_drop": 0.605263,
      "mean_non_target_refraction_drop": 0.483083,
      "frame_specificity_score": 0.12218
    },
    "environment_frame": {
      "top_node": 17,
      "target_refraction_drop": 0.590226,
      "mean_non_target_refraction_drop": 0.490602,
      "frame_specificity_score": 0.099624
    },
    "visibility_frame": {
      "top_node": 17,
      "target_refraction_drop": 0.37594,
      "mean_non_target_refraction_drop": 0.597744,
      "frame_specificity_score": -0.221805
    }
  },
  "mean_frame_specificity_score": -0.0
}
```

### `latent_refraction` seed `1` topology `hub_degree_preserving_random`

```json
{
  "by_frame": {
    "danger_frame": {
      "top_node": 11,
      "target_refraction_drop": 0.552632,
      "mean_non_target_refraction_drop": 0.56203,
      "frame_specificity_score": -0.009398
    },
    "environment_frame": {
      "top_node": 11,
      "target_refraction_drop": 0.402256,
      "mean_non_target_refraction_drop": 0.637218,
      "frame_specificity_score": -0.234962
    },
    "visibility_frame": {
      "top_node": 11,
      "target_refraction_drop": 0.721805,
      "mean_non_target_refraction_drop": 0.477444,
      "frame_specificity_score": 0.244361
    }
  },
  "mean_frame_specificity_score": 0.0
}
```

### `latent_refraction` seed `2` topology `hub_degree_preserving_random`

```json
{
  "by_frame": {
    "danger_frame": {
      "top_node": 16,
      "target_refraction_drop": 0.458647,
      "mean_non_target_refraction_drop": 0.362782,
      "frame_specificity_score": 0.095865
    },
    "environment_frame": {
      "top_node": 46,
      "target_refraction_drop": 0.428571,
      "mean_non_target_refraction_drop": 0.383459,
      "frame_specificity_score": 0.045113
    },
    "visibility_frame": {
      "top_node": 37,
      "target_refraction_drop": 0.409774,
      "mean_non_target_refraction_drop": 0.218045,
      "frame_specificity_score": 0.191729
    }
  },
  "mean_frame_specificity_score": 0.110902
}
```

## Runtime Notes

- total runtime seconds: `79.844153`
- smoke mode: `False`
- node ablation budget per model: `24`

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, or production validation.
