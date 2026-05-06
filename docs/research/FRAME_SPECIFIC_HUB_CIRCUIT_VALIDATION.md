# Frame-Specific Hub Circuit Validation

## Goal

Test whether load-bearing hubs act as global traffic bottlenecks, frame-specific authority routes, or suppressor/gate candidates in the existing `latent_refraction` toy probe.

No new semantics, architecture, FlyWire biology, or wave/pointer mechanisms are added.

## Completed Configs

| Topology | Runs | Accuracy | Authority | Refraction | Global Hubs | Frame-Specific Nodes | Suppressor Candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hub_degree_preserving_random` | `3` | `0.987886` | `0.502089` | `0.505848` | `5.000000` | `35.333334` | `10.000000` |
| `hub_rich` | `3` | `0.971596` | `0.461988` | `0.470760` | `3.000000` | `29.666666` | `10.000000` |
| `random_sparse` | `3` | `0.950710` | `0.428989` | `0.446533` | `4.000000` | `31.000000` | `10.000000` |

## Run Configuration

```json
{
  "experiment": "latent_refraction",
  "input_mode": "entangled",
  "seeds": 3,
  "hidden": 64,
  "steps": 5,
  "epochs": 200,
  "train_size": 1600,
  "test_size": 800,
  "topology_modes": [
    "hub_degree_preserving_random",
    "hub_rich",
    "random_sparse"
  ],
  "max_node_ablation": 64,
  "edge_group_fraction": 0.05,
  "random_edge_controls": 3,
  "top_k_fractions": [
    0.05,
    0.1,
    0.2
  ],
  "smoke": false
}
```

## Verdict

```json
{
  "supports_global_hub_bottlenecks": "true",
  "supports_frame_specific_hubs": "true",
  "supports_hub_edge_authority_routes": "true",
  "supports_suppressor_hub_candidates": "true",
  "supports_shared_core_plus_frame_routes": "true"
}
```

## Interpretation

- Some ablated hubs hurt multiple frames, supporting global bottleneck/integrator candidates.
- Some nodes are more frame-selective, supporting frame-specific authority route candidates.
- Edge-count matched hub-route ablations beat random edge controls in at least one route group.
- Some ablations raise inactive-group influence, marking suppressor/gate candidates rather than pure amplifiers.
- Treat all positives as toy circuit candidates, not unique circuits or biological claims.

## Hub Edge Route Ablation

| Topology | Route | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop | Inactive Rise |
|---|---|---:|---:|---:|---:|---:|
| `hub_degree_preserving_random` | `hub_any_edges` | `0.375104` | `0.124478` | `0.360485` | `0.112643` | `0.178363` |
| `hub_degree_preserving_random` | `hub_incoming_edges` | `0.392231` | `0.130465` | `0.375940` | `0.116959` | `0.200501` |
| `hub_degree_preserving_random` | `hub_outgoing_edges` | `0.334586` | `0.096352` | `0.327485` | `0.087023` | `0.154553` |
| `hub_degree_preserving_random` | `hub_to_hub_edges` | `0.212197` | `0.162629` | `0.220551` | `0.148009` | `0.089390` |
| `hub_rich` | `hub_any_edges` | `0.307018` | `0.119187` | `0.303258` | `0.107491` | `0.150794` |
| `hub_rich` | `hub_incoming_edges` | `0.324979` | `0.133668` | `0.321637` | `0.121972` | `0.167502` |
| `hub_rich` | `hub_outgoing_edges` | `0.314119` | `0.110693` | `0.299081` | `0.102478` | `0.105681` |
| `hub_rich` | `hub_to_hub_edges` | `0.114035` | `0.104845` | `0.101504` | `0.087580` | `0.058897` |
| `random_sparse` | `hub_any_edges` | `0.253968` | `0.104706` | `0.265664` | `0.105820` | `0.122389` |
| `random_sparse` | `hub_incoming_edges` | `0.212615` | `0.102478` | `0.223058` | `0.094124` | `0.105263` |
| `random_sparse` | `hub_outgoing_edges` | `0.172097` | `0.105820` | `0.157895` | `0.103035` | `0.083542` |
| `random_sparse` | `hub_to_hub_edges` | `0.079365` | `0.065581` | `0.065998` | `0.063910` | `0.025063` |

## Per-Frame Node Saliency

### `hub_degree_preserving_random` seed `0`

Top overall authority nodes:

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `17` | `108.000000` | `danger_specific` | `danger_frame` | `0.458647` | `0.523810` | `0.308271` | `0.122180` |
| `25` | `41.000000` | `environment_specific` | `environment_frame` | `0.419799` | `0.403509` | `0.214286` | `0.167293` |
| `24` | `55.000000` | `global_hub` | `environment_frame` | `0.407268` | `0.383459` | `0.251880` | `0.022556` |
| `32` | `28.000000` | `environment_specific` | `environment_frame` | `0.269424` | `0.253133` | `0.187970` | `0.093985` |
| `33` | `75.000000` | `environment_specific` | `environment_frame` | `0.251880` | `0.228070` | `0.157895` | `0.131579` |
| `9` | `8.000000` | `danger_specific` | `danger_frame` | `0.220551` | `0.199248` | `0.124060` | `0.343985` |
| `11` | `24.000000` | `environment_specific` | `environment_frame` | `0.192982` | `0.160401` | `0.150376` | `0.233083` |
| `46` | `10.000000` | `environment_specific` | `environment_frame` | `0.191729` | `0.177945` | `0.229323` | `0.201128` |
| `12` | `15.000000` | `visibility_specific` | `visibility_frame` | `0.162907` | `0.159148` | `0.090226` | `0.082707` |
| `37` | `13.000000` | `global_hub` | `environment_frame` | `0.159148` | `0.157895` | `0.075188` | `0.056391` |

Top nodes for `danger_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `17` | `108.000000` | `danger_specific` | `0.293233` | `0.605263` | `0.296992` | `0.308271` | `0.122180` |
| `25` | `41.000000` | `environment_specific` | `0.278195` | `0.481203` | `0.266917` | `0.214286` | `0.167293` |
| `9` | `8.000000` | `danger_specific` | `0.278195` | `0.428571` | `0.304511` | `0.124060` | `0.343985` |
| `24` | `55.000000` | `global_hub` | `0.488722` | `0.398496` | `0.146617` | `0.251880` | `0.022556` |
| `32` | `28.000000` | `environment_specific` | `0.248120` | `0.300752` | `0.112782` | `0.187970` | `0.093985` |

Top nodes for `environment_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `17` | `108.000000` | `danger_specific` | `0.345865` | `0.590226` | `0.285714` | `0.304511` | `0.122180` |
| `25` | `41.000000` | `environment_specific` | `0.533835` | `0.515038` | `0.353383` | `0.161654` | `0.167293` |
| `24` | `55.000000` | `global_hub` | `0.345865` | `0.398496` | `0.289474` | `0.109023` | `0.022556` |
| `11` | `24.000000` | `environment_specific` | `0.154135` | `0.315789` | `0.165414` | `0.150376` | `0.233083` |
| `32` | `28.000000` | `environment_specific` | `0.184211` | `0.315789` | `0.195489` | `0.120301` | `0.093985` |

Top nodes for `visibility_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `17` | `108.000000` | `danger_specific` | `0.180451` | `0.375940` | `0.187970` | `0.187970` | `0.122180` |
| `24` | `55.000000` | `global_hub` | `0.146617` | `0.353383` | `0.187970` | `0.165414` | `0.022556` |
| `12` | `15.000000` | `visibility_specific` | `0.060150` | `0.214286` | `0.127820` | `0.086466` | `0.082707` |
| `25` | `41.000000` | `environment_specific` | `0.109023` | `0.214286` | `0.101504` | `0.112782` | `0.167293` |
| `34` | `12.000000` | `global_hub` | `0.060150` | `0.165414` | `0.060150` | `0.105263` | `0.016917` |

### `hub_degree_preserving_random` seed `1`

Top overall authority nodes:

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `11` | `72.000000` | `visibility_specific` | `visibility_frame` | `0.644110` | `0.558897` | `0.383459` | `0.244361` |
| `8` | `36.000000` | `danger_specific` | `danger_frame` | `0.263158` | `0.278195` | `0.187970` | `0.101504` |
| `27` | `104.000000` | `danger_specific` | `danger_frame` | `0.258145` | `0.240602` | `0.195489` | `0.400376` |
| `53` | `49.000000` | `danger_specific` | `danger_frame` | `0.248120` | `0.258145` | `0.270677` | `0.250000` |
| `32` | `34.000000` | `danger_specific` | `danger_frame` | `0.179198` | `0.170426` | `0.221805` | `0.387218` |
| `57` | `9.000000` | `global_hub` | `danger_frame` | `0.152882` | `0.137845` | `0.120301` | `0.052632` |
| `1` | `8.000000` | `danger_specific` | `danger_frame` | `0.141604` | `0.112782` | `0.131579` | `0.270677` |
| `5` | `12.000000` | `environment_specific` | `environment_frame` | `0.139098` | `0.120301` | `0.135338` | `0.067669` |
| `18` | `10.000000` | `danger_specific` | `danger_frame` | `0.134085` | `0.120301` | `0.093985` | `0.078947` |
| `56` | `12.000000` | `danger_specific` | `danger_frame` | `0.134085` | `0.132832` | `0.101504` | `0.139098` |

Top nodes for `danger_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `11` | `72.000000` | `visibility_specific` | `0.462406` | `0.552632` | `0.552632` | `0.000000` | `0.244361` |
| `27` | `104.000000` | `danger_specific` | `0.300752` | `0.507519` | `0.312030` | `0.195489` | `0.400376` |
| `32` | `34.000000` | `danger_specific` | `0.120301` | `0.428571` | `0.206767` | `0.221805` | `0.387218` |
| `53` | `49.000000` | `danger_specific` | `0.263158` | `0.424812` | `0.154135` | `0.270677` | `0.250000` |
| `8` | `36.000000` | `danger_specific` | `0.383459` | `0.345865` | `0.206767` | `0.139098` | `0.101504` |

Top nodes for `environment_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `11` | `72.000000` | `visibility_specific` | `0.323308` | `0.402256` | `0.274436` | `0.127820` | `0.244361` |
| `8` | `36.000000` | `danger_specific` | `0.165414` | `0.274436` | `0.086466` | `0.187970` | `0.101504` |
| `4` | `7.000000` | `environment_specific` | `0.165414` | `0.255639` | `0.127820` | `0.127820` | `0.221805` |
| `53` | `49.000000` | `danger_specific` | `0.387218` | `0.244361` | `0.075188` | `0.169173` | `0.250000` |
| `31` | `24.000000` | `environment_specific` | `0.071429` | `0.184211` | `0.105263` | `0.078947` | `0.163534` |

Top nodes for `visibility_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `11` | `72.000000` | `visibility_specific` | `0.443609` | `0.721805` | `0.338346` | `0.383459` | `0.244361` |
| `8` | `36.000000` | `danger_specific` | `0.109023` | `0.214286` | `0.090226` | `0.124060` | `0.101504` |
| `27` | `104.000000` | `danger_specific` | `0.048872` | `0.142857` | `0.090226` | `0.052632` | `0.400376` |
| `44` | `21.000000` | `danger_specific` | `0.041353` | `0.112782` | `0.045113` | `0.067669` | `0.084586` |
| `53` | `49.000000` | `danger_specific` | `0.033835` | `0.105263` | `0.018797` | `0.086466` | `0.250000` |

### `hub_degree_preserving_random` seed `2`

Top overall authority nodes:

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `16` | `59.000000` | `danger_specific` | `danger_frame` | `0.416040` | `0.394737` | `0.214286` | `0.095865` |
| `46` | `104.000000` | `global_hub` | `environment_frame` | `0.353383` | `0.398496` | `0.274436` | `0.045113` |
| `37` | `46.000000` | `visibility_specific` | `visibility_frame` | `0.279449` | `0.281955` | `0.236842` | `0.191729` |
| `42` | `76.000000` | `danger_specific` | `danger_frame` | `0.157895` | `0.122807` | `0.139098` | `0.216165` |
| `9` | `10.000000` | `environment_specific` | `environment_frame` | `0.155388` | `0.122807` | `0.131579` | `0.221805` |
| `33` | `24.000000` | `danger_specific` | `danger_frame` | `0.130326` | `0.100251` | `0.116541` | `0.080827` |
| `43` | `19.000000` | `suppressor_candidate` | `visibility_frame` | `0.112782` | `0.106516` | `0.071429` | `0.048872` |
| `36` | `13.000000` | `danger_specific` | `danger_frame` | `0.109023` | `0.092732` | `0.139098` | `0.086466` |
| `10` | `14.000000` | `global_hub` | `visibility_frame` | `0.101504` | `0.100251` | `0.048872` | `0.013158` |
| `1` | `10.000000` | `suppressor_candidate` | `visibility_frame` | `0.095238` | `0.083960` | `0.060150` | `0.048872` |

Top nodes for `danger_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `16` | `59.000000` | `danger_specific` | `0.436090` | `0.458647` | `0.398496` | `0.060150` | `0.095865` |
| `46` | `104.000000` | `global_hub` | `0.259398` | `0.398496` | `0.139098` | `0.259398` | `0.045113` |
| `42` | `76.000000` | `danger_specific` | `0.360902` | `0.266917` | `0.127820` | `0.139098` | `0.216165` |
| `33` | `24.000000` | `danger_specific` | `0.112782` | `0.154135` | `0.037594` | `0.116541` | `0.080827` |
| `37` | `46.000000` | `visibility_specific` | `0.154135` | `0.150376` | `0.075188` | `0.075188` | `0.191729` |

Top nodes for `environment_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `46` | `104.000000` | `global_hub` | `0.221805` | `0.428571` | `0.154135` | `0.274436` | `0.045113` |
| `16` | `59.000000` | `danger_specific` | `0.462406` | `0.323308` | `0.109023` | `0.214286` | `0.095865` |
| `37` | `46.000000` | `visibility_specific` | `0.251880` | `0.285714` | `0.116541` | `0.169173` | `0.191729` |
| `9` | `10.000000` | `environment_specific` | `0.127820` | `0.270677` | `0.139098` | `0.131579` | `0.221805` |
| `43` | `19.000000` | `suppressor_candidate` | `0.105263` | `0.124060` | `0.052632` | `0.071429` | `0.048872` |

Top nodes for `visibility_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `37` | `46.000000` | `visibility_specific` | `0.146617` | `0.409774` | `0.172932` | `0.236842` | `0.191729` |
| `16` | `59.000000` | `danger_specific` | `0.165414` | `0.402256` | `0.199248` | `0.203008` | `0.095865` |
| `46` | `104.000000` | `global_hub` | `0.266917` | `0.368421` | `0.270677` | `0.097744` | `0.045113` |
| `51` | `27.000000` | `visibility_specific` | `0.037594` | `0.184211` | `0.105263` | `0.078947` | `0.142857` |
| `62` | `37.000000` | `visibility_specific` | `0.018797` | `0.146617` | `0.105263` | `0.041353` | `0.148496` |

### `hub_rich` seed `0`

Top overall authority nodes:

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `17` | `108.000000` | `danger_specific` | `danger_frame` | `0.395990` | `0.486216` | `0.319549` | `0.257519` |
| `33` | `75.000000` | `environment_specific` | `environment_frame` | `0.334586` | `0.310777` | `0.161654` | `0.187970` |
| `24` | `55.000000` | `danger_specific` | `danger_frame` | `0.248120` | `0.206767` | `0.150376` | `0.225564` |
| `20` | `46.000000` | `danger_specific` | `danger_frame` | `0.206767` | `0.197995` | `0.131579` | `0.148496` |
| `43` | `13.000000` | `danger_specific` | `danger_frame` | `0.206767` | `0.186717` | `0.161654` | `0.199248` |
| `34` | `12.000000` | `global_hub` | `environment_frame` | `0.175439` | `0.167920` | `0.090226` | `0.046992` |
| `4` | `31.000000` | `environment_specific` | `environment_frame` | `0.155388` | `0.161654` | `0.037594` | `0.225564` |
| `47` | `19.000000` | `global_hub` | `environment_frame` | `0.145363` | `0.130326` | `0.109023` | `0.058271` |
| `37` | `13.000000` | `environment_specific` | `environment_frame` | `0.129073` | `0.107769` | `0.093985` | `0.103383` |
| `5` | `23.000000` | `environment_specific` | `environment_frame` | `0.122807` | `0.117794` | `0.063910` | `0.071429` |

Top nodes for `danger_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `17` | `108.000000` | `danger_specific` | `0.323308` | `0.657895` | `0.338346` | `0.319549` | `0.257519` |
| `33` | `75.000000` | `environment_specific` | `0.248120` | `0.387218` | `0.225564` | `0.161654` | `0.187970` |
| `24` | `55.000000` | `danger_specific` | `0.263158` | `0.357143` | `0.206767` | `0.150376` | `0.225564` |
| `43` | `13.000000` | `danger_specific` | `0.206767` | `0.319549` | `0.157895` | `0.161654` | `0.199248` |
| `20` | `46.000000` | `danger_specific` | `0.120301` | `0.296992` | `0.165414` | `0.131579` | `0.148496` |

Top nodes for `environment_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `17` | `108.000000` | `danger_specific` | `0.327068` | `0.560150` | `0.278195` | `0.281955` | `0.257519` |
| `33` | `75.000000` | `environment_specific` | `0.383459` | `0.436090` | `0.424812` | `0.011278` | `0.187970` |
| `4` | `31.000000` | `environment_specific` | `0.473684` | `0.312030` | `0.278195` | `0.033835` | `0.225564` |
| `43` | `13.000000` | `danger_specific` | `0.109023` | `0.203008` | `0.124060` | `0.078947` | `0.199248` |
| `34` | `12.000000` | `global_hub` | `0.150376` | `0.199248` | `0.154135` | `0.045113` | `0.046992` |

Top nodes for `visibility_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `17` | `108.000000` | `danger_specific` | `0.146617` | `0.240602` | `0.120301` | `0.120301` | `0.257519` |
| `20` | `46.000000` | `danger_specific` | `0.022556` | `0.139098` | `0.086466` | `0.052632` | `0.148496` |
| `34` | `12.000000` | `global_hub` | `0.045113` | `0.127820` | `0.071429` | `0.056391` | `0.046992` |
| `13` | `7.000000` | `global_hub` | `0.018797` | `0.116541` | `0.063910` | `0.052632` | `0.020677` |
| `55` | `8.000000` | `danger_specific` | `0.018797` | `0.112782` | `0.082707` | `0.030075` | `0.092105` |

### `hub_rich` seed `1`

Top overall authority nodes:

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `27` | `104.000000` | `environment_specific` | `environment_frame` | `0.394737` | `0.383459` | `0.214286` | `0.180451` |
| `11` | `72.000000` | `danger_specific` | `danger_frame` | `0.335840` | `0.357143` | `0.195489` | `0.219925` |
| `38` | `44.000000` | `environment_specific` | `environment_frame` | `0.334586` | `0.309524` | `0.312030` | `0.280075` |
| `44` | `21.000000` | `environment_specific` | `environment_frame` | `0.177945` | `0.144110` | `0.199248` | `0.156015` |
| `32` | `34.000000` | `environment_specific` | `environment_frame` | `0.171679` | `0.167920` | `0.154135` | `0.159774` |
| `16` | `11.000000` | `danger_specific` | `danger_frame` | `0.136591` | `0.119048` | `0.109023` | `0.137218` |
| `56` | `12.000000` | `environment_specific` | `environment_frame` | `0.132832` | `0.124060` | `0.124060` | `0.157895` |
| `10` | `5.000000` | `danger_specific` | `danger_frame` | `0.125313` | `0.109023` | `0.086466` | `0.129699` |
| `6` | `33.000000` | `danger_specific` | `danger_frame` | `0.109023` | `0.093985` | `0.093985` | `0.157895` |
| `53` | `49.000000` | `global_hub` | `environment_frame` | `0.109023` | `0.119048` | `0.120301` | `0.013158` |

Top nodes for `danger_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `11` | `72.000000` | `danger_specific` | `0.308271` | `0.503759` | `0.375940` | `0.127820` | `0.219925` |
| `27` | `104.000000` | `environment_specific` | `0.488722` | `0.454887` | `0.315789` | `0.139098` | `0.180451` |
| `38` | `44.000000` | `environment_specific` | `0.312030` | `0.398496` | `0.342105` | `0.056391` | `0.280075` |
| `32` | `34.000000` | `environment_specific` | `0.120301` | `0.233083` | `0.157895` | `0.075188` | `0.159774` |
| `1` | `8.000000` | `danger_specific` | `0.086466` | `0.218045` | `0.127820` | `0.090226` | `0.201128` |

Top nodes for `environment_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `27` | `104.000000` | `environment_specific` | `0.349624` | `0.503759` | `0.289474` | `0.214286` | `0.180451` |
| `38` | `44.000000` | `environment_specific` | `0.436090` | `0.496241` | `0.184211` | `0.312030` | `0.280075` |
| `11` | `72.000000` | `danger_specific` | `0.150376` | `0.334586` | `0.139098` | `0.195489` | `0.219925` |
| `32` | `34.000000` | `environment_specific` | `0.285714` | `0.274436` | `0.120301` | `0.154135` | `0.159774` |
| `44` | `21.000000` | `environment_specific` | `0.157895` | `0.248120` | `0.048872` | `0.199248` | `0.156015` |

Top nodes for `visibility_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `11` | `72.000000` | `danger_specific` | `0.124060` | `0.233083` | `0.093985` | `0.139098` | `0.219925` |
| `27` | `104.000000` | `environment_specific` | `0.056391` | `0.191729` | `0.105263` | `0.086466` | `0.180451` |
| `46` | `12.000000` | `suppressor_candidate` | `0.030075` | `0.105263` | `0.037594` | `0.067669` | `0.056391` |
| `53` | `49.000000` | `global_hub` | `0.041353` | `0.105263` | `0.033835` | `0.071429` | `0.013158` |
| `44` | `21.000000` | `environment_specific` | `0.026316` | `0.078947` | `0.030075` | `0.048872` | `0.156015` |

### `hub_rich` seed `2`

Top overall authority nodes:

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `16` | `59.000000` | `danger_specific` | `danger_frame` | `0.305764` | `0.317043` | `0.139098` | `0.122180` |
| `46` | `104.000000` | `global_hub` | `visibility_frame` | `0.289474` | `0.334586` | `0.240602` | `0.028195` |
| `32` | `51.000000` | `visibility_specific` | `visibility_frame` | `0.248120` | `0.253133` | `0.169173` | `0.133459` |
| `42` | `76.000000` | `danger_specific` | `danger_frame` | `0.142857` | `0.144110` | `0.093985` | `0.088346` |
| `37` | `46.000000` | `environment_specific` | `environment_frame` | `0.117794` | `0.111529` | `0.078947` | `0.154135` |
| `26` | `10.000000` | `visibility_specific` | `visibility_frame` | `0.112782` | `0.102757` | `0.090226` | `0.212406` |
| `62` | `37.000000` | `visibility_specific` | `visibility_frame` | `0.112782` | `0.105263` | `0.093985` | `0.259398` |
| `9` | `10.000000` | `environment_specific` | `environment_frame` | `0.088972` | `0.071429` | `0.052632` | `0.107143` |
| `43` | `19.000000` | `environment_specific` | `environment_frame` | `0.081454` | `0.078947` | `0.063910` | `0.101504` |
| `1` | `10.000000` | `visibility_specific` | `visibility_frame` | `0.081454` | `0.071429` | `0.033835` | `0.095865` |

Top nodes for `danger_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `16` | `59.000000` | `danger_specific` | `0.330827` | `0.398496` | `0.259398` | `0.139098` | `0.122180` |
| `46` | `104.000000` | `global_hub` | `0.191729` | `0.304511` | `0.063910` | `0.240602` | `0.028195` |
| `42` | `76.000000` | `danger_specific` | `0.469925` | `0.203008` | `0.169173` | `0.033835` | `0.088346` |
| `32` | `51.000000` | `visibility_specific` | `0.120301` | `0.116541` | `0.063910` | `0.052632` | `0.133459` |
| `18` | `20.000000` | `suppressor_candidate` | `0.011278` | `0.071429` | `0.015038` | `0.056391` | `0.039474` |

Top nodes for `environment_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `46` | `104.000000` | `global_hub` | `0.236842` | `0.345865` | `0.127820` | `0.218045` | `0.028195` |
| `16` | `59.000000` | `danger_specific` | `0.575188` | `0.334586` | `0.210526` | `0.124060` | `0.122180` |
| `32` | `51.000000` | `visibility_specific` | `0.263158` | `0.300752` | `0.195489` | `0.105263` | `0.133459` |
| `37` | `46.000000` | `environment_specific` | `0.184211` | `0.214286` | `0.135338` | `0.078947` | `0.154135` |
| `42` | `76.000000` | `danger_specific` | `0.067669` | `0.146617` | `0.052632` | `0.093985` | `0.088346` |

Top nodes for `visibility_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `46` | `104.000000` | `global_hub` | `0.172932` | `0.353383` | `0.225564` | `0.127820` | `0.028195` |
| `32` | `51.000000` | `visibility_specific` | `0.161654` | `0.342105` | `0.172932` | `0.169173` | `0.133459` |
| `62` | `37.000000` | `visibility_specific` | `0.067669` | `0.278195` | `0.184211` | `0.093985` | `0.259398` |
| `26` | `10.000000` | `visibility_specific` | `0.086466` | `0.244361` | `0.154135` | `0.090226` | `0.212406` |
| `56` | `11.000000` | `visibility_specific` | `0.097744` | `0.244361` | `0.124060` | `0.120301` | `0.266917` |

### `random_sparse` seed `0`

Top overall authority nodes:

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `24` | `12.000000` | `danger_specific` | `danger_frame` | `0.209273` | `0.211779` | `0.101504` | `0.060150` |
| `35` | `17.000000` | `global_hub` | `environment_frame` | `0.199248` | `0.213033` | `0.116541` | `0.030075` |
| `34` | `18.000000` | `environment_specific` | `environment_frame` | `0.169173` | `0.165414` | `0.067669` | `0.090226` |
| `9` | `16.000000` | `danger_specific` | `danger_frame` | `0.161654` | `0.139098` | `0.139098` | `0.191729` |
| `32` | `17.000000` | `danger_specific` | `danger_frame` | `0.144110` | `0.109023` | `0.127820` | `0.214286` |
| `11` | `15.000000` | `global_hub` | `environment_frame` | `0.130326` | `0.127820` | `0.045113` | `0.022556` |
| `1` | `18.000000` | `global_hub` | `visibility_frame` | `0.126566` | `0.131579` | `0.071429` | `0.056391` |
| `31` | `10.000000` | `global_hub` | `danger_frame` | `0.124060` | `0.127820` | `0.056391` | `0.022556` |
| `39` | `22.000000` | `environment_specific` | `environment_frame` | `0.121554` | `0.127820` | `0.075188` | `0.157895` |
| `7` | `19.000000` | `visibility_specific` | `visibility_frame` | `0.117794` | `0.129073` | `0.063910` | `0.082707` |

Top nodes for `danger_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `9` | `16.000000` | `danger_specific` | `0.146617` | `0.266917` | `0.127820` | `0.139098` | `0.191729` |
| `24` | `12.000000` | `danger_specific` | `0.135338` | `0.251880` | `0.150376` | `0.101504` | `0.060150` |
| `32` | `17.000000` | `danger_specific` | `0.187970` | `0.251880` | `0.124060` | `0.127820` | `0.214286` |
| `35` | `17.000000` | `global_hub` | `0.101504` | `0.233083` | `0.116541` | `0.116541` | `0.030075` |
| `15` | `23.000000` | `danger_specific` | `0.120301` | `0.157895` | `0.078947` | `0.078947` | `0.073308` |

Top nodes for `environment_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `35` | `17.000000` | `global_hub` | `0.086466` | `0.233083` | `0.135338` | `0.097744` | `0.030075` |
| `39` | `22.000000` | `environment_specific` | `0.097744` | `0.233083` | `0.157895` | `0.075188` | `0.157895` |
| `34` | `18.000000` | `environment_specific` | `0.086466` | `0.225564` | `0.161654` | `0.063910` | `0.090226` |
| `24` | `12.000000` | `danger_specific` | `0.097744` | `0.180451` | `0.101504` | `0.078947` | `0.060150` |
| `4` | `26.000000` | `environment_specific` | `0.124060` | `0.142857` | `0.071429` | `0.071429` | `0.086466` |

Top nodes for `visibility_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `0` | `17.000000` | `visibility_specific` | `0.112782` | `0.218045` | `0.063910` | `0.154135` | `0.161654` |
| `24` | `12.000000` | `danger_specific` | `0.075188` | `0.203008` | `0.109023` | `0.093985` | `0.060150` |
| `12` | `14.000000` | `visibility_specific` | `0.048872` | `0.195489` | `0.120301` | `0.075188` | `0.116541` |
| `7` | `19.000000` | `visibility_specific` | `0.082707` | `0.184211` | `0.120301` | `0.063910` | `0.082707` |
| `35` | `17.000000` | `global_hub` | `0.075188` | `0.172932` | `0.063910` | `0.109023` | `0.030075` |

### `random_sparse` seed `1`

Top overall authority nodes:

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `21` | `16.000000` | `danger_specific` | `danger_frame` | `0.180451` | `0.174185` | `0.135338` | `0.122180` |
| `20` | `16.000000` | `environment_specific` | `environment_frame` | `0.159148` | `0.176692` | `0.116541` | `0.118421` |
| `18` | `15.000000` | `environment_specific` | `environment_frame` | `0.154135` | `0.137845` | `0.157895` | `0.227444` |
| `51` | `14.000000` | `danger_specific` | `danger_frame` | `0.117794` | `0.111529` | `0.090226` | `0.131579` |
| `63` | `27.000000` | `danger_specific` | `danger_frame` | `0.110276` | `0.132832` | `0.086466` | `0.105263` |
| `22` | `12.000000` | `danger_specific` | `danger_frame` | `0.098997` | `0.093985` | `0.082707` | `0.152256` |
| `37` | `10.000000` | `global_hub` | `environment_frame` | `0.097744` | `0.095238` | `0.045113` | `0.026316` |
| `25` | `16.000000` | `environment_specific` | `environment_frame` | `0.095238` | `0.102757` | `0.078947` | `0.105263` |
| `44` | `24.000000` | `danger_specific` | `danger_frame` | `0.095238` | `0.083960` | `0.056391` | `0.110902` |
| `60` | `18.000000` | `danger_specific` | `danger_frame` | `0.095238` | `0.098997` | `0.097744` | `0.218045` |

Top nodes for `danger_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `21` | `16.000000` | `danger_specific` | `0.124060` | `0.255639` | `0.187970` | `0.067669` | `0.122180` |
| `60` | `18.000000` | `danger_specific` | `0.082707` | `0.244361` | `0.146617` | `0.097744` | `0.218045` |
| `63` | `27.000000` | `danger_specific` | `0.056391` | `0.203008` | `0.120301` | `0.082707` | `0.105263` |
| `51` | `14.000000` | `danger_specific` | `0.131579` | `0.199248` | `0.109023` | `0.090226` | `0.131579` |
| `22` | `12.000000` | `danger_specific` | `0.018797` | `0.195489` | `0.112782` | `0.082707` | `0.152256` |

Top nodes for `environment_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `18` | `15.000000` | `environment_specific` | `0.180451` | `0.289474` | `0.131579` | `0.157895` | `0.227444` |
| `20` | `16.000000` | `environment_specific` | `0.172932` | `0.255639` | `0.139098` | `0.116541` | `0.118421` |
| `21` | `16.000000` | `danger_specific` | `0.187970` | `0.255639` | `0.120301` | `0.135338` | `0.122180` |
| `25` | `16.000000` | `environment_specific` | `0.116541` | `0.172932` | `0.093985` | `0.078947` | `0.105263` |
| `2` | `22.000000` | `environment_specific` | `0.078947` | `0.172932` | `0.082707` | `0.090226` | `0.125940` |

Top nodes for `visibility_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `20` | `16.000000` | `environment_specific` | `0.139098` | `0.180451` | `0.063910` | `0.116541` | `0.118421` |
| `63` | `27.000000` | `danger_specific` | `0.086466` | `0.127820` | `0.041353` | `0.086466` | `0.105263` |
| `5` | `14.000000` | `global_hub` | `0.071429` | `0.109023` | `0.022556` | `0.086466` | `0.022556` |
| `9` | `17.000000` | `danger_specific` | `0.071429` | `0.109023` | `0.037594` | `0.071429` | `0.099624` |
| `49` | `14.000000` | `unclear` | `0.022556` | `0.109023` | `0.082707` | `0.026316` | `0.043233` |

### `random_sparse` seed `2`

Top overall authority nodes:

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `35` | `16.000000` | `environment_specific` | `environment_frame` | `0.260652` | `0.218045` | `0.161654` | `0.135338` |
| `56` | `19.000000` | `environment_specific` | `environment_frame` | `0.191729` | `0.164160` | `0.154135` | `0.103383` |
| `55` | `19.000000` | `danger_specific` | `danger_frame` | `0.154135` | `0.145363` | `0.165414` | `0.109023` |
| `26` | `25.000000` | `environment_specific` | `environment_frame` | `0.139098` | `0.126566` | `0.176692` | `0.125940` |
| `10` | `15.000000` | `environment_specific` | `environment_frame` | `0.129073` | `0.111529` | `0.101504` | `0.075188` |
| `31` | `20.000000` | `environment_specific` | `environment_frame` | `0.121554` | `0.140351` | `0.135338` | `0.077068` |
| `13` | `24.000000` | `environment_specific` | `environment_frame` | `0.116541` | `0.104010` | `0.135338` | `0.103383` |
| `1` | `20.000000` | `suppressor_candidate` | `visibility_frame` | `0.114035` | `0.096491` | `0.075188` | `0.058271` |
| `5` | `23.000000` | `suppressor_candidate` | `visibility_frame` | `0.112782` | `0.105263` | `0.112782` | `0.045113` |
| `48` | `17.000000` | `global_hub` | `environment_frame` | `0.110276` | `0.104010` | `0.112782` | `0.024436` |

Top nodes for `danger_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `35` | `16.000000` | `environment_specific` | `0.165414` | `0.281955` | `0.120301` | `0.161654` | `0.135338` |
| `55` | `19.000000` | `danger_specific` | `0.127820` | `0.218045` | `0.052632` | `0.165414` | `0.109023` |
| `33` | `16.000000` | `danger_specific` | `0.127820` | `0.169173` | `0.037594` | `0.131579` | `0.133459` |
| `34` | `14.000000` | `danger_specific` | `0.090226` | `0.161654` | `0.030075` | `0.131579` | `0.114662` |
| `29` | `13.000000` | `danger_specific` | `0.045113` | `0.093985` | `0.045113` | `0.048872` | `0.062030` |

Top nodes for `environment_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `35` | `16.000000` | `environment_specific` | `0.319549` | `0.308271` | `0.154135` | `0.154135` | `0.135338` |
| `56` | `19.000000` | `environment_specific` | `0.131579` | `0.233083` | `0.078947` | `0.154135` | `0.103383` |
| `26` | `25.000000` | `environment_specific` | `0.263158` | `0.210526` | `0.033835` | `0.176692` | `0.125940` |
| `31` | `20.000000` | `environment_specific` | `0.090226` | `0.191729` | `0.056391` | `0.135338` | `0.077068` |
| `13` | `24.000000` | `environment_specific` | `0.206767` | `0.172932` | `0.037594` | `0.135338` | `0.103383` |

Top nodes for `visibility_frame`:

| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |
|---:|---:|---|---:|---:|---:|---:|---:|
| `22` | `13.000000` | `visibility_specific` | `0.018797` | `0.184211` | `0.146617` | `0.037594` | `0.140977` |
| `31` | `20.000000` | `environment_specific` | `0.063910` | `0.184211` | `0.090226` | `0.093985` | `0.077068` |
| `56` | `19.000000` | `environment_specific` | `0.048872` | `0.180451` | `0.105263` | `0.075188` | `0.103383` |
| `55` | `19.000000` | `danger_specific` | `0.022556` | `0.165414` | `0.086466` | `0.078947` | `0.109023` |
| `8` | `12.000000` | `visibility_specific` | `0.011278` | `0.142857` | `0.109023` | `0.033835` | `0.101504` |

## Suppression / Leakage Candidates

### `hub_degree_preserving_random` seed `0`

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `17` | `108.000000` | `danger_specific` | `danger_frame` | `0.458647` | `0.523810` | `0.308271` | `0.122180` |
| `24` | `55.000000` | `global_hub` | `environment_frame` | `0.407268` | `0.383459` | `0.251880` | `0.022556` |
| `46` | `10.000000` | `environment_specific` | `environment_frame` | `0.191729` | `0.177945` | `0.229323` | `0.201128` |
| `25` | `41.000000` | `environment_specific` | `environment_frame` | `0.419799` | `0.403509` | `0.214286` | `0.167293` |
| `32` | `28.000000` | `environment_specific` | `environment_frame` | `0.269424` | `0.253133` | `0.187970` | `0.093985` |
| `33` | `75.000000` | `environment_specific` | `environment_frame` | `0.251880` | `0.228070` | `0.157895` | `0.131579` |
| `11` | `24.000000` | `environment_specific` | `environment_frame` | `0.192982` | `0.160401` | `0.150376` | `0.233083` |
| `15` | `24.000000` | `danger_specific` | `danger_frame` | `0.100251` | `0.086466` | `0.127820` | `0.186090` |
| `34` | `12.000000` | `global_hub` | `visibility_frame` | `0.151629` | `0.154135` | `0.112782` | `0.016917` |
| `55` | `8.000000` | `danger_specific` | `danger_frame` | `0.139098` | `0.119048` | `0.112782` | `0.114662` |

### `hub_degree_preserving_random` seed `1`

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `11` | `72.000000` | `visibility_specific` | `visibility_frame` | `0.644110` | `0.558897` | `0.383459` | `0.244361` |
| `53` | `49.000000` | `danger_specific` | `danger_frame` | `0.248120` | `0.258145` | `0.270677` | `0.250000` |
| `32` | `34.000000` | `danger_specific` | `danger_frame` | `0.179198` | `0.170426` | `0.221805` | `0.387218` |
| `27` | `104.000000` | `danger_specific` | `danger_frame` | `0.258145` | `0.240602` | `0.195489` | `0.400376` |
| `8` | `36.000000` | `danger_specific` | `danger_frame` | `0.263158` | `0.278195` | `0.187970` | `0.101504` |
| `5` | `12.000000` | `environment_specific` | `environment_frame` | `0.139098` | `0.120301` | `0.135338` | `0.067669` |
| `1` | `8.000000` | `danger_specific` | `danger_frame` | `0.141604` | `0.112782` | `0.131579` | `0.270677` |
| `4` | `7.000000` | `environment_specific` | `environment_frame` | `0.125313` | `0.107769` | `0.127820` | `0.221805` |
| `57` | `9.000000` | `global_hub` | `danger_frame` | `0.152882` | `0.137845` | `0.120301` | `0.052632` |
| `37` | `9.000000` | `danger_specific` | `danger_frame` | `0.126566` | `0.107769` | `0.112782` | `0.092105` |

### `hub_degree_preserving_random` seed `2`

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `46` | `104.000000` | `global_hub` | `environment_frame` | `0.353383` | `0.398496` | `0.274436` | `0.045113` |
| `37` | `46.000000` | `visibility_specific` | `visibility_frame` | `0.279449` | `0.281955` | `0.236842` | `0.191729` |
| `16` | `59.000000` | `danger_specific` | `danger_frame` | `0.416040` | `0.394737` | `0.214286` | `0.095865` |
| `36` | `13.000000` | `danger_specific` | `danger_frame` | `0.109023` | `0.092732` | `0.139098` | `0.086466` |
| `42` | `76.000000` | `danger_specific` | `danger_frame` | `0.157895` | `0.122807` | `0.139098` | `0.216165` |
| `9` | `10.000000` | `environment_specific` | `environment_frame` | `0.155388` | `0.122807` | `0.131579` | `0.221805` |
| `33` | `24.000000` | `danger_specific` | `danger_frame` | `0.130326` | `0.100251` | `0.116541` | `0.080827` |
| `34` | `13.000000` | `visibility_specific` | `visibility_frame` | `0.058897` | `0.056391` | `0.078947` | `0.073308` |
| `40` | `6.000000` | `suppressor_candidate` | `visibility_frame` | `0.070175` | `0.062657` | `0.078947` | `0.052632` |
| `51` | `27.000000` | `visibility_specific` | `visibility_frame` | `0.083960` | `0.088972` | `0.078947` | `0.142857` |

### `hub_rich` seed `0`

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `17` | `108.000000` | `danger_specific` | `danger_frame` | `0.395990` | `0.486216` | `0.319549` | `0.257519` |
| `43` | `13.000000` | `danger_specific` | `danger_frame` | `0.206767` | `0.186717` | `0.161654` | `0.199248` |
| `24` | `55.000000` | `danger_specific` | `danger_frame` | `0.248120` | `0.206767` | `0.150376` | `0.225564` |
| `20` | `46.000000` | `danger_specific` | `danger_frame` | `0.206767` | `0.197995` | `0.131579` | `0.148496` |
| `47` | `19.000000` | `global_hub` | `environment_frame` | `0.145363` | `0.130326` | `0.109023` | `0.058271` |
| `60` | `13.000000` | `danger_specific` | `danger_frame` | `0.117794` | `0.104010` | `0.105263` | `0.103383` |
| `49` | `11.000000` | `danger_specific` | `danger_frame` | `0.095238` | `0.092732` | `0.101504` | `0.103383` |
| `37` | `13.000000` | `environment_specific` | `environment_frame` | `0.129073` | `0.107769` | `0.093985` | `0.103383` |
| `12` | `15.000000` | `danger_specific` | `danger_frame` | `0.104010` | `0.098997` | `0.090226` | `0.071429` |
| `34` | `12.000000` | `global_hub` | `environment_frame` | `0.175439` | `0.167920` | `0.090226` | `0.046992` |

### `hub_rich` seed `1`

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `38` | `44.000000` | `environment_specific` | `environment_frame` | `0.334586` | `0.309524` | `0.312030` | `0.280075` |
| `27` | `104.000000` | `environment_specific` | `environment_frame` | `0.394737` | `0.383459` | `0.214286` | `0.180451` |
| `44` | `21.000000` | `environment_specific` | `environment_frame` | `0.177945` | `0.144110` | `0.199248` | `0.156015` |
| `11` | `72.000000` | `danger_specific` | `danger_frame` | `0.335840` | `0.357143` | `0.195489` | `0.219925` |
| `32` | `34.000000` | `environment_specific` | `environment_frame` | `0.171679` | `0.167920` | `0.154135` | `0.159774` |
| `56` | `12.000000` | `environment_specific` | `environment_frame` | `0.132832` | `0.124060` | `0.124060` | `0.157895` |
| `53` | `49.000000` | `global_hub` | `environment_frame` | `0.109023` | `0.119048` | `0.120301` | `0.013158` |
| `16` | `11.000000` | `danger_specific` | `danger_frame` | `0.136591` | `0.119048` | `0.109023` | `0.137218` |
| `45` | `5.000000` | `environment_specific` | `environment_frame` | `0.092732` | `0.071429` | `0.109023` | `0.101504` |
| `8` | `36.000000` | `environment_specific` | `environment_frame` | `0.101504` | `0.088972` | `0.101504` | `0.114662` |

### `hub_rich` seed `2`

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `46` | `104.000000` | `global_hub` | `visibility_frame` | `0.289474` | `0.334586` | `0.240602` | `0.028195` |
| `32` | `51.000000` | `visibility_specific` | `visibility_frame` | `0.248120` | `0.253133` | `0.169173` | `0.133459` |
| `16` | `59.000000` | `danger_specific` | `danger_frame` | `0.305764` | `0.317043` | `0.139098` | `0.122180` |
| `56` | `11.000000` | `visibility_specific` | `visibility_frame` | `0.071429` | `0.066416` | `0.120301` | `0.266917` |
| `42` | `76.000000` | `danger_specific` | `danger_frame` | `0.142857` | `0.144110` | `0.093985` | `0.088346` |
| `62` | `37.000000` | `visibility_specific` | `visibility_frame` | `0.112782` | `0.105263` | `0.093985` | `0.259398` |
| `26` | `10.000000` | `visibility_specific` | `visibility_frame` | `0.112782` | `0.102757` | `0.090226` | `0.212406` |
| `37` | `46.000000` | `environment_specific` | `environment_frame` | `0.117794` | `0.111529` | `0.078947` | `0.154135` |
| `51` | `27.000000` | `visibility_specific` | `visibility_frame` | `0.045113` | `0.045113` | `0.075188` | `0.248120` |
| `20` | `31.000000` | `suppressor_candidate` | `environment_frame` | `0.072682` | `0.055138` | `0.067669` | `0.058271` |

### `random_sparse` seed `0`

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `0` | `17.000000` | `visibility_specific` | `visibility_frame` | `0.097744` | `0.110276` | `0.154135` | `0.161654` |
| `9` | `16.000000` | `danger_specific` | `danger_frame` | `0.161654` | `0.139098` | `0.139098` | `0.191729` |
| `32` | `17.000000` | `danger_specific` | `danger_frame` | `0.144110` | `0.109023` | `0.127820` | `0.214286` |
| `35` | `17.000000` | `global_hub` | `environment_frame` | `0.199248` | `0.213033` | `0.116541` | `0.030075` |
| `24` | `12.000000` | `danger_specific` | `danger_frame` | `0.209273` | `0.211779` | `0.101504` | `0.060150` |
| `3` | `15.000000` | `visibility_specific` | `visibility_frame` | `0.073935` | `0.100251` | `0.093985` | `0.092105` |
| `19` | `17.000000` | `suppressor_candidate` | `danger_frame` | `0.062657` | `0.051378` | `0.082707` | `0.024436` |
| `15` | `23.000000` | `danger_specific` | `danger_frame` | `0.114035` | `0.109023` | `0.078947` | `0.073308` |
| `37` | `12.000000` | `visibility_specific` | `visibility_frame` | `0.087719` | `0.107769` | `0.078947` | `0.092105` |
| `12` | `14.000000` | `visibility_specific` | `visibility_frame` | `0.107769` | `0.117794` | `0.075188` | `0.116541` |

### `random_sparse` seed `1`

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `18` | `15.000000` | `environment_specific` | `environment_frame` | `0.154135` | `0.137845` | `0.157895` | `0.227444` |
| `21` | `16.000000` | `danger_specific` | `danger_frame` | `0.180451` | `0.174185` | `0.135338` | `0.122180` |
| `47` | `12.000000` | `environment_specific` | `environment_frame` | `0.090226` | `0.090226` | `0.135338` | `0.101504` |
| `15` | `21.000000` | `danger_specific` | `danger_frame` | `0.092732` | `0.083960` | `0.127820` | `0.133459` |
| `20` | `16.000000` | `environment_specific` | `environment_frame` | `0.159148` | `0.176692` | `0.116541` | `0.118421` |
| `12` | `15.000000` | `danger_specific` | `danger_frame` | `0.073935` | `0.073935` | `0.105263` | `0.125940` |
| `60` | `18.000000` | `danger_specific` | `danger_frame` | `0.095238` | `0.098997` | `0.097744` | `0.218045` |
| `2` | `22.000000` | `environment_specific` | `environment_frame` | `0.091479` | `0.088972` | `0.090226` | `0.125940` |
| `39` | `10.000000` | `danger_specific` | `danger_frame` | `0.071429` | `0.071429` | `0.090226` | `0.157895` |
| `51` | `14.000000` | `danger_specific` | `danger_frame` | `0.117794` | `0.111529` | `0.090226` | `0.131579` |

### `random_sparse` seed `2`

| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |
|---:|---:|---|---|---:|---:|---:|---:|
| `26` | `25.000000` | `environment_specific` | `environment_frame` | `0.139098` | `0.126566` | `0.176692` | `0.125940` |
| `55` | `19.000000` | `danger_specific` | `danger_frame` | `0.154135` | `0.145363` | `0.165414` | `0.109023` |
| `35` | `16.000000` | `environment_specific` | `environment_frame` | `0.260652` | `0.218045` | `0.161654` | `0.135338` |
| `56` | `19.000000` | `environment_specific` | `environment_frame` | `0.191729` | `0.164160` | `0.154135` | `0.103383` |
| `13` | `24.000000` | `environment_specific` | `environment_frame` | `0.116541` | `0.104010` | `0.135338` | `0.103383` |
| `31` | `20.000000` | `environment_specific` | `environment_frame` | `0.121554` | `0.140351` | `0.135338` | `0.077068` |
| `33` | `16.000000` | `danger_specific` | `danger_frame` | `0.109023` | `0.080201` | `0.131579` | `0.133459` |
| `34` | `14.000000` | `danger_specific` | `danger_frame` | `0.086466` | `0.085213` | `0.131579` | `0.114662` |
| `12` | `12.000000` | `global_hub` | `environment_frame` | `0.105263` | `0.104010` | `0.116541` | `0.035714` |
| `5` | `23.000000` | `suppressor_candidate` | `visibility_frame` | `0.112782` | `0.105263` | `0.112782` | `0.045113` |

## Route Overlap Matrix

### `hub_degree_preserving_random` seed `0`

Node top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.5,
        "visibility_frame": 0.2
      },
      "environment_frame": {
        "danger_frame": 0.5,
        "environment_frame": 1.0,
        "visibility_frame": 0.5
      },
      "visibility_frame": {
        "danger_frame": 0.2,
        "environment_frame": 0.5,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 1,
    "unique_count_by_frame": {
      "danger_frame": 1,
      "environment_frame": 0,
      "visibility_frame": 1
    },
    "set_size_by_frame": {
      "danger_frame": 3,
      "environment_frame": 3,
      "visibility_frame": 3
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.714286,
        "visibility_frame": 0.5
      },
      "environment_frame": {
        "danger_frame": 0.714286,
        "environment_frame": 1.0,
        "visibility_frame": 0.5
      },
      "visibility_frame": {
        "danger_frame": 0.5,
        "environment_frame": 0.5,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 4,
    "unique_count_by_frame": {
      "danger_frame": 1,
      "environment_frame": 1,
      "visibility_frame": 2
    },
    "set_size_by_frame": {
      "danger_frame": 6,
      "environment_frame": 6,
      "visibility_frame": 6
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.3,
        "visibility_frame": 0.3
      },
      "environment_frame": {
        "danger_frame": 0.3,
        "environment_frame": 1.0,
        "visibility_frame": 0.529412
      },
      "visibility_frame": {
        "danger_frame": 0.3,
        "environment_frame": 0.529412,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 6,
    "unique_count_by_frame": {
      "danger_frame": 7,
      "environment_frame": 4,
      "visibility_frame": 4
    },
    "set_size_by_frame": {
      "danger_frame": 13,
      "environment_frame": 13,
      "visibility_frame": 13
    }
  }
}
```

Edge proxy top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.724138,
        "visibility_frame": 0.515152
      },
      "environment_frame": {
        "danger_frame": 0.724138,
        "environment_frame": 1.0,
        "visibility_frame": 0.515152
      },
      "visibility_frame": {
        "danger_frame": 0.515152,
        "environment_frame": 0.515152,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 15,
    "unique_count_by_frame": {
      "danger_frame": 2,
      "environment_frame": 2,
      "visibility_frame": 6
    },
    "set_size_by_frame": {
      "danger_frame": 25,
      "environment_frame": 25,
      "visibility_frame": 25
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.754386,
        "visibility_frame": 0.694915
      },
      "environment_frame": {
        "danger_frame": 0.754386,
        "environment_frame": 1.0,
        "visibility_frame": 0.612903
      },
      "visibility_frame": {
        "danger_frame": 0.694915,
        "environment_frame": 0.612903,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 38,
    "unique_count_by_frame": {
      "danger_frame": 4,
      "environment_frame": 7,
      "visibility_frame": 9
    },
    "set_size_by_frame": {
      "danger_frame": 50,
      "environment_frame": 50,
      "visibility_frame": 50
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.736842,
        "visibility_frame": 0.663866
      },
      "environment_frame": {
        "danger_frame": 0.736842,
        "environment_frame": 1.0,
        "visibility_frame": 0.622951
      },
      "visibility_frame": {
        "danger_frame": 0.663866,
        "environment_frame": 0.622951,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 71,
    "unique_count_by_frame": {
      "danger_frame": 7,
      "environment_frame": 10,
      "visibility_frame": 15
    },
    "set_size_by_frame": {
      "danger_frame": 99,
      "environment_frame": 99,
      "visibility_frame": 99
    }
  }
}
```

### `hub_degree_preserving_random` seed `1`

Node top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.2,
        "visibility_frame": 0.5
      },
      "environment_frame": {
        "danger_frame": 0.2,
        "environment_frame": 1.0,
        "visibility_frame": 0.5
      },
      "visibility_frame": {
        "danger_frame": 0.5,
        "environment_frame": 0.5,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 1,
    "unique_count_by_frame": {
      "danger_frame": 1,
      "environment_frame": 1,
      "visibility_frame": 0
    },
    "set_size_by_frame": {
      "danger_frame": 3,
      "environment_frame": 3,
      "visibility_frame": 3
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.333333,
        "visibility_frame": 0.5
      },
      "environment_frame": {
        "danger_frame": 0.333333,
        "environment_frame": 1.0,
        "visibility_frame": 0.333333
      },
      "visibility_frame": {
        "danger_frame": 0.5,
        "environment_frame": 0.333333,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 3,
    "unique_count_by_frame": {
      "danger_frame": 2,
      "environment_frame": 3,
      "visibility_frame": 2
    },
    "set_size_by_frame": {
      "danger_frame": 6,
      "environment_frame": 6,
      "visibility_frame": 6
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.368421,
        "visibility_frame": 0.444444
      },
      "environment_frame": {
        "danger_frame": 0.368421,
        "environment_frame": 1.0,
        "visibility_frame": 0.368421
      },
      "visibility_frame": {
        "danger_frame": 0.444444,
        "environment_frame": 0.368421,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 5,
    "unique_count_by_frame": {
      "danger_frame": 3,
      "environment_frame": 4,
      "visibility_frame": 3
    },
    "set_size_by_frame": {
      "danger_frame": 13,
      "environment_frame": 13,
      "visibility_frame": 13
    }
  }
}
```

Edge proxy top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.371429,
        "visibility_frame": 0.230769
      },
      "environment_frame": {
        "danger_frame": 0.371429,
        "environment_frame": 1.0,
        "visibility_frame": 0.411765
      },
      "visibility_frame": {
        "danger_frame": 0.230769,
        "environment_frame": 0.411765,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 8,
    "unique_count_by_frame": {
      "danger_frame": 10,
      "environment_frame": 5,
      "visibility_frame": 9
    },
    "set_size_by_frame": {
      "danger_frame": 24,
      "environment_frame": 24,
      "visibility_frame": 24
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.454545,
        "visibility_frame": 0.263158
      },
      "environment_frame": {
        "danger_frame": 0.454545,
        "environment_frame": 1.0,
        "visibility_frame": 0.352113
      },
      "visibility_frame": {
        "danger_frame": 0.263158,
        "environment_frame": 0.352113,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 19,
    "unique_count_by_frame": {
      "danger_frame": 17,
      "environment_frame": 12,
      "visibility_frame": 22
    },
    "set_size_by_frame": {
      "danger_frame": 48,
      "environment_frame": 48,
      "visibility_frame": 48
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.5,
        "visibility_frame": 0.381295
      },
      "environment_frame": {
        "danger_frame": 0.5,
        "environment_frame": 1.0,
        "visibility_frame": 0.548387
      },
      "visibility_frame": {
        "danger_frame": 0.381295,
        "environment_frame": 0.548387,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 47,
    "unique_count_by_frame": {
      "danger_frame": 26,
      "environment_frame": 11,
      "visibility_frame": 22
    },
    "set_size_by_frame": {
      "danger_frame": 96,
      "environment_frame": 96,
      "visibility_frame": 96
    }
  }
}
```

### `hub_degree_preserving_random` seed `2`

Node top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.5,
        "visibility_frame": 0.5
      },
      "environment_frame": {
        "danger_frame": 0.5,
        "environment_frame": 1.0,
        "visibility_frame": 1.0
      },
      "visibility_frame": {
        "danger_frame": 0.5,
        "environment_frame": 1.0,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 2,
    "unique_count_by_frame": {
      "danger_frame": 1,
      "environment_frame": 0,
      "visibility_frame": 0
    },
    "set_size_by_frame": {
      "danger_frame": 3,
      "environment_frame": 3,
      "visibility_frame": 3
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.333333,
        "visibility_frame": 0.333333
      },
      "environment_frame": {
        "danger_frame": 0.333333,
        "environment_frame": 1.0,
        "visibility_frame": 0.5
      },
      "visibility_frame": {
        "danger_frame": 0.333333,
        "environment_frame": 0.5,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 3,
    "unique_count_by_frame": {
      "danger_frame": 3,
      "environment_frame": 2,
      "visibility_frame": 2
    },
    "set_size_by_frame": {
      "danger_frame": 6,
      "environment_frame": 6,
      "visibility_frame": 6
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.238095,
        "visibility_frame": 0.238095
      },
      "environment_frame": {
        "danger_frame": 0.238095,
        "environment_frame": 1.0,
        "visibility_frame": 0.238095
      },
      "visibility_frame": {
        "danger_frame": 0.238095,
        "environment_frame": 0.238095,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 4,
    "unique_count_by_frame": {
      "danger_frame": 7,
      "environment_frame": 7,
      "visibility_frame": 7
    },
    "set_size_by_frame": {
      "danger_frame": 13,
      "environment_frame": 13,
      "visibility_frame": 13
    }
  }
}
```

Edge proxy top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.333333,
        "visibility_frame": 0.405405
      },
      "environment_frame": {
        "danger_frame": 0.333333,
        "environment_frame": 1.0,
        "visibility_frame": 0.575758
      },
      "visibility_frame": {
        "danger_frame": 0.405405,
        "environment_frame": 0.575758,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 12,
    "unique_count_by_frame": {
      "danger_frame": 10,
      "environment_frame": 6,
      "visibility_frame": 4
    },
    "set_size_by_frame": {
      "danger_frame": 26,
      "environment_frame": 26,
      "visibility_frame": 26
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.485714,
        "visibility_frame": 0.464789
      },
      "environment_frame": {
        "danger_frame": 0.485714,
        "environment_frame": 1.0,
        "visibility_frame": 0.650794
      },
      "visibility_frame": {
        "danger_frame": 0.464789,
        "environment_frame": 0.650794,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 30,
    "unique_count_by_frame": {
      "danger_frame": 15,
      "environment_frame": 7,
      "visibility_frame": 8
    },
    "set_size_by_frame": {
      "danger_frame": 52,
      "environment_frame": 52,
      "visibility_frame": 52
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.6,
        "visibility_frame": 0.637795
      },
      "environment_frame": {
        "danger_frame": 0.6,
        "environment_frame": 1.0,
        "visibility_frame": 0.762712
      },
      "visibility_frame": {
        "danger_frame": 0.637795,
        "environment_frame": 0.762712,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 73,
    "unique_count_by_frame": {
      "danger_frame": 18,
      "environment_frame": 9,
      "visibility_frame": 6
    },
    "set_size_by_frame": {
      "danger_frame": 104,
      "environment_frame": 104,
      "visibility_frame": 104
    }
  }
}
```

### `hub_rich` seed `0`

Node top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.5,
        "visibility_frame": 0.2
      },
      "environment_frame": {
        "danger_frame": 0.5,
        "environment_frame": 1.0,
        "visibility_frame": 0.2
      },
      "visibility_frame": {
        "danger_frame": 0.2,
        "environment_frame": 0.2,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 1,
    "unique_count_by_frame": {
      "danger_frame": 1,
      "environment_frame": 1,
      "visibility_frame": 2
    },
    "set_size_by_frame": {
      "danger_frame": 3,
      "environment_frame": 3,
      "visibility_frame": 3
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.714286,
        "visibility_frame": 0.5
      },
      "environment_frame": {
        "danger_frame": 0.714286,
        "environment_frame": 1.0,
        "visibility_frame": 0.333333
      },
      "visibility_frame": {
        "danger_frame": 0.5,
        "environment_frame": 0.333333,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 3,
    "unique_count_by_frame": {
      "danger_frame": 0,
      "environment_frame": 1,
      "visibility_frame": 2
    },
    "set_size_by_frame": {
      "danger_frame": 6,
      "environment_frame": 6,
      "visibility_frame": 6
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.368421,
        "visibility_frame": 0.3
      },
      "environment_frame": {
        "danger_frame": 0.368421,
        "environment_frame": 1.0,
        "visibility_frame": 0.238095
      },
      "visibility_frame": {
        "danger_frame": 0.3,
        "environment_frame": 0.238095,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 4,
    "unique_count_by_frame": {
      "danger_frame": 4,
      "environment_frame": 5,
      "visibility_frame": 6
    },
    "set_size_by_frame": {
      "danger_frame": 13,
      "environment_frame": 13,
      "visibility_frame": 13
    }
  }
}
```

Edge proxy top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.5625,
        "visibility_frame": 0.515152
      },
      "environment_frame": {
        "danger_frame": 0.5625,
        "environment_frame": 1.0,
        "visibility_frame": 0.428571
      },
      "visibility_frame": {
        "danger_frame": 0.515152,
        "environment_frame": 0.428571,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 14,
    "unique_count_by_frame": {
      "danger_frame": 4,
      "environment_frame": 6,
      "visibility_frame": 7
    },
    "set_size_by_frame": {
      "danger_frame": 25,
      "environment_frame": 25,
      "visibility_frame": 25
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.587302,
        "visibility_frame": 0.538462
      },
      "environment_frame": {
        "danger_frame": 0.587302,
        "environment_frame": 1.0,
        "visibility_frame": 0.428571
      },
      "visibility_frame": {
        "danger_frame": 0.538462,
        "environment_frame": 0.428571,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 29,
    "unique_count_by_frame": {
      "danger_frame": 7,
      "environment_frame": 12,
      "visibility_frame": 14
    },
    "set_size_by_frame": {
      "danger_frame": 50,
      "environment_frame": 50,
      "visibility_frame": 50
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.636364,
        "visibility_frame": 0.534884
      },
      "environment_frame": {
        "danger_frame": 0.636364,
        "environment_frame": 1.0,
        "visibility_frame": 0.571429
      },
      "visibility_frame": {
        "danger_frame": 0.534884,
        "environment_frame": 0.571429,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 62,
    "unique_count_by_frame": {
      "danger_frame": 15,
      "environment_frame": 12,
      "visibility_frame": 20
    },
    "set_size_by_frame": {
      "danger_frame": 99,
      "environment_frame": 99,
      "visibility_frame": 99
    }
  }
}
```

### `hub_rich` seed `1`

Node top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 1.0,
        "visibility_frame": 0.5
      },
      "environment_frame": {
        "danger_frame": 1.0,
        "environment_frame": 1.0,
        "visibility_frame": 0.5
      },
      "visibility_frame": {
        "danger_frame": 0.5,
        "environment_frame": 0.5,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 2,
    "unique_count_by_frame": {
      "danger_frame": 0,
      "environment_frame": 0,
      "visibility_frame": 1
    },
    "set_size_by_frame": {
      "danger_frame": 3,
      "environment_frame": 3,
      "visibility_frame": 3
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.5,
        "visibility_frame": 0.2
      },
      "environment_frame": {
        "danger_frame": 0.5,
        "environment_frame": 1.0,
        "visibility_frame": 0.333333
      },
      "visibility_frame": {
        "danger_frame": 0.2,
        "environment_frame": 0.333333,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 2,
    "unique_count_by_frame": {
      "danger_frame": 2,
      "environment_frame": 1,
      "visibility_frame": 3
    },
    "set_size_by_frame": {
      "danger_frame": 6,
      "environment_frame": 6,
      "visibility_frame": 6
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.3,
        "visibility_frame": 0.130435
      },
      "environment_frame": {
        "danger_frame": 0.3,
        "environment_frame": 1.0,
        "visibility_frame": 0.181818
      },
      "visibility_frame": {
        "danger_frame": 0.130435,
        "environment_frame": 0.181818,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 2,
    "unique_count_by_frame": {
      "danger_frame": 6,
      "environment_frame": 5,
      "visibility_frame": 8
    },
    "set_size_by_frame": {
      "danger_frame": 13,
      "environment_frame": 13,
      "visibility_frame": 13
    }
  }
}
```

Edge proxy top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.548387,
        "visibility_frame": 0.371429
      },
      "environment_frame": {
        "danger_frame": 0.548387,
        "environment_frame": 1.0,
        "visibility_frame": 0.263158
      },
      "visibility_frame": {
        "danger_frame": 0.371429,
        "environment_frame": 0.263158,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 7,
    "unique_count_by_frame": {
      "danger_frame": 1,
      "environment_frame": 4,
      "visibility_frame": 8
    },
    "set_size_by_frame": {
      "danger_frame": 24,
      "environment_frame": 24,
      "visibility_frame": 24
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.745455,
        "visibility_frame": 0.476923
      },
      "environment_frame": {
        "danger_frame": 0.745455,
        "environment_frame": 1.0,
        "visibility_frame": 0.371429
      },
      "visibility_frame": {
        "danger_frame": 0.476923,
        "environment_frame": 0.371429,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 24,
    "unique_count_by_frame": {
      "danger_frame": 0,
      "environment_frame": 5,
      "visibility_frame": 15
    },
    "set_size_by_frame": {
      "danger_frame": 48,
      "environment_frame": 48,
      "visibility_frame": 48
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.714286,
        "visibility_frame": 0.613445
      },
      "environment_frame": {
        "danger_frame": 0.714286,
        "environment_frame": 1.0,
        "visibility_frame": 0.52381
      },
      "visibility_frame": {
        "danger_frame": 0.613445,
        "environment_frame": 0.52381,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 62,
    "unique_count_by_frame": {
      "danger_frame": 5,
      "environment_frame": 12,
      "visibility_frame": 19
    },
    "set_size_by_frame": {
      "danger_frame": 96,
      "environment_frame": 96,
      "visibility_frame": 96
    }
  }
}
```

### `hub_rich` seed `2`

Node top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.5,
        "visibility_frame": 0.2
      },
      "environment_frame": {
        "danger_frame": 0.5,
        "environment_frame": 1.0,
        "visibility_frame": 0.5
      },
      "visibility_frame": {
        "danger_frame": 0.2,
        "environment_frame": 0.5,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 1,
    "unique_count_by_frame": {
      "danger_frame": 1,
      "environment_frame": 0,
      "visibility_frame": 1
    },
    "set_size_by_frame": {
      "danger_frame": 3,
      "environment_frame": 3,
      "visibility_frame": 3
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.714286,
        "visibility_frame": 0.333333
      },
      "environment_frame": {
        "danger_frame": 0.714286,
        "environment_frame": 1.0,
        "visibility_frame": 0.333333
      },
      "visibility_frame": {
        "danger_frame": 0.333333,
        "environment_frame": 0.333333,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 3,
    "unique_count_by_frame": {
      "danger_frame": 1,
      "environment_frame": 1,
      "visibility_frame": 3
    },
    "set_size_by_frame": {
      "danger_frame": 6,
      "environment_frame": 6,
      "visibility_frame": 6
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.238095,
        "visibility_frame": 0.181818
      },
      "environment_frame": {
        "danger_frame": 0.238095,
        "environment_frame": 1.0,
        "visibility_frame": 0.238095
      },
      "visibility_frame": {
        "danger_frame": 0.181818,
        "environment_frame": 0.238095,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 3,
    "unique_count_by_frame": {
      "danger_frame": 7,
      "environment_frame": 6,
      "visibility_frame": 7
    },
    "set_size_by_frame": {
      "danger_frame": 13,
      "environment_frame": 13,
      "visibility_frame": 13
    }
  }
}
```

Edge proxy top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.444444,
        "visibility_frame": 0.181818
      },
      "environment_frame": {
        "danger_frame": 0.444444,
        "environment_frame": 1.0,
        "visibility_frame": 0.333333
      },
      "visibility_frame": {
        "danger_frame": 0.181818,
        "environment_frame": 0.333333,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 7,
    "unique_count_by_frame": {
      "danger_frame": 9,
      "environment_frame": 4,
      "visibility_frame": 12
    },
    "set_size_by_frame": {
      "danger_frame": 26,
      "environment_frame": 26,
      "visibility_frame": 26
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.464789,
        "visibility_frame": 0.316456
      },
      "environment_frame": {
        "danger_frame": 0.464789,
        "environment_frame": 1.0,
        "visibility_frame": 0.507246
      },
      "visibility_frame": {
        "danger_frame": 0.316456,
        "environment_frame": 0.507246,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 22,
    "unique_count_by_frame": {
      "danger_frame": 16,
      "environment_frame": 6,
      "visibility_frame": 14
    },
    "set_size_by_frame": {
      "danger_frame": 52,
      "environment_frame": 52,
      "visibility_frame": 52
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.56391,
        "visibility_frame": 0.475177
      },
      "environment_frame": {
        "danger_frame": 0.56391,
        "environment_frame": 1.0,
        "visibility_frame": 0.664
      },
      "visibility_frame": {
        "danger_frame": 0.475177,
        "environment_frame": 0.664,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 63,
    "unique_count_by_frame": {
      "danger_frame": 25,
      "environment_frame": 9,
      "visibility_frame": 17
    },
    "set_size_by_frame": {
      "danger_frame": 104,
      "environment_frame": 104,
      "visibility_frame": 104
    }
  }
}
```

### `random_sparse` seed `0`

Node top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.0,
        "visibility_frame": 0.2
      },
      "environment_frame": {
        "danger_frame": 0.0,
        "environment_frame": 1.0,
        "visibility_frame": 0.0
      },
      "visibility_frame": {
        "danger_frame": 0.2,
        "environment_frame": 0.0,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 0,
    "unique_count_by_frame": {
      "danger_frame": 2,
      "environment_frame": 3,
      "visibility_frame": 2
    },
    "set_size_by_frame": {
      "danger_frame": 3,
      "environment_frame": 3,
      "visibility_frame": 3
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.333333,
        "visibility_frame": 0.2
      },
      "environment_frame": {
        "danger_frame": 0.333333,
        "environment_frame": 1.0,
        "visibility_frame": 0.2
      },
      "visibility_frame": {
        "danger_frame": 0.2,
        "environment_frame": 0.2,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 2,
    "unique_count_by_frame": {
      "danger_frame": 3,
      "environment_frame": 3,
      "visibility_frame": 4
    },
    "set_size_by_frame": {
      "danger_frame": 6,
      "environment_frame": 6,
      "visibility_frame": 6
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.3,
        "visibility_frame": 0.368421
      },
      "environment_frame": {
        "danger_frame": 0.3,
        "environment_frame": 1.0,
        "visibility_frame": 0.368421
      },
      "visibility_frame": {
        "danger_frame": 0.368421,
        "environment_frame": 0.368421,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 4,
    "unique_count_by_frame": {
      "danger_frame": 4,
      "environment_frame": 4,
      "visibility_frame": 3
    },
    "set_size_by_frame": {
      "danger_frame": 13,
      "environment_frame": 13,
      "visibility_frame": 13
    }
  }
}
```

Edge proxy top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.282051,
        "visibility_frame": 0.315789
      },
      "environment_frame": {
        "danger_frame": 0.282051,
        "environment_frame": 1.0,
        "visibility_frame": 0.428571
      },
      "visibility_frame": {
        "danger_frame": 0.315789,
        "environment_frame": 0.428571,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 10,
    "unique_count_by_frame": {
      "danger_frame": 12,
      "environment_frame": 9,
      "visibility_frame": 8
    },
    "set_size_by_frame": {
      "danger_frame": 25,
      "environment_frame": 25,
      "visibility_frame": 25
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.428571,
        "visibility_frame": 0.315789
      },
      "environment_frame": {
        "danger_frame": 0.428571,
        "environment_frame": 1.0,
        "visibility_frame": 0.428571
      },
      "visibility_frame": {
        "danger_frame": 0.315789,
        "environment_frame": 0.428571,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 20,
    "unique_count_by_frame": {
      "danger_frame": 16,
      "environment_frame": 10,
      "visibility_frame": 16
    },
    "set_size_by_frame": {
      "danger_frame": 50,
      "environment_frame": 50,
      "visibility_frame": 50
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.445255,
        "visibility_frame": 0.394366
      },
      "environment_frame": {
        "danger_frame": 0.445255,
        "environment_frame": 1.0,
        "visibility_frame": 0.5
      },
      "visibility_frame": {
        "danger_frame": 0.394366,
        "environment_frame": 0.5,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 45,
    "unique_count_by_frame": {
      "danger_frame": 27,
      "environment_frame": 17,
      "visibility_frame": 22
    },
    "set_size_by_frame": {
      "danger_frame": 99,
      "environment_frame": 99,
      "visibility_frame": 99
    }
  }
}
```

### `random_sparse` seed `1`

Node top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.2,
        "visibility_frame": 0.2
      },
      "environment_frame": {
        "danger_frame": 0.2,
        "environment_frame": 1.0,
        "visibility_frame": 0.2
      },
      "visibility_frame": {
        "danger_frame": 0.2,
        "environment_frame": 0.2,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 0,
    "unique_count_by_frame": {
      "danger_frame": 1,
      "environment_frame": 1,
      "visibility_frame": 1
    },
    "set_size_by_frame": {
      "danger_frame": 3,
      "environment_frame": 3,
      "visibility_frame": 3
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.090909,
        "visibility_frame": 0.090909
      },
      "environment_frame": {
        "danger_frame": 0.090909,
        "environment_frame": 1.0,
        "visibility_frame": 0.090909
      },
      "visibility_frame": {
        "danger_frame": 0.090909,
        "environment_frame": 0.090909,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 0,
    "unique_count_by_frame": {
      "danger_frame": 4,
      "environment_frame": 4,
      "visibility_frame": 4
    },
    "set_size_by_frame": {
      "danger_frame": 6,
      "environment_frame": 6,
      "visibility_frame": 6
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.083333,
        "visibility_frame": 0.130435
      },
      "environment_frame": {
        "danger_frame": 0.083333,
        "environment_frame": 1.0,
        "visibility_frame": 0.181818
      },
      "visibility_frame": {
        "danger_frame": 0.130435,
        "environment_frame": 0.181818,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 0,
    "unique_count_by_frame": {
      "danger_frame": 8,
      "environment_frame": 7,
      "visibility_frame": 6
    },
    "set_size_by_frame": {
      "danger_frame": 13,
      "environment_frame": 13,
      "visibility_frame": 13
    }
  }
}
```

Edge proxy top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.297297,
        "visibility_frame": 0.2
      },
      "environment_frame": {
        "danger_frame": 0.297297,
        "environment_frame": 1.0,
        "visibility_frame": 0.090909
      },
      "visibility_frame": {
        "danger_frame": 0.2,
        "environment_frame": 0.090909,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 1,
    "unique_count_by_frame": {
      "danger_frame": 6,
      "environment_frame": 10,
      "visibility_frame": 13
    },
    "set_size_by_frame": {
      "danger_frame": 24,
      "environment_frame": 24,
      "visibility_frame": 24
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.352113,
        "visibility_frame": 0.28
      },
      "environment_frame": {
        "danger_frame": 0.352113,
        "environment_frame": 1.0,
        "visibility_frame": 0.263158
      },
      "visibility_frame": {
        "danger_frame": 0.28,
        "environment_frame": 0.263158,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 11,
    "unique_count_by_frame": {
      "danger_frame": 13,
      "environment_frame": 14,
      "visibility_frame": 18
    },
    "set_size_by_frame": {
      "danger_frame": 48,
      "environment_frame": 48,
      "visibility_frame": 48
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.391304,
        "visibility_frame": 0.391304
      },
      "environment_frame": {
        "danger_frame": 0.391304,
        "environment_frame": 1.0,
        "visibility_frame": 0.306122
      },
      "visibility_frame": {
        "danger_frame": 0.391304,
        "environment_frame": 0.306122,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 30,
    "unique_count_by_frame": {
      "danger_frame": 18,
      "environment_frame": 27,
      "visibility_frame": 27
    },
    "set_size_by_frame": {
      "danger_frame": 96,
      "environment_frame": 96,
      "visibility_frame": 96
    }
  }
}
```

### `random_sparse` seed `2`

Node top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.2,
        "visibility_frame": 0.0
      },
      "environment_frame": {
        "danger_frame": 0.2,
        "environment_frame": 1.0,
        "visibility_frame": 0.2
      },
      "visibility_frame": {
        "danger_frame": 0.0,
        "environment_frame": 0.2,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 0,
    "unique_count_by_frame": {
      "danger_frame": 2,
      "environment_frame": 1,
      "visibility_frame": 2
    },
    "set_size_by_frame": {
      "danger_frame": 3,
      "environment_frame": 3,
      "visibility_frame": 3
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.090909,
        "visibility_frame": 0.090909
      },
      "environment_frame": {
        "danger_frame": 0.090909,
        "environment_frame": 1.0,
        "visibility_frame": 0.333333
      },
      "visibility_frame": {
        "danger_frame": 0.090909,
        "environment_frame": 0.333333,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 0,
    "unique_count_by_frame": {
      "danger_frame": 4,
      "environment_frame": 2,
      "visibility_frame": 2
    },
    "set_size_by_frame": {
      "danger_frame": 6,
      "environment_frame": 6,
      "visibility_frame": 6
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.3,
        "visibility_frame": 0.181818
      },
      "environment_frame": {
        "danger_frame": 0.3,
        "environment_frame": 1.0,
        "visibility_frame": 0.368421
      },
      "visibility_frame": {
        "danger_frame": 0.181818,
        "environment_frame": 0.368421,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 2,
    "unique_count_by_frame": {
      "danger_frame": 5,
      "environment_frame": 2,
      "visibility_frame": 4
    },
    "set_size_by_frame": {
      "danger_frame": 13,
      "environment_frame": 13,
      "visibility_frame": 13
    }
  }
}
```

Edge proxy top-K overlap:

```json
{
  "5pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.238095,
        "visibility_frame": 0.181818
      },
      "environment_frame": {
        "danger_frame": 0.238095,
        "environment_frame": 1.0,
        "visibility_frame": 0.268293
      },
      "visibility_frame": {
        "danger_frame": 0.181818,
        "environment_frame": 0.268293,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 4,
    "unique_count_by_frame": {
      "danger_frame": 12,
      "environment_frame": 9,
      "visibility_frame": 11
    },
    "set_size_by_frame": {
      "danger_frame": 26,
      "environment_frame": 26,
      "visibility_frame": 26
    }
  },
  "10pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.3,
        "visibility_frame": 0.316456
      },
      "environment_frame": {
        "danger_frame": 0.3,
        "environment_frame": 1.0,
        "visibility_frame": 0.386667
      },
      "visibility_frame": {
        "danger_frame": 0.316456,
        "environment_frame": 0.386667,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 14,
    "unique_count_by_frame": {
      "danger_frame": 17,
      "environment_frame": 13,
      "visibility_frame": 12
    },
    "set_size_by_frame": {
      "danger_frame": 52,
      "environment_frame": 52,
      "visibility_frame": 52
    }
  },
  "20pct": {
    "jaccard_matrix": {
      "danger_frame": {
        "danger_frame": 1.0,
        "environment_frame": 0.350649,
        "visibility_frame": 0.414966
      },
      "environment_frame": {
        "danger_frame": 0.350649,
        "environment_frame": 1.0,
        "visibility_frame": 0.414966
      },
      "visibility_frame": {
        "danger_frame": 0.414966,
        "environment_frame": 0.414966,
        "visibility_frame": 1.0
      }
    },
    "shared_count": 39,
    "unique_count_by_frame": {
      "danger_frame": 28,
      "environment_frame": 28,
      "visibility_frame": 21
    },
    "set_size_by_frame": {
      "danger_frame": 104,
      "environment_frame": 104,
      "visibility_frame": 104
    }
  }
}
```

## Runtime Notes

- total runtime seconds: `73.126964`
- smoke mode: `False`
- node ablation budget per model: `64`

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, FlyWire validation, or production validation.
