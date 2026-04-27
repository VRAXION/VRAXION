# Phase D8.1 Spherical Cell Coherence Audit

Verdict: **CELL_MAP_WEAK**

## Summary

- D8.1 is offline-only and tests whether hypersphere cells have predictive meaning.
- The primary verdict uses behavior features only; score and time are excluded from cell assignment.
- Same-cell leave-one-out prediction is compared against global, same-time-bucket, and random-cell controls.

## Coverage

```json
{
  "input": "output\\phase_d8_archive_psi_replay_20260427\\analysis\\panel_state_dataset.csv",
  "rows": 5840,
  "H_values": [
    128,
    256,
    384
  ],
  "anchor_counts": [
    16,
    32,
    64,
    128
  ],
  "anchor_seeds": [
    11,
    23,
    37
  ],
  "time_buckets": 10,
  "primary_features": [
    "stable_rank",
    "kernel_rank",
    "separation_sp",
    "collision_rate",
    "f_active",
    "unique_predictions",
    "edges",
    "accept_rate_window"
  ],
  "diagnostic_features": [
    "stable_rank",
    "kernel_rank",
    "separation_sp",
    "collision_rate",
    "f_active",
    "unique_predictions",
    "edges",
    "accept_rate_window",
    "main_peak_acc",
    "panel_probe_acc",
    "time_pct"
  ]
}
```

## Decision

```json
{
  "required_h": 2,
  "core": {
    "pass_h": 0,
    "weak_h": 3,
    "data_fail_h": 0,
    "by_h": {
      "128": {
        "strong_counts": 0,
        "weak_counts": 4,
        "reliable_counts": 4,
        "required_counts": 3,
        "anchor_results": [
          {
            "anchor_count": 16,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 32,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 64,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 128,
            "reliable": true,
            "strong": false,
            "weak": true
          }
        ],
        "median_same_mae": 0.005446017782521717,
        "median_time_mae": 0.00397527830287754,
        "median_random_mae": 0.0055322512218955195,
        "median_variance_ratio": 0.08463274829566027,
        "median_basin_lift": 1.4219714506172838
      },
      "256": {
        "strong_counts": 0,
        "weak_counts": 3,
        "reliable_counts": 4,
        "required_counts": 3,
        "anchor_results": [
          {
            "anchor_count": 16,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 32,
            "reliable": true,
            "strong": false,
            "weak": false
          },
          {
            "anchor_count": 64,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 128,
            "reliable": true,
            "strong": false,
            "weak": true
          }
        ],
        "median_same_mae": 0.0073199283164106415,
        "median_time_mae": 0.005509411690443124,
        "median_random_mae": 0.007335766422602148,
        "median_variance_ratio": 0.14692485194988422,
        "median_basin_lift": 1.6201844312647098
      },
      "384": {
        "strong_counts": 0,
        "weak_counts": 4,
        "reliable_counts": 4,
        "required_counts": 3,
        "anchor_results": [
          {
            "anchor_count": 16,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 32,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 64,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 128,
            "reliable": true,
            "strong": false,
            "weak": true
          }
        ],
        "median_same_mae": 0.009694066378495236,
        "median_time_mae": 0.008212538227403239,
        "median_random_mae": 0.010390743139622585,
        "median_variance_ratio": 0.18996255483020585,
        "median_basin_lift": 1.4828952531151312
      }
    }
  },
  "plus": {
    "pass_h": 0,
    "weak_h": 3,
    "data_fail_h": 0,
    "by_h": {
      "128": {
        "strong_counts": 0,
        "weak_counts": 4,
        "reliable_counts": 4,
        "required_counts": 3,
        "anchor_results": [
          {
            "anchor_count": 16,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 32,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 64,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 128,
            "reliable": true,
            "strong": false,
            "weak": true
          }
        ],
        "median_same_mae": 0.004934511757984536,
        "median_time_mae": 0.00397527830287754,
        "median_random_mae": 0.005230787696436582,
        "median_variance_ratio": 0.24174415229020083,
        "median_basin_lift": 1.6347108942471693
      },
      "256": {
        "strong_counts": 0,
        "weak_counts": 4,
        "reliable_counts": 4,
        "required_counts": 3,
        "anchor_results": [
          {
            "anchor_count": 16,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 32,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 64,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 128,
            "reliable": true,
            "strong": false,
            "weak": true
          }
        ],
        "median_same_mae": 0.006399554709186448,
        "median_time_mae": 0.005509411690443124,
        "median_random_mae": 0.006864551582461586,
        "median_variance_ratio": 0.2738162998321365,
        "median_basin_lift": 1.7603796525285296
      },
      "384": {
        "strong_counts": 0,
        "weak_counts": 4,
        "reliable_counts": 4,
        "required_counts": 3,
        "anchor_results": [
          {
            "anchor_count": 16,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 32,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 64,
            "reliable": true,
            "strong": false,
            "weak": true
          },
          {
            "anchor_count": 128,
            "reliable": true,
            "strong": false,
            "weak": true
          }
        ],
        "median_same_mae": 0.008782112279882459,
        "median_time_mae": 0.008212538227403239,
        "median_random_mae": 0.00978319803756086,
        "median_variance_ratio": 0.2737453656703378,
        "median_basin_lift": 1.6804800652739778
      }
    }
  }
}
```

## Core Feature Summary

```text
  H                 feature_set  anchor_count  anchor_seed    n                                                                                             used_features  nonempty_cells  empty_cells  nonempty_rate  singleton_cells  singleton_rate  median_count_nonempty  same_cell_mae  global_mae  time_bucket_mae  random_cell_mae  same_cell_spearman  global_spearman  time_bucket_spearman  random_cell_spearman  between_cell_variance_ratio  top_cell_basin_rate  global_basin_rate  top_cell_basin_lift  top_cells  beats_time_mae  beats_random_mae  beats_time_spearman  beats_random_spearman  positive_enrichment
128 behavior_core_no_score_time            16           11 1380 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              15            1       0.937500                1        0.066667                   63.0       0.005514    0.005484         0.003975         0.005546           -0.055471              0.0              0.677950             -0.006874                     0.021235             0.666667           0.469565             1.419753          3           False              True                False                  False                 True
128 behavior_core_no_score_time            16           23 1380 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              16            0       1.000000                0        0.000000                   69.0       0.005523    0.005484         0.003975         0.005513           -0.020286              0.0              0.677950              0.036643                     0.035010             0.586957           0.469565             1.250000          4           False             False                False                  False                 True
128 behavior_core_no_score_time            16           37 1380 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              16            0       1.000000                0        0.000000                   61.0       0.005478    0.005484         0.003975         0.005501            0.015324              0.0              0.677950              0.008718                     0.031930             0.573099           0.469565             1.220489          4           False              True                False                   True                 True
128 behavior_core_no_score_time            32           11 1380 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              29            3       0.906250                2        0.068966                   48.0       0.005474    0.005484         0.003975         0.005551            0.055927              0.0              0.677950              0.022744                     0.060159             0.684932           0.469565             1.458650          6           False              True                False                   True                 True
128 behavior_core_no_score_time            32           23 1380 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              28            4       0.875000                0        0.000000                   34.0       0.005437    0.005484         0.003975         0.005518            0.086843              0.0              0.677950              0.092458                     0.076336             0.687075           0.469565             1.463215          6           False              True                False                  False                 True
128 behavior_core_no_score_time            32           37 1380 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              29            3       0.906250                1        0.034483                   31.0       0.005455    0.005484         0.003975         0.005492            0.095013              0.0              0.677950              0.083922                     0.057012             0.661017           0.469565             1.407721          6           False              True                False                   True                 True
128 behavior_core_no_score_time            64           11 1380 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              50           14       0.781250                6        0.120000                   17.0       0.005433    0.005484         0.003975         0.005585            0.069773              0.0              0.677950             -0.011551                     0.099769             0.621795           0.469565             1.324193         10           False              True                False                   True                 True
128 behavior_core_no_score_time            64           23 1380 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              50           14       0.781250                3        0.060000                   19.5       0.005389    0.005484         0.003975         0.005510            0.163415              0.0              0.677950              0.144033                     0.119343             0.668750           0.469565             1.424190         10           False              True                False                   True                 True
128 behavior_core_no_score_time            64           37 1380 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              54           10       0.843750                6        0.111111                   16.5       0.005394    0.005484         0.003975         0.005601            0.142219              0.0              0.677950              0.089154                     0.092930             0.663158           0.469565             1.412281         11           False              True                False                   True                 True
128 behavior_core_no_score_time           128           11 1380 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              82           46       0.640625               11        0.134146                   10.5       0.005463    0.005484         0.003975         0.005600            0.166812              0.0              0.677950              0.083135                     0.100252             0.676768           0.469565             1.441264         17           False              True                False                   True                 True
128 behavior_core_no_score_time           128           23 1380 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              92           36       0.718750               12        0.130435                    7.5       0.005305    0.005484         0.003975         0.005510            0.211152              0.0              0.677950              0.154011                     0.169027             0.682353           0.469565             1.453159         19           False              True                False                   True                 True
128 behavior_core_no_score_time           128           37 1380 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              89           39       0.695312                9        0.101124                    8.0       0.005324    0.005484         0.003975         0.005611            0.177799              0.0              0.677950              0.101908                     0.133450             0.712329           0.469565             1.516996         18           False              True                False                   True                 True
256 behavior_core_no_score_time            16           11 1780 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              16            0       1.000000                0        0.000000                   73.0       0.007418    0.007921         0.005509         0.007432            0.155422              0.0              0.656125              0.127740                     0.118811             0.575581           0.391011             1.472033          4           False              True                False                   True                 True
256 behavior_core_no_score_time            16           23 1780 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              16            0       1.000000                0        0.000000                   96.0       0.007376    0.007921         0.005509         0.007361            0.197643              0.0              0.656125              0.160418                     0.127808             0.564784           0.391011             1.444419          4           False             False                False                   True                 True
256 behavior_core_no_score_time            16           37 1780 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              16            0       1.000000                0        0.000000                   98.5       0.007302    0.007921         0.005509         0.007331            0.197202              0.0              0.656125              0.145121                     0.138399             0.549051           0.391011             1.404181          4           False              True                False                   True                 True
256 behavior_core_no_score_time            32           11 1780 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              32            0       1.000000                1        0.031250                   32.5       0.007407    0.007921         0.005509         0.007449            0.217482              0.0              0.656125              0.133814                     0.128651             0.643868           0.391011             1.646674          7           False              True                False                   True                 True
256 behavior_core_no_score_time            32           23 1780 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              29            3       0.906250                0        0.000000                   43.0       0.007341    0.007921         0.005509         0.007341            0.227145              0.0              0.656125              0.165231                     0.139537             0.597426           0.391011             1.527901          6           False             False                False                   True                 True
256 behavior_core_no_score_time            32           37 1780 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              31            1       0.968750                2        0.064516                   41.0       0.007381    0.007921         0.005509         0.007294            0.210960              0.0              0.656125              0.189752                     0.135435             0.623153           0.391011             1.593695          7           False             False                False                   True                 True
256 behavior_core_no_score_time            64           11 1780 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              59            5       0.921875                2        0.033898                   16.0       0.007337    0.007921         0.005509         0.007430            0.213281              0.0              0.656125              0.185184                     0.154313             0.685185           0.391011             1.752341         12           False              True                False                   True                 True
256 behavior_core_no_score_time            64           23 1780 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              54           10       0.843750                3        0.055556                   12.0       0.007264    0.007921         0.005509         0.007266            0.269025              0.0              0.656125              0.219257                     0.161943             0.606498           0.391011             1.551102         11           False              True                False                   True                 True
256 behavior_core_no_score_time            64           37 1780 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              55            9       0.859375                2        0.036364                   16.0       0.007259    0.007921         0.005509         0.007230            0.251867              0.0              0.656125              0.221897                     0.160638             0.660000           0.391011             1.687931         11           False             False                False                   True                 True
256 behavior_core_no_score_time           128           11 1780 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window             101           27       0.789062               11        0.108911                    9.0       0.007238    0.007921         0.005509         0.007359            0.263679              0.0              0.656125              0.226462                     0.192069             0.655329           0.391011             1.675985         21           False              True                False                   True                 True
256 behavior_core_no_score_time           128           23 1780 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              94           34       0.734375               10        0.106383                    9.0       0.007217    0.007921         0.005509         0.007316            0.287069              0.0              0.656125              0.226467                     0.188350             0.663223           0.391011             1.696174         19           False              True                False                   True                 True
256 behavior_core_no_score_time           128           37 1780 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window             103           25       0.804688                8        0.077670                    8.0       0.007172    0.007921         0.005509         0.007275            0.277450              0.0              0.656125              0.233797                     0.204727             0.668874           0.391011             1.710626         21           False              True                False                   True                 True
384 behavior_core_no_score_time            16           11 2680 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              16            0       1.000000                0        0.000000                   79.5       0.010367    0.011060         0.008213         0.010682            0.164754              0.0              0.618246              0.077388                     0.078930             0.714549           0.485075             1.473070          4           False              True                False                   True                 True
384 behavior_core_no_score_time            16           23 2680 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              15            1       0.937500                0        0.000000                  124.0       0.010704    0.011060         0.008213         0.010557            0.056681              0.0              0.618246              0.090937                     0.064309             0.720126           0.485075             1.484567          3           False             False                False                  False                 True
384 behavior_core_no_score_time            16           37 2680 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              15            1       0.937500                0        0.000000                  144.0       0.010029    0.011060         0.008213         0.010498            0.271139              0.0              0.618246              0.166118                     0.125348             0.737143           0.485075             1.519648          3           False              True                False                   True                 True
384 behavior_core_no_score_time            32           11 2680 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              28            4       0.875000                3        0.107143                   31.0       0.009569    0.011060         0.008213         0.010393            0.370918              0.0              0.618246              0.207112                     0.196855             0.718504           0.485075             1.481224          6           False              True                False                   True                 True
384 behavior_core_no_score_time            32           23 2680 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              28            4       0.875000                1        0.035714                   50.5       0.010317    0.011060         0.008213         0.010336            0.209171              0.0              0.618246              0.216441                     0.099742             0.707368           0.485075             1.458267          6           False              True                False                  False                 True
384 behavior_core_no_score_time            32           37 2680 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              28            4       0.875000                1        0.035714                   52.5       0.009806    0.011060         0.008213         0.010442            0.326222              0.0              0.618246              0.212684                     0.166755             0.693512           0.485075             1.429702          6           False              True                False                   True                 True
384 behavior_core_no_score_time            64           11 2680 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              47           17       0.734375                6        0.127660                   15.0       0.009457    0.011060         0.008213         0.010388            0.392157              0.0              0.618246              0.234009                     0.219556             0.737374           0.485075             1.520124         10           False              True                False                   True                 True
384 behavior_core_no_score_time            64           23 2680 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              47           17       0.734375                2        0.042553                   24.0       0.009761    0.011060         0.008213         0.010354            0.361472              0.0              0.618246              0.225274                     0.186344             0.694986           0.485075             1.432741         10           False              True                False                   True                 True
384 behavior_core_no_score_time            64           37 2680 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              46           18       0.718750                4        0.086957                   25.5       0.009627    0.011060         0.008213         0.010341            0.345341              0.0              0.618246              0.204698                     0.193581             0.662757           0.485075             1.366298         10           False              True                False                   True                 True
384 behavior_core_no_score_time           128           11 2680 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              73           55       0.570312               12        0.164384                    6.0       0.009410    0.011060         0.008213         0.010421            0.418973              0.0              0.618246              0.246865                     0.238435             0.767143           0.485075             1.581495         15           False              True                False                   True                 True
384 behavior_core_no_score_time           128           23 2680 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              77           51       0.601562               11        0.142857                   10.0       0.009394    0.011060         0.008213         0.010352            0.400231              0.0              0.618246              0.259421                     0.241501             0.751613           0.485075             1.549479         16           False              True                False                   True                 True
384 behavior_core_no_score_time           128           37 2680 stable_rank,kernel_rank,separation_sp,collision_rate,f_active,unique_predictions,edges,accept_rate_window              80           48       0.625000               13        0.162500                    9.5       0.009507    0.011060         0.008213         0.010329            0.406322              0.0              0.618246              0.260985                     0.220360             0.752039           0.485075             1.550358         16           False              True                False                   True                 True
```

## Interpretation

- The cell map has weak signal, but not enough to treat cells as a reliable pointer substrate.
