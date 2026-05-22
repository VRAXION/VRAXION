# PILOT_SENSOR_OPERATOR_TRAJECTORY_001 Result

## Goal

Test whether the operator-state/collapse formalism adds temporal Pilot value beyond the final v0 mapper.

This probe separates provisional hypotheses from hard commits. Hard execution is allowed only at `END` or explicit commit signals in the main arm.

## Metrics

```json
{
  "final_only_operator_mapper": {
    "ambiguous_hold_accuracy": 1.0,
    "correction_recollapse_accuracy": 1.0,
    "entropy_at_commit": 0.41795860775833726,
    "false_execution_rate": 0.0,
    "final_action_accuracy": 0.9166666666666666,
    "mean_commit_step": 1.5,
    "mention_suppression_accuracy": 1.0,
    "negation_suppression_accuracy": 0.5,
    "per_step_expected_action_accuracy": 0.3235294117647059,
    "premature_hard_commit_rate": 0.0,
    "purity_at_commit": 0.8034896573998862,
    "state_margin_at_commit": 0.8168736626762279,
    "unknown_reject_accuracy": 1.0,
    "weak_hold_accuracy": 1.0
  },
  "no_correction_operator": {
    "ambiguous_hold_accuracy": 1.0,
    "correction_recollapse_accuracy": 0.0,
    "entropy_at_commit": 0.5652103811388252,
    "false_execution_rate": 0.0,
    "final_action_accuracy": 0.8333333333333334,
    "mean_commit_step": 1.8,
    "mention_suppression_accuracy": 1.0,
    "negation_suppression_accuracy": 1.0,
    "per_step_expected_action_accuracy": 0.8823529411764706,
    "premature_hard_commit_rate": 0.0,
    "purity_at_commit": 0.7157279655590195,
    "state_margin_at_commit": 0.7203964141789403,
    "unknown_reject_accuracy": 1.0,
    "weak_hold_accuracy": 1.0
  },
  "no_mention_suppressor": {
    "ambiguous_hold_accuracy": 1.0,
    "correction_recollapse_accuracy": 1.0,
    "entropy_at_commit": 0.44284891269870336,
    "false_execution_rate": 0.08333333333333333,
    "final_action_accuracy": 0.9166666666666666,
    "mean_commit_step": 1.8333333333333333,
    "mention_suppression_accuracy": 0.0,
    "negation_suppression_accuracy": 1.0,
    "per_step_expected_action_accuracy": 0.9411764705882353,
    "premature_hard_commit_rate": 0.0,
    "purity_at_commit": 0.783535747061438,
    "state_margin_at_commit": 0.7900740772558494,
    "unknown_reject_accuracy": 1.0,
    "weak_hold_accuracy": 1.0
  },
  "no_negation_operator": {
    "ambiguous_hold_accuracy": 1.0,
    "correction_recollapse_accuracy": 1.0,
    "entropy_at_commit": 0.4570864102423534,
    "false_execution_rate": 0.08333333333333333,
    "final_action_accuracy": 0.8333333333333334,
    "mean_commit_step": 1.6,
    "mention_suppression_accuracy": 1.0,
    "negation_suppression_accuracy": 0.0,
    "per_step_expected_action_accuracy": 0.8529411764705882,
    "premature_hard_commit_rate": 0.0,
    "purity_at_commit": 0.7822173242549766,
    "state_margin_at_commit": 0.79543441401195,
    "unknown_reject_accuracy": 1.0,
    "weak_hold_accuracy": 1.0
  },
  "no_provisional_commit_delay": {
    "ambiguous_hold_accuracy": 1.0,
    "correction_recollapse_accuracy": 1.0,
    "entropy_at_commit": 0.40869917120276045,
    "false_execution_rate": 0.0,
    "final_action_accuracy": 1.0,
    "mean_commit_step": 1.8,
    "mention_suppression_accuracy": 1.0,
    "negation_suppression_accuracy": 1.0,
    "per_step_expected_action_accuracy": 0.6470588235294118,
    "premature_hard_commit_rate": 0.35294117647058826,
    "purity_at_commit": 0.800817298138658,
    "state_margin_at_commit": 0.8061534088360515,
    "unknown_reject_accuracy": 1.0,
    "weak_hold_accuracy": 1.0
  },
  "no_weak_ambiguity_operator": {
    "ambiguous_hold_accuracy": 1.0,
    "correction_recollapse_accuracy": 1.0,
    "entropy_at_commit": 0.44284891269870336,
    "false_execution_rate": 0.08333333333333333,
    "final_action_accuracy": 0.9166666666666666,
    "mean_commit_step": 1.8333333333333333,
    "mention_suppression_accuracy": 1.0,
    "negation_suppression_accuracy": 1.0,
    "per_step_expected_action_accuracy": 0.9117647058823529,
    "premature_hard_commit_rate": 0.0,
    "purity_at_commit": 0.783535747061438,
    "state_margin_at_commit": 0.7900740772558494,
    "unknown_reject_accuracy": 1.0,
    "weak_hold_accuracy": 0.0
  },
  "operator_trajectory": {
    "ambiguous_hold_accuracy": 1.0,
    "correction_recollapse_accuracy": 1.0,
    "entropy_at_commit": 0.40869917120276045,
    "false_execution_rate": 0.0,
    "final_action_accuracy": 1.0,
    "mean_commit_step": 1.8,
    "mention_suppression_accuracy": 1.0,
    "negation_suppression_accuracy": 1.0,
    "per_step_expected_action_accuracy": 1.0,
    "premature_hard_commit_rate": 0.0,
    "purity_at_commit": 0.800817298138658,
    "state_margin_at_commit": 0.8061534088360515,
    "unknown_reject_accuracy": 1.0,
    "weak_hold_accuracy": 1.0
  }
}
```

## Robustness

```json
{
  "10pct": {
    "false_execution_under_jitter": 0.0,
    "min_purity_by_case": {
      "ambiguous_add_mul": 0.3245950580811747,
      "clean_add": 0.6216100949078632,
      "clean_mul": 0.623976262387234,
      "correction_add_to_mul": 0.854535163820419,
      "correction_mul_to_add": 0.847346534603986,
      "mention_trap": 0.995627330116575,
      "multistep_unsupported": 0.3500644282930118,
      "negated_add": 0.4854352975303311,
      "negated_add_then_mul": 0.5759954281623134,
      "no_evidence": 0.25,
      "unknown_div": 0.6181699901110242,
      "weak_add": 0.44613819253670506
    },
    "min_state_margin_by_case": {
      "ambiguous_add_mul": 0.00035122892099270775,
      "clean_add": 0.616601746420191,
      "clean_mul": 0.619458323893553,
      "correction_add_to_mul": 0.8680000940346023,
      "correction_mul_to_add": 0.8677733456635341,
      "mention_trap": 0.995719565306483,
      "multistep_unsupported": 0.13092218264604527,
      "negated_add": 0.4583738977558418,
      "negated_add_then_mul": 0.3931165237649892,
      "no_evidence": 0.0,
      "unknown_div": 0.6136418763862048,
      "weak_add": 0.0035505028082308265
    },
    "pass_rate_under_jitter": 0.992,
    "premature_commit_under_jitter": 0.0,
    "samples": 500,
    "weakest_case": [
      [
        "negated_add_then_mul",
        4
      ]
    ]
  },
  "20pct": {
    "false_execution_under_jitter": 0.0,
    "min_purity_by_case": {
      "ambiguous_add_mul": 0.32227090922089013,
      "clean_add": 0.5388298981405242,
      "clean_mul": 0.5491923465128921,
      "correction_add_to_mul": 0.7266260928345387,
      "correction_mul_to_add": 0.7477365770679453,
      "mention_trap": 0.9918892367511578,
      "multistep_unsupported": 0.3301799881080787,
      "negated_add": 0.3913459178635086,
      "negated_add_then_mul": 0.4986948167555773,
      "no_evidence": 0.25,
      "unknown_div": 0.5495833268868833,
      "weak_add": 0.42518047071072723
    },
    "min_state_margin_by_case": {
      "ambiguous_add_mul": 8.650355066680548e-05,
      "clean_add": 0.49600774767547623,
      "clean_mul": 0.504567403915293,
      "correction_add_to_mul": 0.737250850336649,
      "correction_mul_to_add": 0.7598509581580984,
      "mention_trap": 0.9920072003379435,
      "multistep_unsupported": 0.0017261762113353618,
      "negated_add": 0.2130329485020523,
      "negated_add_then_mul": 0.03629167950318285,
      "no_evidence": 0.0,
      "unknown_div": 0.5081958007368318,
      "weak_add": 0.00321674252968307
    },
    "pass_rate_under_jitter": 0.75,
    "premature_commit_under_jitter": 0.0,
    "samples": 500,
    "weakest_case": [
      [
        "negated_add_then_mul",
        56
      ],
      [
        "clean_add",
        39
      ],
      [
        "unknown_div",
        30
      ],
      [
        "clean_mul",
        22
      ]
    ]
  },
  "30pct": {
    "false_execution_under_jitter": 0.006,
    "min_purity_by_case": {
      "ambiguous_add_mul": 0.31806796822844924,
      "clean_add": 0.46484192596120166,
      "clean_mul": 0.45770768026282965,
      "correction_add_to_mul": 0.622695428925513,
      "correction_mul_to_add": 0.6808464890209623,
      "mention_trap": 0.9852675199971043,
      "multistep_unsupported": 0.33129452598226866,
      "negated_add": 0.3260302968213506,
      "negated_add_then_mul": 0.48587607724144455,
      "no_evidence": 0.25,
      "unknown_div": 0.47614113657139256,
      "weak_add": 0.3692221737950602
    },
    "min_state_margin_by_case": {
      "ambiguous_add_mul": 0.0015406849936878686,
      "clean_add": 0.3254789972644701,
      "clean_mul": 0.338417345762335,
      "correction_add_to_mul": 0.6576825248637723,
      "correction_mul_to_add": 0.6362072324414789,
      "mention_trap": 0.9857517916008686,
      "multistep_unsupported": 0.001801376314144787,
      "negated_add": 0.01024582037327737,
      "negated_add_then_mul": 0.0013262379096393717,
      "no_evidence": 0.0,
      "unknown_div": 0.3598680542246324,
      "weak_add": 0.0021301401647209617
    },
    "pass_rate_under_jitter": 0.472,
    "premature_commit_under_jitter": 0.0,
    "samples": 500,
    "weakest_case": [
      [
        "negated_add_then_mul",
        136
      ],
      [
        "clean_mul",
        109
      ],
      [
        "clean_add",
        91
      ],
      [
        "unknown_div",
        83
      ],
      [
        "ambiguous_add_mul",
        2
      ]
    ]
  }
}
```

## Parameter Search Diagnostic

```json
{
  "best": [
    {
      "metrics": {
        "ambiguous_hold_accuracy": 1.0,
        "correction_recollapse_accuracy": 1.0,
        "entropy_at_commit": 0.35779294987172683,
        "false_execution_rate": 0.0,
        "final_action_accuracy": 1.0,
        "mean_commit_step": 1.8,
        "mention_suppression_accuracy": 1.0,
        "negation_suppression_accuracy": 1.0,
        "per_step_expected_action_accuracy": 1.0,
        "premature_hard_commit_rate": 0.0,
        "purity_at_commit": 0.8370579157694689,
        "state_margin_at_commit": 0.8568953503145662,
        "unknown_reject_accuracy": 1.0,
        "weak_hold_accuracy": 1.0
      },
      "operators": {
        "ADD": [
          2.4211,
          0.5052,
          0.5329,
          0.8243
        ],
        "AMBIGUITY": [
          1.1047,
          1.3793,
          0.8249,
          2.0251
        ],
        "CORR_ADD": [
          3.2827,
          0.1101,
          0.0942,
          0.4143
        ],
        "CORR_MUL": [
          0.095,
          4.439,
          0.1147,
          0.3135
        ],
        "CORR_UNKNOWN": [
          0.1023,
          0.1183,
          4.0431,
          0.3304
        ],
        "MENTION": [
          0.0575,
          0.043,
          0.0541,
          4.0942
        ],
        "MUL": [
          0.4979,
          2.5961,
          0.479,
          0.7597
        ],
        "MULTI_STEP": [
          1.3003,
          0.9724,
          0.8128,
          2.9978
        ],
        "NOT_ADD": [
          0.0528,
          0.9237,
          0.8327,
          1.5287
        ],
        "NOT_MUL": [
          0.9896,
          0.0599,
          0.8566,
          1.4623
        ],
        "NOT_UNKNOWN": [
          1.1254,
          1.168,
          0.0404,
          1.2335
        ],
        "UNKNOWN": [
          0.3648,
          0.4292,
          2.6365,
          0.8109
        ],
        "WEAK": [
          0.6504,
          0.6025,
          1.1385,
          2.5918
        ]
      },
      "score": [
        0.0,
        0.0,
        -1.0,
        -0.0,
        -1.0
      ]
    },
    {
      "metrics": {
        "ambiguous_hold_accuracy": 1.0,
        "correction_recollapse_accuracy": 1.0,
        "entropy_at_commit": 0.4247393841781951,
        "false_execution_rate": 0.0,
        "final_action_accuracy": 1.0,
        "mean_commit_step": 1.8,
        "mention_suppression_accuracy": 1.0,
        "negation_suppression_accuracy": 1.0,
        "per_step_expected_action_accuracy": 1.0,
        "premature_hard_commit_rate": 0.0,
        "purity_at_commit": 0.7889979522125112,
        "state_margin_at_commit": 0.790310634210091,
        "unknown_reject_accuracy": 1.0,
        "weak_hold_accuracy": 1.0
      },
      "operators": {
        "ADD": [
          2.1765,
          0.5016,
          0.3947,
          0.9374
        ],
        "AMBIGUITY": [
          1.0801,
          1.1349,
          0.909,
          1.8958
        ],
        "CORR_ADD": [
          4.4879,
          0.1168,
          0.1124,
          0.2869
        ],
        "CORR_MUL": [
          0.117,
          3.3485,
          0.1108,
          0.3975
        ],
        "CORR_UNKNOWN": [
          0.1063,
          0.0961,
          3.32,
          0.2919
        ],
        "MENTION": [
          0.0546,
          0.0462,
          0.0415,
          4.7789
        ],
        "MUL": [
          0.46,
          2.4513,
          0.3833,
          0.9571
        ],
        "MULTI_STEP": [
          1.2804,
          0.8891,
          0.8205,
          2.7934
        ],
        "NOT_ADD": [
          0.0481,
          1.0927,
          1.1839,
          1.3469
        ],
        "NOT_MUL": [
          0.9566,
          0.0439,
          0.9006,
          1.1184
        ],
        "NOT_UNKNOWN": [
          0.8442,
          0.885,
          0.0568,
          1.3192
        ],
        "UNKNOWN": [
          0.525,
          0.4174,
          2.2388,
          1.0473
        ],
        "WEAK": [
          0.6704,
          0.5909,
          0.8039,
          2.4229
        ]
      },
      "score": [
        0.0,
        0.0,
        -1.0,
        -0.0,
        -1.0
      ]
    },
    {
      "metrics": {
        "ambiguous_hold_accuracy": 1.0,
        "correction_recollapse_accuracy": 1.0,
        "entropy_at_commit": 0.48651443074027106,
        "false_execution_rate": 0.0,
        "final_action_accuracy": 1.0,
        "mean_commit_step": 1.8,
        "mention_suppression_accuracy": 1.0,
        "negation_suppression_accuracy": 1.0,
        "per_step_expected_action_accuracy": 1.0,
        "premature_hard_commit_rate": 0.0,
        "purity_at_commit": 0.7440498759932311,
        "state_margin_at_commit": 0.7250139566532154,
        "unknown_reject_accuracy": 1.0,
        "weak_hold_accuracy": 1.0
      },
      "operators": {
        "ADD": [
          2.1228,
          0.4081,
          0.4532,
          0.9717
        ],
        "AMBIGUITY": [
          1.2187,
          1.3112,
          1.1176,
          1.7482
        ],
        "CORR_ADD": [
          4.0614,
          0.0897,
          0.0998,
          0.3197
        ],
        "CORR_MUL": [
          0.085,
          3.6,
          0.1072,
          0.3235
        ],
        "CORR_UNKNOWN": [
          0.088,
          0.0946,
          3.5418,
          0.4178
        ],
        "MENTION": [
          0.0412,
          0.0508,
          0.0438,
          4.0619
        ],
        "MUL": [
          0.365,
          2.3728,
          0.5231,
          0.9404
        ],
        "MULTI_STEP": [
          0.9986,
          1.0328,
          0.814,
          2.9324
        ],
        "NOT_ADD": [
          0.0595,
          0.9622,
          0.803,
          1.4301
        ],
        "NOT_MUL": [
          0.9357,
          0.0552,
          1.1322,
          1.0574
        ],
        "NOT_UNKNOWN": [
          1.0923,
          0.9139,
          0.0464,
          1.2138
        ],
        "UNKNOWN": [
          0.5388,
          0.3778,
          2.7923,
          1.0215
        ],
        "WEAK": [
          0.5268,
          0.6595,
          1.1037,
          1.9554
        ]
      },
      "score": [
        0.0,
        0.0,
        -1.0,
        -0.0,
        -1.0
      ]
    },
    {
      "metrics": {
        "ambiguous_hold_accuracy": 1.0,
        "correction_recollapse_accuracy": 1.0,
        "entropy_at_commit": 0.4328115637341358,
        "false_execution_rate": 0.0,
        "final_action_accuracy": 1.0,
        "mean_commit_step": 1.8,
        "mention_suppression_accuracy": 1.0,
        "negation_suppression_accuracy": 1.0,
        "per_step_expected_action_accuracy": 1.0,
        "premature_hard_commit_rate": 0.0,
        "purity_at_commit": 0.7763300943080501,
        "state_margin_at_commit": 0.7625702197615124,
        "unknown_reject_accuracy": 1.0,
        "weak_hold_accuracy": 1.0
      },
      "operators": {
        "ADD": [
          2.6757,
          0.3873,
          0.4129,
          0.8769
        ],
        "AMBIGUITY": [
          1.1073,
          1.2037,
          0.8675,
          1.9524
        ],
        "CORR_ADD": [
          4.4218,
          0.0929,
          0.1186,
          0.3841
        ],
        "CORR_MUL": [
          0.0923,
          4.5815,
          0.1169,
          0.352
        ],
        "CORR_UNKNOWN": [
          0.0857,
          0.0842,
          4.5087,
          0.4043
        ],
        "MENTION": [
          0.0556,
          0.0485,
          0.0464,
          4.4129
        ],
        "MUL": [
          0.4551,
          2.2929,
          0.4542,
          0.9936
        ],
        "MULTI_STEP": [
          1.2102,
          1.2317,
          0.7324,
          2.3323
        ],
        "NOT_ADD": [
          0.045,
          0.8764,
          0.8028,
          1.1037
        ],
        "NOT_MUL": [
          0.9971,
          0.0486,
          1.1559,
          1.2367
        ],
        "NOT_UNKNOWN": [
          0.8681,
          1.1723,
          0.0415,
          1.2167
        ],
        "UNKNOWN": [
          0.3877,
          0.4607,
          2.0779,
          0.9125
        ],
        "WEAK": [
          0.6024,
          0.7288,
          1.1524,
          2.6019
        ]
      },
      "score": [
        0.0,
        0.0,
        -1.0,
        -0.0,
        -1.0
      ]
    },
    {
      "metrics": {
        "ambiguous_hold_accuracy": 1.0,
        "correction_recollapse_accuracy": 1.0,
        "entropy_at_commit": 0.34168255269579306,
        "false_execution_rate": 0.0,
        "final_action_accuracy": 1.0,
        "mean_commit_step": 1.8,
        "mention_suppression_accuracy": 1.0,
        "negation_suppression_accuracy": 1.0,
        "per_step_expected_action_accuracy": 1.0,
        "premature_hard_commit_rate": 0.0,
        "purity_at_commit": 0.8456956025123314,
        "state_margin_at_commit": 0.8613385644043791,
        "unknown_reject_accuracy": 1.0,
        "weak_hold_accuracy": 1.0
      },
      "operators": {
        "ADD": [
          2.5477,
          0.4993,
          0.5029,
          0.8054
        ],
        "AMBIGUITY": [
          0.9476,
          1.2305,
          0.9968,
          1.7447
        ],
        "CORR_ADD": [
          4.2341,
          0.0853,
          0.1151,
          0.3875
        ],
        "CORR_MUL": [
          0.0891,
          3.8429,
          0.0994,
          0.3993
        ],
        "CORR_UNKNOWN": [
          0.1077,
          0.0896,
          4.7218,
          0.4102
        ],
        "MENTION": [
          0.0452,
          0.0461,
          0.0579,
          4.7052
        ],
        "MUL": [
          0.3984,
          2.7529,
          0.5111,
          0.877
        ],
        "MULTI_STEP": [
          1.2109,
          0.9737,
          0.7066,
          2.4573
        ],
        "NOT_ADD": [
          0.0575,
          1.1771,
          0.892,
          1.3694
        ],
        "NOT_MUL": [
          0.9519,
          0.0491,
          0.8791,
          1.4152
        ],
        "NOT_UNKNOWN": [
          1.1128,
          1.1758,
          0.0497,
          1.2697
        ],
        "UNKNOWN": [
          0.4086,
          0.3742,
          2.3083,
          1.0472
        ],
        "WEAK": [
          0.5729,
          0.6021,
          0.8141,
          2.2606
        ]
      },
      "score": [
        0.0,
        0.0,
        -1.0,
        -0.0,
        -1.0
      ]
    }
  ],
  "objective_order": [
    "minimize false_execution",
    "minimize premature_hard_commit",
    "maximize final_action_accuracy",
    "maximize margin safety",
    "prefer HOLD over wrong EXEC"
  ],
  "samples": 2000
}
```

## Verdict

```json
[
  "OPERATOR_TRAJECTORY_POSITIVE",
  "COLLAPSE_DELAY_CAUSAL",
  "FINAL_ONLY_LACKS_TRAJECTORY_TIMING",
  "OPERATOR_FORM_VALID_BUT_CALIBRATION_SENSITIVE"
]
```

## Failure Examples

- `no_provisional_commit_delay` `clean_add` step `0` `ADD_CUE`: `HARD_EXEC_ADD` expected `PROVISIONAL_ADD`; final `HARD_EXEC_ADD` expected `HARD_EXEC_ADD`; failure `premature_hard_commit`.
- `no_provisional_commit_delay` `clean_mul` step `0` `MUL_CUE`: `HARD_EXEC_MUL` expected `PROVISIONAL_MUL`; final `HARD_EXEC_MUL` expected `HARD_EXEC_MUL`; failure `premature_hard_commit`.
- `no_provisional_commit_delay` `ambiguous_add_mul` step `0` `ADD_CUE`: `HARD_EXEC_ADD` expected `PROVISIONAL_ADD`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `premature_hard_commit`.
- `no_provisional_commit_delay` `correction_add_to_mul` step `0` `ADD_CUE`: `HARD_EXEC_ADD` expected `PROVISIONAL_ADD`; final `HARD_EXEC_MUL` expected `HARD_EXEC_MUL`; failure `premature_hard_commit`.
- `no_provisional_commit_delay` `correction_add_to_mul` step `1` `CORRECTION_MUL`: `HARD_EXEC_MUL` expected `PROVISIONAL_MUL`; final `HARD_EXEC_MUL` expected `HARD_EXEC_MUL`; failure `premature_hard_commit`.
- `no_provisional_commit_delay` `correction_mul_to_add` step `0` `MUL_CUE`: `HARD_EXEC_MUL` expected `PROVISIONAL_MUL`; final `HARD_EXEC_ADD` expected `HARD_EXEC_ADD`; failure `premature_hard_commit`.
- `no_provisional_commit_delay` `correction_mul_to_add` step `1` `CORRECTION_ADD`: `HARD_EXEC_ADD` expected `PROVISIONAL_ADD`; final `HARD_EXEC_ADD` expected `HARD_EXEC_ADD`; failure `premature_hard_commit`.
- `no_provisional_commit_delay` `negated_add` step `0` `ADD_CUE`: `HARD_EXEC_ADD` expected `PROVISIONAL_ADD`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `premature_hard_commit`.
- `no_provisional_commit_delay` `negated_add_then_mul` step `0` `ADD_CUE`: `HARD_EXEC_ADD` expected `PROVISIONAL_ADD`; final `HARD_EXEC_MUL` expected `HARD_EXEC_MUL`; failure `premature_hard_commit`.
- `no_provisional_commit_delay` `negated_add_then_mul` step `2` `MUL_CUE`: `HARD_EXEC_MUL` expected `PROVISIONAL_MUL`; final `HARD_EXEC_MUL` expected `HARD_EXEC_MUL`; failure `premature_hard_commit`.
- `no_provisional_commit_delay` `mention_trap` step `0` `ADD_CUE`: `HARD_EXEC_ADD` expected `PROVISIONAL_ADD`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `premature_hard_commit`.
- `no_provisional_commit_delay` `multistep_unsupported` step `0` `ADD_CUE`: `HARD_EXEC_ADD` expected `PROVISIONAL_ADD`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `premature_hard_commit`.
- `final_only_operator_mapper` `clean_add` step `0` `ADD_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `PROVISIONAL_ADD`; final `HARD_EXEC_ADD` expected `HARD_EXEC_ADD`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `clean_mul` step `0` `MUL_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `PROVISIONAL_MUL`; final `HARD_EXEC_MUL` expected `HARD_EXEC_MUL`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `weak_add` step `0` `WEAK`: `FINAL_ONLY_NO_STEP_ACTION` expected `HOLD_ASK_RESEARCH`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `weak_add` step `1` `ADD_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `HOLD_ASK_RESEARCH`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `ambiguous_add_mul` step `0` `ADD_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `PROVISIONAL_ADD`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `ambiguous_add_mul` step `1` `AMBIGUITY`: `FINAL_ONLY_NO_STEP_ACTION` expected `HOLD_ASK_RESEARCH`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `ambiguous_add_mul` step `2` `MUL_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `HOLD_ASK_RESEARCH`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `correction_add_to_mul` step `0` `ADD_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `PROVISIONAL_ADD`; final `HARD_EXEC_MUL` expected `HARD_EXEC_MUL`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `correction_add_to_mul` step `1` `CORRECTION_MUL`: `FINAL_ONLY_NO_STEP_ACTION` expected `PROVISIONAL_MUL`; final `HARD_EXEC_MUL` expected `HARD_EXEC_MUL`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `correction_mul_to_add` step `0` `MUL_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `PROVISIONAL_MUL`; final `HARD_EXEC_ADD` expected `HARD_EXEC_ADD`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `correction_mul_to_add` step `1` `CORRECTION_ADD`: `FINAL_ONLY_NO_STEP_ACTION` expected `PROVISIONAL_ADD`; final `HARD_EXEC_ADD` expected `HARD_EXEC_ADD`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `negated_add` step `0` `ADD_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `PROVISIONAL_ADD`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `negated_add` step `1` `NOT_ADD`: `FINAL_ONLY_NO_STEP_ACTION` expected `HOLD_ASK_RESEARCH`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `negated_add_then_mul` step `0` `ADD_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `PROVISIONAL_ADD`; final `HOLD_ASK_RESEARCH` expected `HARD_EXEC_MUL`; failure `negation_suppression_error`.
- `final_only_operator_mapper` `negated_add_then_mul` step `1` `NOT_ADD`: `FINAL_ONLY_NO_STEP_ACTION` expected `HOLD_ASK_RESEARCH`; final `HOLD_ASK_RESEARCH` expected `HARD_EXEC_MUL`; failure `negation_suppression_error`.
- `final_only_operator_mapper` `negated_add_then_mul` step `2` `MUL_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `PROVISIONAL_MUL`; final `HOLD_ASK_RESEARCH` expected `HARD_EXEC_MUL`; failure `negation_suppression_error`.
- `final_only_operator_mapper` `negated_add_then_mul` step `3` `END`: `HOLD_ASK_RESEARCH` expected `HARD_EXEC_MUL`; final `HOLD_ASK_RESEARCH` expected `HARD_EXEC_MUL`; failure `negation_suppression_error`.
- `final_only_operator_mapper` `mention_trap` step `0` `ADD_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `PROVISIONAL_ADD`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `mention_trap` step `1` `MENTION_CONTEXT`: `FINAL_ONLY_NO_STEP_ACTION` expected `HOLD_ASK_RESEARCH`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `unknown_div` step `0` `UNKNOWN_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `REJECT_UNKNOWN`; final `REJECT_UNKNOWN` expected `REJECT_UNKNOWN`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `multistep_unsupported` step `0` `ADD_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `PROVISIONAL_ADD`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `multistep_unsupported` step `1` `THEN`: `FINAL_ONLY_NO_STEP_ACTION` expected `HOLD_ASK_RESEARCH`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `per_step_timing_error`.
- `final_only_operator_mapper` `multistep_unsupported` step `2` `MUL_CUE`: `FINAL_ONLY_NO_STEP_ACTION` expected `HOLD_ASK_RESEARCH`; final `HOLD_ASK_RESEARCH` expected `HOLD_ASK_RESEARCH`; failure `per_step_timing_error`.
- ... 20 more in `failure_examples.jsonl`.

## Interpretation

A positive trajectory result means the operator-state formalism supports time-resolved evidence accumulation, provisional hypotheses, delayed hard commit, and re-collapse after correction in this toy command domain.

The `no_provisional_commit_delay` ablation tests whether delaying hard collapse is causal. The `final_only_operator_mapper` control tests whether final-state accuracy alone is insufficient evidence for trajectory behavior.

## Claim Boundary

Toy command domain only. No real quantum behavior, quantum hardware requirement, general NLU, full PilotPulse, production VRAXION/INSTNCT, biology, or consciousness claim.
