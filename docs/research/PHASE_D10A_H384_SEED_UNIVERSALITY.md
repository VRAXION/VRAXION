# Phase D10a H=384 Seed Universality Scout

Date: 2026-04-29

## Goal

Determine whether the beta.8 edge+threshold generalist mechanism is seed2042-local or repeats across H=384 D7 baseline checkpoints.

## Checkpoints Tested

All available H=384 D7 baseline checkpoints under `output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/` were included:

| Seed | Checkpoint |
| --- | --- |
| seed_42 | `output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_42/final.ckpt` |
| seed_1042 | `output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_1042/final.ckpt` |
| seed_2042 | `output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt` |
| seed_3042 | `output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_3042/final.ckpt` |
| seed_4042 | `output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_4042/final.ckpt` |

No additional H=384 baseline seed directories were present.

## Commands And Settings

Build:

```powershell
cargo build -p instnct-core --example d9_direct_landscape --release
```

Smoke:

```powershell
target\release\examples\d9_direct_landscape.exe --checkpoints output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_42\final.ckpt,output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_2042\final.ckpt --H 384 --mode scaling-universality-scout --eval-len 1000 --mo-eval-seeds 980001,980002 --mo-climbers 2 --mo-steps 2 --radii 4,8 --out output\phase_d10a_h384_seed_universality_20260429\smoke
```

Scout:

```powershell
target\release\examples\d9_direct_landscape.exe --checkpoints output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_42\final.ckpt,output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_1042\final.ckpt,output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_2042\final.ckpt,output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_3042\final.ckpt,output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_4042\final.ckpt --H 384 --mode scaling-universality-scout --eval-len 4000 --mo-eval-seeds 980001,980002,980003,980004 --mo-climbers 4 --mo-steps 5 --radii 4,8,16 --mutation-types edge,threshold --out output\phase_d10a_h384_seed_universality_20260429\scout
```

The suggested 8x20 proposal budget was not used because the smoke timing extrapolated to hours. The completed scout used 20 proposals per checkpoint, edge+threshold only, radii 4/8/16, 4 eval seeds, eval_len 4000.

Runtime:

| Run | Runtime | Rows |
| --- | ---: | ---: |
| smoke | 9.4s | 8 |
| scout | 751.9s | 100 |

## Results

Output root: `output/phase_d10a_h384_seed_universality_20260429/`

Scout artifacts:

| Artifact | Path |
| --- | --- |
| Candidate rows | `output/phase_d10a_h384_seed_universality_20260429/scout/candidate_summary.csv` |
| Universality matrix | `output/phase_d10a_h384_seed_universality_20260429/scout/universality_matrix.csv` |
| Run summary | `output/phase_d10a_h384_seed_universality_20260429/scout/run_summary.json` |

Strict-pass criteria in code:

```text
smooth_delta >= 0.0120
accuracy_delta >= 0.0020
abs(echo_delta) <= 0.0010
FULL_GENERALIST additionally requires unigram_delta >= 0.0.
```

Top row by checkpoint:

| Seed | Strict passes | Positive unigram rows | Best class | Best mo_score | smooth_delta | accuracy_delta | echo_delta | unigram_delta |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| seed_42 | 0 | 9 | FAIL_RETAIN | 0.009291477984 | 0.000007017675 | -0.000687500000 | 0.000000000000 | 0.006418806873 |
| seed_1042 | 0 | 13 | FAIL_RETAIN | 0.000280874229 | 0.000002315212 | 0.000000000000 | 0.000000000000 | 0.000185706011 |
| seed_2042 | 0 | 10 | FAIL_RETAIN | 0.121962236165 | -0.011935576305 | -0.001062500000 | 0.043750000000 | 0.096911041647 |
| seed_3042 | 0 | 6 | FAIL_RETAIN | 0.000009501319 | 0.000000000000 | 0.000000000000 | 0.000000000000 | 0.000006334213 |
| seed_4042 | 0 | 11 | FAIL_RETAIN | 0.059940264813 | -0.032600335525 | -0.006375000000 | 0.043750000000 | 0.071110400225 |

Best near-pass row:

| Seed | Step | Radius | Mutation | Class | mo_score | smooth_delta | accuracy_delta | echo_delta | unigram_delta | Reason |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| seed_2042 | 7 | 8 | threshold | FAIL_RETAIN | 0.013444672036 | 0.008444891451 | 0.002062500000 | 0.000000000002 | 0.002645687057 | Passes accuracy, echo, and unigram; misses smooth threshold by 0.003555108549. |

All 100 scout rows were `FAIL_RETAIN`; accepted row count was 0. The binary's internal strict-count verdict was `NO_GENERAL_BASIN`.

## Verdict

`LOCAL_H384_BASIN_ONLY`

The scout did not find a strict `FULL_GENERALIST` or `MULTI_OBJECTIVE_SUCCESS` row on any H=384 checkpoint, so there is no evidence for a universal basin across seeds at this budget. The only meaningful near-pass is seed2042-local: it satisfies accuracy, echo, and positive unigram but does not clear the smooth threshold. Other seeds show positive unigram rows, but not a comparable near-pass to the strict multi-objective gate.

## Next Gate

Run a focused D10b seed2042-local falsification gate rather than a broad universality claim:

1. Re-test the seed2042 near-pass neighborhood with a higher local proposal budget and eval_len 16000.
2. Include seed_42 and seed_4042 as controls because they showed positive unigram or high raw mo_score without satisfying the gate.
3. Promote only if a strict pass appears and survives endpoint robustness plus causal-diff confirmation.
