# Phase D9.2 Multi-Objective Search Design

Status: design-ready, not yet implemented.

## Summary

D9 found a robust H=384 specialist checkpoint:

```text
seed2042_improved_v1
smooth   positive
accuracy positive
echo     neutral
unigram  negative
```

D9.1 showed that post-hoc local repair can reduce the unigram regression only weakly. D9.2 should therefore make unigram part of the search objective from the start, rather than trying to repair it after a smooth-only climb.

Target question:

```text
Can multi-objective acceptance find a candidate that preserves the D9 smooth/accuracy gain
while avoiding or strongly reducing unigram regression?
```

## Implementation Changes

Add a new mode to `instnct-core/examples/d9_direct_landscape.rs`:

```text
--mode multi-objective-climb
```

Inputs:

```text
--checkpoint <baseline.ckpt>
--repair-start <start.ckpt>
--H 384
--mutation-types edge,threshold
--radii 4,8,16
--mo-climbers <N>
--mo-steps <N>
--mo-eval-seeds <csv>
--mo-export-top <N>
```

Default start checkpoint:

```text
output/phase_d9_0y_seed2042_improved_v1_candidate_20260429/seed2042_improved_v1.ckpt
```

Baseline:

```text
output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt
```

Mutation scope:

```text
edge,threshold only
```

No polarity, channel, or projection mutation in D9.2a.

## Multi-Objective Acceptance

Evaluate each proposal on:

```text
smooth
accuracy
echo
unigram
```

Use baseline-relative deltas:

```text
smooth_delta
accuracy_delta
echo_delta
unigram_delta
```

Candidate constraints:

```text
smooth_delta   >= +0.0120
accuracy_delta >= +0.0020
abs(echo_delta) <= 0.0010
```

Composite objective:

```text
mo_score =
  1.00 * smooth_delta
+ 0.50 * accuracy_delta
+ 1.50 * max(unigram_delta, -0.0120)
- 0.25 * abs(echo_delta)
```

Acceptance rule:

```text
accept if:
  constraints pass
  AND mo_score >= current_mo_score - 0.00025
```

Reasoning:

- Smooth remains the primary capability signal.
- Accuracy is retained as a second positive task view.
- Unigram gets enough weight to prevent silent regression.
- Echo is a guard, not an objective to optimize.
- The small negative tolerance keeps the climb from becoming completely frozen.

## D9.2a Microprobe

Run shape:

```powershell
target\release\examples\d9_direct_landscape.exe `
  --checkpoint output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_2042\final.ckpt `
  --repair-start output\phase_d9_0y_seed2042_improved_v1_candidate_20260429\seed2042_improved_v1.ckpt `
  --H 384 `
  --mode multi-objective-climb `
  --mutation-types edge,threshold `
  --radii 4,8,16 `
  --mo-climbers 12 `
  --mo-steps 40 `
  --eval-len 4000 `
  --mo-eval-seeds 960001,960002,960003,960004 `
  --mo-export-top 8 `
  --out output\phase_d9_2a_multi_objective_microprobe_20260429
```

Runtime guard:

- First run 2 climbers x 10 steps as a timing probe.
- If projected runtime for 12 x 40 exceeds 45 minutes, reduce to 8 climbers x 25 steps.
- Do not increase eval seeds until D9.2a shows a signal.

Outputs:

```text
multi_objective_paths.csv
multi_objective_candidates.csv
candidates/top_<N>.ckpt
D9_2A_MULTI_OBJECTIVE_REPORT.md
```

## Verdicts

```text
D9_2_MULTI_OBJECTIVE_SUCCESS
  At least one exported candidate keeps smooth/accuracy, echo-safe,
  and unigram_delta >= -0.0044 in microprobe and confirm.

D9_2_FULL_GENERALIST_FOUND
  Same, but unigram lower95 >= 0.0 on confirm.

D9_2_TRADEOFF_FRONTIER_CONFIRMED
  Pareto frontier exists, but unigram repair requires giving up smooth/accuracy.

D9_2_NO_SIGNAL
  No candidate improves unigram beyond D9.1b while retaining smooth/accuracy.
```

## Confirm Gate

Only run D9.2b confirm if D9.2a exports at least one candidate with:

```text
smooth_delta >= +0.0120
accuracy_delta >= +0.0020
abs(echo_delta) <= 0.0010
unigram_delta > -0.008735006
```

D9.2b confirm:

```text
top candidates = up to 8
eval_lens = 4000,16000
eval_seeds = 970001..970030
metrics = smooth,accuracy,echo,unigram
```

## Assumptions

- D9.2 is a new research section, not beta.7 release work.
- `seed2042_improved_v1` remains a specialist checkpoint unless D9.2 confirm proves otherwise.
- D9.2a uses the same corpus and VCBP table as D9.0/D9.1.
- Generated output data stays uncommitted by default.
- Only code and compact research docs should be committed.
