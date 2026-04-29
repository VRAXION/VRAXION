# Phase D10b Seed Replication Ladder

Date: 2026-04-29

Status: scaffolded, smoke-tested, and timing-probed; main ladder requires an unattended run.

## Goal

D10a found `LOCAL_H384_BASIN_ONLY` at scout budget: no strict H=384 seed-universality signal, with only a seed2042 near-pass. D10b adds a deeper replication ladder to test whether other H=384 seeds are truly dead or whether D10a was too shallow and too hard-gated.

The D10b mode does not claim a new basin. It creates the runnable gate needed to falsify or support replication.

## Implemented Mode

New `d9_direct_landscape` mode:

```text
--mode seed-replication-ladder
```

Behavior:

- Accepts multiple H=384 baseline checkpoints via `--checkpoints`.
- Uses each checkpoint as its own baseline and start state.
- Mutates only `edge,threshold`.
- Computes smooth, accuracy, echo, unigram deltas for every proposal.
- Uses a ladder score that rewards progress toward the strict multi-objective gate rather than requiring an immediate strict pass.
- Exports per-seed candidate checkpoints under `candidates/<seed>/top_<N>.ckpt`.

Outputs:

```text
replication_paths.csv
replication_candidates.csv
universality_matrix.csv
run_summary.json
D10B_SEED_REPLICATION_LADDER_REPORT.md
```

## Gate Definitions

Strict gate:

```text
smooth_delta >= +0.0120
accuracy_delta >= +0.0020
abs(echo_delta) <= 0.0010
unigram_delta >= 0.0
```

Near-strict gate:

```text
Exactly one strict gate may miss, and that miss must be within 25% of the relevant threshold.
```

Verdicts:

| verdict | meaning |
|---|---|
| `D10B_REPLICABLE_GENERALIST_BASIN` | At least 2 non-seed2042 checkpoints produce strict candidates. |
| `D10B_SEED_SENSITIVE_BUT_NOT_UNIQUE` | seed2042 plus at least 1 other seed produces strict or near-strict candidates. |
| `D10B_SEED2042_ONLY` | seed2042 has signal, all other seeds fail. |
| `D10B_NO_REPLICATION_SIGNAL` | no seed produces strict or near-strict candidates. |

## Smoke Run

Command:

```powershell
target\release\examples\d9_direct_landscape.exe --checkpoints output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_2042\final.ckpt,output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_42\final.ckpt --H 384 --mode seed-replication-ladder --eval-len 1000 --mo-eval-seeds 990001,990002 --mo-climbers 2 --mo-steps 5 --radii 4,8,16 --mutation-types edge,threshold --mo-export-top 2 --out output\phase_d10b_h384_seed_replication_ladder_20260429\smoke
```

Smoke result:

- Runtime: 20.8s
- Rows: 20
- Exported candidates: 4
- Candidate reload smoke: passed
- Smoke verdict: `D10B_NO_REPLICATION_SIGNAL`

The smoke is not scientific evidence. It only proves that the D10b mode, output schema, checkpoint export, and reload path work.

## Main Run Command

The planned main ladder is intentionally not run interactively because timing-probe runtime implies a multi-hour run at the full requested budget.

Timing probe command shape:

```text
5 checkpoints x 2 climbers x 10 steps x eval_len 4000 x 4 eval seeds
```

Timing probe result:

- Output: `output/phase_d10b_h384_seed_replication_ladder_20260429/timing_probe`
- Rows: 100
- Runtime: 824.0s
- Observed rate: about 8.24s/proposal
- Probe verdict: `D10B_NO_REPLICATION_SIGNAL`
- Strict candidates: 0
- Near-strict candidates: 0
- Best probe seed: `seed_42`
- Best probe row: `WEAK_LADDER`, smooth `+0.0017995`, accuracy `+0.00025`, echo `0`, unigram `+0.0011831`

The probe is not scientific evidence. It is only a runtime and schema sanity check. It says the full D10b main should be run unattended, not foregrounded in an interactive turn.

Projected runtimes from the timing probe:

| run | proposals | projected runtime |
|---|---:|---:|
| full main: 5 seeds x 12 climbers x 80 steps | 4800 | about 11.0h |
| bounded main: 5 seeds x 8 climbers x 50 steps | 2000 | about 4.6h |

```powershell
target\release\examples\d9_direct_landscape.exe --checkpoints output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_42\final.ckpt,output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_1042\final.ckpt,output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_2042\final.ckpt,output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_3042\final.ckpt,output\phase_d7_operator_bandit_20260427\H_384\D7_BASELINE\seed_4042\final.ckpt --H 384 --mode seed-replication-ladder --eval-len 4000 --mo-eval-seeds 990001,990002,990003,990004 --mo-climbers 12 --mo-steps 80 --radii 4,8,16,32 --mutation-types edge,threshold --mo-export-top 8 --out output\phase_d10b_h384_seed_replication_ladder_20260429\main
```

If runtime must be bounded, use:

```text
--mo-climbers 8 --mo-steps 50
```

## Verification

Passed:

- `cargo build -p instnct-core --example d9_direct_landscape --release`
- `cargo test -p instnct-core --lib`
- `python tools/check_public_surface.py`
- `python -m compileall tools`
- D10b smoke generated all required CSV/JSON outputs.
- Exported smoke candidate checkpoint reloaded through `d9_direct_landscape`.
- D10b timing probe generated `replication_paths.csv`, `replication_candidates.csv`, `universality_matrix.csv`, `run_summary.json`, and a report.
- Exported timing-probe candidate checkpoint reloaded through `endpoint-robustness`.

## Progress Map

```text
GLOBAL AI PLAN MAP

[0] D7 H=384 baseline checkpoint bank
        |
        v
[1] Find real improvement
    status: DONE
    evidence: beta.8 seed2042_improved_generalist_v1
        |
        v
[2] Explain why it works
    status: DONE
    evidence: D9.4b 16k causal confirm
    mechanism: edge + threshold co-adaptation
        |
        v
[3] Does the mechanism replicate across H=384 seeds?
    status: CURRENT
    D10a scout: shallow negative / LOCAL_H384_BASIN_ONLY
    D10b smoke: code path works
    D10b timing probe: full main ~= 11h, bounded ~= 4.6h
        |
        |-- if D10b main positive ----------------.
        |                                        v
        |                         [4] 16k confirm + causal-diff
        |                                        |
        |                                        v
        |                         [5] H512 / GPU scaling
        |
        '-- if D10b main negative ----------------.
                                                 v
                                  [4B] Local finding only
                                      shift strategy:
                                      checkpoint quality /
                                      decoder-projection /
                                      task design

WE ARE HERE:
[3] D10b timing probe complete; unattended main is the next evidence run.
```

## Next Gate

Run the main D10b ladder as an unattended CPU job, capped at 3 seed shards in parallel. H=512/GPU remains blocked until D10b reaches at least `D10B_SEED_SENSITIVE_BUT_NOT_UNIQUE`.
