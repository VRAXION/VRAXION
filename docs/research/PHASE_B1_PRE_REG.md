# Phase B.1 Pre-Registration: Horizon x Tie-Policy Ablation

## Question

Phase B found a directional rescue for `mutual_inhibition, H=384` when the horizon increased from 20k to 40k steps. Phase B.1 tests whether that is a stable horizon trend and whether `accept_ties` changes the outcome.

## Fixed Scope

- Fixture: `mutual_inhibition`
- H: `384`
- Jackpot: `9`
- Ticks: `6`
- Input scatter: `false`
- Seeds: `42, 1042, 2042, 3042, 4042`
- No C19/int-weight/motif/bytepair-prune changes

## Arms

| Arm | Steps | accept_ties |
|---|---:|---|
| `B1_S20_STRICT` | 20000 | false |
| `B1_S20_TIES` | 20000 | true |
| `B1_S40_STRICT` | 40000 | false |
| `B1_S40_TIES` | 40000 | true |
| `B1_S80_STRICT` | 80000 | false |
| `B1_S80_TIES` | 80000 | true |

Total: 30 runs.

## Measurements

Each run writes the same Phase B artifact set:

- `stdout.txt`, `stderr.txt`
- `candidates.csv`
- `final.ckpt`
- `run_meta.json`
- `panel_timeseries.csv`
- `panel_summary.json`

Primary endpoint: `peak_acc`.

Secondary diagnostics: `final_acc`, `accept_rate_pct`, `alive_frac_mean`, `C_K_window_ratio`, `V_raw`, `V_sel`, `M_pos`, `R_neg`, operator productivity, and accepted small-delta/neutral counts.

## Decision Rules

- If strict 40k or 80k recovers or exceeds the Phase A `H=256` reference mean of `5.28%`, the horizon-confound interpretation is strengthened.
- If strict 80k > strict 40k, H=384 is still horizon-limited.
- If strict 80k <= strict 40k, H=384 likely hits plateau/drift/policy limits after rescue.
- If `accept_ties=true` improves mean or reduces variance consistently at the same horizon, tie policy remains active.
- If `accept_ties=true` worsens or mostly increases small-delta drift, strict remains the canon policy.
- If tie-policy effects are weak or inconsistent, prioritize horizon and operator-schedule follow-up.

## Non-Claims

- This does not test `bytepair_proj`; that fixture has separate prune/collapse dynamics.
- This does not validate `C_K` as a scalar objective.
- This does not claim criticality, resonance, or edge-of-chaos behavior.
