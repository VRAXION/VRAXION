# STABLE_LOOP_PHASE_LOCK_015_MICRO_BARRAGE Result

Status: implemented, static validation complete, quick barrage complete.

## Question

```text
Which new transport principle has signal against wrong-phase echo?
```

## Verdict

```text
DUAL_LAYER_NO_SIGNAL
MECHANISM_OVERPOWERS_RULE_CONTROL
NO_MICRO_MECHANIC_RESCUES
PRODUCTION_API_NOT_READY
READOUT_WINDOW_ONLY_NOT_TRANSPORT
REFRACTORY_REDUCES_STALE_PHASE
SIGNED_CANCELLATION_KILLS_CORRECT_SIGNAL
```

Interpretation:

```text
none of the tested micro-mechanics passed the micro-signal gate

arrival-window readout is high, but the matching random-control is also high
enough to mark it as a readout/control effect rather than solved transport

consume/refractory/no-reentry reduce wrong-if-arrived, but mainly by killing
or thinning arrival rather than improving stable long-chain accuracy
```

## Planned Runs

Static:

```powershell
cargo check -p instnct-core --example phase_lane_micro_barrage
cargo test -p instnct-core jackpot_traced_emits_candidate_rows_and_accept_invariants
git diff --check
```

Quick barrage:

```powershell
cargo run -p instnct-core --example phase_lane_micro_barrage --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_015_micro_barrage/quick ^
  --seeds 2026 ^
  --eval-examples 256 ^
  --widths 8,12 ^
  --path-lengths 4,8,16,24 ^
  --ticks-list 8,16,24,32 ^
  --heartbeat-sec 15
```

## Quick Barrage Summary

Quick root:

```text
target/pilot_wave/stable_loop_phase_lock_015_micro_barrage/quick
```

Ranking:

| Mechanism | Acc | Long | Family min | Wrong-if-arrived | Wrong drop | Random | Gap | Signal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EMIT_ONCE_CONSUME | 0.533 | 0.545 | 0.453 | 0.000 | 0.232 | 0.487 | 0.346 | false |
| REFRACTORY_CELL | 0.533 | 0.545 | 0.453 | 0.000 | 0.232 | 0.487 | 0.346 | false |
| MOMENTUM_PLUS_CONSUME | 0.533 | 0.545 | 0.453 | 0.000 | 0.232 | 0.487 | 0.346 | false |
| ARRIVAL_WINDOW_READOUT_DIAGNOSTIC | 0.962 | 0.949 | 0.875 | 0.022 | 0.210 | 0.692 | 0.000 | false |
| NO_REENTRY_MOMENTUM | 0.656 | 0.708 | 0.406 | 0.096 | 0.136 | 0.408 | 0.306 | false |
| BEST_PUBLIC_COMBO_014 | 0.804 | 0.738 | 0.641 | 0.181 | 0.051 | 0.406 | 0.083 | false |
| PHASE_COMPETITION_PER_CELL | 0.799 | 0.833 | 0.438 | 0.185 | 0.047 | 0.353 | 0.174 | false |
| BASELINE_FULL16_014 | 0.752 | 0.801 | 0.484 | 0.232 | 0.000 | 0.346 | 0.210 | false |
| MOMENTUM_LANES | 0.752 | 0.801 | 0.484 | 0.232 | 0.000 | 0.346 | 0.210 | false |
| DUAL_LAYER_EB_FIELD | 0.743 | 0.789 | 0.500 | 0.241 | -0.009 | 0.346 | 0.192 | false |
| SIGNED_PHASE_CANCELLATION | 0.719 | 0.792 | 0.500 | 0.263 | -0.031 | 0.382 | 0.263 | false |
| DUAL_LAYER_PLUS_DAMPING | 0.717 | 0.765 | 0.500 | 0.268 | -0.036 | 0.346 | 0.219 | false |
| SIGNED_PLUS_CELL_NORMALIZATION | 0.676 | 0.735 | 0.328 | 0.308 | -0.076 | 0.359 | 0.301 | false |

## Output Artifacts

The quick run wrote:

```text
queue.json
progress.jsonl
metrics.jsonl
mechanism_metrics.jsonl
random_control_metrics.jsonl
family_metrics.jsonl
counterfactual_metrics.jsonl
mechanism_ranking.json
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

No raw `target/` outputs are committed.

## Claim Boundary

015 micro-barrage is a mechanism selector only. It cannot claim stable
transport solved, production architecture, full VRAXION, consciousness,
language grounding, Prismion uniqueness, or physical quantum behavior.
