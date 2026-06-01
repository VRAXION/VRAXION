# D68 Counter-Support Triage Repair Contract

## Purpose

D67 scale-confirmed Rust sparse aggregation-backed support scoring, but routed to
D68 because it still requested too much counter-support in clean and mixed cases.

D68 asks:

```text
Can runtime diagnostics reduce unnecessary counter-support while preserving hard
correlated/adversarial recall, external-test handling, abstain behavior, and
false-confidence safety?
```

## Boundary

```text
D68 only tests counter-support triage for Rust sparse aggregation-backed support
scoring in controlled symbolic joint formula discovery. The formula solver
remains symbolic. It does not prove full VRAXION brain, raw visual Raven
reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture
superiority, or production readiness.
```

## Required Files

Tracked files:

```text
docs/research/D68_COUNTER_SUPPORT_TRIAGE_REPAIR_CONTRACT.md
docs/research/D68_COUNTER_SUPPORT_TRIAGE_REPAIR_RESULT.md
scripts/probes/run_d68_counter_support_triage_repair.py
scripts/probes/run_d68_counter_support_triage_repair_check.py
```

Generated artifacts:

```text
target/pilot_wave/d68_counter_support_triage_repair/
```

## Policy Arms

```text
D67_BEST_REPLAY
COUNTER_TRIAGE_MARGIN_GATE
COUNTER_TRIAGE_ENTROPY_MARGIN_GATE
COUNTER_TRIAGE_CONFIDENCE_STABILITY_GATE
COUNTER_TRIAGE_SUPPORT_INDEPENDENCE_GATE
COUNTER_TRIAGE_ADVERSARIAL_PRESSURE_GATE
COUNTER_TRIAGE_MULTI_SIGNAL_GATE
TRAINED_THRESHOLD_TRIAGE_GATE
COUNTER_TRIAGE_CONSERVATIVE_HIGH_RECALL
COUNTER_TRIAGE_COST_OPTIMIZED
CAP_7_CONTROL
CAP_9_CONTROL
ALWAYS_COUNTER_CONTROL
NEVER_COUNTER_CONTROL
RANDOM_COUNTER_CONTROL
SHUFFLED_TRIAGE_SIGNAL_CONTROL
BAD_TRIAGE_SIGNAL_CONTROL
AGGREGATION_ABLATION_CONTROL
SUPPORT_CONTENT_CORRUPTION_CONTROL
TRUTH_LEAK_SENTINEL_REFERENCE_ONLY
REGIME_LABEL_ORACLE_REFERENCE_ONLY
```

Reference-only arms are not fair arms.

## Metric Definitions

```text
counter_needed =
  DECIDE is wrong and an internal counter action fixes the row

external_test_needed =
  DECIDE is wrong and an available external-test action fixes the row

unnecessary_counter =
  internal counter or external test requested while DECIDE was already correct

missed_counter =
  needed internal counter or needed external test was not requested
```

Internal counter and external-test metrics must be reported separately:

```text
counter_precision
counter_recall
external_test_precision
external_test_recall
unnecessary_internal_counter_rate
unnecessary_external_test_rate
missed_internal_counter_rate
missed_external_test_rate
```

## Hard Gates

Fair triage arms must not use:

```text
truth labels
support_regime
split
seed
row_id
expected labels
Python hash()
fixed synthetic accuracy tables
fake hit=random.random()<p sampling
```

Rust gates:

```text
rust_path_invoked = true
fallback_rows = 0
python_precomputed_final_aggregate_label_rows = 0
failed_jobs = []
```

No black-box run:

```text
queue.json is written immediately
progress.jsonl is appended throughout
partial metric snapshots are written during evaluation
blocking Rust bridge waits emit heartbeat artifacts
```

## Positive Gate

Best fair D68 triage arm must satisfy:

```text
exact >= same_run_D67_BEST_REPLAY_exact - 0.003
correlated_echo >= 0.995
adversarial_distractor >= 0.995
external_test_required >= 0.990
indistinguishable_abstain >= 0.99
false_confidence <= 0.01

unnecessary_counter_support_rate <= same_run_D67_BEST_REPLAY - 0.25
clean_unnecessary <= 0.50
mixed_unnecessary <= 0.60

overall_missed_counter <= 0.02
correlated_missed_counter <= 0.02
adversarial_missed_counter <= 0.02
distinguishable_false_missed_counter <= 0.02
external_test_missed <= 0.02

support_saved_vs_same_run_D67 >= 0.75
random/shuffled/bad/never/ablation controls worse
```

## Decisions

```text
counter_support_triage_repair_confirmed -> D69_COUNTER_SUPPORT_TRIAGE_SCALE_CONFIRM
counter_triage_high_recall_high_cost -> D68C_COUNTER_COST_OPTIMIZATION
counter_triage_recall_failure -> D68R_COUNTER_RECALL_REPAIR
counter_support_triage_repair_not_confirmed -> D68_REPAIR
```

## Validation

```powershell
python -m py_compile scripts/probes/run_d68_counter_support_triage_repair.py
python -m py_compile scripts/probes/run_d68_counter_support_triage_repair_check.py

python scripts/probes/run_d68_counter_support_triage_repair.py --out target/pilot_wave/d68_counter_support_triage_repair/smoke --seeds 12701,12702,12703,12704,12705 --train-rows-per-seed 240 --test-rows-per-seed 240 --ood-rows-per-seed 240 --workers auto --cpu-target 50-75 --heartbeat-sec 20 --scale-mode healthy-240

python scripts/probes/run_d68_counter_support_triage_repair_check.py --check-only --out target/pilot_wave/d68_counter_support_triage_repair/smoke
git diff --check
```
