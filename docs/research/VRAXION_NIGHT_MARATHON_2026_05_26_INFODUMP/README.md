# VRAXION Night Marathon Infodump

Date: 2026-05-26

Purpose: this package is a catch-up dump for a future local/GitHub agent. Read this first before continuing the VRAXION pocket-routing / DNA-search work.

## One-Line State

Clean pocket selection is learnable by direct VRAXION mutation, but the current DNA/genome search is not yet good enough, equality is weak, and extra input noise still breaks routing.

```text
clean selector lane:
  WANTED_POCKET=P# -> output P#
  direct VRAXION mutation reached 100% train / 100% test

current DNS / genome / GA:
  same simple selector plateaued around 52-56%

symbol equality:
  WANTED=A, CANDIDATE=A/B -> yes/no
  around 56.5% vs 50% dumb baseline

noise/grid:
  direct target hint plus extra grid/pocket lamps stayed low, around 11-18% in smoke runs
```

## Human Translation

The "folyoso -> zseb" output path is not dead.

If the input is clean and says which pocket is wanted, the real VRAXION mutation engine can learn to point to the right pocket perfectly.

The failure is elsewhere:

1. The current DNA/genome methods do not reliably rediscover that clean wiring.
2. The system does not yet robustly answer "are these two symbols the same?"
3. Extra lamps/noise confuse the network badly.

So the next work should not jump back to full Raven/IQ tasks. Split the problem:

```text
1. selector lane only
2. equality lane only
3. noise/gating only
4. then recombine
```

## What Was Built

Experimental runner:

```text
instnct-core/examples/raven_pocket_smoke.rs
```

Important status: at dump time this file is untracked in git. If this package is committed to GitHub, add that runner too or preserve a code snapshot, otherwise the future agent will have results without the exact runner.

The runner is intentionally artifact-heavy:

```text
queue.json
progress.jsonl
training_metrics.jsonl
summary.json
report.md
row_level_predictions.jsonl
checkpoint_latest.ckpt
checkpoint_final.ckpt
```

It writes partial progress continuously. This follows the project rule: no black-box runs.

## Main Code Modes

Task modes added/tested:

```text
full
pocket_lookup
pocket_only_lookup
pocket_id_hint
pocket_id_only
pocket_match_hint
symbol_match_only
full_match_hint
pocket_id_grid_noise
pattern_fixed_pocket
```

Genome/DNA modes added/tested:

```text
blind
meta
string_rule
u64_barcode
u64_slot_barcode
u64_gate_sampled
rule_dna_gate
rule_dna_gate_mid
rule_dna_gate_strict
```

The important clean controls:

```text
pocket_id_only:
  prompt has WANTED_POCKET=P#
  no grid
  no pocket list
  output is selected pocket

symbol_match_only:
  prompt has WANTED=<symbol>
  prompt has CANDIDATE=<symbol>
  output P1=yes or P2=no

pocket_id_grid_noise:
  prompt has grid noise plus WANTED_POCKET=P#
  tests whether the model can ignore irrelevant lamps
```

## Main Result Anchors

Current best human report:

```text
target/codex_smoke/night_dna_iter_20260526/overnight_iterative_findings_20260526_v4.md
target/codex_smoke/night_dna_iter_20260526/overnight_iterative_findings_20260526_v4.json
```

Breakthrough control:

```text
target/codex_smoke/night_dna_iter_20260526/pocket_id_only_direct_evo_9181/summary.json
target/codex_smoke/night_dna_iter_20260526/pocket_id_only_direct_evo_9181/progress.jsonl
target/codex_smoke/night_dna_iter_20260526/pocket_id_only_direct_evo_9181/row_level_predictions.jsonl
```

Partial aborted equality-direct run:

```text
target/codex_smoke/night_dna_iter_20260526/symbol_match_only_direct_evo_9182/
```

This run was intentionally stopped when the user needed to shut down. It is not a final result. It only reached step 13 and showed early rise to about 38.75% test accuracy.

## Strong Evidence

### Direct selector solved

Run:

```text
pocket_id_only_direct_evo_9181
```

Key metrics:

```text
train_accuracy = 1.0
test_accuracy = 1.0
random_baseline_accuracy = 0.1111111111111111
step_reached_100_percent = 120 / 140
elapsed_sec = 761.34
```

Meaning:

```text
The official VRAXION direct mutation engine can learn clean wanted-pocket -> output-pocket routing.
```

### DNS/genome search failed to match direct mutation

Long genome/GA selector controls stayed around:

```text
mean_final ~= 52-53%
mean_best ~= 56%
```

Meaning:

```text
The current DNA/genome search is not enough for mainline use yet.
```

## Weak Or Incomplete Evidence

### Equality direct mutation is still unverified

An equality direct run was started:

```text
symbol_match_only_direct_evo_9182
```

It was stopped early. Last observed line:

```text
step = 13
test_accuracy = 0.3875
train_accuracy = 0.39075
```

Do not use this as a conclusion. Continue it or rerun cleanly.

### Full Raven-style task is not solved

The full grid + pockets task is still not solved. Current evidence points to:

```text
selector works when clean
equality weak
noise/gating weak
DNA search weak
```

## What Not To Claim

Do not claim:

```text
Raven solved
general reasoning solved
natural language reasoning
Gemma-like assistant ability
production readiness
DNA search proven
```

Correct claim:

```text
Direct VRAXION mutation can solve clean pocket selection.
The current DNA/genome search does not yet recover that reliably.
The next bottlenecks are equality and noise/gating.
```

## Recommended Next Run

Run direct equality to completion:

```powershell
target\release\examples\raven_pocket_smoke.exe `
  --out target\codex_smoke\night_dna_iter_20260526\symbol_match_only_direct_evo_FULL `
  --seed 9182 `
  --task-mode symbol_match_only `
  --train-rows 4000 `
  --test-rows 1440 `
  --steps 140 `
  --eval-every 10 `
  --candidates 64 `
  --heartbeat-sec 20
```

Decision rule:

```text
If direct equality reaches high accuracy:
  equality is learnable; current DNA/search/noise path is the issue.

If direct equality also stays near 50-60%:
  equality needs a special small lane/unit before full Raven recombination.
```

## Current Git Hygiene Warning

At dump time, unrelated AnchorWeave files are dirty/untracked. Do not mix them into VRAXION commits unless intentionally curating them.

Relevant VRAXION files for this package:

```text
instnct-core/examples/raven_pocket_smoke.rs
docs/research/VRAXION_NIGHT_MARATHON_2026_05_26_INFODUMP/
```

