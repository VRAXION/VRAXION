# Code And Commands

Date: 2026-05-26

## Main Runner

```text
instnct-core/examples/raven_pocket_smoke.rs
```

This is the experimental runner used for the night marathon.

Core constants in that runner:

```rust
const SYMBOLS: [char; 9] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'];
const POCKET_CLASSES: usize = 9;

const GRID_BASE: usize = 0;      // 9 cells * 9 symbols = 81
const MISSING_BASE: usize = 81;  // 9 cells = 9
const POCKET_BASE: usize = 90;   // 9 pockets * 9 symbols = 81
const FAMILY_BASE: usize = 171;  // 4 families
const TARGET_BASE: usize = FAMILY_BASE + FAMILY_COUNT;
const INPUT_FEATURES_USED: usize = TARGET_BASE + 9;
const INPUT_STRENGTH: i32 = 7;
```

Important modes:

```text
--task-mode pocket_id_only
--task-mode symbol_match_only
--task-mode pocket_id_grid_noise
--task-mode full_match_hint
--task-mode pocket_match_hint
```

Important direct mutation knobs:

```text
--steps
--eval-every
--candidates
--heartbeat-sec
--h
```

Important genome/GA knobs:

```text
--genome-population
--genome-generations
--genome-mode
--genome-len
--genome-edges-per-neuron
--genome-mutation-bytes
--genome-random-fraction
```

## Compile

```powershell
cargo check --manifest-path instnct-core\Cargo.toml --example raven_pocket_smoke
cargo build --release --manifest-path instnct-core\Cargo.toml --example raven_pocket_smoke
```

## Strong Positive Control

This run reached 100% train/test:

```powershell
target\release\examples\raven_pocket_smoke.exe `
  --out target\codex_smoke\night_dna_iter_20260526\pocket_id_only_direct_evo_9181 `
  --seed 9181 `
  --task-mode pocket_id_only `
  --train-rows 4000 `
  --test-rows 1440 `
  --steps 140 `
  --eval-every 10 `
  --candidates 64 `
  --heartbeat-sec 20
```

Result:

```text
train_accuracy = 1.0
test_accuracy = 1.0
```

## Equality Direct Test To Rerun

This was started but stopped early. Rerun it cleanly:

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

Expected artifact files:

```text
queue.json
progress.jsonl
curriculum_train.jsonl
curriculum_test.jsonl
training_metrics.jsonl
summary.json
report.md
row_level_predictions.jsonl
checkpoint_latest.ckpt
checkpoint_final.ckpt
```

## Useful Existing Result Files

```text
target/codex_smoke/night_dna_iter_20260526/overnight_iterative_findings_20260526.md
target/codex_smoke/night_dna_iter_20260526/overnight_iterative_findings_20260526_v2.md
target/codex_smoke/night_dna_iter_20260526/overnight_iterative_findings_20260526_v3.md
target/codex_smoke/night_dna_iter_20260526/overnight_iterative_findings_20260526_v4.md
```

## Run Hygiene

Every real run must write:

```text
queue.json immediately
progress.jsonl repeatedly
summary/report at the end
row_level_predictions.jsonl when eval completes
```

No black-box run is acceptable.

