# VRAXION Public Beta — How To Start Training

This is the "what do I run first?" doc for the v5.0.0 Public Beta
(grower-based) open beta. It describes exactly the commands that start
training the frozen canonical stack, what output to expect, when
training is "done", and where the artifacts land.

It supplements [`../BETA.md`](../BETA.md), which defines the public-beta
contract, and the two canonical contract docs
[`./GROWER_RUN_CONTRACT.md`](./GROWER_RUN_CONTRACT.md) (B0) and
[`./BYTE_OPCODE_V1_CONTRACT.md`](./BYTE_OPCODE_V1_CONTRACT.md) (B1).
Everything below references the frozen commands and golden snapshots in
those files verbatim.

## 1. The 3-minute training quickstart

The single canonical entry point is the B0 grower regression harness:

```bash
python tools/run_grower_regression.py
```

The first invocation pays a ~30-60 s Cargo build overhead. End-to-end
wall-clock on a modern laptop is roughly 2-4 minutes. Subsequent runs
reuse the release binary and are faster.

Artifacts land under `target/grower-regression/<UTC-timestamp>/`. A green
run ends with a rendered `## Golden Check` block:

```text
## Golden Check

PASS
```

Exit code `0` and an empty `golden_check.json` `errors` array are the
machine-readable pass signal. If either is wrong, see Section 9.

## 2. What "training" means here

"Training" in this stack is **not** gradient descent. There is no loss
surface being walked by SGD, and no floating-point weight matrix being
updated each batch. Instead:

- **Grower loop**: the builder adds neurons one at a time. For each
  candidate neuron it runs a **scout oracle** (single-signal score,
  connect-all backprop probe, pair-lift shortlist), then a **ternary
  search** over `{-1, 0, +1}` weights and `i32` threshold, then a
  **non-strict accept gate** that only rejects if the new ensemble
  validation score is strictly worse.
- **Persist**: every accepted neuron is appended to `state.tsv`
  (authoritative resume file) and snapshotted into `checkpoints/`.
- **B1 byte+opcode "training"**: train 8 bit-head growers, freeze their
  hidden binary activations, then build an **exact LUT translator**
  keyed on the concatenated hidden binary signatures. The LUT is not
  trainable — it is a deterministic lookup over the frozen latent.

Canonical neuron form is bias-free: `dot >= threshold`, ternary `i8`
weights, `i32` threshold. No bias search, no float weights. See
[`./GROWER_RUN_CONTRACT.md`](./GROWER_RUN_CONTRACT.md) for the frozen
defaults.

## 3. The two canonical pipelines

### 3.1 B0 Grower Regression

```bash
python tools/run_grower_regression.py
```

- Matrix: `four_parity`, `four_popcount_2`, `is_digit_gt_4`,
  `diagonal_xor`, `full_parity_4`, `digit_parity`
- Seeds: `data_seed=42`, `search_seed=42`
- Bundle: `run_cmd.txt`, `env.json`, `metrics.json`, `summary.md`,
  `golden_check.json`, per-task `<task>.stdout.txt`, and a `runs/`
  subtree with each task's persistent `state.tsv` + `checkpoints/`.
- Pass condition: exit 0 **and** `golden_check.json` has `errors: []`.
- Contract: [`./GROWER_RUN_CONTRACT.md`](./GROWER_RUN_CONTRACT.md)

### 3.2 B1 Byte + Opcode Acceptance

```bash
python tools/run_byte_opcode_acceptance.py
```

- Pipeline stages: build example `byte_opcode_grower` -> train +
  `--export-json stack.json` -> `--reload-json stack.json` replay ->
  full 1024-sample domain diff -> golden diff.
- Bundle: `run_cmd.txt`, `env.json`, `train.stdout.txt`,
  `reload.stdout.txt`, `train_metrics.json`, `reload_metrics.json`,
  `summary.md`, `golden_check.json`, `reload_check.json`, `stack.json`.
- Pass condition: exit 0, `golden_check.json` `errors: []`, **and**
  `reload_check.json` `errors: []` (export/reload must be exact).
- Contract: [`./BYTE_OPCODE_V1_CONTRACT.md`](./BYTE_OPCODE_V1_CONTRACT.md)

## 4. Expected trajectory (golden metrics)

These are the frozen reference numbers from the golden fixtures. They
are reproducible baselines, not targets to beat.

### B0 — `instnct-core/tests/fixtures/grower_regression_golden.json`

| Task             | Train | Val   | Test  | Neurons | Depth | Hidden | Stalled |
|------------------|------:|------:|------:|--------:|------:|--------|---------|
| four_parity      | 100.0 | 100.0 | 100.0 |       6 |     6 | true   | false   |
| four_popcount_2  | 100.0 | 100.0 | 100.0 |       7 |     6 | true   | false   |
| is_digit_gt_4    |  77.2 |  71.5 |  76.5 |       7 |     4 | true   | false   |
| diagonal_xor     |  86.5 |  88.5 |  89.5 |       1 |     1 | false  | false   |
| full_parity_4    |  79.3 |  80.5 |  82.0 |       1 |     1 | false  | false   |
| digit_parity     |  94.3 |  90.0 |  89.5 |      12 |     6 | true   | false   |

B0 summary:

| Metric        | Value   |
|---------------|--------:|
| task_count    |       6 |
| mean_val      |  88.417 |
| median_val    |   89.25 |
| max_val       |   100.0 |
| mean_test     |  89.583 |
| median_test   |    89.5 |
| max_test      |   100.0 |
| mean_neurons  |   5.667 |
| max_depth     |       6 |
| stall_count   |       0 |

### B1 — `instnct-core/tests/fixtures/byte_opcode_golden.json`

| Metric                | Value  |
|-----------------------|-------:|
| direct_byte_acc       |   75.0 |
| translator_byte_acc   |  100.0 |
| distinct_keys         |   1024 |
| conflicting_keys      |      0 |
| key_bits              |     61 |
| total_neurons         |     61 |
| max_depth             |      7 |

The direct-bitbank 75.0% line is a **negative control** — raw-input
shortcuts are contractually forbidden in the canonical beta-path. Only
the 100.0% exact LUT translator over the 61-bit frozen latent counts as
a pass.

## 5. Termination criteria

Training is "done" for the public-beta gate when **all** of the
following hold:

- The harness process exits with code `0`.
- The append-only evidence bundle is fully written.
- `golden_check.json` has `errors: []`.
- **B1 only**: `reload_check.json` has `errors: []` (the reloaded stack
  must replay identically to the freshly-exported one).
- **B0 only**: every task's `<task>.stdout.txt` shows a `FINAL:` line
  matching the golden row, and no task reports `Stalled`.
- Rerunning with the same seeds produces byte-identical metrics.

There is no early-stopping patience knob to tune: the grower stops when
it runs out of accepted proposals within the per-task `stall` budget
defined in `tools/run_grower_regression.py`.

## 6. What you get as output

### B0 lane — `target/grower-regression/<ts>/`

| Path                          | Role                                               |
|-------------------------------|----------------------------------------------------|
| `run_cmd.txt`                 | Exact cargo command lines used per task            |
| `env.json`                    | Host, cargo, rustc, Python, workspace version      |
| `metrics.json`                | Full per-task + aggregate numbers for this run     |
| `summary.md`                  | Rendered human-readable report incl. golden check  |
| `golden_check.json`           | `{"errors": [...]}` — empty list == pass           |
| `<task>.stdout.txt`           | Raw stdout from each grower task run (6 files)     |
| `runs/<task>/state.tsv`       | Authoritative trained network (reload input)       |
| `runs/<task>/checkpoints/`    | Append-only JSON snapshots of the build trajectory |

### B1 lane — `target/byte-opcode-acceptance/<ts>/`

| Path                   | Role                                                  |
|------------------------|-------------------------------------------------------|
| `run_cmd.txt`          | Build + train + reload command lines                  |
| `env.json`             | Host, cargo, rustc, Python, harness knobs             |
| `train.stdout.txt`     | Raw train-phase stdout                                |
| `reload.stdout.txt`    | Raw reload-phase stdout                               |
| `train_metrics.json`   | Metrics from the training run                         |
| `reload_metrics.json`  | Metrics from replaying the exported stack             |
| `summary.md`           | Rendered report incl. both golden and reload checks   |
| `golden_check.json`    | Train-vs-golden diff — empty `errors` list == pass    |
| `reload_check.json`    | Reload-vs-train diff — empty `errors` list == pass    |
| `stack.json`           | Frozen 8-bit-head + exact LUT export, reload input    |

Section 7 describes how to consume `state.tsv` and `stack.json`.

## 7. Using the trained network

### B0: consume a grower `state.tsv`

Each B0 run emits an authoritative `state.tsv` under
`runs/<task>/state.tsv`. The canonical read-side CLI is
[`../instnct-core/examples/neuron_infer.rs`](../instnct-core/examples/neuron_infer.rs).

```bash
cargo run --release --example neuron_infer -p instnct-core -- \
  --state target/grower-regression/<ts>/runs/digit_parity/state.tsv \
  --input "1 1 1 1 0 1 1 1 1"
```

Optional flags:

- `--input "<bits>"` — repeatable; accepts whitespace- or comma-split
  0/1 tokens.
- `--task <NAME>` — guard: exits non-zero if the loaded state is for a
  different task.
- `--scores` — also print the raw AdaBoost ensemble score alongside the
  `{0,1}` label.
- `--format tsv|human` — output shape.

`neuron_infer` does no training — it loads the persisted ternary
network and evaluates supplied bit inputs. Input shape must match the
task's `n_in` (4 for `four_*`, 9 for everything else).

### B1: replay a frozen byte+opcode stack

The B1 exported `stack.json` can be reloaded directly by the training
example in replay-only mode:

```bash
cargo run --release --example byte_opcode_grower -p instnct-core -- \
  --reload-json target/byte-opcode-acceptance/<ts>/stack.json \
  --metrics-json /tmp/replay_metrics.json
```

The real reload flag is `--reload-json <path>` (verified against
`tools/run_byte_opcode_acceptance.py` line 134). This runs the full
1024-sample domain through the reloaded 8 frozen bit-heads plus the
exact LUT translator and writes metrics identical to the original
training metrics. Any drift is a contract violation under
[`./BYTE_OPCODE_V1_CONTRACT.md`](./BYTE_OPCODE_V1_CONTRACT.md).

## 8. Failure modes

| Symptom                            | Meaning                                            | What to do                                                 |
|------------------------------------|----------------------------------------------------|------------------------------------------------------------|
| `panic: 101` / unknown task        | bad CLI args or stale script                       | use only the canonical harness command                     |
| OOM during B0                      | `max_neurons` set too high outside the matrix      | stick to the frozen matrix in `run_grower_regression.py`   |
| `NaN` in `alpha`                   | stale `state.tsv` AdaBoost collapse                | delete the task's `state.tsv` and rerun                    |
| `golden_check.json` has errors     | metrics drifted from the frozen fixture            | verify commit hash + seeds, then rerun                     |
| `reload_check.json` has errors     | B1 export/reload is no longer byte-exact           | file a contract-violation bug, do not ship                 |
| "incompatible schema" load error   | old bias-bearing `state.tsv` on disk               | delete it and rerun (current schema is bias-free, 9 cols)  |
| "task mismatch" on infer/reload    | loaded state for a different task than requested   | use a fresh `--out-dir` or `--task` argument               |
| harness exits 0 but bundle missing | filesystem permission / disk full                  | check `target/` is writable, free disk, rerun              |

## 9. Next

- Open beta feedback: file an issue tagged `beta-feedback`. Include the
  full evidence bundle path and your `env.json`.
- The current next build target is
  [Issue #114](https://github.com/VRAXION/VRAXION/issues/114).
- Custom tasks during open beta are fork-and-edit: clone
  `neuron_grower.rs`, add your task to `task_n_in` and the label
  function, and run it outside the canonical regression harness.
- Future directions: broader B1 opcode set, structured multi-byte
  outputs, and a first-class user-task config are being tracked for
  v5.1. None of them are in scope for the current beta gate.

For the authoritative pass/fail contract, always fall back to
[`../BETA.md`](../BETA.md) — this doc is a runbook, not a contract.
