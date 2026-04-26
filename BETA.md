# `v5.0.0 Public Beta` Contract

This document defines the current public-beta freeze target on `main`.

The key transition is simple:

- **released tag**: `v5.0.0-beta.4` is the current shipped Rust public beta (Phase D2 cross-H verdict + repo consolidation pass); `v5.0.0-beta.3` is the prior consolidated mainline + Phase A→B→D research line release; `v5.0.0-beta.2` is the prior grower-based beta and `v5.0.0-beta.1` the original language-evolution beta as historical references
- **current mainline on `main`**: the bias-free Rust grower (`neuron_grower.rs` builder + `run_grower_regression.py` B0 regression bundle), with the Phase A→B→D mutation-selection research line on `evolve_mutual_inhibition.rs`/`evolve_bytepair_proj.rs` running as the active research-and-development surface alongside it
- **next public milestone**: grower-based `v5.0.0 Public Beta`

## Current canonical path

- Builder: [`instnct-core/examples/neuron_grower.rs`](instnct-core/examples/neuron_grower.rs)
- Run contract: [`docs/GROWER_RUN_CONTRACT.md`](docs/GROWER_RUN_CONTRACT.md)
- Canonical regression harness: [`tools/run_grower_regression.py`](tools/run_grower_regression.py)

The grower is the only public mainline path on `main`. Python remains a support/reference lane.

## B0 engine-freeze contract

The current hard gate is the grower regression bundle:

```powershell
python tools/run_grower_regression.py
```

This command must emit an append-only evidence bundle under `target/grower-regression/<timestamp>/`.

### Required artifacts

- `run_cmd.txt`
- `env.json`
- `metrics.json`
- `summary.md`
- `golden_check.json`

### B0 pass conditions

- the regression matrix completes without panic
- the evidence bundle is written
- the results match the golden snapshot
- rerunning with the same seeds reproduces the same metrics

## B1 promotion gate

Public beta is **not** green until the grower also has a locked computation benchmark:

- `1 byte data + 4 opcode -> 1 byte`
- ops: `COPY`, `NOT`, `INC`, `DEC`
- exact translator contract:
  - `8` frozen grower bit-heads remain the trunk
  - translator key = concatenated hidden binary activations
  - translator = fixed exact LUT
  - no raw-input shortcut and no trainable beta-path readout

B1 must also ship:

- direct bitbank negative control
- full truth-table + adversarial suite
- frozen export/reload path for the exact translator stack

The canonical contract is [`docs/BYTE_OPCODE_V1_CONTRACT.md`](docs/BYTE_OPCODE_V1_CONTRACT.md).

## What does not count as public beta

- ad hoc overnight logs without the evidence bundle
- untracked scratch examples or playground HTML
- claims based on the released beta.1 language runner alone
- new benchmark lines without the exact translator freeze
