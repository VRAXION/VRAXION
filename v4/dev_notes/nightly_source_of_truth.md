# Nightly Source of Truth

`origin/nightly` is the current authoritative branch for active VRAXION v4 work.
Agents and engineers should treat this branch as the canonical runtime until a later deliberate consolidation into `main`.
Do not infer branch state from scratch telemetry, archived notes, or ad hoc helper scripts.

## Canonical Entrypoints

- `v4/training/train.py`
  - Real training entrypoint.
  - Uses `v4/config/vraxion_config.yaml` defaults plus CLI overrides.
  - This is the path for real CUDA training runs and checkpoint production.
- `v4/tests/nightly_research_runner.py`
  - Canonical nightly research harness.
  - Use this for fixed research surfaces such as `small_wikitext_fresh`, `fast_memory_carry`, and `wikitext_sequential_carry`.
  - New nightly claims should go through this runner, not through one-off scripts.
- `v4/config/vraxion_config.yaml`
  - Runtime default source for model/training policy.
  - Compile policy, sequential carry defaults, and production-ish training defaults live here.

## Current Recommended Setups

### GPU production-ish training

Locked defaults for the current nightly path:

- `seq_len=256`
- `sequential=true`
- `N=1`
- `write_mode=replace`
- `replace_impl=dense`
- `bb_enabled=false`
- compile auto enabled via `--compile` or `training.compile: true`

This is the current best-supported long-sequence training path. It is the one validated by the T=256 compile stabilization work.

### CPU research

Use the canonical runner surfaces from `v4/tests/nightly_research_runner.py`.
Do not use ad hoc `training/train.py` runs with the production-scale `hidden_dim=4096` config for CPU experiments unless the goal is explicitly to measure that slow path.
For CPU WikiText carry smoke or architectural probes, use the small research surfaces and fixed presets.

## Compile Stabilization

### Root Cause

Full-forward `torch.compile` was tracing `_process_chunk()` with `C=T`, which caused Dynamo to unroll the full timestep loop into one very large graph.
This was acceptable around `T=48`, but became unusable at `T=128` and production `T=256`.

### Current Solution

- Compile `_process_chunk()` at chunk granularity.
- Keep the outer timestep/chunk loop in Python.
- Auto policy:
  - `seq_len <= 48` -> full-model compile
  - `seq_len > 48` -> chunk compile of `_process_chunk`
- Compile policy lives in the training/runtime layer, not in the model build spec.

### Current Limits

- `bb_enabled=true` falls back to eager.
- `io_split_mode!='off'` falls back to eager.
- Proxy overlay is disabled in compile-active paths.
- CPU remains eager fallback in this pass.
- Only scalar diagnostics needed by the training loop/CSV path are preserved in compiled mode.

### Verified Benchmark

Validated on the nightly T=256 compile benchmark:

- eager: `3981.1 ms/step`, `warmup=23.1s`, `vram=6468 MB`, `final_loss=5.5116`
- compile auto/chunk: `205.3 ms/step`, `warmup=200.1s`, `vram=3223 MB`, `final_loss=5.5439`

Practical conclusion:

- the old T=256 compile hang is fixed
- steady-state training is dramatically faster
- compile warmup is still expensive and remains open work

## What Changed And Why

- Chunk compile was chosen because it bounds graph size while preserving sequential timestep semantics.
- Compile policy is a runtime/training concern, not a model architecture concern, so it stays outside the serialized build spec.
- Compiled paths keep only scalar diagnostics because `.item()` and rich list/trace formatting inside the graph cause graph breaks or retracing.
- Gradient checkpointing and chunk compile are mutually exclusive in the active path; chunk compile takes precedence.

## Known Failed Or Deferred Paths

- Full-model compile at `T>=128` is not acceptable in the current architecture.
- BB compile support is deferred because Python scalar state and associated logic are not yet tensorized.
- Rich list/trace telemetry inside compiled chunks is intentionally not preserved in this pass.
- Compile warmup is still long even though steady-state performance is fixed.
- CPU compile is not part of the supported performance path in this pass.

## Nightly Runner State

`v4/tests/nightly_research_runner.py` is the canonical nightly research entrypoint.
The current curated runner state includes the fixed-head auxiliary-offset variants:

- `LLT3F4SG`
- `LLT3F8SG`
- `LLT3F16SG`
- `LLT3F32SG`
- `LLT3F216SG`
- `LLT3F232SG`
- `LLT3F432SG`

These variants are passed through the canonical runner path and corresponding test surface.
They are research variants, not locked production defaults.

## How To Run

### GPU training with compile auto

From `v4/`:

```powershell
python training/train.py --device cuda --compile
```

### T=256 compile validation benchmark

From `v4/`:

```powershell
python bench_compile_wikitext_sim.py
```

### CPU WikiText carry smoke

From `v4/`:

```powershell
python tests/nightly_research_runner.py --surface wikitext_sequential_carry --variant LL --steps 200
```

## Branch Hygiene

Canonical runtime code:

- `v4/model/*`
- `v4/training/*`
- `v4/config/*`
- `v4/tests/nightly_research_runner.py`
- tests that directly validate the canonical runtime or nightly runner

Telemetry and scratch:

- `v4/dev_notes/telemetry/*`
- raw benchmark dumps
- one-off helper scripts not wired into canonical workflows

Expectations:

- New experiments should go through `dev_notes` plus the canonical nightly runner where possible.
- Scratch evidence should stay local or in telemetry, then be summarized into notes if it matters.
- Do not promote ad hoc scripts into source-of-truth status without documenting their intended workflow.

## Next Work, In Order

1. Reduce chunk-compile warmup cost.
2. Add BB/tensorized state compile support only if the nightly production path needs it.
3. Optimize ring clone/write-path overhead.
4. Continue the fixed-head/aux-offset runner follow-up through the canonical nightly runner surfaces before promoting any new variant to default status.
