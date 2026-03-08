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
- On the supported production path, chunk compile now targets the single-expert fast helper directly instead of the generic dispatcher.
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

### Current Hot-Path Specialization

- The active nightly production shape is now special-cased in `instnct.py`:
  - `N=1`
  - `R=1`
  - `pointer_mode='sequential'`
  - `write_mode='replace'`
  - `replace_impl='dense'`
  - `bb_enabled=false`
  - `io_split_mode='off'`
  - `read_kernel_mode='vshape'`
- This removes generic single-expert overhead from `_process_chunk()`:
  - no per-step `hidden_lst` materialization for `N=1`
  - no `torch.stack(hidden_lst).mean(0)` on the hot path
  - no list-based output accumulation before final `torch.stack`
  - cached strict reader indices instead of per-step `.nonzero().flatten().tolist()`
- The generic chunk path keeps the same behavior and now also uses the cheaper output assembly path.

### Verified Benchmark

Validated on the nightly T=256 compile benchmark after the hot-path specialization:

- eager: `3819.3 ms/step`, `warmup=24.0s`, `vram=6468 MB`, `final_loss=5.5116`
- compile auto/chunk: `101.0 ms/step`, `warmup=249.3s`, `vram=2978 MB`, `final_loss=5.5210`

Practical conclusion:

- the old T=256 compile hang is fixed
- steady-state training is dramatically faster and materially better than the earlier `~205 ms/step` chunk-compile baseline
- compile warmup is still expensive and remains open work

### Recorded No-Go: Chunk-Local Ring Strip Cache

Do not treat a generic or fast-path-only ring strip cache as the current best next fix.

- A narrow strip-cache attempt was tested on `2026-03-08` for the validated fast path:
  - one contiguous strip gathered per chunk
  - read+write performed against the strip
  - one flush back into the dense ring at chunk end
- It preserved numerics in tests but regressed the real compiled path:
  - previous stable compiled path: `~101.0 ms/step`, `warmup=249.3s`
  - strip-cache attempt: `127.6 ms/step`, `warmup=703.7s`
  - profiler also worsened `write_replace`
- Current policy:
  - keep the existing dense write path
  - record the strip-cache idea as a tried-and-reverted path
  - do not reintroduce it without a materially different implementation strategy

## What Changed And Why

- Chunk compile was chosen because it bounds graph size while preserving sequential timestep semantics.
- Compile policy is a runtime/training concern, not a model architecture concern, so it stays outside the serialized build spec.
- Compiled paths keep only scalar diagnostics because `.item()` and rich list/trace formatting inside the graph cause graph breaks or retracing.
- Gradient checkpointing and chunk compile are mutually exclusive in the active path; chunk compile takes precedence.
- The single-expert fast path was added because the validated production shape is narrow enough to specialize safely without changing model semantics.

## Known Failed Or Deferred Paths

- Full-model compile at `T>=128` is not acceptable in the current architecture.
- BB compile support is deferred because Python scalar state and associated logic are not yet tensorized.
- Rich list/trace telemetry inside compiled chunks is intentionally not preserved in this pass.
- Compile warmup is still long even though steady-state performance is fixed.
- CPU compile is not part of the supported performance path in this pass.
- The chunk-local strip-cache write experiment is a recorded no-go in its current form and was reverted after benchmark/profiler regression.

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

1. Run a `compile_chunk_size` sweep (`16/24/32/48/64`) to optimize the current chunk-compile wall-clock and warmup tradeoff without touching model math.
2. Replace top-level `forward()` chunk output list accumulation with direct output preallocation/slice writes.
3. Run an isolated `cudagraph_mark_step_begin()` A/B probe to see whether the remaining CUDAGraph warning can be reduced without destabilizing training.
4. Add sampled/minimal telemetry modes to the canonical nightly runner to improve research turnaround without changing model behavior.
5. Revisit deeper write-path work only through materially different approaches (for example fused/custom ops), not by reviving the reverted strip-cache patch as-is.
