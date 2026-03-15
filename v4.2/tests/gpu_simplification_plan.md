# GPU Simplification Plan

## Current state

On the current GPU path:

- mask is already ternary `int8`
- controller knobs are already small ints
- the eval kernel still uses dense floating-point math for:
  - `acts`
  - `charges`
  - `softmax`
  - `weff = mask -> float`

This is the right place to be today. The hot path is not yet a good candidate
for a naive "full int everywhere" rewrite.

## What current PyTorch/CUDA reality implies

The current official PyTorch direction does **not** say that arbitrary
CUDA-side int8 or binary matmul is a default free win.

- `torch.mm` documents CUDA-specific `out_dtype` support for
  `float16` / `bfloat16` inputs, which is the main easy fast path in regular
  eager/compiled PyTorch.
- Current torchao quantized inference docs are focused on `functional.linear`
  style inference, not a custom recurrent graph kernel like ours.
- torchao's published H100 benchmarks even show `int8_rowwise` underperforming
  the `bfloat16` baseline in at least one reported setup.

Conclusion:

- "smaller dtype" is not equal to "faster" on GPU
- for our graph kernel, **bf16/fp16-style simplification is a more realistic
  next step than int8/binary execution**

## Canonical simplification goals

### Already good

- ternary `int8` mask
- int controller state
- merged constants (`DRIVE`, `SELF_DRIVE`)
- compiled eval runner

### Still realistic on GPU

1. keep the proposal path random-first
2. remove host-sync and Python overhead from mutation
3. reduce eval precision only where CUDA has a real fast path
4. delay true integer/binary execution until a custom kernel phase

## Staged plan

### Phase 1: Freeze the current baseline

Baseline to defend:

- random-first proposal
- ternary `int8` mask
- float eval buffers
- compiled eval

Acceptance metric:

- keep current quality while improving attempts/sec

### Phase 2: Mixed-precision eval probe

Goal:

- test whether `acts` / `charges` can move from `float32` to `bfloat16`
  or `float16` without unacceptable quality loss

Variants:

- `fp32` baseline
- `bf16 acts+charges`
- `fp16 acts+charges`
- optional `bf16 acts, fp32 charges`

Keep:

- mask ternary `int8`
- logits/softmax accumulation in `float32`
- controller logic unchanged

Why this is the best next simplification:

- it maps to actual CUDA fast paths
- it is easy to isolate
- it keeps the math structurally identical

### Phase 3: Materialization simplification

Goal:

- reduce or eliminate repeated `mask -> float` overhead

Candidates:

- keep dense `weff` but reuse buffers harder
- compare `mask.float()` materialization against
  - sign-decomposed `pos/neg` bool buffers
  - cached float weight buffer refreshed only on accepted mutations

Important:

- this phase must be measured, not assumed
- more "integer-looking" code is not automatically faster on CUDA

### Phase 4: GPU-native mutation backend

Goal:

- move the mutation hot path fully on-device

Needed:

- on-device alive-edge metadata
- on-device random picks
- no `.item()`
- no `torch.nonzero(...)` in the inner loop

This is likely a larger win than forcing more math into int8 too early.

### Phase 5: Fixed-point experiment

Only after Phases 2-4:

- try fixed-point state/charge
- likely `int16` / `int32` accumulators
- keep logits path high precision if needed

Reason:

- accumulators are the part that actually need range
- a true "full binary" network is unlikely to preserve the current dynamics

### Phase 6: Binary / custom-kernel research

This is the far end of the roadmap, not the next sprint.

Binary or bitpacked execution only makes sense if we are willing to build a
custom CUDA/Triton-style kernel. Regular PyTorch eager/compile is not the
right environment to expect an automatic win from this.

## Practical recommendation

If the next simplification sprint starts now, the order should be:

1. `bf16/fp16` eval probe
2. cached materialization probe
3. GPU-native mutation work
4. only then fixed-point / binary research

## Decision rule

A simplification only advances if:

1. quality is not worse by more than a small tolerance
2. attempts/sec improves materially
3. code complexity does not explode

If a change is "more integer" but slower or less stable on GPU, it is not a
real simplification for this project.
