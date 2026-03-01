# Consolidation Plan - Single Canonical Runtime (Claude Task Board)

**Date**: 2026-02-28  
**Author**: Codex (adversarial audit pass)  
**Status**: PLAN ONLY (no feature changes in this pass)

---

## 1) Context

Recent refactors fixed key behavior (fixed `R`, C19 init scope bug, tests green), but the repo still has fragmentation risk.
This pass is for hard consolidation into one production-grade source tree.

Canonical runtime target:
- `v4/model/*`
- `v4/training/*`
- `v4/config/*`
- `v4/tests/*`
- `v4/datagen/*`

Policy decisions:
1. One canonical runtime tree only.
2. Hard consolidation (not minimal patching).
3. Break-now checkpoint compatibility (no legacy shim).

---

## 2) Adversarial Findings

### 2.1 Green signals
1. `pytest -q` currently passes (`115/115`).
2. Runtime model uses fixed-R buffer (`_R_eff`) instead of learnable `R_param`.
3. Checkpoint version gating exists in `training/train.py`.

### 2.2 Red signals
1. Structural fragmentation remains (legacy flat traces vs structured runtime tree).
2. Stale bench scripts still reference removed API (`R_param`) and can mislead.
3. C19 bound doc mismatch remains:
   - `_C19_C_MAX = 50.0`
   - `_C_from_raw` docstring currently says `[1.0, 10.0]` (outdated)

### 2.3 Confirmed critical runtime files
- `v4/model/instnct.py`
- `v4/training/train.py`
- `v4/training/eval.py`
- `v4/config/vraxion_config.yaml`

### 2.4 Confirmed stale files to port or archive
- `v4/tests/bench_R_sweep_0123.py`
- `v4/tests/bench_kernel_R_matrix.py`
- `v4/tests/bench_learnable_R.py`

---

## 3) Locked Decisions

1. Keep structured subfolder runtime layout as single source of truth.
2. Remove/retire legacy duplicate runtime surfaces.
3. No backward compatibility layer for old checkpoint schema.
4. No architectural experiment mixed into this consolidation pass.

---

## 4) Claude Execution Plan (Decision Complete)

## Phase A - Canonicalization
1. Ensure entrypoints import only from `v4/model`, `v4/training`, `v4/config`.
2. Remove or archive legacy flat runtime duplicates from active tracked paths.
3. Keep experiments and archive material outside active runtime import paths.

## Phase B - Fixed-R closure
1. Remove any remaining runtime dependency on `R_param`.
2. Keep `R` config as sole radius source.
3. Keep `_R_eff` as explicit model buffer and assert in tests.

## Phase C - Stale bench handling
1. For each stale bench file:
   - either port to fixed-R API, or
   - move to `ARCHIVE/legacy_bench/` and mark unsupported.
2. Do not leave dead `R_param` benches inside active `tests/`.

## Phase D - Checkpoint contract
1. Bump checkpoint schema version after consolidation.
2. Reject old versions loudly with explicit error.
3. Keep format strict and deterministic.

## Phase E - Logging correctness
1. Replace formula-based param estimates in BOOT log with exact `requires_grad` counts.
2. Report by subsystem (`input`, `core`, `output`, `c19_meta`) for diagnostics.

## Phase F - Drift guards
1. Add static anti-legacy test:
   - fail if forbidden symbols appear in runtime files (`R_param`, etc.).
2. Add import-path guard:
   - runtime cannot import from archived/legacy paths.

## Phase G - Verification gates
1. `pytest -q` green.
2. Drift/static guard tests green.
3. Coherent runtime tree only (no active duplicate runtime modules).
4. Smoke train boot/save/load works with current schema.

---

## 5) Concrete Snippets for Claude

### 5.1 C19 docstring bound fix (`v4/model/instnct.py`)

```python
def _C_from_raw(raw_C):
    """Sigmoid-bounded C: raw float -> [1.0, 50.0]."""
    return _sigmoid_bounded(raw_C, _C19_C_MIN, _C19_C_MAX)
```

### 5.2 Exact trainable param counting (`v4/training/train.py`)

```python
def _count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

n_total = _count_trainable(model)
print(f"[BOOT] Params     {n_total:,} trainable")

n_inp = _count_trainable(model.inp) if hasattr(model, "inp") and isinstance(model.inp, torch.nn.Module) else 0
n_out = _count_trainable(model.out) if hasattr(model, "out") and isinstance(model.out, torch.nn.Module) else 0
n_core = n_total - n_inp - n_out
print(f"[BOOT] ParamsBy   input={n_inp:,} core={n_core:,} output={n_out:,}")
```

### 5.3 Break-now checkpoint gate (`v4/training/train.py`)

```python
CKPT_VERSION = 2

def func_loadckpt_dct(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Expected checkpoint dict, got {type(ckpt).__name__}")
    saved_ver = ckpt.get("ckpt_version", 0)
    if saved_ver != CKPT_VERSION:
        raise RuntimeError(
            f"Checkpoint version mismatch: file v{saved_ver}, code v{CKPT_VERSION}. "
            "Legacy checkpoints are intentionally unsupported in this build."
        )
    return ckpt
```

### 5.4 Anti-legacy runtime test (`v4/tests/test_no_legacy_runtime_symbols.py`)

```python
from pathlib import Path

RUNTIME_FILES = [
    Path("model/instnct.py"),
    Path("training/train.py"),
    Path("training/eval.py"),
]
FORBIDDEN = ["R_param"]

def test_no_legacy_symbols_in_runtime():
    root = Path(__file__).resolve().parents[1]
    for rel in RUNTIME_FILES:
        text = (root / rel).read_text(encoding="utf-8")
        for sym in FORBIDDEN:
            assert sym not in text, f"{sym} found in runtime file: {rel}"
```

### 5.5 Legacy bench archive marker

```python
"""
LEGACY BENCH (archived)
Reason: depends on removed learnable-R API (R_param).
Status: not part of active CI/runtime validation.
"""
```

---

## 6) Acceptance Criteria (all required)

| Gate | Condition |
|---|---|
| Unit tests | `pytest -q` passes |
| Drift guard | anti-legacy runtime symbol test passes |
| Checkpoint contract | old ckpt hard-fails with clear message |
| Boot diagnostics | exact trainable counts (no stale formulas) |
| Repo hygiene | one canonical runtime tree only |

---

## 7) First A/B After Consolidation

Run only after consolidation lock:
- A: current best baseline config
- B: identical config except `R=0`

Hold constant:
- data
- LR schedule
- batch
- seq_len
- seed policy

Track:
- `masked_acc`
- `masked_loss`
- step time
- stability spikes

Ship B only if quality is >= A and speed/compute is measurably better.

---

## 8) Notes to Claude

1. Keep this PR/pass surgical: structure + contracts + anti-drift guards.
2. Do not mix pointer/content-address or other architecture experiments into this consolidation pass.
3. After merge, cut a fresh baseline tag for reproducible follow-up experiments.

---

