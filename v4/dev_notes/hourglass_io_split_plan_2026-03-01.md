# Hourglass I/O Split Plan (WRITE-only -> RING -> READ-only)

Date: 2026-03-01  
Owner: Claude + Codex handoff  
Status: Design-ready (implementation spec)

## 1) Why this test matters

The observed "expert collapse" (one expert mostly writes, the other mostly reads) was **emergent** and unstable.  
This plan proposes a **forced, clean separation**:

- Writer expert(s): input -> hidden -> write ring, **no ring read**
- Reader expert(s): ring read -> hidden -> output, **no ring write**

This directly tests the hypothesis:

> If a strict memory highway is useful, forcing roles should improve credit assignment and eval robustness.  
> If it is not useful, results stay flat or drop.

Important: this is not a cosmetic refactor. It is a causal experiment about architecture.

---

## 2) Critical distinction: collapse vs strict split

### Emergent collapse (what we already saw)
- No hard constraints.
- One branch can silently become dead (grad ~0).
- Model can still bypass ring through hidden/output shortcuts.

### Strict hourglass split (this plan)
- Hard role masks.
- Output comes from **reader-only** path (optional strictness toggle).
- Writer cannot leak directly to output.

If strict split still gives no gain, then "role split" is not the bottleneck.

---

## 3) Minimal architecture change

Target file: [`model/instnct.py`](S:/AI/work/VRAXION_DEV/v4/model/instnct.py)

Add config-driven role mode:

- `io_split_mode`: `off | strict`
- `io_writer_count`: integer, default `1`
- `io_output_from_readers_only`: bool, default `true` in strict mode

Current `N=2` default:
- expert 0 = writer
- expert 1 = reader

General rule for `N>2`:
- first `io_writer_count` experts are writers
- remaining experts are readers

---

## 4) Exact forward semantics (strict mode)

Within `_process_chunk(...)` expert loop:

### Writer expert
- Skip ring read (`read_vec = 0`, `ring_signal = 0`)
- Hidden update uses `input + phase + prev_hidden` (and optional BB if enabled)
- Performs write as usual
- Pointer moves as usual

### Reader expert
- Performs ring read as usual
- Hidden update uses `input + S*ring_signal + phase + prev_hidden` (and optional BB)
- **Skip write**
- Pointer moves as usual

### Output aggregation
- If `io_output_from_readers_only=true`: average only reader hidden states
- Else: legacy average over all experts

This is the key anti-cheat guard.

---

## 5) Snippet-level implementation sketch

File: `model/instnct.py`, `__init__`:

```python
self.io_split_mode = _cfg.get('io_split_mode', 'off')  # off|strict
self.io_writer_count = int(_cfg.get('io_writer_count', 1))
self.io_output_from_readers_only = bool(_cfg.get('io_output_from_readers_only', True))

writer_mask = torch.zeros(self.N, dtype=torch.bool)
writer_mask[: max(0, min(self.N, self.io_writer_count))] = True
reader_mask = ~writer_mask
if self.io_split_mode == 'strict' and not reader_mask.any():
    raise ValueError("strict io_split requires at least one reader expert")

self.register_buffer('_writer_mask', writer_mask)
self.register_buffer('_reader_mask', reader_mask)
```

File: `model/instnct.py`, `_process_chunk(...)`, inside `for i in range(N):`

```python
is_writer = bool(self._writer_mask[i]) if self.io_split_mode == 'strict' else False
is_reader = bool(self._reader_mask[i]) if self.io_split_mode == 'strict' else True

if is_reader:
    # existing read path (vshape/dotprod/topk)
    ...
    ring_signal = self.read_proj[i](read_vec_tns)
    blended_ring = S_flt * ring_signal if S_flt != 'dotprod' else ...
else:
    # writer-only: no ring read contribution
    ring_signal = torch.zeros_like(input_vec_tns)
    blended_ring = torch.zeros_like(input_vec_tns)

# hidden update unchanged except blended_ring maybe zero
hidden_lst[i] = _c19_activation(input_vec_tns + blended_ring + bb_ctx + phase_tns + hidden_lst[i], ...)

if not (self.io_split_mode == 'strict' and is_reader):
    # writer (or legacy) writes
    ring_tns = func_softwrit_tns(...)
```

File: `model/instnct.py`, output section:

```python
if self.io_split_mode == 'strict' and self.io_output_from_readers_only:
    reader_idx = self._reader_mask.nonzero(as_tuple=False).flatten().tolist()
    mean_hidden = torch.stack([hidden_lst[j] for j in reader_idx]).mean(0)
else:
    mean_hidden = torch.stack(hidden_lst).mean(0)
```

---

## 6) Telemetry required (must-add)

Add these to `self._diag` and CSV logging path:

- `io_writer_hidden_norm`
- `io_reader_hidden_norm`
- `io_writer_write_norm`
- `io_reader_write_norm` (should be ~0 in strict)
- `io_writer_ring_signal_norm` (should be ~0 in strict)
- `io_reader_ring_signal_norm`
- `io_output_reader_only` (0/1)
- `io_mode` encoded integer (`0=off,1=strict`)

Also keep existing:
- `ring_norm`, `ring_slot_mean`
- `masked_acc`, `masked_loss`
- fresh-state eval metric

---

## 7) Acceptance criteria (hard)

For strict mode sanity:

1. Writer `ring_signal_norm ~ 0` throughout.
2. Reader write contribution ~0.
3. Output uses only readers (`io_output_reader_only=1`).
4. No NaN/Inf, no shape regression.

Performance decision rules:

- If strict split >= baseline by `+1.0pp` masked_acc at equal steps: keep exploring.
- If strict split in `[-1.0pp, +1.0pp]`: neutral, not primary bottleneck.
- If strict split <= baseline by `-1.0pp`: drop as main path.

---

## 8) Experiment matrix (short + decisive)

Hold constant:
- same data shard(s), same seed, same seq_len, same batch, same lr schedule
- same model size and kernel mode

Runs:

1. `baseline_off`  
   - `io_split_mode=off`

2. `hourglass_strict`  
   - `io_split_mode=strict`
   - `io_writer_count=1`
   - `io_output_from_readers_only=true`

3. `hourglass_strict_S0_probe`  
   - same as #2 + eval with `S->0` ablation  
   - checks whether reader is truly dependent on ring

Steps:
- quick gate: 1k steps
- confirm: 3k steps

Primary metrics:
- masked_acc, masked_loss
- train/eval gap
- fresh-state eval

---

## 9) Adversarial risks and mitigations

### Risk A: writer dumps unbounded noise
Cause: additive `scatter_add` write accumulates forever.
Mitigation:
- keep `S` conservative
- optionally add write damping later:
  - `slot_new = (1 - w)*slot_old + w*write_vec` (future patch if needed)

### Risk B: reader ignores ring anyway
Mitigation:
- enforce reader-only output path
- run `S->0` eval probe to test causal dependency

### Risk C: pointer policy dominates result
Mitigation:
- keep pointer mode fixed during A/B
- do not co-change pointer algorithm in same run

---

## 10) Recommendation for next action

Implement strict split exactly as above, run 1k + 3k A/B, and decide fast.

This is a high-value falsification test:
- If it wins: we got a cleaner memory routing primitive.
- If it fails: we can stop debating role split and move to reward/credit-path redesign.

