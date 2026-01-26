# VRAXION Architectural Locks

These choices are **architectural**. Changing them later invalidates existing
checkpoints and "tenured" experts. Adjust only by training a new model variant.

## Locked Decisions (Phase C)

### 1) Byte Vocabulary (In-band tokens)
- **Vocab size:** 272
- **Token map:**
  - `0–255`: raw bytes
  - `256`: `<BOS>`
  - `257`: `<EOS>`
  - `258`: `<PAD>`
  - `259`: `<SEP>`
  - `260`: `<CODE>`
  - `261`: `<TEXT>`
  - `262`: `<VISION>`
  - `263`: `<AUDIO>`
  - `264–271`: reserved

### 2) Model Width / Slot Dim
- **`slot_dim` / `d_model`: 576**
- Rationale: best CPU sweet spot in bench (speed + loss) vs. 512/640/768.

### 3) Main Ring Length (Working Memory)
- **`ring_len`: 8192**
- Rationale: intelligence‑first profile on this hardware; lower jitter and
  better long‑horizon stability than 2048/4096 at the cost of throughput.

### 4) Short‑Term Memory Vault (STM)
- **`vault_len`: 4096** (ratio **0.5** of `ring_len`)
- **`vault_dim`: 64**
- Rationale: empirical vault sweep showed lowest mean loss at `vault_dim=64`
  for `slot_dim=576`. Ratio 0.5 consistently reduced jitter.

### 5) Precision Split
- **Weights / model dtype:** **fp32**
- **Pointer dtype:** **fp64**
- Rationale: fp32 weights are ~2.2× faster on Ryzen 3900X while keeping fp64
  pointer stability.
