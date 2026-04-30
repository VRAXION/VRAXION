# Beta.8 Checkpoint Notes

Checkpoint: `output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt`
Inspected: 2026-04-30
SHA256: `d63200504c5b4a6ea2134fd26e3e3d7cb75ff05884236de2cc6e206bb4ba8d54` — verified MATCH

---

## On-disk format

**Type:** Custom binary — Rust `bincode` 1.3 serialization. No file magic bytes, no JSON, no numpy/pickle.

**Bincode 1.3 wire rules (what Python must implement):**
- All integers: fixed-width, little-endian
- `usize` / `isize` in struct fields: serialized as `u64` / `i64` (8 bytes)
- `Vec<T>`: `u64` element-count (8 bytes), then `count` elements back-to-back
- `String`: `u64` byte-count (8 bytes), then raw UTF-8 bytes
- `f64`: IEEE 754 little-endian (8 bytes)
- Struct fields are laid out in declaration order, no padding, no tags

**Two-layer nesting (key gotcha):** the outer struct stores `network_bytes` and `projection_bytes` as `Vec<u8>`. Each of those blobs is itself a second, independent bincode serialization of its respective inner struct. A Python loader must decode the outer struct first, then decode each blob separately.

---

## Struct layout (byte-level)

### Outer `CheckpointDisk` (checkpoint.rs)

| Field | Type | Width | Notes |
|---|---|---|---|
| `version` | `u8` | 1 B | Wire format version; currently `1` |
| `network_bytes` length | `u64 LE` | 8 B | Length of the following blob |
| `network_bytes` payload | `[u8; n]` | n B | Bincode of `NetworkDiskV1` (see below) |
| `projection_bytes` length | `u64 LE` | 8 B | Length of the following blob |
| `projection_bytes` payload | `[u8; n]` | n B | Bincode of `ProjectionDisk` (see below) |
| `meta.step` | `u64 LE` | 8 B | Evolution step counter |
| `meta.accuracy` | `f64 LE` | 8 B | Best known accuracy at save time |
| `meta.label` length | `u64 LE` | 8 B | UTF-8 byte count |
| `meta.label` payload | `[u8; n]` | n B | UTF-8 string |

### Inner `NetworkDiskV1` (network/disk.rs) — lives inside `network_bytes`

| Field | Type | Notes |
|---|---|---|
| `version` | `u8` | Inner network version; currently `1` |
| `graph.neuron_count` | `u64 LE` | `H` (number of neurons) |
| `graph.sources` | `u64` len + `[u64]` | Edge source indices |
| `graph.targets` | `u64` len + `[u64]` | Edge target indices; same count as sources |
| `threshold` | `u64` len + `[u32]` | Per-neuron firing threshold, 1 per neuron, range `[0, 15]` |
| `channel` | `u64` len + `[u8]` | Per-neuron phase channel, 1 per neuron, range `[1, 8]` |
| `polarity` | `u64` len + `[i32]` | Per-neuron polarity, 1 per neuron, values `{-1, +1}` |

Invariants enforced by `disk::validate()`:
- `sources.len() == targets.len()`
- `threshold.len() == channel.len() == polarity.len() == neuron_count`
- All edge endpoints in `[0, neuron_count)`, no self-loops, no duplicate edges
- `threshold[i] <= 15`, `channel[i] in 1..=8`, `polarity[i] in {-1, +1}`

### Inner `ProjectionDisk` (projection.rs) — lives inside `projection_bytes`

| Field | Type | Notes |
|---|---|---|
| `weights` | `u64` len + `[i8]` | Row-major weight matrix, `input_dim × output_classes` elements |
| `input_dim` | `u64 LE` | Output-zone neuron count |
| `output_classes` | `u64 LE` | Number of prediction classes |

Invariant: `weights.len() == input_dim * output_classes`

---

## Beta.8 specific values

| Property | Value |
|---|---|
| File size | 259,981 bytes (253.9 KB) |
| Outer version | 1 |
| Inner network version | 1 |
| `neuron_count` (H) | 384 |
| `edge_count` | 10,140 |
| Threshold range | 0 – 15 (all valid values present) |
| Channel set | 1, 2, 3, 4, 5, 6, 7, 8 (full range) |
| Polarity set | {-1, +1} |
| `input_dim` (projection) | 237 |
| `output_classes` | 397 |
| `weight_count` | 94,089 |
| Nonzero weights | 93,742 (99.6%) |
| `step` at save | 24 |
| `accuracy` at save | ~4.89% (0.048875) |
| Label | `D9.2a mo rank=1 class=FULL_GENERALIST smooth_delta=0.016599 unigram_delta=0.004900` |
| network_bytes blob | offset 0x000001 – 0x028779 (165,745 B) |
| projection_bytes blob | offset 0x02877A – 0x03F722 (94,113 B) |
| meta tail | offset 0x03F723 – EOF |

---

## Compatibility table

| Field | Category | Notes |
|---|---|---|
| outer `version` | Static metadata | Bump = breaking change; Python loader must check `== 1` |
| inner network `version` | Static metadata | Bump = breaking change; check `== 1` |
| `neuron_count` | Static metadata | Architecture constant for this checkpoint |
| `edge_count` | Static metadata | Topology; not adjustable post-load |
| `sources`, `targets` | Static metadata | Topology arrays; fixed at load time |
| `threshold` | Static metadata | Learned genome parameter; fixed at inference time |
| `channel` | Static metadata | Learned genome parameter; fixed at inference time |
| `polarity` | Static metadata | Learned genome parameter; fixed at inference time |
| `weights` (projection) | Static metadata | Learned weight matrix; fixed at inference time |
| `input_dim`, `output_classes` | Static metadata | Architecture constants for the projection |
| `step`, `accuracy`, `label` | Informational | Provenance only; not used at inference time |
| Activation state | Runtime-computed | Not stored in checkpoint; always initialized to zero on load |
| Charge vector | Runtime-computed | Not stored; computed during propagation |
| Refractory timers | Runtime-computed | Not stored; reset to zero on load |
| SDR input patterns | Runtime-computed | Provided by caller; not part of the checkpoint |

---

## What would a Python loader need

```python
def load_beta8_checkpoint(
    path: str | Path,
) -> tuple["NetworkGenome", "Int8Projection", "CheckpointMeta"]:
    """
    Load a VRAXION .ckpt file (bincode 1.3, version 1).

    Parameters
    ----------
    path
        Filesystem path to the .ckpt file.

    Returns
    -------
    genome
        Frozen network topology + per-neuron parameters.
        Fields: neuron_count, sources, targets, threshold, channel, polarity.
    projection
        Int8 output weight matrix.
        Fields: weights (row-major i8[]), input_dim, output_classes.
    meta
        Provenance metadata (step, accuracy, label). Not used at inference.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If outer or inner version != 1, or any validation invariant fails.
    struct.error
        If file is truncated or otherwise malformed.

    Notes
    -----
    - Do NOT pass a .sha256 path; verify separately with verify_sha256().
    - The file has no magic bytes. Version check is the only format guard.
    - The returned types carry no runtime state (activation/charge/refractory).
      Those are initialized to zero and computed by the propagation engine.
    - Two bincode decode passes are required: one for the outer CheckpointDisk,
      then separate passes for network_bytes and projection_bytes.
    """
    ...
```

---

## Gotchas for the future loader

1. **No magic bytes.** There is nothing at the file start to distinguish a
   valid checkpoint from random garbage. The first byte `0x01` is the version
   field. A loader must catch `struct.error` / `EOFError` gracefully.

2. **Two-pass bincode decoding.** `network_bytes` and `projection_bytes` are
   opaque blobs inside the outer struct. Each requires its own separate decode.
   Do not attempt to parse all three structs as a single flat byte stream.

3. **`usize` is `u64` on disk.** Rust `usize` is always encoded as 8 bytes by
   bincode 1.x regardless of the host platform. Python's `struct` must use `<Q`
   (unsigned 64-bit LE) wherever the Rust source says `usize`.

4. **`Vec<u8>` (channel array) is length-prefixed.** Even though its elements
   are single bytes, bincode still writes a `u64` count followed by the bytes.
   The `read_vec_u8` helper in the inspection script shows the correct decode.

5. **Step counter = 24, accuracy = 4.89%.** These are training-phase values.
   The label string (`D9.2a mo rank=1 class=FULL_GENERALIST ...`) identifies
   the experiment that produced the checkpoint. Neither figure is an inference
   parameter.

6. **Projection is nearly fully dense.** Only 347 of 94,089 weights are zero
   (0.4% sparsity). Sparsify is available on the Rust side but was not applied
   to this checkpoint.

7. **Output zone width (input_dim = 237) < neuron_count (384).** The projection
   reads only the output zone of the charge vector, not the full network state.
   The split point is `neuron_count - input_dim = 147`, meaning neurons 0–146
   are hidden neurons and neurons 147–383 are the output zone.
   (Verify this against `InitConfig::output_start()` in the Rust source before
   hardcoding.)
