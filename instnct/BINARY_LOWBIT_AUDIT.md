# INSTNCT Binary / Low-Bit Audit

This audit is the current engineering verdict for the **active mainline + quantized lanes**.

Rules used here:

- `green`: already minmaxed for its current role
- `yellow`: semantically correct, but still carries avoidable widening or mixed representation
- `red`: active path still uses the wrong physical representation and should be cleaned first
- `gray`: compatibility-only boundary; keep for now, but do not expand

A surface counts as **minmaxed** only if:

- runtime dtype is the narrow truthful representation
- checkpoint dtype is equally compact
- compatibility is isolated at the boundary
- hot paths do not widen unless the math truly requires it
- topology presence is kept separate from sign/magnitude unless mixing is required
- self-edge constraints stay explicit and cheap

## Mainline Audit

| Surface | Purpose | Runtime dtype | Checkpoint dtype | Widening / baggage | Class | Recommended action |
|---|---|---|---|---|---|---|
| `graph.py` topology mask | Hidden-to-hidden connection presence | `bool` | `rows/cols uint16|uint32`, `vals bool` | None on active storage path | `green` | Keep as the canonical binary truth |
| `graph.py` sparse binary cache | Multiply-free active forward path | `(rows, cols)` index arrays | n/a | None on the active `edge_magnitude == 1.0` path | `green` | Keep as the default path |
| `graph.py` polarity | Per-neuron sign | `int8` + cached `_polarity_f32` | `int8` | One cached float companion for math | `green` | Keep; this widening is justified and amortized |
| `graph.py` refractory | Per-neuron cooldown | `int8` | not checkpointed | None | `green` | Keep |
| `graph.py` control scalars (`loss_pct`, `mutation_drive`) | Low-bit search control | `int8` | currently saved via Python `int` scalars | Save path is wider than runtime truth | `yellow` | Save as explicit low-bit dtype in a later cleanup |
| `graph.py` legacy sparse/dense compat path | Backward compat for old float/ternary or non-unit magnitude | mixed `float32` branch | old checkpoints still load | Active rollout methods still contain float fallback logic | `yellow` | Keep load compat, but push legacy widening to the boundary only |
| `train_english_1024n_18w.py` topology mask | Canonical English structural truth | effectively `bool` from `net.mask`, but treated like float in worker/eval code | recipe-local, inherited from graph checkpointing | `mask != 0`, `1.0/0.0` writes, `mask * polarity` mixes topology with sign | `yellow` | Make the bool contract explicit and stop writing float-style mask logic |
| `train_english_c19_truth_probe.py` topology mask | Truth-probe structural state | `float32` | recipe-local only | Active path initializes `mask` as `float32` and mutates via `1.0/0.0` | `red` | First cleanup target: convert to explicit `bool` topology |
| `gen_viz_connection_matrix.py` loader | Binary consumer for checkpoints | `rows/cols int32` in-memory for UI payload | consumes compact rows/cols/vals or `mmag` | Filters old `vals != 0`, but does not widen topology meaning | `green` | Keep |
| `gen_viz_layer_workbench.py` derived binary layers | One-page binary/scalar workbench | matrix layers stay index-based; binary scalar strips are converted to `float32` for generic rendering | HTML payload only | `active/sink/source/isolated` derived layers go through float scalar path | `yellow` | Keep for now; later give binary strips their own non-float renderer/data kind |

## Quantized Lane Audit

| Surface | Purpose | Runtime dtype | Checkpoint dtype | Widening / baggage | Class | Recommended action |
|---|---|---|---|---|---|---|
| `train_wordpairs_loglik.py` sign mask (`msign`) | Edge sign | `bool` | `bool` | Converted to float in eval math | `green` | Keep storage contract; only review math-boundary widening |
| `train_wordpairs_loglik.py` magnitude mask (`mmag`) | Edge magnitude | `uint8` | `uint8` | Converted to float in eval math | `green` | Keep storage contract |
| `train_wordpairs_loglik.py` `inj_table` | Quantized input projection | `int8` | `int8` | Per-step `.astype(np.float32) / 128.0` inside eval loops | `yellow` | Hoist float companions outside inner loops later |
| `train_wordpairs_loglik.py` `output_projection_int8` | Quantized readout projection | `int8` with one derived `output_projection_f` companion | `int8` | Clean one-time widening only | `green` | Keep |
| `train_wordpairs_loglik.py` eval rollout | Quantized runtime math | mixed low-bit storage + `float32` math | n/a | Repeated `astype(np.float32)` on `msign`, `mmag`, and `inj_table` per eval path | `yellow` | Cache scaled float views per run/worker instead of rebuilding repeatedly |
| `ops_train_bigram_english.py` sign+mag storage | Overnight quantized lane | `msign bool`, `mmag uint8` | same | Same pattern as task-memory lane | `green` | Keep storage contract |
| `ops_train_bigram_english.py` quantized projections | Input/readout quantization | `inj_table int8`, `output_projection_int8 int8`, one `output_projection_f` companion | same | Clean one-time widening for readout, repeated widening for input in eval loop | `yellow` | Same cleanup as task-memory lane: hoist scaled float companions |

## Compatibility Boundary

These are intentionally **not** first-pass cleanup targets.

| Surface | Why it exists | Class | Decision |
|---|---|---|---|
| `graph.py::load()` converting old `vals != 0` | Old checkpoints stored float or ternary edge values | `gray` | Keep |
| `graph.py` legacy `drive` field fallback | Old checkpoints saved `drive` instead of `mutation_drive` | `gray` | Keep |
| Viz loaders accepting both `rows/cols[/vals]` and `mmag` | Current workbench consumes both mainline and quantized checkpoints | `gray` | Keep |

## Prioritized Fix Order

### 1. Truth-probe mask cleanup

- **Target representation:** `mask: bool` end-to-end inside `train_english_c19_truth_probe.py`
- **Keep compatibility:** none beyond local recipe behavior
- **Must not change:** the three rollout modes and report structure
- **Proof:** short seeded 3-mode probe still runs, same report keys, no float mask state remains

### 2. Canonical English mask cleanup

- **Target representation:** explicit `bool` topology in worker/eval code; sign comes only from `polarity`
- **Keep compatibility:** recipe remains compatible with `SelfWiringGraph` checkpoints and current schedule
- **Must not change:** `2 add / 1 flip / 5 decay` schedule behavior, self-edge block, reported accuracy flow
- **Proof:** short seeded recipe smoke, unchanged edge counts, no float-style mask writes or mask-sign mixing

### 3. Core compat isolation

- **Target representation:** `graph.py` active default remains pure-binary topology with binary sparse cache
- **Keep compatibility:** old checkpoint load stays; non-unit `edge_magnitude` support may remain if still needed
- **Must not change:** `test_model.py` round-trips, current checkpoint schema, current forward behavior on active saves
- **Proof:** `test_model.py`, old checkpoint load smoke, new compact save/load round-trip

### 4. Quantized hot-path cleanup

- **Target representation:** keep `msign bool`, `mmag uint8`, `inj_table int8`, `output_projection_int8 int8`
- **Keep compatibility:** checkpoint schema unchanged
- **Must not change:** exact quantized math semantics, final save fields, worker proposal behavior
- **Proof:** short smoke for `train_wordpairs_loglik.py` and `ops_train_bigram_english.py`; dtype assertions on saved checkpoints

### 5. Workbench binary-strip cleanup

- **Target representation:** binary derived neuron layers stay binary/int-like until the UI serialization boundary
- **Keep compatibility:** one-page workbench layout and current layer ids stay stable
- **Must not change:** counts and visible structure in the HTML output
- **Proof:** regenerated workbench shows identical stats for connection/reciprocal/sink/source/isolated layers

## Immediate Verdict

- The **core checkpointed topology** is already in good shape.
- The **biggest active mismatch** is the truth-probe still storing topology as `float32`.
- The **canonical English recipe** is closer than it looks because its source mask is effectively bool, but the code still expresses it as a float-style mask workflow.
- The **quantized lanes** are structurally clean on storage, but still have obvious float32 rebuild points inside evaluation loops.
