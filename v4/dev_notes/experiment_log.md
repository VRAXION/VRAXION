# INSTNCT v4 — Experiment Log

Newest entries at top. Copy from [TEMPLATE.md](TEMPLATE.md) for new entries.

---

## 2026-03-09 — Repo-Local Results Tables Published

Added a tracked, repo-local results store for current `nightly` runs:

- `v4/results/runs_master.csv`
- `v4/results/derived/runs_golden.csv`
- `v4/results/derived/runs_quarantined.csv`
- rebuild via `python tools/results_ingest.py`

Intent:

- keep canonical run summaries online on `origin/nightly`
- keep raw telemetry JSONs local/untracked
- make Steam Deck / low-bandwidth review possible without digging through `training_output/**`

The initial golden table includes the current-corpus CPU A/B runs, the long `Config C` run summary, and the meaningful needle-task memory/pointer/slot ablations.

---

## 2026-03-09 — Overnight Config C Run (5.19M params, WikiText-103)

**Setup**: Config C (hidden_dim=6144, slot_dim=256, M=1024, N=1), sequential mode,
AMP fp16, chunk-level torch.compile (reduce-overhead), WikiText-103 byte-level.
Ran ~9800 steps overnight before NaN crash.

### Results

| Metric       | Step 1   | Step 1000 | Step 5000 | Step 9000 | Step 9790 (crash) |
|-------------|----------|-----------|-----------|-----------|-------------------|
| loss        | 5.63     | 1.58      | 1.44      | 1.42      | 1.37 (best)       |
| accuracy    | 0.1%     | 54.3%     | 58.2%     | 58.3%     | 60.2% (best)      |
| alpha_0     | 0.596    | 0.586     | 0.500     | 0.299     | 0.274             |
| ring_norm   | 281      | 3651      | 3186      | 4962      | 5548              |
| grad_norm   | 5.63     | 1.71      | 1.55      | 1.57      | 1.54              |
| LR          | 0.00001  | 0.001     | 0.000990  | 0.000961  | 0.000961          |

### Key Findings

**1. Ring is actively useful (alpha × ring_signal = gain control)**
- alpha decreased 0.60 → 0.27, BUT ring_signal_norm grew 3.4 → 429
- product (alpha × ring_signal_norm) monotonically INCREASED: 2 → 128
- This is automatic gain control — the model reduces the gate as the signal strengthens
- The ring is not being shut off; it's contributing more over time

**2. Learnable parameter dynamics**
- `gate_tau`: 4.3 → 8.8 (monoton increase, actively shaping alpha)
- `write_proj` norm: 10 → 141 (ring writes getting stronger, 14× growth)
- `read_proj` norm: 47 → 64 (steady growth)
- `phase_sin/cos`: stabilized early, norms ~1.3

**3. Write gate collapse (gradient starvation)**
- `write_gate` weight/bias byte-identical between step 7000 and 9000
- Adam `exp_avg` = 0.0, `exp_avg_sq` ≈ 1e-7 → gradient flow fully dried up
- The gate IS in the optimizer (confirmed), but sigmoid saturation kills gradients
- Output always ≈ sigmoid(-0.88) ≈ 0.29, independent of context
- Note: this run had no `min_write_strength` floor (now added in nightly)

**4. Dead parameters confirmed (zero gradient, no Adam state)**
- `S_raw`: legacy from scalar ring gate, not used in dotprod mode
- `ring_signal_norm` (LayerNorm, 12,288 params): defined but never called in forward
- `c19_rho_input` + `c19_C_input` (12,288 params): dead in `learned` embed mode
  (active in `bitlift` mode — DO NOT delete, make conditional if cleaning up)

**5. Training plateau**
- Fast learning step 1-1000 (loss 5.63 → 1.58)
- Slow grind step 1000-9800 (loss 1.58 → 1.37, only 0.21 improvement over 8800 steps)
- LR plateau scheduler triggered 4× (0.001 → 0.000961), minimal effect
- Diminishing returns consistent with previous 710K run pattern

### NaN Crash Analysis (step 9793-9803)

**Root cause**: CUDA graph memory reuse defeats NaN rollback.

Chain of events:
1. Step 9793: forward OK, backward produces inf gradients → GRAD-INF, optimizer skipped
2. Step 9794: CUDA graph warning "pending uninvoked backwards", forward with clean
   ring_pre=5546 produces ring_post=nan
3. Steps 9795-9803: ring_pre=nan (rollback failed), 10 consecutive NaN → fatal crash

**Mechanism**: `reduce-overhead` CUDA graphs record fixed memory addresses. When the
optimizer step is skipped (GRAD-INF), the graph state becomes inconsistent. On the next
forward, the graph replays and overwrites input tensor memory. The rollback line
`prev_state = state` was a shallow dict reference — both point to the same tensor memory,
which CUDA graphs have now corrupted.

**Fix** (applied to work/VRAXION_DEV/v4, needs porting to nightly):
1. Deep-clone carry state before forward: `prev_state = {k: v.clone() ...}`
2. `torch.compiler.cudagraph_mark_step_begin()` before each forward pass

**Checkpoint**: ckpt_step_009000.pt verified healthy (loadable, step=9000, optimizer OK).
Run is resumable from 9K with the fix applied.

### Data
- Training CSV: `dev_notes/telemetry/overnight_configC_9800steps.csv`
- Checkpoints: work/VRAXION_DEV/v4/training_output/ckpt_step_{001000..009000}.pt

---

## 2026-03-09 — CPU A/B Canonical Defaults

- Ran clean CPU/no-compile A/B on the current nightly corpus with matched training settings (`steps=120`, `batch=2`, `seq=64`, sequential carry).
- `embed_encoding='learned'` beat `bitlift` on the current production-style smoke:
  - learned: `best_loss=2.578349`, `final_loss=2.748368`
  - bitlift: `best_loss=2.723226`, `final_loss=2.948056`
- Speed was effectively tied in this short CPU run (`~0.6s/step` both cases), so the quality win was not offset by a practical CPU slowdown.
- Updated canonical nightly default:
  - `embed_encoding: learned`
- Also rechecked pointer mode on the same current-corpus CPU smoke:
  - sequential beat pilot (`best_loss=2.723226` vs `2.844055`)
  - pilot remains promising on synthetic/needle tasks, but not the current canonical default for production-style training

---

## 2026-03-08 — Verified Nightly Runtime Corrections

- Verified current-nightly issues from the broader audit and fixed only the ones supported by the active `v4` worktree:
  - `training/inference.py` legacy checkpoint-key usage
  - hardcoded inference model construction
  - `hybrid_heads_spaced_scalar_gate` auxiliary-head overwrite
  - eval batch-mean aggregation bias
- Shipped ring-write-collapse mitigation:
  - `min_write_strength: 0.002`
  - additional write gate diagnostics (`write_strength_*`, `write_gate_logit_*`)
- Fixed read-side `S` was tested on CPU probes and rejected; it materially worsened loss.
- The NaN sequential-carry rollback behavior remains part of the active nightly path.
- Deferred from the broader Claude audit:
  - findings tied to files missing from the current `v4/nightly` tree
  - broader repo/research cleanup outside the active runtime

---

## 2026-03-08 — Chunk-Local Ring Strip Cache (No-Go, Reverted)

- Tried a narrow fast-path-only chunk-local strip cache for the validated nightly shape:
  - `N=1`, `R=1`, sequential pointer, replace write, dense replace impl, BB off, strict I/O off
- Goal:
  - gather one contiguous strip per chunk
  - do read+write against that strip
  - flush once at chunk end instead of cloning/scattering the full ring each timestep
- Result:
  - parity and compile-related tests passed
  - real GPU validation regressed badly enough to reject the change
- Measured regression versus the active nightly baseline:
  - previous stable compiled path: `~101.0 ms/step`, `warmup=249.3s`
  - strip-cache attempt: `127.6 ms/step`, `warmup=703.7s`
  - profiler also worsened `write_replace` (`~1431.1ms -> 1488.8ms`)
- Practical conclusion:
  - the extra strip gather/flush and backward/compile costs outweighed the smaller per-step write footprint
  - this should not be treated as the next low-risk optimization target in its current form
- Reverted locally after validation; not promoted to `origin/nightly` runtime code
- Better next tests:
  - `compile_chunk_size` sweep (`16/24/32/48/64`)
  - top-level `forward()` output prealloc
  - isolated `cudagraph_mark_step_begin()` A/B probe

---

## 2026-03-08 — Single-Expert Hot-Path Specialization

- Nightly production path now has a narrow fast path for the validated shape:
  - `N=1`, `R=1`, sequential pointer, replace-write, dense replace impl, BB off, strict I/O off, local vshape read
- `_process_chunk()` now dispatches between the generic path and a specialized single-expert helper.
- Chunk compile on the supported path compiles the fast helper directly.
- Added adversarial parity coverage for:
  - eager train + backward
  - `torch.no_grad()`
  - chunk-compile with identity-patched `torch.compile`
  - sequential carry across consecutive batches
  - explicit fallback cases (`N>1`, `R!=1`, BB, mtaps, strict I/O, proxy overlay, gated write, topk)
- Verified T=256 benchmark on nightly after the change:
  - eager `3819.3 ms/step`
  - compile-auto `101.0 ms/step`
  - warmup `249.3s`
  - final loss delta `0.0094`
- Practical result: steady-state chunk-compiled training improved again over the earlier `~205 ms/step` baseline, while keeping the compile/checkpoint/carry contracts green.

---

## 2026-03-08 — Nightly Source-of-Truth Consolidation

- `origin/nightly` is now the active source-of-truth branch for VRAXION v4 runtime and nightly research work.
- Canonical handoff/setup doc: [nightly_source_of_truth.md](nightly_source_of_truth.md)
- Compile stabilization is now part of the curated nightly runtime:
  - `seq_len <= 48` -> full-model compile
  - `seq_len > 48` -> chunk compile of `_process_chunk`
- Verified T=256 benchmark on nightly:
  - eager `3981.1 ms/step`
  - compile-auto `205.3 ms/step`
  - warmup still expensive (`200.1s`) but the old compile hang is gone
- CPU WikiText carry smoke also passed through the canonical nightly runner.

---

## 2026-03-01 — Pilot Seek-First: Pointer Before Read

### Motivation
Original pipeline: READ → HIDDEN → WRITE → POINTER MOVE.
Problem: pilot pointer moves AFTER read, using info from the current step — but reads at a position chosen by the PREVIOUS step (blind to current input).
User insight: "Why read first, then move? Should seek first, then read."

### Change
Moved pilot pointer block from post-write to pre-read in `_process_chunk`.
New order (pilot only): **SEEK → READ → HIDDEN → WRITE**.
Sequential/learned pointer unchanged (still post-write).

### Results (N=1, replace write, period=8 synthetic)

| | Sequential | Pilot seek-first |
|---|---|---|
| Step 200 loss | 0.9846 | **0.2076** |
| Step 400 loss | 0.0834 | **0.0261** |
| Step 200 acc | 99.2% | **100.0%** |
| adj_cos (stream) | 0.78 | 0.86 |
| ptr_coverage | 78% | 54% |

~3× faster learning maintained. Pipeline order is now logically correct: seek the right slot, THEN read it.

### Status
- Code committed to `instnct.py` (pilot block in pre-read section, post-write has `pass`)
- **NOT validated on WikiText** — synthetic only

---

## 2026-03-01 — Session Summary: Ring Write Fix + Pilot Pointer

### Phase 1: Diagnosing the Blob

**Mini head gated write (Linear→2 gate) FAILED.**
- Probe: rank=1, adj_cos=0.985 after 500 steps — WORSE than baseline (rank dropped from 2→1)
- Root cause: hidden states converge (cos_AB=0.95), write_head gets identical inputs → identical erase/gate decisions
- **Key learning: the write MECHANISM wasn't the problem — the write CONTENT was**
- The hidden state is dominated by prev_hidden (norm 27 vs input norm 10), so all writes are near-identical

### Phase 2: HDD Analogy → Replace Write

**User insight:** Ring should behave like a HDD, not a tape.
- HDD **REPLACES** sectors — current ring **ACCUMULATES** (scatter_add) → blob
- HDD **SEEKS** to address — current ring always ptr+=1
- HDD has a **FILE TABLE** (FAT) — current ring has no index

**Implementation:** `func_hdd_write_tns()` — weighted lerp replace instead of scatter_add.
- `slot_new = w * write_vec + (1 - w) * slot_old` (center slot full replace, edges partial)
- New param: `write_mode='replace'` in INSTNCT constructor

**Probe results (replace write, N=1, 200 streaming steps):**

| Config | adj_cos@150 | adj_cos@200 | ptr_cov | Gate |
|--------|-------------|-------------|---------|------|
| Baseline (scatter_add, N=2) | 0.994 | 0.994 | 100% | FAIL |
| Mini head gate (N=2) | 0.985 | 0.985 | 100% | FAIL |
| **Replace write (N=1)** | **0.585** | **0.780** | **78%** | **PASS** |
| Replace write (N=1) +500 train | 0.580 | 0.774 | 78% | PASS |
| Replace write (N=2) | 0.993 | 0.993 | 100% | FAIL |

**Key findings:**
- Replace write **eliminates accumulation blob** — adj_cos 0.99 → 0.78 (N=1)
- Stable before AND after training (0.78 → 0.77)
- Model still learns (100% acc on period=8 synthetic task)
- N=2 still blobs because experts overwrite each other's slots → needs write buffer (future)

### Phase 3: Pilot Pointer

**User's original vision (pilot pulse):**
- Ring slots = NEURONS, not memory slots
- Pointer = data pulse traveling through neurons
- Each neuron has a learned identity ("I'm about history", "I'm about math")
- Pulse compares itself to current neuron → similar = stay, different = jump
- Jumps create bonds/connections between neurons (synaptic formation)

**Implementation:** `pointer_mode='pilot'`
- `slot_identity`: Parameter(M, id_dim) — learned identity per neuron
- `ptr_query`: Linear(hidden_dim, id_dim) per expert — pulse query
- Cosine similarity → sigmoid → jump distance
- `tau`: learnable temperature with softplus constraint

**Bug found and fixed during adversarial review:**
- Original code had `ptr += jump` → pointer STALLS when jump≈0 (match)
- Stalling = same slot overwritten every step → temporal memory destroyed
- Fix: `ptr += 1 + jump` → always advance at least 1, plus learned jump
- Three agents (Claude, other agent, GPT) independently confirmed this fix

**Probe results (replace + pilot +1 fix, N=1):**

| Config | adj_cos@200 | ptr_cov@200 | loss@200 | loss@400 |
|--------|-------------|-------------|----------|----------|
| sequential + replace | 0.774 | 78.1% | 0.985 | 0.083 |
| **pilot +1 + replace** | **0.837** | **51.6%** | **0.211** | **0.027** |

**Key findings:**
- Pilot learns **~3× faster** (loss 0.027 vs 0.083 at step 400)
- Lower ptr_coverage (51% vs 78%) — but this is SELECTIVE, not wasteful
- adj_cos slightly worse (0.84 vs 0.77) but still PASSES gate (<0.95)
- Both configs pass all critical gates after training

### What IS proven vs what is NOT

**PROVEN:**
- Replace write eliminates accumulation blob (structural fix)
- Pilot +1 pointer enables faster learning on synthetic periodic task
- Both fixes are compatible and stable together
- Gradient paths alive, model trains normally

**NOT PROVEN (do not over-interpret):**
- "Pilot is a better model" — only tested on period=8 synthetic, not WikiText
- "Sequential is obsolete" — pilot uses only 51% of ring, may lose long-range memory
- "Slot identity self-organizes" — 500 steps too few to confirm, need 3000+
- "3× faster loss = 3× better model" — early lead ≠ final accuracy
- N=2 write problem solved — needs write buffer (experts overwrite each other)

### Architecture Ideas (designed, NOT implemented)

**Hierarchical action head (policy C):**
- Level 1: sigmoid → "act or skip?"
- Level 2: softmax → "write_full / write_delta / erase"
- Level 3: sigmoid → "how strong?"
- Cost: ~5K params, ~1% compute overhead

**Write buffer (multi-expert, N>1):**
- Each expert writes to individual cache, then MERGE + FLUSH to ring
- Merge options: strongest wins, average, or partition
- Partition risks no cross-expert talk

**LCX-style hash index:**
- Existing LCX from VRAXION_WORLD uses cosine similarity + top-K + LRU eviction
- Could replace or supplement ring addressing
- Key = Linear(hidden) → normalized vector, read = cosine lookup over all slots
- Ring topology optional with hash — but angle-based addressing preserves ring structure

**Angle-based addressing (ring + content-based hybrid):**
- hidden → Linear → angle (0°-360°) → ring position
- Similar content clusters at similar angles (self-organizing)
- Preserves ring topology while enabling content-based access
- vshape local read becomes meaningful (neighbors = similar topics)

### Files Modified
- `model/instnct.py`: `func_hdd_write_tns()`, `write_mode` param, `write_head` ModuleList (gated write), pilot +1 fix
- `tests/probe_dataflow.py`: `--write-mode`, `--kernel-mode`, `--pointer-mode`, `--n-experts` CLI flags
- `dev_notes/experiment_log.md`: this entry

### Status
- **Replace write: STABLE FOUNDATION** — use as default for all future experiments
- **Pilot pointer: PROMISING, Phase 2** — needs WikiText validation + longer runs
- **Next: decide ring architecture direction** — pure ring + pilot, LCX-style flat, or hybrid angle-based

---

## 2026-03-01 — Run 33 Post-Mortem: Deep Signal Probe (Claude Opus, manual)

**Context:** Run 33 (N=1, plateau LR, 50K steps) peaked at ~57% masked_acc (best_loss=1.395 at step 11K), then flatlined. LR auto-reduced 6 times to floor (1e-5). The model stopped learning. Why?

**Method:** Manual signal probe (`tests/probe_signals.py`) — loaded best checkpoint (step 11K), ran 50 forward passes with state carry on random byte input, then manually computed all intermediate signal magnitudes without hooks.

### Raw Probe Results

```text
Checkpoint: step=11000, best_loss=1.395481
S=0.2929, N=1, M=1024, hidden=2048, slot=64, R=1

HIDDEN UPDATE COMPOSITION
  hidden_new = c19(input + S*ring_signal + phase + hidden_old)
  input:          39.86   (10.0%)
  S*ring_signal: 252.49   (63.0%)
  phase:           3.97   ( 1.0%)
  hidden_old:    104.28   (26.0%)
  total pre-c19: 400.59

  Ring contribution:  63.0% of hidden update
  Ring vs input:      633.3%
  Ring vs hidden:     242.2%

RING STATE ANALYSIS
  slot norm:  mean=1.7776  std=0.0965
              min=1.3792  max=2.1730
  active slots: 100.0%
  adjacent cos_sim:  mean=0.9806  std=0.0084
  random cos_sim:    mean=0.9455  std=0.0187
  SVD effective rank: 90%=1  95%=2  99%=24  (max=64)
  top 10 SV:    13.2 2.1 1.8 1.5 1.2 1.2 1.0 1.0 0.9 0.9

OUTPUT PATH ANALYSIS
  out.0 (2048->64): condition=14.5x
    top 5 SV: 0.118 0.090 0.067 0.057 0.048
  out.2 (64->256):  condition=17.6x
    top 5 SV: 0.193 0.142 0.119 0.094 0.085
  round-trip cos_sim (2048->64->2048): 0.1367
  round-trip norm ratio: 0.0156

C19 SATURATION ANALYSIS
  tanh argument: mean=-3.0430  std=1.3430
  tanh argument: |arg| > 2.0: 78.5%
  tanh argument: |arg| > 3.0: 55.1%
  tanh output:   mean=-0.9157  std=0.1957
  saturation (|tanh| > 0.95): 69.7%
  rho_hidden: mean=1.5529  std=0.5729
  C_hidden:   mean=-3.1924  std=0.7851
```

### Diagnosis: 4 Smoking Guns

**1. Ring Collapse (CRITICAL)**
SVD effective rank = 1 (90% variance in first singular vector). All 1024 slots converged to nearly the same vector (adjacent cos_sim=0.98, random cos_sim=0.95). The ring is a uniform blob, not structured memory. Cause: scatter_add random walk — cumulative writes without erasure converge to a mean. This is the primary failure: the "memory" has nothing different to remember.

**2. C19 Saturation (CRITICAL — NEW FINDING)**
69.7% of hidden neurons are saturated (|tanh| > 0.95). C_hidden learned a mean of -3.19, which pushes the tanh argument deep into the negative saturation zone. The optimizer found a local minimum where most neurons are dead (zero gradient). This was NOT flagged by any of the 3 external research reports (GPT, Qwen, Claude Opus) because they lacked probe data.

**3. Ring Signal Dominance with Degenerate Content (SEVERE)**
S*ring_signal = 63% of hidden update magnitude, but the ring content is rank-1 degenerate. The model is 63% driven by a signal that's effectively constant noise. Fresh input (the actual information) contributes only 10%. The model listens to its own static instead of the data.

**4. Output Bottleneck (MODERATE)**
2048→64 compression in the output head loses 86% of information (round-trip cos_sim=0.14). The 64-dim bottleneck is too narrow to preserve the structure of a 2048-dim hidden state. However, this is less critical because the output projection maps to 256 classes, not back to hidden space.

### Correlation with External Research Reports

| Finding | GPT Report | Qwen Report | Opus Report | Probe Data |
|---------|-----------|-------------|-------------|-----------|
| Write instability | "stabilize writes first" | "write-path before pointer" | "write dynamics focus" | **CONFIRMED**: ring SVD rank=1 |
| Dead gradients | not flagged | not flagged | "write gradients dead" (wrong mechanism) | **CONFIRMED**: C19 saturation 70% |
| Ring importance | implied | implied | implied | **QUANTIFIED**: 63% of update |
| Output bottleneck | not flagged | not flagged | not flagged | **NEW**: 86% info loss |

### Consolidated Action Priority (all sources + probe)

1. **Phase 0 — C19 saturation fix** (probe-driven, not in any report): clamp C_hidden, or revert to pure tanh (ablation showed tanh > c19)
2. **Phase A — Write stabilization** (all 3 reports agree): gated write / EMA / erase+add to prevent ring collapse
3. **Phase B — S rebalancing**: reduce S or let Phase A fix ring content naturally
4. **Phase C — Soft addressing / pointer fix**: only after ring has diverse content

### Files
- Probe script: `tests/probe_signals.py`
- Best checkpoint: `ARCHIVE/33 - N1 plateau LR 25K peak57pct/ckpt_best_step_011000.pt`
- Training log: `ARCHIVE/33 - N1 plateau LR 25K peak57pct/train_log.csv`

---

## 2026-03-01 — Fast Bench: Hourglass I/O Split + N=1 vs N=2

**Hypothesis:** GPT's gradient analysis of Run 35 showed expert collapse — one expert became write-only, the other read-only. What if this is intentional? Force a strict "hourglass" architecture: Expert 0 = writer (no ring read), Expert 1 = reader (no ring write), output from reader only. Also: is N=1 better than N=2?

**Setup:** New fast bench (`tests/bench_fast_memory.py`) — repeating pattern task, in-RAM data, ~97% supervised positions. Period=128, seq=32, hidden=256, M=256, 500 steps, vshape kernel, CUDA. Includes fresh-state eval and S=0 ring dependency probe.

**Results:**

```text
Config              | Peak  | Fresh | S=0   | Ring Dep | Params  | Speed
--------------------|-------|-------|-------|----------|---------|-------
N=1 baseline        | 39.2% |  0.4% |  0.4% | +38.8pp |  70,658 | 0.31s
N=2 baseline        | 33.6% |  0.4% |  0.4% | +33.2pp | 103,746 | 0.46s
N=2 strict split    | 17.9% |  0.4% |  0.5% | +17.4pp | 103,746 | 0.44s
```

Earlier session (hidden=512, period=64, seq=32):
```text
N=1: 55.3%   N=2: 49.8%   Transformer: 99.1%
```

**Key findings:**

1. **N=1 > N=2 on every test** — the second expert hurts. Likely cause: scatter_add interference (experts overwrite each other's ring entries) + hidden averaging dilutes signal.
2. **Hourglass strict split is catastrophic** — 17.9% vs 39.2% (N=1). Writer-only expert writes blind (no ring feedback), can't adapt to ring contents. The "natural collapse" in Run 35 was a symptom of broken topK dynamics, not a useful architecture.
3. **Ring dependency is real** — S=0 probe drops to 0.4% (random) for all configs. The ring is genuinely storing and recalling patterns across sequences.
4. **Fresh-state eval = 0.4% everywhere** — confirms cross-sequence memory is required (period=128 > seq=32). Without carry state, the model can't see the full pattern.

**Decision:** Hourglass split dropped (-21pp vs N=1). N=1 is the new default for fast iteration. Full WikiText A/B (N=1 vs N=2) recommended to confirm at scale.

**Files:** `tests/bench_fast_memory.py` (updated with `--io-split` flag, fresh-state eval, S=0 probe), `model/instnct.py` (io_split_mode implementation).

---

## 2026-03-01 — Run 35: TopK Ring Read (content-based global search)

**Hypothesis:** INSTNCT's ring stores useful information but can't find it. The vshape kernel reads 3 adjacent slots around the pointer — maybe content-based search over all 1024 slots lets the model find relevant memories like a transformer's attention.

**Setup:** kernel_mode=topk, K=8, query_proj per expert (Linear(2048, 64)). Write stays pointer-based (vshape scatter_add). Read: score all M slots via q·slot dot product, softmax top 8, weighted sum. 973K params (+37% vs vshape 711K).

**Result:** WORSE than vshape by -4.6% (42.3% vs 46.9% final).

```text
                    Vshape INSTNCT   TopK INSTNCT    Transformer
Params              711,210          973,506         710,208
Peak accuracy       47.8% @s2960     43.6% @s2670    52.9% @s2530
Final accuracy      46.9% @s3000     42.3% @s3000    50.7% @s3000
Last 200 avg        46.6%            42.2%           51.5%
BPB (final)         2.69             2.93            2.39
```

**Diagnosis:** The problem is NOT how the ring is read — it's what's IN the ring. Scatter_add writes create lossy superpositions, not clean entries. Global search over 1024 noisy slots finds distractors. Temporal proximity (vshape) is actually the best relevance signal: recent = useful.

**Decision:** Revert to vshape. TopK code stays in instnct.py for reference. The ring write mechanism needs fixing before read improvements can help.

**Files:** `training_output/run35_topk_3000steps/`

---

## 2026-02-28 — Run 31: Cosine Gate, NO LayerNorm (A/B vs Run 30)

**Hypothesis:** Run 30's 4% gap vs Run 24 was caused by LayerNorm stripping magnitude information from ring_signal. Test: keep cosine gate (scale-invariant alpha), remove LayerNorm (let raw ring_signal magnitude through).

**Config:** Identical to Run 30 except: `blended_ring = alpha * ring_signal` (no LayerNorm).

**Results:**

| Metric | Run 30 (w/ LN) | Run 31 (no LN) | Run 24 (champion) |
|--------|----------------|----------------|-------------------|
| Peak masked_acc | 37.6% (step 880) | **39.1%** (step 860) | **41.4%** |
| Final masked_acc | 37.2% | **37.9%** | 41.4% |
| Final masked_loss | 2.271 | **2.239** | — |
| blend_norm | ~22 (clamped by LN) | ~45-55 (raw) | — |
| ring_norm | 7,695 | 9,254 | — |
| alpha_0 mean | 0.51 | 0.45 | — |
| alpha_1 mean | 0.48 | 0.53 | — |
| alpha range | [0.42, 0.68] | [0.29, 0.70] | — |
| hidden_norm | ~50-52 | ~44 | — |
| ring_signal_raw | 68-80 | 95-105 | — |

**Key findings:**

1. **LayerNorm removal: +1.5% improvement** (37.6% → 39.1% peak). Confirms LN was stripping useful magnitude info.

2. **Wider alpha range** [0.29, 0.70] vs [0.42, 0.68] — the model makes bolder input-vs-ring decisions without LN constraining the signal.

3. **Expert specialization reversed:** Expert 0 now input-preferent (α≈0.45), Expert 1 ring-preferent (α≈0.53). The model finds different specialization without LN's homogenizing effect.

4. **Ring norm higher** (9,254 vs 7,695) — without LN the ring accumulates more, but the cosine gate handles it (alpha stays bounded).

5. **Still 2.3% behind Run 24.** The cosine gate itself adds overhead vs simple S=0.3 fixed scalar. At this model size, the adaptive gate doesn't earn back its cost.

**Verdict:** LayerNorm was indeed the primary culprit (-1.5%). Cosine gate alone is better than cosine gate + LN, but still trails the simple S=0.3 baseline by 2.3%. The cosine gate is a valid mechanism but not yet justified at this scale.

**Decision:** For future experiments (bulletin board, etc.), revert to **S=0.3 fixed** as baseline — it's the proven champion. The cosine gate can be revisited when the model is larger or the task is harder.

**Archived:** `ARCHIVE/run31_cosine_gate_no_layernorm/` (train_log.csv, ckpt_latest.pt, config)

---

## 2026-02-28 — Run 30: Cosine Gate + LayerNorm (GPT Adversarial Fix)

**Hypothesis:** Run 29's gate saturation (alpha crashed to 0.20/0.87) was caused by scale sensitivity — raw dot-product gate saw ring_signal norm 84-124 vs input_norm ~18. Two fixes from GPT adversarial analysis:
1. **Cosine gate**: `α = sigmoid(τ · cosine(input, ring_signal))` — always [-1,+1] regardless of norms
2. **LayerNorm on ring_signal**: normalize magnitude before blending into hidden state

**Config:**

| Param | Value |
|-------|-------|
| M / hidden_dim / slot_dim | 1024 / 2048 / 64 |
| N / R | 2 / 1 |
| kernel_mode | vshape |
| Gate | cosine similarity + learnable τ (init 4.0) |
| Ring signal | LayerNorm(hidden_dim) before blend |
| Steps | 1000 |
| LR | 1e-3 (warmup 100, cosine decay) |
| Batch / seq_len | 128 / 64 |
| sequential | true |
| embed_encoding | bitlift |
| output_encoding | lowrank_c19 |
| Params | 711,234 |
| Device | CUDA |

**Code changes (model/instnct.py):**
```python
# Init:
self.gate_tau = nn.Parameter(torch.tensor(4.0))       # learnable temperature
self.ring_signal_norm = nn.LayerNorm(hidden_dim)       # normalizes ring signal

# Forward (gate computation):
cos_sim = F.cosine_similarity(input_vec_tns, ring_signal, dim=-1).unsqueeze(-1)
alpha = torch.sigmoid(self.gate_tau * cos_sim)         # bounded [0, 1]
ring_signal_n = self.ring_signal_norm(ring_signal)     # normalize magnitude
blended_ring = alpha * ring_signal_n
```

**Results:**

| Metric | Step 100 | Step 500 | Step 880 (peak) | Step 1000 |
|--------|----------|----------|-----------------|-----------|
| masked_acc | 18.9% | 33.4% | **37.6%** | 37.2% |
| masked_loss | 3.280 | 2.403 | 2.234 | 2.271 |
| ring_norm | 3,565 | 7,709 | 7,701 | 7,695 |
| alpha_0 (mean) | 0.520 | 0.500 | 0.505 | 0.514 |
| alpha_1 (mean) | 0.443 | 0.452 | 0.476 | 0.478 |
| blended_norm_0 | 23.6 | 22.9 | 23.3 | 23.7 |
| blended_norm_1 | 20.1 | 20.7 | 21.9 | 22.0 |
| ring_signal_raw_0 | 20.1 | 80.2 | 78.5 | 79.7 |
| ring_signal_raw_1 | 33.1 | 64.2 | 69.4 | 68.1 |
| input_norm | 18.0 | 19.7 | 19.7 | 19.5 |

**vs Run 24 Champion (vshape, S=0.3 fixed):** 37.2% vs 41.4% = **-4.2% gap**

**Diagnostic Analysis:**

1. **Cosine gate: SUCCESS.** Alpha stays centered at ~0.50 throughout training (range [0.42, 0.68]). No saturation — both experts keep balanced input/ring mixing. This completely solves Run 29's gate collapse.

2. **LayerNorm: SUCCESS (stability) but COST (performance).** Blended norm controlled at ~22-23 regardless of ring_signal growing from 6 to 80+. The normalization prevents magnitude explosion. However, the 4% gap vs Run 24 suggests LayerNorm is too aggressive — it strips useful magnitude information from the ring signal.

3. **Ring norm: grows 628→7700 then stabilizes.** Random walk effect confirmed — even with balanced +/- writes (diag_write_signs.py showed ~45-55% negative), scatter_add accumulates. LayerNorm makes this irrelevant to model behavior, but the underlying growth remains.

4. **Expert specialization:**
   - Expert 0: α≈0.51 (slight ring preference), ring_signal_raw oscillates 55-93
   - Expert 1: α≈0.47 (slight input preference), ring_signal_raw oscillates 53-80
   - The two experts are nearly symmetric — cosine gate prevents the divergent specialization seen in Run 29

5. **Hidden state stability:** hidden_norm ≈ 50-52 for both experts, completely stable throughout. No growth, no collapse.

6. **Pointer positions:** Fixed at 63 and 575 (sequential mode, staggered init). φ-jump spacing = int(1024 × 0.618) = 632 → 63+632=695≠575, so the staggering is from init, not phi.

**Why 4% behind Run 24?**

Run 24 used simple `hidden += S * read_proj(read_vec)` with S=0.3. This is a fixed, calibrated blend. The cosine gate + LayerNorm combination:
- Normalizes away magnitude → loses "how confident is this ring read?"
- Cosine similarity only measures direction → two ring signals with same direction but 10× different magnitudes get same alpha
- The gate adds parameters (gate_tau) but the softmax-like cosine similarity may not be the right inductive bias for this task

**Verdict:** Cosine gate solves stability, LayerNorm controls magnitude, but the combination is **too conservative**. The model plays it safe (alpha≈0.50) instead of making bold ring/input choices. Run 24's simple S=0.3 was actually better because it let the ring signal's natural magnitude through.

**Key insight:** The problem was never "gate saturation" per se — it was that the raw dot-product gate SAW the ring norm grow and reacted. The fix should preserve magnitude information while controlling it, not strip it entirely. A softer normalization (e.g., RMSNorm, or just dividing by running mean) might work better than LayerNorm.

**Archived:** `ARCHIVE/run30_cosine_gate_layernorm/` (train_log.csv, ckpt_latest.pt, config)

---

## 2026-02-27 — DETOUR: Encoding Experiments (Sideways, Not Upward)

**What happened:** 2 days spent testing hadamard/learned/sincos I/O encoding. This was NOT on the upward build path — it was a sideways exploration that didn't advance the architecture.

**Lesson:** Stick to the upward build plan. Don't chase tangential ideas before the core stack is built.

**Findings (kept for reference, not actionable for upward build):**

1. **Ring = within-sequence scratch pad.** Gets zeroed every forward pass. Cross-sequence memory is not a v4 feature and wasn't planned to be — the ring was always designed as working memory within a sequence. Long-term memory (LCX) belongs on a HIGHER floor of the build.

2. **Learned I/O is required.** Hadamard encoding fails completely (3.6% on echo vs 63% learned). Fixed encodings don't work for this architecture. This is settled — use learned, move on.

3. **R_param specialization.** In the successful echo run (learned I/O), experts specialized: Expert 0 R_eff=467 (wide), Expert 1 R_eff=85 (narrow). Learnable R IS valuable at scale, contradicting the tiny-model 3x3 benchmark.

**Archived runs:** 03 (hadamard real code, 26%), 04 (learned real code, 29.7% @ 60 steps), 05 (hadamard echo, 3.6%). See archive.csv for full data.

---

*(Raw data for the encoding detour runs is in archive.csv and ARCHIVE/03, 04, 05. See FAILURE_REPORT_2026-02-27.md for full post-mortem.)*

---

## 2026-02-26 — 3x3 Kernel x R Strategy Exhaustive Benchmark

**Hypothesis:** Systematically determine the optimal attention kernel shape and radius strategy by testing ALL 9 combinations of {uniform, vshape, gaussian} x {fixed R~1, fixed R~M/2, learnable R per-expert}.

**Motivation:** After individually proving tanh (activation), phase ON (embeddings), depth OFF (harmful), and V-shape + R=1 as the default, we needed to exhaust all alternatives to make a definitive, mathematically complete decision on the attention mechanism.

**Config (all conditions):**

| Param | Value |
|-------|-------|
| M / D / N | 64 / 64 / 2 |
| Steps | 3000 |
| LR | 1e-3 |
| Batch | 4 (train), 8 (eval) |
| Seed | 42 (train), 9999 (eval) |
| Phase | ON (sin/cos) |
| Activation | tanh |
| Depth | OFF |
| Device | cpu |

**Kernel definitions:**
- **Uniform**: soft sigmoid cutoff (k=10) for differentiable boundary
- **V-shape**: linear triangle decay `max(0, 1 - |d|/R_eff)`, zero beyond R_eff
- **Gaussian**: bell curve `exp(-d^2/(2*sigma^2))`, sigma = R_eff/2.5, clamp(min=0.3)

**R strategies:**
- Fixed R~1: R_param frozen at -3.43 (sigmoid -> R_eff ~ 1.0)
- Fixed R~M/2: R_param frozen at +7.0 (sigmoid -> R_eff ~ 31.97)
- Learnable: R_param init -2.0 (sigmoid -> R_eff ~ 3.8), trained via backprop

**Results — Full 3x3 Matrix:**

XOR (seq=36):

|            | Fixed R~1 | Fixed R~M/2 | Learnable R |
|------------|-----------|-------------|-------------|
| uniform    | 85.81%    | 86.33%      | 86.20% R=[2.6, 4.4] |
| **vshape** | **86.59%**| 85.42%      | 86.33% R=[2.0, 4.3] |
| gaussian   | 86.20%    | 85.55%      | 86.20% R=[1.9, 4.5] |

Delayed XOR (seq=44):

|            | Fixed R~1    | Fixed R~M/2 | Learnable R |
|------------|--------------|-------------|-------------|
| uniform    | 56.64%       | 55.47%      | 56.64% R=[3.5, 3.8] |
| **vshape** | **57.81%**   | 55.86%      | 57.42% R=[3.5, 4.0] |
| **gaussian** | **57.81%** | 56.64%      | 57.42% R=[3.5, 4.0] |

Chained XOR (seq=48):

|            | Fixed R~1    | Fixed R~M/2 | Learnable R |
|------------|--------------|-------------|-------------|
| uniform    | 88.37%       | 88.19%      | 88.11% R=[2.6, 4.5] |
| vshape     | 88.45%       | 88.54%      | **88.80%** R=[2.0, 6.6] |
| **gaussian** | **88.72%** | 88.45%      | 88.63% R=[2.0, 6.2] |

Combined (geometric mean across 3 tasks):

|            | Fixed R~1    | Fixed R~M/2 | Learnable R |
|------------|--------------|-------------|-------------|
| uniform    | 75.45%       | 75.03%      | 75.49%      |
| **vshape** | **76.22%**   | 75.03%      | 76.07%      |
| gaussian   | 76.18%       | 75.40%      | 75.98%      |

**WINNER: V-shape + Fixed R~1 = 76.22% geometric mean**

**Learnable R — what the network chose:**

| Kernel | XOR | Delayed | Chained |
|--------|-----|---------|---------|
| uniform | [2.6, 4.4] | [3.5, 3.8] | [2.6, 4.5] |
| vshape | [2.0, 4.3] | [3.5, 4.0] | [2.0, 6.6] |
| gaussian | [1.9, 4.5] | [3.5, 4.0] | [2.0, 6.2] |

Consistent pattern: expert 0 prefers narrow (R~2), expert 1 prefers wider (R~4-6).
On chained XOR, expert 1 goes even wider (R~6-7) for multi-hop reasoning.

**Key findings:**

1. **Kernel shape barely matters.** V-shape and gaussian are within 0.04% geometric mean. Uniform is ~0.8% behind.

2. **Wide fixed R hurts consistently.** R~M/2 is the worst R strategy across ALL kernels on XOR and Delayed XOR. Only on Chained XOR is it competitive.

3. **Learnable R doesn't help.** Despite learning sensible radii (2-6), the learned R never beats fixed R~1 on the combined metric. The overhead of gradient computation through weights adds ~25% training time.

4. **The network naturally prefers R~2-4.** When free to choose, both experts settle on small radii (2-4), confirming that narrow windows are optimal. The learnable mechanism simply rediscovers "R should be small."

5. **V-shape + R=1 is the Pareto optimum.** Best accuracy (76.22% gmean), fastest training (~280s/condition), fewest parameters (R_param frozen = 2 fewer trainable params).

**Verdict:** KEEP V-shape + Fixed R=1. Remove learnable R complexity from the model — it adds training cost without benefit.

**Total benchmark time:** 140 min (27 conditions on CPU).

**Follow-up:** Simplify model by removing R_param learnable machinery. The kernel_mode parameter can stay for future experiments, but default to 'vshape'. Fixed R=1 is the final architecture choice for the attention window.

---

## 2026-02-26 — Phase Ablation: ON vs OFF

**Hypothesis:** Sin/cos phase embeddings (position encoding from pointer position on ring) should help the model by telling hidden state WHERE each expert currently is.

**Config:** Same as 3x3 benchmark above (M=64, D=64, N=2, tanh, 3000 steps).

**Method:** Phase OFF = freeze phase_cos and phase_sin to zero, remove from optimizer.

**Results:**

| Task | Phase OFF | Phase ON | Delta |
|------|-----------|----------|-------|
| XOR | 85.42% | 85.94% | +0.52% |
| Delayed XOR | 55.08% | 56.64% | +1.56% |
| Chained XOR | 87.67% | 88.54% | +0.87% |

**Verdict:** PASS — Phase consistently improves all 3 tasks. Largest gain on Delayed XOR (+1.56%) where position context helps most.

---

## 2026-02-26 — Depth Signal Ablation (helix_z)

**Hypothesis:** Encoding "total distance traveled" (helix_z) as a depth signal should give experts a sense of temporal progress.

**Config:** Same base config. Depth init 0.1, no gate.

**Result:** ALL three activations (tanh, c19, silu) produced IDENTICAL results per task (83.20% XOR, 56.64% Delayed, 85.94% Chained). The depth signal completely dominated the model, making the activation function irrelevant.

**Verdict:** FAIL — Depth signal is harmful. Uniformizes model behavior. **Removed from code.**

---

## 2026-02-26 — Activation Function Benchmark (tanh vs c19 vs silu)

**Hypothesis:** Compare top 3 activation functions on the same 3-task battery.

**Results (with phase ON, no depth):**

| Task | tanh | c19 | silu |
|------|------|-----|------|
| XOR | **85.94%** | 84.38% | 82.81% |
| Delayed XOR | 56.64% | **57.81%** | 55.08% |
| Chained XOR | **88.54%** | 75.78% | 87.50% |

**Verdict:** tanh wins overall. c19 has a small edge on Delayed XOR but catastrophically underperforms on Chained XOR (-12.76%). tanh is the safest, most consistent choice.

---

## 2026-02-25 ~22:00 — C19 Activation: XOR Breakthrough

**Hypothesis:** Adding C19 activation to the hidden state recurrence should break the 50% XOR ceiling that the linear model couldn't pass. The nonlinearity allows the model to compute A^B for unseen inputs.

**Change:** Single line in `instnct.py` forward pass:
```python
# BEFORE: hidden_tns[i] += input_vec_tns; hidden_tns[i] += S * read_proj(read)
# AFTER:  hidden_tns[i] = c19_activation(input + S*read_proj(read) + hidden[i])
```

**Config (small — failed):**

| Param | Value |
|-------|-------|
| M / D / N / R | 32 / 16 / 2 / 1 |
| Steps | 3000 |
| LR | 1e-3 |
| Batch | 4 |
| Device | cpu |

**Result:** Best eval 58.07% — barely above random. D=16 too small for C19 periodic waves.

**Config (medium — success):**

| Param | Value |
|-------|-------|
| M / D / N / R | 64 / 64 / 2 / 1 |
| Steps | 3000 |
| LR | 1e-3 |
| Batch | 4 |
| Params | 9,416 |
| Device | cpu |

**Result:**

| Metric | Value |
|--------|-------|
| Final loss | 0.080 |
| Final train acc | 89.84% |
| Final eval acc | 86.07% |
| Best eval acc | 86.07% (still climbing) |
| Duration | ~4 min |

**Verdict:** PASS — C19 **breaks the 50% XOR ceiling**. Linear model was stuck at ~50% (random guess). C19 model reaches 86% on held-out XOR data and still improving. Mathematical proof that the nonlinearity is working.

**Key insight:** D=16 is too small for C19 — the periodic parabolic wave needs dimensional room. D=64 gives enough space.

**Follow-up:** Add phase embeddings (step 2 of the plan). Then test on real training data.

---

## 2026-02-25 ~16:00 — XOR Generalization Ceiling (linear baseline)

**Hypothesis:** A purely linear model cannot generalize XOR (a nonlinear function) to unseen inputs. Expected: eval accuracy stuck at ~50% (random guess).

**Config:**

| Param | Value |
|-------|-------|
| M (ring slots) | 32 |
| D (slot dims) | 16 |
| N (experts) | 2 |
| R (attn radius) | 1 |
| S (context scale) | 0.05 |
| Steps | 3000 |
| LR | 1e-3 |
| Batch | 4 |
| Seq len | 36 |
| Mode | binary |
| Device | cpu |

**Result:**

| Metric | Value |
|--------|-------|
| Final loss | ~0.25 |
| Final accuracy (train) | varies per random batch |
| Best eval accuracy | 52.86% |
| Converged at step | N/A (noise floor) |
| Duration | ~2 min |

**Verdict:** PASS — Linear model stuck at ~50% on held-out XOR data. Mathematical proof that nonlinearity is required for nonlinear tasks. Model cannot learn A^B for unseen A,B — only random guess.

**Note:** First attempt used fixed batch — model memorized it to 98.96% (824 params > 192 supervised bits). Fixed by using random batches every step + separate held-out eval set.

**Follow-up:** Add C19 activation, re-run XOR test to prove it breaks the 50% ceiling.

---

## 2026-02-25 ~14:00 — Overfit Diagnostic (Echo Task)

**Hypothesis:** Linear model + MSE loss on a linear task (echo = time-delayed identity) should converge to near-100% if given enough steps. Tests whether the architecture can memorize 1 fixed batch.

**Config:**

| Param | Value |
|-------|-------|
| M (ring slots) | 32 |
| D (slot dims) | 16 |
| N (experts) | 2 |
| R (attn radius) | 1 |
| S (context scale) | 0.05 |
| Steps | 10000 |
| LR | 1e-3 |
| Batch | 2 |
| Seq len | 32 |
| Mode | binary |
| Device | cpu |

**Result:**

| Metric | Value |
|--------|-------|
| Final loss | 0.00385 |
| Final accuracy | 97.66% |
| Best eval accuracy | N/A (overfit test — no eval set) |
| Converged at step | ~8000 (still improving slowly) |
| Duration | ~7 min |

**Verdict:** PASS — Architecture CAN memorize echo data. 97.7% accuracy proves the ring-buffer pointer mechanism works for linear pattern recall. Bottleneck is generalization, not capacity.

**Note:** Earlier CUDA run with full-size model (M=256, D=256, N=6) had loss spikes at ~0.85s/step. Tiny model on CPU was stable and 20x faster for diagnostics.

**Follow-up:** XOR test to prove linear ceiling on nonlinear tasks.

---

# Architecture Evolution — How We Got Here

This section documents the lineage from the original concept to v4.

---

## v0.x — The Linear Hallway (original concept)

The starting idea: a **ruler** (linear tape) of neurons. Data enters at neuron 1 and each neuron has an intrinsic **jump probability** and a **destination**. Example: video input arrives at position 1, gets tossed toward the middle (neurons 29→30→31→32), gets processed there, then neuron 32 jumps the result to the output neuron at position 100.

```
Input                Processing zone               Output
  │                  ┌──────────┐                    │
  ▼                  │          │                    ▼
 [1]──jump──→[29][30][31][32]──jump──→            [100]
       p=0.8              p=0.7
```

Key properties:
- Linear topology (finite tape, hard boundaries at 0 and end)
- Each neuron has its own jump probability and target
- Data flows through hops, not layers
- No shared memory — each neuron holds its own state

---

## v1.0 — Circular Ring + Continuous Pointer

The hallway became **circular** — slot M wraps to slot 0, no edge effects. The pointer became a **continuous float** in [0, M) instead of discrete integer positions. Read/write uses a **Gaussian kernel** (soft window) around the pointer.

Key changes from v0:
- Ring buffer `[B, L, S]` — L slots, S dims each
- Float pointer with Gaussian soft-read/write (tau=8.0, kernel width ±2)
- Per-position learned parameters: `theta_ptr` (target), `theta_gate` (jump bias)
- Pointer movement: `p = sigmoid(jump_score(hidden) + theta_gate)`

---

## v1.2 — Möbius Strip / Helix Extension (phase depth)

Once the ring was circular, the next question: **what if we break it into a helix?** A Möbius strip — twist the ring so that after one full revolution, the phase flips. This gives each slot a **phase depth** of +1 or -1, effectively doubling the ring's capacity without doubling the slot count.

```
                    phase +1
                  ╭──────────╮
                 ╱  slot 0    ╲
               ╱   slot 1      ╲        ← top surface
             ╱    ...            ╲
           ╱     slot M/2          ╲
          ╰─── twist ──────────────╯
           ╲     slot M/2          ╱
             ╲    ...            ╱        ← bottom surface (phase flipped)
               ╲   slot 1     ╱
                 ╲  slot 0   ╱
                  ╰──────────╯
                    phase -1
```

Implementation in `platinum/hallway.py`:
```python
self.mobius = bool(MOBIUS_ENABLED)
self.mobius_scale = 2 if self.mobius else 1
self.ring_range = int(self.ring_len * self.mobius_scale)  # 2x effective range

# Phase embedding: pointer position → sin/cos phase signal
theta = (ptr_float / ring_len) * pi
phase_cos = torch.cos(theta)
phase_sin = torch.sin(theta)
cur = cur + phase_cos * phase_embed[0] + phase_sin * phase_embed[1]
```

The phase embedding injected a position-dependent signal into the hidden state — the same slot "felt different" depending on which phase (+1 or -1) the pointer arrived from.

**Probe results (2026-02-10):**

| Implementation | Accuracy | Jump Gate | Params |
|----------------|----------|-----------|--------|
| No Möbius (baseline) | 75.2% | 0% | 516 |
| **2-Layer "Möbius" (simplified)** | **78.4%** | 7% | 644 |
| TRUE Möbius (fixed +1) | 72.4% | 0-1% | 644 |
| TRUE Möbius (random ±1) | 74.6% | 0% | 644 |

**Verdict:** The mathematically "correct" Möbius topology actually performed WORSE than baseline. Problems:
- Gradient disruption from holonomy sign flips
- Limited wrap exposure (1 flip per 64-step sequence)
- Jump gate suppression (pointer stopped jumping to avoid phase discontinuity)

**Decision:** Abandoned true Möbius topology. Kept the **simplified phase embeddings** (sin/cos position encoding) which gave +3.2% without the instability. This evolved into the golden ratio phase embeddings used later in SwarmByteRingModel.

---

## v1.3 — Spiral Topology (ring + golden chords)

Parallel research explored a different way to extend the ring: instead of twisting it (Möbius), add **long-range chord connections** using the golden ratio. Each node connects to `±1` (ring neighbors) and `±k` (golden jump), where `k = round(N/phi)` with `gcd(N,k)=1` enforced.

```
         ╭────── golden chord (k=158) ──────╮
         │                                   │
    ─[slot 0]─[1]─[2]─ ... ─[slot 158]─ ... ─[255]─
         │                                   │
         ╰─── ring neighbor (±1) ───╯
```

Offsets per node: `(+1, -1, +k, -k)` — 4 connections total.

This kept the ring topology intact but added skip connections for fast information propagation. The spiral heartbeat mixer used a nonlinear update:
```python
ctx = x_bn + a_local*(x_p1 + x_m1) + a_jump*(x_pk + x_mk)
state = (1-beta)*state + beta*tanh(ctx + base_bn)
```

Key finding: **nonlinearity in the update step was required** — without `tanh`, the spiral collapsed to a linear average (the "linear trap"). This was an early signal that pure linear operations wouldn't be enough.

---

## v1.5 — Soft Pointer Dynamics

Pointer movement became a **hierarchical soft blend** with inertia and deadzone:

```
ptr_next = CIRC_LERP(
    stay,                    # don't move
    walk (ptr + 1),          # short step
    jump (theta_ptr target), # long leap
)
→ inertia smoothing (60% old, 40% new)
→ deadzone filter (ignore moves < 0.02)
```

This prevented pointer jitter and gradient spikes. The jump destination was still a **learned parameter** per neuron.

---

## v2.0 — Experts (Beings) + LocationExpertRouter

Multiple independent agents ("beings") now share the ring:
- Each being has its own pointer, hidden state, and read projection
- **LocationExpertRouter**: routes input to different expert heads based on pointer address
- Pointer position determines specialization — beings at different ring positions see different context

---

## v2.5 — Prismion Swarm + Fibonacci Budget

The single ring cell became a **swarm of mini-hallways** (Prismions):
- Each Prismion is a full ring cell at smaller scale
- **Fibonacci halving**: satellite Prismions get 0.5^i of the primary budget
- Shared weights + per-Prismion ID embeddings
- Loop topology: A → (B → C → A) × iters → D

---

## v2.7 — Auxiliary Rings (Sensory, Vault, Think/LCX)

Three additional ring layers added:

| Ring | Purpose |
|------|---------|
| **Sensory** | Pre-processes raw input, feeds context into main ring |
| **Vault** | Long-term memory with adaptive gate (learns what to retain) |
| **Think (LCX)** | Dual fast/slow scratchpad for internal reasoning |

The Think Ring used phi-derived decay rates:
- Slow channel: `alpha = phi^-4 ≈ 0.146`
- Fast channel: `alpha = phi^-3 ≈ 0.236`
- **BrainstemMixer**: entropy-driven blending between fast and slow

C19 activation was present at **all projection points** throughout the system.

---

## v2.10.574 — Full System (pre-v4 strip-down)

Peak complexity. The full system included:
- Prismion swarm with loop topology
- All 4 ring layers (main + sensory + vault + think)
- Brainstem governance (entropy-driven gating)
- Panic recovery (gradient spike detection + rollback)
- AGC (Adaptive Gradient Control) — later proven unnecessary (TOT-H007)
- Thermodynamic governors (pointer velocity caps)
- LCX associative memory (hash-bucketed cosine search)
- C19 activation at every projection point

Key proofs at this stage:
- **TOT-H008**: sqrt(N) receptive field scaling law — confirmed
- **TOT-H010**: LCX scratchpad enables cross-being coordination — confirmed
- **TOT-H007**: AGC unnecessary, Adam self-regulates — confirmed

---

## v4.0 — Minimal Reference → Upward Build

Stripped to bare ring mechanism, then systematically proving & adding components upward.

**Proved & locked in (ground floor):**

| Component | Decision | Evidence |
|-----------|----------|----------|
| Ring buffer | YES | Overfit test: 97.7% echo memorization |
| N=2 experts | YES | Baseline established |
| Activation | tanh | Beats c19 on combined metric (+12.76% chained XOR) |
| Phase embeddings | ON (sin/cos) | +0.52% to +1.56% all tasks |
| Depth signal (helix_z) | REMOVED | Harmful — uniformizes behavior |
| Attention kernel | V-shape | Best geometric mean (76.22%) |
| Attention radius | Learnable R_param | Fixed R=1 won tiny benchmark, BUT at scale experts specialize (R_eff=467 vs 85) |
| I/O encoding | Learned | Hadamard fails completely (3.6% vs 63%) |
| Expert write weighting | ON | Gradient-based confidence works (conf=[1.24, 0.76]) |
| Dynamic window | ON | 30% speedup via win=min(ceil(R_eff×1.5)+5, M/2) |

**Current state:** Ground floor solid. Ready to build upward.

**Upward build path (next floors, user decides order):**
- Tanult jump gate (fix Je/Jw → nn.Linear(hidden→1))
- Tanult pointer destinations (fix φ-table → nn.Parameter)
- Per-expert context strength (fix S → tanult)
- Weight tying (shared I/O, -2.1M params)
- LCX long-term memory
- Think ticks
- Binary-bits mód (PF-013)

---

## 2026-03-01 — Multi-AI Research Triage (GPT/Qwen/Grok + local adversarial review)

### Facts (observed in local runs, accepted as ground truth)

1. INSTNCT current ceiling is around high-40s masked accuracy in the recent setup.
2. Equal-size tiny transformer baseline outperforms INSTNCT on current short-context byte task.
3. BB (bulletin board) variants did not provide stable gains and frequently showed gate instability.
4. TopK/content-read path showed train/eval pathology and slot concentration issues.
5. Strict hourglass split (writer-only + reader-only) failed strongly:
   - `N=2 strict split` underperformed both `N=1 baseline` and `N=2 baseline`.
6. `S->0` probes in memory-required setups can collapse performance, so ring path can be causally relevant.

### Clarifications from adversarial review

1. Emergent expert collapse in prior runs is **not** equivalent to a successful intentional role architecture.
2. Hard role separation removes corrective feedback (writer blindness), and was empirically harmful.
3. Several external reports included useful directionality but some over-optimistic gain estimates and loose citations.
4. We should prioritize mechanism-anchored A/B tests over broad architecture jumps.

### Consolidated bottleneck hypothesis (current)

Primary:
- Credit assignment through recurrent memory path remains weak/inconsistent.

Secondary:
- Write accumulation + slot concentration effects reduce effective memory quality.

Tertiary:
- Pointer/read policy mismatch may contribute, but should be tested with minimal deltas only.

### Decisions locked (as of this entry)

1. Drop strict writer/reader split from main roadmap.
2. Keep BB disabled unless a new mechanism has clear falsifiable metric targets.
3. Avoid hard TopK route in near-term optimization path.
4. Keep one-variable-at-a-time experimental discipline.

### Next 3 minimal-risk A/B experiments (recommended)

1. Baseline + diagnostics only (no behavior change)
   - Add/keep probes:
     - per-expert read/write grad norms
     - slot coverage entropy
     - write overwrite ratio
     - read concentration
     - fresh-state eval
     - `S->0` probe

2. Baseline + **soft write gate only**
   - No hard thresholding.
   - Write path uses continuous gate multiplier (`write = g * write_vec`).

3. Baseline + **learned pointer delta only**
   - Keep read/write kernel unchanged (`vshape` baseline).
   - No BB, no TopK, no role split co-change.

### Promotion criteria

Quick gate (1k steps):
- Promote only if `>= +1.5pp` masked accuracy and no severe instability signals.

Confirm (3k steps):
- Keep only if `>= +2.5pp` over matched baseline and diagnostics improve (entropy/overwrite/grad distribution).

Failure rule:
- If neither intervention clears confirm threshold, pivot away from micro-tuning and evaluate larger design moves.

### Important metric interpretation

1. `S->0` probe:
   - On memory tasks, significant degradation is expected if ring is actually used.
   - Therefore, large degradation is **not automatically bad**; it indicates ring dependency.

2. Fresh-state eval:
   - Should be interpreted relative to task design (carry-state vs non-carry-state setup), not as a universal absolute threshold.

---

## 2026-03-01 — GPT Deep Research Report (v2) Interpretation

### What is strong in the report

1. Correctly anchors on observed failures:
   - plateau vs equal-size transformer gap
   - BB instability/no robust gain
   - topK train/eval pathology
   - strict split catastrophic failure
2. Prioritizes mechanism-level bottlenecks over random sweeping.
3. Proposes a sensible intervention order:
   - stabilize write dynamics first
   - then stabilized soft addressing
   - enforce train/eval state parity and causal probes
4. Emphasizes deterministic A/B protocol and fresh-state + `S->0` checks.

### What to treat as uncertain / over-claimed

1. Numerical gain estimates are indicative, not guaranteed.
2. Some literature mapping is mechanism-level (useful) but not direct proof for this exact architecture.
3. Report notes incomplete code visibility in its own environment; local repo truth remains primary.

### Decisions after GPT report (locked)

1. Keep strict split dropped.
2. Keep BB off for main line.
3. Keep hard topK out of near-term path.
4. Execute intervention path with one-variable-at-a-time discipline:
   - Phase A: write stabilization (erase/add/gate/norm controls)
   - Phase B: soft addressing + anti-collapse regularizers
   - Protocol hardening (state parity suite) remains mandatory throughout.

### Minimal implementation sequence (next)

1. Protocol-only baseline run with full probes.
2. Baseline + stabilized write (only change).
3. If pass, add soft addressing controls (only next change).
4. Promote only on confirmed 3k-step gains with better diagnostics.

---

## 2026-03-01 — QWEN Deep Research Interpretation

### What is strong

1. Correctly elevates instrumentation to highest priority (before more architecture churn).
2. Correctly interprets strict split failure as a **diagnostic confirmation**, not a tunable near-win.
3. Correctly warns against random sweeps on known-failing branches (topK/hard split/extra cache complexity).
4. Uses explicit pass/fail framing and reproducibility constraints.

### What is risky / needs correction

1. Proposed ordering (`pointer upgrade` before recurrence stabilization) is risky for this project:
   - if write/read dynamics are unstable, richer addressing can amplify collapse.
   - safer order in this codebase: stabilize write/gradient path first, pointer policy second.
2. Content-attention pointer suggestion appears full-ring flavored in places; this is not "very low cost" at `M=1024` and can reintroduce concentration issues.
3. Fresh-state thresholds must remain task-relative:
   - absolute thresholds (e.g., fixed minimum %) are not universally valid for carry-state memory benches.

### Decision taken from QWEN report

1. Keep QWEN's diagnostics-first principle.
2. Keep detour avoid-list (hard split, cache stacking, broad sweeps on broken variants).
3. Adjust intervention order for local reality:
   - Step 1: protocol + probes
   - Step 2: write-path stabilization
   - Step 3: pointer/addressing upgrade only after stability evidence

### Net consensus impact

QWEN report is accepted as a strong process guide (instrumentation + gating discipline), with implementation order corrected to match observed local failure patterns.

---
## 2026-03-01 — Claude Opus Report (wf-40dc...) Interpretation

### Strong points accepted

1. Good discipline on diagnostics-first: proposes concrete probes (read/write grad norms, read entropy, slot utilization, S->0, fresh-state eval).
2. Correctly flags hard TopK and strict writer/reader split as high-risk based on observed collapse patterns.
3. Correctly pushes 1k gate + 3k confirm protocol with explicit kill rules.
4. Useful focus on write dynamics (overwrite/retention) instead of only read kernel sweeps.

### Critical mismatch with current v4 code (adversarial check)

1. Opus core claim says memory writes are detached and write-content gradients are dead.
2. In current `v4/model/instnct.py`, write path is differentiable:
   - write vector: `write_vec = self.write_proj[i](hidden_lst[i])` (or identity)
   - write op: `ring_tns = func_softwrit_tns(...)`
   - `func_softwrit_tns` uses `scatter_add` on `weights * write_val` without `.detach()` on `write_vec`.
3. Therefore, the exact "dead write gradient due to detach" diagnosis is not valid as a primary explanation for current branch results.
4. The report is still useful as a design prompt, but this main mechanism claim is stale for this code state.

### Practical conclusions after Opus triage

1. Do not pivot solely based on the "detached write" thesis.
2. Keep using Opus protocol ideas (instrumentation and stricter A/B gates).
3. Prioritize interventions that can still help even with differentiable writes:
   - write retention control (EMA/gated write)
   - anti-concentration diagnostics and penalties
   - stable soft addressing only after write-side stability is verified
4. Keep BB off and strict split off in the mainline unless a new variant beats baseline under 3k confirm.

### Immediate next action (consensus)

1. Run baseline with full probes enabled and archive probe traces.
2. Run one write-side intervention only (EMA/gated write).
3. Promote only if 1k gate and 3k confirm both pass against matched baseline.

## 2026-03-01 — Adversarial Sanity Check on `probe_dataflow.py` Claims

### Reproduction status

1. Claude-shared `probe_dataflow.py` fresh-init numbers were reproduced locally on CPU with matching trend:
   - `rank90` stayed very low (`~2`),
   - `adj_cos` increased to ~`0.995` by ~150-200 stream steps,
   - indicating rapid ring homogenization in the tested setup.
2. Backward micro-probe confirmed non-zero grads on major paths (no immediate dead-path bug in this bench config).

### Extra sanity sweep (fresh stream, no training, 200 steps)

Same bench family (`M=256`, `hidden=512`, `slot=64`, `R=1`, bitlift, lowrank_c19):

- `N=1`: `adj_cos=0.7811`, `rank90=1`, `slot_norm=4.513`, `active=78.9%`
- `N=2`: `adj_cos=0.9953`, `rank90=2`, `slot_norm=7.447`, `active=100.0%`
- `N=3`: `adj_cos=0.9940`, `rank90=3`, `slot_norm=8.428`, `active=100.0%`

Interpretation:
- Multi-expert setup (N>=2) strongly amplifies homogenization under current additive write dynamics.
- N=1 is materially less collapsed by adjacency metric, even before training.

### Important caveats (avoid over-claim)

1. This probe demonstrates a strong failure mode in the tested bench regime, but is not alone sufficient to prove universal failure on every config/task.
2. `rank90` and `adj_cos` should be interpreted together; low rank alone can occur even when adjacency collapse is weaker.
3. Forward stage ratio at `t=0` can be affected by projection biases (non-zero `read_proj` output even with empty ring read vector), so these ratios are useful but not absolute.

### Decision impact

1. Treat the "scatter/add write dynamics + multi-expert interference" hypothesis as high-priority and evidence-backed.
2. Keep strict split and hard topK out of mainline.
3. Next A/B should isolate write stabilization first (single change), then pointer/read upgrades.
