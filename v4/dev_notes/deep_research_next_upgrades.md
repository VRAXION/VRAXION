# Deep Research Consolidation: Next Architecture Upgrades

**Date**: 2026-02-28
**Context**: Run 19 done (52.4% best masked acc). Collecting deep research from multiple AI agents to decide what to build next.
**Goal**: Pick the highest-ROI change(s) and implement.

---

## Research Sources

| # | Source | Status | Key Recommendation |
|---|--------|--------|--------------------|
| 1 | Qwen Deep Research | DONE | Learned embedding + differentiable pointer (linear interp) |
| 2 | Ring Memory Expert Architecture Design (PDF) | DONE | Differentiable pointer (Gaussian/continuous), multi-timescale taps, partitioned writes |
| 3 | GPT Online Deep Research | DONE | Continuous pointer kernel (R_max=16) + bounded erase/add writes + shared multi-tap buffer |
| 4 | Grok Deep Research | DONE | Fractional pointers + hybrid content/location addressing + multi-timescale taps |
| 5 | Claude Opus Online Deep Research | DONE | Replace bitlift (#1! — rank-8 bottleneck) + bilinear pointer interp (#2) + content addressing (#3) |

---

## Source 1: Qwen — "Maximizing Masked Accuracy: Architectural Blueprint"

### Ranked Options (by ROI)

| Option | Expected Gain | Cost | Risk | Verdict |
|--------|--------------|------|------|---------|
| Learned Embedding + Continuous Pointer | 52% -> ~60-65% | +524K params, negl. FLOPs | Low | **#1 — DO THIS** |
| Multi-Timescale Taps | 52% -> ~55-58% | ~0 params | Low | #2 — after pointer fix |
| Separate Input/Output Encoders | 52% -> ~54-57% | +296K params | Low | #3 — minor |
| Partitioned Buffers | Negative | saves ~1MB | High | **NO** — breaks MoE sharing |

### Two Core Problems Identified

**Problem A — Input Representation (bitlift is too rigid)**
- `bitlift` = `Linear(8, 2048)` — fixed mapping, byte->8bits->hidden
- Can't learn task-specific byte representations
- Solution: standard `nn.Embedding(256, 2048)` = 524K params
- Qwen says this is the single fastest win

**Problem B — Pointer gradient is severed (discrete indexing)**
- `torch.floor()` / integer index kills gradient flow to pointer position
- Model can learn WHAT to write, but not WHERE to read/write
- Ring becomes "glorified cache with random access patterns"
- Solution: linear interpolation between two nearest slots
  - `f = p - floor(p)` (fractional part)
  - `read = (1-f) * buffer[i] + f * buffer[i+1]`
  - Fully differentiable w.r.t. pointer position `p`
  - Same for write operation
  - Zero extra params, minimal FLOPs

### Alternatives Considered (for pointer)
- **Full softmax attention over all M slots** — O(M) per step, too expensive for M=1024
- **Gumbel-Softmax** — adds temperature annealing complexity, unstable
- **Linear interpolation** — simple, stable, cheap = WINNER

### Multi-Timescale Taps (secondary)
- Hard-coded reads from t-1, t-8, t-16, t-32 in addition to pointer read
- Zero params, negligible compute
- BUT: only useful AFTER pointer is differentiable (otherwise pointer can't learn to navigate to landmarks)
- Recommendation: defer until after pointer fix

### Buffer Sharing
- Shared ring = correct (MoE principle: experts collaborate via shared memory)
- Partitioned = bad (experts isolated, can't collaborate)
- Confirms our existing design is right

### A/B Test Protocol Proposed
- Control: current model (bitlift + discrete pointer)
- Variant: learned embedding + linear interpolation pointer
- Hold constant: everything else (C19, output head, optimizer, data)
- Metric: masked_acc after 6500 steps
- Pass threshold: >1% absolute improvement
- 3 runs with different seeds for statistical confidence

### Failure Modes Flagged
1. Accuracy doesn't improve → revert incrementally, check gradients
2. Loss explosion → gradient clipping (already have max_grad_norm=10)
3. Pointer diverges → use smaller LR for pointer updates
4. OOM → reduce batch (unlikely, only +524K params)
5. Interpolation bug → unit test with torch.autograd.gradcheck

### My Assessment
- **Embedding change**: AGREE. Bitlift was an experiment, learned embedding is the standard for good reason. +524K params is nothing.
- **Pointer interpolation**: AGREE in principle. This is the big one. Currently our pointer movement is via jump/walk probabilities but the ring read/write uses `torch.floor()` which indeed kills gradients to position. Linear interpolation is clean and correct.
- **NOTE**: We already have soft attention (V-shape kernel over 2R+1=5 slots). Qwen's linear interp would replace the discrete floor() in the kernel CENTER, making the whole read/write differentiable. Need to check exactly how our current `_read_ring` and `_write_ring` work vs what Qwen proposes.
- **Partitioned buffers**: AGREE to reject. We discussed this last session and reached the same conclusion.
- **Multi-timescale taps**: Interesting but secondary. Defer.

---

## Codex Adversarial Addendum (2026-02-28)

### What Qwen got right
- **Discrete pointer indexing is the main gradient bottleneck** in the current ring access path (`ptr.long()` / integer gather center).
- **Shared memory over partitioned memory** is the right default for this MoE-style ring.
- **Linear interpolation pointer read/write** is the most pragmatic first differentiable fix (cheap, low-risk, testable).

### What is inaccurate for our current codebase
- Qwen text says bitlift input is "fixed / non-learnable."  
  **Not true in our code**: `embed_encoding: bitlift` uses `nn.Linear(8, hidden_dim)` + learnable C19 params.
- Qwen flags "~707k params" as suspicious.  
  **Not true for current config**: total params are indeed ~707k for bitlift + lowrank_c19 setup.
- "Pointer untrainable => random memory" is **too strong**.  
  More precise: pointer-position gradient is cut at discrete indexing, but content pathway still trains and ring can still help.

### Code-level caveat found during review
- In `model/instnct.py`, `_C19_C_MAX = 50.0` but `_C_from_raw` docstring says `[1.0, 10.0]`.  
  This is a documentation mismatch that can confuse research conclusions; actual bound in code is 50.

### Consensus update (Codex)
1. **Proceed with A/B**: current baseline vs `learned embedding + pointer interpolation`.
2. Keep all else fixed for first pass (data, LR schedule, batch, seq_len, output head, C19 settings).
3. **Do not implement partitioned buffers** in this phase.
4. Multi-timescale taps stay **Phase 2** after pointer fix validates.

---

## Source 2: PDF — "Architectural Audit and Differentiable Design Specification for Recurrent Ring-Memory Models"

### Headline Recommendations
- Replace discrete pointer indexing with continuous differentiable addressing (fractional Gaussian or Gumbel-Softmax relaxation).
- Add multi-timescale taps.
- Partition shared ring writes by expert (read remains shared).

### Codex Adversarial Assessment

#### What aligns with code reality
- Correctly identifies the pointer-position gradient cut risk from integer slot indexing.
- Correctly prioritizes differentiable addressing as a high-impact direction.
- Multi-timescale taps are plausible as a Phase 2 upgrade.

#### What is overstated or inaccurate
- "Fatal flaw" language is too strong for our observed behavior; current model still trains and reaches low-50% masked accuracy.
- Claims C19 is "unbounded/perilous" are inaccurate for current code path:
  - rho is sigmoid-bounded.
  - C is sigmoid-bounded.
- Recommends partitioned buffers as mandatory; this conflicts with shared-memory MoE collaboration and lacks direct evidence on our setup.
- Absolute gain claims (+15-20%) are speculative and not backed by controlled A/B on this repo.

#### Important compatibility note
- We already use kernel-weighted ring reads/writes over a local window.
- The remaining gap is the **discrete center selection** (`ptr.long()`), not absence of weighting.
- Therefore, the actionable change is to make the center addressing continuous, not to rewrite the entire ring mechanism.

### Net Verdict (Source 2)
- **Keep**: differentiable pointer addressing proposal (candidate for A/B).
- **Keep later**: multi-timescale taps (after pointer A/B).
- **Reject for now**: partitioned expert writes.
- **Treat cautiously**: large numeric gain projections.

---

## Source 3: GPT Online — "Evidence-Backed Architecture Recommendation for a Differentiable Byte-Level Ring-Memory Model"

### Ranked Options (by ROI)

| Option | Expected Gain | Compute Cost | Risk | Verdict |
|--------|--------------|-------------|------|---------|
| Continuous pointer kernel + bounded writes + shared buffer | +3 to +8 acc pts | 1.1–1.4× | Medium | **#1 — best ROI** |
| Continuous pointer kernel + bounded writes (no buffer) | +2 to +6 acc pts | 1.05–1.2× | Low–Med | #2 — minimum viable fix |
| Full NTM-style soft attention over all M slots | +4 to +10 acc pts | 2–5× | High | #3 — too expensive |
| Shared buffer only (keep discrete pointer) | +1 to +3 acc pts | 1.05–1.2× | Low | #4 — doesn't fix root cause |
| Remove input projection | -2 to +1 acc pts | Low | Medium | **NO** — byte models still embed |

### Three Core Changes Proposed

**Change 1 — Continuous Pointer Kernel (the big one)**
- Fix R_max=16, window W=33 slots around pointer
- Gaussian weighting: `w_{i,k} = softmax_k( -d_{i,k}² / (2σ_i²) )`
- Learn σ_i per expert: `σ_i = σ_min + softplus(r_i)`
- Gradients flow into `p_i` and `σ_i` through kernel weights — no `.long()` dependency
- Key difference from Qwen: uses Gaussian kernel over W=33 slots, not linear interp over 2 slots

**Change 2 — Bounded Erase/Add Writes (NEW — not in Source 1 or 2)**
- Current additive writes can blow up under multi-expert contention
- NTM-style: `R[j] ← R[j] · (1 - g_i·w·e_i) + g_i·w·a_i`
  - `a_i = tanh(W_a h_i)` — add vector (bounded by tanh)
  - `e_i = sigmoid(W_e h_i)` — erase gate (0–1)
  - `g_i = sigmoid(w_g^T h_i)` — write strength scalar
- Prevents norm explosion and reduces memory blur

**Change 3 — Shared Multi-Tap Read-Only Buffer**
- 11 taps: 8 recent `{t-1,...,t-8}` + 3 dilated `{t-16, t-32, t-64}`
- Single-head dot-product attention: `q_i = W_q h_i`, keys/values = buffer vectors
- Read-only to experts — cheap coordination substrate
- Cost: ~131K params (W_q), O(L·S) FLOPs per expert per step

### Input Encoding Position
- Keep small, keep information-preserving
- Both bitlift `Linear(8→S)` and byte embedding `Embed(256→S)` are valid
- Do NOT remove input projection

### Adversarial Risk Register (5 risks)
1. **Soft-address blur** — kernel weights go near-uniform → ring = low-pass average
2. **Write collisions** — experts overwrite each other
3. **Expert collapse** — experts become redundant
4. **Pointer oscillation** — gradients jitter at cell boundaries
5. **Train/deploy state mismatch** — TBPTT carry-over vs fresh state

### Codex Adversarial Call
- High practical value, but citation markers are internal-style placeholders and not directly auditable.
- Numerical gain ranges should be treated as hypotheses, not forecasts.
- Main actionable consensus remains: (1) differentiable pointer, (2) bounded writes, (3) shared ring, (4) defer multi-timescale.

### Claude Assessment (Source 3)
- **Most thorough and cautious** of the three. Proper evidence anchors, adversarial risk register, 4-stage A/B protocol.
- **Bounded writes (erase+add)** is genuinely new — not raised by Qwen or Gemini. Valid concern in theory, but we haven't seen ring norm explosion in runs 18-20. **Monitor, don't pre-optimize**.
- **R_max=16 Gaussian kernel** is overkill for us — we already have V-shape over R=2 (5 slots). The continuous CENTER is what matters, not a 33-slot window. **Linear interp is still simpler and sufficient for first A/B**.
- **Shared multi-tap buffer** is essentially multi-timescale taps with attention — already in our Phase 2 plan. GPT packages it more formally.
- **Write stability** concern: valid flag, but premature to implement. If pointer fix succeeds and ring norms stay stable, no need.

### Net Verdict (Source 3)
- **Keep**: continuous pointer direction, bounded write concepts (as Phase 2 if needed).
- **Keep as optional**: shared multi-tap buffer after Phase 1.
- **Treat cautiously**: specific gain numbers without repo-local A/B.
- **New consideration**: bounded writes → add to monitoring checklist, implement only if ring norm issues appear.

---

## Source 4: Grok — Deep Research on Differentiable Ring-Memory Architecture

### Ranked Options (by ROI)

| Option | Expected Gain | Compute Cost | Risk | Verdict |
|--------|--------------|-------------|------|---------|
| Fractional pointers + hybrid soft attention | +15-25% acc | +3-5% FLOPs, +5K params | Low | **#1 — fix gradient cut** |
| Multi-timescale short-term buffers | +10-15% acc | +5-10% FLOPs, +10K params | Medium | #2 — add hierarchy |
| Per-expert partitioned buffers | +5-15% acc | +10-20% FLOPs, params×N | High | #3 — coordination loss risk |
| No input projection (raw bytes) | +0-5% acc | -5% FLOPs, -10K params | Low | #4 — dimensionality mismatch |
| Shared read-only buffer | +5-10% acc | +2% FLOPs, +5K params | Low | #5 — coordination |

### Key Proposals

**Fractional Pointers (bilinear interpolation)**
- `torch.nn.functional.interpolate` for 1D ring addressing (soft gather/scatter)
- Hybrid content+location addressing per NTM design
- Claims +20% acc based on DNC literature
- Suggests `torch.interpolate` — different implementation from Qwen's manual linear interp

**Multi-timescale Taps**
- Recent taps t-1..8 + sparse t-16/32/64
- 128-slot shared buffer with soft read via attention
- Ranked #2, pairs with fractional pointers

**"Ship in 1 week" Recommendation**
- Fractional pointers with bilinear interpolation
- Drop input projection if params tight (controversial — contradicts other sources)
- Keep shared buffer, add recent taps only
- Expect +10-15% acc at <3% compute

### Suspicious Claims Flagged (by Grok itself)

**C19 Activation**
- "No primary sources found for C19 activation in neural networks"
- Correctly identifies it as custom/experimental — C19 is our proprietary bounded activation
- Not a problem — it IS custom, intentionally so

**52% Accuracy Context**
- Compares to ByT5 (~20-50% byte prediction), EvaByte 6.5B (~75-85%), Bolmo 1B (~60-70%)
- Notes 52% is "plausible for under-optimized 707k-param model"
- **Important context**: our model is 707K params vs billions — the comparison is informative but unfair

**707K Params with hidden=2048**
- Flags as "unusual" — correct, but explained by our low-rank architecture (slot_dim=64, bitlift, lowrank_c19 head)
- Not suspicious once you understand the factored design

### Adversarial Risk Register (5 risks)

1. **Pointer stagnation** — fractional blend stuck in local minima, overwriting same slots
2. **Information overwriting** — soft writes collide in shared ring
3. **Gradient explosion/vanishing** — C19 bounds fail + multi-timescale amplifies
4. **Expert divergence** — one expert dominates (N=2 too small concern)
5. **Memory overflow** — ring wraps without deallocation, diluting relevance

Mitigations: pointer entropy regularizer, usage-based deallocation (DNC), expert balancing loss

### Claude Assessment (Source 4)

- **Most aggressive gain projections** of all sources (+15-25% from fractional pointers alone). Based on DNC/NTM literature which uses much larger models on algorithmic tasks — NOT directly transferable to our 707K byte-level setup.
- **Hybrid content+location addressing** is NTM/DNC feature creep — we don't need content-based addressing yet. Our pointer already has jump/walk dynamics. Adding content addressing changes the architecture fundamentally. **Reject for Phase 1**.
- **`torch.nn.functional.interpolate`** suggestion — interesting implementation idea but not directly applicable to our ring buffer (designed for image/grid interpolation, not circular 1D buffer indexing). Manual linear interp is cleaner.
- **Usage-based deallocation + temporal links** — DNC features that add massive complexity. Our ring is circular by design (natural deallocation via overwrite). **Reject**.
- **"Drop input projection if params tight"** — contradicts GPT Online, Gemini, and even Grok's own ranked table where removal scored lowest. Inconsistent.
- **C19 flag** is valid observation (it IS custom) but doesn't affect the recommendation — C19 works and is bounded.
- **52% context vs SOTA** is useful perspective: shows our model isn't broken, just small and gradient-limited.

### Net Verdict (Source 4)

- **Keep**: fractional pointer direction (agrees with all sources).
- **Reject**: hybrid content+location addressing, DNC temporal links, usage-based deallocation — overengineering.
- **Reject**: drop input projection suggestion.
- **Treat cautiously**: +15-25% gain projections — most aggressive and least grounded of all sources.
- **Useful context**: 52% comparison to SOTA byte models validates that our model isn't fundamentally broken, just needs the gradient fix.

---

## Source 5: Claude Opus Online — "Adversarial Architecture Audit for VRAXION Ring-Memory Model"

### Ranked Options — DIFFERENT PRIORITY ORDER THAN ALL OTHER SOURCES

| Rank | Option | Expected Gain | Cost | Risk | Verdict |
|------|--------|--------------|------|------|---------|
| 1 | Replace bitlift with Embed(256,64) | +3-8% acc | 0 extra FLOPs, ~16K params | Very low | **#1 — rank-8 bottleneck** |
| 2 | Bilinear pointer interpolation | +1-4% acc | ~130 FLOPs/read | Very low | #2 — restore pointer gradients |
| 3 | Content-based + Gaussian addressing | +2-5% acc | ~65K FLOPs (~2%) | Low-Med | #3 — global search |
| 4 | Multi-timescale taps (exponential) | +1-3% acc | ~4% FLOPs | Moderate | #4 — hierarchy |
| 5 | Per-expert scratchpads + shared ring | +0.5-2% acc | ~24KB, ~4K params | Low | #5 — eliminate write interference |

### Key Insight: Ranks Input Fix ABOVE Pointer Fix (unique)

This is the **ONLY source** that puts embedding change as #1 over pointer fix.

The argument: bitlift `Linear(8, 2048)` constrains all 256 byte representations to a **rank-8 subspace** — only 8 independent directions in 2048-dim space. Bytes that share bit patterns (e.g. 'A'=0x41 and 'a'=0x61 differ by 1 bit) get similar representations regardless of linguistic role. Standard Embedding gives each byte an independent vector.

### Content-Based Addressing (#3 — now 2/6 sources)
- Combine location-based (Gaussian kernel) with content-based (cosine similarity over ring)
- 3-way interpolation gate: location vs content vs blend
- Now Grok + Claude Opus both suggest this

### Per-Expert Scratchpads (#5)
- Hybrid: experts write to private scratchpads, merged to shared ring via gated projection
- References P-NTM (2025): dimension-partitioned writes (expert 0 → dims 0-31, expert 1 → dims 32-63)

### Suspicious Claims Section (most adversarial of all sources)

**"hidden_dim=2048 with 707K params is impossible"**
- Claims GRU at 2048 would need 25M+ params
- **WRONG**: we don't use GRU. Experts use C19 activation on hidden_dim=2048 directly, with factored read_proj/write_proj between slot_dim=64 and hidden_dim=2048. Low param count is by design.

**"Gaussian kernel + long() is contradictory"**
- Correctly identifies that `long()` kills gradient regardless of kernel applied after it
- **CONFIRMS our diagnosis**: V-shape kernel weights work fine, discrete center selection is the problem

**"seq_len=64 is 8× too short"**
- Literature shows 512+ bytes for byte-level LM
- **Partially valid but ignores TBPTT**: sequential=true carries state across sequences. seq_len controls gradient depth, not effective context. Ring IS the long-range memory.

**"52% is only 7-12 pts above frequency baseline"**
- Valid: top-5 byte frequencies could get ~40-45%
- **BUT**: we use masked accuracy on UNPREDICTABLE positions (mask=1), not all positions

**"C19 needs ablation vs GELU/SiLU"**
- Already done in our v4 ablation suite. tanh > C19 > GELU > SiLU. Claude didn't have this data.

### Codex Adversarial Assessment
- Correctly catches GRU assumption as wrong
- Correctly notes Embed(256,64) is not drop-in for hidden_dim=2048 architecture
- Keeps pointer fix and embedding A/B as valid, rejects content addressing bundle for Phase 1

### Claude Assessment (Source 5)
- **Most rigorous adversarial analysis** — catches real issues (rank-8 subspace, seq_len)
- **Rank-8 argument is mathematically valid** but may be overstated: 8 basis vectors CAN point anywhere in 2048-space, question is empirical
- **WRONG about architecture**: assumes GRU, doesn't understand factored C19 design. Discount dependent conclusions.
- **Suggests Embed(256,64) targeting slot_dim** — wrong for our arch. Would need Embed(256, 2048) for hidden_dim input.
- **Content-based addressing**: now 2/6 suggest it. Still reject for Phase 1.
- **seq_len concern**: valid in theory, mitigated by TBPTT. Worth testing longer seq in Phase 2.
- **Best quote**: "The architecture does not need to be replaced. It needs its gradient pathways unblocked and its information bottlenecks widened."

### Net Verdict (Source 5)
- **Keep**: bilinear pointer interp (agrees with all), rank-8 argument strengthens input A/B case
- **Reconsider priority**: embedding change maybe deserves earlier testing (rank-8 is a strong argument)
- **Reject for Phase 1**: content addressing, scratchpads, seq_len increase
- **Architecture misunderstanding**: GRU assumption invalidates several specific conclusions
- **Treat cautiously**: gain numbers (+3-8% for embedding, +1-4% for pointer) — most conservative of all sources but still speculative

---

## Cross-Source Disagreement Matrix (Claude assessment — all 6 sources)

| Topic | Qwen | Gemini | GPT Online | Grok | Opus | Codex | Claude Call |
|-------|------|--------|------------|------|------|-------|------------|
| Input encoding | Embedding(256,H) | Keep bitlift | Keep small (either) | Keep or drop | Replace bitlift with embed | Keep bitlift | **TEST BOTH** — easy A/B |
| Buffer sharing | Shared | Write-partition | Shared | Shared | Hybrid scratchpads | Shared | **Keep shared** (4+ vs 2) |
| Pointer fix | Linear interp (2) | Gaussian (W) | Gaussian (W=33) | Bilinear interp | Bilinear interp | Linear interp | **Linear interp first** |
| Write mechanism | — | — | NTM erase+add | Erase+add (NTM) | Scratchpads | — | **Monitor** — raised by some, no evidence yet |
| Content addressing | — | — | — | Hybrid content+loc | Add content addressing | — | **Reject** — overengineering |
| Multi-timescale | Phase 2 | #2 priority | Phase 2 (buffer) | #2 priority | Add later | Phase 2 | **Phase 2** (near-unanimous) |
| C19 bounds | — | Clamp 4.0 | — | "Suspicious/custom" | Ablate/uncertain | Sigmoid-safe | **Monitor** — it's custom, it works |
| Accuracy projection | +8-13% | +15-20% | +3-8 pts | +15-25% | +4-10 pts | "Cautiously" | **Speculative** — A/B will tell |

### Key Observations

**Near-unanimous (5/6 as #1, 1/6 as #2):** discrete pointer indexing is the top bottleneck. Only Claude Opus ranks embedding change above it.

**Input encoding split:** Qwen + Claude Opus say replace bitlift (rank-8 argument). Gemini + Codex say keep. GPT Online + Grok say either is fine. **TEST BOTH** — the A/B will settle this.

**Bounded writes:** GPT Online + Grok + Claude Opus (scratchpads) = 3/6 raise write stability. No empirical evidence of problem yet. **Monitor**.

**Pointer method split:** Qwen + Codex = linear interp. Gemini + GPT Online = Gaussian kernel. Grok + Opus = bilinear. Our V-shape kernel R=2 handles windowing — only CENTER needs fixing. **Linear interp = simplest first step**.

**Content addressing:** Grok + Claude Opus (2/6) suggest it. Still too much architectural change for Phase 1. **Reject**.

**C19 flag:** Grok + Claude Opus note C19 is custom/unverified. Correct — it IS custom. Already ablated (tanh > C19 > GELU). Not a problem.

**Architecture misunderstanding:** Claude Opus assumes GRU, concludes hidden_dim=2048 impossible with 707K params. Wrong — our factored C19 design is real. Several of Opus's numeric conclusions depend on this wrong assumption.

---

## DECISION LOG

| # | Decision | What | Why | Status |
|---|----------|------|-----|--------|
| 1 | Differentiable pointer | Make ring read/write center continuous (linear interp) | 6/6 sources agree: discrete floor() kills gradient to position | **APPROVED** |
| 2 | Input encoding A/B | Test bitlift vs learned Embedding | Qwen+Opus say replace (rank-8 argument), others say keep/either — need data | **APPROVED** |
| 3 | Keep shared ring | No write-partitioning | 4/6 sources + phi-offsets already separate experts | **APPROVED** |
| 4 | Multi-timescale taps | Defer to Phase 2 | 6/6 sources agree it's secondary to pointer fix | **DEFERRED** |
| 5 | C19 rho bounds | Monitor, don't tighten yet | Empirical data shows no explosion; C19 already ablated (tanh > C19 > GELU) | **MONITOR** |
| 6 | Bounded writes (erase+add) | Monitor ring norms, implement if needed | GPT Online + Grok + Opus raised this (3/6); no empirical evidence of problem | **MONITOR** |
| 7 | Content addressing (NTM) | Reject for Phase 1 | Grok + Opus suggest (2/6); fundamentally changes architecture | **REJECTED** |
| 8 | DNC features (temporal links, deallocation) | Reject | Only Grok; massive complexity, ring already handles via overwrite | **REJECTED** |
| 9 | Per-expert scratchpads | Reject for Phase 1 | Only Opus (P-NTM reference); phi-offsets already separate experts | **REJECTED** |
| 10 | Increase seq_len 64→256+ | Defer, test later | Only Opus; TBPTT sequential mode mitigates short gradient depth | **DEFERRED** |

---

## IMPLEMENTATION PLAN

### Phase 1A: Differentiable Pointer (run 21)
- Make ring read/write center selection continuous via linear interpolation
- Currently: `ptr.long()` -> integer index -> V-shape kernel around it
- New: fractional part blends between two adjacent slots, kernel still applies on top
- Zero extra params, minimal compute
- **Must verify**: check exactly how `_read_ring` / `_write_ring` use the pointer in instnct.py

### Phase 1B: Input Encoding A/B (run 22 vs run 21)
- Run 21: bitlift + differentiable pointer (isolate pointer fix)
- Run 22: learned Embedding + differentiable pointer (test input change on top)
- Compare at same sample count

### Phase 2: Multi-Timescale Taps (after Phase 1 validates)
- Only if pointer fix shows clear improvement
- Logarithmic offsets: t-1, t-4, t-16, t-64
- Requires small fusion projection layer

### NOT doing:
- Buffer partitioning (rejected)
- Gumbel-Softmax (too complex, O(M) logits)
- C19 rho clamping (no evidence of problem)

---

## Codex Deep-Research Refresh (Code-Verified, 2026-02-28)

### New hard findings from current code (line-level)

1. **Pointer center is still discretized in read path (gradient cut at addressing center).**
   - `center = ptr_tns[i].long().clamp(...)` in `model/instnct.py` line 450.
   - Window indices then built from this integer center (line 451).
   - Conclusion: kernel weights are soft, but the center index is still hard-cast.

2. **Pointer move also uses integer table lookup.**
   - `current_slot_tns = ptr_tns.long().clamp(...)` in `func_movepntr_tns` (line 258).
   - Jump target looked up from `dests[current_slot_tns]` (line 261).
   - Conclusion: jump target selection is content-trainable, position-gradient path is partially blocked by integer lookup.

3. **TBPTT/sequential carry is implemented and active in both train and eval flows.**
   - Model resumes from provided `state` in forward (lines 559-563) and returns detached carry state (625-629).
   - Train loop carries `state` when `sequential=true` (training/train.py lines 833-856).
   - Eval loop also carries `state` when `sequential=true` in checkpoint config (training/eval.py lines 87-99).
   - Dataset sequential reader is circular (`offset % n_samples`) by design (training/train.py line 258).
   - Conclusion: wrap-around with persistent state is intentional semantics for circular corpus mode.

4. **BOOT logging currently misreports parameter counts for non-`learned` encodings.**
   - `if enc != 'learned': print(...0 input params...)` (training/train.py lines 675-677).
   - `if out_enc != 'learned': print(...0 output params...)` (lines 680-682).
   - This is false for trainable non-learned modes (`bitlift`, `lowrank_c19`).
   - Conclusion: reporting bug only (not training bug), but it can mislead interpretation.

5. **C19 C-bound comment/doc mismatch still present.**
   - `_C19_C_MAX = 50.0` in code (model/instnct.py line 36).
   - `_C_from_raw` docstring says `[1.0, 10.0]` (line 47).
   - Conclusion: actual bound is `[1, 50]`; docs should be aligned.

### Python/PyTorch implementation feature-space (practical)

| Option | Pointer-position gradient | Compute delta | Complexity | Phase fit |
|---|---|---|---|---|
| Integer index (`long` center) | No | Baseline | Very low | Current baseline |
| 2-point linear interpolation (floor + alpha) | Yes (through `alpha`) | Very low | Low | **Phase 1 recommended** |
| Wider soft kernel around fractional center | Yes | Low-Med | Med | Phase 1.5 / 2 |
| Content+location hybrid addressing | Yes | Med | High | Phase 2+ only |
| STE/Gumbel hard routing | Approximate | Med | High | Later experimental only |

### Why this matters for next run decisions

- The repo is **not blocked by missing infrastructure** (TBPTT is present, carry state is present, checkpoint carry is present).
- The highest-confidence unresolved bottleneck remains **pointer discretization at the addressing center**.
- Input-encoding debate (bitlift vs learned) should remain **A/B-driven**, but pointer-center fix is orthogonal and should be landed first.

### Immediate documentation/action hygiene

- Fix BOOT param reporting for `bitlift` and `lowrank_c19` (truthful trainable param counts).
- Align C19 C-bound docs with actual bounds to avoid research confusion.
- Keep explicit note in run reports: `sequential=true` implies circular-stream semantics unless reset policy is added.

---

## Run Forensics Addendum (2026-02-28): Learned Parameter Behavior Across Recent Runs

### Scope audited
- `ARCHIVE/16..20` checkpoints + `train_log.csv`
- current `training_output` checkpoints + `train_log.csv`
- main aggregate CSV: `ARCHIVE/archive.csv` sanity

### End-state comparison (latest checkpoints)

| Run | Key variant | Best masked acc | Final masked acc | R_eff (E0,E1) | Notable learned meta |
|---|---|---:|---:|---|---|
| 16 | c19 everywhere, fixed C=pi | 47.36% | 46.99% | [47.7, 44.7] | no learnable c19 meta tensors |
| 17 | c19 everywhere, fixed C=phi | 42.55% | 42.55% | [48.1, 43.7] | no learnable c19 meta tensors |
| 18 | learnable rho (input+hidden+head rho) | 56.73% | 54.81% | [68.3, 54.2] | rho spread widened strongly |
| 19 | learnable rho+C | 52.39% | 49.57% | [48.3, 39.2] | rho+C learned, bounded OK |
| 20 | learnable rho+C + separate c19 LR group | 51.77% | 48.88% | [47.7, 40.6] | rho+C learned, bounded OK |

### What the optimizer-state sanity check proved

- Run 19: **single optimizer group**, all params decayed together to near-zero LR by end.
- Run 20: **two optimizer groups**:
  - group0 (normal weights) decayed to near-zero LR,
  - group1 (c19 meta params) stayed at constant `0.001`.
- This confirms run20 did apply the meta-LR split behavior (not a naming artifact).

### Run 19 vs Run 20 (same horizon: 6500 steps, same batch/seq/data)

- Common points compared: 651.
- Average delta (run20 - run19):
  - `+0.058` percentage points masked acc (tiny),
  - `-0.00441` masked loss (tiny improvement).
- Win rate:
  - run20 better acc at 54.7% of logged points,
  - run20 better loss at 57.1% of logged points.
- But peaks:
  - run19 best acc higher (`52.39%` vs `51.77%`),
  - run20 slightly smoother tail volatility.

Interpretation: meta-LR split changed optimization trajectory modestly, not a large regime shift.

### R-parameter behavior (important)

- In current training_output (run20 lineage), checkpoints show:
  - step 100: `R_eff ~ [61.1, 61.4]`
  - step 3300: `R_eff ~ [47.3, 42.8]`
  - step 6500: `R_eff ~ [47.8, 40.6]`
- So learnable `R` is active and moving (not frozen), but it converges to moderately wide windows (40–50 slots), not to narrow `R≈2`.
- Correlation sanity (within run20):
  - as `R_mean` shrank early, acc improved strongly; late-phase this coupling weakened (time confound + plateau).
  - do **not** treat this as causal proof alone.

### C19 meta parameter sanity

- All learned c19 parameters remained within configured bounds:
  - rho in `[0.5, 8.0]`,
  - C in `[1.0, 50.0]`.
- No NaN / out-of-bound anomalies detected.
- Practical observation:
  - `C_hidden` stayed concentrated low (~2–4),
  - `C_input` used a wider range (up to ~10 in run20).
  - This suggests the broad `[1,50]` range is largely unused in these runs.

### Aggregate CSV (`ARCHIVE/archive.csv`) integrity check

- File has mixed-schema rows:
  - early rows include full run metadata,
  - from run18 onward, many rows are appended in raw `train_log.csv` shape.
- Audit stats:
  - total rows: 9943
  - unique `(run_id, step)`: 7376
  - duplicates: 131 rows (~1.32%)
  - rows missing expected metadata fields: 2436
- Conclusion: `archive.csv` is usable for coarse trend mining, but not a strict normalized source of truth without cleaning.

### Practical conclusion from this forensic slice

1. Learnable meta pathways are functioning and stable (bounds respected, no divergence artifact).
2. Run20’s meta-LR policy is real, but gain vs run19 is incremental, not dramatic.
3. `R` is learnable but lands in medium-wide regime; this remains compatible with the pointer-center differentiability priority.
4. For reproducible cross-run claims, prefer per-run `train_log.csv` + checkpoint data over raw `archive.csv`.
