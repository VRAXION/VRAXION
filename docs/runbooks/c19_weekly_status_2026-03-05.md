# C19 / Ring Nightly Status — Canonical Summary

This note is the canonical nightly state for the current C19 + ring-memory research line.

Rules for reading it:
- `main` is untouched.
- This note is about `nightly` research only.
- Only canonical runner surfaces count as evidence:
  - `small_wikitext_fresh`
  - `fast_memory_carry`
  - `wikitext_sequential_carry`
- Old one-off probes and scratch files are not treated as primary evidence.

Primary entrypoint:
- [nightly_research_runner.py](../../v4/tests/nightly_research_runner.py)

## 1. What Is Actually Settled

### C19 core
- Standalone C19 work does support `dual-phi` as the lead activation variant.
- `C_init = pi` stays the default.
- `C` should remain learnable.
- Tail changes are not priority:
  - current tasks mostly stay inside the periodic core
  - tail-heavy ideas did not show useful signal

This note does not repeat the full standalone C19 sweep history. The current practical outcome is:
- keep `dual-phi`
- keep `C_init = pi`
- keep `C` learnable

### Ring research surfaces are now separated correctly

We previously mixed together two different regimes:

1. `small_wikitext_fresh`
- fresh state each batch
- pointer resets each batch
- not a full-ring carry verdict

2. `fast_memory_carry`
- carry is active
- valid mechanistic long-memory surface

3. `wikitext_sequential_carry`
- carry is active on real data
- valid real-data carry surface

This separation is now enforced by the canonical runner and baked into every artifact.

## 2. Canonical Verdicts

### A. Small real-data fresh surface
Surface:
- `small_wikitext_fresh`
- CPU
- small model
- `state=None` each batch

Current verdict:
- `LL` is the best branch on this surface.
- `GL` and `GG` are not better.
- This verdict applies only to the fresh-start surface.

Key evidence:
- `LL`:
  - [nightly_runner_small_wikitext_fresh_LL_20260306_212253.json](../../v4/dev_notes/telemetry/nightly_runner_small_wikitext_fresh_LL_20260306_212253.json)
- `GL`:
  - [nightly_runner_small_wikitext_fresh_GL_20260306_212254.json](../../v4/dev_notes/telemetry/nightly_runner_small_wikitext_fresh_GL_20260306_212254.json)
- `GG`:
  - [nightly_runner_small_wikitext_fresh_GG_20260306_212253.json](../../v4/dev_notes/telemetry/nightly_runner_small_wikitext_fresh_GG_20260306_212253.json)

Interpretation:
- read-only or full global topk does not win on the current pooled-topk design
- this does not prove that global retrieval is useless in general
- it proves that this specific pooled-topk path is not a near-term mainline candidate on the fresh real-data surface

### B. Fast memory carry surface
Surface:
- `fast_memory_carry`
- CPU
- repeating-pattern long-memory bench
- carry active

Current verdict:
- `GL` is mechanistically viable here.
- global retrieval is not dead in principle.
- `GG` is not the main upgrade path.

Key evidence:
- `LL`:
  - [bench_fast_memory_ll_10k_cpu_20260306.json](../../v4/dev_notes/telemetry/bench_fast_memory_ll_10k_cpu_20260306.json)
- `GL`:
  - [bench_fast_memory_gl_10k_cpu_20260306.json](../../v4/dev_notes/telemetry/bench_fast_memory_gl_10k_cpu_20260306.json)
- `GG`:
  - [bench_fast_memory_gg_10k_cpu_20260306.json](../../v4/dev_notes/telemetry/bench_fast_memory_gg_10k_cpu_20260306.json)

Interpretation:
- the global branch can work when the task genuinely forces non-local memory use
- this is why `GL` should not be described as "dead" in general
- however, this bench is mechanistic evidence, not a production recommendation

### C. Small standard baseline
Param-matched tiny transformer on the small fresh real-data surface:
- [bench_tiny_transformer_wikitext_small_20260306_193120.json](../../v4/dev_notes/telemetry/bench_tiny_transformer_wikitext_small_20260306_193120.json)

Current read:
- the small ring baseline beat the tiny transformer in final quality on that tiny surface
- but the transformer was much faster

Interpretation:
- the ring path is not placebo
- but its efficiency cost is still real

## 3. What We Closed

### A. Pooled hard topK is not the near-term path

This is the most important cleanup point.

What failed:
- `GL` on small fresh real-data did not beat `LL`
- `GG` on small fresh real-data did not beat `LL`
- `K=1` did not rescue the topk path
- full global read+write still did not become the winner

Why this matters:
- the current topk path is a pooled read:
  - several retrieved slots are collapsed into one weighted sum vector
- this is exactly the kind of design that blurs multiple associations together
- this branch should stay out of the near-term mainline path

Practical decision:
- do not spend more time on `K` sweeps for the current pooled-topk design
- do not treat `GL/GG` as the next integration branch

### B. Shortest-arc seam mode is not the current bottleneck

Implemented as a nightly-only option:
- `pointer_seam_mode = mod | shortest_arc`

Quick gate verdict:
- seam crossings definitely happened
- coverage was full
- shortest-arc still did not help

Meaning:
- the current learned-pointer regime is not seam-limited
- keep the option as research code
- do not treat it as a priority branch

### C. Write-path perf rewrites did not survive full validation

What happened:
- several write-path and micro-opt rewrites looked promising in microbench
- they did not hold up in the real proxy step

Meaning:
- the active perf problem is not solved by small algebra rewrites
- the main pain remains tensor churn / copy / scatter / index traffic

## 4. What Pointer Interpolation Actually Means

This is where confusion happened, so this is explicit.

### What pointer interpolation does fix
- local pointer-center discreteness
- `ptr.long()` / integer-center gradient cut

### What pointer interpolation does not fix
- long-range retrieval by itself
- "answer is on the far side of the ring" problems
- pooled-topk blur

So both statements are true:
- pointer interpolation is a valid local gradient fix
- pointer interpolation is not the solution to late-game far-memory retrieval

Evidence:
- on `fast_memory_carry`, pointer interpolation helped
- on `wikitext_sequential_carry`, it did not become the main win

Key artifacts:
- mechanistic carry:
  - [nightly_runner_fast_memory_carry_LL_20260306_221755_793650.json](../../v4/dev_notes/telemetry/nightly_runner_fast_memory_carry_LL_20260306_221755_793650.json)
  - [nightly_runner_fast_memory_carry_LL_20260306_222517_866783.json](../../v4/dev_notes/telemetry/nightly_runner_fast_memory_carry_LL_20260306_222517_866783.json)
- real-data sequential carry:
  - [nightly_runner_wikitext_sequential_carry_LL_20260306_223327_017337.json](../../v4/dev_notes/telemetry/nightly_runner_wikitext_sequential_carry_LL_20260306_223327_017337.json)
  - [nightly_runner_wikitext_sequential_carry_LL_20260306_224150_937913.json](../../v4/dev_notes/telemetry/nightly_runner_wikitext_sequential_carry_LL_20260306_224150_937913.json)

## 5. What The Current Bottleneck Looks Like

The strongest current read is:
- not "topk is the answer"
- not "shortest-arc is the answer"
- not "pointer interpolation alone solves long-memory"

The likely remaining issue is:
- structured retrieval bandwidth / memory hierarchy
- not just wider addressing range

Why:
- global topk could become genuinely non-local and still not win on small fresh real-data
- full global read+write still did not become the winner there
- pointer interpolation helped on the mechanistic carry task, but not on the small real-data carry task

This points away from:
- more pooled-topk tuning
- more seam tweaks

And toward:
- structured multi-timescale reads
- non-pooled retrieval structure
- or other explicitly hierarchical memory access

## 6. The One Next Architecture Branch

The next branch should be:
- **multi-timescale taps**

Not:
- more pooled topk
- more `K` sweeps
- more shortest-arc work

Reason:
- this matches the older deep-research ranking
- it also matches the newer nightly evidence
- it targets long-range access without collapsing everything into one pooled global read

## 7. Operational Rules Going Forward

1. Any new claim must name the surface:
- `small_wikitext_fresh`
- `fast_memory_carry`
- `wikitext_sequential_carry`

2. No artifact may be used as evidence unless it is generated by:
- [nightly_research_runner.py](../../v4/tests/nightly_research_runner.py)
- or explicitly marked as non-canonical scratch

3. Fresh-start verdicts are not general ring verdicts.

4. Carry verdicts are the only valid basis for long-range memory claims.

5. Pooled hard topk stays closed for near-term mainline work.

## Final Summary

Current best consolidated read:

- `dual-phi` remains the lead C19 form
- `C_init = pi`, `C` learnable
- small fresh real-data: `LL` wins
- carry mechanistic bench: `GL` is viable, so global retrieval is not dead in principle
- pooled topk is still not the near-term solution
- pointer interpolation is a local fix, not a late-game memory fix
- shortest-arc is not the current bottleneck
- the next serious architecture branch is **multi-timescale taps**
