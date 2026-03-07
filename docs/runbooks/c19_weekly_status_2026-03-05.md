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

Canonical overnight wrapper:
- [nightly_orchestrator.py](../../v4/tools/nightly_orchestrator.py)
- [overnight_llt_validation.json](../../v4/tools/queues/overnight_llt_validation.json)

Overnight runtime contract:
- runtime outputs live under `bench_vault/night_runs/...`
- each run writes `status.json`, `summary.json`, `wake_trigger.json`
- per-job logs live under `jobs/<job_id>/`
- detached launch is supported through the orchestrator `launch` subcommand
- lane defaults now include restart budget and no-output watchdogs
- the overnight path has been smoke-tested in detached mode, not just dry-run mode

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

### D. Fresh-start vs sequential-carry on the same small real-data `LL` path

This was rerun directly on the canonical runner with the same small WikiText `LL` setup and the same sequential pointer. Only the surface changed:

- `small_wikitext_fresh`
  - [nightly_runner_small_wikitext_fresh_LL_10k_seqptr_20260306.json](../../v4/dev_notes/telemetry/nightly_runner_small_wikitext_fresh_LL_10k_seqptr_20260306.json)
- `wikitext_sequential_carry`
  - [nightly_runner_wikitext_sequential_carry_LL_10k_seqptr_20260306.json](../../v4/dev_notes/telemetry/nightly_runner_wikitext_sequential_carry_LL_10k_seqptr_20260306.json)

Observed result:
- fresh:
  - final acc `0.356`
  - best acc `0.578`
  - final BPC `3.303`
  - `ptr_unique_frac = 0.125`
- carry:
  - final acc `0.355`
  - best acc `0.632`
  - final BPC `3.318`
  - `ptr_unique_frac = 1.000`

Interpretation:
- removing the reset does exactly what it should geometrically: pointer coverage expands from a tiny front segment to the full ring;
- but on this exact small real-data `LL` setup the final quality barely changes;
- so the previous fresh-surface conclusions were surface-limited, but the reset itself was not hiding a large `LL` quality win here.

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

## 6A. Multi-Timescale Taps: First Deterministic Verdict

Nightly-only implementation:
- `LLT = LL + fixed lag taps`
- taps stay as separate channels
- no pooled topk averaging
- current lag set: `[1, 2, 4, 8, 16, 32]`

What was tested:
- `fast_memory_carry`, CPU, `10k`
  - `LL` vs `LLT`
- `wikitext_sequential_carry`, CPU, `10k`
  - `LL` vs `LLT`

Artifacts:
- mechanistic carry:
  - [nightly_runner_fast_memory_carry_LL_mtaps_baseline_20260306.json](../../v4/dev_notes/telemetry/nightly_runner_fast_memory_carry_LL_mtaps_baseline_20260306.json)
  - [nightly_runner_fast_memory_carry_LLT_10k_20260306.json](../../v4/dev_notes/telemetry/nightly_runner_fast_memory_carry_LLT_10k_20260306.json)
- real-data sequential carry:
  - [nightly_runner_wikitext_sequential_carry_LL_10k_seqptr_20260306.json](../../v4/dev_notes/telemetry/nightly_runner_wikitext_sequential_carry_LL_10k_seqptr_20260306.json)
  - [nightly_runner_wikitext_sequential_carry_LLT_10k_20260306.json](../../v4/dev_notes/telemetry/nightly_runner_wikitext_sequential_carry_LLT_10k_20260306.json)

Verdict:
- `fast_memory_carry`
  - `LL`: final `100%`, `358.2s`
  - `LLT`: final `100%`, `353.5s`
  - meaning: taps do not hurt the mechanistic carry surface, but they do not create a new ceiling there
- `wikitext_sequential_carry`
  - `LL`: final acc `0.3551`, BPC `3.3185`, `376.5s`
  - `LLT`: final acc `0.3686`, BPC `3.2487`, `414.4s`
  - delta: `+1.34 pt` final accuracy, `-0.0698` BPC, slower by about `10%`

Trace evidence:
- `LLT` taps were genuinely active, not decorative
- `tap_unique_frac = 1.0`
- `tap_center_dist_mean = 10.5`

Meaning:
- this is the first nightly branch after pooled topk that gives a clean carry-surface quality win
- the gain is not coming from global search
- it is coming from structured, non-pooled extra retrieval bandwidth

## 6B. `LLT7` Long Confirmation and Knee Check

The first `LLT` win was not enough on its own, so the branch was pushed through longer deterministic carry runs.

Long CPU confirmations on `wikitext_sequential_carry`:
- seed `43`
  - [nightly_runner_wikitext_sequential_carry_LLT7_20260307_044641_296480_summary.json](../../v4/dev_notes/telemetry/nightly_runner_wikitext_sequential_carry_LLT7_20260307_044641_296480_summary.json)
  - final acc `0.405`
  - final BPC `2.966`
- seed `44`
  - [nightly_runner_wikitext_sequential_carry_LLT7_20260307_051820_497573_summary.json](../../v4/dev_notes/telemetry/nightly_runner_wikitext_sequential_carry_LLT7_20260307_051820_497573_summary.json)
  - final acc `0.407`
  - best acc `0.703`
  - final BPC `2.987`
  - time `1088.1s`

Observed result:
- `LLT7` keeps the long-run gain stable across multiple seeds;
- the branch does not show late collapse on the canonical carry surface;
- the gain is not a single-seed artifact.

Non-aliased extra-lag knee check:
- [nightly_runner_wikitext_sequential_carry_LLT48_20260307_042510_607837_summary.json](../../v4/dev_notes/telemetry/nightly_runner_wikitext_sequential_carry_LLT48_20260307_042510_607837_summary.json)
- `LLT48 = [1,2,4,8,16,32,48]`
- final acc `0.373`
- final BPC `3.268`

Meaning:
- a meaningful extra long lag did not improve over `LLT7`;
- the current evidence points to `LLT7` being at or very near the useful tap knee on this `M=64` carry surface;
- the near-term task is no longer "add more taps blindly", but "confirm `LLT7` and scale it".

## 6C. CPU Pareto Needle-Poke: `seq=8,batch=8` vs `seq=16,batch=4`

Goal:
- start a CPU-smartness / CPU-speed frontier search without broad sweeps;
- keep runs sequential and budgeted to about `5` minutes each;
- compare one factor at a time, then stop and interpret.

Method:
- canonical surface: `wikitext_sequential_carry`
- canonical branch: `LLT7`
- equal token budget per step: `64 tokens/step`
- candidate A:
  - `seq=8`
  - `batch=8`
- candidate B:
  - `seq=16`
  - `batch=4`
- probe script:
  - [cpu_pareto_probe.py](../../v4/tests/cpu_pareto_probe.py)
- summary artifact:
  - [cpu_pareto_probe_seq_tradeoff_20260307_175928_639660.json](../../v4/dev_notes/telemetry/cpu_pareto_probe_seq_tradeoff_20260307_175928_639660.json)

Results:
- `A_seq8_b8`
  - target steps `7300`
  - final acc `0.3744`
  - final BPC `3.2167`
  - time `320.8s`
- `B_seq16_b4`
  - target steps `3700`
  - final acc `0.3438`
  - final BPC `3.3883`
  - time `309.4s`

Verdict:
- longer `seq` at the same token budget was worse on both quality metrics;
- this is not a tiny edge:
  - accuracy drops by `3.06 pt`
  - BPC worsens by `+0.1716`
- the extra in-sequence context did not pay for itself on this small CPU surface.

Meaning:
- the next CPU-smart frontier probe should not spend more budget on longer `seq`;
- on this surface, the better use of the same budget is still the shorter, denser update schedule;
- the next single-factor probe should target model capacity or read bandwidth instead of sequence length.

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

6. Overnight orchestration now uses an explicit runner heartbeat, not just stdout/stderr log mtimes.
- This was added after a false-positive `watchdog:no_output` failure on a long CPU scale-up run.
- Canonical overnight jobs now write a `heartbeat.json` sidecar under each job directory.
- The orchestrator watches that heartbeat first, and only falls back to log mtimes second.
- CPU lane env now explicitly pins thread counts in the queue JSON so unattended launches do not depend on the parent shell state.
- Windows launch mode now uses a separate orchestrator console instead of a fully detached stdio-redirected process.
- This was added after adversarial smoke tests showed that the fully detached Windows launch path could start child jobs that stayed idle with no heartbeat and no progress.

## Final Summary

Current best consolidated read:

- `dual-phi` remains the lead C19 form
- `C_init = pi`, `C` learnable
- small fresh real-data: `LL` wins
- carry mechanistic bench: `GL` is viable, so global retrieval is not dead in principle
- pooled topk is still not the near-term solution
- pointer interpolation is a local fix, not a late-game memory fix
- shortest-arc is not the current bottleneck
- **multi-timescale taps is now the active winning nightly branch**


