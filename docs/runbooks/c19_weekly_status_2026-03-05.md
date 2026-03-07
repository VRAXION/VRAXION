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

## 0. Active Correction: `S` Wiring

There is one active correction that must stay visible until the canonical runner/train path is cleaned up.

Current fact pattern:
- the canonical train path does **not** pass `S` into `model(...)`
- `forward(..., S=None, ...)` falls back to `S = 'dotprod'`
- the canonical nightly runner now passes `context_mode='dotprod'` **explicitly**
- therefore the YAML/config `S: 0.3` is currently **not** the active mode on the canonical train path, and old nightly evidence before the runner fix should be read as implicit `dotprod`

What this means:
- current canonical nightly/train evidence should be read as **implicit `dotprod` evidence**
- any old claim that a fixed scalar `S` from config was actively used is **not settled**
- relative branch comparisons that all used the same path remain valid
- explicit probes that pass `S` directly are still meaningful and should be marked separately

Practical rule:
- do **not** describe the canonical train/nightly results as "fixed `S=0.3`"
- do describe old nightly/train results as "implicit `dotprod`"
- do describe current canonical runner results as "explicit `dotprod`"
- treat fixed-`S` conclusions as pending re-validation

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

## 6D. CPU Pareto Needle-Poke: `hidden_dim=32` vs `hidden_dim=512`

With sequence length ruled out as the next frontier lever, the next direct question was whether the real limit was internal capacity.

Method:
- canonical surface: `wikitext_sequential_carry`
- canonical branch: `LLT7`
- fixed:
  - `seq=8`
  - `batch=8`
- time budget: about `5` minutes per run
- summary artifact:
  - [cpu_pareto_probe_seq_tradeoff_20260307_182020_815561.json](../../v4/dev_notes/telemetry/cpu_pareto_probe_seq_tradeoff_20260307_182020_815561.json)

Results:
- `A_h32`
  - `hidden_dim=32`
  - final acc `0.3744`
  - final BPC `3.2167`
  - time `348.2s`
- `B_h512`
  - `hidden_dim=512`
  - final acc `0.4731`
  - final BPC `2.7097`
  - time `296.3s`

Verdict:
- this is a large, clean win for more internal capacity on the same CPU budget;
- delta:
  - accuracy `+9.87 pt`
  - BPC `-0.5070`
- the current bottleneck at this stage was not sequence length;
- it was internal representation capacity.

Meaning:
- once `LLT7` fixed the retrieval structure, the next limiting factor became "how much model can actually use what it reads";
- on this small CPU frontier, more "brain" mattered far more than more in-sequence context.

## 6E. CPU Pareto Needle-Poke: `slot_dim=8` vs `slot_dim=16`

After the `hidden_dim=512` win, the next question became whether the memory/read path itself was now the bottleneck.

Method:
- canonical surface: `wikitext_sequential_carry`
- canonical branch: `LLT7`
- fixed:
  - `hidden_dim=512`
  - `seq=8`
  - `batch=8`
- compare:
  - `slot_dim=8`
  - `slot_dim=16`
- summary artifact:
  - [cpu_pareto_probe_seq_tradeoff_20260307_184037_121413.json](../../v4/dev_notes/telemetry/cpu_pareto_probe_seq_tradeoff_20260307_184037_121413.json)

Results:
- `A_slot8`
  - final acc `0.4313`
  - final BPC `2.8448`
  - time `329.8s`
- `B_slot16`
  - final acc `0.4532`
  - final BPC `2.7707`
  - time `275.3s`

Verdict:
- `slot_dim=16` is a clean win over `slot_dim=8`;
- delta:
  - accuracy `+2.19 pt`
  - BPC `-0.0741`
- this means the next bottleneck after `hidden_dim=512` was indeed read / memory bandwidth.

Meaning:
- retrieval structure alone was not enough;
- once there was enough hidden capacity, wider slot bandwidth also paid off;
- the frontier question then changed from "is more bandwidth helpful at all?" to "where is the knee?"

## 6F. Coarse Hidden Frontier on the Better Memory Width

With `slot_dim=16` already beating `8`, a coarse hidden sweep was used to stop guessing and bracket the hidden-capacity knee.

Method:
- canonical surface: `wikitext_sequential_carry`
- canonical branch: `LLT7`
- fixed:
  - `slot_dim=16`
  - `seq=8`
  - `batch=8`
- candidates:
  - `hidden_dim = 128, 256, 512, 768`
- time budget: about `5` minutes each
- summary artifact:
  - [cpu_pareto_probe_seq_tradeoff_20260307_191847_013563.json](../../v4/dev_notes/telemetry/cpu_pareto_probe_seq_tradeoff_20260307_191847_013563.json)

Results:
- `H128`
  - final acc `0.4480`
  - final BPC `2.8735`
  - time `376.7s`
- `H256`
  - final acc `0.4267`
  - final BPC `2.9007`
  - time `306.2s`
- `H512`
  - final acc `0.4414`
  - final BPC `2.8187`
  - time `284.5s`
- `H768`
  - final acc `0.4456`
  - final BPC `2.7947`
  - time `299.8s`

Verdict:
- the frontier is no longer monotonic in a naive way;
- `H256` did not beat `H128`;
- `H512` beat `H256` strongly on BPC;
- `H768` only improved modestly over `H512`:
  - accuracy `+0.43 pt`
  - BPC `-0.0240`

Meaning:
- the hidden-capacity knee now appears to be in the `512 -> 768` region, not down near `128/256`;
- `512` is still a very strong quality/time point;
- `768` may still be a viable frontier move, but it is already much closer to the knee than the earlier `32 -> 512` jump;
- the next deterministic question is no longer "is more hidden useful at all?" but rather whether `768` is enough, or whether the next real limit after that is ring capacity `M`.

## 6G. CPU Pareto Needle-Poke: `M=64` vs `M=128`

After the hidden and slot sweeps, the next frontier hypothesis was that ring capacity / horizon might be the next bottleneck.

Method:
- canonical surface: `wikitext_sequential_carry`
- canonical branch: `LLT7`
- fixed:
  - `hidden_dim=512`
  - `slot_dim=16`
  - `seq=8`
  - `batch=8`
- compare:
  - `M=64`
  - `M=128`
- time budget: about `5` minutes each
- summary artifact:
  - [cpu_pareto_probe_seq_tradeoff_20260307_195200_668300.json](../../v4/dev_notes/telemetry/cpu_pareto_probe_seq_tradeoff_20260307_195200_668300.json)

Results:
- `M64`
  - final acc `0.4599`
  - final BPC `2.7996`
  - time `415.1s`
- `M128`
  - final acc `0.4474`
  - final BPC `2.8154`
  - time `328.6s`

Verdict:
- `M=128` did not improve over `M=64`;
- it was slightly worse on both quality metrics:
  - accuracy `-1.25 pt`
  - BPC `+0.0158`
- on this canonical CPU carry surface, raw ring capacity is not the next limiting factor.

Meaning:
- the current frontier does not appear to be horizon-limited at `M=64`;
- the next useful branch should not blindly push `M` higher;
- the stronger remaining question is whether the post-`512` hidden knee is already at `768`, or whether the next gains must come from a better tap mixer / non-size architectural refinement rather than just scaling ring size.

## 6H. CPU Pareto Needle-Poke: `M=256` Early-Stop Check

Because the hidden sweep was non-monotonic, we explicitly checked one higher-capacity `M` point before closing the ring-capacity axis.

Method:
- canonical surface: `wikitext_sequential_carry`
- canonical branch: `LLT7`
- fixed:
  - `hidden_dim=512`
  - `slot_dim=16`
  - `seq=8`
  - `batch=8`
- compare:
  - current winner `M=64`
  - one higher-capacity probe `M=256`
- time budget: about `5` minutes
- summary artifact:
  - [cpu_pareto_probe_seq_tradeoff_20260307_201122_925029.json](../../v4/dev_notes/telemetry/cpu_pareto_probe_seq_tradeoff_20260307_201122_925029.json)

Results:
- `M64`
  - final acc `0.4599`
  - final BPC `2.7996`
- `M256`
  - final acc `0.4158`
  - final BPC `2.9942`
  - time `279.3s`

Verdict:
- `M=256` was substantially worse than `M=64`:
  - accuracy `-4.41 pt`
  - BPC `+0.1946`
- this is strong enough to early-stop the upward `M` sweep for now;
- on this canonical CPU carry surface, raw ring capacity is not the next bottleneck.

Meaning:
- the non-monotonic hidden sweep does not generalize into evidence that `M` should keep scaling upward;
- at least up to the next meaningful higher point, more ring capacity is not buying quality;
- the next frontier question should move away from `M` and toward either:
  - confirming the hidden knee (`512 -> 768`),
  - or probing tap-mixer / architectural expressivity.

## 6I. CPU Pareto Needle-Poke: `slot_dim=16` vs `32` vs `64` at `H=512`

After confirming that `slot_dim=16` beat `8`, and that raw ring size `M` was not the next limit, the next question was whether the memory/read bandwidth axis itself had a sweet spot.

Method:
- canonical surface: `wikitext_sequential_carry`
- canonical branch: `LLT7`
- fixed:
  - `hidden_dim=512`
  - `M=64`
  - `seq=8`
  - `batch=8`
- compare:
  - `slot_dim=16`
  - `slot_dim=32`
  - `slot_dim=64`
- time budget: about `5` minutes each
- summary artifact:
  - [cpu_pareto_probe_seq_tradeoff_20260307_202412_877309.json](../../v4/dev_notes/telemetry/cpu_pareto_probe_seq_tradeoff_20260307_202412_877309.json)

Results:
- `slot16`
  - final acc `0.4308`
  - final BPC `2.8747`
  - time `320.8s`
- `slot32`
  - final acc `0.4697`
  - final BPC `2.7091`
  - time `277.2s`
- `slot64`
  - final acc `0.4192`
  - final BPC `2.9376`
  - time `299.3s`

Verdict:
- the slot-width frontier is now clearly non-monotonic;
- `slot32` is the current sweet spot on this canonical CPU carry surface;
- `slot64` is worse than both `slot16` and `slot32`, so the next bottleneck is not "keep widening slots indefinitely";
- relative to `slot16`, `slot32` improved:
  - accuracy `+3.89 pt`
  - BPC `-0.1656`
  - while also running faster in this budgeted probe

Meaning:
- memory/read bandwidth was still a real bottleneck beyond `slot16`, but it has now hit an internal optimum;
- the next frontier move should not be more raw `slot_dim`;
- the remaining likely bottleneck is now the tap mixer / architectural expressivity rather than another simple size axis.

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


