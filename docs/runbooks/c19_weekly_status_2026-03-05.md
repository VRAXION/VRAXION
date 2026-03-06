# C19 Weekly Status — 2026-03-05

This note records the current state of the standalone C19 investigation that ran alongside the main nightly line this week.

Important:
- This is a research status note, not a merged runtime change.
- Most of the work happened in standalone A/B and sweep scripts outside the current nightly codepath.
- The current goal is to decide which C19 asymmetry pattern deserves promotion into the next integration pass.

## Scope

This week focused on four questions:

1. Is symmetric C19 too weak or too ambiguous in recurrent use?
2. Does negative-side asymmetry help?
3. Is full dual-phi better than simple `neg*phi only`?
4. Are `C` regularization and tail-limit changes worth carrying forward?

## What Was Tested

- C19 cost / profiling:
  - activation bottleneck profiling
  - cheaper alternatives and inference timing
- Standalone asymmetry sweeps:
  - `original`
  - `asym neg=1.5`
  - `asym neg=phi`
  - `asym neg=2.0`
  - `asym neg=phi^2`
  - `asym neg=3.0`
- Standalone dual-family tests:
  - `neg*phi only`
  - `dual-phi`
  - `dual-phi-inv`
  - `dual-sqrt-phi`
  - `dual-phi-same`
- Learnable-`C` checks:
  - causal swap test
  - regularization A/B
  - flat vs phi-structured regularization
  - tail-limit sweeps
- Real-data GPU A/B:
  - baseline vs `dual-phi`
  - `neg-phi` vs `dual-phi`
- Real-data GPU telemetry pilot:
  - `dual-phi` vs `dual-phi-envelope(alpha)`
  - `alpha = 0.02, 0.05, 0.10`
  - tail-hit and `|x|/C` quantile logging
- Fixed-`C` core-geometry pilot:
  - `C = phi^2, pi, 2pi`
  - `tail = linear` vs `tail = periodic`
  - deterministic single-seed WikiText comparison
- Fixed-`C` upper-sweep follow-up:
  - `C = 2.8, 3.0, pi, 3.3, 3.5`
  - `C = 3.5, 3.8, 4.2, 4.8, 5.5, 2pi`
  - deterministic single-seed WikiText comparison
- Learnable-`C` synth probe:
  - `rho` frozen
  - `C` left learnable
  - task-wise telemetry on `count1`, `alternate2`, `echo8`

## What Looks Confirmed

### 1) Plain symmetric C19 is not the best learner

Short local CPU A/B on the echo task showed the symmetric baseline lagging the asymmetric variants by a large margin over 100 steps.

Observed quick-run pattern:
- `original (1.0)`: 31.5% best acc
- `asym neg=phi`: 56.8% best acc
- `asym neg=2.0`: 63.0% best acc
- `asym neg=phi^2`: 69.0% best acc

Interpretation:
- breaking symmetry helps;
- the model appears to benefit from "knowing which side it is on";
- stronger asymmetry increases gradient volatility.

### 2) The sign matters: stronger negative side is useful, stronger positive side is dangerous

Local dual-family A/B on the echo task showed:

- `neg*phi only`: 56.8% best acc, stable enough
- `dual-phi`: 30.7% best acc, extremely stable but too damped in this short regime
- `dual-phi-inv` (`neg*1/phi`, `pos*phi`): catastrophic instability
- `dual-phi-same` (`neg*phi`, `pos*phi`): catastrophic instability

Interpretation:
- it is not enough to say "some phi asymmetry";
- the useful direction is "negative side gets the stronger gain";
- making the positive side stronger is currently treated as a bad direction.

### 3) Dual-phi is the current standalone winner in longer real-data runs

The decisive 500-step WikiText GPU A/B was reproduced locally this week using the same real-data script family as the devlog note in [v4/dev_notes/ab_c19_dualphi_results_2026-03-05.md](../../v4/dev_notes/ab_c19_dualphi_results_2026-03-05.md).

Observed head-to-head result:

| Metric | Neg-Phi | Dual-Phi | Delta |
|---|---:|---:|---:|
| Final Acc | 51.7% | 53.4% | +1.65% |
| Best Acc | 54.0% | 55.9% | +1.9% |
| Final Loss | 1.6979 | 1.6264 | -4.2% |
| BPC | 2.450 | 2.346 | -0.104 |
| Wall Time | 1175s | 1128s | -47s |
| Max GradNorm | 7.4 | 3.5 | 2x lower |
| Grad Spikes | 0 | 0 | both stable |

Key read:
- `dual-phi` beat the old symmetric baseline in the prior A/B;
- `dual-phi` also beat `neg-phi only` in the direct same-config comparison;
- `dual-phi` took the lead around step 14 and kept it;
- the shorter CPU echo regime understated `dual-phi`, but the longer real-data run paid back its extra damping.

### 4) Learnable `C` looks real

The causal `C` swap test passed in prior checking, which is evidence that `C` carries task-specific signal rather than acting as pure noise.

Current read:
- learnable `C` is not yet fully production-qualified;
- but it is no longer reasonable to treat it as decorative.

### 5) Light outer-loop damping is inert in the current regime

A local 200-step WikiText telemetry pilot compared plain `dual-phi` against `dual-phi-envelope(alpha)` with `alpha = 0.02, 0.05, 0.10`.

Observed pattern:
- all four variants finished at essentially the same place (`45.5%` final acc);
- `tail_hit = 0.0000%` for every variant;
- `p99 |x|/C = 1.05`;
- `max |x|/C ~= 2.20-2.21`, still far below the `6C` tail boundary.

Interpretation:
- the model is operating deep inside the periodic core;
- light damping of farther arches does not meaningfully change the active regime;
- this makes "more loops before tail" and "small soft envelope before tail" low-priority ideas for the current task.

### 6) `pi` remains the best default `C` initialization among the tested fixed scales

A fixed-`C` core-geometry pilot compared `C = phi^2`, `pi`, and `2pi`, with both standard linear tail and pure periodic/no-tail mode.

Observed pattern:
- `phi^2`: `34.8%` final acc, `p99 |x|/C = 1.31`
- `pi`: `35.4%` final acc, `p99 |x|/C = 1.04`
- `2pi`: `35.0%` final acc, `p99 |x|/C = 0.57`
- linear tail and periodic/no-tail were identical at all three `C` values

Interpretation:
- smaller `C` makes the geometry too dense and hurts;
- larger `C` makes the geometry too loose and slightly underuses the internal structure;
- `pi` is currently the best-tested compromise for the core geometry;
- this supports keeping `C_init = pi` as the default starting point.

Follow-up upper-sweep:
- `2.8`: `35.0%`
- `3.0`: `35.3%`
- `pi`: `35.4%`
- `3.3`: `35.4%`
- `3.5`: `35.5%`
- `3.8`: `35.6%`
- `4.2`: `35.7%`
- `4.8`: `35.4%`
- `5.5`: `35.2%`
- `2pi`: `35.0%`

Read:
- the fixed-`C` curve currently looks like a broad, smooth hump rather than a phi-like spiky resonance pattern;
- the best 100-step single-seed point so far is `C = 4.2`;
- the improvement over `pi` is tiny, so it is not enough evidence to change the default init;
- the safer current interpretation is "wide safe band, likely peaking somewhere around `4.0-4.4`", not "special irrational magic value".

### 7) Learnable `C` adapts in a task-dependent way

A synthetic `learnable C` probe was run with `rho` frozen and `bitlift` input active so both input-side and hidden-side `C` remained trainable.

Observed pattern:
- `count1`:
  - `C_in: +0.101`
  - `C_h: +0.069`
- `alternate2`:
  - `C_in: -0.017`
  - `C_h: +0.061`
- `echo8`:
  - `C_in: +0.001`
  - `C_h: -0.030`
- `tail_hit = 0%` on all three tasks

Interpretation:
- `C` is not decorative;
- the model does move `C`, and not always in the same direction;
- input-side and hidden-side `C` can diverge by task;
- the current best read is to keep `C` learnable, but initialize it at `pi`.

## What Is Still Open

### 1) The dual-phi verdict is strong, but still based on a narrow validation slice

What is already true:
- `dual-phi > symmetric baseline`;
- `dual-phi > neg-phi only`;
- `dual-phi` does this while staying at least as stable.

What is still missing before a mainline promotion:
- more than one seed;
- at least one longer or sequential run;
- confirmation that the same gain holds after integration into the active model path.

### 2) Learnable `C` regularization is still under research

The current best read is that simple flat regularization around `lambda ~= 1e-4` is the strongest practical candidate.

Still open:
- whether phi-structured regularization has any real edge;
- how much regularization survives after moving from toy tests to production training;
- whether the task-dependent `C` drift seen in synth probes survives longer real-data runs in the same direction.

### 3) Tail-limit changes are not yet compelling

The current read is:
- there is no strong evidence yet that the current `6C` tail boundary needs to change;
- new telemetry says the active distribution sits far below the tail (`p99 |x|/C ~= 1.05`, `max ~= 2.21`);
- light outer-loop damping also failed to change behavior in this regime;
- fixed-`C` comparison also found no measurable gap between linear-tail and pure-periodic variants at tested scales;
- tail-limit work remains lower priority than asymmetry and `C` regularization.

## Current Best Read

If we had to summarize the week in one sentence:

> C19 wants asymmetry, and the best asymmetry found so far is dual-phi: stronger negative arches, weaker positive arches.

Practical version:
- symmetric baseline is too weak;
- `neg*phi only` helped expose the direction of the effect;
- `dual-phi` is now the current lead standalone variant, not just the prettier hypothesis;
- the sign of the asymmetry matters more than the raw amount of scaling;
- `pi` remains the best-tested default `C` init;
- the fixed-`C` surface looks smooth, not strongly resonant;
- `C` itself should remain learnable.

## Performance Investigation

### Batch 0 — Proxy-step baseline and closed C19 micro-opt check

Purpose:
- establish a deterministic runtime baseline for the current short WikiText proxy;
- close the current exact dual-phi activation micro-opt branch before touching the write path.

Proxy-step baseline (`seed=42`, `batch=32`, `seq=256`, `C=pi`, `N=1`, `R=1`, `write_mode=replace`):
- `forward_loss ~= 2.14s`
- `backward ~= 1.55s`
- main logical hotspots:
  - `_c19_activation`
  - `func_hdd_write_tns`
  - `func_softread_tns`

Artifacts:
- baseline JSON: [profile_sweep_step_wikitext_20260306_112140.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_112140.json)
- baseline op table: [profile_sweep_step_wikitext_20260306_112140_ops.txt](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_112140_ops.txt)
- activation microbench: [bench_c19_dualphi_optimize_20260306_1130.txt](../../v4/dev_notes/telemetry/bench_c19_dualphi_optimize_20260306_1130.txt)
- rejected activation variant JSON: [profile_sweep_step_wikitext_20260306_112321.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_112321.json)
- rejected activation op table: [profile_sweep_step_wikitext_20260306_112321_ops.txt](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_112321_ops.txt)

Closed finding:
- isolated dual-phi activation microbench (`gain_v2`) looked better in forward-only timing;
- the same variant regressed on the full proxy step because backward became significantly slower;
- verdict: do not promote the current exact `gain_v2` activation rewrite.

Next action:
- move to deterministic write-path A/B (`func_hdd_write_tns`) before opening any new activation rewrites.

### Batch 1 — HDD write microbench

Purpose:
- measure whether the current replace-write path has cheap implementation headroom before touching the full proxy step.

Scripts used:
- `v4/tests/bench_hdd_write_optimize.py`

Variants:
- `current`
- `lerp_v2 = torch.lerp(current, write_vec, w)`
- `delta_v3 = current + w * (write_vec - current)`

Artifact:
- microbench JSON: [bench_hdd_write_optimize_20260306_113735.json](../../v4/dev_notes/telemetry/bench_hdd_write_optimize_20260306_113735.json)
- microbench text: [bench_hdd_write_optimize_20260306_113735.txt](../../v4/dev_notes/telemetry/bench_hdd_write_optimize_20260306_113735.txt)

Result:
- both rewrites matched the current implementation within tolerance (`max diff <= 4.768e-07`);
- `lerp_v2` improved forward by `1.705x` and backward by `1.201x`;
- `delta_v3` improved forward by `1.431x` and backward by `1.255x`.

Verdict:
- both rewrites clear the Batch 1 promotion threshold;
- carry only `delta_v3` into Batch 2 because it had the best combined forward+backward write time.

Next action:
- run deterministic proxy-step validation for `current` vs `delta_v3`.

### Batch 2 — Proxy-step validation for `delta_v3`

Purpose:
- verify whether the microbench winner survives the actual short WikiText proxy step.

Scripts used:
- `v4/tests/profile_sweep_step_wikitext.py --impl current --write-impl current`
- `v4/tests/profile_sweep_step_wikitext.py --impl current --write-impl delta_v3`

Artifacts:
- baseline JSON: [profile_sweep_step_wikitext_20260306_113911.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_113911.json)
- baseline op table: [profile_sweep_step_wikitext_20260306_113911_ops.txt](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_113911_ops.txt)
- baseline breakdown chart: [profile_sweep_step_wikitext_20260306_113911_breakdown.png](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_113911_breakdown.png)
- candidate JSON: [profile_sweep_step_wikitext_20260306_114037.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_114037.json)
- candidate op table: [profile_sweep_step_wikitext_20260306_114037_ops.txt](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_114037_ops.txt)

Observed proxy-step timings:
- `current`: `forward ~= 2.096s`, `backward ~= 1.345s`
- `delta_v3`: `forward ~= 2.185s`, `backward ~= 1.353s`

Verdict:
- the microbench winner did not survive the real proxy step;
- total proxy-step time regressed slightly instead of improving by the required `>= 5%`;
- do not promote the `delta_v3` write rewrite.

Next action:
- move to Batch 3 and target op-level overhead (`aten::empty`, `aten::copy_`, `aten::fill_`, scatter/index-related ops) using the existing profiler tables.

### Batch 3 — Next op-level target after write rewrite failure

Purpose:
- choose the next optimization target from the existing profiler evidence instead of opening another algebra rewrite branch.

Evidence source:
- [profile_sweep_step_wikitext_20260306_113911_ops.txt](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_113911_ops.txt)
- [profile_sweep_step_wikitext_20260306_114037_ops.txt](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_114037_ops.txt)

Read:
- the failed write rewrite did not change the overall op picture;
- the dominant non-model-shell costs remain allocation/copy/scatter-heavy:
  - `aten::empty`
  - `aten::empty_strided`
  - `aten::resize_`
  - `aten::copy_`
  - `aten::_to_copy`
  - `aten::scatter_add_`
  - `aten::_index_put_impl_`
  - `aten::scatter_`
- this points to temporary tensor churn and scatter/index traffic, not a missing activation algebra trick.

Verdict:
- close the current activation-rewrite branch;
- close the current write-algebra rewrite branch;
- the next performance round should target allocation/copy/scatter reduction on the same proxy path.

Next action:
- start from intermediate-tensor reduction and write-path memory traffic, not from new C19 formulas or expert-vectorization.

### Batch 4 — Source-map baseline for the proxy step

Purpose:
- replace guesswork with source-level attribution inside the actual proxy forward path;
- identify which exact block owns the bulk of the scoped CUDA time before picking a new low-risk optimization.

Scripts used:
- `v4/tests/profile_sweep_step_wikitext.py --impl current --write-impl current --source-map`
- `v4/tests/plot_profile_breakdown.py --json ...121153.json --ops ...121153_ops.txt`

Artifacts:
- source-map JSON: [profile_sweep_step_wikitext_20260306_121153.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_121153.json)
- source-map op table: [profile_sweep_step_wikitext_20260306_121153_ops.txt](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_121153_ops.txt)
- source scope totals: [profile_sweep_step_wikitext_20260306_121153_scopes.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_121153_scopes.json)
- source scope op attribution: [profile_sweep_step_wikitext_20260306_121153_scope_ops.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_121153_scope_ops.json)
- updated breakdown chart: [profile_sweep_step_wikitext_20260306_121153_breakdown.png](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_121153_breakdown.png)

Observed scope totals:
- `write_replace`: `1222.5ms` scoped CUDA (`62.4%`)
- `write_prepare`: `261.8ms` (`13.4%`)
- `output_head`: `187.8ms` (`9.6%`)
- `window_prepare`: `141.4ms` (`7.2%`)
- `softread`: `92.9ms` (`4.7%`)
- `pointer_update`: `53.0ms` (`2.7%`)
- `state_init`: `0.5ms` (`~0.0%`)

Key read:
- the dominant scoped cost is not `state_init` and not `pointer_update`;
- the live hotspot is the current `write_replace` path around `func_hdd_write_tns`;
- `_index_put_impl_`, `copy_`, `fill_`, and `empty` all show their largest scoped share under `write_replace`;
- `source_map_complete` is still `false` because some global top ops are outside the forward source scopes (`random_`, backward-heavy ops like `scatter_add_`, and other non-model-shell overhead).

Verdict:
- close the earlier `state_init` and `pointer_update` hypotheses for this proxy config;
- promote the safest semantically equivalent replace-write candidate into full proxy validation next.

Next action:
- run `current` vs `lerp_v2` on the same deterministic proxy config and accept only if total proxy-step time improves by at least `5%`.

### Batch 5 — Proxy-step validation for `lerp_v2` after source-map attribution

Purpose:
- test the safest semantically equivalent replace-write implementation against the clean, non-source-map proxy path;
- make sure the validation is not polluted by the disabled source-map scaffolding itself.

Setup note:
- the first disabled source-map implementation allocated a fresh no-op context on every scope entry and measurably distorted the plain proxy path;
- validation was therefore rerun only after switching the off-path to a singleton no-op scope.

Scripts used:
- `v4/tests/profile_sweep_step_wikitext.py --impl current --write-impl current`
- `v4/tests/profile_sweep_step_wikitext.py --impl current --write-impl lerp_v2`

Artifacts:
- clean baseline JSON: [profile_sweep_step_wikitext_20260306_122326.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_122326.json)
- clean baseline op table: [profile_sweep_step_wikitext_20260306_122326_ops.txt](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_122326_ops.txt)
- `lerp_v2` JSON: [profile_sweep_step_wikitext_20260306_122458.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_122458.json)
- `lerp_v2` op table: [profile_sweep_step_wikitext_20260306_122458_ops.txt](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_122458_ops.txt)

Observed proxy-step timings:
- `current`: `forward ~= 2.067s`, `backward ~= 1.403s`, total `~= 3.470s`
- `lerp_v2`: `forward ~= 2.312s`, `backward ~= 1.576s`, total `~= 3.888s`

Observed helper timings:
- `func_hdd_write_tns`
  - `current ~= 315.3ms`
  - `lerp_v2 ~= 368.0ms`
- `_c19_activation`
  - `current ~= 878.1ms`
  - `lerp_v2 ~= 955.5ms`

Key read:
- the apparent earlier `lerp_v2` win vanished once the disabled source-map overhead was removed;
- on the clean proxy path, `lerp_v2` is slower in both forward and backward;
- native `torch.lerp` also needed an autocast-safe float-cast fallback to run in this path at all, which further weakens it as a production-quality candidate.

Verdict:
- reject `lerp_v2` for this proxy;
- the current write algebra still wins on the real proxy step even though microbench and scoped attribution made the write path look promising.

Next action:
- close write-algebra rewrites for this round and move to tensor-churn reduction inside the current `write_replace` path (`copy_`, `fill_`, `empty`, `_index_put_impl_`, `scatter_`), not new formulas.

### Batch 6 — Nightly-only `proxy_overlay` fast path implementation and correctness validation

Purpose:
- test a narrow `nightly`-only fast path for the exact proxy regime instead of another algebra rewrite;
- avoid cloning the full ring on every replace write while preserving immediate read-after-write visibility.

Implementation scope:
- new runtime switch: `replace_impl = dense | proxy_overlay`
- proxy overlay is gated behind the exact proxy regime only:
  - `N = 1`
  - `R = 1`
  - `write_mode = replace`
  - `kernel_mode = vshape`
  - `pointer_mode = sequential`
  - `bb_enabled = false`
  - `io_split_mode = off`
  - `checkpoint_chunks = 0`
- implementation style: dense base ring + small contiguous overlay segment with periodic flushes
- this is `nightly`-only and not for `main` yet

Scripts used:
- `v4/tests/validate_replace_overlay_proxy.py`

Artifacts:
- validation JSON: [validate_replace_overlay_proxy_20260306_125335.json](../../v4/dev_notes/telemetry/validate_replace_overlay_proxy_20260306_125335.json)

Observed correctness:
- fixed-batch fp32 loss diff: `0.00000000`
- fixed-batch logit max abs diff: `0.00000000`
- 16-step chunk logit max abs diff: `0.00000000`
- 16-step final ring max abs diff: `0.00000000`
- 16-step final ptr max abs diff: `0.00000000`
- 16-step final hidden max abs diff: `0.00000000`

Verdict:
- the proxy overlay path is numerically exact on the current validation harness;
- correctness is not the blocker for promotion.

Next action:
- validate the overlay path on the clean deterministic proxy benchmark and keep it only if total step time improves by at least `5%`.

### Batch 7 — Proxy-step and source-map validation for `replace_impl=proxy_overlay`

Purpose:
- determine whether the nightly proxy overlay is worth keeping after full proxy-step timing, not just correctness.

Scripts used:
- `v4/tests/profile_sweep_step_wikitext.py --impl current --write-impl current --replace-impl dense`
- `v4/tests/profile_sweep_step_wikitext.py --impl current --write-impl current --replace-impl proxy_overlay`
- `v4/tests/profile_sweep_step_wikitext.py --impl current --write-impl current --replace-impl dense --source-map`
- `v4/tests/profile_sweep_step_wikitext.py --impl current --write-impl current --replace-impl proxy_overlay --source-map`
- `v4/tests/plot_profile_breakdown.py --json ...125954.json --ops ...125954_ops.txt`
- `v4/tests/plot_profile_breakdown.py --json ...125712.json --ops ...125712_ops.txt`

Artifacts:
- dense plain JSON: [profile_sweep_step_wikitext_20260306_125359.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125359.json)
- dense plain ops: [profile_sweep_step_wikitext_20260306_125359_ops.txt](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125359_ops.txt)
- overlay plain JSON: [profile_sweep_step_wikitext_20260306_125530.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125530.json)
- overlay plain ops: [profile_sweep_step_wikitext_20260306_125530_ops.txt](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125530_ops.txt)
- dense source-map JSON: [profile_sweep_step_wikitext_20260306_125954.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125954.json)
- dense source-map ops: [profile_sweep_step_wikitext_20260306_125954_ops.txt](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125954_ops.txt)
- dense source scopes: [profile_sweep_step_wikitext_20260306_125954_scopes.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125954_scopes.json)
- dense source scope ops: [profile_sweep_step_wikitext_20260306_125954_scope_ops.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125954_scope_ops.json)
- dense source-map chart: [profile_sweep_step_wikitext_20260306_125954_breakdown.png](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125954_breakdown.png)
- overlay source-map JSON: [profile_sweep_step_wikitext_20260306_125712.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125712.json)
- overlay source-map ops: [profile_sweep_step_wikitext_20260306_125712_ops.txt](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125712_ops.txt)
- overlay source scopes: [profile_sweep_step_wikitext_20260306_125712_scopes.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125712_scopes.json)
- overlay source scope ops: [profile_sweep_step_wikitext_20260306_125712_scope_ops.json](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125712_scope_ops.json)
- overlay source-map chart: [profile_sweep_step_wikitext_20260306_125712_breakdown.png](../../v4/dev_notes/telemetry/profile_sweep_step_wikitext_20260306_125712_breakdown.png)

Observed plain proxy timings:
- `dense`: `forward ~= 2.269s`, `backward ~= 1.352s`, total `~= 3.624s`
- `proxy_overlay`: `forward ~= 1.917s`, `backward ~= 1.600s`, total `~= 3.519s`
- total-step improvement: `~2.9%`

Observed source-scope shift:
- dense `write_replace`: `1292.5ms` scoped CUDA (`60.5%`)
- overlay `write_replace`: `251.6ms` scoped CUDA (`19.3%`)
- overlay now shifts more scoped time into:
  - `softread`: `362.7ms` (`27.9%`)
  - `write_prepare`: `292.4ms` (`22.5%`)
  - `output_head`: `214.1ms` (`16.5%`)
- `source_map_complete` remains `false` in both runs because some high-cost ops still live outside the forward source scopes.

Key read:
- the overlay path does exactly what it was meant to do for the scoped hotspot: `write_replace` drops sharply;
- the saved write time is partly paid back in extra `softread` / overlay merge work and backward cost;
- the full proxy step does improve, but not enough to clear the acceptance threshold.

Verdict:
- reject `replace_impl=proxy_overlay` for promotion in this round;
- keep it as a documented nightly experiment only;
- do not merge it into `main` and do not make it the default nightly replace path.

Next action:
- keep the conclusion from Batch 4/5: the next worthwhile performance work is tensor-churn reduction inside the current write path and surrounding read/write prep (`copy_`, `empty`, `fill_`, `as_strided`, `_index_put_impl_`, `scatter_`), not more replace-formula variants.

### Batch 8 — `kernel_mode=vshape` vs `kernel_mode=topk(K=8)` on the current nightly proxy

Purpose:
- test whether the existing global content-read branch is worth revisiting on the current architecture;
- compare the current local pointer-window read (`vshape`) against the existing `topk` global-read path with the same C19, write mode, and proxy config.

Setup:
- same fixed-C dual-phi activation in both variants
- `C = pi`
- `tail_mode = linear`
- `replace_impl = dense`
- `topk_K = 8`
- current `topk` semantics are hybrid:
  - read is global content-based topK over the whole ring
  - write still uses the pointer-centered vshape window

Quality script used:
- `v4/tests/sweep_c19_core_geometry_wikitext.py --steps 60 --batch 16 --seq 256 --seed 42 --c-values pi --tail-modes linear --kernel-modes vshape,topk --topk-k 8 --replace-impl dense`

Quality artifact:
- [sweep_c19_core_geometry_wikitext_20260306_135900.json](../../v4/dev_notes/telemetry/sweep_c19_core_geometry_wikitext_20260306_135900.json)

Observed short-train quality:
- `vshape`
  - final acc `0.315`
  - best acc `0.393`
  - final loss `2.5777`
  - wall time `269s`
- `topk(K=8)`
  - final acc `0.307`
  - best acc `0.370`
  - final loss `2.6022`
  - wall time `286s`

Perf scripts used:
- `v4/tests/profile_sweep_step_wikitext.py --impl current --write-impl current --replace-impl dense --kernel-mode vshape --topk-k 8`
- `v4/tests/profile_sweep_step_wikitext.py --impl current --write-impl current --replace-impl dense --kernel-mode topk --topk-k 8`

Perf artifacts:
- vshape JSON: [profile_kernel_vshape_20260306.json](../../v4/dev_notes/telemetry/profile_kernel_vshape_20260306.json)
- vshape ops: [profile_kernel_vshape_20260306_ops.txt](../../v4/dev_notes/telemetry/profile_kernel_vshape_20260306_ops.txt)
- topk JSON: [profile_kernel_topk_20260306.json](../../v4/dev_notes/telemetry/profile_kernel_topk_20260306.json)
- topk ops: [profile_kernel_topk_20260306_ops.txt](../../v4/dev_notes/telemetry/profile_kernel_topk_20260306_ops.txt)

Observed one-step perf:
- `vshape`: `forward ~= 2.643s`, `backward ~= 1.902s`, total `~= 4.547s`
- `topk(K=8)`: `forward ~= 2.346s`, `backward ~= 2.102s`, total `~= 4.449s`

Read:
- the quality result is the important one here: the current global `topk` read did not beat the local `vshape` baseline on this architecture;
- perf is mixed:
  - one-step proxy timing gives `topk` a small total-step edge,
  - but the 60-step short-train wall time is still worse for `topk`;
- this is not strong enough evidence to reopen global-read work as the next priority.

Verdict:
- do not promote `kernel_mode=topk` as the next refinement path right now;
- keep `vshape` as the working baseline;
- only revisit global read if a later task shows a clear quality ceiling that locality cannot reach.

Next action:
- continue with deterministic ring-path minmaxing around the local read/write pipeline (`window_prepare`, `softread`, rolling local cache, tensor churn), not global topK search.

### Batch 9 — `topk_K=1` sanity check against the current `topk_K=8` result

Purpose:
- test the simplest reduction of the current global-read branch:
  - same `kernel_mode=topk`
  - same current architecture
  - only reduce `topk_K` from `8` to `1`
- check whether the old intuition ("smaller K might keep the benefit but get cheaper") still holds here.

Quality script used:
- `v4/tests/sweep_c19_core_geometry_wikitext.py --steps 60 --batch 16 --seq 256 --seed 42 --c-values pi --tail-modes linear --kernel-modes topk --topk-k 1 --replace-impl dense`

Quality artifact:
- [sweep_kernel_topk1_20260306.json](../../v4/dev_notes/telemetry/sweep_kernel_topk1_20260306.json)

Observed short-train quality:
- `topk_K=8`
  - final acc `0.307`
  - best acc `0.370`
  - final loss `2.6022`
  - wall time `286s`
- `topk_K=1`
  - final acc `0.298`
  - best acc `0.361`
  - final loss `2.6358`
  - wall time `290s`

Perf script used:
- `v4/tests/profile_sweep_step_wikitext.py --impl current --write-impl current --replace-impl dense --kernel-mode topk --topk-k 1`

Perf artifacts:
- [profile_kernel_topk1_20260306.json](../../v4/dev_notes/telemetry/profile_kernel_topk1_20260306.json)
- [profile_kernel_topk1_20260306_ops.txt](../../v4/dev_notes/telemetry/profile_kernel_topk1_20260306_ops.txt)

Observed one-step perf:
- `topk_K=8`: total `~= 4.449s`
- `topk_K=1`: total `~= 5.099s`

Read:
- on this architecture, shrinking `K` from `8` to `1` did not preserve quality;
- it also did not buy a clean perf win in the one-step proxy;
- that fits the current code path: the branch still scores the whole ring before taking topK, so reducing `K` does not remove the main global-search cost.

Verdict:
- reject `topk_K=1` as a promising cheap substitute for the current `topk_K=8` branch;
- do not reopen global topK search as the next minmax target.

Next action:
- stay on the local pointer-window path and optimize the deterministic local read/write pipeline instead of global topK variants.

### Batch 10 — `topk_K=2` check after the failed `K=1` sanity run

Purpose:
- test whether the `K=1` failure was just too aggressive, or whether a smaller global topK can recover the `K=8` quality while staying cheaper.

Quality script used:
- `v4/tests/sweep_c19_core_geometry_wikitext.py --steps 60 --batch 16 --seq 256 --seed 42 --c-values pi --tail-modes linear --kernel-modes topk --topk-k 2 --replace-impl dense`

Quality artifact:
- [sweep_kernel_topk2_20260306.json](../../v4/dev_notes/telemetry/sweep_kernel_topk2_20260306.json)

Observed short-train quality:
- `topk_K=8`
  - final acc `0.307`
  - best acc `0.370`
  - final loss `2.6022`
  - wall time `286s`
- `topk_K=2`
  - final acc `0.307`
  - best acc `0.372`
  - final loss `2.6030`
  - wall time `260s`
- `topk_K=1`
  - final acc `0.298`
  - best acc `0.361`
  - final loss `2.6358`
  - wall time `290s`

Perf script used:
- `v4/tests/profile_sweep_step_wikitext.py --impl current --write-impl current --replace-impl dense --kernel-mode topk --topk-k 2`

Perf artifacts:
- [profile_kernel_topk2_20260306.json](../../v4/dev_notes/telemetry/profile_kernel_topk2_20260306.json)
- [profile_kernel_topk2_20260306_ops.txt](../../v4/dev_notes/telemetry/profile_kernel_topk2_20260306_ops.txt)

Observed one-step perf:
- `topk_K=8`: total `~= 4.449s`
- `topk_K=2`: total `~= 3.948s`

Read:
- `K=2` is materially better than `K=1`;
- on this short nightly proxy, `K=2` essentially matches the current `K=8` quality;
- it also beats `K=8` on both short-run wall time and one-step proxy time.

Verdict:
- if the global topK branch is revisited later, `topk_K=2` is the only currently defensible value;
- but it still does not beat the local `vshape` baseline on quality, so this does not reopen global topK as the mainline refinement path.

Next action:
- keep `vshape` as the baseline;
- if a future hybrid/global retrieval experiment is opened, use `topk_K=2` as the global-read ceiling candidate, not `K=1` or `K=8`.

## Planned Next Tests

The next tests should be about confidence, not rediscovery:

- repeat the `neg-phi` vs `dual-phi` WikiText A/B across more seeds;
- run at least one longer or sequential validation;
- carry the winning activation into the active model path and confirm the gain survives integration;
- if fixed-`C` tuning is revisited, do a narrow `3.8-4.6` multi-seed sweep to confirm whether `4.2` is real or just noise;
- continue `C` regularization work only after the activation verdict is stable;
- if tail work is revisited, do it with a forced-tail stress task or much stronger envelope, not with more light damping;
- rerun the learnable-`C` telemetry on longer synth or mixed-data tasks to see whether the early task-specific drift persists.

## Promotion Guidance

Safe to say now:
- asymmetry is real;
- negative-side amplification is meaningful;
- positive-side amplification is dangerous;
- `dual-phi` is the current standalone winner and belongs in nightly-level experiment notes.
- light outer-loop damping does not buy anything in the current WikiText regime.
- keep `C_init = pi` as the default.
- keep `C` learnable.
- treat the current `4.2` fixed-`C` win as suggestive, not promotion-grade.

Not safe to say yet:
- that `dual-phi` should already replace the active mainline C19 in production;
- that tail-limit changes should be merged;
- that phi-structured regularization has beaten the simpler flat regularization.
