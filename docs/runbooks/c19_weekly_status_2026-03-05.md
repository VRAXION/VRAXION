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

### Batch 11 — large-step closure check for read-only `topk_K=2`

Purpose:
- run the current best global-read candidate (`kernel_mode=topk`, `topk_K=2`) at a larger deterministic horizon;
- add nightly-only topK read telemetry to verify whether the candidate is truly using non-local slots;
- close the read-only global topK branch early if it is still clearly worse than the local `vshape` baseline.

Quality script used:
- `v4/tests/sweep_c19_core_geometry_wikitext.py --steps 200 --batch 16 --seq 256 --seed 42 --c-values 3.141592653589793 --tail-modes linear --kernel-modes vshape,topk --topk-k 2 --replace-impl dense --topk-read-diag --json-out v4/dev_notes/telemetry/sweep_kernel_topk2_stage1_20260306_rerun.json`

Quality artifact:
- [sweep_kernel_topk2_stage1_20260306_rerun.json](../../v4/dev_notes/telemetry/sweep_kernel_topk2_stage1_20260306_rerun.json)

Observed 200-step quality:
- `vshape`
  - final acc `0.442`
  - best acc `0.479`
  - final loss `1.9590`
  - wall time `849s`
- `topk_K=2`
  - final acc `0.401`
  - best acc `0.439`
  - final loss `2.0956`
  - wall time `1142s`

Observed topK read telemetry:
- `topk_mean_abs_circ_dist = 67.08`
- `topk_outside_local_frac = 0.9738`
- `topk_attn_entropy = 0.6423`
- `topk_unique_slot_frac = 1.0000`

Read:
- the telemetry confirms this is a genuinely non-local read path, not a disguised local read;
- despite that, the candidate misses the explicit Stage 1 keep-alive thresholds by a wide margin:
  - final acc delta vs `vshape`: `-4.09 pt`
  - best acc delta vs `vshape`: `-4.03 pt`
- because the close condition was `final < -1.5 pt` and `best < -1.0 pt`, the branch closes here without a 500-step or 3-seed continuation.

Verdict:
- close the read-only global topK branch for this round;
- do not run Stage 2 (`500` step) or Stage 3 (`3` seeds), because the candidate is already decisively worse at Stage 1;
- keep `vshape` as the active baseline.

Next action:
- if global retrieval is revisited later, it should be via a hybrid read/write coupling design, not read-only topK.

### Batch 12 — `10k step < 10 min` CPU mechanistic probe

Purpose:
- establish a single-run probe that can hit `10k` steps inside the `10` minute budget;
- use a dense, deterministic memory task instead of the sparse delayed-recall task;
- re-check the `GL` branch on a longer horizon where the model can fully train.

Harness:
- script: `v4/tests/bench_fast_memory.py`
- task: repeating-pattern long-memory bench
- device: `cpu`
- config:
  - `N=1`
  - `hidden_dim=32`
  - `M=64`
  - `slot_dim=8`
  - `batch=8`
  - `seq=8`
  - `period=64`
  - `steps=10000`
  - `seed=42`
- addressing:
  - `LL = read_kernel_mode=vshape, write_address_mode=pointer`
  - `GL = read_kernel_mode=topk, topk_K=2, write_address_mode=pointer`

Artifacts:
- `LL`: [bench_fast_memory_ll_10k_cpu_20260306.json](../../v4/dev_notes/telemetry/bench_fast_memory_ll_10k_cpu_20260306.json)
- `GL`: [bench_fast_memory_gl_10k_cpu_20260306.json](../../v4/dev_notes/telemetry/bench_fast_memory_gl_10k_cpu_20260306.json)

Observed runtime:
- `LL`: `384.0s` total, `0.0384 s/step`
- `GL`: `473.1s` total, `0.0473 s/step`

Observed quality:
- `LL`
  - final acc `100.0%`
  - peak acc `100.0% @ 7000`
  - fresh eval `1.09%`
  - `S=0` probe `1.88%`
  - ring dependency `+98.13pp`
- `GL`
  - final acc `100.0%`
  - peak acc `100.0% @ 2000`
  - fresh eval `0.47%`
  - `S=0` probe `0.31%`
  - ring dependency `+99.69pp`

Observed topK telemetry (`GL`):
- `topk_mean_abs_circ_dist = 17.20`
- `topk_outside_local_frac = 0.9625`
- `topk_attn_entropy = 0.1345`
- `topk_unique_slot_frac = 1.0000`

Read:
- this CPU bench is a valid `10k step < 10 min` mechanistic probe;
- the `GL` branch does not die on the longer dense-memory horizon;
- the telemetry confirms the `GL` branch is genuinely non-local here, not a disguised local read;
- `GL` is slower than `LL`, but on this probe it reaches the same final quality and even peaks earlier.

Verdict:
- keep both `LL` and `GL` alive on this specific CPU bench;
- do not treat this as a production promotion signal;
- this bench is useful as a fast mechanistic truth surface, not as a real-data winner pick.

Next action:
- run `GG` on the same `10k` CPU bench only if we explicitly want to test whether global write adds anything on top of a now-surviving global read;
- do not use the sparse delayed-recall harness as the first `10k` budget probe.

### Batch 13 — `GG` check on the same `10k` CPU bench

Purpose:
- test whether global write adds anything on top of the now-surviving `GL` branch;
- resolve the original mismatch suspicion on the same dense, fully trained mechanistic horizon.

Artifact:
- [bench_fast_memory_gg_10k_cpu_20260306.json](../../v4/dev_notes/telemetry/bench_fast_memory_gg_10k_cpu_20260306.json)

Observed `GG` result:
- final acc `92.2%`
- peak acc `100.0% @ 7000`
- fresh eval `1.6%`
- `S=0` probe `0.5%`
- ring dependency `+91.7pp`
- wall time `492.7s`
- `0.049 s/step`

Observed topK telemetry (`GG`):
- `topk_mean_abs_circ_dist ~= 15.9`
- `topk_outside_local_frac ~= 0.945`
- `write_topk_mean_abs_circ_dist ~= 15.9`
- `write_topk_outside_local_frac ~= 0.945`

Read:
- `GG` is genuinely non-local on both read and write;
- it can solve the task transiently (`100%` peak), but it does not hold the solution as cleanly as `GL`;
- compared to `GL`, it is slower and less stable at the end of training.

Verdict:
- on this `10k` CPU mechanistic bench, `GL` is better than `GG`;
- the original "global read needs global write to work" suspicion is not supported here;
- the current best read on this fast truth-surface is:
  - `LL` and `GL` are both viable;
  - `GG` is not the upgrade path.

Next action:
- keep `GL` alive as a mechanistic branch;
- do not pursue `GG` as the next refinement on this bench;
- if global retrieval is revisited beyond `GL`, it should be via a more selective hybrid write, not full content-topk write.

### Batch 14 — small-model WikiText `LL` baseline

Purpose:
- take the same small `10k` CPU model used on the mechanistic memory bench;
- place it on real English WikiText shards;
- measure where the local baseline (`LL`) actually plateaus before comparing it to `GL`.

Harness:
- script: `v4/tests/sweep_c19_core_geometry_wikitext.py`
- device: `cpu`
- config:
  - `steps=10000`
  - `batch=8`
  - `seq=8`
  - `hidden_dim=32`
  - `M=64`
  - `slot_dim=8`
  - `N=1`
  - `R=1`
  - fixed `C = pi`
  - `tail_mode = linear`
  - `read_kernel_mode = vshape`
  - `write_address_mode = pointer`

Artifact:
- [sweep_wikitext_ll_small_10k_cpu_20260306.json](../../v4/dev_notes/telemetry/sweep_wikitext_ll_small_10k_cpu_20260306.json)

Observed result:
- final acc `0.356`
- best acc `0.578`
- final loss `2.2893`
- final BPC `3.303`
- wall time `395s`
- `0.0395 s/step`

Observed training shape:
- strong early gains up to roughly the `2k-4k` region;
- after that the run enters a broad, noisy plateau around `0.34-0.36` final-window accuracy;
- no instability signal:
  - grad norm stayed around `0.5-0.6`
  - max grad norm stayed below `1.0`

Observed C19 telemetry:
- `tail_hit = 0.0000%`
- `p99 |x|/C = 1.19`
- `p99-ring = 1.00`

Read:
- this small real-data baseline is stable, not spiky;
- it does learn real byte-level English regularities, but it appears capacity-limited;
- the plateau looks like a small-model ceiling, not a training-instability failure;
- just like the earlier probes, the active C19 regime still stays deep inside the inner core and does not use the tail.

Verdict:
- this is a valid real-data baseline for a same-size `LL vs GL` comparison;
- the local baseline does not collapse, but it also does not continue improving sharply after the mid-run;
- the next useful comparison on this exact surface is `GL`, not a larger `LL`.

Next action:
- run the matching small-model WikiText `GL` baseline on the same `10k` CPU surface if we want to know whether global read breaks this local plateau or merely converges to the same ceiling.

### Batch 15 — param-matched TinyTransformer baseline on the same small WikiText surface

Purpose:
- establish an apples-to-apples standard-language-model baseline on the exact same small WikiText CPU surface;
- check whether the small ring model is merely "working" or actually stronger than a standard transformer at the same parameter budget.
- this batch supersedes the earlier rough transformer check that landed near `23k` params; the verdict below is the corrected `~17.5k` parameter match.

Harness:
- script: `v4/tests/bench_tiny_transformer_wikitext_small.py`
- device: `cpu`
- config:
  - `steps=10000`
  - `batch=8`
  - `seq=8`
  - `seed=42`
  - `d_model=24`
  - `n_layers=1`
  - `n_heads=2`
  - `d_ff=48`
  - `max_seq=16`

Artifact:
- [bench_tiny_transformer_wikitext_small_20260306_194722.json](../../v4/dev_notes/telemetry/bench_tiny_transformer_wikitext_small_20260306_194722.json)

Observed result:
- final acc `0.309`
- best acc `0.531`
- final loss `2.4125`
- final BPC `3.480`
- wall time `46.1s`
- `0.00461 s/step`

Comparison vs the small `LL` ring baseline:
- params:
  - `TinyTransformer`: `17,464`
  - `LL`: `17,483`
- quality:
  - final acc delta: `LL +4.7 pt`
  - final BPC delta: `LL -0.177`
- speed:
  - `TinyTransformer` is about `8.6x` faster wall-clock on this CPU surface

Read:
- the standard tiny transformer is much faster, but it plateaus lower on final-window quality;
- the ring model is materially slower, but on this tiny real-data surface it does appear to buy real predictive quality, not just a different style of training dynamics;
- the ring baseline holds a stronger late-run plateau, while the transformer remains the cheap speed reference.

Verdict:
- on this exact small WikiText surface, the ring `LL` baseline is stronger than a param-matched tiny transformer in final quality;
- the transformer remains a valuable speed baseline and sanity reference;
- this result supports continuing the `LL vs GL` comparison on the small real-data surface instead of dismissing the ring path as obviously inferior.

Next action:
- keep the tiny transformer result as the standard non-ring baseline for this small surface;
- if we continue the real-data comparison, the next useful run is still the matching small-model `GL` baseline.

### Batch 16 — small-model WikiText `GL` baseline

Purpose:
- test whether the same small real-data surface benefits from global topk read once the `LL` local baseline and the standard transformer baseline are both fixed;
- check whether the local plateau around `0.34-0.36` is a true local-read ceiling or just a retriever bottleneck.

Harness:
- script: `v4/tests/sweep_c19_core_geometry_wikitext.py`
- device: `cpu`
- config:
  - `steps=10000`
  - `batch=8`
  - `seq=8`
  - `seed=42`
  - `hidden_dim=32`
  - `M=64`
  - `slot_dim=8`
  - `N=1`
  - `R=1`
  - fixed `C = pi`
  - `tail_mode = linear`
  - `kernel_mode = topk`
  - `topk_K = 2`
  - `write_address_mode = pointer` via the existing default path
  - `topk_read_diag = on`

Artifact:
- [sweep_c19_core_geometry_wikitext_20260306_194510.json](../../v4/dev_notes/telemetry/sweep_c19_core_geometry_wikitext_20260306_194510.json)

Observed result:
- final acc `0.352`
- best acc `0.625`
- final loss `2.3037`
- final BPC `3.324`
- wall time `481.2s`
- `0.0481 s/step`

Comparison vs `LL`:
- params:
  - `GL`: `17,747`
  - `LL`: `17,483`
- quality:
  - final acc delta vs `LL`: `-0.41 pt`
  - final BPC delta vs `LL`: `+0.021`
  - best acc delta vs `LL`: `+4.69 pt`
- speed:
  - `GL` is about `1.22x` slower than `LL`

Observed topK telemetry:
- `topk_mean_abs_circ_dist = 1.97`
- `topk_outside_local_frac = 0.479`
- `topk_attn_entropy = 0.691`
- `topk_unique_slot_frac = 1.000`

Observed C19 telemetry:
- `tail_hit = 0.0000%`
- `p99 |x|/C = 1.29`
- `p99-ring = 1.00`

Read:
- on this real-data surface, `GL` does not unlock a clearly more global retrieval regime;
- compared to the mechanistic `10k` memory bench, the learned topk read stays much closer to the pointer neighborhood;
- the candidate does not break the small `LL` plateau:
  - it reaches a slightly higher transient peak;
  - but its final-window quality is slightly worse and its wall time is worse.

Verdict:
- keep `LL` as the active small real-data baseline;
- do not promote `GL` as the better real-data small-model read path;
- the result suggests that, on this surface, topk behaves more like a soft local read than a genuinely useful global retriever.

Next action:
- if global retrieval is revisited on real data, it should likely be through a hybrid or delayed-use design, not this direct `topk_K=2` read-only path;
- otherwise the more useful next step is to keep `LL` as the small real-data baseline and compare larger-capacity or integrated variants against it.

### Batch 17 — small-model WikiText `GG` baseline

Purpose:
- close the remaining coupling question on the same small real-data surface;
- test whether the `GL` branch missed the `LL` plateau only because write stayed pointer-local.

Harness:
- script: `v4/tests/sweep_c19_core_geometry_wikitext.py`
- device: `cpu`
- config:
  - `steps=10000`
  - `batch=8`
  - `seq=8`
  - `seed=42`
  - `hidden_dim=32`
  - `M=64`
  - `slot_dim=8`
  - `N=1`
  - `R=1`
  - fixed `C = pi`
  - `tail_mode = linear`
  - `kernel_mode = topk`
  - `topk_K = 2`
  - `write_address_mode = content_topk`
  - `write_topk_K = 2`
  - `topk_read_diag = on`

Artifact:
- [sweep_c19_core_geometry_wikitext_20260306_200228.json](../../v4/dev_notes/telemetry/sweep_c19_core_geometry_wikitext_20260306_200228.json)

Observed result:
- final acc `0.350`
- best acc `0.547`
- final loss `2.3153`
- final BPC `3.340`
- wall time `523.6s`
- `0.0524 s/step`

Comparison vs `LL` and `GL`:
- final acc:
  - `LL`: `0.356`
  - `GL`: `0.352`
  - `GG`: `0.350`
- final BPC:
  - `LL`: `3.303`
  - `GL`: `3.324`
  - `GG`: `3.340`
- wall time:
  - `LL`: `395s`
  - `GL`: `481s`
  - `GG`: `524s`

Observed topK telemetry:
- read:
  - `topk_mean_abs_circ_dist = 5.02`
  - `topk_outside_local_frac = 0.744`
- write:
  - `write_topk_mean_abs_circ_dist = 5.02`
  - `write_topk_outside_local_frac = 0.744`

Observed C19 telemetry:
- `tail_hit = 0.0000%`
- `p99 |x|/C = 1.33`
- `p99-ring = 1.00`

Read:
- unlike `GL`, the `GG` branch is now genuinely non-local on both read and write;
- despite that, it still does not beat the local baseline and does not beat `GL`;
- this means the small real-data surface is not primarily bottlenecked by pointer-local addressing.

Verdict:
- close the "maybe `GL` only lost because write stayed local" hypothesis on this surface;
- the current ordering is:
  - `LL` best final-window quality;
  - `GL` slightly worse;
  - `GG` slightly worse again and slower;
- on this small WikiText setup, relaxing addressing range does not improve the objective.

Next action:
- if global retrieval is revisited, it should be because a different task or a larger-capacity model makes it worth it;
- for this exact small real-data surface, the stronger hypothesis is now capacity / representation bottleneck, not addressing bottleneck.

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

### Batch 18 — small-model WikiText `GL` with `topk_K = 1`

Purpose:
- test the quick "pooling blur" hypothesis on the same small real-data surface;
- check whether collapsing global retrieval from `topk_K = 2` to `topk_K = 1` helps by avoiding weighted averaging of two remote slots.

Harness:
- script: `v4/tests/sweep_c19_core_geometry_wikitext.py`
- device: `cpu`
- config:
  - `steps=10000`
  - `batch=8`
  - `seq=8`
  - `seed=42`
  - `hidden_dim=32`
  - `M=64`
  - `slot_dim=8`
  - `N=1`
  - `R=1`
  - fixed `C = pi`
  - `tail_mode = linear`
  - `kernel_mode = topk`
  - `topk_K = 1`
  - `write_address_mode = pointer`
  - `topk_read_diag = on`

Artifact:
- [sweep_c19_core_geometry_wikitext_20260306_202009.json](../../v4/dev_notes/telemetry/sweep_c19_core_geometry_wikitext_20260306_202009.json)

Observed result:
- final acc `0.345`
- best acc `0.562`
- final loss `2.3324`
- final BPC `3.365`
- wall time `480s`
- `0.0480 s/step`

Comparison vs the earlier small real-data baselines:
- final acc:
  - `LL`: `0.356`
  - `GL, K=2`: `0.352`
  - `GL, K=1`: `0.345`
- final BPC:
  - `LL`: `3.303`
  - `GL, K=2`: `3.324`
  - `GL, K=1`: `3.365`
- wall time:
  - `LL`: `395s`
  - `GL, K=2`: `481s`
  - `GL, K=1`: `480s`

Observed topK telemetry:
- `topk_mean_abs_circ_dist = 1.84`
- `topk_outside_local_frac = 0.328`
- `p99 |x|/C = 1.30`
- `p99-ring = 1.00`

Read:
- `topk_K = 1` did not improve the small real-data result over `topk_K = 2`;
- it was slightly worse in final-window quality and did not recover the `LL` baseline;
- importantly, the `K = 1` read became even more local than the `K = 2` run, with much smaller average ring distance and much lower outside-local fraction.

Verdict:
- the quick "topK blur" explanation is not supported on this surface;
- simply removing the weighted average over two slots does not rescue the global-read path;
- the evidence now points more toward limited useful non-local signal / representation bandwidth than toward topK pooling blur as the primary issue.

Next action:
- keep `LL` as the active small real-data baseline;
- do not spend more time on `topk_K` sweeps on this exact surface;
- if global retrieval is revisited, it should be through a different retrieval shape (for example differentiated pointer or multi-timescale taps), not by tuning `topk_K` within this pooled read design.

### Batch 19 — full ring trace on small WikiText surface (`LL` vs `GG`)

Purpose:
- remove the remaining ambiguity around how the ring is actually used on the small real-data surface;
- collect a full per-step pointer/read/write trace plus slot histograms on the same objective where `LL > GG` was already observed.

Harness:
- script: `v4/tests/sweep_c19_core_geometry_wikitext.py`
- device: `cpu`
- config shared by both runs:
  - `steps=10000`
  - `batch=8`
  - `seq=8`
  - `seed=42`
  - `hidden_dim=32`
  - `M=64`
  - `slot_dim=8`
  - `N=1`
  - `R=1`
  - fixed `C = pi`
  - `tail_mode = linear`
  - `ring_trace = on`
- variants:
  - `LL = vshape read + pointer write`
  - `GG = topk_K=2 read + content_topk write`

Artifacts:
- [LL trace run](../../v4/dev_notes/telemetry/sweep_c19_core_geometry_wikitext_20260306_203626.json)
- [GG trace run](../../v4/dev_notes/telemetry/sweep_c19_core_geometry_wikitext_20260306_204520.json)

Observed result:
- `LL`
  - final acc `0.356`
  - final BPC `3.303`
  - wall time `407s`
  - trace summary:
    - `ptr_unique_frac = 0.125`
    - `read_unique_frac = 0.156`
    - `write_unique_frac = 0.156`
    - `ptr_jump_mean = 1.75`
    - `read_center_dist_mean = 0.67`
    - `write_center_dist_mean = 0.67`
    - `read_write_overlap_mean = 1.000`
- `GG`
  - final acc `0.350`
  - final BPC `3.340`
  - wall time `514s`
  - trace summary:
    - `ptr_unique_frac = 0.125`
    - `read_unique_frac = 0.250`
    - `write_unique_frac = 0.250`
    - `ptr_jump_mean = 1.75`
    - `read_center_dist_mean = 5.02`
    - `write_center_dist_mean = 5.02`
    - `read_write_overlap_mean = 1.000`

Read:
- both variants traverse the same pointer path; the pointer itself is not the differentiator here;
- `LL` keeps both read and write tightly local around the pointer-centered working set;
- `GG` genuinely spreads both read and write much farther across the ring while still following the same pointer timeline;
- despite that much broader ring usage, `GG` still finishes slightly worse than `LL`.

Verdict:
- the missing information on this surface is no longer "how the ring is being used";
- the ring trace now makes the failure mode explicit:
  - `GG` is using a broader non-local ring footprint,
  - but that extra address range does not convert into a better objective;
- this is strong evidence that the small real-data surface is bottlenecked more by representation / bandwidth than by pointer-local reach.

Next action:
- stop treating addressing range as the main unknown on this exact small WikiText surface;
- if global retrieval is revisited, the next candidate should be a different retrieval shape rather than another topK sweep;
- the more promising next proof target is either differentiated pointer interpolation or a multi-timescale tap design.

### Batch 20 — canonical nightly runner + adversarial hardening

Purpose:
- consolidate the nightly research path into one canonical entrypoint;
- prevent fresh-start vs carry and local vs global claims from being mixed silently;
- make every canonical artifact self-describing enough that the surface semantics are obvious from the JSON alone.

Canonical entrypoint:
- script: [v4/tests/nightly_research_runner.py](../../v4/tests/nightly_research_runner.py)
- allowed surfaces only:
  - `small_wikitext_fresh`
  - `fast_memory_carry`
  - `wikitext_sequential_carry`
- allowed variants only:
  - `LL`
  - `GL`
  - `GG`

Hardening changes:
- all canonical artifacts now record:
  - `surface_kind`
  - `state_mode`
  - `read_mode`
  - `write_mode`
  - `seq`
  - `steps`
  - `ring_slots`
  - `reset_each_batch`
  - `pooled_topk_read`
- all canonical runs now emit ring-trace summaries with:
  - `ptr_unique_frac`
  - `read_unique_frac`
  - `write_unique_frac`
  - `read_center_dist_mean`
  - `write_center_dist_mean`
  - `read_write_overlap_mean`
- `GL/GG` runs are forced to expose topk telemetry, or the runner fails loudly;
- `_scratch` proof and smoke leftovers were separated from the canonical path and ignored in git.

Validation:
- unit guards: [v4/tests/test_nightly_research_runner.py](../../v4/tests/test_nightly_research_runner.py)
- `pytest` result: `5 passed`
- smoke artifacts:
  - small fresh:
    - [LL](../../v4/dev_notes/telemetry/nightly_runner_small_wikitext_fresh_LL_20260306_212253.json)
    - [GL](../../v4/dev_notes/telemetry/nightly_runner_small_wikitext_fresh_GL_20260306_212254.json)
    - [GG](../../v4/dev_notes/telemetry/nightly_runner_small_wikitext_fresh_GG_20260306_212253.json)
  - fast memory carry:
    - [LL](../../v4/dev_notes/telemetry/nightly_runner_fast_memory_carry_LL_20260306_212542.json)
    - [GL](../../v4/dev_notes/telemetry/nightly_runner_fast_memory_carry_GL_20260306_212542.json)
    - [GG](../../v4/dev_notes/telemetry/nightly_runner_fast_memory_carry_GG_20260306_212542.json)
  - sequential real-data carry:
    - [LL](../../v4/dev_notes/telemetry/nightly_runner_wikitext_sequential_carry_LL_20260306_212457.json)
    - [GG](../../v4/dev_notes/telemetry/nightly_runner_wikitext_sequential_carry_GG_20260306_212459.json)

Observed result:
- `small_wikitext_fresh` now makes the fresh-start limitation explicit:
  - reset each batch is encoded in the artifact;
  - pointer coverage is capped and guarded.
- `fast_memory_carry` now produces full-step ring traces instead of log-point fragments;
  - globality claims are now backed by complete trace counts.
- `wikitext_sequential_carry` now exists as a separate canonical real-data surface;
  - it proves state carry and broad pointer coverage without being confused with the fresh-start surface.

Verdict:
- the nightly research path now has one official runner and one official artifact format;
- fresh-start and carry surfaces are no longer easy to confuse;
- the prior small-WikiText `GL/GG` verdict remains valid, but only as a `fresh-start` surface verdict, not a general ring verdict.

Next action:
- keep using the canonical runner for all nightly evidence;
- open no new architecture branch until the next candidate is tested through this runner;
- the next architecture candidate remains `pointer interpolation`, followed by `multi-timescale taps`.

### Batch 21 — pointer interpolation quick gate (`learned` pointer, `LL` path)

Purpose:
- test the first real post-hardening architecture candidate;
- isolate whether the discrete pointer center (`ptr.long()`) is a real bottleneck once the pointer is allowed to move fractionally.

Implementation:
- nightly-only `pointer_interp_mode = off|linear` added to [v4/model/instnct.py](../../v4/model/instnct.py)
- local positional read/write now supports a merged fractional support:
  - pointer floor window
  - pointer ceil window
  - aggregated unique support with blended weights
- no topK changes
- no new write mechanism
- no perf-opt changes

Correctness checks:
- [v4/tests/test_nightly_research_runner.py](../../v4/tests/test_nightly_research_runner.py) now includes integer-case equivalence coverage for the interpolation helper
- `pytest` status after patch: `6 passed`

Quick gate A/B:
- runner: [v4/tests/nightly_research_runner.py](../../v4/tests/nightly_research_runner.py)
- variant: `LL`
- pointer mode: `learned`
- compare:
  - `pointer_interp_mode = off`
  - `pointer_interp_mode = linear`

Artifacts:
- fast memory carry:
  - [off](../../v4/dev_notes/telemetry/nightly_runner_fast_memory_carry_LL_20260306_220336_443934.json)
  - [linear](../../v4/dev_notes/telemetry/nightly_runner_fast_memory_carry_LL_20260306_220429_002560.json)
- small WikiText fresh:
  - [off](../../v4/dev_notes/telemetry/nightly_runner_small_wikitext_fresh_LL_20260306_220648_975520.json)
  - [linear](../../v4/dev_notes/telemetry/nightly_runner_small_wikitext_fresh_LL_20260306_220754_769265.json)

Observed result:
- fast memory carry, `1000` steps:
  - `off`: final acc `0.094`, best acc `0.203`, time `42.5s`
  - `linear`: final acc `0.719`, best acc `0.719`, time `39.0s`
- small WikiText fresh, `1000` steps:
  - `off`: final acc `0.274`, best acc `0.484`, BPC `3.904`, time `45.7s`
  - `linear`: final acc `0.275`, best acc `0.500`, BPC `3.897`, time `50.9s`

Read:
- on the mechanistic carry task, pointer interpolation is a major win;
- on the small fresh real-data surface, the gain is present but marginal;
- this strongly suggests the discrete center is a real bottleneck for learned memory routing, but not the dominant limit on the current small fresh WikiText surface.

Verdict:
- keep `pointer interpolation` alive as the next architecture branch;
- do not promote from the `1000`-step quick gate alone;
- the next confirmation run should be:
  - `fast_memory_carry`, longer confirm
  - then `wikitext_sequential_carry`
- no reason to reopen pooled topK before this pointer branch is properly confirmed.

### Batch 22 — pointer interpolation long confirm (`learned` pointer, `LL` path)

Purpose:
- confirm whether the Batch 21 win survives a full `10k` carry run;
- separate the mechanistic routing signal from the real-data carry signal before promoting pointer interpolation further.

Runner:
- [v4/tests/nightly_research_runner.py](../../v4/tests/nightly_research_runner.py)
- variant: `LL`
- pointer mode: `learned`
- compare:
  - `pointer_interp_mode = off`
  - `pointer_interp_mode = linear`

Artifacts:
- fast memory carry, `10k`:
  - [off](../../v4/dev_notes/telemetry/nightly_runner_fast_memory_carry_LL_20260306_221755_793650.json)
  - [linear](../../v4/dev_notes/telemetry/nightly_runner_fast_memory_carry_LL_20260306_222517_866783.json)
- sequential WikiText carry, `10k`:
  - [off](../../v4/dev_notes/telemetry/nightly_runner_wikitext_sequential_carry_LL_20260306_223327_017337.json)
  - [linear](../../v4/dev_notes/telemetry/nightly_runner_wikitext_sequential_carry_LL_20260306_224150_937913.json)

Observed result:
- fast memory carry, `10k`:
  - `off`: final acc `0.891`, best acc `1.000`, time `386.7s`, ring dependency `+88.3pp`
  - `linear`: final acc `1.000`, best acc `1.000`, time `429.5s`, ring dependency `+99.2pp`
- sequential WikiText carry, `10k`:
  - `off`: final acc `0.367`, best acc `0.719`, BPC `3.261`, time `460.6s`
  - `linear`: final acc `0.363`, best acc `0.688`, BPC `3.279`, time `486.1s`

Read:
- on the mechanistic carry surface, pointer interpolation survives the longer run and remains clearly stronger:
  - earlier convergence
  - better final accuracy
  - stronger ring dependence
  - cleaner ring health
- on the real-data sequential carry surface, the same change does not help:
  - final accuracy is slightly worse
  - best accuracy is lower
  - BPC is slightly worse
  - runtime is slightly slower

Verdict:
- the discrete pointer center is a real bottleneck for learned routing on the mechanistic carry task;
- it is not the dominant bottleneck on the current small real-data sequential carry surface;
- pointer interpolation stays alive as a research branch, but it is not ready for promotion as a general nightly default;
- the next architecture branch should not reopen pooled topK;
- the next serious candidate after this is still `multi-timescale taps`, unless a larger-capacity real-data surface is introduced first.
