# D111T Trig Periodic Targeted Repair Prototype Contract

D111T is an adapter-only targeted repair prototype for `TRIG_PERIODIC_SYMBOLIC_FAMILY` after D111A localized the failure to phase/top1-top2 ambiguity in controlled symbolic formula-discovery tasks.

## Boundary

- Preserve the symbolic formula solver, dense baseline, protected symbolic router, protected components, and the D102 8% light protected sparse core.
- Train only targeted adapter deltas: `recurrent_state_adapter_phase_aware_repair_delta` and `calibration_scalar_adapter_margin_delta`; the optional `halting_head_adapter_threshold_delta` remains gated.
- Do not include symbolic-sequence or language-like families, raw visual Raven tasks, natural-language pretraining, or Gemma-class training.
- Keep trig excluded from the healthy claim unless explicit promotion gates pass; if promotion does not pass, trig remains repair-only.

## Required execution

D111T must audit or restore D111A, record `d111a_upstream_manifest.json`, execute the unreduced requested repair scale, run checkpointed targeted adapter updates, replay the D111A baseline, evaluate trig repair metrics, evaluate Lane A/B/D preservation, run shortcut/leak sentinels, and write all required reports under `target/pilot_wave/d111t_trig_periodic_targeted_repair_prototype/`.

## Positive decision criteria

The healthy repair prototype decision is allowed only when D111A replay is valid, scale is unreduced, sparse identity and protected freezes hold, repair updates execute without rollback, trig failing-case rate is reduced by at least 20%, phase/top1-top2/harmonic/loop/mask/calibration metrics improve, worst-seed score improves, Lane A/B/D preservation gates pass, Rust path is invoked with zero fallback, leak sentinels pass, and reports cross-check.
