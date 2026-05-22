# STABLE_LOOP_PHASE_LOCK_128_POST_CALIBRATION_REPAIR_CEILING_AND_GAP_REMAP_RESULT

## Result Contract

The 128 runner writes generated artifacts under:

```text
target/pilot_wave/stable_loop_phase_lock_128_post_calibration_repair_ceiling_and_gap_remap/
```

The expected positive decision is:

```text
decision = post_calibration_repair_ceiling_gap_map_complete
next = 129_TARGETED_POST_CALIBRATION_REPAIR_OR_SCALE_PLAN
```

The result must include:

- ceiling status and first breakpoint or `ceiling_not_reached_within_config`
- `first_breakpoint_tier`
- `first_breakpoint_family`
- `primary_next_repair_target`
- failure mode map with complete labels
- capability gap map
- post-calibration delta versus 124
- prior repair preservation report
- retention, collapse, namespace, leakage, control, and boundary reports

The expected first-breakpoint candidates are:

```text
format + prompt injection / instruction priority
long-context format/injection combined stress
multi-doc ambiguity / priority conflict
harder combined post-calibration stress
```

First breakpoint outranks global failure count unless root-vs-symptom evidence
proves a later tier is upstream.

## Boundary

128 is eval-only post-calibration ceiling/gap remap. It is not GPT-like
assistant readiness, not open-domain assistant readiness, not production chat,
not public API, not deployment readiness, not safety alignment, and not
Hungarian assistant readiness.
