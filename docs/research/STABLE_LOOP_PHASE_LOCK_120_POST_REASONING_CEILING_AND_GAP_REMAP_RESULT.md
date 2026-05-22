# STABLE_LOOP_PHASE_LOCK_120_POST_REASONING_CEILING_AND_GAP_REMAP_RESULT

120 is implemented by:

- `scripts/probes/run_stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap.py`
- `scripts/probes/run_stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap_check.py`

The runner writes generated outputs only under
`target/pilot_wave/stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap/`.

Expected positive route:

- `POST_REASONING_CEILING_AND_GAP_REMAP_POSITIVE`
- `UPSTREAM_119_REASONING_CONFIRM_VERIFIED`
- `POST_REASONING_CEILING_MAP_COMPLETE`
- `FAILURE_MODE_MAP_WRITTEN`
- `NEW_BREAKPOINT_WRITTEN`
- `REASONING_REGRESSION_REJECTED`
- `RETENTION_PRESERVED`
- `COLLAPSE_REJECTED`
- `CONTROLS_FAILED`
- `LEAKAGE_REJECTED`
- `BOUNDED_RELEASE_UNCHANGED`
- `121_TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN`

The result is an eval-only ceiling/gap remap. It is not GPT-like assistant
readiness, not open-domain assistant readiness, not production chat, not public
API, not deployment readiness, not safety alignment, and not Hungarian assistant
readiness.
