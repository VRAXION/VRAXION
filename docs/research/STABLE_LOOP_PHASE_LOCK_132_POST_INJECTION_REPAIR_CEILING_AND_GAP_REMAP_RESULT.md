# STABLE_LOOP_PHASE_LOCK_132_POST_INJECTION_REPAIR_CEILING_AND_GAP_REMAP_RESULT

## Result Contract

The 132 runner writes generated artifacts under:

```text
target/pilot_wave/stable_loop_phase_lock_132_post_injection_repair_ceiling_and_gap_remap/
```

Expected positive result:

```text
POST_INJECTION_REPAIR_CEILING_AND_GAP_REMAP_POSITIVE
decision = post_injection_repair_ceiling_gap_map_complete
next = 133_TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN
```

The result must include:

- tier and family metrics
- ceiling by tier
- failure mode map
- capability gap map
- post-injection delta versus 128
- prior repair preservation report
- retention, collapse, namespace, leakage, control, and boundary reports

Positive verdicts include:

```text
POST_INJECTION_CEILING_MAP_COMPLETE
UPSTREAM_131_INJECTION_PRIORITY_SCALE_CONFIRM_VERIFIED
REASONING_REPAIR_PRESERVED
STATE_REPAIR_PRESERVED
CALIBRATION_REPAIR_PRESERVED
INJECTION_PRIORITY_REPAIR_PRESERVED
RETENTION_PRESERVED
COLLAPSE_REJECTED
NAMESPACE_MEMORIZATION_REJECTED
CONTROLS_FAILED
LEAKAGE_REJECTED
BOUNDED_RELEASE_UNCHANGED
PRODUCTION_CHAT_NOT_CLAIMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

The new first breakpoint is machine-readable in `decision.json`:

```text
ceiling_status
first_breakpoint_tier OR ceiling_not_reached_within_config
first_breakpoint_family
top_failure_families
primary_next_repair_target
reasoning_preserved
state_preserved
calibration_preserved
injection_priority_preserved
next = 133_TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN
```

## Boundary

132 is eval-only post-injection ceiling/gap remap. It is not GPT-like assistant
readiness, not open-domain assistant readiness, not production chat, not public
API, not deployment readiness, not safety alignment, and not Hungarian
assistant readiness.
