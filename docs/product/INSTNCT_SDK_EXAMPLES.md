# INSTNCT SDK Examples

Status: 057 SDK release-candidate engineering artifact.

The first runnable example is the bounded smoke runner:

```powershell
cargo run -p instnct-core --example instnct_sdk_candidate_smoke -- --out target/pilot_wave/stable_loop_phase_lock_057_instnct_sdk_release_candidate/smoke
```

Expected output directory:

```text
target/pilot_wave/stable_loop_phase_lock_057_instnct_sdk_release_candidate/smoke
```

Expected smoke sequence:

```text
train -> save -> load -> infer -> evaluate -> export_visual -> rollback
```

The smoke also validates:

```text
invalid_schema_rejection
regulated_use_rejection
```

Expected positive verdict:

```text
SDK_RELEASE_CANDIDATE_POSITIVE
```

## Claim Boundary

057 supports SDK release-candidate engineering only. It does not support
production API readiness, public beta, clinical use, high-stakes education use,
full VRAXION, language grounding, consciousness, biological/FlyWire equivalence,
or physical quantum behavior.

Exact boundary tokens:

```text
no production API readiness
no public beta
no clinical use
no high-stakes education use
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```
