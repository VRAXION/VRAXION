# INSTNCT Core Install Smoke Security Ops Support Checklist

Status: 066 install/smoke/security/ops/support checklist.

This checklist aggregates existing gates for the Core private/self-hosted
release readiness candidate.

## Checklist

- Install guide exists in `docs/releases/INSTNCT_RC_001_INSTALL_GUIDE.md`.
- Smoke guide exists in `docs/releases/INSTNCT_RC_001_SMOKE_TEST_GUIDE.md`.
- SDK candidate evidence exists in 057 result.
- Deployment harness evidence exists in 058 result.
- Service/API alpha evidence exists in 062 result.
- Visual audit package evidence exists in 055 result.
- Security/supply-chain gate evidence exists in 063 result.
- Ops-readiness gate evidence exists in 064 result.
- Support boundary exists in `docs/releases/INSTNCT_RC_001_SUPPORT_BOUNDARY.md`.
- Private pilot support policy exists in 065 docs.
- License draft/counsel-review boundary exists in 060 docs.
- Claim boundary exists in 061 RC_001 docs.

## Required Checker Commands

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_066_core_ga_private_readiness_check.py
python scripts/probes/run_stable_loop_phase_lock_066_core_ga_private_readiness_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_065_private_pilot_readiness_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_064_observability_incident_backup_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_063_security_supply_chain_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_062_service_api_alpha_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_061_release_candidate_check.py --check-only
git diff --check
```

No GA launched.
No production release.
No production deployment.
No hosted SaaS.
No public beta.
No public launch.
No partner approved.
No pilot launched.
No production API readiness.
No production SRE readiness.
No SLA.
No clinical use.
No high-stakes education use.
No PHI/student records without separate written agreement.
No final legal terms.
No commercial launch.
No full VRAXION.
No language grounding.
No consciousness.
No biological/FlyWire equivalence.
No physical quantum behavior.

## Boundary

Exact boundary tokens:

```text
no GA launched
no production release
no production deployment
no hosted SaaS
no public beta
no public launch
no partner approved
no pilot launched
no production API readiness
no production SRE readiness
no SLA
no clinical use
no high-stakes education use
no PHI/student records without separate written agreement
no final legal terms
no commercial launch
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```
