# INSTNCT Deployment Security Baseline

Status: 058 security baseline.

The 058 harness is a local/private evaluation harness. It is not a hardened
production service.

## Baseline Controls

- Reject regulated deployment modes before SDK side effects.
- Reject production flag contamination.
- Restrict outputs to `target/pilot_wave/...`.
- Avoid shell-concatenated user config values.
- Record exact child command as argument list.
- Record normalized-LF config SHA-256.
- Require fresh child SDK smoke artifacts.
- Keep healthcheck side-effect-light.

## Out Of Scope

- Hosted SaaS isolation.
- Tenant isolation.
- Secrets management.
- Production incident response.
- Clinical or high-stakes education compliance.

## Claim Boundary

058 supports local/private deployment harness engineering only.

Exact boundary tokens:

```text
no production deployment
no hosted SaaS
no public beta
no clinical use
no high-stakes education use
no production API readiness
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```
