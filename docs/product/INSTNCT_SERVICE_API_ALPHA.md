# INSTNCT Service API Alpha

Status: 062 service/API alpha planning and engineering artifact.

The service/API alpha wraps the 057 SDK candidate and 058 deployment harness
behind a localhost-only HTTP surface. It is local/private evaluation
infrastructure only: no production API readiness, no production deployment, no
hosted SaaS, no public beta, no multi-tenant IAM, no clinical use, no
high-stakes education use, no commercial launch, no full VRAXION, no language
grounding, no consciousness, no biological/FlyWire equivalence, and no physical
quantum behavior.

## Alpha Routes

```text
GET /v1/health
POST /v1/policy/check
POST /v1/jobs
GET /v1/jobs/{job_id}
POST /v1/infer
POST /v1/evaluate
POST /v1/visual-export
GET /v1/artifacts/{job_id}/{artifact_name}
```

## Canonical Implementation

```text
tools/instnct_service_alpha/instnct_service_alpha.py
```

The alpha implementation uses Python stdlib HTTP only. It does not add Rust
HTTP dependencies and does not widen the public `instnct-core` crate surface.

## Boundary

Exact boundary tokens:

```text
no production API readiness
no production deployment
no hosted SaaS
no public beta
no multi-tenant IAM
no clinical use
no high-stakes education use
no commercial launch
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```

