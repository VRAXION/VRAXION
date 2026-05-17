# INSTNCT Service Alpha Artifact Retrieval

Status: 062 artifact retrieval alpha boundary.

Artifact retrieval is allowlist-only:

```text
GET /v1/artifacts/{job_id}/{artifact_name}
```

The service may return only artifacts registered in that job's manifest.

Rejected:

- `../` traversal
- absolute paths
- repository-root files
- source files
- arbitrary `target/` files
- unregistered artifact names

Artifact retrieval is no production API readiness, no production deployment, no
hosted SaaS, no public beta, no multi-tenant IAM, no clinical use, no
high-stakes education use, no commercial launch, no full VRAXION, no language
grounding, no consciousness, no biological/FlyWire equivalence, and no physical
quantum behavior.

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

