# INSTNCT Private Evaluation Runbook

Status: 058 private evaluation runbook.

Private evaluation mode is for isolated technical evaluation only. It does not
permit production automation, hosted SaaS, clinical use, or high-stakes
education use.

## Command

```powershell
python tools/instnct_deploy/instnct_deploy.py smoke --config tools/instnct_deploy/config/example.private_eval.json
```

## Operator Responsibilities

- Use only approved private evaluation data.
- Keep outputs under `target/pilot_wave/...`.
- Review `audit_log.jsonl` and `summary.json`.
- Confirm production flags remain false.
- Stop immediately if a requested use becomes regulated or production-like.

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
