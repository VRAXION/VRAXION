# Public Repo Cleanup Report

Status: guardrail + scanner-hardening pass on `public-surface-cleanup-001`.

This report is path-only. It does not copy artifact contents, row samples, secrets, or private data.

## Repo identity

```text
repo = VRAXION/VRAXION
visibility = public
cleanup branch = public-surface-cleanup-001
history rewrite = not performed
force push = not performed
private repo touched = no
```

## Immediate finding

Recent public work appears to have committed operational research sample packs and runner/checker scripts. This is treated as a public-surface exposure, not as a confirmed credential incident.

## High-priority remove candidates

```text
removed current-tree artifact sample directory
```

Reason codes:

```text
committed_artifact_sample_pack
jsonl_public_boundary_violation
row_level_or_trace_like_artifact_surface
```

## High-priority sanitize or remove candidates

```text
private_probe_runner_removed*
private_probe_runner_removed*
private_probe_runner_removed*
private_probe_runner_removed*
private_probe_runner_removed*
private_probe_runner_removed*
private_probe_runner_removed*
private_probe_runner_removed*
private_probe_runner_removed*
private_probe_runner_removed*
private_probe_runner_removed*
private_dataset_preparation_tool_removed*
```

Reason codes:

```text
operational_runner_public_surface
private_like_experiment_reproducibility_surface
fineweb_operational_surface
```

## Sanitize candidates

```text
docs/research/*_RESULT.md
docs/research/*_CONTRACT.md
README.md
CHANGELOG.md
VALIDATED_FINDINGS.md
```

Reason codes:

```text
exact_metric_surface
exact_runner_command_surface
private_artifact_reference_surface
implementation_detail_surface
```

## Keep-public candidates

```text
LICENSE
SECURITY.md
PUBLIC_BOUNDARY.md
README.md after sanitize
legal/
docs/legal/
docs public landing pages after sanitize
```

## Secret incident status

```text
confirmed_secret_detected = unknown
credential_rotation_required = unknown
history_rewrite_recommended_now = no
```

A separate scan is still required before declaring that no secrets or raw dataset rows were exposed.

## Work completed in this guardrail pass

```text
PUBLIC_BOUNDARY.md added
scripts/audit_public_surface.py added
.github/workflows/public-surface-audit.yml added
.gitignore hardened
SECURITY.md hardened
scanner self-scan false positive fixed
E18-E23 operational runner paths promoted from WARN to FAIL
FineWeb operational runner/tool paths promoted to FAIL
```

## Reverted mistake

```text
A temporary tombstone file was accidentally created under the removed artifact
sample directory and then deleted on the same cleanup branch.
```

No force push or history rewrite was performed.

## Next cleanup pass

1. Remove tracked artifact sample files from current public tree.
2. Remove or stub operational E18-E23 and FineWeb probe/checker scripts.
3. Sanitize exact result and contract docs into public summaries.
4. Run `python scripts/audit_public_surface.py`.
5. Open a draft PR for review.

## Acceptance target for next pass

```text
git ls-files contains no tracked artifact sample path
git ls-files contains no *.jsonl
public audit script passes
no operational FineWeb runner remains public
no E18-E23 operational runner/checker remains public
result docs no longer reference full artifact sample packs
```
