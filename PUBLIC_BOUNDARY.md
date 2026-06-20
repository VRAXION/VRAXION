# Public Boundary Policy

VRAXION has two separate surfaces:

```text
public repo    = brand, prior-art record, safe docs, public roadmap, toy demos
private repo   = frontier implementation, datasets, runners, ledgers, exact configs, raw artifacts
```

This repository is the public surface. It must stay useful for trust, citation, licensing, and safe onboarding, but it must not function as a working frontier experiment warehouse.

## Allowed public content

Allowed content is material that explains the project without enabling direct reproduction of private frontier work:

- project thesis and high-level architecture language
- public roadmap and status summaries
- licensing, trademark, contributor, and prior-art documents
- public-safe diagrams and landing pages
- toy examples that do not contain frontier logic
- sanitized research summaries with clear claim boundaries
- non-operational pseudocode
- issue templates and contribution process documents

## Private-only content

The following must stay out of the public repo:

- frontier experiment code and runners
- private training or evaluation datasets
- dataset manifests and reader manifests
- raw or row-level ledgers
- `.jsonl` research artifacts
- exact canary configs, thresholds, and run commands for private experiments
- real operator registries, negative-memory ledgers, and lifecycle logs
- private checkpoint, resume, and trace files
- local absolute paths, especially Windows dataset paths
- API keys, tokens, secrets, credentials, or private endpoint names
- raw corpus text, corpus-derived snippets, copied prompt batches, or generated training rows
- private repo names or private local paths when they reveal operational layout

## File classes

Each tracked file should be classified as one of:

```text
KEEP_PUBLIC       safe to keep as-is
SANITIZE_PUBLIC   keep path, reduce implementation/detail exposure
MOVE_PRIVATE      belongs in the private repository only
DELETE_PUBLIC     remove from current public tree
HISTORY_REVIEW    current tree can be cleaned, but prior exposure may need review
SECRET_INCIDENT   possible credential or sensitive data leak, rotate/revoke first
```

## Public result rule

Public research result docs may state:

```text
what was tested
high-level outcome
claim boundary
whether artifacts are private pending review
```

They must not include:

```text
row-level samples
full ledgers
exact private runner commands
private dataset paths
full operational manifests
large exact metric dumps that reconstruct the experiment
```

## Artifact sample rule

Committed public artifact sample packs are disallowed unless they are intentionally tiny toy fixtures reviewed for public release.

Default rule:

```text
docs/research/artifact_samples/ = not public
*.jsonl                         = not public
```

## Emergency response

If a secret, token, credential, private dataset row, or raw corpus text appears in public:

1. Stop public merges.
2. Rotate or revoke the secret first, if applicable.
3. Preserve a path-only incident report.
4. Remove the current-tree exposure.
5. Decide separately whether history rewrite is required.

Do not force-push or rewrite history without explicit approval.
