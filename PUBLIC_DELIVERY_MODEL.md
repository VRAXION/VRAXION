# VRAXION Public Delivery Model

Status: public delivery summary for release `public-sdk-p11-20260629`.

This repository remains the public SDK and documentation surface. It does not
contain the private engine source.

## Current Direction

The preferred public delivery sequence is:

```text
1. controlled early-access signed artifact path after release review
2. any API or hosted service path only after separate public review
3. thin public SDK, docs, and wrapper glue where useful
```

This is not a full engine release, not a hosted availability promise, and not
an uncontrolled local assistant-style executable.

## What This Repo Provides

```text
public compatibility SDK
release documentation
public CI and audit scripts
Pages status docs
```

## What This Repo Does Not Provide

```text
private engine source
private binary internals
production training service
non-public persistence services
hosted SaaS availability
medical or clinical claims
```

## Trust Surface

Future public releases should be paired with:

```text
clear release notes
signed artifacts or checksums
versioned release docs
security contact
deterministic compatibility fixtures where available
```

The public theory and prior-art trail can stay public. The integrated engine
can remain protected while the project is still in early-access stabilization.
