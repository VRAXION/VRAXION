# VRAXION Public Delivery Model

Status: public P11 boundary summary.

This repository remains the public SDK and documentation boundary. It does not
contain the private engine source.

## Current Direction

The preferred public delivery sequence is:

```text
1. controlled early-access signed binary
2. hosted API / SaaS later
3. thin public SDK, docs, and wrapper glue where useful
```

This is not a full engine source release and not an uncontrolled local
assistant-style executable.

## What This Repo Provides

```text
source-visible compatibility SDK
boundary documentation
public CI and audit scripts
Pages status docs
```

## What This Repo Does Not Provide

```text
private engine source
private binary internals
production training service
skill persistence store
hosted SaaS availability
medical or clinical claims
```

## Trust Surface

Future public releases should be paired with:

```text
clear release notes
signed artifacts or checksums
versioned boundary docs
security contact
deterministic compatibility fixtures where available
```

The public theory and prior-art trail can stay public. The integrated engine
can remain protected while the project is still in early-access stabilization.
