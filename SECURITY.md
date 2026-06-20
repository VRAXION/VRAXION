# Security Policy

VRAXION is a research repository. Security is taken seriously, but expectations should match the public-surface scope.

## Reporting a vulnerability or leak

If you believe you have found a vulnerability, secret, credential, private artifact exposure, dataset leak, raw corpus leak, or exploit path:

1. Use a private GitHub Security Advisory:
   https://github.com/VRAXION/VRAXION/security/advisories/new
2. If advisories are unavailable, use a private contact channel with the repository owner.

Do not open public Issues or Discussions for suspected vulnerabilities, secrets, dataset leaks, private artifact exposure, or exploit paths.

Please include:

- A clear description of the issue
- Affected paths or commits, using path references only when sensitive content is involved
- Steps to reproduce, if safe to share privately
- Impact assessment
- Any mitigations or patches you propose

## In scope

- Vulnerabilities that affect users running public VRAXION tooling or published artifacts
- Accidental exposure of private artifacts, datasets, local paths, secrets, or credentials
- Public-boundary violations that could expose frontier implementation details

## Out of scope

- Attacks requiring full local machine compromise
- Requests to disclose private frontier code, datasets, or unpublished artifacts

## Supported versions

This repo is under active development. Only `main` is considered supported.
