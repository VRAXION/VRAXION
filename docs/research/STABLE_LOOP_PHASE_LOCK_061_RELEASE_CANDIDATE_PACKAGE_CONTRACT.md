# STABLE_LOOP_PHASE_LOCK_061_RELEASE_CANDIDATE_PACKAGE Contract

061 creates `INSTNCT_RC_001`, a static release-candidate documentation package
under `docs/releases/`. It packages the 050-060 evidence chain into release
manifest, install, smoke, checksum, doc-index, limitation, support, and claim
boundary docs.

061 is not GA, not production deployment, not hosted SaaS launch, not public
beta, not final legal terms, not commercial launch, and not a new model
experiment.

## Required Artifacts

```text
docs/releases/INSTNCT_RC_001_RELEASE_MANIFEST.md
docs/releases/INSTNCT_RC_001_INSTALL_GUIDE.md
docs/releases/INSTNCT_RC_001_SMOKE_TEST_GUIDE.md
docs/releases/INSTNCT_RC_001_KNOWN_LIMITATIONS.md
docs/releases/INSTNCT_RC_001_SUPPORT_BOUNDARY.md
docs/releases/INSTNCT_RC_001_CHECKSUMS.json
docs/releases/INSTNCT_RC_001_DOC_INDEX.md
docs/releases/INSTNCT_RC_001_CLAIM_BOUNDARY.md
docs/research/STABLE_LOOP_PHASE_LOCK_061_RELEASE_CANDIDATE_PACKAGE_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_061_RELEASE_CANDIDATE_PACKAGE_RESULT.md
scripts/probes/run_stable_loop_phase_lock_061_release_candidate_check.py
```

## Static Validation

The 061 checker must validate committed files only. It must not run smoke
commands, run a model experiment, create a release archive, create a binary, or
create a checkpoint.

Required validation commands:

```text
python -m py_compile scripts/probes/run_stable_loop_phase_lock_061_release_candidate_check.py
python scripts/probes/run_stable_loop_phase_lock_061_release_candidate_check.py --check-only
git diff --check
```

## Boundary

Exact boundary tokens:

```text
no GA
no production deployment
no hosted SaaS launch
no public beta
no production API readiness
no production readiness
no clinical use
no high-stakes education use
no final legal terms
no commercial launch
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```

