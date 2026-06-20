# Prior Art And Provenance Checklist

This checklist is for release hygiene and defensive publication discipline. It
is not legal advice.

## Release proof bundle

For each public checkpoint that should be easy to cite later, keep these items
aligned:

- public Git commit SHA;
- signed or GitHub-visible tag;
- GitHub release notes;
- source archive generated from the release;
- root `LICENSE`;
- `README.md`;
- `CITATION.cff`;
- `CODEX_HANDOVER.md`;
- relevant result documents under `docs/research/`;
- relevant sanitized artifact sample pack when explicitly included in a release;
- deterministic checker command and result;
- timestamped dashboard or report artifact, if applicable.

## Zenodo / DOI update

If Zenodo is connected to the GitHub repository, trigger a Zenodo archive from
the GitHub release and record:

- concept DOI;
- version DOI;
- Git tag;
- commit SHA;
- release URL;
- Zenodo archive URL;
- date and time of archive.

After the DOI exists, update:

- `CITATION.cff`;
- release notes if needed;
- `README.md` if the DOI should be visible from the front door.

## Optional backup publication

For especially important architecture claims, consider an additional public
timestamp:

- Internet Archive snapshot;
- OSF preregistration or archive;
- arXiv-style technical report;
- signed PDF or release manifest;
- independent mirror.

## Caveats

Prior art/provenance records help show what was disclosed and when. They do not
by themselves enforce the commercial license, and they only protect what is
actually disclosed in enough detail to be understandable and reproducible.

The license governs use of the Project Materials. It does not automatically
control independent clean-room work that does not copy, adapt, use, link to,
derive from, or incorporate Project Materials.
