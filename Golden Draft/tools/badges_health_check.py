#!/usr/bin/env python3
"""
VRAXION Badge Health Check (deterministic, stdlib-only)

Validates the repo-hosted SVG label/badge library under:
  docs/assets/badges/{mono,neon}/*.svg

Guards against:
- missing badge files (both themes)
- unsafe SVG features (scripts, external hrefs)
- spec drift (height/viewBox, missing title/desc/accessibility metadata)
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


THEMES = ("mono", "neon")
BADGE_IDS = (
    # Governance / doc-type
    "canonical",
    "locked",
    "protocol",
    "policy",
    "spec",
    "guide",
    "note",
    # Lifecycle / maturity
    "status",
    "roadmap",
    "releases",
    "draft",
    "wip",
    "experimental",
    "legacy",
    "deprecated",
    # Rigor / contracts
    "evidence_required",
    "reproducible",
    "telemetry",
    "fail_gates",
    # Domain
    "engineering",
    "research",
    "gpu",
    "scaling",
    # Public surface
    "wiki",
    "pages",
    # Provenance / license
    "doi_10_5281_zenodo_18332532",
    "noncommercial",
)

EXTERNAL_HREF_RE = re.compile(r"""(?:xlink:)?href\s*=\s*["']https?://""", re.IGNORECASE)
EXTERNAL_URLFUNC_RE = re.compile(r"""url\(\s*["']?\s*https?://""", re.IGNORECASE)


@dataclass(frozen=True)
class Issue:
    kind: str
    file: str
    message: str


def _localname(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_floatish(value: str) -> float | None:
    s = value.strip()
    if s.endswith("px"):
        s = s[: -len("px")]
    try:
        return float(s)
    except ValueError:
        return None


def _check_svg(path: Path) -> list[Issue]:
    issues: list[Issue] = []
    raw = path.read_text(encoding="utf-8", errors="replace")

    if "<script" in raw.lower():
        issues.append(Issue(kind="unsafe_svg", file=str(path), message="Contains <script>"))

    if EXTERNAL_HREF_RE.search(raw) is not None:
        issues.append(Issue(kind="unsafe_svg", file=str(path), message="Contains external href (http/https)"))

    if EXTERNAL_URLFUNC_RE.search(raw) is not None:
        issues.append(Issue(kind="unsafe_svg", file=str(path), message="Contains url(http...) reference"))

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        issues.append(Issue(kind="xml_parse", file=str(path), message=f"XML parse error: {e}"))
        return issues

    if _localname(root.tag) != "svg":
        issues.append(Issue(kind="bad_root", file=str(path), message=f"Root element is not <svg>: {root.tag}"))

    role = (root.attrib.get("role") or "").strip().lower()
    if role != "img":
        issues.append(Issue(kind="a11y", file=str(path), message='Missing/incorrect root attribute: role="img"'))

    aria = (root.attrib.get("aria-labelledby") or "").strip()
    if not aria:
        issues.append(Issue(kind="a11y", file=str(path), message='Missing root attribute: aria-labelledby="..."'))

    height = root.attrib.get("height", "")
    height_f = _parse_floatish(height)
    if height_f is None or abs(height_f - 24.0) > 1e-6:
        issues.append(Issue(kind="spec", file=str(path), message=f'Expected height="24", got "{height}"'))

    view_box = root.attrib.get("viewBox", "")
    parts = [p for p in view_box.replace(",", " ").split() if p]
    if len(parts) != 4:
        issues.append(Issue(kind="spec", file=str(path), message=f'Expected viewBox="0 0 <W> 24", got "{view_box}"'))
    else:
        vb_h = _parse_floatish(parts[3])
        if vb_h is None or abs(vb_h - 24.0) > 1e-6:
            issues.append(
                Issue(
                    kind="spec",
                    file=str(path),
                    message=f'Expected viewBox height 24, got "{view_box}"',
                )
            )

    has_title = False
    has_desc = False
    for el in root.iter():
        ln = _localname(el.tag).lower()
        if ln == "script":
            issues.append(Issue(kind="unsafe_svg", file=str(path), message="Contains <script> element"))
        elif ln == "title":
            has_title = True
        elif ln == "desc":
            has_desc = True

    if not has_title:
        issues.append(Issue(kind="a11y", file=str(path), message="Missing <title>"))
    if not has_desc:
        issues.append(Issue(kind="a11y", file=str(path), message="Missing <desc>"))

    return issues


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    badges_root = repo_root / "docs" / "assets" / "badges"

    issues: list[Issue] = []

    # 1) Existence check (expected library completeness)
    for theme in THEMES:
        for badge_id in BADGE_IDS:
            p = badges_root / theme / f"{badge_id}.svg"
            if not p.exists():
                issues.append(Issue(kind="missing_file", file=str(p), message="Missing badge file"))

    # 2) Spec/safety check (all files in expected set)
    for theme in THEMES:
        for badge_id in BADGE_IDS:
            p = badges_root / theme / f"{badge_id}.svg"
            if p.exists():
                issues.extend(_check_svg(p))

    if issues:
        print("\nBADGES HEALTH CHECK FAILURES:")
        for i in issues:
            print(f"- {i.kind}: {i.file}: {i.message}")
        print("\nSUMMARY:")
        print(f"- issues: {len(issues)}")
        return 1

    print("\nSUMMARY:")
    print("- issues: 0")
    print(f"- badge_ids: {len(BADGE_IDS)}")
    print(f"- themes: {len(THEMES)}")
    print(f"- total_expected_files: {len(BADGE_IDS) * len(THEMES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

