"""Lightweight public-surface consistency checks for CI and local audits."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "v4.2" / "model" / "graph.py"
README = ROOT / "README.md"
V42_README = ROOT / "v4.2" / "README.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
FINDINGS = ROOT / "VALIDATED_FINDINGS.md"
LANDING = ROOT / "docs" / "index.html"
ARCHIVE = ROOT / "ARCHIVE.md"
PR_TEMPLATE = ROOT / ".github" / "pull_request_template.md"
PUBLIC_UPDATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "public_update.md"

WIKI_SOURCE_DIR = ROOT / "docs" / "wiki"
WIKI_HOME_SRC = WIKI_SOURCE_DIR / "Home.md"
WIKI_CH01_SRC = WIKI_SOURCE_DIR / "Chapter-01---Vision-and-Scope.md"
WIKI_SWG_SRC = WIKI_SOURCE_DIR / "SWG-v4.2-Architecture.md"
WIKI_FINDINGS_SRC = WIKI_SOURCE_DIR / "Validated-Findings.md"
WIKI_PROVEN_SRC = WIKI_SOURCE_DIR / "Proven-Findings.md"
WIKI_ENGINEERING_SRC = WIKI_SOURCE_DIR / "Engineering.md"
WIKI_GOVERNANCE_SRC = WIKI_SOURCE_DIR / "Governance.md"
WIKI_SIDEBAR_SRC = WIKI_SOURCE_DIR / "_Sidebar.md"
WIKI_FOOTER_SRC = WIKI_SOURCE_DIR / "_Footer.md"
WIKI_MIRROR_DIR = ROOT / "VRAXION.wiki"
PRIMARY_WIKI_SOURCE_FILES = {
    WIKI_HOME_SRC,
    WIKI_CH01_SRC,
    WIKI_SWG_SRC,
    WIKI_FINDINGS_SRC,
    WIKI_PROVEN_SRC,
    WIKI_ENGINEERING_SRC,
    WIKI_GOVERNANCE_SRC,
    WIKI_SIDEBAR_SRC,
    WIKI_FOOTER_SRC,
}

MARKDOWN_FILES = [
    README,
    V42_README,
    CONTRIBUTING,
    FINDINGS,
    ARCHIVE,
    WIKI_HOME_SRC,
    WIKI_CH01_SRC,
    WIKI_SWG_SRC,
    WIKI_FINDINGS_SRC,
    WIKI_PROVEN_SRC,
    WIKI_ENGINEERING_SRC,
    WIKI_GOVERNANCE_SRC,
    WIKI_SIDEBAR_SRC,
    WIKI_FOOTER_SRC,
    PR_TEMPLATE,
    PUBLIC_UPDATE,
]
TAXONOMY_FILES = [README, V42_README, FINDINGS, WIKI_HOME_SRC, WIKI_SWG_SRC, WIKI_FINDINGS_SRC, WIKI_ENGINEERING_SRC]
FRONT_DOOR_TEXTS = [README, V42_README, FINDINGS, WIKI_HOME_SRC, WIKI_SWG_SRC, WIKI_FINDINGS_SRC, WIKI_ENGINEERING_SRC]
WIKI_MIRROR_MAP = {
    WIKI_HOME_SRC: WIKI_MIRROR_DIR / "Home.md",
    WIKI_CH01_SRC: WIKI_MIRROR_DIR / "Chapter-01---Vision-and-Scope.md",
    WIKI_SWG_SRC: WIKI_MIRROR_DIR / "SWG-v4.2-Architecture.md",
    WIKI_FINDINGS_SRC: WIKI_MIRROR_DIR / "Validated-Findings.md",
    WIKI_PROVEN_SRC: WIKI_MIRROR_DIR / "Proven-Findings.md",
    WIKI_ENGINEERING_SRC: WIKI_MIRROR_DIR / "Engineering.md",
    WIKI_GOVERNANCE_SRC: WIKI_MIRROR_DIR / "Governance.md",
    WIKI_SIDEBAR_SRC: WIKI_MIRROR_DIR / "_Sidebar.md",
    WIKI_FOOTER_SRC: WIKI_MIRROR_DIR / "_Footer.md",
}
REQUIRED_LOCAL_LINK_TARGETS = {
    README: ["VALIDATED_FINDINGS.md", "v4.2/README.md"],
    V42_README: ["../VALIDATED_FINDINGS.md"],
}
BANNED_ANYWHERE = {
    re.compile(r"\bnew default\b", re.IGNORECASE): "use 'validated finding' or 'experimental branch' instead of 'new default'",
    re.compile(r"\bcurrent defaults?\b", re.IGNORECASE): "avoid 'current default' phrasing on canonical public surfaces; use 'current mainline' or 'live mainline default'",
    re.compile(r"cleaned research mainline", re.IGNORECASE): "old branch-cleanup phrasing should not appear on public surfaces",
    re.compile(r"only learnable structure", re.IGNORECASE): "public surfaces should not claim the graph is the only learnable structure; theta/decay are co-evolved in mainline",
}
FRONT_DOOR_ONLY_BANNED = {
    re.compile(r"credit-guided rewiring", re.IGNORECASE): "credit-guided rewiring should not appear on front-door public surfaces",
    re.compile(r"Diamond Code v3", re.IGNORECASE): "legacy Diamond Code references should not appear on primary public surfaces",
}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def extract_constant(name: str, text: str) -> str:
    match = re.search(rf"^\s*{name}\s*=\s*([0-9.]+)", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find constant {name} in {GRAPH}")
    return match.group(1)


def check_required_terms(path: Path, text: str, errors: list[str]) -> None:
    for term in ["Current mainline", "Validated finding", "Experimental branch"]:
        if term not in text:
            fail(f"{path.name}: missing required taxonomy term {term!r}", errors)


def resolve_local_target(path: Path, href: str) -> Path | None:
    target = (path.parent / href).resolve()
    candidates = [target]
    if not href.lower().endswith(".md"):
        candidates.append(Path(str(target) + ".md"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def check_links(path: Path, text: str, errors: list[str]) -> None:
    for rel_target in REQUIRED_LOCAL_LINK_TARGETS.get(path, []):
        if resolve_local_target(path, rel_target) is None:
            fail(f"{path.name}: required link target does not exist: {rel_target}", errors)

    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
        href = match.group(1).strip()
        if not href or href.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if resolve_local_target(path, href) is None:
            fail(f"{path.name}: broken relative markdown link: {href}", errors)

    if path in PRIMARY_WIKI_SOURCE_FILES and re.search(r"\[\[[^\]]+\]\]", text):
        fail(f"{path.name}: primary wiki sources must use markdown links, not [[...]] alias links", errors)


def check_banned_phrases(path: Path, text: str, errors: list[str]) -> None:
    for pattern, message in BANNED_ANYWHERE.items():
        if pattern.search(text):
            fail(f"{path.name}: {message}", errors)


def check_front_door_phrases(path: Path, text: str, errors: list[str]) -> None:
    for pattern, message in FRONT_DOOR_ONLY_BANNED.items():
        if pattern.search(text):
            fail(f"{path.name}: {message}", errors)


def check_landing(text: str, errors: list[str]) -> None:
    for term in ["Current mainline", "Validated finding", "Experimental branch", "INSTNCT"]:
        if term not in text:
            fail(f"docs/index.html: missing required term {term!r}", errors)


def check_findings_constants(threshold: str, inj_scale: str, errors: list[str]) -> None:
    findings_text = read(FINDINGS)
    if f"`THRESHOLD = {threshold}`" not in findings_text:
        fail("VALIDATED_FINDINGS.md: threshold line is not aligned to graph.py", errors)
    if f"`INJ_SCALE = {inj_scale}`" not in findings_text:
        fail("VALIDATED_FINDINGS.md: INJ_SCALE line is not aligned to graph.py", errors)


def check_archive(errors: list[str]) -> None:
    text = read(ARCHIVE)
    required = [
        "Only the active self-wiring graph line belongs on `main`:",
        "public documentation for the current self-wiring direction",
        "If a line is not part of the current self-wiring mainline doctrine, archive it.",
    ]
    for term in required:
        if term not in text:
            fail(f"ARCHIVE.md: missing expected archive-policy term {term!r}", errors)


def check_templates(errors: list[str]) -> None:
    pr_text = read(PR_TEMPLATE)
    if "tools/check_public_surface.py" not in pr_text:
        fail("pull_request_template.md: missing public-surface verification command", errors)
    if "Taxonomy label" not in pr_text:
        fail("pull_request_template.md: missing taxonomy label prompt", errors)

    public_update_text = read(PUBLIC_UPDATE)
    for term in ["Current mainline", "Validated finding", "Experimental branch"]:
        if term not in public_update_text:
            fail(f"public_update.md: missing taxonomy option {term!r}", errors)


def check_contributing(errors: list[str]) -> None:
    text = read(CONTRIBUTING)
    if "public technical repo" not in text:
        fail("CONTRIBUTING.md: expected public-technical-repo posture text", errors)
    if "docs/wiki/" not in text:
        fail("CONTRIBUTING.md: expected docs/wiki guidance", errors)
    if "VRAXION.wiki/" not in text:
        fail("CONTRIBUTING.md: expected mirrored wiki guidance", errors)


def check_proven_stub(errors: list[str]) -> None:
    text = read(WIKI_PROVEN_SRC)
    for term in ["historical", "Validated-Findings"]:
        if term not in text:
            fail(f"Proven-Findings.md: missing expected stub term {term!r}", errors)


def check_ch01_stub(errors: list[str]) -> None:
    text = read(WIKI_CH01_SRC)
    for term in ["retired", "Home"]:
        if term not in text:
            fail(f"Chapter-01---Vision-and-Scope.md: missing expected stub term {term!r}", errors)


def check_wiki_sources(errors: list[str]) -> None:
    sidebar_text = read(WIKI_SIDEBAR_SRC)
    if "## Primary" not in sidebar_text:
        fail("_Sidebar.md: missing Primary navigation section", errors)
    for href in ["Home", "SWG-v4.2-Architecture", "Validated-Findings", "Engineering"]:
        if not re.search(rf"\[[^\]]+\]\({re.escape(href)}\)", sidebar_text):
            fail(f"_Sidebar.md: missing markdown navigation link target {href!r}", errors)
    if not re.search(r"\[[^\]]+\]\(Governance\)", sidebar_text):
        fail("_Sidebar.md: missing markdown navigation link target 'Governance'", errors)
    if "Proven-Findings" in sidebar_text or "Proven Findings" in sidebar_text:
        fail("_Sidebar.md: Proven Findings should not appear in the current wiki navigation", errors)
    if "Chapter-01---Vision-and-Scope" in sidebar_text or "Chapter 01 - Vision and Scope" in sidebar_text:
        fail("_Sidebar.md: Chapter 01 should not appear in the current wiki navigation", errors)

    footer_text = read(WIKI_FOOTER_SRC)
    if "Nav:" not in footer_text:
        fail("_Footer.md: missing Nav line", errors)
    for href in ["Home", "SWG-v4.2-Architecture", "Validated-Findings", "Engineering", "Governance"]:
        if not re.search(rf"\[[^\]]+\]\({re.escape(href)}\)", footer_text):
            fail(f"_Footer.md: missing footer markdown navigation link target {href!r}", errors)
    if "INSTNCT" not in footer_text:
        fail("_Footer.md: missing footer primary navigation label 'INSTNCT'", errors)


def check_wiki_mirror(errors: list[str]) -> None:
    if not WIKI_MIRROR_DIR.exists():
        return

    for src, mirror in WIKI_MIRROR_MAP.items():
        if not mirror.exists():
            fail(f"Wiki mirror missing expected file: {mirror}", errors)
            continue
        if read(src) != read(mirror):
            fail(f"Wiki mirror drift: {mirror.name} is out of sync with docs/wiki source", errors)


def main() -> int:
    errors: list[str] = []

    graph_text = read(GRAPH)
    threshold = extract_constant("THRESHOLD", graph_text)
    inj_scale = extract_constant("INJ_SCALE", graph_text)

    for path in MARKDOWN_FILES:
        text = read(path)
        if path in TAXONOMY_FILES:
            check_required_terms(path, text, errors)
        check_links(path, text, errors)
        if path in FRONT_DOOR_TEXTS:
            check_banned_phrases(path, text, errors)
            check_front_door_phrases(path, text, errors)

    landing_text = read(LANDING)
    check_landing(landing_text, errors)
    check_banned_phrases(LANDING, landing_text, errors)
    check_front_door_phrases(LANDING, landing_text, errors)

    check_findings_constants(threshold, inj_scale, errors)
    check_archive(errors)
    check_templates(errors)
    check_contributing(errors)
    check_proven_stub(errors)
    check_ch01_stub(errors)
    check_wiki_sources(errors)
    check_wiki_mirror(errors)

    if errors:
        print("Public surface check FAILED:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Public surface check passed.")
    print(f"Current mainline constants: THRESHOLD={threshold}, INJ_SCALE={inj_scale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
