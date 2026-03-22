"""Lightweight public-surface consistency checks for CI and local audits."""

from __future__ import annotations

import re
import sys
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "v4.2" / "model" / "graph.py"
README = ROOT / "README.md"
V42_README = ROOT / "v4.2" / "README.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
FINDINGS = ROOT / "VALIDATED_FINDINGS.md"
LANDING = ROOT / "docs" / "index.html"
LANDING_STACK_MAP = ROOT / "docs" / "assets" / "vraxion-public-stack-map.png"
ARCHIVE = ROOT / "ARCHIVE.md"
PR_TEMPLATE = ROOT / ".github" / "pull_request_template.md"
PUBLIC_UPDATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "public_update.md"

WIKI_SOURCE_DIR = ROOT / "docs" / "wiki"
WIKI_HOME_SRC = WIKI_SOURCE_DIR / "Home.md"
WIKI_CH01_SRC = WIKI_SOURCE_DIR / "Chapter-01---Vision-and-Scope.md"
WIKI_RELEASE_NOTES_SRC = WIKI_SOURCE_DIR / "Release-Notes.md"
WIKI_SWG_SRC = WIKI_SOURCE_DIR / "SWG-v4.2-Architecture.md"
WIKI_FINDINGS_SRC = WIKI_SOURCE_DIR / "Validated-Findings.md"
WIKI_ENGINEERING_SRC = WIKI_SOURCE_DIR / "Engineering.md"
WIKI_GOVERNANCE_SRC = WIKI_SOURCE_DIR / "Governance.md"
WIKI_SIDEBAR_SRC = WIKI_SOURCE_DIR / "_Sidebar.md"
WIKI_FOOTER_SRC = WIKI_SOURCE_DIR / "_Footer.md"
ENGLISH_RECIPE = ROOT / "v4.2" / "english_1024n_18w.py"
HOME_STACK_MAP_FILE = ROOT / "docs" / "assets" / "home-public-stack.svg"
HOME_ANATOMY_FILE = ROOT / "docs" / "assets" / "home-instnct-anatomy.svg"
HOME_LOGO_ASSET = "https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/vraxion-instnct-spiral.png"
HOME_STACK_MAP_ASSET = "https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/home-public-stack.svg"
HOME_ANATOMY_ASSET = "https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/home-instnct-anatomy.svg"
HOME_STACK_MAP_LABELS = [
    "VRAXION Home",
    "INSTNCT Architecture",
    "Validated Findings",
    "Engineering Protocol",
    "Project Timeline",
]
HOME_STACK_MAP_DESCRIPTORS = [
    "orientation hub",
    "active technical line",
    "evidence board",
    "run contract",
    "changes and terms",
]
HOME_STACK_MAP_BANNED_GENERIC_LABELS = [
    r">\s*Timeline\s*<",
    r">\s*Architecture\s*<",
    r">\s*Findings\s*<",
    r">\s*Engineering\s*<",
    r">\s*Home\s*<",
]
HOME_ANATOMY_LABELS = [
    "Passive I/O Projections",
    "Self-Wiring Hidden Graph",
    "persistent internal state",
    "Mutation-Selection Training",
    "not fixed-graph backprop",
]
REMOVED_WIKI_SOURCE_FILES = [
    WIKI_GOVERNANCE_SRC,
    WIKI_SOURCE_DIR / "Chapter-11---Roadmap.md",
    WIKI_SOURCE_DIR / "Theory-of-Thought.md",
    WIKI_SOURCE_DIR / "Hypotheses.md",
    WIKI_SOURCE_DIR / "Legacy-Vault.md",
    WIKI_SOURCE_DIR / "Glossary.md",
    WIKI_SOURCE_DIR / "Diamond-Code-v3-Architecture.md",
    WIKI_SOURCE_DIR / "Proven-Findings.md",
]
WIKI_MIRROR_DIR = ROOT / "VRAXION.wiki"
PRIMARY_WIKI_SOURCE_FILES = {
    WIKI_HOME_SRC,
    WIKI_CH01_SRC,
    WIKI_RELEASE_NOTES_SRC,
    WIKI_SWG_SRC,
    WIKI_FINDINGS_SRC,
    WIKI_ENGINEERING_SRC,
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
    WIKI_RELEASE_NOTES_SRC,
    WIKI_SWG_SRC,
    WIKI_FINDINGS_SRC,
    WIKI_ENGINEERING_SRC,
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
    WIKI_RELEASE_NOTES_SRC: WIKI_MIRROR_DIR / "Release-Notes.md",
    WIKI_SWG_SRC: WIKI_MIRROR_DIR / "SWG-v4.2-Architecture.md",
    WIKI_FINDINGS_SRC: WIKI_MIRROR_DIR / "Validated-Findings.md",
    WIKI_ENGINEERING_SRC: WIKI_MIRROR_DIR / "Engineering.md",
    WIKI_SIDEBAR_SRC: WIKI_MIRROR_DIR / "_Sidebar.md",
    WIKI_FOOTER_SRC: WIKI_MIRROR_DIR / "_Footer.md",
}
WIKI_HOME_LABEL = "VRAXION Home"
WIKI_ARCH_LABEL = "INSTNCT Architecture"
OLD_WIKI_ARCH_LABEL = "VRAXION Architecture (INSTNCT)"
PRIMARY_NAV_TARGETS = ["Home", "SWG-v4.2-Architecture", "Validated-Findings", "Engineering", "Release-Notes"]
LANDING_CTA_LABELS = [
    "VRAXION Home",
    "INSTNCT Architecture",
    "Validated Findings",
    "Engineering Protocol",
    "Project Timeline",
]
PRIMARY_PAGE_SECTION_RULES = {
    WIKI_HOME_SRC: ["At a Glance", "Use This Page When", "Start Here"],
    WIKI_SWG_SRC: ["At a Glance", "Use This Page When", "Current Shipped Facts"],
    WIKI_FINDINGS_SRC: ["Best Current Evidence", "Use This Page When", "Evidence Table"],
    WIKI_ENGINEERING_SRC: ["At a Glance", "Use This Page When", "Run Contract", "Required Evidence"],
    WIKI_RELEASE_NOTES_SRC: [
        "Current Snapshot",
        "What Matters Now",
        "Preparing for v5.0.0 Public Beta",
        "Project Timeline",
        "Open Questions and Promotion Gates",
        "Key Terms",
    ],
}
HOME_ONLY_META_PHRASES = [
    "Repo-tracked docs are canonical",
    "mirrored secondary surface",
]
FOOTER_ONLY_META_PHRASES = [
    "if the GitHub wiki render is incomplete",
]
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
STALE_RECIPE_TEXT = "add/add/add/flip/theta/decay"
STALE_RECIPE_SLOT_TEXT = "decay slot"
CURRENT_CANDIDATE_PHRASE = "triangle-derived `2 add / 1 flip / 5 decay` schedule"
EDGE_FORMAT_FINDING_PHRASE = "sign+mag + magnitude resample"
EDGE_FORMAT_RESULT_PHRASE = "`18.69%` at `155` edges (`q=0.121`)"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def extract_constant(name: str, text: str) -> str:
    match = re.search(rf"^\s*{name}\s*=\s*([0-9.]+)", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find constant {name} in {GRAPH}")
    return match.group(1)


def extract_schedule_list(text: str) -> list[str]:
    match = re.search(r"^\s*SCHEDULE\s*=\s*(\[[^\n]+\])", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find SCHEDULE in {ENGLISH_RECIPE}")
    return ast.literal_eval(match.group(1))


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
    for term in ["Current mainline", "Validated finding", "Experimental branch", "INSTNCT", "Best validated evidence", "Current next target"]:
        if term not in text:
            fail(f"docs/index.html: missing required term {term!r}", errors)
    if WIKI_ARCH_LABEL not in text:
        fail(f"docs/index.html: missing architecture landing label {WIKI_ARCH_LABEL!r}", errors)
    for label in LANDING_CTA_LABELS:
        if label not in text:
            fail(f"docs/index.html: missing CTA label {label!r}", errors)
    if "./assets/vraxion-public-stack-map.png" not in text:
        fail("docs/index.html: missing public stack map asset reference", errors)
    if "VRAXION public stack hierarchy" not in text:
        fail("docs/index.html: missing public stack map alt text", errors)
    if "mutation policy, and edge representation still under active review" not in text:
        fail("docs/index.html: current-next-target card should mention edge representation review", errors)


def require_sections(path: Path, text: str, sections: list[str], errors: list[str]) -> None:
    for section in sections:
        if section not in text:
            fail(f"{path.name}: missing expected editorial section {section!r}", errors)


def require_details_section(path: Path, text: str, summary: str, errors: list[str]) -> None:
    pattern = rf"<details>\s*<summary>{re.escape(summary)}</summary>.*?</details>"
    if not re.search(pattern, text, re.DOTALL):
        fail(f"{path.name}: expected {summary!r} to remain inside a <details> block", errors)


def check_primary_editorial_shape(errors: list[str]) -> None:
    for path, sections in PRIMARY_PAGE_SECTION_RULES.items():
        require_sections(path, read(path), sections, errors)

    release_notes_text = read(WIKI_RELEASE_NOTES_SRC)
    if not re.search(r"^#\s+Project Timeline\s*$", release_notes_text, re.MULTILINE):
        fail("Release-Notes.md: visible page title must be 'Project Timeline'", errors)
    require_details_section(WIKI_RELEASE_NOTES_SRC, release_notes_text, "Open retired surface map", errors)
    require_details_section(WIKI_RELEASE_NOTES_SRC, release_notes_text, "Open key terms", errors)

    findings_text = read(WIKI_FINDINGS_SRC)
    require_details_section(WIKI_FINDINGS_SRC, findings_text, "Historical Context", errors)


def check_home_orientation_graphic(errors: list[str]) -> None:
    home_text = read(WIKI_HOME_SRC)
    asset_text = read(HOME_STACK_MAP_FILE)
    if HOME_LOGO_ASSET not in home_text:
        fail("Home.md: top spiral logo reference must remain intact", errors)
    if HOME_STACK_MAP_ASSET not in home_text:
        fail("Home.md: missing home-public-stack.svg orientation graphic", errors)
    if "Use this stack map to jump by question, not by page name." not in home_text:
        fail("Home.md: missing refined lead-in line for the public stack map", errors)
    if "Use this stack map to jump by intent." in home_text:
        fail("Home.md: old stack-map lead-in should not remain", errors)

    for label in HOME_STACK_MAP_LABELS:
        if label not in asset_text:
            fail(f"home-public-stack.svg: missing exact public label {label!r}", errors)
    for descriptor in HOME_STACK_MAP_DESCRIPTORS:
        if descriptor not in asset_text:
            fail(f"home-public-stack.svg: missing intent descriptor {descriptor!r}", errors)
    for pattern in HOME_STACK_MAP_BANNED_GENERIC_LABELS:
        if re.search(pattern, asset_text):
            fail(f"home-public-stack.svg: old generic standalone label pattern {pattern!r} should not remain", errors)

    other_primary_pages = [
        WIKI_SWG_SRC,
        WIKI_FINDINGS_SRC,
        WIKI_ENGINEERING_SRC,
        WIKI_RELEASE_NOTES_SRC,
    ]
    for path in other_primary_pages:
        if HOME_STACK_MAP_ASSET in read(path):
            fail(f"{path.name}: home-public-stack.svg should stay on Home only", errors)


def check_home_anatomy_graphic(errors: list[str]) -> None:
    home_text = read(WIKI_HOME_SRC)
    asset_text = read(HOME_ANATOMY_FILE)
    if HOME_ANATOMY_ASSET not in home_text:
        fail("Home.md: missing home-instnct-anatomy.svg anatomy graphic", errors)
    if "INSTNCT in one glance:" not in home_text:
        fail("Home.md: missing anatomy-graphic lead-in line", errors)
    for label in HOME_ANATOMY_LABELS:
        if label not in asset_text:
            fail(f"home-instnct-anatomy.svg: missing anatomy label {label!r}", errors)

    other_primary_pages = [
        WIKI_SWG_SRC,
        WIKI_FINDINGS_SRC,
        WIKI_ENGINEERING_SRC,
        WIKI_RELEASE_NOTES_SRC,
    ]
    for path in other_primary_pages:
        if HOME_ANATOMY_ASSET in read(path):
            fail(f"{path.name}: home-instnct-anatomy.svg should stay on Home only", errors)


def check_meta_copy_boundaries(errors: list[str]) -> None:
    non_home_primary_pages = [
        WIKI_SWG_SRC,
        WIKI_FINDINGS_SRC,
        WIKI_ENGINEERING_SRC,
        WIKI_RELEASE_NOTES_SRC,
    ]
    for path in non_home_primary_pages:
        text = read(path)
        for phrase in HOME_ONLY_META_PHRASES:
            if phrase in text:
                fail(f"{path.name}: long canonical/mirror explanation should stay on Home only", errors)
        for phrase in FOOTER_ONLY_META_PHRASES:
            if phrase in text:
                fail(f"{path.name}: GitHub render fallback note should stay in the footer only", errors)


def check_findings_constants(threshold: str, inj_scale: str, errors: list[str]) -> None:
    findings_text = read(FINDINGS)
    if f"`THRESHOLD = {threshold}`" not in findings_text:
        fail("VALIDATED_FINDINGS.md: threshold line is not aligned to graph.py", errors)
    if f"`INJ_SCALE = {inj_scale}`" not in findings_text:
        fail("VALIDATED_FINDINGS.md: INJ_SCALE line is not aligned to graph.py", errors)


def check_recipe_candidate_sync(errors: list[str]) -> None:
    recipe_text = read(ENGLISH_RECIPE)
    schedule = extract_schedule_list(recipe_text)
    schedule_counts = {kind: schedule.count(kind) for kind in sorted(set(schedule))}
    header_phrase = "Schedule: triangle-derived 2 add / 1 flip / 5 decay (8-step fixed approximation)"

    if schedule != ['add', 'add', 'flip', 'decay', 'decay', 'decay', 'decay', 'decay']:
        fail(f"english_1024n_18w.py: unexpected current candidate schedule {schedule_counts}", errors)

    if header_phrase not in recipe_text:
        fail("english_1024n_18w.py: header schedule description is not aligned to the current SCHEDULE", errors)

    current_candidate_surfaces = [README, V42_README, FINDINGS, WIKI_HOME_SRC, WIKI_SWG_SRC, WIKI_FINDINGS_SRC, WIKI_RELEASE_NOTES_SRC]
    for path in current_candidate_surfaces:
        text = read(path)
        if STALE_RECIPE_TEXT in text:
            fail(f"{path.name}: stale recipe schedule text {STALE_RECIPE_TEXT!r} should not appear on active public surfaces", errors)
        if STALE_RECIPE_SLOT_TEXT in text.lower():
            fail(f"{path.name}: stale recipe phrasing {STALE_RECIPE_SLOT_TEXT!r} should not appear on active public surfaces", errors)
        if CURRENT_CANDIDATE_PHRASE not in text:
            fail(f"{path.name}: current recipe candidate summary must match the triangle-derived 2/1/5 schedule", errors)


def check_edge_representation_sync(errors: list[str]) -> None:
    for path in [FINDINGS, WIKI_FINDINGS_SRC, WIKI_RELEASE_NOTES_SRC]:
        text = read(path)
        if EDGE_FORMAT_FINDING_PHRASE not in text:
            fail(f"{path.name}: missing sign+mag edge-representation finding", errors)
        if EDGE_FORMAT_RESULT_PHRASE not in text:
            fail(f"{path.name}: missing sign+mag result metrics {EDGE_FORMAT_RESULT_PHRASE}", errors)


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
    for term in [
        "canonical public source",
        "mirror output only",
        "tools/sync_wiki_from_repo.py",
        "tools/check_public_surface.py",
        "VERSION.json",
        "CITATION.cff",
    ]:
        if term not in text:
            fail(f"CONTRIBUTING.md: missing governance term {term!r}", errors)


def check_ch01_stub(errors: list[str]) -> None:
    text = read(WIKI_CH01_SRC)
    for term in ["retired", "Home"]:
        if term not in text:
            fail(f"Chapter-01---Vision-and-Scope.md: missing expected stub term {term!r}", errors)


def check_release_notes_page(errors: list[str]) -> None:
    text = read(WIKI_RELEASE_NOTES_SRC)
    if not re.search(r"^#\s+Project Timeline\s*$", text, re.MULTILINE):
        fail("Release-Notes.md: visible page title must be 'Project Timeline'", errors)
    for term in [
        "What This Page Is",
        "How To Read It",
        "Current Snapshot",
        "What Matters Now",
        "Preparing for v5.0.0 Public Beta",
        "Project Timeline",
        "Retired Surfaces and Replacements",
        "Open Questions and Promotion Gates",
        "Key Terms",
        "Published Releases",
        "Read Next",
    ]:
        if term not in text:
            fail(f"Release-Notes.md: missing expected project-timeline section {term!r}", errors)
    if not re.search(r"Current canonical public release:\s*(?:\[[^\]]+\]\([^)]+\)|`?v4\.2\.0`?)", text):
        fail("Release-Notes.md: missing current canonical public release framing for v4.2.0", errors)
    if not re.search(r"Next public milestone:\s*preparation toward\s*`?v5\.0\.0 Public Beta`?", text):
        fail("Release-Notes.md: missing next public milestone framing for v5.0.0 Public Beta", errors)
    for banned in [
        "v5.0.0 Public Beta is live",
        "v5.0.0 is the current release",
    ]:
        if banned in text:
            fail(f"Release-Notes.md: beta-prep framing should not claim {banned!r}", errors)


def check_removed_wiki_sources(errors: list[str]) -> None:
    for path in REMOVED_WIKI_SOURCE_FILES:
        if path.exists():
            fail(f"{path.name}: retired wiki source should be deleted, not kept as a stub", errors)
        mirror = WIKI_MIRROR_DIR / path.name
        if mirror.exists():
            fail(f"{mirror.name}: mirrored retired wiki page should be deleted", errors)


def check_wiki_sources(errors: list[str]) -> None:
    sidebar_text = read(WIKI_SIDEBAR_SRC)
    if "## Primary" not in sidebar_text:
        fail("_Sidebar.md: missing Primary navigation section", errors)
    for href in PRIMARY_NAV_TARGETS:
        if not re.search(rf"\[[^\]]+\]\({re.escape(href)}\)", sidebar_text):
            fail(f"_Sidebar.md: missing markdown navigation link target {href!r}", errors)
    if not re.search(rf"\[{re.escape(WIKI_HOME_LABEL)}\]\(Home\)", sidebar_text):
        fail(f"_Sidebar.md: missing primary navigation label {WIKI_HOME_LABEL!r}", errors)
    if not re.search(rf"\[{re.escape(WIKI_ARCH_LABEL)}\]\(SWG-v4\.2-Architecture\)", sidebar_text):
        fail(f"_Sidebar.md: missing primary navigation label {WIKI_ARCH_LABEL!r}", errors)
    if "Project Timeline" not in sidebar_text:
        fail("_Sidebar.md: missing primary navigation label 'Project Timeline'", errors)
    if OLD_WIKI_ARCH_LABEL in sidebar_text:
        fail(f"_Sidebar.md: old architecture label {OLD_WIKI_ARCH_LABEL!r} should not appear in active nav", errors)
    if "**ACTIVE**" in sidebar_text or " ACTIVE" in sidebar_text:
        fail("_Sidebar.md: active nav should not contain ad hoc status badges", errors)
    if "Wiki Graph" in sidebar_text:
        fail("_Sidebar.md: Wiki Graph should not appear in the current wiki navigation", errors)
    if "Governance" in sidebar_text or "Documentation Governance" in sidebar_text:
        fail("_Sidebar.md: Governance should not appear in the current wiki navigation", errors)
    if "Proven-Findings" in sidebar_text or "Proven Findings" in sidebar_text:
        fail("_Sidebar.md: Proven Findings should not appear in the current wiki navigation", errors)
    if "Chapter-01---Vision-and-Scope" in sidebar_text or "Chapter 01 - Vision and Scope" in sidebar_text:
        fail("_Sidebar.md: Chapter 01 should not appear in the current wiki navigation", errors)
    if "Chapter-11---Roadmap" in sidebar_text or "Chapter 11 - Roadmap" in sidebar_text:
        fail("_Sidebar.md: Chapter 11 - Roadmap should not appear in the current wiki navigation", errors)
    if "Theory-of-Thought" in sidebar_text or "Theory of Thought" in sidebar_text:
        fail("_Sidebar.md: Theory of Thought should not appear in the current wiki navigation", errors)
    if "Hypotheses" in sidebar_text:
        fail("_Sidebar.md: Hypotheses should not appear in the current wiki navigation", errors)
    if "Diamond-Code-v3-Architecture" in sidebar_text or "Diamond Code v3 Architecture" in sidebar_text:
        fail("_Sidebar.md: Diamond Code v3 Architecture should not appear in the current wiki navigation", errors)
    if "Legacy-Vault" in sidebar_text or "Legacy Vault" in sidebar_text:
        fail("_Sidebar.md: Legacy Vault should not appear in the current wiki navigation", errors)
    if "Glossary" in sidebar_text:
        fail("_Sidebar.md: Glossary should not appear in the current wiki navigation", errors)

    footer_text = read(WIKI_FOOTER_SRC)
    if "Nav:" not in footer_text:
        fail("_Footer.md: missing Nav line", errors)
    for href in PRIMARY_NAV_TARGETS:
        if not re.search(rf"\[[^\]]+\]\({re.escape(href)}\)", footer_text):
            fail(f"_Footer.md: missing footer markdown navigation link target {href!r}", errors)
    if not re.search(rf"\[{re.escape(WIKI_HOME_LABEL)}\]\(Home\)", footer_text):
        fail(f"_Footer.md: missing footer navigation label {WIKI_HOME_LABEL!r}", errors)
    if not re.search(rf"\[{re.escape(WIKI_ARCH_LABEL)}\]\(SWG-v4\.2-Architecture\)", footer_text):
        fail(f"_Footer.md: missing footer navigation label {WIKI_ARCH_LABEL!r}", errors)
    if "Project Timeline" not in footer_text:
        fail("_Footer.md: missing footer navigation label 'Project Timeline'", errors)
    if OLD_WIKI_ARCH_LABEL in footer_text:
        fail(f"_Footer.md: old architecture label {OLD_WIKI_ARCH_LABEL!r} should not appear in active nav", errors)
    if "Governance" in footer_text or "Documentation Governance" in footer_text:
        fail("_Footer.md: Governance should not appear in the current wiki footer", errors)
    if "Chapter-11---Roadmap" in footer_text or "Chapter 11 - Roadmap" in footer_text:
        fail("_Footer.md: Chapter 11 - Roadmap should not appear in the current wiki footer", errors)
    if "Legacy-Vault" in footer_text or "Legacy Vault" in footer_text:
        fail("_Footer.md: Legacy Vault should not appear in the current wiki footer", errors)
    if "Glossary" in footer_text:
        fail("_Footer.md: Glossary should not appear in the current wiki footer", errors)


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
    if not LANDING_STACK_MAP.exists():
        fail(f"Missing landing stack-map asset: {LANDING_STACK_MAP}", errors)
    check_landing(landing_text, errors)
    check_banned_phrases(LANDING, landing_text, errors)
    check_front_door_phrases(LANDING, landing_text, errors)

    check_findings_constants(threshold, inj_scale, errors)
    check_recipe_candidate_sync(errors)
    check_edge_representation_sync(errors)
    check_archive(errors)
    check_templates(errors)
    check_contributing(errors)
    check_ch01_stub(errors)
    check_release_notes_page(errors)
    check_primary_editorial_shape(errors)
    check_home_orientation_graphic(errors)
    check_home_anatomy_graphic(errors)
    check_meta_copy_boundaries(errors)
    check_removed_wiki_sources(errors)
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
