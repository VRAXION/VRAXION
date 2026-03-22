"""Lightweight public-surface consistency checks for CI and local audits."""

from __future__ import annotations

import re
import sys
import ast
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "instnct" / "model" / "graph.py"
README = ROOT / "README.md"
V42_README = ROOT / "instnct" / "README.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
FINDINGS = ROOT / "VALIDATED_FINDINGS.md"
LANDING = ROOT / "docs" / "index.html"
VERSION_FILE = ROOT / "VERSION.json"
LANDING_STACK_MAP = ROOT / "docs" / "assets" / "vraxion-public-stack-map.png"
ARCHIVE = ROOT / "ARCHIVE.md"
PR_TEMPLATE = ROOT / ".github" / "pull_request_template.md"
PUBLIC_UPDATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "public_update.md"

WIKI_SOURCE_DIR = ROOT / "docs" / "wiki"
WIKI_FLOWCHART_SRC = WIKI_SOURCE_DIR / "AI-Logic-Flowchart.md"
WIKI_HOME_SRC = WIKI_SOURCE_DIR / "Home.md"
WIKI_CH01_SRC = WIKI_SOURCE_DIR / "Chapter-01---Vision-and-Scope.md"
WIKI_RELEASE_NOTES_SRC = WIKI_SOURCE_DIR / "Release-Notes.md"
WIKI_SWG_SRC = WIKI_SOURCE_DIR / "INSTNCT-Architecture.md"
WIKI_FINDINGS_SRC = WIKI_SOURCE_DIR / "Validated-Findings.md"
WIKI_ENGINEERING_SRC = WIKI_SOURCE_DIR / "Engineering.md"
WIKI_GOVERNANCE_SRC = WIKI_SOURCE_DIR / "Governance.md"
WIKI_SIDEBAR_SRC = WIKI_SOURCE_DIR / "_Sidebar.md"
WIKI_FOOTER_SRC = WIKI_SOURCE_DIR / "_Footer.md"
ENGLISH_RECIPE = ROOT / "instnct" / "recipes" / "english_1024n_18w.py"
HOME_ANATOMY_FILE = ROOT / "docs" / "assets" / "instnct-at-a-glance-core.png"
HOME_ANATOMY_SOURCE_FILE = ROOT / "docs" / "assets" / "source" / "wiki-home-graphics.drawio"
HOME_MISSION_ILLUSTRATION_FILE = ROOT / "docs" / "assets" / "long-horizon-mission.jpg"
HOME_HERO_FILE = ROOT / "docs" / "assets" / "vraxion-home-hero.jpg"
ARCH_TRAINING_LOOP_FILE = ROOT / "docs" / "assets" / "instnct-at-a-glance-training.png"
HOME_LOGO_ASSET = "https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/vraxion-instnct-spiral.png"
HOME_HERO_ASSET = "https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/vraxion-home-hero.jpg"
HOME_STACK_MAP_ASSET = "https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/vraxion-public-stack-map.png"
HOME_ANATOMY_ASSET = "https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/instnct-at-a-glance-core.png"
HOME_MISSION_ILLUSTRATION_ASSET = "https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/long-horizon-mission.jpg"
ARCH_TRAINING_LOOP_ASSET = "https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/instnct-at-a-glance-training.png"
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
    WIKI_FLOWCHART_SRC,
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
    WIKI_FLOWCHART_SRC: WIKI_MIRROR_DIR / "AI-Logic-Flowchart.md",
    WIKI_HOME_SRC: WIKI_MIRROR_DIR / "Home.md",
    WIKI_CH01_SRC: WIKI_MIRROR_DIR / "Chapter-01---Vision-and-Scope.md",
    WIKI_RELEASE_NOTES_SRC: WIKI_MIRROR_DIR / "Release-Notes.md",
    WIKI_SWG_SRC: WIKI_MIRROR_DIR / "INSTNCT-Architecture.md",
    WIKI_FINDINGS_SRC: WIKI_MIRROR_DIR / "Validated-Findings.md",
    WIKI_ENGINEERING_SRC: WIKI_MIRROR_DIR / "Engineering.md",
    WIKI_SIDEBAR_SRC: WIKI_MIRROR_DIR / "_Sidebar.md",
    WIKI_FOOTER_SRC: WIKI_MIRROR_DIR / "_Footer.md",
}
WIKI_HOME_LABEL = "VRAXION Home"
WIKI_ARCH_LABEL = "INSTNCT Architecture"
OLD_WIKI_ARCH_LABEL = "VRAXION Architecture (INSTNCT)"
PRIMARY_NAV_TARGETS = ["Home", "INSTNCT-Architecture", "Validated-Findings", "Engineering", "Release-Notes"]
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
    README: ["VALIDATED_FINDINGS.md", "instnct/README.md"],
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
STALE_NEXT_TARGET_PHRASE = "18-worker swarm"
CURRENT_NEXT_TARGET_PHRASE = "context-dependent task learning"
DIAMOND_ARCHIVE_BRANCH = "archive/diamond-code-era-20260322"
SURFACE_FREEZE_BRANCH = "archive/instnct-surface-freeze-20260322"
EXPECTED_ACTIVE_RECIPES = {"english_1024n_18w.py", "train_wordpairs_ll.py"}
EXPECTED_ACTIVE_PROBES = {"generate_text.py"}
BANNED_TRACKED_GLOBS = [
    "instnct/sweeps/**",
    "instnct/tests/archive/**",
    "instnct/tests/results/**",
    "instnct/tests/viz/**",
    "instnct/tests/viz_interference/**",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def is_tracked(relpath: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", relpath],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def tracked_glob(pattern: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def load_version_info(errors: list[str]) -> dict[str, str]:
    if not VERSION_FILE.exists():
        fail(f"Missing version source of truth: {VERSION_FILE}", errors)
        return {}
    try:
        data = json.loads(read(VERSION_FILE))
    except Exception as exc:
        fail(f"VERSION.json: failed to parse JSON ({exc})", errors)
        return {}

    required = [
        "current_release",
        "current_channel",
        "next_milestone",
        "next_channel",
        "internal_code_path",
    ]
    for key in required:
        if key not in data:
            fail(f"VERSION.json: missing required field {key!r}", errors)

    if data.get("current_release") != "v4.2.0":
        fail("VERSION.json: current_release must be 'v4.2.0'", errors)
    if data.get("current_channel") != "stable":
        fail("VERSION.json: current_channel must be 'stable'", errors)
    if data.get("next_milestone") != "v5.0.0 Public Beta":
        fail("VERSION.json: next_milestone must be 'v5.0.0 Public Beta'", errors)
    if data.get("next_channel") != "preparation":
        fail("VERSION.json: next_channel must be 'preparation'", errors)
    if data.get("internal_code_path") != "instnct/":
        fail("VERSION.json: internal_code_path must be 'instnct/'", errors)

    return data


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
    if CURRENT_NEXT_TARGET_PHRASE not in text.lower():
        fail(f"docs/index.html: current-next-target card should mention {CURRENT_NEXT_TARGET_PHRASE!r}", errors)
    if "v4.2.0" not in text:
        fail("docs/index.html: missing current release framing for v4.2.0", errors)
    if "v5.0.0 Public Beta" not in text:
        fail("docs/index.html: missing next milestone framing for v5.0.0 Public Beta", errors)
    if "instnct/" not in text:
        fail("docs/index.html: missing internal code path framing for instnct/", errors)


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
    if HOME_HERO_ASSET not in home_text:
        fail("Home.md: top front-door hero reference must remain intact", errors)
    if 'alt="VRAXION front-door illustration"' not in home_text:
        fail("Home.md: missing front-door hero alt text", errors)
    if '<em>The engineering of the "I"</em>' not in home_text:
        fail('Home.md: missing front-door hero quote "The engineering of the \\"I\\""', errors)
    if not HOME_HERO_FILE.exists():
        fail(f"Missing Home hero asset: {HOME_HERO_FILE}", errors)
    if HOME_LOGO_ASSET in home_text:
        fail("Home.md: spiral logo should no longer be the top Home visual", errors)
    if HOME_STACK_MAP_ASSET not in home_text:
        fail("Home.md: missing PNG public-stack orientation graphic", errors)
    if "home-public-stack.svg" in home_text:
        fail("Home.md: old home-public-stack.svg reference should not remain", errors)
    if "VRAXION public stack hierarchy" not in home_text:
        fail("Home.md: missing updated public-stack image alt text", errors)
    if "Use this stack map to jump by question, not by page name." not in home_text:
        fail("Home.md: missing refined lead-in line for the public stack map", errors)
    if "Use this stack map to jump by intent." in home_text:
        fail("Home.md: old stack-map lead-in should not remain", errors)

    other_primary_pages = [
        WIKI_SWG_SRC,
        WIKI_FINDINGS_SRC,
        WIKI_ENGINEERING_SRC,
        WIKI_RELEASE_NOTES_SRC,
    ]
    for path in other_primary_pages:
        if HOME_STACK_MAP_ASSET in read(path):
            fail(f"{path.name}: PNG public-stack graphic should stay on Home only", errors)


def check_home_anatomy_graphic(errors: list[str]) -> None:
    home_text = read(WIKI_HOME_SRC)
    if HOME_ANATOMY_ASSET not in home_text:
        fail("Home.md: missing instnct-at-a-glance-core.png anatomy graphic", errors)
    if "INSTNCT core anatomy at a glance:" not in home_text:
        fail("Home.md: missing anatomy-graphic lead-in line", errors)
    if "home-instnct-anatomy.svg" in home_text:
        fail("Home.md: old home-instnct-anatomy.svg reference should not remain", errors)
    if not HOME_ANATOMY_FILE.exists():
        fail(f"Missing home anatomy asset: {HOME_ANATOMY_FILE}", errors)
    if not HOME_ANATOMY_SOURCE_FILE.exists():
        fail(f"Missing editable source for home anatomy asset: {HOME_ANATOMY_SOURCE_FILE}", errors)
    if (ROOT / "docs" / "assets" / "home-instnct-anatomy.svg").exists():
        fail("docs/assets/home-instnct-anatomy.svg: retired anatomy SVG should be deleted", errors)

    other_primary_pages = [
        WIKI_SWG_SRC,
        WIKI_FINDINGS_SRC,
        WIKI_ENGINEERING_SRC,
        WIKI_RELEASE_NOTES_SRC,
    ]
    for path in other_primary_pages:
        if HOME_ANATOMY_ASSET in read(path):
            fail(f"{path.name}: home-instnct-anatomy.svg should stay on Home only", errors)


def check_home_mission_illustration(errors: list[str]) -> None:
    home_text = read(WIKI_HOME_SRC)
    if HOME_MISSION_ILLUSTRATION_ASSET not in home_text:
        fail("Home.md: missing long-horizon mission illustration", errors)
    if 'alt="Long-horizon mission illustration"' not in home_text:
        fail("Home.md: missing long-horizon mission illustration alt text", errors)
    if not HOME_MISSION_ILLUSTRATION_FILE.exists():
        fail(f"Missing long-horizon mission asset: {HOME_MISSION_ILLUSTRATION_FILE}", errors)

    other_primary_pages = [
        WIKI_SWG_SRC,
        WIKI_FINDINGS_SRC,
        WIKI_ENGINEERING_SRC,
        WIKI_RELEASE_NOTES_SRC,
    ]
    for path in other_primary_pages:
        if HOME_MISSION_ILLUSTRATION_ASSET in read(path):
            fail(f"{path.name}: long-horizon mission illustration should stay on Home only", errors)


def check_architecture_training_graphic(errors: list[str]) -> None:
    swg_text = read(WIKI_SWG_SRC)
    if HOME_LOGO_ASSET not in swg_text:
        fail("INSTNCT-Architecture.md: missing INSTNCT spiral logo", errors)
    if 'alt="INSTNCT spiral logo"' not in swg_text:
        fail("INSTNCT-Architecture.md: missing INSTNCT spiral logo alt text", errors)
    if HOME_HERO_ASSET in swg_text:
        fail("INSTNCT-Architecture.md: Home front-door hero should stay on Home only", errors)
    if ARCH_TRAINING_LOOP_ASSET not in swg_text:
        fail("INSTNCT-Architecture.md: missing instnct-at-a-glance-training.png training graphic", errors)
    if "Mutation-selection loop at a glance:" not in swg_text:
        fail("INSTNCT-Architecture.md: missing training-graphic lead-in line", errors)
    if "Mutation-selection training loop at a glance" not in swg_text:
        fail("INSTNCT-Architecture.md: missing training-graphic alt text", errors)
    if not ARCH_TRAINING_LOOP_FILE.exists():
        fail(f"Missing architecture training asset: {ARCH_TRAINING_LOOP_FILE}", errors)

    other_primary_pages = [
        WIKI_HOME_SRC,
        WIKI_FINDINGS_SRC,
        WIKI_ENGINEERING_SRC,
        WIKI_RELEASE_NOTES_SRC,
    ]
    for path in other_primary_pages:
        if ARCH_TRAINING_LOOP_ASSET in read(path):
            fail(f"{path.name}: instnct-at-a-glance-training.png should stay on Architecture only", errors)


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


def check_findings_constants(
    theta_default: str,
    projection_scale_default: str,
    edge_magnitude_default: str,
    errors: list[str],
) -> None:
    findings_text = read(FINDINGS)
    if f"`DEFAULT_THETA = {theta_default}`" not in findings_text:
        fail("VALIDATED_FINDINGS.md: DEFAULT_THETA line is not aligned to graph.py", errors)
    if f"`DEFAULT_PROJECTION_SCALE = {projection_scale_default}`" not in findings_text:
        fail("VALIDATED_FINDINGS.md: DEFAULT_PROJECTION_SCALE line is not aligned to graph.py", errors)
    if f"`DEFAULT_EDGE_MAGNITUDE = {edge_magnitude_default}`" not in findings_text:
        fail("VALIDATED_FINDINGS.md: DEFAULT_EDGE_MAGNITUDE line is not aligned to graph.py", errors)


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


def check_current_next_target_sync(errors: list[str]) -> None:
    current_target_surfaces = [README, V42_README, LANDING, WIKI_HOME_SRC, WIKI_SWG_SRC, WIKI_FINDINGS_SRC, WIKI_RELEASE_NOTES_SRC]
    for path in current_target_surfaces:
        text = read(path)
        lowered = text.lower()
        if STALE_NEXT_TARGET_PHRASE in lowered:
            fail(f"{path.name}: stale next-target phrasing {STALE_NEXT_TARGET_PHRASE!r} should not appear on active public surfaces", errors)
        if CURRENT_NEXT_TARGET_PHRASE not in lowered:
            fail(f"{path.name}: current next-target framing should mention {CURRENT_NEXT_TARGET_PHRASE!r}", errors)


def check_archive(errors: list[str]) -> None:
    text = read(ARCHIVE)
    required = [
        "Only the active self-wiring graph line belongs on `main`:",
        "public documentation for the current self-wiring direction",
        "If a line is not part of the current self-wiring mainline doctrine, archive it.",
        DIAMOND_ARCHIVE_BRANCH,
        SURFACE_FREEZE_BRANCH,
    ]
    for term in required:
        if term not in text:
            fail(f"ARCHIVE.md: missing expected archive-policy term {term!r}", errors)


def check_surface_freeze(errors: list[str]) -> None:
    for pattern in BANNED_TRACKED_GLOBS:
        hits = tracked_glob(pattern)
        if hits:
            fail(f"{pattern}: retired research surface must not remain tracked on main", errors)

    recipe_hits = {Path(path).name for path in tracked_glob("instnct/recipes/*.py")}
    if recipe_hits != EXPECTED_ACTIVE_RECIPES:
        fail(
            "instnct/recipes: tracked active recipe set must be exactly "
            f"{sorted(EXPECTED_ACTIVE_RECIPES)} (found {sorted(recipe_hits)})",
            errors,
        )

    probe_hits = {Path(path).name for path in tracked_glob("instnct/probes/*.py")}
    if probe_hits != EXPECTED_ACTIVE_PROBES:
        fail(
            "instnct/probes: tracked active probe set must be exactly "
            f"{sorted(EXPECTED_ACTIVE_PROBES)} (found {sorted(probe_hits)})",
            errors,
        )

    gitignore_text = read(ROOT / ".gitignore")
    for term in ["instnct/tests/results/", "instnct/tests/viz/", "instnct/tests/viz_interference/"]:
        if term not in gitignore_text:
            fail(f".gitignore: missing surface-freeze ignore protection {term!r}", errors)


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


def check_front_door_history_truth(errors: list[str]) -> None:
    for path in [README, CONTRIBUTING, V42_README]:
        text = read(path)
        if "Historical code lines remain in-repo for reference only" in text:
            fail(f"{path.name}: should not claim retired historical lines remain tracked in-repo", errors)
        for stale in ["`v4/`", "`v22_ternary/`", "`v23_instnct_lm/`"]:
            if stale in text:
                fail(f"{path.name}: stale historical path {stale} should not appear on front-door/governance surfaces", errors)
        if "Diamond Code/" in text:
            fail(f"{path.name}: should not describe Diamond Code/ as a live path on main", errors)


def check_diamond_extraction(errors: list[str]) -> None:
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", "Diamond Code/**"],
        capture_output=True,
        text=True,
        check=False,
    )
    tracked_files = [line for line in tracked.stdout.splitlines() if line.strip()]
    if tracked_files:
        fail("Diamond Code/**: historical tree must not remain tracked on main", errors)
    if "**/.influx_token" not in read(ROOT / ".gitignore"):
        fail(".gitignore: missing .influx_token ignore protection", errors)
    if "Diamond Code/" not in read(ROOT / ".gitignore"):
        fail(".gitignore: missing local Diamond Code ignore after extraction", errors)


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
    if "VERSION.json" not in text:
        fail("Release-Notes.md: missing VERSION.json source-of-truth reference", errors)
    for term in ["`v4`, `v22_ternary`, `v23_instnct_lm`", "not as active tracked public lines on `main`"]:
        if term not in text:
            fail(f"Release-Notes.md: missing historical line-context term {term!r}", errors)


def check_release_framing(version_info: dict[str, str], errors: list[str]) -> None:
    if not version_info:
        return
    release = version_info["current_release"]
    milestone = version_info["next_milestone"]
    path = version_info["internal_code_path"]

    readme_text = read(README)
    if not re.search(rf"\*{{0,2}}Current canonical public release:\*{{0,2}}\s*(?:\[[^\]]+\]\([^)]+\)|`?{re.escape(release)}`?)", readme_text):
        fail(f"README.md: missing current canonical public release framing for {release}", errors)
    if not re.search(rf"\*{{0,2}}Next public milestone:\*{{0,2}}\s*preparation toward\s*`?{re.escape(milestone)}`?", readme_text):
        fail(f"README.md: missing next public milestone framing for {milestone}", errors)
    if not re.search(rf"\*{{0,2}}Internal code path:\*{{0,2}}\s*(?:\[[^\]]+\]\([^)]+\)|`?{re.escape(path)}`?)", readme_text):
        fail(f"README.md: missing internal code path framing for {path}", errors)

    home_text = read(WIKI_HOME_SRC)
    if not re.search(rf"Current canonical public release:\s*(?:\[[^\]]+\]\([^)]+\)|`?{re.escape(release)}`?)", home_text):
        fail(f"Home.md: missing current canonical public release framing for {release}", errors)
    if not re.search(rf"Next public milestone:\s*preparation toward\s*`?{re.escape(milestone)}`?", home_text):
        fail(f"Home.md: missing next public milestone framing for {milestone}", errors)
    if not re.search(rf"Internal code path:\s*(?:\[[^\]]+\]\([^)]+\)|`?{re.escape(path)}`?)", home_text):
        fail(f"Home.md: missing internal code path framing for {path}", errors)

    contributing_text = read(CONTRIBUTING)
    if "VERSION.json" in contributing_text and not VERSION_FILE.exists():
        fail("CONTRIBUTING.md: claims VERSION.json is canonical but VERSION.json is missing", errors)

    release_notes_text = read(WIKI_RELEASE_NOTES_SRC)
    if "VERSION.json" in release_notes_text and not VERSION_FILE.exists():
        fail("Release-Notes.md: claims VERSION.json is canonical but VERSION.json is missing", errors)


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
    if not re.search(rf"\[{re.escape(WIKI_ARCH_LABEL)}\]\(INSTNCT-Architecture\)", sidebar_text):
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
    if not re.search(rf"\[{re.escape(WIKI_ARCH_LABEL)}\]\(INSTNCT-Architecture\)", footer_text):
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
    version_info = load_version_info(errors)

    graph_text = read(GRAPH)
    theta_default = extract_constant("DEFAULT_THETA", graph_text)
    projection_scale_default = extract_constant("DEFAULT_PROJECTION_SCALE", graph_text)
    edge_magnitude_default = extract_constant("DEFAULT_EDGE_MAGNITUDE", graph_text)

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

    check_findings_constants(
        theta_default,
        projection_scale_default,
        edge_magnitude_default,
        errors,
    )
    check_recipe_candidate_sync(errors)
    check_edge_representation_sync(errors)
    check_current_next_target_sync(errors)
    check_archive(errors)
    check_surface_freeze(errors)
    check_templates(errors)
    check_contributing(errors)
    check_front_door_history_truth(errors)
    check_diamond_extraction(errors)
    check_release_framing(version_info, errors)
    check_ch01_stub(errors)
    check_release_notes_page(errors)
    check_primary_editorial_shape(errors)
    check_home_orientation_graphic(errors)
    check_home_anatomy_graphic(errors)
    check_home_mission_illustration(errors)
    check_architecture_training_graphic(errors)
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
    print(
        "Current mainline defaults: "
        f"DEFAULT_THETA={theta_default}, "
        f"DEFAULT_PROJECTION_SCALE={projection_scale_default}, "
        f"DEFAULT_EDGE_MAGNITUDE={edge_magnitude_default}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
