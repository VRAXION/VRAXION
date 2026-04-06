"""Public-surface consistency checks for the current website/wiki shape.

This checker intentionally validates the *current* public contract:
- GitHub Pages served from docs/
- 4 public website routes (Home / INSTNCT / Research / Rust)
- moved wiki stub under docs/wiki/
- VERSION.json as lightweight status source

It does not enforce legacy labels, retired pages, or historical website structure.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
VERSION_FILE = DOCS / "VERSION.json"
NOJEKYLL = DOCS / ".nojekyll"

HOME = DOCS / "index.html"
INSTNCT = DOCS / "instnct" / "index.html"
RESEARCH = DOCS / "research" / "index.html"
RUST = DOCS / "rust" / "index.html"
WIKI_STUB = DOCS / "wiki" / "index.html"

REQUIRED_PAGES = [HOME, INSTNCT, RESEARCH, RUST, WIKI_STUB]
REQUIRED_ASSETS = [
    ASSETS / "favicon.svg",
    ASSETS / "hero-instnct-field.svg",
    ASSETS / "instnct-at-a-glance-core.png",
    ASSETS / "site.css",
    ASSETS / "site.js",
    ASSETS / "vraxion-home-hero.jpg",
    ASSETS / "vraxion-instnct-spiral.png",
]

REQUIRED_NAV_LABELS = ["Home", "INSTNCT", "Research", "Rust", "Wiki", "GitHub"]

STALE_TERMS = [
    "Validated Findings",
    "Engineering Protocol",
    "Project Timeline",
    "Rust v5 Beta Surface",
    "5-Minute Proof",
    "Issue #114",
]

PAGE_EXPECTATIONS = {
    HOME: [
        "Vraxion builds INSTNCT.",
        "Enter the wiki",
        "Open the wiki",
        "Open INSTNCT",
        "Open Research",
        "Open Rust",
    ],
    INSTNCT: [
        "INSTNCT treats topology as part of the learnable object.",
        "Open the full architecture page",
        "Theory of Thought",
    ],
    RESEARCH: [
        "Research is where claims get earned.",
        "Open the archive",
        "Research Process &amp; Archive",
    ],
    RUST: [
        "Rust is no longer the experiment. It is the implementation frontier.",
        "Open the Rust wiki page",
        "17-18%",
    ],
    WIKI_STUB: [
        "Legacy docs/wiki moved",
        "Go to wiki",
        "https://github.com/VRAXION/VRAXION/wiki/Home",
    ],
}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def require_exists(path: Path, errors: list[str]) -> None:
    if not path.exists():
        fail(f"Missing required file: {path}", errors)


def load_version(errors: list[str]) -> dict[str, str]:
    require_exists(VERSION_FILE, errors)
    if not VERSION_FILE.exists():
        return {}

    try:
        data = json.loads(read(VERSION_FILE))
    except Exception as exc:  # pragma: no cover - CI-facing error path
        fail(f"VERSION.json parse failure: {exc}", errors)
        return {}

    required = [
        "current_release",
        "current_channel",
        "next_milestone",
        "next_channel",
        "architecture_line",
        "rust_surface",
    ]
    for key in required:
        if key not in data:
            fail(f"VERSION.json: missing required key {key!r}", errors)

    if data.get("current_channel") != "stable":
        fail("VERSION.json: current_channel must be 'stable'", errors)
    if data.get("next_channel") != "preparation":
        fail("VERSION.json: next_channel must be 'preparation'", errors)
    if data.get("architecture_line") != "INSTNCT":
        fail("VERSION.json: architecture_line must be 'INSTNCT'", errors)
    if data.get("rust_surface") != "Rust Implementation Surface":
        fail("VERSION.json: rust_surface must be 'Rust Implementation Surface'", errors)

    if "internal_code_path" in data and data.get("internal_code_path") != "instnct/":
        fail("VERSION.json: internal_code_path, if present, must be 'instnct/'", errors)

    return {k: str(v) for k, v in data.items()}


def check_page(path: Path, errors: list[str]) -> None:
    require_exists(path, errors)
    if not path.exists():
        return

    text = read(path)

    if "?v=" not in text:
        fail(f"{path.relative_to(ROOT)}: expected cache-busted asset URLs with ?v=", errors)

    if "site.css" not in text:
        fail(f"{path.relative_to(ROOT)}: missing site.css reference", errors)
    if path != WIKI_STUB and "site.js" not in text:
        fail(f"{path.relative_to(ROOT)}: missing site.js reference", errors)

    for label in REQUIRED_NAV_LABELS:
        if label not in text:
            fail(f"{path.relative_to(ROOT)}: missing nav label {label!r}", errors)

    for term in STALE_TERMS:
        if term in text:
            fail(f"{path.relative_to(ROOT)}: stale term still present: {term!r}", errors)

    for expected in PAGE_EXPECTATIONS.get(path, []):
        if expected not in text:
            fail(f"{path.relative_to(ROOT)}: missing expected text {expected!r}", errors)


def check_assets(errors: list[str]) -> None:
    require_exists(NOJEKYLL, errors)
    for path in REQUIRED_ASSETS:
        require_exists(path, errors)


def check_cross_links(errors: list[str]) -> None:
    home_text = read(HOME)
    if "./instnct/" not in home_text or "./research/" not in home_text or "./rust/" not in home_text:
        fail("docs/index.html: missing one or more internal route links", errors)

    for path in [INSTNCT, RESEARCH, RUST]:
        text = read(path)
        if "../" not in text:
            fail(f"{path.relative_to(ROOT)}: expected relative route links", errors)


def main() -> int:
    errors: list[str] = []

    load_version(errors)
    check_assets(errors)
    for path in REQUIRED_PAGES:
        check_page(path, errors)
    check_cross_links(errors)

    if errors:
        print("Public surface check FAILED:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Public surface check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
