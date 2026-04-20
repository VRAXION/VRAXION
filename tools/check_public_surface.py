"""Public-surface consistency checks for the current website/wiki shape.

This checker validates the current public contract (2026-04-20):
- GitHub Pages served from docs/
- Home + 5 Block pages (A Byte Unit → E Brain) + Legacy detail view + Wiki stub
- docs/VERSION.json as lightweight status source
- Required asset files present

It does not enforce legacy labels, retired pages, or historical website structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
VERSION_FILE = DOCS / "VERSION.json"
NOJEKYLL = DOCS / ".nojekyll"

HOME = DOCS / "index.html"
LEGACY = DOCS / "legacy.html"
WIKI_STUB = DOCS / "wiki" / "index.html"

BLOCK_PAGES = [
    DOCS / "blocks" / "a-byte-unit.html",
    DOCS / "blocks" / "b-merger.html",
    DOCS / "blocks" / "c-tokenizer.html",
    DOCS / "blocks" / "d-embedder.html",
    DOCS / "blocks" / "e-brain.html",
]

BLOCK_VARIANT_PAGES = [
    DOCS / "blocks" / "b-merger-native-7bit.html",
]

REQUIRED_PAGES = [HOME, LEGACY, WIKI_STUB] + BLOCK_PAGES + BLOCK_VARIANT_PAGES

REQUIRED_ASSETS = [
    ASSETS / "favicon.svg",
    ASSETS / "site.css",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def require_exists(path: Path, errors: list[str]) -> None:
    if not path.exists():
        fail(f"Missing required file: {path.relative_to(ROOT)}", errors)


def load_version(errors: list[str]) -> dict[str, str]:
    require_exists(VERSION_FILE, errors)
    if not VERSION_FILE.exists():
        return {}

    try:
        data = json.loads(read(VERSION_FILE))
    except Exception as exc:
        fail(f"VERSION.json parse failure: {exc}", errors)
        return {}

    required = [
        "current_release",
        "current_channel",
        "architecture_line",
    ]
    for key in required:
        if key not in data:
            fail(f"VERSION.json: missing required key {key!r}", errors)

    if data.get("current_channel") not in ("stable", "beta"):
        fail("VERSION.json: current_channel must be 'stable' or 'beta'", errors)
    if data.get("architecture_line") != "INSTNCT":
        fail("VERSION.json: architecture_line must be 'INSTNCT'", errors)

    return {k: str(v) for k, v in data.items()}


def check_home(errors: list[str]) -> None:
    require_exists(HOME, errors)
    if not HOME.exists():
        return
    text = read(HOME)
    for link in ["./blocks/a-byte-unit.html", "./blocks/b-merger.html",
                 "./blocks/c-tokenizer.html", "./blocks/d-embedder.html",
                 "./blocks/e-brain.html"]:
        if link not in text:
            fail(f"docs/index.html: missing Block link {link!r}", errors)


def check_blocks(errors: list[str]) -> None:
    for path in BLOCK_PAGES:
        require_exists(path, errors)
        if not path.exists():
            continue


def check_assets(errors: list[str]) -> None:
    require_exists(NOJEKYLL, errors)
    for path in REQUIRED_ASSETS:
        require_exists(path, errors)


def main() -> int:
    errors: list[str] = []

    load_version(errors)
    check_assets(errors)
    for path in REQUIRED_PAGES:
        require_exists(path, errors)
    check_home(errors)
    check_blocks(errors)

    if errors:
        print("Public surface check FAILED:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Public surface check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
