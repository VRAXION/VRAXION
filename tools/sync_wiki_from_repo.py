"""Sync repo-tracked wiki sources into the checked-out GitHub wiki repo."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs" / "wiki"
# Canonical wiki is typically a sibling of the main repo (e.g. ../VRAXION.wiki/),
# but fall back to the in-repo path if the sibling layout isn't present.
_SIBLING_WIKI = ROOT.parent / "VRAXION.wiki"
WIKI_DIR = _SIBLING_WIKI if _SIBLING_WIKI.exists() else (ROOT / "VRAXION.wiki")
FILES = [
    "Home.md",
    "INSTNCT-Architecture.md",
    "Timeline-Archive.md",
    "COMPRESSION_LOOP.md",
    "Theory-of-Thought.md",
    "v5-Rust-Port-Benchmarks.md",
    "_Sidebar.md",
    "_Footer.md",
    "pipeline-architecture.svg",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def sync_file(src: Path, dst: Path, dry_run: bool) -> tuple[bool, str]:
    src_text = read_text(src)
    if not dst.exists():
        if not dry_run:
            write_text(dst, src_text)
        return True, "create"

    dst_text = read_text(dst)
    if dst_text == src_text:
        return False, "unchanged"

    if not dry_run:
        write_text(dst, src_text)
    return True, "update"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned sync actions without writing files.")
    args = parser.parse_args()

    missing_sources = [name for name in FILES if not (SOURCE_DIR / name).exists()]
    if missing_sources:
        raise SystemExit(f"Missing repo-tracked wiki sources: {', '.join(missing_sources)}")

    if not WIKI_DIR.exists():
        if args.dry_run:
            print(f"Dry-run: wiki checkout not found at {WIKI_DIR}. Source files validated; no mirror written.")
            for name in FILES:
                print(f"would sync {SOURCE_DIR / name} -> {WIKI_DIR / name}")
            return 0
        raise SystemExit(
            f"Wiki checkout not found at {WIKI_DIR}. Clone the GitHub wiki next to the repo or rerun with --dry-run."
        )

    changed = 0
    for name in FILES:
        src = SOURCE_DIR / name
        dst = WIKI_DIR / name
        did_change, action = sync_file(src, dst, args.dry_run)
        if did_change:
            changed += 1
        print(f"{action:9} {src} -> {dst}")

    if args.dry_run:
        print(f"Dry-run complete: {changed} file(s) would change.")
    else:
        print(f"Sync complete: {changed} file(s) changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
