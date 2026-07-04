#!/usr/bin/env python3
"""Validate the public zero-state repository surface."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def marker(*parts: str) -> str:
    return "".join(parts)


FORBIDDEN_PATH_PARTS = [
    marker("golden_", "refactor"),
    marker("golden-", "refactor"),
    marker("alphasync-", "selftrain"),
    marker("alphasync-", "skillstore"),
    marker("golden_", "connector"),
    marker("golden_", "legacy_parity"),
    marker("vraxion-", "runtime"),
    marker("docs/", "research"),
    marker("tools/", "private_data_adapters"),
    marker("docs/", "vn", "gard"),
    marker("red", "b"),
]

FORBIDDEN_TEXT = [
    marker("Fine", "Web"),
    marker("E", "136"),
    marker("final_", "train"),
    marker("current_", "best"),
    marker("Perma", "Core"),
    marker("True", "Golden"),
    marker("json", "l"),
    marker("check", "point"),
    marker("row_", "level"),
    marker("private ", "dataset"),
    marker("S:", "\\"),
    marker("C:", "\\"),
]

LEGAL_TEXT_FILES = {
    "LICENSE",
    "crates/alphasync-core/LICENSE",
    "crates/alphasync-runtime/LICENSE",
}

PUBLIC_COPY_SUFFIXES = (".md", ".html", ".cff")

FORBIDDEN_PUBLIC_COPY = [
    marker("source", "-available"),
    marker("source ", "available"),
    marker("source ", "snapshot"),
    marker("source ", "archive"),
    marker("public source ", "archive"),
    marker("boundary ", "snapshot"),
    marker("boundary ", "archive"),
    marker("P11 ", "SDK ", "bound", "ary"),
    marker("P11 ", "delivery decision"),
    marker("zero-state ", "SDK"),
    marker("docs/", "vn", "gard"),
    marker("docs\\", "vn", "gard"),
]

EXPECTED_CRATES = {"alphasync-core", "alphasync-runtime"}

PUBLIC_BINARY_ASSETS = {
    "docs/assets/vraxion-home-hero.jpg",
    "docs/assets/vraxion-wordmark.png",
    "docs/assets/fonts/geist-sans-variable.woff2",
    "docs/instnct/assets/instnct-hero-bg.png",
    "docs/instnct/assets/instnct-logo.png",
    "docs/instnct/assets/t1-reflex-bg.jpg",
    "docs/instnct/assets/vraxion-note-bg.png",
}

MAX_PUBLIC_BINARY_ASSET_BYTES = 4 * 1024 * 1024


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    files = tracked_files()
    file_set = {relative.replace("\\", "/") for relative in files}

    crate_root = ROOT / "crates"
    crates = {
        child.name
        for child in crate_root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    }
    if crates != EXPECTED_CRATES:
        failures.append(f"unexpected crate set: {sorted(crates)}")

    for relative in files:
        normalized = relative.replace("\\", "/")
        lower = normalized.lower()
        if normalized.endswith(marker(".", "json", "l")):
            failures.append(f"forbidden log suffix: {relative}")
        for part in FORBIDDEN_PATH_PARTS:
            if part.lower() in lower:
                failures.append(f"forbidden path marker {part!r}: {relative}")

        path = ROOT / relative
        if path.is_file():
            if normalized in PUBLIC_BINARY_ASSETS:
                if path.stat().st_size > MAX_PUBLIC_BINARY_ASSET_BYTES:
                    failures.append(f"public binary asset too large: {relative}")
                continue
            text = read_text(path)
            for needle in FORBIDDEN_TEXT:
                if needle in text:
                    failures.append(f"forbidden text marker {needle!r}: {relative}")
            if (
                normalized not in LEGAL_TEXT_FILES
                and normalized.endswith(PUBLIC_COPY_SUFFIXES)
            ):
                lower_text = text.lower()
                for needle in FORBIDDEN_PUBLIC_COPY:
                    if needle.lower() in lower_text:
                        failures.append(f"forbidden public copy {needle!r}: {relative}")

    version_path = ROOT / "docs" / "VERSION.json"
    try:
        version = json.loads(read_text(version_path))
    except Exception as exc:  # noqa: BLE001 - audit should report the parse failure.
        failures.append(f"invalid docs/VERSION.json: {exc}")
        version = {}

    latest_release = version.get("latest_public_release")
    if not isinstance(latest_release, str) or not latest_release.startswith("public-sdk-"):
        failures.append(f"docs/VERSION.json latest_public_release is invalid: {latest_release!r}")

    index_html = read_text(ROOT / "docs" / "index.html")
    expected_release_url = "https://github.com/VRAXION/VRAXION/releases/tag/" + str(
        latest_release or ""
    )
    if latest_release and expected_release_url not in index_html:
        failures.append("docs/index.html does not link to the latest public release")

    for match in re.finditer(
        r"https://github\.com/VRAXION/VRAXION/blob/main/([^\"#?]+)",
        index_html,
    ):
        linked_path = match.group(1)
        if linked_path not in file_set:
            failures.append(f"docs/index.html links to missing repo path: {linked_path}")

    print("PUBLIC_SURFACE_AUDIT")
    print(f"tracked_files={len(files)}")
    print(f"failure_count={len(failures)}")
    print(f"warning_count={len(warnings)}")
    for failure in failures:
        print(f"failure: {failure}")
    for warning in warnings:
        print(f"warning: {warning}")
    if failures:
        return 1
    print("public_surface_audit=pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
