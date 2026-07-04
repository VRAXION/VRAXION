#!/usr/bin/env python3
"""Validate the public zero-state repository surface."""

from __future__ import annotations

import pathlib
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

EXPECTED_CRATES = {"alphasync-core", "alphasync-runtime"}

PUBLIC_BINARY_ASSETS = {
    "docs/assets/vraxion-home-hero.jpg",
    "docs/assets/vraxion-wordmark.png",
    "docs/instnct/assets/instnct-hero-bg.png",
    "docs/instnct/assets/instnct-wordmark.png",
    "docs/instnct/assets/t1-reflex-bg.jpg",
    "docs/instnct/assets/vraxion-note-bg.png",
    "docs/vngard/assets/alpha-sync-fabric-card.jpg",
    "docs/vngard/assets/mutation-core-card.jpg",
    "docs/vngard/assets/prismion-atom-card.jpg",
    "docs/vngard/assets/vngard-guardian-hero-bg.png",
    "docs/vngard/assets/vngard-wordmark-logo.png",
    "docs/vngard/assets/fonts/geist-sans-variable.woff2",
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
