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
    marker("open ", "source"),
    marker("open-", "source"),
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
    "local reasoning runtime",
    "runtime project",
    "governed runtime frame",
    "founder-led runtime work",
    "vraxion runtime principles",
    "preview live",
    "signed offline verification",
    "hosted api / saas later",
    "hosted api/saas later",
    "skill libraries",
    "make local intelligence",
    "production backend target",
]

EXPECTED_CRATES = {"alphasync-core", "alphasync-runtime"}

REQUIRED_TRACKED_FILES = {
    ".github/dependabot.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/public-surface-report.yml",
    ".github/pull_request_template.md",
    ".gitattributes",
    ".gitignore",
    "README.md",
    "PUBLIC_BOUNDARY.md",
    "PACKAGE_BOUNDARY.md",
    "PUBLIC_DELIVERY_MODEL.md",
    "PUBLIC_GITHUB_STATE.md",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "LICENSE_BOUNDARY.md",
    "SUPPORT.md",
}

REQUIRED_PR_TEMPLATE_MARKERS = {
    "PUBLIC_GITHUB_STATE.md",
    "What exactly becomes public?",
    "What stays private?",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "workers/instnct-notify/wrangler.jsonc",
    "powershell -ExecutionPolicy Bypass -File scripts\\check_public_export.ps1",
}

REQUIRED_DEPENDABOT_MARKERS = {
    "directory: \"/\"",
    "github-actions",
    "open-pull-requests-limit: 5",
    "package-ecosystem: \"cargo\"",
    "public-github-actions",
    "public-surface",
    "rust-public-dependencies",
    "timezone: \"Europe/Budapest\"",
}

REQUIRED_GITHUB_STATE_MARKERS = {
    "## Current Public Truth",
    "## Historical GitHub Records",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "docs/VERSION.json",
    "gh release list --limit 20",
    "latest public release: `public-sdk-p11-20260629`",
    "non-public training data",
    "public manifests and GitHub",
    "release metadata and assets",
}

REQUIRED_ISSUE_TEMPLATE_MARKERS = {
    "Do not paste secrets",
    "SECURITY.md",
    "local machine paths",
    "non-public training data",
    "public surface",
    "raw operator output",
}

REQUIRED_ISSUE_CONFIG_MARKERS = {
    "PUBLIC_RELEASE_CHECKLIST.md",
    "blank_issues_enabled: false",
    "security/policy",
}

REQUIRED_SUPPORT_MARKERS = {
    "## Do Not Include",
    "## Use Public Issues For",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "SECURITY.md",
    "non-public training data",
    "public files and public releases",
    "raw operator output",
}

REQUIRED_GITATTRIBUTES_ENTRIES = {
    "* text=auto eol=lf",
    "*.jpg binary",
    "*.jpeg binary",
    "*.png binary",
    "*.webp binary",
    "*.woff2 binary",
}

REQUIRED_GITIGNORE_ENTRIES = {
    "target/",
    ".codex/",
    "disabled-surfaces/",
    ".env",
    ".env.*",
    "!.env.example",
    "workers/instnct-notify/wrangler.jsonc",
}

PUBLIC_BINARY_ASSETS = {
    "docs/assets/vraxion-home-hero.jpg",
    "docs/assets/vraxion-home-hero.webp",
    "docs/assets/vraxion-wordmark.webp",
    "docs/assets/fonts/geist-sans-variable.woff2",
    "docs/instnct/assets/engine-scope-bg.jpg",
    "docs/instnct/assets/engine-scope-bg.webp",
    "docs/instnct/assets/exact-mode-bg.jpg",
    "docs/instnct/assets/exact-mode-bg.webp",
    "docs/instnct/assets/cli-proof-bg.jpg",
    "docs/instnct/assets/cli-proof-bg.webp",
    "docs/instnct/assets/constraints-founder-bg.jpg",
    "docs/instnct/assets/constraints-founder-bg.webp",
    "docs/instnct/assets/fabric-result-bg.jpg",
    "docs/instnct/assets/fabric-result-bg.webp",
    "docs/instnct/assets/instnct-hero-bg.jpg",
    "docs/instnct/assets/instnct-hero-bg.webp",
    "docs/instnct/assets/instnct-logo.webp",
    "docs/instnct/assets/proof-pack-bg.jpg",
    "docs/instnct/assets/proof-pack-bg.webp",
    "docs/instnct/assets/release-claim-bg.jpg",
    "docs/instnct/assets/release-claim-bg.webp",
    "docs/instnct/assets/t1-reflex-bg.jpg",
    "docs/instnct/assets/vraxion-note-bg.jpg",
    "docs/instnct/assets/vraxion-note-bg.webp",
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

    for required in sorted(REQUIRED_TRACKED_FILES):
        if required not in file_set:
            failures.append(f"required public repo file is missing: {required}")

    pr_template = read_text(ROOT / ".github" / "pull_request_template.md")
    for required in sorted(REQUIRED_PR_TEMPLATE_MARKERS):
        if required not in pr_template:
            failures.append(f"pull request template missing release guard marker: {required}")

    dependabot_config = read_text(ROOT / ".github" / "dependabot.yml")
    for required in sorted(REQUIRED_DEPENDABOT_MARKERS):
        if required not in dependabot_config:
            failures.append(f"dependabot config missing public dependency marker: {required}")

    github_state_doc = read_text(ROOT / "PUBLIC_GITHUB_STATE.md")
    for required in sorted(REQUIRED_GITHUB_STATE_MARKERS):
        if required not in github_state_doc:
            failures.append(f"github state doc missing public-state marker: {required}")

    issue_template = read_text(
        ROOT / ".github" / "ISSUE_TEMPLATE" / "public-surface-report.yml"
    )
    for required in sorted(REQUIRED_ISSUE_TEMPLATE_MARKERS):
        if required not in issue_template:
            failures.append(f"issue template missing public-safety marker: {required}")

    issue_config = read_text(ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml")
    for required in sorted(REQUIRED_ISSUE_CONFIG_MARKERS):
        if required not in issue_config:
            failures.append(f"issue template config missing intake marker: {required}")

    support_doc = read_text(ROOT / "SUPPORT.md")
    for required in sorted(REQUIRED_SUPPORT_MARKERS):
        if required not in support_doc:
            failures.append(f"support doc missing public-support marker: {required}")

    gitattributes_path = ROOT / ".gitattributes"
    gitattributes_entries = {
        line.strip()
        for line in read_text(gitattributes_path).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for required in sorted(REQUIRED_GITATTRIBUTES_ENTRIES):
        if required not in gitattributes_entries:
            failures.append(f".gitattributes missing public hygiene entry: {required}")

    gitignore_path = ROOT / ".gitignore"
    gitignore_entries = {
        line.strip()
        for line in read_text(gitignore_path).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for required in sorted(REQUIRED_GITIGNORE_ENTRIES):
        if required not in gitignore_entries:
            failures.append(f".gitignore missing public hygiene entry: {required}")

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
