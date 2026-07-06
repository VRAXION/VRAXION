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
    marker("docs/", "van", "guard"),
    marker("docs/", "vn", "gard"),
    marker("red", "b"),
]

REQUIRED_FORBIDDEN_PATH_PARTS = {
    marker("docs/", "van", "guard"),
    marker("tools/", "private_data_adapters"),
}

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

FORBIDDEN_TEXT_REGEX = [
    (
        "absolute_drive_path",
        re.compile(r"(?:^|[^A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'<>|]+"),
    ),
    (
        "unc_local_path",
        re.compile(r"\\\\[A-Za-z0-9_.-]+\\[A-Za-z0-9_.-]+"),
    ),
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
    marker("SDK ", "bound", "ary"),
    marker("P11 ", "SDK ", "bound", "ary"),
    marker("P11 ", "delivery decision"),
    marker("zero-state ", "SDK"),
    marker("docs/", "vn", "gard"),
    marker("docs\\", "vn", "gard"),
    marker("van", "guard"),
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

REQUIRED_FORBIDDEN_PUBLIC_COPY_MARKERS = {
    marker("source", "-available"),
    marker("source ", "available"),
    marker("source ", "archive"),
    marker("public source ", "archive"),
    marker("boundary ", "archive"),
    marker("boundary ", "snapshot"),
    marker("SDK ", "bound", "ary"),
    marker("van", "guard"),
    "hosted api / saas later",
    "hosted api/saas later",
    "preview live",
    "signed offline verification",
}

EXPECTED_CRATES = {"alphasync-core", "alphasync-runtime"}

REQUIRED_TRACKED_FILES = {
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/public-surface-report.yml",
    ".github/pull_request_template.md",
    ".github/workflows/ci.yml",
    ".github/workflows/deploy-instnct-notify.yml",
    ".github/workflows/public-pages-smoke.yml",
    ".github/workflows/public-surface-audit.yml",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "DEPLOYMENT.md",
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "README.md",
    "PUBLIC_BOUNDARY.md",
    "PACKAGE_BOUNDARY.md",
    "PUBLIC_DELIVERY_MODEL.md",
    "PUBLIC_GITHUB_STATE.md",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "releases/README.md",
    "releases/public-sdk-p11-20260629.manifest.json",
    "releases/public-release-manifest.example.json",
    "releases/public-release-manifest.schema.json",
    "scripts/audit_public_secrets.mjs",
    "scripts/audit_public_github_state.mjs",
    "scripts/validate_public_release_manifests.mjs",
    "scripts/validate_public_release_state.mjs",
    "scripts/sync_public_release_links.mjs",
    "LICENSE_BOUNDARY.md",
    "SECURITY.md",
    "SUPPORT.md",
    "TRADEMARK_POLICY.md",
    "docs/.well-known/security.txt",
}

REQUIRED_CHANGELOG_MARKERS = {
    "## 2026-07-06",
    "release-link sync coverage",
    "contributor gates",
    "security policy",
    "deployment runbook",
    "generated Wrangler config",
    ".dev.vars",
    "Worker secrets",
    "Worker local config hygiene",
    "operator config and export/delete output",
    "workflow hygiene",
    "CI concurrency controls",
    "job timeouts",
    "public export guard",
    "live Pages state",
    "main GitHub Actions",
    "security.txt endpoint",
    "vulnerability disclosure routing",
}

REQUIRED_SECURITY_TXT_MARKERS = {
    "Contact: https://github.com/VRAXION/VRAXION/security/policy",
    "Policy: https://github.com/VRAXION/VRAXION/security/policy",
    "Preferred-Languages: en, hu",
    "Expires: 2027-01-06T00:00:00Z",
    "Canonical: https://vraxion.github.io/VRAXION/.well-known/security.txt",
}

REQUIRED_CITATION_MARKERS = {
    "cff-version: 1.2.0",
    "If you use this public SDK/docs release, cite VRAXION.",
    'title: "VRAXION Public SDK"',
    'repository-code: "https://github.com/VRAXION/VRAXION"',
    'license: "LicenseRef-VRAXION-Community-Source-1.0"',
    'date-released: "2026-06-28"',
}

REQUIRED_PR_TEMPLATE_MARKERS = {
    "PUBLIC_GITHUB_STATE.md",
    "What exactly becomes public?",
    "What stays private?",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "absolute local or UNC paths",
    "artifact_release",
    "required published artifact",
    "Release manifest checked:",
    "releases/public-release-manifest.schema.json",
    "node scripts\\validate_public_release_manifests.mjs",
    "node scripts\\validate_public_release_state.mjs",
    "node scripts\\audit_public_secrets.mjs",
    "workers/instnct-notify/wrangler.jsonc",
    "powershell -ExecutionPolicy Bypass -File scripts\\check_public_export.ps1",
}

REQUIRED_CONTRIBUTING_MARKERS = {
    "PUBLIC_RELEASE_CHECKLIST.md",
    "SECURITY.md",
    "SUPPORT.md",
    "node scripts\\sync_public_release_links.mjs --check",
    "node scripts\\validate_public_release_manifests.mjs",
    "node scripts\\validate_public_release_state.mjs",
    "node scripts\\audit_public_secrets.mjs",
    "node scripts\\audit_instnct_static_site.mjs",
    "node scripts\\audit_instnct_notify_worker.mjs",
    "python scripts\\audit_public_surface.py",
    "node scripts\\smoke_public_pages_links.mjs",
    "node scripts\\audit_public_github_state.mjs",
    "cargo fmt --all -- --check",
    "cargo test --workspace --all-features",
    "cargo test --doc --workspace --all-features",
    "cargo clippy --workspace --all-targets --all-features -- -D warnings",
    "powershell -ExecutionPolicy Bypass -File scripts/check_public_export.ps1",
    "non-public training data",
    "absolute local or UNC paths",
}

REQUIRED_SECURITY_MARKERS = {
    "GitHub security advisory",
    "open a public issue",
    "vulnerabilities, suspected secret exposure",
    "private material to explain the impact",
    "Public Scope",
    "non-public training data",
    "raw operator output",
    "absolute local or UNC machine",
    "production config",
    "private dashboards",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "node scripts\\audit_public_secrets.mjs",
    "node scripts\\audit_public_github_state.mjs",
    "before opening the final PR and",
    "again after merge",
}

REQUIRED_DEPLOYMENT_MARKERS = {
    "Public deployment runbook",
    "docs/index.html",
    "docs/instnct/",
    "node scripts\\sync_public_release_links.mjs --check",
    "node scripts\\validate_public_release_manifests.mjs",
    "node scripts\\validate_public_release_state.mjs",
    "node scripts\\audit_public_secrets.mjs",
    "node scripts\\audit_instnct_static_site.mjs",
    "python scripts\\audit_public_surface.py",
    "node scripts\\smoke_instnct_browser.mjs",
    "node scripts\\smoke_public_pages_links.mjs",
    "powershell -ExecutionPolicy Bypass -File scripts\\check_public_export.ps1",
    "Deploy INSTNCT Notify Worker",
    "workers/instnct-notify/wrangler.example.jsonc",
    "workers/instnct-notify/wrangler.jsonc",
    ".dev.vars",
    "CLOUDFLARE_API_TOKEN",
    "INSTNCT_NOTIFY_D1_DATABASE_ID",
    "INSTNCT_NOTIFY_EMAIL_HASH_PEPPER",
    "INSTNCT_NOTIFY_ADMIN_TOKEN",
    "scripts\\smoke_instnct_notify_live.mjs",
    "node scripts\\audit_public_github_state.mjs",
    "Do not add an active email form",
    "connect-src 'none'; form-action 'none'",
    "link-only release tracking block",
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

REQUIRED_CODEOWNERS_MARKERS = {
    "* @Kenessy",
    "/.github/ @Kenessy",
    "/scripts/ @Kenessy",
    "/docs/ @Kenessy",
    "/workers/instnct-notify/ @Kenessy",
    "/crates/ @Kenessy",
    "/PUBLIC_*.md @Kenessy",
    "/docs/VERSION.json @Kenessy",
}

REQUIRED_GITHUB_STATE_MARKERS = {
    "## Current Public Truth",
    "## Historical GitHub Records",
    ".github/CODEOWNERS",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "docs/VERSION.json",
    "gh release list --limit 20",
    "latest public release: `public-sdk-p11-20260629`",
    "release manifest: `releases/public-sdk-p11-20260629.manifest.json`",
    "non-public training data",
    "node scripts\\audit_public_github_state.mjs",
    "latest Pages build",
    "public manifests and GitHub",
    "release metadata and assets",
    "releases/public-release-manifest.schema.json",
    "node scripts\\validate_public_release_state.mjs",
    "node scripts\\audit_public_secrets.mjs",
}

REQUIRED_RELEASE_MANIFEST_README_MARKERS = {
    "artifact_release",
    "public-sdk-p11-20260629.manifest.json",
    "public-release-manifest.example.json",
    "public-release-manifest.schema.json",
    "releases/<release-slug>.manifest.json",
    "node scripts\\validate_public_release_manifests.mjs",
    "published `proof_pack`",
    "published non-documentation artifact",
    "policy_self_tests",
    "policy self-tests",
    "signature_path_or_url",
    "schema contract",
    "private engine source",
    "non-public training data",
    "raw operator output",
}

REQUIRED_RELEASE_LINK_SYNC_MARKERS = {
    "requiredCurrentStateFiles",
    "README.md",
    "PUBLIC_GITHUB_STATE.md",
    "docs/CURRENT_STATUS.md",
    "docs/CURRENT_CAPABILITIES.md",
    "current-state release link file is not scanned",
    "public_release_link_files",
}

REQUIRED_RELEASE_MANIFEST_EXCLUSIONS = {
    "private_engine_source",
    "non_public_training_data",
    "raw_operator_output",
    "local_machine_paths",
    "secrets_or_tokens",
    "filled_production_config",
    "private_dashboards",
}

REQUIRED_ISSUE_TEMPLATE_MARKERS = {
    "absolute local or UNC machine paths",
    "Do not paste secrets",
    "SECURITY.md",
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
    "absolute local or UNC machine paths",
}

REQUIRED_WORKFLOW_PERMISSION_MARKERS = {
    ".github/workflows/ci.yml": "permissions:\n  contents: read",
    ".github/workflows/deploy-instnct-notify.yml": "permissions:\n  contents: read",
    ".github/workflows/public-pages-smoke.yml": "permissions:\n  contents: read",
    ".github/workflows/public-surface-audit.yml": "permissions:\n  contents: read",
}

REQUIRED_WORKFLOW_OPERATIONAL_MARKERS = {
    ".github/workflows/ci.yml": {
        "concurrency:",
        "group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}",
        "cancel-in-progress: true",
        "timeout-minutes: 20",
        "timeout-minutes: 45",
    },
    ".github/workflows/public-surface-audit.yml": {
        "concurrency:",
        "group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}",
        "cancel-in-progress: true",
        "timeout-minutes: 10",
    },
    ".github/workflows/public-pages-smoke.yml": {
        "concurrency:",
        "group: public-pages-smoke-${{ github.ref }}",
        "cancel-in-progress: true",
        "timeout-minutes: 10",
    },
    ".github/workflows/deploy-instnct-notify.yml": {
        "concurrency:",
        "group: deploy-instnct-notify-production",
        "cancel-in-progress: false",
        "timeout-minutes: 20",
    },
}

REQUIRED_PUBLIC_SURFACE_AUDIT_WORKFLOW_MARKERS = {
    "Validate public release manifests",
    "Validate public release state",
    "Sync public release links",
    "node scripts/sync_public_release_links.mjs --check",
    "Audit public secrets",
    "Audit public surface",
}

FORBIDDEN_WORKFLOW_PERMISSION_MARKERS = {
    "actions: write",
    "checks: write",
    "contents: write",
    "deployments: write",
    "discussions: write",
    "id-token: write",
    "issues: write",
    "packages: write",
    "pages: write",
    "pull-requests: write",
    "repository-projects: write",
    "security-events: write",
    "statuses: write",
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
    ".codex/",
    ".dev.vars",
    ".dev.vars.*",
    ".env",
    ".env.*",
    ".envrc",
    ".wrangler/",
    "!.env.example",
    "*.crt",
    "*.csr",
    "*.db",
    "*.db-*",
    "*.key",
    "*.log",
    "*.p12",
    "*.pem",
    "*.pfx",
    "*.py[cod]",
    "*.sqlite",
    "*.sqlite3",
    "*.tmp",
    "*$py.class",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".venv/",
    "coverage/",
    "disabled-surfaces/",
    "dist/",
    "node_modules/",
    "target/",
    "venv/",
    "__pycache__/",
    "workers/instnct-notify/wrangler.jsonc",
    "workers/instnct-notify/wrangler.toml",
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
    forbidden_public_copy_markers = {
        marker_text.lower() for marker_text in FORBIDDEN_PUBLIC_COPY
    }
    forbidden_path_parts = {path_part.lower() for path_part in FORBIDDEN_PATH_PARTS}

    for required in sorted(REQUIRED_FORBIDDEN_PATH_PARTS):
        if required.lower() not in forbidden_path_parts:
            failures.append(f"forbidden path marker is not guarded: {required}")
    for required in sorted(REQUIRED_FORBIDDEN_PUBLIC_COPY_MARKERS):
        if required.lower() not in forbidden_public_copy_markers:
            failures.append(f"forbidden public copy marker is not guarded: {required}")

    for required in sorted(REQUIRED_TRACKED_FILES):
        if required not in file_set:
            failures.append(f"required public repo file is missing: {required}")

    changelog = read_text(ROOT / "CHANGELOG.md")
    for required in sorted(REQUIRED_CHANGELOG_MARKERS):
        if required not in changelog:
            failures.append(f"changelog missing public hardening marker: {required}")

    citation = read_text(ROOT / "CITATION.cff")
    for required in sorted(REQUIRED_CITATION_MARKERS):
        if required not in citation:
            failures.append(f"citation metadata missing marker: {required}")

    pr_template = read_text(ROOT / ".github" / "pull_request_template.md")
    for required in sorted(REQUIRED_PR_TEMPLATE_MARKERS):
        if required not in pr_template:
            failures.append(f"pull request template missing release guard marker: {required}")

    contributing_doc = read_text(ROOT / "CONTRIBUTING.md")
    for required in sorted(REQUIRED_CONTRIBUTING_MARKERS):
        if required not in contributing_doc:
            failures.append(f"contributing doc missing public-gate marker: {required}")

    security_doc = read_text(ROOT / "SECURITY.md")
    for required in sorted(REQUIRED_SECURITY_MARKERS):
        if required not in security_doc:
            failures.append(f"security doc missing public-security marker: {required}")

    security_txt = read_text(ROOT / "docs" / ".well-known" / "security.txt")
    for required in sorted(REQUIRED_SECURITY_TXT_MARKERS):
        if required not in security_txt:
            failures.append(f"security.txt missing public disclosure marker: {required}")

    deployment_doc = read_text(ROOT / "DEPLOYMENT.md")
    for required in sorted(REQUIRED_DEPLOYMENT_MARKERS):
        if required not in deployment_doc:
            failures.append(f"deployment doc missing public-deployment marker: {required}")

    dependabot_config = read_text(ROOT / ".github" / "dependabot.yml")
    for required in sorted(REQUIRED_DEPENDABOT_MARKERS):
        if required not in dependabot_config:
            failures.append(f"dependabot config missing public dependency marker: {required}")

    codeowners = read_text(ROOT / ".github" / "CODEOWNERS")
    for required in sorted(REQUIRED_CODEOWNERS_MARKERS):
        if required not in codeowners:
            failures.append(f"CODEOWNERS missing public ownership marker: {required}")

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

    for relative_path, required in sorted(REQUIRED_WORKFLOW_PERMISSION_MARKERS.items()):
        workflow_text = read_text(ROOT / relative_path)
        if required not in workflow_text:
            failures.append(f"workflow missing read-only permission marker: {relative_path}")
        workflow_lower = workflow_text.lower()
        for forbidden in sorted(FORBIDDEN_WORKFLOW_PERMISSION_MARKERS):
            if forbidden in workflow_lower:
                failures.append(
                    f"workflow contains forbidden write permission {forbidden!r}: {relative_path}"
                )

    for relative_path, markers in sorted(REQUIRED_WORKFLOW_OPERATIONAL_MARKERS.items()):
        workflow_text = read_text(ROOT / relative_path)
        for required in sorted(markers):
            if required not in workflow_text:
                failures.append(
                    f"workflow missing operational hygiene marker {required!r}: {relative_path}"
                )

    public_surface_workflow = read_text(
        ROOT / ".github" / "workflows" / "public-surface-audit.yml"
    )
    for required in sorted(REQUIRED_PUBLIC_SURFACE_AUDIT_WORKFLOW_MARKERS):
        if required not in public_surface_workflow:
            failures.append(f"public surface audit workflow missing marker: {required}")

    release_manifest_readme = read_text(ROOT / "releases" / "README.md")
    for required in sorted(REQUIRED_RELEASE_MANIFEST_README_MARKERS):
        if required not in release_manifest_readme:
            failures.append(f"release manifest readme missing marker: {required}")

    release_link_sync = read_text(ROOT / "scripts" / "sync_public_release_links.mjs")
    for required in sorted(REQUIRED_RELEASE_LINK_SYNC_MARKERS):
        if required not in release_link_sync:
            failures.append(f"release link sync missing coverage marker: {required}")

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
            for label, pattern in FORBIDDEN_TEXT_REGEX:
                match = pattern.search(text)
                if match:
                    failures.append(
                        f"forbidden text regex {label!r}: {relative}"
                    )
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

    schema_path = ROOT / "releases" / "public-release-manifest.schema.json"
    example_path = ROOT / "releases" / "public-release-manifest.example.json"
    try:
        release_manifest_schema = json.loads(read_text(schema_path))
    except Exception as exc:  # noqa: BLE001 - audit should report the parse failure.
        failures.append(f"invalid public release manifest schema: {exc}")
        release_manifest_schema = {}

    try:
        release_manifest_example = json.loads(read_text(example_path))
    except Exception as exc:  # noqa: BLE001 - audit should report the parse failure.
        failures.append(f"invalid public release manifest example: {exc}")
        release_manifest_example = {}

    manifest_const = (
        release_manifest_schema.get("properties", {})
        .get("schema", {})
        .get("const")
    )
    if manifest_const != "vraxion.public.release-manifest.v1":
        failures.append("release manifest schema const is not vraxion.public.release-manifest.v1")
    if release_manifest_example.get("schema") != "vraxion.public.release-manifest.v1":
        failures.append("release manifest example uses the wrong schema id")
    example_slug = release_manifest_example.get("release_slug")
    if not isinstance(example_slug, str) or not re.fullmatch(
        r"public-[a-z0-9][a-z0-9-]*-[0-9]{8}",
        example_slug,
    ):
        failures.append(f"release manifest example slug is invalid: {example_slug!r}")
    if not isinstance(release_manifest_example.get("artifacts"), list):
        failures.append("release manifest example artifacts must be a list")

    schema_exclusions = (
        release_manifest_schema.get("properties", {})
        .get("exclusions", {})
        .get("properties", {})
    )
    example_exclusions = release_manifest_example.get("exclusions")
    if not isinstance(example_exclusions, dict):
        failures.append("release manifest example exclusions must be an object")
        example_exclusions = {}
    for exclusion in sorted(REQUIRED_RELEASE_MANIFEST_EXCLUSIONS):
        if schema_exclusions.get(exclusion, {}).get("const") is not False:
            failures.append(f"release manifest schema exclusion is not const false: {exclusion}")
        if example_exclusions.get(exclusion) is not False:
            failures.append(f"release manifest example exclusion is not false: {exclusion}")

    verification = release_manifest_example.get("verification")
    commands = verification.get("commands") if isinstance(verification, dict) else None
    if not isinstance(commands, list) or not any(
        "scripts\\audit_public_github_state.mjs" in command for command in commands
    ):
        failures.append("release manifest example must include the live GitHub state audit command")
    if not isinstance(commands, list) or not any(
        "scripts\\check_public_export.ps1" in command for command in commands
    ):
        failures.append("release manifest example must include the public export guard command")
    if not isinstance(commands, list) or not any(
        "scripts\\validate_public_release_manifests.mjs" in command
        for command in commands
    ):
        failures.append("release manifest example must include the release manifest validator command")
    if not isinstance(commands, list) or not any(
        "scripts\\validate_public_release_state.mjs" in command
        for command in commands
    ):
        failures.append("release manifest example must include the release state validator command")
    if not isinstance(commands, list) or not any(
        "scripts\\audit_public_secrets.mjs" in command for command in commands
    ):
        failures.append("release manifest example must include the public secret scan command")
    if not isinstance(commands, list) or not any(
        "scripts\\smoke_public_pages_links.mjs" in command for command in commands
    ):
        failures.append("release manifest example must include the live public Pages smoke command")

    index_html = read_text(ROOT / "docs" / "index.html")
    expected_release_url = "https://github.com/VRAXION/VRAXION/releases/tag/" + str(
        latest_release or ""
    )
    if latest_release and expected_release_url not in index_html:
        failures.append("docs/index.html does not link to the latest public release")

    latest_manifest_path = f"releases/{latest_release}.manifest.json"
    if latest_release and latest_manifest_path not in file_set:
        failures.append(f"latest public release manifest is missing: {latest_manifest_path}")

    for match in re.finditer(
        r"https://github\.com/VRAXION/VRAXION/blob/main/([^\"#?]+)",
        index_html,
    ):
        linked_path = match.group(1)
        if linked_path not in file_set:
            failures.append(f"docs/index.html links to missing repo path: {linked_path}")

    print("PUBLIC_SURFACE_AUDIT")
    print(f"tracked_files={len(files)}")
    print(f"forbidden_path_markers={len(FORBIDDEN_PATH_PARTS)}")
    print(f"forbidden_public_copy_markers={len(FORBIDDEN_PUBLIC_COPY)}")
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
