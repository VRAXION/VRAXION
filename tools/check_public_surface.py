"""Lightweight public-surface consistency checks for CI."""

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

MARKDOWN_FILES = [README, V42_README, CONTRIBUTING, FINDINGS]
REQUIRED_LOCAL_LINK_TARGETS = {
    README: ["VALIDATED_FINDINGS.md", "v4.2/README.md"],
    V42_README: ["../VALIDATED_FINDINGS.md"],
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
    required = ["Current mainline", "Validated finding", "Experimental branch"]
    for term in required:
        if term not in text:
            fail(f"{path.name}: missing required taxonomy term {term!r}", errors)


def check_links(path: Path, text: str, errors: list[str]) -> None:
    for rel_target in REQUIRED_LOCAL_LINK_TARGETS.get(path, []):
        target = (path.parent / rel_target).resolve()
        if not target.exists():
            fail(f"{path.name}: required link target does not exist: {rel_target}", errors)

    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
        href = match.group(1).strip()
        if not href or href.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target = (path.parent / href).resolve()
        if not target.exists():
            fail(f"{path.name}: broken relative markdown link: {href}", errors)


def check_landing(text: str, threshold: str, inj_scale: str, errors: list[str]) -> None:
    for term in ["Current mainline", "Validated finding", "Experimental branch", "INSTNCT / SWG v4.2"]:
        if term not in text:
            fail(f"docs/index.html: missing required term {term!r}", errors)
    if f"`THRESHOLD = {threshold}`" not in read(FINDINGS):
        fail("VALIDATED_FINDINGS.md: threshold line is not aligned to graph.py", errors)
    if f"`INJ_SCALE = {inj_scale}`" not in read(FINDINGS):
        fail("VALIDATED_FINDINGS.md: INJ_SCALE line is not aligned to graph.py", errors)


def main() -> int:
    errors: list[str] = []

    graph_text = read(GRAPH)
    threshold = extract_constant("THRESHOLD", graph_text)
    inj_scale = extract_constant("INJ_SCALE", graph_text)

    for path in MARKDOWN_FILES:
        text = read(path)
        if path in {README, V42_README, FINDINGS}:
            check_required_terms(path, text, errors)
        check_links(path, text, errors)

    landing_text = read(LANDING)
    check_landing(landing_text, threshold, inj_scale, errors)

    if "public technical repo" not in read(CONTRIBUTING):
        fail("CONTRIBUTING.md: expected public-technical-repo posture text", errors)

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
