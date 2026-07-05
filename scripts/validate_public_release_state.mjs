#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const VERSION_PATH = "docs/VERSION.json";
const VERSION_SCHEMA = "vraxion.public.version.v1";
const RELEASE_SLUG_RE = /^public-sdk-p[0-9]+-[0-9]{8}$/;
const DATE_RE = /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/;
const SIMPLE_VERSION_RE = /^[a-z0-9][a-z0-9._-]*$/;
const PAGES_URL = "https://vraxion.github.io/VRAXION/";

const REQUIRED_VERSION_KEYS = [
  "schema",
  "status",
  "date",
  "public_surface",
  "pages_surface",
  "latest_public_release",
  "home_asset_version",
  "instnct_asset_version",
  "anchorcell_asset_version",
  "delivery_direction",
  "crates",
];
const REQUIRED_RELEASE_REFERENCES = [
  "README.md",
  "PUBLIC_GITHUB_STATE.md",
  "docs/CURRENT_STATUS.md",
  "docs/CURRENT_CAPABILITIES.md",
  "docs/index.html",
];
const REQUIRED_PUBLIC_STATE_MARKERS = [
  "default branch: `main`",
  "version record: `docs/VERSION.json`",
  `Pages URL: \`${PAGES_URL}\``,
  "gh release list --limit 20",
  "node scripts\\validate_public_release_state.mjs",
];

const failures = [];

function fail(message) {
  failures.push(message);
}

function readText(relativePath) {
  return fs.readFileSync(path.join(ROOT, relativePath), "utf8");
}

function readJson(relativePath) {
  try {
    return JSON.parse(readText(relativePath));
  } catch (error) {
    fail(`${relativePath}: invalid JSON: ${error.message}`);
    return null;
  }
}

function assertPlainObject(label, value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(`${label}: must be an object`);
    return false;
  }
  return true;
}

function assertRealDate(label, value) {
  if (typeof value !== "string" || !DATE_RE.test(value)) {
    fail(`${label}: invalid date string: ${value}`);
    return;
  }
  const parsed = new Date(`${value}T00:00:00.000Z`);
  if (Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== value) {
    fail(`${label}: not a real calendar date: ${value}`);
  }
}

function assertSimpleField(label, value) {
  if (typeof value !== "string" || !SIMPLE_VERSION_RE.test(value)) {
    fail(`${label}: must be a non-empty simple identifier`);
  }
}

function validateVersionRecord(version) {
  if (!assertPlainObject(VERSION_PATH, version)) {
    return null;
  }
  const expected = new Set(REQUIRED_VERSION_KEYS);
  for (const key of Object.keys(version)) {
    if (!expected.has(key)) {
      fail(`${VERSION_PATH}: unsupported key: ${key}`);
    }
  }
  for (const key of REQUIRED_VERSION_KEYS) {
    if (!(key in version)) {
      fail(`${VERSION_PATH}: missing required key: ${key}`);
    }
  }

  if (version.schema !== VERSION_SCHEMA) {
    fail(`${VERSION_PATH}: schema must be ${VERSION_SCHEMA}`);
  }
  assertSimpleField(`${VERSION_PATH}: status`, version.status);
  assertRealDate(`${VERSION_PATH}: date`, version.date);
  assertSimpleField(`${VERSION_PATH}: public_surface`, version.public_surface);
  assertSimpleField(`${VERSION_PATH}: pages_surface`, version.pages_surface);
  assertSimpleField(`${VERSION_PATH}: home_asset_version`, version.home_asset_version);
  assertSimpleField(`${VERSION_PATH}: instnct_asset_version`, version.instnct_asset_version);
  assertSimpleField(`${VERSION_PATH}: anchorcell_asset_version`, version.anchorcell_asset_version);
  assertSimpleField(`${VERSION_PATH}: delivery_direction`, version.delivery_direction);

  if (typeof version.latest_public_release !== "string" || !RELEASE_SLUG_RE.test(version.latest_public_release)) {
    fail(`${VERSION_PATH}: latest_public_release is invalid: ${version.latest_public_release}`);
  }

  if (!Array.isArray(version.crates) || version.crates.length === 0) {
    fail(`${VERSION_PATH}: crates must be a non-empty array`);
  } else {
    const crateSet = new Set();
    for (const crateName of version.crates) {
      if (typeof crateName !== "string" || !/^[a-z0-9][a-z0-9-]*$/.test(crateName)) {
        fail(`${VERSION_PATH}: invalid crate name: ${crateName}`);
      }
      if (crateSet.has(crateName)) {
        fail(`${VERSION_PATH}: duplicate crate name: ${crateName}`);
      }
      crateSet.add(crateName);
    }
    const crateDirs = fs
      .readdirSync(path.join(ROOT, "crates"), { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort();
    const listedCrates = [...crateSet].sort();
    if (JSON.stringify(crateDirs) !== JSON.stringify(listedCrates)) {
      fail(`${VERSION_PATH}: crates ${listedCrates.join(", ")} do not match crates/ directories ${crateDirs.join(", ")}`);
    }
    for (const crateName of listedCrates) {
      const crateManifest = `crates/${crateName}/Cargo.toml`;
      if (!fs.existsSync(path.join(ROOT, crateManifest))) {
        fail(`${VERSION_PATH}: listed crate is missing Cargo.toml: ${crateManifest}`);
      }
    }
  }

  return version.latest_public_release;
}

function validateReleaseReferences(latestPublicRelease) {
  if (!latestPublicRelease || !RELEASE_SLUG_RE.test(latestPublicRelease)) {
    return;
  }
  const releaseUrl = `https://github.com/VRAXION/VRAXION/releases/tag/${latestPublicRelease}`;

  for (const relativePath of REQUIRED_RELEASE_REFERENCES) {
    const text = readText(relativePath);
    if (!text.includes(latestPublicRelease)) {
      fail(`${relativePath}: missing latest public release slug ${latestPublicRelease}`);
    }
    if (!text.includes(releaseUrl)) {
      fail(`${relativePath}: missing latest public release URL ${releaseUrl}`);
    }
  }

  const publicState = readText("PUBLIC_GITHUB_STATE.md");
  for (const marker of REQUIRED_PUBLIC_STATE_MARKERS) {
    if (!publicState.includes(marker)) {
      fail(`PUBLIC_GITHUB_STATE.md: missing public-state marker: ${marker}`);
    }
  }

  const latestManifestPath = `releases/${latestPublicRelease}.manifest.json`;
  const latestManifestAbsolutePath = path.join(ROOT, latestManifestPath);
  if (!fs.existsSync(latestManifestAbsolutePath)) {
    fail(`${latestManifestPath}: latest public release manifest is missing`);
  } else {
    const latestManifestText = readText(latestManifestPath);
    if (!latestManifestText.includes(latestPublicRelease)) {
      fail(`${latestManifestPath}: missing latest public release slug ${latestPublicRelease}`);
    }
    if (!latestManifestText.includes(releaseUrl)) {
      fail(`${latestManifestPath}: missing latest public release URL ${releaseUrl}`);
    }
  }

  const indexHtml = readText("docs/index.html");
  if (!indexHtml.includes(`<strong>${latestPublicRelease}</strong>`)) {
    fail("docs/index.html: current artifact chip does not render the latest public release slug");
  }
}

const version = readJson(VERSION_PATH);
const latestPublicRelease = validateVersionRecord(version);
validateReleaseReferences(latestPublicRelease);

console.log("PUBLIC_RELEASE_STATE_VALIDATION");
console.log(`latest_public_release=${latestPublicRelease || "unknown"}`);
console.log(`failure_count=${failures.length}`);
for (const failure of failures) {
  console.log(`failure: ${failure}`);
}
if (failures.length > 0) {
  process.exit(1);
}
console.log("public_release_state_validation=pass");
