#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RELEASES_DIR = path.join(ROOT, "releases");
const SCHEMA_ID = "vraxion.public.release-manifest.v1";
const RELEASE_SLUG_RE = /^public-[a-z0-9][a-z0-9-]*-[0-9]{8}$/;
const RELEASE_DATE_RE = /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/;
const REPO_RELATIVE_RE = /^(?![A-Za-z]:)(?!\/)(?!.*\.\.)(?!.*\/\/)[A-Za-z0-9._/-]+$/;
const SHA256_RE = /^[a-f0-9]{64}$/;
const REVIEWER_RE = /^@[A-Za-z0-9-]+$/;

const RELEASE_KINDS = new Set([
  "docs_only",
  "sdk_docs",
  "proof_pack",
  "artifact_release",
]);
const ARTIFACT_KINDS = new Set([
  "documentation",
  "repository_zip",
  "proof_pack",
  "checksum",
  "signature",
  "binary",
  "other",
]);
const ARTIFACT_STATUSES = new Set([
  "published",
  "pending_review",
  "not_included",
]);
const REQUIRED_EXCLUSIONS = [
  "private_engine_source",
  "non_public_training_data",
  "raw_operator_output",
  "local_machine_paths",
  "secrets_or_tokens",
  "filled_production_config",
  "private_dashboards",
];
const REQUIRED_COMMANDS = [
  "node scripts\\audit_public_github_state.mjs",
  "node scripts\\sync_public_release_links.mjs --check",
  "node scripts\\validate_public_release_manifests.mjs",
  "node scripts\\validate_public_release_state.mjs",
  "node scripts\\audit_public_secrets.mjs",
  "python scripts\\audit_public_surface.py",
  "node scripts\\smoke_public_pages_links.mjs",
  "powershell -ExecutionPolicy Bypass -File scripts\\check_public_export.ps1",
];
const REQUIRED_GITHUB_CHECKS = [
  "Public Surface Audit",
  "Public SDK CI",
  "CodeQL",
  "Cloudflare Pages",
];
const PUBLISHED_ARTIFACT_KINDS_REQUIRING_SHA256 = new Set([
  "repository_zip",
  "proof_pack",
  "checksum",
  "signature",
  "binary",
  "other",
]);
const PUBLISHED_ARTIFACT_KINDS_REQUIRING_SIGNATURE = new Set([
  "proof_pack",
  "binary",
]);

const failures = [];
const trackedFiles = new Set(
  execFileSync("git", ["ls-files"], { cwd: ROOT, encoding: "utf8" })
    .split(/\r?\n/)
    .map((line) => line.trim().replaceAll("\\", "/"))
    .filter(Boolean),
);

function fail(file, message) {
  failures.push(`${file}: ${message}`);
}

function readJson(relativePath) {
  const absolutePath = path.join(ROOT, relativePath);
  try {
    return JSON.parse(fs.readFileSync(absolutePath, "utf8"));
  } catch (error) {
    fail(relativePath, `invalid JSON: ${error.message}`);
    return null;
  }
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function assertExactKeys(file, label, value, requiredKeys, optionalKeys = []) {
  if (!isPlainObject(value)) {
    fail(file, `${label} must be an object`);
    return false;
  }
  const expected = new Set([...requiredKeys, ...optionalKeys]);
  for (const key of Object.keys(value)) {
    if (!expected.has(key)) {
      fail(file, `${label} has unsupported key: ${key}`);
    }
  }
  for (const key of requiredKeys) {
    if (!(key in value)) {
      fail(file, `${label} missing required key: ${key}`);
    }
  }
  return true;
}

function assertUniqueStringArray(file, label, value, { minItems = 0 } = {}) {
  if (!Array.isArray(value)) {
    fail(file, `${label} must be an array`);
    return [];
  }
  if (value.length < minItems) {
    fail(file, `${label} must include at least ${minItems} item(s)`);
  }
  const seen = new Set();
  const strings = [];
  value.forEach((item, index) => {
    if (typeof item !== "string" || item.trim() === "") {
      fail(file, `${label}[${index}] must be a non-empty string`);
      return;
    }
    if (seen.has(item)) {
      fail(file, `${label} contains duplicate item: ${item}`);
    }
    seen.add(item);
    strings.push(item);
  });
  return strings;
}

function isRepoRelativePath(value) {
  if (typeof value !== "string" || value.trim() === "") {
    return false;
  }
  return REPO_RELATIVE_RE.test(value) && !value.split("/").includes("..");
}

function isHttpUrl(value) {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function hasLocalPathMarker(value) {
  return (
    /^[A-Za-z]:[\\/]/.test(value) ||
    /^[A-Za-z]:$/.test(value) ||
    value.startsWith("\\\\") ||
    value.startsWith("//") ||
    /(^|[\\/])\.\.($|[\\/])/.test(value)
  );
}

function assertTrackedRepoPath(file, label, value) {
  if (!isRepoRelativePath(value)) {
    fail(file, `${label} must be a repo-relative public path: ${value}`);
    return;
  }
  if (!trackedFiles.has(value)) {
    fail(file, `${label} is not tracked in this public repo: ${value}`);
  }
}

function assertDate(file, releaseDate, releaseSlug) {
  if (typeof releaseDate !== "string" || !RELEASE_DATE_RE.test(releaseDate)) {
    fail(file, `release_date is invalid: ${releaseDate}`);
    return;
  }
  const parsed = new Date(`${releaseDate}T00:00:00.000Z`);
  if (Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== releaseDate) {
    fail(file, `release_date is not a real calendar date: ${releaseDate}`);
  }
  if (typeof releaseSlug === "string") {
    const slugDate = releaseSlug.slice(-8);
    if (slugDate !== releaseDate.replaceAll("-", "")) {
      fail(file, `release_slug date ${slugDate} does not match release_date ${releaseDate}`);
    }
  }
}

function validateSchemaContract(schema) {
  const file = "releases/public-release-manifest.schema.json";
  if (!schema) {
    return;
  }
  const schemaConst = schema?.properties?.schema?.const;
  if (schemaConst !== SCHEMA_ID) {
    fail(file, `schema const must be ${SCHEMA_ID}`);
  }
  const exclusions = schema?.properties?.exclusions?.properties;
  if (!isPlainObject(exclusions)) {
    fail(file, "exclusions properties must be defined");
    return;
  }
  for (const exclusion of REQUIRED_EXCLUSIONS) {
    if (exclusions?.[exclusion]?.const !== false) {
      fail(file, `exclusion must be const false: ${exclusion}`);
    }
  }
}

function validateArtifact(file, artifact, index) {
  if (
    !assertExactKeys(file, `artifacts[${index}]`, artifact, [
      "name",
      "kind",
      "status",
      "path_or_url",
      "sha256",
      "signature_path_or_url",
      "notes",
    ])
  ) {
    return;
  }

  if (typeof artifact.name !== "string" || !/^[A-Za-z0-9._-]+$/.test(artifact.name)) {
    fail(file, `artifacts[${index}].name is invalid: ${artifact.name}`);
  }
  if (!ARTIFACT_KINDS.has(artifact.kind)) {
    fail(file, `artifacts[${index}].kind is invalid: ${artifact.kind}`);
  }
  if (!ARTIFACT_STATUSES.has(artifact.status)) {
    fail(file, `artifacts[${index}].status is invalid: ${artifact.status}`);
  }

  if (typeof artifact.path_or_url !== "string" || artifact.path_or_url.trim() === "") {
    fail(file, `artifacts[${index}].path_or_url must be a non-empty string`);
  } else if (hasLocalPathMarker(artifact.path_or_url)) {
    fail(file, `artifacts[${index}].path_or_url contains a local or parent path`);
  } else if (!isHttpUrl(artifact.path_or_url)) {
    assertTrackedRepoPath(file, `artifacts[${index}].path_or_url`, artifact.path_or_url);
  }

  if (artifact.sha256 !== null && (typeof artifact.sha256 !== "string" || !SHA256_RE.test(artifact.sha256))) {
    fail(file, `artifacts[${index}].sha256 must be null or lowercase 64-character hex`);
  }
  if (
    artifact.status === "published" &&
    PUBLISHED_ARTIFACT_KINDS_REQUIRING_SHA256.has(artifact.kind) &&
    artifact.sha256 === null
  ) {
    fail(file, `artifacts[${index}] is published and must include sha256`);
  }
  if (
    artifact.status === "published" &&
    PUBLISHED_ARTIFACT_KINDS_REQUIRING_SIGNATURE.has(artifact.kind) &&
    artifact.signature_path_or_url === null
  ) {
    fail(file, `artifacts[${index}] is published and must include signature_path_or_url`);
  }

  if (artifact.signature_path_or_url !== null) {
    if (typeof artifact.signature_path_or_url !== "string" || artifact.signature_path_or_url.trim() === "") {
      fail(file, `artifacts[${index}].signature_path_or_url must be null or a non-empty string`);
    } else if (hasLocalPathMarker(artifact.signature_path_or_url)) {
      fail(file, `artifacts[${index}].signature_path_or_url contains a local or parent path`);
    } else if (!isHttpUrl(artifact.signature_path_or_url)) {
      assertTrackedRepoPath(
        file,
        `artifacts[${index}].signature_path_or_url`,
        artifact.signature_path_or_url,
      );
    }
  }

  if (typeof artifact.notes !== "string" || artifact.notes.length > 500) {
    fail(file, `artifacts[${index}].notes must be a string up to 500 characters`);
  }
}

function validateManifest(file, manifest) {
  if (!manifest) {
    return;
  }
  assertExactKeys(file, "manifest", manifest, [
    "schema",
    "release_slug",
    "release_date",
    "release_kind",
    "public_claim",
    "source_of_truth",
    "public_files",
    "artifacts",
    "exclusions",
    "verification",
  ], ["$schema"]);

  if (manifest.schema !== SCHEMA_ID) {
    fail(file, `schema must be ${SCHEMA_ID}`);
  }
  if ("$schema" in manifest && (typeof manifest.$schema !== "string" || manifest.$schema.trim() === "")) {
    fail(file, "$schema must be a non-empty string when present");
  }
  if (typeof manifest.release_slug !== "string" || !RELEASE_SLUG_RE.test(manifest.release_slug)) {
    fail(file, `release_slug is invalid: ${manifest.release_slug}`);
  } else if (
    file !== "releases/public-release-manifest.example.json" &&
    file !== `releases/${manifest.release_slug}.manifest.json`
  ) {
    fail(file, `manifest filename must be releases/${manifest.release_slug}.manifest.json`);
  }
  assertDate(file, manifest.release_date, manifest.release_slug);
  if (!RELEASE_KINDS.has(manifest.release_kind)) {
    fail(file, `release_kind is invalid: ${manifest.release_kind}`);
  }
  if (
    typeof manifest.public_claim !== "string" ||
    manifest.public_claim.trim() === "" ||
    manifest.public_claim.length > 500
  ) {
    fail(file, "public_claim must be 1-500 visible characters");
  }

  if (
    assertExactKeys(file, "source_of_truth", manifest.source_of_truth, [
      "version_file",
      "github_release",
      "status_docs",
    ])
  ) {
    assertTrackedRepoPath(file, "source_of_truth.version_file", manifest.source_of_truth.version_file);
    if (manifest.source_of_truth.version_file !== "docs/VERSION.json") {
      fail(file, "source_of_truth.version_file must be docs/VERSION.json");
    }
    const expectedReleaseUrl = `https://github.com/VRAXION/VRAXION/releases/tag/${manifest.release_slug}`;
    if (manifest.source_of_truth.github_release !== expectedReleaseUrl) {
      fail(file, `source_of_truth.github_release must be ${expectedReleaseUrl}`);
    }
    for (const statusDoc of assertUniqueStringArray(
      file,
      "source_of_truth.status_docs",
      manifest.source_of_truth.status_docs,
      { minItems: 1 },
    )) {
      assertTrackedRepoPath(file, "source_of_truth.status_docs", statusDoc);
    }
  }

  for (const publicFile of assertUniqueStringArray(file, "public_files", manifest.public_files, { minItems: 1 })) {
    assertTrackedRepoPath(file, "public_files", publicFile);
  }

  if (!Array.isArray(manifest.artifacts)) {
    fail(file, "artifacts must be an array");
  } else {
    manifest.artifacts.forEach((artifact, index) => validateArtifact(file, artifact, index));
    const publishedArtifacts = manifest.artifacts.filter((artifact) => artifact?.status === "published");
    if (
      manifest.release_kind === "artifact_release" &&
      !publishedArtifacts.some((artifact) => artifact?.kind !== "documentation")
    ) {
      fail(file, "artifact_release manifests must include at least one published non-documentation artifact");
    }
    if (
      manifest.release_kind === "proof_pack" &&
      !publishedArtifacts.some((artifact) => artifact?.kind === "proof_pack")
    ) {
      fail(file, "proof_pack manifests must include at least one published proof_pack artifact");
    }
  }

  if (assertExactKeys(file, "exclusions", manifest.exclusions, REQUIRED_EXCLUSIONS)) {
    for (const exclusion of REQUIRED_EXCLUSIONS) {
      if (manifest.exclusions[exclusion] !== false) {
        fail(file, `exclusions.${exclusion} must be false`);
      }
    }
  }

  if (
    assertExactKeys(file, "verification", manifest.verification, [
      "commands",
      "github_checks",
      "reviewer",
    ])
  ) {
    const commands = assertUniqueStringArray(file, "verification.commands", manifest.verification.commands, {
      minItems: 1,
    });
    for (const required of REQUIRED_COMMANDS) {
      if (!commands.includes(required)) {
        fail(file, `verification.commands missing required command: ${required}`);
      }
    }

    const githubChecks = assertUniqueStringArray(
      file,
      "verification.github_checks",
      manifest.verification.github_checks,
      { minItems: 1 },
    );
    for (const required of REQUIRED_GITHUB_CHECKS) {
      if (!githubChecks.includes(required)) {
        fail(file, `verification.github_checks missing required check: ${required}`);
      }
    }

    if (typeof manifest.verification.reviewer !== "string" || !REVIEWER_RE.test(manifest.verification.reviewer)) {
      fail(file, `verification.reviewer is invalid: ${manifest.verification.reviewer}`);
    }
  }
}

const schema = readJson("releases/public-release-manifest.schema.json");
validateSchemaContract(schema);

const manifestFiles = fs
  .readdirSync(RELEASES_DIR)
  .filter((name) => name === "public-release-manifest.example.json" || name.endsWith(".manifest.json"))
  .sort()
  .map((name) => `releases/${name}`);

const version = readJson("docs/VERSION.json");
const latestPublicRelease = version?.latest_public_release;
if (typeof latestPublicRelease !== "string" || !RELEASE_SLUG_RE.test(latestPublicRelease)) {
  fail("docs/VERSION.json", `latest_public_release is invalid: ${latestPublicRelease}`);
} else {
  const latestManifestFile = `releases/${latestPublicRelease}.manifest.json`;
  if (!manifestFiles.includes(latestManifestFile)) {
    fail("docs/VERSION.json", `latest_public_release is missing release manifest: ${latestManifestFile}`);
  }
}

for (const manifestFile of manifestFiles) {
  validateManifest(manifestFile, readJson(manifestFile));
}

console.log("PUBLIC_RELEASE_MANIFEST_VALIDATION");
console.log(`manifest_files=${manifestFiles.length}`);
console.log(`failure_count=${failures.length}`);
for (const failure of failures) {
  console.log(`failure: ${failure}`);
}
if (failures.length > 0) {
  process.exit(1);
}
console.log("public_release_manifest_validation=pass");
