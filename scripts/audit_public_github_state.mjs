#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const VERSION_PATH = "docs/VERSION.json";
const EXPECTED_REPOSITORY = "VRAXION/VRAXION";
const EXPECTED_DEFAULT_BRANCH = "main";
const EXPECTED_PAGES_URL = "https://vraxion.github.io/VRAXION/";
const EXPECTED_PAGES_SOURCE_BRANCH = "main";
const EXPECTED_PAGES_SOURCE_PATH = "/docs";
const ALLOWED_REMOTE_BRANCHES = new Set(["origin", "origin/HEAD", "origin/main"]);
const PUBLIC_URL_TIMEOUT_MS = 20000;
const args = new Set(process.argv.slice(2));

if (args.has("--help")) {
  console.log(
    "Usage: node scripts/audit_public_github_state.mjs [--allow-open-prs] [--allow-extra-remote-branches]",
  );
  process.exit(0);
}

const allowOpenPullRequests = args.has("--allow-open-prs");
const allowExtraRemoteBranches = args.has("--allow-extra-remote-branches");
const failures = [];
const warnings = [];

function fail(message) {
  failures.push(message);
}

function warn(message) {
  warnings.push(message);
}

function runCommand(label, file, commandArgs) {
  try {
    return execFileSync(file, commandArgs, {
      cwd: ROOT,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    }).trim();
  } catch (error) {
    const stderr = error.stderr?.toString("utf8").trim();
    fail(`${label}: ${stderr || error.message}`);
    return null;
  }
}

function runJson(label, file, commandArgs) {
  const output = runCommand(label, file, commandArgs);
  if (output === null) {
    return null;
  }
  try {
    return JSON.parse(output);
  } catch (error) {
    fail(`${label}: invalid JSON: ${error.message}`);
    return null;
  }
}

function readJsonFile(relativePath) {
  try {
    return JSON.parse(fs.readFileSync(path.join(ROOT, relativePath), "utf8"));
  } catch (error) {
    fail(`${relativePath}: invalid JSON: ${error.message}`);
    return null;
  }
}

function normalizeTrailingSlash(value) {
  return value.endsWith("/") ? value : `${value}/`;
}

function getRemoteMainCommit() {
  const output = runCommand("git remote main commit", "git", ["ls-remote", "origin", "refs/heads/main"]);
  const commit = output?.split(/\s+/)[0] || "";
  if (!/^[a-f0-9]{40}$/i.test(commit)) {
    fail(`origin/main remote commit could not be resolved: ${commit || "missing"}`);
    return "unknown";
  }
  return commit.toLowerCase();
}

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PUBLIC_URL_TIMEOUT_MS);
  try {
    return await fetch(url, {
      redirect: "follow",
      ...options,
      signal: controller.signal,
      headers: {
        "user-agent": "VRAXION public GitHub state audit",
        ...(options.headers || {}),
      },
    });
  } finally {
    clearTimeout(timer);
  }
}

async function fetchText(url, label) {
  try {
    const response = await fetchWithTimeout(url);
    if (!response.ok) {
      fail(`${label} returned HTTP ${response.status}: ${url}`);
      return { status: response.status, text: "" };
    }
    return { status: response.status, text: await response.text() };
  } catch (error) {
    fail(`${label} could not be fetched: ${url} ${error.message}`);
    return { status: "error", text: "" };
  }
}

function validateOriginRemote() {
  const originUrl = runCommand("git origin remote", "git", ["remote", "get-url", "origin"]);
  if (originUrl === null) {
    return;
  }
  if (!/[:/]VRAXION\/VRAXION(?:\.git)?$/i.test(originUrl)) {
    fail(`origin remote does not point at ${EXPECTED_REPOSITORY}`);
  }
}

function validateRemoteBranches() {
  const output = runCommand("git remote branches", "git", ["branch", "-r", "--format=%(refname:short)"]);
  if (output === null) {
    return [];
  }
  const branches = output.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (!branches.includes("origin/main")) {
    fail("origin/main remote branch is missing");
  }
  const unexpectedBranches = branches.filter((branch) => !ALLOWED_REMOTE_BRANCHES.has(branch));
  if (unexpectedBranches.length > 0) {
    const message = `unexpected remote branch(es): ${unexpectedBranches.join(", ")}`;
    if (allowExtraRemoteBranches) {
      warn(message);
    } else {
      fail(message);
    }
  }
  return branches;
}

function validateRepository() {
  const repository = runJson("GitHub repository", "gh", [
    "repo",
    "view",
    "--json",
    "nameWithOwner,defaultBranchRef,url",
  ]);
  if (repository === null) {
    return null;
  }
  if (repository.nameWithOwner !== EXPECTED_REPOSITORY) {
    fail(`GitHub repository must be ${EXPECTED_REPOSITORY}, got ${repository.nameWithOwner}`);
  }
  if (repository.defaultBranchRef?.name !== EXPECTED_DEFAULT_BRANCH) {
    fail(`GitHub default branch must be ${EXPECTED_DEFAULT_BRANCH}, got ${repository.defaultBranchRef?.name}`);
  }
  return repository;
}

function validateOpenPullRequests() {
  const pullRequests = runJson("GitHub open pull requests", "gh", [
    "pr",
    "list",
    "--state",
    "open",
    "--limit",
    "100",
    "--json",
    "number,title,headRefName,baseRefName,url",
  ]);
  if (!Array.isArray(pullRequests)) {
    return [];
  }
  if (pullRequests.length > 0) {
    const summary = pullRequests
      .map((pullRequest) => `#${pullRequest.number} ${pullRequest.headRefName}->${pullRequest.baseRefName}`)
      .join(", ");
    const message = `open pull request(s) present during public release audit: ${summary}`;
    if (allowOpenPullRequests) {
      warn(message);
    } else {
      fail(message);
    }
  }
  return pullRequests;
}

function validateRelease(version) {
  const latestPublicRelease = version?.latest_public_release;
  if (typeof latestPublicRelease !== "string" || latestPublicRelease.length === 0) {
    fail(`${VERSION_PATH}: latest_public_release must be a non-empty string`);
    return { latestPublicRelease: "unknown", latestReleaseTag: "unknown" };
  }

  const expectedReleaseUrl = `https://github.com/${EXPECTED_REPOSITORY}/releases/tag/${latestPublicRelease}`;
  const releaseView = runJson("GitHub release view", "gh", [
    "release",
    "view",
    latestPublicRelease,
    "--json",
    "tagName,isPrerelease,name,url,publishedAt,targetCommitish,isDraft",
  ]);
  if (releaseView !== null) {
    if (releaseView.tagName !== latestPublicRelease) {
      fail(`GitHub release tag mismatch: expected ${latestPublicRelease}, got ${releaseView.tagName}`);
    }
    if (releaseView.url !== expectedReleaseUrl) {
      fail(`GitHub release URL mismatch: expected ${expectedReleaseUrl}, got ${releaseView.url}`);
    }
    if (releaseView.isDraft) {
      fail(`GitHub release ${latestPublicRelease} is still a draft`);
    }
    if (releaseView.isPrerelease) {
      fail(`GitHub release ${latestPublicRelease} is marked as prerelease`);
    }
    if (typeof releaseView.name !== "string" || !releaseView.name.includes(latestPublicRelease)) {
      fail(`GitHub release name must include ${latestPublicRelease}`);
    }
  }

  const releaseList = runJson("GitHub release list", "gh", [
    "release",
    "list",
    "--limit",
    "20",
    "--json",
    "tagName,isLatest,isPrerelease,name,publishedAt,isDraft",
  ]);
  let latestReleaseTag = "unknown";
  if (Array.isArray(releaseList)) {
    const latestEntries = releaseList.filter((release) => release.isLatest === true);
    if (latestEntries.length !== 1) {
      fail(`GitHub release list must report exactly one latest release, got ${latestEntries.length}`);
    } else {
      latestReleaseTag = latestEntries[0].tagName;
      if (latestReleaseTag !== latestPublicRelease) {
        fail(`GitHub latest release must be ${latestPublicRelease}, got ${latestReleaseTag}`);
      }
    }
    if (!releaseList.some((release) => release.tagName === latestPublicRelease)) {
      fail(`GitHub release list did not include ${latestPublicRelease} in the first 20 releases`);
    }
  }

  return { latestPublicRelease, latestReleaseTag };
}

async function validatePagesState(version, remoteMainCommit) {
  const pages = runJson("GitHub Pages config", "gh", ["api", `repos/${EXPECTED_REPOSITORY}/pages`]);
  const latestBuild = runJson("GitHub Pages latest build", "gh", [
    "api",
    `repos/${EXPECTED_REPOSITORY}/pages/builds/latest`,
  ]);
  let pagesStatus = pages?.status || "unknown";
  let pagesBuildStatus = latestBuild?.status || "unknown";
  let pagesBuildCommit = latestBuild?.commit || "unknown";
  let publicPagesHttpStatus = "unknown";
  let liveVersionRelease = "unknown";

  if (pages !== null) {
    const pagesUrl = normalizeTrailingSlash(String(pages.html_url || ""));
    if (pagesUrl !== EXPECTED_PAGES_URL) {
      fail(`GitHub Pages URL must be ${EXPECTED_PAGES_URL}, got ${pages.html_url || "missing"}`);
    }
    if (pages.status !== "built") {
      fail(`GitHub Pages status must be built, got ${pages.status || "missing"}`);
    }
    if (pages.public !== true) {
      fail("GitHub Pages must be public");
    }
    if (pages.https_enforced !== true) {
      fail("GitHub Pages must enforce HTTPS");
    }
    if (pages.source?.branch !== EXPECTED_PAGES_SOURCE_BRANCH || pages.source?.path !== EXPECTED_PAGES_SOURCE_PATH) {
      fail(
        `GitHub Pages source must be ${EXPECTED_PAGES_SOURCE_BRANCH}:${EXPECTED_PAGES_SOURCE_PATH}, got ${
          pages.source?.branch || "missing"
        }:${pages.source?.path || "missing"}`,
      );
    }
  }

  if (latestBuild !== null) {
    if (latestBuild.status !== "built") {
      fail(`latest GitHub Pages build must be built, got ${latestBuild.status || "missing"}`);
    }
    if (latestBuild.error?.message) {
      fail(`latest GitHub Pages build reports an error: ${latestBuild.error.message}`);
    }
    if (remoteMainCommit !== "unknown" && latestBuild.commit?.toLowerCase() !== remoteMainCommit) {
      fail(`latest GitHub Pages build commit must match origin/main ${remoteMainCommit}, got ${latestBuild.commit}`);
    }
  }

  const homeFetch = await fetchText(EXPECTED_PAGES_URL, "public Pages home");
  publicPagesHttpStatus = String(homeFetch.status);
  const versionFetch = await fetchText(`${EXPECTED_PAGES_URL}VERSION.json`, "public Pages VERSION.json");
  if (versionFetch.text) {
    try {
      const liveVersion = JSON.parse(versionFetch.text);
      liveVersionRelease = String(liveVersion.latest_public_release || "");
      if (version?.latest_public_release && liveVersionRelease !== version.latest_public_release) {
        fail(
          `live Pages VERSION latest_public_release must be ${version.latest_public_release}, got ${
            liveVersionRelease || "missing"
          }`,
        );
      }
      if (version?.date && liveVersion.date !== version.date) {
        fail(`live Pages VERSION date must be ${version.date}, got ${liveVersion.date || "missing"}`);
      }
    } catch (error) {
      fail(`public Pages VERSION.json is invalid JSON: ${error.message}`);
    }
  }

  return {
    pagesStatus,
    pagesBuildStatus,
    pagesBuildCommit,
    publicPagesHttpStatus,
    liveVersionRelease: liveVersionRelease || "unknown",
  };
}

runCommand("GitHub auth status", "gh", ["auth", "status"]);
validateOriginRemote();
const remoteBranches = validateRemoteBranches();
const remoteMainCommit = getRemoteMainCommit();
const repository = validateRepository();
const openPullRequests = validateOpenPullRequests();
const version = readJsonFile(VERSION_PATH);
const { latestPublicRelease, latestReleaseTag } = validateRelease(version);
const pagesState = await validatePagesState(version, remoteMainCommit);

console.log("PUBLIC_GITHUB_STATE_AUDIT");
console.log(`repository=${repository?.nameWithOwner || "unknown"}`);
console.log(`default_branch=${repository?.defaultBranchRef?.name || "unknown"}`);
console.log(`origin_main_commit=${remoteMainCommit}`);
console.log(`latest_public_release=${latestPublicRelease}`);
console.log(`github_latest_release=${latestReleaseTag}`);
console.log(`pages_status=${pagesState.pagesStatus}`);
console.log(`pages_latest_build_status=${pagesState.pagesBuildStatus}`);
console.log(`pages_latest_build_commit=${pagesState.pagesBuildCommit}`);
console.log(`pages_live_version_release=${pagesState.liveVersionRelease}`);
console.log(`public_pages_http_status=${pagesState.publicPagesHttpStatus}`);
console.log(`open_pull_request_count=${openPullRequests.length}`);
console.log(`remote_branch_count=${remoteBranches.length}`);
console.log(`failure_count=${failures.length}`);
console.log(`warning_count=${warnings.length}`);
for (const failure of failures) {
  console.log(`failure: ${failure}`);
}
for (const warning of warnings) {
  console.log(`warning: ${warning}`);
}
if (failures.length > 0) {
  process.exit(1);
}
console.log("public_github_state_audit=pass");
