#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const VERSION_PATH = "docs/VERSION.json";
const EXPECTED_REPOSITORY = "VRAXION/VRAXION";
const EXPECTED_DEFAULT_BRANCH = "main";
const ALLOWED_REMOTE_BRANCHES = new Set(["origin", "origin/HEAD", "origin/main"]);
const args = new Set(process.argv.slice(2));

if (args.has("--help")) {
  console.log("Usage: node scripts/audit_public_github_state.mjs [--allow-open-prs] [--allow-extra-remote-branches]");
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

runCommand("GitHub auth status", "gh", ["auth", "status"]);
validateOriginRemote();
const remoteBranches = validateRemoteBranches();
const repository = validateRepository();
const openPullRequests = validateOpenPullRequests();
const version = readJsonFile(VERSION_PATH);
const { latestPublicRelease, latestReleaseTag } = validateRelease(version);

console.log("PUBLIC_GITHUB_STATE_AUDIT");
console.log(`repository=${repository?.nameWithOwner || "unknown"}`);
console.log(`default_branch=${repository?.defaultBranchRef?.name || "unknown"}`);
console.log(`latest_public_release=${latestPublicRelease}`);
console.log(`github_latest_release=${latestReleaseTag}`);
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
