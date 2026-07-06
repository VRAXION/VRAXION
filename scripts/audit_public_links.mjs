#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pagesBaseUrl = "https://vraxion.github.io/VRAXION/";
const githubBlobPrefix = "https://github.com/VRAXION/VRAXION/blob/main/";
const scannedExtensions = new Set([".md", ".html", ".yml", ".yaml"]);
const markdownExtensions = new Set([".md"]);
const htmlExtensions = new Set([".html"]);
const yamlExtensions = new Set([".yml", ".yaml"]);
const failures = [];

const trackedFiles = execFileSync("git", ["ls-files"], { cwd: root, encoding: "utf8" })
  .split(/\r?\n/)
  .map((line) => line.trim().replaceAll("\\", "/"))
  .filter(Boolean);
const trackedFileSet = new Set(trackedFiles);
const trackedDirSet = new Set(["."]);

for (const file of trackedFiles) {
  const parts = file.split("/");
  for (let index = 1; index < parts.length; index += 1) {
    trackedDirSet.add(parts.slice(0, index).join("/"));
  }
}

function fail(relativePath, line, message) {
  failures.push(`${relativePath}:${line}: ${message}`);
}

function lineNumber(text, index) {
  return text.slice(0, index).split(/\r?\n/).length;
}

function isExternalReference(target) {
  return /^(?:https?:|mailto:|tel:|data:|javascript:)/i.test(target);
}

function withoutFragmentAndQuery(target) {
  return target.split("#", 1)[0].split("?", 1)[0];
}

function normalizeRepoPath(candidate) {
  return path.posix.normalize(candidate.replaceAll("\\", "/")).replace(/^\.\//, "").replace(/\/+$/, "");
}

function trackedTargetExists(repoPath) {
  const normalized = normalizeRepoPath(repoPath);
  if (!normalized || normalized === ".") return true;
  if (trackedFileSet.has(normalized) || trackedDirSet.has(normalized)) return true;
  if (trackedFileSet.has(`${normalized}/index.html`)) return true;
  return false;
}

function checkRepoPath(relativePath, line, repoPath, originalTarget) {
  if (/[A-Za-z]:[\\/]|^\/\//.test(repoPath)) {
    fail(relativePath, line, `local link uses an absolute machine path: ${originalTarget}`);
    return;
  }
  const normalized = normalizeRepoPath(repoPath);
  if (normalized.startsWith("../") || normalized.includes("/../")) {
    fail(relativePath, line, `local link escapes the public repository: ${originalTarget}`);
    return;
  }
  if (!trackedTargetExists(normalized)) {
    fail(relativePath, line, `local link target is missing: ${originalTarget}`);
  }
}

function checkPagesUrl(relativePath, line, target) {
  const url = new URL(target);
  if (url.origin !== "https://vraxion.github.io" || !url.pathname.startsWith("/VRAXION/")) return;
  const pagesPath = decodeURIComponent(url.pathname.slice("/VRAXION/".length));
  const docsPath = pagesPath ? `docs/${pagesPath}` : "docs/index.html";
  checkRepoPath(relativePath, line, docsPath.endsWith("/") ? `${docsPath}index.html` : docsPath, target);
}

function checkGithubBlobUrl(relativePath, line, target) {
  if (!target.startsWith(githubBlobPrefix)) return;
  const repoPath = decodeURIComponent(withoutFragmentAndQuery(target.slice(githubBlobPrefix.length)));
  checkRepoPath(relativePath, line, repoPath, target);
}

function checkLocalTarget(relativePath, line, target) {
  const cleanTarget = withoutFragmentAndQuery(target.trim());
  if (!cleanTarget || cleanTarget.startsWith("#")) return;
  if (isExternalReference(cleanTarget)) {
    checkGithubBlobUrl(relativePath, line, cleanTarget);
    checkPagesUrl(relativePath, line, cleanTarget);
    return;
  }

  const baseDir = path.posix.dirname(relativePath);
  const repoPath = cleanTarget.startsWith("/")
    ? cleanTarget.slice(1)
    : path.posix.join(baseDir === "." ? "" : baseDir, cleanTarget);
  checkRepoPath(relativePath, line, repoPath, target);
}

function collectMarkdownLinks(text) {
  const links = [];
  for (const match of text.matchAll(/(?<!!)\[[^\]\n]+\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g)) {
    links.push({ index: match.index ?? 0, target: match[1] });
  }
  return links;
}

function collectHtmlLinks(text) {
  const links = [];
  for (const match of text.matchAll(/\s(?:href|src)=["']([^"']+)["']/gi)) {
    links.push({ index: match.index ?? 0, target: match[1] });
  }
  return links;
}

function collectYamlGithubBlobLinks(text) {
  const links = [];
  for (const match of text.matchAll(/https:\/\/github\.com\/VRAXION\/VRAXION\/blob\/main\/[^\s"')]+/g)) {
    links.push({ index: match.index ?? 0, target: match[0] });
  }
  return links;
}

let scannedFiles = 0;
let checkedLinks = 0;

for (const relativePath of trackedFiles) {
  const extension = path.extname(relativePath).toLowerCase();
  if (!scannedExtensions.has(extension)) continue;

  const text = fs.readFileSync(path.join(root, relativePath), "utf8");
  const links = [];
  if (markdownExtensions.has(extension)) links.push(...collectMarkdownLinks(text));
  if (htmlExtensions.has(extension)) links.push(...collectHtmlLinks(text));
  if (yamlExtensions.has(extension)) links.push(...collectYamlGithubBlobLinks(text));

  scannedFiles += 1;
  for (const link of links) {
    checkedLinks += 1;
    checkLocalTarget(relativePath, lineNumber(text, link.index), link.target);
  }
}

console.log("PUBLIC_LINK_AUDIT");
console.log(`tracked_files=${trackedFiles.length}`);
console.log(`scanned_files=${scannedFiles}`);
console.log(`checked_links=${checkedLinks}`);
console.log(`failure_count=${failures.length}`);
for (const failure of failures) console.log(`failure: ${failure}`);
if (failures.length) process.exit(1);
console.log("public_link_audit=pass");
