import { execFileSync } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const versionPath = path.join(root, "docs", "VERSION.json");
const releaseSlugPattern = /public-sdk-p\d+-\d{8}/g;
const publicTextExtensions = new Set([".html", ".md", ".json", ".txt", ".xml"]);
const args = new Set(process.argv.slice(2));
const write = args.has("--write");

function fail(message) {
  console.error(message);
  process.exitCode = 1;
}

function trackedFiles() {
  const output = execFileSync("git", ["ls-files"], { cwd: root, encoding: "utf8" });
  return output
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.replaceAll("\\", "/"));
}

function isPublicTextFile(relative) {
  if (!relative.startsWith("docs/") && !["LICENSE_BOUNDARY.md", "PUBLIC_DELIVERY_MODEL.md"].includes(relative)) {
    return false;
  }
  return publicTextExtensions.has(path.extname(relative).toLowerCase());
}

function repoBlobPathFromUrl(urlText) {
  const match = urlText.match(/https:\/\/github\.com\/VRAXION\/VRAXION\/blob\/main\/([^"#?]+)/);
  return match ? decodeURIComponent(match[1]) : "";
}

const version = JSON.parse(await fs.readFile(versionPath, "utf8"));
const latestRelease = String(version.latest_public_release || "");
if (!/^public-sdk-p\d+-\d{8}$/.test(latestRelease)) {
  fail(`docs/VERSION.json latest_public_release is invalid: ${latestRelease || "missing"}`);
}

const files = trackedFiles();
const fileSet = new Set(files);
const releaseFiles = files.filter(isPublicTextFile);
const changed = [];

for (const relative of releaseFiles) {
  const absolute = path.join(root, relative);
  const original = await fs.readFile(absolute, "utf8");
  const next = original.replace(releaseSlugPattern, latestRelease);
  if (next !== original) {
    if (write) {
      await fs.writeFile(absolute, next);
      changed.push(relative);
    } else {
      const stale = [...new Set(original.match(releaseSlugPattern) || [])].filter((slug) => slug !== latestRelease);
      if (stale.length) fail(`${relative} contains stale public release slug(s): ${stale.join(", ")}`);
    }
  }

  const text = write ? next : original;
  for (const match of text.matchAll(/https:\/\/github\.com\/VRAXION\/VRAXION\/blob\/main\/[^"'<)\s]+/g)) {
    const repoPath = repoBlobPathFromUrl(match[0]);
    if (repoPath && !fileSet.has(repoPath)) fail(`${relative} links to missing repo path: ${repoPath}`);
  }
}

const indexHtml = await fs.readFile(path.join(root, "docs", "index.html"), "utf8");
const instnctHtml = await fs.readFile(path.join(root, "docs", "instnct", "index.html"), "utf8");
const releaseUrl = `https://github.com/VRAXION/VRAXION/releases/tag/${latestRelease}`;
const archiveUrl = `https://github.com/VRAXION/VRAXION/archive/refs/tags/${latestRelease}.zip`;

if (!indexHtml.includes(releaseUrl)) fail("docs/index.html does not link to the VERSION latest release URL");
if (!instnctHtml.includes(releaseUrl)) fail("docs/instnct/index.html does not link to the VERSION latest release URL");
if (!instnctHtml.includes(archiveUrl)) fail("docs/instnct/index.html does not link to the VERSION source archive URL");

if (write) {
  console.log(`public_release_links_write=${changed.length ? changed.join(",") : "none"}`);
} else if (process.exitCode) {
  process.exit(process.exitCode);
} else {
  console.log("public_release_links=pass");
}
