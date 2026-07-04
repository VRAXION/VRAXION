import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const version = JSON.parse(await fs.readFile(path.join(root, "docs", "VERSION.json"), "utf8"));
const latestRelease = String(version.latest_public_release || "");
const instnctAssetVersion = String(version.instnct_asset_version || "");
const baseUrl = (process.env.PUBLIC_PAGES_BASE_URL || "https://vraxion.github.io/VRAXION").replace(/\/+$/, "");
const hiddenSurfaceSlug = ["vn", "gard"].join("");
const token = (...parts) => parts.join("");
const unsafePublicCopyPattern = new RegExp(
  [
    "Not AI",
    "Not ever",
    "Runs locally",
    "microsecond-class reasoning core",
    "Hallucination,",
    "hallucination, toggleable",
    "fabric that reasons",
    "decentralized intelligence",
    "Scales by dimension",
    "No weights",
    "No probabilities",
    "T1 is coming",
    "local runnable",
    token("source", "-available"),
    token("source ", "available"),
    token("source ", "snapshot"),
    token("source ", "archive"),
    token("public source ", "archive"),
    token("page ", "source"),
    token("boundary ", "snapshot"),
    token("boundary ", "archive"),
    token("P11 SDK ", "boundary"),
    token("binary ", "boundary"),
    token("release ", "boundary"),
    token("boundary text ", "versioned in repo"),
    token("boundary and ", "release target"),
    token("release ", "boundaries"),
    token("mode ", "boundary"),
    token("returns a ", "boundary"),
    token("public mark ", "boundary"),
    token("opt-in ", "boundary"),
    token(">bound", "ary<\\/span>"),
  ].join("|"),
  "i"
);
const failures = [];

function fail(message) {
  failures.push(message);
}

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Number(process.env.PUBLIC_PAGES_LINK_TIMEOUT_MS || 20000));
  try {
    return await fetch(url, {
      redirect: "follow",
      ...options,
      signal: controller.signal,
      headers: {
        "user-agent": "VRAXION public-pages-smoke",
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
      return "";
    }
    return await response.text();
  } catch (err) {
    fail(`${label} could not be fetched: ${url} ${err.message}`);
    return "";
  }
}

async function assertReachable(url, label) {
  try {
    let response = await fetchWithTimeout(url, { method: "HEAD" });
    if (response.status === 405 || response.status === 403) {
      response = await fetchWithTimeout(url, { headers: { range: "bytes=0-0" } });
    }
    if (!response.ok) fail(`${label} returned HTTP ${response.status}: ${url}`);
  } catch (err) {
    fail(`${label} could not be reached: ${url} ${err.message}`);
  }
}

function collectUrls(html, pageUrl) {
  const urls = new Set();
  for (const match of html.matchAll(/\s(?:href|src)=["']([^"']+)["']/g)) {
    const raw = match[1].trim();
    if (!raw || raw.startsWith("#") || /^(mailto|tel|data|javascript):/i.test(raw)) continue;
    const url = new URL(raw, pageUrl);
    url.hash = "";
    urls.add(url.href);
  }
  return urls;
}

const homeUrl = `${baseUrl}/`;
const instnctUrl = `${baseUrl}/instnct/`;
const hiddenSurfaceUrl = `${baseUrl}/${hiddenSurfaceSlug}/`;
const robotsUrl = `${baseUrl}/robots.txt`;
const sitemapUrl = `${baseUrl}/sitemap.xml`;

const home = await fetchText(homeUrl, "home");
const instnct = await fetchText(instnctUrl, "INSTNCT");
const robots = await fetchText(robotsUrl, "robots.txt");
const sitemap = await fetchText(sitemapUrl, "sitemap.xml");
let hiddenSurfaceStatus = 0;
try {
  const hiddenSurfaceResponse = await fetchWithTimeout(hiddenSurfaceUrl, { redirect: "manual" });
  hiddenSurfaceStatus = hiddenSurfaceResponse.status;
} catch (err) {
  fail(`hidden roadmap surface URL could not be checked: ${hiddenSurfaceUrl} ${err.message}`);
}

if (home && !home.includes(`releases/tag/${latestRelease}`)) fail("home does not expose the VERSION latest release");
if (instnct && !instnct.includes(`archive/refs/tags/${latestRelease}.zip`)) {
  fail("INSTNCT does not expose the VERSION GitHub tag ZIP");
}
if (!/^release-\d+$/.test(instnctAssetVersion)) {
  fail(`VERSION instnct_asset_version is invalid: ${instnctAssetVersion || "missing"}`);
}
if (
  instnct &&
  (!instnct.includes(`styles.css?v=${instnctAssetVersion}`) ||
    !instnct.includes(`instnct.js?v=${instnctAssetVersion}`))
) {
  fail("INSTNCT live page does not load the VERSION asset cache key");
}
if (instnct && unsafePublicCopyPattern.test(instnct)) {
  fail("INSTNCT public copy exposes unsafe or internal release wording");
}
if (instnct && !instnct.includes("artifact-status")) fail("INSTNCT artifact status block is missing on live Pages");
if (instnct && /github\.com\/VRAXION\/VRAXION\/blob\/main\/(?:CURRENT_|PUBLIC_SURFACE_POLICY)/.test(instnct)) {
  fail("INSTNCT live page links public docs at the repository root instead of docs/");
}
if (home && new RegExp(`href=["'][^"']*${hiddenSurfaceSlug}/|hidden roadmap retained|Open roadmap concept`, "i").test(home)) {
  fail("home page exposes hidden roadmap surface");
}
if (instnct && new RegExp(`href=["'][^"']*${hiddenSurfaceSlug}/|hidden roadmap`, "i").test(instnct)) {
  fail("INSTNCT page exposes hidden roadmap surface");
}
if (hiddenSurfaceStatus !== 404) fail(`hidden roadmap surface URL must be absent from Pages, got HTTP ${hiddenSurfaceStatus}`);
if (robots && !robots.includes(`${baseUrl}/sitemap.xml`)) fail("robots.txt does not point at the live sitemap");
if (sitemap && !sitemap.includes(`${baseUrl}/instnct/`)) fail("sitemap.xml does not include INSTNCT");
if (sitemap && sitemap.includes(`${baseUrl}/${hiddenSurfaceSlug}/`)) fail("sitemap.xml exposes hidden roadmap surface");

const urls = new Set([
  ...collectUrls(home, homeUrl),
  ...collectUrls(instnct, instnctUrl),
  `${baseUrl}/robots.txt`,
  `${baseUrl}/sitemap.xml`,
  `https://github.com/VRAXION/VRAXION/releases/tag/${latestRelease}`,
  `https://github.com/VRAXION/VRAXION/archive/refs/tags/${latestRelease}.zip`,
]);

for (const url of urls) {
  const parsed = new URL(url);
  const isSameSite = url.startsWith(`${baseUrl}/`) || url === `${baseUrl}/`;
  const isGitHub = parsed.hostname === "github.com";
  if (isSameSite || isGitHub) {
    await assertReachable(url, isSameSite ? "live Pages link" : "GitHub link");
  }
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log("public_pages_link_smoke=pass");
