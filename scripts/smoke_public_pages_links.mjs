import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const version = JSON.parse(await fs.readFile(path.join(root, "docs", "VERSION.json"), "utf8"));
const latestRelease = String(version.latest_public_release || "");
const instnctAssetVersion = String(version.instnct_asset_version || "");
const baseUrl = (process.env.PUBLIC_PAGES_BASE_URL || "https://vraxion.github.io/VRAXION").replace(/\/+$/, "");
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
const vngardUrl = `${baseUrl}/vngard/`;
const robotsUrl = `${baseUrl}/robots.txt`;
const sitemapUrl = `${baseUrl}/sitemap.xml`;

const home = await fetchText(homeUrl, "home");
const instnct = await fetchText(instnctUrl, "INSTNCT");
const robots = await fetchText(robotsUrl, "robots.txt");
const sitemap = await fetchText(sitemapUrl, "sitemap.xml");
let vngardStatus = 0;
try {
  const vngardResponse = await fetchWithTimeout(vngardUrl, { redirect: "manual" });
  vngardStatus = vngardResponse.status;
} catch (err) {
  fail(`VNGARD public URL could not be checked: ${vngardUrl} ${err.message}`);
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
if (
  instnct &&
  /Not AI|Not ever|Runs locally|microsecond-class reasoning core|Hallucination,|hallucination, toggleable|fabric that reasons|decentralized intelligence|Scales by dimension|No weights|No probabilities|T1 is coming|local runnable|source-available|source available|source snapshot|source archive|public source archive|page source|boundary snapshot|boundary archive|P11 SDK boundary|binary boundary|release boundary|boundary text versioned in repo|boundary and release target|release boundaries|mode boundary|returns a boundary|public mark boundary|opt-in boundary|>boundary<\/span>/i.test(
    instnct
  )
) {
  fail("INSTNCT public copy exposes unsafe or internal release wording");
}
if (instnct && !instnct.includes("artifact-status")) fail("INSTNCT artifact status block is missing on live Pages");
if (instnct && /github\.com\/VRAXION\/VRAXION\/blob\/main\/(?:CURRENT_|PUBLIC_SURFACE_POLICY)/.test(instnct)) {
  fail("INSTNCT live page links public docs at the repository root instead of docs/");
}
if (home && /href=["'][^"']*vngard\/|VNGARD retained|Open roadmap concept/i.test(home)) {
  fail("home page exposes hidden VNGARD surface");
}
if (instnct && /href=["'][^"']*vngard\/|VNGARD roadmap/i.test(instnct)) {
  fail("INSTNCT page exposes hidden VNGARD surface");
}
if (vngardStatus !== 404) fail(`VNGARD public URL must be absent from Pages, got HTTP ${vngardStatus}`);
if (robots && !robots.includes(`${baseUrl}/sitemap.xml`)) fail("robots.txt does not point at the live sitemap");
if (sitemap && !sitemap.includes(`${baseUrl}/instnct/`)) fail("sitemap.xml does not include INSTNCT");
if (sitemap && sitemap.includes(`${baseUrl}/vngard/`)) fail("sitemap.xml exposes hidden VNGARD");

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
