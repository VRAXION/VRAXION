import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const version = JSON.parse(await fs.readFile(path.join(root, "docs", "VERSION.json"), "utf8"));
const latestRelease = String(version.latest_public_release || "");
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
const robotsUrl = `${baseUrl}/robots.txt`;
const sitemapUrl = `${baseUrl}/sitemap.xml`;

const home = await fetchText(homeUrl, "home");
const instnct = await fetchText(instnctUrl, "INSTNCT");
const robots = await fetchText(robotsUrl, "robots.txt");
const sitemap = await fetchText(sitemapUrl, "sitemap.xml");

if (home && !home.includes(`releases/tag/${latestRelease}`)) fail("home does not expose the VERSION latest release");
if (instnct && !instnct.includes(`archive/refs/tags/${latestRelease}.zip`)) {
  fail("INSTNCT does not expose the VERSION GitHub tag ZIP");
}
if (
  instnct &&
  /source-available|source available|source snapshot|source archive|public source archive|page source|boundary snapshot|boundary archive|P11 SDK boundary/i.test(
    instnct
  )
) {
  fail("INSTNCT public copy implies source availability");
}
if (instnct && !instnct.includes("artifact-status")) fail("INSTNCT artifact status block is missing on live Pages");
if (instnct && /github\.com\/VRAXION\/VRAXION\/blob\/main\/(?:CURRENT_|PUBLIC_SURFACE_POLICY)/.test(instnct)) {
  fail("INSTNCT live page links public docs at the repository root instead of docs/");
}
if (robots && !robots.includes(`${baseUrl}/sitemap.xml`)) fail("robots.txt does not point at the live sitemap");
if (sitemap && !sitemap.includes(`${baseUrl}/instnct/`)) fail("sitemap.xml does not include INSTNCT");

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
