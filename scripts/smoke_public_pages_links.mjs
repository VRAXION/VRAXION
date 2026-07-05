const baseUrl = (process.env.PUBLIC_PAGES_BASE_URL || "https://vraxion.github.io/VRAXION").replace(/\/+$/, "");
const canonicalBaseUrl = (process.env.PUBLIC_PAGES_CANONICAL_BASE_URL || "https://vraxion.github.io/VRAXION").replace(/\/+$/, "");
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

function attr(tag, name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(`\\s${escaped}=(["'])(.*?)\\1`, "i");
  return tag.match(pattern)?.[2] || "";
}

function metaContent(html, name) {
  for (const match of html.matchAll(/<meta\b[^>]*>/gi)) {
    const tag = match[0];
    const metaName = attr(tag, "name") || attr(tag, "property") || attr(tag, "http-equiv");
    if (metaName.toLowerCase() === name.toLowerCase()) return attr(tag, "content");
  }
  return "";
}

function assertCsp(html, label, requirements) {
  const csp = metaContent(html, "Content-Security-Policy");
  if (!csp) {
    fail(`${label} is missing a Content-Security-Policy meta tag`);
    return;
  }
  for (const required of requirements) {
    if (!csp.includes(required)) fail(`${label} CSP is missing ${required}`);
  }
}

const homeUrl = `${baseUrl}/`;
const instnctUrl = `${baseUrl}/instnct/`;
const anchorcellUrl = `${baseUrl}/anchorcell/`;
const hiddenSurfaceUrl = `${baseUrl}/${hiddenSurfaceSlug}/`;
const robotsUrl = `${baseUrl}/robots.txt`;
const sitemapUrl = `${baseUrl}/sitemap.xml`;
const versionUrl = `${baseUrl}/VERSION.json`;

const home = await fetchText(homeUrl, "home");
const instnct = await fetchText(instnctUrl, "INSTNCT");
const anchorcell = await fetchText(anchorcellUrl, "AnchorCell");
const robots = await fetchText(robotsUrl, "robots.txt");
const sitemap = await fetchText(sitemapUrl, "sitemap.xml");
const versionText = await fetchText(versionUrl, "VERSION.json");
let latestRelease = "";
let instnctAssetVersion = "";
let anchorcellAssetVersion = "";
if (versionText) {
  try {
    const version = JSON.parse(versionText);
    latestRelease = String(version.latest_public_release || "");
    instnctAssetVersion = String(version.instnct_asset_version || "");
    anchorcellAssetVersion = String(version.anchorcell_asset_version || "");
  } catch (err) {
    fail(`live VERSION.json is invalid JSON: ${err.message}`);
  }
}
let hiddenSurfaceStatus = 0;
try {
  const hiddenSurfaceResponse = await fetchWithTimeout(hiddenSurfaceUrl, { redirect: "manual" });
  hiddenSurfaceStatus = hiddenSurfaceResponse.status;
} catch (err) {
  fail(`hidden roadmap surface URL could not be checked: ${hiddenSurfaceUrl} ${err.message}`);
}

if (!/^public-sdk-p\d+-\d{8}$/.test(latestRelease)) {
  fail(`live VERSION latest_public_release is invalid: ${latestRelease || "missing"}`);
}
if (home && latestRelease && !home.includes(`releases/tag/${latestRelease}`)) {
  fail("home does not expose the VERSION latest release");
}
for (const required of [
  "VRAXION / INSTNCT T1 Reflex Engine",
  "solo-built, AI-assisted",
  "Signed T1 Proof Pack pending",
  "Meet INSTNCT, the first public VRAXION engine target.",
  "The engine contract: answer on-path, refuse off-path.",
  "AnchorCell studies the format before the model.",
]) {
  if (home && !home.includes(required)) fail(`home live positioning copy is missing: ${required}`);
}
for (const required of [
  "Training data with its trust boundaries intact.",
  "AnchorCell is a Vraxion research direction",
  "candidate_primary",
  "public_redacted",
  "The proof is not rhetoric. It is a validator stack.",
]) {
  if (anchorcell && !anchorcell.includes(required)) fail(`AnchorCell live copy is missing: ${required}`);
}
if (instnct && latestRelease && !instnct.includes(`archive/refs/tags/${latestRelease}.zip`)) {
  fail("INSTNCT does not expose the VERSION GitHub tag ZIP");
}
if (!/^release-\d+$/.test(instnctAssetVersion)) {
  fail(`VERSION instnct_asset_version is invalid: ${instnctAssetVersion || "missing"}`);
}
if (!/^research-\d+$/.test(anchorcellAssetVersion)) {
  fail(`VERSION anchorcell_asset_version is invalid: ${anchorcellAssetVersion || "missing"}`);
}
if (
  instnct &&
  (!instnct.includes(`styles.css?v=${instnctAssetVersion}`) ||
    !instnct.includes(`instnct.js?v=${instnctAssetVersion}`))
) {
  fail("INSTNCT live page does not load the VERSION asset cache key");
}
if (
  anchorcell &&
  (!anchorcell.includes(`styles.css?v=${anchorcellAssetVersion}`) ||
    !anchorcell.includes(`anchorcell.js?v=${anchorcellAssetVersion}`))
) {
  fail("AnchorCell live page does not load the VERSION asset cache key");
}
if (instnct && unsafePublicCopyPattern.test(instnct)) {
  fail("INSTNCT public copy exposes unsafe or internal release wording");
}
if (home && unsafePublicCopyPattern.test(home)) {
  fail("home public copy exposes unsafe or internal release wording");
}
if (anchorcell && unsafePublicCopyPattern.test(anchorcell)) {
  fail("AnchorCell public copy exposes unsafe or internal release wording");
}
if (home) {
  assertCsp(home, "home", [
    "script-src 'none'",
    "connect-src 'none'",
    "form-action 'none'",
    "upgrade-insecure-requests",
  ]);
  if (!home.includes(`<link rel="canonical" href="${canonicalBaseUrl}/">`)) fail("home canonical URL is missing or stale");
}
if (instnct) {
  assertCsp(instnct, "INSTNCT", [
    "script-src 'self'",
    "connect-src 'none'",
    "form-action 'none'",
    "upgrade-insecure-requests",
  ]);
  if (!instnct.includes(`<link rel="canonical" href="${canonicalBaseUrl}/instnct/">`)) {
    fail("INSTNCT canonical URL is missing or stale");
  }
}
if (anchorcell) {
  assertCsp(anchorcell, "AnchorCell", [
    "script-src 'self'",
    "style-src 'self'",
    "connect-src 'none'",
    "form-action 'none'",
    "upgrade-insecure-requests",
  ]);
  if (!anchorcell.includes(`<link rel="canonical" href="${canonicalBaseUrl}/anchorcell/">`)) {
    fail("AnchorCell canonical URL is missing or stale");
  }
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
if (robots && !robots.includes(`${canonicalBaseUrl}/sitemap.xml`)) fail("robots.txt does not point at the live sitemap");
if (sitemap && !sitemap.includes(`${canonicalBaseUrl}/instnct/`)) fail("sitemap.xml does not include INSTNCT");
if (sitemap && !sitemap.includes(`${canonicalBaseUrl}/anchorcell/`)) fail("sitemap.xml does not include AnchorCell");
if (sitemap && sitemap.includes(`${canonicalBaseUrl}/${hiddenSurfaceSlug}/`)) fail("sitemap.xml exposes hidden roadmap surface");

const urls = new Set([
  ...collectUrls(home, homeUrl),
  ...collectUrls(instnct, instnctUrl),
  ...collectUrls(anchorcell, anchorcellUrl),
  `${baseUrl}/robots.txt`,
  `${baseUrl}/sitemap.xml`,
  ...(latestRelease
    ? [
        `https://github.com/VRAXION/VRAXION/releases/tag/${latestRelease}`,
        `https://github.com/VRAXION/VRAXION/archive/refs/tags/${latestRelease}.zip`,
      ]
    : []),
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
