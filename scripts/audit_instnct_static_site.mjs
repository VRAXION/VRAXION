import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const root = process.cwd();
const homePath = path.join(root, "docs", "index.html");
const htmlPath = path.join(root, "docs", "instnct", "index.html");
const anchorcellPath = path.join(root, "docs", "anchorcell", "index.html");
const hiddenSurfaceSlug = ["vn", "gard"].join("");
const publishedHiddenSurfacePath = path.join(root, "docs", hiddenSurfaceSlug);
const notFoundPath = path.join(root, "docs", "404.html");
const redirectsPath = path.join(root, "docs", "_redirects");
const jsPath = path.join(root, "docs", "instnct", "instnct.js");
const cssPath = path.join(root, "docs", "instnct", "styles.css");
const anchorcellJsPath = path.join(root, "docs", "anchorcell", "anchorcell.js");
const anchorcellCssPath = path.join(root, "docs", "anchorcell", "styles.css");
const anchorcellSchemaPath = path.join(root, "docs", "anchorcell", "anchorcell.v2.schema.json");
const anchorcellExamplePath = path.join(root, "docs", "anchorcell", "anchorcell.v2.example.json");
const browserSmokePath = path.join(root, "scripts", "smoke_instnct_browser.mjs");
const versionPath = path.join(root, "docs", "VERSION.json");
const robotsPath = path.join(root, "docs", "robots.txt");
const sitemapPath = path.join(root, "docs", "sitemap.xml");
const currentCapabilitiesPath = path.join(root, "docs", "CURRENT_CAPABILITIES.md");
const benchmarkNotesPath = path.join(root, "docs", "INSTNCT_BENCHMARK_NOTES.md");
const docsRoot = path.join(root, "docs");
const siteRoot = path.join(root, "docs", "instnct");

const html = fs.readFileSync(htmlPath, "utf8");
const home = fs.readFileSync(homePath, "utf8");
const anchorcell = fs.readFileSync(anchorcellPath, "utf8");
const js = fs.readFileSync(jsPath, "utf8");
const css = fs.readFileSync(cssPath, "utf8");
const anchorcellJs = fs.readFileSync(anchorcellJsPath, "utf8");
const anchorcellCss = fs.readFileSync(anchorcellCssPath, "utf8");
const anchorcellSchemaText = fs.readFileSync(anchorcellSchemaPath, "utf8");
const anchorcellExampleText = fs.readFileSync(anchorcellExamplePath, "utf8");
const notFound = fs.readFileSync(notFoundPath, "utf8");
const redirects = fs.readFileSync(redirectsPath, "utf8");
const browserSmoke = fs.readFileSync(browserSmokePath, "utf8");
const currentCapabilities = fs.readFileSync(currentCapabilitiesPath, "utf8");
const benchmarkNotes = fs.readFileSync(benchmarkNotesPath, "utf8");
const token = (...parts) => parts.join("");
let latestRelease = "";
let instnctAssetVersion = "";
let anchorcellAssetVersion = "";
const failures = [];

function fail(message) {
  failures.push(message);
}

function attr(tag, name) {
  const match = tag.match(new RegExp(`\\s${name}="([^"]*)"`, "i"));
  return match ? match[1] : "";
}

function isExternalRef(value) {
  return /^(?:https?:)?\/\//i.test(value);
}

function isLocalOrDataRef(value) {
  return (
    value.startsWith("./") ||
    value.startsWith("../") ||
    value.startsWith("#") ||
    value.startsWith("data:image/")
  );
}

function parseCsp(content) {
  const directives = new Map();
  for (const part of content.split(";")) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    const [name, ...tokens] = trimmed.split(/\s+/);
    directives.set(name, tokens);
  }
  return directives;
}

function metaContent(name, value) {
  const pattern = new RegExp(`<meta\\s+[^>]*(?:name|property)="${name}"[^>]*content="([^"]*)"[^>]*>`, "i");
  const match = html.match(pattern);
  if (!match) fail(`missing meta: ${name}`);
  return match ? match[1] : "";
}

function metaContentFor(label, markup, name) {
  const pattern = new RegExp(`<meta\\s+[^>]*(?:name|property)="${name}"[^>]*content="([^"]*)"[^>]*>`, "i");
  const match = markup.match(pattern);
  if (!match) fail(`${label} missing meta: ${name}`);
  return match ? match[1] : "";
}

function pngSize(filePath) {
  const buf = fs.readFileSync(filePath);
  const signature = "89504e470d0a1a0a";
  if (buf.subarray(0, 8).toString("hex") !== signature) {
    throw new Error(`${filePath} is not a PNG`);
  }
  return {
    width: buf.readUInt32BE(16),
    height: buf.readUInt32BE(20),
  };
}

function jpegSize(filePath) {
  const buf = fs.readFileSync(filePath);
  if (buf[0] !== 0xff || buf[1] !== 0xd8) {
    throw new Error(`${filePath} is not a JPEG`);
  }
  let offset = 2;
  while (offset < buf.length) {
    while (buf[offset] === 0xff) offset += 1;
    const marker = buf[offset];
    offset += 1;
    if (marker === 0xd9 || marker === 0xda) break;
    const length = buf.readUInt16BE(offset);
    if (length < 2) throw new Error(`${filePath} has an invalid JPEG segment length`);
    if ((marker >= 0xc0 && marker <= 0xc3) || (marker >= 0xc5 && marker <= 0xc7) || (marker >= 0xc9 && marker <= 0xcb) || (marker >= 0xcd && marker <= 0xcf)) {
      return {
        height: buf.readUInt16BE(offset + 3),
        width: buf.readUInt16BE(offset + 5),
      };
    }
    offset += length;
  }
  throw new Error(`${filePath} does not expose JPEG dimensions`);
}

function imageSize(filePath) {
  const lower = filePath.toLowerCase();
  if (lower.endsWith(".png")) return pngSize(filePath);
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return jpegSize(filePath);
  throw new Error(`${filePath} is not a supported social image format`);
}

function validateDocumentRefs(label, markup) {
  const ids = new Set([...markup.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]));
  for (const tagMatch of markup.matchAll(/<[^>]+>/g)) {
    const tag = tagMatch[0];
    for (const attrName of ["aria-controls", "aria-labelledby", "aria-describedby"]) {
      const value = attr(tag, attrName);
      if (!value) continue;
      for (const id of value.split(/\s+/).filter(Boolean)) {
        if (!ids.has(id)) fail(`${label} broken ${attrName} reference: #${id}`);
      }
    }
  }
  for (const match of markup.matchAll(/\shref="#([^"]+)"/g)) {
    if (!ids.has(match[1])) fail(`${label} broken hash link: #${match[1]}`);
  }
}

function validateSocialImage({ label, markup, prefix, rootDir, expectedAlt }) {
  const siteName = metaContentFor(label, markup, "og:site_name");
  const ogImage = metaContentFor(label, markup, "og:image");
  const twitterImage = metaContentFor(label, markup, "twitter:image");
  const ogAlt = metaContentFor(label, markup, "og:image:alt");
  const twitterAlt = metaContentFor(label, markup, "twitter:image:alt");
  const ogWidth = Number(metaContentFor(label, markup, "og:image:width"));
  const ogHeight = Number(metaContentFor(label, markup, "og:image:height"));
  if (siteName !== "VRAXION") fail(`${label} og:site_name must be VRAXION`);
  if (twitterImage !== ogImage) fail(`${label} twitter:image must match og:image`);
  if (ogAlt !== expectedAlt) fail(`${label} og:image:alt mismatch: ${ogAlt}`);
  if (twitterAlt !== expectedAlt) fail(`${label} twitter:image:alt mismatch: ${twitterAlt}`);
  if (!ogImage.startsWith(prefix)) {
    fail(`${label} og:image must use Pages prefix: ${ogImage}`);
    return;
  }
  const local = path.join(rootDir, ogImage.slice(prefix.length).replaceAll("/", path.sep));
  if (!fs.existsSync(local)) {
    fail(`${label} missing og:image asset: ${local}`);
    return;
  }
  const size = imageSize(local);
  if (size.width !== ogWidth || size.height !== ogHeight) {
    fail(`${label} og:image dimensions ${ogWidth}x${ogHeight} do not match asset ${size.width}x${size.height}`);
  }
}

try {
  new vm.Script(js, { filename: jsPath });
} catch (err) {
  fail(`INSTNCT JS syntax error: ${err.message}`);
}
try {
  new vm.Script(anchorcellJs, { filename: anchorcellJsPath });
} catch (err) {
  fail(`AnchorCell JS syntax error: ${err.message}`);
}

try {
  const version = JSON.parse(fs.readFileSync(versionPath, "utf8"));
  const versionDate = String(version.date || "");
  if (versionDate !== "2026-07-05") {
    fail(`docs/VERSION.json date must match the current public site release date: ${versionDate || "missing"}`);
  }
  latestRelease = String(version.latest_public_release || "");
  if (!latestRelease) fail("docs/VERSION.json must define latest_public_release");
  instnctAssetVersion = String(version.instnct_asset_version || "");
  if (!/^release-\d+$/.test(instnctAssetVersion)) {
    fail(`docs/VERSION.json instnct_asset_version is invalid: ${instnctAssetVersion || "missing"}`);
  }
  anchorcellAssetVersion = String(version.anchorcell_asset_version || "");
  if (!/^research-\d+$/.test(anchorcellAssetVersion)) {
    fail(`docs/VERSION.json anchorcell_asset_version is invalid: ${anchorcellAssetVersion || "missing"}`);
  }
  for (const [field, value] of Object.entries(version)) {
    if (typeof value === "string" && /\bboundary\b/i.test(value)) {
      fail(`docs/VERSION.json public field ${field} must use scope wording, not boundary`);
    }
  }
} catch (err) {
  fail(`invalid docs/VERSION.json: ${err.message}`);
}

try {
  const schema = JSON.parse(anchorcellSchemaText);
  const serialized = JSON.stringify(schema);
  if (schema.$schema !== "https://json-schema.org/draft/2020-12/schema") {
    fail(`AnchorCell schema must use JSON Schema Draft 2020-12, found ${schema.$schema || "missing"}`);
  }
  if (schema.$id !== "https://vraxion.github.io/VRAXION/anchorcell/anchorcell.v2.schema.json") {
    fail(`AnchorCell schema $id must use the public VRAXION URL, found ${schema.$id || "missing"}`);
  }
  if (schema.title !== "AnchorCell Finalized Authoring Schema") {
    fail(`AnchorCell schema title mismatch: ${schema.title || "missing"}`);
  }
  if (schema.unevaluatedProperties !== false) {
    fail("AnchorCell schema must close unevaluated top-level properties");
  }
  if (schema.properties?.schema_version?.const !== "alphasync.anchorcell.v2") {
    fail("AnchorCell schema_version const must be alphasync.anchorcell.v2");
  }
  const requiredTop = new Set(schema.required || []);
  for (const required of [
    "schema_version",
    "schema_revision",
    "cell_id",
    "public_export_status",
    "context_packet",
    "branches",
    "gold",
    "review",
    "security",
  ]) {
    if (!requiredTop.has(required)) fail(`AnchorCell schema missing required top-level field: ${required}`);
  }
  if (schema.properties?.branches?.minItems !== 4 || schema.properties?.branches?.maxItems !== 8) {
    fail("AnchorCell schema branch bounds must require 4..8 branches");
  }
  if (schema.$defs?.branch?.properties?.evidence?.minItems !== 1) {
    fail("AnchorCell branch evidence must require at least one evidence ref");
  }
  for (const defName of ["contextPacket", "branch", "gold", "review", "security", "projection"]) {
    if (!schema.$defs?.[defName]) fail(`AnchorCell schema missing $defs.${defName}`);
  }
  for (const requiredText of [
    "chain_of_thought",
    "hidden_reasoning",
    "raw_prompt",
    "system_prompt",
    "public_redacted",
    "public_full",
    "allow_training_export",
    "allow_public_export",
    "Private or internal-only authoring records must not be marked as public exportable.",
    "Training-exportable records must already be accepted, reviewed, and assigned to a training/public export state.",
    "train_internal_only",
    "branch_role",
    "adversarial",
    "naive_bad",
  ]) {
    if (!serialized.includes(requiredText)) fail(`AnchorCell schema missing invariant text: ${requiredText}`);
  }
} catch (err) {
  fail(`invalid AnchorCell schema JSON: ${err.message}`);
}

try {
  const example = JSON.parse(anchorcellExampleText);
  if (example.schema_version !== "alphasync.anchorcell.v2") {
    fail(`AnchorCell example schema_version mismatch: ${example.schema_version || "missing"}`);
  }
  if (example.schema_revision !== "2.0.0") {
    fail(`AnchorCell example schema_revision mismatch: ${example.schema_revision || "missing"}`);
  }
  if (example.synthetic !== true) fail("AnchorCell example must remain explicitly synthetic");
  if (example.status !== "accepted") fail(`AnchorCell example status must be accepted, found ${example.status || "missing"}`);
  if (example.public_export_status !== "public_redacted") {
    fail(`AnchorCell example must be public_redacted, found ${example.public_export_status || "missing"}`);
  }
  if (!/^ac_[a-z0-9_]+$/.test(example.cell_id || "")) {
    fail(`AnchorCell example cell_id has unexpected format: ${example.cell_id || "missing"}`);
  }
  if (!Array.isArray(example.branches) || example.branches.length !== 4) {
    fail("AnchorCell example must include exactly four branch records");
  }
  const branchRoles = new Set((example.branches || []).map((branch) => branch.branch_role));
  for (const role of ["human", "assistant", "naive_bad", "adversarial"]) {
    if (!branchRoles.has(role)) fail(`AnchorCell example missing branch_role ${role}`);
  }
  for (const branch of example.branches || []) {
    if (!Array.isArray(branch.evidence) || branch.evidence.length < 1) {
      fail(`AnchorCell example branch ${branch.branch_id || "unknown"} must include evidence refs`);
    }
  }
  if (example.gold?.decision !== "ESCALATE") fail("AnchorCell example gold decision must be ESCALATE");
  if (!example.gold?.forbidden_decisions?.includes("EXECUTE")) {
    fail("AnchorCell example must explicitly forbid EXECUTE");
  }
  for (const [field, expected] of Object.entries({
    contains_pii: false,
    contains_secrets: false,
    requires_redaction: false,
    allow_training_export: true,
    allow_public_export: true,
    prompt_injection_tested: true,
    poisoning_risk_reviewed: true,
  })) {
    if (example.security?.[field] !== expected) {
      fail(`AnchorCell example security.${field} must be ${expected}`);
    }
  }
  const riskyStrings = [];
  const scanStrings = (value, pathParts = []) => {
    if (Array.isArray(value)) {
      value.forEach((item, index) => scanStrings(item, pathParts.concat(index)));
    } else if (value && typeof value === "object") {
      for (const [key, item] of Object.entries(value)) scanStrings(item, pathParts.concat(key));
    } else if (
      typeof value === "string" &&
      /((?:^|[\s"'(])(?:C|S):[\\/]|github\.com|VRAXION_DEV|private|secret|token|api[_-]?key|kenes|kenessy|localhost|127\.0\.0\.1|filecite|turn\d+)/i.test(value)
    ) {
      riskyStrings.push(`${pathParts.join(".")}: ${value.slice(0, 120)}`);
    }
  };
  scanStrings(example);
  if (riskyStrings.length) fail(`AnchorCell example contains public-risk strings: ${riskyStrings.join("; ")}`);
} catch (err) {
  fail(`invalid AnchorCell example JSON: ${err.message}`);
}

for (const required of [
  "hero-mesh-canvas",
  "icon-sprite",
  "#icon-terminal",
  "mode-switch",
  "data-cli-demo",
  "data-manifesto",
  "fabric-flow-panel",
  "fabric-flow-canvas",
  "keyboard-dialog",
  'main id="main" tabindex="-1"',
  "release-snapshot-pill",
  "data-benchmark",
  "terminal-actions",
  "terminal-note",
  "artifact-status",
  "mobile-section-readout",
  "data-copy-status",
  "data-mobile-indicator-number",
  "no form fields",
  "planned local flow",
  "preview command shape:",
  "../INSTNCT_BENCHMARK_NOTES.md",
  "T1 REFLEX ENGINE",
  "Exact Mode",
  "Path Selector",
  "Proof Pack",
  "scoped to avoid hosted LLM/model-wrapper dependencies",
  "runnable Proof Pack is still pending",
  "Public SDK/docs ZIP",
  "engine note / T1",
  "public artifact terms",
  "Proof Pack pending",
  "Is the engine implementation public today?",
  "The T1 Proof Pack target is concrete",
  "The first runnable release is defined as a Proof Pack",
  "Runtime telemetry claims belong with a signed artifact",
  "founder-led, not committee-built",
  "ai-assisted, human-owned",
  "claims ship with evidence, not vibes",
  "local artifact target",
]) {
  if (!html.includes(required)) fail(`missing INSTNCT markup: ${required}`);
}

for (const required of [
  '<a class="skip-link" href="#main">Skip to content</a>',
  '<main id="main" tabindex="-1">',
  ".skip-link:focus,",
  ".skip-link:focus-visible",
  "VRAXION / INSTNCT T1 Reflex Engine",
  "home-badge-row",
  "hero-wordmark",
  "./assets/vraxion-wordmark.png?v=brand-pass-1",
  "Founder-built. AI-assisted. Evidence-gated.",
  "./assets/fonts/geist-sans-variable.woff2",
  "Exact Mode refusal",
  "solo-built, AI-assisted",
  "Signed T1 Proof Pack pending",
  "Meet INSTNCT, the first public VRAXION engine target.",
  "The engine contract: answer on-path, refuse off-path.",
  "AnchorCell studies the format before the model.",
  "This path is not a model announcement.",
  "v2 authoring schema baseline",
  "./anchorcell/",
  "./ANCHORCELL_RESEARCH_BRIEF.md",
  "./anchorcell/anchorcell.v2.schema.json",
  "./anchorcell/anchorcell.v2.example.json",
  "Path Selector",
  "Exact Mode",
  "Proof Pack",
  "local reflex reasoning engine that says yes inside known paths and no outside them",
]) {
  if (!home.includes(required)) fail(`missing homepage positioning copy: ${required}`);
}

for (const required of [
  '<a class="skip-link" href="#main">Skip to content</a>',
  '<main id="main" tabindex="-1">',
  '<link rel="canonical" href="https://vraxion.github.io/VRAXION/anchorcell/">',
  "AnchorCell is a Vraxion research direction",
  "Training data with its trust boundaries intact.",
  "one cell per decision",
  "trusted policy separated from untrusted input",
  "section-indicator",
  "mobile-section-readout",
  "hero-cursor-glow",
  "candidate_primary",
  "naive_bad",
  "adversarial",
  "public_redacted",
  "JSON Schema Draft 2020-12 authoring baseline",
  "not a finished model claim",
  "../ANCHORCELL_RESEARCH_BRIEF.md",
  "./anchorcell.v2.schema.json",
  "./anchorcell.v2.example.json",
  "Inspect v2 schema",
  "Open example cell",
]) {
  if (!anchorcell.includes(required)) fail(`missing AnchorCell markup: ${required}`);
}

const keyboardTrigger = html.match(/<button class="keyboard-help-trigger"[^>]*>/)?.[0] || "";
if (
  attr(keyboardTrigger, "aria-haspopup") !== "dialog" ||
  attr(keyboardTrigger, "aria-controls") !== "keyboard-dialog" ||
  attr(keyboardTrigger, "aria-expanded") !== "false"
) {
  fail("keyboard shortcut trigger must expose dialog controls and closed state");
}
if (!html.includes('id="keyboard-dialog" class="keyboard-dialog" role="dialog"')) {
  fail("keyboard shortcut dialog must expose a stable id for aria-controls");
}

for (const required of [
  "installHeroMesh",
  "installFabricFlow",
  "installCliDemo",
  "installWallpaperParallax",
  "animateReadout",
  "installReveals",
  "shouldAnimateMesh",
  "shouldAnimateFlow",
  "data-copy-command",
  "data-copy-status",
  "α-SYNC",
  "setKeyboardBackgroundInert",
  "canUsePageShortcut",
  'smoothScrollTo("#get-notified", true)',
  'smoothScrollTo("#main", true)',
  'keyboardTrigger?.setAttribute("aria-expanded", "true")',
  'keyboardTrigger?.setAttribute("aria-expanded", "false")',
  "dataset.lineType",
  "aria-hidden",
  "data-wallpaper-section",
  "is-indicator-open",
]) {
  if (!js.includes(required)) fail(`missing INSTNCT enhancement: ${required}`);
}

for (const required of [
  "data-section-link",
  "is-pointer-active",
  "is-revealed",
  "prefers-reduced-motion",
  "--hero-bg-x",
  "--hero-glow-x",
  "data-indicator-current",
  "data-mobile-label",
]) {
  if (!anchorcellJs.includes(required)) fail(`missing AnchorCell enhancement: ${required}`);
}

for (const required of [
  ".fabric-flow-panel",
  "#not-ai::after",
  "engine-scope-bg.jpg",
  "#hallucination::after",
  "exact-mode-bg.jpg",
  "#trust::after",
  "proof-pack-bg.jpg",
  "#grounding::after",
  "release-claim-bg.jpg",
  "constraints-founder-bg.jpg",
  "fabric-result-bg.jpg",
  "cli-proof-bg.jpg",
  ".release-snapshot-pill",
  ".button-icon",
  ".card-icon",
  ".section-indicator.is-indicator-open ol",
  "::-webkit-scrollbar-thumb",
  ".has-js [data-reveal]",
  "@keyframes readoutSwap",
  ".terminal-line[data-line-type=\"ok\"]",
  ".terminal-bar em",
  ".faq-item button",
  ".terminal-note",
  ".artifact-status",
  ".mobile-section-readout",
  ".wallpaper-section",
  "--wallpaper-scroll-y",
  ".has-js .section-indicator",
  ".indicator-track:hover ~ ol",
  ".has-js .keyboard-help-trigger",
  "@media (max-width: 1360px)",
  ".has-js .faq-panel",
  ".faq-item button[aria-expanded=\"true\"]::after",
  "@media (max-width: 420px) and (max-height: 700px)",
]) {
  if (!css.includes(required)) fail(`missing INSTNCT style: ${required}`);
}

for (const required of [
  ".section-indicator",
  ".mobile-section-readout",
  ".hero-cursor-glow",
  ".hero-background img",
  ".cell-diagram",
  ".branch-grid",
  ".export-list",
  ".proof-grid",
  ".has-js .hero.is-booted",
  "@media (max-width: 1360px)",
  "@media (prefers-reduced-motion: reduce)",
]) {
  if (!anchorcellCss.includes(required)) fail(`missing AnchorCell style: ${required}`);
}

for (const required of [
  "byteSizeForServedPath",
  "probeInstnctPerformanceBudget",
  "probeInstnctNoJs",
  "hero pointer interaction did not update visual state",
  "wallpaper parallax did not update visual state",
  "section rail is missing or overlaps content",
  "hero scroll motion should move downward and fade",
  "section indicator did not track fabric section",
  "mobile section readout did not track fabric section",
  "no-js JS-only controls should be hidden",
  "skip link should exist offscreen before focus",
  "skip link is not the first visible keyboard target",
  "totalBytes",
  "externalResources",
]) {
  if (!browserSmoke.includes(required)) fail(`missing INSTNCT browser smoke guard: ${required}`);
}

for (const forbidden of [
  /\bfetch\s*\(/,
  /\bXMLHttpRequest\b/,
  /\bsendBeacon\b/,
  /\blocalStorage\b/,
  /\bsessionStorage\b/,
  /\beval\s*\(/,
  /\bnew\s+Function\b/,
]) {
  if (forbidden.test(js)) fail(`forbidden client API in INSTNCT JS: ${forbidden}`);
}
for (const forbidden of [
  /\bfetch\s*\(/,
  /\bXMLHttpRequest\b/,
  /\bsendBeacon\b/,
  /\blocalStorage\b/,
  /\bsessionStorage\b/,
  /\beval\s*\(/,
  /\bnew\s+Function\b/,
]) {
  if (forbidden.test(anchorcellJs)) fail(`forbidden client API in AnchorCell JS: ${forbidden}`);
}

for (const forbidden of [
  "/api/notify",
  "instnct-website.zip",
  "<form",
  'type="email"',
]) {
  if (html.includes(forbidden)) fail(`unsafe static-page boundary token: ${forbidden}`);
}

for (const forbidden of [
  `href="./${hiddenSurfaceSlug}/"`,
  "hidden roadmap retained",
  "Open roadmap concept",
]) {
  if (home.includes(forbidden)) fail(`hidden roadmap surface should not be visible on the public homepage: ${forbidden}`);
}
for (const forbidden of [
  "future product concepts",
  "Product paths",
  "Start from VRAXION, then branch by surface",
]) {
  if (home.includes(forbidden)) fail(`stale homepage path copy should not be visible: ${forbidden}`);
}
for (const forbidden of [
  `href="../${hiddenSurfaceSlug}/"`,
  "hidden roadmap",
]) {
  if (html.includes(forbidden)) fail(`hidden roadmap surface should not be visible on the INSTNCT page: ${forbidden}`);
}
if (fs.existsSync(publishedHiddenSurfacePath)) {
  fail("hidden roadmap surface must not exist under docs/ because GitHub Pages publishes that path");
}
for (const required of [
  `/${hiddenSurfaceSlug} /404.html 404`,
  `/${hiddenSurfaceSlug}/ /404.html 404`,
  `/${hiddenSurfaceSlug}/* /404.html 404`,
]) {
  if (!redirects.includes(required)) fail(`Cloudflare hidden-route redirect is missing: ${required}`);
}
for (const required of [
  "Page not found.",
  "This public site only exposes the current VRAXION, INSTNCT, and AnchorCell surfaces.",
  "default-src 'none'",
  "noindex,nofollow",
]) {
  if (!notFound.includes(required)) fail(`404 page is missing public-safe marker: ${required}`);
}

if (latestRelease && !html.includes(latestRelease)) {
  fail("INSTNCT page must link to the latest public boundary release");
}
if (latestRelease && !html.includes(`archive/refs/tags/${latestRelease}.zip`)) {
  fail("INSTNCT page must expose the safe public GitHub tag ZIP");
}
if (latestRelease) {
  const releaseSlugs = [...new Set([...html.matchAll(/public-sdk-p\d+-\d{8}/g)].map((match) => match[0]))];
  for (const slug of releaseSlugs) {
    if (slug !== latestRelease) fail(`release slug ${slug} does not match docs/VERSION.json ${latestRelease}`);
  }
}
if (instnctAssetVersion) {
  if (!html.includes(`./styles.css?v=${instnctAssetVersion}`)) {
    fail("INSTNCT stylesheet cache key must match docs/VERSION.json instnct_asset_version");
  }
  if (!html.includes(`./instnct.js?v=${instnctAssetVersion}`)) {
    fail("INSTNCT script cache key must match docs/VERSION.json instnct_asset_version");
  }
  const assetVersions = [
    ...html.matchAll(/\.(?:css|js)\?v=(release-\d+)/g),
  ].map((match) => match[1]);
  for (const version of new Set(assetVersions)) {
    if (version !== instnctAssetVersion) {
      fail(`INSTNCT asset cache key ${version} does not match docs/VERSION.json ${instnctAssetVersion}`);
    }
  }
  if (!browserSmoke.includes("instnctAssetVersion")) {
    fail("INSTNCT browser smoke must read the asset cache key from docs/VERSION.json");
  }
}
if (!html.includes("tag ZIP contains the public SDK/docs snapshot only")) {
  fail("INSTNCT tag ZIP note must state the public SDK/docs scope");
}
if (!html.includes("not a runnable T1 artifact") || !html.includes("excludes non-public implementation materials")) {
  fail("INSTNCT tag ZIP note must state the non-public implementation and runnable-artifact scope");
}
for (const forbiddenCopy of [
  "Not AI",
  "Not ever",
  "Runs locally",
  "Local Reasoning Runtime",
  "public proof target",
  "proof surface",
  "runtime project",
  "governed runtime frame",
  "Founder-led runtime work",
  "VRAXION runtime principles",
  "preview live",
  "INSTNCT preview live",
  "signed offline verification",
  "T1 Reflex release target",
  "The demo should be a local command",
  "Skill libraries",
  "Node mesh",
  "Make local intelligence",
  "production backend target",
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
  "model note / T1",
  "terms pending",
  "final release terms",
  "What are the public release terms?",
  "should eventually",
  "should arrive",
  "should ship",
  "should be made",
  "release-14",
  "release-13",
  token("source", "-available"),
  token("source", " available"),
  token("open ", "source"),
  token("open-", "source"),
  token("Source ", "snapshot"),
  token("source ", "snapshot"),
  token("source ", "archive"),
  token("public source ", "archive"),
  token("page ", "source"),
  token("Boundary ", "snapshot"),
  token("boundary", "-first language"),
  token("boundary ", "archive"),
  token("P11 boundary ", "archive"),
  token("P11 SDK ", "boundary"),
  token("binary ", "boundary"),
  token("Claim ", "Boundary"),
  token("claim ", "boundaries"),
  token("release ", "boundary"),
  token("Release ", "boundary"),
  token("Release ", "Boundary"),
  token("boundary text ", "versioned in repo"),
  token("boundary and ", "release target"),
  token("release ", "boundaries"),
  token("mode ", "boundary"),
  token("returns a ", "boundary"),
  token("public mark ", "boundary"),
  token("opt-in ", "boundary"),
  token(">bound", "ary</span>"),
]) {
  if (html.includes(forbiddenCopy)) fail(`unsafe or internal public copy is visible: ${forbiddenCopy}`);
  if (home.includes(forbiddenCopy)) fail(`unsafe or internal public homepage copy is visible: ${forbiddenCopy}`);
  if (js.includes(forbiddenCopy)) fail(`unsafe or internal public JS copy is visible: ${forbiddenCopy}`);
  if (anchorcell.includes(forbiddenCopy)) fail(`unsafe or internal AnchorCell copy is visible: ${forbiddenCopy}`);
  if (anchorcellJs.includes(forbiddenCopy)) fail(`unsafe or internal AnchorCell JS copy is visible: ${forbiddenCopy}`);
  if (currentCapabilities.includes(forbiddenCopy)) {
    fail(`unsafe or internal public capabilities copy is visible: ${forbiddenCopy}`);
  }
  if (benchmarkNotes.includes(forbiddenCopy)) {
    fail(`unsafe or internal public benchmark copy is visible: ${forbiddenCopy}`);
  }
}

if (!html.includes("connect-src 'none'")) fail("CSP must keep connect-src 'none'");
if (!html.includes("form-action 'none'")) fail("CSP must keep form-action 'none'");
if (!anchorcell.includes("connect-src 'none'")) fail("AnchorCell CSP must keep connect-src 'none'");
if (!anchorcell.includes("form-action 'none'")) fail("AnchorCell CSP must keep form-action 'none'");

const homeCspTag = home.match(/<meta\s+[^>]*http-equiv="Content-Security-Policy"[^>]*>/i)?.[0] || "";
const homeCsp = attr(homeCspTag, "content");
if (!homeCsp) fail("homepage missing Content-Security-Policy meta");
const homeCspDirectives = parseCsp(homeCsp);
const homeStyleMatch = home.match(/<style>([\s\S]*?)<\/style>/i);
if (!homeStyleMatch) {
  fail("homepage inline style block is missing");
} else {
  const homeStyleHash = createHash("sha256").update(homeStyleMatch[1].replace(/\r\n/g, "\n")).digest("base64");
  if (!homeCsp.includes(`'sha256-${homeStyleHash}'`)) {
    fail(`homepage CSP style hash mismatch: expected 'sha256-${homeStyleHash}'`);
  }
}
for (const [name, expected] of [
  ["default-src", ["'self'"]],
  ["script-src", ["'none'"]],
  ["img-src", ["'self'", "data:"]],
  ["connect-src", ["'none'"]],
  ["object-src", ["'none'"]],
  ["base-uri", ["'none'"]],
  ["form-action", ["'none'"]],
]) {
  const actual = homeCspDirectives.get(name) || [];
  if (actual.join(" ") !== expected.join(" ")) {
    fail(`homepage CSP ${name} mismatch: ${actual.join(" ") || "missing"}`);
  }
}
const homeStyleSrc = homeCspDirectives.get("style-src") || [];
if (homeStyleSrc.length !== 2 || homeStyleSrc[0] !== "'self'" || !/^'sha256-[^']+'$/.test(homeStyleSrc[1])) {
  fail(`homepage CSP style-src mismatch: ${homeStyleSrc.join(" ") || "missing"}`);
}
if (!homeCspDirectives.has("upgrade-insecure-requests")) {
  fail("homepage CSP missing upgrade-insecure-requests");
}
if (homeCsp.includes("'unsafe-inline'") || homeCsp.includes("'unsafe-eval'")) {
  fail("homepage CSP must not allow unsafe inline/eval");
}

const cspTag = html.match(/<meta\s+[^>]*http-equiv="Content-Security-Policy"[^>]*>/i)?.[0] || "";
const csp = attr(cspTag, "content");
if (!csp) fail("missing Content-Security-Policy meta");
const cspDirectives = parseCsp(csp);
for (const [name, expected] of [
  ["default-src", ["'self'"]],
  ["style-src", ["'self'"]],
  ["img-src", ["'self'", "data:"]],
  ["connect-src", ["'none'"]],
  ["object-src", ["'none'"]],
  ["base-uri", ["'none'"]],
  ["form-action", ["'none'"]],
]) {
  const actual = cspDirectives.get(name) || [];
  if (actual.join(" ") !== expected.join(" ")) {
    fail(`CSP ${name} mismatch: ${actual.join(" ") || "missing"}`);
  }
}
if (!cspDirectives.has("upgrade-insecure-requests")) fail("CSP missing upgrade-insecure-requests");

const anchorcellCspTag = anchorcell.match(/<meta\s+[^>]*http-equiv="Content-Security-Policy"[^>]*>/i)?.[0] || "";
const anchorcellCsp = attr(anchorcellCspTag, "content");
if (!anchorcellCsp) fail("AnchorCell missing Content-Security-Policy meta");
const anchorcellCspDirectives = parseCsp(anchorcellCsp);
for (const [name, expected] of [
  ["default-src", ["'self'"]],
  ["script-src", ["'self'"]],
  ["style-src", ["'self'"]],
  ["img-src", ["'self'", "data:"]],
  ["connect-src", ["'none'"]],
  ["object-src", ["'none'"]],
  ["base-uri", ["'none'"]],
  ["form-action", ["'none'"]],
]) {
  const actual = anchorcellCspDirectives.get(name) || [];
  if (actual.join(" ") !== expected.join(" ")) {
    fail(`AnchorCell CSP ${name} mismatch: ${actual.join(" ") || "missing"}`);
  }
}
if (!anchorcellCspDirectives.has("upgrade-insecure-requests")) fail("AnchorCell CSP missing upgrade-insecure-requests");

const jsonLdMatch = html.match(/<script\s+type="application\/ld\+json">([\s\S]*?)<\/script\b[^>]*>/i);
if (!jsonLdMatch) {
  fail("missing JSON-LD script");
} else {
  const hash = createHash("sha256").update(jsonLdMatch[1]).digest("base64");
  if (!csp.includes(`'sha256-${hash}'`)) {
    fail(`CSP JSON-LD hash mismatch: expected 'sha256-${hash}'`);
  }
  try {
    const data = JSON.parse(jsonLdMatch[1]);
    if (data["@type"] !== "WebPage") fail("JSON-LD @type must be WebPage");
    if (data.name !== "INSTNCT") fail("JSON-LD name must be INSTNCT");
    if (!String(data.description || "").includes("No public runnable T1 binary")) {
      fail("JSON-LD description must state the public T1 binary scope");
    }
    if ("offers" in data) fail("JSON-LD must not expose an offer before a runnable artifact exists");
  } catch (err) {
    fail(`invalid JSON-LD: ${err.message}`);
  }
}

for (const scriptTag of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\b[^>]*>/gi)) {
  const tag = scriptTag[0];
  const type = attr(tag, "type");
  const src = attr(tag, "src");
  if (!src && type !== "application/ld+json") {
    fail("unexpected inline script without JSON-LD type");
  }
  if (src && (!isLocalOrDataRef(src) || isExternalRef(src))) {
    fail(`external script source is not allowed: ${src}`);
  }
}
for (const scriptTag of anchorcell.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\b[^>]*>/gi)) {
  const tag = scriptTag[0];
  const type = attr(tag, "type");
  const src = attr(tag, "src");
  if (!src && type !== "application/ld+json") {
    fail("AnchorCell unexpected inline script without JSON-LD type");
  }
  if (src && (!isLocalOrDataRef(src) || isExternalRef(src))) {
    fail(`AnchorCell external script source is not allowed: ${src}`);
  }
}

for (const tagMatch of html.matchAll(/<(img|source|video|audio|iframe|object)\b[^>]*>/gi)) {
  const tag = tagMatch[0];
  const tagName = tagMatch[1].toLowerCase();
  for (const name of ["src", "poster", "data"]) {
    const value = attr(tag, name);
    if (value && (!isLocalOrDataRef(value) || isExternalRef(value))) {
      fail(`external ${tagName} ${name} is not allowed: ${value}`);
    }
  }
  const srcset = attr(tag, "srcset");
  if (srcset) {
    for (const candidate of srcset.split(",")) {
      const value = candidate.trim().split(/\s+/)[0] || "";
      if (value && (!isLocalOrDataRef(value) || isExternalRef(value))) {
        fail(`external ${tagName} srcset is not allowed: ${value}`);
      }
    }
  }
}
for (const tagMatch of anchorcell.matchAll(/<(img|source|video|audio|iframe|object)\b[^>]*>/gi)) {
  const tag = tagMatch[0];
  const tagName = tagMatch[1].toLowerCase();
  for (const name of ["src", "poster", "data"]) {
    const value = attr(tag, name);
    if (value && (!isLocalOrDataRef(value) || isExternalRef(value))) {
      fail(`AnchorCell external ${tagName} ${name} is not allowed: ${value}`);
    }
  }
  const srcset = attr(tag, "srcset");
  if (srcset) {
    for (const candidate of srcset.split(",")) {
      const value = candidate.trim().split(/\s+/)[0] || "";
      if (value && (!isLocalOrDataRef(value) || isExternalRef(value))) {
        fail(`AnchorCell external ${tagName} srcset is not allowed: ${value}`);
      }
    }
  }
}

for (const linkTag of html.matchAll(/<link\b[^>]*>/gi)) {
  const tag = linkTag[0];
  const rel = attr(tag, "rel").toLowerCase();
  const href = attr(tag, "href");
  if (!href) continue;
  const isSubresource = /\b(?:stylesheet|preload|modulepreload|icon|apple-touch-icon|manifest)\b/.test(rel);
  if (isSubresource && (!isLocalOrDataRef(href) || isExternalRef(href))) {
    fail(`external link subresource is not allowed: ${href}`);
  }
}
for (const linkTag of anchorcell.matchAll(/<link\b[^>]*>/gi)) {
  const tag = linkTag[0];
  const rel = attr(tag, "rel").toLowerCase();
  const href = attr(tag, "href");
  if (!href) continue;
  const isSubresource = /\b(?:stylesheet|preload|modulepreload|icon|apple-touch-icon|manifest)\b/.test(rel);
  if (isSubresource && (!isLocalOrDataRef(href) || isExternalRef(href))) {
    fail(`AnchorCell external link subresource is not allowed: ${href}`);
  }
}

for (const importMatch of css.matchAll(/@import\s+(?:url\()?["']?([^"')\s]+)["']?\)?/gi)) {
  const value = importMatch[1];
  if (!isLocalOrDataRef(value) || isExternalRef(value)) {
    fail(`external CSS import is not allowed: ${value}`);
  }
}

for (const urlMatch of css.matchAll(/url\(["']?([^"')]+)["']?\)/gi)) {
  const value = urlMatch[1];
  if (!isLocalOrDataRef(value) || isExternalRef(value)) {
    fail(`external CSS url is not allowed: ${value}`);
  }
}
for (const urlMatch of anchorcellCss.matchAll(/url\(["']?([^"')]+)["']?\)/gi)) {
  const value = urlMatch[1];
  if (!isLocalOrDataRef(value) || isExternalRef(value)) {
    fail(`AnchorCell external CSS url is not allowed: ${value}`);
  }
}

validateDocumentRefs("homepage", home);
validateDocumentRefs("INSTNCT", html);
validateDocumentRefs("AnchorCell", anchorcell);

const sectionLinks = [...html.matchAll(/data-section-link="/g)].length;
const initialTotal = html.match(/data-indicator-total>\s*\/\s*(\d+)/)?.[1];
if (sectionLinks !== 13) fail(`expected 13 section links, found ${sectionLinks}`);
if (initialTotal !== "13") fail(`initial indicator total should be 13, found ${initialTotal || "missing"}`);

const faqItems = [...html.matchAll(/class="faq-item/g)].length;
if (faqItems < 8) fail(`expected at least 8 FAQ items, found ${faqItems}`);
for (let i = 1; i <= 8; i += 1) {
  const buttonId = `instnct-faq-button-${i}`;
  const panelId = `instnct-faq-${i}`;
  if (!html.includes(`id="${buttonId}" type="button" aria-expanded="true" aria-controls="${panelId}"`)) {
    fail(`FAQ button ${i} is missing static aria-controls wiring`);
  }
  if (!html.includes(`id="${panelId}" class="faq-panel" role="region" aria-labelledby="${buttonId}"`)) {
    fail(`FAQ panel ${i} is missing static region/label wiring`);
  }
}

const refs = [
  ...html.matchAll(/\s(?:src|href)="(\.{1,2}\/[^"#?]+)(?:[?#][^"]*)?"/g),
  ...css.matchAll(/url\("(\.{1,2}\/[^"#?]+)(?:[?#][^"]*)?"\)/g),
].map((match) => match[1]);

for (const ref of refs) {
  const base = ref.startsWith("../") ? siteRoot : path.dirname(htmlPath);
  const target = path.resolve(base, ref);
  if (!target.startsWith(docsRoot + path.sep)) {
    fail(`asset escapes docs: ${ref}`);
  } else if (!fs.existsSync(target)) {
    fail(`missing local asset: ${ref}`);
  }
}

const anchorcellRefs = [
  ...anchorcell.matchAll(/\s(?:src|href)="(\.{1,2}\/[^"#?]+)(?:[?#][^"]*)?"/g),
  ...anchorcellCss.matchAll(/url\("(\.{1,2}\/[^"#?]+)(?:[?#][^"]*)?"\)/g),
].map((match) => match[1]);

for (const ref of anchorcellRefs) {
  const target = path.resolve(path.dirname(anchorcellPath), ref);
  if (!target.startsWith(docsRoot + path.sep)) {
    fail(`AnchorCell asset escapes docs: ${ref}`);
  } else if (!fs.existsSync(target)) {
    fail(`AnchorCell missing local asset: ${ref}`);
  }
}

validateSocialImage({
  label: "homepage",
  markup: home,
  prefix: "https://vraxion.github.io/VRAXION/",
  rootDir: docsRoot,
  expectedAlt: "VRAXION local reflex reasoning hero surface",
});
validateSocialImage({
  label: "INSTNCT",
  markup: html,
  prefix: "https://vraxion.github.io/VRAXION/instnct/",
  rootDir: siteRoot,
  expectedAlt: "INSTNCT local reflex reasoning hero surface",
});
validateSocialImage({
  label: "AnchorCell",
  markup: anchorcell,
  prefix: "https://vraxion.github.io/VRAXION/",
  rootDir: docsRoot,
  expectedAlt: "AnchorCell training data research surface",
});

if (anchorcellAssetVersion) {
  if (!anchorcell.includes(`./styles.css?v=${anchorcellAssetVersion}`)) {
    fail("AnchorCell stylesheet cache key must match docs/VERSION.json anchorcell_asset_version");
  }
  if (!anchorcell.includes(`./anchorcell.js?v=${anchorcellAssetVersion}`)) {
    fail("AnchorCell script cache key must match docs/VERSION.json anchorcell_asset_version");
  }
}

if (/\.hero-mesh\s*\{[^}]*display:\s*none/i.test(css)) {
  fail("hero mesh should not be fully disabled in responsive CSS");
}
if (/\.fabric-diagram::before/.test(css)) {
  fail("old fabric-diagram connector pseudo-element should not remain");
}
if (/\.icon-dot\b/.test(css) || html.includes("icon-dot")) {
  fail("old dot-only card icon language should not remain");
}
if (/\.nav-soon\b/.test(css) || html.includes("nav-soon")) {
  fail("old hidden nav-soon shell should not remain");
}

if (!fs.existsSync(robotsPath)) fail("docs/robots.txt is missing");
else {
  const robots = fs.readFileSync(robotsPath, "utf8");
  if (!robots.includes("Sitemap: https://vraxion.github.io/VRAXION/sitemap.xml")) {
    fail("robots.txt must point at the VRAXION sitemap");
  }
}

if (!fs.existsSync(sitemapPath)) fail("docs/sitemap.xml is missing");
else {
  const sitemap = fs.readFileSync(sitemapPath, "utf8");
  for (const url of [
    "https://vraxion.github.io/VRAXION/",
    "https://vraxion.github.io/VRAXION/instnct/",
    "https://vraxion.github.io/VRAXION/anchorcell/",
  ]) {
    if (!sitemap.includes(`<loc>${url}</loc>`)) fail(`sitemap missing ${url}`);
  }
  if (sitemap.includes(`/${hiddenSurfaceSlug}/`)) fail("sitemap must not expose hidden roadmap surface");
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log("instnct_static_site_audit=pass");
