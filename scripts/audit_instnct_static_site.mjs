import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const root = process.cwd();
const htmlPath = path.join(root, "docs", "instnct", "index.html");
const jsPath = path.join(root, "docs", "instnct", "instnct.js");
const cssPath = path.join(root, "docs", "instnct", "styles.css");
const docsRoot = path.join(root, "docs");
const siteRoot = path.join(root, "docs", "instnct");

const html = fs.readFileSync(htmlPath, "utf8");
const js = fs.readFileSync(jsPath, "utf8");
const css = fs.readFileSync(cssPath, "utf8");
const failures = [];

function fail(message) {
  failures.push(message);
}

function attr(tag, name) {
  const match = tag.match(new RegExp(`\\s${name}="([^"]*)"`, "i"));
  return match ? match[1] : "";
}

function metaContent(name, value) {
  const pattern = new RegExp(`<meta\\s+[^>]*(?:name|property)="${name}"[^>]*content="([^"]*)"[^>]*>`, "i");
  const match = html.match(pattern);
  if (!match) fail(`missing meta: ${name}`);
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

try {
  new vm.Script(js, { filename: jsPath });
} catch (err) {
  fail(`INSTNCT JS syntax error: ${err.message}`);
}

for (const required of [
  "hero-mesh-canvas",
  "mode-switch",
  "data-cli-demo",
  "data-manifesto",
  "fabric-flow-panel",
  "fabric-flow-canvas",
  "keyboard-dialog",
  "data-benchmark",
  "terminal-actions",
]) {
  if (!html.includes(required)) fail(`missing INSTNCT markup: ${required}`);
}

for (const required of [
  "installHeroMesh",
  "installFabricFlow",
  "installCliDemo",
  "data-copy-command",
  "setKeyboardBackgroundInert",
  "dataset.lineType",
  "aria-hidden",
]) {
  if (!js.includes(required)) fail(`missing INSTNCT enhancement: ${required}`);
}

for (const required of [
  ".fabric-flow-panel",
  ".terminal-line[data-line-type=\"ok\"]",
  ".terminal-bar em",
  ".faq-item button",
]) {
  if (!css.includes(required)) fail(`missing INSTNCT style: ${required}`);
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
  "/api/notify",
  "instnct-website.zip",
  "<form",
  'type="email"',
]) {
  if (html.includes(forbidden)) fail(`unsafe static-page boundary token: ${forbidden}`);
}

if (!html.includes("public-sdk-p11-20260629")) {
  fail("INSTNCT page must link to the latest public boundary release");
}
if (!html.includes("archive/refs/tags/public-sdk-p11-20260629.zip")) {
  fail("INSTNCT page must expose the safe public source snapshot archive");
}
if (!html.includes("not the private engine source")) {
  fail("INSTNCT source snapshot CTA must state the private-engine boundary");
}

if (!html.includes("connect-src 'none'")) fail("CSP must keep connect-src 'none'");
if (!html.includes("form-action 'none'")) fail("CSP must keep form-action 'none'");

const cspTag = html.match(/<meta\s+[^>]*http-equiv="Content-Security-Policy"[^>]*>/i)?.[0] || "";
const csp = attr(cspTag, "content");
if (!csp) fail("missing Content-Security-Policy meta");

const jsonLdMatch = html.match(/<script\s+type="application\/ld\+json">([\s\S]*?)<\/script>/i);
if (!jsonLdMatch) {
  fail("missing JSON-LD script");
} else {
  const hash = createHash("sha256").update(jsonLdMatch[1]).digest("base64");
  if (!csp.includes(`'sha256-${hash}'`)) {
    fail(`CSP JSON-LD hash mismatch: expected 'sha256-${hash}'`);
  }
  try {
    const data = JSON.parse(jsonLdMatch[1]);
    if (data["@type"] !== "SoftwareApplication") fail("JSON-LD @type must be SoftwareApplication");
    if (data.name !== "INSTNCT") fail("JSON-LD name must be INSTNCT");
  } catch (err) {
    fail(`invalid JSON-LD: ${err.message}`);
  }
}

for (const scriptTag of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)) {
  const tag = scriptTag[0];
  const type = attr(tag, "type");
  const src = attr(tag, "src");
  if (!src && type !== "application/ld+json") {
    fail("unexpected inline script without JSON-LD type");
  }
}

const ids = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]));
for (const match of html.matchAll(/\shref="#([^"]+)"/g)) {
  if (!ids.has(match[1])) fail(`broken hash link: #${match[1]}`);
}

const sectionLinks = [...html.matchAll(/data-section-link="/g)].length;
const initialTotal = html.match(/data-indicator-total>\s*\/\s*(\d+)/)?.[1];
if (sectionLinks !== 12) fail(`expected 12 section links, found ${sectionLinks}`);
if (initialTotal !== "12") fail(`initial indicator total should be 12, found ${initialTotal || "missing"}`);

const faqItems = [...html.matchAll(/class="faq-item/g)].length;
if (faqItems < 8) fail(`expected at least 8 FAQ items, found ${faqItems}`);

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

const ogImage = metaContent("og:image", "");
const ogWidth = Number(metaContent("og:image:width", ""));
const ogHeight = Number(metaContent("og:image:height", ""));
const expectedPrefix = "https://vraxion.github.io/VRAXION/instnct/";
if (ogImage.startsWith(expectedPrefix)) {
  const local = path.join(siteRoot, ogImage.slice(expectedPrefix.length).replaceAll("/", path.sep));
  if (!fs.existsSync(local)) {
    fail(`missing og:image asset: ${local}`);
  } else if (local.toLowerCase().endsWith(".png")) {
    const size = pngSize(local);
    if (size.width !== ogWidth || size.height !== ogHeight) {
      fail(`og:image dimensions ${ogWidth}x${ogHeight} do not match asset ${size.width}x${size.height}`);
    }
  }
} else {
  fail(`og:image must use INSTNCT Pages prefix: ${ogImage}`);
}

if (/\.hero-mesh\s*\{[^}]*display:\s*none/i.test(css)) {
  fail("hero mesh should not be fully disabled in responsive CSS");
}
if (/\.fabric-diagram::before/.test(css)) {
  fail("old fabric-diagram connector pseudo-element should not remain");
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log("instnct_static_site_audit=pass");
