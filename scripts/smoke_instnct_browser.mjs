import { createRequire } from "node:module";
import http from "node:http";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import fs from "node:fs/promises";

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const docsRoot = path.join(root, "docs");
const failures = [];

function fail(message) {
  failures.push(message);
}

let latestRelease = "";
try {
  const version = JSON.parse(await fs.readFile(path.join(docsRoot, "VERSION.json"), "utf8"));
  latestRelease = String(version.latest_public_release || "");
  if (!latestRelease) fail("VERSION.json latest_public_release is missing");
} catch (err) {
  fail(`VERSION.json could not be read: ${err.message}`);
}

const latestReleasePath = `releases/tag/${latestRelease}`;
const latestArchivePath = `archive/refs/tags/${latestRelease}.zip`;
const criticalResourceTypes = new Set(["document", "stylesheet", "script", "image", "font"]);

function trackPageFailures(page, origin, label) {
  const errors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));
  page.on("requestfailed", (request) => {
    if (request.url().startsWith(origin) && criticalResourceTypes.has(request.resourceType())) {
      errors.push(`${label} request failed ${request.resourceType()} ${request.url()}`);
    }
  });
  page.on("response", (response) => {
    const request = response.request();
    if (
      response.url().startsWith(origin) &&
      criticalResourceTypes.has(request.resourceType()) &&
      response.status() >= 400
    ) {
      errors.push(`${label} bad response ${response.status()} ${request.resourceType()} ${response.url()}`);
    }
  });
  return errors;
}

async function loadPlaywright() {
  try {
    return require("playwright");
  } catch (err) {
    const moduleRoot = process.env.PLAYWRIGHT_MODULE_ROOT;
    if (moduleRoot) {
      const mod = await import(pathToFileURL(path.join(moduleRoot, "playwright", "index.js")).href);
      return mod.default || mod;
    }
    throw err;
  }
}

function contentType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return (
    {
      ".css": "text/css; charset=utf-8",
      ".html": "text/html; charset=utf-8",
      ".js": "text/javascript; charset=utf-8",
      ".json": "application/json; charset=utf-8",
      ".md": "text/markdown; charset=utf-8",
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".svg": "image/svg+xml",
      ".woff2": "font/woff2",
    }[ext] || "application/octet-stream"
  );
}

async function startServer() {
  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url || "/", "http://127.0.0.1");
      let pathname = decodeURIComponent(url.pathname);
      if (pathname.endsWith("/")) pathname += "index.html";
      const target = path.resolve(docsRoot, `.${pathname}`);
      if (!target.startsWith(docsRoot + path.sep)) {
        res.writeHead(403);
        res.end("forbidden");
        return;
      }
      const body = await fs.readFile(target);
      res.writeHead(200, { "content-type": contentType(target) });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end("not found");
    }
  });

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  return {
    origin: `http://127.0.0.1:${address.port}`,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}

async function canvasSamples(page) {
  return page.evaluate(() =>
    [...document.querySelectorAll("canvas")].map((canvas) => {
      const ctx = canvas.getContext("2d");
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let nonblank = 0;
      for (let i = 3; i < data.length; i += 16) {
        if (data[i] !== 0) nonblank += 1;
      }
      return { width: canvas.width, height: canvas.height, nonblank };
    })
  );
}

async function assertActiveModePanelFits(page, label) {
  const fit = await page.evaluate(() => {
    const active = document.querySelector(".mode-panel.is-active");
    return active
      ? {
          mode: active.dataset.modePanel,
          hidden: active.hidden,
          scrollHeight: active.scrollHeight,
          clientHeight: active.clientHeight,
          rectHeight: Math.round(active.getBoundingClientRect().height),
        }
      : null;
  });
  if (!fit || fit.hidden || fit.scrollHeight > fit.clientHeight + 1) {
    fail(`${label} active mode panel clips content: ${JSON.stringify(fit)}`);
  }
}

async function probeHome(browser, origin) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = trackPageFailures(page, origin, "home");
  await page.goto(`${origin}/`, { waitUntil: "networkidle" });
  const result = await page.evaluate(() => ({
    overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
    latestRelease: document.body.textContent.includes("P11 SDK boundary release"),
    releaseHrefs: [...document.querySelectorAll("a")].map((a) => a.href),
    oldReleaseHref: [...document.querySelectorAll("a")].some((a) => a.href.includes("releases/tag/v6.1.7")),
    capabilitiesHref: [...document.querySelectorAll("a")].some((a) => a.href.includes("CURRENT_CAPABILITIES.md")),
    instnctLive: document.body.textContent.includes("INSTNCT live"),
  }));
  await page.close();

  if (errors.length) fail(`home browser errors: ${errors.join(" | ")}`);
  if (result.overflow) fail("home page has horizontal overflow");
  if (!result.latestRelease || !result.releaseHrefs.some((href) => href.includes(latestReleasePath))) {
    fail("home page latest release link is missing");
  }
  if (result.oldReleaseHref) fail("home page still links to old v6.1.7 release");
  if (!result.capabilitiesHref) fail("home page current capabilities link is missing");
  if (!result.instnctLive) fail("home page still frames INSTNCT as not-live");
}

async function probeInstnctDesktop(browser, origin) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = trackPageFailures(page, origin, "INSTNCT desktop");
  await page.goto(`${origin}/instnct/`, { waitUntil: "networkidle" });
  await page.waitForTimeout(400);

  const top = await page.evaluate(() => ({
    overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
    release2: [...document.querySelectorAll("link,script")].some((el) =>
      String(el.href || el.src || "").includes("release-2")
    ),
    heroMeshDisplay: getComputedStyle(document.querySelector(".hero-mesh")).display,
    heroGlowDisplay: getComputedStyle(document.querySelector(".hero-cursor-glow")).display,
    sourceHrefs: [...document.querySelectorAll("a")].map((a) => a.href),
    boundaryNote: document.querySelector(".notify-note")?.textContent.includes("not the private engine source"),
    logoAsset: document.querySelector(".wordmark img")?.getAttribute("src") || "",
    schemaType: JSON.parse(document.querySelector('script[type="application/ld+json"]').textContent)["@type"],
  }));
  if (top.overflow) fail("INSTNCT desktop has horizontal overflow");
  if (!top.release2) fail("INSTNCT desktop is not loading release-2 assets");
  if (top.heroMeshDisplay === "none" || top.heroGlowDisplay === "none") {
    fail("INSTNCT desktop hero mesh/glow is hidden");
  }
  if (!top.sourceHrefs.some((href) => href.includes(latestArchivePath)) || !top.boundaryNote) {
    fail("INSTNCT source snapshot boundary CTA is missing");
  }
  if (!top.logoAsset.includes("instnct-logo.png")) fail("INSTNCT hero is not using the GLM final logo asset");
  if (top.schemaType !== "WebPage") fail(`INSTNCT JSON-LD should be WebPage, found ${top.schemaType}`);

  await page.locator("#hallucination").scrollIntoViewIfNeeded();
  await page.waitForTimeout(200);
  await assertActiveModePanelFits(page, "desktop exact");
  const switchHit = await page.evaluate(() => {
    const el = document.querySelector(".mode-switch");
    const rect = el.getBoundingClientRect();
    const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
    return !!hit?.closest(".mode-switch");
  });
  if (!switchHit) fail("mode switch is not the clickable element at its center point");
  await page.locator(".mode-switch").click();
  const mode = await page.evaluate(() => ({
    card: document.querySelector(".mode-card").dataset.mode,
    checked: document.querySelector(".mode-switch").getAttribute("aria-checked"),
    exactHidden: document.querySelector('[data-mode-panel="exact"]').getAttribute("aria-hidden"),
    imaginationHidden: document.querySelector('[data-mode-panel="imagination"]').getAttribute("aria-hidden"),
  }));
  if (mode.card !== "imagination" || mode.checked !== "true" || mode.exactHidden !== "true" || mode.imaginationHidden !== "false") {
    fail(`mode switch did not expose imagination mode correctly: ${JSON.stringify(mode)}`);
  }
  await assertActiveModePanelFits(page, "desktop imagination");

  await page.locator(".indicator-track").hover();
  await page.waitForTimeout(240);
  const indicator = await page.evaluate(() => {
    const track = document.querySelector(".indicator-track");
    const list = document.querySelector(".section-indicator ol");
    const number = document.querySelector("[data-indicator-number]");
    const rect = track.getBoundingClientRect();
    const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
    return {
      hitIndicator: !!hit?.closest(".section-indicator"),
      listOpacity: Number(getComputedStyle(list).opacity),
      numberColor: getComputedStyle(number).color,
    };
  });
  if (!indicator.hitIndicator || indicator.listOpacity < 0.8) {
    fail(`section indicator hover target is broken: ${JSON.stringify(indicator)}`);
  }

  await page.keyboard.press("?");
  for (let i = 0; i < 8; i += 1) await page.keyboard.press("Tab");
  const dialog = await page.evaluate(() => ({
    open: !document.querySelector(".keyboard-dialog").hidden,
    focusInside: document.querySelector(".keyboard-dialog").contains(document.activeElement),
    mainInert: document.querySelector("main").hasAttribute("inert"),
  }));
  if (!dialog.open || !dialog.focusInside || !dialog.mainInert) {
    fail(`keyboard dialog is not modal: ${JSON.stringify(dialog)}`);
  }
  await page.keyboard.press("Escape");
  await page.locator("header .nav a").first().focus();
  const focusedBeforeEscape = await page.evaluate(() => document.activeElement?.textContent?.trim());
  await page.keyboard.press("Escape");
  const focusedAfterEscape = await page.evaluate(() => document.activeElement?.textContent?.trim());
  if (focusedBeforeEscape !== focusedAfterEscape) {
    fail(`Escape changed focus while keyboard dialog was closed: ${focusedBeforeEscape} -> ${focusedAfterEscape}`);
  }

  await page.locator("#fabric").scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  const fabric = await page.evaluate(() => {
    const panel = document.querySelector(".fabric-flow-panel").getBoundingClientRect();
    const diagram = document.querySelector(".fabric-diagram").getBoundingClientRect();
    return {
      panelWidth: Math.round(panel.width),
      panelHeight: Math.round(panel.height),
      diagramAfterPanel: diagram.top > panel.bottom - 1,
    };
  });
  if (fabric.panelWidth < 300 || fabric.panelHeight < 160 || !fabric.diagramAfterPanel) {
    fail(`fabric flow panel layout is invalid: ${JSON.stringify(fabric)}`);
  }

  const samples = await canvasSamples(page);
  if (samples.length !== 2 || samples.some((sample) => sample.width === 0 || sample.height === 0 || sample.nonblank === 0)) {
    fail(`canvas render check failed: ${JSON.stringify(samples)}`);
  }

  await page.locator("#faq").scrollIntoViewIfNeeded();
  const faq = await page.evaluate(() => ({
    count: document.querySelectorAll(".faq-item").length,
    openCount: document.querySelectorAll(".faq-item.is-open").length,
    hiddenCount: [...document.querySelectorAll(".faq-panel")].filter((panel) => panel.getAttribute("aria-hidden") === "true").length,
  }));
  if (faq.count < 8 || faq.openCount !== 1 || faq.hiddenCount !== faq.count - 1) {
    fail(`FAQ accessibility state is invalid: ${JSON.stringify(faq)}`);
  }

  await page.close();
  if (errors.length) fail(`INSTNCT desktop browser errors: ${errors.join(" | ")}`);
}

async function probeInstnctReducedMotion(browser, origin) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: "reduce" });
  const errors = trackPageFailures(page, origin, "INSTNCT reduced-motion");
  await page.goto(`${origin}/instnct/`, { waitUntil: "networkidle" });
  await page.locator("#dev-trail").scrollIntoViewIfNeeded();
  const terminal = await page.evaluate(() => ({
    lineCount: document.querySelectorAll(".terminal-line").length,
    types: [...new Set([...document.querySelectorAll(".terminal-line")].map((el) => el.dataset.lineType))],
    counts: [...document.querySelectorAll("[data-count-to]")].map((el) => el.textContent.trim()),
  }));
  const expectedTypes = ["prompt", "output", "trace", "ok", "query", "warn"];
  if (terminal.lineCount !== 14 || expectedTypes.some((type) => !terminal.types.includes(type))) {
    fail(`terminal reduced-motion fallback is incomplete: ${JSON.stringify(terminal)}`);
  }
  if (terminal.counts.join(",") !== "5,261,52") {
    fail(`reduced-motion benchmark counts are wrong: ${terminal.counts.join(",")}`);
  }
  await page.close();
  if (errors.length) fail(`INSTNCT reduced-motion browser errors: ${errors.join(" | ")}`);
}

async function probeInstnctMobile(browser, origin) {
  const page = await browser.newPage({ viewport: { width: 360, height: 760 } });
  const errors = trackPageFailures(page, origin, "INSTNCT mobile");
  await page.goto(`${origin}/instnct/`, { waitUntil: "networkidle" });
  await page.waitForTimeout(300);
  const mobile = await page.evaluate(() => ({
    overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
    indicatorHidden: getComputedStyle(document.querySelector(".section-indicator")).display === "none",
    keyboardTriggerHidden: getComputedStyle(document.querySelector(".keyboard-help-trigger")).display === "none",
    sourcePillHidden: getComputedStyle(document.querySelector(".source-snapshot-pill")).display === "none",
    heroMeshDisplay: getComputedStyle(document.querySelector(".hero-mesh")).display,
    sourceHrefs: [...document.querySelectorAll("a")].map((a) => a.href),
  }));
  if (mobile.overflow) fail("INSTNCT mobile has horizontal overflow");
  if (!mobile.indicatorHidden || !mobile.keyboardTriggerHidden || !mobile.sourcePillHidden) {
    fail(`INSTNCT mobile fixed controls are not hidden: ${JSON.stringify(mobile)}`);
  }
  if (mobile.heroMeshDisplay === "none") fail("INSTNCT mobile hero mesh is hidden");
  if (!mobile.sourceHrefs.some((href) => href.includes(latestArchivePath))) {
    fail("INSTNCT mobile source snapshot link is missing");
  }
  await page.close();
  if (errors.length) fail(`INSTNCT mobile browser errors: ${errors.join(" | ")}`);
}

async function probeResponsiveViewports(browser, origin) {
  const viewports = [
    { width: 320, height: 740 },
    { width: 412, height: 915 },
    { width: 768, height: 1024 },
    { width: 1024, height: 768 },
    { width: 1440, height: 900 },
  ];

  for (const viewport of viewports) {
    const label = `${viewport.width}x${viewport.height}`;
    const page = await browser.newPage({ viewport });
    const errors = trackPageFailures(page, origin, `responsive ${label}`);

    await page.goto(`${origin}/instnct/`, { waitUntil: "networkidle" });
    await page.waitForTimeout(250);
    const instnct = await page.evaluate(() => {
      const active = document.querySelector(".mode-panel.is-active");
      return {
        overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
        activePanelClipped: active ? active.scrollHeight > active.clientHeight + 1 : true,
        heroMeshDisplay: getComputedStyle(document.querySelector(".hero-mesh")).display,
      };
    });
    if (instnct.overflow) fail(`INSTNCT ${label} has horizontal overflow`);
    if (instnct.activePanelClipped) fail(`INSTNCT ${label} exact mode panel clips`);
    if (instnct.heroMeshDisplay === "none") fail(`INSTNCT ${label} hero mesh is hidden`);

    await page.locator("#hallucination").scrollIntoViewIfNeeded();
    await page.locator(".mode-switch").click();
    await assertActiveModePanelFits(page, `responsive ${label} imagination`);

    await page.goto(`${origin}/`, { waitUntil: "networkidle" });
    const home = await page.evaluate(() => {
      const nav = document.querySelector(".nav");
      return {
        overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
        navDisplay: getComputedStyle(nav).display,
        navText: nav.textContent,
      };
    });
    if (home.overflow) fail(`home ${label} has horizontal overflow`);
    if (viewport.width <= 980 && (home.navDisplay === "none" || !home.navText.includes("INSTNCT"))) {
      fail(`home ${label} mobile nav does not expose INSTNCT: ${JSON.stringify(home)}`);
    }

    await page.close();
    if (errors.length) fail(`responsive ${label} browser errors: ${errors.join(" | ")}`);
  }
}

let server;
let browser;
try {
  const playwright = await loadPlaywright();
  server = await startServer();
  browser = await playwright.chromium.launch({ headless: true });
  await probeHome(browser, server.origin);
  await probeInstnctDesktop(browser, server.origin);
  await probeInstnctReducedMotion(browser, server.origin);
  await probeInstnctMobile(browser, server.origin);
  await probeResponsiveViewports(browser, server.origin);
} finally {
  if (browser) await browser.close();
  if (server) await server.close();
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log("instnct_browser_smoke=pass");
