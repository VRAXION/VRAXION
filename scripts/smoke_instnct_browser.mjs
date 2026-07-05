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
let instnctAssetVersion = "";
try {
  const version = JSON.parse(await fs.readFile(path.join(docsRoot, "VERSION.json"), "utf8"));
  latestRelease = String(version.latest_public_release || "");
  if (!latestRelease) fail("VERSION.json latest_public_release is missing");
  instnctAssetVersion = String(version.instnct_asset_version || "");
  if (!/^release-\d+$/.test(instnctAssetVersion)) {
    fail(`VERSION.json instnct_asset_version is invalid: ${instnctAssetVersion || "missing"}`);
  }
} catch (err) {
  fail(`VERSION.json could not be read: ${err.message}`);
}

const latestReleasePath = `releases/tag/${latestRelease}`;
const latestArchivePath = `archive/refs/tags/${latestRelease}.zip`;
const criticalResourceTypes = new Set(["document", "stylesheet", "script", "image", "font"]);
const token = (...parts) => parts.join("");
const unsafePublicCopyPatternSource = [
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
  token("source ", "available"),
  token("open ", "source"),
  token("open-", "source"),
  "governed runtime frame",
  "Founder-led runtime work",
  "VRAXION runtime principles",
  token("source ", "snapshot"),
  token("source ", "archive"),
  token("public source ", "archive"),
  token("page ", "source"),
  token("boundary ", "snapshot"),
  token("boundary ", "archive"),
  token("P11 SDK ", "boundary"),
  String.raw`\b${token("bound", "ary")}\b`,
].join("|");

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
    const home = process.env.USERPROFILE || process.env.HOME || "";
    const candidateRoots = [
      process.env.PLAYWRIGHT_MODULE_ROOT,
      process.env.CODEX_NODE_MODULES,
      home
        ? path.join(
            home,
            ".cache",
            "codex-runtimes",
            "codex-primary-runtime",
            "dependencies",
            "node",
            "node_modules"
          )
        : "",
    ].filter(Boolean);

    const expandedRoots = [];
    for (const moduleRoot of candidateRoots) {
      expandedRoots.push(moduleRoot);
      try {
        const pnpmRoot = path.join(moduleRoot, ".pnpm");
        const entries = await fs.readdir(pnpmRoot, { withFileTypes: true });
        for (const entry of entries) {
          if (entry.isDirectory() && /^playwright@/.test(entry.name)) {
            expandedRoots.push(path.join(pnpmRoot, entry.name, "node_modules"));
          }
        }
      } catch {
        // Non-pnpm installs do not have a .pnpm virtual store.
      }
    }

    for (const moduleRoot of expandedRoots) {
      try {
        const mod = await import(pathToFileURL(path.join(moduleRoot, "playwright", "index.js")).href);
        return mod.default || mod;
      } catch {
        // Try the next known runtime location before surfacing the original require error.
      }
    }

    err.message = `${err.message}. Install the repo Playwright dev dependency or set PLAYWRIGHT_MODULE_ROOT to a node_modules directory containing playwright.`;
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

async function byteSizeForServedPath(pathname) {
  let cleanPath = decodeURIComponent(pathname);
  if (cleanPath.endsWith("/")) cleanPath += "index.html";
  const target = path.resolve(docsRoot, `.${cleanPath}`);
  if (!target.startsWith(docsRoot + path.sep)) return 0;
  try {
    const stat = await fs.stat(target);
    return stat.isFile() ? stat.size : 0;
  } catch {
    return 0;
  }
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
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: "no-preference" });
  const errors = trackPageFailures(page, origin, "home");
  await page.goto(`${origin}/`, { waitUntil: "networkidle" });
  const result = await page.evaluate((latestReleaseText) => ({
    overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
    latestRelease: document.body.textContent.includes(latestReleaseText),
    releaseHrefs: [...document.querySelectorAll("a")].map((a) => a.href),
    oldReleaseHref: [...document.querySelectorAll("a")].some((a) => a.href.includes("releases/tag/v6.1.7")),
    capabilitiesHref: [...document.querySelectorAll("a")].some((a) => a.href.includes("CURRENT_CAPABILITIES.md")),
    anchorcellSchemaHref: [...document.querySelectorAll("a")].some((a) =>
      a.href.endsWith("/anchorcell/anchorcell.v2.schema.json")
    ),
    anchorcellExampleHref: [...document.querySelectorAll("a")].some((a) =>
      a.href.endsWith("/anchorcell/anchorcell.v2.example.json")
    ),
    instnctPublished: document.body.textContent.includes("INSTNCT T1 Reflex Engine preview"),
    anchorcellPublished: document.body.textContent.includes("AnchorCell studies the format before the model."),
  }), latestRelease);
  await page.close();

  if (errors.length) fail(`home browser errors: ${errors.join(" | ")}`);
  if (result.overflow) fail("home page has horizontal overflow");
  if (!result.latestRelease || !result.releaseHrefs.some((href) => href.includes(latestReleasePath))) {
    fail("home page latest release link is missing");
  }
  if (result.oldReleaseHref) fail("home page still links to old v6.1.7 release");
  if (!result.capabilitiesHref) fail("home page current capabilities link is missing");
  if (!result.anchorcellSchemaHref) fail("home page AnchorCell v2 schema link is missing");
  if (!result.anchorcellExampleHref) fail("home page AnchorCell v2 example link is missing");
  if (!result.instnctPublished) fail("home page does not frame INSTNCT as the T1 Reflex Engine preview");
  if (!result.anchorcellPublished) fail("home page does not expose the AnchorCell research path");
}

async function probeInstnctDesktop(browser, origin) {
  const page = await browser.newPage({ viewport: { width: 1600, height: 940 }, reducedMotion: "no-preference" });
  const errors = trackPageFailures(page, origin, "INSTNCT desktop");
  await page.goto(`${origin}/instnct/`, { waitUntil: "networkidle" });
  await page.waitForTimeout(400);

  const top = await page.evaluate(({ unsafeCopyPattern, assetVersion }) => ({
    overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
    currentAssetVersion: [...document.querySelectorAll("link,script")].some((el) =>
      String(el.href || el.src || "").includes(assetVersion)
    ),
    heroMeshDisplay: getComputedStyle(document.querySelector(".hero-mesh")).display,
    heroGlowDisplay: getComputedStyle(document.querySelector(".hero-cursor-glow")).display,
    boundaryHrefs: [...document.querySelectorAll("a")].map((a) => a.href),
    boundaryNote: document.querySelector(".notify-note")?.textContent.includes(
      "tag ZIP contains the public SDK/docs snapshot only"
    ),
    unsafePublicCopy: new RegExp(unsafeCopyPattern, "i").test(document.body.textContent),
    logoAsset: document.querySelector(".wordmark img")?.getAttribute("src") || "",
    schemaType: JSON.parse(document.querySelector('script[type="application/ld+json"]').textContent)["@type"],
    engineScopeWallpaper: getComputedStyle(document.querySelector("#not-ai"), "::after").backgroundImage,
    exactModeWallpaper: getComputedStyle(document.querySelector("#hallucination"), "::after").backgroundImage,
    proofPackWallpaper: getComputedStyle(document.querySelector("#trust"), "::after").backgroundImage,
    releaseClaimWallpaper: getComputedStyle(document.querySelector("#grounding"), "::after").backgroundImage,
    wallpaperSectionCount: document.querySelectorAll("[data-wallpaper-section]").length,
    proofPackHeight: Math.round(document.querySelector("#trust").getBoundingClientRect().height),
    releaseClaimHeight: Math.round(document.querySelector("#grounding").getBoundingClientRect().height),
    heroFirstImpression: (() => {
      const hero = document.querySelector(".hero");
      const headline = document.querySelector(".hero-headline");
      const actions = document.querySelector(".hero-actions");
      const principles = document.querySelector(".principle-list");
      const headlineRect = headline?.getBoundingClientRect();
      const actionsRect = actions?.getBoundingClientRect();
      return {
        booted: !!hero?.classList.contains("is-booted"),
        headlineOpacity: Number(getComputedStyle(headline).opacity),
        actionsOpacity: Number(getComputedStyle(actions).opacity),
        principlesOpacity: Number(getComputedStyle(principles).opacity),
        headlineVisible:
          !!headlineRect &&
          headlineRect.top >= 0 &&
          headlineRect.bottom <= window.innerHeight &&
          headlineRect.width > 0 &&
          headlineRect.height > 0,
        actionsVisible:
          !!actionsRect &&
          actionsRect.top >= 0 &&
          actionsRect.top < window.innerHeight &&
          actionsRect.width > 0 &&
          actionsRect.height > 0,
      };
    })(),
  }), { unsafeCopyPattern: unsafePublicCopyPatternSource, assetVersion: instnctAssetVersion });
  if (top.overflow) fail("INSTNCT desktop has horizontal overflow");
  if (!top.currentAssetVersion) fail(`INSTNCT desktop is not loading VERSION asset version ${instnctAssetVersion}`);
  if (top.heroMeshDisplay === "none" || top.heroGlowDisplay === "none") {
    fail("INSTNCT desktop hero mesh/glow is hidden");
  }
  if (!top.boundaryHrefs.some((href) => href.includes(latestArchivePath)) || !top.boundaryNote) {
    fail("INSTNCT GitHub tag ZIP CTA is missing");
  }
  if (top.unsafePublicCopy) fail("INSTNCT desktop copy exposes unsafe or internal release wording");
  if (!top.logoAsset.includes("instnct-logo.png")) fail("INSTNCT hero is not using the GLM final logo asset");
  if (top.schemaType !== "WebPage") fail(`INSTNCT JSON-LD should be WebPage, found ${top.schemaType}`);
  if (
    !top.heroFirstImpression.booted ||
    !top.heroFirstImpression.headlineVisible ||
    !top.heroFirstImpression.actionsVisible ||
    top.heroFirstImpression.headlineOpacity < 0.9 ||
    top.heroFirstImpression.actionsOpacity < 0.55 ||
    top.heroFirstImpression.principlesOpacity < 0.3
  ) {
    fail(`INSTNCT desktop hero first impression is too slow or hidden: ${JSON.stringify(top.heroFirstImpression)}`);
  }
  if (!top.engineScopeWallpaper.includes("engine-scope-bg.jpg")) {
    fail(`INSTNCT engine scope wallpaper is missing: ${top.engineScopeWallpaper}`);
  }
  if (!top.exactModeWallpaper.includes("exact-mode-bg.jpg")) {
    fail(`INSTNCT exact mode wallpaper is missing: ${top.exactModeWallpaper}`);
  }
  if (!top.proofPackWallpaper.includes("proof-pack-bg.jpg")) {
    fail(`INSTNCT proof pack wallpaper is missing: ${top.proofPackWallpaper}`);
  }
  if (!top.releaseClaimWallpaper.includes("release-claim-bg.jpg")) {
    fail(`INSTNCT release claim wallpaper is missing: ${top.releaseClaimWallpaper}`);
  }
  if (top.wallpaperSectionCount !== 6 || top.proofPackHeight < 940 || top.releaseClaimHeight < 660) {
    fail(`INSTNCT wallpaper scene sections are not expanded: ${JSON.stringify(top)}`);
  }

  const heroRect = await page.locator(".hero").boundingBox();
  if (!heroRect) {
    fail("INSTNCT hero bounding box is missing");
  } else {
    const heroBefore = await page.evaluate(() => {
      const hero = document.querySelector(".hero");
      const glow = document.querySelector(".hero-cursor-glow");
      const style = getComputedStyle(hero);
      return {
        active: hero.classList.contains("is-pointer-active"),
        bgX: style.getPropertyValue("--hero-bg-x").trim(),
        meshX: style.getPropertyValue("--hero-mesh-x").trim(),
        markX: style.getPropertyValue("--hero-mark-x").trim(),
        glowTransform: getComputedStyle(glow).transform,
      };
    });
    await page.mouse.move(heroRect.x + 120, heroRect.y + 120);
    await page.waitForTimeout(220);
    await page.mouse.move(heroRect.x + heroRect.width - 140, heroRect.y + 250);
    await page.waitForTimeout(320);
    const heroAfter = await page.evaluate(() => {
      const hero = document.querySelector(".hero");
      const glow = document.querySelector(".hero-cursor-glow");
      const style = getComputedStyle(hero);
      return {
        active: hero.classList.contains("is-pointer-active"),
        bgX: style.getPropertyValue("--hero-bg-x").trim(),
        meshX: style.getPropertyValue("--hero-mesh-x").trim(),
        markX: style.getPropertyValue("--hero-mark-x").trim(),
        glowTransform: getComputedStyle(glow).transform,
      };
    });
    if (
      !heroAfter.active ||
      (heroBefore.bgX === heroAfter.bgX && heroBefore.meshX === heroAfter.meshX && heroBefore.markX === heroAfter.markX) ||
      heroBefore.glowTransform === heroAfter.glowTransform
    ) {
      fail(`hero pointer interaction did not update visual state: ${JSON.stringify({ heroBefore, heroAfter })}`);
    }
  }

  await page.evaluate(() => {
    document.documentElement.style.scrollBehavior = "auto";
    document.body.style.scrollBehavior = "auto";
    window.scrollTo({ left: 0, top: Math.round(window.innerHeight * 0.72), behavior: "auto" });
  });
  await page.waitForFunction(() => window.scrollY >= Math.round(window.innerHeight * 0.68), null, { timeout: 1000 });
  await page.waitForFunction(() => {
    const hero = document.querySelector(".hero");
    const style = getComputedStyle(hero);
    return (
      Number.parseFloat(style.getPropertyValue("--hero-scroll-y")) > 32 &&
      Number.parseFloat(style.getPropertyValue("--hero-scroll-opacity")) < 0.78
    );
  }, null, { timeout: 1000 });
  const heroScroll = await page.evaluate(() => {
    const hero = document.querySelector(".hero");
    const style = getComputedStyle(hero);
    return {
      y: Number.parseFloat(style.getPropertyValue("--hero-scroll-y")),
      opacity: Number.parseFloat(style.getPropertyValue("--hero-scroll-opacity")),
    };
  });
  if (!(heroScroll.y > 32) || !(heroScroll.opacity < 0.78)) {
    fail(`hero scroll motion should move downward and fade: ${JSON.stringify(heroScroll)}`);
  }

  await page.locator("#trust").scrollIntoViewIfNeeded();
  await page.waitForTimeout(220);
  const wallpaperBefore = await page.evaluate(() => {
    const section = document.querySelector("#trust");
    const style = getComputedStyle(section);
    return {
      pointerX: style.getPropertyValue("--wallpaper-pointer-x").trim(),
      pointerY: style.getPropertyValue("--wallpaper-pointer-y").trim(),
      scrollY: style.getPropertyValue("--wallpaper-scroll-y").trim(),
      transform: getComputedStyle(section, "::after").transform,
    };
  });
  const trustRect = await page.locator("#trust").boundingBox();
  if (!trustRect) {
    fail("INSTNCT proof section bounding box is missing");
  } else {
    await page.mouse.move(trustRect.x + trustRect.width * 0.24, trustRect.y + trustRect.height * 0.28);
    await page.waitForTimeout(220);
    await page.mouse.move(trustRect.x + trustRect.width * 0.76, trustRect.y + trustRect.height * 0.52);
    await page.waitForTimeout(360);
  }
  const wallpaperAfter = await page.evaluate(() => {
    const section = document.querySelector("#trust");
    const style = getComputedStyle(section);
    return {
      pointerX: style.getPropertyValue("--wallpaper-pointer-x").trim(),
      pointerY: style.getPropertyValue("--wallpaper-pointer-y").trim(),
      scrollY: style.getPropertyValue("--wallpaper-scroll-y").trim(),
      transform: getComputedStyle(section, "::after").transform,
    };
  });
  if (
    wallpaperBefore.pointerX === wallpaperAfter.pointerX &&
    wallpaperBefore.pointerY === wallpaperAfter.pointerY &&
    wallpaperBefore.transform === wallpaperAfter.transform
  ) {
    fail(`wallpaper parallax did not update visual state: ${JSON.stringify({ wallpaperBefore, wallpaperAfter })}`);
  }
  await page.evaluate(() => window.scrollTo({ left: 0, top: 0, behavior: "auto" }));
  await page.waitForTimeout(120);

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

  await page.locator(".mode-switch").focus();
  await page.keyboard.press("?");
  const shortcutBlockedOnControl = await page.evaluate(() => ({
    dialogOpen: !document.querySelector(".keyboard-dialog").hidden,
    activeClass: document.activeElement?.className || "",
  }));
  if (shortcutBlockedOnControl.dialogOpen) {
    fail(`single-key shortcut opened while a control was focused: ${JSON.stringify(shortcutBlockedOnControl)}`);
  }
  await page.locator("header .nav a").first().focus();
  const scrollBeforeFocusedShortcut = await page.evaluate(() => window.scrollY);
  await page.keyboard.press("j");
  await page.waitForTimeout(120);
  const scrollAfterFocusedShortcut = await page.evaluate(() => window.scrollY);
  if (Math.abs(scrollAfterFocusedShortcut - scrollBeforeFocusedShortcut) > 2) {
    fail(`scroll shortcut fired while a link was focused: ${scrollBeforeFocusedShortcut} -> ${scrollAfterFocusedShortcut}`);
  }

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

  await page.locator("#trust").scrollIntoViewIfNeeded();
  await page.waitForTimeout(220);
  const railOverlap = await page.evaluate(() => {
    const rail = document.querySelector(".section-indicator");
    const railStyle = rail ? getComputedStyle(rail) : null;
    if (!rail || railStyle.display === "none" || railStyle.visibility === "hidden") {
      return { railVisible: false, overlaps: [] };
    }
    const railRect = rail.getBoundingClientRect();
    const targets = [...document.querySelectorAll("#trust .trust-card, #trust .center-heading, #trust .benchmark-card")];
    const overlaps = targets
      .map((target) => {
        const rect = target.getBoundingClientRect();
        const width = Math.max(0, Math.min(railRect.right, rect.right) - Math.max(railRect.left, rect.left));
        const height = Math.max(0, Math.min(railRect.bottom, rect.bottom) - Math.max(railRect.top, rect.top));
        return {
          target: target.className,
          area: Math.round(width * height),
        };
      })
      .filter((entry) => entry.area > 0);
    return { railVisible: true, overlaps };
  });
  if (!railOverlap.railVisible || railOverlap.overlaps.length) {
    fail(`section rail is missing or overlaps content: ${JSON.stringify(railOverlap)}`);
  }
  const releasePillOverlap = await page.evaluate(() => {
    const pill = document.querySelector(".release-snapshot-pill");
    const pillStyle = pill ? getComputedStyle(pill) : null;
    if (!pill || pillStyle.display === "none" || pillStyle.visibility === "hidden") {
      return { visible: false, overlaps: [] };
    }
    const pillRect = pill.getBoundingClientRect();
    const targets = [...document.querySelectorAll("#trust .trust-card, #trust .center-heading")];
    const overlaps = targets
      .map((target) => {
        const rect = target.getBoundingClientRect();
        const width = Math.max(0, Math.min(pillRect.right, rect.right) - Math.max(pillRect.left, rect.left));
        const height = Math.max(0, Math.min(pillRect.bottom, rect.bottom) - Math.max(pillRect.top, rect.top));
        return {
          target: target.className,
          area: Math.round(width * height),
          pill: {
            left: Math.round(pillRect.left),
            top: Math.round(pillRect.top),
            right: Math.round(pillRect.right),
            bottom: Math.round(pillRect.bottom),
          },
          rect: {
            left: Math.round(rect.left),
            top: Math.round(rect.top),
            right: Math.round(rect.right),
            bottom: Math.round(rect.bottom),
          },
        };
      })
      .filter((entry) => entry.area > 0);
    return { visible: true, overlaps };
  });
  if (releasePillOverlap.overlaps.length) {
    fail(`release ZIP pill overlaps proof content: ${JSON.stringify(releasePillOverlap)}`);
  }

  await page.evaluate(() => document.activeElement?.blur());
  await page.keyboard.press("?");
  for (let i = 0; i < 8; i += 1) await page.keyboard.press("Tab");
  const dialog = await page.evaluate(() => ({
    open: !document.querySelector(".keyboard-dialog").hidden,
    focusInside: document.querySelector(".keyboard-dialog").contains(document.activeElement),
    mainInert: document.querySelector("main").hasAttribute("inert"),
    dialogId: document.querySelector(".keyboard-dialog")?.id,
    triggerControls: document.querySelector(".keyboard-help-trigger")?.getAttribute("aria-controls"),
    triggerExpanded: document.querySelector(".keyboard-help-trigger")?.getAttribute("aria-expanded"),
  }));
  if (
    !dialog.open ||
    !dialog.focusInside ||
    !dialog.mainInert ||
    dialog.dialogId !== "keyboard-dialog" ||
    dialog.triggerControls !== "keyboard-dialog" ||
    dialog.triggerExpanded !== "true"
  ) {
    fail(`keyboard dialog is not modal: ${JSON.stringify(dialog)}`);
  }
  await page.keyboard.press("Escape");
  const keyboardTriggerExpandedAfterClose = await page.evaluate(() =>
    document.querySelector(".keyboard-help-trigger")?.getAttribute("aria-expanded"),
  );
  if (keyboardTriggerExpandedAfterClose !== "false") {
    fail(`keyboard dialog trigger did not return to closed state: ${keyboardTriggerExpandedAfterClose}`);
  }
  await page.locator("header .nav a").first().focus();
  const focusedBeforeEscape = await page.evaluate(() => document.activeElement?.textContent?.trim());
  await page.keyboard.press("Escape");
  const focusedAfterEscape = await page.evaluate(() => document.activeElement?.textContent?.trim());
  if (focusedBeforeEscape !== focusedAfterEscape) {
    fail(`Escape changed focus while keyboard dialog was closed: ${focusedBeforeEscape} -> ${focusedAfterEscape}`);
  }

  await page.locator("#fabric").scrollIntoViewIfNeeded();
  await page
    .waitForFunction(() => {
      const active = document.querySelector(".section-indicator a.is-active");
      const fill = document.querySelector(".indicator-track span");
      const track = document.querySelector(".indicator-track");
      if (!active || !fill || !track || active.getAttribute("href") !== "#fabric") return false;
      return Number.parseFloat(getComputedStyle(fill).height) >= Number.parseFloat(getComputedStyle(track).height) * 0.45;
    }, null, { timeout: 1600 })
    .catch(() => {});
  const sectionState = await page.evaluate(() => {
    const fill = document.querySelector(".indicator-track span");
    const active = document.querySelector(".section-indicator a.is-active");
    return {
      number: document.querySelector("[data-indicator-number]")?.textContent.trim(),
      label: document.querySelector("[data-indicator-label]")?.textContent.trim(),
      activeHref: active?.getAttribute("href"),
      activeCurrent: active?.getAttribute("aria-current"),
      fillHeight: Number.parseFloat(getComputedStyle(fill).height),
      trackHeight: Number.parseFloat(getComputedStyle(document.querySelector(".indicator-track")).height),
    };
  });
  if (
    sectionState.number !== "07" ||
    sectionState.label !== "path" ||
    sectionState.activeHref !== "#fabric" ||
    sectionState.activeCurrent !== "true" ||
    sectionState.fillHeight < sectionState.trackHeight * 0.45
  ) {
    fail(`section indicator did not track fabric section: ${JSON.stringify(sectionState)}`);
  }
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
    wiredCount: [...document.querySelectorAll(".faq-item")].filter((item) => {
      const button = item.querySelector("button");
      const panel = item.querySelector(".faq-panel");
      return (
        button &&
        panel &&
        button.getAttribute("aria-controls") === panel.id &&
        panel.getAttribute("role") === "region" &&
        panel.getAttribute("aria-labelledby") === button.id
      );
    }).length,
  }));
  if (faq.count < 8 || faq.openCount !== 1 || faq.hiddenCount !== faq.count - 1 || faq.wiredCount !== faq.count) {
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
    heroCount: document.querySelector(".live-readout [data-count-to]")?.textContent.trim() || "",
    benchmarkCounts: [...document.querySelectorAll("[data-benchmark] [data-count-to]")].map((el) =>
      el.textContent.trim()
    ),
  }));
  const expectedTypes = ["prompt", "output", "trace", "ok", "query", "warn"];
  if (terminal.lineCount !== 14 || expectedTypes.some((type) => !terminal.types.includes(type))) {
    fail(`terminal reduced-motion fallback is incomplete: ${JSON.stringify(terminal)}`);
  }
  if (terminal.heroCount !== "5" || terminal.benchmarkCounts.join(",") !== "5,261,52") {
    fail(`reduced-motion counts are wrong: ${JSON.stringify(terminal)}`);
  }
  await page.close();
  if (errors.length) fail(`INSTNCT reduced-motion browser errors: ${errors.join(" | ")}`);
}

async function probeInstnctMobile(browser, origin) {
  const page = await browser.newPage({ viewport: { width: 360, height: 760 } });
  const errors = trackPageFailures(page, origin, "INSTNCT mobile");
  await page.goto(`${origin}/instnct/`, { waitUntil: "networkidle" });
  await page.waitForTimeout(300);
  const mobile = await page.evaluate((unsafeCopyPattern) => ({
    overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
    indicatorHidden: getComputedStyle(document.querySelector(".section-indicator")).display === "none",
    keyboardTriggerHidden: getComputedStyle(document.querySelector(".keyboard-help-trigger")).display === "none",
    boundaryPillHidden: getComputedStyle(document.querySelector(".release-snapshot-pill")).display === "none",
    heroMeshDisplay: getComputedStyle(document.querySelector(".hero-mesh")).display,
    boundaryHrefs: [...document.querySelectorAll("a")].map((a) => a.href),
    unsafePublicCopy: new RegExp(unsafeCopyPattern, "i").test(document.body.textContent),
    mobileReadoutHiddenInHero:
      document.querySelector(".mobile-section-readout")?.classList.contains("is-hidden") &&
      Number(getComputedStyle(document.querySelector(".mobile-section-readout")).opacity) < 0.1,
    mobileReadoutOverlapsHeroCard: (() => {
      const el = document.querySelector(".mobile-section-readout");
      const style = el ? getComputedStyle(el) : null;
      if (!el || style.display === "none" || Number(style.opacity) < 0.1 || el.classList.contains("is-hidden")) return false;
      const readout = el.getBoundingClientRect();
      return [...document.querySelectorAll(".principle-list li")].some((card) => {
        const rect = card.getBoundingClientRect();
        return !(readout.right <= rect.left || readout.left >= rect.right || readout.bottom <= rect.top || readout.top >= rect.bottom);
      });
    })(),
  }), unsafePublicCopyPatternSource);
  if (mobile.overflow) fail("INSTNCT mobile has horizontal overflow");
  if (!mobile.indicatorHidden || !mobile.keyboardTriggerHidden || !mobile.boundaryPillHidden) {
    fail(`INSTNCT mobile fixed controls are not hidden: ${JSON.stringify(mobile)}`);
  }
  if (mobile.heroMeshDisplay === "none") fail("INSTNCT mobile hero mesh is hidden");
  if (!mobile.boundaryHrefs.some((href) => href.includes(latestArchivePath))) {
    fail("INSTNCT mobile GitHub tag ZIP link is missing");
  }
  if (mobile.unsafePublicCopy) fail("INSTNCT mobile copy exposes unsafe or internal release wording");
  if (!mobile.mobileReadoutHiddenInHero || mobile.mobileReadoutOverlapsHeroCard) {
    fail(`INSTNCT mobile section readout overlaps hero state: ${JSON.stringify(mobile)}`);
  }
  await page.locator("#fabric").scrollIntoViewIfNeeded();
  await page.waitForTimeout(280);
  const mobileIndicator = await page.evaluate(() => ({
    number: document.querySelector("[data-mobile-indicator-number]")?.textContent.trim(),
    label: document.querySelector("[data-mobile-indicator-label]")?.textContent.trim(),
    hidden: document.querySelector(".mobile-section-readout")?.classList.contains("is-hidden"),
    opacity: Number(getComputedStyle(document.querySelector(".mobile-section-readout")).opacity),
  }));
  if (mobileIndicator.number !== "07" || mobileIndicator.label !== "path" || mobileIndicator.hidden || mobileIndicator.opacity < 0.8) {
    fail(`INSTNCT mobile section readout did not track fabric section: ${JSON.stringify(mobileIndicator)}`);
  }
  const mobileFixedControls = await page.evaluate(() => {
    const readout = document.querySelector(".mobile-section-readout");
    const top = document.querySelector(".back-to-top");
    const help = document.querySelector(".keyboard-help-trigger");
    const visible = (el) => {
      if (!el) return false;
      const style = getComputedStyle(el);
      return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity) > 0.25;
    };
    if (!visible(readout)) return { overlap: false, overlaps: [] };
    const a = readout.getBoundingClientRect();
    const controls = [
      ["top", top],
      ["help", help],
    ].filter(([, el]) => visible(el));
    const overlaps = controls
      .map(([name, el]) => {
        const b = el.getBoundingClientRect();
        const overlap = !(a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom);
        return {
          name,
          overlap,
          rect: { left: Math.round(b.left), right: Math.round(b.right), top: Math.round(b.top), bottom: Math.round(b.bottom) },
        };
      })
      .filter((entry) => entry.overlap);
    return {
      overlap: overlaps.length > 0,
      overlaps,
      readout: { left: Math.round(a.left), right: Math.round(a.right), top: Math.round(a.top), bottom: Math.round(a.bottom) },
    };
  });
  if (mobileFixedControls.overlap) {
    fail(`INSTNCT mobile fixed controls overlap after scroll: ${JSON.stringify(mobileFixedControls)}`);
  }
  await page.close();
  if (errors.length) fail(`INSTNCT mobile browser errors: ${errors.join(" | ")}`);
}

async function probeInstnctNoJs(browser, origin) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, javaScriptEnabled: false });
  const errors = trackPageFailures(page, origin, "INSTNCT no-js");
  await page.goto(`${origin}/instnct/`, { waitUntil: "networkidle" });
  const noJs = await page.evaluate(() => ({
    sectionIndicatorDisplay: getComputedStyle(document.querySelector(".section-indicator")).display,
    keyboardTriggerDisplay: getComputedStyle(document.querySelector(".keyboard-help-trigger")).display,
    modeSwitchDisplay: getComputedStyle(document.querySelector(".mode-switch")).display,
    modePanels: [...document.querySelectorAll(".mode-panel")].map((panel) => ({
      mode: panel.dataset.modePanel,
      hidden: panel.hidden,
      ariaHidden: panel.getAttribute("aria-hidden"),
      height: Math.round(panel.getBoundingClientRect().height),
      scrollHeight: panel.scrollHeight,
      clientHeight: panel.clientHeight,
    })),
    faqExpanded: [...document.querySelectorAll(".faq-item button")].map((button) => button.getAttribute("aria-expanded")),
    faqPanelHeights: [...document.querySelectorAll(".faq-panel")].map((panel) => Math.round(panel.getBoundingClientRect().height)),
    mobileReadoutDesktopDisplay: getComputedStyle(document.querySelector(".mobile-section-readout")).display,
  }));
  if (noJs.sectionIndicatorDisplay !== "none" || noJs.keyboardTriggerDisplay !== "none") {
    fail(`no-js JS-only controls should be hidden: ${JSON.stringify(noJs)}`);
  }
  if (noJs.modeSwitchDisplay !== "none") fail(`no-js mode switch should be hidden: ${JSON.stringify(noJs)}`);
  if (
    noJs.modePanels.length !== 2 ||
    noJs.modePanels.some(
      (panel) => panel.hidden || panel.ariaHidden === "true" || panel.height <= 0 || panel.scrollHeight > panel.clientHeight + 1
    )
  ) {
    fail(`no-js mode panels are not both readable: ${JSON.stringify(noJs)}`);
  }
  if (noJs.faqExpanded.some((value) => value !== "true") || noJs.faqPanelHeights.some((height) => height <= 0)) {
    fail(`no-js FAQ state is not truthful/readable: ${JSON.stringify(noJs)}`);
  }
  if (noJs.mobileReadoutDesktopDisplay !== "none") fail(`no-js mobile readout should not render on desktop: ${JSON.stringify(noJs)}`);
  await page.close();
  if (errors.length) fail(`INSTNCT no-js browser errors: ${errors.join(" | ")}`);
}

async function probeResponsiveViewports(browser, origin) {
  const viewports = [
    { width: 320, height: 568 },
    { width: 320, height: 740 },
    { width: 390, height: 844 },
    { width: 412, height: 915 },
    { width: 768, height: 1024 },
    { width: 1024, height: 768 },
    { width: 1280, height: 900 },
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
      const hero = document.querySelector(".hero");
      const heroRect = hero?.getBoundingClientRect();
      const nextRect = hero?.nextElementSibling?.getBoundingClientRect();
      const nextSignal = hero?.nextElementSibling?.querySelector(".section-label, .section-heading, .center-heading");
      const nextSignalRect = nextSignal?.getBoundingClientRect();
      const nextSignalStyle = nextSignal ? getComputedStyle(nextSignal) : null;
      const heroLead = document.querySelector(".hero .lead");
      const nav = document.querySelector(".site-header .nav");
      const navRect = nav.getBoundingClientRect();
      const visibleTarget = (el) => {
        const style = getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        return (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          Number(style.opacity || 1) !== 0 &&
          rect.width > 0 &&
          rect.height > 0
        );
      };
      return {
        overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
        activePanelClipped: active ? active.scrollHeight > active.clientHeight + 1 : true,
        heroMeshDisplay: getComputedStyle(document.querySelector(".hero-mesh")).display,
        heroHeight: heroRect ? Math.round(heroRect.height) : 0,
        heroNextTop: nextRect ? Math.round(nextRect.top) : null,
        heroNextSignalTop: nextSignalRect ? Math.round(nextSignalRect.top) : null,
        heroNextSignalVisible: nextSignalRect
          ? nextSignalRect.top <= document.documentElement.clientHeight - 12 &&
            Number(nextSignalStyle.opacity) > 0.8 &&
            nextSignalStyle.visibility !== "hidden"
          : false,
        heroLeadClipped: heroLead ? heroLead.scrollHeight > heroLead.clientHeight + 1 : true,
        heroLeadBox: heroLead
          ? {
              scrollHeight: heroLead.scrollHeight,
              clientHeight: heroLead.clientHeight,
              lineClamp: getComputedStyle(heroLead).webkitLineClamp,
              overflow: getComputedStyle(heroLead).overflow,
            }
          : null,
        headerNavClipped:
          nav.scrollWidth > nav.clientWidth + 1 ||
          navRect.left < -1 ||
          navRect.right > document.documentElement.clientWidth + 1,
        shortTargets: [...document.querySelectorAll('a[href], button, [role="button"], [role="switch"]')]
          .filter(visibleTarget)
          .map((el) => {
            const rect = el.getBoundingClientRect();
            return {
              text: (el.textContent || el.getAttribute("aria-label") || el.getAttribute("href") || "")
                .trim()
                .replace(/\s+/g, " ")
                .slice(0, 80),
              width: Math.round(rect.width),
              height: Math.round(rect.height),
            };
          })
          .filter((target) => target.width < 44 || target.height < 44),
      };
    });
    if (instnct.overflow) fail(`INSTNCT ${label} has horizontal overflow`);
    if (instnct.activePanelClipped) fail(`INSTNCT ${label} exact mode panel clips`);
    if (viewport.width <= 420 && instnct.heroLeadClipped) {
      fail(`INSTNCT ${label} hero lead clips on mobile: ${JSON.stringify(instnct)}`);
    }
    if (instnct.heroMeshDisplay === "none") fail(`INSTNCT ${label} hero mesh is hidden`);
    if (!instnct.heroNextSignalVisible) {
      fail(`INSTNCT ${label} hero does not reveal next-section content in the first viewport: ${JSON.stringify(instnct)}`);
    }
    if (viewport.width <= 420 && instnct.heroHeight > Math.max(920, viewport.height * 1.2)) {
      fail(`INSTNCT ${label} mobile hero is too tall: ${JSON.stringify(instnct)}`);
    }
    if (viewport.width <= 420 && instnct.headerNavClipped) {
      fail(`INSTNCT ${label} header nav is clipped: ${JSON.stringify(instnct)}`);
    }
    if (instnct.shortTargets.length) {
      fail(`INSTNCT ${label} visible action targets are too small: ${JSON.stringify(instnct.shortTargets)}`);
    }

    await page.evaluate(() => {
      document.documentElement.style.scrollBehavior = "auto";
      document.body.style.scrollBehavior = "auto";
    });
    await page.locator("#trust").scrollIntoViewIfNeeded();
    if (viewport.width <= 1360) {
      await page
        .waitForFunction(() => {
          const readout = document.querySelector(".mobile-section-readout");
          if (!readout) return false;
          const style = getComputedStyle(readout);
          return (
            style.display !== "none" &&
            style.visibility !== "hidden" &&
            Number(style.opacity) > 0.25 &&
            !readout.classList.contains("is-hidden")
          );
        }, null, { timeout: 1500 })
        .catch(() => {});
    } else {
      await page.waitForTimeout(260);
    }
    const sectionRailMode = await page.evaluate(() => {
      const rail = document.querySelector(".section-indicator");
      const readout = document.querySelector(".mobile-section-readout");
      const railStyle = rail ? getComputedStyle(rail) : null;
      const readoutStyle = readout ? getComputedStyle(readout) : null;
      const railVisible = !!rail && railStyle.display !== "none" && railStyle.visibility !== "hidden";
      const readoutVisible =
        !!readout &&
        readoutStyle.display !== "none" &&
        readoutStyle.visibility !== "hidden" &&
        Number(readoutStyle.opacity) > 0.25 &&
        !readout.classList.contains("is-hidden");
      const railRect = railVisible ? rail.getBoundingClientRect() : null;
      const targets = [...document.querySelectorAll("#trust .trust-card, #trust .center-heading, #trust .benchmark-card")];
      const overlaps = railRect
        ? targets
            .map((target) => {
              const rect = target.getBoundingClientRect();
              const width = Math.max(0, Math.min(railRect.right, rect.right) - Math.max(railRect.left, rect.left));
              const height = Math.max(0, Math.min(railRect.bottom, rect.bottom) - Math.max(railRect.top, rect.top));
              return { target: target.className, area: Math.round(width * height) };
            })
            .filter((entry) => entry.area > 0)
        : [];
      return { railVisible, readoutVisible, overlaps };
    });
    if (viewport.width <= 1360) {
      if (sectionRailMode.railVisible || !sectionRailMode.readoutVisible) {
        fail(`INSTNCT ${label} should use compact section readout: ${JSON.stringify(sectionRailMode)}`);
      }
    } else if (!sectionRailMode.railVisible || sectionRailMode.overlaps.length) {
      fail(`INSTNCT ${label} section rail is missing or overlaps content: ${JSON.stringify(sectionRailMode)}`);
    }

    if (viewport.width <= 1360) {
      await page.locator("#fabric").scrollIntoViewIfNeeded();
      await page.waitForTimeout(260);
      const fixedControls = await page.evaluate(() => {
        const readout = document.querySelector(".mobile-section-readout");
        const top = document.querySelector(".back-to-top");
        const help = document.querySelector(".keyboard-help-trigger");
        const visible = (el) => {
          if (!el) return false;
          const style = getComputedStyle(el);
          return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity) > 0.25;
        };
        if (!visible(readout)) return { overlap: false, overlaps: [] };
        const a = readout.getBoundingClientRect();
        const controls = [
          ["top", top],
          ["help", help],
        ].filter(([, el]) => visible(el));
        const overlaps = controls
          .map(([name, el]) => {
            const b = el.getBoundingClientRect();
            const overlap = !(a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom);
            return {
              name,
              overlap,
              rect: { left: Math.round(b.left), right: Math.round(b.right), top: Math.round(b.top), bottom: Math.round(b.bottom) },
            };
          })
          .filter((entry) => entry.overlap);
        return {
          overlap: overlaps.length > 0,
          overlaps,
          readout: { left: Math.round(a.left), right: Math.round(a.right), top: Math.round(a.top), bottom: Math.round(a.bottom) },
        };
      });
      if (fixedControls.overlap) {
        fail(`INSTNCT ${label} fixed controls overlap after scroll: ${JSON.stringify(fixedControls)}`);
      }

      if (viewport.width <= 760) {
        await page.locator("#t1-reflex-class .boundary-band").scrollIntoViewIfNeeded();
        await page.waitForTimeout(260);
        const t1FixedOverlap = await page.evaluate(() => {
          const readout = document.querySelector(".mobile-section-readout");
          const top = document.querySelector(".back-to-top");
          const visible = (el) => {
            if (!el) return false;
            const style = getComputedStyle(el);
            return (
              style.display !== "none" &&
              style.visibility !== "hidden" &&
              Number(style.opacity || 1) > 0.25 &&
              !el.classList.contains("is-hidden")
            );
          };
          const intersects = (a, b) => !(a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom);
          const controls = [
            ["readout", readout],
            ["top", top],
          ].filter(([, el]) => visible(el));
          const targets = [...document.querySelectorAll("#t1-reflex-class .boundary-band h3, #t1-reflex-class .boundary-band p, #t1-reflex-class .boundary-band .button")]
            .filter(visible);
          const overlaps = [];
          for (const [controlName, control] of controls) {
            const controlRect = control.getBoundingClientRect();
            for (const target of targets) {
              const targetRect = target.getBoundingClientRect();
              if (intersects(controlRect, targetRect)) {
                overlaps.push({
                  control: controlName,
                  target: target.tagName.toLowerCase(),
                  text: target.textContent.trim().replace(/\s+/g, " ").slice(0, 80),
                });
              }
            }
          }
          return { overlaps };
        });
        if (t1FixedOverlap.overlaps.length) {
          fail(`INSTNCT ${label} T1 fixed readout overlaps scope card content: ${JSON.stringify(t1FixedOverlap)}`);
        }
      }
    }

    await page.locator("#dev-trail").scrollIntoViewIfNeeded();
    await page.waitForTimeout(180);
    const actionTargets = await page.evaluate(() => {
      const rectFor = (el) => {
        const rect = el.getBoundingClientRect();
        const style = getComputedStyle(el);
        return {
          text: el.textContent.trim(),
          width: Math.round(rect.width),
          height: Math.round(rect.height),
          display: style.display,
          visibility: style.visibility,
        };
      };
      return {
        terminalButtons: [...document.querySelectorAll(".terminal-actions button")].map(rectFor),
        footerLinks: [...document.querySelectorAll(".footer-inner a:not(:first-child)")].map(rectFor),
      };
    });
    const shortTerminalTargets = actionTargets.terminalButtons.filter(
      (button) => button.display === "none" || button.visibility === "hidden" || button.height < 44 || button.width < 44
    );
    const shortFooterTargets = actionTargets.footerLinks.filter(
      (link) => link.display === "none" || link.visibility === "hidden" || link.height < 44 || link.width < 44
    );
    if (shortTerminalTargets.length || shortFooterTargets.length) {
      fail(`INSTNCT ${label} action targets are too small: ${JSON.stringify({ shortTerminalTargets, shortFooterTargets })}`);
    }

    await page.locator("#hallucination").scrollIntoViewIfNeeded();
    await page.locator(".mode-switch").click();
    await assertActiveModePanelFits(page, `responsive ${label} imagination`);

    await page.goto(`${origin}/`, { waitUntil: "networkidle" });
    const home = await page.evaluate(() => {
      const nav = document.querySelector(".nav");
      const navRect = nav.getBoundingClientRect();
      const hero = document.querySelector(".hero");
      const heroRect = hero?.getBoundingClientRect();
      const nextSignal = hero?.nextElementSibling?.querySelector(".system-label, h2");
      const nextSignalRect = nextSignal?.getBoundingClientRect();
      const nextSignalStyle = nextSignal ? getComputedStyle(nextSignal) : null;
      const clippedLinks = [...nav.querySelectorAll("a")]
        .filter((link) => {
          const style = getComputedStyle(link);
          return style.display !== "none" && style.visibility !== "hidden";
        })
        .map((link) => ({ text: link.textContent.trim(), rect: link.getBoundingClientRect() }))
        .filter(({ rect }) => rect.left < navRect.left - 1 || rect.right > navRect.right + 1)
        .map(({ text, rect }) => ({ text, left: Math.round(rect.left), right: Math.round(rect.right) }));
      const visibleAction = (el) => {
        const style = getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        return (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          Number(style.opacity || 1) > 0.25 &&
          rect.width > 0 &&
          rect.height > 0
        );
      };
      const shortHeaderTargets = [...document.querySelectorAll(".site-header a")]
        .filter(visibleAction)
        .map((el) => {
          const rect = el.getBoundingClientRect();
          return {
            text: el.textContent.trim().replace(/\s+/g, " "),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
          };
        })
        .filter((target) => target.width < 44 || target.height < 44);
      return {
        overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
        navDisplay: getComputedStyle(nav).display,
        navText: nav.textContent,
        clippedLinks,
        shortHeaderTargets,
        brandImage: document.querySelector(".brand img")?.getAttribute("src") || "",
        heroHeight: heroRect ? Math.round(heroRect.height) : 0,
        heroNextSignalTop: nextSignalRect ? Math.round(nextSignalRect.top) : null,
        heroNextSignalVisible: nextSignalRect
          ? nextSignalRect.top <= document.documentElement.clientHeight - 12 &&
            Number(nextSignalStyle.opacity) > 0.8 &&
            nextSignalStyle.visibility !== "hidden"
          : false,
      };
    });
    if (home.overflow) fail(`home ${label} has horizontal overflow`);
    if (!home.brandImage.includes("vraxion-wordmark.png")) {
      fail(`home ${label} header brand does not use the VRAXION wordmark asset: ${JSON.stringify(home)}`);
    }
    if (!home.heroNextSignalVisible) {
      fail(`home ${label} hero does not reveal next-section content in the first viewport: ${JSON.stringify(home)}`);
    }
    if (viewport.width <= 980 && (home.navDisplay === "none" || !home.navText.includes("INSTNCT"))) {
      fail(`home ${label} mobile nav does not expose INSTNCT: ${JSON.stringify(home)}`);
    }
    if (home.clippedLinks.length) {
      fail(`home ${label} mobile nav links clip outside nav: ${JSON.stringify(home)}`);
    }
    if (home.shortHeaderTargets.length) {
      fail(`home ${label} header action targets are too small: ${JSON.stringify(home.shortHeaderTargets)}`);
    }

    await page.close();
    if (errors.length) fail(`responsive ${label} browser errors: ${errors.join(" | ")}`);
  }
}

async function probeInstnctScrollReveals(browser, origin) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = trackPageFailures(page, origin, "INSTNCT reveal");
  await page.goto(`${origin}/instnct/`, { waitUntil: "networkidle" });
  await page.waitForTimeout(300);

  const revealCount = await page.evaluate(() => document.querySelectorAll("[data-reveal]").length);
  for (let index = 0; index < revealCount; index += 1) {
    await page.evaluate((targetIndex) => {
      document.querySelectorAll("[data-reveal]")[targetIndex]?.scrollIntoView({ block: "center" });
    }, index);
    await page
      .waitForFunction(
        (targetIndex) => document.querySelectorAll("[data-reveal]")[targetIndex]?.classList.contains("is-revealed"),
        index,
        { timeout: 1500 }
      )
      .catch(() => {});
  }
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
  await page.waitForTimeout(800);

  const reveal = await page.evaluate(() => {
    const targets = [...document.querySelectorAll("[data-reveal]")];
    return {
      count: targets.length,
      hidden: targets
        .filter((target) => !target.classList.contains("is-revealed") || Number(getComputedStyle(target).opacity) < 0.95)
        .map((target) => ({
          tag: target.tagName.toLowerCase(),
          id: target.id || "",
          className: target.className,
          text: target.textContent.trim().replace(/\s+/g, " ").slice(0, 80),
          opacity: getComputedStyle(target).opacity,
        })),
    };
  });

  if (reveal.count < 20 || reveal.hidden.length > 0) {
    fail(`scroll reveal targets did not all become visible: ${JSON.stringify(reveal)}`);
  }
  await page.close();
  if (errors.length) fail(`INSTNCT reveal browser errors: ${errors.join(" | ")}`);
}

function resourceKind(pathname, resourceType) {
  const ext = path.extname(pathname).toLowerCase();
  if (resourceType === "script" || ext === ".js") return "script";
  if (resourceType === "stylesheet" || ext === ".css") return "style";
  if (resourceType === "font" || ext === ".woff2") return "font";
  if (resourceType === "image" || [".png", ".jpg", ".jpeg", ".svg"].includes(ext)) return "image";
  return resourceType || "resource";
}

async function probeInstnctPerformanceBudget(browser, origin) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = trackPageFailures(page, origin, "INSTNCT performance");
  const sameOriginResources = new Map();
  const externalResources = new Set();

  function rememberResource(urlText, resourceType) {
    let url;
    try {
      url = new URL(urlText);
    } catch {
      return;
    }
    if (!criticalResourceTypes.has(resourceType)) return;
    if (url.origin !== origin) {
      if (resourceType !== "document") externalResources.add(url.href);
      return;
    }
    const previous = sameOriginResources.get(url.pathname);
    sameOriginResources.set(url.pathname, previous || resourceType);
  }

  page.on("response", (response) => {
    rememberResource(response.url(), response.request().resourceType());
  });

  await page.goto(`${origin}/instnct/`, { waitUntil: "networkidle" });
  for (const y of [0, 700, 1500, 2400, 3400, 4600, 6200, 8200, 10000]) {
    await page.evaluate((nextY) => window.scrollTo(0, nextY), y);
    await page.waitForTimeout(90);
  }
  await page.waitForLoadState("networkidle");

  const performanceResources = await page.evaluate(() =>
    performance.getEntriesByType("resource").map((entry) => ({
      name: entry.name,
      initiatorType: entry.initiatorType || "resource",
    }))
  );
  for (const entry of performanceResources) {
    const resourceType = entry.initiatorType === "link" ? "stylesheet" : entry.initiatorType;
    rememberResource(entry.name, resourceType);
  }

  const resources = [];
  for (const [pathname, resourceType] of sameOriginResources.entries()) {
    const bytes = await byteSizeForServedPath(pathname);
    resources.push({
      path: pathname,
      kind: resourceKind(pathname, resourceType),
      bytes,
    });
  }

  const totalBytes = resources.reduce((sum, resource) => sum + resource.bytes, 0);
  const scriptStyleBytes = resources
    .filter((resource) => resource.kind === "script" || resource.kind === "style")
    .reduce((sum, resource) => sum + resource.bytes, 0);
  const mediaFontBytes = resources
    .filter((resource) => ["image", "font"].includes(resource.kind))
    .reduce((sum, resource) => sum + resource.bytes, 0);
  const missingBytePaths = resources.filter((resource) => resource.bytes <= 0).map((resource) => resource.path);
  const largest = [...resources].sort((a, b) => b.bytes - a.bytes).slice(0, 5);
  const budget = {
    resourceCount: resources.length,
    totalBytes,
    scriptStyleBytes,
    mediaFontBytes,
    missingBytePaths,
    externalResources: [...externalResources],
    largest,
  };

  if (resources.length > 18 || totalBytes > 5_600_000 || scriptStyleBytes > 360_000 || mediaFontBytes > 5_200_000) {
    fail(`INSTNCT performance budget exceeded: ${JSON.stringify(budget)}`);
  }
  if (missingBytePaths.length) fail(`INSTNCT performance missing byte sizes: ${JSON.stringify(budget)}`);
  if (externalResources.size) fail(`INSTNCT loaded external subresources: ${JSON.stringify(budget)}`);

  await page.close();
  if (errors.length) fail(`INSTNCT performance browser errors: ${errors.join(" | ")}`);
}

async function probeAnchorCell(browser, origin) {
  const viewports = [
    { width: 390, height: 844 },
    { width: 768, height: 1024 },
    { width: 1440, height: 900 },
  ];

  for (const viewport of viewports) {
    const label = `${viewport.width}x${viewport.height}`;
    const page = await browser.newPage({ viewport, reducedMotion: "no-preference" });
    const errors = trackPageFailures(page, origin, `AnchorCell ${label}`);
    await page.goto(`${origin}/anchorcell/`, { waitUntil: "networkidle" });
    await page.waitForTimeout(350);

    const first = await page.evaluate(() => {
      const visibleTarget = (el) => {
        const style = getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        return (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          Number(style.opacity || 1) !== 0 &&
          rect.width > 0 &&
          rect.height > 0
        );
      };
      const hero = document.querySelector(".hero");
      const heroRect = hero?.getBoundingClientRect();
      const nextSignal = [...(hero?.nextElementSibling?.querySelectorAll(".section-label, h2") || [])].find((candidate) => {
        const rect = candidate.getBoundingClientRect();
        return rect.top <= document.documentElement.clientHeight - 12 && rect.bottom > 0;
      });
      const nextSignalRect = nextSignal?.getBoundingClientRect();
      const nextSignalStyle = nextSignal ? getComputedStyle(nextSignal) : null;
      const rail = document.querySelector(".section-indicator");
      const readout = document.querySelector(".mobile-section-readout");
      const railStyle = rail ? getComputedStyle(rail) : null;
      const readoutStyle = readout ? getComputedStyle(readout) : null;
      return {
        overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
        h1Count: [...document.querySelectorAll("h1")].filter(visibleTarget).length,
        copyOk:
          document.body.textContent.includes("Training data with its trust boundaries intact.") &&
          document.body.textContent.includes("not a finished model claim"),
        schemaHref: [...document.querySelectorAll("a")].some((a) => a.href.endsWith("/anchorcell/anchorcell.v2.schema.json")),
        exampleHref: [...document.querySelectorAll("a")].some((a) => a.href.endsWith("/anchorcell/anchorcell.v2.example.json")),
        booted: hero?.classList.contains("is-booted"),
        heroHeight: heroRect ? Math.round(heroRect.height) : 0,
        heroNextSignalVisible: nextSignalRect
          ? nextSignalRect.top <= document.documentElement.clientHeight - 12 &&
            Number(nextSignalStyle.opacity) > 0.75 &&
            nextSignalStyle.visibility !== "hidden"
          : false,
        railVisible: !!rail && railStyle.display !== "none" && railStyle.visibility !== "hidden",
        readoutVisible:
          !!readout &&
          readoutStyle.display !== "none" &&
          readoutStyle.visibility !== "hidden" &&
          Number(readoutStyle.opacity || 1) > 0.25,
        shortTargets: [...document.querySelectorAll('a[href], button, [role="button"]')]
          .filter(visibleTarget)
          .map((el) => {
            const rect = el.getBoundingClientRect();
            return {
              text: (el.textContent || el.getAttribute("aria-label") || el.getAttribute("href") || "")
                .trim()
                .replace(/\s+/g, " ")
                .slice(0, 80),
              width: Math.round(rect.width),
              height: Math.round(rect.height),
            };
          })
          .filter((target) => target.width < 44 || target.height < 44),
      };
    });

    if (first.overflow) fail(`AnchorCell ${label} has horizontal overflow`);
    if (first.h1Count !== 1) fail(`AnchorCell ${label} must expose one visible h1: ${JSON.stringify(first)}`);
    if (!first.copyOk) fail(`AnchorCell ${label} required copy is missing`);
    if (!first.schemaHref) fail(`AnchorCell ${label} schema CTA is missing: ${JSON.stringify(first)}`);
    if (!first.exampleHref) fail(`AnchorCell ${label} example CTA is missing: ${JSON.stringify(first)}`);
    if (!first.booted) fail(`AnchorCell ${label} hero did not boot`);
    if (!first.heroNextSignalVisible) {
      fail(`AnchorCell ${label} hero does not reveal next-section content in the first viewport: ${JSON.stringify(first)}`);
    }
    if (viewport.width <= 420 && first.heroHeight > Math.max(920, viewport.height * 1.2)) {
      fail(`AnchorCell ${label} mobile hero is too tall: ${JSON.stringify(first)}`);
    }
    if (viewport.width <= 1360) {
      if (first.railVisible || !first.readoutVisible) fail(`AnchorCell ${label} should use compact section readout: ${JSON.stringify(first)}`);
    } else if (!first.railVisible || first.readoutVisible) {
      fail(`AnchorCell ${label} desktop section rail/readout mode is wrong: ${JSON.stringify(first)}`);
    }
    if (first.shortTargets.length) {
      fail(`AnchorCell ${label} visible action targets are too small: ${JSON.stringify(first.shortTargets)}`);
    }

    await page.locator("#branches").scrollIntoViewIfNeeded();
    await page.waitForTimeout(260);
    const tracking = await page.evaluate(() => ({
      active: document.querySelector(".section-indicator a.is-active")?.textContent.trim() || "",
      mobileLabel: document.querySelector("[data-mobile-label]")?.textContent.trim() || "",
      revealed: [...document.querySelectorAll(".section")].filter((section) => section.classList.contains("is-revealed")).length,
    }));
    if (!tracking.active.includes("Branches") && !tracking.mobileLabel.includes("Branches")) {
      fail(`AnchorCell ${label} section tracking did not follow scroll: ${JSON.stringify(tracking)}`);
    }
    if (tracking.revealed < 2) {
      fail(`AnchorCell ${label} scroll reveal did not mark sections: ${JSON.stringify(tracking)}`);
    }

    if (viewport.width <= 420) {
      await page.goto(`${origin}/anchorcell/#branches`, { waitUntil: "networkidle" });
      await page.waitForTimeout(360);
      const branchFragment = await page.evaluate(() => {
        const section = document.querySelector("#branches");
        const title = document.querySelector("#branches-title");
        const readout = document.querySelector(".mobile-section-readout");
        const sectionStyle = section ? getComputedStyle(section) : null;
        const sectionRect = section?.getBoundingClientRect();
        const titleRect = title?.getBoundingClientRect();
        const readoutStyle = readout ? getComputedStyle(readout) : null;
        const readoutRect =
          readout &&
          readoutStyle.display !== "none" &&
          readoutStyle.visibility !== "hidden" &&
          Number(readoutStyle.opacity || 1) > 0.25
            ? readout.getBoundingClientRect()
            : null;
        const overlapsCritical = readoutRect
          ? [...document.querySelectorAll("#branches h2, #branches h3, #branches p, #branches .section-label")]
              .map((target) => {
                const rect = target.getBoundingClientRect();
                const width = Math.max(0, Math.min(readoutRect.right, rect.right) - Math.max(readoutRect.left, rect.left));
                const height = Math.max(0, Math.min(readoutRect.bottom, rect.bottom) - Math.max(readoutRect.top, rect.top));
                return {
                  text: target.textContent.trim().replace(/\s+/g, " ").slice(0, 48),
                  area: Math.round(width * height),
                };
              })
              .filter((entry) => entry.area > 0)
          : [];
        return {
          overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
          outlineStyle: sectionStyle?.outlineStyle || "",
          outlineWidth: sectionStyle?.outlineWidth || "",
          sectionLeft: sectionRect ? Math.round(sectionRect.left) : null,
          titleLeft: titleRect ? Math.round(titleRect.left) : null,
          titleTop: titleRect ? Math.round(titleRect.top) : null,
          overlapsCritical,
        };
      });
      if (
        branchFragment.overflow ||
        branchFragment.outlineStyle !== "none" ||
        branchFragment.sectionLeft < 0 ||
        branchFragment.titleLeft < 0 ||
        branchFragment.titleTop < 0 ||
        branchFragment.overlapsCritical.length
      ) {
        fail(`AnchorCell ${label} branch fragment target is visually clipped: ${JSON.stringify(branchFragment)}`);
      }
    }

    await page.close();
    if (errors.length) fail(`AnchorCell ${label} browser errors: ${errors.join(" | ")}`);
  }
}

async function probeAccessibilitySemantics(browser, origin) {
  const targets = [
    { path: "/", label: "home" },
    { path: "/instnct/", label: "INSTNCT" },
    { path: "/anchorcell/", label: "AnchorCell" },
  ];

  for (const target of targets) {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const errors = trackPageFailures(page, origin, `${target.label} accessibility`);
    await page.goto(`${origin}${target.path}`, { waitUntil: "networkidle" });
    await page.waitForTimeout(300);

    const result = await page.evaluate(() => {
      const focusableSelector = [
        "a[href]",
        "button:not([disabled])",
        "input:not([disabled])",
        "select:not([disabled])",
        "textarea:not([disabled])",
        "summary",
        "[tabindex]:not([tabindex='-1'])",
      ].join(",");
      const isHidden = (el) => {
        if (!el) return true;
        if (el.hidden || el.closest("[hidden]") || el.closest("[inert]")) return true;
        const style = getComputedStyle(el);
        return style.display === "none" || style.visibility === "hidden";
      };
      const visible = (el) => !isHidden(el) && (el.offsetWidth > 0 || el.offsetHeight > 0);
      const nameOf = (el) =>
        (el.getAttribute("aria-label") ||
          (el.getAttribute("aria-labelledby") || "")
            .split(/\s+/)
            .map((id) => document.getElementById(id)?.textContent || "")
            .join(" ") ||
          el.textContent ||
          el.getAttribute("title") ||
          "")
          .trim()
          .replace(/\s+/g, " ");

      const ids = [...document.querySelectorAll("[id]")].map((el) => el.id);
      const duplicates = ids.filter((id, index) => ids.indexOf(id) !== index);
      const headings = [...document.querySelectorAll("h1,h2,h3,h4,h5,h6")]
        .filter(visible)
        .map((heading) => ({
          level: Number(heading.tagName.slice(1)),
          text: heading.textContent.trim().replace(/\s+/g, " "),
        }));
      const headingSkips = [];
      for (let i = 1; i < headings.length; i += 1) {
        if (headings[i].level > headings[i - 1].level + 1) {
          headingSkips.push(`${headings[i - 1].text} -> ${headings[i].text}`);
        }
      }

      const interactiveMissingNames = [...document.querySelectorAll("a[href],button")]
        .filter(visible)
        .filter((el) => !nameOf(el))
        .map((el) => el.outerHTML.slice(0, 140));
      const imageIssues = [...document.images]
        .filter(visible)
        .filter((img) => !img.hasAttribute("alt") && img.getAttribute("aria-hidden") !== "true")
        .map((img) => img.currentSrc || img.src);
      const ariaHiddenFocusable = [...document.querySelectorAll('[aria-hidden="true"]')]
        .flatMap((el) => [...el.querySelectorAll(focusableSelector)].filter(visible))
        .map((el) => el.outerHTML.slice(0, 140));
      const brokenAriaRefs = [];
      for (const el of document.querySelectorAll("[aria-controls],[aria-labelledby],[aria-describedby]")) {
        for (const attr of ["aria-controls", "aria-labelledby", "aria-describedby"]) {
          const value = el.getAttribute(attr);
          if (!value) continue;
          for (const id of value.split(/\s+/)) {
            if (id && !document.getElementById(id)) brokenAriaRefs.push(`${attr}=${id}`);
          }
        }
      }

      return {
        landmarks: {
          header: document.querySelectorAll("header").length,
          main: document.querySelectorAll("main").length,
          footer: document.querySelectorAll("footer").length,
          navPrimary: document.querySelectorAll('nav[aria-label="Primary"]').length,
        },
        h1Count: headings.filter((heading) => heading.level === 1).length,
        headingSkips,
        duplicates: [...new Set(duplicates)],
        interactiveMissingNames,
        imageIssues,
        ariaHiddenFocusable,
        brokenAriaRefs: [...new Set(brokenAriaRefs)],
      };
    });

    if (result.landmarks.header !== 1 || result.landmarks.main !== 1 || result.landmarks.footer !== 1) {
      fail(`${target.label} landmark count is invalid: ${JSON.stringify(result.landmarks)}`);
    }
    if (result.landmarks.navPrimary !== 1) {
      fail(`${target.label} primary nav landmark is missing: ${JSON.stringify(result.landmarks)}`);
    }
    if (result.h1Count !== 1) fail(`${target.label} must expose exactly one visible h1, found ${result.h1Count}`);
    if (result.headingSkips.length) fail(`${target.label} heading levels skip: ${result.headingSkips.join(" | ")}`);
    if (result.duplicates.length) fail(`${target.label} duplicate ids: ${result.duplicates.join(", ")}`);
    if (result.interactiveMissingNames.length) {
      fail(`${target.label} interactive elements without names: ${result.interactiveMissingNames.join(" | ")}`);
    }
    if (result.imageIssues.length) fail(`${target.label} images missing alt/aria-hidden: ${result.imageIssues.join(" | ")}`);
    if (result.ariaHiddenFocusable.length) {
      fail(`${target.label} aria-hidden contains focusable elements: ${result.ariaHiddenFocusable.join(" | ")}`);
    }
    if (result.brokenAriaRefs.length) fail(`${target.label} broken aria refs: ${result.brokenAriaRefs.join(", ")}`);

    await page.keyboard.press("Home");
    const skipBeforeFocus = await page.evaluate(() => {
      const link = document.querySelector(".skip-link");
      const rect = link?.getBoundingClientRect();
      return {
        exists: !!link,
        href: link?.getAttribute("href") || "",
        bottom: rect ? Math.round(rect.bottom) : 0,
      };
    });
    if (!skipBeforeFocus.exists || skipBeforeFocus.href !== "#main" || skipBeforeFocus.bottom > 0) {
      fail(`${target.label} skip link should exist offscreen before focus: ${JSON.stringify(skipBeforeFocus)}`);
    }
    await page.keyboard.press("Tab");
    await page.waitForTimeout(500);
    const skipFocus = await page.evaluate(() => {
      const el = document.activeElement;
      const rect = el?.getBoundingClientRect();
      return {
        isSkip: !!el?.classList.contains("skip-link"),
        top: rect ? Math.round(rect.top) : 0,
        bottom: rect ? Math.round(rect.bottom) : 0,
        transform: el ? getComputedStyle(el).transform : "",
        matchesFocus: !!el?.matches(":focus"),
        matchesFocusVisible: !!el?.matches(":focus-visible"),
      };
    });
    if (!skipFocus.isSkip || skipFocus.top < 0 || skipFocus.bottom <= 0) {
      fail(`${target.label} skip link is not the first visible keyboard target: ${JSON.stringify(skipFocus)}`);
    }
    await page.keyboard.press("Enter");
    await page.waitForTimeout(120);
    const skipState = await page.evaluate(() => ({
      hash: window.location.hash,
      activeId: document.activeElement?.id || "",
      mainTabIndex: document.querySelector("main")?.getAttribute("tabindex"),
    }));
    if (skipState.hash !== "#main" || skipState.activeId !== "main" || skipState.mainTabIndex !== "-1") {
      fail(`${target.label} skip link does not move focus to main: ${JSON.stringify(skipState)}`);
    }

    await page.goto(`${origin}${target.path}`, { waitUntil: "networkidle" });
    await page.waitForTimeout(120);
    await page.keyboard.press("Home");
    const focusProblems = [];
    const focusableCount = await page.evaluate(() => {
      const focusableSelector = [
        "a[href]",
        "area[href]",
        "button:not([disabled])",
        "input:not([disabled])",
        "select:not([disabled])",
        "textarea:not([disabled])",
        "summary",
        "[tabindex]:not([tabindex='-1'])",
      ].join(",");
      return [...document.querySelectorAll(focusableSelector)].filter((el) => {
        const style = getComputedStyle(el);
        return (
          el !== document.body &&
          !el.hidden &&
          !el.closest("[hidden]") &&
          !el.closest("[inert]") &&
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          el.getBoundingClientRect().width > 0 &&
          el.getBoundingClientRect().height > 0
        );
      }).length;
    });
    for (let i = 0; i < Math.min(18, focusableCount); i += 1) {
      await page.keyboard.press("Tab");
      const state = await page.evaluate(() => {
        const el = document.activeElement;
        const rect = el?.getBoundingClientRect();
        const style = el ? getComputedStyle(el) : null;
        return {
          tag: el?.tagName || "",
          text: el?.textContent?.trim().replace(/\s+/g, " ").slice(0, 80) || "",
          hidden:
            !el ||
            el === document.body ||
            el.hidden ||
            !!el.closest("[hidden]") ||
            !!el.closest("[inert]") ||
            style?.display === "none" ||
            style?.visibility === "hidden",
          visibleRect: !!rect && rect.width > 0 && rect.height > 0,
        };
      });
      if (state.hidden || !state.visibleRect) focusProblems.push(state);
    }
    if (focusProblems.length) {
      fail(`${target.label} keyboard tab reached hidden/nonvisible controls: ${JSON.stringify(focusProblems)}`);
    }

    await page.close();
    if (errors.length) fail(`${target.label} accessibility browser errors: ${errors.join(" | ")}`);
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
  await probeInstnctNoJs(browser, server.origin);
  await probeResponsiveViewports(browser, server.origin);
  await probeInstnctScrollReveals(browser, server.origin);
  await probeInstnctPerformanceBudget(browser, server.origin);
  await probeAnchorCell(browser, server.origin);
  await probeAccessibilitySemantics(browser, server.origin);
} finally {
  if (browser) await browser.close();
  if (server) await server.close();
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log("instnct_browser_smoke=pass");
