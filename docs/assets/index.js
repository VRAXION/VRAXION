(function () {
  "use strict";

  const data = window.BYTE_EMBEDDER_DATA;
  if (!data) {
    throw new Error("BYTE_EMBEDDER_DATA is missing.");
  }

  const {
    C19_C,
    C19_RHO,
    W1,
    W2,
    BIAS1,
    BIAS2,
    LUT_SCALE,
    LUT,
    SCALE_W1,
    SCALE_W2,
  } = data;

  const canonicalSlides = ["hero", "architecture", "breakthrough", "results", "demo", "inspect"];
  const slideAliases = {
    pareto: "results",
    neurons: "inspect",
    heatmaps: "inspect",
    similarity: "inspect",
  };

  const slides = canonicalSlides.map((id) => document.getElementById(id));
  const progressDots = document.getElementById("progressDots");
  const navLinks = [...document.querySelectorAll("[data-slide-target]")];
  const mobilePrev = document.getElementById("mobilePrev");
  const mobileNext = document.getElementById("mobileNext");
  const mobileSlideTitle = document.getElementById("mobileSlideTitle");
  const mobileSlideCount = document.getElementById("mobileSlideCount");
  const mobileSheet = document.getElementById("mobileSheet");
  const mobileMenuToggle = document.getElementById("mobileMenuToggle");
  const mobileSlideList = document.getElementById("mobileSlideList");
  const inspectTabs = [...document.querySelectorAll(".inspect-tab")];
  const inspectPanels = [...document.querySelectorAll(".inspect-panel")];

  let activeIndex = 0;
  let navLockedUntil = 0;
  let currentDemoMode = "float";
  let touchState = null;
  const verificationCache = { float: null, lut: null };

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function c19(x, c, rho) {
    c = Math.max(c, 0.1);
    rho = Math.max(rho, 0.0);
    const L = 6.0 * c;
    if (x >= L) return x - L;
    if (x <= -L) return x + L;
    const s = x / c;
    const n = Math.floor(s);
    const t = s - n;
    const h = t * (1.0 - t);
    const sgn = n % 2 === 0 ? 1.0 : -1.0;
    return c * (sgn * h + rho * h * h);
  }

  function byteToBits(value) {
    const bits = [];
    for (let i = 0; i < 8; i += 1) {
      bits.push(((value >> i) & 1) ? 1 : -1);
    }
    return bits;
  }

  function neuralForward(bits) {
    const hidden = new Array(24).fill(0);
    for (let j = 0; j < 24; j += 1) {
      let sum = BIAS1[j];
      for (let i = 0; i < 8; i += 1) {
        sum += bits[i] * W1[i][j] * SCALE_W1;
      }
      hidden[j] = sum;
    }

    const activated = hidden.map((h, j) => c19(h, C19_C[j], C19_RHO[j]));
    const latent = new Array(16).fill(0);
    for (let j = 0; j < 16; j += 1) {
      let sum = BIAS2[j];
      for (let i = 0; i < 24; i += 1) {
        sum += activated[i] * W2[i][j] * SCALE_W2;
      }
      latent[j] = sum;
    }
    return latent;
  }

  function neuralDecode(latent) {
    const decoderHidden = new Array(24).fill(0);
    for (let j = 0; j < 24; j += 1) {
      let sum = 0;
      for (let i = 0; i < 16; i += 1) {
        sum += latent[i] * W2[j][i] * SCALE_W2;
      }
      decoderHidden[j] = sum;
    }

    const recon = new Array(8).fill(0);
    for (let j = 0; j < 8; j += 1) {
      let sum = 0;
      for (let i = 0; i < 24; i += 1) {
        sum += decoderHidden[i] * W1[j][i] * SCALE_W1;
      }
      recon[j] = sum;
    }
    return recon.map((value) => value > 0 ? 1 : -1);
  }

  function getFloatLatent(byteValue) {
    return neuralForward(byteToBits(byteValue));
  }

  function getLutLatent(byteValue) {
    return LUT[byteValue].map((value) => value * LUT_SCALE);
  }

  function maxAbsDelta(a, b) {
    let best = 0;
    for (let i = 0; i < a.length; i += 1) {
      best = Math.max(best, Math.abs(a[i] - b[i]));
    }
    return best;
  }

  function verifyMode(mode) {
    if (verificationCache[mode]) {
      return verificationCache[mode];
    }
    let ok = 0;
    for (let byteValue = 0; byteValue < 256; byteValue += 1) {
      const bits = byteToBits(byteValue);
      const latent = mode === "float" ? getFloatLatent(byteValue) : getLutLatent(byteValue);
      const decoded = neuralDecode(latent);
      const match = decoded.every((value, index) => value === bits[index]);
      if (match) ok += 1;
    }
    verificationCache[mode] = `${ok}/256 verified`;
    return verificationCache[mode];
  }

  function makeBitPill(value) {
    const span = document.createElement("span");
    span.className = `bit-pill ${value === 1 ? "bit-pill--pos" : "bit-pill--neg"}`;
    span.textContent = value === 1 ? "+1" : "-1";
    return span;
  }

  function makeLatentBar(value, maxValue, className) {
    const bar = document.createElement("div");
    bar.className = className;
    const ratio = maxValue ? Math.abs(value) / maxValue : 0;
    const alpha = 0.28 + ratio * 0.7;
    const height = Math.max(6, ratio * 74);
    bar.style.height = `${height}px`;
    bar.style.background = value >= 0
      ? `rgba(74,222,128,${alpha})`
      : `rgba(248,113,113,${alpha})`;
    return bar;
  }

  function renderHeroPreview() {
    const byteValue = 65;
    const bits = byteToBits(byteValue);
    const floatLatent = getFloatLatent(byteValue);
    const lutLatent = getLutLatent(byteValue);
    const bitsEl = document.getElementById("heroBits");
    const floatEl = document.getElementById("heroLatentFloat");
    const lutEl = document.getElementById("heroLatentLut");
    const deltaEl = document.getElementById("heroPreviewDelta");

    bitsEl.innerHTML = "";
    bits.forEach((value) => bitsEl.appendChild(makeBitPill(value)));

    const maxValue = Math.max(
      ...floatLatent.map((value) => Math.abs(value)),
      ...lutLatent.map((value) => Math.abs(value)),
      1,
    );

    floatEl.innerHTML = "";
    lutEl.innerHTML = "";
    floatLatent.forEach((value) => floatEl.appendChild(makeLatentBar(value, maxValue, "latent-mini__bar")));
    lutLatent.forEach((value) => lutEl.appendChild(makeLatentBar(value, maxValue, "latent-mini__bar")));
    deltaEl.textContent = `Δ max ${maxAbsDelta(floatLatent, lutLatent).toFixed(4)}`;
  }

  function renderProgressDots() {
    progressDots.innerHTML = "";
    mobileSlideList.innerHTML = "";

    canonicalSlides.forEach((id, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "progress-dot";
      button.setAttribute("aria-label", `Go to ${slides[index].dataset.title}`);
      button.dataset.slideIndex = String(index);
      progressDots.appendChild(button);

      const item = document.createElement("button");
      item.type = "button";
      item.className = "mobile-sheet__item";
      item.dataset.slideIndex = String(index);
      item.innerHTML = `<span>${slides[index].dataset.title}</span><span class="panel__mono">${String(index + 1).padStart(2, "0")}</span>`;
      mobileSlideList.appendChild(item);
    });
  }

  function resolveHash(hash) {
    const cleaned = String(hash || "").replace(/^#/, "");
    if (!cleaned) return "hero";
    if (canonicalSlides.includes(cleaned)) return cleaned;
    if (slideAliases[cleaned]) return slideAliases[cleaned];
    return "hero";
  }

  function maybeUseScrollRegion(target, deltaY) {
    if (!(target instanceof Element)) return false;
    const region = target.closest(".scroll-region");
    if (!region || region.scrollHeight <= region.clientHeight + 4) return false;
    if (deltaY > 0 && region.scrollTop + region.clientHeight < region.scrollHeight - 1) return true;
    if (deltaY < 0 && region.scrollTop > 0) return true;
    return false;
  }

  function isInteractiveTarget(target) {
    if (!(target instanceof Element)) return false;
    const tag = target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
  }

  function setSlideTransforms() {
    slides.forEach((slide, index) => {
      const offset = index - activeIndex;
      const translateY = offset * 100;
      const scale = index === activeIndex ? 1 : 0.965;
      slide.style.transform = `translate3d(0, ${translateY}%, 0) scale(${scale})`;
      slide.classList.toggle("is-active", index === activeIndex);
      slide.setAttribute("aria-hidden", index === activeIndex ? "false" : "true");
    });

    [...progressDots.children].forEach((dot, index) => {
      dot.classList.toggle("is-active", index === activeIndex);
    });

    [...mobileSlideList.children].forEach((item, index) => {
      item.classList.toggle("is-active", index === activeIndex);
    });

    navLinks.forEach((link) => {
      link.classList.toggle("is-active", link.dataset.slideTarget === canonicalSlides[activeIndex]);
    });

    mobileSlideTitle.textContent = slides[activeIndex].dataset.title;
    mobileSlideCount.textContent = `${activeIndex + 1} / ${slides.length}`;
  }

  function setHashForIndex(index, replace) {
    const canonical = canonicalSlides[index];
    const nextHash = `#${canonical}`;
    if (replace) {
      history.replaceState({ slide: canonical }, "", nextHash);
    } else if (location.hash !== nextHash) {
      history.pushState({ slide: canonical }, "", nextHash);
    }
  }

  function goToIndex(index, options = {}) {
    const { updateHash = true, replaceHash = false } = options;
    const nextIndex = clamp(index, 0, slides.length - 1);
    if (nextIndex === activeIndex && !options.force) return;
    activeIndex = nextIndex;
    navLockedUntil = performance.now() + 740;
    setSlideTransforms();
    if (updateHash) {
      setHashForIndex(nextIndex, replaceHash);
    }
  }

  function goToHash(hash, options = {}) {
    const canonicalId = resolveHash(hash);
    const nextIndex = canonicalSlides.indexOf(canonicalId);
    const replaceHash = options.replaceHash || canonicalId !== String(hash || "").replace(/^#/, "");
    goToIndex(nextIndex, { updateHash: true, replaceHash, force: options.force });
  }

  function maybeNavigate(delta) {
    if (performance.now() < navLockedUntil) return;
    goToIndex(activeIndex + delta, { updateHash: true });
  }

  function setInspectTab(tabId) {
    inspectTabs.forEach((button) => {
      const active = button.dataset.inspectTab === tabId;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });

    inspectPanels.forEach((panel) => {
      const active = panel.dataset.inspectPanel === tabId;
      panel.hidden = !active;
      panel.classList.toggle("is-active", active);
    });
  }

  function drawPareto() {
    const canvas = document.getElementById("paretoCanvas");
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const pad = { t: 44, r: 34, b: 64, l: 76 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;
    const dataPoints = [
      { label: "float32", acc: 41.78, bytes: 2304, color: "#94a3b8", current: true },
      { label: "int8 LUT", acc: 41.74, bytes: 576, color: "#38bdf8", current: true },
      { label: "int4 staged", acc: 40.72, bytes: 288, color: "#4ade80", current: true },
      { label: "int3", acc: 38.06, bytes: 216, color: "#a78bfa", current: false },
      { label: "ternary", acc: 35.42, bytes: 144, color: "#fbbf24", current: false },
      { label: "binary", acc: 35.42, bytes: 72, color: "#f87171", current: false },
    ];

    const xMax = 2500;
    const yMin = 33;
    const yMax = 43;
    const sx = (bytes) => pad.l + (bytes / xMax) * plotW;
    const sy = (acc) => pad.t + (1 - (acc - yMin) / (yMax - yMin)) * plotH;

    ctx.strokeStyle = "rgba(148,163,184,0.14)";
    ctx.lineWidth = 1;
    ctx.font = "12px Inter";
    ctx.fillStyle = "#6b7893";
    ctx.textAlign = "right";
    for (let acc = 34; acc <= 42; acc += 2) {
      const y = sy(acc);
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(W - pad.r, y);
      ctx.stroke();
      ctx.fillText(`${acc}%`, pad.l - 10, y + 4);
    }

    ctx.textAlign = "center";
    [0, 500, 1000, 1500, 2000, 2500].forEach((bytes) => {
      const x = sx(bytes);
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, H - pad.b);
      ctx.stroke();
      ctx.fillText(`${bytes}B`, x, H - pad.b + 24);
    });

    const frontier = [...dataPoints].sort((a, b) => a.bytes - b.bytes);
    ctx.strokeStyle = "rgba(74,222,128,0.3)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    frontier.forEach((point, index) => {
      const x = sx(point.bytes);
      const y = sy(point.acc);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    dataPoints.forEach((point) => {
      const x = sx(point.bytes);
      const y = sy(point.acc);
      if (point.current) {
        ctx.beginPath();
        ctx.fillStyle = "rgba(255,255,255,0.06)";
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.beginPath();
      ctx.fillStyle = point.color;
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = point.current ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.32)";
      ctx.lineWidth = point.current ? 2.5 : 2;
      ctx.stroke();

      ctx.fillStyle = point.color;
      ctx.font = "700 11px JetBrains Mono";
      ctx.textAlign = "center";
      ctx.fillText(point.label, x, y - 18);
      ctx.fillStyle = "#d7e4f8";
      ctx.font = "11px Inter";
      ctx.fillText(`${point.acc.toFixed(2)}%`, x, y + 24);
    });

    ctx.fillStyle = "#9ca8c2";
    ctx.font = "12px Inter";
    ctx.textAlign = "center";
    ctx.fillText("Storage (bytes)", W / 2, H - 16);
    ctx.save();
    ctx.translate(20, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Downstream accuracy", 0, 0);
    ctx.restore();
  }

  function drawHeatmap(canvasId, matrix) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext("2d");
    const rows = matrix.length;
    const cols = matrix[0].length;
    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;
    const maxAbs = Math.max(...matrix.flat().map((value) => Math.abs(value)), 1);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(6,8,20,0.8)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const value = matrix[r][c];
        const intensity = Math.abs(value) / maxAbs;
        ctx.fillStyle = value >= 0
          ? `rgba(74,222,128,${0.08 + intensity * 0.9})`
          : `rgba(248,113,113,${0.08 + intensity * 0.9})`;
        ctx.fillRect(c * cellW, r * cellH, cellW - 1, cellH - 1);
        if (cellW > 14 && cellH > 14) {
          ctx.fillStyle = intensity > 0.48 ? "#ffffff" : "#97a6c4";
          ctx.font = `${Math.min(cellW * 0.45, 11)}px JetBrains Mono`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(value), c * cellW + cellW / 2, r * cellH + cellH / 2);
        }
      }
    }
  }

  function drawHistogram() {
    const canvas = document.getElementById("histCanvas");
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const weights = [...W1.flat(), ...W2.flat()];
    const bins = new Map();
    weights.forEach((value) => bins.set(value, (bins.get(value) || 0) + 1));
    const keys = [...bins.keys()].sort((a, b) => a - b);
    const maxCount = Math.max(...keys.map((key) => bins.get(key)));

    const pad = { l: 44, r: 18, t: 18, b: 34 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;
    const slotW = plotW / keys.length;

    ctx.fillStyle = "rgba(6,8,20,0.8)";
    ctx.fillRect(0, 0, W, H);

    keys.forEach((key, index) => {
      const count = bins.get(key);
      const barH = (count / maxCount) * plotH;
      const x = pad.l + index * slotW + slotW * 0.15;
      const y = pad.t + plotH - barH;
      ctx.fillStyle = key === 0 ? "#94a3b8" : key > 0 ? "#4ade80" : "#f87171";
      ctx.fillRect(x, y, slotW * 0.7, barH);
      ctx.fillStyle = "#6b7893";
      ctx.font = "9px JetBrains Mono";
      ctx.textAlign = "center";
      ctx.fillText(String(key), x + slotW * 0.35, H - 12);
    });

    ctx.fillStyle = "#9ca8c2";
    ctx.font = "12px Inter";
    ctx.textAlign = "center";
    ctx.fillText("Quantized weight value", W / 2, H - 2);
  }

  function renderNeurons() {
    const grid = document.getElementById("neuronGrid");
    grid.innerHTML = "";

    for (let index = 0; index < 24; index += 1) {
      const card = document.createElement("div");
      card.className = "neuron-card";
      const canvas = document.createElement("canvas");
      canvas.width = 320;
      canvas.height = 160;
      card.appendChild(canvas);
      const label = document.createElement("div");
      label.className = "neuron-label";
      label.textContent = `#${index}  c=${C19_C[index].toFixed(2)}  rho=${C19_RHO[index].toFixed(2)}`;
      card.appendChild(label);
      grid.appendChild(card);

      const ctx = canvas.getContext("2d");
      const W = canvas.width;
      const H = canvas.height;
      ctx.fillStyle = "rgba(4,8,18,0.9)";
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = "rgba(148,163,184,0.14)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, H / 2);
      ctx.lineTo(W, H / 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(W / 2, 0);
      ctx.lineTo(W / 2, H);
      ctx.stroke();

      ctx.strokeStyle = "#a78bfa";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let px = 0; px < W; px += 1) {
        const x = (px / W - 0.5) * 20;
        const y = c19(x, C19_C[index], C19_RHO[index]);
        const py = H / 2 - y * (H / 3.4);
        if (px === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
  }

  function drawSimilarity() {
    const canvas = document.getElementById("simCanvas");
    const ctx = canvas.getContext("2d");
    const letters = [];
    const embeddings = [];
    for (let code = 65; code <= 90; code += 1) {
      letters.push(String.fromCharCode(code));
      embeddings.push(LUT[code]);
    }
    const n = letters.length;
    const cellSize = Math.floor(600 / (n + 1));
    const offset = cellSize;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(4,8,18,0.92)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    function dot(a, b) {
      let sum = 0;
      for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i];
      return sum;
    }

    function norm(vector) {
      return Math.sqrt(dot(vector, vector));
    }

    function cosine(a, b) {
      const na = norm(a);
      const nb = norm(b);
      return na && nb ? dot(a, b) / (na * nb) : 0;
    }

    ctx.fillStyle = "#9ca8c2";
    ctx.font = "700 13px JetBrains Mono";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = 0; i < n; i += 1) {
      ctx.fillText(letters[i], offset + i * cellSize + cellSize / 2, cellSize / 2);
      ctx.fillText(letters[i], cellSize / 2, offset + i * cellSize + cellSize / 2);
    }

    for (let row = 0; row < n; row += 1) {
      for (let col = 0; col < n; col += 1) {
        const sim = cosine(embeddings[row], embeddings[col]);
        const norm01 = (sim + 1) / 2;
        if (row === col) {
          ctx.fillStyle = "rgba(56,189,248,0.72)";
        } else if (sim > 0.7) {
          ctx.fillStyle = `rgba(74,222,128,${norm01 * 0.8 + 0.08})`;
        } else if (sim > 0.3) {
          ctx.fillStyle = `rgba(56,189,248,${norm01 * 0.55 + 0.08})`;
        } else if (sim > 0) {
          ctx.fillStyle = `rgba(148,163,184,${norm01 * 0.35 + 0.08})`;
        } else {
          ctx.fillStyle = `rgba(248,113,113,${(1 - norm01) * 0.46 + 0.08})`;
        }
        ctx.fillRect(offset + col * cellSize, offset + row * cellSize, cellSize - 1, cellSize - 1);
      }
    }
  }

  function updateDemo() {
    const numInput = document.getElementById("demoNum");
    const rangeInput = document.getElementById("demoRange");
    const value = clamp(parseInt(numInput.value || "0", 10) || 0, 0, 255);
    numInput.value = String(value);
    rangeInput.value = String(value);

    const bits = byteToBits(value);
    const floatLatent = getFloatLatent(value);
    const lutLatent = getLutLatent(value);
    const activeLatent = currentDemoMode === "float" ? floatLatent : lutLatent;
    const decoded = neuralDecode(activeLatent);
    const match = decoded.every((bit, index) => bit === bits[index]);

    const charLabel = (value >= 32 && value < 127)
      ? String.fromCharCode(value)
      : (value === 0 ? "NUL" : `0x${value.toString(16).toUpperCase().padStart(2, "0")}`);
    document.getElementById("demoChar").textContent = charLabel;
    document.getElementById("demoByteLabel").textContent = `0x${value.toString(16).toUpperCase().padStart(2, "0")}`;
    document.getElementById("demoPathLabel").textContent = currentDemoMode === "float"
      ? "byte → bits → neural latent → decoded bits"
      : "byte → LUT latent → decoded bits";
    document.getElementById("demoVerifyBadge").textContent = verifyMode(currentDemoMode);
    document.getElementById("demoDeltaLabel").textContent = `Δ max ${maxAbsDelta(floatLatent, lutLatent).toFixed(4)}`;

    const maxValue = Math.max(
      ...activeLatent.map((entry) => Math.abs(entry)),
      1,
    );

    const bitsHtml = bits.map((bit) => `<span class="bit ${bit === 1 ? "bit--pos" : "bit--neg"}">${bit === 1 ? "+1" : "-1"}</span>`).join("");
    const latentHtml = activeLatent.map((entry) => {
      const ratio = Math.abs(entry) / maxValue;
      const alpha = 0.28 + ratio * 0.7;
      const height = Math.max(8, ratio * 78);
      const color = entry >= 0
        ? `rgba(74,222,128,${alpha})`
        : `rgba(248,113,113,${alpha})`;
      return `<div class="latent-col" style="height:${height}px;background:${color}"></div>`;
    }).join("");
    const decodeHtml = decoded.map((bit, index) => {
      const ok = bit === bits[index];
      const bitClass = ok ? (bit === 1 ? "bit--pos" : "bit--neg") : "bit--warn";
      return `<span class="bit ${bitClass}">${bit === 1 ? "+1" : "-1"}</span>`;
    }).join("");
    const modeInline = currentDemoMode === "float"
      ? `<span class="panel__mono">Mode: neural encoder output</span>`
      : `<span class="panel__mono">Mode: baked LUT deploy output</span>`;

    const resultHtml = [
      `<div class="demo-row"><span class="demo-row-label">Bits</span><div class="bit-row">${bitsHtml}</div></div>`,
      `<div class="demo-row"><span class="demo-row-label">Source</span><div class="demo-inline-list">${modeInline}<span class="panel__mono">other path Δmax ${maxAbsDelta(floatLatent, lutLatent).toFixed(4)}</span></div></div>`,
      `<div class="demo-row"><span class="demo-row-label">16D latent</span><div class="latent-bar">${latentHtml}</div></div>`,
      `<div class="demo-row"><span class="demo-row-label">Decoded</span><div class="bit-row">${decodeHtml}</div></div>`,
      `<div class="demo-row"><span class="demo-row-label">Match</span><div><span class="match-badge ${match ? "match-badge--ok" : "match-badge--fail"}">${match ? "PERFECT MATCH" : "MISMATCH"}</span></div></div>`,
    ].join("");

    document.getElementById("demoResult").innerHTML = resultHtml;
  }

  function wireDemo() {
    const numInput = document.getElementById("demoNum");
    const rangeInput = document.getElementById("demoRange");
    const modeButtons = [...document.querySelectorAll("[data-demo-mode]")];

    numInput.addEventListener("input", () => {
      rangeInput.value = numInput.value;
      updateDemo();
    });
    rangeInput.addEventListener("input", () => {
      numInput.value = rangeInput.value;
      updateDemo();
    });

    modeButtons.forEach((button) => {
      button.addEventListener("click", () => {
        currentDemoMode = button.dataset.demoMode;
        modeButtons.forEach((other) => {
          const active = other === button;
          other.classList.toggle("is-active", active);
          other.setAttribute("aria-selected", active ? "true" : "false");
        });
        updateDemo();
      });
    });

    updateDemo();
  }

  function closeMobileSheet() {
    mobileSheet.hidden = true;
  }

  function openMobileSheet() {
    mobileSheet.hidden = false;
  }

  function bindNavigation() {
    navLinks.forEach((link) => {
      link.addEventListener("click", (event) => {
        event.preventDefault();
        const target = link.dataset.slideTarget;
        goToIndex(canonicalSlides.indexOf(target), { updateHash: true });
        closeMobileSheet();
      });
    });

    progressDots.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement) || !target.dataset.slideIndex) return;
      goToIndex(Number(target.dataset.slideIndex), { updateHash: true });
    });

    mobileSlideList.addEventListener("click", (event) => {
      const target = event.target instanceof HTMLElement ? event.target.closest("[data-slide-index]") : null;
      if (!(target instanceof HTMLElement)) return;
      goToIndex(Number(target.dataset.slideIndex), { updateHash: true });
      closeMobileSheet();
    });

    mobilePrev.addEventListener("click", () => maybeNavigate(-1));
    mobileNext.addEventListener("click", () => maybeNavigate(1));
    mobileMenuToggle.addEventListener("click", () => {
      if (mobileSheet.hidden) openMobileSheet();
      else closeMobileSheet();
    });
    mobileSheet.addEventListener("click", (event) => {
      const target = event.target;
      if (target instanceof HTMLElement && target.hasAttribute("data-sheet-close")) {
        closeMobileSheet();
      }
    });

    window.addEventListener("hashchange", () => {
      goToHash(location.hash, { replaceHash: false, force: true });
    });

    window.addEventListener("keydown", (event) => {
      if (isInteractiveTarget(document.activeElement)) return;
      if (event.key === "ArrowDown" || event.key === "PageDown") {
        event.preventDefault();
        maybeNavigate(1);
      } else if (event.key === "ArrowUp" || event.key === "PageUp") {
        event.preventDefault();
        maybeNavigate(-1);
      } else if (event.key === "Home") {
        event.preventDefault();
        goToIndex(0, { updateHash: true });
      } else if (event.key === "End") {
        event.preventDefault();
        goToIndex(slides.length - 1, { updateHash: true });
      } else if (event.key === "Escape") {
        closeMobileSheet();
      }
    }, { passive: false });

    window.addEventListener("wheel", (event) => {
      if (maybeUseScrollRegion(event.target, event.deltaY)) return;
      if (Math.abs(event.deltaY) < 24) return;
      event.preventDefault();
      maybeNavigate(event.deltaY > 0 ? 1 : -1);
    }, { passive: false });

    window.addEventListener("touchstart", (event) => {
      const touch = event.changedTouches[0];
      touchState = {
        y: touch.clientY,
        target: event.target,
      };
    }, { passive: true });

    window.addEventListener("touchend", (event) => {
      if (!touchState) return;
      if (touchState.target instanceof Element && touchState.target.closest(".scroll-region")) {
        touchState = null;
        return;
      }
      const touch = event.changedTouches[0];
      const dy = touch.clientY - touchState.y;
      if (Math.abs(dy) > 70) {
        maybeNavigate(dy < 0 ? 1 : -1);
      }
      touchState = null;
    }, { passive: true });
  }

  function bindInspectTabs() {
    inspectTabs.forEach((button) => {
      button.addEventListener("click", () => {
        setInspectTab(button.dataset.inspectTab);
      });
    });
    setInspectTab("weights");
  }

  function initialise() {
    renderProgressDots();
    bindNavigation();
    bindInspectTabs();
    renderHeroPreview();
    wireDemo();
    drawPareto();
    drawHeatmap("heatW1", W1);
    drawHeatmap("heatW2", W2);
    drawHistogram();
    renderNeurons();
    drawSimilarity();

    const initialHash = location.hash || "#hero";
    const canonical = resolveHash(initialHash);
    const initialIndex = canonicalSlides.indexOf(canonical);
    activeIndex = clamp(initialIndex, 0, slides.length - 1);
    setSlideTransforms();

    if (canonical !== String(initialHash).replace(/^#/, "")) {
      setHashForIndex(activeIndex, true);
    } else if (!location.hash) {
      setHashForIndex(activeIndex, true);
    }
  }

  initialise();
})();
