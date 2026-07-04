(() => {
  "use strict";

  const doc = document.documentElement;
  const safeMatch = (query) => {
    if (typeof window.matchMedia !== "function") {
      return { matches: false, addEventListener() {}, removeEventListener() {} };
    }
    return window.matchMedia(query);
  };
  const reduceMotionQuery = safeMatch("(prefers-reduced-motion: reduce)");
  const finePointerQuery = safeMatch("(pointer: fine)");
  let reduceMotion = reduceMotionQuery.matches;
  const finePointer = finePointerQuery.matches;
  const raf = window.requestAnimationFrame
    ? window.requestAnimationFrame.bind(window)
    : (callback) => window.setTimeout(() => callback(Date.now()), 16);
  const caf = window.cancelAnimationFrame
    ? window.cancelAnimationFrame.bind(window)
    : window.clearTimeout.bind(window);
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

  function isVisible(el) {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    return style.visibility !== "hidden" && style.display !== "none";
  }

  function getFocusable(container) {
    if (!container) return [];
    return Array.from(container.querySelectorAll(focusableSelector)).filter((el) => {
      if (el.hasAttribute("disabled") || el.getAttribute("aria-hidden") === "true") return false;
      return isVisible(el) && (el.offsetWidth > 0 || el.offsetHeight > 0 || el === document.activeElement);
    });
  }

  function setElementInert(el, inert) {
    if (!el) return;
    try {
      el.inert = inert;
    } catch (_err) {
      // Older browsers may only honor the inert attribute.
    }
    el.toggleAttribute("inert", inert);
  }

  doc.classList.add("has-js");

  const progressBar = document.querySelector(".scroll-progress span");
  const backToTop = document.querySelector(".back-to-top");
  const indicator = document.querySelector(".section-indicator");
  const indicatorReadout = document.querySelector(".indicator-readout");
  const indicatorFill = document.querySelector(".indicator-track span");
  const indicatorThumb = document.querySelector(".indicator-track i");
  const indicatorNumber = document.querySelector("[data-indicator-number]");
  const indicatorTotal = document.querySelector("[data-indicator-total]");
  const indicatorLabel = document.querySelector("[data-indicator-label]");
  const mobileIndicatorNumber = document.querySelector("[data-mobile-indicator-number]");
  const mobileIndicatorTotal = document.querySelector("[data-mobile-indicator-total]");
  const mobileIndicatorLabel = document.querySelector("[data-mobile-indicator-label]");
  const mobileSectionReadout = document.querySelector(".mobile-section-readout");
  const sectionLinks = Array.from(document.querySelectorAll("[data-section-link]"));
  const hero = document.querySelector(".hero");
  const heroGlow = document.querySelector(".hero-cursor-glow");

  const sections = sectionLinks
    .map((link) => {
      const id = (link.getAttribute("href") || "").replace("#", "");
      return {
        id,
        label: link.dataset.sectionLink || link.textContent.trim().toLowerCase(),
        el: document.getElementById(id),
        link,
      };
    })
    .filter((section) => section.id && section.el);

  let activeIndex = -1;
  let scrollTicking = false;
  const readoutAnimationTimers = new WeakMap();

  function animateReadout(el) {
    if (!el || reduceMotion) return;
    const existing = readoutAnimationTimers.get(el);
    if (existing) window.clearTimeout(existing);
    el.classList.remove("is-changing");
    void el.offsetWidth;
    el.classList.add("is-changing");
    readoutAnimationTimers.set(
      el,
      window.setTimeout(() => {
        el.classList.remove("is-changing");
        readoutAnimationTimers.delete(el);
      }, 280)
    );
  }

  function updateActiveSection(nextIndex) {
    if (nextIndex === activeIndex || sections.length === 0) return;

    activeIndex = nextIndex;
    const section = sections[activeIndex] || sections[0];
    const total = sections.length;
    const progress = ((activeIndex + 1) / total) * 100;
    const number = String(activeIndex + 1).padStart(2, "0");
    const previousLabel = indicatorLabel?.textContent || "";

    if (indicatorFill) indicatorFill.style.height = `${progress}%`;
    if (indicatorThumb) indicatorThumb.style.top = `calc(${progress}% - 3px)`;
    if (indicatorNumber) indicatorNumber.textContent = number;
    if (indicatorTotal) indicatorTotal.textContent = `/ ${total}`;
    if (indicatorLabel) indicatorLabel.textContent = section.label;
    if (mobileIndicatorNumber) mobileIndicatorNumber.textContent = number;
    if (mobileIndicatorTotal) mobileIndicatorTotal.textContent = `/ ${total}`;
    if (mobileIndicatorLabel) mobileIndicatorLabel.textContent = section.label;
    if (mobileSectionReadout) mobileSectionReadout.classList.toggle("is-hidden", activeIndex === 0);
    if (previousLabel && previousLabel !== section.label) {
      animateReadout(indicatorReadout);
      animateReadout(mobileSectionReadout);
    }

    sectionLinks.forEach((link) => {
      const isActive = link === section.link;
      link.classList.toggle("is-active", isActive);
      if (isActive) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    });
  }

  function updateScrollState() {
    const maxScroll = Math.max(1, document.body.scrollHeight - window.innerHeight);
    const scrollRatio = Math.min(1, Math.max(0, window.scrollY / maxScroll));

    if (progressBar) progressBar.style.transform = `scaleX(${scrollRatio})`;
    if (backToTop) {
      backToTop.classList.toggle("is-visible", window.scrollY > window.innerHeight * 1.25);
    }
    if (hero) {
      const rect = hero.getBoundingClientRect();
      const range = Math.max(1, rect.height * 0.8);
      const heroProgress = Math.min(1, Math.max(0, -rect.top / range));
      const heroOpacity = Math.max(0, 1 - heroProgress);
      hero.style.setProperty("--hero-scroll-y", reduceMotion ? "0px" : `${(120 * heroProgress).toFixed(2)}px`);
      hero.style.setProperty("--hero-scroll-bg-y", reduceMotion ? "0px" : `${(42 * heroProgress).toFixed(2)}px`);
      hero.style.setProperty("--hero-scroll-opacity", reduceMotion ? "1" : heroOpacity.toFixed(3));
    }

    if (indicator && sections.length > 0) {
      const trigger = window.innerHeight * 0.56;
      let nextIndex = 0;

      sections.forEach((section, index) => {
        if (section.el.getBoundingClientRect().top <= trigger) nextIndex = index;
      });

      if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 4) {
        nextIndex = sections.length - 1;
      }

      updateActiveSection(nextIndex);
    }

    scrollTicking = false;
  }

  function requestScrollUpdate() {
    if (scrollTicking) return;
    scrollTicking = true;
    raf(updateScrollState);
  }

  if (backToTop) {
    backToTop.addEventListener("click", () => {
      window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
    });
  }

  window.addEventListener("scroll", requestScrollUpdate, { passive: true });
  window.addEventListener("resize", requestScrollUpdate);
  updateScrollState();

  if (hero) {
    raf(() => hero.classList.add("is-booted"));
  }

  if (hero && heroGlow && finePointer && !reduceMotion) {
    const target = {
      x: window.innerWidth / 2,
      y: window.innerHeight / 2,
      bgX: 0,
      bgY: 0,
      meshX: 0,
      meshY: 0,
      markX: 0,
      markY: 0,
    };
    const current = { ...target };
    let heroRaf = 0;
    let active = false;

    function setHeroTarget(event) {
      const rect = hero.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const nx = x / Math.max(1, rect.width) - 0.5;
      const ny = y / Math.max(1, rect.height) - 0.5;

      target.x = x;
      target.y = y;
      target.bgX = nx * -18;
      target.bgY = ny * -12;
      target.meshX = nx * 30;
      target.meshY = ny * 22;
      target.markX = nx * 8;
      target.markY = ny * 5;
    }

    function ease(currentValue, targetValue, factor) {
      return currentValue + (targetValue - currentValue) * factor;
    }

    function tickHero() {
      current.x = ease(current.x, target.x, 0.09);
      current.y = ease(current.y, target.y, 0.09);
      current.bgX = ease(current.bgX, target.bgX, 0.08);
      current.bgY = ease(current.bgY, target.bgY, 0.08);
      current.meshX = ease(current.meshX, target.meshX, 0.08);
      current.meshY = ease(current.meshY, target.meshY, 0.08);
      current.markX = ease(current.markX, target.markX, 0.1);
      current.markY = ease(current.markY, target.markY, 0.1);

      heroGlow.style.transform = `translate3d(${current.x - 300}px, ${current.y - 300}px, 0)`;
      hero.style.setProperty("--hero-bg-x", `${current.bgX.toFixed(2)}px`);
      hero.style.setProperty("--hero-bg-y", `${current.bgY.toFixed(2)}px`);
      hero.style.setProperty("--hero-mesh-x", `${current.meshX.toFixed(2)}px`);
      hero.style.setProperty("--hero-mesh-y", `${current.meshY.toFixed(2)}px`);
      hero.style.setProperty("--hero-mark-x", `${current.markX.toFixed(2)}px`);
      hero.style.setProperty("--hero-mark-y", `${current.markY.toFixed(2)}px`);

      const moving =
        Math.abs(current.x - target.x) +
          Math.abs(current.y - target.y) +
          Math.abs(current.bgX - target.bgX) +
          Math.abs(current.meshX - target.meshX) +
          Math.abs(current.markX - target.markX) >
        0.15;

      if (active || moving) heroRaf = raf(tickHero);
      else heroRaf = 0;
    }

    function startHeroLoop() {
      if (!heroRaf) heroRaf = raf(tickHero);
    }

    hero.addEventListener(
      "pointerenter",
      (event) => {
        active = true;
        hero.classList.add("is-pointer-active");
        setHeroTarget(event);
        startHeroLoop();
      },
      { passive: true }
    );

    hero.addEventListener(
      "pointermove",
      (event) => {
        setHeroTarget(event);
        startHeroLoop();
      },
      { passive: true }
    );

    hero.addEventListener(
      "pointerleave",
      () => {
        const rect = hero.getBoundingClientRect();
        active = false;
        hero.classList.remove("is-pointer-active");
        target.x = rect.width * 0.55;
        target.y = rect.height * 0.45;
        target.bgX = 0;
        target.bgY = 0;
        target.meshX = 0;
        target.meshY = 0;
        target.markX = 0;
        target.markY = 0;
        startHeroLoop();
      },
      { passive: true }
    );
  }

  function installHeroMesh() {
    const canvas = document.querySelector(".hero-mesh-canvas");
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const mouse = { x: -1000, y: -1000, active: false };
    let width = 0;
    let height = 0;
    let nodes = [];
    let meshRaf = 0;
    let meshVisible = true;

    function shouldAnimateMesh() {
      return !reduceMotion && meshVisible && document.visibilityState !== "hidden";
    }

    function stopMesh() {
      if (meshRaf) {
        caf(meshRaf);
        meshRaf = 0;
      }
    }

    function queueMesh() {
      if (!meshRaf && shouldAnimateMesh()) meshRaf = raf(draw);
    }

    function renderMeshFrame() {
      stopMesh();
      draw();
      if (!shouldAnimateMesh()) stopMesh();
    }

    function init() {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const count = Math.max(24, Math.min(90, Math.floor(width * height * 0.00008)));
      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.18,
        vy: (Math.random() - 0.5) * 0.18,
        r: 1 + Math.random() * 1.8,
        hue: Math.random() < 0.7 ? 0 : 1,
        pulse: Math.random() * Math.PI * 2,
      }));
    }

    function draw() {
      meshRaf = 0;
      ctx.clearRect(0, 0, width, height);
      const maxDist = 140;

      for (let i = 0; i < nodes.length; i += 1) {
        for (let j = i + 1; j < nodes.length; j += 1) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d >= maxDist) continue;

          const alpha = (1 - d / maxDist) * 0.18;
          const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
          grad.addColorStop(0, `rgba(${a.hue ? "255, 61, 129" : "0, 229, 255"}, ${alpha})`);
          grad.addColorStop(1, `rgba(${b.hue ? "255, 61, 129" : "0, 229, 255"}, ${alpha})`);
          ctx.strokeStyle = grad;
          ctx.lineWidth = 0.7;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }

      nodes.forEach((node) => {
        if (!reduceMotion) {
          node.x += node.vx;
          node.y += node.vy;
          node.pulse += 0.02;

          if (node.x < -10) node.x = width + 10;
          if (node.x > width + 10) node.x = -10;
          if (node.y < -10) node.y = height + 10;
          if (node.y > height + 10) node.y = -10;

          if (mouse.active) {
            const dx = node.x - mouse.x;
            const dy = node.y - mouse.y;
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d < 160 && d > 0.1) {
              const force = (1 - d / 160) * 0.6;
              node.x += (dx / d) * force;
              node.y += (dy / d) * force;
            }
          }
        }

        const pulse = 0.6 + Math.sin(node.pulse) * 0.4;
        const color = node.hue ? "255, 61, 129" : "0, 229, 255";
        const glow = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.r * 6);
        glow.addColorStop(0, `rgba(${color}, ${0.5 * pulse})`);
        glow.addColorStop(1, `rgba(${color}, 0)`);
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.r * 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = `rgba(${color}, ${0.9 * pulse})`;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
        ctx.fill();
      });

      if (shouldAnimateMesh()) meshRaf = raf(draw);
    }

    function onMove(event) {
      const rect = canvas.getBoundingClientRect();
      mouse.x = event.clientX - rect.left;
      mouse.y = event.clientY - rect.top;
      mouse.active = true;
    }

    function onLeave() {
      mouse.active = false;
      mouse.x = -1000;
      mouse.y = -1000;
    }

    init();
    draw();
    window.addEventListener("resize", () => {
      init();
      if (shouldAnimateMesh()) queueMesh();
      else renderMeshFrame();
    });
    document.addEventListener("visibilitychange", () => {
      if (shouldAnimateMesh()) queueMesh();
      else stopMesh();
    });
    reduceMotionQuery.addEventListener("change", () => {
      reduceMotion = reduceMotionQuery.matches;
      if (shouldAnimateMesh()) queueMesh();
      else renderMeshFrame();
    });
    if ("IntersectionObserver" in window) {
      const obs = new IntersectionObserver(
        (entries) => {
          meshVisible = entries.some((entry) => entry.isIntersecting);
          if (shouldAnimateMesh()) queueMesh();
          else stopMesh();
        },
        { rootMargin: "160px" }
      );
      obs.observe(canvas);
    }
    if (!reduceMotion && hero) {
      hero.addEventListener("pointermove", onMove, { passive: true });
      hero.addEventListener("pointerleave", onLeave, { passive: true });
    }
  }

  installHeroMesh();

  const modeCard = document.querySelector(".mode-card");
  if (modeCard) {
    const modeSwitch = modeCard.querySelector(".mode-switch");
    const modeState = modeCard.querySelector("[data-mode-state]");
    const panels = Array.from(modeCard.querySelectorAll("[data-mode-panel]"));

    function setMode(mode) {
      const isImagination = mode === "imagination";
      modeCard.dataset.mode = mode;
      if (modeSwitch) modeSwitch.setAttribute("aria-checked", String(isImagination));
      if (modeState) modeState.textContent = isImagination ? "imagination opt-in" : "exact by default";
      panels.forEach((panel) => {
        const isActive = panel.dataset.modePanel === mode;
        panel.hidden = !isActive;
        panel.classList.toggle("is-active", isActive);
        panel.setAttribute("aria-hidden", String(!isActive));
        setElementInert(panel, !isActive);
      });
    }

    if (modeSwitch) {
      modeSwitch.addEventListener("click", () => {
        setMode(modeCard.dataset.mode === "imagination" ? "exact" : "imagination");
      });
    }
    setMode("exact");
  }

  function countTo(el, value, duration) {
    if (reduceMotion) {
      el.textContent = String(value);
      return;
    }
    const start = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = String(Math.round(value * eased));
      if (t < 1) raf(tick);
    }
    el.textContent = "0";
    raf(tick);
  }

  const countSections = Array.from(document.querySelectorAll("[data-benchmark], .live-readout"));
  countSections.forEach((benchmark) => {
    const counters = Array.from(benchmark.querySelectorAll("[data-count-to]"));
    counters.forEach((counter) => {
      if (!reduceMotion) counter.textContent = "0";
    });

    const revealBenchmark = () => {
      if (benchmark.classList.contains("is-visible")) return;
      benchmark.classList.add("is-visible");
      counters.forEach((counter, index) => {
        countTo(counter, Number(counter.dataset.countTo || "0"), 900 + index * 220);
      });
    };

    if ("IntersectionObserver" in window && !reduceMotion) {
      const obs = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            revealBenchmark();
            obs.disconnect();
          }
        },
        { threshold: 0.32 }
      );
      obs.observe(benchmark);
    } else {
      revealBenchmark();
    }
  });

  function installManifesto() {
    const wrap = document.querySelector("[data-manifesto]");
    if (!wrap) return;
    const lines = Array.from(wrap.querySelectorAll("[data-manifesto-line]"));
    const signature = wrap.querySelector(".manifesto-signature");
    const texts = lines.map((line) => line.textContent.trim());

    if (reduceMotion) {
      wrap.classList.add("is-complete");
      return;
    }

    lines.forEach((line) => {
      line.textContent = "";
    });
    if (signature) signature.setAttribute("aria-hidden", "true");

    const caret = document.createElement("span");
    caret.className = "manifesto-caret";
    caret.setAttribute("aria-hidden", "true");

    function renderLine(index, length) {
      const line = lines[index];
      if (!line) return;
      line.replaceChildren(document.createTextNode(texts[index].slice(0, length)), caret);
    }

    function typeLine(index, length) {
      if (index >= lines.length) {
        caret.remove();
        wrap.classList.add("is-complete");
        if (signature) signature.removeAttribute("aria-hidden");
        return;
      }

      renderLine(index, length);
      if (length <= texts[index].length) {
        window.setTimeout(() => typeLine(index, length + 1), texts[index].length ? 34 : 180);
      } else {
        window.setTimeout(() => typeLine(index + 1, 0), 360);
      }
    }

    const start = () => typeLine(0, 0);
    if ("IntersectionObserver" in window) {
      const obs = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            start();
            obs.disconnect();
          }
        },
        { threshold: 0.35 }
      );
      obs.observe(wrap);
    } else {
      start();
    }
  }

  installManifesto();

  function installReveals() {
    const revealTargets = Array.from(
      document.querySelectorAll(
        [
          ".section-label",
          ".section-heading",
          ".center-heading",
          ".pillar",
          ".trust-card",
          ".benchmark-card",
          ".statement-panel",
          ".fabric-flow-panel",
          ".fabric-diagram article",
          ".terminal-panel",
          ".info-card",
          ".roadmap-list li",
          ".faq-item",
          ".get-notified .button",
        ].join(",")
      )
    );

    if (revealTargets.length === 0) return;
    revealTargets.forEach((target, index) => {
      target.dataset.reveal = "";
      target.style.setProperty("--reveal-delay", `${Math.min(index % 4, 3) * 45}ms`);
    });

    if (reduceMotion || !("IntersectionObserver" in window)) {
      revealTargets.forEach((target) => target.classList.add("is-revealed"));
      return;
    }

    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-revealed");
          obs.unobserve(entry.target);
        });
      },
      { rootMargin: "0px 0px -12% 0px", threshold: 0.14 }
    );

    revealTargets.forEach((target) => obs.observe(target));
  }

  installReveals();

  function installFabricFlow() {
    const canvas = document.querySelector(".fabric-flow-canvas");
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = 0;
    let height = 0;
    let t = 0;
    let flowRaf = 0;
    let flowVisible = true;
    const nodes = [
      { label: "INPUT", sub: "data", x: 0.08, color: "0, 229, 255" },
      { label: "PRISMION", sub: "prism + neuron", x: 0.32, color: "255, 61, 129" },
      { label: "α-SYNC", sub: "fabric", x: 0.56, color: "0, 229, 255" },
      { label: "BOUNDED", sub: "result path", x: 0.8, color: "255, 61, 129" },
    ];

    function shouldAnimateFlow() {
      return !reduceMotion && flowVisible && document.visibilityState !== "hidden";
    }

    function stopFlow() {
      if (flowRaf) {
        caf(flowRaf);
        flowRaf = 0;
      }
    }

    function queueFlow() {
      if (!flowRaf && shouldAnimateFlow()) flowRaf = raf(draw);
    }

    function renderFlowFrame() {
      stopFlow();
      draw();
      if (!shouldAnimateFlow()) stopFlow();
    }

    function init() {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function draw() {
      flowRaf = 0;
      if (!reduceMotion) t += 0.012;
      ctx.clearRect(0, 0, width, height);
      const pad = 16;
      const usableW = width - pad * 2;
      const cy = height / 2;

      for (let i = 0; i < nodes.length - 1; i += 1) {
        const a = nodes[i];
        const b = nodes[i + 1];
        const ax = pad + a.x * usableW;
        const bx = pad + b.x * usableW;
        const ay = cy + Math.sin(t + i) * 6;
        const by = cy + Math.sin(t + i + 1) * 6;
        const grad = ctx.createLinearGradient(ax, ay, bx, by);
        grad.addColorStop(0, `rgba(${a.color}, 0.5)`);
        grad.addColorStop(1, `rgba(${b.color}, 0.5)`);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      }

      const pos = (t * 0.5) % (nodes.length - 1);
      const segIdx = Math.floor(pos);
      const segFrac = pos - segIdx;
      const a = nodes[segIdx];
      const b = nodes[segIdx + 1] || a;
      const px = pad + (a.x + (b.x - a.x) * segFrac) * usableW;
      const py = cy + Math.sin(t + segIdx + segFrac) * 6;
      const pulseGrad = ctx.createRadialGradient(px, py, 0, px, py, 18);
      pulseGrad.addColorStop(0, "rgba(255, 255, 255, 0.9)");
      pulseGrad.addColorStop(0.4, `rgba(${a.color}, 0.6)`);
      pulseGrad.addColorStop(1, `rgba(${a.color}, 0)`);
      ctx.fillStyle = pulseGrad;
      ctx.beginPath();
      ctx.arc(px, py, 18, 0, Math.PI * 2);
      ctx.fill();

      nodes.forEach((node, index) => {
        const nx = pad + node.x * usableW;
        const ny = cy + Math.sin(t + index) * 6;
        const pulse = 0.7 + Math.sin(t * 2 + index) * 0.3;
        const glow = ctx.createRadialGradient(nx, ny, 0, nx, ny, 40);
        glow.addColorStop(0, `rgba(${node.color}, ${0.25 * pulse})`);
        glow.addColorStop(1, `rgba(${node.color}, 0)`);
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(nx, ny, 40, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = `rgba(${node.color}, 0.6)`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(nx, ny, 16, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = `rgba(${node.color}, 0.9)`;
        ctx.beginPath();
        ctx.arc(nx, ny, 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "rgba(240, 244, 248, 0.92)";
        ctx.font = "700 10px SFMono-Regular, Cascadia Code, Consolas, monospace";
        ctx.textAlign = "center";
        ctx.fillText(node.label, nx, ny + 38);
        ctx.fillStyle = "rgba(138, 147, 164, 0.72)";
        ctx.font = "500 9px SFMono-Regular, Cascadia Code, Consolas, monospace";
        ctx.fillText(node.sub, nx, ny + 52);
      });

      if (shouldAnimateFlow()) flowRaf = raf(draw);
    }

    init();
    draw();
    window.addEventListener("resize", () => {
      init();
      if (shouldAnimateFlow()) queueFlow();
      else renderFlowFrame();
    });
    document.addEventListener("visibilitychange", () => {
      if (shouldAnimateFlow()) queueFlow();
      else stopFlow();
    });
    reduceMotionQuery.addEventListener("change", () => {
      reduceMotion = reduceMotionQuery.matches;
      if (shouldAnimateFlow()) queueFlow();
      else renderFlowFrame();
    });
    if ("IntersectionObserver" in window) {
      const obs = new IntersectionObserver(
        (entries) => {
          flowVisible = entries.some((entry) => entry.isIntersecting);
          if (shouldAnimateFlow()) queueFlow();
          else stopFlow();
        },
        { rootMargin: "160px" }
      );
      obs.observe(canvas);
    }
  }

  installFabricFlow();

  function installCliDemo() {
    const pre = document.querySelector("[data-cli-demo]");
    if (!pre) return;
    const code = pre.querySelector("code");
    if (!code) return;

    const script = [
      { type: "prompt", text: "$ instnct --version" },
      { type: "output", text: "INSTNCT T1 Reflex Engine - v0.1.0-preview" },
      { type: "trace", text: "VRAXION local proof target - network disabled" },
      { type: "prompt", text: "$ instnct run --mode exact-recall" },
      { type: "ok", text: "ok selector timing: see signed benchmark notes" },
      { type: "ok", text: "ok exact/refusal contract: ON (approved paths only)" },
      { type: "query", text: '$ "What is the capital of Hungary?"' },
      { type: "output", text: "Budapest" },
      { type: "trace", text: "[trace: selector -> approved path - 1 hop - grounded]" },
      { type: "query", text: '$ "Invent a city that does not exist"' },
      { type: "warn", text: "refused - no approved pattern matches" },
      { type: "trace", text: "[exact-recall mode does not invent. flip toggle to compose.]" },
      { type: "prompt", text: "$ instnct toggle --imagination on" },
      { type: "ok", text: "ok imagination: ON (bounded drift budget: 0.3)" },
    ];

    function createLine(entry, text, includeCursor) {
      const span = document.createElement("span");
      span.className = "terminal-line";
      span.dataset.lineType = entry.type;
      span.append(document.createTextNode(text));
      if (includeCursor) span.append(cursor);
      return span;
    }

    function renderLines(limit, currentText, includeCursor) {
      const frag = document.createDocumentFragment();
      for (let i = 0; i < limit; i += 1) {
        frag.append(createLine(script[i], script[i].text, false));
      }
      if (limit < script.length) {
        frag.append(createLine(script[limit], currentText, includeCursor));
      }
      code.replaceChildren(frag);
      pre.scrollTop = pre.scrollHeight;
    }

    if (reduceMotion) {
      renderLines(script.length, "", false);
      return;
    }

    const cursor = document.createElement("span");
    cursor.className = "terminal-cursor";
    cursor.textContent = "|";
    cursor.setAttribute("aria-hidden", "true");

    let line = 0;
    let char = 0;
    function render() {
      const current = script[line] || { text: "", type: "output" };
      renderLines(line, current.text.slice(0, char), true);
    }

    function tick() {
      if (line >= script.length) {
        window.setTimeout(() => {
          line = 0;
          char = 0;
          tick();
        }, 4200);
        return;
      }

      const current = script[line];
      render();
      if (char <= current.text.length) {
        char += 1;
        window.setTimeout(tick, current.type === "prompt" || current.type === "query" ? 28 : 12);
      } else {
        line += 1;
        char = 0;
        window.setTimeout(tick, current.type === "prompt" || current.type === "query" ? 420 : 260);
      }
    }

    code.textContent = "";
    tick();
  }

  installCliDemo();

  function fallbackCopyText(text) {
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.top = "-1000px";
    document.body.append(area);
    area.select();
    try {
      const copied = document.execCommand("copy");
      return copied ? Promise.resolve() : Promise.reject(new Error("copy failed"));
    } finally {
      area.remove();
    }
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).catch(() => fallbackCopyText(text));
    }
    return fallbackCopyText(text);
  }

  const copyStatus = document.querySelector("[data-copy-status]");
  document.querySelectorAll("[data-copy-command]").forEach((button) => {
    button.addEventListener("click", () => {
      const original = button.textContent;
      copyText(button.dataset.copyCommand || "")
        .then(() => {
          button.classList.add("is-copied");
          button.textContent = "copied";
          if (copyStatus) copyStatus.textContent = `${original} copied`;
        })
        .catch(() => {
          button.classList.add("is-copy-failed");
          button.textContent = "copy failed";
          if (copyStatus) copyStatus.textContent = `${original} copy failed`;
        })
        .finally(() => {
          window.setTimeout(() => {
            button.textContent = original;
            button.classList.remove("is-copied", "is-copy-failed");
            if (copyStatus) copyStatus.textContent = "";
          }, 1300);
        });
    });
  });

  const faqItems = Array.from(document.querySelectorAll(".faq-item"));
  faqItems.forEach((item, index) => {
    const button = item.querySelector("button");
    const panel = item.querySelector(".faq-panel");
    if (!button || !panel) return;
    const panelId = `instnct-faq-${index + 1}`;
    const buttonId = `instnct-faq-button-${index + 1}`;
    panel.id = panelId;
    if (!button.id) button.id = buttonId;
    button.setAttribute("aria-controls", panelId);
    panel.setAttribute("role", "region");
    panel.setAttribute("aria-labelledby", button.id);

    function setFaqOpen(target, open) {
      const targetButton = target.querySelector("button");
      const targetPanel = target.querySelector(".faq-panel");
      target.classList.toggle("is-open", open);
      if (targetButton) targetButton.setAttribute("aria-expanded", String(open));
      if (targetPanel) {
        targetPanel.setAttribute("aria-hidden", String(!open));
        setElementInert(targetPanel, !open);
      }
    }

    setFaqOpen(item, item.classList.contains("is-open"));

    button.addEventListener("click", () => {
      const nextOpen = !item.classList.contains("is-open");
      faqItems.forEach((other) => setFaqOpen(other, false));
      setFaqOpen(item, nextOpen);
    });
  });

  const keyboardTrigger = document.querySelector(".keyboard-help-trigger");
  const keyboardDialog = document.querySelector(".keyboard-dialog");
  const keyboardPanel = document.querySelector(".keyboard-dialog-panel");
  const keyboardBackgroundState = new Map();
  let lastKeyboardFocus = null;
  let pendingGo = false;
  let pendingTimer = 0;

  function setKeyboardBackgroundInert(open) {
    if (!keyboardDialog) return;
    Array.from(document.body.children).forEach((el) => {
      if (el === keyboardDialog) return;
      if (open) {
        if (!keyboardBackgroundState.has(el)) {
          keyboardBackgroundState.set(el, {
            inert: el.hasAttribute("inert"),
            ariaHidden: el.getAttribute("aria-hidden"),
          });
        }
        setElementInert(el, true);
        el.setAttribute("aria-hidden", "true");
      } else {
        const state = keyboardBackgroundState.get(el);
        if (!state) return;
        setElementInert(el, state.inert);
        if (state.ariaHidden === null) el.removeAttribute("aria-hidden");
        else el.setAttribute("aria-hidden", state.ariaHidden);
      }
    });
    if (!open) keyboardBackgroundState.clear();
  }

  function restoreKeyboardFocus() {
    const candidate =
      lastKeyboardFocus && lastKeyboardFocus.focus && isVisible(lastKeyboardFocus)
        ? lastKeyboardFocus
        : keyboardTrigger;
    lastKeyboardFocus = null;
    if (candidate && candidate.focus && isVisible(candidate) && !candidate.closest("[inert]")) {
      candidate.focus({ preventScroll: true });
    }
  }

  function openKeyboardDialog() {
    if (!keyboardDialog) return;
    lastKeyboardFocus = document.activeElement;
    keyboardDialog.hidden = false;
    keyboardTrigger?.setAttribute("aria-expanded", "true");
    doc.classList.add("keyboard-dialog-open");
    setKeyboardBackgroundInert(true);
    const close = keyboardDialog.querySelector(".keyboard-close");
    if (close) close.focus({ preventScroll: true });
  }

  function closeKeyboardDialog() {
    if (!keyboardDialog || keyboardDialog.hidden) return;
    setKeyboardBackgroundInert(false);
    keyboardDialog.hidden = true;
    keyboardTrigger?.setAttribute("aria-expanded", "false");
    doc.classList.remove("keyboard-dialog-open");
    restoreKeyboardFocus();
  }

  function toggleKeyboardDialog() {
    if (!keyboardDialog) return;
    if (keyboardDialog.hidden) openKeyboardDialog();
    else closeKeyboardDialog();
  }

  if (keyboardTrigger) keyboardTrigger.addEventListener("click", toggleKeyboardDialog);
  if (keyboardDialog) {
    keyboardDialog.querySelectorAll("[data-keyboard-close]").forEach((el) => {
      el.addEventListener("click", closeKeyboardDialog);
    });
    if (keyboardPanel) {
      keyboardPanel.addEventListener("click", (event) => event.stopPropagation());
    }
  }

  function trapKeyboardDialogFocus(event) {
    if (!keyboardDialog || keyboardDialog.hidden || event.key !== "Tab") return false;
    const focusable = getFocusable(keyboardPanel || keyboardDialog);
    if (focusable.length === 0) {
      event.preventDefault();
      return true;
    }

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus({ preventScroll: true });
      return true;
    }
    if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus({ preventScroll: true });
      return true;
    }
    return false;
  }

  function canUsePageShortcut(target) {
    if (!target || target === document.body || target === document.documentElement) return true;
    if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable) return false;
    return !target.closest("a[href], button, input, select, textarea, summary, [role], [contenteditable]");
  }

  function smoothScrollTo(target, focusTarget = false) {
    const el = typeof target === "string" ? document.querySelector(target) : target;
    if (!el) return;
    el.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    if (focusTarget && el.focus) el.focus({ preventScroll: true });
  }

  window.addEventListener("keydown", (event) => {
    if (trapKeyboardDialogFocus(event)) return;

    const target = event.target;
    const typing =
      target &&
      (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);
    if (typing) return;

    if (event.key === "Escape") {
      if (keyboardDialog && !keyboardDialog.hidden) {
        event.preventDefault();
        closeKeyboardDialog();
      }
      pendingGo = false;
      return;
    }

    if (keyboardDialog && !keyboardDialog.hidden) return;

    if (!canUsePageShortcut(target)) {
      pendingGo = false;
      return;
    }

    if (event.key === "?" || (event.key === "/" && event.shiftKey)) {
      event.preventDefault();
      toggleKeyboardDialog();
      return;
    }

    if (pendingGo) {
      pendingGo = false;
      window.clearTimeout(pendingTimer);
      if (event.key.toLowerCase() === "n") {
        event.preventDefault();
        smoothScrollTo("#get-notified", true);
      } else if (event.key.toLowerCase() === "h") {
        event.preventDefault();
        smoothScrollTo("#main", true);
      }
      return;
    }

    if (event.key.toLowerCase() === "g") {
      pendingGo = true;
      pendingTimer = window.setTimeout(() => {
        pendingGo = false;
      }, 1100);
      return;
    }

    if (event.key.toLowerCase() === "j") {
      event.preventDefault();
      window.scrollBy({ top: window.innerHeight * 0.82, behavior: reduceMotion ? "auto" : "smooth" });
    } else if (event.key.toLowerCase() === "k") {
      event.preventDefault();
      window.scrollBy({ top: -window.innerHeight * 0.82, behavior: reduceMotion ? "auto" : "smooth" });
    }
  });
})();
