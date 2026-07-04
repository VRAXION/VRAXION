(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = window.matchMedia("(pointer: fine)").matches;
  const doc = document.documentElement;

  doc.classList.add("has-js");

  const progressBar = document.querySelector(".scroll-progress span");
  const backToTop = document.querySelector(".back-to-top");
  const indicator = document.querySelector(".section-indicator");
  const indicatorFill = document.querySelector(".indicator-track span");
  const indicatorThumb = document.querySelector(".indicator-track i");
  const indicatorNumber = document.querySelector(".indicator-readout strong");
  const indicatorLabel = document.querySelector(".indicator-readout small");
  const sectionLinks = Array.from(document.querySelectorAll("[data-section-link]"));

  const sections = [
    { id: "intro", label: "intro" },
    { id: "not-ai", label: "position" },
    { id: "hallucination", label: "flagship" },
    { id: "trust", label: "verify" },
    { id: "grounding", label: "claim" },
    { id: "fabric", label: "structure" },
    { id: "dev-trail", label: "terminal" },
    { id: "t1-reflex-class", label: "model" },
    { id: "roadmap", label: "direction" },
    { id: "vraxion-note", label: "vraxion" },
    { id: "faq", label: "questions" },
    { id: "get-notified", label: "signal" },
  ]
    .map((section) => ({ ...section, el: document.getElementById(section.id) }))
    .filter((section) => section.el);

  let activeIndex = -1;
  let scrollTicking = false;

  function updateActiveSection(nextIndex) {
    if (nextIndex === activeIndex || sections.length === 0) return;

    activeIndex = nextIndex;
    const section = sections[activeIndex] || sections[0];
    const total = sections.length;
    const progress = ((activeIndex + 1) / total) * 100;
    const number = String(activeIndex + 1).padStart(2, "0");

    if (indicatorFill) indicatorFill.style.height = `${progress}%`;
    if (indicatorThumb) indicatorThumb.style.top = `calc(${progress}% - 3px)`;
    if (indicatorNumber) indicatorNumber.innerHTML = `${number} <span>/ ${total}</span>`;
    if (indicatorLabel) indicatorLabel.textContent = section.label;

    sectionLinks.forEach((link) => {
      const isActive = link.getAttribute("href") === `#${section.id}`;
      link.classList.toggle("is-active", isActive);
      if (isActive) {
        link.setAttribute("aria-current", "true");
      } else {
        link.removeAttribute("aria-current");
      }
    });
  }

  function updateScrollState() {
    const maxScroll = Math.max(1, document.body.scrollHeight - window.innerHeight);
    const scrollRatio = Math.min(1, Math.max(0, window.scrollY / maxScroll));

    if (progressBar) {
      progressBar.style.transform = `scaleX(${scrollRatio})`;
    }

    if (backToTop) {
      backToTop.classList.toggle("is-visible", window.scrollY > window.innerHeight * 1.25);
    }

    if (indicator && sections.length > 0) {
      const trigger = window.innerHeight * 0.56;
      let nextIndex = 0;

      for (let i = 0; i < sections.length; i += 1) {
        const rect = sections[i].el.getBoundingClientRect();
        if (rect.top <= trigger) nextIndex = i;
      }

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
    window.requestAnimationFrame(updateScrollState);
  }

  if (backToTop) {
    backToTop.addEventListener("click", () => {
      window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
    });
  }

  window.addEventListener("scroll", requestScrollUpdate, { passive: true });
  window.addEventListener("resize", requestScrollUpdate);
  updateScrollState();

  const hero = document.querySelector(".hero");
  const heroGlow = document.querySelector(".hero-cursor-glow");

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
        Math.abs(current.markX - target.markX) > 0.15;

      if (active || moving) {
        heroRaf = window.requestAnimationFrame(tickHero);
      } else {
        heroRaf = 0;
      }
    }

    function startHeroLoop() {
      if (!heroRaf) heroRaf = window.requestAnimationFrame(tickHero);
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
})();
