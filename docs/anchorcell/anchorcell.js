(() => {
  "use strict";

  const doc = document.documentElement;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = window.matchMedia("(pointer: fine)").matches;
  const hero = document.querySelector(".hero");
  const glow = document.querySelector(".hero-cursor-glow");
  const sectionLinks = Array.from(document.querySelectorAll("[data-section-link]"));
  const sections = sectionLinks
    .map((link, index) => {
      const id = link.getAttribute("href")?.slice(1) || "";
      return {
        index,
        id,
        label: link.dataset.sectionLink || link.textContent.trim(),
        link,
        el: document.getElementById(id),
      };
    })
    .filter((section) => section.el);
  const current = document.querySelector("[data-indicator-current]");
  const mobileCurrent = document.querySelector("[data-mobile-current]");
  const mobileLabel = document.querySelector("[data-mobile-label]");
  let activeIndex = -1;

  doc.classList.add("has-js");
  requestAnimationFrame(() => hero?.classList.add("is-booted"));

  function setActive(index) {
    if (!sections.length || index === activeIndex) return;
    activeIndex = index;
    const section = sections[index];
    const number = String(index + 1).padStart(2, "0");
    sectionLinks.forEach((link) => link.classList.toggle("is-active", link === section.link));
    section.el.classList.add("is-revealed");
    if (current) current.textContent = number;
    if (mobileCurrent) mobileCurrent.textContent = number;
    if (mobileLabel) mobileLabel.textContent = section.label;
  }

  function onScroll() {
    if (hero) {
      const rect = hero.getBoundingClientRect();
      const range = Math.max(1, rect.height * 0.7);
      const progress = Math.min(1, Math.max(0, -rect.top / range));
      hero.style.setProperty("--hero-scroll-y", reduceMotion ? "0px" : `${(90 * progress).toFixed(2)}px`);
      hero.style.setProperty("--hero-scroll-opacity", reduceMotion ? "1" : String(Math.max(0, 1 - progress * 0.78).toFixed(3)));
    }

    let nextIndex = 0;
    const trigger = Math.max(140, window.innerHeight * 0.34);
    sections.forEach((section, index) => {
      const rect = section.el.getBoundingClientRect();
      if (rect.top <= trigger) nextIndex = index;
      if (rect.top <= window.innerHeight + 80 && rect.bottom >= 0) {
        section.el.classList.add("is-revealed");
      }
    });
    setActive(nextIndex);
  }

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) entry.target.classList.add("is-revealed");
        });
      },
      { rootMargin: "0px 0px -12% 0px", threshold: 0.12 }
    );
    document.querySelectorAll(".section").forEach((section) => observer.observe(section));
  } else {
    document.querySelectorAll(".section").forEach((section) => section.classList.add("is-revealed"));
  }
  sections[0]?.el.classList.add("is-revealed");

  if (hero && finePointer && !reduceMotion) {
    hero.addEventListener(
      "pointermove",
      (event) => {
        const rect = hero.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const nx = (x / Math.max(1, rect.width) - 0.5) * 2;
        const ny = (y / Math.max(1, rect.height) - 0.5) * 2;
        hero.classList.add("is-pointer-active");
        hero.style.setProperty("--hero-bg-x", `${(nx * -18).toFixed(2)}px`);
        hero.style.setProperty("--hero-bg-y", `${(ny * -12).toFixed(2)}px`);
        hero.style.setProperty("--hero-glow-x", `${((x / Math.max(1, rect.width)) * 100).toFixed(2)}%`);
        hero.style.setProperty("--hero-glow-y", `${((y / Math.max(1, rect.height)) * 100).toFixed(2)}%`);
        if (glow) glow.style.transform = `translate3d(${(x - 260).toFixed(2)}px, ${(y - 260).toFixed(2)}px, 0)`;
      },
      { passive: true }
    );

    hero.addEventListener(
      "pointerleave",
      () => {
        hero.classList.remove("is-pointer-active");
        hero.style.setProperty("--hero-bg-x", "0px");
        hero.style.setProperty("--hero-bg-y", "0px");
      },
      { passive: true }
    );
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll, { passive: true });
  onScroll();
})();
