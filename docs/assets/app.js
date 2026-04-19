/* ============================================================
   VRAXION — Shared JS v5.0.0-β.2
   - Active nav highlighting
   - Mobile menu toggle
   - Keyboard navigation (← → h)
   ============================================================ */

(function () {
  'use strict';

  /* ── Active nav tab ──────────────────────────────────────── */
  function highlightNav() {
    const path = window.location.pathname;

    // All nav links
    const links = document.querySelectorAll('.topnav-tabs a, .topnav-drawer a');

    links.forEach(function (a) {
      const href = a.getAttribute('href');
      if (!href) return;

      let active = false;

      if (href === '/' || href === './' || href === '../' ||
          href.endsWith('/index.html')) {
        // Home tab — only active on root
        active = (path === '/' || path.endsWith('/') ||
                  path.endsWith('/index.html'));
      } else if (href.includes('/blocks/') || href === 'blocks/') {
        // Blocks tab — active on any /blocks/ page
        active = path.includes('/blocks/');
      } else if (href.includes('wiki') || href.includes('Wiki')) {
        active = path.includes('wiki');
      }

      a.classList.toggle('active', active);
    });

    // Also highlight the drop button
    const dropBtn = document.querySelector('.topnav-tab-btn[data-tab="blocks"]');
    if (dropBtn) {
      dropBtn.classList.toggle('active', window.location.pathname.includes('/blocks/'));
    }
  }

  /* ── Mobile hamburger ────────────────────────────────────── */
  function initMobileMenu() {
    const burger = document.querySelector('.topnav-burger');
    const drawer = document.querySelector('.topnav-drawer');
    if (!burger || !drawer) return;

    burger.addEventListener('click', function () {
      const isOpen = drawer.classList.toggle('open');
      burger.setAttribute('aria-expanded', String(isOpen));
    });

    // Close on outside click
    document.addEventListener('click', function (e) {
      if (!burger.contains(e.target) && !drawer.contains(e.target)) {
        drawer.classList.remove('open');
        burger.setAttribute('aria-expanded', 'false');
      }
    });

    // Close on link click
    drawer.querySelectorAll('a').forEach(function (a) {
      a.addEventListener('click', function () {
        drawer.classList.remove('open');
        burger.setAttribute('aria-expanded', 'false');
      });
    });
  }

  /* ── Keyboard navigation ─────────────────────────────────── */
  function initKeyboardNav() {
    // Pages order (flat list for ← →)
    var PAGES = [
      '/',
      '/blocks/a-byte-unit.html',
      '/blocks/b-merger.html',
      '/blocks/c-tokenizer.html',
      '/blocks/d-embedder.html',
      '/blocks/e-brain.html',
    ];

    // Normalise current path to base /VRAXION/ prefix
    var path = window.location.pathname;
    // strip trailing index
    path = path.replace(/\/index\.html$/, '/');

    // find current index in pages
    var current = -1;
    for (var i = 0; i < PAGES.length; i++) {
      if (path.endsWith(PAGES[i]) || path === PAGES[i]) {
        current = i;
        break;
      }
    }

    // resolve a page path to a navigable URL preserving the repo base
    function resolve(page) {
      var base = window.location.origin + window.location.pathname;
      // walk up to the site root (above /blocks/ if we're in it)
      base = base.replace(/\/blocks\/[^/]*$/, '/');
      base = base.replace(/\/index\.html$/, '/');
      if (!base.endsWith('/')) base += '/';

      if (page === '/') return base;
      // page is like /blocks/a-byte-unit.html
      return base + page.replace(/^\//, '');
    }

    document.addEventListener('keydown', function (e) {
      // Skip if focus is inside an input
      var tag = document.activeElement && document.activeElement.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      if (e.key === 'h' || e.key === 'H') {
        window.location.href = resolve('/');
        return;
      }

      if (current === -1) return;

      if (e.key === 'ArrowRight' && current < PAGES.length - 1) {
        window.location.href = resolve(PAGES[current + 1]);
      } else if (e.key === 'ArrowLeft' && current > 0) {
        window.location.href = resolve(PAGES[current - 1]);
      }
    });
  }

  /* ── Init ────────────────────────────────────────────────── */
  document.addEventListener('DOMContentLoaded', function () {
    highlightNav();
    initMobileMenu();
    initKeyboardNav();
  });
})();
