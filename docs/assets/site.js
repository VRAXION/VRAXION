(() => {
  const body = document.body;
  const root = body.dataset.root || ".";
  document.querySelectorAll("[data-site-year]").forEach((node) => {
    node.textContent = String(new Date().getFullYear());
  });

  const targets = document.querySelectorAll("[data-version-field]");
  if (!targets.length) return;

  fetch(`${root}/VERSION.json`)
    .then((res) => (res.ok ? res.json() : null))
    .then((data) => {
      if (!data) return;
      targets.forEach((node) => {
        const key = node.getAttribute("data-version-field");
        if (key && Object.prototype.hasOwnProperty.call(data, key)) {
          node.textContent = String(data[key]);
        }
      });
    })
    .catch(() => {});
})();
