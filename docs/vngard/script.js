(() => {
  const canvas = document.getElementById("fieldCanvas");
  const ctx = canvas.getContext("2d");
  const particles = [];
  const particleCount = 42;
  let width = 0;
  let height = 0;
  let pixelRatio = 1;

  function resize() {
    pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * pixelRatio);
    canvas.height = Math.floor(height * pixelRatio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  }

  function seedParticles() {
    particles.length = 0;
    for (let i = 0; i < particleCount; i += 1) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.16,
        vy: (Math.random() - 0.5) * 0.16,
        r: 1 + Math.random() * 1.8,
        phase: Math.random() * Math.PI * 2,
        seal: Math.random() > 0.92
      });
    }
  }

  function step() {
    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 1;

    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      p.phase += 0.012;

      if (p.x < -20) p.x = width + 20;
      if (p.x > width + 20) p.x = -20;
      if (p.y < -20) p.y = height + 20;
      if (p.y > height + 20) p.y = -20;
    }

    for (let i = 0; i < particles.length; i += 1) {
      for (let j = i + 1; j < particles.length; j += 1) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d = Math.hypot(dx, dy);
        if (d < 150) {
          const alpha = (1 - d / 150) * 0.16;
          ctx.strokeStyle = `rgba(0, 255, 183, ${alpha})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    for (const p of particles) {
      const pulse = 0.55 + Math.sin(p.phase) * 0.25;
      ctx.fillStyle = p.seal
        ? `rgba(255, 0, 72, ${0.18 + pulse * 0.12})`
        : `rgba(0, 255, 183, ${0.18 + pulse * 0.16})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }

    requestAnimationFrame(step);
  }

  window.addEventListener("resize", () => {
    resize();
    seedParticles();
  });

  resize();
  seedParticles();
  step();
})();
