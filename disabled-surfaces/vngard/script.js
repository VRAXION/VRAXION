(() => {
  const canvas = document.getElementById('fieldCanvas');
  const reduceMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

  if (!canvas || reduceMotionQuery.matches) {
    if (canvas) {
      canvas.setAttribute('hidden', '');
    }
    return;
  }

  const ctx = canvas.getContext('2d');

  if (!ctx) {
    return;
  }

  const particles = [];
  const particleCount = 42;
  let width = 0;
  let height = 0;
  let pixelRatio = 1;
  let animationId = 0;
  let running = false;

  function resize() {
    pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;

    canvas.width = Math.floor(width * pixelRatio);
    canvas.height = Math.floor(height * pixelRatio);

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

  function drawConnections() {
    for (let i = 0; i < particles.length; i += 1) {
      for (let j = i + 1; j < particles.length; j += 1) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d = Math.hypot(dx, dy);

        if (d < 150) {
          const alpha = (1 - d / 150) * 0.16;
          ctx.strokeStyle = `rgba(0, 229, 255, ${alpha})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
  }

  function drawParticles() {
    for (const p of particles) {
      const pulse = 0.55 + Math.sin(p.phase) * 0.25;
      ctx.fillStyle = p.seal
        ? `rgba(255, 0, 72, ${0.18 + pulse * 0.12})`
        : `rgba(0, 229, 255, ${0.18 + pulse * 0.16})`;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function step() {
    if (!running) {
      return;
    }

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

    drawConnections();
    drawParticles();

    animationId = requestAnimationFrame(step);
  }

  function start() {
    if (running || document.hidden || reduceMotionQuery.matches) {
      return;
    }

    running = true;
    animationId = requestAnimationFrame(step);
  }

  function stop() {
    running = false;

    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = 0;
    }
  }

  window.addEventListener('resize', () => {
    resize();
    seedParticles();
  });

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stop();
    } else {
      start();
    }
  });

  if (typeof reduceMotionQuery.addEventListener === 'function') {
    reduceMotionQuery.addEventListener('change', event => {
      if (event.matches) {
        stop();
        canvas.setAttribute('hidden', '');
      } else {
        canvas.removeAttribute('hidden');
        resize();
        seedParticles();
        start();
      }
    });
  }

  resize();
  seedParticles();
  start();
})();
