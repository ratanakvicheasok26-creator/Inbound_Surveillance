type Particle = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  alpha: number;
  twinkle: number;
  phase: number;
  green: boolean;
};

export function startPageParticles(canvas: HTMLCanvasElement) {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) return;

  const particles: Particle[] = [];
  let width = 0;
  let height = 0;
  let raf = 0;
  let running = true;

  const count = () => (width < 700 ? 56 : 96);

  const make = (): Particle => ({
    x: Math.random() * width,
    y: Math.random() * height,
    vx: (Math.random() - 0.5) * 0.22,
    vy: -0.08 - Math.random() * 0.22,
    size: 0.6 + Math.random() * 1.8,
    alpha: 0.18 + Math.random() * 0.45,
    twinkle: 0.008 + Math.random() * 0.018,
    phase: Math.random() * Math.PI * 2,
    green: Math.random() > 0.38,
  });

  const seed = () => {
    particles.length = 0;
    const n = count();
    for (let i = 0; i < n; i += 1) particles.push(make());
  };

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const cssW = Math.max(1, window.innerWidth);
    const cssH = Math.max(1, window.innerHeight);
    const nextW = cssW;
    const nextH = cssH;
    const changed = nextW !== width || nextH !== height;
    width = nextW;
    height = nextH;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (changed) seed();
  };

  const wrap = (p: Particle) => {
    if (p.x < -8) p.x = width + 8;
    if (p.x > width + 8) p.x = -8;
    if (p.y < -8) p.y = height + 8;
    if (p.y > height + 8) p.y = -8;
  };

  const draw = () => {
    if (!running) return;
    raf = requestAnimationFrame(draw);
    if (document.hidden) return;
    ctx.clearRect(0, 0, width, height);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      p.phase += p.twinkle;
      wrap(p);
      const pulse = 0.55 + 0.45 * Math.sin(p.phase);
      ctx.beginPath();
      ctx.fillStyle = p.green
        ? `rgba(74, 222, 128, ${p.alpha * pulse})`
        : `rgba(186, 190, 196, ${p.alpha * pulse * 0.7})`;
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const onVisibility = () => {
    if (document.hidden) {
      running = false;
      cancelAnimationFrame(raf);
      return;
    }
    running = true;
    raf = requestAnimationFrame(draw);
  };

  resize();
  raf = requestAnimationFrame(draw);
  window.addEventListener("resize", resize);
  document.addEventListener("visibilitychange", onVisibility);
}
