import { useEffect, useRef } from "react";

type Particle = {
  a: number;
  r: number;
  s: number;
  size: number;
  alpha: number;
  tilt: number;
  drift: number;
  green: boolean;
};

export function HeroParticles({
  focus,
  count,
}: {
  focus: [number, number];
  count: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return;

    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    const particles: Particle[] = [];
    let raf = 0;
    let visible = true;
    let width = 0;
    let height = 0;

    const seed = () => {
      particles.length = 0;
      for (let i = 0; i < count; i += 1) {
        particles.push({
          a: Math.random() * Math.PI * 2,
          r: 28 + Math.random() * 210,
          s: 0.0012 + Math.random() * 0.0045,
          size: 0.5 + Math.random() * 1.8,
          alpha: 0.14 + Math.random() * 0.5,
          tilt: 0.32 + Math.random() * 0.22,
          drift: (Math.random() - 0.5) * 0.08,
          green: Math.random() > 0.32,
        });
      }
    };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      width = Math.max(1, Math.round(rect.width));
      height = Math.max(1, Math.round(rect.height));
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = () => {
      raf = requestAnimationFrame(draw);
      if (!visible || document.hidden) return;
      ctx.clearRect(0, 0, width, height);
      const cx = width * focus[0];
      const cy = height * focus[1];
      for (const p of particles) {
        p.a += p.s;
        p.r += p.drift;
        if (p.r > 240) p.r = 30;
        if (p.r < 24) p.r = 230;
        const x = cx + Math.cos(p.a) * p.r;
        const y = cy + Math.sin(p.a) * p.r * p.tilt;
        const pulse = 0.55 + 0.45 * Math.sin(p.a * 3);
        ctx.beginPath();
        ctx.fillStyle = p.green
          ? `rgba(74, 222, 128, ${p.alpha * pulse})`
          : `rgba(186, 190, 196, ${p.alpha * pulse * 0.7})`;
        ctx.arc(x, y, p.size, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    seed();
    resize();
    draw();

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    const io = new IntersectionObserver(
      (entries) => {
        visible = entries[0]?.isIntersecting ?? true;
      },
      { threshold: 0 }
    );
    io.observe(canvas);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
    };
  }, [count, focus]);

  return (
    <canvas
      ref={canvasRef}
      className="hero-particles"
      aria-hidden="true"
    />
  );
}
