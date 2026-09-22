import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { BlackHoleHeroSection } from "@/components/ui/blackhole-hero-section";
import { HeroParticles } from "./hero-particles";
import { startPageParticles } from "./page-particles";
import "./index.css";

function HeroBlackHole() {
  const [narrow, setNarrow] = useState(false);
  const [low, setLow] = useState(false);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 767px)");
    const sync = () => setNarrow(media.matches);
    sync();
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  useEffect(() => {
    let frames = 0;
    let start = performance.now();
    let raf = 0;
    const probe = (now: number) => {
      frames += 1;
      if (now - start < 1800) {
        raf = requestAnimationFrame(probe);
        return;
      }
      const fps = frames / ((now - start) / 1000);
      if (fps < 32) setLow(true);
    };
    raf = requestAnimationFrame(probe);
    return () => cancelAnimationFrame(raf);
  }, []);

  const focus: [number, number] = narrow ? [0.5, 0.76] : [0.72, 0.46];

  return (
    <div className="relative h-full w-full">
      <BlackHoleHeroSection
        className="absolute inset-0 h-full w-full bg-transparent"
        focus={focus}
        scrim={narrow ? "top" : "left"}
        scrimStrength={0.88}
        distance={24}
        elevation={narrow ? -7 : -5.5}
        roll={-20}
        fov={narrow ? 58 : 42}
        glow={low ? 0.35 : 0.55}
        vignette={0.34}
        exposure={0.92}
        steps={low ? 90 : narrow ? 130 : 160}
        resolution={low ? 0.36 : narrow ? 0.42 : 0.48}
        maxDpr={1}
        starBrightness={0}
        hotColor="#F7FEE7"
        midColor="#4ADE80"
        coolColor="#14532D"
        doppler={0.28}
        spinSpeed={0.05}
        aria-hidden="true"
      />
      <HeroParticles focus={focus} count={narrow ? 42 : 72} />
    </div>
  );
}

const bg = document.getElementById("page-bg") as HTMLCanvasElement | null;
if (bg) startPageParticles(bg);

const host = document.getElementById("hero-art");
if (host) {
  createRoot(host).render(
    <StrictMode>
      <HeroBlackHole />
    </StrictMode>
  );
}
