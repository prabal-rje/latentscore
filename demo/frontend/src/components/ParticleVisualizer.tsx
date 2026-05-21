import { useRef, useEffect } from "react";
import { useStore } from "../store";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  colorIdx: number;
  alpha: number;
  life: number;
  maxLife: number;
  type: "float" | "burst" | "orbit" | "trail";
}

function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
    : [100, 90, 75];
}

const TEMPO_MAP: Record<string, number> = {
  very_slow: 0.15, slow: 0.3, medium: 0.5, fast: 0.8, very_fast: 1.2,
};
const MOTION_MAP: Record<string, number> = {
  static: 0.05, slow: 0.2, medium: 0.5, fast: 0.8, chaotic: 1.5,
};

export default function ParticleVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animationRef = useRef<number>(0);
  const timeRef = useRef(0);
  const mouseRef = useRef({ x: -1000, y: -1000, active: false });

  // Single useEffect — runs once, reads state from store each frame
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY, active: true };
    };
    const handleMouseLeave = () => {
      mouseRef.current = { ...mouseRef.current, active: false };
    };
    window.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseleave", handleMouseLeave);

    const spawnParticle = (
      type: Particle["type"],
      w: number,
      h: number,
      motion: number,
    ): Particle => {
      const colorIdx = Math.floor(Math.random() * 3);
      const cx = w / 2;
      const cy = h / 2;

      // Read playing state for intensity-aware spawning
      const isActive = useStore.getState().isPlaying;

      switch (type) {
        case "burst": {
          const angle = Math.random() * Math.PI * 2;
          const speed = (1.0 + Math.random() * 3) * motion;
          return {
            x: cx + (Math.random() - 0.5) * 200,
            y: cy + (Math.random() - 0.5) * 200,
            vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
            size: isActive ? 2.5 + Math.random() * 5 : 0.8 + Math.random() * 2.2, colorIdx,
            alpha: isActive ? 0.7 + Math.random() * 0.3 : 0.15 + Math.random() * 0.1,
            life: 0, maxLife: 100 + Math.random() * 200, type: "burst",
          };
        }
        case "orbit": {
          const angle = Math.random() * Math.PI * 2;
          const dist = 200 + Math.random() * 350;
          return {
            x: cx + Math.cos(angle) * dist, y: cy + Math.sin(angle) * dist,
            vx: angle, vy: dist,
            size: isActive ? 2 + Math.random() * 4 : 0.8 + Math.random() * 1.5, colorIdx,
            alpha: isActive ? 0.5 + Math.random() * 0.4 : 0.08 + Math.random() * 0.1,
            life: 0, maxLife: 200 + Math.random() * 300, type: "orbit",
          };
        }
        case "trail":
          return {
            x: Math.random() * w, y: h + 10,
            vx: (Math.random() - 0.5) * 0.5,
            vy: -(0.3 + Math.random() * 0.7) * motion,
            size: isActive ? 1.5 + Math.random() * 3.5 : 0.4 + Math.random() * 1.2, colorIdx,
            alpha: isActive ? 0.45 + Math.random() * 0.35 : 0.06 + Math.random() * 0.1,
            life: 0, maxLife: 200 + Math.random() * 400, type: "trail",
          };
        default:
          return {
            x: Math.random() * w, y: Math.random() * h,
            vx: (Math.random() - 0.5) * 0.3 * motion,
            vy: (Math.random() - 0.5) * 0.3 * motion,
            size: isActive ? 1.5 + Math.random() * 4 : 0.5 + Math.random() * 1.5, colorIdx,
            alpha: isActive ? 0.35 + Math.random() * 0.4 : 0.04 + Math.random() * 0.1,
            life: 0, maxLife: 300 + Math.random() * 500, type: "float",
          };
      }
    };

    const animate = () => {
      // Read all state fresh each frame — no stale closures
      const state = useStore.getState();
      const palette = state.activePalette;
      const playing = state.isPlaying;
      const config = state.activeConfig;

      const speed = TEMPO_MAP[(config.tempo as string) ?? "medium"] ?? 0.5;
      const density = ((config.density as number) ?? 4) * 15;
      const motion = MOTION_MAP[(config.motion as string) ?? "medium"] ?? 0.5;

      timeRef.current += 1;
      const t = timeRef.current;
      const particles = particlesRef.current;
      const mouse = mouseRef.current;
      const w = canvas.width;
      const h = canvas.height;

      // Background — darken palette bg
      const rawBg = palette[3] ?? "#1a1816";
      // Darken the palette background by blending toward black
      const [bgR, bgG, bgB] = hexToRgb(rawBg);
      const darkFactor = playing ? 0.3 : 0.55;
      ctx.fillStyle = `rgb(${Math.round(bgR * darkFactor)},${Math.round(bgG * darkFactor)},${Math.round(bgB * darkFactor)})`;
      ctx.fillRect(0, 0, w, h);

      // Subtle radial glow — positioned at edges, not behind the card
      const glow = palette[4] ?? "#8a7d6b";
      const [gr, gg, gb] = hexToRgb(glow);
      const gradient = ctx.createRadialGradient(
        w * 0.15, h * 0.35, 0,
        w * 0.15, h * 0.35, w * 0.6,
      );
      gradient.addColorStop(0, `rgba(${gr},${gg},${gb},${playing ? 0.18 : 0.015})`);
      gradient.addColorStop(1, "transparent");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, w, h);

      // Second glow from opposite corner when playing
      if (playing) {
        const c2 = palette[0] ?? "#8a7d6b";
        const [r2, g2, b2] = hexToRgb(c2);
        const grad2 = ctx.createRadialGradient(
          w * 0.85, h * 0.65, 0, w * 0.85, h * 0.65, w * 0.55,
        );
        grad2.addColorStop(0, `rgba(${r2},${g2},${b2},0.12)`);
        grad2.addColorStop(1, "transparent");
        ctx.fillStyle = grad2;
        ctx.fillRect(0, 0, w, h);

        // Bottom glow
        const c3 = palette[2] ?? "#8a7d6b";
        const [r3, g3, b3] = hexToRgb(c3);
        const grad3 = ctx.createRadialGradient(
          w * 0.5, h * 0.9, 0, w * 0.5, h * 0.9, w * 0.5,
        );
        grad3.addColorStop(0, `rgba(${r3},${g3},${b3},0.08)`);
        grad3.addColorStop(1, "transparent");
        ctx.fillStyle = grad3;
        ctx.fillRect(0, 0, w, h);
      }

      // Mouse glow
      if (mouse.active) {
        const mouseGlow = ctx.createRadialGradient(
          mouse.x, mouse.y, 0, mouse.x, mouse.y, playing ? 250 : 140,
        );
        mouseGlow.addColorStop(0, `rgba(${gr},${gg},${gb},${playing ? 0.18 : 0.04})`);
        mouseGlow.addColorStop(1, "transparent");
        ctx.fillStyle = mouseGlow;
        ctx.fillRect(0, 0, w, h);
      }

      // Spawn particles — density scales with screen area
      const screenArea = w * h;
      const baseCount = Math.min(100, Math.max(18, Math.floor(screenArea / 18000)));
      const targetCount = playing ? Math.min(baseCount * 3, density * 1.5) : baseCount;
      while (particles.length < targetCount) {
        if (playing) {
          const r = Math.random();
          const type: Particle["type"] =
            r < 0.3 ? "burst" : r < 0.5 ? "orbit" : r < 0.7 ? "trail" : "float";
          particles.push(spawnParticle(type, w, h, motion));
        } else {
          particles.push(spawnParticle("float", w, h, motion));
        }
      }

      // Periodic bursts when playing — explosive color bursts
      if (playing && t % Math.max(8, Math.floor(50 / speed)) === 0) {
        for (let i = 0; i < 12 + Math.floor(speed * 16); i++) {
          particles.push(spawnParticle("burst", w, h, motion));
        }
      }

      const cx = w / 2;
      const cy = h / 2;
      const MOUSE_RADIUS = 200;
      const MOUSE_STRENGTH = 1.2;
      // Center repulsion — push particles away from the card area
      const CARD_REPEL_X = 280; // half card width + margin
      const CARD_REPEL_Y = 320; // half card height + margin
      const REPEL_STRENGTH = playing ? 0.8 : 0.3;

      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.life += 1;
        if (p.life > p.maxLife) { particles.splice(i, 1); continue; }

        const lifeRatio = p.life / p.maxLife;
        const fadeIn = Math.min(1, p.life / 20);
        const fadeOut = lifeRatio > 0.7 ? 1 - (lifeRatio - 0.7) / 0.3 : 1;
        const currentAlpha = p.alpha * fadeIn * fadeOut;

        // Center repulsion — push away from the card
        const dcx = p.x - cx;
        const dcy = p.y - cy;
        const normX = Math.abs(dcx) / CARD_REPEL_X;
        const normY = Math.abs(dcy) / CARD_REPEL_Y;
        if (normX < 1 && normY < 1) {
          const overlap = (1 - Math.max(normX, normY));
          const repelForce = overlap * REPEL_STRENGTH;
          const dist = Math.sqrt(dcx * dcx + dcy * dcy);
          if (dist > 1) {
            p.vx += (dcx / dist) * repelForce;
            p.vy += (dcy / dist) * repelForce;
          }
        }

        // Mouse attraction
        if (mouse.active) {
          const dx = mouse.x - p.x;
          const dy = mouse.y - p.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < MOUSE_RADIUS && dist > 1) {
            const force = (1 - dist / MOUSE_RADIUS) * MOUSE_STRENGTH;
            p.vx += (dx / dist) * force * 0.4;
            p.vy += (dy / dist) * force * 0.4;
            p.vx *= 0.94;
            p.vy *= 0.94;
          }
        }

        // Physics
        switch (p.type) {
          case "orbit": {
            const angle = p.vx + t * 0.005 * speed * motion;
            const wobble = Math.sin(t * 0.02 + p.vy) * 10;
            p.x = cx + Math.cos(angle) * (p.vy + wobble);
            p.y = cy + Math.sin(angle) * (p.vy + wobble) * 0.6;
            break;
          }
          case "burst":
            p.vx *= 0.985; p.vy *= 0.985;
            p.x += p.vx * speed; p.y += p.vy * speed;
            break;
          case "trail":
            p.x += p.vx + Math.sin(t * 0.01 + p.x * 0.01) * 0.3 * motion;
            p.y += p.vy * speed;
            break;
          default:
            p.x += p.vx + Math.sin(t * 0.008 + p.y * 0.005) * 0.2 * motion;
            p.y += p.vy + Math.cos(t * 0.006 + p.x * 0.005) * 0.2 * motion;
        }

        // Wrap
        if (p.x < -10) p.x = w + 10;
        if (p.x > w + 10) p.x = -10;
        if (p.y < -10 && p.type !== "trail") p.y = h + 10;

        // Render — use live palette color for the particle's index
        const color = palette[p.colorIdx] ?? palette[0] ?? "#8a7d6b";
        const [r, g, b] = hexToRgb(color);
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r},${g},${b},${currentAlpha})`;
        ctx.fill();

        // Glow halos — big colorful auras when playing
        if (playing && p.size > 1.5 && currentAlpha > 0.1) {
          // Outer soft glow
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * 6, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${r},${g},${b},${currentAlpha * 0.08})`;
          ctx.fill();
          // Inner bright halo
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * 3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${r},${g},${b},${currentAlpha * 0.25})`;
          ctx.fill();
        } else if (!playing && p.size > 1 && currentAlpha > 0.08) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * 3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${r},${g},${b},${currentAlpha * 0.06})`;
          ctx.fill();
        }
      }

      // Connection lines — vivid colored web when playing
      if (playing && particles.length > 0) {
        const maxDist = 160;
        const subset = particles.slice(0, Math.min(particles.length, 60));
        for (let i = 0; i < subset.length; i++) {
          for (let j = i + 1; j < subset.length; j++) {
            const dx = subset[i].x - subset[j].x;
            const dy = subset[i].y - subset[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < maxDist) {
              const alpha = (1 - dist / maxDist) * 0.25;
              const c = palette[subset[i].colorIdx] ?? palette[0] ?? "#8a7d6b";
              const [r, g, b] = hexToRgb(c);
              ctx.beginPath();
              ctx.moveTo(subset[i].x, subset[i].y);
              ctx.lineTo(subset[j].x, subset[j].y);
              ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
              ctx.lineWidth = 1;
              ctx.stroke();
            }
          }
        }
      }

      // Mouse connection lines — vivid interactive tendrils
      if (mouse.active && particles.length > 0) {
        const mouseRange = playing ? 220 : 120;
        const nearby = particles
          .filter((p) => {
            const dx = p.x - mouse.x, dy = p.y - mouse.y;
            return Math.sqrt(dx * dx + dy * dy) < mouseRange;
          })
          .slice(0, playing ? 14 : 6);
        for (const p of nearby) {
          const dx = p.x - mouse.x, dy = p.y - mouse.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const alpha = (1 - dist / mouseRange) * (playing ? 0.4 : 0.1);
          const c = palette[p.colorIdx] ?? palette[0] ?? "#8a7d6b";
          const [r, g, b] = hexToRgb(c);
          ctx.beginPath();
          ctx.moveTo(mouse.x, mouse.y);
          ctx.lineTo(p.x, p.y);
          ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
          ctx.lineWidth = playing ? 1.2 : 0.4;
          ctx.stroke();
        }
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseleave", handleMouseLeave);
      cancelAnimationFrame(animationRef.current);
    };
  }, []); // Empty deps — runs once, reads store each frame

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        top: 0, left: 0,
        width: "100%", height: "100%",
        zIndex: 0,
      }}
    />
  );
}
