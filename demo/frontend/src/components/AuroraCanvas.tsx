import { useEffect, useRef } from "react";
import { useStore } from "../store";

interface CurtainSpec {
  baseColor: [number, number, number];
  alpha: number;
  yBase: number;
  amp: number;
  freq: number;
  speed: number;
  phase: number;
  height: number;
}

const DAYLIGHT_CURTAINS: CurtainSpec[] = [
  { baseColor: [56, 189, 248], alpha: 0.46, yBase: 0.16, amp: 0.10, freq: 1.9, speed: 0.28, phase: 0.2, height: 0.30 },
  { baseColor: [52, 211, 153], alpha: 0.42, yBase: 0.24, amp: 0.11, freq: 1.5, speed: -0.22, phase: 2.0, height: 0.28 },
  { baseColor: [253, 186, 116], alpha: 0.40, yBase: 0.12, amp: 0.09, freq: 2.3, speed: 0.19, phase: 4.1, height: 0.26 },
  { baseColor: [251, 113, 133], alpha: 0.32, yBase: 0.28, amp: 0.10, freq: 1.7, speed: -0.17, phase: 5.2, height: 0.25 },
];

const TEMPO_SPEED: Record<string, number> = {
  very_slow: 0.55,
  slow: 0.78,
  medium: 1.0,
  fast: 1.35,
  very_fast: 1.75,
};

const FRAME_INTERVAL_MS = 33; // ~30 fps cap to stay lightweight behind full-screen CSS blur

function hexToRgb(hex: string): [number, number, number] | null {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!m) return null;
  return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)];
}

/** Lift dark track palette colors into airy daylight pastels so contrast stays crisp. */
function liftToDaylight(rgb: [number, number, number], fallback: [number, number, number]): [number, number, number] {
  const lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
  const mix = lum < 110 ? 0.52 : 0.32;
  const r = Math.round(rgb[0] * (1 - mix) + fallback[0] * mix * 0.55 + 255 * mix * 0.45);
  const g = Math.round(rgb[1] * (1 - mix) + fallback[1] * mix * 0.55 + 255 * mix * 0.45);
  const b = Math.round(rgb[2] * (1 - mix) + fallback[2] * mix * 0.55 + 255 * mix * 0.45);
  return [Math.min(255, r), Math.min(255, g), Math.min(255, b)];
}

interface Ripple {
  x: number;
  y: number;
  r: number;
  maxR: number;
  alpha: number;
  color: [number, number, number];
}

interface AuroraCanvasProps {
  reactive?: boolean;
}

export default function AuroraCanvas({ reactive = false }: AuroraCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    const motionQuery = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    let reducedMotion = motionQuery?.matches ?? false;
    const BUF_W = 240;
    const BUF_H = 160;
    canvas.width = BUF_W;
    canvas.height = BUF_H;

    let rafId = 0;
    let t = 0.8;
    let lastTs = 0;
    let energy = 0.22;
    let burst = 0;

    const currentColors: [number, number, number][] = DAYLIGHT_CURTAINS.map((c) => [...c.baseColor]);
    const ripples: Ripple[] = [];
    const pointer = { x: 0.55, y: 0.22, tx: 0.55, ty: 0.22, active: 0.35, targetActive: 0.35 };

    const addRipple = (nx: number, ny: number, strength = 1) => {
      if (reducedMotion) return;
      const color = currentColors[Math.floor(Math.random() * currentColors.length)];
      ripples.push({
        x: nx * BUF_W,
        y: ny * BUF_H,
        r: 4,
        maxR: BUF_W * 0.45 * strength,
        alpha: 0.44 * strength,
        color: [...color],
      });
      if (ripples.length > 5) ripples.shift();
    };

    const onPointerMove = (e: PointerEvent) => {
      pointer.tx = e.clientX / Math.max(1, window.innerWidth);
      pointer.ty = e.clientY / Math.max(1, window.innerHeight);
      pointer.targetActive = 1;
    };

    const onPointerLeave = () => {
      pointer.targetActive = 0.25;
    };

    const onPointerDown = (e: PointerEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      addRipple(
        e.clientX / Math.max(1, window.innerWidth),
        e.clientY / Math.max(1, window.innerHeight),
        0.9
      );
      burst = Math.min(0.9, burst + 0.25);
    };

    const onVisibilityChange = () => {
      lastTs = 0;
    };

    const drawCurtain = (c: CurtainSpec, color: [number, number, number], time: number) => {
      const steps = 12;
      const dx = BUF_W / steps;
      const px = pointer.x * BUF_W;
      const py = pointer.y * BUF_H;

      const dynAmp = c.amp * BUF_H * (1 + energy * 0.5 + burst * 0.65);
      const bandH = c.height * BUF_H * (reactive ? 1.18 : 0.98) * (1 + energy * 0.22 + burst * 0.18);
      const baseY = (c.yBase + (reactive ? 0.14 : 0)) * BUF_H;
      const [r, g, b] = color;
      const a = c.alpha * (reactive ? 1.12 : 1.0) * (0.78 + energy * 0.34 + burst * 0.26);

      for (let i = -1; i <= steps + 1; i++) {
        const x = i * dx;
        const nx = x / BUF_W;
        const wave1 = Math.sin(nx * Math.PI * c.freq + time * c.speed + c.phase) * dynAmp;
        const wave2 =
          Math.cos(nx * Math.PI * (c.freq * 1.6) - time * c.speed * 1.2 + c.phase) *
          (dynAmp * 0.45);
        const distX = (x - px) / BUF_W;
        const influence = Math.exp(-(distX * distX) / 0.065) * pointer.active;
        const y = baseY + wave1 + wave2 + (py - baseY) * 0.35 * influence;
        const rad = bandH * (0.68 + 0.26 * Math.sin(nx * 5 + time * 0.8 + c.phase));

        const grad = ctx.createRadialGradient(x, y, 0, x, y, rad);
        grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${(a * 0.48).toFixed(3)})`);
        grad.addColorStop(0.55, `rgba(${r}, ${g}, ${b}, ${(a * 0.18).toFixed(3)})`);
        grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(x, y, rad, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    const drawFrame = (dt: number, snapColors = false) => {
      const state = useStore.getState();
      const playing = reactive && state.isPlaying;
      const tempoKey = (state.activeConfig.tempo as string | undefined) ?? "medium";
      const tempoMult = reactive ? (TEMPO_SPEED[tempoKey] ?? 1.0) : 1.0;

      const hasSession = reactive && Boolean(state.activeSessionId);
      const pal = snapColors ? state.targetPalette : state.activePalette;
      const colorIndices = [0, 1, 4, 2];
      const blend = snapColors ? 1 : Math.min(1, dt * 2.5);
      for (let idx = 0; idx < DAYLIGHT_CURTAINS.length; idx++) {
        const fallback = DAYLIGHT_CURTAINS[idx].baseColor;
        const parsed = hasSession ? hexToRgb(pal[colorIndices[idx]] ?? "") : null;
        const target = parsed ? liftToDaylight(parsed, fallback) : fallback;
        currentColors[idx][0] += (target[0] - currentColors[idx][0]) * blend;
        currentColors[idx][1] += (target[1] - currentColors[idx][1]) * blend;
        currentColors[idx][2] += (target[2] - currentColors[idx][2]) * blend;
      }

      const targetEnergy = playing ? 0.85 : hasSession ? 0.42 : 0.22;
      pointer.x += (pointer.tx - pointer.x) * Math.min(1, dt * 5.5);
      pointer.y += (pointer.ty - pointer.y) * Math.min(1, dt * 5.5);
      pointer.active += (pointer.targetActive - pointer.active) * Math.min(1, dt * 3.5);
      energy += (targetEnergy - energy) * (snapColors ? 1 : Math.min(1, dt * 3.0));
      burst *= Math.pow(0.12, dt);

      t += dt * (playing ? 1.55 * tempoMult : 0.95) + burst * dt * 1.4;

      ctx.clearRect(0, 0, BUF_W, BUF_H);
      for (let i = 0; i < DAYLIGHT_CURTAINS.length; i++) {
        const c: [number, number, number] = [
          Math.round(currentColors[i][0]),
          Math.round(currentColors[i][1]),
          Math.round(currentColors[i][2]),
        ];
        drawCurtain(DAYLIGHT_CURTAINS[i], c, t);
      }

      if (pointer.active >= 0.05) {
        const x = pointer.x * BUF_W;
        const y = pointer.y * BUF_H;
        const rad = BUF_W * (0.24 + burst * 0.08);
        const a = 0.22 * pointer.active * (1 + burst * 0.45);
        const grad = ctx.createRadialGradient(x, y, 0, x, y, rad);
        grad.addColorStop(0, `rgba(56, 189, 248, ${a.toFixed(3)})`);
        grad.addColorStop(0.45, `rgba(52, 211, 153, ${(a * 0.65).toFixed(3)})`);
        grad.addColorStop(0.75, `rgba(253, 186, 116, ${(a * 0.35).toFixed(3)})`);
        grad.addColorStop(1, "rgba(255, 255, 255, 0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(x, y, rad, 0, Math.PI * 2);
        ctx.fill();
      }

      for (let i = ripples.length - 1; i >= 0; i--) {
        const rp = ripples[i];
        rp.r += (rp.maxR - rp.r) * Math.min(1, dt * 3.2) + dt * 16;
        rp.alpha *= Math.pow(0.22, dt);
        if (rp.alpha < 0.01 || rp.r >= rp.maxR) {
          ripples.splice(i, 1);
          continue;
        }
        const [r, g, b] = rp.color;
        const grad = ctx.createRadialGradient(rp.x, rp.y, rp.r * 0.35, rp.x, rp.y, rp.r);
        grad.addColorStop(0, `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, 0)`);
        grad.addColorStop(0.65, `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${rp.alpha.toFixed(3)})`);
        grad.addColorStop(1, `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, 0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(rp.x, rp.y, rp.r, 0, Math.PI * 2);
        ctx.fill();
      }

      const veilStart = reactive ? 0.54 : 0.34;
      const veil = ctx.createLinearGradient(0, BUF_H * veilStart, 0, BUF_H);
      veil.addColorStop(0, "rgba(255, 255, 255, 0)");
      veil.addColorStop(0.64, reactive ? "rgba(255, 255, 255, 0.58)" : "rgba(255, 255, 255, 0.76)");
      veil.addColorStop(1, reactive ? "rgba(255, 255, 255, 0.92)" : "rgba(255, 255, 255, 0.97)");
      ctx.fillStyle = veil;
      ctx.fillRect(0, 0, BUF_W, BUF_H);
    };

    const loop = (ts: number) => {
      if (reducedMotion) return;
      if (!lastTs) lastTs = ts;
      const elapsedMs = ts - lastTs;
      if (elapsedMs >= FRAME_INTERVAL_MS) {
        const dt = Math.min(0.06, elapsedMs / 1000);
        lastTs = ts;
        if (!document.hidden) drawFrame(dt, false);
      }
      rafId = requestAnimationFrame(loop);
    };

    const onMotionChange = (e: MediaQueryListEvent) => {
      reducedMotion = e.matches;
      if (reducedMotion) {
        if (rafId) cancelAnimationFrame(rafId);
        rafId = 0;
        drawFrame(0, true);
      } else {
        lastTs = 0;
        rafId = requestAnimationFrame(loop);
      }
    };

    // Update static frame whenever palette changes while reduced-motion is enabled.
    const unsubStore = useStore.subscribe((next, prev) => {
      if (reducedMotion && (next.targetPalette !== prev.targetPalette || next.activeSessionId !== prev.activeSessionId)) {
        drawFrame(0, true);
      }
    });

    window.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("pointerleave", onPointerLeave, { passive: true });
    window.addEventListener("pointerdown", onPointerDown, { passive: true });
    document.addEventListener("visibilitychange", onVisibilityChange);
    motionQuery?.addEventListener("change", onMotionChange);

    drawFrame(0, true);
    if (!reducedMotion) rafId = requestAnimationFrame(loop);

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      unsubStore();
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerleave", onPointerLeave);
      window.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      motionQuery?.removeEventListener("change", onMotionChange);
    };
  }, [reactive]);

  return (
    <>
      <canvas ref={canvasRef} className="aurora-canvas" aria-hidden="true" />
      <div className="film-grain" aria-hidden="true" />
    </>
  );
}
