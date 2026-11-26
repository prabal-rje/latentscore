import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

interface CharmItem {
  src: string;
  term: string;
  alt: string;
}

const CHARM_ITEMS: CharmItem[] = [
  { src: "/art/charm-vinyl.png", term: "resonance", alt: "Iridescent pixel vinyl disc" },
  { src: "/art/charm-note.png", term: "arpeggio", alt: "Pixel music note" },
  { src: "/art/charm-clef.png", term: "counterpoint", alt: "Pixel treble clef" },
  { src: "/art/charm-wave.png", term: "harmonics", alt: "Prismatic pixel soundwave" },
  { src: "/art/charm-synth.png", term: "voicing", alt: "Pixel synthesizer keyboard" },
  { src: "/art/charm-cassette.png", term: "ostinato", alt: "Pastel pixel cassette tape" },
  { src: "/art/charm-bell.png", term: "overtone", alt: "Sunlit pixel chime bell" },
  { src: "/art/charm-prism.png", term: "timbre", alt: "Holographic pixel prism" },
];

const SPARKLE_COLORS = ["#0284C7", "#0D9488", "#EA580C", "#E11D48", "#EAB308"];

interface Sparkle {
  id: number;
  x: number;
  y: number;
  dx: string;
  dy: string;
  color: string;
}

interface PixelCharmProps {
  initialIndex?: number;
  size?: number;
  isPlaying?: boolean;
  autoCycleMs?: number;
}

let nextSparkleId = 1;

export default function PixelCharm({
  initialIndex = 0,
  size = 58,
  isPlaying = false,
  autoCycleMs = 6500,
}: PixelCharmProps) {
  const [index, setIndex] = useState(initialIndex % CHARM_ITEMS.length);
  const [popping, setPopping] = useState(false);
  const [sparkles, setSparkles] = useState<Sparkle[]>([]);
  const popTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sparkleTimersRef = useRef<Set<ReturnType<typeof setTimeout>>>(new Set());

  useEffect(() => {
    const timers = sparkleTimersRef.current;
    return () => {
      if (popTimerRef.current !== null) clearTimeout(popTimerRef.current);
      for (const timer of timers) clearTimeout(timer);
      timers.clear();
    };
  }, []);

  const stepCharm = useCallback(() => {
    setIndex((prev) => (prev + 1) % CHARM_ITEMS.length);
    setPopping(true);
    if (popTimerRef.current !== null) clearTimeout(popTimerRef.current);
    popTimerRef.current = setTimeout(() => {
      setPopping(false);
      popTimerRef.current = null;
    }, 420);
  }, []);

  useEffect(() => {
    if (autoCycleMs <= 0) return;
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    let intervalId: ReturnType<typeof setInterval> | null = null;

    const start = () => {
      if (intervalId !== null) return;
      intervalId = setInterval(() => {
        if (!document.hidden) stepCharm();
      }, autoCycleMs);
    };
    const stop = () => {
      if (intervalId !== null) {
        clearInterval(intervalId);
        intervalId = null;
      }
    };

    if (!mq?.matches) start();
    const onChange = (e: MediaQueryListEvent) => {
      if (e.matches) stop();
      else start();
    };
    mq?.addEventListener("change", onChange);
    return () => {
      stop();
      mq?.removeEventListener("change", onChange);
    };
  }, [autoCycleMs, stepCharm]);

  const handleCycle = useCallback(
    (e: React.MouseEvent<HTMLButtonElement>) => {
      stepCharm();
      const rect = e.currentTarget.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const created: Sparkle[] = Array.from({ length: 8 }, (_, i) => {
        const angle = (Math.PI * 2 * i) / 8 + (Math.random() - 0.5) * 0.35;
        const dist = 28 + Math.random() * 26;
        return {
          id: nextSparkleId++,
          x: cx,
          y: cy,
          dx: `${Math.round(Math.cos(angle) * dist)}px`,
          dy: `${Math.round(Math.sin(angle) * dist)}px`,
          color: SPARKLE_COLORS[i % SPARKLE_COLORS.length],
        };
      });
      const ids = new Set(created.map((s) => s.id));
      setSparkles((prev) => [...prev, ...created]);
      const timer = setTimeout(() => {
        sparkleTimersRef.current.delete(timer);
        setSparkles((prev) => prev.filter((s) => !ids.has(s.id)));
      }, 660);
      sparkleTimersRef.current.add(timer);
    },
    [stepCharm]
  );

  const charm = CHARM_ITEMS[index];

  return (
    <>
      <button
        type="button"
        className="pixel-charm-btn"
        onClick={handleCycle}
        title={`Tap to switch (${charm.term})`}
        aria-label={`Cycle musical charm (${charm.term})`}
      >
        <img
          src={charm.src}
          width={size}
          height={size}
          alt={charm.alt}
          className={`pixel-charm-img${popping ? " is-popping" : ""}${isPlaying ? " is-playing" : ""}`}
        />
      </button>

      {sparkles.length > 0 &&
        typeof document !== "undefined" &&
        createPortal(
          <div className="sparkle-layer" aria-hidden="true">
            {sparkles.map((s) => (
              <span
                key={s.id}
                className="px-sparkle"
                style={
                  {
                    left: s.x,
                    top: s.y,
                    backgroundColor: s.color,
                    "--dx": s.dx,
                    "--dy": s.dy,
                  } as React.CSSProperties
                }
              />
            ))}
          </div>,
          document.body
        )}
    </>
  );
}
