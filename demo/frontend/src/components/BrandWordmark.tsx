import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

interface MusicalMark {
  src: string;
  term: string;
  alt: string;
}

const MUSICAL_MARKS: MusicalMark[] = [
  { src: "/art/charm-vinyl.png", term: "resonance", alt: "Pixel vinyl disc" },
  { src: "/art/charm-note.png", term: "arpeggio", alt: "Pixel eighth note" },
  { src: "/art/charm-wave.png", term: "harmonics", alt: "Pixel soundwave" },
  { src: "/art/charm-synth.png", term: "voicing", alt: "Pixel synthesizer keyboard" },
  { src: "/art/charm-cassette.png", term: "ostinato", alt: "Pixel cassette tape" },
  { src: "/art/charm-bell.png", term: "overtone", alt: "Pixel chime bell" },
  { src: "/art/charm-clef.png", term: "counterpoint", alt: "Pixel treble clef" },
  { src: "/art/charm-prism.png", term: "timbre", alt: "Pixel prism" },
];

interface BrandWordmarkProps {
  isPlaying?: boolean;
}

export default function BrandWordmark({ isPlaying = false }: BrandWordmarkProps) {
  const [idx, setIdx] = useState(0);
  const [swapping, setSwapping] = useState(false);
  const popTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const triggerSwap = useCallback(() => {
    setIdx((prev) => (prev + 1) % MUSICAL_MARKS.length);
    setSwapping(true);
    if (popTimerRef.current !== null) clearTimeout(popTimerRef.current);
    popTimerRef.current = setTimeout(() => {
      setSwapping(false);
      popTimerRef.current = null;
    }, 380);
  }, []);

  useEffect(() => {
    return () => {
      if (popTimerRef.current !== null) clearTimeout(popTimerRef.current);
    };
  }, []);

  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    let intervalId: ReturnType<typeof setInterval> | null = null;

    const start = () => {
      if (intervalId !== null) return;
      intervalId = setInterval(() => {
        if (!document.hidden) triggerSwap();
      }, 4500);
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
  }, [triggerSwap]);

  const mark = MUSICAL_MARKS[idx];

  return (
    <div className="wordmark-group">
      <button
        type="button"
        className={`brand-badge${swapping ? " is-swapping" : ""}${isPlaying ? " is-playing" : ""}`}
        onClick={triggerSwap}
        title={`Musical motif: ${mark.term} (tap to cycle)`}
        aria-label={`Musical icon: ${mark.term}. Click to switch.`}
      >
        <img
          src={mark.src}
          width={28}
          height={28}
          alt={mark.alt}
          className="brand-badge-pixel"
        />
      </button>

      <Link to="/" className="wordmark" aria-label="LatentScore home">
        <span>
          Latent<em>Score</em>
        </span>
      </Link>
    </div>
  );
}
