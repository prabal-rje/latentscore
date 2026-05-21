import { useQuery, useMutation } from "@tanstack/react-query";
import { useState, useEffect, useCallback } from "react";
import { fetchPresets, generateFromModel, detectProvider } from "../api";
import { useStore } from "../store";
import type { Preset } from "../types";

const ICON_MAP: Record<string, string> = {
  // Familiar presets
  ocean: "\u{1F30A}", sunrise: "\u{1F305}", city: "\u{1F303}", zen: "\u{1F9D8}",
  space: "\u{1F680}", fire: "\u{1F525}",
  // Diversity-sampled high-CLAP presets
  mystery: "\u{1F52E}", tension: "\u{1F48E}", bittersweet: "\u{1F342}",
  stars: "\u2728", thanks: "\u{1F31F}", doubt: "\u{1F3AD}",
};

const VISIBLE_COUNT = 5;
const CYCLE_INTERVAL = 6000;

export default function PresetGrid() {
  const setActiveSession = useStore((s) => s.setActiveSession);
  const addPendingTrack = useStore((s) => s.addPendingTrack);
  const resolvePlaylistTrack = useStore((s) => s.resolvePlaylistTrack);
  const failPlaylistTrack = useStore((s) => s.failPlaylistTrack);
  const selectedModel = useStore((s) => s.selectedModel);
  const customModelId = useStore((s) => s.customModelId);
  const providerApiKeys = useStore((s) => s.providerApiKeys);
  const setError = useStore((s) => s.setError);
  const setWarning = useStore((s) => s.setWarning);
  const setPhase = useStore((s) => s.setGeneratingPhase);
  const [page, setPage] = useState(0);
  const [fading, setFading] = useState(false);
  const [loadingId, setLoadingId] = useState<string | null>(null);

  const { data: presets = [], isLoading } = useQuery({
    queryKey: ["presets"],
    queryFn: fetchPresets,
  });

  const totalPages = Math.max(1, Math.ceil(presets.length / VISIBLE_COUNT));

  const cycleTo = useCallback((nextPage: number) => {
    setFading(true);
    setTimeout(() => {
      setPage(nextPage % totalPages);
      setFading(false);
    }, 300);
  }, [totalPages]);

  const apiKey = providerApiKeys[detectProvider(customModelId).provider] || "";

  // Play now — blocks UI, shows phase indicator
  const playMutation = useMutation({
    mutationFn: (v: string) =>
      generateFromModel(v, selectedModel, customModelId, apiKey, setPhase),
    onSuccess: (data, v) => {
      if (data.warning) setWarning(data.warning);
      setActiveSession({
        sessionId: data.session_id,
        title: data.title,
        justification: data.justification,
        palette: data.palette,
        config: data.config,
        vibe: v,
      });
      setLoadingId(null);
      setPhase(null);
    },
    onError: (err: Error) => { setLoadingId(null); setPhase(null); setError(err.message || "Generation failed"); },
  });

  // Only play blocks the UI
  const isPlayPending = playMutation.isPending;

  // Pause carousel during play generation
  useEffect(() => {
    if (presets.length <= VISIBLE_COUNT) return;
    if (isPlayPending) return;
    const timer = setInterval(() => {
      cycleTo(page + 1);
    }, CYCLE_INTERVAL);
    return () => clearInterval(timer);
  }, [page, presets.length, cycleTo, isPlayPending]);

  const handlePlay = (preset: Preset) => {
    if (isPlayPending) return;
    setLoadingId(preset.id + "-play");
    playMutation.mutate(preset.vibe);
  };

  const handleEnqueue = (e: React.MouseEvent, preset: Preset) => {
    e.stopPropagation();
    const trackId = addPendingTrack(preset.vibe);
    generateFromModel(preset.vibe, selectedModel, customModelId, apiKey)
      .then((data) => {
        resolvePlaylistTrack(trackId, {
          sessionId: data.session_id,
          title: data.title,
          palette: data.palette,
          justification: data.justification,
          config: data.config,
        });
      })
      .catch((err: Error) => {
        failPlaylistTrack(trackId, err.message || "Generation failed");
      });
  };

  if (isLoading) {
    return <div className="preset-strip-loading">Loading presets...</div>;
  }

  const start = page * VISIBLE_COUNT;
  const visible = presets.slice(start, start + VISIBLE_COUNT);

  return (
    <div className="preset-carousel">
      {totalPages > 1 && (
        <div className="preset-dots">
          {Array.from({ length: totalPages }).map((_, i) => (
            <button
              key={i}
              className={`preset-dot ${i === page ? "active" : ""}`}
              onClick={() => cycleTo(i)}
              aria-label={`Page ${i + 1}`}
            />
          ))}
        </div>
      )}
      <div className={`preset-carousel-items ${fading ? "fading" : ""}`}>
        {visible.map((preset: Preset) => {
          const isThisLoading = loadingId?.startsWith(preset.id) && isPlayPending;
          return (
            <div
              key={preset.id}
              className={`preset-chip ${isThisLoading ? "loading" : ""} ${isPlayPending && !isThisLoading ? "disabled" : ""}`}
              onClick={() => handlePlay(preset)}
              role="button"
              tabIndex={0}
            >
              <span className="preset-chip-icon">
                {ICON_MAP[preset.icon] ?? "\u{1F3B5}"}
              </span>
              <span className="preset-chip-name">{preset.name}</span>
              <span
                className="preset-chip-enqueue"
                onClick={(e) => handleEnqueue(e, preset)}
                title="Add to playlist"
                role="button"
                tabIndex={0}
                style={isThisLoading ? { opacity: 0.3, pointerEvents: "none" } : undefined}
              >
                +
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
