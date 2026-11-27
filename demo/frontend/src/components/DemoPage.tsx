import { useState, useEffect, useCallback, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import AuroraCanvas from "./AuroraCanvas";
import BrandWordmark from "./BrandWordmark";
import PixelCharm from "./PixelCharm";
import PresetGrid from "./PresetGrid";
import ShareButton from "./ShareButton";
import VibeInput from "./VibeInput";
import ParamPanel from "./ParamPanel";
import { useStore } from "../store";
import { useAudioPlayer } from "../hooks/useAudioPlayer";
import { DEMO_MODE, fetchCapabilities, detectProvider, generateFromModel } from "../api";
import { usePageSeo } from "../seo";
import {
  browserNextAutoDrift,
  browserWarmup,
  resetAutoDriftAnchor,
  subscribeEngineStatus,
  type EngineStatus,
} from "../browser/engine";
import type { PlaylistTrack } from "../types";

const PROVIDER_MODELS: Record<string, { label: string; models: { label: string; id: string }[] }> = {
  openai: {
    label: "OpenAI",
    models: [
      { label: "GPT-6 Luna", id: "gpt-6-luna" },
      { label: "GPT-6.2 Sol", id: "gpt-6.2-sol" },
      { label: "GPT-6.2 Astra", id: "gpt-6.2-astra" },
    ],
  },
  anthropic: {
    label: "Anthropic",
    models: [
      { label: "Haiku 5.5", id: "anthropic/claude-haiku-5-5" },
      { label: "Sonnet 5.5", id: "anthropic/claude-sonnet-5-5" },
      { label: "Opus 5.5", id: "anthropic/claude-opus-5-5" },
    ],
  },
  google: {
    label: "Gemini",
    models: [
      { label: "Gemini 3.8 Flash", id: "gemini/gemini-3.8-flash" },
      { label: "Gemini 4 Carbon", id: "gemini/gemini-4-carbon" },
      { label: "Gemini 4 Neon", id: "gemini/gemini-4-neon" },
    ],
  },
};

const ANONYMOUS = import.meta.env.VITE_ANONYMOUS === "true";

const IS_IOS =
  typeof navigator !== "undefined" &&
  (/iPhone|iPad|iPod/.test(navigator.userAgent) ||
    (navigator.userAgent.includes("Mac") && navigator.maxTouchPoints > 1));

export default function DemoPage() {
  usePageSeo({
    title: "LatentScore Live Demo: Free Browser Ambient Music Studio from Text",
    description:
      "Sketch procedural ambient music from text prompts directly in your browser. Zero GPU, 14,035 moods, live 18-parameter tuning, and endless similarity-gradient Auto Drift.",
    canonicalPath: "/demo",
  });
  const [iosBannerDismissed, setIosBannerDismissed] = useState(false);
  const activeSessionId = useStore((s) => s.activeSessionId);
  const activePalette = useStore((s) => s.activePalette);
  const activeTitle = useStore((s) => s.activeTitle);
  const activeJustification = useStore((s) => s.activeJustification);
  const activeVibe = useStore((s) => s.activeVibe);
  const isPlaying = useStore((s) => s.isPlaying);
  const playlist = useStore((s) => s.playlist);
  const removeFromPlaylist = useStore((s) => s.removeFromPlaylist);
  const setActiveSession = useStore((s) => s.setActiveSession);
  const toggleParamPanel = useStore((s) => s.toggleParamPanel);
  const showParamPanel = useStore((s) => s.showParamPanel);
  const selectedModel = useStore((s) => s.selectedModel);
  const setSelectedModel = useStore((s) => s.setSelectedModel);
  const customModelId = useStore((s) => s.customModelId);
  const setCustomModelId = useStore((s) => s.setCustomModelId);
  const providerApiKeys = useStore((s) => s.providerApiKeys);
  const setProviderApiKey = useStore((s) => s.setProviderApiKey);
  const streamMode = useStore((s) => s.streamMode);
  const setStreamMode = useStore((s) => s.setStreamMode);
  const loopMode = useStore((s) => s.loopMode);
  const setLoopMode = useStore((s) => s.setLoopMode);
  const autoMode = useStore((s) => s.autoMode);
  const setAutoMode = useStore((s) => s.setAutoMode);
  const autoMatchScore = useStore((s) => s.autoMatchScore);
  const setAutoMatchScore = useStore((s) => s.setAutoMatchScore);
  const errorMessage = useStore((s) => s.errorMessage);
  const setError = useStore((s) => s.setError);
  const warningMessage = useStore((s) => s.warningMessage);
  const setWarning = useStore((s) => s.setWarning);
  const generatingPhase = useStore((s) => s.generatingPhase);

  // Keep the generating banner visible for at least 1s so it does not flicker.
  const [delayedPhase, setDelayedPhase] = useState<"llm" | "evaluating" | "music" | null>(null);
  const phaseStartRef = useRef(0);
  useEffect(() => {
    if (generatingPhase !== null) {
      if (phaseStartRef.current === 0) phaseStartRef.current = Date.now();
      const captured = generatingPhase;
      const timer = setTimeout(() => setDelayedPhase(captured), 0);
      return () => clearTimeout(timer);
    }
    if (phaseStartRef.current === 0) return;
    const elapsed = Date.now() - phaseStartRef.current;
    const remaining = Math.max(0, 1000 - elapsed);
    const timer = setTimeout(() => {
      setDelayedPhase(null);
      phaseStartRef.current = 0;
    }, remaining);
    return () => clearTimeout(timer);
  }, [generatingPhase]);

  const visiblePhase = generatingPhase ?? delayedPhase;

  const [engineStatus, setEngineStatus] = useState<EngineStatus | null>(null);
  useEffect(() => {
    if (DEMO_MODE !== "browser") return;
    browserWarmup();
    return subscribeEngineStatus(setEngineStatus);
  }, []);

  const isGenerating = visiblePhase !== null;
  const generatingLabel =
    visiblePhase === "llm"
      ? "Calling LLM"
      : visiblePhase === "evaluating"
        ? "Evaluating Audio"
        : "Generating Music";
  const engineLoadingNote =
    DEMO_MODE === "browser" && engineStatus?.detail ? engineStatus.detail : null;

  const { data: capabilities } = useQuery({
    queryKey: ["capabilities"],
    queryFn: fetchCapabilities,
    staleTime: Infinity,
  });
  const expressiveAvailable = capabilities?.expressive_available ?? false;
  const heavyAvailable = capabilities?.models.includes("fast_heavy") ?? false;

  useEffect(() => {
    if (capabilities && !heavyAvailable && selectedModel === "fast_heavy") {
      setSelectedModel("fast");
    }
  }, [capabilities, heavyAvailable, selectedModel, setSelectedModel]);

  const autoDriftBusyRef = useRef(false);
  const serverDriftIdxRef = useRef(0);

  const advanceAutoDrift = useCallback(async (reset = false) => {
    if (autoDriftBusyRef.current) return;
    autoDriftBusyRef.current = true;
    try {
      const state = useStore.getState();
      if (!state.autoMode) return;
      if (DEMO_MODE === "browser") {
        const next = await browserNextAutoDrift(
          state.activeSessionId ? state.activeConfig : undefined,
          reset
        );
        if (!useStore.getState().autoMode) return;
        setAutoMatchScore(next.similarity);
        setActiveSession({
          sessionId: next.session_id,
          title: next.title,
          justification: next.justification,
          palette: next.palette,
          config: next.config,
          vibe: next.vibe,
        });
      } else {
        const fallbackVibes = [
          "calm twilight rain over glass",
          "warm lo-fi vinyl dusk in a quiet cafe",
          "misty cedar forest morning sunlight",
          "floating acoustic harp in deep cavern",
          "ethereal starlight nebula horizon",
          "soft neon rain on empty midnight boulevard",
          "gentle ocean tide under pale moonlight",
          "crystalline snow falling in stillness",
        ];
        const vibe = fallbackVibes[serverDriftIdxRef.current % fallbackVibes.length];
        serverDriftIdxRef.current += 1;
        const apiKey =
          state.providerApiKeys[detectProvider(state.customModelId).provider] || "";
        const res = await generateFromModel(
          vibe,
          state.selectedModel,
          state.customModelId,
          apiKey
        );
        if (!useStore.getState().autoMode) return;
        setAutoMatchScore(null);
        setActiveSession({
          sessionId: res.session_id,
          title: res.title,
          justification: res.justification,
          palette: res.palette,
          config: res.config,
          vibe,
        });
      }
    } catch (err) {
      console.error("Auto drift step failed", err);
      useStore
        .getState()
        .setError(err instanceof Error ? err.message : "Auto Drift failed");
    } finally {
      autoDriftBusyRef.current = false;
    }
  }, [setActiveSession, setAutoMatchScore]);

  const handleStreamEnd = useCallback(() => {
    const state = useStore.getState();
    if (state.autoMode) {
      void advanceAutoDrift(false);
      return;
    }
    const pl = state.playlist;
    const currentId = state.activeSessionId;
    if (pl.length === 0) return;
    const currentIdx = pl.findIndex((t) => t.sessionId === currentId);
    const startIdx = currentIdx >= 0 ? currentIdx + 1 : 0;
    const next = pl.slice(startIdx).find((t) => t.status === "ready");
    if (next) {
      state.setActiveSession({
        sessionId: next.sessionId,
        title: next.title,
        justification: next.justification,
        palette: next.palette,
        config: next.config,
        vibe: next.vibe,
      });
    }
  }, [advanceAutoDrift]);

  const { startStream, stopStream, downloadAudio, hasRecordedAudio } =
    useAudioPlayer(handleStreamEnd, advanceAutoDrift);

  const handleToggleAutoMode = useCallback(() => {
    const state = useStore.getState();
    if (state.autoMode) {
      setAutoMode(false);
      return;
    }
    setAutoMode(true);
    if (!state.activeSessionId) {
      resetAutoDriftAnchor(null);
      void advanceAutoDrift(true);
    } else if (!state.isPlaying) {
      startStream();
    }
  }, [advanceAutoDrift, setAutoMode, startStream]);

  const [showMore, setShowMore] = useState(false);
  const [showModelDialog, setShowModelDialog] = useState(false);
  const [activeProvider, setActiveProvider] = useState(
    () => detectProvider(customModelId).provider
  );
  const [customModelInput, setCustomModelInput] = useState("");
  const restreamTrigger = useStore((s) => s.restreamTrigger);

  useEffect(() => {
    if (ANONYMOUS) document.title = "Live Demo";
  }, []);

  // Support shared deep links like /demo?vibe=midnight%20rain
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sharedVibe = params.get("vibe")?.trim();
    if (sharedVibe) {
      useStore.getState().setVibeText(sharedVibe);
    }
  }, []);

  useEffect(() => {
    if (activeSessionId) startStream();
  }, [activeSessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (restreamTrigger > 0 && activeSessionId) startStream();
  }, [restreamTrigger]); // eslint-disable-line react-hooks/exhaustive-deps

  // Smoothly crossfade to the newly selected Render/Live mode if playback is active.
  const prevStreamModeRef = useRef(streamMode);
  useEffect(() => {
    if (prevStreamModeRef.current === streamMode) return;
    prevStreamModeRef.current = streamMode;
    if (activeSessionId && useStore.getState().isPlaying) {
      startStream();
    }
  }, [streamMode, activeSessionId, startStream]);

  useEffect(() => {
    if (!errorMessage) return;
    const t = setTimeout(() => setError(null), 5000);
    return () => clearTimeout(t);
  }, [errorMessage, setError]);

  useEffect(() => {
    if (!warningMessage) return;
    const t = setTimeout(() => setWarning(null), 6000);
    return () => clearTimeout(t);
  }, [warningMessage, setWarning]);

  useEffect(() => {
    if (!showMore && !showModelDialog) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setShowMore(false);
        setShowModelDialog(false);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [showMore, showModelDialog]);

  const handlePlayTrack = (track: PlaylistTrack) => {
    if (track.status !== "ready") return;
    setActiveSession({
      sessionId: track.sessionId,
      title: track.title,
      justification: track.justification,
      palette: track.palette,
      config: track.config,
      vibe: track.vibe,
    });
  };

  const handleSkip = () => {
    if (playlist.length > 0) {
      const currentIdx = playlist.findIndex((t) => t.sessionId === activeSessionId);
      const startIdx = currentIdx >= 0 ? currentIdx + 1 : 0;
      const next = playlist.slice(startIdx).find((t) => t.status === "ready");
      if (next) {
        setActiveSession({
          sessionId: next.sessionId,
          title: next.title,
          justification: next.justification,
          palette: next.palette,
          config: next.config,
          vibe: next.vibe,
        });
        return;
      }
    }
    if (useStore.getState().autoMode) {
      void advanceAutoDrift(false);
      return;
    }
    stopStream();
  };

  const handleDownload = () => {
    if (hasRecordedAudio()) {
      downloadAudio(activeTitle || "audio");
    }
  };

  const modelLabel =
    selectedModel === "custom"
      ? "Custom LLM"
      : selectedModel === "expressive"
        ? "Expressive"
        : selectedModel === "fast_heavy"
          ? "Fast (Heavy)"
          : DEMO_MODE === "browser"
            ? "Fast (Local)"
            : "Fast";

  return (
    <>
      <AuroraCanvas reactive={true} />

      <div className="alba-page">
        <header className="masthead">
          <div className="measure-demo masthead-inner">
            {!ANONYMOUS ? (
              <BrandWordmark isPlaying={isPlaying} />
            ) : (
              <span className="wordmark">Live Demo</span>
            )}

            <div className="masthead-right">
              {!ANONYMOUS && (
                <ShareButton
                  variant="pill"
                  trackTitle={activeSessionId ? activeTitle : undefined}
                  trackVibe={activeSessionId ? activeVibe : undefined}
                />
              )}
              <button
                type="button"
                className="status-trigger"
                onClick={() => setShowModelDialog(true)}
                aria-label="Choose generation model"
              >
                <span className={`status-dot${engineLoadingNote ? " busy" : ""}`} />
                <span>{modelLabel}</span>
              </button>
            </div>
          </div>
        </header>

        <main className="demo-main">
          <div className="measure-demo">
            {!activeSessionId && (
              <section className="demo-hero">
                <PixelCharm initialIndex={1} size={56} isPlaying={isPlaying} />
                <h1 className="demo-heading">
                  Sketch a mood, hear it{" "}
                  <span className="no-orphan">
                    <em>come alive</em>.
                  </span>
                </h1>
                <p className="demo-subtitle">Describe any scene or feeling.</p>
              </section>
            )}

            {engineLoadingNote && (
              <div className="ios-silent-banner" role="status">
                <span className="spinner-sm" />
                <span>{engineLoadingNote}</span>
              </div>
            )}

            {IS_IOS && !iosBannerDismissed && (
              <div className="ios-silent-banner" role="status">
                <svg
                  className="ios-silent-banner-icon"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M13.73 21a2 2 0 0 1-3.46 0" />
                  <path d="M18.63 13A17.89 17.89 0 0 1 18 8" />
                  <path d="M6.26 6.26A5.86 5.86 0 0 0 6 8c0 7-3 9-3 9h14" />
                  <path d="M18 8a6 6 0 0 0-9.33-5" />
                  <line x1="1" y1="1" x2="23" y2="23" />
                </svg>
                <span>Turn off Silent Mode to hear audio</span>
                <button
                  type="button"
                  className="ios-silent-banner-close"
                  onClick={() => setIosBannerDismissed(true)}
                  aria-label="Dismiss"
                >
                  &times;
                </button>
              </div>
            )}

            <div className="player-card">
              {activeSessionId && (
                <div>
                  <div className="np-palette">
                    {activePalette.map((c, i) => (
                      <div key={i} className="np-palette-strip" style={{ backgroundColor: c }} />
                    ))}
                  </div>

                  <div className="np-body">
                    {isGenerating && (
                      <div className="generating-banner">
                        <span className="generating-dots">
                          <span />
                          <span />
                          <span />
                        </span>
                        <span>{generatingLabel}</span>
                      </div>
                    )}

                    <div className="np-top-row">
                      <div key={activeSessionId} className="np-info">
                        <div className="np-title">{activeTitle}</div>
                        <div className="np-vibe">
                          {activeVibe}
                          {autoMode && (
                            <span className="np-auto-badge">
                              {autoMatchScore !== null
                                ? `Auto · ${Math.round(autoMatchScore * 100)}% gradient`
                                : "Auto Drift"}
                            </span>
                          )}
                        </div>
                      </div>
                      <PixelCharm initialIndex={2} size={42} isPlaying={isPlaying} />
                    </div>

                    <div className="np-controls">
                      <button
                        type="button"
                        className={`play-btn ${isPlaying ? "playing" : ""}`}
                        onClick={isPlaying ? stopStream : startStream}
                        title={isPlaying ? "Stop" : "Play"}
                        disabled={isGenerating}
                      >
                        {isGenerating ? (
                          <span className="spinner" />
                        ) : isPlaying ? (
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                            <rect x="6" y="4" width="4" height="16" rx="1" />
                            <rect x="14" y="4" width="4" height="16" rx="1" />
                          </svg>
                        ) : (
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M8 5v14l11-7z" />
                          </svg>
                        )}
                      </button>

                      <button
                        type="button"
                        className="ctrl-btn"
                        onClick={handleSkip}
                        title={
                          autoMode
                            ? "Glide to Next Similar Genre"
                            : playlist.length > 0
                              ? "Next Track"
                              : "Stop"
                        }
                        disabled={isGenerating || (!isPlaying && !autoMode && playlist.length === 0)}
                      >
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M5 4l10 8-10 8V4z" />
                          <rect x="17" y="4" width="2.5" height="16" rx="0.5" />
                        </svg>
                      </button>

                      <button
                        type="button"
                        className={`ctrl-btn${!showParamPanel ? " param-glow" : ""}`}
                        onClick={toggleParamPanel}
                        title="Tune Parameters"
                      >
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <line x1="4" y1="21" x2="4" y2="14" />
                          <line x1="4" y1="10" x2="4" y2="3" />
                          <line x1="12" y1="21" x2="12" y2="12" />
                          <line x1="12" y1="8" x2="12" y2="3" />
                          <line x1="20" y1="21" x2="20" y2="16" />
                          <line x1="20" y1="12" x2="20" y2="3" />
                          <line x1="1" y1="14" x2="7" y2="14" />
                          <line x1="9" y1="8" x2="15" y2="8" />
                          <line x1="17" y1="16" x2="23" y2="16" />
                        </svg>
                      </button>

                      <button
                        type="button"
                        className="mode-toggle"
                        onClick={() => setStreamMode(!streamMode)}
                        title={
                          streamMode
                            ? "Live mode (continuous streamed audio)"
                            : "Render mode (full clip synthesis)"
                        }
                      >
                        <span className={`mode-toggle-dot ${streamMode ? "stream" : "fetch"}`} />
                        {streamMode ? "Live" : "Render"}
                      </button>

                      <button
                        type="button"
                        className={`mode-toggle ${loopMode ? "active" : ""}`}
                        onClick={() => setLoopMode(!loopMode)}
                        title={loopMode ? "Looping current track" : "Play once, then fade out"}
                      >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="17 1 21 5 17 9" />
                          <path d="M3 11V9a4 4 0 0 1 4-4h14" />
                          <polyline points="7 23 3 19 7 15" />
                          <path d="M21 13v2a4 4 0 0 1-4 4H3" />
                        </svg>
                        Loop
                      </button>

                      <button
                        type="button"
                        className={`mode-toggle ${autoMode ? "active auto-active" : ""}`}
                        onClick={handleToggleAutoMode}
                        title="Auto Drift: endlessly crossfade through 14,000+ genres by similarity gradient"
                      >
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M18.178 8c5.096 0 5.096 8 0 8-5.095 0-7.133-8-12.739-8-4.585 0-4.585 8 0 8 5.606 0 7.644-8 12.74-8z" />
                        </svg>
                        Auto
                      </button>

                      <div style={{ flex: 1 }} />

                      <button
                        type="button"
                        className="ctrl-btn"
                        onClick={() => setShowMore(!showMore)}
                        title="More info"
                      >
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                          <circle cx="5" cy="12" r="2" />
                          <circle cx="12" cy="12" r="2" />
                          <circle cx="19" cy="12" r="2" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {activeSessionId && <div className="card-divider" />}

              <div className="card-input">
                {!activeSessionId && isGenerating && (
                  <div className="generating-banner">
                    <span className="generating-dots">
                      <span />
                      <span />
                      <span />
                    </span>
                    <span>{generatingLabel}</span>
                  </div>
                )}
                {activeSessionId && <div className="input-label">generate another</div>}
                <VibeInput />
              </div>

              <div className="card-divider" />

              <div className="card-presets">
                <div className="presets-header-row">
                  <div className="section-label">
                    {activeSessionId ? "or try a preset" : "or pick a preset"}
                  </div>
                  {!activeSessionId && (
                    <button
                      type="button"
                      className={`auto-drift-pill ${autoMode ? "active" : ""}`}
                      onClick={handleToggleAutoMode}
                      title="Endlessly glide through 14,000+ genres by similarity gradient"
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M18.178 8c5.096 0 5.096 8 0 8-5.095 0-7.133-8-12.739-8-4.585 0-4.585 8 0 8 5.606 0 7.644-8 12.74-8z" />
                      </svg>
                      <span>{autoMode ? "Auto Drifting" : "Auto Drift"}</span>
                    </button>
                  )}
                </div>
                <PresetGrid />
              </div>

              {playlist.length > 0 && (
                <>
                  <div className="card-divider" />
                  <div className="card-playlist">
                    <div className="section-label">playlist</div>
                    <div className="playlist-scroll">
                      {playlist.map((track, i) => {
                        const isCurrent =
                          track.status === "ready" && track.sessionId === activeSessionId;
                        const isPending = track.status === "pending";
                        const isError = track.status === "error";
                        return (
                          <div
                            key={track.id}
                            className={`queue-track ${isCurrent ? "active-track" : ""}`}
                            onClick={() => handlePlayTrack(track)}
                            onKeyDown={(e) => {
                              if (e.target === e.currentTarget && (e.key === "Enter" || e.key === " ")) {
                                e.preventDefault();
                                handlePlayTrack(track);
                              }
                            }}
                            role="button"
                            tabIndex={track.status === "ready" ? 0 : -1}
                            style={isPending ? { cursor: "default", opacity: 0.6 } : undefined}
                          >
                            {isCurrent && isPlaying ? (
                              <div className="playing-indicator">
                                <span />
                                <span />
                                <span />
                              </div>
                            ) : isPending ? (
                              <span className="spinner-sm" />
                            ) : isError ? (
                              <div className="queue-track-error-icon">!</div>
                            ) : (
                              <div className="queue-track-colors">
                                {track.palette.slice(0, 3).map((c, j) => (
                                  <span key={j} className="q-dot" style={{ backgroundColor: c }} />
                                ))}
                              </div>
                            )}
                            <div className="queue-track-info">
                              <span className="queue-track-title">
                                {isPending
                                  ? "Generating..."
                                  : isError
                                    ? "Failed"
                                    : track.title}
                              </span>
                              <span className="queue-track-vibe">{track.vibe}</span>
                            </div>
                            <button
                              type="button"
                              className="queue-track-remove"
                              onClick={(e) => {
                                e.stopPropagation();
                                removeFromPlaylist(i);
                              }}
                              aria-label="Remove track"
                            >
                              &times;
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </>
              )}
            </div>

            {!ANONYMOUS && (
              <div className="social-row">
                <a
                  href="https://www.linkedin.com/in/prabal1997"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="social-icon"
                  title="LinkedIn"
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                  </svg>
                </a>
                <a
                  href="https://x.com/prabal_"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="social-icon"
                  title="X"
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                  </svg>
                </a>
                <a
                  href="https://github.com/prabal-rje/latentscore"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="social-icon"
                  title="GitHub"
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                  </svg>
                </a>
              </div>
            )}
          </div>
        </main>
      </div>

      <ParamPanel />

      {warningMessage && (
        <div className="warning-toast" onClick={() => setWarning(null)}>
          <span className="warning-toast-icon">!</span>
          <span>{warningMessage}</span>
        </div>
      )}

      {errorMessage && (
        <div className="error-toast" onClick={() => setError(null)}>
          <span className="error-toast-icon">!</span>
          <span>{errorMessage}</span>
        </div>
      )}

      {showModelDialog && (
        <div
          className="modal-overlay"
          onClick={() => setShowModelDialog(false)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="model-dialog-title"
        >
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <span id="model-dialog-title" className="modal-header-title">Choose Model</span>
              <button
                type="button"
                className="modal-close"
                onClick={() => setShowModelDialog(false)}
                aria-label="Close"
                autoFocus
              >
                &times;
              </button>
            </div>
            <div className="modal-body">
              {heavyAvailable && (
                <button
                  type="button"
                  className={`model-option ${selectedModel === "fast_heavy" ? "active" : ""}`}
                  onClick={() => {
                    setSelectedModel("fast_heavy");
                    setShowModelDialog(false);
                  }}
                >
                  <span className="model-name">Fast (Heavy)</span>
                  <span className="model-desc-long">
                    CLAP audio-grounded matching. Matches your text against how each configuration
                    sounds.
                  </span>
                </button>
              )}

              <button
                type="button"
                className={`model-option ${selectedModel === "fast" ? "active" : ""}`}
                onClick={() => {
                  setSelectedModel("fast");
                  setShowModelDialog(false);
                }}
              >
                <span className="model-name">
                  {DEMO_MODE === "browser" ? "Fast (Local in Browser)" : "Fast"}
                </span>
                <span className="model-desc-long">
                  Semantic retrieval across 14,035 curated configurations using quantized MiniLM
                  embeddings.
                </span>
              </button>

              <button
                type="button"
                className={`model-option ${selectedModel === "expressive" ? "active" : ""} ${!expressiveAvailable ? "disabled" : ""}`}
                onClick={() => {
                  if (expressiveAvailable) {
                    setSelectedModel("expressive");
                    setShowModelDialog(false);
                  }
                }}
                disabled={!expressiveAvailable}
              >
                <span className="model-name">Expressive</span>
                <span className="model-desc-long">
                  Fine-tuned Gemma 3 1B on CPU for custom parameter generation. Available when
                  running the Python library locally.
                </span>
                {!expressiveAvailable && !ANONYMOUS && (
                  <span className="model-local-link">
                    <a
                      href="https://github.com/prabal-rje/latentscore"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Run locally to enable &rarr;
                    </a>
                  </span>
                )}
              </button>

              <div
                className={`model-option ${selectedModel === "custom" ? "active" : ""}`}
                onClick={() => setSelectedModel("custom")}
                onKeyDown={(e) => {
                  if (e.target === e.currentTarget && (e.key === "Enter" || e.key === " ")) {
                    e.preventDefault();
                    setSelectedModel("custom");
                  }
                }}
                role="button"
                tabIndex={0}
              >
                <span className="model-name">Custom LLM</span>
                <span className="model-desc-long">
                  Bring your own API key. Calls your chosen LLM directly from the browser for full
                  creative control.
                </span>

                {selectedModel === "custom" && (
                  <div className="custom-llm-config" onClick={(e) => e.stopPropagation()}>
                    <div className="custom-llm-providers">
                      {Object.entries(PROVIDER_MODELS).map(([key, p]) => (
                        <button
                          key={key}
                          type="button"
                          className={`custom-llm-provider-tab ${activeProvider === key ? "active" : ""}`}
                          onClick={() => {
                            setActiveProvider(key);
                            setCustomModelId(p.models[0].id);
                            setCustomModelInput("");
                          }}
                        >
                          {p.label}
                        </button>
                      ))}
                    </div>

                    {(() => {
                      const presetModels = PROVIDER_MODELS[activeProvider]?.models ?? [];
                      const matchesPreset = presetModels.some((m) => m.id === customModelId);
                      const selectVal =
                        customModelInput !== "" || !matchesPreset ? "__custom__" : customModelId;
                      return (
                        <>
                          <select
                            className="custom-llm-select"
                            value={selectVal}
                            onChange={(e) => {
                              if (e.target.value === "__custom__") {
                                setCustomModelInput(customModelId);
                              } else {
                                setCustomModelInput("");
                                setCustomModelId(e.target.value);
                              }
                            }}
                          >
                            {presetModels.map((m) => (
                              <option key={m.id} value={m.id}>
                                {m.label}
                              </option>
                            ))}
                            <option value="__custom__">
                              {!matchesPreset && customModelId
                                ? `Custom (${customModelId})`
                                : "Custom..."}
                            </option>
                          </select>

                          {(customModelInput !== "" || !matchesPreset) && (
                            <input
                              type="text"
                              className="custom-llm-input"
                              placeholder={
                                activeProvider === "google"
                                  ? "gemini/gemini-..."
                                  : activeProvider === "anthropic"
                                    ? "anthropic/claude-..."
                                    : "gpt-..."
                              }
                              value={customModelInput !== "" ? customModelInput : customModelId}
                              autoFocus
                              onChange={(e) => {
                                const rawVal = e.target.value;
                                setCustomModelInput(rawVal);
                                const trimmed = rawVal.trim();
                                if (!trimmed) return;
                                if (activeProvider === "google" && !trimmed.startsWith("gemini/")) {
                                  setCustomModelId(`gemini/${trimmed}`);
                                } else if (
                                  activeProvider === "anthropic" &&
                                  !trimmed.startsWith("anthropic/") &&
                                  !trimmed.startsWith("claude")
                                ) {
                                  setCustomModelId(`anthropic/${trimmed}`);
                                } else {
                                  setCustomModelId(trimmed);
                                }
                              }}
                            />
                          )}
                        </>
                      );
                    })()}

                    <input
                      type="password"
                      className="custom-llm-input"
                      placeholder={`${PROVIDER_MODELS[activeProvider]?.label ?? "Provider"} API Key`}
                      value={providerApiKeys[activeProvider] || ""}
                      onChange={(e) => setProviderApiKey(activeProvider, e.target.value)}
                    />
                    <span className="custom-llm-note">
                      Key stays in your browser only and goes straight to{" "}
                      {PROVIDER_MODELS[activeProvider]?.label ?? "the provider"}.
                    </span>
                    <button
                      type="button"
                      className="custom-llm-done"
                      onClick={() => setShowModelDialog(false)}
                      disabled={!customModelId || !providerApiKeys[activeProvider]}
                    >
                      Done
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {showMore && activeSessionId && (
        <div
          className="modal-overlay"
          onClick={() => setShowMore(false)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="about-piece-title"
        >
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <span id="about-piece-title" className="modal-header-title">About This Piece</span>
              <button
                type="button"
                className="modal-close"
                onClick={() => setShowMore(false)}
                aria-label="Close"
                autoFocus
              >
                &times;
              </button>
            </div>
            <div className="modal-body">
              <button
                type="button"
                className="more-download-btn"
                onClick={handleDownload}
                disabled={!hasRecordedAudio()}
                title={hasRecordedAudio() ? "Download as WAV" : "Play first to record audio"}
                style={{ marginBottom: 20 }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                {hasRecordedAudio() ? "Download WAV" : "Play to enable download"}
              </button>

              <div className="more-label">palette</div>
              <div className="more-palette-row">
                {activePalette.map((c, i) => (
                  <div key={i} className="more-swatch">
                    <div className="more-swatch-dot" style={{ backgroundColor: c }} />
                    <span className="more-swatch-hex">{c}</span>
                  </div>
                ))}
              </div>

              <div className="more-label">justification</div>
              <div className="more-justification">{activeJustification}</div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
