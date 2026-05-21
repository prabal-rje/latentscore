import { useState, useEffect, useCallback, useRef } from "react";
import { Link } from "react-router-dom";
import ParticleVisualizer from "./ParticleVisualizer";
import PresetGrid from "./PresetGrid";
import VibeInput from "./VibeInput";
import ParamPanel from "./ParamPanel";
import { useQuery } from "@tanstack/react-query";
import { useStore } from "../store";
import { useAudioPlayer } from "../hooks/useAudioPlayer";
import { fetchCapabilities, detectProvider } from "../api";
import type { PlaylistTrack } from "../types";

const PROVIDER_MODELS: Record<string, { label: string; models: { label: string; id: string }[] }> = {
  openai: {
    label: "OpenAI",
    models: [
      { label: "GPT-5 Nano", id: "gpt-5-nano-2025-08-07" },
      { label: "GPT-5 Mini", id: "gpt-5-mini-2025-08-07" },
      { label: "GPT-5.2", id: "gpt-5.2-2025-12-11" },
    ],
  },
  anthropic: {
    label: "Anthropic",
    models: [
      { label: "Haiku 3", id: "anthropic/claude-3-haiku-20240307" },
      { label: "Sonnet 4.5", id: "anthropic/claude-sonnet-4-5-20250929" },
      { label: "Opus 4.5", id: "anthropic/claude-opus-4-5-20251101" },
    ],
  },
  google: {
    label: "Gemini",
    models: [
      { label: "Gemini 3 Flash", id: "gemini/gemini-3-flash-preview" },
      { label: "Gemini 3 Pro", id: "gemini/gemini-3-pro-preview" },
    ],
  },
};

const ANONYMOUS = import.meta.env.VITE_ANONYMOUS === "true";

const IS_IOS =
  typeof navigator !== "undefined" &&
  (/iPhone|iPad|iPod/.test(navigator.userAgent) ||
    (navigator.userAgent.includes("Mac") && navigator.maxTouchPoints > 1));

function DemoPage() {
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
  const errorMessage = useStore((s) => s.errorMessage);
  const setError = useStore((s) => s.setError);
  const warningMessage = useStore((s) => s.warningMessage);
  const setWarning = useStore((s) => s.setWarning);
  const generatingPhase = useStore((s) => s.generatingPhase);

  // Keep the label visible for at least 1s so it doesn't flash/jitter
  const [visiblePhase, setVisiblePhase] = useState<"llm" | "evaluating" | "music" | null>(null);
  const phaseStartRef = useRef(0);
  useEffect(() => {
    if (generatingPhase !== null) {
      setVisiblePhase(generatingPhase);
      if (phaseStartRef.current === 0) phaseStartRef.current = Date.now();
    } else {
      const elapsed = Date.now() - phaseStartRef.current;
      const remaining = Math.max(0, 1000 - elapsed);
      const timer = setTimeout(() => {
        setVisiblePhase(null);
        phaseStartRef.current = 0;
      }, remaining);
      return () => clearTimeout(timer);
    }
  }, [generatingPhase]);

  const isGenerating = visiblePhase !== null;
  const generatingLabel = visiblePhase === "llm" ? "Calling LLM" : visiblePhase === "evaluating" ? "Evaluating Audio" : "Generating Music";

  const { data: capabilities } = useQuery({
    queryKey: ["capabilities"],
    queryFn: fetchCapabilities,
    staleTime: Infinity,
  });
  const expressiveAvailable = capabilities?.expressive_available ?? false;

  // Auto-advance to next ready track when stream ends naturally
  const handleStreamEnd = useCallback(() => {
    const state = useStore.getState();
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
  }, []);

  const { startStream, stopStream, downloadAudio, hasRecordedAudio } = useAudioPlayer(handleStreamEnd);

  const [showMore, setShowMore] = useState(false);
  const [showModelDialog, setShowModelDialog] = useState(false);
  const [activeProvider, setActiveProvider] = useState(() => detectProvider(customModelId).provider);
  const [customModelInput, setCustomModelInput] = useState("");

  const restreamTrigger = useStore((s) => s.restreamTrigger);

  // Override document title in anonymous mode
  useEffect(() => {
    if (ANONYMOUS) document.title = "Live Demo";
  }, []);

  // Auto-play whenever the active session changes
  useEffect(() => {
    if (activeSessionId) startStream();
  }, [activeSessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Restart stream when params are applied
  useEffect(() => {
    if (restreamTrigger > 0 && activeSessionId) startStream();
  }, [restreamTrigger]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-dismiss error toast
  useEffect(() => {
    if (!errorMessage) return;
    const t = setTimeout(() => setError(null), 5000);
    return () => clearTimeout(t);
  }, [errorMessage, setError]);

  // Auto-dismiss warning toast
  useEffect(() => {
    if (!warningMessage) return;
    const t = setTimeout(() => setWarning(null), 6000);
    return () => clearTimeout(t);
  }, [warningMessage, setWarning]);

  // Close modals on Escape
  useEffect(() => {
    if (!showMore && !showModelDialog) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setShowMore(false); setShowModelDialog(false); }
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
    if (playlist.length === 0) {
      stopStream();
      return;
    }
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
    } else {
      stopStream();
    }
  };

  const handleDownload = () => {
    if (hasRecordedAudio()) {
      downloadAudio(activeTitle || "audio");
    }
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --c1: ${activePalette[0] ?? "#d4a574"};
          --c2: ${activePalette[1] ?? "#8b6e55"};
          --c3: ${activePalette[2] ?? "#c49670"};
          --bg: ${activePalette[3] ?? "#1c1a17"};
          --c4: ${activePalette[4] ?? "#e8c9a0"};

          --ivory: #f0ece4;
          --ivory-dim: rgba(240, 236, 228, 0.45);
          --ivory-faint: rgba(240, 236, 228, 0.12);
          --ivory-ghost: rgba(240, 236, 228, 0.06);
          --warm-black: #141210;

          --font-display: 'DM Serif Display', Georgia, serif;
          --font-body: 'IBM Plex Mono', 'Menlo', monospace;
        }

        body {
          font-family: var(--font-body);
          background: var(--warm-black);
          color: var(--ivory);
          overflow-x: hidden;
          min-height: 100vh;
        }

        /* --- Noise grain --- */
        .grain {
          position: fixed;
          top: 0; left: 0; right: 0; bottom: 0;
          z-index: 1;
          pointer-events: none;
          opacity: 0.035;
          background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
          background-repeat: repeat;
          background-size: 256px 256px;
        }

        /* --- App wrapper --- */
        .app-wrapper {
          position: relative;
          z-index: 2;
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: flex-start;
          flex-direction: column;
          padding: 120px 24px 48px;
        }

        /* --- Brand mark (top-left) --- */
        .brand-mark {
          position: fixed;
          top: 24px;
          left: 28px;
          z-index: 50;
          font-family: var(--font-display);
          font-size: 26px;
          font-weight: 400;
          color: var(--ivory-dim);
          letter-spacing: -0.5px;
          text-decoration: none;
          transition: color 0.3s ease;
        }

        .brand-mark:hover {
          color: var(--ivory);
        }

        .brand-mark em {
          font-style: italic;
          color: var(--c4);
          transition: color 1s ease;
        }

        /* --- Staggered reveal --- */
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(12px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .reveal { opacity: 0; animation: fadeUp 0.6s ease forwards; }
        .r1 { animation-delay: 0.1s; }
        .r2 { animation-delay: 0.2s; }
        .r3 { animation-delay: 0.35s; }
        .r4 { animation-delay: 0.5s; }

        /* --- Player card --- */
        .player-card {
          width: 100%;
          background: rgba(26, 24, 22, 0.72);
          border: 1px solid var(--ivory-ghost);
          border-radius: 12px;
          backdrop-filter: blur(40px);
          -webkit-backdrop-filter: blur(40px);
          overflow: visible;
        }

        /* --- Now Playing section --- */
        .np-section {
          padding: 0;
          overflow: visible;
        }

        .np-palette {
          display: flex;
          height: 4px;
          width: 100%;
          border-radius: 12px 12px 0 0;
          overflow: hidden;
        }

        .np-palette-strip {
          flex: 1;
          transition: background-color 1s ease;
        }

        .np-body {
          padding: 20px 28px 18px;
        }

        .np-top-row {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 16px;
        }

        .np-info {
          flex: 1;
          min-width: 0;
        }

        .np-title {
          font-family: var(--font-display);
          font-size: 24px;
          font-weight: 400;
          line-height: 1.15;
          color: var(--ivory);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .np-vibe {
          font-size: 11px;
          font-weight: 300;
          color: var(--ivory-dim);
          margin-top: 3px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          font-style: italic;
        }

        .np-controls {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .play-btn {
          width: 44px; height: 44px;
          border-radius: 50%;
          border: 1.5px solid var(--ivory-faint);
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          transition: all 0.25s ease;
          background: transparent;
          color: var(--ivory);
          flex-shrink: 0;
        }

        .play-btn:hover:not(:disabled) {
          border-color: var(--ivory);
          background: rgba(240, 236, 228, 0.06);
        }

        .play-btn:disabled {
          opacity: 0.15;
          cursor: not-allowed;
        }

        .play-btn.playing {
          border-color: var(--c1);
          background: rgba(240, 236, 228, 0.04);
        }

        .play-btn.playing:hover {
          background: rgba(240, 236, 228, 0.08);
        }

        .ctrl-btn {
          width: 34px; height: 34px;
          border-radius: 4px;
          border: 1px solid transparent;
          background: transparent;
          color: var(--ivory-dim);
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s ease;
          flex-shrink: 0;
        }

        .ctrl-btn:hover:not(:disabled) {
          color: var(--ivory);
          border-color: var(--ivory-faint);
          background: var(--ivory-ghost);
        }

        .ctrl-btn:disabled {
          opacity: 0.15;
          cursor: not-allowed;
        }

        @keyframes param-pulse {
          0%, 100% { box-shadow: 0 0 3px rgba(218, 165, 32, 0.1), 0 0 6px rgba(218, 165, 32, 0.05); }
          50% { box-shadow: 0 0 5px rgba(218, 165, 32, 0.3), 0 0 12px rgba(218, 165, 32, 0.1); }
        }

        .param-glow {
          animation: param-pulse 3s ease-in-out infinite;
          border-color: rgba(218, 165, 32, 0.2) !important;
          color: rgba(218, 165, 32, 0.7) !important;
        }

        .mode-toggle {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 4px 10px;
          border-radius: 4px;
          border: 1px solid var(--ivory-ghost);
          background: transparent;
          color: var(--ivory-dim);
          cursor: pointer;
          font-family: var(--font-body);
          font-size: 10px;
          font-weight: 400;
          letter-spacing: 0.5px;
          transition: all 0.2s ease;
          white-space: nowrap;
        }

        .mode-toggle:hover {
          border-color: var(--ivory-faint);
          color: var(--ivory);
        }

        .mode-toggle.active {
          border-color: var(--c4);
          color: var(--c4);
        }

        .mode-toggle-dot {
          width: 5px;
          height: 5px;
          border-radius: 50%;
          transition: background 0.2s ease;
        }

        .mode-toggle-dot.stream {
          background: var(--c4);
        }

        .mode-toggle-dot.fetch {
          background: var(--c1);
        }

        /* --- Info modal overlay --- */
        .modal-overlay {
          position: fixed;
          top: 0; left: 0; right: 0; bottom: 0;
          background: rgba(0, 0, 0, 0.6);
          backdrop-filter: blur(6px);
          -webkit-backdrop-filter: blur(6px);
          z-index: 200;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 24px;
          animation: fadeOverlay 0.2s ease;
        }

        @keyframes fadeOverlay {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        .modal-card {
          width: 400px;
          max-width: 100%;
          max-height: 70vh;
          background: rgba(22, 20, 18, 0.97);
          border: 1px solid var(--ivory-faint);
          border-radius: 12px;
          backdrop-filter: blur(40px);
          -webkit-backdrop-filter: blur(40px);
          overflow: hidden;
          display: flex;
          flex-direction: column;
          animation: modalIn 0.25s ease;
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
        }

        @keyframes modalIn {
          from { opacity: 0; transform: scale(0.95) translateY(10px); }
          to { opacity: 1; transform: scale(1) translateY(0); }
        }

        .modal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 18px 24px 14px;
          border-bottom: 1px solid var(--ivory-ghost);
          flex-shrink: 0;
        }

        .modal-header-title {
          font-family: var(--font-display);
          font-size: 18px;
          font-weight: 400;
          color: var(--ivory);
        }

        .modal-close {
          width: 28px; height: 28px;
          border-radius: 4px;
          border: 1px solid var(--ivory-faint);
          background: transparent;
          color: var(--ivory-dim);
          cursor: pointer;
          font-size: 16px;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.15s ease;
        }

        .modal-close:hover {
          color: var(--ivory);
          border-color: var(--ivory-dim);
        }

        .modal-body {
          flex: 1;
          overflow-y: auto;
          padding: 20px 24px 24px;
        }

        .modal-body::-webkit-scrollbar { width: 3px; }
        .modal-body::-webkit-scrollbar-track { background: transparent; }
        .modal-body::-webkit-scrollbar-thumb { background: var(--ivory-ghost); border-radius: 2px; }

        .more-label {
          font-size: 10px;
          letter-spacing: 2.5px;
          text-transform: uppercase;
          color: var(--ivory-dim);
          margin-bottom: 8px;
          font-weight: 400;
          opacity: 0.7;
        }

        .more-justification {
          font-size: 13px;
          font-weight: 300;
          color: rgba(240, 236, 228, 0.65);
          line-height: 1.7;
          margin-bottom: 22px;
        }

        .more-palette-row {
          display: flex;
          gap: 10px;
          margin-bottom: 22px;
          flex-wrap: wrap;
        }

        .more-swatch {
          display: flex;
          align-items: center;
          gap: 7px;
        }

        .more-swatch-dot {
          width: 18px;
          height: 18px;
          border-radius: 4px;
          border: 1px solid var(--ivory-ghost);
          flex-shrink: 0;
        }

        .more-swatch-hex {
          font-size: 11px;
          font-weight: 300;
          color: var(--ivory-dim);
          font-family: var(--font-body);
        }

        .more-download-btn {
          width: 100%;
          padding: 12px;
          background: rgba(240, 236, 228, 0.04);
          border: 1px solid var(--ivory-faint);
          border-radius: 6px;
          color: var(--ivory-dim);
          font-family: var(--font-body);
          font-size: 12px;
          font-weight: 400;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          transition: all 0.2s ease;
        }

        .more-download-btn:hover:not(:disabled) {
          color: var(--ivory);
          border-color: var(--ivory-dim);
          background: rgba(240, 236, 228, 0.06);
        }

        .more-download-btn:disabled {
          opacity: 0.25;
          cursor: not-allowed;
        }

        /* --- Card divider --- */
        .card-divider {
          height: 1px;
          background: var(--ivory-ghost);
          margin: 0;
        }

        /* --- Input section --- */
        .card-input {
          padding: 24px 28px 20px;
        }

        /* --- Model pill (floating above card, centered) --- */
        .player-card-wrapper {
          position: relative;
          width: 480px;
          max-width: 100%;
        }

        .social-row {
          display: flex;
          justify-content: center;
          gap: 16px;
          margin-top: 14px;
        }

        .social-icon {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 32px;
          height: 32px;
          border-radius: 50%;
          background: rgba(26, 24, 22, 0.6);
          border: 1px solid var(--ivory-ghost);
          color: var(--ivory-dim);
          transition: color 0.2s, border-color 0.2s, background 0.2s;
          text-decoration: none;
        }

        .social-icon:hover {
          color: var(--ivory);
          border-color: var(--ivory-dim);
          background: rgba(26, 24, 22, 0.85);
        }

        .model-pill {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 5px 14px;
          margin-bottom: 16px;
          background: rgba(26, 24, 22, 0.85);
          backdrop-filter: blur(16px);
          border: 1px solid var(--ivory-ghost);
          border-radius: 16px;
          cursor: pointer;
          font-family: var(--font-body);
          font-size: 11px;
          font-weight: 400;
          color: var(--ivory-dim);
          transition: all 0.2s ease;
          white-space: nowrap;
        }

        .model-pill:hover {
          border-color: var(--ivory-faint);
          color: var(--ivory);
        }

        .model-pill-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: var(--c4);
        }

        /* --- Model dialog options --- */
        .model-option {
          width: 100%;
          padding: 14px 16px;
          background: transparent;
          border: 1px solid var(--ivory-faint);
          border-radius: 8px;
          cursor: pointer;
          text-align: left;
          font-family: var(--font-body);
          transition: all 0.2s ease;
          display: flex;
          flex-direction: column;
          gap: 6px;
          margin-bottom: 8px;
        }

        .model-option:last-child { margin-bottom: 0; }

        .model-option:hover:not(:disabled):not(.disabled) {
          border-color: var(--ivory-dim);
          background: var(--ivory-ghost);
        }

        .model-option.active {
          border-color: var(--c1);
          background: rgba(240, 236, 228, 0.04);
        }

        .model-option.disabled {
          opacity: 0.45;
          cursor: not-allowed;
        }

        .model-name {
          font-size: 13px;
          font-weight: 500;
          color: var(--ivory);
        }

        .model-desc-long {
          font-size: 11px;
          font-weight: 300;
          color: var(--ivory-dim);
          line-height: 1.5;
        }

        .model-local-link {
          margin-top: 4px;
        }

        .custom-llm-config {
          display: flex;
          flex-direction: column;
          gap: 8px;
          margin-top: 10px;
          padding-top: 10px;
          border-top: 1px solid var(--ivory-ghost);
        }

        .custom-llm-providers {
          display: flex;
          gap: 0;
          border: 1px solid var(--ivory-ghost);
          border-radius: 4px;
          overflow: hidden;
        }

        .custom-llm-provider-tab {
          flex: 1;
          padding: 7px 8px;
          border: none;
          border-right: 1px solid var(--ivory-ghost);
          background: transparent;
          color: var(--ivory-dim);
          font-family: var(--font-body);
          font-size: 11px;
          font-weight: 400;
          cursor: pointer;
          transition: all 0.15s ease;
        }

        .custom-llm-provider-tab:last-child { border-right: none; }

        .custom-llm-provider-tab:hover {
          color: var(--ivory);
          background: var(--ivory-ghost);
        }

        .custom-llm-provider-tab.active {
          background: rgba(240, 236, 228, 0.08);
          color: var(--ivory);
          font-weight: 500;
        }

        .custom-llm-select {
          padding: 8px 12px;
          background: rgba(240, 236, 228, 0.04);
          border: 1px solid var(--ivory-faint);
          border-radius: 4px;
          color: var(--ivory);
          font-size: 12px;
          font-family: var(--font-body);
          font-weight: 300;
          outline: none;
          cursor: pointer;
          transition: border-color 0.2s ease;
          -webkit-appearance: none;
          appearance: none;
          background-image: url("data:image/svg+xml,%3Csvg width='10' height='6' viewBox='0 0 10 6' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1L5 5L9 1' stroke='%23f0ece4' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");
          background-repeat: no-repeat;
          background-position: right 12px center;
          padding-right: 32px;
        }

        .custom-llm-select:focus {
          border-color: var(--ivory-dim);
        }

        .custom-llm-select option {
          background: #1a1816;
          color: var(--ivory);
        }

        .custom-llm-input {
          padding: 8px 12px;
          background: rgba(240, 236, 228, 0.04);
          border: 1px solid var(--ivory-faint);
          border-radius: 4px;
          color: var(--ivory);
          font-size: 12px;
          font-family: var(--font-body);
          font-weight: 300;
          outline: none;
          transition: border-color 0.2s ease;
        }

        .custom-llm-input:focus {
          border-color: var(--ivory-dim);
        }

        .custom-llm-input::placeholder {
          color: var(--ivory-dim);
          font-style: italic;
        }

        .custom-llm-note {
          font-size: 10px;
          font-weight: 300;
          color: var(--ivory-dim);
          opacity: 0.7;
          line-height: 1.4;
        }

        .custom-llm-done {
          padding: 8px 16px;
          border-radius: 4px;
          border: 1px solid var(--c1);
          background: rgba(240, 236, 228, 0.06);
          color: var(--ivory);
          font-family: var(--font-body);
          font-size: 12px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .custom-llm-done:hover:not(:disabled) {
          background: rgba(240, 236, 228, 0.12);
          border-color: var(--c4);
        }

        .custom-llm-done:disabled {
          opacity: 0.35;
          cursor: not-allowed;
        }

        .model-local-link a {
          font-size: 11px;
          color: var(--c4);
          text-decoration: none;
          border-bottom: 1px solid rgba(232, 201, 160, 0.3);
          transition: border-color 0.2s;
        }

        .model-local-link a:hover {
          border-color: var(--c4);
        }

        .card-brand {
          font-family: var(--font-display);
          font-size: 32px;
          font-weight: 400;
          line-height: 1;
          color: var(--ivory);
          margin-bottom: 8px;
        }

        .card-sub {
          font-size: 13px;
          font-weight: 300;
          color: var(--ivory-dim);
          line-height: 1.5;
          margin-bottom: 24px;
          max-width: 380px;
        }

        .input-label {
          font-size: 10px;
          letter-spacing: 2.5px;
          text-transform: uppercase;
          color: var(--ivory-dim);
          margin-bottom: 10px;
          font-weight: 400;
          opacity: 0.6;
        }

        .vibe-input-container {
          display: flex;
          gap: 8px;
          width: 100%;
        }

        .vibe-input {
          flex: 1;
          padding: 12px 16px;
          background: rgba(240, 236, 228, 0.04);
          border: 1px solid var(--ivory-faint);
          border-radius: 6px;
          color: var(--ivory);
          font-size: 13px;
          font-family: var(--font-body);
          font-weight: 300;
          outline: none;
          transition: border-color 0.3s ease;
        }

        .vibe-input:focus {
          border-color: var(--ivory-dim);
        }

        .vibe-input::placeholder {
          color: var(--ivory-dim);
          font-style: italic;
        }

        .vibe-submit {
          padding: 12px 14px;
          background: rgba(240, 236, 228, 0.04);
          border: 1px solid var(--ivory-faint);
          border-radius: 6px;
          color: var(--ivory-dim);
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s ease;
        }

        .vibe-submit:hover:not(:disabled) {
          color: var(--ivory);
          border-color: var(--ivory-dim);
        }

        .vibe-submit:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .spinner {
          width: 16px; height: 16px;
          border: 1.5px solid transparent;
          border-top-color: var(--ivory);
          border-radius: 50%;
          animation: spin 0.6s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        /* --- Presets section --- */
        .card-presets {
          padding: 16px 28px 24px;
        }

        .section-label {
          font-size: 10px;
          letter-spacing: 3px;
          text-transform: uppercase;
          color: var(--ivory-dim);
          margin-bottom: 12px;
          font-weight: 400;
          opacity: 0.6;
        }

        .preset-carousel {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .preset-strip-loading {
          color: var(--ivory-dim);
          font-size: 12px;
        }

        .preset-carousel-items {
          display: flex;
          gap: 6px;
          transition: opacity 0.3s ease;
          flex-wrap: wrap;
        }

        .preset-carousel-items.fading {
          opacity: 0;
        }

        .preset-dots {
          display: flex;
          gap: 6px;
        }

        .preset-dot {
          width: 5px; height: 5px;
          border-radius: 50%;
          border: none;
          background: var(--ivory-faint);
          cursor: pointer;
          padding: 0;
          transition: all 0.3s ease;
        }

        .preset-dot.active {
          background: var(--ivory-dim);
          transform: scale(1.3);
        }

        .preset-dot:hover {
          background: var(--ivory-dim);
        }

        .preset-chip {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 0;
          padding-left: 14px;
          background: transparent;
          border: 1px solid var(--ivory-faint);
          border-radius: 20px;
          color: var(--ivory-dim);
          cursor: pointer;
          font-family: var(--font-body);
          font-size: 11px;
          font-weight: 400;
          white-space: nowrap;
          transition: all 0.2s ease;
          flex-shrink: 0;
          height: 34px;
        }

        .preset-chip:hover:not(.loading) {
          color: var(--ivory);
          border-color: var(--ivory-dim);
          background: var(--ivory-ghost);
        }

        .preset-chip.loading {
          pointer-events: none;
          border-color: var(--c4);
          background: rgba(240, 236, 228, 0.04);
        }

        .preset-chip.disabled {
          opacity: 0.3;
          pointer-events: none;
        }

        /* --- Generating dots animation --- */
        .generating-dots {
          display: inline-flex;
          align-items: center;
          gap: 3px;
        }

        .generating-dots span {
          width: 5px;
          height: 5px;
          border-radius: 50%;
          background: var(--c4);
          animation: dotPulse 1.2s ease-in-out infinite;
        }

        .generating-dots span:nth-child(2) { animation-delay: 0.2s; }
        .generating-dots span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes dotPulse {
          0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
          40% { opacity: 1; transform: scale(1.1); }
        }

        /* --- Generating banner in now-playing --- */
        .generating-banner {
          display: flex;
          align-items: center;
          gap: 8px;
          padding-bottom: 8px;
          margin-bottom: 4px;
          font-size: 11px;
          font-weight: 400;
          color: var(--c4);
          letter-spacing: 1px;
          text-transform: uppercase;
          animation: fadeUp 0.3s ease;
        }

        .generating-banner .generating-dots span {
          width: 4px;
          height: 4px;
          background: var(--c4);
        }

        /* --- iOS silent-mode pill above the player --- */
        .ios-silent-banner {
          display: flex;
          align-items: center;
          gap: 8px;
          width: fit-content;
          max-width: 100%;
          margin: 0 auto 12px;
          padding: 6px 6px 6px 12px;
          background: rgba(26, 24, 22, 0.85);
          border: 1px solid var(--ivory-ghost);
          border-radius: 999px;
          font-family: var(--font-body);
          font-size: 11px;
          line-height: 1;
          color: var(--ivory-dim);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          animation: fadeUp 0.4s ease;
        }

        .ios-silent-banner-icon {
          flex-shrink: 0;
          color: var(--c4);
        }

        .ios-silent-banner-close {
          flex-shrink: 0;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 20px;
          height: 20px;
          background: transparent;
          border: none;
          color: var(--ivory-dim);
          font-size: 14px;
          line-height: 1;
          cursor: pointer;
          border-radius: 50%;
          transition: color 0.2s ease, background 0.2s ease;
        }

        .ios-silent-banner-close:hover {
          color: var(--ivory);
          background: rgba(240, 236, 228, 0.06);
        }

        .preset-chip-icon { font-size: 14px; line-height: 1; }
        .preset-chip-name { font-weight: 400; }

        .preset-chip-enqueue {
          display: flex;
          align-items: center;
          justify-content: center;
          align-self: stretch;
          width: 32px;
          margin-left: 8px;
          border-left: 1px solid var(--ivory-ghost);
          font-size: 18px;
          font-weight: 300;
          line-height: 1;
          color: var(--ivory-faint);
          transition: all 0.15s ease;
          border-radius: 0 20px 20px 0;
        }

        .preset-chip-enqueue:hover {
          color: var(--ivory);
          background: var(--ivory-ghost);
        }

        .vibe-enqueue {
          background: transparent;
          border-color: var(--ivory-ghost);
        }

        .vibe-enqueue:hover:not(:disabled) {
          border-color: var(--ivory-faint);
        }

        /* --- Playlist section --- */
        .card-playlist {
          padding: 16px 28px 20px;
        }

        .playlist-scroll {
          max-height: 240px;
          overflow-y: auto;
        }

        .playlist-scroll::-webkit-scrollbar { width: 3px; }
        .playlist-scroll::-webkit-scrollbar-track { background: transparent; }
        .playlist-scroll::-webkit-scrollbar-thumb { background: var(--ivory-ghost); border-radius: 2px; }

        .queue-track {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px 10px;
          border-radius: 6px;
          cursor: pointer;
          transition: background 0.15s ease;
        }

        .queue-track:hover {
          background: var(--ivory-ghost);
        }

        .queue-track.active-track {
          background: rgba(240, 236, 228, 0.04);
          border-left: 2px solid var(--c1);
          padding-left: 8px;
        }

        .playing-indicator {
          display: flex;
          align-items: flex-end;
          gap: 1.5px;
          height: 12px;
          flex-shrink: 0;
        }

        .playing-indicator span {
          width: 2px;
          background: var(--c1);
          border-radius: 1px;
          animation: eqBar 0.8s ease-in-out infinite;
        }

        .playing-indicator span:nth-child(1) { height: 40%; animation-delay: 0s; }
        .playing-indicator span:nth-child(2) { height: 70%; animation-delay: 0.15s; }
        .playing-indicator span:nth-child(3) { height: 50%; animation-delay: 0.3s; }

        @keyframes eqBar {
          0%, 100% { transform: scaleY(0.4); }
          50% { transform: scaleY(1); }
        }

        .queue-track-colors {
          display: flex;
          gap: 2px;
          flex-shrink: 0;
        }

        .q-dot {
          width: 4px; height: 4px;
          border-radius: 50%;
          transition: background-color 0.5s ease;
        }

        .queue-track-info {
          flex: 1;
          min-width: 0;
          display: flex;
          flex-direction: column;
          gap: 1px;
        }

        .queue-track-title {
          font-family: var(--font-display);
          font-size: 13px;
          font-weight: 400;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .queue-track-vibe {
          font-size: 10px;
          font-weight: 300;
          color: var(--ivory-dim);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .queue-track-remove {
          width: 20px; height: 20px;
          border-radius: 3px;
          border: none;
          background: transparent;
          color: var(--ivory-dim);
          cursor: pointer;
          font-size: 14px;
          opacity: 0;
          transition: opacity 0.15s ease;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .queue-track:hover .queue-track-remove { opacity: 1; }
        .queue-track-remove:hover { color: #c45; }

        .queue-track-pending {
          opacity: 0.55;
        }
        .queue-track-error {
          opacity: 0.65;
        }
        .queue-track-error-icon {
          width: 22px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #c45;
          font-weight: 700;
          font-size: 13px;
          flex-shrink: 0;
        }
        .queue-track-spinner {
          width: 22px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }
        .spinner-sm {
          width: 12px;
          height: 12px;
          border: 2px solid var(--ivory-ghost);
          border-top-color: var(--ivory);
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }

        /* --- Param panel --- */
        .param-panel {
          position: fixed;
          right: 0; top: 0; bottom: 0;
          width: 340px;
          background: rgba(26, 24, 22, 0.92);
          border-left: 1px solid var(--ivory-ghost);
          backdrop-filter: blur(30px);
          -webkit-backdrop-filter: blur(30px);
          z-index: 90;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          animation: slideIn 0.25s ease;
        }

        @keyframes slideIn {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }

        .param-panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px 20px 14px;
          border-bottom: 1px solid var(--ivory-ghost);
        }

        .param-panel-header h3 {
          font-family: var(--font-display);
          font-size: 18px;
          font-weight: 400;
          color: var(--ivory);
        }

        .param-close {
          width: 24px; height: 24px;
          border-radius: 2px;
          border: 1px solid var(--ivory-faint);
          background: transparent;
          color: var(--ivory-dim);
          cursor: pointer;
          font-size: 14px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .param-close:hover {
          color: var(--ivory);
          border-color: var(--ivory-dim);
        }

        .param-categories {
          display: flex;
          gap: 2px;
          padding: 10px 10px 6px;
          border-bottom: 1px solid var(--ivory-ghost);
        }

        .param-cat-btn {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 3px;
          padding: 7px 4px;
          border-radius: 3px;
          border: 1px solid transparent;
          background: transparent;
          color: var(--ivory-dim);
          font-size: 9px;
          font-family: var(--font-body);
          cursor: pointer;
          transition: all 0.15s ease;
          letter-spacing: 0.5px;
          text-transform: uppercase;
        }

        .param-cat-btn:hover { color: var(--ivory); }

        .param-cat-btn.active {
          border-color: var(--ivory-faint);
          color: var(--ivory);
        }

        .param-list {
          flex: 1;
          overflow-y: auto;
          padding: 12px 16px;
        }

        .param-list::-webkit-scrollbar { width: 2px; }
        .param-list::-webkit-scrollbar-track { background: transparent; }
        .param-list::-webkit-scrollbar-thumb { background: var(--ivory-ghost); }

        .param-row { margin-bottom: 14px; }

        .param-label {
          display: block;
          font-size: 10px;
          font-weight: 400;
          letter-spacing: 1px;
          text-transform: uppercase;
          color: var(--ivory-dim);
          margin-bottom: 6px;
        }

        .param-options {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
        }

        .param-option {
          padding: 4px 10px;
          border-radius: 2px;
          border: 1px solid var(--ivory-ghost);
          background: transparent;
          color: var(--ivory-dim);
          font-size: 11px;
          font-family: var(--font-body);
          font-weight: 300;
          cursor: pointer;
          transition: all 0.15s ease;
          white-space: nowrap;
        }

        .param-option:hover {
          border-color: var(--ivory-faint);
          color: var(--ivory);
        }

        .param-option.active {
          border-color: var(--c1);
          color: var(--ivory);
          font-weight: 400;
        }

        .param-modified-dot {
          display: inline-block;
          width: 5px;
          height: 5px;
          border-radius: 50%;
          background: var(--c4);
          margin-left: 5px;
          vertical-align: middle;
        }

        .param-apply-footer {
          display: flex;
          gap: 8px;
          padding: 12px 16px;
          border-top: 1px solid var(--ivory-ghost);
          background: rgba(20, 18, 16, 0.95);
          flex-shrink: 0;
          animation: fadeUp 0.2s ease;
        }

        .param-apply-btn {
          flex: 1;
          padding: 10px 16px;
          border-radius: 4px;
          border: 1px solid var(--c1);
          background: rgba(240, 236, 228, 0.06);
          color: var(--ivory);
          font-family: var(--font-body);
          font-size: 12px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
          letter-spacing: 0.5px;
        }

        .param-apply-btn:hover:not(:disabled) {
          background: rgba(240, 236, 228, 0.12);
          border-color: var(--c4);
        }

        .param-apply-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .param-reset-btn {
          padding: 10px 14px;
          border-radius: 4px;
          border: 1px solid var(--ivory-ghost);
          background: transparent;
          color: var(--ivory-dim);
          font-family: var(--font-body);
          font-size: 11px;
          font-weight: 400;
          cursor: pointer;
          transition: all 0.15s ease;
        }

        .param-reset-btn:hover:not(:disabled) {
          color: var(--ivory);
          border-color: var(--ivory-faint);
        }

        .param-reset-btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        /* --- Error toast --- */
        .warning-toast {
          position: fixed;
          bottom: 28px;
          left: 0;
          right: 0;
          margin: 0 auto;
          width: fit-content;
          z-index: 300;
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 12px 20px;
          background: rgba(160, 120, 20, 0.92);
          border: 1px solid rgba(255, 200, 50, 0.3);
          border-radius: 8px;
          backdrop-filter: blur(16px);
          color: #fff;
          font-size: 13px;
          cursor: pointer;
          animation: toast-in 0.3s ease;
        }
        .warning-toast-icon {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          border: 1.5px solid rgba(255, 220, 100, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          font-weight: 600;
          flex-shrink: 0;
        }
        .error-toast {
          position: fixed;
          bottom: 28px;
          left: 0;
          right: 0;
          margin: 0 auto;
          width: fit-content;
          z-index: 300;
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 12px 20px;
          background: rgba(180, 40, 40, 0.92);
          border: 1px solid rgba(255, 100, 100, 0.3);
          border-radius: 8px;
          backdrop-filter: blur(16px);
          color: #fdd;
          font-family: var(--font-body);
          font-size: 13px;
          font-weight: 400;
          cursor: pointer;
          animation: fadeUp 0.3s ease;
          max-width: 480px;
        }

        .error-toast-icon {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          border: 1.5px solid rgba(255, 150, 150, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          font-weight: 600;
          flex-shrink: 0;
        }

        /* --- Responsive --- */
        @media (max-width: 768px) {
          .app-wrapper { padding: 24px 12px; }
          .player-card { border-radius: 8px; }
          .np-body { padding: 16px 20px 14px; }
          .np-title { font-size: 20px; }
          .card-input { padding: 20px 20px 16px; }
          .card-presets { padding: 12px 20px 20px; }
          .card-queue { padding: 12px 20px 16px; }
          .brand-mark { position: absolute; top: 16px; left: 16px; font-size: 22px; }
          .param-panel { width: 100%; }
        }
      `}</style>

      <ParticleVisualizer />
      <div className="grain" />

      <div className="app-wrapper">
        {!ANONYMOUS && (
          <Link to="/" className="brand-mark reveal r1">
            Latent<em>Score</em>
          </Link>
        )}

        {/* Model pill */}
        <button className="model-pill" onClick={() => setShowModelDialog(true)}>
          <span className="model-pill-dot" />
          {selectedModel === "custom"
            ? "Custom LLM"
            : selectedModel === "expressive"
              ? "Expressive"
              : selectedModel === "fast_heavy"
                ? "Fast (Heavy)"
                : "Fast"}
        </button>

        <div className="player-card-wrapper">
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
              className="ios-silent-banner-close"
              onClick={() => setIosBannerDismissed(true)}
              aria-label="Dismiss"
            >
              ×
            </button>
          </div>
        )}
        <div className="player-card">
          {/* --- Now Playing (compact) --- */}
          {activeSessionId && (
            <div className="np-section reveal r1">
              <div className="np-palette">
                {activePalette.map((c, i) => (
                  <div key={i} className="np-palette-strip" style={{ backgroundColor: c }} />
                ))}
              </div>

              <div className="np-body">
                {isGenerating && (
                  <div className="generating-banner">
                    <span className="generating-dots">
                      <span /><span /><span />
                    </span>
                    <span>{generatingLabel}</span>
                  </div>
                )}
                <div className="np-top-row">
                  <div className="np-info">
                    <div className="np-title">{activeTitle}</div>
                    <div className="np-vibe">{activeVibe}</div>
                  </div>
                </div>

                <div className="np-controls">
                  <button
                    className={`play-btn ${isPlaying ? "playing" : ""}`}
                    onClick={isPlaying ? stopStream : startStream}
                    title={isPlaying ? "Stop" : "Play"}
                    disabled={isGenerating}
                  >
                    {isGenerating ? (
                      <span className="spinner" />
                    ) : isPlaying ? (
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <rect x="6" y="4" width="4" height="16" rx="1" />
                        <rect x="14" y="4" width="4" height="16" rx="1" />
                      </svg>
                    ) : (
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M8 5v14l11-7z" />
                      </svg>
                    )}
                  </button>

                  {/* Skip / Next */}
                  <button
                    className="ctrl-btn"
                    onClick={handleSkip}
                    title={playlist.length > 0 ? "Next Track" : "Stop"}
                    disabled={isGenerating || (!isPlaying && playlist.length === 0)}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M5 4l10 8-10 8V4z" />
                      <rect x="17" y="4" width="2.5" height="16" rx="0.5" />
                    </svg>
                  </button>

                  {/* Tune Parameters */}
                  <button
                    className={`ctrl-btn${!showParamPanel ? ' param-glow' : ''}`}
                    onClick={toggleParamPanel}
                    title="Tune Parameters"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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

                  {/* Stream / Render toggle */}
                  <button
                    className="mode-toggle"
                    onClick={() => setStreamMode(!streamMode)}
                    title={streamMode ? "Live stream (phase-continuous WebSocket)" : "Render full track (60s fetch-and-play)"}
                  >
                    <span className={`mode-toggle-dot ${streamMode ? "stream" : "fetch"}`} />
                    {streamMode ? "Live" : "Render"}
                  </button>

                  {/* Loop toggle */}
                  <button
                    className={`mode-toggle ${loopMode ? "active" : ""}`}
                    onClick={() => setLoopMode(!loopMode)}
                    title={loopMode ? "Looping current track" : "Play once (60s), then advance"}
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="17 1 21 5 17 9" />
                      <path d="M3 11V9a4 4 0 0 1 4-4h14" />
                      <polyline points="7 23 3 19 7 15" />
                      <path d="M21 13v2a4 4 0 0 1-4 4H3" />
                    </svg>
                    Loop
                  </button>

                  {/* Spacer */}
                  <div style={{ flex: 1 }} />

                  {/* More button */}
                  <button
                    className="ctrl-btn"
                    onClick={() => setShowMore(!showMore)}
                    title="More info"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
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

          {/* --- Input section --- */}
          <div className="card-input reveal r2">
            {!activeSessionId ? (
              <>
                <div className="card-brand">Live Performance</div>
                <div className="card-sub">
                  Describe a mood, scene, or feeling and get unique ambient music generated in real-time.
                </div>
                {isGenerating && (
                  <div className="generating-banner" style={{ marginBottom: 12 }}>
                    <span className="generating-dots">
                      <span /><span /><span />
                    </span>
                    <span>{generatingLabel}</span>
                  </div>
                )}
              </>
            ) : (
              <div className="input-label">generate another</div>
            )}
            <VibeInput />
          </div>

          <div className="card-divider" />

          {/* --- Presets --- */}
          <div className="card-presets reveal r3">
            <div className="section-label">
              {activeSessionId ? "or try a preset" : "or pick a preset"}
            </div>
            <PresetGrid />
          </div>

          {/* --- Playlist --- */}
          {playlist.length > 0 && (
            <>
              <div className="card-divider" />
              <div className="card-playlist reveal r4">
                <div className="section-label">playlist</div>
                <div className="playlist-scroll">
                  {playlist.map((track, i) => {
                    const isCurrent = track.status === "ready" && track.sessionId === activeSessionId;
                    const isPending = track.status === "pending";
                    const isError = track.status === "error";
                    return (
                      <div
                        key={track.id}
                        className={`queue-track ${isCurrent ? "active-track" : ""} ${isPending ? "queue-track-pending" : ""} ${isError ? "queue-track-error" : ""}`}
                        onClick={() => handlePlayTrack(track)}
                        style={isPending ? { cursor: "default" } : undefined}
                      >
                        {isCurrent && isPlaying ? (
                          <div className="playing-indicator">
                            <span /><span /><span />
                          </div>
                        ) : isPending ? (
                          <div className="queue-track-spinner">
                            <span className="spinner-sm" />
                          </div>
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
                            {isPending ? "Generating..." : isError ? "Failed" : track.title}
                          </span>
                          <span className="queue-track-vibe">{track.vibe}</span>
                        </div>
                        <button
                          className="queue-track-remove"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeFromPlaylist(i);
                          }}
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

        {/* Social links */}
        <div className="social-row">
          <a href="https://www.linkedin.com/in/prabal1997" target="_blank" rel="noopener noreferrer" className="social-icon" title="LinkedIn">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
            </svg>
          </a>
          <a href="https://x.com/prabal_" target="_blank" rel="noopener noreferrer" className="social-icon" title="X">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
            </svg>
          </a>
          <a href="https://github.com/prabal-rje/latentscore" target="_blank" rel="noopener noreferrer" className="social-icon" title="GitHub">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
            </svg>
          </a>
        </div>
        </div>
      </div>

      <ParamPanel />

      {/* --- Warning Toast --- */}
      {warningMessage && (
        <div className="warning-toast" onClick={() => setWarning(null)}>
          <span className="warning-toast-icon">!</span>
          <span>{warningMessage}</span>
        </div>
      )}

      {/* --- Error Toast --- */}
      {errorMessage && (
        <div className="error-toast" onClick={() => setError(null)}>
          <span className="error-toast-icon">!</span>
          <span>{errorMessage}</span>
        </div>
      )}

      {/* --- Model Dialog --- */}
      {showModelDialog && (
        <div className="modal-overlay" onClick={() => setShowModelDialog(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <span className="modal-header-title">Choose Model</span>
              <button className="modal-close" onClick={() => setShowModelDialog(false)}>
                &times;
              </button>
            </div>
            <div className="modal-body">
              <button
                className={`model-option ${selectedModel === "fast_heavy" ? "active" : ""}`}
                onClick={() => { setSelectedModel("fast_heavy"); setShowModelDialog(false); }}
              >
                <span className="model-name">Fast (Heavy)</span>
                <span className="model-desc-long">
                  CLAP audio-grounded matching. Matches your vibe text against how each
                  pre-computed configuration actually sounds. Best quality, instant results.
                </span>
              </button>

              <button
                className={`model-option ${selectedModel === "fast" ? "active" : ""}`}
                onClick={() => { setSelectedModel("fast"); setShowModelDialog(false); }}
              >
                <span className="model-name">Fast</span>
                <span className="model-desc-long">
                  Embedding-based matching. Maps your vibe to the closest pre-computed
                  configuration using semantic similarity. Instant results.
                </span>
              </button>

              <button
                className={`model-option ${selectedModel === "expressive" ? "active" : ""} ${!expressiveAvailable ? "disabled" : ""}`}
                onClick={() => {
                  if (expressiveAvailable) { setSelectedModel("expressive"); setShowModelDialog(false); }
                }}
                disabled={!expressiveAvailable}
              >
                <span className="model-name">Expressive</span>
                <span className="model-desc-long">
                  {expressiveAvailable
                    ? "Fine-tuned Gemma 3 1B running on CPU. Generates unique configurations with richer, more nuanced choices. Takes 2-5 seconds."
                    : "Fine-tuned Gemma 3 1B running on CPU. Richer, more nuanced parameter choices. Only available when running locally."}
                </span>
                {!expressiveAvailable && !ANONYMOUS && (
                  <span className="model-local-link">
                    <a href="https://github.com/prabal-rje/latentscore" target="_blank" rel="noopener noreferrer">
                      Run locally to enable &rarr;
                    </a>
                  </span>
                )}
              </button>

              <div
                className={`model-option ${selectedModel === "custom" ? "active" : ""}`}
                onClick={() => setSelectedModel("custom")}
                role="button"
                tabIndex={0}
              >
                <span className="model-name">Custom LLM</span>
                <span className="model-desc-long">
                  Bring your own API key. Uses any LLM to interpret your vibe and
                  craft a unique configuration with full creative control.
                </span>

                {selectedModel === "custom" && (
                  <div className="custom-llm-config" onClick={(e) => e.stopPropagation()}>
                    {/* Provider tabs */}
                    <div className="custom-llm-providers">
                      {Object.entries(PROVIDER_MODELS).map(([key, p]) => (
                        <button
                          key={key}
                          className={`custom-llm-provider-tab ${activeProvider === key ? "active" : ""}`}
                          onClick={() => {
                            setActiveProvider(key);
                            // Auto-select first model of this provider
                            setCustomModelId(p.models[0].id);
                            setCustomModelInput("");
                          }}
                        >
                          {p.label}
                        </button>
                      ))}
                    </div>

                    {/* Model dropdown */}
                    <select
                      className="custom-llm-select"
                      value={customModelInput ? "__custom__" : customModelId}
                      onChange={(e) => {
                        if (e.target.value === "__custom__") {
                          setCustomModelInput(customModelId);
                        } else {
                          setCustomModelInput("");
                          setCustomModelId(e.target.value);
                        }
                      }}
                    >
                      {PROVIDER_MODELS[activeProvider]?.models.map((m) => (
                        <option key={m.id} value={m.id}>{m.label}</option>
                      ))}
                      <option value="__custom__">Custom...</option>
                    </select>
                    {customModelInput !== "" && (
                      <input
                        type="text"
                        className="custom-llm-input"
                        placeholder="Model ID"
                        value={customModelInput}
                        autoFocus
                        onChange={(e) => {
                          setCustomModelInput(e.target.value);
                          if (e.target.value.trim()) setCustomModelId(e.target.value.trim());
                        }}
                      />
                    )}

                    {/* API key for active provider */}
                    <input
                      type="password"
                      className="custom-llm-input"
                      placeholder={`${PROVIDER_MODELS[activeProvider]?.label ?? "Provider"} API Key`}
                      value={providerApiKeys[activeProvider] || ""}
                      onChange={(e) => setProviderApiKey(activeProvider, e.target.value)}
                    />
                    <span className="custom-llm-note">
                      Key stored in your browser only. Sent directly to {PROVIDER_MODELS[activeProvider]?.label ?? "the provider"}.
                    </span>
                    <button
                      className="custom-llm-done"
                      onClick={() => setShowModelDialog(false)}
                      disabled={!customModelId || !(providerApiKeys[activeProvider])}
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

      {/* --- Info Modal --- */}
      {showMore && activeSessionId && (
        <div className="modal-overlay" onClick={() => setShowMore(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <span className="modal-header-title">About This Piece</span>
              <button className="modal-close" onClick={() => setShowMore(false)}>
                &times;
              </button>
            </div>
            <div className="modal-body">
              <button
                className="more-download-btn"
                onClick={handleDownload}
                disabled={!hasRecordedAudio()}
                title={hasRecordedAudio() ? "Download as WAV" : "Play first to record audio"}
                style={{ marginBottom: 22 }}
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

export default DemoPage;
