import { create } from "zustand";
import type { PlaylistTrack, ModelChoice } from "./types";

function normalizeHex(hex: string): string {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex.trim());
  if (!m) return "#000000";
  return `#${m[1].toLowerCase()}${m[2].toLowerCase()}${m[3].toLowerCase()}`;
}

function normalizePalette(palette: string[]): string[] {
  return palette.map(normalizeHex);
}

interface AppState {
  activeSessionId: string | null;
  activeTitle: string;
  activeJustification: string;
  activePalette: string[];       // currently displayed (animated)
  targetPalette: string[];       // where we're animating toward
  activeConfig: Record<string, unknown>;
  activeVibe: string;
  isPlaying: boolean;
  streamMode: boolean;
  loopMode: boolean;
  autoMode: boolean;
  autoMatchScore: number | null;

  vibeText: string;
  selectedModel: ModelChoice;
  customModelId: string;
  providerApiKeys: Record<string, string>;

  playlist: PlaylistTrack[];
  playlistName: string;
  currentTrackIndex: number;

  showPlaylistPanel: boolean;
  showParamPanel: boolean;
  restreamTrigger: number;
  errorMessage: string | null;
  warningMessage: string | null;
  generatingPhase: "llm" | "evaluating" | "music" | null;

  setActiveSession: (data: {
    sessionId: string;
    title: string;
    justification: string;
    palette: string[];
    config: Record<string, unknown>;
    vibe: string;
  }) => void;
  setPlaying: (playing: boolean) => void;
  setStreamMode: (mode: boolean) => void;
  setLoopMode: (loop: boolean) => void;
  setAutoMode: (auto: boolean) => void;
  setAutoMatchScore: (score: number | null) => void;
  setVibeText: (text: string) => void;
  setSelectedModel: (model: ModelChoice) => void;
  setCustomModelId: (id: string) => void;
  setProviderApiKey: (provider: string, key: string) => void;
  updateConfig: (config: Record<string, unknown>) => void;
  updatePalette: (palette: string[]) => void;
  updateTitle: (title: string) => void;
  commitSessionParams: (
    sessionId: string,
    data: { config: Record<string, unknown>; palette: string[]; title: string }
  ) => void;
  addToPlaylist: (track: PlaylistTrack) => void;
  addPendingTrack: (vibe: string) => string;
  resolvePlaylistTrack: (id: string, data: Omit<PlaylistTrack, "id" | "vibe" | "status" | "duration">) => void;
  failPlaylistTrack: (id: string, errorMessage: string) => void;
  removeFromPlaylist: (index: number) => void;
  setPlaylist: (tracks: PlaylistTrack[], name: string) => void;
  setCurrentTrackIndex: (index: number) => void;
  togglePlaylistPanel: () => void;
  toggleParamPanel: () => void;
  triggerRestream: () => void;
  setError: (msg: string | null) => void;
  setWarning: (msg: string | null) => void;
  setGeneratingPhase: (phase: "llm" | "evaluating" | "music" | null) => void;
}

const DEFAULT_PALETTE = ["#d4a574", "#8b6e55", "#c49670", "#0e0d0b", "#e8c9a0"];

export const useStore = create<AppState>((set) => ({
  activeSessionId: null,
  activeTitle: "",
  activeJustification: "",
  activePalette: DEFAULT_PALETTE,
  targetPalette: DEFAULT_PALETTE,
  activeConfig: {},
  activeVibe: "",
  isPlaying: false,
  streamMode: localStorage.getItem("ls-stream-mode") === "true",
  loopMode: localStorage.getItem("ls-loop-mode") === "true",
  autoMode: false,
  autoMatchScore: null,

  vibeText: "",
  selectedModel: (localStorage.getItem("ls-model") as ModelChoice) || "fast",
  customModelId: localStorage.getItem("ls-custom-model-id") || "gpt-6-luna",
  providerApiKeys: {
    openai: localStorage.getItem("ls-api-key-openai") || "",
    anthropic: localStorage.getItem("ls-api-key-anthropic") || "",
    google: localStorage.getItem("ls-api-key-google") || "",
  },

  playlist: [],
  playlistName: "My Playlist",
  currentTrackIndex: -1,

  showPlaylistPanel: false,
  showParamPanel: false,
  restreamTrigger: 0,
  errorMessage: null,
  warningMessage: null,
  generatingPhase: null,

  setActiveSession: (data) => {
    const norm = normalizePalette(data.palette);
    set({
      activeSessionId: data.sessionId,
      activeTitle: data.title,
      activeJustification: data.justification,
      activePalette: norm,
      targetPalette: norm,
      activeConfig: data.config,
      activeVibe: data.vibe,
    });
  },

  setPlaying: (playing) => set({ isPlaying: playing }),
  setStreamMode: (mode) => {
    localStorage.setItem("ls-stream-mode", String(mode));
    set({ streamMode: mode });
  },
  setLoopMode: (loop) => {
    localStorage.setItem("ls-loop-mode", String(loop));
    set((state) => ({
      loopMode: loop,
      autoMode: loop ? false : state.autoMode,
      autoMatchScore: loop ? null : state.autoMatchScore,
    }));
  },
  setAutoMode: (auto) => {
    if (auto) localStorage.setItem("ls-loop-mode", "false");
    set((state) => ({
      autoMode: auto,
      loopMode: auto ? false : state.loopMode,
      autoMatchScore: auto ? state.autoMatchScore : null,
    }));
  },
  setAutoMatchScore: (score) => set({ autoMatchScore: score }),
  setVibeText: (text) => set({ vibeText: text }),
  setSelectedModel: (model) => {
    localStorage.setItem("ls-model", model);
    set({ selectedModel: model });
  },
  setCustomModelId: (id) => {
    localStorage.setItem("ls-custom-model-id", id);
    set({ customModelId: id });
  },
  setProviderApiKey: (provider, key) => {
    localStorage.setItem(`ls-api-key-${provider}`, key);
    set((state) => ({
      providerApiKeys: { ...state.providerApiKeys, [provider]: key },
    }));
  },
  updateConfig: (config) => set({ activeConfig: config }),
  updatePalette: (palette) => {
    const norm = normalizePalette(palette);
    set({ activePalette: norm, targetPalette: norm });
  },
  updateTitle: (title) => set({ activeTitle: title }),
  commitSessionParams: (sessionId, data) => {
    const norm = normalizePalette(data.palette);
    set((state) => ({
      activeConfig:
        state.activeSessionId === sessionId ? data.config : state.activeConfig,
      activePalette:
        state.activeSessionId === sessionId ? norm : state.activePalette,
      targetPalette:
        state.activeSessionId === sessionId ? norm : state.targetPalette,
      activeTitle:
        state.activeSessionId === sessionId ? data.title : state.activeTitle,
      playlist: state.playlist.map((track) =>
        track.sessionId === sessionId
          ? {
              ...track,
              config: data.config,
              palette: norm,
              title: data.title,
            }
          : track
      ),
    }));
  },

  addToPlaylist: (track) =>
    set((state) => ({ playlist: [...state.playlist, track] })),

  addPendingTrack: (vibe) => {
    const id = `pending-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    set((state) => ({
      playlist: [...state.playlist, {
        id,
        sessionId: "",
        vibe,
        title: "",
        palette: [],
        justification: "",
        config: {},
        duration: 30,
        status: "pending" as const,
      }],
    }));
    return id;
  },

  resolvePlaylistTrack: (id, data) =>
    set((state) => ({
      playlist: state.playlist.map((t) =>
        t.id === id
          ? { ...t, ...data, status: "ready" as const }
          : t
      ),
    })),

  failPlaylistTrack: (id, errorMessage) =>
    set((state) => ({
      playlist: state.playlist.map((t) =>
        t.id === id
          ? { ...t, status: "error" as const, errorMessage }
          : t
      ),
    })),

  removeFromPlaylist: (index) =>
    set((state) => ({
      playlist: state.playlist.filter((_, i) => i !== index),
    })),

  setPlaylist: (tracks, name) => set({ playlist: tracks, playlistName: name }),
  setCurrentTrackIndex: (index) => set({ currentTrackIndex: index }),
  togglePlaylistPanel: () =>
    set((state) => ({ showPlaylistPanel: !state.showPlaylistPanel })),
  toggleParamPanel: () =>
    set((state) => ({ showParamPanel: !state.showParamPanel })),
  triggerRestream: () =>
    set((state) => ({ restreamTrigger: state.restreamTrigger + 1 })),
  setError: (msg) => set({ errorMessage: msg }),
  setWarning: (msg) => set({ warningMessage: msg }),
  setGeneratingPhase: (phase) => set({ generatingPhase: phase }),
}));
