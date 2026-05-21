import { create } from "zustand";
import type { PlaylistTrack, ModelChoice } from "./types";

// Smooth color interpolation for palette transitions
function lerpColor(a: string, b: string, t: number): string {
  const parse = (hex: string) => {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [0, 0, 0];
  };
  const [ar, ag, ab] = parse(a);
  const [br, bg, bb] = parse(b);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const b2 = Math.round(ab + (bb - ab) * t);
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b2.toString(16).padStart(2, "0")}`;
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
  setVibeText: (text: string) => void;
  setSelectedModel: (model: ModelChoice) => void;
  setCustomModelId: (id: string) => void;
  setProviderApiKey: (provider: string, key: string) => void;
  updateConfig: (config: Record<string, unknown>) => void;
  updatePalette: (palette: string[]) => void;
  updateTitle: (title: string) => void;
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

// Palette animation controller
let animationFrame: number | null = null;
function animatePalette() {
  if (animationFrame) cancelAnimationFrame(animationFrame);

  const step = () => {
    const state = useStore.getState();
    const current = state.activePalette;
    const target = state.targetPalette;
    let done = true;

    const next = current.map((c, i) => {
      const t = target[i] ?? c;
      if (c === t) return c;
      done = false;
      return lerpColor(c, t, 0.06); // smooth ~60 frame transition
    });

    if (!done) {
      useStore.setState({ activePalette: next });
      animationFrame = requestAnimationFrame(step);
    } else {
      animationFrame = null;
    }
  };
  animationFrame = requestAnimationFrame(step);
}

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

  vibeText: "",
  selectedModel: (localStorage.getItem("ls-model") as ModelChoice) || "fast",
  customModelId: localStorage.getItem("ls-custom-model-id") || "gpt-5-nano-2025-08-07",
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
    set({
      activeSessionId: data.sessionId,
      activeTitle: data.title,
      activeJustification: data.justification,
      targetPalette: data.palette,
      activeConfig: data.config,
      activeVibe: data.vibe,
    });
    animatePalette();
  },

  setPlaying: (playing) => set({ isPlaying: playing }),
  setStreamMode: (mode) => {
    localStorage.setItem("ls-stream-mode", String(mode));
    set({ streamMode: mode });
  },
  setLoopMode: (loop) => {
    localStorage.setItem("ls-loop-mode", String(loop));
    set({ loopMode: loop });
  },
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
    set({ targetPalette: palette });
    animatePalette();
  },
  updateTitle: (title) => set({ activeTitle: title }),

  addToPlaylist: (track) =>
    set((state) => ({ playlist: [track, ...state.playlist] })),

  addPendingTrack: (vibe) => {
    const id = `pending-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    set((state) => ({
      playlist: [{
        id,
        sessionId: "",
        vibe,
        title: "",
        palette: [],
        justification: "",
        config: {},
        duration: 30,
        status: "pending" as const,
      }, ...state.playlist],
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
