/**
 * Browser-native zero-backend engine coordinating instant preset lookups,
 * the MiniLM vibe Web Worker, and the Pyodide NumPy synth Web Worker.
 */

import type { Capabilities, GenerateResponse, ParamOptions, Preset } from "../types";
import {
  PARAM_OPTIONS,
  coerceConfig,
  configToRecord,
  generateJustification,
  generatePalette,
  generateTitle,
  heuristicConfig,
  toInternalSynthConfig,
  type MusicConfig,
  type PrecomputedPreset,
} from "./musicLogic";

export interface EngineStatus {
  vibeStage: string;
  synthStage: string;
  detail: string | null;
  progress: number | null;
}

type StatusListener = (status: EngineStatus) => void;

interface LocalSession {
  sessionId: string;
  vibe: string;
  config: MusicConfig;
}

const localSessions = new Map<string, LocalSession>();
const MAX_LOCAL_SESSIONS = 256;
const WORKER_TIMEOUT_MS = 90_000;

let presetsPromise: Promise<PrecomputedPreset[]> | null = null;
let vibeWorker: Worker | null = null;
let synthWorker: Worker | null = null;
let nextReqId = 1;

interface PendingEntry<T> {
  resolve: (val: T) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

const pendingVibe = new Map<
  number,
  PendingEntry<{
    title: string;
    palette: string[];
    config: Record<string, unknown>;
    rowIndex?: number;
    similarity?: number | null;
  }>
>();
const pendingSynth = new Map<number, PendingEntry<Float32Array>>();

const MAX_DRIFT_HISTORY_ROWS = 600;
const MAX_DRIFT_HISTORY_TITLES = 180;
let autoDriftRowIndex: number | null = null;
const autoDriftVisitedRows: number[] = [];
const autoDriftVisitedTitles: string[] = [];

function recordDriftVisit(rowIndex: number, title: string): void {
  autoDriftRowIndex = rowIndex;
  autoDriftVisitedRows.push(rowIndex);
  if (autoDriftVisitedRows.length > MAX_DRIFT_HISTORY_ROWS) {
    autoDriftVisitedRows.shift();
  }
  const clean = title.trim();
  if (clean) {
    autoDriftVisitedTitles.push(clean);
    if (autoDriftVisitedTitles.length > MAX_DRIFT_HISTORY_TITLES) {
      autoDriftVisitedTitles.shift();
    }
  }
}

export function resetAutoDriftAnchor(rowIndex: number | null = null): void {
  autoDriftRowIndex = rowIndex;
}

let currentStatus: EngineStatus = {
  vibeStage: "idle",
  synthStage: "idle",
  detail: null,
  progress: null,
};
const listeners = new Set<StatusListener>();

function emitStatus() {
  for (const fn of listeners) {
    fn({ ...currentStatus });
  }
}

export function subscribeEngineStatus(listener: StatusListener): () => void {
  listeners.add(listener);
  listener({ ...currentStatus });
  return () => {
    listeners.delete(listener);
  };
}

function makeSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID().slice(0, 8);
  }
  return Math.random().toString(36).slice(2, 10);
}

function setBoundedSession(sessionId: string, vibe: string, config: MusicConfig): LocalSession {
  // Delete and re-insert to maintain LRU insertion order.
  localSessions.delete(sessionId);
  const session: LocalSession = { sessionId, vibe, config };
  localSessions.set(sessionId, session);
  while (localSessions.size > MAX_LOCAL_SESSIONS) {
    const oldest = localSessions.keys().next().value;
    if (oldest === undefined) break;
    localSessions.delete(oldest);
  }
  return session;
}

function storeLocalSession(vibe: string, config: MusicConfig): string {
  const sessionId = makeSessionId();
  setBoundedSession(sessionId, vibe, config);
  return sessionId;
}

export function getLocalSessionConfig(sessionId: string): MusicConfig | null {
  return localSessions.get(sessionId)?.config ?? null;
}

function getVibeWorker(): Worker {
  if (!vibeWorker) {
    const worker = new Worker(new URL("./vibeWorker.ts", import.meta.url), { type: "module" });
    vibeWorker = worker;
    worker.onmessage = (event: MessageEvent) => {
      const msg = event.data as Record<string, unknown>;
      if (msg.type === "status") {
        currentStatus = {
          ...currentStatus,
          vibeStage: String(msg.stage ?? "idle"),
          detail:
            msg.stage === "loading"
              ? String(msg.detail ?? "")
              : currentStatus.synthStage === "loading"
                ? currentStatus.detail
                : null,
          progress: typeof msg.progress === "number" ? msg.progress : null,
        };
        emitStatus();
        return;
      }
      const id = Number(msg.id);
      const entry = pendingVibe.get(id);
      if (!entry) return;
      clearTimeout(entry.timer);
      pendingVibe.delete(id);
      if (msg.type === "match_result" || msg.type === "drift_result") {
        entry.resolve(
          msg.result as {
            title: string;
            palette: string[];
            config: Record<string, unknown>;
            rowIndex?: number;
            similarity?: number | null;
          }
        );
      } else {
        entry.reject(new Error(String(msg.error ?? "Vibe lookup failed")));
      }
    };
    worker.onerror = (ev) => {
      const err = new Error(ev.message || "Vibe worker crashed");
      worker.terminate();
      if (vibeWorker === worker) vibeWorker = null;
      for (const [, entry] of pendingVibe) {
        clearTimeout(entry.timer);
        entry.reject(err);
      }
      pendingVibe.clear();
    };
  }
  return vibeWorker;
}

function getSynthWorker(): Worker {
  if (!synthWorker) {
    const worker = new Worker(new URL("./synthWorker.ts", import.meta.url), { type: "module" });
    synthWorker = worker;
    worker.onmessage = (event: MessageEvent) => {
      const msg = event.data as Record<string, unknown>;
      if (msg.type === "status") {
        currentStatus = {
          ...currentStatus,
          synthStage: String(msg.stage ?? "idle"),
          detail:
            msg.stage === "loading"
              ? String(msg.detail ?? "")
              : currentStatus.vibeStage === "loading"
                ? currentStatus.detail
                : null,
          progress: typeof msg.progress === "number" ? msg.progress : null,
        };
        emitStatus();
        return;
      }
      const id = Number(msg.id);
      const entry = pendingSynth.get(id);
      if (!entry) return;
      clearTimeout(entry.timer);
      pendingSynth.delete(id);
      if (msg.type === "render_result") {
        entry.resolve(new Float32Array(msg.buffer as ArrayBuffer));
      } else if (msg.type === "render_cancelled") {
        entry.reject(new DOMException("Render cancelled", "AbortError"));
      } else {
        entry.reject(new Error(String(msg.error ?? "Synth render failed")));
      }
    };
    worker.onerror = (ev) => {
      const err = new Error(ev.message || "Synth worker crashed");
      worker.terminate();
      if (synthWorker === worker) synthWorker = null;
      for (const [, entry] of pendingSynth) {
        clearTimeout(entry.timer);
        entry.reject(err);
      }
      pendingSynth.clear();
    };
  }
  return synthWorker;
}

async function loadPrecomputedPresets(): Promise<PrecomputedPreset[]> {
  if (!presetsPromise) {
    presetsPromise = fetch("/data/presets.json", { cache: "no-cache" })
      .then((res) => {
        if (!res.ok) throw new Error(`Presets fetch failed: ${res.status}`);
        return res.json() as Promise<{ presets: PrecomputedPreset[] }>;
      })
      .then((data) => data.presets)
      .catch((err) => {
        presetsPromise = null;
        throw err;
      });
  }
  return presetsPromise;
}

/** Lazily warm both workers when /demo mounts (never called on landing page). */
export function browserWarmup(): void {
  void loadPrecomputedPresets().catch(() => {});
  getSynthWorker().postMessage({ type: "warmup" });
  getVibeWorker().postMessage({ type: "warmup" });
}

export async function browserFetchPresets(): Promise<Preset[]> {
  const full = await loadPrecomputedPresets();
  return full.map(({ id, name, description, vibe, icon }) => ({
    id,
    name,
    description,
    vibe,
    icon,
  }));
}

export async function browserFetchParamOptions(): Promise<ParamOptions> {
  return PARAM_OPTIONS;
}

export async function browserFetchCapabilities(): Promise<Capabilities> {
  return {
    models: ["fast"],
    expressive_available: false,
    best_of_n: false,
  };
}

export async function browserGenerateFromVibe(vibe: string): Promise<GenerateResponse> {
  // Fast path: exact match against precomputed presets resolves in 0 ms even before MiniLM loads.
  const norm = vibe.trim().toLowerCase();
  try {
    const presets = await loadPrecomputedPresets();
    const hit = presets.find((p) => p.vibe.trim().toLowerCase() === norm);
    if (hit) {
      const cfg = coerceConfig(hit.config);
      const sessionId = storeLocalSession(vibe, cfg);
      const palette = hit.palette.length === 5 ? hit.palette : generatePalette(cfg, vibe);
      return {
        session_id: sessionId,
        title: hit.title || generateTitle(vibe, cfg),
        justification: generateJustification(vibe, cfg),
        palette,
        config: configToRecord(cfg),
      };
    }
  } catch {
    // Continue to worker lookup if presets.json failed.
  }

  try {
    const worker = getVibeWorker();
    const id = nextReqId++;
    const matched = await new Promise<{
      title: string;
      palette: string[];
      config: Record<string, unknown>;
      rowIndex?: number;
    }>((resolve, reject) => {
      const timer = setTimeout(() => {
        pendingVibe.delete(id);
        reject(new Error("Vibe encoder timed out"));
      }, WORKER_TIMEOUT_MS);
      pendingVibe.set(id, { resolve, reject, timer });
      worker.postMessage({ type: "match", id, vibe });
    });

    const cfg = coerceConfig(matched.config);
    const sessionId = storeLocalSession(vibe, cfg);
    if (typeof matched.rowIndex === "number") {
      autoDriftRowIndex = matched.rowIndex;
    }
    const palette = matched.palette.length === 5 ? matched.palette : generatePalette(cfg, vibe);
    return {
      session_id: sessionId,
      title: matched.title || generateTitle(vibe, cfg),
      justification: generateJustification(vibe, cfg),
      palette,
      config: configToRecord(cfg),
    };
  } catch (err) {
    console.warn("Browser MiniLM lookup failed, falling back to heuristic:", err);
    const cfg = heuristicConfig(vibe);
    const sessionId = storeLocalSession(vibe, cfg);
    return {
      session_id: sessionId,
      title: generateTitle(vibe, cfg),
      justification: generateJustification(vibe, cfg),
      palette: generatePalette(cfg, vibe),
      config: configToRecord(cfg),
    };
  }
}

export async function browserCreateFromConfig(
  vibe: string,
  rawConfig: Record<string, unknown>,
  justification: string,
  title?: string,
  palette?: string[]
): Promise<GenerateResponse> {
  const cfg = coerceConfig(rawConfig);
  const sessionId = storeLocalSession(vibe, cfg);
  return {
    session_id: sessionId,
    title: title || generateTitle(vibe, cfg),
    justification: justification || generateJustification(vibe, cfg),
    palette: palette && palette.length === 5 ? palette : generatePalette(cfg, vibe),
    config: configToRecord(cfg),
  };
}

export async function browserUpdateParams(
  sessionId: string,
  updates: Record<string, unknown>,
  fallbackConfig?: Record<string, unknown>,
  fallbackVibe?: string
): Promise<{ config: Record<string, unknown>; palette: string[]; title: string }> {
  const existing = localSessions.get(sessionId);
  const base = existing ? existing.config : coerceConfig(fallbackConfig ?? {});
  const vibe = existing ? existing.vibe : fallbackVibe ?? "";
  const next = coerceConfig({ ...base, ...updates });
  setBoundedSession(sessionId, vibe, next);
  return {
    config: configToRecord(next),
    palette: generatePalette(next, vibe),
    title: generateTitle(vibe, next),
  };
}

export interface AutoDriftStepResponse extends GenerateResponse {
  vibe: string;
  similarity: number | null;
}

/**
 * Step smoothly to the next semantically similar soundscape along a continuous
 * gradient across all 14,035 curated genres in the embedding manifold.
 */
export async function browserNextAutoDrift(
  currentConfig?: Record<string, unknown>,
  reset = false
): Promise<AutoDriftStepResponse> {
  if (reset) {
    autoDriftRowIndex = null;
  }
  const worker = getVibeWorker();
  const id = nextReqId++;
  const matched = await new Promise<{
    title: string;
    palette: string[];
    config: Record<string, unknown>;
    rowIndex?: number;
    similarity?: number | null;
  }>((resolve, reject) => {
    const timer = setTimeout(() => {
      pendingVibe.delete(id);
      reject(new Error("Auto Drift lookup timed out"));
    }, WORKER_TIMEOUT_MS);
    pendingVibe.set(id, { resolve, reject, timer });
    worker.postMessage({
      type: "drift_next",
      id,
      currentRowIndex: autoDriftRowIndex,
      currentConfig,
      visitedIndices: autoDriftVisitedRows,
      visitedTitles: autoDriftVisitedTitles,
      reset,
    });
  });

  const cfg = coerceConfig(matched.config);
  const title = matched.title?.trim() || generateTitle("ambient drift", cfg);
  const moodParts = [
    cfg.mode.replace(/_/g, " "),
    cfg.tempo.replace(/_/g, " "),
    cfg.pad.replace(/_/g, " "),
    cfg.texture !== "none" ? cfg.texture.replace(/_/g, " ") : cfg.space.replace(/_/g, " "),
  ];
  const vibeLabel = moodParts.join(" · ");
  if (typeof matched.rowIndex === "number") {
    recordDriftVisit(matched.rowIndex, title);
  }
  const sessionId = storeLocalSession(vibeLabel, cfg);
  const palette = matched.palette.length === 5 ? matched.palette : generatePalette(cfg, title);
  return {
    session_id: sessionId,
    title,
    vibe: vibeLabel,
    justification: generateJustification(title, cfg),
    palette,
    config: configToRecord(cfg),
    similarity: typeof matched.similarity === "number" ? matched.similarity : null,
  };
}

export async function browserRenderPcm(
  rawConfig: Record<string, unknown>,
  duration = 60,
  tOffset = 0,
  normalize = true,
  seed: number | null = null,
  signal?: AbortSignal
): Promise<Float32Array> {
  if (signal?.aborted) {
    throw new DOMException("Render cancelled", "AbortError");
  }
  const internal = toInternalSynthConfig(rawConfig);
  const worker = getSynthWorker();
  const id = nextReqId++;
  return new Promise<Float32Array>((resolve, reject) => {
    const onAbort = () => {
      const entry = pendingSynth.get(id);
      if (!entry) return;
      clearTimeout(entry.timer);
      pendingSynth.delete(id);
      worker.postMessage({ type: "cancel", id });
      reject(new DOMException("Render cancelled", "AbortError"));
    };

    const cleanup = () => {
      signal?.removeEventListener("abort", onAbort);
    };

    const timer = setTimeout(() => {
      pendingSynth.delete(id);
      cleanup();
      worker.postMessage({ type: "cancel", id });
      reject(new Error("Audio render timed out"));
    }, WORKER_TIMEOUT_MS);

    pendingSynth.set(id, {
      resolve: (pcm) => {
        cleanup();
        resolve(pcm);
      },
      reject: (err) => {
        cleanup();
        reject(err);
      },
      timer,
    });

    signal?.addEventListener("abort", onAbort, { once: true });
    worker.postMessage({
      type: "render",
      id,
      configJson: JSON.stringify(internal),
      duration,
      tOffset,
      normalize,
      seed,
    });
  });
}
