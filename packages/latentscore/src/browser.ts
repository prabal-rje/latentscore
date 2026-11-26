/**
 * LatentScore Browser Web Audio entry point ("latentscore/browser").
 *
 * Audio-safety guarantees:
 * - AudioContext is created and resumed SYNCHRONOUSLY inside `play()` before
 *   any await or synthesis work so browser autoplay policies allow sound.
 * - Each playback session owns an isolated gain/bus graph and generation id so
 *   rapid `stop() -> play()` or `stop()` during pending `resume()` never mixes
 *   old and new audio sources or clicks.
 * - Audio clock is re-sampled AFTER rendering so slow synthesis blocks never
 *   cause overlapping buffers.
 * - Headroom-safe Mid-Side stereo widener + soft-clipping WaveShaper ceiling.
 * - Single cubic fade-into-nothingness `(1 - u)^3` on `stop()`.
 * - Resumes automatically on `visibilitychange` when `ctx.state !== "running"`
 *   (covers iOS Safari `"interrupted"` state).
 */

import { ConfigInput, MusicConfig, MusicConfigUpdate, SAMPLE_RATE } from "./config.js";
import {
  AutoDriftHandle,
  AutoDriftOptions,
  Score,
  ScoreOptions,
  SteerOptions,
} from "./score.js";

export * from "./index.js";

const DEFAULT_BROWSER_BLOCK_SECONDS = 1.5;
const PARAM_GUARD_SECONDS = 0.004;

export interface BrowserScoreOptions extends ScoreOptions {
  /** Initial master volume in [0, 1] (default 0.85). */
  volume?: number;
  /** Debounce window in ms when calling `steerOnType(text)` (default 280). */
  typeDebounceMs?: number;
}

interface PlaybackSession {
  readonly id: number;
  readonly bus: GainNode;
  readonly duck: GainNode;
  readonly sources: Set<AudioBufferSourceNode>;
  stopped: boolean;
  nextStartTime: number;
  scheduleTimer: ReturnType<typeof setInterval> | null;
}

function makeSoftClipCurve(samples = 1024): Float32Array<ArrayBuffer> {
  const curve = new Float32Array(new ArrayBuffer(samples * 4));
  for (let i = 0; i < samples; i++) {
    const x = (i * 2) / (samples - 1) - 1;
    const ax = Math.abs(x);
    if (ax <= 0.78) {
      curve[i] = x;
    } else {
      const y = 0.78 + 0.14 * Math.tanh((ax - 0.78) / 0.14);
      curve[i] = x >= 0 ? y : -y;
    }
  }
  return curve;
}

/**
 * Interactive Web Audio ambient music player wrapping the pure-TS Score engine.
 */
export class BrowserScore {
  readonly score: Score;
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private midGain: GainNode | null = null;
  private sideGain: GainNode | null = null;
  private stereoInput: GainNode | null = null;
  private activeSession: PlaybackSession | null = null;
  private sessionSeq = 0;
  private disposed = false;
  private volume: number;
  private typeTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly typeDebounceMs: number;
  private visibilityListener: (() => void) | null = null;

  constructor(options?: BrowserScoreOptions) {
    this.score = new Score({
      blockSeconds: options?.blockSeconds ?? DEFAULT_BROWSER_BLOCK_SECONDS,
      ...options,
    });
    this.volume = Math.max(0, Math.min(1, options?.volume ?? 0.85));
    this.typeDebounceMs = Math.max(40, options?.typeDebounceMs ?? 280);
  }

  get isPlaying(): boolean {
    return this.activeSession !== null && !this.activeSession.stopped;
  }

  get currentConfig(): MusicConfig {
    return this.score.currentConfig;
  }

  get targetConfig(): MusicConfig {
    return this.score.targetConfig;
  }

  private cancelPendingType(): void {
    if (this.typeTimer !== null) {
      clearTimeout(this.typeTimer);
      this.typeTimer = null;
    }
  }

  private ensureAudioGraphSync(): AudioContext {
    if (this.ctx) return this.ctx;
    const AudioCtx =
      globalThis.AudioContext ??
      (globalThis as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) {
      throw new Error("Web Audio API is unavailable in this environment");
    }

    let ctx: AudioContext;
    try {
      ctx = new AudioCtx({ sampleRate: SAMPLE_RATE });
    } catch {
      ctx = new AudioCtx();
    }

    const stereoInput = ctx.createGain();
    const midGain = ctx.createGain();
    const sideDelay = ctx.createDelay(0.05);
    sideDelay.delayTime.value = 0.011;
    const sideHp = ctx.createBiquadFilter();
    sideHp.type = "highpass";
    sideHp.frequency.value = 300;
    const sideGain = ctx.createGain();
    const sideInvert = ctx.createGain();
    sideInvert.gain.value = -1;

    const merger = ctx.createChannelMerger(2);
    const shaper = ctx.createWaveShaper();
    shaper.curve = makeSoftClipCurve();
    shaper.oversample = "2x";

    const masterGain = ctx.createGain();
    masterGain.gain.value = this.volume;

    // Headroom-safe Mid-Side stereo matrix: L = M + S, R = M - S (mono-safe sum).
    stereoInput.connect(midGain);
    midGain.connect(merger, 0, 0);
    midGain.connect(merger, 0, 1);

    stereoInput.connect(sideDelay);
    sideDelay.connect(sideHp);
    sideHp.connect(sideGain);
    sideGain.connect(merger, 0, 0);
    sideGain.connect(sideInvert);
    sideInvert.connect(merger, 0, 1);

    merger.connect(shaper);
    shaper.connect(masterGain);
    masterGain.connect(ctx.destination);

    this.ctx = ctx;
    this.stereoInput = stereoInput;
    this.midGain = midGain;
    this.sideGain = sideGain;
    this.masterGain = masterGain;
    this.updateStereoNodes(ctx.currentTime);

    if (typeof document !== "undefined") {
      this.visibilityListener = () => {
        if (
          document.visibilityState === "visible" &&
          this.isPlaying &&
          this.ctx &&
          this.ctx.state !== "running"
        ) {
          void this.ctx.resume();
        }
      };
      document.addEventListener("visibilitychange", this.visibilityListener);
    }

    return ctx;
  }

  private updateStereoNodes(now: number): void {
    if (!this.midGain || !this.sideGain) return;
    const width = Math.max(0, Math.min(1, this.score.internalConfig.stereo));
    const sGain = 0.28 * width;
    const mGain = 1.0 / (1.0 + sGain);
    const t = now + PARAM_GUARD_SECONDS;
    this.midGain.gain.setTargetAtTime(mGain, t, 0.08);
    this.sideGain.gain.setTargetAtTime(sGain, t, 0.08);
  }

  private retireSession(session: PlaybackSession, fadeOutSeconds: number): void {
    if (session.stopped) return;
    session.stopped = true;
    if (session.scheduleTimer !== null) {
      clearInterval(session.scheduleTimer);
      session.scheduleTimer = null;
    }

    const ctx = this.ctx;
    const fadeSec = Math.max(0.03, fadeOutSeconds);
    if (!ctx) return;

    const now = ctx.currentTime;
    const startVal = Math.max(0.0, Math.min(1.0, session.duck.gain.value));
    const steps = 32;
    const curve = new Float32Array(steps);
    for (let i = 0; i < steps; i++) {
      const u = i / (steps - 1);
      curve[i] = startVal * (1 - u) * (1 - u) * (1 - u);
    }
    curve[steps - 1] = 0.0;

    session.duck.gain.cancelScheduledValues(now);
    session.duck.gain.setValueAtTime(startVal, now);
    session.duck.gain.setValueCurveAtTime(curve, now + PARAM_GUARD_SECONDS, fadeSec);

    const stopAt = now + fadeSec + 0.05;
    for (const src of session.sources) {
      try {
        src.stop(stopAt);
      } catch {
        // Already ended.
      }
    }
    const cleanupMs = Math.ceil((fadeSec + 0.12) * 1000);
    setTimeout(() => {
      try {
        session.bus.disconnect();
        session.duck.disconnect();
      } catch {
        // Already disconnected.
      }
      session.sources.clear();
    }, cleanupMs);
  }

  private scheduleForSession(session: PlaybackSession): void {
    if (session.stopped || this.activeSession !== session || !this.ctx) return;
    const ctx = this.ctx;
    const blockSec = this.score.blockSeconds;
    const lookaheadHorizon = Math.max(blockSec * 1.25, 1.8);

    // Cap rendered blocks per tick so slow devices never starve the browser event loop.
    let renderedBlocks = 0;
    while (
      !session.stopped &&
      renderedBlocks < 2 &&
      session.nextStartTime < ctx.currentTime + lookaheadHorizon
    ) {
      const pcm = this.score.nextBlock(blockSec);
      renderedBlocks += 1;
      // Re-check session validity and audio clock AFTER synchronous render.
      if (session.stopped || this.activeSession !== session) return;

      const postRenderNow = ctx.currentTime;
      const startAt = Math.max(session.nextStartTime, postRenderNow + 0.05);
      const buffer = ctx.createBuffer(1, pcm.length, SAMPLE_RATE);
      buffer.copyToChannel(pcm as Float32Array<ArrayBuffer>, 0);

      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(session.bus);
      src.start(startAt);
      session.sources.add(src);
      src.onended = () => {
        session.sources.delete(src);
        try {
          src.disconnect();
        } catch {
          // Already disconnected.
        }
      };

      session.nextStartTime = startAt + pcm.length / SAMPLE_RATE;
      this.updateStereoNodes(postRenderNow);
    }
  }

  /**
   * Start live playback. Invoke from a user click/key gesture so AudioContext
   * creation and `resume()` run synchronously before any await.
   */
  async play(): Promise<this> {
    if (this.disposed) {
      throw new Error("BrowserScore has been disposed");
    }
    if (this.isPlaying) return this;

    const ctx = this.ensureAudioGraphSync();
    const myGen = ++this.sessionSeq;

    // Synchronous resume inside user gesture BEFORE any await.
    const resumePromise = ctx.state !== "running" ? ctx.resume() : Promise.resolve();
    await resumePromise;

    // Abort cleanly if stop(), dispose(), or a newer play() fired while awaiting resume.
    if (this.disposed || this.sessionSeq !== myGen || !this.stereoInput) {
      return this;
    }

    const bus = ctx.createGain();
    const duck = ctx.createGain();
    const now = ctx.currentTime;
    duck.gain.setValueAtTime(0.0001, now);
    duck.gain.linearRampToValueAtTime(1.0, now + 0.14);
    bus.connect(duck);
    duck.connect(this.stereoInput);

    const session: PlaybackSession = {
      id: myGen,
      bus,
      duck,
      sources: new Set(),
      stopped: false,
      nextStartTime: now + 0.06,
      scheduleTimer: null,
    };
    this.activeSession = session;

    this.scheduleForSession(session);
    const pollMs = Math.max(150, Math.min(400, Math.round(this.score.blockSeconds * 220)));
    session.scheduleTimer = setInterval(() => this.scheduleForSession(session), pollMs);
    return this;
  }

  /**
   * Steer toward a mood idempotently (cancels pending typing debounce and active auto-drift).
   */
  steer(mood: string | ConfigInput, options?: SteerOptions): MusicConfig {
    this.cancelPendingType();
    const resolved = this.score.steer(mood, options);
    if (this.ctx && this.isPlaying) {
      this.updateStereoNodes(this.ctx.currentTime);
    }
    return resolved;
  }

  /**
   * Debounced steering tailored for live `<input>` / `<textarea>` typing events.
   */
  steerOnType(
    text: string,
    options?: SteerOptions & { onApplied?: (config: MusicConfig) => void },
  ): void {
    this.cancelPendingType();
    this.typeTimer = setTimeout(() => {
      this.typeTimer = null;
      const cfg = this.steer(text, options);
      options?.onApplied?.(cfg);
    }, this.typeDebounceMs);
  }

  /** Apply relative Step() or absolute field tweaks (cancels pending typing debounce). */
  nudge(update: MusicConfigUpdate): MusicConfig {
    this.cancelPendingType();
    const resolved = this.score.nudge(update);
    if (this.ctx && this.isPlaying) {
      this.updateStereoNodes(this.ctx.currentTime);
    }
    return resolved;
  }

  /** Set master playback volume in [0, 1] smoothly. */
  setVolume(value: number): void {
    if (!Number.isFinite(value)) return;
    this.volume = Math.max(0, Math.min(1, value));
    if (this.ctx && this.masterGain) {
      const t = this.ctx.currentTime + PARAM_GUARD_SECONDS;
      this.masterGain.gain.setTargetAtTime(this.volume, t, 0.04);
    }
  }

  /** Start endless smooth auto-drift; returns a cancellable handle. */
  startAutoDrift(options?: AutoDriftOptions): AutoDriftHandle {
    this.cancelPendingType();
    return this.score.startAutoDrift(options);
  }

  /**
   * Fade the current playback session smoothly into silence using a single
   * cubic `(1 - u)^3` curve. Safe to call repeatedly or while `play()` is pending.
   */
  stop(fadeOutSeconds = 0.45): void {
    this.sessionSeq += 1;
    this.cancelPendingType();
    const session = this.activeSession;
    if (!session) return;
    this.activeSession = null;
    this.retireSession(session, fadeOutSeconds);
  }

  /** Stop audio, wait for the fade-out to complete, remove listeners, and close AudioContext. */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    const hadSession = this.activeSession !== null;
    this.stop(0.12);
    this.score.dispose();
    if (this.visibilityListener && typeof document !== "undefined") {
      document.removeEventListener("visibilitychange", this.visibilityListener);
      this.visibilityListener = null;
    }
    if (hadSession) {
      await new Promise((resolve) => setTimeout(resolve, 180));
    }
    if (this.ctx) {
      const ctx = this.ctx;
      this.ctx = null;
      this.stereoInput = null;
      this.midGain = null;
      this.sideGain = null;
      this.masterGain = null;
      await ctx.close();
    }
  }
}

/**
 * Convenience factory for interactive browser apps:
 * `const music = createBrowserScore("quiet morning desk");`
 */
export function createBrowserScore(
  initial?: string | ConfigInput,
  options?: Omit<BrowserScoreOptions, "initial">,
): BrowserScore {
  return new BrowserScore({ ...options, initial });
}
