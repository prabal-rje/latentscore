/**
 * DOM-free stateful reactive Score engine.
 * Works identically in Node.js, Bun, Deno, Web Workers, SSR, and browsers.
 */

import {
  ConfigInput,
  DEFAULT_CONFIG,
  MusicConfig,
  MusicConfigUpdate,
  SAMPLE_RATE,
  SynthConfig,
  applyUpdateToInternal,
  interpolateConfigs,
  toInternalConfig,
  toPublicConfig,
} from "./config.js";
import { assemble, softLimit } from "./dsp.js";
import { AutoDrift, EmbeddingIndex, moodFromText } from "./mood.js";
import { EncodeWavOptions, encodeWavBytes, fadeIntoNothingness } from "./wav.js";

const OVERLAP_SECONDS = 0.45;
const FIXED_SESSION_GAIN = 0.62;

export interface ScoreOptions {
  /** Initial prompt string or MusicConfig / SynthConfig. */
  initial?: string | ConfigInput;
  /** Deterministic synthesis seed (default 42). */
  seed?: number;
  /** Default glide duration in seconds when steering toward a new mood (default 3.0). */
  glideSeconds?: number;
  /** Internal synthesis block duration in seconds (default 1.5). */
  blockSeconds?: number;
  /** Optional EmbeddingIndex for 14,035-mood gradient auto-drift. */
  index?: EmbeddingIndex;
}

export interface SteerOptions {
  /** Override glide duration in seconds for this steer call. */
  glideSeconds?: number;
}

export interface AutoDriftHandle {
  /** Stop scheduled auto-drift immediately. */
  stop(): void;
  /** True while auto-drift is active. */
  readonly active: boolean;
}

export interface AutoDriftOptions {
  /** Milliseconds between smooth gradient hops (default 14000). */
  intervalMs?: number;
  /** Glide duration per hop in seconds (default 4.5). */
  glideSeconds?: number;
  /** Optional callback invoked whenever auto-drift selects a new mood. */
  onMood?: (mood: { config: MusicConfig; title?: string; palette?: readonly string[] }) => void;
  /** Optional callback invoked when manual steer/nudge cancels auto-drift. */
  onStop?: () => void;
}

function resolveTarget(mood: string | ConfigInput): SynthConfig {
  return typeof mood === "string" ? toInternalConfig(moodFromText(mood)) : toInternalConfig(mood);
}

function synthConfigsEqual(a: SynthConfig, b: SynthConfig): boolean {
  const keys = Object.keys(a) as (keyof SynthConfig)[];
  for (const k of keys) {
    if (a[k] !== b[k]) return false;
  }
  return true;
}

/**
 * Stateful pull-based ambient synthesizer with idempotent target steering
 * and guaranteed glide convergence across arbitrary block sizes.
 */
export class Score {
  readonly sampleRate: number = SAMPLE_RATE;
  readonly blockSeconds: number;
  private current: SynthConfig;
  private glideFrom: SynthConfig;
  private target: SynthConfig;
  private glideElapsed = 0.0;
  private glideSeconds: number;
  private readonly sessionSeed: number;
  private readonly drifter: AutoDrift;
  private tOffset = 0.0;
  private prevTail: Float64Array = new Float64Array(0);
  private fifo: Float32Array = new Float32Array(0);
  private fifoRead = 0;
  private activeDrift: AutoDriftHandle | null = null;

  constructor(options?: ScoreOptions) {
    const blockSec = options?.blockSeconds ?? 1.5;
    const glideSec = options?.glideSeconds ?? 3.0;
    if (!Number.isFinite(blockSec) || blockSec <= 0.1) {
      throw new RangeError("blockSeconds must be finite and > 0.1");
    }
    if (!Number.isFinite(glideSec) || glideSec < 0) {
      throw new RangeError("glideSeconds must be finite and >= 0");
    }

    const start =
      options?.initial !== undefined
        ? resolveTarget(options.initial)
        : toInternalConfig(DEFAULT_CONFIG);
    this.current = start;
    this.glideFrom = start;
    this.target = start;
    this.glideSeconds = glideSec;
    this.blockSeconds = blockSec;
    this.sessionSeed = ((options?.seed ?? 42) >>> 0) & 0x7fffffff;
    this.drifter = new AutoDrift({
      initial: start,
      index: options?.index,
      seed: this.sessionSeed ^ 0x517cc1b7,
    });
  }

  /** Nearest public MusicConfig for the current audible synthesis state. */
  get currentConfig(): MusicConfig {
    return toPublicConfig(this.current);
  }

  /** Nearest public MusicConfig for the current steering target. */
  get targetConfig(): MusicConfig {
    return toPublicConfig(this.target);
  }

  /** Frozen snapshot of the current numeric SynthConfig. */
  get internalConfig(): Readonly<SynthConfig> {
    return Object.freeze({ ...this.current });
  }

  /**
   * Set a new target mood idempotently and glide smoothly toward it.
   * Manual steer always cancels any active AutoDrift so user intent wins.
   */
  steer(mood: string | ConfigInput, options?: SteerOptions): MusicConfig {
    if (
      options?.glideSeconds !== undefined &&
      (!Number.isFinite(options.glideSeconds) || options.glideSeconds < 0)
    ) {
      throw new RangeError("glideSeconds must be finite and >= 0");
    }
    const resolved = resolveTarget(mood);
    if (this.activeDrift?.active) {
      this.activeDrift.stop();
    }
    this.applyTarget(resolved, options?.glideSeconds);
    this.drifter.anchorToConfig(resolved);
    return toPublicConfig(resolved);
  }

  private applyTarget(resolved: SynthConfig, glideOverride?: number): void {
    if (glideOverride !== undefined) {
      this.glideSeconds = glideOverride;
    }
    if (!synthConfigsEqual(resolved, this.target)) {
      this.glideFrom = this.current;
      this.target = resolved;
      this.glideElapsed = 0.0;
    } else if (synthConfigsEqual(this.current, this.target)) {
      this.glideFrom = this.target;
      this.glideElapsed = this.glideSeconds;
    }
  }

  /**
   * Apply absolute fields or relative Step() deltas to the target config,
   * preserving all untouched continuous parameters.
   */
  nudge(update: MusicConfigUpdate): MusicConfig {
    const nextInternal = applyUpdateToInternal(this.target, update);
    if (this.activeDrift?.active) {
      this.activeDrift.stop();
    }
    this.applyTarget(nextInternal);
    this.drifter.anchorToConfig(nextInternal);
    return toPublicConfig(nextInternal);
  }

  /**
   * Advance one smooth step along the similarity / harmonic gradient,
   * preserving continuous numeric drift offsets.
   */
  driftNext(options?: SteerOptions): {
    config: MusicConfig;
    title?: string;
    palette?: readonly string[];
  } {
    if (
      options?.glideSeconds !== undefined &&
      (!Number.isFinite(options.glideSeconds) || options.glideSeconds < 0)
    ) {
      throw new RangeError("glideSeconds must be finite and >= 0");
    }
    const hop = this.drifter.next(this.target);
    this.applyTarget(hop.synthConfig, options?.glideSeconds);
    return { config: hop.config, title: hop.title, palette: hop.palette };
  }

  /**
   * Begin endless smooth auto-drift; returns a cancellable handle.
   * Any manual `steer()` or `nudge()` cancels auto-drift cleanly.
   */
  startAutoDrift(options?: AutoDriftOptions): AutoDriftHandle {
    if (this.activeDrift?.active) {
      this.activeDrift.stop();
    }
    const intervalMs = Math.max(1500, options?.intervalMs ?? 14000);
    const glideSeconds = options?.glideSeconds ?? 4.5;
    let running = true;

    const tick = (): void => {
      if (!running) return;
      const hop = this.driftNext({ glideSeconds });
      options?.onMood?.(hop);
    };

    const timer: unknown = setInterval(tick, intervalMs);
    if (
      typeof timer === "object" &&
      timer !== null &&
      "unref" in timer &&
      typeof (timer as { unref?: unknown }).unref === "function"
    ) {
      (timer as { unref: () => void }).unref();
    }

    const handle: AutoDriftHandle = {
      get active(): boolean {
        return running;
      },
      stop: (): void => {
        if (!running) return;
        running = false;
        clearInterval(timer as ReturnType<typeof setInterval>);
        if (this.activeDrift === handle) this.activeDrift = null;
        options?.onStop?.();
      },
    };
    this.activeDrift = handle;
    return handle;
  }

  private advanceConfig(stepSec: number): SynthConfig {
    if (this.glideSeconds <= 1e-6 || synthConfigsEqual(this.current, this.target)) {
      this.current = this.target;
      this.glideFrom = this.target;
      this.glideElapsed = this.glideSeconds;
      return this.current;
    }
    this.glideElapsed = Math.min(this.glideSeconds, this.glideElapsed + stepSec);
    const progress = this.glideElapsed / this.glideSeconds;
    if (progress >= 1.0) {
      this.current = this.target;
      this.glideFrom = this.target;
    } else {
      this.current = interpolateConfigs(this.glideFrom, this.target, progress);
    }
    return this.current;
  }

  private renderInternalBlock(dur: number): Float32Array {
    const cfg = this.advanceConfig(dur);
    const overlapSec = Math.min(OVERLAP_SECONDS, dur * 0.35);
    const overlapN = Math.max(16, Math.round(overlapSec * this.sampleRate));
    const blockN = Math.max(1, Math.round(dur * this.sampleRate));
    const renderDur = (blockN + overlapN) / this.sampleRate;

    const raw = assemble(cfg, {
      duration: renderDur,
      normalize: false,
      seed: this.sessionSeed,
      tOffset: this.tOffset,
    });
    const scaled = new Float64Array(raw.length);
    for (let i = 0; i < raw.length; i++) scaled[i] = raw[i]! * FIXED_SESSION_GAIN;
    const calibrated = softLimit(scaled, 0.92);

    const head = new Float64Array(blockN);
    head.set(calibrated.subarray(0, Math.min(blockN, calibrated.length)));
    const tail = new Float64Array(
      calibrated.subarray(blockN, Math.min(blockN + overlapN, calibrated.length)),
    );

    if (this.prevTail.length > 0) {
      const blendN = Math.min(head.length, this.prevTail.length);
      const denom = Math.max(1, blendN - 1);
      for (let i = 0; i < blendN; i++) {
        const fadeIn = 0.5 * (1.0 - Math.cos((Math.PI * i) / denom));
        head[i] = this.prevTail[i]! * (1.0 - fadeIn) + head[i]! * fadeIn;
      }
    }

    this.prevTail = tail;
    this.tOffset += blockN / this.sampleRate;

    const out = new Float32Array(blockN);
    for (let i = 0; i < blockN; i++) {
      out[i] = Math.max(-1, Math.min(1, head[i]!));
    }
    return out;
  }

  /**
   * Pull an exact number of mono Float32 frames sequentially through the
   * canonical timeline (never skips or rewinds when mixed with `nextBlock()`).
   */
  fill(targetOrFrames: Float32Array | number): Float32Array {
    if (typeof targetOrFrames === "number") {
      if (!Number.isFinite(targetOrFrames) || targetOrFrames < 0) {
        throw new RangeError("Frame count must be finite and >= 0");
      }
    }
    const out =
      typeof targetOrFrames === "number"
        ? new Float32Array(Math.floor(targetOrFrames))
        : targetOrFrames;
    let written = 0;
    while (written < out.length) {
      const avail = this.fifo.length - this.fifoRead;
      if (avail <= 0) {
        this.fifo = this.renderInternalBlock(this.blockSeconds);
        this.fifoRead = 0;
        continue;
      }
      const take = Math.min(out.length - written, avail);
      out.set(this.fifo.subarray(this.fifoRead, this.fifoRead + take), written);
      this.fifoRead += take;
      written += take;
    }
    return out;
  }

  /**
   * Pull `seconds` of seamless mono Float32 audio through the canonical timeline.
   */
  nextBlock(seconds?: number): Float32Array {
    const dur = seconds ?? this.blockSeconds;
    if (!Number.isFinite(dur) || dur <= 0.05) {
      throw new RangeError("Block duration must be finite and > 0.05 seconds");
    }
    const frames = Math.max(1, Math.round(dur * this.sampleRate));
    return this.fill(frames);
  }

  /**
   * Render `durationSeconds` smoothly with a cubic fade-into-nothingness and
   * encode as WAV bytes locked to `this.sampleRate` (44,100 Hz).
   */
  toWavBytes(durationSeconds = 16.0, options?: Omit<EncodeWavOptions, "sampleRate">): Uint8Array {
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
      throw new RangeError("durationSeconds must be finite and > 0");
    }
    const frames = Math.max(1, Math.round(durationSeconds * this.sampleRate));
    const pcm = this.fill(frames);
    return encodeWavBytes(pcm, {
      ...options,
      sampleRate: this.sampleRate,
      stereoWidth: options?.stereoWidth ?? this.current.stereo,
    });
  }

  /** Release any running auto-drift timer cleanly. */
  dispose(): void {
    this.activeDrift?.stop();
  }
}

/**
 * Convenience factory: `const score = createScore("quiet morning desk")`.
 */
export function createScore(
  initial?: string | ConfigInput,
  options?: Omit<ScoreOptions, "initial">,
): Score {
  return new Score({ ...options, initial });
}

/**
 * One-shot helper: render a text prompt or MusicConfig to mono Float32 PCM.
 */
export function render(
  promptOrConfig: string | ConfigInput,
  options?: { duration?: number; seed?: number; fadeOutSeconds?: number },
): Float32Array {
  const cfg = typeof promptOrConfig === "string" ? moodFromText(promptOrConfig) : promptOrConfig;
  const duration = options?.duration ?? 8.0;
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new RangeError("duration must be finite and > 0");
  }
  const f64 = assemble(cfg, { duration, normalize: true, seed: options?.seed ?? 42 });
  const fadeOut = options?.fadeOutSeconds ?? 0.0;
  if (fadeOut > 0) {
    return fadeIntoNothingness(f64, fadeOut, SAMPLE_RATE);
  }
  const f32 = new Float32Array(f64.length);
  for (let i = 0; i < f64.length; i++) f32[i] = Math.max(-1, Math.min(1, f64[i]!));
  return f32;
}
