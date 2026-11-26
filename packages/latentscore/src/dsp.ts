/**
 * Pure TypeScript Float64 procedural ambient synthesizer matching Python
 * latentscore/synth.py (all Round 1 & Round 2 studio DSP repairs included).
 * Zero runtime dependencies, 100% DOM-free.
 */

import {
  ConfigInput,
  DensityLevel,
  ModeName,
  RootNote,
  SAMPLE_RATE,
  SynthConfig,
  bpmFromTempo,
  toInternalConfig,
} from "./config.js";

const TWO_PI = Math.PI * 2;
const SQRT2 = Math.SQRT2;

const NOTE_FREQS: Readonly<Record<RootNote, number>> = {
  c: 261.63,
  "c#": 277.18,
  d: 293.66,
  "d#": 311.13,
  e: 329.63,
  f: 349.23,
  "f#": 369.99,
  g: 392.0,
  "g#": 415.3,
  a: 440.0,
  "a#": 466.16,
  b: 493.88,
};

const MODE_INTERVALS: Readonly<Record<ModeName, readonly number[]>> = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  mixolydian: [0, 2, 4, 5, 7, 9, 10],
};

const ATTACK_MULT: Readonly<Record<string, number>> = {
  soft: 2.5,
  medium: 1.0,
  sharp: 0.3,
};

const DENSITY_LAYERS: Readonly<Record<DensityLevel, readonly string[]>> = {
  2: ["bass", "pad"],
  3: ["bass", "pad", "melody"],
  4: ["bass", "pad", "melody", "rhythm"],
  5: ["bass", "pad", "melody", "rhythm", "texture"],
  6: ["bass", "pad", "melody", "rhythm", "texture", "accent"],
};

const REVERB_TAPS: readonly (readonly [number, number])[] = [
  [0.0233, 0.24],
  [0.0317, -0.22],
  [0.0431, 0.2],
  [0.0571, -0.18],
  [0.0733, 0.16],
  [0.0947, -0.14],
  [0.1181, 0.12],
  [0.1489, -0.1],
  [0.1847, 0.085],
  [0.2269, -0.07],
  [0.2741, 0.055],
  [0.3313, -0.04],
];

const DIFFUSION_TAPS: readonly (readonly [number, number])[] = [
  [0.0073, 0.35],
  [0.0131, -0.28],
  [0.0197, 0.22],
];

const HARMONY_TEMPLATES: Readonly<Record<string, readonly (readonly number[])[]>> = {
  pop_major: [
    [0, 5, 3, 4],
    [0, 3, 4, 0],
    [0, 4, 5, 3],
  ],
  pop_minor: [
    [0, 5, 3, 6],
    [0, 3, 6, 4],
    [0, 6, 3, 5],
    [0, 4, 5, 3],
  ],
  jazz_major: [
    [1, 4, 0, 0],
    [1, 4, 0, 5],
    [5, 1, 4, 0],
  ],
  jazz_minor: [
    [1, 4, 0, 0],
    [5, 1, 4, 0],
  ],
  cinematic_minor: [
    [0, 6, 5, 3],
    [0, 5, 6, 4],
    [0, 6, 3, 4],
  ],
  ambient: [
    [0, 5, 0, 3],
    [0, 4, 0, 6],
    [0, 3, 0, 5],
  ],
};

// === Deterministic PRNG ===

export class SeededRng {
  private state: number;
  private spare: number | null = null;

  constructor(seed: number) {
    this.state = (seed >>> 0) || 0x9e3779b9;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  intRange(lo: number, hiExclusive: number): number {
    if (hiExclusive <= lo) return lo;
    return lo + Math.floor(this.next() * (hiExclusive - lo));
  }

  uniform(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next();
  }

  normal(mean = 0, std = 1): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return mean + std * v;
    }
    let u = 0;
    let v = 0;
    while (u <= 1e-12) u = this.next();
    v = this.next();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    const z0 = mag * Math.cos(TWO_PI * v);
    this.spare = mag * Math.sin(TWO_PI * v);
    return mean + std * z0;
  }
}

export function deriveSeed(baseSeed: number, salt: number): number {
  let x = ((baseSeed >>> 0) ^ Math.imul(salt >>> 0, 0x9e3779b1)) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0;
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
  return (x ^ (x >>> 16)) & 0x7fffffff;
}

// === Biquad Butterworth filter + FIR decimation ===

interface BiquadCoeffs {
  readonly b0: number;
  readonly b1: number;
  readonly b2: number;
  readonly a1: number;
  readonly a2: number;
}

const BIQUAD_CACHE = new Map<string, BiquadCoeffs>();

function butterCached(kind: "low" | "high", normalizedCutoff: number): BiquadCoeffs {
  const wn = Math.round(Math.max(0.0005, Math.min(0.99, normalizedCutoff)) * 1e5) / 1e5;
  const key = `${kind}:${wn}`;
  const hit = BIQUAD_CACHE.get(key);
  if (hit) return hit;

  const k = Math.tan(Math.PI * wn * 0.5);
  const k2 = k * k;
  const norm = 1.0 / (1.0 + SQRT2 * k + k2);
  const a1 = 2.0 * (k2 - 1.0) * norm;
  const a2 = (1.0 - SQRT2 * k + k2) * norm;

  const b0 = kind === "low" ? k2 * norm : norm;
  const b1 = kind === "low" ? 2.0 * b0 : -2.0 * b0;
  const coeffs: BiquadCoeffs = { b0, b1, b2: b0, a1, a2 };
  if (BIQUAD_CACHE.size < 512) BIQUAD_CACHE.set(key, coeffs);
  return coeffs;
}

function runBiquad(signal: Float64Array, c: BiquadCoeffs): Float64Array {
  const n = signal.length;
  const out = new Float64Array(n);
  const { b0, b1, b2, a1, a2 } = c;
  let z1 = 0.0;
  let z2 = 0.0;
  for (let i = 0; i < n; i++) {
    const x = signal[i]!;
    const y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    out[i] = y;
  }
  return out;
}

export function applyLowpass(
  signal: Float64Array,
  cutoff: number,
  sr: number = SAMPLE_RATE,
): Float64Array {
  if (signal.length === 0) return new Float64Array(0);
  const wn = Math.max(0.0005, Math.min(0.99, cutoff / (sr * 0.5)));
  return runBiquad(signal, butterCached("low", wn));
}

export function applyHighpass(
  signal: Float64Array,
  cutoff: number,
  sr: number = SAMPLE_RATE,
): Float64Array {
  if (signal.length === 0) return new Float64Array(0);
  const wn = Math.max(0.0005, Math.min(0.99, cutoff / (sr * 0.5)));
  return runBiquad(signal, butterCached("high", wn));
}

const FIR_CACHE = new Map<number, Float64Array>();

function firLowpassTaps(factor: number): Float64Array {
  const cached = FIR_CACHE.get(factor);
  if (cached) return cached;
  const numTaps = 20 * factor + 1;
  const half = (numTaps - 1) * 0.5;
  const cutoff = 1.0 / factor;
  const taps = new Float64Array(numTaps);
  let total = 0.0;
  for (let i = 0; i < numTaps; i++) {
    const n = i - half;
    const x = cutoff * n;
    const sinc = Math.abs(x) < 1e-12 ? 1.0 : Math.sin(Math.PI * x) / (Math.PI * x);
    const hamming = 0.54 - 0.46 * Math.cos((TWO_PI * i) / (numTaps - 1));
    const v = sinc * hamming;
    taps[i] = v;
    total += v;
  }
  for (let i = 0; i < numTaps; i++) taps[i]! /= total;
  FIR_CACHE.set(factor, taps);
  return taps;
}

function decimateFirZeroPhase(signal: Float64Array, factor: number, targetLen: number): Float64Array {
  if (factor <= 1 || signal.length === 0) {
    return matchLength(signal, targetLen);
  }
  const taps = firLowpassTaps(factor);
  const n = signal.length;
  const half = taps.length >> 1;
  const out = new Float64Array(targetLen);
  const s0 = signal[0]!;
  const sEnd = signal[n - 1]!;

  for (let i = 0; i < targetLen; i++) {
    const center = i * factor;
    let acc = 0.0;
    for (let k = 0; k < taps.length; k++) {
      const idx = center + k - half;
      let sample: number;
      if (idx < 0) {
        const mirror = Math.min(n - 1, -idx);
        sample = 2.0 * s0 - signal[mirror]!;
      } else if (idx >= n) {
        const mirror = Math.max(0, 2 * (n - 1) - idx);
        sample = 2.0 * sEnd - signal[mirror]!;
      } else {
        sample = signal[idx]!;
      }
      acc += sample * taps[k]!;
    }
    out[i] = acc;
  }
  return out;
}

// === Oscillators & primitives ===

function sampleCount(duration: number, sr: number): number {
  return Math.max(0, Math.floor(sr * duration));
}

function matchLength(sig: Float64Array, targetLen: number): Float64Array {
  if (sig.length === targetLen) return sig;
  const out = new Float64Array(targetLen);
  out.set(sig.subarray(0, Math.min(sig.length, targetLen)));
  return out;
}

function polyBlep2pt(phase: number, dt: number): number {
  if (dt <= 0 || dt >= 0.5) return 0;
  if (phase < dt) {
    const u = phase / dt;
    return u + u - u * u - 1.0;
  }
  if (phase > 1.0 - dt) {
    const u = (phase - 1.0) / dt;
    return u * u + u + u + 1.0;
  }
  return 0;
}

export function freqFromNote(root: RootNote, semitones = 0, octave = 4): number {
  const base = NOTE_FREQS[root] ?? 261.63;
  return base * Math.pow(2, octave - 4) * Math.pow(2, semitones / 12);
}

function semitoneFromDegree(mode: ModeName, degree: number): number {
  const intervals = MODE_INTERVALS[mode] ?? MODE_INTERVALS.minor;
  const len = intervals.length;
  const octShift = Math.floor(degree / len);
  const idx = ((degree % len) + len) % len;
  return intervals[idx]! + 12 * octShift;
}

function scaleFreq(cfg: SynthConfig, degree: number, octave = 4): number {
  return freqFromNote(cfg.root, semitoneFromDegree(cfg.mode, degree), octave);
}

export function generateSine(
  freq: number,
  duration: number,
  sr: number = SAMPLE_RATE,
  amp = 0.3,
  tOffset = 0.0,
): Float64Array {
  const n = sampleCount(duration, sr);
  const out = new Float64Array(n);
  const invSr = 1.0 / sr;
  const w = TWO_PI * freq;
  for (let i = 0; i < n; i++) {
    out[i] = amp * Math.sin(w * (tOffset + i * invSr));
  }
  return out;
}

export function generateTriangle(
  freq: number,
  duration: number,
  sr: number = SAMPLE_RATE,
  amp = 0.3,
  tOffset = 0.0,
): Float64Array {
  const n = sampleCount(duration, sr);
  const out = new Float64Array(n);
  if (n === 0 || freq <= 0) return out;
  const maxHarmonic = Math.min(15, Math.floor((sr * 0.45) / freq));
  if (maxHarmonic < 1) return out;

  const invSr = 1.0 / sr;
  const w = TWO_PI * freq;
  const scale = (8.0 / (Math.PI * Math.PI)) * amp;
  let sign = 1.0;
  for (let h = 1; h <= maxHarmonic; h += 2) {
    const coeff = (sign / (h * h)) * scale;
    const hw = h * w;
    for (let i = 0; i < n; i++) {
      out[i]! += coeff * Math.sin(hw * (tOffset + i * invSr));
    }
    sign = -sign;
  }
  return out;
}

export function generateSawtooth(
  freq: number,
  duration: number,
  sr: number = SAMPLE_RATE,
  amp = 0.3,
  tOffset = 0.0,
  oversample = 2,
): Float64Array {
  const n = sampleCount(duration, sr);
  if (n === 0 || freq <= 0) return new Float64Array(n);
  const os = Math.max(1, Math.round(oversample));
  const srHigh = sr * os;
  const dt = freq / srHigh;
  if (dt >= 0.45) return generateSine(freq, duration, sr, amp, tOffset);

  const nHigh = n * os;
  const high = new Float64Array(nHigh);
  const invSrHigh = 1.0 / srHigh;
  for (let i = 0; i < nHigh; i++) {
    const t = tOffset + i * invSrHigh;
    const rawPhase = t * freq;
    const phase = rawPhase - Math.floor(rawPhase);
    high[i] = 2.0 * phase - 1.0 - polyBlep2pt(phase, dt);
  }
  const dec = decimateFirZeroPhase(high, os, n);
  for (let i = 0; i < n; i++) dec[i]! *= amp;
  return dec;
}

export function generateSquare(
  freq: number,
  duration: number,
  sr: number = SAMPLE_RATE,
  amp = 0.3,
  tOffset = 0.0,
  oversample = 2,
): Float64Array {
  const n = sampleCount(duration, sr);
  if (n === 0 || freq <= 0) return new Float64Array(n);
  const os = Math.max(1, Math.round(oversample));
  const srHigh = sr * os;
  const dt = freq / srHigh;
  if (dt >= 0.45) return generateSine(freq, duration, sr, amp, tOffset);

  const nHigh = n * os;
  const high = new Float64Array(nHigh);
  const invSrHigh = 1.0 / srHigh;
  for (let i = 0; i < nHigh; i++) {
    const t = tOffset + i * invSrHigh;
    const rawPhase = t * freq;
    const phase = rawPhase - Math.floor(rawPhase);
    const shifted = phase + 0.5 >= 1.0 ? phase - 0.5 : phase + 0.5;
    const naive = phase < 0.5 ? 1.0 : -1.0;
    high[i] = naive + polyBlep2pt(phase, dt) - polyBlep2pt(shifted, dt);
  }
  const dec = decimateFirZeroPhase(high, os, n);
  for (let i = 0; i < n; i++) dec[i]! *= amp;
  return dec;
}

function oscForGrain(
  grain: string,
  freq: number,
  dur: number,
  sr: number,
  amp: number,
  tOffset: number,
  fallback: "sine" | "triangle" = "sine",
): Float64Array {
  if (grain === "gritty") return generateSawtooth(freq, dur, sr, amp, tOffset);
  if (grain === "warm") return generateTriangle(freq, dur, sr, amp, tOffset);
  if (grain === "clean") return generateSine(freq, dur, sr, amp, tOffset);
  return fallback === "triangle"
    ? generateTriangle(freq, dur, sr, amp, tOffset)
    : generateSine(freq, dur, sr, amp, tOffset);
}

function generateLfo(
  duration: number,
  rate: number,
  sr: number = SAMPLE_RATE,
  tOffset = 0.0,
): Float64Array {
  const n = sampleCount(duration, sr);
  const out = new Float64Array(n);
  const invSr = 1.0 / sr;
  const w = TWO_PI * rate;
  for (let i = 0; i < n; i++) {
    out[i] = 0.5 + 0.5 * Math.sin(w * (tOffset + i * invSr));
  }
  return out;
}

// === Envelopes, humanize, limiter, spatial ===

function writeRaisedCosineRamp(
  dst: Float64Array,
  offset: number,
  length: number,
  start: number,
  end: number,
): void {
  if (length <= 0) return;
  if (length === 1) {
    dst[offset] = end;
    return;
  }
  const denom = length - 1;
  const span = end - start;
  for (let i = 0; i < length; i++) {
    const curve = 0.5 * (1.0 - Math.cos((Math.PI * i) / denom));
    dst[offset + i] = start + span * curve;
  }
}

export function applyAdsr(
  signal: Float64Array,
  attack: number,
  decay: number,
  sustain: number,
  release: number,
  sr: number = SAMPLE_RATE,
): Float64Array {
  const total = signal.length;
  if (total === 0) return new Float64Array(0);
  if (total === 1) return new Float64Array(1);

  const sus = Math.max(0.0, Math.min(1.0, sustain));
  const minA = Math.min(Math.max(1, Math.round(0.005 * sr)), Math.max(1, total >> 2));
  const minR = Math.min(Math.max(1, Math.round(0.018 * sr)), Math.max(1, Math.floor(total / 3)));

  const aReq = Math.max(minA, Math.round(Math.max(attack, 0.005) * sr));
  const dReq = Math.max(0, Math.round(Math.max(decay, 0.0) * sr));
  const rReq = Math.max(minR, Math.round(Math.max(release, 0.018) * sr));

  let aLen: number;
  let dLen: number;
  let sLen: number;
  let rLen: number;
  const stageSum = aReq + dReq + rReq;
  if (stageSum <= total) {
    aLen = aReq;
    dLen = dReq;
    sLen = total - stageSum;
    rLen = rReq;
  } else {
    const scale = total / stageSum;
    rLen = Math.min(Math.max(minR, Math.round(rReq * scale)), Math.max(1, total - minA));
    const rem = total - rLen;
    const adSum = Math.max(1, aReq + dReq);
    aLen = Math.min(Math.max(minA, Math.round(rem * (aReq / adSum))), rem);
    dLen = Math.max(0, rem - aLen);
    sLen = 0;
  }

  const env = new Float64Array(total);
  let pos = 0;
  writeRaisedCosineRamp(env, pos, aLen, 0.0, 1.0);
  pos += aLen;
  if (dLen > 0) {
    writeRaisedCosineRamp(env, pos, dLen, 1.0, sus);
    pos += dLen;
  }
  if (sLen > 0) {
    env.fill(sus, pos, pos + sLen);
    pos += sLen;
  }
  const relStart = dLen > 0 || sLen > 0 ? sus : 1.0;
  writeRaisedCosineRamp(env, pos, total - pos, relStart, 0.0);
  env[0] = 0.0;
  env[total - 1] = 0.0;

  const out = new Float64Array(total);
  for (let i = 0; i < total; i++) out[i] = signal[i]! * env[i]!;
  return out;
}

export function applyHumanize(
  signal: Float64Array,
  amount: number,
  sr: number = SAMPLE_RATE,
  rng?: SeededRng,
): Float64Array {
  const n = signal.length;
  if (amount <= 0 || n < 4) return signal;
  const gen = rng ?? new SeededRng(1337);
  const depth = Math.max(0, Math.min(1, amount)) * 0.075;
  const numKnots = Math.max(3, Math.round((n / sr) * 1.4) + 2);
  const knots = new Float64Array(numKnots);
  for (let i = 0; i < numKnots; i++) {
    knots[i] = Math.max(-0.09, Math.min(0.09, gen.normal(0, depth)));
  }

  const out = new Float64Array(n);
  const scale = (numKnots - 1) / Math.max(1, n - 1);
  for (let i = 0; i < n; i++) {
    const pos = i * scale;
    const idx0 = Math.min(Math.floor(pos), numKnots - 2);
    const frac = pos - idx0;
    const smooth = 0.5 * (1.0 - Math.cos(Math.PI * frac));
    const drift = knots[idx0]! * (1.0 - smooth) + knots[idx0 + 1]! * smooth;
    out[i] = signal[i]! * (1.0 + drift);
  }
  return out;
}

export function softLimit(signal: Float64Array, ceiling = 0.92): Float64Array {
  if (signal.length === 0 || ceiling <= 0) return signal;
  const knee = ceiling * 0.78;
  let maxAbs = 0;
  for (let i = 0; i < signal.length; i++) {
    const a = Math.abs(signal[i]!);
    if (a > maxAbs) maxAbs = a;
  }
  if (maxAbs <= knee) return signal;
  const span = Math.max(1e-6, ceiling - knee);
  const out = new Float64Array(signal.length);
  for (let i = 0; i < signal.length; i++) {
    const x = signal[i]!;
    const a = Math.abs(x);
    if (a <= knee) {
      out[i] = x;
    } else {
      const comp = knee + span * Math.tanh((a - knee) / span);
      out[i] = x >= 0 ? comp : -comp;
    }
  }
  return out;
}

export function softFadeEdges(
  signal: Float64Array,
  fadeInSec = 0.0,
  fadeOutSec = 0.0,
  sr: number = SAMPLE_RATE,
): Float64Array {
  const n = signal.length;
  const out = new Float64Array(signal);
  if (n < 2) return out;
  const inN = Math.min(n >> 1, Math.max(0, Math.round(fadeInSec * sr)));
  const outN = Math.min(n - inN, Math.max(0, Math.round(fadeOutSec * sr)));
  if (inN >= 2) {
    const denom = inN - 1;
    for (let i = 0; i < inN; i++) {
      out[i]! *= 0.5 * (1.0 - Math.cos((Math.PI * i) / denom));
    }
  }
  if (outN >= 2) {
    const start = n - outN;
    const denom = outN - 1;
    for (let i = 0; i < outN; i++) {
      out[start + i]! *= 0.5 * (1.0 + Math.cos((Math.PI * i) / denom));
    }
  }
  return out;
}

function applyDelay(
  signal: Float64Array,
  delayTime: number,
  feedback: number,
  wet: number,
  sr: number = SAMPLE_RATE,
): Float64Array {
  const n = signal.length;
  if (n === 0 || wet <= 0 || delayTime <= 0) return signal;
  const safeFb = Math.max(0, Math.min(0.72, feedback));
  const safeWet = Math.max(0, Math.min(0.55, wet));
  const delaySamples = Math.round(delayTime * sr);
  if (delaySamples <= 0 || delaySamples >= n) return signal;

  const tapSource = applyLowpass(signal, 2800.0, sr);
  const out = new Float64Array(signal);
  for (let tap = 1; tap <= 5; tap++) {
    const off = delaySamples * tap;
    if (off >= n) break;
    const gain = Math.pow(safeFb, tap) * safeWet;
    if (gain < 0.005) break;
    for (let i = off; i < n; i++) {
      out[i]! += tapSource[i - off]! * gain;
    }
  }
  return out;
}

function applyReverb(
  signal: Float64Array,
  room: number,
  size: number,
  sr: number = SAMPLE_RATE,
): Float64Array {
  const n = signal.length;
  const safeRoom = Math.max(0, Math.min(1, room));
  const safeSize = Math.max(0, Math.min(1, size));
  if (n === 0 || safeRoom <= 0.01 || safeSize <= 0.01) return signal;

  const diffused = new Float64Array(signal);
  for (const [dSec, dGain] of DIFFUSION_TAPS) {
    const off = Math.round(dSec * sr);
    if (off > 0 && off < n) {
      for (let i = off; i < n; i++) diffused[i]! += signal[i - off]! * dGain;
    }
  }

  const sizeScale = 0.45 + 0.9 * safeSize;
  const wet = new Float64Array(n);
  for (const [tSec, tGain] of REVERB_TAPS) {
    const off = Math.round(tSec * sizeScale * sr);
    if (off > 0 && off < n) {
      for (let i = off; i < n; i++) wet[i]! += diffused[i - off]! * tGain;
    }
  }

  const damped = applyLowpass(wet, 4200.0 - 2000.0 * safeSize, sr);
  const wetMix = safeRoom * 0.48;
  const dryMix = 1.0 - safeRoom * 0.16;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = signal[i]! * dryMix + damped[i]! * wetMix;
  }
  return out;
}

function applyStereoWidth(
  signal: Float64Array,
  width: number,
  sr: number = SAMPLE_RATE,
): Float64Array {
  const safeWidth = Math.max(0, Math.min(1, width));
  const n = signal.length;
  if (safeWidth <= 0 || n < 16) return signal;
  const air = applyHighpass(signal, 320.0, sr);
  const out = new Float64Array(signal);
  const taps: readonly (readonly [number, number])[] = [
    [0.0097, 0.055 * safeWidth],
    [0.0153, -0.042 * safeWidth],
  ];
  for (const [dSec, gain] of taps) {
    const off = Math.round(dSec * sr);
    if (off > 0 && off < n) {
      for (let i = off; i < n; i++) out[i]! += air[i - off]! * gain;
    }
  }
  return out;
}

function addNote(signal: Float64Array, note: Float64Array, startIdx: number): void {
  if (startIdx >= signal.length || note.length === 0) return;
  const endIdx = startIdx + note.length;
  if (endIdx <= signal.length) {
    for (let i = 0; i < note.length; i++) signal[startIdx + i]! += note[i]!;
    return;
  }
  const available = signal.length - startIdx;
  const fadeSamples = Math.min(Math.round(SAMPLE_RATE * 0.025), Math.max(2, Math.floor(available / 3)));
  const fadeStart = Math.max(0, available - fadeSamples);
  const denom = Math.max(1, fadeSamples - 1);
  for (let i = 0; i < available; i++) {
    let sample = note[i]!;
    if (fadeSamples >= 2 && i >= fadeStart) {
      const k = i - fadeStart;
      sample *= 0.5 * (1.0 + Math.cos((Math.PI * k) / denom));
    }
    signal[startIdx + i]! += sample;
  }
}

// === Radix-2 FFT + calibrated master hall reverb ===

function fftRadix2(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      const tr = re[i]!;
      re[i] = re[j]!;
      re[j] = tr;
      const ti = im[i]!;
      im[i] = im[j]!;
      im[j] = ti;
    }
  }
  const sign = inverse ? 1.0 : -1.0;
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (sign * TWO_PI) / len;
    const wCos = Math.cos(ang);
    const wSin = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1.0;
      let curIm = 0.0;
      for (let j = 0; j < half; j++) {
        const uIdx = i + j;
        const vIdx = uIdx + half;
        const uRe = re[uIdx]!;
        const uIm = im[uIdx]!;
        const vRe = re[vIdx]! * curRe - im[vIdx]! * curIm;
        const vIm = re[vIdx]! * curIm + im[vIdx]! * curRe;
        re[uIdx] = uRe + vRe;
        im[uIdx] = uIm + vIm;
        re[vIdx] = uRe - vRe;
        im[vIdx] = uIm - vIm;
        const nextRe = curRe * wCos - curIm * wSin;
        curIm = curRe * wSin + curIm * wCos;
        curRe = nextRe;
      }
    }
  }
  if (inverse) {
    const invN = 1.0 / n;
    for (let i = 0; i < n; i++) {
      re[i]! *= invN;
      im[i]! *= invN;
    }
  }
}

function nextPow2(v: number): number {
  let p = 1;
  while (p < v) p <<= 1;
  return p;
}

const HALL_IR_CACHE = new Map<number, Float64Array>();

function masterReverbIr(sizeBucket: number, sr: number = SAMPLE_RATE): Float64Array {
  const key = sizeBucket * 100000 + sr;
  const cached = HALL_IR_CACHE.get(key);
  if (cached) return cached;

  const size = sizeBucket / 10.0;
  const rt60 = 0.9 + 1.8 * size;
  const irLen = Math.floor(sr * Math.min(1.5, rt60 * 0.7));
  const rng = new SeededRng(20261125 + sizeBucket);
  const raw = new Float64Array(irLen);
  for (let i = 0; i < irLen; i++) {
    const t = i / sr;
    raw[i] = rng.normal(0, 1) * Math.pow(10, (-3.0 * t) / rt60);
  }

  const predelay = Math.floor(0.022 * sr);
  const onset = Math.floor(0.03 * sr);
  let ir: Float64Array = new Float64Array(irLen);
  if (predelay < irLen) {
    const bodyLen = irLen - predelay;
    const rise = Math.min(bodyLen, onset);
    const denom = Math.max(1, rise - 1);
    for (let i = 0; i < bodyLen; i++) {
      let s = raw[i]!;
      if (rise >= 2 && i < rise) {
        s *= 0.5 * (1.0 - Math.cos((Math.PI * i) / denom));
      }
      ir[predelay + i] = s;
    }
  }

  const tailN = Math.min(irLen >> 2, Math.floor(0.18 * sr));
  if (tailN >= 2) {
    const start = irLen - tailN;
    const denom = tailN - 1;
    for (let i = 0; i < tailN; i++) {
      ir[start + i]! *= 0.5 * (1.0 + Math.cos((Math.PI * i) / denom));
    }
  }

  ir = applyHighpass(ir, 180.0, sr);
  ir = applyLowpass(ir, 3600.0 - 1500.0 * size, sr);

  const fftLen = nextPow2(Math.max(256, irLen * 2));
  const re = new Float64Array(fftLen);
  const im = new Float64Array(fftLen);
  re.set(ir);
  fftRadix2(re, im, false);
  const half = (fftLen >> 1) + 1;
  let sumSq = 0;
  let peakResp = 0;
  for (let i = 0; i < half; i++) {
    const mag = Math.hypot(re[i]!, im[i]!);
    sumSq += mag * mag;
    if (mag > peakResp) peakResp = mag;
  }
  const rmsResp = Math.sqrt(sumSq / half);
  const norm = Math.max(rmsResp, peakResp * 0.55) + 1e-12;
  for (let i = 0; i < irLen; i++) ir[i]! /= norm;

  HALL_IR_CACHE.set(key, ir);
  return ir;
}

export function applyMasterHallReverb(
  signal: Float64Array,
  space: number,
  sr: number = SAMPLE_RATE,
): Float64Array {
  const safeSpace = Math.max(0, Math.min(1, space));
  const n = signal.length;
  if (n < 64 || safeSpace <= 0.12) return signal;

  const bucket = Math.max(1, Math.min(10, Math.round(safeSpace * 10.0)));
  const ir = masterReverbIr(bucket, sr);
  const fftN = nextPow2(n + ir.length - 1);

  const sigRe = new Float64Array(fftN);
  const sigIm = new Float64Array(fftN);
  sigRe.set(signal);
  fftRadix2(sigRe, sigIm, false);

  const irRe = new Float64Array(fftN);
  const irIm = new Float64Array(fftN);
  irRe.set(ir);
  fftRadix2(irRe, irIm, false);

  for (let i = 0; i < fftN; i++) {
    const srRe = sigRe[i]!;
    const srIm = sigIm[i]!;
    const hRe = irRe[i]!;
    const hIm = irIm[i]!;
    sigRe[i] = srRe * hRe - srIm * hIm;
    sigIm[i] = srRe * hIm + srIm * hRe;
  }
  fftRadix2(sigRe, sigIm, true);

  const wetGain = 0.08 + 0.22 * safeSpace;
  const dryGain = 1.0 - 0.14 * safeSpace;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = signal[i]! * dryGain + sigRe[i]! * wetGain;
  }
  return out;
}

// === Harmony + Pattern Generators ===

interface ChordEvent {
  readonly startBeat: number;
  readonly durationBeats: number;
  readonly rootDegree: number;
}

interface HarmonyPlan {
  readonly chords: readonly ChordEvent[];
  readonly beatsPerBar: number;
}

function secondsPerBeat(cfg: SynthConfig): number {
  return 60.0 / bpmFromTempo(cfg.tempo);
}

function selectHarmonyTemplate(cfg: SynthConfig, rng: SeededRng): readonly number[] {
  let style = (cfg.harmony_style || "auto").toLowerCase();
  if (style === "auto") {
    if (cfg.rhythm === "none" && cfg.space >= 0.7) style = "ambient";
    else if (cfg.pad === "cinematic" || cfg.bass === "drone" || cfg.bass === "sub_pulse") style = "cinematic";
    else if (cfg.texture === "vinyl_crackle" || cfg.melody === "ornamental") style = "jazz";
    else style = "pop";
  }
  const majorish = cfg.mode === "major" || cfg.mode === "mixolydian";
  let key: string;
  if (style === "ambient") key = "ambient";
  else if (style === "jazz") key = majorish ? "jazz_major" : "jazz_minor";
  else if (style === "cinematic") key = majorish ? "pop_major" : "cinematic_minor";
  else key = majorish ? "pop_major" : "pop_minor";
  const choices = HARMONY_TEMPLATES[key]!;
  return choices[rng.intRange(0, choices.length)]!;
}

function buildHarmonyPlan(
  cfg: SynthConfig,
  duration: number,
  rng: SeededRng,
  tOffset = 0.0,
): HarmonyPlan {
  const spb = secondsPerBeat(cfg);
  const chordBeats = Math.max(1, Math.round(cfg.chord_change_bars)) * 4;
  const template = selectHarmonyTemplate(cfg, rng);
  const startBeatG = Math.max(0.0, tOffset / spb);
  const endBeatG = Math.max(startBeatG + 1e-4, (tOffset + duration) / spb);
  const firstIdx = Math.floor(startBeatG / chordBeats);
  const lastIdx = Math.max(firstIdx + 1, Math.ceil(endBeatG / chordBeats));

  const chords: ChordEvent[] = [];
  for (let idx = firstIdx; idx < lastIdx; idx++) {
    const segStartG = Math.max(startBeatG, idx * chordBeats);
    const segEndG = Math.min(endBeatG, (idx + 1) * chordBeats);
    const segDur = segEndG - segStartG;
    if (segDur <= 1e-6) continue;
    const tIdx = ((idx % template.length) + template.length) % template.length;
    chords.push({
      startBeat: segStartG - startBeatG,
      durationBeats: segDur,
      rootDegree: ((template[tIdx]! % 7) + 7) % 7,
    });
  }
  return { chords, beatsPerBar: 4 };
}

function chordRootAtBeat(beat: number, harmony: HarmonyPlan): number {
  if (harmony.chords.length === 0) return 0;
  for (const ch of harmony.chords) {
    if (beat >= ch.startBeat && beat < ch.startBeat + ch.durationBeats) {
      return ch.rootDegree;
    }
  }
  if (beat < harmony.chords[0]!.startBeat) return harmony.chords[0]!.rootDegree;
  return harmony.chords[harmony.chords.length - 1]!.rootDegree;
}

function chordSegments(
  cfg: SynthConfig,
  duration: number,
  harmony: HarmonyPlan,
): { start: number; dur: number; root: number }[] {
  if (harmony.chords.length === 0) return [{ start: 0, dur: duration, root: 0 }];
  const spb = secondsPerBeat(cfg);
  const out: { start: number; dur: number; root: number }[] = [];
  for (const ch of harmony.chords) {
    const start = ch.startBeat * spb;
    if (start >= duration) break;
    const end = Math.min(duration, start + ch.durationBeats * spb);
    const dur = Math.max(0, end - start);
    if (dur > 0) out.push({ start, dur, root: ch.rootDegree });
  }
  return out;
}

function chordTonesAscending(ext: string, root: number): number[] {
  const tones = [root, root + 2, root + 4];
  if (ext === "sevenths" || ext === "lush") tones.push(root + 6);
  if (ext === "lush") tones.push(root + 8);
  return tones;
}

function chordToneClasses(root: number, ext: string): number[] {
  const r = ((root % 7) + 7) % 7;
  const raw = [r, (r + 2) % 7, (r + 4) % 7];
  if (ext === "sevenths" || ext === "lush") raw.push((r + 6) % 7);
  if (ext === "lush") raw.push((r + 1) % 7);
  return [...new Set(raw)];
}

function firstGridSlot(tOffset: number, slotSec: number): number {
  const slot = Math.floor(tOffset / slotSec);
  return slot * slotSec < tOffset - 1e-9 ? slot + 1 : slot;
}

function renderSoftKick(
  baseFreq: number,
  kickDur: number,
  amp: number,
  decayRate: number,
  lowpassHz: number,
  sr: number = SAMPLE_RATE,
): Float64Array {
  const n = sampleCount(kickDur, sr);
  if (n < 4) return new Float64Array(n);
  const kick = new Float64Array(n);
  const invSr = 1.0 / sr;
  let phase = 0.0;
  for (let i = 0; i < n; i++) {
    const t = i * invSr;
    const fInst = baseFreq * (1.0 + 0.75 * Math.exp(-t * 32.0));
    phase += (TWO_PI * fInst) * invSr;
    const env = Math.exp((-decayRate * i) / Math.max(1, n - 1));
    kick[i] = amp * Math.sin(phase) * env;
  }
  const atk = Math.min(n >> 2, Math.max(2, Math.round(0.004 * sr)));
  const rel = Math.min(Math.floor(n / 3), Math.max(2, Math.round(0.015 * sr)));
  for (let i = 0; i < atk; i++) kick[i]! *= 0.5 * (1.0 - Math.cos((Math.PI * i) / (atk - 1)));
  for (let i = 0; i < rel; i++) {
    kick[n - rel + i]! *= 0.5 * (1.0 + Math.cos((Math.PI * i) / (rel - 1)));
  }
  const filtered = applyLowpass(kick, lowpassHz, sr);
  filtered[0] = 0;
  filtered[n - 1] = 0;
  return filtered;
}

// --- Bass ---

function renderBassSubPulse(
  cfg: SynthConfig,
  duration: number,
  harmony: HarmonyPlan,
  tOffset: number,
): Float64Array {
  const sr = SAMPLE_RATE;
  const atkMult = ATTACK_MULT[cfg.attack] ?? 1.0;
  const sig = new Float64Array(sampleCount(duration, sr));
  const spb = secondsPerBeat(cfg);
  const pulseSec = 2.0 * spb;
  if (pulseSec <= 0) return sig;
  let slotIdx = firstGridSlot(tOffset, pulseSec);
  while (true) {
    const localSec = slotIdx * pulseSec - tOffset;
    if (localSec >= duration) break;
    const root = chordRootAtBeat(localSec / spb, harmony);
    const freq = scaleFreq(cfg, root, 1);
    let note = generateSine(freq, pulseSec * 0.95, sr, 0.4, tOffset + localSec);
    note = applyLowpass(note, 55.0, sr);
    note = applyAdsr(note, 0.25 * atkMult, 0.2, 0.9, 0.7 * atkMult, sr);
    addNote(sig, note, Math.round(localSec * sr));
    slotIdx += 1;
  }
  return sig;
}

function renderBass(
  cfg: SynthConfig,
  duration: number,
  harmony: HarmonyPlan,
  rng: SeededRng,
  tOffset: number,
): Float64Array {
  const sr = SAMPLE_RATE;
  const atkMult = ATTACK_MULT[cfg.attack] ?? 1.0;
  const sig = new Float64Array(sampleCount(duration, sr));
  const style = cfg.bass;

  if (style === "sub_pulse") return renderBassSubPulse(cfg, duration, harmony, tOffset);

  if (style === "pulsing" || style === "arp_bass") {
    const spb = secondsPerBeat(cfg);
    const pulseSec = 0.5 * spb;
    if (pulseSec <= 0) return sig;
    let slotIdx = firstGridSlot(tOffset, pulseSec);
    while (true) {
      const localSec = slotIdx * pulseSec - tOffset;
      if (localSec >= duration) break;
      const root = chordRootAtBeat(localSec / spb, harmony);
      const freq = scaleFreq(cfg, root, 2);
      let note = generateSine(freq, pulseSec * 0.82, sr, 0.35, tOffset + localSec);
      note = applyLowpass(note, 90.0 * cfg.brightness + 20.0, sr);
      note = applyAdsr(note, 0.02 * atkMult, 0.08, 0.6, 0.25 * atkMult, sr);
      addNote(sig, note, Math.round(localSec * sr));
      slotIdx += 1;
    }
    return sig;
  }

  if (style === "walking") {
    const spb = secondsPerBeat(cfg);
    if (spb <= 0) return sig;
    const walkRel = [0, 2, 4, 2] as const;
    let slotIdx = firstGridSlot(tOffset, spb);
    while (true) {
      const localSec = slotIdx * spb - tOffset;
      if (localSec >= duration) break;
      const root = chordRootAtBeat(localSec / spb, harmony);
      const rel = walkRel[((slotIdx % 4) + 4) % 4]!;
      const freq = scaleFreq(cfg, root + rel, 2);
      let note = generateTriangle(freq, spb * 0.92, sr, 0.3, tOffset + localSec);
      note = applyLowpass(note, 120.0 * cfg.brightness + 40.0, sr);
      note = applyAdsr(note, 0.04 * atkMult, 0.12, 0.7, 0.22 * atkMult, sr);
      note = applyHumanize(note, cfg.human, sr, rng);
      addNote(sig, note, Math.round(localSec * sr));
      slotIdx += 1;
    }
    return applyReverb(sig, cfg.space * 0.3, cfg.space * 0.8, sr);
  }

  if (style === "sustained" || style === "octave") {
    const pattern = [0, 0, 4, 0] as const;
    for (const seg of chordSegments(cfg, duration, harmony)) {
      const noteDur = seg.dur / pattern.length;
      for (let i = 0; i < pattern.length; i++) {
        const startSec = seg.start + i * noteDur;
        const freq = scaleFreq(cfg, seg.root + pattern[i]!, 2);
        let note = generateSine(freq, noteDur * 0.95, sr, 0.32, tOffset + startSec);
        note = applyLowpass(note, 100.0 * cfg.brightness + 30.0, sr);
        note = applyAdsr(note, 0.7 * atkMult, 0.3, 0.85, 1.2 * atkMult, sr);
        addNote(sig, note, Math.round(startSec * sr));
      }
    }
    return applyReverb(sig, cfg.space * 0.4, cfg.space, sr);
  }

  if (style === "fifth_drone") {
    for (const seg of chordSegments(cfg, duration, harmony)) {
      const noteDur = Math.min(seg.dur * 1.15, duration - seg.start);
      let root = generateSine(scaleFreq(cfg, seg.root, 2), noteDur, sr, 0.26, tOffset + seg.start);
      root = applyLowpass(root, 70.0 * cfg.brightness + 20.0, sr);
      root = applyAdsr(root, 2.2 * atkMult, 0.5, 0.95, 2.6 * atkMult, sr);
      let fifth = generateSine(scaleFreq(cfg, seg.root + 4, 2), noteDur, sr, 0.18, tOffset + seg.start);
      fifth = applyLowpass(fifth, 100.0 * cfg.brightness + 30.0, sr);
      fifth = applyAdsr(fifth, 2.4 * atkMult, 0.5, 0.9, 2.6 * atkMult, sr);
      for (let i = 0; i < root.length; i++) root[i]! += fifth[i]!;
      addNote(sig, root, Math.round(seg.start * sr));
    }
    return applyReverb(sig, cfg.space * 0.5, cfg.space, sr);
  }

  // Default: drone
  for (const seg of chordSegments(cfg, duration, harmony)) {
    const noteDur = Math.min(seg.dur * 1.15, duration - seg.start);
    let note = generateSine(scaleFreq(cfg, seg.root, 2), noteDur, sr, 0.35, tOffset + seg.start);
    note = applyLowpass(note, 80.0 * cfg.brightness + 20.0, sr);
    note = applyAdsr(note, 1.8 * atkMult, 0.5, 0.95, 2.2 * atkMult, sr);
    addNote(sig, note, Math.round(seg.start * sr));
  }
  return applyReverb(sig, cfg.space * 0.5, cfg.space, sr);
}

// --- Pads ---

function renderPad(
  cfg: SynthConfig,
  duration: number,
  harmony: HarmonyPlan,
  rng: SeededRng,
  tOffset: number,
): Float64Array {
  const sr = SAMPLE_RATE;
  const atkMult = ATTACK_MULT[cfg.attack] ?? 1.0;
  const echoMult = cfg.echo / 0.5;
  const sig = new Float64Array(sampleCount(duration, sr));
  const style = cfg.pad;

  if (style === "dark_sustained") {
    for (const seg of chordSegments(cfg, duration, harmony)) {
      const noteDur = Math.min(seg.dur * 1.15, duration - seg.start);
      for (const deg of chordTonesAscending(cfg.chord_extensions, seg.root).slice(0, 3)) {
        let tone = generateSawtooth(scaleFreq(cfg, deg, 3), noteDur, sr, 0.12, tOffset + seg.start);
        tone = applyLowpass(tone, 200.0 * cfg.brightness + 80.0, sr);
        tone = applyAdsr(tone, 2.0 * atkMult, 1.0, 0.9, 3.0 * atkMult, sr);
        addNote(sig, tone, Math.round(seg.start * sr));
      }
    }
    return applyReverb(sig, cfg.space * 0.8, cfg.space, sr);
  }

  if (style === "cinematic") {
    const voicings: readonly (readonly [number, number])[] = [
      [0, 3],
      [2, 3],
      [4, 3],
      [0, 4],
      [4, 4],
    ];
    for (const seg of chordSegments(cfg, duration, harmony)) {
      const noteDur = Math.min(seg.dur * 1.1, duration - seg.start);
      for (const [rel, oct] of voicings) {
        const freq = scaleFreq(cfg, seg.root + rel, oct);
        const saw = generateSawtooth(freq, noteDur, sr, 0.08, tOffset + seg.start);
        const tri = generateTriangle(freq * 1.002, noteDur, sr, 0.06, tOffset + seg.start);
        for (let i = 0; i < saw.length; i++) saw[i]! += tri[i]!;
        let tone = applyLowpass(saw, 400.0 * cfg.brightness + 150.0, sr);
        tone = applyAdsr(tone, 1.8 * atkMult, 0.8, 0.88, 2.8 * atkMult, sr);
        addNote(sig, tone, Math.round(seg.start * sr));
      }
    }
    return applyDelay(
      applyReverb(sig, cfg.space * 0.85, cfg.space, sr),
      0.4,
      0.35 * echoMult,
      0.3 * echoMult,
      sr,
    );
  }

  if (style === "thin_high" || style === "bright_open") {
    for (const seg of chordSegments(cfg, duration, harmony)) {
      const noteDur = Math.min(seg.dur * 1.15, duration - seg.start);
      for (const rel of [0, 4]) {
        let tone = generateSine(scaleFreq(cfg, seg.root + rel, 4), noteDur, sr, 0.12, tOffset + seg.start);
        tone = applyHighpass(applyLowpass(tone, 800.0 * cfg.brightness + 200.0, sr), 200.0, sr);
        tone = applyAdsr(tone, 1.2 * atkMult, 0.6, 0.8, 2.0 * atkMult, sr);
        addNote(sig, tone, Math.round(seg.start * sr));
      }
    }
    return applyReverb(sig, cfg.space * 0.75, cfg.space * 0.95, sr);
  }

  if (style === "ambient_drift") {
    for (const seg of chordSegments(cfg, duration, harmony)) {
      const noteDur = Math.min(seg.dur * 1.35, duration - seg.start);
      const tones = chordTonesAscending(cfg.chord_extensions, seg.root).slice(0, 3);
      if (cfg.motion > 0.4 && rng.next() < 0.15 + 0.25 * cfg.motion) {
        tones.push(seg.root + 5);
      }
      for (const deg of tones) {
        let tone = generateSine(scaleFreq(cfg, deg, 3), noteDur, sr, 0.14, tOffset + seg.start);
        tone = applyLowpass(tone, 350.0 * cfg.brightness + 100.0, sr);
        tone = applyAdsr(tone, 1.6 * atkMult, 0.5, 0.85, 2.2 * atkMult, sr);
        addNote(sig, tone, Math.round(seg.start * sr));
      }
    }
    return applyDelay(
      applyReverb(sig, cfg.space * 0.8, cfg.space, sr),
      0.5,
      0.4 * echoMult,
      0.35 * echoMult,
      sr,
    );
  }

  if (style === "stacked_fifths") {
    const voicings: readonly (readonly [number, number])[] = [
      [0, 3],
      [4, 3],
      [0, 4],
      [4, 4],
    ];
    for (const seg of chordSegments(cfg, duration, harmony)) {
      const noteDur = Math.min(seg.dur * 1.1, duration - seg.start);
      for (const [rel, oct] of voicings) {
        let tone = oscForGrain(
          cfg.grain,
          scaleFreq(cfg, seg.root + rel, oct),
          noteDur,
          sr,
          0.1,
          tOffset + seg.start,
          "triangle",
        );
        tone = applyLowpass(tone, 500.0 * cfg.brightness + 150.0, sr);
        tone = applyAdsr(tone, 1.3 * atkMult, 0.7, 0.88, 2.2 * atkMult, sr);
        addNote(sig, tone, Math.round(seg.start * sr));
      }
    }
    return applyReverb(sig, cfg.space * 0.7, cfg.space * 0.85, sr);
  }

  // Default: warm_slow (repaired proportional motion LFO + continuous tOffset)
  const lfoRate = 0.06 + 0.28 * Math.max(0, Math.min(1, cfg.motion));
  for (const seg of chordSegments(cfg, duration, harmony)) {
    const noteDur = Math.min(seg.dur * 1.15, duration - seg.start);
    for (const deg of chordTonesAscending(cfg.chord_extensions, seg.root).slice(0, 3)) {
      const raw = oscForGrain(
        cfg.grain,
        scaleFreq(cfg, deg, 3),
        noteDur,
        sr,
        0.15,
        tOffset + seg.start,
        "sine",
      );
      const lfo = generateLfo(noteDur, lfoRate, sr, tOffset + seg.start);
      const baseCutoff = 300.0 * cfg.brightness + 100.0;
      const lo = applyLowpass(raw, baseCutoff * 0.5, sr);
      const hi = applyLowpass(raw, baseCutoff * 1.5, sr);
      const blended = new Float64Array(raw.length);
      for (let i = 0; i < raw.length; i++) {
        blended[i] = lo[i]! * (1.0 - lfo[i]!) + hi[i]! * lfo[i]!;
      }
      const env = applyAdsr(blended, 1.2 * atkMult, 0.6, 0.85, 1.6 * atkMult, sr);
      addNote(sig, env, Math.round(seg.start * sr));
    }
  }
  const rev = applyReverb(sig, cfg.space * 0.7, cfg.space * 0.9, sr);
  const del = applyDelay(rev, 0.35, 0.3 * echoMult, 0.25 * echoMult, sr);
  return applyHumanize(del, cfg.human, sr, rng);
}

// --- Melodies (BPM-locked patterns + procedural engine) ---

function renderBpmPatternMelody(
  cfg: SynthConfig,
  duration: number,
  harmony: HarmonyPlan,
  pattern: readonly number[],
  stepBeats: number,
  octave: number,
  amp: number,
  durMult: number,
  cutoff: number,
  adsr: readonly [number, number, number, number],
  delayCfg: readonly [number, number, number],
  reverbScale: readonly [number, number],
  rng: SeededRng,
  tOffset: number,
): Float64Array {
  const sr = SAMPLE_RATE;
  const sig = new Float64Array(sampleCount(duration, sr));
  const atkMult = ATTACK_MULT[cfg.attack] ?? 1.0;
  const echoMult = cfg.echo / 0.5;
  const spb = secondsPerBeat(cfg);
  const slotSec = stepBeats * spb;
  if (slotSec <= 0 || pattern.length === 0) return sig;

  let slotIdx = firstGridSlot(tOffset, slotSec);
  while (true) {
    const localSec = slotIdx * slotSec - tOffset;
    if (localSec >= duration) break;
    const relDeg = pattern[((slotIdx % pattern.length) + pattern.length) % pattern.length]!;
    if (relDeg >= 0) {
      const root = chordRootAtBeat(localSec / spb, harmony);
      const freq = scaleFreq(cfg, root + relDeg, octave);
      const noteDur = Math.min(slotSec * durMult, Math.max(0.06, duration - localSec));
      let note = oscForGrain(cfg.grain, freq, noteDur, sr, amp, tOffset + localSec, "triangle");
      note = applyLowpass(note, cutoff, sr);
      note = applyAdsr(note, adsr[0] * atkMult, adsr[1], adsr[2], adsr[3] * atkMult, sr);
      note = applyHumanize(note, cfg.human, sr, rng);
      addNote(sig, note, Math.round(localSec * sr));
    }
    slotIdx += 1;
  }

  const del = applyDelay(sig, delayCfg[0], delayCfg[1] * echoMult, delayCfg[2] * echoMult, sr);
  return applyReverb(del, cfg.space * reverbScale[0], cfg.space * reverbScale[1], sr);
}

function tensionValue(curve: string, xRaw: number): number {
  const x = Math.max(0, Math.min(1, xRaw));
  if (curve === "ramp") return x;
  if (curve === "waves") return 0.5 - 0.5 * Math.cos(TWO_PI * 2.0 * x);
  return Math.sin(Math.PI * x);
}

function weightedChoice(rng: SeededRng, items: number[], weights: number[]): number {
  let sum = 0;
  for (const w of weights) sum += w > 0 ? w : 0;
  if (sum <= 0) return items[rng.intRange(0, items.length)]!;
  let r = rng.next() * sum;
  for (let i = 0; i < items.length; i++) {
    r -= Math.max(0, weights[i]!);
    if (r <= 0) return items[i]!;
  }
  return items[items.length - 1]!;
}

function nearestChordTone(
  pitch: number,
  chordRoot: number,
  minPitch: number,
  maxPitch: number,
  ext: string,
): number {
  const tones = chordToneClasses(chordRoot, ext);
  let best = Math.max(minPitch, Math.min(maxPitch, pitch));
  let bestDist = Infinity;
  const maxOct = Math.floor(maxPitch / 7) + 1;
  for (const tone of tones) {
    for (let o = 0; o <= maxOct; o++) {
      const cand = tone + 7 * o;
      if (cand >= minPitch && cand <= maxPitch) {
        const d = Math.abs(cand - pitch);
        if (d < bestDist) {
          bestDist = d;
          best = cand;
        }
      }
    }
  }
  return best;
}

function barAnchorPitch(
  cfg: SynthConfig,
  barIdx: number,
  baseSeed: number,
  chordRoot: number,
  minPitch: number,
  maxPitch: number,
  phraseLen: number,
): number {
  const posInPhrase = ((barIdx % phraseLen) + phraseLen) % phraseLen;
  const x = posInPhrase / Math.max(1, phraseLen - 1);
  const isCadence = posInPhrase === phraseLen - 1 && phraseLen > 1;
  const tension = isCadence ? 0.0 : tensionValue(cfg.tension_curve, x);
  const cadStr = isCadence ? cfg.cadence_strength : 0.1;

  const tones = chordToneClasses(chordRoot, cfg.chord_extensions);
  const prev = Math.floor((maxPitch - minPitch) / 2);
  const mid = 0.5 * (minPitch + maxPitch);
  const desired = mid + (tension - 0.5) * 0.35 * (maxPitch - minPitch);
  const items: number[] = [];
  const weights: number[] = [];
  const rootMod = ((chordRoot % 7) + 7) % 7;
  for (const tone of tones) {
    let baseW = 0.75;
    if (tone === rootMod) baseW = 1.0 + 1.8 * cadStr;
    else if (tone === (rootMod + 2) % 7) baseW = 1.0;
    else if (tone === (rootMod + 4) % 7) baseW = 0.95;
    else if (tone === (rootMod + 6) % 7) baseW = 0.85;
    for (let o = 0; o <= Math.floor(maxPitch / 7); o++) {
      const cand = tone + 7 * o;
      if (cand < minPitch || cand > maxPitch) continue;
      items.push(cand);
      weights.push(
        baseW * Math.exp(-Math.abs(cand - prev) / 2.6) * Math.exp(-Math.abs(cand - desired) / 4.0),
      );
    }
  }
  const barRng = new SeededRng(deriveSeed(baseSeed, 3001 + barIdx * 17));
  return items.length > 0 ? weightedChoice(barRng, items, weights) : prev;
}

function renderMelodyProcedural(
  cfg: SynthConfig,
  duration: number,
  harmony: HarmonyPlan,
  rng: SeededRng,
  tOffset: number,
): Float64Array {
  const sr = SAMPLE_RATE;
  const sig = new Float64Array(sampleCount(duration, sr));
  const atkMult = ATTACK_MULT[cfg.attack] ?? 1.0;
  const echoMult = cfg.echo / 0.5;
  const spb = secondsPerBeat(cfg);
  const barSec = 4.0 * spb;
  if (barSec <= 0) return sig;
  const cutoff = 700.0 * cfg.brightness + 180.0;

  const baseSeed = rng.intRange(1, 0x7fffffff);
  const phraseLen = Math.max(1, Math.min(16, Math.round(cfg.phrase_len_bars)));
  const density = Math.max(0, Math.min(1, cfg.melody_density));
  const regMin = Math.min(cfg.register_min_oct, cfg.register_max_oct);
  const regMax = Math.max(cfg.register_min_oct, cfg.register_max_oct);
  const minPitch = 0;
  const maxPitch = Math.max(0, regMax - regMin) * 7 + 6;
  const stepBeats = density < 0.25 ? 1.0 : density < 0.6 ? 0.5 : 0.25;
  const stepsPerBar = Math.max(1, Math.min(64, Math.round(4 / stepBeats)));

  const firstBar = Math.max(0, Math.floor(Math.max(0, tOffset - 2.0 * spb) / barSec));
  const lastBar = Math.max(firstBar + 1, Math.ceil((tOffset + duration) / barSec));

  for (let bar = firstBar; bar < lastBar; bar++) {
    const barStartSec = bar * barSec;
    const localBarBeat = (barStartSec - tOffset) / spb;
    const chordRoot = chordRootAtBeat(Math.max(0, localBarBeat), harmony);
    const nextChordRoot = chordRootAtBeat(Math.max(0, localBarBeat + 4.0), harmony);

    const posInPhrase = bar % phraseLen;
    const x = posInPhrase / Math.max(1, phraseLen - 1);
    const tension = tensionValue(cfg.tension_curve, x);
    const isCadence = posInPhrase === phraseLen - 1 && phraseLen > 1;
    const anchorPitch = barAnchorPitch(cfg, bar, baseSeed, chordRoot, minPitch, maxPitch, phraseLen);
    const nextAnchor = barAnchorPitch(
      cfg,
      bar + 1,
      baseSeed,
      nextChordRoot,
      minPitch,
      maxPitch,
      phraseLen,
    );

    const barRng = new SeededRng(deriveSeed(baseSeed, 7001 + bar * 29));
    const humRng = new SeededRng(deriveSeed(baseSeed, 9001 + bar * 37));
    let needResolve = false;
    let currentPitch = anchorPitch;

    for (let s = 0; s < stepsPerBar; s++) {
      const posBeats = s * stepBeats;
      let startSec = barStartSec + posBeats * spb;
      if (stepBeats === 0.5 && s % 2 === 1) startSec += cfg.swing * 0.12 * spb;

      let play: boolean;
      let isAnchor = false;
      if (s === 0) {
        play = true;
        isAnchor = true;
      } else {
        const offbeat = Math.abs(posBeats - Math.round(posBeats)) > 1e-6;
        let prob = 0.1 + 0.8 * density + 0.2 * tension;
        prob += offbeat ? 0.25 * cfg.syncopation : -0.15 * cfg.syncopation;
        if (isCadence && posBeats >= 2.0) prob *= 0.35 + 0.4 * (1.0 - cfg.cadence_strength);
        play = barRng.next() < Math.max(0, Math.min(1, prob));
      }
      if (!play) continue;

      let pitch: number;
      if (s === 0) {
        pitch = anchorPitch;
      } else {
        const delta = nextAnchor - currentPitch;
        const dir = delta === 0 ? (barRng.next() < 0.5 ? -1 : 1) : delta > 0 ? 1 : -1;
        if (needResolve) {
          pitch = nearestChordTone(currentPitch, chordRoot, minPitch, maxPitch, cfg.chord_extensions);
          needResolve = false;
        } else {
          const leapProb = (1.0 - cfg.step_bias) * (0.25 + 0.35 * tension);
          let stepSize = 1;
          if (barRng.next() < 0.06 + 0.1 * tension) stepSize = 0;
          else if (Math.abs(delta) >= 4 && barRng.next() < leapProb) stepSize = 2 + barRng.intRange(0, 3);
          pitch = Math.max(minPitch, Math.min(maxPitch, currentPitch + dir * stepSize));
        }
        const tones = chordToneClasses(chordRoot, cfg.chord_extensions);
        if (!tones.includes(((pitch % 7) + 7) % 7)) {
          if (barRng.next() < 0.55 - 0.45 * tension) {
            pitch = nearestChordTone(pitch, chordRoot, minPitch, maxPitch, cfg.chord_extensions);
          } else {
            needResolve = true;
          }
        }
      }

      let durBeats = stepBeats * (0.7 + 0.25 * barRng.next());
      if (isAnchor) durBeats = Math.max(durBeats, isCadence ? 1.6 : 1.0);
      const durSec = Math.max(0.02, durBeats * spb);
      let jitter = 0;
      if (cfg.human > 0 && !isAnchor) {
        jitter = Math.max(-0.02 * spb, Math.min(0.02 * spb, barRng.normal(0, cfg.human * 0.006 * spb)));
      }
      const evStart = Math.max(0, startSec + jitter);
      let amp = 0.12 + 0.05 * tension;
      if (isCadence && posBeats >= 2.0) amp *= 0.85;
      if (isAnchor) amp *= 1.1;

      const useChromatic =
        cfg.chromatic_prob > 0 &&
        !isAnchor &&
        Math.abs(nextAnchor - pitch) <= 2 &&
        barRng.next() < cfg.chromatic_prob * (0.35 + 0.65 * tension);
      const freq = useChromatic
        ? freqFromNote(
            cfg.root,
            semitoneFromDegree(cfg.mode, nextAnchor) - (nextAnchor >= pitch ? 1 : -1),
            regMin,
          )
        : scaleFreq(cfg, pitch, regMin);

      if (evStart < tOffset + duration && evStart + durSec > tOffset) {
        let note = oscForGrain(cfg.grain, freq, durSec, sr, amp, evStart, "triangle");
        note = applyLowpass(note, cutoff, sr);
        const atk = (isAnchor ? 0.05 : 0.03) * atkMult;
        const rel = (isAnchor ? 0.22 : 0.16) * atkMult;
        note = applyAdsr(note, atk, 0.12, 0.55, rel, sr);
        note = applyHumanize(note, cfg.human * 0.6, sr, humRng);

        const localStart = Math.round((evStart - tOffset) * sr);
        if (localStart < 0) {
          const skip = -localStart;
          if (skip < note.length) addNote(sig, note.subarray(skip), 0);
        } else {
          addNote(sig, note, localStart);
        }
      }
      currentPitch = pitch;
    }
  }

  const del = applyDelay(sig, 0.33, 0.35 * echoMult, 0.28 * echoMult, sr);
  return applyReverb(del, cfg.space * 0.55, cfg.space * 0.75, sr);
}

function renderMelody(
  cfg: SynthConfig,
  duration: number,
  harmony: HarmonyPlan,
  rng: SeededRng,
  tOffset: number,
): Float64Array {
  if (cfg.melody_engine === "procedural" || cfg.melody === "procedural") {
    return renderMelodyProcedural(cfg, duration, harmony, rng, tOffset);
  }
  switch (cfg.melody) {
    case "rising":
    case "heroic":
      return renderBpmPatternMelody(
        cfg, duration, harmony,
        [0, -1, 2, -1, 4, -1, -1, 5, -1, -1, 6, -1, 7, -1, -1, -1],
        1.0, 5, 0.16, 1.7, 900.0 * cfg.brightness + 250.0,
        [0.07, 0.22, 0.55, 0.75], [0.33, 0.34, 0.26], [0.55, 0.75], rng, tOffset,
      );
    case "falling":
      return renderBpmPatternMelody(
        cfg, duration, harmony,
        [6, -1, -1, 5, -1, -1, 4, -1, 2, -1, -1, 0, -1, -1, -1, -1],
        1.0, 5, 0.16, 1.6, 850.0 * cfg.brightness + 220.0,
        [0.08, 0.25, 0.52, 0.75], [0.38, 0.36, 0.28], [0.58, 0.78], rng, tOffset,
      );
    case "minimal":
      return renderBpmPatternMelody(
        cfg, duration, harmony,
        [4, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1],
        1.0, 5, 0.19, 2.4, 620.0 * cfg.brightness + 160.0,
        [0.12, 0.35, 0.5, 1.0], [0.5, 0.42, 0.35], [0.7, 0.9], rng, tOffset,
      );
    case "ornamental":
      return renderBpmPatternMelody(
        cfg, duration, harmony,
        [2, 0, -1, -1, 5, 4, -1, -1, 3, 2, -1, -1, 2, 0, -1, -1],
        0.5, 5, 0.16, 1.6, 900.0 * cfg.brightness + 220.0,
        [0.04, 0.22, 0.52, 0.6], [0.3, 0.34, 0.25], [0.5, 0.7], rng, tOffset,
      );
    case "arp_melody":
      return renderBpmPatternMelody(
        cfg, duration, harmony,
        [0, 2, 4, 2],
        0.5, 4, 0.14, 0.85, 1150.0 * cfg.brightness + 280.0,
        [0.015, 0.08, 0.4, 0.14], [0.25, 0.3, 0.24], [0.45, 0.65], rng, tOffset,
      );
    default:
      return renderBpmPatternMelody(
        cfg, duration, harmony,
        [0, -1, 2, -1, 4, -1, 2, -1, 0, -1, -1, -1, 2, -1, -1, -1],
        1.0, 5, 0.17, 1.5, 800.0 * cfg.brightness + 220.0,
        [0.08, 0.25, 0.6, 0.65], [0.35, 0.38, 0.28], [0.6, 0.8], rng, tOffset,
      );
  }
}

// --- Rhythms ---

function gridSlotTimes(
  cfg: SynthConfig,
  duration: number,
  slotsPerBeat: number,
  tOffset: number,
): { times: number[]; firstSlot: number } {
  const slotDur = secondsPerBeat(cfg) / slotsPerBeat;
  if (slotDur <= 0) return { times: [], firstSlot: 0 };
  const firstSlot = firstGridSlot(tOffset, slotDur);
  const times: number[] = [];
  let cur = firstSlot;
  while (true) {
    const localT = cur * slotDur - tOffset;
    if (localT >= duration) break;
    times.push(localT);
    cur += 1;
  }
  return { times, firstSlot };
}

function renderKickGrid(
  cfg: SynthConfig,
  duration: number,
  slotsPerBeat: number,
  pattern: readonly number[],
  baseFreq: number,
  kickDur: number,
  amp: number,
  decayRate: number,
  lowpassHz: number,
  rng: SeededRng,
  tOffset: number,
): Float64Array {
  const sr = SAMPLE_RATE;
  const sig = new Float64Array(sampleCount(duration, sr));
  const { times, firstSlot } = gridSlotTimes(cfg, duration, slotsPerBeat, tOffset);
  for (let i = 0; i < times.length; i++) {
    if (!pattern[(firstSlot + i) % pattern.length]) continue;
    const vel = 1.0 + rng.uniform(-0.07, 0.07) * cfg.human;
    addNote(sig, renderSoftKick(baseFreq, kickDur, amp * vel, decayRate, lowpassHz, sr), Math.round(times[i]! * sr));
  }
  return sig;
}

function renderRhythm(
  cfg: SynthConfig,
  duration: number,
  rng: SeededRng,
  tOffset: number,
): Float64Array {
  const sr = SAMPLE_RATE;
  const style = cfg.rhythm;
  if (style === "none") return new Float64Array(sampleCount(duration, sr));

  if (style === "heartbeat" || style === "tabla_essence") {
    return renderKickGrid(cfg, duration, 2, [1, 1, 0, 0, 0, 0, 0, 0], 54, 0.14, 0.3, 9.0, 120, rng, tOffset);
  }
  if (style === "soft_four" || style === "kit_medium" || style === "military") {
    const sig = renderKickGrid(cfg, duration, 1, [1], 50, 0.19, 0.26, 6.0, 110, rng, tOffset);
    return applyReverb(sig, cfg.space * 0.22, cfg.space * 0.4, sr);
  }
  if (style === "hats_only") {
    const sig = new Float64Array(sampleCount(duration, sr));
    const { times } = gridSlotTimes(cfg, duration, 4, tOffset);
    const hatN = sampleCount(0.045, sr);
    for (const tSec of times) {
      const vel = 1.0 + rng.uniform(-0.12, 0.12) * cfg.human;
      let hat: Float64Array = new Float64Array(hatN);
      for (let i = 0; i < hatN; i++) hat[i] = rng.normal(0, 1) * (0.026 * vel);
      hat = applyLowpass(applyHighpass(hat, 5800, sr), 11500, sr);
      hat = applyAdsr(hat, 0.005, 0.015, 0.06, 0.025, sr);
      addNote(sig, hat, Math.round(tSec * sr));
    }
    return sig;
  }
  if (style === "electronic") {
    const sig = new Float64Array(sampleCount(duration, sr));
    const kicks = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];
    const hats = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0];
    const { times, firstSlot } = gridSlotTimes(cfg, duration, 4, tOffset);
    const hatN = sampleCount(0.05, sr);
    for (let i = 0; i < times.length; i++) {
      const idx = (firstSlot + i) % 16;
      const pos = Math.round(times[i]! * sr);
      if (kicks[idx]) addNote(sig, renderSoftKick(46, 0.2, 0.3, 7.5, 110, sr), pos);
      if (hats[idx]) {
        let hat: Float64Array = new Float64Array(hatN);
        for (let k = 0; k < hatN; k++) hat[k] = rng.normal(0, 1) * 0.028;
        hat = applyLowpass(applyHighpass(hat, 6200, sr), 11500, sr);
        hat = applyAdsr(hat, 0.005, 0.014, 0.06, 0.028, sr);
        addNote(sig, hat, pos);
      }
    }
    return sig;
  }

  // Default: minimal / kit_light / brush
  const sig = renderKickGrid(cfg, duration, 2, [1, 0, 0, 0, 0, 0, 0, 0], 58, 0.17, 0.32, 7.5, 140, rng, tOffset);
  return applyReverb(sig, cfg.space * 0.25, cfg.space * 0.45, sr);
}

// --- Textures ---

function renderTexture(
  cfg: SynthConfig,
  duration: number,
  rng: SeededRng,
  tOffset: number,
): Float64Array {
  const sr = SAMPLE_RATE;
  const n = sampleCount(duration, sr);
  const style = cfg.texture;
  if (style === "none") return new Float64Array(n);

  const atkMult = ATTACK_MULT[cfg.attack] ?? 1.0;
  const echoMult = cfg.echo / 0.5;

  if (style === "vinyl_crackle") {
    const sig = new Float64Array(n);
    const grainN = Math.max(24, Math.round(0.004 * sr));
    const win = new Float64Array(grainN);
    for (let i = 0; i < grainN; i++) win[i] = 0.5 * (1.0 - Math.cos((TWO_PI * i) / (grainN - 1)));
    const count = Math.floor(duration * 6);
    for (let c = 0; c < count; c++) {
      if (n <= grainN + 2) break;
      const pos = rng.intRange(0, n - grainN - 1);
      const amp = rng.uniform(0.002, 0.008);
      let g: Float64Array = new Float64Array(grainN);
      for (let i = 0; i < grainN; i++) g[i] = rng.normal(0, 1) * amp;
      g = applyLowpass(applyHighpass(g, 1600, sr), 5500, sr);
      for (let i = 0; i < grainN; i++) g[i]! *= win[i]!;
      addNote(sig, g, pos);
    }
    let air: Float64Array = new Float64Array(n);
    for (let i = 0; i < n; i++) air[i] = rng.normal(0, 1) * 0.0018;
    air = applyLowpass(applyHighpass(air, 700, sr), 4200, sr);
    for (let i = 0; i < n; i++) sig[i]! += air[i]!;
    return applyAdsr(sig, tOffset > 1e-6 ? 0.025 : 0.25, 0.0, 1.0, 0.05, sr);
  }

  if (style === "breath" || style === "noise_wash" || style === "pad_whisper") {
    let sig: Float64Array = new Float64Array(n);
    for (let i = 0; i < n; i++) sig[i] = rng.normal(0, 1) * 0.032;
    const freq = scaleFreq(cfg, 0, 3);
    sig = applyHighpass(applyLowpass(sig, Math.min(freq * 2.2, 1600), sr), Math.max(freq * 0.5, 90), sr);
    const lfo = generateLfo(duration, 0.08 + 0.24 * Math.max(0, Math.min(1, cfg.motion)), sr, tOffset);
    for (let i = 0; i < n; i++) sig[i]! *= lfo[i]!;
    sig = applyAdsr(sig, tOffset > 1e-6 ? 0.025 : 0.35, 0.0, 1.0, 0.05, sr);
    return applyReverb(sig, cfg.space * 0.6, cfg.space * 0.8, sr);
  }

  if (style === "stars" || style === "crystal") {
    const sig = new Float64Array(n);
    const numStars = Math.floor(duration * 3);
    const degrees = [0, 2, 4, 5] as const;
    for (let i = 0; i < numStars; i++) {
      const pos = rng.intRange(0, Math.max(1, n - sr));
      const deg = degrees[rng.intRange(0, degrees.length)]!;
      const freq = scaleFreq(cfg, deg, 6);
      let star = generateSine(freq, 0.3, sr, rng.uniform(0.02, 0.05), tOffset + pos / sr);
      star = applyAdsr(star, 0.01, 0.1, 0.1, 0.2, sr);
      addNote(sig, star, pos);
    }
    const del = applyDelay(sig, 0.4, 0.5 * echoMult, 0.4 * echoMult, sr);
    return applyReverb(del, cfg.space * 0.9, cfg.space, sr);
  }

  // shimmer / shimmer_slow / glitch
  const slow = style === "shimmer_slow";
  const degrees = slow ? [0, 4] : [0, 2, 4];
  const amp = slow ? 0.026 : 0.03;
  const baseRate = slow ? 0.12 : 0.35;
  const rateSpan = slow ? 0.5 : 1.6;
  const lfoRate = baseRate + rateSpan * Math.max(0, Math.min(1, cfg.motion));
  const atkSec = tOffset > 1e-6 ? 0.025 : 0.35 * atkMult;
  const sig = new Float64Array(n);
  for (let idx = 0; idx < degrees.length; idx++) {
    const tone = generateSine(scaleFreq(cfg, degrees[idx]!, 6), duration, sr, amp, tOffset);
    const lfo = generateLfo(duration, lfoRate * (1.0 + 0.13 * idx), sr, tOffset);
    for (let i = 0; i < n; i++) tone[i]! *= 0.45 + 0.55 * lfo[i]!;
    const env = applyAdsr(tone, atkSec, 0.0, 1.0, 0.05, sr);
    for (let i = 0; i < n; i++) sig[i]! += env[i]!;
  }
  const del = applyDelay(
    sig,
    slow ? 0.5 : 0.4,
    (slow ? 0.48 : 0.45) * echoMult,
    (slow ? 0.34 : 0.3) * echoMult,
    sr,
  );
  return applyReverb(del, cfg.space * (slow ? 0.8 : 0.75), cfg.space, sr);
}

// --- Accents ---

function renderAccent(
  cfg: SynthConfig,
  duration: number,
  harmony: HarmonyPlan,
  rng: SeededRng,
  tOffset: number,
): Float64Array {
  const sr = SAMPLE_RATE;
  const n = sampleCount(duration, sr);
  const style = cfg.accent;
  if (style === "none") return new Float64Array(n);

  const atkMult = ATTACK_MULT[cfg.attack] ?? 1.0;
  const spb = secondsPerBeat(cfg);
  const echoMult = cfg.echo / 0.5;
  const sig = new Float64Array(n);

  if (style === "chime" || style === "blip_random" || style === "wind") {
    const numChimes = Math.floor(duration * 1.5);
    const degrees = [0, 2, 4, 5, 6] as const;
    for (let i = 0; i < numChimes; i++) {
      const maxStart = n - sr;
      const pos = maxStart <= 0 ? 0 : rng.intRange(0, maxStart);
      const deg = degrees[rng.intRange(0, degrees.length)]!;
      const freq = scaleFreq(cfg, deg, 5);
      const chime = generateSine(freq, 1.2, sr, rng.uniform(0.06, 0.12), tOffset + pos / sr);
      const harm = generateSine(freq * 2.0, 1.2, sr, 0.03, tOffset + pos / sr);
      for (let k = 0; k < chime.length; k++) chime[k]! += harm[k]!;
      addNote(sig, applyAdsr(chime, 0.005, 0.3, 0.05, 0.9, sr), pos);
    }
    const del = applyDelay(sig, 0.3, 0.4 * echoMult, 0.3 * echoMult, sr);
    return applyReverb(del, cfg.space * 0.75, cfg.space * 0.9, sr);
  }

  if (style === "pluck" || style === "arp_accent" || style === "piano_note") {
    const slotSec = spb * 0.5;
    const pattern = [0, -1, -1, 4, -1, -1, 2, -1, -1, -1, 0, -1, -1, -1, -1, -1] as const;
    let slotIdx = firstGridSlot(tOffset, slotSec);
    while (true) {
      const localSec = slotIdx * slotSec - tOffset;
      if (localSec >= duration) break;
      const rel = pattern[((slotIdx % pattern.length) + pattern.length) % pattern.length]!;
      if (rel >= 0) {
        const root = chordRootAtBeat(localSec / spb, harmony);
        const freq = scaleFreq(cfg, root + rel, 4);
        const dur = Math.min(0.55, Math.max(0.08, duration - localSec));
        let p = generateTriangle(freq, dur, sr, 0.14, 0.0);
        p = applyLowpass(p, 1500.0 * cfg.brightness + 400.0, sr);
        p = applyAdsr(p, 0.006 * atkMult, 0.14, 0.04, 0.36 * atkMult, sr);
        p = applyHumanize(p, cfg.human, sr, rng);
        addNote(sig, p, Math.round(localSec * sr));
      }
      slotIdx += 1;
    }
    return applyReverb(sig, cfg.space * 0.5, cfg.space * 0.7, sr);
  }

  // bells / bells_dense / blip / brass_hit
  const pattern = [0, -1, -1, -1, 4, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1] as const;
  let slotIdx = firstGridSlot(tOffset, spb);
  while (true) {
    const localSec = slotIdx * spb - tOffset;
    if (localSec >= duration) break;
    const rel = pattern[((slotIdx % pattern.length) + pattern.length) % pattern.length]!;
    if (rel >= 0) {
      const root = chordRootAtBeat(localSec / spb, harmony);
      const freq = scaleFreq(cfg, root + rel, 5);
      const dur = Math.min(1.1, Math.max(0.12, duration - localSec));
      const b1 = generateSine(freq, dur, sr, 0.11, 0.0);
      const b2 = generateSine(freq * 2.0, dur, sr, 0.042, 0.0);
      const b3 = generateSine(freq * 3.0, dur, sr, 0.018, 0.0);
      for (let k = 0; k < b1.length; k++) b1[k]! += b2[k]! + b3[k]!;
      let bell = applyAdsr(b1, 0.008 * atkMult, 0.18, 0.08, 0.65 * atkMult, sr);
      bell = applyHumanize(bell, cfg.human, sr, rng);
      addNote(sig, bell, Math.round(localSec * sr));
    }
    slotIdx += 1;
  }
  return applyReverb(sig, cfg.space * 0.6, cfg.space * 0.8, sr);
}

// === Master Assembler ===

export interface AssembleOptions {
  /** Duration in seconds (default 8.0). */
  duration?: number;
  /** Peak-normalize output to 0.78 (default true; set false for seamless streaming). */
  normalize?: boolean;
  /** Deterministic session seed (default 42). */
  seed?: number;
  /** Continuous timeline offset in seconds (default 0.0). */
  tOffset?: number;
}

/**
 * Assemble all active synthesis layers into headroom-safe, click-free mono Float64 PCM.
 */
export function assemble(input: ConfigInput, options?: AssembleOptions): Float64Array {
  const cfg = toInternalConfig(input);
  const duration = options?.duration ?? 8.0;
  const normalize = options?.normalize ?? true;
  const tOffset = options?.tOffset ?? 0.0;
  const sr = SAMPLE_RATE;
  const n = sampleCount(duration, sr);
  if (n === 0) return new Float64Array(0);

  const seedRng = new SeededRng(options?.seed ?? 42);
  const sessionSeed = seedRng.intRange(1, 0x7fffffff);
  const harmony = buildHarmonyPlan(cfg, duration, new SeededRng(deriveSeed(sessionSeed, 101)), tOffset);
  const activeLayers = DENSITY_LAYERS[cfg.density];
  const blockSalt = Math.round(tOffset * 10.0) * 31;

  const out = new Float64Array(n);
  const mixIn = (layer: Float64Array, gain = 1.0): void => {
    const len = Math.min(n, layer.length);
    for (let i = 0; i < len; i++) out[i]! += layer[i]! * gain;
  };

  if (cfg.depth) {
    mixIn(renderBassSubPulse(cfg, duration, harmony, tOffset), 0.6);
  }
  if (activeLayers.includes("bass")) {
    mixIn(renderBass(cfg, duration, harmony, new SeededRng(deriveSeed(sessionSeed, 11 + blockSalt)), tOffset));
  }
  if (activeLayers.includes("pad")) {
    mixIn(renderPad(cfg, duration, harmony, new SeededRng(deriveSeed(sessionSeed, 23 + blockSalt)), tOffset));
  }
  if (activeLayers.includes("melody")) {
    const melSeed =
      cfg.melody_engine === "procedural"
        ? deriveSeed(sessionSeed, 101)
        : deriveSeed(sessionSeed, 41 + blockSalt);
    mixIn(renderMelody(cfg, duration, harmony, new SeededRng(melSeed), tOffset));
  }
  if (activeLayers.includes("rhythm") && cfg.rhythm !== "none") {
    mixIn(renderRhythm(cfg, duration, new SeededRng(deriveSeed(sessionSeed, 53 + blockSalt)), tOffset));
  }
  if (activeLayers.includes("texture") && cfg.texture !== "none") {
    mixIn(renderTexture(cfg, duration, new SeededRng(deriveSeed(sessionSeed, 67 + blockSalt)), tOffset));
  }
  if (activeLayers.includes("accent") && cfg.accent !== "none") {
    mixIn(renderAccent(cfg, duration, harmony, new SeededRng(deriveSeed(sessionSeed, 79 + blockSalt)), tOffset));
  }

  let master = applyStereoWidth(out, cfg.stereo, sr);
  master = applyMasterHallReverb(master, cfg.space, sr);
  master = applyHighpass(master, 20.0, sr);
  master = softLimit(master, 0.92);

  if (normalize && master.length > 0) {
    let peak = 0;
    for (let i = 0; i < master.length; i++) {
      const a = Math.abs(master[i]!);
      if (a > peak) peak = a;
    }
    if (peak > 1e-6) {
      const gain = 0.78 / peak;
      for (let i = 0; i < master.length; i++) master[i]! *= gain;
    }
  }

  const fadeSec = tOffset > 1e-6 ? 0.008 : 0.055;
  return softFadeEdges(master, fadeSec, fadeSec, sr);
}

/**
 * Render a prompt or config directly into mono Float32 PCM in [-1, 1].
 */
export function renderPcm(input: ConfigInput, options?: AssembleOptions): Float32Array {
  const f64 = assemble(input, options);
  const f32 = new Float32Array(f64.length);
  for (let i = 0; i < f64.length; i++) {
    f32[i] = Math.max(-1, Math.min(1, f64[i]!));
  }
  return f32;
}
