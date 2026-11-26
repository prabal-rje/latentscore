/**
 * Zero-dependency mono and mono-compatible M-S stereo WAV encoder.
 */

import { SAMPLE_RATE } from "./config.js";

export interface EncodeWavOptions {
  /** Sample rate in Hz (default 44100). */
  sampleRate?: number;
  /** Output channels: 1 for mono, 2 for mono-compatible M-S stereo (default 1). */
  channels?: 1 | 2;
  /** Stereo width in [0, 1] when channels === 2 (default 0.5). */
  stereoWidth?: number;
  /** Raised-cosine fade-in duration in seconds (default 0.015). */
  fadeInSeconds?: number;
  /** Cubic fade-into-nothingness duration in seconds (default 0.25). */
  fadeOutSeconds?: number;
}

/**
 * Apply a smooth cubic fade-out (1 - u)^3 ending at exact zero.
 */
export function fadeIntoNothingness(
  samples: Float32Array | Float64Array,
  fadeSeconds = 0.35,
  sampleRate: number = SAMPLE_RATE,
): Float32Array {
  const out = new Float32Array(samples.length);
  out.set(samples);
  applyFadeOutInPlace(out, fadeSeconds, sampleRate);
  return out;
}

function applyFadeInInPlace(samples: Float32Array, fadeSeconds: number, sampleRate: number): void {
  const n = samples.length;
  if (n < 2 || fadeSeconds <= 0) return;
  const fadeN = Math.min(Math.floor(n / 2), Math.max(2, Math.round(fadeSeconds * sampleRate)));
  const denom = Math.max(1, fadeN - 1);
  for (let i = 0; i < fadeN; i++) {
    samples[i]! *= 0.5 * (1 - Math.cos((Math.PI * i) / denom));
  }
  samples[0] = 0;
}

function applyFadeOutInPlace(samples: Float32Array, fadeSeconds: number, sampleRate: number): void {
  const n = samples.length;
  if (n < 2 || fadeSeconds <= 0) return;
  const fadeN = Math.min(n, Math.max(2, Math.round(fadeSeconds * sampleRate)));
  const start = n - fadeN;
  const denom = Math.max(1, fadeN - 1);
  for (let i = 0; i < fadeN; i++) {
    const u = i / denom;
    samples[start + i]! *= (1 - u) * (1 - u) * (1 - u);
  }
  samples[n - 1] = 0;
}

/**
 * Encode mono PCM samples into a standards-compliant 16-bit PCM WAV Uint8Array.
 * When channels === 2, builds a mono-compatible Mid-Side stereo field
 * (L = M + S, R = M - S) and applies the final fade AFTER stereo decorrelation
 * so both channels end at exact zero silence.
 */
export function encodeWavBytes(
  samples: Float32Array | Float64Array,
  options?: EncodeWavOptions,
): Uint8Array {
  const sr = options?.sampleRate ?? SAMPLE_RATE;
  const channels = options?.channels ?? 1;
  const stereoWidth = Math.max(0, Math.min(1, options?.stereoWidth ?? 0.5));
  const fadeIn = options?.fadeInSeconds ?? 0.015;
  const fadeOut = options?.fadeOutSeconds ?? 0.25;

  const numFrames = samples.length;
  const bytesPerSample = 2;
  const dataBytes = numFrames * channels * bytesPerSample;
  if (dataBytes > 0xffff_ff00) {
    throw new RangeError("WAV payload exceeds 4 GiB RIFF limit");
  }
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  const writeStr = (offset: number, text: string): void => {
    for (let i = 0; i < text.length; i++) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataBytes, true);

  let pos = 44;
  if (channels === 1) {
    const mono = new Float32Array(numFrames);
    mono.set(samples);
    applyFadeInInPlace(mono, fadeIn, sr);
    applyFadeOutInPlace(mono, fadeOut, sr);
    for (let i = 0; i < numFrames; i++) {
      const s = Math.max(-1, Math.min(1, mono[i]!));
      view.setInt16(pos, Math.round(s * 32767), true);
      pos += 2;
    }
  } else {
    const left = new Float32Array(numFrames);
    const right = new Float32Array(numFrames);
    const delaySamples = Math.max(1, Math.round(0.011 * sr));
    const sideGain = 0.28 * stereoWidth;
    const midGain = 1 / (1 + sideGain);
    const alpha = Math.exp((-2 * Math.PI * 300) / sr);
    let prevIn = 0;
    let prevOut = 0;

    for (let i = 0; i < numFrames; i++) {
      const m = samples[i]!;
      const del = i >= delaySamples ? samples[i - delaySamples]! : 0;
      const hp = alpha * (prevOut + del - prevIn);
      prevIn = del;
      prevOut = hp;
      const side = hp * sideGain;
      left[i] = m * midGain + side;
      right[i] = m * midGain - side;
    }

    // Apply entry and exit fades AFTER stereo decorrelation so final L & R are exact zero.
    applyFadeInInPlace(left, fadeIn, sr);
    applyFadeInInPlace(right, fadeIn, sr);
    applyFadeOutInPlace(left, fadeOut, sr);
    applyFadeOutInPlace(right, fadeOut, sr);

    for (let i = 0; i < numFrames; i++) {
      const l = Math.max(-1, Math.min(1, left[i]!));
      const r = Math.max(-1, Math.min(1, right[i]!));
      view.setInt16(pos, Math.round(l * 32767), true);
      view.setInt16(pos + 2, Math.round(r * 32767), true);
      pos += 4;
    }
  }

  return new Uint8Array(buffer);
}
