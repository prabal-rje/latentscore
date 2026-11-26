/**
 * Pure TypeScript music configuration, palette, title, justification,
 * heuristic fallback, and WAV encoding logic matching backend/main.py
 * and latentscore/config.py for zero-backend browser execution.
 */

import type { ParamOptions, Preset } from "../types";

export const SAMPLE_RATE = 44100;

export const PARAM_OPTIONS: ParamOptions = {
  tempo: ["very_slow", "slow", "medium", "fast", "very_fast"],
  root: ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"],
  mode: ["major", "minor", "dorian", "mixolydian"],
  brightness: ["very_dark", "dark", "medium", "bright", "very_bright"],
  space: ["dry", "small", "medium", "large", "vast"],
  density: ["2", "3", "4", "5", "6"],
  bass: ["drone", "sustained", "pulsing", "walking", "fifth_drone", "sub_pulse", "octave", "arp_bass"],
  pad: ["warm_slow", "dark_sustained", "cinematic", "thin_high", "ambient_drift", "stacked_fifths", "bright_open"],
  melody: [
    "procedural",
    "contemplative",
    "rising",
    "falling",
    "minimal",
    "ornamental",
    "arp_melody",
    "contemplative_minor",
    "call_response",
    "heroic",
  ],
  rhythm: [
    "none",
    "minimal",
    "heartbeat",
    "soft_four",
    "hats_only",
    "electronic",
    "kit_light",
    "kit_medium",
    "military",
    "tabla_essence",
    "brush",
  ],
  texture: [
    "none",
    "shimmer",
    "shimmer_slow",
    "vinyl_crackle",
    "breath",
    "stars",
    "glitch",
    "noise_wash",
    "crystal",
    "pad_whisper",
  ],
  accent: [
    "none",
    "bells",
    "pluck",
    "chime",
    "bells_dense",
    "blip",
    "blip_random",
    "brass_hit",
    "wind",
    "arp_accent",
    "piano_note",
  ],
  motion: ["static", "slow", "medium", "fast", "chaotic"],
  attack: ["soft", "medium", "sharp"],
  stereo: ["mono", "narrow", "medium", "wide", "ultra_wide"],
  echo: ["none", "subtle", "medium", "heavy", "infinite"],
  human: ["robotic", "tight", "natural", "loose", "drunk"],
  grain: ["clean", "warm", "gritty"],
};

export interface MusicConfig {
  tempo: string;
  root: string;
  mode: string;
  brightness: string;
  space: string;
  density: number;
  bass: string;
  pad: string;
  melody: string;
  rhythm: string;
  texture: string;
  accent: string;
  motion: string;
  attack: string;
  stereo: string;
  depth: boolean;
  echo: string;
  human: string;
  grain: string;
  melody_engine: string;
  phrase_len_bars: number;
  melody_density: number;
  syncopation: number;
  swing: number;
  motif_repeat_prob: number;
  step_bias: number;
  chromatic_prob: number;
  cadence_strength: number;
  register_min_oct: number;
  register_max_oct: number;
  tension_curve: string;
  harmony_style: string;
  chord_change_bars: number;
  chord_extensions: string;
}

export interface InternalSynthConfig extends Omit<
  MusicConfig,
  "tempo" | "brightness" | "space" | "motion" | "stereo" | "echo" | "human"
> {
  tempo: number;
  brightness: number;
  space: number;
  motion: number;
  stereo: number;
  echo: number;
  human: number;
}

export interface PrecomputedPreset extends Preset {
  title: string;
  palette: string[];
  config: Record<string, unknown>;
}

export const DEFAULT_MUSIC_CONFIG: MusicConfig = {
  tempo: "medium",
  root: "c",
  mode: "minor",
  brightness: "medium",
  space: "medium",
  density: 5,
  bass: "drone",
  pad: "warm_slow",
  melody: "contemplative",
  rhythm: "minimal",
  texture: "shimmer",
  accent: "bells",
  motion: "medium",
  attack: "medium",
  stereo: "medium",
  depth: false,
  echo: "medium",
  human: "robotic",
  grain: "clean",
  melody_engine: "pattern",
  phrase_len_bars: 4,
  melody_density: 0.45,
  syncopation: 0.2,
  swing: 0.0,
  motif_repeat_prob: 0.5,
  step_bias: 0.75,
  chromatic_prob: 0.05,
  cadence_strength: 0.65,
  register_min_oct: 4,
  register_max_oct: 6,
  tension_curve: "arc",
  harmony_style: "auto",
  chord_change_bars: 1,
  chord_extensions: "triads",
};

const TEMPO_MAP: Record<string, number> = {
  very_slow: 0.15,
  slow: 0.3,
  medium: 0.5,
  fast: 0.7,
  very_fast: 0.9,
};
const BRIGHTNESS_MAP: Record<string, number> = {
  very_dark: 0.1,
  dark: 0.3,
  medium: 0.5,
  bright: 0.7,
  very_bright: 0.9,
};
const SPACE_MAP: Record<string, number> = {
  dry: 0.1,
  small: 0.3,
  medium: 0.5,
  large: 0.7,
  vast: 0.95,
};
const MOTION_MAP: Record<string, number> = {
  static: 0.1,
  slow: 0.3,
  medium: 0.5,
  fast: 0.7,
  chaotic: 0.9,
};
const STEREO_MAP: Record<string, number> = {
  mono: 0.0,
  narrow: 0.25,
  medium: 0.5,
  wide: 0.75,
  ultra_wide: 1.0,
};
const ECHO_MAP: Record<string, number> = {
  none: 0.0,
  subtle: 0.25,
  medium: 0.5,
  heavy: 0.75,
  infinite: 0.95,
};
const HUMAN_MAP: Record<string, number> = {
  robotic: 0.0,
  tight: 0.15,
  natural: 0.3,
  loose: 0.5,
  drunk: 0.8,
};

const LABEL_TO_NUM: Record<string, Record<string, number>> = {
  melody_density: { very_sparse: 0.15, sparse: 0.3, medium: 0.5, busy: 0.7, very_busy: 0.85 },
  syncopation: { straight: 0.0, light: 0.2, medium: 0.5, heavy: 0.8 },
  swing: { none: 0.0, light: 0.2, medium: 0.5, heavy: 0.8 },
  motif_repeat_prob: { rare: 0.2, sometimes: 0.5, often: 0.8 },
  step_bias: { step: 0.9, balanced: 0.7, leapy: 0.4 },
  chromatic_prob: { none: 0.0, light: 0.05, medium: 0.12, heavy: 0.25 },
  cadence_strength: { weak: 0.3, medium: 0.6, strong: 0.9 },
  chord_change_bars: { very_slow: 4, slow: 2, medium: 1, fast: 1 },
};

const EXTRA_ENUMS: Record<string, readonly string[]> = {
  melody_engine: ["pattern", "procedural"],
  tension_curve: ["arc", "ramp", "waves"],
  harmony_style: ["auto", "pop", "jazz", "cinematic", "ambient"],
  chord_extensions: ["triads", "sevenths", "lush"],
};

function clampNumber(val: unknown, min: number, max: number, fallback: number): number {
  const num = typeof val === "number" ? val : typeof val === "string" ? Number(val) : NaN;
  if (!Number.isFinite(num)) return fallback;
  return Math.max(min, Math.min(max, num));
}

function validEnum(key: string, val: unknown, fallback: string): string {
  const allowed = PARAM_OPTIONS[key] ?? EXTRA_ENUMS[key];
  if (typeof val === "string" && allowed && allowed.includes(val)) {
    return val;
  }
  return fallback;
}

function coerceLabelFloat(key: string, val: unknown, fallback: number, max = 1.0): number {
  const table = LABEL_TO_NUM[key];
  if (typeof val === "string" && table && val in table) {
    return table[val];
  }
  return clampNumber(val, 0.0, max, fallback);
}

/** Validate and normalize raw/LLM config fields into a complete MusicConfig. */
export function coerceConfig(raw: Record<string, unknown>): MusicConfig {
  const d = DEFAULT_MUSIC_CONFIG;
  const density = Math.round(clampNumber(raw.density, 2, 6, d.density));
  const rawPhrase = Math.round(clampNumber(raw.phrase_len_bars, 2, 8, d.phrase_len_bars));
  const phraseLen = ([2, 4, 8] as const).reduce((best, cand) =>
    Math.abs(cand - rawPhrase) < Math.abs(best - rawPhrase) ? cand : best
  );
  const rawBars = coerceLabelFloat("chord_change_bars", raw.chord_change_bars, d.chord_change_bars, 8);
  const chordChangeBars = Math.max(1, Math.min(4, Math.round(rawBars)));
  const minOct = Math.round(clampNumber(raw.register_min_oct, 1, 8, d.register_min_oct));
  const maxOct = Math.max(
    minOct,
    Math.round(clampNumber(raw.register_max_oct, 1, 8, d.register_max_oct))
  );

  return {
    tempo: validEnum("tempo", raw.tempo, d.tempo),
    root: validEnum("root", raw.root, d.root),
    mode: validEnum("mode", raw.mode, d.mode),
    brightness: validEnum("brightness", raw.brightness, d.brightness),
    space: validEnum("space", raw.space, d.space),
    density,
    bass: validEnum("bass", raw.bass, d.bass),
    pad: validEnum("pad", raw.pad, d.pad),
    melody: validEnum("melody", raw.melody, d.melody),
    rhythm: validEnum("rhythm", raw.rhythm, d.rhythm),
    texture: validEnum("texture", raw.texture, d.texture),
    accent: validEnum("accent", raw.accent, d.accent),
    motion: validEnum("motion", raw.motion, d.motion),
    attack: validEnum("attack", raw.attack, d.attack),
    stereo: validEnum("stereo", raw.stereo, d.stereo),
    depth: typeof raw.depth === "boolean" ? raw.depth : d.depth,
    echo: validEnum("echo", raw.echo, d.echo),
    human: validEnum("human", raw.human, d.human),
    grain: validEnum("grain", raw.grain, d.grain),
    melody_engine: validEnum("melody_engine", raw.melody_engine, d.melody_engine),
    phrase_len_bars: phraseLen,
    melody_density: coerceLabelFloat("melody_density", raw.melody_density, d.melody_density),
    syncopation: coerceLabelFloat("syncopation", raw.syncopation, d.syncopation),
    swing: coerceLabelFloat("swing", raw.swing, d.swing),
    motif_repeat_prob: coerceLabelFloat("motif_repeat_prob", raw.motif_repeat_prob, d.motif_repeat_prob),
    step_bias: coerceLabelFloat("step_bias", raw.step_bias, d.step_bias),
    chromatic_prob: coerceLabelFloat("chromatic_prob", raw.chromatic_prob, d.chromatic_prob),
    cadence_strength: coerceLabelFloat("cadence_strength", raw.cadence_strength, d.cadence_strength),
    register_min_oct: minOct,
    register_max_oct: maxOct,
    tension_curve: validEnum("tension_curve", raw.tension_curve, d.tension_curve),
    harmony_style: validEnum("harmony_style", raw.harmony_style, d.harmony_style),
    chord_change_bars: chordChangeBars,
    chord_extensions: validEnum("chord_extensions", raw.chord_extensions, d.chord_extensions),
  };
}

/** Convert public MusicConfig labels to numeric SynthConfig expected by latentscore/synth.py. */
export function toInternalSynthConfig(raw: Record<string, unknown>): InternalSynthConfig {
  const cfg = coerceConfig(raw);
  return {
    ...cfg,
    tempo: TEMPO_MAP[cfg.tempo] ?? 0.5,
    brightness: BRIGHTNESS_MAP[cfg.brightness] ?? 0.5,
    space: SPACE_MAP[cfg.space] ?? 0.5,
    motion: MOTION_MAP[cfg.motion] ?? 0.5,
    stereo: STEREO_MAP[cfg.stereo] ?? 0.5,
    echo: ECHO_MAP[cfg.echo] ?? 0.5,
    human: HUMAN_MAP[cfg.human] ?? 0.0,
  };
}

// ─── Palette, title, justification, heuristic fallback ──────────────────────

const MODE_HUES: Record<string, number> = {
  major: 45,
  minor: 230,
  dorian: 160,
  mixolydian: 25,
};
const BRIGHTNESS_LIGHTNESS: Record<string, number> = {
  very_dark: 0.18,
  dark: 0.28,
  medium: 0.42,
  bright: 0.56,
  very_bright: 0.68,
};
const SPACE_SATURATION: Record<string, number> = {
  dry: 0.65,
  small: 0.55,
  medium: 0.45,
  large: 0.35,
  vast: 0.25,
};
const TEMPO_HUE_SHIFT: Record<string, number> = {
  very_slow: -12,
  slow: -6,
  medium: 0,
  fast: 8,
  very_fast: 16,
};
const GRAIN_SAT_MULT: Record<string, number> = {
  clean: 1.0,
  warm: 0.85,
  gritty: 0.65,
};

function vibeHash(vibe: string): number {
  let h = 0;
  for (let i = 0; i < vibe.length; i++) {
    h = (Math.imul(h, 31) + vibe.charCodeAt(i)) >>> 0;
  }
  return (h % 3600) / 10.0;
}

function hslToHex(hDeg: number, sIn: number, lIn: number): string {
  const h = (((hDeg % 360) + 360) % 360) / 360;
  const s = Math.max(0, Math.min(1, sIn));
  const l = Math.max(0, Math.min(1, lIn));
  let r: number;
  let g: number;
  let b: number;

  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p: number, q: number, tIn: number): number => {
      let t = tIn;
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }

  const hex = (v: number) => Math.floor(v * 255).toString(16).padStart(2, "0");
  return `#${hex(r)}${hex(g)}${hex(b)}`;
}

export function generatePalette(cfg: MusicConfig, vibe = ""): string[] {
  const baseH = MODE_HUES[cfg.mode] ?? 200;
  const vibeOffset = (vibeHash(vibe) % 50) - 25;
  const tempoShift = TEMPO_HUE_SHIFT[cfg.tempo] ?? 0;
  const h = (((baseH + vibeOffset + tempoShift) % 360) + 360) % 360;

  const lBase = BRIGHTNESS_LIGHTNESS[cfg.brightness] ?? 0.42;
  const sBase = (SPACE_SATURATION[cfg.space] ?? 0.45) * (GRAIN_SAT_MULT[cfg.grain] ?? 1.0);

  return [
    hslToHex(h, sBase + 0.1, lBase),
    hslToHex(h + 30, sBase, lBase * 0.85 + 0.1),
    hslToHex(h + 180, sBase * 0.55, lBase * 0.7 + 0.12),
    hslToHex(h, sBase * 0.3, 0.06),
    hslToHex(h + 160, Math.min(0.85, sBase + 0.3), Math.min(0.7, lBase + 0.18)),
  ];
}

const STOP_WORDS = new Set([
  "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
  "and", "or", "but", "is", "was", "are", "it", "its", "my", "your",
  "from", "by", "as", "like", "very", "really", "just", "some", "that",
  "this", "over", "into", "through", "about",
]);

const MODE_ADJECTIVES: Record<string, string[]> = {
  major: ["Golden", "Radiant", "Luminous", "Sunlit", "Bright"],
  minor: ["Moonlit", "Veiled", "Twilight", "Dusted", "Shadowed"],
  dorian: ["Ancient", "Sage", "Woven", "Mystic", "Verdant"],
  mixolydian: ["Amber", "Wandering", "Drifting", "Hazy", "Burnished"],
};

export function generateTitle(vibe: string, cfg: MusicConfig): string {
  const words = vibe
    .split(/\s+/)
    .map((w) => w.replace(/^[,.'";:!?()[\]]+|[,.'";:!?()[\]]+$/g, ""));
  const meaningful = words.filter((w) => w.length > 2 && !STOP_WORDS.has(w.toLowerCase()));
  if (meaningful.length >= 2) {
    return meaningful
      .slice(0, 4)
      .map((w) => w[0].toUpperCase() + w.slice(1).toLowerCase())
      .join(" ")
      .slice(0, 60);
  }
  const adjs = MODE_ADJECTIVES[cfg.mode] ?? ["Untitled"];
  const idx = Math.floor(vibeHash(vibe)) % adjs.length;
  const adj = adjs[idx];
  if (meaningful.length === 1) {
    const first = meaningful[0];
    return `${adj} ${first[0].toUpperCase()}${first.slice(1).toLowerCase()}`.slice(0, 60);
  }
  return `${adj} Soundscape`.slice(0, 60);
}

const MODE_FEELING: Record<string, string> = {
  major: "bright and uplifting",
  minor: "deep and emotional",
  dorian: "mystical and modal",
  mixolydian: "warm and adventurous",
};
const TEMPO_FEELING: Record<string, string> = {
  very_slow: "glacial, meditative pace",
  slow: "contemplative, unhurried flow",
  medium: "balanced, steady movement",
  fast: "driven, energetic pulse",
  very_fast: "intense, urgent momentum",
};
const SPACE_FEELING: Record<string, string> = {
  dry: "intimate, close-up presence",
  small: "cozy, contained warmth",
  medium: "natural room ambience",
  large: "expansive, open atmosphere",
  vast: "cathedral-like, infinite space",
};

export function generateJustification(vibe: string, cfg: MusicConfig): string {
  const feeling = MODE_FEELING[cfg.mode] ?? "balanced";
  const pace = TEMPO_FEELING[cfg.tempo] ?? "moderate movement";
  const room = SPACE_FEELING[cfg.space] ?? "natural space";
  const textureDesc =
    cfg.texture !== "none" ? ` through ${cfg.texture.replace(/_/g, " ")} textures` : "";
  const depthDesc = cfg.depth ? " Sub-bass reinforcement adds physical weight." : "";

  return (
    `"${vibe}" suggests a ${feeling} soundscape. ` +
    `${cfg.root.toUpperCase()} ${cfg.mode} provides the harmonic foundation with a ${pace}. ` +
    `A ${cfg.bass.replace(/_/g, " ")} bass line and ${cfg.pad.replace(/_/g, " ")} pad ` +
    `build the harmonic bed, while ${cfg.melody.replace(/_/g, " ")} melody adds movement${textureDesc}. ` +
    `The spatial character is ${room} with ${cfg.echo} delay and ` +
    `${cfg.stereo.replace(/_/g, " ")} stereo spread.${depthDesc}`
  );
}

export function heuristicConfig(vibe: string): MusicConfig {
  const lower = vibe.toLowerCase();
  const hasAny = (words: string[]) => words.some((w) => lower.includes(w));
  const cfg: MusicConfig = { ...DEFAULT_MUSIC_CONFIG };

  if (hasAny(["dark", "sad", "night", "mysterious"])) {
    cfg.root = "d";
    cfg.mode = "dorian";
    cfg.brightness = "dark";
  } else if (hasAny(["happy", "bright", "joy"])) {
    cfg.root = "c";
    cfg.mode = "major";
    cfg.brightness = "bright";
  } else if (hasAny(["epic", "cinematic", "powerful"])) {
    cfg.root = "d";
    cfg.mode = "minor";
    cfg.brightness = "medium";
    cfg.depth = true;
  } else if (hasAny(["indian", "spiritual", "meditation", "zen"])) {
    cfg.root = "d";
    cfg.mode = "dorian";
    cfg.human = "natural";
    cfg.grain = "warm";
  }

  if (hasAny(["slow", "calm", "meditation", "zen"])) {
    cfg.tempo = "slow";
  } else if (hasAny(["fast", "energy", "drive", "cyberpunk"])) {
    cfg.tempo = "fast";
  } else {
    cfg.tempo = "medium";
  }

  return cfg;
}

export function configToRecord(cfg: MusicConfig): Record<string, unknown> {
  return { ...cfg };
}

/**
 * Compute an exact integer-bar duration (in seconds) near ``targetSec`` so
 * loop boundaries and Live continuations always land on Beat 1 of a bar.
 */
export function computeBarAlignedDuration(
  rawConfig: Record<string, unknown>,
  targetSec = 28,
  minSec = 10,
  maxSec = 36
): number {
  const internal = toInternalSynthConfig(rawConfig);
  const bpm = 55 + 110 * Math.max(0, Math.min(1, internal.tempo));
  const barSec = (4 * 60) / bpm;
  const minBars = Math.max(1, Math.ceil(minSec / barSec));
  const maxBars = Math.max(minBars, Math.floor(maxSec / barSec));
  const targetBars = Math.max(minBars, Math.min(maxBars, Math.round(targetSec / barSec)));

  const chordBars = Math.max(1, internal.chord_change_bars);
  const fullProgBars = chordBars * 4;

  const pickClosestMultiple = (step: number): number | null => {
    let best: number | null = null;
    let bestDist = Infinity;
    for (let b = step; b <= maxBars; b += step) {
      if (b < minBars) continue;
      const d = Math.abs(b - targetBars);
      if (d < bestDist) {
        best = b;
        bestDist = d;
      }
    }
    return best;
  };

  const bars =
    pickClosestMultiple(fullProgBars) ??
    pickClosestMultiple(chordBars) ??
    targetBars;
  return bars * barSec;
}
