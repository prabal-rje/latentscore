/**
 * 34-field LatentScore configuration types, label-to-numeric mappings,
 * relative Step updates, interpolation, and bar-aligned phrase duration math.
 * Matches spec/config.schema.json (schema_version: 1).
 */

export const SCHEMA_VERSION = 1 as const;
export const SAMPLE_RATE = 44100 as const;

export type TempoLabel = "very_slow" | "slow" | "medium" | "fast" | "very_fast";
export type RootNote = "c" | "c#" | "d" | "d#" | "e" | "f" | "f#" | "g" | "g#" | "a" | "a#" | "b";
export type ModeName = "major" | "minor" | "dorian" | "mixolydian";
export type BrightnessLabel = "very_dark" | "dark" | "medium" | "bright" | "very_bright";
export type SpaceLabel = "dry" | "small" | "medium" | "large" | "vast";
export type DensityLevel = 2 | 3 | 4 | 5 | 6;
export type BassStyle =
  | "drone"
  | "sustained"
  | "pulsing"
  | "walking"
  | "fifth_drone"
  | "sub_pulse"
  | "octave"
  | "arp_bass";
export type PadStyle =
  | "warm_slow"
  | "dark_sustained"
  | "cinematic"
  | "thin_high"
  | "ambient_drift"
  | "stacked_fifths"
  | "bright_open";
export type MelodyStyle =
  | "contemplative"
  | "contemplative_minor"
  | "rising"
  | "falling"
  | "minimal"
  | "ornamental"
  | "arp_melody"
  | "call_response"
  | "heroic"
  | "procedural";
export type RhythmStyle =
  | "none"
  | "minimal"
  | "heartbeat"
  | "soft_four"
  | "hats_only"
  | "electronic"
  | "kit_light"
  | "kit_medium"
  | "military"
  | "tabla_essence"
  | "brush";
export type TextureStyle =
  | "none"
  | "shimmer"
  | "shimmer_slow"
  | "vinyl_crackle"
  | "breath"
  | "stars"
  | "glitch"
  | "noise_wash"
  | "crystal"
  | "pad_whisper";
export type AccentStyle =
  | "none"
  | "bells"
  | "bells_dense"
  | "pluck"
  | "chime"
  | "blip"
  | "blip_random"
  | "brass_hit"
  | "wind"
  | "arp_accent"
  | "piano_note";
export type MotionLabel = "static" | "slow" | "medium" | "fast" | "chaotic";
export type AttackStyle = "soft" | "medium" | "sharp";
export type StereoLabel = "mono" | "narrow" | "medium" | "wide" | "ultra_wide";
export type EchoLabel = "none" | "subtle" | "medium" | "heavy" | "infinite";
export type HumanFeelLabel = "robotic" | "tight" | "natural" | "loose" | "drunk";
export type GrainStyle = "clean" | "warm" | "gritty";
export type MelodyEngine = "pattern" | "procedural";
export type PhraseLengthBars = 2 | 4 | 8;
export type TensionCurve = "arc" | "ramp" | "waves";
export type HarmonyStyle = "auto" | "pop" | "jazz" | "cinematic" | "ambient";
export type ChordExtensions = "triads" | "sevenths" | "lush";

export interface Step {
  readonly kind: "step";
  readonly delta: number;
}

export function step(delta: number): Step {
  if (!Number.isInteger(delta)) {
    throw new RangeError(`step(delta) expects an integer, got ${delta}`);
  }
  return { kind: "step", delta };
}

export function isStep(value: unknown): value is Step {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: unknown }).kind === "step" &&
    typeof (value as { delta?: unknown }).delta === "number"
  );
}

export interface MusicConfig {
  readonly schema_version: 1;
  readonly tempo: TempoLabel;
  readonly root: RootNote;
  readonly mode: ModeName;
  readonly brightness: BrightnessLabel;
  readonly space: SpaceLabel;
  readonly density: DensityLevel;
  readonly bass: BassStyle;
  readonly pad: PadStyle;
  readonly melody: MelodyStyle;
  readonly rhythm: RhythmStyle;
  readonly texture: TextureStyle;
  readonly accent: AccentStyle;
  readonly motion: MotionLabel;
  readonly attack: AttackStyle;
  readonly stereo: StereoLabel;
  readonly depth: boolean;
  readonly echo: EchoLabel;
  readonly human: HumanFeelLabel;
  readonly grain: GrainStyle;
  readonly melody_engine: MelodyEngine;
  readonly phrase_len_bars: PhraseLengthBars;
  readonly melody_density: number;
  readonly syncopation: number;
  readonly swing: number;
  readonly motif_repeat_prob: number;
  readonly step_bias: number;
  readonly chromatic_prob: number;
  readonly cadence_strength: number;
  readonly register_min_oct: number;
  readonly register_max_oct: number;
  readonly tension_curve: TensionCurve;
  readonly harmony_style: HarmonyStyle;
  readonly chord_change_bars: number;
  readonly chord_extensions: ChordExtensions;
}

export interface SynthConfig {
  readonly tempo: number;
  readonly root: RootNote;
  readonly mode: ModeName;
  readonly brightness: number;
  readonly space: number;
  readonly density: DensityLevel;
  readonly bass: BassStyle;
  readonly pad: PadStyle;
  readonly melody: MelodyStyle;
  readonly rhythm: RhythmStyle;
  readonly texture: TextureStyle;
  readonly accent: AccentStyle;
  readonly motion: number;
  readonly attack: AttackStyle;
  readonly stereo: number;
  readonly depth: boolean;
  readonly echo: number;
  readonly human: number;
  readonly grain: GrainStyle;
  readonly melody_engine: MelodyEngine;
  readonly phrase_len_bars: number;
  readonly melody_density: number;
  readonly syncopation: number;
  readonly swing: number;
  readonly motif_repeat_prob: number;
  readonly step_bias: number;
  readonly chromatic_prob: number;
  readonly cadence_strength: number;
  readonly register_min_oct: number;
  readonly register_max_oct: number;
  readonly tension_curve: TensionCurve;
  readonly harmony_style: HarmonyStyle;
  readonly chord_change_bars: number;
  readonly chord_extensions: ChordExtensions;
}

export type MusicConfigUpdate = {
  tempo?: TempoLabel | Step;
  root?: RootNote;
  mode?: ModeName;
  brightness?: BrightnessLabel | Step;
  space?: SpaceLabel | Step;
  density?: DensityLevel | Step;
  bass?: BassStyle;
  pad?: PadStyle;
  melody?: MelodyStyle;
  rhythm?: RhythmStyle;
  texture?: TextureStyle;
  accent?: AccentStyle;
  motion?: MotionLabel | Step;
  attack?: AttackStyle | Step;
  stereo?: StereoLabel | Step;
  depth?: boolean;
  echo?: EchoLabel | Step;
  human?: HumanFeelLabel | Step;
  grain?: GrainStyle | Step;
  melody_engine?: MelodyEngine;
  phrase_len_bars?: PhraseLengthBars;
  melody_density?: number;
  syncopation?: number;
  swing?: number;
  motif_repeat_prob?: number;
  step_bias?: number;
  chromatic_prob?: number;
  cadence_strength?: number;
  register_min_oct?: number;
  register_max_oct?: number;
  tension_curve?: TensionCurve;
  harmony_style?: HarmonyStyle;
  chord_change_bars?: number;
  chord_extensions?: ChordExtensions;
};

export type ConfigInput = Partial<MusicConfig> | SynthConfig;

export const DEFAULT_CONFIG: MusicConfig = Object.freeze({
  schema_version: 1,
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
});

export const TEMPO_MAP: Readonly<Record<TempoLabel, number>> = Object.freeze({
  very_slow: 0.15,
  slow: 0.3,
  medium: 0.5,
  fast: 0.7,
  very_fast: 0.9,
});

export const BRIGHTNESS_MAP: Readonly<Record<BrightnessLabel, number>> = Object.freeze({
  very_dark: 0.1,
  dark: 0.3,
  medium: 0.5,
  bright: 0.7,
  very_bright: 0.9,
});

export const SPACE_MAP: Readonly<Record<SpaceLabel, number>> = Object.freeze({
  dry: 0.1,
  small: 0.3,
  medium: 0.5,
  large: 0.7,
  vast: 0.95,
});

export const MOTION_MAP: Readonly<Record<MotionLabel, number>> = Object.freeze({
  static: 0.1,
  slow: 0.3,
  medium: 0.5,
  fast: 0.7,
  chaotic: 0.9,
});

export const STEREO_MAP: Readonly<Record<StereoLabel, number>> = Object.freeze({
  mono: 0.0,
  narrow: 0.25,
  medium: 0.5,
  wide: 0.75,
  ultra_wide: 1.0,
});

export const ECHO_MAP: Readonly<Record<EchoLabel, number>> = Object.freeze({
  none: 0.0,
  subtle: 0.25,
  medium: 0.5,
  heavy: 0.75,
  infinite: 0.95,
});

export const HUMAN_MAP: Readonly<Record<HumanFeelLabel, number>> = Object.freeze({
  robotic: 0.0,
  tight: 0.15,
  natural: 0.3,
  loose: 0.5,
  drunk: 0.8,
});

const ORDERED_SCALES = {
  tempo: ["very_slow", "slow", "medium", "fast", "very_fast"] as const,
  brightness: ["very_dark", "dark", "medium", "bright", "very_bright"] as const,
  space: ["dry", "small", "medium", "large", "vast"] as const,
  density: [2, 3, 4, 5, 6] as const,
  motion: ["static", "slow", "medium", "fast", "chaotic"] as const,
  attack: ["soft", "medium", "sharp"] as const,
  stereo: ["mono", "narrow", "medium", "wide", "ultra_wide"] as const,
  echo: ["none", "subtle", "medium", "heavy", "infinite"] as const,
  human: ["robotic", "tight", "natural", "loose", "drunk"] as const,
  grain: ["clean", "warm", "gritty"] as const,
} as const;

const VALID_UPDATE_KEYS = new Set<string>(Object.keys(DEFAULT_CONFIG));

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function clampDensity(v: number): DensityLevel {
  const n = Math.round(v);
  if (n <= 2) return 2;
  if (n === 3) return 3;
  if (n === 4) return 4;
  if (n === 5) return 5;
  return 6;
}

function closestLabel<T extends string>(val: number, map: Readonly<Record<T, number>>): T {
  const entries = Object.entries(map) as [T, number][];
  let best = entries[0]!;
  let bestDist = Math.abs(best[1] - val);
  for (let i = 1; i < entries.length; i++) {
    const entry = entries[i]!;
    const d = Math.abs(entry[1] - val);
    if (d < bestDist) {
      best = entry;
      bestDist = d;
    }
  }
  return best[0];
}

function resolveNumeric(
  value: string | number | undefined,
  map: Readonly<Record<string, number>>,
  fallback: number,
): number {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new RangeError(`Numeric config value must be finite, got ${value}`);
    }
    return clamp(value, 0, 1);
  }
  if (typeof value === "string" && value in map) {
    return map[value]!;
  }
  return fallback;
}

function assertFinite(name: string, v: number): number {
  if (!Number.isFinite(v)) {
    throw new RangeError(`Field "${name}" must be a finite number, got ${v}`);
  }
  return v;
}

export function isSynthConfig(cfg: ConfigInput): cfg is SynthConfig {
  return typeof (cfg as SynthConfig).tempo === "number";
}

/**
 * Convert a public or partial MusicConfig (or numeric SynthConfig) into a
 * normalized numeric SynthConfig ready for synthesis.
 */
export function toInternalConfig(input: ConfigInput = DEFAULT_CONFIG): SynthConfig {
  if (isSynthConfig(input)) {
    return {
      ...input,
      tempo: clamp(assertFinite("tempo", input.tempo), 0, 1),
      brightness: clamp(assertFinite("brightness", input.brightness), 0, 1),
      space: clamp(assertFinite("space", input.space), 0, 1),
      motion: clamp(assertFinite("motion", input.motion), 0, 1),
      stereo: clamp(assertFinite("stereo", input.stereo), 0, 1),
      echo: clamp(assertFinite("echo", input.echo), 0, 1),
      human: clamp(assertFinite("human", input.human), 0, 1),
      density: clampDensity(assertFinite("density", input.density)),
      melody_density: clamp(assertFinite("melody_density", input.melody_density), 0, 1),
      syncopation: clamp(assertFinite("syncopation", input.syncopation), 0, 1),
      swing: clamp(assertFinite("swing", input.swing), 0, 1),
      motif_repeat_prob: clamp(assertFinite("motif_repeat_prob", input.motif_repeat_prob), 0, 1),
      step_bias: clamp(assertFinite("step_bias", input.step_bias), 0, 1),
      chromatic_prob: clamp(assertFinite("chromatic_prob", input.chromatic_prob), 0, 1),
      cadence_strength: clamp(assertFinite("cadence_strength", input.cadence_strength), 0, 1),
    };
  }

  const m: MusicConfig = { ...DEFAULT_CONFIG, ...input, schema_version: 1 };
  return {
    tempo: resolveNumeric(m.tempo, TEMPO_MAP, 0.5),
    root: m.root,
    mode: m.mode,
    brightness: resolveNumeric(m.brightness, BRIGHTNESS_MAP, 0.5),
    space: resolveNumeric(m.space, SPACE_MAP, 0.5),
    density: clampDensity(assertFinite("density", m.density)),
    bass: m.bass,
    pad: m.pad,
    melody: m.melody,
    rhythm: m.rhythm,
    texture: m.texture,
    accent: m.accent,
    motion: resolveNumeric(m.motion, MOTION_MAP, 0.5),
    attack: m.attack,
    stereo: resolveNumeric(m.stereo, STEREO_MAP, 0.5),
    depth: Boolean(m.depth),
    echo: resolveNumeric(m.echo, ECHO_MAP, 0.5),
    human: resolveNumeric(m.human, HUMAN_MAP, 0.0),
    grain: m.grain,
    melody_engine: m.melody_engine,
    phrase_len_bars: Math.max(1, Math.min(16, Math.round(assertFinite("phrase_len_bars", m.phrase_len_bars)))),
    melody_density: clamp(assertFinite("melody_density", m.melody_density), 0, 1),
    syncopation: clamp(assertFinite("syncopation", m.syncopation), 0, 1),
    swing: clamp(assertFinite("swing", m.swing), 0, 1),
    motif_repeat_prob: clamp(assertFinite("motif_repeat_prob", m.motif_repeat_prob), 0, 1),
    step_bias: clamp(assertFinite("step_bias", m.step_bias), 0, 1),
    chromatic_prob: clamp(assertFinite("chromatic_prob", m.chromatic_prob), 0, 1),
    cadence_strength: clamp(assertFinite("cadence_strength", m.cadence_strength), 0, 1),
    register_min_oct: Math.round(assertFinite("register_min_oct", m.register_min_oct)),
    register_max_oct: Math.round(assertFinite("register_max_oct", m.register_max_oct)),
    tension_curve: m.tension_curve,
    harmony_style: m.harmony_style,
    chord_change_bars: Math.max(1, Math.min(4, Math.round(assertFinite("chord_change_bars", m.chord_change_bars)))),
    chord_extensions: m.chord_extensions,
  };
}

/**
 * Convert a numeric SynthConfig back to its nearest public MusicConfig labels.
 */
export function toPublicConfig(cfg: SynthConfig): MusicConfig {
  const phraseLen = cfg.phrase_len_bars <= 3 ? 2 : cfg.phrase_len_bars <= 6 ? 4 : 8;
  return {
    schema_version: 1,
    tempo: closestLabel(cfg.tempo, TEMPO_MAP),
    root: cfg.root,
    mode: cfg.mode,
    brightness: closestLabel(cfg.brightness, BRIGHTNESS_MAP),
    space: closestLabel(cfg.space, SPACE_MAP),
    density: clampDensity(cfg.density),
    bass: cfg.bass,
    pad: cfg.pad,
    melody: cfg.melody,
    rhythm: cfg.rhythm,
    texture: cfg.texture,
    accent: cfg.accent,
    motion: closestLabel(cfg.motion, MOTION_MAP),
    attack: cfg.attack,
    stereo: closestLabel(cfg.stereo, STEREO_MAP),
    depth: cfg.depth,
    echo: closestLabel(cfg.echo, ECHO_MAP),
    human: closestLabel(cfg.human, HUMAN_MAP),
    grain: cfg.grain,
    melody_engine: cfg.melody_engine,
    phrase_len_bars: phraseLen,
    melody_density: cfg.melody_density,
    syncopation: cfg.syncopation,
    swing: cfg.swing,
    motif_repeat_prob: cfg.motif_repeat_prob,
    step_bias: cfg.step_bias,
    chromatic_prob: cfg.chromatic_prob,
    cadence_strength: cfg.cadence_strength,
    register_min_oct: cfg.register_min_oct,
    register_max_oct: cfg.register_max_oct,
    tension_curve: cfg.tension_curve,
    harmony_style: cfg.harmony_style,
    chord_change_bars: cfg.chord_change_bars,
    chord_extensions: cfg.chord_extensions,
  };
}

/**
 * Strictly apply a MusicConfigUpdate (supporting Step deltas on ordered fields).
 * Rejects unknown keys per spec/config.schema.json.
 */
export function applyUpdate(base: MusicConfig, update: MusicConfigUpdate): MusicConfig {
  const next: Record<string, unknown> = { ...base };
  for (const [key, val] of Object.entries(update)) {
    if (val === undefined) continue;
    if (!VALID_UPDATE_KEYS.has(key) || key === "schema_version") {
      throw new Error(`Unknown MusicConfigUpdate field: "${key}"`);
    }
    if (isStep(val)) {
      if (!Number.isInteger(val.delta)) {
        throw new RangeError(`Step delta for "${key}" must be an integer, got ${val.delta}`);
      }
      if (!(key in ORDERED_SCALES)) {
        throw new Error(`Field "${key}" does not support Step() updates`);
      }
      const scale = ORDERED_SCALES[key as keyof typeof ORDERED_SCALES] as readonly unknown[];
      const curIdx = scale.indexOf(next[key]);
      const nextIdx = clamp((curIdx === -1 ? 0 : curIdx) + val.delta, 0, scale.length - 1);
      next[key] = scale[nextIdx];
    } else {
      if (typeof val === "number" && !Number.isFinite(val)) {
        throw new RangeError(`Field "${key}" must be a finite number`);
      }
      next[key] = val;
    }
  }
  return next as unknown as MusicConfig;
}

const NUMERIC_LABEL_MAPS: Readonly<Record<string, Readonly<Record<string, number>>>> = {
  tempo: TEMPO_MAP,
  brightness: BRIGHTNESS_MAP,
  space: SPACE_MAP,
  motion: MOTION_MAP,
  stereo: STEREO_MAP,
  echo: ECHO_MAP,
  human: HUMAN_MAP,
};

/**
 * Apply only touched fields from a MusicConfigUpdate onto a numeric SynthConfig
 * without snapping unrelated continuous parameters to discrete labels.
 */
export function applyUpdateToInternal(base: SynthConfig, update: MusicConfigUpdate): SynthConfig {
  const pubBase = toPublicConfig(base);
  const pubNext = applyUpdate(pubBase, update);
  const nextRec: Record<string, unknown> = { ...base };
  for (const [key, val] of Object.entries(update)) {
    if (val === undefined) continue;
    const resolvedVal = (pubNext as unknown as Record<string, unknown>)[key];
    const map = NUMERIC_LABEL_MAPS[key];
    if (map && typeof resolvedVal === "string" && resolvedVal in map) {
      nextRec[key] = map[resolvedVal]!;
    } else if (key === "density" && typeof resolvedVal === "number") {
      nextRec[key] = clampDensity(resolvedVal);
    } else {
      nextRec[key] = resolvedVal;
    }
  }
  return toInternalConfig(nextRec as unknown as SynthConfig);
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Interpolate smoothly between two SynthConfigs (t in [0, 1]) with staggered
 * layer handoffs matching Python latentscore.synth.interpolate_configs.
 */
export function interpolateConfigs(a: SynthConfig, b: SynthConfig, tRaw: number): SynthConfig {
  const t = clamp(tRaw, 0, 1);
  return {
    tempo: lerp(a.tempo, b.tempo, t),
    root: t < 0.5 ? a.root : b.root,
    mode: t < 0.5 ? a.mode : b.mode,
    brightness: lerp(a.brightness, b.brightness, t),
    space: lerp(a.space, b.space, t),
    density: clampDensity(lerp(a.density, b.density, t)),
    bass: t < 0.4 ? a.bass : b.bass,
    pad: t < 0.5 ? a.pad : b.pad,
    melody: t < 0.6 ? a.melody : b.melody,
    rhythm: t < 0.5 ? a.rhythm : b.rhythm,
    texture: t < 0.7 ? a.texture : b.texture,
    accent: t < 0.8 ? a.accent : b.accent,
    motion: lerp(a.motion, b.motion, t),
    attack: t < 0.5 ? a.attack : b.attack,
    stereo: lerp(a.stereo, b.stereo, t),
    depth: t < 0.5 ? a.depth : b.depth,
    echo: lerp(a.echo, b.echo, t),
    human: lerp(a.human, b.human, t),
    grain: t < 0.5 ? a.grain : b.grain,
    melody_engine: t < 0.5 ? a.melody_engine : b.melody_engine,
    phrase_len_bars: Math.round(lerp(a.phrase_len_bars, b.phrase_len_bars, t)),
    melody_density: lerp(a.melody_density, b.melody_density, t),
    syncopation: lerp(a.syncopation, b.syncopation, t),
    swing: lerp(a.swing, b.swing, t),
    motif_repeat_prob: lerp(a.motif_repeat_prob, b.motif_repeat_prob, t),
    step_bias: lerp(a.step_bias, b.step_bias, t),
    chromatic_prob: lerp(a.chromatic_prob, b.chromatic_prob, t),
    cadence_strength: lerp(a.cadence_strength, b.cadence_strength, t),
    register_min_oct: Math.round(lerp(a.register_min_oct, b.register_min_oct, t)),
    register_max_oct: Math.round(lerp(a.register_max_oct, b.register_max_oct, t)),
    tension_curve: t < 0.5 ? a.tension_curve : b.tension_curve,
    harmony_style: t < 0.5 ? a.harmony_style : b.harmony_style,
    chord_change_bars: Math.max(1, Math.round(lerp(a.chord_change_bars, b.chord_change_bars, t))),
    chord_extensions: t < 0.5 ? a.chord_extensions : b.chord_extensions,
  };
}

/** Alias for interpolateConfigs accepting public or internal configs. */
export function blendConfigs(a: ConfigInput, b: ConfigInput, alpha: number): SynthConfig {
  return interpolateConfigs(toInternalConfig(a), toInternalConfig(b), alpha);
}

/**
 * Compute BPM from numeric tempo in [0, 1]: 55 + 110 * tempo.
 */
export function bpmFromTempo(tempo: number): number {
  return 55.0 + 110.0 * clamp(tempo, 0, 1);
}

/**
 * Snap a target clip duration to an exact 4-bar musical phrase boundary.
 */
export function computeBarAlignedDuration(tempo: number, targetSeconds = 16.0): number {
  const bpm = bpmFromTempo(tempo);
  const barSec = 4.0 * (60.0 / bpm);
  const phraseSec = 4.0 * barSec;
  const phrases = Math.max(1, Math.round(Math.max(barSec, targetSeconds) / phraseSec));
  return phrases * phraseSec;
}
