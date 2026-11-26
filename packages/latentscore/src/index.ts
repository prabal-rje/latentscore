/**
 * LatentScore: DOM-free core entry point ("latentscore").
 * Works in Node.js, Bun, Deno, Web Workers, SSR, and browsers.
 * For Web Audio browser playback, import from "latentscore/browser".
 */

export {
  BRIGHTNESS_MAP,
  DEFAULT_CONFIG,
  ECHO_MAP,
  HUMAN_MAP,
  MOTION_MAP,
  SAMPLE_RATE,
  SCHEMA_VERSION,
  SPACE_MAP,
  STEREO_MAP,
  TEMPO_MAP,
  applyUpdate,
  applyUpdateToInternal,
  blendConfigs,
  bpmFromTempo,
  computeBarAlignedDuration,
  interpolateConfigs,
  isStep,
  isSynthConfig,
  step,
  toInternalConfig,
  toPublicConfig,
  type AccentStyle,
  type AttackStyle,
  type BassStyle,
  type BrightnessLabel,
  type ChordExtensions,
  type ConfigInput,
  type DensityLevel,
  type EchoLabel,
  type GrainStyle,
  type HarmonyStyle,
  type HumanFeelLabel,
  type MelodyEngine,
  type MelodyStyle,
  type ModeName,
  type MotionLabel,
  type MusicConfig,
  type MusicConfigUpdate,
  type PadStyle,
  type PhraseLengthBars,
  type RhythmStyle,
  type RootNote,
  type SpaceLabel,
  type Step,
  type StereoLabel,
  type SynthConfig,
  type TempoLabel,
  type TensionCurve,
  type TextureStyle,
} from "./config.js";

export {
  SeededRng,
  applyAdsr,
  applyHighpass,
  applyHumanize,
  applyLowpass,
  applyMasterHallReverb,
  assemble,
  deriveSeed,
  freqFromNote,
  generateSawtooth,
  generateSine,
  generateSquare,
  generateTriangle,
  renderPcm,
  softFadeEdges,
  softLimit,
  type AssembleOptions,
} from "./dsp.js";

export {
  AutoDrift,
  EmbeddingIndex,
  moodFromText,
  type EmbeddingIndexManifest,
  type IndexedMood,
} from "./mood.js";

export {
  Score,
  createScore,
  render,
  type AutoDriftHandle,
  type AutoDriftOptions,
  type ScoreOptions,
  type SteerOptions,
} from "./score.js";

export {
  encodeWavBytes,
  fadeIntoNothingness,
  type EncodeWavOptions,
} from "./wav.js";
