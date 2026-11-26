//! 34-field LatentScore configuration schema, label-to-numeric mappings,
//! relative `Step` updates, interpolation, and bar-aligned phrase durations.
//! Matches `spec/config.schema.json` (`schema_version = 1`).

use serde::{Deserialize, Serialize};
use std::fmt;

pub const SCHEMA_VERSION: u32 = 1;
pub const SAMPLE_RATE: u32 = 44_100;

/// Relative delta for ordered label fields inside [`MusicConfigUpdate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step(pub i32);

/// Either an absolute value or a relative [`Step`] adjustment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FieldOrStep<T> {
    Absolute(T),
    Delta(Step),
}

impl<T> From<T> for FieldOrStep<T> {
    fn from(value: T) -> Self {
        Self::Absolute(value)
    }
}

/// Error returned when applying an invalid configuration update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ConfigError {}

macro_rules! string_enum {
    ($name:ident, $default:ident, [ $( ($variant:ident, $str:literal) ),+ $(,)? ]) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum $name {
            $(
                #[serde(rename = $str)]
                $variant,
            )+
        }

        impl Default for $name {
            fn default() -> Self {
                Self::$default
            }
        }

        impl $name {
            pub const ALL: &'static [Self] = &[ $( Self::$variant ),+ ];

            pub fn as_str(self) -> &'static str {
                match self {
                    $( Self::$variant => $str, )+
                }
            }

            pub fn from_str_opt(s: &str) -> Option<Self> {
                match s {
                    $( $str => Some(Self::$variant), )+
                    _ => None,
                }
            }

            pub fn step_by(self, delta: i32) -> Self {
                let idx = Self::ALL.iter().position(|&x| x == self).unwrap_or(0) as i32;
                let max_idx = (Self::ALL.len() as i32) - 1;
                let next = idx.saturating_add(delta).clamp(0, max_idx) as usize;
                Self::ALL[next]
            }
        }
    };
}

string_enum!(TempoLabel, Medium, [
    (VerySlow, "very_slow"),
    (Slow, "slow"),
    (Medium, "medium"),
    (Fast, "fast"),
    (VeryFast, "very_fast"),
]);

string_enum!(RootNote, C, [
    (C, "c"),
    (CSharp, "c#"),
    (D, "d"),
    (DSharp, "d#"),
    (E, "e"),
    (F, "f"),
    (FSharp, "f#"),
    (G, "g"),
    (GSharp, "g#"),
    (A, "a"),
    (ASharp, "a#"),
    (B, "b"),
]);

string_enum!(ModeName, Minor, [
    (Major, "major"),
    (Minor, "minor"),
    (Dorian, "dorian"),
    (Mixolydian, "mixolydian"),
]);

string_enum!(BrightnessLabel, Medium, [
    (VeryDark, "very_dark"),
    (Dark, "dark"),
    (Medium, "medium"),
    (Bright, "bright"),
    (VeryBright, "very_bright"),
]);

string_enum!(SpaceLabel, Medium, [
    (Dry, "dry"),
    (Small, "small"),
    (Medium, "medium"),
    (Large, "large"),
    (Vast, "vast"),
]);

string_enum!(BassStyle, Drone, [
    (Drone, "drone"),
    (Sustained, "sustained"),
    (Pulsing, "pulsing"),
    (Walking, "walking"),
    (FifthDrone, "fifth_drone"),
    (SubPulse, "sub_pulse"),
    (Octave, "octave"),
    (ArpBass, "arp_bass"),
]);

string_enum!(PadStyle, WarmSlow, [
    (WarmSlow, "warm_slow"),
    (DarkSustained, "dark_sustained"),
    (Cinematic, "cinematic"),
    (ThinHigh, "thin_high"),
    (AmbientDrift, "ambient_drift"),
    (StackedFifths, "stacked_fifths"),
    (BrightOpen, "bright_open"),
]);

string_enum!(MelodyStyle, Contemplative, [
    (Contemplative, "contemplative"),
    (ContemplativeMinor, "contemplative_minor"),
    (Rising, "rising"),
    (Falling, "falling"),
    (Minimal, "minimal"),
    (Ornamental, "ornamental"),
    (ArpMelody, "arp_melody"),
    (CallResponse, "call_response"),
    (Heroic, "heroic"),
    (Procedural, "procedural"),
]);

string_enum!(RhythmStyle, Minimal, [
    (None, "none"),
    (Minimal, "minimal"),
    (Heartbeat, "heartbeat"),
    (SoftFour, "soft_four"),
    (HatsOnly, "hats_only"),
    (Electronic, "electronic"),
    (KitLight, "kit_light"),
    (KitMedium, "kit_medium"),
    (Military, "military"),
    (TablaEssence, "tabla_essence"),
    (Brush, "brush"),
]);

string_enum!(TextureStyle, Shimmer, [
    (None, "none"),
    (Shimmer, "shimmer"),
    (ShimmerSlow, "shimmer_slow"),
    (VinylCrackle, "vinyl_crackle"),
    (Breath, "breath"),
    (Stars, "stars"),
    (Glitch, "glitch"),
    (NoiseWash, "noise_wash"),
    (Crystal, "crystal"),
    (PadWhisper, "pad_whisper"),
]);

string_enum!(AccentStyle, Bells, [
    (None, "none"),
    (Bells, "bells"),
    (BellsDense, "bells_dense"),
    (Pluck, "pluck"),
    (Chime, "chime"),
    (Blip, "blip"),
    (BlipRandom, "blip_random"),
    (BrassHit, "brass_hit"),
    (Wind, "wind"),
    (ArpAccent, "arp_accent"),
    (PianoNote, "piano_note"),
]);

string_enum!(MotionLabel, Medium, [
    (Static, "static"),
    (Slow, "slow"),
    (Medium, "medium"),
    (Fast, "fast"),
    (Chaotic, "chaotic"),
]);

string_enum!(AttackStyle, Medium, [
    (Soft, "soft"),
    (Medium, "medium"),
    (Sharp, "sharp"),
]);

string_enum!(StereoLabel, Medium, [
    (Mono, "mono"),
    (Narrow, "narrow"),
    (Medium, "medium"),
    (Wide, "wide"),
    (UltraWide, "ultra_wide"),
]);

string_enum!(EchoLabel, Medium, [
    (None, "none"),
    (Subtle, "subtle"),
    (Medium, "medium"),
    (Heavy, "heavy"),
    (Infinite, "infinite"),
]);

string_enum!(HumanFeelLabel, Robotic, [
    (Robotic, "robotic"),
    (Tight, "tight"),
    (Natural, "natural"),
    (Loose, "loose"),
    (Drunk, "drunk"),
]);

string_enum!(GrainStyle, Clean, [
    (Clean, "clean"),
    (Warm, "warm"),
    (Gritty, "gritty"),
]);

string_enum!(MelodyEngine, Pattern, [
    (Pattern, "pattern"),
    (Procedural, "procedural"),
]);

string_enum!(TensionCurve, Arc, [
    (Arc, "arc"),
    (Ramp, "ramp"),
    (Waves, "waves"),
]);

string_enum!(HarmonyStyle, Auto, [
    (Auto, "auto"),
    (Pop, "pop"),
    (Jazz, "jazz"),
    (Cinematic, "cinematic"),
    (Ambient, "ambient"),
]);

string_enum!(ChordExtensions, Triads, [
    (Triads, "triads"),
    (Sevenths, "sevenths"),
    (Lush, "lush"),
]);

impl TempoLabel {
    pub fn to_f64(self) -> f64 {
        match self {
            Self::VerySlow => 0.15,
            Self::Slow => 0.30,
            Self::Medium => 0.50,
            Self::Fast => 0.70,
            Self::VeryFast => 0.90,
        }
    }

    pub fn closest(v: f64) -> Self {
        *Self::ALL
            .iter()
            .min_by(|a, b| (a.to_f64() - v).abs().total_cmp(&(b.to_f64() - v).abs()))
            .unwrap_or(&Self::Medium)
    }
}

impl BrightnessLabel {
    pub fn to_f64(self) -> f64 {
        match self {
            Self::VeryDark => 0.10,
            Self::Dark => 0.30,
            Self::Medium => 0.50,
            Self::Bright => 0.70,
            Self::VeryBright => 0.90,
        }
    }

    pub fn closest(v: f64) -> Self {
        *Self::ALL
            .iter()
            .min_by(|a, b| (a.to_f64() - v).abs().total_cmp(&(b.to_f64() - v).abs()))
            .unwrap_or(&Self::Medium)
    }
}

impl SpaceLabel {
    pub fn to_f64(self) -> f64 {
        match self {
            Self::Dry => 0.10,
            Self::Small => 0.30,
            Self::Medium => 0.50,
            Self::Large => 0.70,
            Self::Vast => 0.95,
        }
    }

    pub fn closest(v: f64) -> Self {
        *Self::ALL
            .iter()
            .min_by(|a, b| (a.to_f64() - v).abs().total_cmp(&(b.to_f64() - v).abs()))
            .unwrap_or(&Self::Medium)
    }
}

impl MotionLabel {
    pub fn to_f64(self) -> f64 {
        match self {
            Self::Static => 0.10,
            Self::Slow => 0.30,
            Self::Medium => 0.50,
            Self::Fast => 0.70,
            Self::Chaotic => 0.90,
        }
    }

    pub fn closest(v: f64) -> Self {
        *Self::ALL
            .iter()
            .min_by(|a, b| (a.to_f64() - v).abs().total_cmp(&(b.to_f64() - v).abs()))
            .unwrap_or(&Self::Medium)
    }
}

impl StereoLabel {
    pub fn to_f64(self) -> f64 {
        match self {
            Self::Mono => 0.0,
            Self::Narrow => 0.25,
            Self::Medium => 0.50,
            Self::Wide => 0.75,
            Self::UltraWide => 1.0,
        }
    }

    pub fn closest(v: f64) -> Self {
        *Self::ALL
            .iter()
            .min_by(|a, b| (a.to_f64() - v).abs().total_cmp(&(b.to_f64() - v).abs()))
            .unwrap_or(&Self::Medium)
    }
}

impl EchoLabel {
    pub fn to_f64(self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Subtle => 0.25,
            Self::Medium => 0.50,
            Self::Heavy => 0.75,
            Self::Infinite => 0.95,
        }
    }

    pub fn closest(v: f64) -> Self {
        *Self::ALL
            .iter()
            .min_by(|a, b| (a.to_f64() - v).abs().total_cmp(&(b.to_f64() - v).abs()))
            .unwrap_or(&Self::Medium)
    }
}

impl HumanFeelLabel {
    pub fn to_f64(self) -> f64 {
        match self {
            Self::Robotic => 0.0,
            Self::Tight => 0.15,
            Self::Natural => 0.30,
            Self::Loose => 0.50,
            Self::Drunk => 0.80,
        }
    }

    pub fn closest(v: f64) -> Self {
        *Self::ALL
            .iter()
            .min_by(|a, b| (a.to_f64() - v).abs().total_cmp(&(b.to_f64() - v).abs()))
            .unwrap_or(&Self::Robotic)
    }
}

fn default_schema_version() -> u32 {
    SCHEMA_VERSION
}
fn default_density() -> u8 {
    5
}
fn default_phrase_len_bars() -> u8 {
    4
}
fn default_melody_density() -> f64 {
    0.45
}
fn default_syncopation() -> f64 {
    0.20
}
fn default_motif_repeat() -> f64 {
    0.50
}
fn default_step_bias() -> f64 {
    0.75
}
fn default_chromatic_prob() -> f64 {
    0.05
}
fn default_cadence_strength() -> f64 {
    0.65
}
fn default_reg_min() -> u8 {
    4
}
fn default_reg_max() -> u8 {
    6
}
fn default_chord_change_bars() -> u8 {
    1
}

/// Public, human-readable 34-field musical configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MusicConfig {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub tempo: TempoLabel,
    pub root: RootNote,
    pub mode: ModeName,
    pub brightness: BrightnessLabel,
    pub space: SpaceLabel,
    #[serde(default = "default_density")]
    pub density: u8,
    pub bass: BassStyle,
    pub pad: PadStyle,
    pub melody: MelodyStyle,
    pub rhythm: RhythmStyle,
    pub texture: TextureStyle,
    pub accent: AccentStyle,
    pub motion: MotionLabel,
    pub attack: AttackStyle,
    pub stereo: StereoLabel,
    pub depth: bool,
    pub echo: EchoLabel,
    pub human: HumanFeelLabel,
    pub grain: GrainStyle,
    pub melody_engine: MelodyEngine,
    #[serde(default = "default_phrase_len_bars")]
    pub phrase_len_bars: u8,
    #[serde(default = "default_melody_density")]
    pub melody_density: f64,
    #[serde(default = "default_syncopation")]
    pub syncopation: f64,
    pub swing: f64,
    #[serde(default = "default_motif_repeat")]
    pub motif_repeat_prob: f64,
    #[serde(default = "default_step_bias")]
    pub step_bias: f64,
    #[serde(default = "default_chromatic_prob")]
    pub chromatic_prob: f64,
    #[serde(default = "default_cadence_strength")]
    pub cadence_strength: f64,
    #[serde(default = "default_reg_min")]
    pub register_min_oct: u8,
    #[serde(default = "default_reg_max")]
    pub register_max_oct: u8,
    pub tension_curve: TensionCurve,
    pub harmony_style: HarmonyStyle,
    #[serde(default = "default_chord_change_bars")]
    pub chord_change_bars: u8,
    pub chord_extensions: ChordExtensions,
}

impl Default for MusicConfig {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            tempo: TempoLabel::Medium,
            root: RootNote::C,
            mode: ModeName::Minor,
            brightness: BrightnessLabel::Medium,
            space: SpaceLabel::Medium,
            density: 5,
            bass: BassStyle::Drone,
            pad: PadStyle::WarmSlow,
            melody: MelodyStyle::Contemplative,
            rhythm: RhythmStyle::Minimal,
            texture: TextureStyle::Shimmer,
            accent: AccentStyle::Bells,
            motion: MotionLabel::Medium,
            attack: AttackStyle::Medium,
            stereo: StereoLabel::Medium,
            depth: false,
            echo: EchoLabel::Medium,
            human: HumanFeelLabel::Robotic,
            grain: GrainStyle::Clean,
            melody_engine: MelodyEngine::Pattern,
            phrase_len_bars: 4,
            melody_density: 0.45,
            syncopation: 0.20,
            swing: 0.0,
            motif_repeat_prob: 0.50,
            step_bias: 0.75,
            chromatic_prob: 0.05,
            cadence_strength: 0.65,
            register_min_oct: 4,
            register_max_oct: 6,
            tension_curve: TensionCurve::Arc,
            harmony_style: HarmonyStyle::Auto,
            chord_change_bars: 1,
            chord_extensions: ChordExtensions::Triads,
        }
    }
}

/// Numeric internal configuration consumed by the DSP engine.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthConfig {
    pub tempo: f64,
    pub root: RootNote,
    pub mode: ModeName,
    pub brightness: f64,
    pub space: f64,
    pub density: u8,
    pub bass: BassStyle,
    pub pad: PadStyle,
    pub melody: MelodyStyle,
    pub rhythm: RhythmStyle,
    pub texture: TextureStyle,
    pub accent: AccentStyle,
    pub motion: f64,
    pub attack: AttackStyle,
    pub stereo: f64,
    pub depth: bool,
    pub echo: f64,
    pub human: f64,
    pub grain: GrainStyle,
    pub melody_engine: MelodyEngine,
    pub phrase_len_bars: u8,
    pub melody_density: f64,
    pub syncopation: f64,
    pub swing: f64,
    pub motif_repeat_prob: f64,
    pub step_bias: f64,
    pub chromatic_prob: f64,
    pub cadence_strength: f64,
    pub register_min_oct: u8,
    pub register_max_oct: u8,
    pub tension_curve: TensionCurve,
    pub harmony_style: HarmonyStyle,
    pub chord_change_bars: u8,
    pub chord_extensions: ChordExtensions,
}

impl Default for SynthConfig {
    fn default() -> Self {
        to_internal(&MusicConfig::default())
    }
}

/// Strict partial update supporting relative [`Step`] adjustments.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicConfigUpdate {
    pub tempo: Option<FieldOrStep<TempoLabel>>,
    pub root: Option<RootNote>,
    pub mode: Option<ModeName>,
    pub brightness: Option<FieldOrStep<BrightnessLabel>>,
    pub space: Option<FieldOrStep<SpaceLabel>>,
    pub density: Option<FieldOrStep<u8>>,
    pub bass: Option<BassStyle>,
    pub pad: Option<PadStyle>,
    pub melody: Option<MelodyStyle>,
    pub rhythm: Option<RhythmStyle>,
    pub texture: Option<TextureStyle>,
    pub accent: Option<AccentStyle>,
    pub motion: Option<FieldOrStep<MotionLabel>>,
    pub attack: Option<FieldOrStep<AttackStyle>>,
    pub stereo: Option<FieldOrStep<StereoLabel>>,
    pub depth: Option<bool>,
    pub echo: Option<FieldOrStep<EchoLabel>>,
    pub human: Option<FieldOrStep<HumanFeelLabel>>,
    pub grain: Option<FieldOrStep<GrainStyle>>,
    pub melody_engine: Option<MelodyEngine>,
    pub phrase_len_bars: Option<u8>,
    pub melody_density: Option<f64>,
    pub syncopation: Option<f64>,
    pub swing: Option<f64>,
    pub motif_repeat_prob: Option<f64>,
    pub step_bias: Option<f64>,
    pub chromatic_prob: Option<f64>,
    pub cadence_strength: Option<f64>,
    pub register_min_oct: Option<u8>,
    pub register_max_oct: Option<u8>,
    pub tension_curve: Option<TensionCurve>,
    pub harmony_style: Option<HarmonyStyle>,
    pub chord_change_bars: Option<u8>,
    pub chord_extensions: Option<ChordExtensions>,
}

fn clamp_unit(v: f64, fallback: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        fallback
    }
}

/// Convert a public [`MusicConfig`] to a numeric [`SynthConfig`].
pub fn to_internal(cfg: &MusicConfig) -> SynthConfig {
    SynthConfig {
        tempo: cfg.tempo.to_f64(),
        root: cfg.root,
        mode: cfg.mode,
        brightness: cfg.brightness.to_f64(),
        space: cfg.space.to_f64(),
        density: cfg.density.clamp(2, 6),
        bass: cfg.bass,
        pad: cfg.pad,
        melody: cfg.melody,
        rhythm: cfg.rhythm,
        texture: cfg.texture,
        accent: cfg.accent,
        motion: cfg.motion.to_f64(),
        attack: cfg.attack,
        stereo: cfg.stereo.to_f64(),
        depth: cfg.depth,
        echo: cfg.echo.to_f64(),
        human: cfg.human.to_f64(),
        grain: cfg.grain,
        melody_engine: cfg.melody_engine,
        phrase_len_bars: cfg.phrase_len_bars.clamp(1, 16),
        melody_density: clamp_unit(cfg.melody_density, 0.45),
        syncopation: clamp_unit(cfg.syncopation, 0.20),
        swing: clamp_unit(cfg.swing, 0.0),
        motif_repeat_prob: clamp_unit(cfg.motif_repeat_prob, 0.50),
        step_bias: clamp_unit(cfg.step_bias, 0.75),
        chromatic_prob: clamp_unit(cfg.chromatic_prob, 0.05),
        cadence_strength: clamp_unit(cfg.cadence_strength, 0.65),
        register_min_oct: cfg.register_min_oct.clamp(1, 8),
        register_max_oct: cfg.register_max_oct.clamp(1, 8),
        tension_curve: cfg.tension_curve,
        harmony_style: cfg.harmony_style,
        chord_change_bars: cfg.chord_change_bars.clamp(1, 4),
        chord_extensions: cfg.chord_extensions,
    }
}

/// Convert a numeric [`SynthConfig`] back to its closest public [`MusicConfig`].
pub fn to_public(cfg: &SynthConfig) -> MusicConfig {
    let phrase_len = if cfg.phrase_len_bars <= 3 {
        2
    } else if cfg.phrase_len_bars <= 6 {
        4
    } else {
        8
    };
    MusicConfig {
        schema_version: SCHEMA_VERSION,
        tempo: TempoLabel::closest(cfg.tempo),
        root: cfg.root,
        mode: cfg.mode,
        brightness: BrightnessLabel::closest(cfg.brightness),
        space: SpaceLabel::closest(cfg.space),
        density: cfg.density.clamp(2, 6),
        bass: cfg.bass,
        pad: cfg.pad,
        melody: cfg.melody,
        rhythm: cfg.rhythm,
        texture: cfg.texture,
        accent: cfg.accent,
        motion: MotionLabel::closest(cfg.motion),
        attack: cfg.attack,
        stereo: StereoLabel::closest(cfg.stereo),
        depth: cfg.depth,
        echo: EchoLabel::closest(cfg.echo),
        human: HumanFeelLabel::closest(cfg.human),
        grain: cfg.grain,
        melody_engine: cfg.melody_engine,
        phrase_len_bars: phrase_len,
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
    }
}

/// Apply a [`MusicConfigUpdate`] (including relative [`Step`] adjustments) to `base`.
pub fn apply_update(base: &MusicConfig, upd: &MusicConfigUpdate) -> MusicConfig {
    let mut out = base.clone();
    if let Some(v) = &upd.tempo {
        out.tempo = match v {
            FieldOrStep::Absolute(x) => *x,
            FieldOrStep::Delta(Step(d)) => out.tempo.step_by(*d),
        };
    }
    if let Some(v) = upd.root {
        out.root = v;
    }
    if let Some(v) = upd.mode {
        out.mode = v;
    }
    if let Some(v) = &upd.brightness {
        out.brightness = match v {
            FieldOrStep::Absolute(x) => *x,
            FieldOrStep::Delta(Step(d)) => out.brightness.step_by(*d),
        };
    }
    if let Some(v) = &upd.space {
        out.space = match v {
            FieldOrStep::Absolute(x) => *x,
            FieldOrStep::Delta(Step(d)) => out.space.step_by(*d),
        };
    }
    if let Some(v) = &upd.density {
        out.density = match v {
            FieldOrStep::Absolute(x) => (*x).clamp(2, 6),
            FieldOrStep::Delta(Step(d)) => (out.density as i32).saturating_add(*d).clamp(2, 6) as u8,
        };
    }
    if let Some(v) = upd.bass {
        out.bass = v;
    }
    if let Some(v) = upd.pad {
        out.pad = v;
    }
    if let Some(v) = upd.melody {
        out.melody = v;
    }
    if let Some(v) = upd.rhythm {
        out.rhythm = v;
    }
    if let Some(v) = upd.texture {
        out.texture = v;
    }
    if let Some(v) = upd.accent {
        out.accent = v;
    }
    if let Some(v) = &upd.motion {
        out.motion = match v {
            FieldOrStep::Absolute(x) => *x,
            FieldOrStep::Delta(Step(d)) => out.motion.step_by(*d),
        };
    }
    if let Some(v) = &upd.attack {
        out.attack = match v {
            FieldOrStep::Absolute(x) => *x,
            FieldOrStep::Delta(Step(d)) => out.attack.step_by(*d),
        };
    }
    if let Some(v) = &upd.stereo {
        out.stereo = match v {
            FieldOrStep::Absolute(x) => *x,
            FieldOrStep::Delta(Step(d)) => out.stereo.step_by(*d),
        };
    }
    if let Some(v) = upd.depth {
        out.depth = v;
    }
    if let Some(v) = &upd.echo {
        out.echo = match v {
            FieldOrStep::Absolute(x) => *x,
            FieldOrStep::Delta(Step(d)) => out.echo.step_by(*d),
        };
    }
    if let Some(v) = &upd.human {
        out.human = match v {
            FieldOrStep::Absolute(x) => *x,
            FieldOrStep::Delta(Step(d)) => out.human.step_by(*d),
        };
    }
    if let Some(v) = &upd.grain {
        out.grain = match v {
            FieldOrStep::Absolute(x) => *x,
            FieldOrStep::Delta(Step(d)) => out.grain.step_by(*d),
        };
    }
    if let Some(v) = upd.melody_engine {
        out.melody_engine = v;
    }
    if let Some(v) = upd.phrase_len_bars {
        out.phrase_len_bars = v;
    }
    if let Some(v) = upd.melody_density {
        out.melody_density = v.clamp(0.0, 1.0);
    }
    if let Some(v) = upd.syncopation {
        out.syncopation = v.clamp(0.0, 1.0);
    }
    if let Some(v) = upd.swing {
        out.swing = v.clamp(0.0, 1.0);
    }
    if let Some(v) = upd.motif_repeat_prob {
        out.motif_repeat_prob = v.clamp(0.0, 1.0);
    }
    if let Some(v) = upd.step_bias {
        out.step_bias = v.clamp(0.0, 1.0);
    }
    if let Some(v) = upd.chromatic_prob {
        out.chromatic_prob = v.clamp(0.0, 1.0);
    }
    if let Some(v) = upd.cadence_strength {
        out.cadence_strength = v.clamp(0.0, 1.0);
    }
    if let Some(v) = upd.register_min_oct {
        out.register_min_oct = v;
    }
    if let Some(v) = upd.register_max_oct {
        out.register_max_oct = v;
    }
    if let Some(v) = upd.tension_curve {
        out.tension_curve = v;
    }
    if let Some(v) = upd.harmony_style {
        out.harmony_style = v;
    }
    if let Some(v) = upd.chord_change_bars {
        out.chord_change_bars = v.clamp(1, 4);
    }
    if let Some(v) = upd.chord_extensions {
        out.chord_extensions = v;
    }
    out
}

/// Apply a [`MusicConfigUpdate`] directly onto an internal [`SynthConfig`] while
/// preserving untouched continuous floats (important during drift or partial glides).
pub fn apply_update_to_internal(base: &SynthConfig, upd: &MusicConfigUpdate) -> SynthConfig {
    let mut out = base.clone();
    if let Some(v) = &upd.tempo {
        out.tempo = match v {
            FieldOrStep::Absolute(x) => x.to_f64(),
            FieldOrStep::Delta(Step(d)) => TempoLabel::closest(out.tempo).step_by(*d).to_f64(),
        };
    }
    if let Some(v) = upd.root {
        out.root = v;
    }
    if let Some(v) = upd.mode {
        out.mode = v;
    }
    if let Some(v) = &upd.brightness {
        out.brightness = match v {
            FieldOrStep::Absolute(x) => x.to_f64(),
            FieldOrStep::Delta(Step(d)) => {
                BrightnessLabel::closest(out.brightness).step_by(*d).to_f64()
            }
        };
    }
    if let Some(v) = &upd.space {
        out.space = match v {
            FieldOrStep::Absolute(x) => x.to_f64(),
            FieldOrStep::Delta(Step(d)) => SpaceLabel::closest(out.space).step_by(*d).to_f64(),
        };
    }
    if let Some(v) = &upd.density {
        out.density = match v {
            FieldOrStep::Absolute(x) => (*x).clamp(2, 6),
            FieldOrStep::Delta(Step(d)) => (out.density as i32).saturating_add(*d).clamp(2, 6) as u8,
        };
    }
    if let Some(v) = upd.bass {
        out.bass = v;
    }
    if let Some(v) = upd.pad {
        out.pad = v;
    }
    if let Some(v) = upd.melody {
        out.melody = v;
    }
    if let Some(v) = upd.rhythm {
        out.rhythm = v;
    }
    if let Some(v) = upd.texture {
        out.texture = v;
    }
    if let Some(v) = upd.accent {
        out.accent = v;
    }
    if let Some(v) = &upd.motion {
        out.motion = match v {
            FieldOrStep::Absolute(x) => x.to_f64(),
            FieldOrStep::Delta(Step(d)) => MotionLabel::closest(out.motion).step_by(*d).to_f64(),
        };
    }
    if let Some(v) = &upd.attack {
        out.attack = match v {
            FieldOrStep::Absolute(x) => *x,
            FieldOrStep::Delta(Step(d)) => out.attack.step_by(*d),
        };
    }
    if let Some(v) = &upd.stereo {
        out.stereo = match v {
            FieldOrStep::Absolute(x) => x.to_f64(),
            FieldOrStep::Delta(Step(d)) => StereoLabel::closest(out.stereo).step_by(*d).to_f64(),
        };
    }
    if let Some(v) = upd.depth {
        out.depth = v;
    }
    if let Some(v) = &upd.echo {
        out.echo = match v {
            FieldOrStep::Absolute(x) => x.to_f64(),
            FieldOrStep::Delta(Step(d)) => EchoLabel::closest(out.echo).step_by(*d).to_f64(),
        };
    }
    if let Some(v) = &upd.human {
        out.human = match v {
            FieldOrStep::Absolute(x) => x.to_f64(),
            FieldOrStep::Delta(Step(d)) => {
                HumanFeelLabel::closest(out.human).step_by(*d).to_f64()
            }
        };
    }
    if let Some(v) = &upd.grain {
        out.grain = match v {
            FieldOrStep::Absolute(x) => *x,
            FieldOrStep::Delta(Step(d)) => out.grain.step_by(*d),
        };
    }
    if let Some(v) = upd.melody_engine {
        out.melody_engine = v;
    }
    if let Some(v) = upd.phrase_len_bars {
        out.phrase_len_bars = v.clamp(1, 16);
    }
    if let Some(v) = upd.melody_density {
        out.melody_density = clamp_unit(v, out.melody_density);
    }
    if let Some(v) = upd.syncopation {
        out.syncopation = clamp_unit(v, out.syncopation);
    }
    if let Some(v) = upd.swing {
        out.swing = clamp_unit(v, out.swing);
    }
    if let Some(v) = upd.motif_repeat_prob {
        out.motif_repeat_prob = clamp_unit(v, out.motif_repeat_prob);
    }
    if let Some(v) = upd.step_bias {
        out.step_bias = clamp_unit(v, out.step_bias);
    }
    if let Some(v) = upd.chromatic_prob {
        out.chromatic_prob = clamp_unit(v, out.chromatic_prob);
    }
    if let Some(v) = upd.cadence_strength {
        out.cadence_strength = clamp_unit(v, out.cadence_strength);
    }
    if let Some(v) = upd.register_min_oct {
        out.register_min_oct = v.clamp(1, 8);
    }
    if let Some(v) = upd.register_max_oct {
        out.register_max_oct = v.clamp(1, 8);
    }
    if let Some(v) = upd.tension_curve {
        out.tension_curve = v;
    }
    if let Some(v) = upd.harmony_style {
        out.harmony_style = v;
    }
    if let Some(v) = upd.chord_change_bars {
        out.chord_change_bars = v.clamp(1, 4);
    }
    if let Some(v) = upd.chord_extensions {
        out.chord_extensions = v;
    }
    out
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Interpolate smoothly between two [`SynthConfig`] instances with staggered
/// layer handoffs matching Python and TypeScript.
pub fn interpolate_configs(a: &SynthConfig, b: &SynthConfig, t_raw: f64) -> SynthConfig {
    let t = t_raw.clamp(0.0, 1.0);
    SynthConfig {
        tempo: lerp(a.tempo, b.tempo, t),
        root: if t < 0.5 { a.root } else { b.root },
        mode: if t < 0.5 { a.mode } else { b.mode },
        brightness: lerp(a.brightness, b.brightness, t),
        space: lerp(a.space, b.space, t),
        density: lerp(a.density as f64, b.density as f64, t).round().clamp(2.0, 6.0) as u8,
        bass: if t < 0.4 { a.bass } else { b.bass },
        pad: if t < 0.5 { a.pad } else { b.pad },
        melody: if t < 0.6 { a.melody } else { b.melody },
        rhythm: if t < 0.5 { a.rhythm } else { b.rhythm },
        texture: if t < 0.7 { a.texture } else { b.texture },
        accent: if t < 0.8 { a.accent } else { b.accent },
        motion: lerp(a.motion, b.motion, t),
        attack: if t < 0.5 { a.attack } else { b.attack },
        stereo: lerp(a.stereo, b.stereo, t),
        depth: if t < 0.5 { a.depth } else { b.depth },
        echo: lerp(a.echo, b.echo, t),
        human: lerp(a.human, b.human, t),
        grain: if t < 0.5 { a.grain } else { b.grain },
        melody_engine: if t < 0.5 { a.melody_engine } else { b.melody_engine },
        phrase_len_bars: lerp(a.phrase_len_bars as f64, b.phrase_len_bars as f64, t)
            .round()
            .clamp(1.0, 16.0) as u8,
        melody_density: lerp(a.melody_density, b.melody_density, t),
        syncopation: lerp(a.syncopation, b.syncopation, t),
        swing: lerp(a.swing, b.swing, t),
        motif_repeat_prob: lerp(a.motif_repeat_prob, b.motif_repeat_prob, t),
        step_bias: lerp(a.step_bias, b.step_bias, t),
        chromatic_prob: lerp(a.chromatic_prob, b.chromatic_prob, t),
        cadence_strength: lerp(a.cadence_strength, b.cadence_strength, t),
        register_min_oct: lerp(a.register_min_oct as f64, b.register_min_oct as f64, t)
            .round()
            .clamp(1.0, 8.0) as u8,
        register_max_oct: lerp(a.register_max_oct as f64, b.register_max_oct as f64, t)
            .round()
            .clamp(1.0, 8.0) as u8,
        tension_curve: if t < 0.5 { a.tension_curve } else { b.tension_curve },
        harmony_style: if t < 0.5 { a.harmony_style } else { b.harmony_style },
        chord_change_bars: lerp(a.chord_change_bars as f64, b.chord_change_bars as f64, t)
            .round()
            .clamp(1.0, 4.0) as u8,
        chord_extensions: if t < 0.5 { a.chord_extensions } else { b.chord_extensions },
    }
}

/// Convert normalized tempo `[0, 1]` to BPM (`55 + 110 * tempo`).
pub fn bpm_from_tempo(tempo: f64) -> f64 {
    55.0 + 110.0 * tempo.clamp(0.0, 1.0)
}

/// Snap a target duration in seconds to an exact 4-bar musical phrase boundary.
pub fn compute_bar_aligned_duration(tempo: f64, target_seconds: f64) -> f64 {
    let bpm = bpm_from_tempo(tempo);
    let bar_sec = 4.0 * (60.0 / bpm);
    let phrase_sec = 4.0 * bar_sec;
    let phrases = (target_seconds.max(bar_sec) / phrase_sec).round().max(1.0);
    phrases * phrase_sec
}
