//! # LatentScore
//!
//! Pure-Rust local procedural ambient music synthesizer and reactive mood engine.
//! Runs 100% locally with zero GPU and zero external runtime assets.
//!
//! ## Quickstart
//!
//! ```no_run
//! use latentscore::{Score, WavOptions};
//!
//! let mut score = Score::from_text("quiet morning desk").with_glide_seconds(3.5);
//! score.steer("rainy midnight coffee shop");
//! score.save_wav("morning-to-rain.wav", 8.0, WavOptions::default()).unwrap();
//! ```

pub mod config;
pub mod dsp;
pub mod mood;
#[cfg(feature = "playback")]
pub mod playback;
pub mod score;
pub mod wav;

pub use config::{
    apply_update, apply_update_to_internal, bpm_from_tempo, compute_bar_aligned_duration,
    interpolate_configs, to_internal, to_public, AccentStyle, AttackStyle, BassStyle,
    BrightnessLabel, ChordExtensions, ConfigError, EchoLabel, FieldOrStep, GrainStyle,
    HarmonyStyle, HumanFeelLabel, MelodyEngine, MelodyStyle, ModeName, MotionLabel, MusicConfig,
    MusicConfigUpdate, PadStyle, RhythmStyle, RootNote, SpaceLabel, Step, StereoLabel, SynthConfig,
    TempoLabel, TensionCurve, TextureStyle, SAMPLE_RATE, SCHEMA_VERSION,
};
pub use dsp::{
    apply_adsr, apply_highpass, apply_humanize, apply_lowpass, apply_master_hall_reverb, assemble,
    fade_into_nothingness, freq_from_note, generate_sawtooth, generate_sine, generate_square,
    generate_triangle, soft_fade_edges, soft_limit, SeededRng,
};
pub use mood::{
    mood_from_text, mood_from_text_with_base, AutoDrift, DriftStep, EmbeddingIndex, IndexedMood,
};
#[cfg(feature = "playback")]
pub use playback::{PlaybackError, PlaybackHandle};
pub use score::Score;
pub use wav::{encode_wav_bytes, save_wav, WavOptions};

/// One-shot helper: render a text prompt into mono float32 PCM samples in `[-1.0, 1.0]`.
pub fn render(prompt: &str, duration_seconds: f64) -> Vec<f32> {
    let cfg = to_internal(&mood_from_text(prompt));
    assemble(&cfg, duration_seconds, true, 42, 0.0)
        .into_iter()
        .map(|s| s.clamp(-1.0, 1.0) as f32)
        .collect()
}

/// One-shot helper: render an explicit [`MusicConfig`] into mono float32 PCM samples.
pub fn render_config(config: &MusicConfig, duration_seconds: f64, seed: u32) -> Vec<f32> {
    let cfg = to_internal(config);
    assemble(&cfg, duration_seconds, true, seed, 0.0)
        .into_iter()
        .map(|s| s.clamp(-1.0, 1.0) as f32)
        .collect()
}
