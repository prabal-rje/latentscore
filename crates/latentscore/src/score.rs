//! Stateful pull-based [`Score`] engine with idempotent target steering,
//! smooth overlap-crossfaded block synthesis, and WAV export.

use crate::config::{
    apply_update_to_internal, interpolate_configs, to_internal, to_public, MusicConfig,
    MusicConfigUpdate, SynthConfig, SAMPLE_RATE,
};
use crate::dsp::{assemble, soft_limit};
use crate::mood::{mood_from_text, AutoDrift, EmbeddingIndex};
use crate::wav::{save_wav, WavOptions};
use std::f64::consts::PI;
use std::io;
use std::path::Path;

const OVERLAP_SECONDS: f64 = 0.65;
const FIXED_SESSION_GAIN: f64 = 0.62;

/// Stateful ambient music engine for interactive local apps and audio hosts.
///
/// Calling [`Score::steer`] sets the target mood idempotently (calling it once
/// or ten times with the same prompt aims for the exact same destination) while
/// the engine smoothly glides from the current audible state over `glide_seconds`.
#[derive(Debug, Clone)]
pub struct Score {
    current: SynthConfig,
    glide_from: SynthConfig,
    target: SynthConfig,
    glide_elapsed: f64,
    glide_seconds: f64,
    block_seconds: f64,
    session_seed: u32,
    t_offset: f64,
    prev_tail: Vec<f64>,
    fifo: Vec<f32>,
    fifo_read: usize,
    drifter: AutoDrift,
}

impl Score {
    /// Create a new Score starting from a [`MusicConfig`].
    pub fn new(initial: MusicConfig) -> Self {
        let internal = to_internal(&initial);
        let seed = 42_u32;
        Self {
            drifter: AutoDrift::new(&internal, (seed ^ 0x517C_C1B7) as u64),
            current: internal.clone(),
            glide_from: internal.clone(),
            target: internal,
            glide_elapsed: 0.0,
            glide_seconds: 3.0,
            block_seconds: 1.5,
            session_seed: seed,
            t_offset: 0.0,
            prev_tail: Vec::new(),
            fifo: Vec::new(),
            fifo_read: 0,
        }
    }

    /// Create a new Score starting from a natural-language mood prompt.
    pub fn from_text(prompt: &str) -> Self {
        Self::new(mood_from_text(prompt))
    }

    /// Set a deterministic session seed (preserves any attached [`EmbeddingIndex`]).
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.session_seed = seed & 0x7FFF_FFFF;
        self.drifter
            .reseed((self.session_seed ^ 0x517C_C1B7) as u64);
        self
    }

    /// Set the default glide duration (seconds) when steering to a new mood.
    pub fn with_glide_seconds(mut self, seconds: f64) -> Self {
        if seconds.is_finite() {
            self.glide_seconds = seconds.max(0.0);
        }
        self
    }

    /// Set internal render block length (seconds, clamped to `>= 0.25`).
    pub fn with_block_seconds(mut self, seconds: f64) -> Self {
        if seconds.is_finite() {
            self.block_seconds = seconds.max(0.25);
        }
        self
    }

    /// Attach an optional [`EmbeddingIndex`] for 14,035-mood similarity drift.
    pub fn with_embedding_index(mut self, index: EmbeddingIndex) -> Self {
        self.drifter = self.drifter.with_index(index);
        self
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    pub fn current_config(&self) -> MusicConfig {
        to_public(&self.current)
    }

    pub fn target_config(&self) -> MusicConfig {
        to_public(&self.target)
    }

    pub fn internal_config(&self) -> &SynthConfig {
        &self.current
    }

    fn set_target_internal(&mut self, next: SynthConfig) {
        if next != self.target {
            self.glide_from = self.current.clone();
            self.target = next;
            self.glide_elapsed = 0.0;
        }
    }

    /// Steer idempotently toward a mood described by `prompt`.
    pub fn steer(&mut self, prompt: &str) -> MusicConfig {
        let cfg = mood_from_text(prompt);
        self.steer_config(cfg)
    }

    /// Steer idempotently toward an explicit [`MusicConfig`].
    pub fn steer_config(&mut self, cfg: MusicConfig) -> MusicConfig {
        let internal = to_internal(&cfg);
        self.set_target_internal(internal);
        self.drifter.anchor_to_config(&self.target);
        to_public(&self.target)
    }

    /// Apply absolute fields or relative [`crate::Step`] adjustments to the target
    /// while preserving untouched continuous parameters.
    pub fn nudge(&mut self, update: &MusicConfigUpdate) -> MusicConfig {
        let next_internal = apply_update_to_internal(&self.target, update);
        self.set_target_internal(next_internal);
        self.drifter.anchor_to_config(&self.target);
        to_public(&self.target)
    }

    /// Advance one smooth step along the similarity / harmonic gradient.
    pub fn drift_next(&mut self) -> MusicConfig {
        let step = self.drifter.next_step(Some(&self.target));
        self.set_target_internal(step.synth_config);
        step.config
    }

    fn advance_config(&mut self, step_sec: f64) -> SynthConfig {
        if self.glide_seconds <= 1e-6 {
            self.current = self.target.clone();
            self.glide_from = self.target.clone();
            self.glide_elapsed = 0.0;
        } else if self.current != self.target {
            self.glide_elapsed = (self.glide_elapsed + step_sec).min(self.glide_seconds);
            let progress = (self.glide_elapsed / self.glide_seconds).clamp(0.0, 1.0);
            if progress >= 1.0 {
                self.current = self.target.clone();
                self.glide_from = self.target.clone();
            } else {
                self.current = interpolate_configs(&self.glide_from, &self.target, progress);
            }
        }
        self.current.clone()
    }

    fn render_internal_block(&mut self, dur: f64) -> Vec<f32> {
        let cfg = self.advance_config(dur);
        let sr = SAMPLE_RATE as f64;
        let overlap_sec = OVERLAP_SECONDS.min(dur * 0.35);
        let overlap_n = ((overlap_sec * sr).round() as usize).max(16);
        let block_n = ((dur * sr).round() as usize).max(1);
        let render_dur = ((block_n + overlap_n) as f64) / sr;

        let raw = assemble(&cfg, render_dur, false, self.session_seed, self.t_offset);
        let scaled: Vec<f64> = raw.into_iter().map(|s| s * FIXED_SESSION_GAIN).collect();
        let calibrated = soft_limit(&scaled, 0.92);

        let head_len = block_n.min(calibrated.len());
        let mut head = vec![0.0_f64; block_n];
        head[..head_len].copy_from_slice(&calibrated[..head_len]);

        let tail_end = (block_n + overlap_n).min(calibrated.len());
        let tail = if block_n < tail_end {
            calibrated[block_n..tail_end].to_vec()
        } else {
            Vec::new()
        };

        if !self.prev_tail.is_empty() {
            let blend_n = head.len().min(self.prev_tail.len());
            let denom = blend_n.saturating_sub(1).max(1) as f64;
            for i in 0..blend_n {
                let fade_in = 0.5 * (1.0 - (PI * (i as f64) / denom).cos());
                head[i] = self.prev_tail[i] * (1.0 - fade_in) + head[i] * fade_in;
            }
        }

        self.prev_tail = tail;
        self.t_offset += (block_n as f64) / sr;
        head.into_iter()
            .map(|s| s.clamp(-1.0, 1.0) as f32)
            .collect()
    }

    /// Return the next seamless mono float32 block (canonical FIFO order with [`Score::fill`]).
    pub fn next_block(&mut self, seconds: Option<f64>) -> Vec<f32> {
        let raw_dur = seconds.unwrap_or(self.block_seconds);
        let dur = if raw_dur.is_finite() {
            raw_dur.max(0.1)
        } else {
            self.block_seconds
        };
        let frames = ((dur * (SAMPLE_RATE as f64)).round() as usize).max(1);
        let mut out = vec![0.0_f32; frames];
        self.fill(&mut out);
        out
    }

    /// Fill `out` with an exact number of seamless mono float32 frames.
    pub fn fill(&mut self, out: &mut [f32]) {
        let mut written = 0_usize;
        while written < out.len() {
            let avail = self.fifo.len().saturating_sub(self.fifo_read);
            if avail == 0 {
                self.fifo = self.render_internal_block(self.block_seconds);
                self.fifo_read = 0;
                continue;
            }
            let take = (out.len() - written).min(avail);
            out[written..written + take]
                .copy_from_slice(&self.fifo[self.fifo_read..self.fifo_read + take]);
            self.fifo_read += take;
            written += take;
        }
    }

    /// Render `duration_seconds` and write a WAV file (mono or M-S stereo).
    /// Always encodes using the engine's native 44,100 Hz sample rate.
    pub fn save_wav(
        &mut self,
        path: impl AsRef<Path>,
        duration_seconds: f64,
        mut options: WavOptions,
    ) -> io::Result<()> {
        let safe_dur = if duration_seconds.is_finite() {
            duration_seconds.max(0.05)
        } else {
            1.0
        };
        let frames = ((safe_dur * (SAMPLE_RATE as f64)).round() as usize).max(1);
        let mut buf = vec![0.0_f32; frames];
        self.fill(&mut buf);
        options.sample_rate = SAMPLE_RATE;
        save_wav(path, &buf, options)
    }

    /// Start non-blocking real-time playback through the system audio output
    /// (requires the `playback` Cargo feature).
    #[cfg(feature = "playback")]
    pub fn play(self) -> Result<crate::playback::PlaybackHandle, crate::playback::PlaybackError> {
        crate::playback::start_playback(self)
    }
}
