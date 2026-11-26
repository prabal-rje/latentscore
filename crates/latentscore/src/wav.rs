//! Zero-dependency mono and mono-compatible M-S stereo 16-bit PCM WAV encoder.

use crate::config::SAMPLE_RATE;
use crate::dsp::fade_into_nothingness;
use std::fs;
use std::io;
use std::path::Path;

/// Options controlling WAV encoding.
#[derive(Debug, Clone, Copy)]
pub struct WavOptions {
    pub sample_rate: u32,
    /// `1` for mono, `2` for mono-compatible Mid-Side stereo.
    pub channels: u16,
    /// Stereo width in `[0.0, 1.0]` when `channels == 2`.
    pub stereo_width: f64,
    /// Raised-cosine fade-in duration in seconds.
    pub fade_in_seconds: f64,
    /// Smooth cubic fade-into-nothingness duration in seconds.
    pub fade_out_seconds: f64,
}

impl Default for WavOptions {
    fn default() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            channels: 1,
            stereo_width: 0.5,
            fade_in_seconds: 0.015,
            fade_out_seconds: 0.25,
        }
    }
}

fn apply_fade_in(samples: &mut [f64], fade_sec: f64, sr: u32) {
    let n = samples.len();
    if n < 2 || !fade_sec.is_finite() || fade_sec <= 0.0 {
        return;
    }
    let fade_n = ((fade_sec * (sr as f64)).round() as usize)
        .max(2)
        .min(n / 2);
    let denom = (fade_n - 1).max(1) as f64;
    for (i, sample) in samples.iter_mut().take(fade_n).enumerate() {
        let gain = 0.5 * (1.0 - (std::f64::consts::PI * (i as f64) / denom).cos());
        *sample *= gain;
    }
    samples[0] = 0.0;
}

fn try_encode_wav_bytes(samples: &[f32], options: WavOptions) -> io::Result<Vec<u8>> {
    let channels = if options.channels == 2 { 2_u16 } else { 1_u16 };
    let sr = options.sample_rate.max(8_000);
    let width = if options.stereo_width.is_finite() {
        options.stereo_width.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let fade_in_sec = if options.fade_in_seconds.is_finite() {
        options.fade_in_seconds.max(0.0)
    } else {
        0.0
    };
    let fade_out_sec = if options.fade_out_seconds.is_finite() {
        options.fade_out_seconds.max(0.0)
    } else {
        0.0
    };

    let num_frames = samples.len();
    let bytes_per_sample: u16 = 2;
    let data_bytes_usize = num_frames
        .checked_mul(channels as usize)
        .and_then(|v| v.checked_mul(bytes_per_sample as usize))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "WAV payload size overflow"))?;
    let data_bytes = u32::try_from(data_bytes_usize).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "WAV payload exceeds 4 GiB RIFF limit",
        )
    })?;
    let riff_chunk_size = 36_u32.checked_add(data_bytes).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "WAV RIFF header exceeds 4 GiB limit",
        )
    })?;

    let mut out = Vec::with_capacity(44 + data_bytes_usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&1_u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&(sr * (channels as u32) * (bytes_per_sample as u32)).to_le_bytes());
    out.extend_from_slice(&(channels * bytes_per_sample).to_le_bytes());
    out.extend_from_slice(&16_u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());

    let raw_mono: Vec<f64> = samples
        .iter()
        .map(|&s| if s.is_finite() { s as f64 } else { 0.0 })
        .collect();

    if channels == 1 {
        let mut mono = fade_into_nothingness(&raw_mono, fade_out_sec, sr);
        apply_fade_in(&mut mono, fade_in_sec, sr);
        for &s in &mono {
            let pcm = (s.clamp(-1.0, 1.0) * 32_767.0).round() as i16;
            out.extend_from_slice(&pcm.to_le_bytes());
        }
    } else {
        // Decorrelate before final entry/exit fades so the stereo tail ends at exact (0, 0).
        let delay_samples = ((0.011 * (sr as f64)).round() as usize).max(1);
        let side_gain = 0.28 * width;
        let mid_gain = 1.0 / (1.0 + side_gain);
        let alpha = (-2.0 * std::f64::consts::PI * 300.0 / (sr as f64)).exp();
        let mut prev_in = 0.0_f64;
        let mut prev_out = 0.0_f64;
        let mut left_raw = vec![0.0_f64; num_frames];
        let mut right_raw = vec![0.0_f64; num_frames];

        for i in 0..num_frames {
            let m = raw_mono[i];
            let del = if i >= delay_samples {
                raw_mono[i - delay_samples]
            } else {
                0.0
            };
            let hp = alpha * (prev_out + del - prev_in);
            prev_in = del;
            prev_out = hp;
            let side = hp * side_gain;
            left_raw[i] = m * mid_gain + side;
            right_raw[i] = m * mid_gain - side;
        }

        let mut left = fade_into_nothingness(&left_raw, fade_out_sec, sr);
        let mut right = fade_into_nothingness(&right_raw, fade_out_sec, sr);
        apply_fade_in(&mut left, fade_in_sec, sr);
        apply_fade_in(&mut right, fade_in_sec, sr);

        for i in 0..num_frames {
            let l_pcm = (left[i].clamp(-1.0, 1.0) * 32_767.0).round() as i16;
            let r_pcm = (right[i].clamp(-1.0, 1.0) * 32_767.0).round() as i16;
            out.extend_from_slice(&l_pcm.to_le_bytes());
            out.extend_from_slice(&r_pcm.to_le_bytes());
        }
    }

    Ok(out)
}

/// Encode mono float32 PCM samples into a complete 16-bit PCM WAV byte buffer.
///
/// When `options.channels == 2`, generates a mono-compatible Mid-Side stereo
/// image (`L = M + S`, `R = M - S`, so `L + R == 2 * M`) and applies entry and
/// exit fades after decorrelation so both channels start and end cleanly at zero.
pub fn encode_wav_bytes(samples: &[f32], options: WavOptions) -> Vec<u8> {
    try_encode_wav_bytes(samples, options).expect("WAV payload fits within 4 GiB RIFF limit")
}

/// Encode and write mono or stereo PCM samples to a `.wav` file on disk.
pub fn save_wav(path: impl AsRef<Path>, samples: &[f32], options: WavOptions) -> io::Result<()> {
    let bytes = try_encode_wav_bytes(samples, options)?;
    fs::write(path, bytes)
}
