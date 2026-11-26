//! Pure-Rust Float64 procedural ambient synthesizer matching Python and TypeScript
//! (including all Round 1 & Round 2 studio DSP repairs).

use crate::config::{
    bpm_from_tempo, AccentStyle, AttackStyle, BassStyle, ChordExtensions, GrainStyle,
    HarmonyStyle, MelodyEngine, MelodyStyle, ModeName, PadStyle, RhythmStyle, RootNote,
    SynthConfig, TensionCurve, TextureStyle, SAMPLE_RATE,
};
use std::collections::HashMap;
use std::f64::consts::{PI, SQRT_2};
use std::sync::{Mutex, OnceLock};

const TWO_PI: f64 = 2.0 * PI;

const REVERB_TAPS: &[(f64, f64)] = &[
    (0.0233, 0.24),
    (0.0317, -0.22),
    (0.0431, 0.20),
    (0.0571, -0.18),
    (0.0733, 0.16),
    (0.0947, -0.14),
    (0.1181, 0.12),
    (0.1489, -0.10),
    (0.1847, 0.085),
    (0.2269, -0.07),
    (0.2741, 0.055),
    (0.3313, -0.04),
];

const DIFFUSION_TAPS: &[(f64, f64)] = &[
    (0.0073, 0.35),
    (0.0131, -0.28),
    (0.0197, 0.22),
];

// === Deterministic PRNG ===

#[derive(Debug, Clone)]
pub struct SeededRng {
    state: u32,
    spare: Option<f64>,
}

impl SeededRng {
    pub fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9 } else { seed },
            spare: None,
        }
    }

    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6D2B_79F5);
        let mut t = (self.state ^ (self.state >> 15)).wrapping_mul(1 | self.state);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        ((t ^ (t >> 14)) as f64) / 4_294_967_296.0
    }

    pub fn int_range(&mut self, lo: usize, hi_exclusive: usize) -> usize {
        if hi_exclusive <= lo {
            return lo;
        }
        let span = hi_exclusive - lo;
        lo + ((self.next_f64() * (span as f64)) as usize).min(span - 1)
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        if let Some(v) = self.spare.take() {
            return mean + std * v;
        }
        let mut u = 0.0_f64;
        while u <= 1e-12 {
            u = self.next_f64();
        }
        let v = self.next_f64();
        let mag = (-2.0 * u.ln()).sqrt();
        let z0 = mag * (TWO_PI * v).cos();
        self.spare = Some(mag * (TWO_PI * v).sin());
        mean + std * z0
    }
}

pub fn derive_seed(base_seed: u32, salt: u32) -> u32 {
    let mut x = base_seed ^ salt.wrapping_mul(0x9E37_79B1);
    x = (x ^ (x >> 16)).wrapping_mul(0x85EB_CA6B);
    x = (x ^ (x >> 13)).wrapping_mul(0xC2B2_AE35);
    (x ^ (x >> 16)) & 0x7FFF_FFFF
}

// === Biquad Butterworth filters + FIR decimation ===

#[derive(Debug, Clone, Copy)]
struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

fn butter_coeffs(highpass: bool, normalized_cutoff: f64) -> BiquadCoeffs {
    let wn = (normalized_cutoff.clamp(0.0005, 0.99) * 1e5).round() / 1e5;
    let k = (PI * wn * 0.5).tan();
    let k2 = k * k;
    let norm = 1.0 / (1.0 + SQRT_2 * k + k2);
    let a1 = 2.0 * (k2 - 1.0) * norm;
    let a2 = (1.0 - SQRT_2 * k + k2) * norm;
    let b0 = if highpass { norm } else { k2 * norm };
    let b1 = if highpass { -2.0 * b0 } else { 2.0 * b0 };
    BiquadCoeffs {
        b0,
        b1,
        b2: b0,
        a1,
        a2,
    }
}

fn run_biquad(signal: &[f64], c: BiquadCoeffs) -> Vec<f64> {
    let mut out = vec![0.0_f64; signal.len()];
    let mut z1 = 0.0_f64;
    let mut z2 = 0.0_f64;
    for (i, &x) in signal.iter().enumerate() {
        let y = c.b0 * x + z1;
        z1 = c.b1 * x - c.a1 * y + z2;
        z2 = c.b2 * x - c.a2 * y;
        out[i] = y;
    }
    out
}

pub fn apply_lowpass(signal: &[f64], cutoff: f64, sr: u32) -> Vec<f64> {
    if signal.is_empty() {
        return Vec::new();
    }
    let wn = cutoff / ((sr as f64) * 0.5);
    run_biquad(signal, butter_coeffs(false, wn))
}

pub fn apply_highpass(signal: &[f64], cutoff: f64, sr: u32) -> Vec<f64> {
    if signal.is_empty() {
        return Vec::new();
    }
    let wn = cutoff / ((sr as f64) * 0.5);
    run_biquad(signal, butter_coeffs(true, wn))
}

fn fir_lowpass_taps(factor: usize) -> Vec<f64> {
    let num_taps = 20 * factor + 1;
    let half = ((num_taps - 1) as f64) * 0.5;
    let cutoff = 1.0 / (factor as f64);
    let mut taps = vec![0.0_f64; num_taps];
    let mut total = 0.0_f64;
    for (i, slot) in taps.iter_mut().enumerate() {
        let n = (i as f64) - half;
        let x = cutoff * n;
        let sinc = if x.abs() < 1e-12 {
            1.0
        } else {
            (PI * x).sin() / (PI * x)
        };
        let hamming = 0.54 - 0.46 * (TWO_PI * (i as f64) / ((num_taps - 1) as f64)).cos();
        let v = sinc * hamming;
        *slot = v;
        total += v;
    }
    for t in &mut taps {
        *t /= total;
    }
    taps
}

fn decimate_fir_zero_phase(signal: &[f64], factor: usize, target_len: usize) -> Vec<f64> {
    if factor <= 1 || signal.is_empty() {
        let mut out = vec![0.0_f64; target_len];
        let copy_n = signal.len().min(target_len);
        out[..copy_n].copy_from_slice(&signal[..copy_n]);
        return out;
    }
    let taps = fir_lowpass_taps(factor);
    let n = signal.len() as isize;
    let half = (taps.len() / 2) as isize;
    let s0 = signal[0];
    let s_end = signal[signal.len() - 1];
    let mut out = vec![0.0_f64; target_len];

    for (i, slot) in out.iter_mut().enumerate() {
        let center = (i * factor) as isize;
        let mut acc = 0.0_f64;
        for (k, &tap) in taps.iter().enumerate() {
            let idx = center + (k as isize) - half;
            let sample = if idx < 0 {
                let mirror = (-idx).min(n - 1) as usize;
                2.0 * s0 - signal[mirror]
            } else if idx >= n {
                let mirror = (2 * (n - 1) - idx).max(0) as usize;
                2.0 * s_end - signal[mirror]
            } else {
                signal[idx as usize]
            };
            acc += sample * tap;
        }
        *slot = acc;
    }
    out
}

// === Oscillators & primitives ===

fn sample_count(duration: f64, sr: u32) -> usize {
    if duration <= 0.0 {
        0
    } else {
        ((sr as f64) * duration) as usize
    }
}

fn poly_blep_2pt(phase: f64, dt: f64) -> f64 {
    if dt <= 0.0 || dt >= 0.5 {
        return 0.0;
    }
    if phase < dt {
        let u = phase / dt;
        u + u - u * u - 1.0
    } else if phase > 1.0 - dt {
        let u = (phase - 1.0) / dt;
        u * u + u + u + 1.0
    } else {
        0.0
    }
}

pub fn freq_from_note(root: RootNote, semitones: i32, octave: i32) -> f64 {
    let base = match root {
        RootNote::C => 261.63,
        RootNote::CSharp => 277.18,
        RootNote::D => 293.66,
        RootNote::DSharp => 311.13,
        RootNote::E => 329.63,
        RootNote::F => 349.23,
        RootNote::FSharp => 369.99,
        RootNote::G => 392.00,
        RootNote::GSharp => 415.30,
        RootNote::A => 440.00,
        RootNote::ASharp => 466.16,
        RootNote::B => 493.88,
    };
    base * 2.0_f64.powi(octave - 4) * 2.0_f64.powf((semitones as f64) / 12.0)
}

fn semitone_from_degree(mode: ModeName, degree: i32) -> i32 {
    let intervals: &[i32] = match mode {
        ModeName::Major => &[0, 2, 4, 5, 7, 9, 11],
        ModeName::Minor => &[0, 2, 3, 5, 7, 8, 10],
        ModeName::Dorian => &[0, 2, 3, 5, 7, 9, 10],
        ModeName::Mixolydian => &[0, 2, 4, 5, 7, 9, 10],
    };
    let len = intervals.len() as i32;
    let oct_shift = degree.div_euclid(len);
    let idx = degree.rem_euclid(len) as usize;
    intervals[idx] + 12 * oct_shift
}

fn scale_freq(cfg: &SynthConfig, degree: i32, octave: i32) -> f64 {
    freq_from_note(cfg.root, semitone_from_degree(cfg.mode, degree), octave)
}

pub fn generate_sine(freq: f64, duration: f64, sr: u32, amp: f64, t_offset: f64) -> Vec<f64> {
    let n = sample_count(duration, sr);
    let inv_sr = 1.0 / (sr as f64);
    let w = TWO_PI * freq;
    let mut out = vec![0.0_f64; n];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = amp * (w * (t_offset + (i as f64) * inv_sr)).sin();
    }
    out
}

pub fn generate_triangle(freq: f64, duration: f64, sr: u32, amp: f64, t_offset: f64) -> Vec<f64> {
    let n = sample_count(duration, sr);
    let mut out = vec![0.0_f64; n];
    if n == 0 || freq <= 0.0 {
        return out;
    }
    let max_h = (((sr as f64) * 0.45 / freq) as usize).min(15);
    if max_h < 1 {
        return out;
    }
    let inv_sr = 1.0 / (sr as f64);
    let w = TWO_PI * freq;
    let scale = (8.0 / (PI * PI)) * amp;
    let mut sign = 1.0_f64;
    let mut h = 1_usize;
    while h <= max_h {
        let coeff = (sign / ((h * h) as f64)) * scale;
        let hw = (h as f64) * w;
        for (i, slot) in out.iter_mut().enumerate() {
            *slot += coeff * (hw * (t_offset + (i as f64) * inv_sr)).sin();
        }
        sign = -sign;
        h += 2;
    }
    out
}

pub fn generate_sawtooth(
    freq: f64,
    duration: f64,
    sr: u32,
    amp: f64,
    t_offset: f64,
    oversample: usize,
) -> Vec<f64> {
    let n = sample_count(duration, sr);
    if n == 0 || freq <= 0.0 {
        return vec![0.0_f64; n];
    }
    let os = oversample.max(1);
    let sr_high = (sr as usize) * os;
    let dt = freq / (sr_high as f64);
    if dt >= 0.45 {
        return generate_sine(freq, duration, sr, amp, t_offset);
    }
    let n_high = n * os;
    let inv_sr = 1.0 / (sr_high as f64);
    let mut high = vec![0.0_f64; n_high];
    for (i, slot) in high.iter_mut().enumerate() {
        let t = t_offset + (i as f64) * inv_sr;
        let raw = t * freq;
        let phase = raw - raw.floor();
        *slot = 2.0 * phase - 1.0 - poly_blep_2pt(phase, dt);
    }
    let mut dec = decimate_fir_zero_phase(&high, os, n);
    for v in &mut dec {
        *v *= amp;
    }
    dec
}

pub fn generate_square(
    freq: f64,
    duration: f64,
    sr: u32,
    amp: f64,
    t_offset: f64,
    oversample: usize,
) -> Vec<f64> {
    let n = sample_count(duration, sr);
    if n == 0 || freq <= 0.0 {
        return vec![0.0_f64; n];
    }
    let os = oversample.max(1);
    let sr_high = (sr as usize) * os;
    let dt = freq / (sr_high as f64);
    if dt >= 0.45 {
        return generate_sine(freq, duration, sr, amp, t_offset);
    }
    let n_high = n * os;
    let inv_sr = 1.0 / (sr_high as f64);
    let mut high = vec![0.0_f64; n_high];
    for (i, slot) in high.iter_mut().enumerate() {
        let t = t_offset + (i as f64) * inv_sr;
        let raw = t * freq;
        let phase = raw - raw.floor();
        let shifted = if phase + 0.5 >= 1.0 {
            phase - 0.5
        } else {
            phase + 0.5
        };
        let naive = if phase < 0.5 { 1.0 } else { -1.0 };
        *slot = naive + poly_blep_2pt(phase, dt) - poly_blep_2pt(shifted, dt);
    }
    let mut dec = decimate_fir_zero_phase(&high, os, n);
    for v in &mut dec {
        *v *= amp;
    }
    dec
}

fn osc_for_grain(
    grain: GrainStyle,
    freq: f64,
    dur: f64,
    sr: u32,
    amp: f64,
    t_offset: f64,
) -> Vec<f64> {
    match grain {
        GrainStyle::Clean => generate_sine(freq, dur, sr, amp, t_offset),
        GrainStyle::Warm => generate_triangle(freq, dur, sr, amp, t_offset),
        GrainStyle::Gritty => generate_sawtooth(freq, dur, sr, amp, t_offset, 2),
    }
}

fn generate_lfo(duration: f64, rate: f64, sr: u32, t_offset: f64) -> Vec<f64> {
    let n = sample_count(duration, sr);
    let inv_sr = 1.0 / (sr as f64);
    let w = TWO_PI * rate;
    let mut out = vec![0.0_f64; n];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = 0.5 + 0.5 * (w * (t_offset + (i as f64) * inv_sr)).sin();
    }
    out
}

fn attack_mult(style: AttackStyle) -> f64 {
    match style {
        AttackStyle::Soft => 2.5,
        AttackStyle::Medium => 1.0,
        AttackStyle::Sharp => 0.3,
    }
}

// === Envelopes, humanize, limiter, spatial ===

fn write_raised_cosine(dst: &mut [f64], start: f64, end: f64) {
    let len = dst.len();
    if len == 0 {
        return;
    }
    if len == 1 {
        dst[0] = end;
        return;
    }
    let denom = (len - 1) as f64;
    let span = end - start;
    for (i, slot) in dst.iter_mut().enumerate() {
        let curve = 0.5 * (1.0 - (PI * (i as f64) / denom).cos());
        *slot = start + span * curve;
    }
}

pub fn apply_adsr(
    signal: &[f64],
    attack: f64,
    decay: f64,
    sustain: f64,
    release: f64,
    sr: u32,
) -> Vec<f64> {
    let total = signal.len();
    if total == 0 {
        return Vec::new();
    }
    if total == 1 {
        return vec![0.0];
    }

    let sus = sustain.clamp(0.0, 1.0);
    let sr_f = sr as f64;
    let min_a = ((0.005 * sr_f).round() as usize).max(1).min((total / 4).max(1));
    let min_r = ((0.018 * sr_f).round() as usize).max(1).min((total / 3).max(1));

    let a_req = ((attack.max(0.005) * sr_f).round() as usize).max(min_a);
    let d_req = (decay.max(0.0) * sr_f).round() as usize;
    let r_req = ((release.max(0.018) * sr_f).round() as usize).max(min_r);

    let stage_sum = a_req + d_req + r_req;
    let (a_len, d_len, s_len) = if stage_sum <= total {
        (a_req, d_req, total - stage_sum)
    } else {
        let scale = (total as f64) / (stage_sum as f64);
        let r_len = (((r_req as f64) * scale).round() as usize)
            .max(min_r)
            .min(total.saturating_sub(min_a).max(1));
        let rem = total - r_len;
        let ad_sum = (a_req + d_req).max(1) as f64;
        let a_len = (((rem as f64) * ((a_req as f64) / ad_sum)).round() as usize)
            .max(min_a)
            .min(rem);
        (a_len, rem - a_len, 0)
    };

    let mut env = vec![0.0_f64; total];
    let mut pos = 0_usize;
    let a_end = (pos + a_len).min(total);
    write_raised_cosine(&mut env[pos..a_end], 0.0, 1.0);
    pos = a_end;
    if d_len > 0 && pos < total {
        let d_end = (pos + d_len).min(total);
        write_raised_cosine(&mut env[pos..d_end], 1.0, sus);
        pos = d_end;
    }
    if s_len > 0 && pos < total {
        let s_end = (pos + s_len).min(total);
        env[pos..s_end].fill(sus);
        pos = s_end;
    }
    let rel_start = if d_len > 0 || s_len > 0 { sus } else { 1.0 };
    if pos < total {
        write_raised_cosine(&mut env[pos..total], rel_start, 0.0);
    }
    env[0] = 0.0;
    env[total - 1] = 0.0;

    signal.iter().zip(env.iter()).map(|(&s, &e)| s * e).collect()
}

pub fn apply_humanize(signal: &[f64], amount: f64, sr: u32, rng: &mut SeededRng) -> Vec<f64> {
    let n = signal.len();
    if amount <= 0.0 || n < 4 {
        return signal.to_vec();
    }
    let depth = amount.clamp(0.0, 1.0) * 0.075;
    let num_knots = (((n as f64) / (sr as f64) * 1.4).round() as usize + 2).max(3);
    let mut knots = vec![0.0_f64; num_knots];
    for k in &mut knots {
        *k = rng.normal(0.0, depth).clamp(-0.09, 0.09);
    }

    let scale = ((num_knots - 1) as f64) / ((n - 1).max(1) as f64);
    let mut out = vec![0.0_f64; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let pos = (i as f64) * scale;
        let idx0 = (pos.floor() as usize).min(num_knots - 2);
        let frac = pos - (idx0 as f64);
        let smooth = 0.5 * (1.0 - (PI * frac).cos());
        let drift = knots[idx0] * (1.0 - smooth) + knots[idx0 + 1] * smooth;
        *slot = signal[i] * (1.0 + drift);
    }
    out
}

pub fn soft_limit(signal: &[f64], ceiling: f64) -> Vec<f64> {
    if signal.is_empty() || ceiling <= 0.0 {
        return signal.to_vec();
    }
    let knee = ceiling * 0.78;
    let max_abs = signal.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    if max_abs <= knee {
        return signal.to_vec();
    }
    let span = (ceiling - knee).max(1e-6);
    signal
        .iter()
        .map(|&x| {
            let a = x.abs();
            if a <= knee {
                x
            } else {
                let comp = knee + span * ((a - knee) / span).tanh();
                if x >= 0.0 { comp } else { -comp }
            }
        })
        .collect()
}

pub fn soft_fade_edges(signal: &[f64], fade_in_sec: f64, fade_out_sec: f64, sr: u32) -> Vec<f64> {
    let n = signal.len();
    let mut out = signal.to_vec();
    if n < 2 {
        return out;
    }
    let in_n = ((fade_in_sec * (sr as f64)).round() as usize).min(n / 2);
    let out_n = ((fade_out_sec * (sr as f64)).round() as usize).min(n - in_n);
    if in_n >= 2 {
        let denom = (in_n - 1) as f64;
        for (i, slot) in out.iter_mut().take(in_n).enumerate() {
            *slot *= 0.5 * (1.0 - (PI * (i as f64) / denom).cos());
        }
    }
    if out_n >= 2 {
        let start = n - out_n;
        let denom = (out_n - 1) as f64;
        for i in 0..out_n {
            out[start + i] *= 0.5 * (1.0 + (PI * (i as f64) / denom).cos());
        }
    }
    out
}

/// Apply a smooth cubic `(1 - u)^3` fade-out ending at exact zero.
pub fn fade_into_nothingness(signal: &[f64], fade_sec: f64, sr: u32) -> Vec<f64> {
    let n = signal.len();
    let mut out = signal.to_vec();
    if n < 2 || fade_sec <= 0.0 {
        return out;
    }
    let fade_n = ((fade_sec * (sr as f64)).round() as usize).max(2).min(n);
    let start = n - fade_n;
    let denom = (fade_n - 1).max(1) as f64;
    for i in 0..fade_n {
        let u = (i as f64) / denom;
        let g = (1.0 - u).powi(3);
        out[start + i] *= g;
    }
    out[n - 1] = 0.0;
    out
}

fn apply_delay(signal: &[f64], delay_time: f64, feedback: f64, wet: f64, sr: u32) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || wet <= 0.0 || delay_time <= 0.0 {
        return signal.to_vec();
    }
    let safe_fb = feedback.clamp(0.0, 0.72);
    let safe_wet = wet.clamp(0.0, 0.55);
    let delay_samples = (delay_time * (sr as f64)).round() as usize;
    if delay_samples == 0 || delay_samples >= n {
        return signal.to_vec();
    }

    let tap_source = apply_lowpass(signal, 2800.0, sr);
    let mut out = signal.to_vec();
    for tap in 1..=5 {
        let off = delay_samples * tap;
        if off >= n {
            break;
        }
        let gain = safe_fb.powi(tap as i32) * safe_wet;
        if gain < 0.005 {
            break;
        }
        for i in off..n {
            out[i] += tap_source[i - off] * gain;
        }
    }
    out
}

fn apply_reverb(signal: &[f64], room: f64, size: f64, sr: u32) -> Vec<f64> {
    let n = signal.len();
    let safe_room = room.clamp(0.0, 1.0);
    let safe_size = size.clamp(0.0, 1.0);
    if n == 0 || safe_room <= 0.01 || safe_size <= 0.01 {
        return signal.to_vec();
    }

    let mut diffused = signal.to_vec();
    for &(d_sec, d_gain) in DIFFUSION_TAPS {
        let off = (d_sec * (sr as f64)).round() as usize;
        if off > 0 && off < n {
            for i in off..n {
                diffused[i] += signal[i - off] * d_gain;
            }
        }
    }

    let size_scale = 0.45 + 0.90 * safe_size;
    let mut wet = vec![0.0_f64; n];
    for &(t_sec, t_gain) in REVERB_TAPS {
        let off = (t_sec * size_scale * (sr as f64)).round() as usize;
        if off > 0 && off < n {
            for i in off..n {
                wet[i] += diffused[i - off] * t_gain;
            }
        }
    }

    let damped = apply_lowpass(&wet, 4200.0 - 2000.0 * safe_size, sr);
    let wet_mix = safe_room * 0.48;
    let dry_mix = 1.0 - safe_room * 0.16;
    signal
        .iter()
        .zip(damped.iter())
        .map(|(&s, &w)| s * dry_mix + w * wet_mix)
        .collect()
}

fn apply_stereo_width(signal: &[f64], width: f64, sr: u32) -> Vec<f64> {
    let safe_width = width.clamp(0.0, 1.0);
    let n = signal.len();
    if safe_width <= 0.0 || n < 16 {
        return signal.to_vec();
    }
    let air = apply_highpass(signal, 320.0, sr);
    let mut out = signal.to_vec();
    for &(d_sec, gain) in &[(0.0097, 0.055 * safe_width), (0.0153, -0.042 * safe_width)] {
        let off = (d_sec * (sr as f64)).round() as usize;
        if off > 0 && off < n {
            for i in off..n {
                out[i] += air[i - off] * gain;
            }
        }
    }
    out
}

fn add_note(signal: &mut [f64], note: &[f64], start_idx: usize) {
    if start_idx >= signal.len() || note.is_empty() {
        return;
    }
    let end_idx = start_idx + note.len();
    if end_idx <= signal.len() {
        for (dst, &src) in signal[start_idx..end_idx].iter_mut().zip(note.iter()) {
            *dst += src;
        }
        return;
    }
    let available = signal.len() - start_idx;
    let fade_n = (((SAMPLE_RATE as f64) * 0.025).round() as usize).min((available / 3).max(2));
    let fade_start = available.saturating_sub(fade_n);
    let denom = (fade_n.saturating_sub(1)).max(1) as f64;
    for i in 0..available {
        let mut s = note[i];
        if fade_n >= 2 && i >= fade_start {
            let k = i - fade_start;
            s *= 0.5 * (1.0 + (PI * (k as f64) / denom).cos());
        }
        signal[start_idx + i] += s;
    }
}

// === Radix-2 FFT + peak-calibrated master hall reverb ===

fn fft_radix2(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let mut j = 0_usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2_usize;
    while len <= n {
        let half = len >> 1;
        let ang = sign * TWO_PI / (len as f64);
        let w_cos = ang.cos();
        let w_sin = ang.sin();
        let mut i = 0;
        while i < n {
            let mut cur_re = 1.0_f64;
            let mut cur_im = 0.0_f64;
            for k in 0..half {
                let u = i + k;
                let v = u + half;
                let u_re = re[u];
                let u_im = im[u];
                let v_re = re[v] * cur_re - im[v] * cur_im;
                let v_im = re[v] * cur_im + im[v] * cur_re;
                re[u] = u_re + v_re;
                im[u] = u_im + v_im;
                re[v] = u_re - v_re;
                im[v] = u_im - v_im;
                let next_re = cur_re * w_cos - cur_im * w_sin;
                cur_im = cur_re * w_sin + cur_im * w_cos;
                cur_re = next_re;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let inv_n = 1.0 / (n as f64);
        for i in 0..n {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
}

type HallIrCache = Mutex<HashMap<(u32, u32), Vec<f64>>>;

fn master_reverb_ir(bucket: u32, sr: u32) -> Vec<f64> {
    static CACHE: OnceLock<HallIrCache> = OnceLock::new();
    let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = map.lock() {
        if let Some(hit) = guard.get(&(bucket, sr)) {
            return hit.clone();
        }
    }

    let size = (bucket as f64) / 10.0;
    let rt60 = 0.9 + 1.8 * size;
    let ir_len = ((sr as f64) * (rt60 * 0.70).min(1.50)) as usize;
    let mut rng = SeededRng::new(2026_1125 + bucket);
    let mut raw = vec![0.0_f64; ir_len];
    for (i, slot) in raw.iter_mut().enumerate() {
        let t = (i as f64) / (sr as f64);
        *slot = rng.normal(0.0, 1.0) * 10.0_f64.powf(-3.0 * t / rt60);
    }

    let predelay = (0.022 * (sr as f64)) as usize;
    let onset = (0.030 * (sr as f64)) as usize;
    let mut ir = vec![0.0_f64; ir_len];
    if predelay < ir_len {
        let body_len = ir_len - predelay;
        let rise = body_len.min(onset);
        let denom = rise.saturating_sub(1).max(1) as f64;
        for i in 0..body_len {
            let mut s = raw[i];
            if rise >= 2 && i < rise {
                s *= 0.5 * (1.0 - (PI * (i as f64) / denom).cos());
            }
            ir[predelay + i] = s;
        }
    }
    let tail_n = (ir_len / 4).min((0.18 * (sr as f64)) as usize);
    if tail_n >= 2 {
        let start = ir_len - tail_n;
        let denom = (tail_n - 1) as f64;
        for i in 0..tail_n {
            ir[start + i] *= 0.5 * (1.0 + (PI * (i as f64) / denom).cos());
        }
    }

    let ir = apply_highpass(&ir, 180.0, sr);
    let mut ir = apply_lowpass(&ir, 3600.0 - 1500.0 * size, sr);

    let fft_len = (ir_len * 2).max(256).next_power_of_two();
    let mut re = vec![0.0_f64; fft_len];
    let mut im = vec![0.0_f64; fft_len];
    re[..ir_len].copy_from_slice(&ir);
    fft_radix2(&mut re, &mut im, false);
    let half = fft_len / 2 + 1;
    let mut sum_sq = 0.0_f64;
    let mut peak_resp = 0.0_f64;
    for i in 0..half {
        let mag = re[i].hypot(im[i]);
        sum_sq += mag * mag;
        if mag > peak_resp {
            peak_resp = mag;
        }
    }
    let rms_resp = (sum_sq / (half as f64)).sqrt();
    let norm = rms_resp.max(peak_resp * 0.55) + 1e-12;
    for v in &mut ir {
        *v /= norm;
    }

    if let Ok(mut guard) = map.lock() {
        guard.insert((bucket, sr), ir.clone());
    }
    ir
}

pub fn apply_master_hall_reverb(signal: &[f64], space: f64, sr: u32) -> Vec<f64> {
    let safe_space = space.clamp(0.0, 1.0);
    let n = signal.len();
    if n < 64 || safe_space <= 0.12 {
        return signal.to_vec();
    }
    let bucket = ((safe_space * 10.0).round() as u32).clamp(1, 10);
    let ir = master_reverb_ir(bucket, sr);
    let fft_n = (n + ir.len() - 1).next_power_of_two();

    let mut sig_re = vec![0.0_f64; fft_n];
    let mut sig_im = vec![0.0_f64; fft_n];
    sig_re[..n].copy_from_slice(signal);
    fft_radix2(&mut sig_re, &mut sig_im, false);

    let mut ir_re = vec![0.0_f64; fft_n];
    let mut ir_im = vec![0.0_f64; fft_n];
    ir_re[..ir.len()].copy_from_slice(&ir);
    fft_radix2(&mut ir_re, &mut ir_im, false);

    for i in 0..fft_n {
        let sr_re = sig_re[i];
        let sr_im = sig_im[i];
        let h_re = ir_re[i];
        let h_im = ir_im[i];
        sig_re[i] = sr_re * h_re - sr_im * h_im;
        sig_im[i] = sr_re * h_im + sr_im * h_re;
    }
    fft_radix2(&mut sig_re, &mut sig_im, true);

    let wet_gain = 0.08 + 0.22 * safe_space;
    let dry_gain = 1.0 - 0.14 * safe_space;
    (0..n)
        .map(|i| signal[i] * dry_gain + sig_re[i] * wet_gain)
        .collect()
}

// === Harmony + Pattern Generators ===

#[derive(Debug, Clone, Copy)]
struct ChordEvent {
    start_beat: f64,
    duration_beats: f64,
    root_degree: i32,
}

#[derive(Debug, Clone)]
struct HarmonyPlan {
    chords: Vec<ChordEvent>,
}

fn seconds_per_beat(cfg: &SynthConfig) -> f64 {
    60.0 / bpm_from_tempo(cfg.tempo)
}

fn select_harmony_template(cfg: &SynthConfig, rng: &mut SeededRng) -> &'static [i32] {
    let style = match cfg.harmony_style {
        HarmonyStyle::Auto => {
            if cfg.rhythm == RhythmStyle::None && cfg.space >= 0.7 {
                HarmonyStyle::Ambient
            } else if cfg.pad == PadStyle::Cinematic
                || matches!(cfg.bass, BassStyle::Drone | BassStyle::SubPulse)
            {
                HarmonyStyle::Cinematic
            } else if cfg.texture == TextureStyle::VinylCrackle
                || cfg.melody == MelodyStyle::Ornamental
            {
                HarmonyStyle::Jazz
            } else {
                HarmonyStyle::Pop
            }
        }
        explicit => explicit,
    };
    let majorish = matches!(cfg.mode, ModeName::Major | ModeName::Mixolydian);
    let templates: &[&[i32]] = match style {
        HarmonyStyle::Ambient => &[&[0, 5, 0, 3], &[0, 4, 0, 6], &[0, 3, 0, 5]],
        HarmonyStyle::Jazz if majorish => &[&[1, 4, 0, 0], &[1, 4, 0, 5], &[5, 1, 4, 0]],
        HarmonyStyle::Jazz => &[&[1, 4, 0, 0], &[5, 1, 4, 0]],
        HarmonyStyle::Cinematic if !majorish => &[&[0, 6, 5, 3], &[0, 5, 6, 4], &[0, 6, 3, 4]],
        _ if majorish => &[&[0, 5, 3, 4], &[0, 3, 4, 0], &[0, 4, 5, 3]],
        _ => &[&[0, 5, 3, 6], &[0, 3, 6, 4], &[0, 6, 3, 5], &[0, 4, 5, 3]],
    };
    templates[rng.int_range(0, templates.len())]
}

fn build_harmony_plan(
    cfg: &SynthConfig,
    duration: f64,
    rng: &mut SeededRng,
    t_offset: f64,
) -> HarmonyPlan {
    let spb = seconds_per_beat(cfg);
    let chord_beats = ((cfg.chord_change_bars.max(1) as usize) * 4) as f64;
    let template = select_harmony_template(cfg, rng);
    let start_g = (t_offset / spb).max(0.0);
    let end_g = ((t_offset + duration) / spb).max(start_g + 1e-4);
    let first_idx = (start_g / chord_beats).floor() as i64;
    let last_idx = ((end_g / chord_beats).ceil() as i64).max(first_idx + 1);

    let mut chords = Vec::new();
    for idx in first_idx..last_idx {
        let seg_start = start_g.max((idx as f64) * chord_beats);
        let seg_end = end_g.min(((idx + 1) as f64) * chord_beats);
        let seg_dur = seg_end - seg_start;
        if seg_dur <= 1e-6 {
            continue;
        }
        let t_idx = idx.rem_euclid(template.len() as i64) as usize;
        chords.push(ChordEvent {
            start_beat: seg_start - start_g,
            duration_beats: seg_dur,
            root_degree: template[t_idx].rem_euclid(7),
        });
    }
    HarmonyPlan { chords }
}

fn chord_root_at_beat(beat: f64, harmony: &HarmonyPlan) -> i32 {
    if harmony.chords.is_empty() {
        return 0;
    }
    for ch in &harmony.chords {
        if beat >= ch.start_beat && beat < ch.start_beat + ch.duration_beats {
            return ch.root_degree;
        }
    }
    if beat < harmony.chords[0].start_beat {
        harmony.chords[0].root_degree
    } else {
        harmony.chords[harmony.chords.len() - 1].root_degree
    }
}

fn chord_segments(cfg: &SynthConfig, duration: f64, harmony: &HarmonyPlan) -> Vec<(f64, f64, i32)> {
    if harmony.chords.is_empty() {
        return vec![(0.0, duration, 0)];
    }
    let spb = seconds_per_beat(cfg);
    let mut out = Vec::new();
    for ch in &harmony.chords {
        let start = ch.start_beat * spb;
        if start >= duration {
            break;
        }
        let end = (start + ch.duration_beats * spb).min(duration);
        let dur = (end - start).max(0.0);
        if dur > 0.0 {
            out.push((start, dur, ch.root_degree));
        }
    }
    out
}

fn chord_tones_ascending(ext: ChordExtensions, root: i32) -> Vec<i32> {
    let mut tones = vec![root, root + 2, root + 4];
    if matches!(ext, ChordExtensions::Sevenths | ChordExtensions::Lush) {
        tones.push(root + 6);
    }
    if ext == ChordExtensions::Lush {
        tones.push(root + 8);
    }
    tones
}

fn chord_tone_classes(root: i32, ext: ChordExtensions) -> Vec<i32> {
    let r = root.rem_euclid(7);
    let mut raw = vec![r, (r + 2) % 7, (r + 4) % 7];
    if matches!(ext, ChordExtensions::Sevenths | ChordExtensions::Lush) {
        raw.push((r + 6) % 7);
    }
    if ext == ChordExtensions::Lush {
        raw.push((r + 1) % 7);
    }
    let mut uniq = Vec::new();
    for t in raw {
        if !uniq.contains(&t) {
            uniq.push(t);
        }
    }
    uniq
}

fn first_grid_slot(t_offset: f64, slot_sec: f64) -> i64 {
    let slot = (t_offset / slot_sec).floor() as i64;
    if (slot as f64) * slot_sec < t_offset - 1e-9 {
        slot + 1
    } else {
        slot
    }
}

fn render_soft_kick(
    base_freq: f64,
    kick_dur: f64,
    amp: f64,
    decay_rate: f64,
    lowpass_hz: f64,
    sr: u32,
) -> Vec<f64> {
    let n = sample_count(kick_dur, sr);
    if n < 4 {
        return vec![0.0_f64; n];
    }
    let inv_sr = 1.0 / (sr as f64);
    let mut phase = 0.0_f64;
    let mut kick = vec![0.0_f64; n];
    for (i, slot) in kick.iter_mut().enumerate() {
        let t = (i as f64) * inv_sr;
        let f_inst = base_freq * (1.0 + 0.75 * (-t * 32.0).exp());
        phase += TWO_PI * f_inst * inv_sr;
        let env = (-decay_rate * (i as f64) / ((n - 1).max(1) as f64)).exp();
        *slot = amp * phase.sin() * env;
    }
    let atk = ((0.004 * (sr as f64)).round() as usize).max(2).min(n / 4);
    let rel = ((0.015 * (sr as f64)).round() as usize).max(2).min(n / 3);
    for (i, sample) in kick.iter_mut().enumerate().take(atk) {
        *sample *= 0.5 * (1.0 - (PI * (i as f64) / ((atk - 1) as f64)).cos());
    }
    for i in 0..rel {
        kick[n - rel + i] *= 0.5 * (1.0 + (PI * (i as f64) / ((rel - 1) as f64)).cos());
    }
    let mut filtered = apply_lowpass(&kick, lowpass_hz, sr);
    filtered[0] = 0.0;
    filtered[n - 1] = 0.0;
    filtered
}

fn render_bass_sub_pulse(
    cfg: &SynthConfig,
    duration: f64,
    harmony: &HarmonyPlan,
    t_offset: f64,
) -> Vec<f64> {
    let sr = SAMPLE_RATE;
    let atk = attack_mult(cfg.attack);
    let mut sig = vec![0.0_f64; sample_count(duration, sr)];
    let spb = seconds_per_beat(cfg);
    let pulse_sec = 2.0 * spb;
    let mut slot = first_grid_slot(t_offset, pulse_sec);
    loop {
        let local_sec = (slot as f64) * pulse_sec - t_offset;
        if local_sec >= duration {
            break;
        }
        let root = chord_root_at_beat(local_sec / spb, harmony);
        let note_dur = (pulse_sec * 0.95).min((duration - local_sec).max(0.06));
        let note = generate_sine(scale_freq(cfg, root, 1), note_dur, sr, 0.4, t_offset + local_sec);
        let note = apply_lowpass(&note, 55.0, sr);
        let note = apply_adsr(&note, 0.25 * atk, 0.2, 0.9, 0.7 * atk, sr);
        add_note(&mut sig, &note, (local_sec * (sr as f64)).round() as usize);
        slot += 1;
    }
    sig
}

fn render_bass(
    cfg: &SynthConfig,
    duration: f64,
    harmony: &HarmonyPlan,
    rng: &mut SeededRng,
    t_offset: f64,
) -> Vec<f64> {
    let sr = SAMPLE_RATE;
    let atk = attack_mult(cfg.attack);
    let mut sig = vec![0.0_f64; sample_count(duration, sr)];
    match cfg.bass {
        BassStyle::SubPulse => render_bass_sub_pulse(cfg, duration, harmony, t_offset),
        BassStyle::Pulsing | BassStyle::ArpBass => {
            let spb = seconds_per_beat(cfg);
            let pulse_sec = 0.5 * spb;
            let mut slot = first_grid_slot(t_offset, pulse_sec);
            loop {
                let local_sec = (slot as f64) * pulse_sec - t_offset;
                if local_sec >= duration {
                    break;
                }
                let root = chord_root_at_beat(local_sec / spb, harmony);
                let note_dur = (pulse_sec * 0.82).min((duration - local_sec).max(0.05));
                let note =
                    generate_sine(scale_freq(cfg, root, 2), note_dur, sr, 0.35, t_offset + local_sec);
                let note = apply_lowpass(&note, 90.0 * cfg.brightness + 20.0, sr);
                let note = apply_adsr(&note, 0.02 * atk, 0.08, 0.6, 0.25 * atk, sr);
                add_note(&mut sig, &note, (local_sec * (sr as f64)).round() as usize);
                slot += 1;
            }
            sig
        }
        BassStyle::Walking => {
            let spb = seconds_per_beat(cfg);
            let walk = [0, 2, 4, 2];
            let mut slot = first_grid_slot(t_offset, spb);
            loop {
                let local_sec = (slot as f64) * spb - t_offset;
                if local_sec >= duration {
                    break;
                }
                let root = chord_root_at_beat(local_sec / spb, harmony);
                let walk_deg = walk[slot.rem_euclid(4) as usize];
                let note_dur = (spb * 0.92).min((duration - local_sec).max(0.06));
                let note = generate_triangle(
                    scale_freq(cfg, root + walk_deg, 2),
                    note_dur,
                    sr,
                    0.30,
                    t_offset + local_sec,
                );
                let note = apply_lowpass(&note, 120.0 * cfg.brightness + 40.0, sr);
                let note = apply_adsr(&note, 0.04 * atk, 0.12, 0.7, 0.22 * atk, sr);
                let note = apply_humanize(&note, cfg.human, sr, rng);
                add_note(&mut sig, &note, (local_sec * (sr as f64)).round() as usize);
                slot += 1;
            }
            apply_reverb(&sig, cfg.space * 0.3, cfg.space * 0.8, sr)
        }
        BassStyle::Sustained | BassStyle::Octave => {
            let pat = [0, 0, 4, 0];
            for (start_seg, seg_dur, root) in chord_segments(cfg, duration, harmony) {
                let note_dur = seg_dur / 4.0;
                for (i, &rel) in pat.iter().enumerate() {
                    let start = start_seg + (i as f64) * note_dur;
                    let note = generate_sine(scale_freq(cfg, root + rel, 2), note_dur * 0.95, sr, 0.32, t_offset + start);
                    let note = apply_lowpass(&note, 100.0 * cfg.brightness + 30.0, sr);
                    let note = apply_adsr(&note, 0.7 * atk, 0.3, 0.85, 1.2 * atk, sr);
                    add_note(&mut sig, &note, (start * (sr as f64)).round() as usize);
                }
            }
            apply_reverb(&sig, cfg.space * 0.4, cfg.space, sr)
        }
        BassStyle::FifthDrone => {
            for (start_seg, seg_dur, root) in chord_segments(cfg, duration, harmony) {
                let note_dur = (seg_dur * 1.15).min(duration - start_seg);
                let r = generate_sine(scale_freq(cfg, root, 2), note_dur, sr, 0.26, t_offset + start_seg);
                let r = apply_adsr(&apply_lowpass(&r, 70.0 * cfg.brightness + 20.0, sr), 2.2 * atk, 0.5, 0.95, 2.6 * atk, sr);
                let f = generate_sine(scale_freq(cfg, root + 4, 2), note_dur, sr, 0.18, t_offset + start_seg);
                let f = apply_adsr(&apply_lowpass(&f, 100.0 * cfg.brightness + 30.0, sr), 2.4 * atk, 0.5, 0.9, 2.6 * atk, sr);
                let sum: Vec<f64> = r.iter().zip(f.iter()).map(|(&a, &b)| a + b).collect();
                add_note(&mut sig, &sum, (start_seg * (sr as f64)).round() as usize);
            }
            apply_reverb(&sig, cfg.space * 0.5, cfg.space, sr)
        }
        BassStyle::Drone => {
            for (start_seg, seg_dur, root) in chord_segments(cfg, duration, harmony) {
                let note_dur = (seg_dur * 1.15).min(duration - start_seg);
                let note = generate_sine(scale_freq(cfg, root, 2), note_dur, sr, 0.35, t_offset + start_seg);
                let note = apply_lowpass(&note, 80.0 * cfg.brightness + 20.0, sr);
                let note = apply_adsr(&note, 1.8 * atk, 0.5, 0.95, 2.2 * atk, sr);
                add_note(&mut sig, &note, (start_seg * (sr as f64)).round() as usize);
            }
            apply_reverb(&sig, cfg.space * 0.5, cfg.space, sr)
        }
    }
}

fn render_pad(
    cfg: &SynthConfig,
    duration: f64,
    harmony: &HarmonyPlan,
    rng: &mut SeededRng,
    t_offset: f64,
) -> Vec<f64> {
    let sr = SAMPLE_RATE;
    let atk = attack_mult(cfg.attack);
    let echo_mult = cfg.echo / 0.5;
    let mut sig = vec![0.0_f64; sample_count(duration, sr)];

    match cfg.pad {
        PadStyle::DarkSustained => {
            for (start, seg_dur, root) in chord_segments(cfg, duration, harmony) {
                let dur = (seg_dur * 1.15).min(duration - start);
                for deg in chord_tones_ascending(cfg.chord_extensions, root).into_iter().take(3) {
                    let t = generate_sawtooth(scale_freq(cfg, deg, 3), dur, sr, 0.12, t_offset + start, 2);
                    let t = apply_lowpass(&t, 200.0 * cfg.brightness + 80.0, sr);
                    let t = apply_adsr(&t, 2.0 * atk, 1.0, 0.9, 3.0 * atk, sr);
                    add_note(&mut sig, &t, (start * (sr as f64)).round() as usize);
                }
            }
            apply_reverb(&sig, cfg.space * 0.8, cfg.space, sr)
        }
        PadStyle::Cinematic => {
            let voicings = [(0, 3), (2, 3), (4, 3), (0, 4), (4, 4)];
            for (start, seg_dur, root) in chord_segments(cfg, duration, harmony) {
                let dur = (seg_dur * 1.10).min(duration - start);
                for &(rel, oct) in &voicings {
                    let freq = scale_freq(cfg, root + rel, oct);
                    let saw = generate_sawtooth(freq, dur, sr, 0.08, t_offset + start, 2);
                    let tri = generate_triangle(freq * 1.002, dur, sr, 0.06, t_offset + start);
                    let mix: Vec<f64> = saw.iter().zip(tri.iter()).map(|(&a, &b)| a + b).collect();
                    let t = apply_lowpass(&mix, 400.0 * cfg.brightness + 150.0, sr);
                    let t = apply_adsr(&t, 1.8 * atk, 0.8, 0.88, 2.8 * atk, sr);
                    add_note(&mut sig, &t, (start * (sr as f64)).round() as usize);
                }
            }
            let rev = apply_reverb(&sig, cfg.space * 0.85, cfg.space, sr);
            apply_delay(&rev, 0.4, 0.35 * echo_mult, 0.3 * echo_mult, sr)
        }
        PadStyle::ThinHigh | PadStyle::BrightOpen => {
            for (start, seg_dur, root) in chord_segments(cfg, duration, harmony) {
                let dur = (seg_dur * 1.15).min(duration - start);
                for &rel in &[0, 4] {
                    let t = generate_sine(scale_freq(cfg, root + rel, 4), dur, sr, 0.12, t_offset + start);
                    let t = apply_highpass(&apply_lowpass(&t, 800.0 * cfg.brightness + 200.0, sr), 200.0, sr);
                    let t = apply_adsr(&t, 1.2 * atk, 0.6, 0.8, 2.0 * atk, sr);
                    add_note(&mut sig, &t, (start * (sr as f64)).round() as usize);
                }
            }
            apply_reverb(&sig, cfg.space * 0.75, cfg.space * 0.95, sr)
        }
        PadStyle::AmbientDrift => {
            for (start, seg_dur, root) in chord_segments(cfg, duration, harmony) {
                let dur = (seg_dur * 1.35).min(duration - start);
                let mut tones: Vec<i32> = chord_tones_ascending(cfg.chord_extensions, root)
                    .into_iter()
                    .take(3)
                    .collect();
                if cfg.motion > 0.4 && rng.next_f64() < 0.15 + 0.25 * cfg.motion {
                    tones.push(root + 5);
                }
                for deg in tones {
                    let t = generate_sine(scale_freq(cfg, deg, 3), dur, sr, 0.14, t_offset + start);
                    let t = apply_lowpass(&t, 350.0 * cfg.brightness + 100.0, sr);
                    let t = apply_adsr(&t, 1.6 * atk, 0.5, 0.85, 2.2 * atk, sr);
                    add_note(&mut sig, &t, (start * (sr as f64)).round() as usize);
                }
            }
            let rev = apply_reverb(&sig, cfg.space * 0.8, cfg.space, sr);
            apply_delay(&rev, 0.5, 0.4 * echo_mult, 0.35 * echo_mult, sr)
        }
        PadStyle::StackedFifths => {
            let voicings = [(0, 3), (4, 3), (0, 4), (4, 4)];
            for (start, seg_dur, root) in chord_segments(cfg, duration, harmony) {
                let dur = (seg_dur * 1.10).min(duration - start);
                for &(rel, oct) in &voicings {
                    let t = osc_for_grain(cfg.grain, scale_freq(cfg, root + rel, oct), dur, sr, 0.10, t_offset + start);
                    let t = apply_lowpass(&t, 500.0 * cfg.brightness + 150.0, sr);
                    let t = apply_adsr(&t, 1.3 * atk, 0.7, 0.88, 2.2 * atk, sr);
                    add_note(&mut sig, &t, (start * (sr as f64)).round() as usize);
                }
            }
            apply_reverb(&sig, cfg.space * 0.7, cfg.space * 0.85, sr)
        }
        PadStyle::WarmSlow => {
            let lfo_rate = 0.06 + 0.28 * cfg.motion.clamp(0.0, 1.0);
            for (start, seg_dur, root) in chord_segments(cfg, duration, harmony) {
                let dur = (seg_dur * 1.15).min(duration - start);
                for deg in chord_tones_ascending(cfg.chord_extensions, root).into_iter().take(3) {
                    let raw = osc_for_grain(cfg.grain, scale_freq(cfg, deg, 3), dur, sr, 0.15, t_offset + start);
                    let lfo = generate_lfo(dur, lfo_rate, sr, t_offset + start);
                    let base = 300.0 * cfg.brightness + 100.0;
                    let lo = apply_lowpass(&raw, base * 0.5, sr);
                    let hi = apply_lowpass(&raw, base * 1.5, sr);
                    let blend: Vec<f64> = (0..raw.len())
                        .map(|i| lo[i] * (1.0 - lfo[i]) + hi[i] * lfo[i])
                        .collect();
                    let env = apply_adsr(&blend, 1.2 * atk, 0.6, 0.85, 1.6 * atk, sr);
                    add_note(&mut sig, &env, (start * (sr as f64)).round() as usize);
                }
            }
            let rev = apply_reverb(&sig, cfg.space * 0.7, cfg.space * 0.9, sr);
            let del = apply_delay(&rev, 0.35, 0.30 * echo_mult, 0.25 * echo_mult, sr);
            apply_humanize(&del, cfg.human, sr, rng)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_bpm_pattern_melody(
    cfg: &SynthConfig,
    duration: f64,
    harmony: &HarmonyPlan,
    pattern: &[i32],
    step_beats: f64,
    octave: i32,
    amp: f64,
    dur_mult: f64,
    cutoff: f64,
    adsr: (f64, f64, f64, f64),
    delay_cfg: (f64, f64, f64),
    reverb_scale: (f64, f64),
    rng: &mut SeededRng,
    t_offset: f64,
) -> Vec<f64> {
    let sr = SAMPLE_RATE;
    let mut sig = vec![0.0_f64; sample_count(duration, sr)];
    let atk = attack_mult(cfg.attack);
    let echo_mult = cfg.echo / 0.5;
    let spb = seconds_per_beat(cfg);
    let slot_sec = step_beats * spb;
    if slot_sec <= 0.0 || pattern.is_empty() {
        return sig;
    }

    let mut slot_idx = first_grid_slot(t_offset, slot_sec);
    loop {
        let local_sec = (slot_idx as f64) * slot_sec - t_offset;
        if local_sec >= duration {
            break;
        }
        let pat_idx = slot_idx.rem_euclid(pattern.len() as i64) as usize;
        let rel = pattern[pat_idx];
        if rel >= 0 {
            let root = chord_root_at_beat(local_sec / spb, harmony);
            let freq = scale_freq(cfg, root + rel, octave);
            let note_dur = (slot_sec * dur_mult).min((duration - local_sec).max(0.06));
            let note = osc_for_grain(cfg.grain, freq, note_dur, sr, amp, t_offset + local_sec);
            let note = apply_lowpass(&note, cutoff, sr);
            let note = apply_adsr(&note, adsr.0 * atk, adsr.1, adsr.2, adsr.3 * atk, sr);
            let note = apply_humanize(&note, cfg.human, sr, rng);
            add_note(&mut sig, &note, (local_sec * (sr as f64)).round() as usize);
        }
        slot_idx += 1;
    }

    let del = apply_delay(&sig, delay_cfg.0, delay_cfg.1 * echo_mult, delay_cfg.2 * echo_mult, sr);
    apply_reverb(&del, cfg.space * reverb_scale.0, cfg.space * reverb_scale.1, sr)
}

fn tension_value(curve: TensionCurve, x_raw: f64) -> f64 {
    let x = x_raw.clamp(0.0, 1.0);
    match curve {
        TensionCurve::Ramp => x,
        TensionCurve::Waves => 0.5 - 0.5 * (TWO_PI * 2.0 * x).cos(),
        TensionCurve::Arc => (PI * x).sin(),
    }
}

fn weighted_choice(rng: &mut SeededRng, items: &[i32], weights: &[f64]) -> i32 {
    let sum: f64 = weights.iter().map(|&w| w.max(0.0)).sum();
    if sum <= 0.0 {
        return items[rng.int_range(0, items.len())];
    }
    let mut r = rng.next_f64() * sum;
    for (&item, &w) in items.iter().zip(weights.iter()) {
        r -= w.max(0.0);
        if r <= 0.0 {
            return item;
        }
    }
    items[items.len() - 1]
}

fn nearest_chord_tone(pitch: i32, chord_root: i32, min_p: i32, max_p: i32, ext: ChordExtensions) -> i32 {
    let tones = chord_tone_classes(chord_root, ext);
    let mut best = pitch.clamp(min_p, max_p);
    let mut best_d = i32::MAX;
    for &t in &tones {
        for o in 0..=(max_p / 7 + 1) {
            let cand = t + 7 * o;
            if cand >= min_p && cand <= max_p {
                let d = (cand - pitch).abs();
                if d < best_d {
                    best_d = d;
                    best = cand;
                }
            }
        }
    }
    best
}

fn bar_anchor_pitch(
    cfg: &SynthConfig,
    bar_idx: usize,
    base_seed: u32,
    chord_root: i32,
    min_p: i32,
    max_p: i32,
    phrase_len: usize,
) -> i32 {
    let pos_in = bar_idx % phrase_len;
    let x = (pos_in as f64) / (phrase_len.saturating_sub(1).max(1) as f64);
    let is_cad = pos_in == phrase_len - 1 && phrase_len > 1;
    let tension = if is_cad { 0.0 } else { tension_value(cfg.tension_curve, x) };
    let cad_str = if is_cad { cfg.cadence_strength } else { 0.1 };

    let tones = chord_tone_classes(chord_root, cfg.chord_extensions);
    let prev = (max_p - min_p) / 2;
    let mid = 0.5 * ((min_p + max_p) as f64);
    let desired = mid + (tension - 0.5) * 0.35 * ((max_p - min_p) as f64);
    let root_mod = chord_root.rem_euclid(7);
    let mut items = Vec::new();
    let mut weights = Vec::new();
    for &tone in &tones {
        let base_w = if tone == root_mod {
            1.0 + 1.8 * cad_str
        } else if tone == (root_mod + 2) % 7 {
            1.0
        } else if tone == (root_mod + 4) % 7 {
            0.95
        } else if tone == (root_mod + 6) % 7 {
            0.85
        } else {
            0.75
        };
        for o in 0..=(max_p / 7) {
            let cand = tone + 7 * o;
            if cand < min_p || cand > max_p {
                continue;
            }
            items.push(cand);
            weights.push(
                base_w
                    * (-((cand - prev).abs() as f64) / 2.6).exp()
                    * (-((cand as f64) - desired).abs() / 4.0).exp(),
            );
        }
    }
    let salt = 3001_u32.wrapping_add((bar_idx as u32).wrapping_mul(17));
    let mut bar_rng = SeededRng::new(derive_seed(base_seed, salt));
    if items.is_empty() {
        prev
    } else {
        weighted_choice(&mut bar_rng, &items, &weights)
    }
}

fn render_melody_procedural(
    cfg: &SynthConfig,
    duration: f64,
    harmony: &HarmonyPlan,
    rng: &mut SeededRng,
    t_offset: f64,
) -> Vec<f64> {
    let sr = SAMPLE_RATE;
    let mut sig = vec![0.0_f64; sample_count(duration, sr)];
    let atk_mult = attack_mult(cfg.attack);
    let echo_mult = cfg.echo / 0.5;
    let spb = seconds_per_beat(cfg);
    let bar_sec = 4.0 * spb;
    if bar_sec <= 0.0 {
        return sig;
    }
    let cutoff = 700.0 * cfg.brightness + 180.0;

    let base_seed = rng.int_range(1, 0x7FFF_FFFF) as u32;
    let phrase_len = (cfg.phrase_len_bars as usize).clamp(1, 16);
    let density = cfg.melody_density.clamp(0.0, 1.0);
    let reg_min = cfg.register_min_oct.min(cfg.register_max_oct) as i32;
    let reg_max = cfg.register_min_oct.max(cfg.register_max_oct) as i32;
    let min_p = 0_i32;
    let max_p = (reg_max - reg_min).max(0) * 7 + 6;
    let step_beats: f64 = if density < 0.25 {
        1.0
    } else if density < 0.60 {
        0.5
    } else {
        0.25
    };
    let steps_per_bar = ((4.0_f64 / step_beats).round() as usize).clamp(1, 64);

    // O(duration) bar-deterministic rendering: only inspect bars intersecting `[t_offset, t_offset + duration)`.
    let first_bar = ((t_offset - 2.0 * spb).max(0.0) / bar_sec).floor() as usize;
    let last_bar = (((t_offset + duration) / bar_sec).ceil() as usize).max(first_bar + 1);

    for bar in first_bar..last_bar {
        let bar_start_sec = (bar as f64) * bar_sec;
        let local_bar_beat = (bar_start_sec - t_offset) / spb;
        let chord_root = chord_root_at_beat(local_bar_beat.max(0.0), harmony);
        let next_chord_root = chord_root_at_beat((local_bar_beat + 4.0).max(0.0), harmony);

        let pos_in = bar % phrase_len;
        let x = (pos_in as f64) / (phrase_len.saturating_sub(1).max(1) as f64);
        let tension = tension_value(cfg.tension_curve, x);
        let is_cad = pos_in == phrase_len - 1 && phrase_len > 1;
        let anchor_p = bar_anchor_pitch(cfg, bar, base_seed, chord_root, min_p, max_p, phrase_len);
        let next_anchor =
            bar_anchor_pitch(cfg, bar + 1, base_seed, next_chord_root, min_p, max_p, phrase_len);

        let bar_salt = 7001_u32.wrapping_add((bar as u32).wrapping_mul(29));
        let hum_salt = 9001_u32.wrapping_add((bar as u32).wrapping_mul(37));
        let mut bar_rng = SeededRng::new(derive_seed(base_seed, bar_salt));
        let mut hum_rng = SeededRng::new(derive_seed(base_seed, hum_salt));
        let mut need_resolve = false;
        let mut cur_p = anchor_p;

        for s in 0..steps_per_bar {
            let pos_beats = (s as f64) * step_beats;
            let mut start_sec = bar_start_sec + pos_beats * spb;
            if (step_beats - 0.5).abs() < 1e-9 && s % 2 == 1 {
                start_sec += cfg.swing * 0.12 * spb;
            }
            let (play, is_anchor) = if s == 0 {
                (true, true)
            } else {
                let offbeat = (pos_beats - pos_beats.round()).abs() > 1e-6;
                let mut prob = 0.10 + 0.80 * density + 0.20 * tension;
                prob += if offbeat {
                    0.25 * cfg.syncopation
                } else {
                    -0.15 * cfg.syncopation
                };
                if is_cad && pos_beats >= 2.0 {
                    prob *= 0.35 + 0.40 * (1.0 - cfg.cadence_strength);
                }
                (bar_rng.next_f64() < prob.clamp(0.0, 1.0), false)
            };
            if !play {
                continue;
            }

            let pitch = if s == 0 {
                anchor_p
            } else {
                let delta = next_anchor - cur_p;
                let dir = if delta == 0 {
                    if bar_rng.next_f64() < 0.5 { -1 } else { 1 }
                } else if delta > 0 {
                    1
                } else {
                    -1
                };
                let mut p = if need_resolve {
                    need_resolve = false;
                    nearest_chord_tone(cur_p, chord_root, min_p, max_p, cfg.chord_extensions)
                } else {
                    let leap_prob = (1.0 - cfg.step_bias) * (0.25 + 0.35 * tension);
                    let step_sz = if bar_rng.next_f64() < 0.06 + 0.10 * tension {
                        0
                    } else if delta.abs() >= 4 && bar_rng.next_f64() < leap_prob {
                        (2 + bar_rng.int_range(0, 3)) as i32
                    } else {
                        1
                    };
                    (cur_p + dir * step_sz).clamp(min_p, max_p)
                };
                let tones = chord_tone_classes(chord_root, cfg.chord_extensions);
                if !tones.contains(&p.rem_euclid(7)) {
                    if bar_rng.next_f64() < 0.55 - 0.45 * tension {
                        p = nearest_chord_tone(p, chord_root, min_p, max_p, cfg.chord_extensions);
                    } else {
                        need_resolve = true;
                    }
                }
                p
            };

            let mut dur_beats = step_beats * (0.70 + 0.25 * bar_rng.next_f64());
            if is_anchor {
                dur_beats = dur_beats.max(if is_cad { 1.6 } else { 1.0 });
            }
            let dur_sec = (dur_beats * spb).max(0.02);
            let jitter = if cfg.human > 0.0 && !is_anchor {
                bar_rng
                    .normal(0.0, cfg.human * 0.006 * spb)
                    .clamp(-0.02 * spb, 0.02 * spb)
            } else {
                0.0
            };
            let ev_start = (start_sec + jitter).max(0.0);
            let mut amp = 0.12 + 0.05 * tension;
            if is_cad && pos_beats >= 2.0 {
                amp *= 0.85;
            }
            if is_anchor {
                amp *= 1.10;
            }

            let use_chrom = cfg.chromatic_prob > 0.0
                && !is_anchor
                && (next_anchor - pitch).abs() <= 2
                && bar_rng.next_f64() < cfg.chromatic_prob * (0.35 + 0.65 * tension);
            let freq = if use_chrom {
                let sign = if next_anchor >= pitch { 1 } else { -1 };
                freq_from_note(cfg.root, semitone_from_degree(cfg.mode, next_anchor) - sign, reg_min)
            } else {
                scale_freq(cfg, pitch, reg_min)
            };

            if ev_start < t_offset + duration && ev_start + dur_sec > t_offset {
                let note = osc_for_grain(cfg.grain, freq, dur_sec, sr, amp, ev_start);
                let note = apply_lowpass(&note, cutoff, sr);
                let atk = (if is_anchor { 0.05 } else { 0.03 }) * atk_mult;
                let rel = (if is_anchor { 0.22 } else { 0.16 }) * atk_mult;
                let note = apply_adsr(&note, atk, 0.12, 0.55, rel, sr);
                let note = apply_humanize(&note, cfg.human * 0.6, sr, &mut hum_rng);

                let local_start = ((ev_start - t_offset) * (sr as f64)).round() as isize;
                if local_start < 0 {
                    let skip = (-local_start) as usize;
                    if skip < note.len() {
                        add_note(&mut sig, &note[skip..], 0);
                    }
                } else {
                    add_note(&mut sig, &note, local_start as usize);
                }
            }

            cur_p = pitch;
        }
    }

    let del = apply_delay(&sig, 0.33, 0.35 * echo_mult, 0.28 * echo_mult, sr);
    apply_reverb(&del, cfg.space * 0.55, cfg.space * 0.75, sr)
}

fn render_melody(
    cfg: &SynthConfig,
    duration: f64,
    harmony: &HarmonyPlan,
    rng: &mut SeededRng,
    t_offset: f64,
) -> Vec<f64> {
    if cfg.melody_engine == MelodyEngine::Procedural || cfg.melody == MelodyStyle::Procedural {
        return render_melody_procedural(cfg, duration, harmony, rng, t_offset);
    }
    match cfg.melody {
        MelodyStyle::Rising | MelodyStyle::Heroic => render_bpm_pattern_melody(
            cfg, duration, harmony,
            &[0, -1, 2, -1, 4, -1, -1, 5, -1, -1, 6, -1, 7, -1, -1, -1],
            1.0, 5, 0.16, 1.7, 900.0 * cfg.brightness + 250.0,
            (0.07, 0.22, 0.55, 0.75), (0.33, 0.34, 0.26), (0.55, 0.75), rng, t_offset,
        ),
        MelodyStyle::Falling => render_bpm_pattern_melody(
            cfg, duration, harmony,
            &[6, -1, -1, 5, -1, -1, 4, -1, 2, -1, -1, 0, -1, -1, -1, -1],
            1.0, 5, 0.16, 1.6, 850.0 * cfg.brightness + 220.0,
            (0.08, 0.25, 0.52, 0.75), (0.38, 0.36, 0.28), (0.58, 0.78), rng, t_offset,
        ),
        MelodyStyle::Minimal => render_bpm_pattern_melody(
            cfg, duration, harmony,
            &[4, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1],
            1.0, 5, 0.19, 2.4, 620.0 * cfg.brightness + 160.0,
            (0.12, 0.35, 0.5, 1.0), (0.50, 0.42, 0.35), (0.7, 0.9), rng, t_offset,
        ),
        MelodyStyle::Ornamental => render_bpm_pattern_melody(
            cfg, duration, harmony,
            &[2, 0, -1, -1, 5, 4, -1, -1, 3, 2, -1, -1, 2, 0, -1, -1],
            0.5, 5, 0.16, 1.6, 900.0 * cfg.brightness + 220.0,
            (0.04, 0.22, 0.52, 0.60), (0.30, 0.34, 0.25), (0.5, 0.7), rng, t_offset,
        ),
        MelodyStyle::ArpMelody => render_bpm_pattern_melody(
            cfg, duration, harmony,
            &[0, 2, 4, 2],
            0.5, 4, 0.14, 0.85, 1150.0 * cfg.brightness + 280.0,
            (0.015, 0.08, 0.4, 0.14), (0.25, 0.30, 0.24), (0.45, 0.65), rng, t_offset,
        ),
        _ => render_bpm_pattern_melody(
            cfg, duration, harmony,
            &[0, -1, 2, -1, 4, -1, 2, -1, 0, -1, -1, -1, 2, -1, -1, -1],
            1.0, 5, 0.17, 1.5, 800.0 * cfg.brightness + 220.0,
            (0.08, 0.25, 0.6, 0.65), (0.35, 0.38, 0.28), (0.6, 0.8), rng, t_offset,
        ),
    }
}

fn grid_slot_times(cfg: &SynthConfig, duration: f64, slots_per_beat: usize, t_offset: f64) -> (Vec<f64>, i64) {
    let slot_dur = seconds_per_beat(cfg) / (slots_per_beat as f64);
    if slot_dur <= 0.0 {
        return (Vec::new(), 0);
    }
    let first = first_grid_slot(t_offset, slot_dur);
    let mut times = Vec::new();
    let mut cur = first;
    loop {
        let local = (cur as f64) * slot_dur - t_offset;
        if local >= duration {
            break;
        }
        times.push(local);
        cur += 1;
    }
    (times, first)
}

#[allow(clippy::too_many_arguments)]
fn render_kick_grid(
    cfg: &SynthConfig,
    duration: f64,
    slots_per_beat: usize,
    pattern: &[u8],
    base_freq: f64,
    kick_dur: f64,
    amp: f64,
    decay_rate: f64,
    lowpass_hz: f64,
    rng: &mut SeededRng,
    t_offset: f64,
) -> Vec<f64> {
    let sr = SAMPLE_RATE;
    let mut sig = vec![0.0_f64; sample_count(duration, sr)];
    let (times, first) = grid_slot_times(cfg, duration, slots_per_beat, t_offset);
    for (i, &t_sec) in times.iter().enumerate() {
        let idx = (first + (i as i64)).rem_euclid(pattern.len() as i64) as usize;
        if pattern[idx] == 0 {
            continue;
        }
        let vel = 1.0 + rng.uniform(-0.07, 0.07) * cfg.human;
        let kick = render_soft_kick(base_freq, kick_dur, amp * vel, decay_rate, lowpass_hz, sr);
        add_note(&mut sig, &kick, (t_sec * (sr as f64)).round() as usize);
    }
    sig
}

fn render_rhythm(cfg: &SynthConfig, duration: f64, rng: &mut SeededRng, t_offset: f64) -> Vec<f64> {
    let sr = SAMPLE_RATE;
    match cfg.rhythm {
        RhythmStyle::None => vec![0.0_f64; sample_count(duration, sr)],
        RhythmStyle::Heartbeat | RhythmStyle::TablaEssence => render_kick_grid(
            cfg, duration, 2, &[1, 1, 0, 0, 0, 0, 0, 0], 54.0, 0.14, 0.30, 9.0, 120.0, rng, t_offset,
        ),
        RhythmStyle::SoftFour | RhythmStyle::KitMedium | RhythmStyle::Military => {
            let s = render_kick_grid(cfg, duration, 1, &[1], 50.0, 0.19, 0.26, 6.0, 110.0, rng, t_offset);
            apply_reverb(&s, cfg.space * 0.22, cfg.space * 0.40, sr)
        }
        RhythmStyle::HatsOnly => {
            let mut sig = vec![0.0_f64; sample_count(duration, sr)];
            let (times, _) = grid_slot_times(cfg, duration, 4, t_offset);
            let hat_n = sample_count(0.045, sr);
            for t_sec in times {
                let vel = 1.0 + rng.uniform(-0.12, 0.12) * cfg.human;
                let hat: Vec<f64> = (0..hat_n).map(|_| rng.normal(0.0, 1.0) * (0.026 * vel)).collect();
                let hat = apply_lowpass(&apply_highpass(&hat, 5800.0, sr), 11500.0, sr);
                let hat = apply_adsr(&hat, 0.005, 0.015, 0.06, 0.025, sr);
                add_note(&mut sig, &hat, (t_sec * (sr as f64)).round() as usize);
            }
            sig
        }
        RhythmStyle::Electronic => {
            let mut sig = vec![0.0_f64; sample_count(duration, sr)];
            let kicks = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];
            let hats = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0];
            let (times, first) = grid_slot_times(cfg, duration, 4, t_offset);
            let hat_n = sample_count(0.05, sr);
            for (i, &t_sec) in times.iter().enumerate() {
                let idx = (first + (i as i64)).rem_euclid(16) as usize;
                let pos = (t_sec * (sr as f64)).round() as usize;
                if kicks[idx] != 0 {
                    add_note(&mut sig, &render_soft_kick(46.0, 0.20, 0.30, 7.5, 110.0, sr), pos);
                }
                if hats[idx] != 0 {
                    let hat: Vec<f64> = (0..hat_n).map(|_| rng.normal(0.0, 1.0) * 0.028).collect();
                    let hat = apply_lowpass(&apply_highpass(&hat, 6200.0, sr), 11500.0, sr);
                    let hat = apply_adsr(&hat, 0.005, 0.014, 0.06, 0.028, sr);
                    add_note(&mut sig, &hat, pos);
                }
            }
            sig
        }
        _ => {
            let s = render_kick_grid(
                cfg, duration, 2, &[1, 0, 0, 0, 0, 0, 0, 0], 58.0, 0.17, 0.32, 7.5, 140.0, rng, t_offset,
            );
            apply_reverb(&s, cfg.space * 0.25, cfg.space * 0.45, sr)
        }
    }
}

fn render_texture(cfg: &SynthConfig, duration: f64, rng: &mut SeededRng, t_offset: f64) -> Vec<f64> {
    let sr = SAMPLE_RATE;
    let n = sample_count(duration, sr);
    let atk = attack_mult(cfg.attack);
    let echo_mult = cfg.echo / 0.5;

    match cfg.texture {
        TextureStyle::None => vec![0.0_f64; n],
        TextureStyle::VinylCrackle => {
            let mut sig = vec![0.0_f64; n];
            let grain_n = ((0.004 * (sr as f64)).round() as usize).max(24);
            let win: Vec<f64> = (0..grain_n)
                .map(|i| 0.5 * (1.0 - (TWO_PI * (i as f64) / ((grain_n - 1) as f64)).cos()))
                .collect();
            for _ in 0..(duration * 6.0) as usize {
                if n <= grain_n + 2 {
                    break;
                }
                let pos = rng.int_range(0, n - grain_n - 1);
                let amp = rng.uniform(0.002, 0.008);
                let g: Vec<f64> = (0..grain_n).map(|_| rng.normal(0.0, 1.0) * amp).collect();
                let g = apply_lowpass(&apply_highpass(&g, 1600.0, sr), 5500.0, sr);
                let gw: Vec<f64> = g.iter().zip(win.iter()).map(|(&a, &w)| a * w).collect();
                add_note(&mut sig, &gw, pos);
            }
            let air: Vec<f64> = (0..n).map(|_| rng.normal(0.0, 1.0) * 0.0018).collect();
            let air = apply_lowpass(&apply_highpass(&air, 700.0, sr), 4200.0, sr);
            for i in 0..n {
                sig[i] += air[i];
            }
            let atk_s = if t_offset > 1e-6 { 0.025 } else { 0.25 };
            apply_adsr(&sig, atk_s, 0.0, 1.0, 0.05, sr)
        }
        TextureStyle::Breath | TextureStyle::NoiseWash | TextureStyle::PadWhisper => {
            let raw: Vec<f64> = (0..n).map(|_| rng.normal(0.0, 1.0) * 0.032).collect();
            let freq = scale_freq(cfg, 0, 3);
            let mut sig = apply_highpass(
                &apply_lowpass(&raw, (freq * 2.2).min(1600.0), sr),
                (freq * 0.5).max(90.0),
                sr,
            );
            let lfo = generate_lfo(duration, 0.08 + 0.24 * cfg.motion.clamp(0.0, 1.0), sr, t_offset);
            for i in 0..n {
                sig[i] *= lfo[i];
            }
            let atk_s = if t_offset > 1e-6 { 0.025 } else { 0.35 };
            let sig = apply_adsr(&sig, atk_s, 0.0, 1.0, 0.05, sr);
            apply_reverb(&sig, cfg.space * 0.6, cfg.space * 0.8, sr)
        }
        TextureStyle::Stars | TextureStyle::Crystal => {
            let mut sig = vec![0.0_f64; n];
            let degrees = [0, 2, 4, 5];
            for _ in 0..(duration * 3.0) as usize {
                let pos = rng.int_range(0, n.saturating_sub(sr as usize).max(1));
                let deg = degrees[rng.int_range(0, degrees.len())];
                let freq = scale_freq(cfg, deg, 6);
                let star = generate_sine(freq, 0.3, sr, rng.uniform(0.02, 0.05), t_offset + (pos as f64) / (sr as f64));
                let star = apply_adsr(&star, 0.01, 0.1, 0.1, 0.2, sr);
                add_note(&mut sig, &star, pos);
            }
            let del = apply_delay(&sig, 0.4, 0.5 * echo_mult, 0.4 * echo_mult, sr);
            apply_reverb(&del, cfg.space * 0.9, cfg.space, sr)
        }
        _ => {
            let slow = cfg.texture == TextureStyle::ShimmerSlow;
            let degrees: &[i32] = if slow { &[0, 4] } else { &[0, 2, 4] };
            let amp = if slow { 0.026 } else { 0.030 };
            let lfo_rate = (if slow { 0.12 } else { 0.35 })
                + (if slow { 0.50 } else { 1.6 }) * cfg.motion.clamp(0.0, 1.0);
            let atk_s = if t_offset > 1e-6 { 0.025 } else { 0.35 * atk };
            let mut sig = vec![0.0_f64; n];
            for (idx, &deg) in degrees.iter().enumerate() {
                let mut tone = generate_sine(scale_freq(cfg, deg, 6), duration, sr, amp, t_offset);
                let lfo = generate_lfo(duration, lfo_rate * (1.0 + 0.13 * (idx as f64)), sr, t_offset);
                for i in 0..n {
                    tone[i] *= 0.45 + 0.55 * lfo[i];
                }
                let env = apply_adsr(&tone, atk_s, 0.0, 1.0, 0.05, sr);
                for i in 0..n {
                    sig[i] += env[i];
                }
            }
            let del = apply_delay(
                &sig,
                if slow { 0.5 } else { 0.4 },
                (if slow { 0.48 } else { 0.45 }) * echo_mult,
                (if slow { 0.34 } else { 0.30 }) * echo_mult,
                sr,
            );
            apply_reverb(&del, cfg.space * if slow { 0.80 } else { 0.75 }, cfg.space, sr)
        }
    }
}

fn render_accent(
    cfg: &SynthConfig,
    duration: f64,
    harmony: &HarmonyPlan,
    rng: &mut SeededRng,
    t_offset: f64,
) -> Vec<f64> {
    let sr = SAMPLE_RATE;
    let n = sample_count(duration, sr);
    if cfg.accent == AccentStyle::None {
        return vec![0.0_f64; n];
    }
    let atk = attack_mult(cfg.attack);
    let spb = seconds_per_beat(cfg);
    let echo_mult = cfg.echo / 0.5;
    let mut sig = vec![0.0_f64; n];

    match cfg.accent {
        AccentStyle::Chime | AccentStyle::BlipRandom | AccentStyle::Wind => {
            let degrees = [0, 2, 4, 5, 6];
            for _ in 0..(duration * 1.5) as usize {
                let pos = if n <= sr as usize {
                    0
                } else {
                    rng.int_range(0, n - (sr as usize))
                };
                let deg = degrees[rng.int_range(0, degrees.len())];
                let freq = scale_freq(cfg, deg, 5);
                let t0 = t_offset + (pos as f64) / (sr as f64);
                let mut c = generate_sine(freq, 1.2, sr, rng.uniform(0.06, 0.12), t0);
                let h = generate_sine(freq * 2.0, 1.2, sr, 0.03, t0);
                for i in 0..c.len() {
                    c[i] += h[i];
                }
                add_note(&mut sig, &apply_adsr(&c, 0.005, 0.3, 0.05, 0.9, sr), pos);
            }
            let del = apply_delay(&sig, 0.3, 0.4 * echo_mult, 0.3 * echo_mult, sr);
            apply_reverb(&del, cfg.space * 0.75, cfg.space * 0.9, sr)
        }
        AccentStyle::Pluck | AccentStyle::ArpAccent | AccentStyle::PianoNote => {
            let slot_sec = spb * 0.5;
            let pat = [0, -1, -1, 4, -1, -1, 2, -1, -1, -1, 0, -1, -1, -1, -1, -1];
            let mut slot = first_grid_slot(t_offset, slot_sec);
            loop {
                let local = (slot as f64) * slot_sec - t_offset;
                if local >= duration {
                    break;
                }
                let rel = pat[slot.rem_euclid(pat.len() as i64) as usize];
                if rel >= 0 {
                    let root = chord_root_at_beat(local / spb, harmony);
                    let dur = (duration - local).clamp(0.08, 0.55);
                    let p = generate_triangle(scale_freq(cfg, root + rel, 4), dur, sr, 0.14, 0.0);
                    let p = apply_lowpass(&p, 1500.0 * cfg.brightness + 400.0, sr);
                    let p = apply_adsr(&p, 0.006 * atk, 0.14, 0.04, 0.36 * atk, sr);
                    let p = apply_humanize(&p, cfg.human, sr, rng);
                    add_note(&mut sig, &p, (local * (sr as f64)).round() as usize);
                }
                slot += 1;
            }
            apply_reverb(&sig, cfg.space * 0.5, cfg.space * 0.7, sr)
        }
        _ => {
            let pat = [0, -1, -1, -1, 4, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1];
            let mut slot = first_grid_slot(t_offset, spb);
            loop {
                let local = (slot as f64) * spb - t_offset;
                if local >= duration {
                    break;
                }
                let rel = pat[slot.rem_euclid(pat.len() as i64) as usize];
                if rel >= 0 {
                    let root = chord_root_at_beat(local / spb, harmony);
                    let freq = scale_freq(cfg, root + rel, 5);
                    let dur = (duration - local).clamp(0.12, 1.1);
                    let mut b = generate_sine(freq, dur, sr, 0.11, 0.0);
                    let h2 = generate_sine(freq * 2.0, dur, sr, 0.042, 0.0);
                    let h3 = generate_sine(freq * 3.0, dur, sr, 0.018, 0.0);
                    for i in 0..b.len() {
                        b[i] += h2[i] + h3[i];
                    }
                    let bell = apply_adsr(&b, 0.008 * atk, 0.18, 0.08, 0.65 * atk, sr);
                    let bell = apply_humanize(&bell, cfg.human, sr, rng);
                    add_note(&mut sig, &bell, (local * (sr as f64)).round() as usize);
                }
                slot += 1;
            }
            apply_reverb(&sig, cfg.space * 0.6, cfg.space * 0.8, sr)
        }
    }
}

/// Assemble all active layers into headroom-safe, click-free mono Float64 PCM.
pub fn assemble(
    cfg: &SynthConfig,
    duration: f64,
    normalize: bool,
    seed: u32,
    t_offset: f64,
) -> Vec<f64> {
    let sr = SAMPLE_RATE;
    let n = sample_count(duration, sr);
    if n == 0 {
        return Vec::new();
    }

    let mut seed_rng = SeededRng::new(seed);
    let session_seed = seed_rng.int_range(1, 0x7FFF_FFFF) as u32;
    let mut harm_rng = SeededRng::new(derive_seed(session_seed, 101));
    let harmony = build_harmony_plan(cfg, duration, &mut harm_rng, t_offset);
    let block_salt = ((t_offset * 10.0).round() as i64).unsigned_abs() as u32 * 31;
    let density = cfg.density.clamp(2, 6);

    let mut out = vec![0.0_f64; n];
    let mut mix_in = |layer: &[f64], gain: f64| {
        for (dst, &src) in out.iter_mut().zip(layer.iter()) {
            *dst += src * gain;
        }
    };

    if cfg.depth {
        let sub = render_bass_sub_pulse(cfg, duration, &harmony, t_offset);
        mix_in(&sub, 0.6);
    }

    // density >= 2: bass + pad
    let mut r_bass = SeededRng::new(derive_seed(session_seed, 11_u32.wrapping_add(block_salt)));
    mix_in(&render_bass(cfg, duration, &harmony, &mut r_bass, t_offset), 1.0);

    let mut r_pad = SeededRng::new(derive_seed(session_seed, 23_u32.wrapping_add(block_salt)));
    mix_in(&render_pad(cfg, duration, &harmony, &mut r_pad, t_offset), 1.0);

    if density >= 3 {
        let mel_seed = if cfg.melody_engine == MelodyEngine::Procedural {
            derive_seed(session_seed, 101)
        } else {
            derive_seed(session_seed, 41_u32.wrapping_add(block_salt))
        };
        let mut r_mel = SeededRng::new(mel_seed);
        mix_in(&render_melody(cfg, duration, &harmony, &mut r_mel, t_offset), 1.0);
    }
    if density >= 4 && cfg.rhythm != RhythmStyle::None {
        let mut r_rhy = SeededRng::new(derive_seed(session_seed, 53_u32.wrapping_add(block_salt)));
        mix_in(&render_rhythm(cfg, duration, &mut r_rhy, t_offset), 1.0);
    }
    if density >= 5 && cfg.texture != TextureStyle::None {
        let mut r_tex = SeededRng::new(derive_seed(session_seed, 67_u32.wrapping_add(block_salt)));
        mix_in(&render_texture(cfg, duration, &mut r_tex, t_offset), 1.0);
    }
    if density >= 6 && cfg.accent != AccentStyle::None {
        let mut r_acc = SeededRng::new(derive_seed(session_seed, 79_u32.wrapping_add(block_salt)));
        mix_in(&render_accent(cfg, duration, &harmony, &mut r_acc, t_offset), 1.0);
    }

    let master = apply_stereo_width(&out, cfg.stereo, sr);
    let master = apply_master_hall_reverb(&master, cfg.space, sr);
    let master = apply_highpass(&master, 20.0, sr);
    let mut master = soft_limit(&master, 0.92);

    if normalize && !master.is_empty() {
        let peak = master.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        if peak > 1e-6 {
            let gain = 0.78 / peak;
            for s in &mut master {
                *s *= gain;
            }
        }
    }

    let fade_sec = if t_offset > 1e-6 { 0.008 } else { 0.055 };
    soft_fade_edges(&master, fade_sec, fade_sec, sr)
}
