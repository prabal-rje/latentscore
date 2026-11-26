# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

"""
Architecture:

1. Primitives: oscillators, envelopes, filters, effects
2. Patterns: pre-baked layer templates (bass, pad, melody, rhythm, texture, accent)
3. Assembler: config → audio conversion with V2 parameter transforms
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, Literal, TypeAlias

import numpy as np
import soundfile as sf  # type: ignore[import]
from numpy.typing import NDArray
from scipy.signal import butter, decimate, lfilter  # type: ignore[import]

from .config import (
    AccentStyle,
    AttackStyle,
    BassStyle,
    ChordExtensions,
    DensityLevel,
    GrainStyle,
    ModeName,
    PadStyle,
    RhythmStyle,
    RootNote,
    SynthConfig,
    TextureStyle,
)

MusicConfig: TypeAlias = SynthConfig

# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 44100

# Feature flag: when True, rhythm patterns use config.tempo for beat spacing.
# When False, rhythm patterns divide the chunk duration into fixed slots (legacy behavior).
# Set to False to revert to old behavior if needed.
TEMPO_AWARE_RHYTHM = True

# Root note frequencies (octave 4) - mapping proxy for immutability
NOTE_FREQS: Mapping[RootNote, float] = MappingProxyType(
    {
        "c": 261.63,
        "c#": 277.18,
        "d": 293.66,
        "d#": 311.13,
        "e": 329.63,
        "f": 349.23,
        "f#": 369.99,
        "g": 392.00,
        "g#": 415.30,
        "a": 440.00,
        "a#": 466.16,
        "b": 493.88,
    }
)

# Root to semitone offset
ROOT_SEMITONES: Mapping[RootNote, int] = MappingProxyType(
    {
        "c": 0,
        "c#": 1,
        "d": 2,
        "d#": 3,
        "e": 4,
        "f": 5,
        "f#": 6,
        "g": 7,
        "g#": 8,
        "a": 9,
        "a#": 10,
        "b": 11,
    }
)

# Mode intervals (semitones from root) - tuples for JIT
MODE_INTERVALS: Mapping[ModeName, tuple[int, ...]] = MappingProxyType(
    {
        "major": (0, 2, 4, 5, 7, 9, 11),
        "minor": (0, 2, 3, 5, 7, 8, 10),
        "dorian": (0, 2, 3, 5, 7, 9, 10),
        "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    }
)

# V2 parameter mappings
ATTACK_MULT: Mapping[AttackStyle, float] = MappingProxyType(
    {"soft": 2.5, "medium": 1.0, "sharp": 0.3}
)
GRAIN_OSC: Mapping[GrainStyle, str] = MappingProxyType(
    {"clean": "sine", "warm": "triangle", "gritty": "sawtooth"}
)

# Density → active layers (tuples)
DENSITY_LAYERS: Mapping[DensityLevel, tuple[str, ...]] = MappingProxyType(
    {
        2: ("bass", "pad"),
        3: ("bass", "pad", "melody"),
        4: ("bass", "pad", "melody", "rhythm"),
        5: ("bass", "pad", "melody", "rhythm", "texture"),
        6: ("bass", "pad", "melody", "rhythm", "texture", "accent"),
    }
)

FloatArray: TypeAlias = NDArray[np.float64]
# OscFn signature: (freq, duration, sr, amp, t_offset) -> audio
OscFn: TypeAlias = Callable[[float, float, int, float, float], FloatArray]


# =============================================================================
# PART 1: SYNTHESIS PRIMITIVES
# =============================================================================


def freq_from_note(root: RootNote, semitones: int = 0, octave: int = 4) -> float:
    """Get frequency for a note."""
    root_value: RootNote = root
    if root_value not in NOTE_FREQS:
        raise ValueError(f"Unknown root note: {root}. Valid: {list(NOTE_FREQS.keys())}")
    base_freq = NOTE_FREQS[root_value]
    octave_shift = octave - 4
    return base_freq * (2**octave_shift) * (2 ** (semitones / 12))


_REVERB_TAPS_SEC: tuple[tuple[float, float], ...] = (
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
)

_DIFFUSION_TAPS_SEC: tuple[tuple[float, float], ...] = (
    (0.0073, 0.35),
    (0.0131, -0.28),
    (0.0197, 0.22),
)


def _sample_count(duration: float, sr: int) -> int:
    """Compute exact non-negative sample count for duration at sample rate."""
    return max(0, int(sr * duration))


def _time_axis(num_samples: int, sr: int, t_offset: float = 0.0) -> FloatArray:
    """Build exact integer-indexed sample timestamps without endpoint stretch."""
    if num_samples <= 0:
        return np.empty(0, dtype=np.float64)
    return t_offset + (np.arange(num_samples, dtype=np.float64) / float(sr))


def _match_length(signal: FloatArray, target_len: int) -> FloatArray:
    """Trim or zero-pad a 1-D signal to exact target length."""
    if len(signal) > target_len:
        return signal[:target_len]
    if len(signal) < target_len:
        return np.pad(signal, (0, target_len - len(signal)))
    return signal


def _poly_blep_2pt(phase: FloatArray, dt: float) -> FloatArray:
    """Evaluate verified 2-point PolyBLEP residual on wrapped phase in [0, 1)."""
    corr = np.zeros_like(phase, dtype=np.float64)
    if dt <= 0.0 or dt >= 0.5:
        return corr

    m1 = phase < dt
    if np.any(m1):
        u1 = phase[m1] / dt
        corr[m1] = u1 + u1 - u1 * u1 - 1.0

    m2 = phase > (1.0 - dt)
    if np.any(m2):
        u2 = (phase[m2] - 1.0) / dt
        corr[m2] = u2 * u2 + u2 + u2 + 1.0

    return corr


def generate_sine(
    freq: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    amp: float = 0.3,
    t_offset: float = 0.0,
) -> FloatArray:
    """Generate phase-accurate sine wave on an integer sample clock."""
    n = _sample_count(duration, sr)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    t = _time_axis(n, sr, t_offset)
    return np.asarray(amp * np.sin(2.0 * math.pi * freq * t), dtype=np.float64)


def generate_triangle(
    freq: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    amp: float = 0.3,
    t_offset: float = 0.0,
) -> FloatArray:
    """Generate band-limited triangle wave via odd harmonics below Nyquist."""
    n = _sample_count(duration, sr)
    if n == 0 or freq <= 0.0:
        return np.zeros(n, dtype=np.float64)

    max_harmonic = min(15, int((sr * 0.45) // freq))
    if max_harmonic < 1:
        return np.zeros(n, dtype=np.float64)

    t = _time_axis(n, sr, t_offset)
    out = np.zeros(n, dtype=np.float64)
    two_pi_f_t = 2.0 * math.pi * freq * t
    scale = (8.0 / (math.pi * math.pi)) * amp
    sign = 1.0
    for h in range(1, max_harmonic + 1, 2):
        out += (sign / float(h * h)) * np.sin(h * two_pi_f_t)
        sign = -sign
    return np.asarray(out * scale, dtype=np.float64)


def generate_sawtooth(
    freq: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    amp: float = 0.3,
    oversample: int = 2,
    t_offset: float = 0.0,
) -> FloatArray:
    """Generate anti-aliased sawtooth via verified 2-point PolyBLEP + FIR decimate."""
    n = _sample_count(duration, sr)
    if n == 0 or freq <= 0.0:
        return np.zeros(n, dtype=np.float64)

    os_factor = max(1, int(oversample))
    sr_high = sr * os_factor
    dt = freq / float(sr_high)
    if dt >= 0.45:
        return generate_sine(freq, duration, sr, amp, t_offset)

    t_high = _time_axis(n * os_factor, sr_high, t_offset)
    phase = np.mod(t_high * freq, 1.0)
    signal_high = (2.0 * phase - 1.0) - _poly_blep_2pt(phase, dt)
    signal = (
        decimate(signal_high, os_factor, ftype="fir", zero_phase=True)
        if os_factor > 1
        else signal_high
    )
    return np.asarray(amp * _match_length(np.asarray(signal, dtype=np.float64), n), dtype=np.float64)


def generate_square(
    freq: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    amp: float = 0.3,
    oversample: int = 2,
    t_offset: float = 0.0,
) -> FloatArray:
    """Generate anti-aliased square wave via verified 2-point PolyBLEP + FIR decimate."""
    n = _sample_count(duration, sr)
    if n == 0 or freq <= 0.0:
        return np.zeros(n, dtype=np.float64)

    os_factor = max(1, int(oversample))
    sr_high = sr * os_factor
    dt = freq / float(sr_high)
    if dt >= 0.45:
        return generate_sine(freq, duration, sr, amp, t_offset)

    t_high = _time_axis(n * os_factor, sr_high, t_offset)
    phase = np.mod(t_high * freq, 1.0)
    naive = np.where(phase < 0.5, 1.0, -1.0)
    corr = _poly_blep_2pt(phase, dt) - _poly_blep_2pt(np.mod(phase + 0.5, 1.0), dt)
    signal_high = naive + corr
    signal = (
        decimate(signal_high, os_factor, ftype="fir", zero_phase=True)
        if os_factor > 1
        else signal_high
    )
    return np.asarray(amp * _match_length(np.asarray(signal, dtype=np.float64), n), dtype=np.float64)


def generate_noise(
    duration: float,
    sr: int = SAMPLE_RATE,
    amp: float = 0.1,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Generate white noise using explicit RNG when supplied."""
    n = _sample_count(duration, sr)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    samples = rng.standard_normal(n) if rng is not None else np.random.randn(n)
    return np.asarray(amp * samples, dtype=np.float64)


# NOTE: sawtooth/square have an extra 'oversample' parameter, so types don't
# match OscFn exactly. Callers use keyword args for t_offset, which works at runtime.
OSC_FUNCTIONS: Mapping[str, OscFn] = MappingProxyType(  # type: ignore[assignment]
    {
        "sine": generate_sine,
        "triangle": generate_triangle,
        "sawtooth": generate_sawtooth,
        "square": generate_square,
    }
)


def _raised_cosine_ramp(length: int, start: float, end: float) -> FloatArray:
    """Build a C1-smooth raised-cosine ramp from start to end."""
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    if length == 1:
        return np.array([end], dtype=np.float64)
    u = np.linspace(0.0, math.pi, length, dtype=np.float64)
    curve = 0.5 * (1.0 - np.cos(u))
    return np.asarray(start + (end - start) * curve, dtype=np.float64)


def _budget_adsr_samples(
    total: int,
    attack: float,
    decay: float,
    release: float,
    sr: int,
) -> tuple[int, int, int, int]:
    """Fit attack, decay, sustain, and release inside total without release truncation."""
    min_a = min(max(1, int(round(0.005 * sr))), max(1, total // 4))
    min_r = min(max(1, int(round(0.018 * sr))), max(1, total // 3))

    a_req = max(min_a, int(round(max(attack, 0.005) * sr)))
    d_req = max(0, int(round(max(decay, 0.0) * sr)))
    r_req = max(min_r, int(round(max(release, 0.018) * sr)))

    stage_sum = a_req + d_req + r_req
    if stage_sum <= total:
        return a_req, d_req, total - stage_sum, r_req

    scale = float(total) / float(stage_sum)
    r_len = min(max(min_r, int(round(r_req * scale))), max(1, total - min_a))
    rem = total - r_len
    ad_sum = max(1, a_req + d_req)
    a_len = min(max(min_a, int(round(rem * (a_req / float(ad_sum))))), rem)
    d_len = max(0, rem - a_len)
    return a_len, d_len, 0, r_len


def apply_adsr(
    signal: FloatArray,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    sr: int = SAMPLE_RATE,
) -> FloatArray:
    """Apply click-free raised-cosine ADSR guaranteed to start and end at zero."""
    total = len(signal)
    if total == 0:
        return np.empty(0, dtype=np.float64)
    if total == 1:
        return np.zeros(1, dtype=np.float64)

    sus = float(np.clip(sustain, 0.0, 1.0))
    a_len, d_len, s_len, r_len = _budget_adsr_samples(total, attack, decay, release, sr)

    parts: list[FloatArray] = [_raised_cosine_ramp(a_len, 0.0, 1.0)]
    if d_len > 0:
        parts.append(_raised_cosine_ramp(d_len, 1.0, sus))
    if s_len > 0:
        parts.append(np.full(s_len, sus, dtype=np.float64))

    release_start = sus if (d_len > 0 or s_len > 0) else 1.0
    parts.append(_raised_cosine_ramp(r_len, release_start, 0.0))

    env = _match_length(np.concatenate(parts), total)
    env[0] = 0.0
    env[-1] = 0.0
    return np.asarray(signal * env, dtype=np.float64)


def soft_limit(signal: FloatArray, ceiling: float = 0.92) -> FloatArray:
    """Apply transparent soft-knee tanh ceiling so sums never hard-clip."""
    if signal.size == 0 or ceiling <= 0.0:
        return signal
    knee = ceiling * 0.78
    abs_x = np.abs(signal)
    if float(np.max(abs_x)) <= knee:
        return signal
    excess = np.maximum(0.0, abs_x - knee)
    span = max(1e-6, ceiling - knee)
    compressed = knee + span * np.tanh(excess / span)
    return np.asarray(
        np.sign(signal) * np.where(abs_x > knee, compressed, abs_x),
        dtype=np.float64,
    )


def soft_fade_edges(
    signal: FloatArray,
    fade_in_sec: float = 0.0,
    fade_out_sec: float = 0.0,
    sr: int = SAMPLE_RATE,
) -> FloatArray:
    """Apply raised-cosine fade-in and fade-out at buffer boundaries."""
    n = len(signal)
    if n < 2:
        return signal.copy()
    out = signal.copy()
    in_n = min(n // 2, max(0, int(round(fade_in_sec * sr))))
    out_n = min(n - in_n, max(0, int(round(fade_out_sec * sr))))
    if in_n >= 2:
        out[:in_n] *= _raised_cosine_ramp(in_n, 0.0, 1.0)
    if out_n >= 2:
        out[-out_n:] *= _raised_cosine_ramp(out_n, 1.0, 0.0)
    return out


def fade_into_nothingness(
    signal: FloatArray,
    fade_sec: float = 0.35,
    sr: int = SAMPLE_RATE,
) -> FloatArray:
    """Apply a smooth cubic fade-out (1 - u)^3 ending at exact zero."""
    n = len(signal)
    if n < 2 or fade_sec <= 0.0:
        return signal.copy()
    out = signal.copy()
    fade_n = min(n, max(2, int(round(fade_sec * sr))))
    u = np.linspace(0.0, 1.0, fade_n, dtype=np.float64)
    curve = (1.0 - u) ** 3
    curve[-1] = 0.0
    out[-fade_n:] *= curve
    return out


def compute_bar_aligned_duration(tempo: float, target_seconds: float = 16.0) -> float:
    """Snap target_seconds to the nearest 4-bar phrase boundary for tempo in [0, 1]."""
    clamped = max(0.0, min(1.0, float(tempo)))
    bpm = 55.0 + 110.0 * clamped
    bar_sec = 4.0 * (60.0 / bpm)
    phrase_sec = 4.0 * bar_sec
    phrases = max(1, int(round(max(bar_sec, target_seconds) / phrase_sec)))
    return phrases * phrase_sec


@lru_cache(maxsize=512)
def _butter_cached(
    kind: str, normalized_cutoff: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    btype: Literal["low", "high"] = "low" if kind == "low" else "high"
    coeffs = butter(2, normalized_cutoff, btype=btype, output="ba")
    assert isinstance(coeffs, tuple)
    assert len(coeffs) == 2
    b_raw, a_raw = coeffs
    assert isinstance(b_raw, np.ndarray)
    assert isinstance(a_raw, np.ndarray)
    b = np.asarray(b_raw, dtype=np.float64)
    a = np.asarray(a_raw, dtype=np.float64)
    return b, a


def apply_lowpass(signal: FloatArray, cutoff: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply 2nd-order Butterworth lowpass filter with fine cutoff precision."""
    if signal.size == 0:
        return np.empty(0, dtype=np.float64)
    nyquist = sr * 0.5
    normalized = min(max(cutoff / nyquist, 0.0005), 0.99)
    b, a = _butter_cached("low", round(normalized, 5))
    filtered = lfilter(b, a, signal)
    return np.asarray(filtered, dtype=np.float64)


def apply_highpass(signal: FloatArray, cutoff: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply 2nd-order Butterworth highpass filter with fine cutoff precision."""
    if signal.size == 0:
        return np.empty(0, dtype=np.float64)
    nyquist = sr * 0.5
    normalized = min(max(cutoff / nyquist, 0.0005), 0.99)
    b, a = _butter_cached("high", round(normalized, 5))
    filtered = lfilter(b, a, signal)
    return np.asarray(filtered, dtype=np.float64)


def apply_delay(
    signal: FloatArray, delay_time: float, feedback: float, wet: float, sr: int = SAMPLE_RATE
) -> FloatArray:
    """Apply stable lowpass-damped tape-style delay (feedback strictly <= 0.72)."""
    n = len(signal)
    if n == 0 or wet <= 0.0 or delay_time <= 0.0:
        return signal

    safe_fb = float(np.clip(feedback, 0.0, 0.72))
    safe_wet = float(np.clip(wet, 0.0, 0.55))
    delay_samples = int(round(delay_time * sr))
    if delay_samples <= 0 or delay_samples >= n:
        return signal

    wet_buf = np.zeros(n, dtype=np.float64)
    tap_source = apply_lowpass(signal, 2800.0, sr)
    for tap_idx in range(1, 6):
        offset = delay_samples * tap_idx
        if offset >= n:
            break
        gain = (safe_fb**tap_idx) * safe_wet
        if gain < 0.005:
            break
        wet_buf[offset:] += tap_source[:-offset] * gain

    return np.asarray(signal + wet_buf, dtype=np.float64)


def apply_reverb(signal: FloatArray, room: float, size: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply diffused, lowpass-damped ambient reverb without metallic comb ringing."""
    n = len(signal)
    safe_room = float(np.clip(room, 0.0, 1.0))
    safe_size = float(np.clip(size, 0.0, 1.0))
    if n == 0 or safe_room <= 0.01 or safe_size <= 0.01:
        return signal

    diffused = signal.copy()
    for d_sec, d_gain in _DIFFUSION_TAPS_SEC:
        off = int(round(d_sec * sr))
        if 0 < off < n:
            diffused[off:] += signal[:-off] * d_gain

    size_scale = 0.45 + 0.90 * safe_size
    wet = np.zeros(n, dtype=np.float64)
    for t_sec, t_gain in _REVERB_TAPS_SEC:
        off = int(round(t_sec * size_scale * sr))
        if 0 < off < n:
            wet[off:] += diffused[:-off] * t_gain

    damp_hz = 4200.0 - 2000.0 * safe_size
    wet = apply_lowpass(wet, damp_hz, sr)
    wet_mix = safe_room * 0.48
    dry_mix = 1.0 - safe_room * 0.16
    return np.asarray(signal * dry_mix + wet * wet_mix, dtype=np.float64)


def apply_stereo_width(signal: FloatArray, width: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply subtle high-passed diffused micro-reflections without low-end comb filtering."""
    safe_width = float(np.clip(width, 0.0, 1.0))
    n = len(signal)
    if safe_width <= 0.0 or n < 16:
        return signal

    air = apply_highpass(signal, 320.0, sr)
    wet = np.zeros(n, dtype=np.float64)
    for delay_sec, gain in ((0.0097, 0.055), (0.0153, -0.042)):
        off = int(round(delay_sec * sr))
        if 0 < off < n:
            wet[off:] += air[:-off] * (gain * safe_width)
    return np.asarray(signal + wet, dtype=np.float64)


def generate_lfo(
    duration: float,
    rate: float,
    sr: int = SAMPLE_RATE,
    t_offset: float = 0.0,
) -> FloatArray:
    """Generate continuous phase-accurate LFO in [0, 1]."""
    n = _sample_count(duration, sr)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    t = _time_axis(n, sr, t_offset)
    return np.asarray(0.5 + 0.5 * np.sin(2.0 * math.pi * rate * t), dtype=np.float64)


def apply_humanize(
    signal: FloatArray,
    amount: float,
    sr: int = SAMPLE_RATE,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Apply smooth sub-2 Hz organic gain breathing with zero high-frequency hiss."""
    n = len(signal)
    if amount <= 0.0 or n < 4:
        return signal

    depth = float(np.clip(amount, 0.0, 1.0)) * 0.075
    num_knots = max(3, int(round((n / float(sr)) * 1.4)) + 2)
    knots = (
        rng.normal(0.0, depth, size=num_knots)
        if rng is not None
        else np.random.randn(num_knots) * depth
    )
    knots = np.clip(knots, -0.09, 0.09)

    knot_pos = np.linspace(0.0, float(num_knots - 1), n, dtype=np.float64)
    idx0 = np.minimum(knot_pos.astype(np.int64), num_knots - 2)
    frac = knot_pos - idx0
    smooth = 0.5 * (1.0 - np.cos(math.pi * frac))
    drift = knots[idx0] * (1.0 - smooth) + knots[idx0 + 1] * smooth
    return np.asarray(signal * (1.0 + drift), dtype=np.float64)


def add_note(signal: FloatArray, note: FloatArray, start_index: int) -> None:
    """Safely adds a note to signal, applying raised-cosine fade-out if truncated."""
    if start_index >= len(signal) or len(note) == 0:
        return

    end_index = start_index + len(note)
    if end_index <= len(signal):
        signal[start_index:end_index] += note
        return

    available = len(signal) - start_index
    clipped = note[:available].copy()
    fade_samples = min(int(round(SAMPLE_RATE * 0.025)), max(2, available // 3))
    if fade_samples >= 2 and available >= fade_samples:
        clipped[-fade_samples:] *= _raised_cosine_ramp(fade_samples, 1.0, 0.0)
    signal[start_index:] += clipped


# =============================================================================
# PART 2: PATTERN GENERATORS
# =============================================================================


# -----------------------------------------------------------------------------
# HARMONY + PROCEDURAL MELODY STRUCTURES (compute-light)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ChordEvent:
    """A chord specified in scale degrees relative to the key, scheduled in beats."""

    start_beat: float
    duration_beats: float
    root_degree: int


@dataclass(frozen=True)
class HarmonyPlan:
    """A sequence of chord events spanning the render duration."""

    chords: tuple[ChordEvent, ...] = ()
    beats_per_bar: int = 4

    def chord_at_beat(self, beat: float) -> ChordEvent:
        """Get the chord event active at the given beat (clamped)."""
        if not self.chords:
            return ChordEvent(0.0, float(self.beats_per_bar), 0)

        for chord in self.chords:
            if chord.start_beat <= beat < (chord.start_beat + chord.duration_beats):
                return chord

        if beat < self.chords[0].start_beat:
            return self.chords[0]
        return self.chords[-1]


@dataclass(frozen=True)
class MelodyPolicy:
    """Knobs for compute-light, phrase-aware, chord-aware melody generation."""

    phrase_len_bars: int = 4
    density: float = 0.45
    syncopation: float = 0.20
    swing: float = 0.0
    motif_repeat_prob: float = 0.50
    step_bias: float = 0.75
    chromatic_prob: float = 0.05
    cadence_strength: float = 0.65
    register_min_oct: int = 4
    register_max_oct: int = 6
    tension_curve: str = "arc"


@dataclass(frozen=True)
class NoteEvent:
    """A rendered melody note event (frequency already resolved)."""

    start_sec: float
    dur_sec: float
    freq: float
    amp: float
    is_anchor: bool = False


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _tension_value(curve: str, x: float) -> float:
    """x in [0..1] -> tension in [0..1]."""
    x = _clamp01(x)
    if curve == "ramp":
        return x
    if curve == "waves":
        return 0.5 - 0.5 * float(np.cos(2.0 * np.pi * 2.0 * x))
    return float(np.sin(np.pi * x))


def _iter_chord_segments(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
) -> Iterable[tuple[float, float, int]]:
    """Yield (start_sec, dur_sec, chord_root_degree) segments."""
    if harmony is None or not harmony.chords:
        yield (0.0, duration, 0)
        return

    seconds_per_beat = _seconds_per_beat(config)
    for chord in harmony.chords:
        start = chord.start_beat * seconds_per_beat
        if start >= duration:
            break
        dur = chord.duration_beats * seconds_per_beat
        end = min(duration, start + dur)
        dur = max(0.0, end - start)
        if dur > 0:
            yield (start, dur, int(chord.root_degree))


def _weighted_choice(
    rng: np.random.Generator, items: Sequence[int], weights: Sequence[float]
) -> int:
    """Small helper around rng.choice for int items."""
    if not items:
        raise ValueError("No items to choose from")
    w = np.asarray(weights, dtype=np.float64)
    if np.all(w <= 0):
        w = np.ones_like(w)
    w = w / np.sum(w)
    idx = int(rng.choice(len(items), p=w))
    return int(items[idx])


def _chord_tones_ascending(
    chord_extensions: ChordExtensions, chord_root_degree: int
) -> tuple[int, ...]:
    """Chord tones as ascending scale degrees above the chord root (best for voicings)."""
    degree = int(chord_root_degree)
    tones: list[int] = [degree, degree + 2, degree + 4]
    if chord_extensions in ("sevenths", "lush"):
        tones.append(degree + 6)
    if chord_extensions == "lush":
        tones.append(degree + 8)
    return tuple(tones)


def _attack_mult(config: SynthConfig) -> float:
    return ATTACK_MULT.get(config.attack, 1.0)


def _osc_type(config: SynthConfig) -> str:
    return GRAIN_OSC.get(config.grain, "sine")


def _echo_mult(config: SynthConfig) -> float:
    return config.echo / 0.5


def _bpm(config: SynthConfig) -> float:
    t = float(np.clip(config.tempo, 0.0, 1.0))
    return 55.0 + 110.0 * t


def _seconds_per_beat(config: SynthConfig) -> float:
    return 60.0 / _bpm(config)


def _rhythm_slot_times(
    config: SynthConfig,
    duration: float,
    slots_per_beat: int = 2,
    t_offset: float = 0.0,
) -> tuple[list[float], int]:
    """
    Generate slot times for rhythm patterns based on tempo.

    Args:
        config: Synth config containing tempo
        slots_per_beat: Number of subdivisions per beat (2 = 8th notes, 4 = 16th notes)
        t_offset: Time offset for pattern continuity across chunks

    Returns:
        Tuple of (slot_times, slot_offset) where:
        - slot_times: List of time positions (in seconds) within the chunk, aligned to global grid
        - slot_offset: Starting pattern index based on t_offset (for pattern continuity)
    """
    if not TEMPO_AWARE_RHYTHM:
        # Legacy: not used in this path, but kept for clarity
        return [], 0

    spb = _seconds_per_beat(config)
    slot_dur = spb / slots_per_beat
    if slot_dur <= 0:
        return [], 0

    # Calculate which global slot contains t_offset
    slot_at_offset = int(t_offset / slot_dur)
    global_slot_start = slot_at_offset * slot_dur

    # If t_offset is exactly at a slot boundary, start there
    # Otherwise, start at the next slot
    if abs(global_slot_start - t_offset) < 1e-9:
        first_slot = slot_at_offset
    else:
        first_slot = slot_at_offset + 1

    # Generate slot times aligned to the global grid
    times: list[float] = []
    current_slot = first_slot
    while True:
        local_time = current_slot * slot_dur - t_offset
        if local_time >= duration:
            break
        times.append(local_time)
        current_slot += 1

    return times, first_slot


def _semitone_from_degree(mode: ModeName, degree: int) -> int:
    intervals = MODE_INTERVALS.get(mode, MODE_INTERVALS["minor"])
    return int(intervals[degree % len(intervals)] + (12 * (degree // len(intervals))))


def _scale_freq(config: SynthConfig, degree: int, octave: int = 4) -> float:
    semitone = _semitone_from_degree(config.mode, degree)
    return freq_from_note(config.root, semitone, octave)


def _chord_tone_classes(
    chord_root_degree: int, chord_extensions: ChordExtensions
) -> tuple[int, ...]:
    root_degree = int(chord_root_degree) % 7
    tones: list[int] = [root_degree, (root_degree + 2) % 7, (root_degree + 4) % 7]
    if chord_extensions in ("sevenths", "lush"):
        tones.append((root_degree + 6) % 7)
    if chord_extensions == "lush":
        tones.append((root_degree + 1) % 7)
    out: list[int] = []
    for tone in tones:
        if tone not in out:
            out.append(tone)
    return tuple(out)


def _chord_root_degree_at_beat(beat: float, harmony: HarmonyPlan | None) -> int:
    if harmony is None:
        return 0
    return int(harmony.chord_at_beat(float(beat)).root_degree) % 7


PatternFn: TypeAlias = Callable[
    [SynthConfig, float, HarmonyPlan | None, MelodyPolicy | None, np.random.Generator, float],
    FloatArray,
]
# PatternFn signature: (config, duration, harmony, melody_policy, rng, t_offset) -> audio


# -----------------------------------------------------------------------------
# BASS PATTERNS
# -----------------------------------------------------------------------------


def bass_drone(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Sustained drone bass (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)

    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)
        freq = _scale_freq(config, chord_root, 2)

        note = generate_sine(freq, note_dur, sr, 0.35, t_offset=t_offset + start_sec)
        note = apply_lowpass(note, 80 * config.brightness + 20, sr)
        note = apply_adsr(
            note,
            1.8 * attack_mult,
            0.5,
            0.95,
            2.2 * attack_mult,
            sr,
        )
        add_note(signal, note, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.5, config.space, sr)
    return signal


def bass_sustained(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Long sustained bass notes that follow chord roots."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        pattern = (0, 0, 4, 0)
        note_dur = seg_dur / len(pattern)

        for i, rel_degree in enumerate(pattern):
            deg = chord_root + rel_degree
            freq = _scale_freq(config, deg, 2)
            start = int((start_sec + i * note_dur) * sr)

            note_start_sec = start_sec + i * note_dur
            note = generate_sine(
                freq, note_dur * 0.95, sr, 0.32, t_offset=t_offset + note_start_sec
            )
            note = apply_lowpass(note, 100 * config.brightness + 30, sr)
            note = apply_adsr(
                note,
                0.7 * attack_mult,
                0.3,
                0.85,
                1.2 * attack_mult,
                sr,
            )
            add_note(signal, note, start)

    signal = apply_reverb(signal, config.space * 0.4, config.space, sr)
    return signal


def bass_pulsing(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Rhythmic pulsing bass aligned to the global beat grid."""
    sr = SAMPLE_RATE
    dur = duration
    attack_mult = _attack_mult(config)
    spb = _seconds_per_beat(config)
    pulse_beats = 0.5
    pulse_sec = pulse_beats * spb

    signal = np.zeros(_sample_count(dur, sr), dtype=np.float64)
    if pulse_sec <= 0.0:
        return signal

    slot_idx = _first_grid_slot(t_offset, pulse_sec)
    while True:
        local_sec = slot_idx * pulse_sec - t_offset
        if local_sec >= dur:
            break

        chord_root = _chord_root_degree_at_beat(local_sec / spb, harmony)
        freq = _scale_freq(config, chord_root, 2)

        note = generate_sine(freq, pulse_sec * 0.82, sr, 0.35, t_offset=t_offset + local_sec)
        note = apply_lowpass(note, 90 * config.brightness + 20, sr)
        note = apply_adsr(
            note,
            0.02 * attack_mult,
            0.08,
            0.6,
            0.25 * attack_mult,
            sr,
        )
        add_note(signal, note, int(round(local_sec * sr)))
        slot_idx += 1

    return signal


def bass_walking(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Walking bass line locked to the global beat grid."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    signal = np.zeros(_sample_count(dur_total, sr), dtype=np.float64)

    note_sec = _seconds_per_beat(config)
    if note_sec <= 0.0:
        return signal

    slot_idx = _first_grid_slot(t_offset, note_sec)
    while True:
        local_sec = slot_idx * note_sec - t_offset
        if local_sec >= dur_total:
            break

        chord_root = _chord_root_degree_at_beat(local_sec / note_sec, harmony)
        rel = (0, 2, 4, 2)[slot_idx % 4]
        degree = chord_root + rel

        freq = _scale_freq(config, degree, 2)
        note = generate_triangle(freq, note_sec * 0.92, sr, 0.30, t_offset=t_offset + local_sec)
        note = apply_lowpass(note, 120 * config.brightness + 40, sr)
        note = apply_adsr(
            note,
            0.04 * attack_mult,
            0.12,
            0.7,
            0.22 * attack_mult,
            sr,
        )
        note = apply_humanize(note, config.human, sr, rng=rng)
        add_note(signal, note, int(round(local_sec * sr)))
        slot_idx += 1

    signal = apply_reverb(signal, config.space * 0.3, config.space * 0.8, sr)
    return signal


def bass_fifth_drone(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Root + fifth drone that follows chord changes."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        root_freq = _scale_freq(config, chord_root, 2)
        fifth_freq = _scale_freq(config, chord_root + 4, 2)

        root = generate_sine(root_freq, note_dur, sr, 0.26, t_offset=t_offset + start_sec)
        root = apply_lowpass(root, 70 * config.brightness + 20, sr)
        root = apply_adsr(root, 2.2 * attack_mult, 0.5, 0.95, 2.6 * attack_mult, sr)

        fifth = generate_sine(fifth_freq, note_dur, sr, 0.18, t_offset=t_offset + start_sec)
        fifth = apply_lowpass(fifth, 100 * config.brightness + 30, sr)
        fifth = apply_adsr(fifth, 2.4 * attack_mult, 0.5, 0.9, 2.6 * attack_mult, sr)

        add_note(signal, root + fifth, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.5, config.space, sr)
    return signal


def bass_sub_pulse(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Deep sub-bass pulse aligned to the global beat grid."""
    sr = SAMPLE_RATE
    dur = duration
    attack_mult = _attack_mult(config)
    signal = np.zeros(_sample_count(dur, sr), dtype=np.float64)

    spb = _seconds_per_beat(config)
    beats_per_pulse = 2.0
    pulse_sec = beats_per_pulse * spb
    if pulse_sec <= 0.0:
        return signal

    slot_idx = _first_grid_slot(t_offset, pulse_sec)
    while True:
        local_sec = slot_idx * pulse_sec - t_offset
        if local_sec >= dur:
            break

        chord_root = _chord_root_degree_at_beat(local_sec / spb, harmony)
        freq = _scale_freq(config, chord_root, 1)
        note = generate_sine(freq, pulse_sec * 0.95, sr, 0.4, t_offset=t_offset + local_sec)
        note = apply_lowpass(note, 55, sr)
        note = apply_adsr(
            note,
            0.25 * attack_mult,
            0.2,
            0.9,
            0.7 * attack_mult,
            sr,
        )
        add_note(signal, note, int(round(local_sec * sr)))
        slot_idx += 1

    return signal


BASS_PATTERNS: Mapping[BassStyle, PatternFn] = MappingProxyType(
    {
        "drone": bass_drone,
        "sustained": bass_sustained,
        "pulsing": bass_pulsing,
        "walking": bass_walking,
        "fifth_drone": bass_fifth_drone,
        "sub_pulse": bass_sub_pulse,
        "octave": bass_sustained,
        "arp_bass": bass_pulsing,
    }
)


# -----------------------------------------------------------------------------
# PAD PATTERNS
# -----------------------------------------------------------------------------


def pad_warm_slow(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Warm evolving pad with proportional LFO motion and click-free ADSR."""
    sr = SAMPLE_RATE
    dur_total = duration
    osc = OSC_FUNCTIONS.get(_osc_type(config), generate_sine)
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)

    signal = np.zeros(_sample_count(dur_total, sr), dtype=np.float64)
    lfo_rate = 0.06 + 0.28 * float(np.clip(config.motion, 0.0, 1.0))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        for degree in _chord_tones_ascending(config.chord_extensions, chord_root)[:3]:
            freq = _scale_freq(config, degree, 3)
            tone = osc(freq, note_dur, sr, 0.15, t_offset=t_offset + start_sec)  # type: ignore[call-arg]
            lfo = generate_lfo(note_dur, lfo_rate, sr, t_offset=t_offset + start_sec)

            base_cutoff = 300.0 * config.brightness + 100.0
            tone_low = apply_lowpass(tone, base_cutoff * 0.5, sr)
            tone_high = apply_lowpass(tone, base_cutoff * 1.5, sr)
            tone = tone_low * (1.0 - lfo) + tone_high * lfo

            tone = apply_adsr(tone, 1.2 * attack_mult, 0.6, 0.85, 1.6 * attack_mult, sr)
            add_note(signal, tone, int(round(start_sec * sr)))

    signal = apply_reverb(signal, config.space * 0.7, config.space * 0.9, sr)
    signal = apply_delay(signal, 0.35, 0.30 * echo_mult, 0.25 * echo_mult, sr)
    return apply_humanize(signal, config.human, sr, rng=rng)


def pad_dark_sustained(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Dark, heavy sustained pad (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        for degree in _chord_tones_ascending(config.chord_extensions, chord_root)[:3]:
            freq = _scale_freq(config, degree, 3)
            tone = generate_sawtooth(freq, note_dur, sr, 0.12, t_offset=t_offset + start_sec)
            tone = apply_lowpass(tone, 200 * config.brightness + 80, sr)
            tone = apply_adsr(tone, 2.0 * attack_mult, 1.0, 0.9, 3.0 * attack_mult, sr)
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.8, config.space, sr)
    return signal


def pad_cinematic(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Big, cinematic pad with movement (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)
    signal = np.zeros(int(sr * dur_total))

    voicings = ((0, 3), (2, 3), (4, 3), (0, 4), (4, 4))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.10, dur_total - start_sec)

        for rel_degree, octave in voicings:
            degree = chord_root + rel_degree
            freq = _scale_freq(config, degree, octave)

            tone = generate_sawtooth(freq, note_dur, sr, 0.08, t_offset=t_offset + start_sec)
            tone += generate_triangle(
                freq * 1.002, note_dur, sr, 0.06, t_offset=t_offset + start_sec
            )

            tone = apply_lowpass(tone, 400 * config.brightness + 150, sr)
            tone = apply_adsr(tone, 1.8 * attack_mult, 0.8, 0.88, 2.8 * attack_mult, sr)
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.85, config.space, sr)
    signal = apply_delay(signal, 0.4, 0.35 * echo_mult, 0.3 * echo_mult, sr)

    return signal


def pad_thin_high(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Thin, high pad (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        for rel_degree in (0, 4):
            degree = chord_root + rel_degree
            freq = _scale_freq(config, degree, 4)
            tone = generate_sine(freq, note_dur, sr, 0.12, t_offset=t_offset + start_sec)
            tone = apply_lowpass(tone, 800 * config.brightness + 200, sr)
            tone = apply_highpass(tone, 200, sr)
            tone = apply_adsr(tone, 1.2 * attack_mult, 0.6, 0.8, 2.0 * attack_mult, sr)
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.75, config.space * 0.95, sr)
    return signal


def pad_ambient_drift(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Slowly drifting ambient pad (chord-aware, with gentle color notes)."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.35, dur_total - start_sec)

        tones = list(_chord_tones_ascending(config.chord_extensions, chord_root)[:3])

        if config.motion > 0.4 and rng.random() < (0.15 + 0.25 * config.motion):
            tones.append(chord_root + 5)

        for degree in tones:
            freq = _scale_freq(config, degree, 3)
            tone = generate_sine(freq, note_dur, sr, 0.14, t_offset=t_offset + start_sec)
            tone = apply_lowpass(tone, 350 * config.brightness + 100, sr)
            tone = apply_adsr(tone, 1.6 * attack_mult, 0.5, 0.85, 2.2 * attack_mult, sr)
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.8, config.space, sr)
    signal = apply_delay(signal, 0.5, 0.4 * echo_mult, 0.35 * echo_mult, sr)

    return signal


def pad_stacked_fifths(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Fifths stacked for a powerful sound (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = duration
    osc = OSC_FUNCTIONS.get(_osc_type(config), generate_triangle)
    attack_mult = _attack_mult(config)

    signal = np.zeros(int(sr * dur_total))

    voicings = ((0, 3), (4, 3), (0, 4), (4, 4))
    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.10, dur_total - start_sec)

        for rel_degree, octave in voicings:
            degree = chord_root + rel_degree
            freq = _scale_freq(config, degree, octave)
            tone = osc(freq, note_dur, sr, 0.10, t_offset=t_offset + start_sec)  # type: ignore[call-arg]
            tone = apply_lowpass(tone, 500 * config.brightness + 150, sr)
            tone = apply_adsr(tone, 1.3 * attack_mult, 0.7, 0.88, 2.2 * attack_mult, sr)
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.7, config.space * 0.85, sr)
    return signal


PAD_PATTERNS: Mapping[PadStyle, PatternFn] = MappingProxyType(
    {
        "warm_slow": pad_warm_slow,
        "dark_sustained": pad_dark_sustained,
        "cinematic": pad_cinematic,
        "thin_high": pad_thin_high,
        "ambient_drift": pad_ambient_drift,
        "stacked_fifths": pad_stacked_fifths,
        "bright_open": pad_thin_high,
    }
)


# -----------------------------------------------------------------------------
# MELODY PATTERNS
# -----------------------------------------------------------------------------


def _resolve_melody_policy(
    policy: MelodyPolicy | None,
    config: SynthConfig,
) -> MelodyPolicy:
    """Return the active MelodyPolicy with safety clamps."""
    base = policy or build_melody_policy(config)
    phrase_len = int(max(1, min(16, base.phrase_len_bars)))
    reg_min = int(min(base.register_min_oct, base.register_max_oct))
    reg_max = int(max(base.register_min_oct, base.register_max_oct))
    return MelodyPolicy(
        phrase_len_bars=phrase_len,
        density=_clamp01(base.density),
        syncopation=_clamp01(base.syncopation),
        swing=_clamp01(base.swing),
        motif_repeat_prob=_clamp01(base.motif_repeat_prob),
        step_bias=_clamp01(base.step_bias),
        chromatic_prob=_clamp01(base.chromatic_prob),
        cadence_strength=_clamp01(base.cadence_strength),
        register_min_oct=reg_min,
        register_max_oct=reg_max,
        tension_curve=base.tension_curve
        if base.tension_curve in ("arc", "ramp", "waves")
        else "arc",
    )


def _grid_step_beats(density: float) -> float:
    """Density->grid step in beats."""
    if density < 0.25:
        return 1.0
    if density < 0.60:
        return 0.5
    return 0.25


def _nearest_chord_tone_pitch(
    pitch: int,
    chord_root_degree: int,
    min_pitch: int,
    max_pitch: int,
    chord_extensions: ChordExtensions,
) -> int:
    """Find the nearest chord-tone pitch (diatonic) to the given pitch."""
    chord_tones = _chord_tone_classes(chord_root_degree, chord_extensions)
    candidates: list[int] = []
    for tone in chord_tones:
        for oct_i in range((max_pitch // 7) + 1):
            cand = tone + 7 * oct_i
            if min_pitch <= cand <= max_pitch:
                candidates.append(cand)
    if not candidates:
        return int(np.clip(pitch, min_pitch, max_pitch))
    return int(min(candidates, key=lambda c: abs(c - pitch)))


def _choose_anchor_pitch(
    rng: np.random.Generator,
    prev_pitch: int,
    chord_root_degree: int,
    min_pitch: int,
    max_pitch: int,
    tension: float,
    cadence_strength: float,
    chord_extensions: ChordExtensions,
) -> int:
    """Choose an anchor pitch (chord tone) with simple voice-leading."""
    chord_root = chord_root_degree % 7
    chord_tones = _chord_tone_classes(chord_root_degree, chord_extensions)

    midpoint = 0.5 * (min_pitch + max_pitch)
    desired = midpoint + (tension - 0.5) * 0.35 * (max_pitch - min_pitch)

    items: list[int] = []
    weights: list[float] = []

    for tone in chord_tones:
        weight = 1.0
        if tone == chord_root:
            weight *= 1.0 + 1.8 * cadence_strength
        elif tone == (chord_root + 2) % 7:
            weight *= 1.0
        elif tone == (chord_root + 4) % 7:
            weight *= 0.95
        elif tone == (chord_root + 6) % 7:
            weight *= 0.85
        else:
            weight *= 0.75

        for oct_i in range((max_pitch // 7) + 1):
            cand = tone + 7 * oct_i
            if cand < min_pitch or cand > max_pitch:
                continue

            d_prev = abs(cand - prev_pitch)
            d_reg = abs(cand - desired)
            w = weight * float(np.exp(-d_prev / 2.6)) * float(np.exp(-d_reg / 4.0))
            items.append(int(cand))
            weights.append(float(w))

    if not items:
        return int(np.clip(prev_pitch, min_pitch, max_pitch))

    return _weighted_choice(rng, items, weights)


def _first_grid_slot(t_offset: float, slot_sec: float) -> int:
    """Return the first beat-grid slot index at or after t_offset."""
    slot = int(math.floor(t_offset / slot_sec))
    return slot + 1 if slot * slot_sec < t_offset - 1e-9 else slot


def _render_procedural_event_slice(
    signal: FloatArray,
    event: NoteEvent,
    osc: OscFn,
    cutoff: float,
    attack_mult: float,
    human: float,
    rng: np.random.Generator,
    t_offset: float,
    duration: float,
) -> None:
    """Render one procedural NoteEvent sliced cleanly into [0, duration)."""
    sr = SAMPLE_RATE
    if event.start_sec >= t_offset + duration or event.start_sec + event.dur_sec <= t_offset:
        return

    note = osc(event.freq, event.dur_sec, sr, event.amp, t_offset=event.start_sec)  # type: ignore[call-arg]
    note = apply_lowpass(note, cutoff, sr)
    atk = (0.05 if event.is_anchor else 0.03) * attack_mult
    rel = (0.22 if event.is_anchor else 0.16) * attack_mult
    note = apply_adsr(note, atk, 0.12, 0.55, rel, sr)
    note = apply_humanize(note, human * 0.6, sr, rng=rng)

    local_start = int(round((event.start_sec - t_offset) * sr))
    if local_start < 0:
        skip = -local_start
        if skip >= len(note):
            return
        note = note[skip:]
        local_start = 0
    add_note(signal, note, local_start)


def _bar_anchor_pitch(
    config: SynthConfig,
    policy: MelodyPolicy,
    bar_idx: int,
    base_seed: int,
    chord_root: int,
    min_pitch: int,
    max_pitch: int,
) -> int:
    """Deterministically choose an anchor pitch for absolute bar_idx in O(1)."""
    phrase_len = policy.phrase_len_bars
    pos_in_phrase = bar_idx % phrase_len
    denom = max(1, phrase_len - 1)
    x = pos_in_phrase / denom
    is_cadence = (pos_in_phrase == phrase_len - 1) and (phrase_len > 1)
    tension = 0.0 if is_cadence else _tension_value(policy.tension_curve, x)
    prev_mid = int(np.clip((max_pitch - min_pitch) // 2, min_pitch, max_pitch))
    bar_rng = np.random.default_rng(_derive_seed(base_seed, 3001 + bar_idx * 17))
    return _choose_anchor_pitch(
        rng=bar_rng,
        prev_pitch=prev_mid,
        chord_root_degree=chord_root,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
        tension=tension,
        cadence_strength=policy.cadence_strength if is_cadence else 0.10,
        chord_extensions=config.chord_extensions,
    )


def melody_procedural(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """O(duration) bar-deterministic procedural melody continuous across blocks."""
    sr = SAMPLE_RATE
    osc = OSC_FUNCTIONS.get(_osc_type(config), generate_triangle)
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)
    cutoff = 700.0 * config.brightness + 180.0
    signal = np.zeros(_sample_count(duration, sr), dtype=np.float64)

    policy = _resolve_melody_policy(melody_policy, config)
    spb = _seconds_per_beat(config)
    bar_sec = 4.0 * spb
    if bar_sec <= 0.0:
        return signal

    base_seed = int(rng.integers(1, 0x7FFFFFFF))
    base_oct = policy.register_min_oct
    min_pitch = 0
    max_pitch = max(0, policy.register_max_oct - policy.register_min_oct) * 7 + 6

    step_beats = _grid_step_beats(policy.density)
    steps_per_bar = max(1, min(64, int(round(4.0 / step_beats))))

    first_bar = max(0, int(math.floor(max(0.0, t_offset - 2.0 * spb) / bar_sec)))
    last_bar = max(first_bar + 1, int(math.ceil((t_offset + duration) / bar_sec)))

    for bar_idx in range(first_bar, last_bar):
        bar_start_sec = bar_idx * bar_sec
        local_bar_beat = (bar_start_sec - t_offset) / spb
        chord_root = _chord_root_degree_at_beat(max(0.0, local_bar_beat), harmony)
        next_chord_root = _chord_root_degree_at_beat(max(0.0, local_bar_beat + 4.0), harmony)

        phrase_len = policy.phrase_len_bars
        pos_in_phrase = bar_idx % phrase_len
        x = pos_in_phrase / max(1, phrase_len - 1)
        tension = _tension_value(policy.tension_curve, x)
        is_cadence_bar = (pos_in_phrase == phrase_len - 1) and (phrase_len > 1)

        anchor_pitch = _bar_anchor_pitch(
            config, policy, bar_idx, base_seed, chord_root, min_pitch, max_pitch
        )
        next_anchor = _bar_anchor_pitch(
            config, policy, bar_idx + 1, base_seed, next_chord_root, min_pitch, max_pitch
        )

        bar_rng = np.random.default_rng(_derive_seed(base_seed, 7001 + bar_idx * 29))
        hum_rng = np.random.default_rng(_derive_seed(base_seed, 9001 + bar_idx * 37))
        need_resolve = False
        current_pitch = anchor_pitch

        for step_idx in range(steps_per_bar):
            pos_beats = step_idx * step_beats
            ev_start = bar_start_sec + pos_beats * spb
            if step_beats == 0.5 and (step_idx % 2 == 1):
                ev_start += policy.swing * 0.12 * spb

            if step_idx == 0:
                play = True
                is_anchor = True
            else:
                is_offbeat = abs(pos_beats - round(pos_beats)) > 1e-6
                prob = 0.10 + 0.80 * policy.density + 0.20 * tension
                prob += (0.25 * policy.syncopation) if is_offbeat else (-0.15 * policy.syncopation)
                if is_cadence_bar and pos_beats >= 2.0:
                    prob *= 0.35 + 0.40 * (1.0 - policy.cadence_strength)
                play = bool(bar_rng.random() < _clamp01(prob))
                is_anchor = False

            if not play:
                continue

            if step_idx == 0:
                pitch = anchor_pitch
            else:
                delta = next_anchor - current_pitch
                direction = int(bar_rng.choice([-1, 1])) if delta == 0 else (1 if delta > 0 else -1)
                if need_resolve:
                    pitch = _nearest_chord_tone_pitch(
                        current_pitch, chord_root, min_pitch, max_pitch, config.chord_extensions
                    )
                    need_resolve = False
                else:
                    leap_prob = (1.0 - policy.step_bias) * (0.25 + 0.35 * tension)
                    step_size = 1
                    if bar_rng.random() < (0.06 + 0.10 * tension):
                        step_size = 0
                    elif abs(delta) >= 4 and (bar_rng.random() < leap_prob):
                        step_size = int(bar_rng.choice([2, 3, 4]))
                    pitch = int(np.clip(current_pitch + direction * step_size, min_pitch, max_pitch))

                chord_tones = _chord_tone_classes(chord_root, config.chord_extensions)
                if (pitch % 7) not in chord_tones:
                    if bar_rng.random() < (0.55 - 0.45 * tension):
                        pitch = _nearest_chord_tone_pitch(
                            pitch, chord_root, min_pitch, max_pitch, config.chord_extensions
                        )
                    else:
                        need_resolve = True

            dur_beats = step_beats * (0.70 + 0.25 * float(bar_rng.random()))
            if is_anchor:
                dur_beats = max(dur_beats, 1.6 if is_cadence_bar else 1.0)
            dur_sec = max(0.02, dur_beats * spb)

            jitter = 0.0
            if config.human > 0 and not is_anchor:
                jitter = float(
                    np.clip(bar_rng.normal(0.0, config.human * 0.006 * spb), -0.02 * spb, 0.02 * spb)
                )
            ev_start_j = max(0.0, ev_start + jitter)

            amp = 0.12 + 0.05 * tension
            if is_cadence_bar and pos_beats >= 2.0:
                amp *= 0.85
            if is_anchor:
                amp *= 1.10

            use_chromatic = (
                (policy.chromatic_prob > 0)
                and (not is_anchor)
                and (abs(next_anchor - pitch) <= 2)
                and (bar_rng.random() < (policy.chromatic_prob * (0.35 + 0.65 * tension)))
            )
            if use_chromatic:
                sign = 1 if (next_anchor - pitch) >= 0 else -1
                semitone = _semitone_from_degree(config.mode, next_anchor) - sign
                freq = freq_from_note(config.root, semitone, base_oct)
            else:
                freq = _scale_freq(config, pitch, base_oct)

            ev = NoteEvent(
                start_sec=ev_start_j,
                dur_sec=dur_sec,
                freq=freq,
                amp=float(amp),
                is_anchor=is_anchor,
            )
            _render_procedural_event_slice(
                signal, ev, osc, cutoff, attack_mult, config.human, hum_rng, t_offset, duration
            )
            current_pitch = pitch

    signal = apply_delay(signal, 0.33, 0.35 * echo_mult, 0.28 * echo_mult, sr)
    return apply_reverb(signal, config.space * 0.55, config.space * 0.75, sr)


def _render_bpm_pattern_melody(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    pattern: tuple[int, ...],
    step_beats: float,
    octave: int,
    amp: float,
    dur_mult: float,
    cutoff: float,
    adsr: tuple[float, float, float, float],
    delay_cfg: tuple[float, float, float],
    reverb_scale: tuple[float, float],
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Schedule a pattern melody on BPM beats locked to active chord harmony."""
    sr = SAMPLE_RATE
    signal = np.zeros(_sample_count(duration, sr), dtype=np.float64)
    osc = OSC_FUNCTIONS.get(_osc_type(config), generate_triangle)
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)
    spb = _seconds_per_beat(config)
    slot_sec = step_beats * spb
    if slot_sec <= 0.0 or not pattern:
        return signal

    slot_idx = _first_grid_slot(t_offset, slot_sec)
    while True:
        local_sec = slot_idx * slot_sec - t_offset
        if local_sec >= duration:
            break
        rel_deg = pattern[slot_idx % len(pattern)]
        if rel_deg >= 0:
            chord_root = _chord_root_degree_at_beat(local_sec / spb, harmony)
            freq = _scale_freq(config, chord_root + rel_deg, octave)
            note_dur = min(slot_sec * dur_mult, max(0.06, duration - local_sec))
            note = osc(freq, note_dur, sr, amp, t_offset=t_offset + local_sec)  # type: ignore[call-arg]
            note = apply_lowpass(note, cutoff, sr)
            note = apply_adsr(
                note, adsr[0] * attack_mult, adsr[1], adsr[2], adsr[3] * attack_mult, sr
            )
            note = apply_humanize(note, config.human, sr, rng=rng)
            add_note(signal, note, int(round(local_sec * sr)))
        slot_idx += 1

    signal = apply_delay(
        signal, delay_cfg[0], delay_cfg[1] * echo_mult, delay_cfg[2] * echo_mult, sr
    )
    return apply_reverb(
        signal, config.space * reverb_scale[0], config.space * reverb_scale[1], sr
    )


def melody_contemplative(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Slow, contemplative BPM-locked harmony-aware melody."""
    return _render_bpm_pattern_melody(
        config,
        duration,
        harmony,
        (0, -1, 2, -1, 4, -1, 2, -1, 0, -1, -1, -1, 2, -1, -1, -1),
        1.0,
        5,
        0.17,
        1.5,
        800.0 * config.brightness + 220.0,
        (0.08, 0.25, 0.6, 0.65),
        (0.35, 0.38, 0.28),
        (0.6, 0.8),
        rng,
        t_offset,
    )


def melody_rising(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Ascending melodic line locked to BPM and active chord."""
    return _render_bpm_pattern_melody(
        config,
        duration,
        harmony,
        (0, -1, 2, -1, 4, -1, -1, 5, -1, -1, 6, -1, 7, -1, -1, -1),
        1.0,
        5,
        0.16,
        1.7,
        900.0 * config.brightness + 250.0,
        (0.07, 0.22, 0.55, 0.75),
        (0.33, 0.34, 0.26),
        (0.55, 0.75),
        rng,
        t_offset,
    )


def melody_falling(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Descending melodic line locked to BPM and active chord."""
    return _render_bpm_pattern_melody(
        config,
        duration,
        harmony,
        (6, -1, -1, 5, -1, -1, 4, -1, 2, -1, -1, 0, -1, -1, -1, -1),
        1.0,
        5,
        0.16,
        1.6,
        850.0 * config.brightness + 220.0,
        (0.08, 0.25, 0.52, 0.75),
        (0.38, 0.36, 0.28),
        (0.58, 0.78),
        rng,
        t_offset,
    )


def melody_minimal(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Very sparse, minimal melody locked to BPM and active chord."""
    return _render_bpm_pattern_melody(
        config,
        duration,
        harmony,
        (4, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1),
        1.0,
        5,
        0.19,
        2.4,
        620.0 * config.brightness + 160.0,
        (0.12, 0.35, 0.5, 1.0),
        (0.50, 0.42, 0.35),
        (0.7, 0.9),
        rng,
        t_offset,
    )


def melody_ornamental(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Ornamental melody locked to BPM and active chord."""
    return _render_bpm_pattern_melody(
        config,
        duration,
        harmony,
        (2, 0, -1, -1, 5, 4, -1, -1, 3, 2, -1, -1, 2, 0, -1, -1),
        0.5,
        5,
        0.16,
        1.6,
        900.0 * config.brightness + 220.0,
        (0.04, 0.22, 0.52, 0.60),
        (0.30, 0.34, 0.25),
        (0.5, 0.7),
        rng,
        t_offset,
    )


def melody_arp(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Arpeggiated melody locked to BPM and active chord."""
    return _render_bpm_pattern_melody(
        config,
        duration,
        harmony,
        (0, 2, 4, 2),
        0.5,
        4,
        0.14,
        0.85,
        1150.0 * config.brightness + 280.0,
        (0.015, 0.08, 0.4, 0.14),
        (0.25, 0.30, 0.24),
        (0.45, 0.65),
        rng,
        t_offset,
    )


MELODY_PATTERNS: Mapping[str, PatternFn] = MappingProxyType(
    {
        "procedural": melody_procedural,
        "contemplative": melody_contemplative,
        "contemplative_minor": melody_contemplative,
        "rising": melody_rising,
        "falling": melody_falling,
        "minimal": melody_minimal,
        "ornamental": melody_ornamental,
        "arp_melody": melody_arp,
        "call_response": melody_contemplative,
        "heroic": melody_rising,
    }
)


# -----------------------------------------------------------------------------
# RHYTHM PATTERNS
# -----------------------------------------------------------------------------


def rhythm_none(
    _config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """No rhythm."""
    return np.zeros(int(SAMPLE_RATE * duration))


def _render_soft_kick(
    base_freq: float,
    kick_dur: float,
    amp: float,
    decay_rate: float,
    lowpass_hz: float,
    sr: int = SAMPLE_RATE,
) -> FloatArray:
    """Render a click-free phase-zero pitch-swept ambient kick drum."""
    n = _sample_count(kick_dur, sr)
    if n < 4:
        return np.zeros(n, dtype=np.float64)
    t = _time_axis(n, sr, 0.0)
    freq_inst = base_freq * (1.0 + 0.75 * np.exp(-t * 32.0))
    phase = 2.0 * math.pi * np.cumsum(freq_inst) / float(sr)
    env_decay = np.exp(-np.linspace(0.0, decay_rate, n, dtype=np.float64))
    kick = amp * np.sin(phase) * env_decay
    atk = min(n // 4, max(2, int(round(0.004 * sr))))
    rel = min(n // 3, max(2, int(round(0.015 * sr))))
    kick[:atk] *= _raised_cosine_ramp(atk, 0.0, 1.0)
    kick[-rel:] *= _raised_cosine_ramp(rel, 1.0, 0.0)
    kick = apply_lowpass(kick, lowpass_hz, sr)
    kick[0] = 0.0
    kick[-1] = 0.0
    return kick


def _rhythm_kick_grid(
    config: SynthConfig,
    duration: float,
    slots_per_beat: int,
    pattern: tuple[int, ...],
    base_freq: float,
    kick_dur: float,
    amp: float,
    decay_rate: float,
    lowpass_hz: float,
    rng: np.random.Generator,
    t_offset: float,
) -> FloatArray:
    """Render soft phase-zero kicks aligned to the global tempo grid."""
    sr = SAMPLE_RATE
    signal = np.zeros(_sample_count(duration, sr), dtype=np.float64)
    slot_times, slot_offset = _rhythm_slot_times(
        config, duration, slots_per_beat=slots_per_beat, t_offset=t_offset
    )
    for i, t_sec in enumerate(slot_times):
        if not pattern[(slot_offset + i) % len(pattern)]:
            continue
        vel = 1.0 + float(rng.uniform(-0.07, 0.07)) * config.human
        kick = _render_soft_kick(base_freq, kick_dur, amp * vel, decay_rate, lowpass_hz, sr)
        add_note(signal, kick, int(round(t_sec * sr)))
    return signal


def rhythm_minimal(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Minimal rhythm with soft phase-zero ambient kicks."""
    sig = _rhythm_kick_grid(
        config,
        duration,
        2,
        (1, 0, 0, 0, 0, 0, 0, 0),
        58.0,
        0.17,
        0.32,
        7.5,
        140.0,
        rng,
        t_offset,
    )
    return apply_reverb(sig, config.space * 0.25, config.space * 0.45, SAMPLE_RATE)


def rhythm_heartbeat(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Heartbeat-like double-hit ambient pulse."""
    return _rhythm_kick_grid(
        config,
        duration,
        2,
        (1, 1, 0, 0, 0, 0, 0, 0),
        54.0,
        0.14,
        0.30,
        9.0,
        120.0,
        rng,
        t_offset,
    )


def rhythm_soft_four(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Soft four-on-the-floor ambient kick pulse."""
    sig = _rhythm_kick_grid(
        config,
        duration,
        1,
        (1,),
        50.0,
        0.19,
        0.26,
        6.0,
        110.0,
        rng,
        t_offset,
    )
    return apply_reverb(sig, config.space * 0.22, config.space * 0.40, SAMPLE_RATE)


def rhythm_hats_only(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Soft brushed ambient hi-hats without harsh white-noise spikes."""
    sr = SAMPLE_RATE
    signal = np.zeros(_sample_count(duration, sr), dtype=np.float64)
    slot_times, _ = _rhythm_slot_times(config, duration, slots_per_beat=4, t_offset=t_offset)
    hat_n = _sample_count(0.045, sr)
    for t_sec in slot_times:
        vel = 1.0 + float(rng.uniform(-0.12, 0.12)) * config.human
        hat = rng.standard_normal(hat_n) * (0.026 * vel)
        hat = apply_lowpass(apply_highpass(hat, 5800.0, sr), 11500.0, sr)
        hat = apply_adsr(hat, 0.005, 0.015, 0.06, 0.025, sr)
        add_note(signal, hat, int(round(t_sec * sr)))
    return signal


def rhythm_electronic(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Electronic beat with soft phase-zero kicks and brushed off-beat hats."""
    sr = SAMPLE_RATE
    signal = np.zeros(_sample_count(duration, sr), dtype=np.float64)
    kicks = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
    hats = (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)
    slot_times, slot_offset = _rhythm_slot_times(
        config, duration, slots_per_beat=4, t_offset=t_offset
    )
    hat_n = _sample_count(0.05, sr)
    for i, t_sec in enumerate(slot_times):
        idx = (slot_offset + i) % len(kicks)
        pos = int(round(t_sec * sr))
        if kicks[idx]:
            add_note(signal, _render_soft_kick(46.0, 0.20, 0.30, 7.5, 110.0, sr), pos)
        if hats[idx]:
            hat = rng.standard_normal(hat_n) * 0.028
            hat = apply_lowpass(apply_highpass(hat, 6200.0, sr), 11500.0, sr)
            hat = apply_adsr(hat, 0.005, 0.014, 0.06, 0.028, sr)
            add_note(signal, hat, pos)
    return signal


RHYTHM_PATTERNS: Mapping[RhythmStyle, PatternFn] = MappingProxyType(
    {
        "none": rhythm_none,
        "minimal": rhythm_minimal,
        "heartbeat": rhythm_heartbeat,
        "soft_four": rhythm_soft_four,
        "hats_only": rhythm_hats_only,
        "electronic": rhythm_electronic,
        "kit_light": rhythm_minimal,
        "kit_medium": rhythm_soft_four,
        "military": rhythm_soft_four,
        "tabla_essence": rhythm_heartbeat,
        "brush": rhythm_minimal,
    }
)


# -----------------------------------------------------------------------------
# TEXTURE PATTERNS
# -----------------------------------------------------------------------------


def texture_none(
    _config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """No texture."""
    del t_offset  # unused
    return np.zeros(int(SAMPLE_RATE * duration))


def _texture_shimmer_core(
    config: SynthConfig,
    duration: float,
    degrees: tuple[int, ...],
    amp: float,
    base_rate: float,
    rate_span: float,
    delay_sec: float,
    fb_scale: float,
    wet_scale: float,
    reverb_room_scale: float,
    t_offset: float,
) -> FloatArray:
    sr = SAMPLE_RATE
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)
    signal = np.zeros(_sample_count(duration, sr), dtype=np.float64)
    lfo_rate = base_rate + rate_span * float(np.clip(config.motion, 0.0, 1.0))
    atk_sec = 0.025 if t_offset > 1e-6 else 0.35 * attack_mult
    rel_sec = 0.05
    for idx, deg in enumerate(degrees):
        freq = _scale_freq(config, deg, 6)
        tone = generate_sine(freq, duration, sr, amp, t_offset=t_offset)
        lfo = generate_lfo(duration, lfo_rate * (1.0 + 0.13 * idx), sr, t_offset=t_offset)
        tone = apply_adsr(tone * (0.45 + 0.55 * lfo), atk_sec, 0.0, 1.0, rel_sec, sr)
        signal += tone
    signal = apply_delay(signal, delay_sec, fb_scale * echo_mult, wet_scale * echo_mult, sr)
    return apply_reverb(signal, config.space * reverb_room_scale, config.space, sr)


def texture_shimmer(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """High, shimmering texture with proportional motion and seamless t_offset."""
    return _texture_shimmer_core(
        config, duration, (0, 2, 4), 0.030, 0.35, 1.6, 0.4, 0.45, 0.30, 0.75, t_offset
    )


def texture_shimmer_slow(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Slow, gentle shimmer with proportional motion and seamless t_offset."""
    return _texture_shimmer_core(
        config, duration, (0, 4), 0.026, 0.12, 0.50, 0.5, 0.48, 0.34, 0.80, t_offset
    )


def texture_vinyl_crackle(
    _config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Gentle Hann-windowed vinyl dust without harsh unwindowed clicks."""
    sr = SAMPLE_RATE
    n = _sample_count(duration, sr)
    signal = np.zeros(n, dtype=np.float64)
    grain_n = max(24, int(round(0.004 * sr)))
    window = np.hanning(grain_n)
    for _ in range(int(duration * 6)):
        if n <= grain_n + 2:
            break
        pos = int(rng.integers(0, n - grain_n - 1))
        grain = rng.standard_normal(grain_n) * float(rng.uniform(0.002, 0.008))
        grain = apply_lowpass(apply_highpass(grain, 1600.0, sr), 5500.0, sr) * window
        add_note(signal, grain, pos)
    air = rng.standard_normal(n) * 0.0018
    air = apply_lowpass(apply_highpass(air, 700.0, sr), 4200.0, sr)
    atk_sec = 0.025 if t_offset > 1e-6 else 0.25
    return apply_adsr(signal + air, atk_sec, 0.0, 1.0, 0.05, sr)


def texture_breath(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Warm breathing air texture with proportional LFO rate."""
    sr = SAMPLE_RATE
    n = _sample_count(duration, sr)
    signal = rng.standard_normal(n) * 0.032
    freq = _scale_freq(config, 0, 3)
    signal = apply_lowpass(signal, min(freq * 2.2, 1600.0), sr)
    signal = apply_highpass(signal, max(freq * 0.5, 90.0), sr)
    breath_rate = 0.08 + 0.24 * float(np.clip(config.motion, 0.0, 1.0))
    lfo = generate_lfo(duration, breath_rate, sr, t_offset=t_offset)
    atk_sec = 0.025 if t_offset > 1e-6 else 0.35
    signal = apply_adsr(signal * lfo, atk_sec, 0.0, 1.0, 0.05, sr)
    return apply_reverb(signal, config.space * 0.6, config.space * 0.8, sr)


def texture_stars(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Twinkling stars texture."""
    sr = SAMPLE_RATE
    dur = duration
    echo_mult = _echo_mult(config)

    signal = np.zeros(int(sr * dur))

    # Random high plinks
    num_stars = int(dur * 3)  # ~3 per second

    # Scale degrees for stars (tuple)
    star_degrees = (0, 2, 4, 5)

    for _ in range(num_stars):
        pos = int(rng.integers(0, max(1, len(signal) - sr)))

        degree = int(rng.choice(star_degrees))
        freq = _scale_freq(config, degree, 6)

        start_sec = pos / sr
        star = generate_sine(
            freq, 0.3, sr, float(rng.uniform(0.02, 0.05)), t_offset=t_offset + start_sec
        )
        star = apply_adsr(star, 0.01, 0.1, 0.1, 0.2, sr)

        add_note(signal, star, pos)

    signal = apply_delay(signal, 0.4, 0.5 * echo_mult, 0.4 * echo_mult, sr)
    signal = apply_reverb(signal, config.space * 0.9, config.space, sr)

    return signal


TEXTURE_PATTERNS: Mapping[TextureStyle, PatternFn] = MappingProxyType(
    {
        "none": texture_none,
        "shimmer": texture_shimmer,
        "shimmer_slow": texture_shimmer_slow,
        "vinyl_crackle": texture_vinyl_crackle,
        "breath": texture_breath,
        "stars": texture_stars,
        "glitch": texture_shimmer,
        "noise_wash": texture_breath,
        "crystal": texture_stars,
        "pad_whisper": texture_breath,
    }
)


# -----------------------------------------------------------------------------
# ACCENT PATTERNS
# -----------------------------------------------------------------------------


def accent_none(
    _config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """No accent."""
    del t_offset  # unused
    return np.zeros(int(SAMPLE_RATE * duration))


def accent_bells(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """BPM-locked harmony-aware bell accents."""
    sr = SAMPLE_RATE
    attack_mult = _attack_mult(config)
    spb = _seconds_per_beat(config)
    signal = np.zeros(_sample_count(duration, sr), dtype=np.float64)
    pattern = (0, -1, -1, -1, 4, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1)
    slot_idx = _first_grid_slot(t_offset, spb)
    while True:
        local_sec = slot_idx * spb - t_offset
        if local_sec >= duration:
            break
        rel_deg = pattern[slot_idx % len(pattern)]
        if rel_deg >= 0:
            chord_root = _chord_root_degree_at_beat(local_sec / spb, harmony)
            freq = _scale_freq(config, chord_root + rel_deg, 5)
            bell_dur = min(1.1, max(0.12, duration - local_sec))
            bell = (
                generate_sine(freq, bell_dur, sr, 0.11, 0.0)
                + generate_sine(freq * 2.0, bell_dur, sr, 0.042, 0.0)
                + generate_sine(freq * 3.0, bell_dur, sr, 0.018, 0.0)
            )
            bell = apply_adsr(bell, 0.008 * attack_mult, 0.18, 0.08, 0.65 * attack_mult, sr)
            bell = apply_humanize(bell, config.human, sr, rng=rng)
            add_note(signal, bell, int(round(local_sec * sr)))
        slot_idx += 1
    return apply_reverb(signal, config.space * 0.6, config.space * 0.8, sr)


def accent_pluck(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """BPM-locked harmony-aware plucked string accents."""
    sr = SAMPLE_RATE
    attack_mult = _attack_mult(config)
    spb = _seconds_per_beat(config)
    slot_sec = spb * 0.5
    signal = np.zeros(_sample_count(duration, sr), dtype=np.float64)
    pattern = (0, -1, -1, 4, -1, -1, 2, -1, -1, -1, 0, -1, -1, -1, -1, -1)
    slot_idx = _first_grid_slot(t_offset, slot_sec)
    while True:
        local_sec = slot_idx * slot_sec - t_offset
        if local_sec >= duration:
            break
        rel_deg = pattern[slot_idx % len(pattern)]
        if rel_deg >= 0:
            chord_root = _chord_root_degree_at_beat(local_sec / spb, harmony)
            freq = _scale_freq(config, chord_root + rel_deg, 4)
            pluck_dur = min(0.55, max(0.08, duration - local_sec))
            pluck = generate_triangle(freq, pluck_dur, sr, 0.14, 0.0)
            pluck = apply_lowpass(pluck, 1500.0 * config.brightness + 400.0, sr)
            pluck = apply_adsr(pluck, 0.006 * attack_mult, 0.14, 0.04, 0.36 * attack_mult, sr)
            pluck = apply_humanize(pluck, config.human, sr, rng=rng)
            add_note(signal, pluck, int(round(local_sec * sr)))
        slot_idx += 1
    return apply_reverb(signal, config.space * 0.5, config.space * 0.7, sr)


def accent_chime(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Wind chime accents."""
    sr = SAMPLE_RATE
    dur = duration
    echo_mult = _echo_mult(config)

    signal = np.zeros(int(sr * dur))

    # Random chime hits
    num_chimes = int(dur * 1.5)

    # Chime degrees (tuple)
    chime_degrees = (0, 2, 4, 5, 6)

    for _ in range(num_chimes):
        max_start = len(signal) - sr
        if max_start <= 0:
            pos = 0
        else:
            pos = int(rng.integers(0, max_start))

        degree = int(rng.choice(chime_degrees))
        freq = _scale_freq(config, degree, 5)

        chime_dur = 1.2
        start_sec = pos / sr
        chime = generate_sine(
            freq, chime_dur, sr, float(rng.uniform(0.06, 0.12)), t_offset=t_offset + start_sec
        )
        chime += generate_sine(freq * 2.0, chime_dur, sr, 0.03, t_offset=t_offset + start_sec)
        chime = apply_adsr(chime, 0.002, 0.3, 0.05, 0.9, sr)

        add_note(signal, chime, pos)

    signal = apply_delay(signal, 0.3, 0.4 * echo_mult, 0.3 * echo_mult, sr)
    signal = apply_reverb(signal, config.space * 0.75, config.space * 0.9, sr)

    return signal


ACCENT_PATTERNS: Mapping[AccentStyle, PatternFn] = MappingProxyType(
    {
        "none": accent_none,
        "bells": accent_bells,
        "bells_dense": accent_bells,
        "pluck": accent_pluck,
        "chime": accent_chime,
        "blip": accent_bells,
        "blip_random": accent_chime,
        "brass_hit": accent_bells,
        "wind": accent_chime,
        "arp_accent": accent_pluck,
        "piano_note": accent_pluck,
    }
)


def build_melody_policy(config: SynthConfig) -> MelodyPolicy:
    """Create a MelodyPolicy from config fields."""
    return MelodyPolicy(
        phrase_len_bars=int(max(1, min(16, config.phrase_len_bars))),
        density=float(np.clip(config.melody_density, 0.0, 1.0)),
        syncopation=float(np.clip(config.syncopation, 0.0, 1.0)),
        swing=float(np.clip(config.swing, 0.0, 1.0)),
        motif_repeat_prob=float(np.clip(config.motif_repeat_prob, 0.0, 1.0)),
        step_bias=float(np.clip(config.step_bias, 0.0, 1.0)),
        chromatic_prob=float(np.clip(config.chromatic_prob, 0.0, 1.0)),
        cadence_strength=float(np.clip(config.cadence_strength, 0.0, 1.0)),
        register_min_oct=int(config.register_min_oct),
        register_max_oct=int(config.register_max_oct),
        tension_curve=config.tension_curve,
    )


_HARMONY_TEMPLATES: Mapping[str, tuple[tuple[int, ...], ...]] = MappingProxyType(
    {
        "pop_major": ((0, 5, 3, 4), (0, 3, 4, 0), (0, 4, 5, 3)),
        "pop_minor": ((0, 5, 3, 6), (0, 3, 6, 4), (0, 6, 3, 5), (0, 4, 5, 3)),
        "jazz_major": ((1, 4, 0, 0), (1, 4, 0, 5), (5, 1, 4, 0)),
        "jazz_minor": ((1, 4, 0, 0), (5, 1, 4, 0)),
        "cinematic_minor": ((0, 6, 5, 3), (0, 5, 6, 4), (0, 6, 3, 4)),
        "ambient": ((0, 5, 0, 3), (0, 4, 0, 6), (0, 3, 0, 5)),
    }
)


def _derive_seed(base_seed: int, salt: int) -> int:
    """Mix a 31-bit deterministic sub-seed from base_seed and salt."""
    x = ((base_seed & 0xFFFFFFFF) ^ ((salt * 0x9E3779B1) & 0xFFFFFFFF)) & 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
    return (x ^ (x >> 16)) & 0x7FFFFFFF


def _select_harmony_template(
    config: SynthConfig, rng: np.random.Generator
) -> tuple[int, ...]:
    """Deterministically select a 4-chord diatonic progression template."""
    style = (config.harmony_style or "auto").lower()
    if style == "auto":
        if config.rhythm == "none" and config.space >= 0.7:
            style = "ambient"
        elif config.pad == "cinematic" or config.bass in ("drone", "sub_pulse"):
            style = "cinematic"
        elif config.texture == "vinyl_crackle" or config.melody == "ornamental":
            style = "jazz"
        else:
            style = "pop"
    majorish = (config.mode or "minor").lower() in ("major", "mixolydian")
    if style == "ambient":
        key = "ambient"
    elif style == "jazz":
        key = "jazz_major" if majorish else "jazz_minor"
    elif style == "cinematic":
        key = "pop_major" if majorish else "cinematic_minor"
    else:
        key = "pop_major" if majorish else "pop_minor"
    choices = _HARMONY_TEMPLATES[key]
    return choices[int(rng.integers(0, len(choices)))]


def build_harmony_plan(
    config: SynthConfig,
    duration: float,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> HarmonyPlan:
    """Build a continuous diatonic chord timeline aligned to global t_offset."""
    beats_per_bar = 4
    spb = _seconds_per_beat(config)
    chord_beats = float(max(1, int(config.chord_change_bars)) * beats_per_bar)
    template = _select_harmony_template(config, rng)

    start_beat_g = max(0.0, t_offset / spb)
    end_beat_g = max(start_beat_g + 1e-4, (t_offset + duration) / spb)
    first_idx = int(math.floor(start_beat_g / chord_beats))
    last_idx = max(first_idx + 1, int(math.ceil(end_beat_g / chord_beats)))

    chords: list[ChordEvent] = []
    for idx in range(first_idx, last_idx):
        seg_start_g = max(start_beat_g, idx * chord_beats)
        seg_end_g = min(end_beat_g, (idx + 1) * chord_beats)
        seg_dur = seg_end_g - seg_start_g
        if seg_dur <= 1e-6:
            continue
        chords.append(
            ChordEvent(
                start_beat=seg_start_g - start_beat_g,
                duration_beats=seg_dur,
                root_degree=int(template[idx % len(template)]) % 7,
            )
        )

    return HarmonyPlan(chords=tuple(chords), beats_per_bar=beats_per_bar)


@lru_cache(maxsize=12)
def _master_reverb_ir(size_bucket: int, sr: int = SAMPLE_RATE) -> FloatArray:
    """Build frequency-peak-calibrated ambient hall IR (<= 1.50 s)."""
    size = float(size_bucket) / 10.0
    rt60 = 0.9 + 1.8 * size
    ir_len = int(sr * min(1.50, rt60 * 0.70))
    rng = np.random.default_rng(20261125 + size_bucket)
    t = np.arange(ir_len, dtype=np.float64) / float(sr)
    decay = np.power(10.0, -3.0 * t / rt60)
    raw = rng.standard_normal(ir_len) * decay

    predelay = int(0.022 * sr)
    onset = int(0.030 * sr)
    ir = np.zeros(ir_len, dtype=np.float64)
    if predelay < ir_len:
        body = raw[: ir_len - predelay].copy()
        rise = min(len(body), onset)
        if rise >= 2:
            body[:rise] *= _raised_cosine_ramp(rise, 0.0, 1.0)
        ir[predelay:] = body

    tail_n = min(ir_len // 4, int(0.18 * sr))
    if tail_n >= 2:
        ir[-tail_n:] *= _raised_cosine_ramp(tail_n, 1.0, 0.0)

    ir = apply_highpass(ir, 180.0, sr)
    ir = apply_lowpass(ir, 3600.0 - 1500.0 * size, sr)

    fft_len = 1 << int(math.ceil(math.log2(max(256, ir_len * 2))))
    spec = np.abs(np.fft.rfft(ir, n=fft_len))
    rms_resp = float(np.sqrt(np.mean(spec * spec)))
    peak_resp = float(np.max(spec))
    norm = max(rms_resp, peak_resp * 0.55) + 1e-12
    return np.asarray(ir / norm, dtype=np.float64)


def _apply_master_hall_reverb(
    signal: FloatArray,
    space: float,
    sr: int = SAMPLE_RATE,
) -> FloatArray:
    """Convolve master mix with a calibrated warm ambient hall IR via NumPy FFT."""
    safe_space = float(np.clip(space, 0.0, 1.0))
    n = len(signal)
    if n < 64 or safe_space <= 0.12:
        return signal

    bucket = max(1, min(10, int(round(safe_space * 10.0))))
    ir = _master_reverb_ir(bucket, sr)
    fft_n = 1 << int(math.ceil(math.log2(n + len(ir) - 1)))
    wet_full = np.fft.irfft(
        np.fft.rfft(signal, fft_n) * np.fft.rfft(ir, fft_n),
        fft_n,
    )
    wet = wet_full[:n]
    wet_gain = 0.08 + 0.22 * safe_space
    dry_gain = 1.0 - 0.14 * safe_space
    return np.asarray(signal * dry_gain + wet * wet_gain, dtype=np.float64)


def _render_layer_group(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan,
    melody_policy: MelodyPolicy,
    session_seed: int,
    t_offset: float,
) -> FloatArray:
    """Sum all active instrument layers with continuous harmony and stable seeds."""
    active_layers = DENSITY_LAYERS[config.density]
    block_salt = int(round(t_offset * 10.0)) * 31
    out = np.zeros(_sample_count(duration, SAMPLE_RATE), dtype=np.float64)

    if config.depth:
        sub_rng = np.random.default_rng(_derive_seed(session_seed, 17 + block_salt))
        out += bass_sub_pulse(config, duration, harmony, melody_policy, sub_rng, t_offset) * 0.6

    if "bass" in active_layers:
        bass_fn = BASS_PATTERNS.get(config.bass, bass_drone)
        bass_rng = np.random.default_rng(_derive_seed(session_seed, 11 + block_salt))
        out += bass_fn(config, duration, harmony, melody_policy, bass_rng, t_offset)

    if "pad" in active_layers:
        pad_fn = PAD_PATTERNS.get(config.pad, pad_warm_slow)
        pad_rng = np.random.default_rng(_derive_seed(session_seed, 23 + block_salt))
        out += pad_fn(config, duration, harmony, melody_policy, pad_rng, t_offset)

    if "melody" in active_layers:
        if config.melody_engine == "procedural":
            mel_rng = np.random.default_rng(_derive_seed(session_seed, 101))
            out += melody_procedural(config, duration, harmony, melody_policy, mel_rng, t_offset)
        else:
            mel_fn = MELODY_PATTERNS.get(config.melody, melody_contemplative)
            mel_rng = np.random.default_rng(_derive_seed(session_seed, 41 + block_salt))
            out += mel_fn(config, duration, harmony, melody_policy, mel_rng, t_offset)

    if "rhythm" in active_layers and config.rhythm != "none":
        rhythm_fn = RHYTHM_PATTERNS.get(config.rhythm, rhythm_none)
        rhythm_rng = np.random.default_rng(_derive_seed(session_seed, 53 + block_salt))
        out += rhythm_fn(config, duration, harmony, melody_policy, rhythm_rng, t_offset)

    if "texture" in active_layers and config.texture != "none":
        texture_fn = TEXTURE_PATTERNS.get(config.texture, texture_none)
        texture_rng = np.random.default_rng(_derive_seed(session_seed, 67 + block_salt))
        out += texture_fn(config, duration, harmony, melody_policy, texture_rng, t_offset)

    if "accent" in active_layers and config.accent != "none":
        accent_fn = ACCENT_PATTERNS.get(config.accent, accent_none)
        accent_rng = np.random.default_rng(_derive_seed(session_seed, 79 + block_salt))
        out += accent_fn(config, duration, harmony, melody_policy, accent_rng, t_offset)

    return out


# =============================================================================
# PART 3: ASSEMBLER - CONFIG → AUDIO
# =============================================================================


def assemble(
    config: SynthConfig,
    duration: float = 16.0,
    normalize: bool = True,
    rng: np.random.Generator | None = None,
    t_offset: float = 0.0,
) -> FloatArray:
    """Assemble all active layers into headroom-safe, click-free mono audio.

    Args:
        config: Numeric SynthConfig.
        duration: Output duration in seconds.
        normalize: If True, peak-normalizes to 0.78; set False for streaming
            chunks so dynamics remain consistent across blocks.
        rng: Optional RNG for deterministic generation.
        t_offset: Time offset in seconds for phase, harmony, and melody continuity.
    """
    sr = SAMPLE_RATE
    local_rng = rng if rng is not None else np.random.default_rng()
    session_seed = int(local_rng.integers(0, 0x7FFFFFFF))
    harmony_rng = np.random.default_rng(_derive_seed(session_seed, 101))
    harmony = build_harmony_plan(config, duration, harmony_rng, t_offset)
    melody_policy = build_melody_policy(config)

    output = _render_layer_group(config, duration, harmony, melody_policy, session_seed, t_offset)
    output = apply_stereo_width(output, config.stereo, sr)
    output = _apply_master_hall_reverb(output, config.space, sr)
    output = apply_highpass(output, 20.0, sr)
    output = soft_limit(output, ceiling=0.92)

    if normalize and output.size > 0:
        max_val = float(np.max(np.abs(output)))
        if max_val > 1e-6:
            output = output * (0.78 / max_val)

    fade_sec = 0.008 if t_offset > 1e-6 else 0.055
    return soft_fade_edges(output, fade_in_sec=fade_sec, fade_out_sec=fade_sec, sr=sr)


def config_to_audio(config: SynthConfig, output_path: str, duration: float = 16.0) -> str:
    """
    Convert a SynthConfig to an audio file.

    Args:
        config: SynthConfig object
        output_path: Path to save the WAV file
        duration: Duration in seconds

    Returns:
        Path to the saved file
    """
    audio = assemble(config, duration)
    assert callable(sf.write)
    write_audio = sf.write
    write_audio(output_path, audio, SAMPLE_RATE)  # type: ignore[reportUnknownMemberType]
    return output_path


def dict_to_audio(
    config_dict: Mapping[str, Any],
    output_path: str,
    duration: float = 16.0,
) -> str:
    """
    Convert a config dict (from JSON) directly to audio.

    Args:
        config_dict: Mapping with config values
        output_path: Path to save the WAV file
        duration: Duration in seconds

    Returns:
        Path to the saved file
    """
    config = SynthConfig.from_dict(config_dict)
    return config_to_audio(config, output_path, duration)


# # =============================================================================
# # PART 4: CONVENIENCE FUNCTIONS
# # =============================================================================


# def generate_from_vibe(vibe: str, output_path: str = "output.wav", duration: float = 16.0) -> str:
#     """
#     Generate audio from a vibe description.

#     This is a placeholder - in production, this would call an LLM to generate the config.
#     For now, uses simple keyword matching.
#     """
#     vibe_lower = vibe.lower()

#     # Simple keyword-based config generation
#     config = SynthConfig()

#     # Root/mode selection
#     if any(w in vibe_lower for w in ("dark", "sad", "night", "mysterious")):
#         config.root = "d"
#         config.mode = "dorian"
#         config.brightness = 0.35
#     elif any(w in vibe_lower for w in ("happy", "bright", "joy")):
#         config.root = "c"
#         config.mode = "major"
#         config.brightness = 0.65
#     elif any(w in vibe_lower for w in ("epic", "cinematic", "powerful")):
#         config.root = "d"
#         config.mode = "minor"
#         config.brightness = 0.5
#         config.depth = True
#     elif any(w in vibe_lower for w in ("indian", "spiritual", "meditation")):
#         config.root = "d"
#         config.mode = "dorian"
#         config.human = 0.2
#         config.grain = "warm"
#     else:
#         config.root = "c"
#         config.mode = "minor"

#     # Tempo
#     if any(w in vibe_lower for w in ("slow", "calm", "meditation")):
#         config.tempo = 0.30
#     elif any(w in vibe_lower for w in ("fast", "energy", "drive")):
#         config.tempo = 0.45
#     else:
#         config.tempo = 0.36

#     # Space
#     if any(w in vibe_lower for w in ("vast", "space", "underwater", "cave")):
#         config.space = 0.85
#         config.echo = 0.75
#     elif any(w in vibe_lower for w in ("intimate", "close")):
#         config.space = 0.4
#         config.echo = 0.3

#     # Layers based on keywords
#     if any(w in vibe_lower for w in ("electronic", "synth")):
#         config.rhythm = "electronic"
#         config.attack = "sharp"
#     elif any(w in vibe_lower for w in ("ambient", "peaceful")):
#         config.rhythm = "none"
#         config.attack = "soft"

#     if any(w in vibe_lower for w in ("minimal", "sparse")):
#         config.density = 3
#         config.melody = "minimal"
#     elif any(w in vibe_lower for w in ("rich", "full", "lush")):
#         config.density = 6

#     return config_to_audio(config, output_path, duration)


# =============================================================================
# PART 5: TRANSITION FUNCTIONS
# =============================================================================


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: t=0 returns a, t=1 returns b"""
    return a + (b - a) * t


def _clamp_density(value: float) -> DensityLevel:
    raw = int(round(value))
    match raw:
        case 2:
            return 2
        case 3:
            return 3
        case 4:
            return 4
        case 5:
            return 5
        case 6:
            return 6
        case _:
            return 2 if raw < 2 else 6


def interpolate_configs(config_a: SynthConfig, config_b: SynthConfig, t: float) -> SynthConfig:
    """
    Interpolate between two configs.
    t=0.0 → config_a
    t=1.0 → config_b
    """
    return SynthConfig(
        tempo=lerp(config_a.tempo, config_b.tempo, t),
        root=config_a.root if t < 0.5 else config_b.root,
        mode=config_a.mode if t < 0.5 else config_b.mode,
        brightness=lerp(config_a.brightness, config_b.brightness, t),
        space=lerp(config_a.space, config_b.space, t),
        density=_clamp_density(lerp(config_a.density, config_b.density, t)),
        # Layer selections: staggered switching
        bass=config_a.bass if t < 0.4 else config_b.bass,
        pad=config_a.pad if t < 0.5 else config_b.pad,
        melody=config_a.melody if t < 0.6 else config_b.melody,
        rhythm=config_a.rhythm if t < 0.5 else config_b.rhythm,
        texture=config_a.texture if t < 0.7 else config_b.texture,
        accent=config_a.accent if t < 0.8 else config_b.accent,
        # V2 parameters
        motion=lerp(config_a.motion, config_b.motion, t),
        attack=config_a.attack if t < 0.5 else config_b.attack,
        stereo=lerp(config_a.stereo, config_b.stereo, t),
        depth=config_a.depth if t < 0.5 else config_b.depth,
        echo=lerp(config_a.echo, config_b.echo, t),
        human=lerp(config_a.human, config_b.human, t),
        grain=config_a.grain if t < 0.5 else config_b.grain,
        melody_engine=config_a.melody_engine if t < 0.5 else config_b.melody_engine,
        phrase_len_bars=round(lerp(config_a.phrase_len_bars, config_b.phrase_len_bars, t)),
        melody_density=lerp(config_a.melody_density, config_b.melody_density, t),
        syncopation=lerp(config_a.syncopation, config_b.syncopation, t),
        swing=lerp(config_a.swing, config_b.swing, t),
        motif_repeat_prob=lerp(config_a.motif_repeat_prob, config_b.motif_repeat_prob, t),
        step_bias=lerp(config_a.step_bias, config_b.step_bias, t),
        chromatic_prob=lerp(config_a.chromatic_prob, config_b.chromatic_prob, t),
        cadence_strength=lerp(config_a.cadence_strength, config_b.cadence_strength, t),
        register_min_oct=round(lerp(config_a.register_min_oct, config_b.register_min_oct, t)),
        register_max_oct=round(lerp(config_a.register_max_oct, config_b.register_max_oct, t)),
        tension_curve=config_a.tension_curve if t < 0.5 else config_b.tension_curve,
        harmony_style=config_a.harmony_style if t < 0.5 else config_b.harmony_style,
        chord_change_bars=round(lerp(config_a.chord_change_bars, config_b.chord_change_bars, t)),
        chord_extensions=config_a.chord_extensions if t < 0.5 else config_b.chord_extensions,
    )


def morph_audio(
    config_a: SynthConfig, config_b: SynthConfig, duration: float = 60.0, segments: int = 8
) -> FloatArray:
    """
    Generate audio that morphs from config_a to config_b over duration.
    """
    segment_duration = duration / segments
    output = []

    for i in range(segments):
        t = i / (segments - 1) if segments > 1 else 0.0
        interpolated = interpolate_configs(config_a, config_b, t)
        # Don't normalize individual segments; preserve relative dynamics
        segment = assemble(interpolated, segment_duration, normalize=False)
        output.append(segment)

    result = np.concatenate(output)

    # Normalize the final result
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val * 0.85

    return result


def crossfade(audio_a: FloatArray, audio_b: FloatArray, crossfade_samples: int) -> FloatArray:
    """Raised-cosine crossfade between two audio arrays at the midpoint."""
    min_len = min(len(audio_a), len(audio_b))
    mid = min_len // 2
    half_cf = min(crossfade_samples // 2, mid)
    cf_len = half_cf * 2
    if cf_len < 2:
        return np.concatenate((audio_a[:mid], audio_b[mid:min_len]))

    fade_in = _raised_cosine_ramp(cf_len, 0.0, 1.0)
    fade_out = 1.0 - fade_in

    result = np.concatenate(
        (
            audio_a[: mid - half_cf],
            audio_a[mid - half_cf : mid + half_cf] * fade_out
            + audio_b[mid - half_cf : mid + half_cf] * fade_in,
            audio_b[mid + half_cf : min_len],
        )
    )

    return result


def transition(config_a: SynthConfig, config_b: SynthConfig, duration: float = 60.0) -> FloatArray:
    """Generate with crossfade transition."""
    # Don't normalize individual tracks; normalize after crossfade
    audio_a = assemble(config_a, duration, normalize=False)
    audio_b = assemble(config_b, duration, normalize=False)

    crossfade_duration = 4.0  # seconds
    crossfade_samples = int(crossfade_duration * SAMPLE_RATE)

    result = crossfade(audio_a, audio_b, crossfade_samples)

    # Normalize the final result
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val * 0.85

    return result


def generate_tween_with_automation(
    config_a: SynthConfig,
    config_b: SynthConfig,
    duration: float = 120.0,
    chunk_seconds: float = 2.0,
    overlap_seconds: float = 0.05,
) -> FloatArray:
    """
    Generate audio with automated parameters using Cached Block Processing.

    Performance: ~8x faster than per-chunk generation.
    Trade-off: Automation updates occur every 16s (Pattern Length) instead of every 2s.
    """
    sr = SAMPLE_RATE
    num_chunks = int(np.ceil(duration / chunk_seconds))  # ceil ensures we cover full duration
    overlap_samples = int(overlap_seconds * sr)

    PATTERN_LEN = 16.0
    chunks_per_pattern = int(PATTERN_LEN / chunk_seconds)

    chunk_len_sec = chunk_seconds + overlap_seconds
    chunk_samples = int(chunk_len_sec * sr)

    # Initialize output buffer
    output = np.zeros(int(duration * sr))

    cached_pattern: FloatArray | None = None
    cached_pattern_idx: int = -1

    for i in range(num_chunks):
        # 1. Determine which 16s Block we are in
        pattern_idx = i // chunks_per_pattern

        # 2. Check Cache
        if pattern_idx != cached_pattern_idx:
            # Interpolate parameters for this specific 16s block
            # Note: This "steps" the parameters every 16s.
            t = (pattern_idx * chunks_per_pattern) / max(1, num_chunks - 1)
            t = np.clip(t, 0.0, 1.0)
            t_eased = 0.5 - 0.5 * np.cos(t * np.pi)

            config = interpolate_configs(config_a, config_b, t_eased)

            # Generate the FULL 16s block
            # normalize=False preserves relative dynamics between blocks
            cached_pattern = assemble(config, PATTERN_LEN, normalize=False)
            cached_pattern_idx = pattern_idx

        # Ensure cache exists (typing safety)
        if cached_pattern is None:
            continue

        # 3. Slice the 2s window + overlap
        local_chunk_idx = i % chunks_per_pattern

        # Calculate indices relative to the cached pattern
        start_idx = int(local_chunk_idx * chunk_seconds * sr)
        end_idx = start_idx + chunk_samples

        # Handle wrapping (Circular Buffer logic)
        pattern_len_samples = len(cached_pattern)

        if end_idx <= pattern_len_samples:
            chunk = cached_pattern[start_idx:end_idx].copy()
        else:
            # Wrap around to start
            # part_a: from start_idx to end of buffer
            # part_b: from 0 to remainder
            part_a = cached_pattern[start_idx:]
            remainder = end_idx - pattern_len_samples
            part_b = cached_pattern[:remainder]
            chunk = np.concatenate((part_a, part_b))

        # 4. Apply Crossfade Envelopes (Overlap-Add)
        if i > 0:
            chunk[:overlap_samples] *= np.linspace(0.0, 1.0, overlap_samples)

        if i < num_chunks - 1:
            chunk[-overlap_samples:] *= np.linspace(1.0, 0.0, overlap_samples)

        # 5. Add to Main Output
        out_start = int(i * chunk_seconds * sr)
        out_end = min(out_start + len(chunk), len(output))
        available = out_end - out_start

        if available > 0:
            output[out_start:out_end] += chunk[:available]

    # Global Normalization
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 0.85

    return output


# =============================================================================
# MAIN - Demo
# =============================================================================

# if __name__ == "__main__":
#     print("VibeSynth V1/V2 - Pure Python Synthesis Engine")
#     print("=" * 50)

#     # Example 3: Bubblegum, sad, dead (from llm_to_synth.py, see @file_context_1)
#     import json

#     # These examples mirror the demo config blocks captured in @file_context_0:
#     demo_configs = [
#         {
#             # "thinking": "The 'Bubblegum'-style sound is characterized by a bright, slightly distorted, and playful melody with a prominent bass line and a smooth, evolving pad.",
#             "tempo": 0.5,
#             "root": "a",
#             "mode": "major",
#             "brightness": 0.75,
#             "space": 0.5,
#             "density": 2,
#             "bass": "drone",
#             "pad": "warm_slow",
#             "melody": "contemplative",
#             "rhythm": "none",
#             "texture": "shimmer",
#             "accent": "bells",
#             "motion": 0.5,
#             "attack": "medium",
#             "stereo": 0.5,
#             "depth": False,
#             "echo": 0.25,
#             "human": 0.5,
#             "grain": "clean",
#         },
#         {
#             # "thinking": "The sad vibe is characterized by a low-key sound and a sense of melancholy. The low intensity and muted tones create a feeling of quiet sorrow. The 'low' intensity of the bass and the 'dark' tone of the pad support this feeling. The 'soft' attack and 'warm' tone of the pad create a sense of longing and a feeling of sadness.",
#             "tempo": 0.25,
#             "root": "e",
#             "mode": "minor",
#             "brightness": 0.25,
#             "space": 0.25,
#             "density": 5,
#             "bass": "drone",
#             "pad": "warm_slow",
#             "melody": "contemplative",
#             "rhythm": "soft_four",
#             "texture": "shimmer_slow",
#             "accent": "bells",
#             "motion": 0.25,
#             "attack": "soft",
#             "stereo": 0.25,
#             "depth": True,
#             "echo": 0.5,
#             "human": 0.5,
#             "grain": "gritty",
#         },
#         {
#             # "thinking": "The 'dead', a low volume synth with a simple sine wave and a low pitch.",
#             "tempo": 0.25,
#             "root": "c",
#             "mode": "minor",
#             "brightness": 0.25,
#             "space": 0.25,
#             "density": 2,
#             "bass": "drone",
#             "pad": "warm_slow",
#             "melody": "contemplative",
#             "rhythm": "none",
#             "texture": "shimmer_slow",
#             "accent": "bells",
#             "motion": 0.25,
#             "attack": "soft",
#             "stereo": 0.25,
#             "depth": False,
#             "echo": 0.25,
#             "human": 0.25,
#             "grain": "clean",
#         },
#     ]
#     demo_names = ["bubblegum.wav", "sad.wav", "dead.wav"]
#     demo_vibes = ["Bubblegum", "sad", "dead"]

#     for vibe, fname, config_dict in zip(demo_vibes, demo_names, demo_configs):
#         print(f"\n{'=' * 60}")
#         print(f"Vibe: {vibe}")
#         print("=" * 60)
#         config = SynthConfig.from_dict(config_dict)
#         print(json.dumps(config_dict, indent=2))
#         config_to_audio(config, fname, duration=20.0)
#         print(f"   Saved: {fname}")

#     print("\nAll demo audio exported.")


# if __name__ == "__main__":
#     print("VibeSynth V1/V2 - Pure Python Synthesis Engine")
#     print("=" * 50)

#     # Example 1: Direct config
#     print("\n1. Generating from direct config (Indian Wedding)...")
#     config = SynthConfig(
#         tempo=0.36,
#         root="d",
#         mode="dorian",
#         brightness=0.5,
#         space=0.75,
#         density=5,
#         bass="drone",
#         pad="warm_slow",
#         melody="ornamental",
#         rhythm="minimal",
#         texture="shimmer_slow",
#         accent="pluck",
#         motion=0.5,
#         attack="soft",
#         stereo=0.65,
#         depth=True,
#         echo=0.55,
#         human=0.18,
#         grain="warm",
#     )
#     config_to_audio(config, "indian_wedding.wav", duration=20.0)
#     print("   Saved: indian_wedding.wav")

#     # Example 2: From dict (simulating JSON from LLM)
#     print("\n2. Generating from dict (Dark Electronic)...")
#     dark_config = {
#         "tempo": 0.42,
#         "root": "a",
#         "mode": "minor",
#         "brightness": 0.4,
#         "space": 0.6,
#         "density": 6,
#         "layers": {
#             "bass": "pulsing",
#             "pad": "dark_sustained",
#             "melody": "arp_melody",
#             "rhythm": "electronic",
#             "texture": "shimmer",
#             "accent": "bells",
#         },
#         "motion": 0.65,
#         "attack": "sharp",
#         "stereo": 0.7,
#         "depth": True,
#         "echo": 0.5,
#         "human": 0.0,
#         "grain": "gritty",
#     }
#     dict_to_audio(dark_config, "dark_electronic.wav", duration=20.0)
#     print("   Saved: dark_electronic.wav")

#     # Example 3: From vibe string
#     print("\n3. Generating from vibe string (Underwater Cave)...")
#     generate_from_vibe(
#         "slow, peaceful, underwater cave, bioluminescence", "underwater_cave.wav", duration=20.0
#     )
#     print("   Saved: underwater_cave.wav")

#     print("\n" + "=" * 50)

#     # Example 4: Morph between two configs
#     print("\n4. Generating morph (Morning → Evening)...")

#     morning = SynthConfig(
#         tempo=0.30,
#         root="c",
#         mode="major",
#         brightness=0.6,
#         space=0.8,
#         bass="drone",
#         pad="warm_slow",
#         melody="minimal",
#         motion=0.3,
#         attack="soft",
#         echo=0.7,
#     )

#     evening = SynthConfig(
#         tempo=0.42,
#         root="a",
#         mode="minor",
#         brightness=0.45,
#         space=0.6,
#         bass="pulsing",
#         pad="dark_sustained",
#         melody="arp_melody",
#         motion=0.65,
#         attack="medium",
#         echo=0.5,
#     )

#     # Generate 2-minute morph with overlap-add (no volume dips)
#     audio = generate_tween_with_automation(morning, evening, duration=120.0)
#     # audio = assemble(morning, duration=30.0)  # Just morning, no morphing

#     sf.write("morning_to_evening.wav", audio, SAMPLE_RATE)
#     print("   Saved: morning_to_evening.wav")

#     print("\nDone! Generated 4 audio files.")
