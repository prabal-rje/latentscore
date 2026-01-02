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

import inspect
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, TypeAlias, TypedDict, cast

import numpy as np
import soundfile as sf  # type: ignore[import]
from numpy.typing import NDArray
from scipy.signal import butter, decimate, lfilter  # type: ignore[import]

# -----------------------------------------------------------------------------
# Optional: Spotify's 'pedalboard' (high-quality C++ DSP effects)
# -----------------------------------------------------------------------------
# We use pedalboard whenever it's available, but keep pure-numpy/scipy fallbacks
# so this file remains usable even without the dependency.
pedalboard_available = False
Pedalboard = None  # type: ignore[assignment]
PBReverb = PBDelay = PBLowpass = PBHighpass = None  # type: ignore[assignment]
PBCompressor = PBLimiter = None  # type: ignore[assignment]
PBBitcrush = PBDistortion = PBChorus = PBGain = None  # type: ignore[assignment]

try:  # pragma: no cover
    from pedalboard import (
        Compressor as PBCompressor,
    )
    from pedalboard import (
        Delay as PBDelay,
    )
    from pedalboard import (
        HighpassFilter as PBHighpass,
    )
    from pedalboard import (
        Limiter as PBLimiter,
    )
    from pedalboard import (
        LowpassFilter as PBLowpass,
    )
    from pedalboard import (
        Pedalboard,  # type: ignore[reportPrivateImportUsage]
    )
    from pedalboard import (
        Reverb as PBReverb,
    )

    pedalboard_available = True

    # Optional extras (not guaranteed to exist in every pedalboard version)
    try:  # pragma: no cover
        from pedalboard import Bitcrush as PBBitcrush  # type: ignore[import]
    except Exception:
        PBBitcrush = None  # type: ignore[assignment]

    try:  # pragma: no cover
        from pedalboard import Distortion as PBDistortion  # type: ignore[import]
    except Exception:
        PBDistortion = None  # type: ignore[assignment]

    try:  # pragma: no cover
        from pedalboard import Chorus as PBChorus  # type: ignore[import]
    except Exception:
        PBChorus = None  # type: ignore[assignment]

    try:  # pragma: no cover
        from pedalboard import Gain as PBGain  # type: ignore[import]
    except Exception:
        PBGain = None  # type: ignore[assignment]

except Exception:
    pedalboard_available = False


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return float(lo)
    if x > hi:
        return float(hi)
    return float(x)


def _pb_make_effect(effect_cls: Any, **kwargs: Any) -> Any:
    """Best-effort instantiate a pedalboard effect across versions."""
    if effect_cls is None:
        return None
    # Try keyword args (filtered by signature)
    try:
        sig = inspect.signature(effect_cls)
        accepted = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        if filtered:
            return effect_cls(**filtered)
    except Exception:
        pass

    # Fallback: try all kwargs directly
    try:
        return effect_cls(**kwargs)
    except Exception:
        pass

    # Last resort: try first value positionally (covers simple constructors)
    try:
        if kwargs:
            return effect_cls(next(iter(kwargs.values())))
        return effect_cls()
    except Exception:
        return None


def _pb_process_mono(signal: FloatArray, sr: int, effects: list[Any]) -> FloatArray:
    """Run a mono buffer through a pedalboard chain."""
    if not pedalboard_available or Pedalboard is None or not effects:
        return signal

    x = np.ascontiguousarray(np.asarray(signal, dtype=np.float32))
    board = Pedalboard([fx for fx in effects if fx is not None])
    y = board(x, sr)
    return np.asarray(y, dtype=np.float64)


@lru_cache(maxsize=256)
def _butter2_cached(norm_cutoff_q: float, btype: str) -> FilterCoefficients:
    """Cached 2nd-order Butterworth coefficients."""
    b, a = cast(FilterCoefficients, butter(2, norm_cutoff_q, btype=btype, output="ba"))
    return b, a


# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 44100

# Root note frequencies (octave 4) - tuple for immutability
NOTE_FREQS: dict[str, float] = {
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

# Root to semitone offset
ROOT_SEMITONES: dict[str, int] = {
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

# Mode intervals (semitones from root) - tuples for JIT
MODE_INTERVALS: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
}

# V2 parameter mappings
ATTACK_MULT: dict[str, float] = {"soft": 2.5, "medium": 1.0, "sharp": 0.3}
GRAIN_OSC: dict[str, str] = {"clean": "sine", "warm": "triangle", "gritty": "sawtooth"}

# Density → active layers (tuples)
DENSITY_LAYERS: dict[int, tuple[str, ...]] = {
    2: ("bass", "pad"),
    3: ("bass", "pad", "melody"),
    4: ("bass", "pad", "melody", "rhythm"),
    5: ("bass", "pad", "melody", "rhythm", "texture"),
    6: ("bass", "pad", "melody", "rhythm", "texture", "accent"),
}

FloatArray: TypeAlias = NDArray[np.float64]
OscFn: TypeAlias = Callable[[float, float, int, float], FloatArray]
FilterCoefficients: TypeAlias = tuple[FloatArray, FloatArray]


class MusicConfigDict(TypedDict, total=False):
    tempo: float
    root: str
    mode: str
    brightness: float
    space: float
    density: int
    bass: str
    pad: str
    melody: str
    rhythm: str
    texture: str
    accent: str
    motion: float
    attack: str
    stereo: float
    depth: bool
    echo: float
    human: float
    grain: str
    melody_engine: str
    phrase_len_bars: int
    melody_density: float
    syncopation: float
    swing: float
    motif_repeat_prob: float
    step_bias: float
    chromatic_prob: float
    cadence_strength: float
    register_min_oct: int
    register_max_oct: int
    tension_curve: str
    harmony_style: str
    chord_change_bars: int
    chord_extensions: str
    seed: int


# =============================================================================
# PART 1: SYNTHESIS PRIMITIVES
# =============================================================================


def freq_from_note(root: str, semitones: int = 0, octave: int = 4) -> float:
    """Get frequency for a note."""
    root_lower = root.lower()
    if root_lower not in NOTE_FREQS:
        raise ValueError(f"Unknown root note: {root}. Valid: {list(NOTE_FREQS.keys())}")
    base_freq = NOTE_FREQS[root_lower]
    octave_shift = octave - 4
    return base_freq * (2**octave_shift) * (2 ** (semitones / 12))


def generate_sine(
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3
) -> FloatArray:
    """Generate sine wave."""
    t = np.linspace(0, duration, int(sr * duration), False)
    return amp * np.sin(2 * np.pi * freq * t)


def generate_triangle(
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3
) -> FloatArray:
    """Generate triangle wave."""
    t = np.linspace(0, duration, int(sr * duration), False)
    return amp * 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - amp


def generate_sawtooth(
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3, oversample: int = 2
) -> FloatArray:
    """Generate anti-aliased sawtooth using 4-point PolyBLEP + oversampling."""

    num_samples = int(sr * duration)
    sr_high = sr * oversample
    num_samples_high = num_samples * oversample  # <- exact multiple
    dt = freq / sr_high

    t = np.linspace(0, duration * freq, num_samples_high, endpoint=False)
    t = t % 1.0
    naive = 2.0 * t - 1.0
    correction = np.zeros(num_samples_high)

    # 4-point PolyBLEP correction
    # Region 1: 0 <= t < dt
    m1 = t < dt
    t1 = t[m1] / dt
    correction[m1] = t1 * t1 * (2 * t1 - 3) + 1

    # Region 2: dt <= t < 2*dt
    m2 = (t >= dt) & (t < 2 * dt)
    t2 = t[m2] / dt - 1
    correction[m2] = t2 * t2 * (2 * t2 - 3)

    # Region 3: 1-2*dt < t <= 1-dt
    m3 = (t > 1 - 2 * dt) & (t <= 1 - dt)
    t3 = (t[m3] - 1) / dt + 1
    correction[m3] = t3 * t3 * (2 * t3 + 3)

    # Region 4: 1-dt < t < 1
    m4 = t > 1 - dt
    t4 = (t[m4] - 1) / dt
    correction[m4] = t4 * t4 * (2 * t4 + 3) + 1

    signal_high = naive - correction
    signal = decimate(signal_high, oversample, ftype="fir", zero_phase=True)

    # Ensure exact output length
    if len(signal) > num_samples:
        signal = signal[:num_samples]
    elif len(signal) < num_samples:
        signal = np.pad(signal, (0, num_samples - len(signal)))

    return cast(FloatArray, amp * signal)


def generate_square(
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3, oversample: int = 2
) -> FloatArray:
    """Generate anti-aliased square wave using 4-point PolyBLEP + oversampling."""

    num_samples = int(sr * duration)
    sr_high = sr * oversample
    num_samples_high = num_samples * oversample  # <- exact multiple
    dt = freq / sr_high

    t = np.linspace(0, duration * freq, num_samples_high, endpoint=False)
    t = t % 1.0
    naive = np.where(t < 0.5, 1.0, -1.0)
    correction = np.zeros(num_samples_high)

    def apply_4pt_blep(phase: FloatArray, corr: FloatArray, sign: float) -> None:
        """Apply 4-point PolyBLEP correction at discontinuity."""
        # Region 1: 0 <= phase < dt
        m1 = phase < dt
        t1 = phase[m1] / dt
        corr[m1] += sign * (t1 * t1 * (2 * t1 - 3) + 1)

        # Region 2: dt <= phase < 2*dt
        m2 = (phase >= dt) & (phase < 2 * dt)
        t2 = phase[m2] / dt - 1
        corr[m2] += sign * (t2 * t2 * (2 * t2 - 3))

        # Region 3: 1-2*dt < phase <= 1-dt
        m3 = (phase > 1 - 2 * dt) & (phase <= 1 - dt)
        t3 = (phase[m3] - 1) / dt + 1
        corr[m3] += sign * (t3 * t3 * (2 * t3 + 3))

        # Region 4: 1-dt < phase < 1
        m4 = phase > 1 - dt
        t4 = (phase[m4] - 1) / dt
        corr[m4] += sign * (t4 * t4 * (2 * t4 + 3) + 1)

    # Rising edge at phase = 0
    apply_4pt_blep(cast(FloatArray, t), correction, 1.0)

    # Falling edge at phase = 0.5
    t_shifted = (t + 0.5) % 1.0
    apply_4pt_blep(cast(FloatArray, t_shifted), correction, -1.0)

    signal_high = naive + correction
    signal = decimate(signal_high, oversample, ftype="fir", zero_phase=True)

    # Ensure exact output length
    if len(signal) > num_samples:
        signal = signal[:num_samples]
    elif len(signal) < num_samples:
        signal = np.pad(signal, (0, num_samples - len(signal)))

    return cast(FloatArray, amp * signal)


def generate_noise(
    duration: float,
    sr: int = SAMPLE_RATE,
    amp: float = 0.1,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Generate white noise."""
    n = int(sr * duration)
    if n <= 0:
        return cast(FloatArray, np.zeros(0, dtype=np.float64))

    if rng is None:
        return amp * np.random.randn(n)

    return amp * rng.standard_normal(n)


OSC_FUNCTIONS: dict[str, OscFn] = {
    "sine": generate_sine,
    "triangle": generate_triangle,
    "sawtooth": generate_sawtooth,
    "square": generate_square,
}


def apply_adsr(
    signal: FloatArray,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    sr: int = SAMPLE_RATE,
) -> FloatArray:
    """Apply ADSR envelope to signal."""
    # Minimum times to prevent clicks (5ms attack, 10ms release)
    attack = max(attack, 0.005)
    release = max(release, 0.01)

    total = len(signal)
    a_samples = int(attack * sr)
    d_samples = int(decay * sr)
    r_samples = int(release * sr)
    s_samples = max(0, total - a_samples - d_samples - r_samples)

    envelope = np.concatenate(
        (
            np.linspace(0, 1, max(1, a_samples)),
            np.linspace(1, sustain, max(1, d_samples)),
            np.ones(max(1, s_samples)) * sustain,
            np.linspace(sustain, 0, max(1, r_samples)),
        )
    )

    # Match length
    if len(envelope) < total:
        envelope = np.pad(envelope, (0, total - len(envelope)))
    else:
        envelope = envelope[:total]

    return signal * envelope


def apply_lowpass(signal: FloatArray, cutoff: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply lowpass filter.

    Prefers Spotify's pedalboard (if installed) for high-quality DSP, with a scipy fallback.
    """
    if cutoff <= 0:
        return signal

    # Pedalboard path
    if pedalboard_available and PBLowpass is not None and Pedalboard is not None:
        cutoff_hz = _clamp(float(cutoff), 5.0, (sr / 2) * 0.99)
        fx = _pb_make_effect(
            PBLowpass,
            cutoff_frequency_hz=cutoff_hz,
            cutoff_hz=cutoff_hz,
        )
        if fx is not None:
            return _pb_process_mono(signal, sr, [fx])

    # Scipy fallback (cached coefficients)
    nyquist = sr / 2
    normalized = min(max(cutoff / nyquist, 0.001), 0.99)
    normalized_q = float(round(normalized, 6))
    b, a = _butter2_cached(normalized_q, "low")
    return cast(FloatArray, lfilter(b, a, signal))


def apply_highpass(signal: FloatArray, cutoff: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply highpass filter.

    Prefers Spotify's pedalboard (if installed) for high-quality DSP, with a scipy fallback.
    """
    if cutoff <= 0:
        return signal

    # Pedalboard path
    if pedalboard_available and PBHighpass is not None and Pedalboard is not None:
        cutoff_hz = _clamp(float(cutoff), 5.0, (sr / 2) * 0.99)
        fx = _pb_make_effect(
            PBHighpass,
            cutoff_frequency_hz=cutoff_hz,
            cutoff_hz=cutoff_hz,
        )
        if fx is not None:
            return _pb_process_mono(signal, sr, [fx])

    # Scipy fallback (cached coefficients)
    nyquist = sr / 2
    normalized = min(max(cutoff / nyquist, 0.001), 0.99)
    normalized_q = float(round(normalized, 6))
    b, a = _butter2_cached(normalized_q, "high")
    return cast(FloatArray, lfilter(b, a, signal))


def apply_delay(
    signal: FloatArray,
    delay_time: float,
    feedback: float,
    wet: float,
    sr: int = SAMPLE_RATE,
) -> FloatArray:
    """Apply delay effect.

    - If pedalboard is available, uses its Delay (feedback + mix).
    - Otherwise, falls back to a lightweight multi-tap delay.
    """
    if delay_time <= 0 or wet <= 0:
        return signal

    # Pedalboard path
    if pedalboard_available and PBDelay is not None and Pedalboard is not None:
        delay_s = _clamp(float(delay_time), 0.0, 2.0)
        fb = _clamp(float(feedback), 0.0, 0.99)
        mix = _clamp(float(wet), 0.0, 1.0)

        fx = _pb_make_effect(
            PBDelay,
            delay_seconds=delay_s,
            feedback=fb,
            mix=mix,
        )
        if fx is not None:
            return _pb_process_mono(signal, sr, [fx])

    # Fallback: simple multi-tap delay
    delay_samples = int(delay_time * sr)
    if delay_samples <= 0:
        return signal

    output = signal.copy()

    for i in range(1, 5):  # 4 delay taps
        offset = delay_samples * i
        if offset < len(signal):
            delayed = np.zeros_like(signal)
            delayed[offset:] = signal[:-offset] * (feedback**i) * wet
            output += delayed

    return output


def apply_reverb(signal: FloatArray, room: float, size: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Reverb.

    - If pedalboard is available, uses its Reverb for better quality.
    - Otherwise, uses a simple multi-delay reverb (fast + dependency-free).
    """
    if room <= 0:
        return signal

    # Pedalboard path
    if pedalboard_available and PBReverb is not None and Pedalboard is not None:
        # Map the engine's controls to pedalboard's parameters (best-effort).
        room_amt = _clamp(float(room), 0.0, 1.0)
        room_size = _clamp(float(size), 0.0, 1.0)

        # Pedalboard reverb can get very wet quickly; keep it conservative.
        wet = _clamp(room_amt * 0.6, 0.0, 1.0)
        dry = _clamp(1.0 - wet, 0.0, 1.0)
        damping = _clamp(0.2 + (1.0 - room_size) * 0.6, 0.0, 1.0)

        fx = _pb_make_effect(
            PBReverb,
            room_size=room_size,
            damping=damping,
            wet_level=wet,
            dry_level=dry,
            width=1.0,
        )
        if fx is not None:
            return _pb_process_mono(signal, sr, [fx])

    # Fallback: multiple delay lines at prime-ish intervals
    output = signal.copy()

    delays = (0.029, 0.037, 0.041, 0.053, 0.067)

    for i, delay in enumerate(delays):
        delay_samples = int(delay * size * sr)
        if 0 < delay_samples < len(signal):
            reverb = np.zeros_like(signal)
            reverb[delay_samples:] = signal[:-delay_samples] * room * (0.7**i)
            output += reverb

    return output


def generate_lfo(duration: float, rate: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Generate LFO signal (0 to 1 range)."""
    t = np.linspace(0, duration, int(sr * duration), False)
    return 0.5 + 0.5 * np.sin(2 * np.pi * rate * t)


def apply_humanize(
    signal: FloatArray,
    amount: float,
    sr: int = SAMPLE_RATE,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Apply subtle timing/amplitude humanization.

    Note: `rng` is optional and only exists to make seeded renders deterministic.
    """
    _ = sr  # (kept for backwards compatibility / signature stability)

    if amount <= 0:
        return signal

    if rng is None:
        jitter = np.random.randn(len(signal))
    else:
        jitter = rng.standard_normal(len(signal))

    # Subtle amplitude variation
    amp_lfo = 1.0 + (jitter * amount * 0.1)
    amp_lfo = np.clip(amp_lfo, 0.9, 1.1)

    return signal * amp_lfo


def add_note(signal: FloatArray, note: FloatArray, start_index: int) -> None:
    """Safely adds a note to the signal buffer, clipping if necessary."""
    if start_index >= len(signal):
        return

    end_index = start_index + len(note)

    if end_index <= len(signal):
        signal[start_index:end_index] += note
    else:
        # Clip the note to fit the remaining signal space
        available = len(signal) - start_index
        clipped = note[:available].copy()

        # Apply quick fade-out to prevent click from abrupt cutoff
        fade_samples = min(int(SAMPLE_RATE * 0.01), available // 4)  # 10ms max
        if fade_samples > 1:
            clipped[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        signal[start_index:] += clipped


# =============================================================================
# PART 2: PATTERN GENERATORS
# =============================================================================


# -----------------------------------------------------------------------------
# HARMONY + PROCEDURAL MELODY STRUCTURES (compute-light)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ChordEvent:
    """A chord specified in *scale degrees* relative to the key, scheduled in beats."""

    start_beat: float
    duration_beats: float
    root_degree: int  # 0-6 (scale degree)


@dataclass(frozen=True)
class HarmonyPlan:
    """A sequence of chord events spanning the render duration."""

    chords: Tuple[ChordEvent, ...] = ()
    beats_per_bar: int = 4

    def chord_at_beat(self, beat: float) -> ChordEvent:
        """Get the chord event active at the given beat (clamped)."""
        if not self.chords:
            return ChordEvent(0.0, float(self.beats_per_bar), 0)

        # Fast path for common small chord counts
        for ch in self.chords:
            if ch.start_beat <= beat < (ch.start_beat + ch.duration_beats):
                return ch

        # If out of range, clamp
        if beat < self.chords[0].start_beat:
            return self.chords[0]
        return self.chords[-1]


@dataclass(frozen=True)
class MelodyPolicy:
    """Knobs for compute-light, phrase-aware, chord-aware melody generation."""

    phrase_len_bars: int = 4
    density: float = 0.45  # 0..1 (probability-ish)
    syncopation: float = 0.20  # 0..1 (offbeat bias)
    swing: float = 0.0  # 0..1 (8th swing feel)
    motif_repeat_prob: float = 0.50  # 0..1
    step_bias: float = 0.75  # 0..1 (1=mostly stepwise, 0=leapy)
    chromatic_prob: float = 0.05  # 0..1 (approach-tone chance)
    cadence_strength: float = 0.65  # 0..1
    register_min_oct: int = 4  # inclusive
    register_max_oct: int = 6  # inclusive
    tension_curve: str = "arc"  # "arc" | "ramp" | "waves"


@dataclass(frozen=True)
class NoteEvent:
    """A rendered melody note event (frequency already resolved)."""

    start_sec: float
    dur_sec: float
    freq: float
    amp: float
    is_anchor: bool = False


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _tension_value(curve: str, x: float) -> float:
    """x in [0..1] -> tension in [0..1]."""
    x = _clamp01(x)
    if curve == "ramp":
        return x
    if curve == "waves":
        # Two gentle waves across a phrase
        return 0.5 - 0.5 * float(np.cos(2.0 * np.pi * 2.0 * x))
    # default: arc
    return float(np.sin(np.pi * x))


def _iter_chord_segments(params: "SynthParams") -> Iterable[Tuple[float, float, int]]:
    """Yield (start_sec, dur_sec, chord_root_degree) segments."""
    if params.harmony is None or not params.harmony.chords:
        yield (0.0, params.duration, 0)
        return

    spb = params.seconds_per_beat
    for ch in params.harmony.chords:
        start = ch.start_beat * spb
        if start >= params.duration:
            break
        dur = ch.duration_beats * spb
        end = min(params.duration, start + dur)
        dur = max(0.0, end - start)
        if dur > 0:
            yield (start, dur, int(ch.root_degree))


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


def _chord_tones_ascending(params: "SynthParams", chord_root_degree: int) -> Tuple[int, ...]:
    """Chord tones as ascending scale degrees above the chord root (best for voicings)."""
    d = int(chord_root_degree)
    tones: List[int] = [d, d + 2, d + 4]  # triad
    if params.chord_extensions in ("sevenths", "lush"):
        tones.append(d + 6)  # 7th
    if params.chord_extensions == "lush":
        tones.append(d + 8)  # 9th (2 an octave up)
    return tuple(tones)


@dataclass
class SynthParams:
    """Parameters passed to all synthesis functions."""

    root: str = "c"
    mode: str = "minor"
    brightness: float = 0.5
    space: float = 0.6
    duration: float = 16.0
    tempo: float = 0.35  # normalized 0..1 (mapped to BPM internally)

    # V2 parameters
    motion: float = 0.5
    attack: str = "medium"
    stereo: float = 0.5
    depth: bool = False
    echo: float = 0.5
    human: float = 0.0
    grain: str = "clean"

    # New: harmony + procedural melody
    harmony: HarmonyPlan | None = None
    chord_extensions: str = "triads"  # "triads" | "sevenths" | "lush"
    melody_policy: MelodyPolicy | None = None

    # New: deterministic randomness per render block if desired
    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)

    # -------------------------------------------------------------------------
    # Derived helpers
    # -------------------------------------------------------------------------

    @property
    def attack_mult(self) -> float:
        return ATTACK_MULT.get(self.attack, 1.0)

    @property
    def osc_type(self) -> str:
        return GRAIN_OSC.get(self.grain, "sine")

    @property
    def echo_mult(self) -> float:
        return self.echo / 0.5  # Normalize around 0.5

    @property
    def pan_width(self) -> float:
        return self.stereo * 0.5

    @property
    def bpm(self) -> float:
        """Map normalized tempo (0..1) to a musical BPM range."""
        # 55..165 feels usable for ambient→upbeat without getting silly
        t = float(np.clip(self.tempo, 0.0, 1.0))
        return 55.0 + 110.0 * t

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def beats_total(self) -> float:
        return self.duration / self.seconds_per_beat

    def beat_to_seconds(self, beat: float) -> float:
        return float(beat) * self.seconds_per_beat

    def seconds_to_beat(self, sec: float) -> float:
        return float(sec) / self.seconds_per_beat

    # -------------------------------------------------------------------------
    # Pitch helpers
    # -------------------------------------------------------------------------

    def semitone_from_degree(self, degree: int) -> int:
        """Scale-degree index → semitone offset from the key root."""
        intervals = MODE_INTERVALS.get(self.mode, MODE_INTERVALS["minor"])
        return int(intervals[degree % len(intervals)] + (12 * (degree // len(intervals))))

    def get_scale_freq(self, degree: int, octave: int = 4) -> float:
        """Get frequency for a scale degree in the current mode."""
        semitone = self.semitone_from_degree(degree)
        return freq_from_note(self.root, semitone, octave)

    def chord_tone_classes(self, chord_root_degree: int) -> Tuple[int, ...]:
        """Return chord tone *classes* (0..6), i.e., diatonic pitch classes."""
        d = int(chord_root_degree) % 7
        tones: List[int] = [d, (d + 2) % 7, (d + 4) % 7]  # triad
        if self.chord_extensions in ("sevenths", "lush"):
            tones.append((d + 6) % 7)  # 7th
        if self.chord_extensions == "lush":
            tones.append((d + 1) % 7)  # 9th
        # de-dupe while preserving order
        out: List[int] = []
        for t in tones:
            if t not in out:
                out.append(t)
        return tuple(out)

    def chord_root_degree_at_beat(self, beat: float) -> int:
        if self.harmony is None:
            return 0
        return int(self.harmony.chord_at_beat(float(beat)).root_degree) % 7


PatternFn: TypeAlias = Callable[[SynthParams], FloatArray]


# -----------------------------------------------------------------------------
# BASS PATTERNS
# -----------------------------------------------------------------------------


def bass_drone(params: SynthParams) -> FloatArray:
    """Sustained drone bass (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = params.duration

    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        # Slight overlap between chord segments to avoid discontinuities
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)
        freq = params.get_scale_freq(chord_root, 2)

        note = generate_sine(freq, note_dur, sr, 0.35)
        note = apply_lowpass(note, 80 * params.brightness + 20, sr)
        note = apply_adsr(
            note,
            1.8 * params.attack_mult,
            0.5,
            0.95,
            2.2 * params.attack_mult,
            sr,
        )
        add_note(signal, note, int(start_sec * sr))

    signal = apply_reverb(signal, params.space * 0.5, params.space, sr)
    return signal


def bass_sustained(params: SynthParams) -> FloatArray:
    """Long sustained bass notes that follow chord roots."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        # A gentle root→fifth→root movement inside each chord region
        pattern = (0, 0, 4, 0)
        note_dur = seg_dur / len(pattern)

        for i, rel_degree in enumerate(pattern):
            deg = chord_root + rel_degree
            freq = params.get_scale_freq(deg, 2)
            start = int((start_sec + i * note_dur) * sr)

            note = generate_sine(freq, note_dur * 0.95, sr, 0.32)
            note = apply_lowpass(note, 100 * params.brightness + 30, sr)
            note = apply_adsr(
                note,
                0.7 * params.attack_mult,
                0.3,
                0.85,
                1.2 * params.attack_mult,
                sr,
            )
            add_note(signal, note, start)

    signal = apply_reverb(signal, params.space * 0.4, params.space, sr)
    return signal


def bass_pulsing(params: SynthParams) -> FloatArray:
    """Rhythmic pulsing bass (chord-aware)."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Pulses in beat space so it tracks tempo naturally
    beats_total = max(1.0, params.beats_total)
    pulses_per_beat = 2.0  # 8th-note pulse
    num_pulses = int(np.ceil(beats_total * pulses_per_beat))
    pulse_beats = 1.0 / pulses_per_beat
    pulse_sec = pulse_beats * params.seconds_per_beat

    signal = np.zeros(int(sr * dur))

    for i in range(num_pulses):
        start_sec = i * pulse_sec
        if start_sec >= dur:
            break

        beat = i * pulse_beats
        chord_root = params.chord_root_degree_at_beat(beat)
        freq = params.get_scale_freq(chord_root, 2)

        note = generate_sine(freq, pulse_sec * 0.82, sr, 0.35)
        note = apply_lowpass(note, 90 * params.brightness + 20, sr)
        note = apply_adsr(
            note,
            0.02 * params.attack_mult,
            0.08,
            0.6,
            0.25 * params.attack_mult,
            sr,
        )
        add_note(signal, note, int(start_sec * sr))

    return signal


def bass_walking(params: SynthParams) -> FloatArray:
    """Walking bass line that follows the chord root (diatonic)."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    signal = np.zeros(int(sr * dur_total))

    # 1 note per beat
    beats_total = int(np.ceil(params.beats_total))
    note_sec = params.seconds_per_beat

    for i in range(beats_total):
        start_sec = i * note_sec
        if start_sec >= dur_total:
            break

        chord_root = params.chord_root_degree_at_beat(float(i))
        # Simple walk inside the triad: root → third → fifth → third
        rel = (0, 2, 4, 2)[i % 4]
        degree = chord_root + rel

        freq = params.get_scale_freq(degree, 2)
        note = generate_triangle(freq, note_sec * 0.92, sr, 0.30)
        note = apply_lowpass(note, 120 * params.brightness + 40, sr)
        note = apply_adsr(
            note,
            0.04 * params.attack_mult,
            0.12,
            0.7,
            0.22 * params.attack_mult,
            sr,
        )
        note = apply_humanize(note, params.human, sr, params.rng)
        add_note(signal, note, int(start_sec * sr))

    signal = apply_reverb(signal, params.space * 0.3, params.space * 0.8, sr)
    return signal


def bass_fifth_drone(params: SynthParams) -> FloatArray:
    """Root + fifth drone that follows chord changes."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        root_freq = params.get_scale_freq(chord_root, 2)
        fifth_freq = params.get_scale_freq(chord_root + 4, 2)

        root = generate_sine(root_freq, note_dur, sr, 0.26)
        root = apply_lowpass(root, 70 * params.brightness + 20, sr)
        root = apply_adsr(root, 2.2 * params.attack_mult, 0.5, 0.95, 2.6 * params.attack_mult, sr)

        fifth = generate_sine(fifth_freq, note_dur, sr, 0.18)
        fifth = apply_lowpass(fifth, 100 * params.brightness + 30, sr)
        fifth = apply_adsr(fifth, 2.4 * params.attack_mult, 0.5, 0.9, 2.6 * params.attack_mult, sr)

        add_note(signal, root + fifth, int(start_sec * sr))

    signal = apply_reverb(signal, params.space * 0.5, params.space, sr)
    return signal


def bass_sub_pulse(params: SynthParams) -> FloatArray:
    """Deep sub-bass pulse (chord-aware)."""
    sr = SAMPLE_RATE
    dur = params.duration
    signal = np.zeros(int(sr * dur))

    beats_total = max(1.0, params.beats_total)
    pulses_per_bar = 2  # half-note pulses
    beats_per_pulse = 4.0 / pulses_per_bar  # assuming 4/4
    pulse_sec = beats_per_pulse * params.seconds_per_beat

    num_pulses = int(np.ceil(beats_total / beats_per_pulse))
    for i in range(num_pulses):
        start_sec = i * pulse_sec
        if start_sec >= dur:
            break

        beat = i * beats_per_pulse
        chord_root = params.chord_root_degree_at_beat(beat)

        freq = params.get_scale_freq(chord_root, 1)  # very low octave
        note = generate_sine(freq, pulse_sec * 0.95, sr, 0.4)
        note = apply_lowpass(note, 55, sr)
        note = apply_adsr(note, 0.25 * params.attack_mult, 0.2, 0.9, 0.7 * params.attack_mult, sr)
        add_note(signal, note, int(start_sec * sr))

    return signal


BASS_PATTERNS: dict[str, PatternFn] = {
    "drone": bass_drone,
    "sustained": bass_sustained,
    "pulsing": bass_pulsing,
    "walking": bass_walking,
    "fifth_drone": bass_fifth_drone,
    "sub_pulse": bass_sub_pulse,
    "octave": bass_sustained,
    "arp_bass": bass_pulsing,
}


# -----------------------------------------------------------------------------
# PAD PATTERNS
# -----------------------------------------------------------------------------


def pad_warm_slow(params: SynthParams) -> FloatArray:
    """Warm, slowly evolving pad (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_sine)

    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        for degree in _chord_tones_ascending(params, chord_root)[:3]:
            freq = params.get_scale_freq(degree, 3)
            tone = osc(freq, note_dur, sr, 0.15)

            # Slow filter movement (affected by motion)
            lfo_rate = 0.1 / (params.motion + 0.1)
            lfo = generate_lfo(note_dur, lfo_rate, sr)

            base_cutoff = 300 * params.brightness + 100
            tone_low = apply_lowpass(tone, base_cutoff * 0.5, sr)
            tone_high = apply_lowpass(tone, base_cutoff * 1.5, sr)
            tone = tone_low * (1 - lfo) + tone_high * lfo

            tone = apply_adsr(
                tone, 1.5 * params.attack_mult, 0.8, 0.85, 2.5 * params.attack_mult, sr
            )
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, params.space * 0.7, params.space * 0.9, sr)
    signal = apply_delay(signal, 0.35, 0.3 * params.echo_mult, 0.25 * params.echo_mult, sr)
    signal = apply_humanize(signal, params.human, sr, params.rng)

    return signal


def pad_dark_sustained(params: SynthParams) -> FloatArray:
    """Dark, heavy sustained pad (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        for degree in _chord_tones_ascending(params, chord_root)[:3]:
            freq = params.get_scale_freq(degree, 3)
            tone = generate_sawtooth(freq, note_dur, sr, 0.12)
            tone = apply_lowpass(tone, 200 * params.brightness + 80, sr)
            tone = apply_adsr(
                tone, 2.0 * params.attack_mult, 1.0, 0.9, 3.0 * params.attack_mult, sr
            )
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, params.space * 0.8, params.space, sr)
    return signal


def pad_cinematic(params: SynthParams) -> FloatArray:
    """Big, cinematic pad with movement (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    signal = np.zeros(int(sr * dur_total))

    # Wider voicing with octave doubling
    voicings = ((0, 3), (2, 3), (4, 3), (0, 4), (4, 4))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        note_dur = min(seg_dur * 1.10, dur_total - start_sec)

        for rel_degree, octave in voicings:
            degree = chord_root + rel_degree
            freq = params.get_scale_freq(degree, octave)

            tone = generate_sawtooth(freq, note_dur, sr, 0.08)
            tone += generate_triangle(freq * 1.002, note_dur, sr, 0.06)  # slight detune

            tone = apply_lowpass(tone, 400 * params.brightness + 150, sr)
            tone = apply_adsr(
                tone, 1.8 * params.attack_mult, 0.8, 0.88, 2.8 * params.attack_mult, sr
            )
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, params.space * 0.85, params.space, sr)
    signal = apply_delay(signal, 0.4, 0.35 * params.echo_mult, 0.3 * params.echo_mult, sr)

    return signal


def pad_thin_high(params: SynthParams) -> FloatArray:
    """Thin, high pad (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        for rel_degree in (0, 4):  # root + fifth
            degree = chord_root + rel_degree
            freq = params.get_scale_freq(degree, 4)
            tone = generate_sine(freq, note_dur, sr, 0.12)
            tone = apply_lowpass(tone, 800 * params.brightness + 200, sr)
            tone = apply_highpass(tone, 200, sr)
            tone = apply_adsr(
                tone, 1.2 * params.attack_mult, 0.6, 0.8, 2.0 * params.attack_mult, sr
            )
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, params.space * 0.75, params.space * 0.95, sr)
    return signal


def pad_ambient_drift(params: SynthParams) -> FloatArray:
    """Slowly drifting ambient pad (chord-aware, with gentle color notes)."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        # More overlap to feel "drifty"
        note_dur = min(seg_dur * 1.35, dur_total - start_sec)

        tones = list(_chord_tones_ascending(params, chord_root)[:3])

        # Occasionally add a gentle color tone (sus4-ish) to create drift
        if params.motion > 0.4 and params.rng.random() < (0.15 + 0.25 * params.motion):
            tones.append(chord_root + 5)

        for degree in tones:
            freq = params.get_scale_freq(degree, 3)
            tone = generate_sine(freq, note_dur, sr, 0.14)
            tone = apply_lowpass(tone, 350 * params.brightness + 100, sr)
            tone = apply_adsr(
                tone, 1.6 * params.attack_mult, 0.5, 0.85, 2.2 * params.attack_mult, sr
            )
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, params.space * 0.8, params.space, sr)
    signal = apply_delay(signal, 0.5, 0.4 * params.echo_mult, 0.35 * params.echo_mult, sr)

    return signal


def pad_stacked_fifths(params: SynthParams) -> FloatArray:
    """Fifths stacked for a powerful sound (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    signal = np.zeros(int(sr * dur_total))

    voicings = ((0, 3), (4, 3), (0, 4), (4, 4))
    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        note_dur = min(seg_dur * 1.10, dur_total - start_sec)

        for rel_degree, octave in voicings:
            degree = chord_root + rel_degree
            freq = params.get_scale_freq(degree, octave)
            tone = osc(freq, note_dur, sr, 0.10)
            tone = apply_lowpass(tone, 500 * params.brightness + 150, sr)
            tone = apply_adsr(
                tone, 1.3 * params.attack_mult, 0.7, 0.88, 2.2 * params.attack_mult, sr
            )
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, params.space * 0.7, params.space * 0.85, sr)
    return signal


PAD_PATTERNS: dict[str, PatternFn] = {
    "warm_slow": pad_warm_slow,
    "dark_sustained": pad_dark_sustained,
    "cinematic": pad_cinematic,
    "thin_high": pad_thin_high,
    "ambient_drift": pad_ambient_drift,
    "stacked_fifths": pad_stacked_fifths,
    "bright_open": pad_thin_high,
}


# -----------------------------------------------------------------------------
# MELODY PATTERNS
# -----------------------------------------------------------------------------


def _melody_policy_from_params(params: SynthParams) -> MelodyPolicy:
    """Return the active MelodyPolicy with safety clamps."""
    p = params.melody_policy or MelodyPolicy()
    # Clamp-ish values (avoid user/LLM weirdness)
    phrase_len = int(max(1, min(16, p.phrase_len_bars)))
    reg_min = int(min(p.register_min_oct, p.register_max_oct))
    reg_max = int(max(p.register_min_oct, p.register_max_oct))
    return MelodyPolicy(
        phrase_len_bars=phrase_len,
        density=_clamp01(p.density),
        syncopation=_clamp01(p.syncopation),
        swing=_clamp01(p.swing),
        motif_repeat_prob=_clamp01(p.motif_repeat_prob),
        step_bias=_clamp01(p.step_bias),
        chromatic_prob=_clamp01(p.chromatic_prob),
        cadence_strength=_clamp01(p.cadence_strength),
        register_min_oct=reg_min,
        register_max_oct=reg_max,
        tension_curve=p.tension_curve if p.tension_curve in ("arc", "ramp", "waves") else "arc",
    )


def _grid_step_beats(density: float) -> float:
    """Density→grid step in beats (compute-light)."""
    if density < 0.25:
        return 1.0  # quarters
    if density < 0.60:
        return 0.5  # 8ths
    return 0.25  # 16ths


def _nearest_chord_tone_pitch(
    pitch: int,
    chord_root_degree: int,
    min_pitch: int,
    max_pitch: int,
    params: SynthParams,
) -> int:
    """Find the nearest chord-tone pitch (diatonic) to the given pitch."""
    pcs = params.chord_tone_classes(chord_root_degree)
    candidates: List[int] = []
    for pc in pcs:
        for oct_i in range((max_pitch // 7) + 1):
            cand = pc + 7 * oct_i
            if min_pitch <= cand <= max_pitch:
                candidates.append(cand)
    if not candidates:
        return int(np.clip(pitch, min_pitch, max_pitch))
    best = min(candidates, key=lambda c: abs(c - pitch))
    return int(best)


def _choose_anchor_pitch(
    rng: np.random.Generator,
    prev_pitch: int,
    chord_root_degree: int,
    min_pitch: int,
    max_pitch: int,
    tension: float,
    cadence_strength: float,
    params: SynthParams,
) -> int:
    """Choose an anchor pitch (chord tone) with simple voice-leading."""
    chord_root = chord_root_degree % 7
    pcs = params.chord_tone_classes(chord_root_degree)

    # Desired register drifts slightly upward with tension
    midpoint = 0.5 * (min_pitch + max_pitch)
    desired = midpoint + (tension - 0.5) * 0.35 * (max_pitch - min_pitch)

    items: List[int] = []
    weights: List[float] = []

    for pc in pcs:
        # base weights: emphasize root at cadences
        w_pc = 1.0
        if pc == chord_root:
            w_pc *= 1.0 + 1.8 * cadence_strength
        elif pc == (chord_root + 2) % 7:  # 3rd
            w_pc *= 1.0
        elif pc == (chord_root + 4) % 7:  # 5th
            w_pc *= 0.95
        elif pc == (chord_root + 6) % 7:  # 7th
            w_pc *= 0.85
        else:  # 9th etc
            w_pc *= 0.75

        for oct_i in range((max_pitch // 7) + 1):
            cand = pc + 7 * oct_i
            if cand < min_pitch or cand > max_pitch:
                continue

            # Voice-leading + register preference
            d_prev = abs(cand - prev_pitch)
            d_reg = abs(cand - desired)
            w = w_pc * float(np.exp(-d_prev / 2.6)) * float(np.exp(-d_reg / 4.0))
            items.append(int(cand))
            weights.append(float(w))

    if not items:
        return int(np.clip(prev_pitch, min_pitch, max_pitch))

    return _weighted_choice(rng, items, weights)


def _generate_procedural_melody_events(params: SynthParams) -> List[NoteEvent]:
    """Generate phrase-aware, chord-aware melody events."""
    sr = SAMPLE_RATE
    _ = sr  # (kept for symmetry; generation is in beat-time)

    policy = _melody_policy_from_params(params)
    rng = params.rng

    beats_per_bar = 4
    spb = params.seconds_per_beat
    total_beats = max(1.0, params.beats_total)
    total_bars = int(np.ceil(total_beats / beats_per_bar))

    # Register in diatonic "degree_total" units, relative to register_min_oct
    base_oct = policy.register_min_oct
    oct_span = max(0, policy.register_max_oct - policy.register_min_oct)
    min_pitch = 0
    max_pitch = oct_span * 7 + 6

    # ---------------------------------------------------------------------
    # 1) Build anchor pitches (1 per bar)
    # ---------------------------------------------------------------------
    anchors: List[int] = []
    prev = int(np.clip((max_pitch - min_pitch) // 2, min_pitch, max_pitch))

    for bar in range(total_bars):
        beat0 = float(bar * beats_per_bar)
        chord_root = params.chord_root_degree_at_beat(beat0)

        phrase_len = policy.phrase_len_bars
        pos_in_phrase = bar % phrase_len
        denom = max(1, phrase_len - 1)
        x = pos_in_phrase / denom
        tension = _tension_value(policy.tension_curve, x)

        # Cadence on the last bar of each phrase
        is_cadence_bar = (pos_in_phrase == phrase_len - 1) and (phrase_len > 1)
        if is_cadence_bar:
            tension = 0.0

        anchor = _choose_anchor_pitch(
            rng=rng,
            prev_pitch=prev,
            chord_root_degree=chord_root,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            tension=tension,
            cadence_strength=policy.cadence_strength if is_cadence_bar else 0.10,
            params=params,
        )
        anchors.append(anchor)
        prev = anchor

    # ---------------------------------------------------------------------
    # 2) Generate note events bar-by-bar, using anchors as targets
    # ---------------------------------------------------------------------
    events: List[NoteEvent] = []

    step_beats = _grid_step_beats(policy.density)
    steps_per_bar = int(round(beats_per_bar / step_beats))
    steps_per_bar = max(1, min(64, steps_per_bar))

    # Motif memory: list of (step_idx, offset_from_anchor)
    motif: List[Tuple[int, int]] = []

    for bar in range(total_bars):
        beat_bar = float(bar * beats_per_bar)
        chord_root = params.chord_root_degree_at_beat(beat_bar)

        phrase_len = policy.phrase_len_bars
        pos_in_phrase = bar % phrase_len
        denom = max(1, phrase_len - 1)
        x = pos_in_phrase / denom
        tension = _tension_value(policy.tension_curve, x)

        is_phrase_start = pos_in_phrase == 0
        is_cadence_bar = (pos_in_phrase == phrase_len - 1) and (phrase_len > 1)

        anchor_pitch = anchors[bar]
        next_anchor = anchors[bar + 1] if bar + 1 < len(anchors) else anchor_pitch

        # Decide if we reuse motif on phrase starts
        use_motif = bool(motif) and is_phrase_start and (rng.random() < policy.motif_repeat_prob)

        need_resolve = False
        current_pitch = anchor_pitch

        # Collect notes for motif capture
        bar_notes_for_motif: List[Tuple[int, int]] = []

        for step_idx in range(steps_per_bar):
            pos_beats = step_idx * step_beats
            beat = beat_bar + pos_beats
            start_sec = beat * spb

            if start_sec >= params.duration:
                break

            # Swing (only makes sense on 8ths)
            if step_beats == 0.5 and (step_idx % 2 == 1):
                start_sec += policy.swing * 0.12 * spb

            # Note on/off decision
            if step_idx == 0:
                play = True
                is_anchor = True
            elif use_motif and any(s == step_idx for s, _ in motif):
                play = True
                is_anchor = False
            else:
                is_offbeat = abs(pos_beats - round(pos_beats)) > 1e-6

                p = 0.10 + 0.80 * policy.density
                p += 0.20 * tension
                p += (0.25 * policy.syncopation) if is_offbeat else (-0.15 * policy.syncopation)

                # Make cadences breathe (more space late in cadence bars)
                if is_cadence_bar and pos_beats >= 2.0:
                    p *= 0.35 + 0.40 * (1.0 - policy.cadence_strength)

                # Avoid machine-gun repeats
                if events and (events[-1].start_sec > (start_sec - 0.001)):
                    p *= 0.6

                play = rng.random() < _clamp01(p)
                is_anchor = False

            if not play:
                continue

            # -----------------------------------------------------------------
            # Pitch selection
            # -----------------------------------------------------------------
            if step_idx == 0:
                pitch = anchor_pitch
            elif use_motif:
                # Find the offset stored for this step
                offset = next((off for s, off in motif if s == step_idx), 0)
                pitch = int(np.clip(anchor_pitch + offset, min_pitch, max_pitch))
            else:
                delta = next_anchor - current_pitch
                if delta == 0:
                    direction = int(rng.choice([-1, 1]))
                else:
                    direction = 1 if delta > 0 else -1

                # If we have a pending non-chord tone, resolve to nearest chord tone
                if need_resolve:
                    pitch = _nearest_chord_tone_pitch(
                        current_pitch, chord_root, min_pitch, max_pitch, params
                    )
                    need_resolve = False
                else:
                    leap_prob = (1.0 - policy.step_bias) * (0.25 + 0.35 * tension)
                    step_size = 1

                    # occasional holds
                    if rng.random() < (0.06 + 0.10 * tension):
                        step_size = 0
                    elif abs(delta) >= 4 and (rng.random() < leap_prob):
                        step_size = int(rng.choice([2, 3, 4]))

                    pitch = int(
                        np.clip(current_pitch + direction * step_size, min_pitch, max_pitch)
                    )

                # If note is non-chord tone, sometimes keep it as tension, but resolve soon
                chord_pcs = params.chord_tone_classes(chord_root)
                if (pitch % 7) not in chord_pcs:
                    if rng.random() < (0.55 - 0.45 * tension):
                        pitch = _nearest_chord_tone_pitch(
                            pitch, chord_root, min_pitch, max_pitch, params
                        )
                    else:
                        need_resolve = True

            # -----------------------------------------------------------------
            # Timing & articulation
            # -----------------------------------------------------------------
            dur_beats = step_beats * (0.70 + 0.25 * float(rng.random()))
            if is_anchor:
                dur_beats = max(dur_beats, 1.0)
                if is_cadence_bar:
                    dur_beats = max(dur_beats, 1.6)

            dur_sec = dur_beats * spb
            # Keep within total duration
            dur_sec = min(dur_sec, max(0.02, params.duration - start_sec))

            # Human timing jitter (keep anchors tighter)
            jitter = 0.0
            if params.human > 0 and not is_anchor:
                jitter = float(rng.normal(0.0, params.human * 0.006 * spb))
                jitter = float(np.clip(jitter, -0.02 * spb, 0.02 * spb))
            start_sec_j = float(np.clip(start_sec + jitter, 0.0, max(0.0, params.duration - 0.01)))

            # Velocity: slightly higher during tension, softer on cadences
            amp = 0.12 + 0.05 * tension
            if is_cadence_bar and pos_beats >= 2.0:
                amp *= 0.85
            if is_anchor:
                amp *= 1.10

            # Optional chromatic approach (near phrase ends / near target)
            use_chromatic = (
                (policy.chromatic_prob > 0)
                and (not is_anchor)
                and (abs((next_anchor - pitch)) <= 2)
                and (rng.random() < (policy.chromatic_prob * (0.35 + 0.65 * tension)))
            )

            if use_chromatic:
                target = next_anchor
                sign = 1 if (target - pitch) >= 0 else -1
                sem = params.semitone_from_degree(target) - sign  # approach by semitone
                freq = freq_from_note(params.root, sem, base_oct)
            else:
                freq = params.get_scale_freq(pitch, base_oct)

            events.append(
                NoteEvent(
                    start_sec=start_sec_j,
                    dur_sec=dur_sec,
                    freq=freq,
                    amp=float(amp),
                    is_anchor=is_anchor,
                )
            )

            if is_phrase_start and not use_motif and not motif and not is_anchor:
                # Capture a first motif only once, using early phrase material
                # (store offset from anchor at fixed grid positions)
                bar_notes_for_motif.append((step_idx, pitch - anchor_pitch))

            current_pitch = pitch

        # If we didn't have a motif yet, capture a small one from the first phrase start bar.
        if not motif and is_phrase_start and bar_notes_for_motif:
            # Keep it short and memorable
            motif = bar_notes_for_motif[: min(5, len(bar_notes_for_motif))]

    return events


def melody_procedural(params: SynthParams) -> FloatArray:
    """Procedural, chord-aware melody with phrases + tension/resolution."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    events = _generate_procedural_melody_events(params)

    signal = np.zeros(int(sr * dur))

    cutoff = 700 * params.brightness + 180
    for ev in events:
        start_idx = int(ev.start_sec * sr)
        if start_idx >= len(signal):
            continue

        note = osc(ev.freq, ev.dur_sec, sr, ev.amp)
        note = apply_lowpass(note, cutoff, sr)

        # Articulation: anchors get slightly softer attack and longer release
        atk = (0.05 if ev.is_anchor else 0.03) * params.attack_mult
        rel = (0.22 if ev.is_anchor else 0.16) * params.attack_mult
        note = apply_adsr(note, atk, 0.12, 0.55, rel, sr)
        note = apply_humanize(note, params.human * 0.6, sr, params.rng)

        add_note(signal, note, start_idx)

    # Space FX
    signal = apply_delay(signal, 0.33, 0.35 * params.echo_mult, 0.28 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.55, params.space * 0.75, sr)

    return signal


def melody_contemplative(params: SynthParams) -> FloatArray:
    """Slow, contemplative melody."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    # Sparse melody pattern (tuple), -1 = rest
    pattern = (0, -1, 2, -1, 4, -1, 2, -1, 0, -1, -1, -1, 2, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = params.get_scale_freq(degree, 5)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 1.5, sr, 0.18)
        note = apply_lowpass(note, 800 * params.brightness + 200, sr)
        note = apply_adsr(note, 0.1 * params.attack_mult, 0.3, 0.6, 0.8 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr, params.rng)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.35, 0.4 * params.echo_mult, 0.3 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.6, params.space * 0.8, sr)

    return signal


def melody_rising(params: SynthParams) -> FloatArray:
    """Ascending melodic line."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    # Rising pattern (tuple)
    pattern = (0, -1, 2, -1, 4, -1, -1, 5, -1, -1, 6, -1, -1, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = params.get_scale_freq(degree, 5)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 1.8, sr, 0.16)
        note = apply_lowpass(note, 900 * params.brightness + 250, sr)
        note = apply_adsr(note, 0.08 * params.attack_mult, 0.25, 0.55, 0.9 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr, params.rng)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.33, 0.35 * params.echo_mult, 0.28 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.55, params.space * 0.75, sr)

    return signal


def melody_falling(params: SynthParams) -> FloatArray:
    """Descending melodic line."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    # Falling pattern (tuple)
    pattern = (6, -1, -1, 5, -1, -1, 4, -1, 2, -1, -1, 0, -1, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = params.get_scale_freq(degree, 5)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 1.6, sr, 0.17)
        note = apply_lowpass(note, 850 * params.brightness + 220, sr)
        note = apply_adsr(note, 0.1 * params.attack_mult, 0.28, 0.52, 0.85 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr, params.rng)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.38, 0.38 * params.echo_mult, 0.3 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.58, params.space * 0.78, sr)

    return signal


def melody_minimal(params: SynthParams) -> FloatArray:
    """Very sparse, minimal melody."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_sine)

    # Extremely sparse (tuple)
    pattern = (4, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = params.get_scale_freq(degree, 5)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 2.5, sr, 0.20)
        note = apply_lowpass(note, 600 * params.brightness + 150, sr)
        note = apply_adsr(note, 0.15 * params.attack_mult, 0.4, 0.5, 1.2 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr, params.rng)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.5, 0.45 * params.echo_mult, 0.4 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.7, params.space * 0.9, sr)

    return signal


def melody_ornamental(params: SynthParams) -> FloatArray:
    """Ornamental melody with grace notes."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    # Pattern with ornaments (tuples)
    main_notes = (0, 4, 2, 0)
    grace_offsets = (2, 5, 3, 2)
    note_dur = dur / (len(main_notes) * 2)
    signal = np.zeros(int(sr * dur))

    for i, (main, grace) in enumerate(zip(main_notes, grace_offsets)):
        start = int(i * 2 * note_dur * sr)

        # Grace note (quick)
        grace_freq = params.get_scale_freq(grace, 5)
        grace_note = osc(grace_freq, note_dur * 0.15, sr, 0.10)
        grace_note = apply_lowpass(grace_note, 1000 * params.brightness, sr)
        grace_note = apply_adsr(grace_note, 0.01, 0.05, 0.3, 0.1, sr)

        # Main note
        main_freq = params.get_scale_freq(main, 5)
        main_note = osc(main_freq, note_dur * 1.5, sr, 0.18)
        main_note = apply_lowpass(main_note, 900 * params.brightness + 200, sr)
        main_note = apply_adsr(
            main_note, 0.08 * params.attack_mult, 0.3, 0.55, 0.8 * params.attack_mult, sr
        )
        main_note = apply_humanize(main_note, params.human, sr, params.rng)

        grace_start = start
        main_start = start + int(note_dur * 0.15 * sr)

        add_note(signal, grace_note, grace_start)
        add_note(signal, main_note, main_start)

    signal = apply_delay(signal, 0.3, 0.35 * params.echo_mult, 0.25 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.5, params.space * 0.7, sr)

    return signal


def melody_arp(params: SynthParams) -> FloatArray:
    """Arpeggiated melody."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    # Fast arpeggio pattern (tuple * 8)
    base_pattern = (0, 2, 4, 2)
    pattern = base_pattern * 8
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        freq = params.get_scale_freq(degree, 4)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 0.8, sr, 0.14)
        note = apply_lowpass(note, 1200 * params.brightness + 300, sr)
        note = apply_adsr(note, 0.02 * params.attack_mult, 0.1, 0.4, 0.15 * params.attack_mult, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.25, 0.3 * params.echo_mult, 0.25 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.4, params.space * 0.6, sr)

    return signal


MELODY_PATTERNS: dict[str, PatternFn] = {
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


# -----------------------------------------------------------------------------
# RHYTHM PATTERNS
# -----------------------------------------------------------------------------


def rhythm_none(params: SynthParams) -> FloatArray:
    """No rhythm."""
    return np.zeros(int(SAMPLE_RATE * params.duration))


def rhythm_minimal(params: SynthParams) -> FloatArray:
    """Minimal rhythm - just occasional hits."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Sparse kicks (tuple)
    pattern = (1, 0, 0, 0, 0, 0, 0, 0) * 2
    hit_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, hit in enumerate(pattern):
        if not hit:
            continue

        start = int(i * hit_dur * sr)

        # Kick: sine with pitch envelope
        kick_dur = 0.15
        kick = generate_sine(60, kick_dur, sr, 0.35)
        pitch_env = np.exp(-np.linspace(0, 8, len(kick)))
        kick = kick * pitch_env
        kick = apply_lowpass(kick, 150, sr)
        kick = apply_humanize(kick, params.human, sr, params.rng)

        add_note(signal, kick, start)

    signal = apply_reverb(signal, params.space * 0.3, params.space * 0.5, sr)
    return signal


def rhythm_heartbeat(params: SynthParams) -> FloatArray:
    """Heartbeat-like rhythm."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Double-hit pattern (tuple)
    pattern = (1, 1, 0, 0, 0, 0, 0, 0) * 2
    hit_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, hit in enumerate(pattern):
        if not hit:
            continue

        start = int(i * hit_dur * sr)

        kick_dur = 0.12
        kick = generate_sine(55, kick_dur, sr, 0.32)
        pitch_env = np.exp(-np.linspace(0, 10, len(kick)))
        kick = kick * pitch_env
        kick = apply_lowpass(kick, 120, sr)

        add_note(signal, kick, start)

    return signal


def rhythm_soft_four(params: SynthParams) -> FloatArray:
    """Soft four-on-the-floor."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Kick on every beat
    num_beats = 8
    beat_dur = dur / num_beats
    signal = np.zeros(int(sr * dur))

    for i in range(num_beats):
        start = int(i * beat_dur * sr)

        # Soft kick
        kick_dur = 0.18
        kick = generate_sine(50, kick_dur, sr, 0.28)
        pitch_env = np.exp(-np.linspace(0, 6, len(kick)))
        kick = kick * pitch_env
        kick = apply_lowpass(kick, 100, sr)
        kick = apply_humanize(kick, params.human, sr, params.rng)

        add_note(signal, kick, start)

    signal = apply_reverb(signal, params.space * 0.25, params.space * 0.4, sr)
    return signal


def rhythm_hats_only(params: SynthParams) -> FloatArray:
    """Just hi-hats."""
    sr = SAMPLE_RATE
    dur = params.duration

    # 16th note hats
    num_hits = 32
    hit_dur = dur / num_hits
    signal = np.zeros(int(sr * dur))

    for i in range(num_hits):
        start = int(i * hit_dur * sr)

        # Hi-hat: filtered noise
        hat_dur = 0.05
        hat = generate_noise(hat_dur, sr, 0.08, params.rng)
        hat = apply_highpass(hat, 6000, sr)
        hat = apply_adsr(hat, 0.001, 0.02, 0.1, 0.03, sr)
        hat = apply_humanize(hat, params.human, sr, params.rng)

        add_note(signal, hat, start)

    return signal


def rhythm_electronic(params: SynthParams) -> FloatArray:
    """Electronic beat."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))
    beat_dur = dur / 16

    # Patterns (tuples)
    kick_pattern = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
    hat_pattern = (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)

    for i in range(16):
        start = int(i * beat_dur * sr)

        if kick_pattern[i]:
            kick = generate_sine(45, 0.2, sr, 0.35)
            pitch_env = np.exp(-np.linspace(0, 8, len(kick)))
            kick = kick * pitch_env
            kick = apply_lowpass(kick, 100, sr)
            add_note(signal, kick, start)

        if hat_pattern[i]:
            hat = generate_noise(0.06, sr, 0.06, params.rng)
            hat = apply_highpass(hat, 7000, sr)
            hat = apply_adsr(hat, 0.001, 0.02, 0.1, 0.04, sr)
            add_note(signal, hat, start)

    return signal


RHYTHM_PATTERNS: dict[str, PatternFn] = {
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


# -----------------------------------------------------------------------------
# TEXTURE PATTERNS
# -----------------------------------------------------------------------------


def texture_none(params: SynthParams) -> FloatArray:
    """No texture."""
    return np.zeros(int(SAMPLE_RATE * params.duration))


def texture_shimmer(params: SynthParams) -> FloatArray:
    """High, shimmering texture."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # High sine clusters with amplitude modulation (tuple)
    for degree in (0, 2, 4):
        freq = params.get_scale_freq(degree, 6)
        tone = generate_sine(freq, dur, sr, 0.04)

        # Amplitude modulation for shimmer
        lfo_rate = 2.0 / (params.motion + 0.2)
        lfo = generate_lfo(dur, lfo_rate, sr)
        tone = tone * (0.5 + 0.5 * lfo)

        tone = apply_adsr(tone, 0.5 * params.attack_mult, 0.3, 0.8, 1.5 * params.attack_mult, sr)
        signal += tone

    signal = apply_delay(signal, 0.4, 0.5 * params.echo_mult, 0.4 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.8, params.space, sr)

    return signal


def texture_shimmer_slow(params: SynthParams) -> FloatArray:
    """Slow, gentle shimmer."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    for degree in (0, 4):
        freq = params.get_scale_freq(degree, 6)
        tone = generate_sine(freq, dur, sr, 0.035)

        # Very slow amplitude modulation
        lfo_rate = 0.5 / (params.motion + 0.2)
        lfo = generate_lfo(dur, lfo_rate, sr)
        tone = tone * (0.4 + 0.6 * lfo)

        tone = apply_adsr(tone, 1.0 * params.attack_mult, 0.5, 0.85, 2.0 * params.attack_mult, sr)
        signal += tone

    signal = apply_delay(signal, 0.5, 0.55 * params.echo_mult, 0.45 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.85, params.space, sr)

    return signal


def texture_vinyl_crackle(params: SynthParams) -> FloatArray:
    """Vinyl crackle texture."""
    sr = SAMPLE_RATE
    dur = params.duration
    rng = params.rng

    # Sparse noise impulses
    signal = np.zeros(int(sr * dur))

    num_crackles = int(dur * 20)  # ~20 crackles per second
    max_pos = len(signal) - 100

    if max_pos > 0:
        for _ in range(num_crackles):
            pos = int(rng.integers(0, max_pos))
            amp = float(rng.uniform(0.01, 0.04))

            crackle = generate_noise(0.002, sr, amp, rng)
            crackle = apply_highpass(crackle, 2000, sr)

            add_note(signal, crackle, pos)

    # Soft background hiss
    hiss = generate_noise(dur, sr, 0.008, rng)
    hiss = apply_lowpass(hiss, 8000, sr)
    hiss = apply_highpass(hiss, 1000, sr)
    signal += hiss

    return signal


def texture_breath(params: SynthParams) -> FloatArray:
    """Breathing texture."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Filtered noise with slow envelope
    signal = generate_noise(dur, sr, 0.06, params.rng)

    # Bandpass around a note frequency
    freq = params.get_scale_freq(0, 3)
    signal = apply_lowpass(signal, freq * 2, sr)
    signal = apply_highpass(signal, freq * 0.5, sr)

    # Breathing envelope (slow LFO)
    breath_rate = 0.2 / (params.motion + 0.1)
    lfo = generate_lfo(dur, breath_rate, sr)
    signal = signal * lfo

    signal = apply_reverb(signal, params.space * 0.6, params.space * 0.8, sr)

    return signal


def texture_stars(params: SynthParams) -> FloatArray:
    """Twinkling stars texture."""
    sr = SAMPLE_RATE
    dur = params.duration
    rng = params.rng

    signal = np.zeros(int(sr * dur))

    # Random high plinks
    num_stars = int(dur * 3)  # ~3 per second

    # Scale degrees for stars (tuple)
    star_degrees = (0, 2, 4, 5)

    max_pos = len(signal) - sr
    if max_pos > 0:
        for _ in range(num_stars):
            pos = int(rng.integers(0, max_pos))

            degree = int(rng.choice(star_degrees))
            freq = params.get_scale_freq(degree, 6)

            amp = float(rng.uniform(0.02, 0.05))
            star = generate_sine(freq, 0.3, sr, amp)
            star = apply_adsr(star, 0.01, 0.1, 0.1, 0.2, sr)

            add_note(signal, star, pos)

    signal = apply_delay(signal, 0.4, 0.5 * params.echo_mult, 0.4 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.9, params.space, sr)

    return signal


TEXTURE_PATTERNS: dict[str, PatternFn] = {
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


# -----------------------------------------------------------------------------
# ACCENT PATTERNS
# -----------------------------------------------------------------------------


def accent_none(params: SynthParams) -> FloatArray:
    """No accent."""
    return np.zeros(int(SAMPLE_RATE * params.duration))


def accent_bells(params: SynthParams) -> FloatArray:
    """Bell-like accents."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # Sparse bell hits (tuple)
    pattern = (0, -1, -1, -1, 4, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1)
    hit_dur = dur / len(pattern)

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        start = int(i * hit_dur * sr)
        freq = params.get_scale_freq(degree, 5)

        # Bell: mix of harmonics with fast decay
        bell_dur = 0.8
        bell = generate_sine(freq, bell_dur, sr, 0.12)
        bell += generate_sine(freq * 2.0, bell_dur, sr, 0.06)
        bell += generate_sine(freq * 3.0, bell_dur, sr, 0.03)
        bell = apply_adsr(bell, 0.005 * params.attack_mult, 0.2, 0.1, 0.6 * params.attack_mult, sr)
        bell = apply_humanize(bell, params.human, sr, params.rng)

        add_note(signal, bell, start)

    signal = apply_reverb(signal, params.space * 0.6, params.space * 0.8, sr)
    return signal


def accent_pluck(params: SynthParams) -> FloatArray:
    """Plucked string accents."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # Pattern (tuple)
    pattern = (0, -1, -1, 4, -1, -1, 2, -1, -1, -1, 0, -1, -1, -1, -1, -1)
    hit_dur = dur / len(pattern)

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        start = int(i * hit_dur * sr)
        freq = params.get_scale_freq(degree, 4)

        # Pluck: sharp attack, quick decay
        pluck = generate_triangle(freq, 0.5, sr, 0.15)
        pluck = apply_lowpass(pluck, 1500 * params.brightness + 400, sr)
        pluck = apply_adsr(
            pluck, 0.003 * params.attack_mult, 0.15, 0.05, 0.4 * params.attack_mult, sr
        )
        pluck = apply_humanize(pluck, params.human, sr, params.rng)

        add_note(signal, pluck, start)

    signal = apply_reverb(signal, params.space * 0.5, params.space * 0.7, sr)
    return signal


def accent_chime(params: SynthParams) -> FloatArray:
    """Wind chime accents."""
    sr = SAMPLE_RATE
    dur = params.duration
    rng = params.rng

    signal = np.zeros(int(sr * dur))

    # Random chime hits
    num_chimes = int(dur * 1.5)

    # Chime degrees (tuple)
    chime_degrees = (0, 2, 4, 5, 6)

    max_pos = len(signal) - sr
    if max_pos > 0:
        for _ in range(num_chimes):
            pos = int(rng.integers(0, max_pos))

            degree = int(rng.choice(chime_degrees))
            freq = params.get_scale_freq(degree, 5)

            chime_dur = 1.2
            amp = float(rng.uniform(0.06, 0.12))
            chime = generate_sine(freq, chime_dur, sr, amp)
            chime += generate_sine(freq * 2.0, chime_dur, sr, 0.03)
            chime = apply_adsr(chime, 0.002, 0.3, 0.05, 0.9, sr)

            add_note(signal, chime, pos)

    signal = apply_delay(signal, 0.3, 0.4 * params.echo_mult, 0.3 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.75, params.space * 0.9, sr)

    return signal


ACCENT_PATTERNS: dict[str, PatternFn] = {
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


# =============================================================================
# PART 3: ASSEMBLER - CONFIG → AUDIO
# =============================================================================


@dataclass
class MusicConfig:
    """Complete V1/V2 configuration."""

    tempo: float = 0.35
    root: str = "c"
    mode: str = "minor"
    brightness: float = 0.5
    space: float = 0.6
    density: int = 5

    # Layer selections
    bass: str = "drone"
    pad: str = "warm_slow"
    melody: str = "contemplative"
    rhythm: str = "minimal"
    texture: str = "shimmer"
    accent: str = "bells"

    # V2 parameters
    motion: float = 0.5
    attack: str = "medium"
    stereo: float = 0.5
    depth: bool = False
    echo: float = 0.5
    human: float = 0.0
    grain: str = "clean"

    # New: procedural melody engine (optional; defaults keep backwards compat)
    melody_engine: str = "pattern"  # "pattern" | "procedural"

    # Procedural melody policy knobs (only used when melody_engine="procedural")
    phrase_len_bars: int = 4
    melody_density: float = 0.45
    syncopation: float = 0.20
    swing: float = 0.0
    motif_repeat_prob: float = 0.50
    step_bias: float = 0.75
    chromatic_prob: float = 0.05
    cadence_strength: float = 0.65
    register_min_oct: int = 4
    register_max_oct: int = 6
    tension_curve: str = "arc"  # "arc" | "ramp" | "waves"

    # Harmony controls (used by bass/pad + procedural melody)
    harmony_style: str = "auto"  # "auto" | "pop" | "jazz" | "cinematic" | "ambient"
    chord_change_bars: int = 1  # bars per chord
    chord_extensions: str = "triads"  # "triads" | "sevenths" | "lush"

    # Random seed for deterministic renders (0 = random)
    seed: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MusicConfig":
        """Create config from dict (e.g., from JSON)."""
        config = cls()

        for key, value in d.items():
            if key == "layers" and isinstance(value, dict):
                layers = cast(dict[str, Any], value)
                for layer_key, layer_value in layers.items():
                    if hasattr(config, layer_key):
                        setattr(config, layer_key, layer_value)

            # Optional nested policy dictionaries (LLM-friendly)
            elif key in ("melody_policy", "procedural_melody") and isinstance(value, dict):
                policy = cast(dict[str, Any], value)
                for k, v in policy.items():
                    if hasattr(config, k):
                        setattr(config, k, v)

            elif key in ("harmony", "harmony_plan") and isinstance(value, dict):
                harmony = cast(dict[str, Any], value)
                for k, v in harmony.items():
                    if hasattr(config, k):
                        setattr(config, k, v)

            elif hasattr(config, key):
                setattr(config, key, value)

        return config


# -----------------------------------------------------------------------------
# HARMONY + POLICY BUILDERS (from config only)
# -----------------------------------------------------------------------------


def build_melody_policy(config: MusicConfig) -> MelodyPolicy:
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


def build_harmony_plan(config: MusicConfig, params: SynthParams) -> HarmonyPlan:
    """
    Generate a simple diatonic chord timeline in scale-degrees.

    This is intentionally compute-light: a handful of templates + repetition.
    """
    beats_per_bar = 4
    bars_total = int(np.ceil(params.beats_total / beats_per_bar))
    bars_total = max(1, bars_total)

    chord_change_bars = int(max(1, config.chord_change_bars))
    n_chords = int(np.ceil(bars_total / chord_change_bars))
    n_chords = max(1, n_chords)

    style = (config.harmony_style or "auto").lower()
    if style == "auto":
        # Heuristic: guess harmony style from the rest of the config
        if config.rhythm == "none" and config.space >= 0.7:
            style = "ambient"
        elif config.pad in ("cinematic",) or config.bass in ("drone", "sub_pulse"):
            style = "cinematic"
        elif config.texture in ("vinyl_crackle",) or config.melody in ("ornamental",):
            style = "jazz"
        else:
            style = "pop"

    mode = (config.mode or "minor").lower()
    majorish = mode in ("major", "mixolydian")

    rng = params.rng

    # Templates are scale-degree roots (0..6)
    pop_major = (
        (0, 5, 3, 4),  # I–vi–IV–V
        (0, 3, 4, 0),  # I–IV–V–I
        (0, 4, 5, 3),  # I–V–vi–IV
    )
    pop_minor = (
        (0, 5, 3, 6),  # i–VI–IV–VII
        (0, 3, 6, 4),  # i–IV–VII–V
        (0, 6, 3, 5),  # i–VII–IV–VI
        (0, 4, 5, 3),  # i–V–VI–IV (works in many minor-ish modes)
    )
    jazz_major = (
        (1, 4, 0, 0),  # ii–V–I–I
        (1, 4, 0, 5),  # ii–V–I–vi
        (5, 1, 4, 0),  # vi–ii–V–I
    )
    jazz_minor = (
        (1, 4, 0, 0),  # ii°–V–i–i (approx)
        (5, 1, 4, 0),  # VI–ii–V–i (approx)
    )
    cinematic_minor = (
        (0, 6, 5, 3),
        (0, 5, 6, 4),
        (0, 6, 3, 4),
    )
    ambient_any = (
        (0, 5, 0, 3),
        (0, 4, 0, 6),
        (0, 3, 0, 5),
    )

    if style == "jazz":
        template = rng.choice(jazz_major if majorish else jazz_minor)
    elif style == "cinematic":
        template = rng.choice(cinematic_minor if not majorish else pop_major)
    elif style == "ambient":
        template = rng.choice(ambient_any)
    else:  # pop / default
        template = rng.choice(pop_major if majorish else pop_minor)

    # Build chord roots repeating the template
    roots: List[int] = []
    while len(roots) < n_chords:
        roots.extend(int(x) for x in template)
    roots = roots[:n_chords]

    chord_beats = float(chord_change_bars * beats_per_bar)

    chords: List[ChordEvent] = []
    for i, deg in enumerate(roots):
        chords.append(
            ChordEvent(
                start_beat=i * chord_beats, duration_beats=chord_beats, root_degree=int(deg) % 7
            )
        )

    return HarmonyPlan(chords=tuple(chords), beats_per_bar=beats_per_bar)


def apply_master_fx(
    signal: FloatArray,
    config: MusicConfig,
    sr: int = SAMPLE_RATE,
    loudness: bool = True,
) -> FloatArray:
    """Optional master-bus FX using Spotify's pedalboard when available.

    This is intentionally subtle and safe:
    - If pedalboard isn't installed, it's a no-op.
    - If `loudness` is False, it skips compressor/limiter to preserve dynamics
      for chunked automation renders.
    """
    if not pedalboard_available or Pedalboard is None:
        return signal

    fx_chain: list[Any] = []

    # --- Tone / color --------------------------------------------------------
    grain = getattr(config, "grain", "clean")
    if grain == "warm":
        if PBDistortion is not None:
            # Gentle saturation
            fx_chain.append(_pb_make_effect(PBDistortion, drive_db=3.0))
    elif grain == "gritty":
        if PBBitcrush is not None:
            # Subtle lo-fi crunch (conservative defaults)
            fx_chain.append(_pb_make_effect(PBBitcrush, bit_depth=10))
        if PBDistortion is not None:
            fx_chain.append(_pb_make_effect(PBDistortion, drive_db=6.0))

    # Add subtle chorus for motion if available
    motion = float(getattr(config, "motion", 0.0))
    stereo = float(getattr(config, "stereo", 0.0))
    if PBChorus is not None and motion > 0.7 and stereo > 0.2:
        # Best-effort: many pedalboard versions support some subset of these params
        fx_chain.append(
            _pb_make_effect(
                PBChorus,
                rate_hz=_clamp(0.15 + motion * 0.8, 0.05, 5.0),
                depth=_clamp(0.05 + motion * 0.25, 0.0, 1.0),
                mix=_clamp(0.06 + stereo * 0.18, 0.0, 0.5),
            )
        )

    # --- Loudness glue -------------------------------------------------------
    if loudness:
        if PBCompressor is not None:
            fx_chain.append(
                _pb_make_effect(
                    PBCompressor,
                    threshold_db=-24.0,
                    ratio=2.0,
                    attack_ms=5.0,
                    release_ms=80.0,
                )
            )
        if PBLimiter is not None:
            fx_chain.append(_pb_make_effect(PBLimiter, threshold_db=-1.0))

    # Filter out Nones and bail if nothing constructed successfully
    fx_chain = [fx for fx in fx_chain if fx is not None]
    if not fx_chain:
        return signal

    return _pb_process_mono(signal, sr, fx_chain)


def assemble(config: MusicConfig, duration: float = 16.0, normalize: bool = True) -> FloatArray:
    """
    Assemble all layers into final audio.

    This is the core function that converts a config into audio.

    Args:
        config: MusicConfig object
        duration: Duration in seconds
        normalize: If True, maximizes volume. Set False for automation chunks
                   to preserve relative dynamics.
    """
    sr = SAMPLE_RATE

    # Build synth params
    params = SynthParams(
        root=config.root,
        mode=config.mode,
        brightness=config.brightness,
        space=config.space,
        duration=duration,
        tempo=config.tempo,
        motion=config.motion,
        attack=config.attack,
        stereo=config.stereo,
        depth=config.depth,
        echo=config.echo,
        human=config.human,
        grain=config.grain,
        # New
        chord_extensions=config.chord_extensions,
        rng=np.random.default_rng(config.seed if config.seed else None),
    )

    # Shared harmony + procedural melody policy (bass/pad/procedural melody use this)
    params.harmony = build_harmony_plan(config, params)
    params.melody_policy = build_melody_policy(config)

    # Determine active layers based on density
    active_layers = DENSITY_LAYERS.get(config.density, DENSITY_LAYERS[5])

    # Initialize output
    output = np.zeros(int(sr * duration))

    # Generate each layer
    if "bass" in active_layers:
        bass_fn = BASS_PATTERNS.get(config.bass, bass_drone)
        output += bass_fn(params)

    if config.depth:
        # Add sub-bass layer
        sub_params = replace(params, duration=duration)
        output += bass_sub_pulse(sub_params) * 0.6

    if "pad" in active_layers:
        pad_fn = PAD_PATTERNS.get(config.pad, pad_warm_slow)
        output += pad_fn(params)

    if "melody" in active_layers:
        if config.melody_engine == "procedural":
            output += melody_procedural(params)
        else:
            melody_fn = MELODY_PATTERNS.get(config.melody, melody_contemplative)
            output += melody_fn(params)

    if "rhythm" in active_layers and config.rhythm != "none":
        rhythm_fn = RHYTHM_PATTERNS.get(config.rhythm, rhythm_none)
        output += rhythm_fn(params)

    if "texture" in active_layers and config.texture != "none":
        texture_fn = TEXTURE_PATTERNS.get(config.texture, texture_none)
        output += texture_fn(params)

    if "accent" in active_layers and config.accent != "none":
        accent_fn = ACCENT_PATTERNS.get(config.accent, accent_none)
        output += accent_fn(params)

    # Master bus FX (pedalboard when available).
    output = apply_master_fx(output, config, sr, loudness=normalize)

    # Normalize only if requested
    if normalize:
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.85

    return output


def config_to_audio(config: MusicConfig, output_path: str, duration: float = 16.0) -> str:
    """
    Convert a MusicConfig to an audio file.

    Args:
        config: MusicConfig object
        output_path: Path to save the WAV file
        duration: Duration in seconds

    Returns:
        Path to the saved file
    """
    audio = assemble(config, duration)
    write_audio = cast(Callable[[str, FloatArray, int], None], sf.write)
    write_audio(output_path, audio, SAMPLE_RATE)
    return output_path


def dict_to_audio(config_dict: Dict[str, Any], output_path: str, duration: float = 16.0) -> str:
    """
    Convert a config dict (from JSON) directly to audio.

    Args:
        config_dict: Dict with config values
        output_path: Path to save the WAV file
        duration: Duration in seconds

    Returns:
        Path to the saved file
    """
    config = MusicConfig.from_dict(config_dict)
    return config_to_audio(config, output_path, duration)


# =============================================================================
# PART 4: CONVENIENCE FUNCTIONS
# =============================================================================


def generate_from_vibe(vibe: str, output_path: str = "output.wav", duration: float = 16.0) -> str:
    """
    Generate audio from a vibe description.

    This is a placeholder - in production, this would call an LLM to generate the config.
    For now, uses simple keyword matching.
    """
    vibe_lower = vibe.lower()

    # Simple keyword-based config generation
    config = MusicConfig()

    # Root/mode selection
    if any(w in vibe_lower for w in ("dark", "sad", "night", "mysterious")):
        config.root = "d"
        config.mode = "dorian"
        config.brightness = 0.35
    elif any(w in vibe_lower for w in ("happy", "bright", "joy")):
        config.root = "c"
        config.mode = "major"
        config.brightness = 0.65
    elif any(w in vibe_lower for w in ("epic", "cinematic", "powerful")):
        config.root = "d"
        config.mode = "minor"
        config.brightness = 0.5
        config.depth = True
    elif any(w in vibe_lower for w in ("indian", "spiritual", "meditation")):
        config.root = "d"
        config.mode = "dorian"
        config.human = 0.2
        config.grain = "warm"
    else:
        config.root = "c"
        config.mode = "minor"

    # Tempo
    if any(w in vibe_lower for w in ("slow", "calm", "meditation")):
        config.tempo = 0.30
    elif any(w in vibe_lower for w in ("fast", "energy", "drive")):
        config.tempo = 0.45
    else:
        config.tempo = 0.36

    # Space
    if any(w in vibe_lower for w in ("vast", "space", "underwater", "cave")):
        config.space = 0.85
        config.echo = 0.75
    elif any(w in vibe_lower for w in ("intimate", "close")):
        config.space = 0.4
        config.echo = 0.3

    # Layers based on keywords
    if any(w in vibe_lower for w in ("electronic", "synth")):
        config.rhythm = "electronic"
        config.attack = "sharp"
    elif any(w in vibe_lower for w in ("ambient", "peaceful")):
        config.rhythm = "none"
        config.attack = "soft"

    if any(w in vibe_lower for w in ("minimal", "sparse")):
        config.density = 3
        config.melody = "minimal"
    elif any(w in vibe_lower for w in ("rich", "full", "lush")):
        config.density = 6

    return config_to_audio(config, output_path, duration)


# =============================================================================
# PART 5: TRANSITION FUNCTIONS
# =============================================================================


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: t=0 returns a, t=1 returns b"""
    return a + (b - a) * t


def interpolate_configs(config_a: MusicConfig, config_b: MusicConfig, t: float) -> MusicConfig:
    """
    Interpolate between two configs.
    t=0.0 → config_a
    t=1.0 → config_b
    """
    return MusicConfig(
        tempo=lerp(config_a.tempo, config_b.tempo, t),
        root=config_a.root if t < 0.5 else config_b.root,
        mode=config_a.mode if t < 0.5 else config_b.mode,
        brightness=lerp(config_a.brightness, config_b.brightness, t),
        space=lerp(config_a.space, config_b.space, t),
        density=round(lerp(config_a.density, config_b.density, t)),
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
        # New: procedural melody engine + policy knobs
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
        # New: harmony knobs
        harmony_style=config_a.harmony_style if t < 0.5 else config_b.harmony_style,
        chord_change_bars=round(lerp(config_a.chord_change_bars, config_b.chord_change_bars, t)),
        chord_extensions=config_a.chord_extensions if t < 0.5 else config_b.chord_extensions,
        seed=config_a.seed if t < 0.5 else config_b.seed,
    )


def morph_audio(
    config_a: MusicConfig, config_b: MusicConfig, duration: float = 60.0, segments: int = 8
) -> FloatArray:
    """
    Generate audio that morphs from config_a to config_b over duration.
    """
    segment_duration = duration / segments
    output = []

    for i in range(segments):
        t = i / (segments - 1) if segments > 1 else 0.0
        interpolated = interpolate_configs(config_a, config_b, t)
        # Don't normalize individual segments — preserve relative dynamics
        segment = assemble(interpolated, segment_duration, normalize=False)
        output.append(segment)

    result = np.concatenate(output)

    # Normalize the final result
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val * 0.85

    return result


def crossfade(audio_a: FloatArray, audio_b: FloatArray, crossfade_samples: int) -> FloatArray:
    """Crossfade between two audio arrays around the midpoint.

    This helper is deliberately defensive:
    - clamps the crossfade length to the available audio
    - handles very short buffers
    - supports odd crossfade lengths
    """
    min_len = min(len(audio_a), len(audio_b))
    if min_len <= 0:
        return np.zeros(0, dtype=np.float64)

    a = audio_a[:min_len]
    b = audio_b[:min_len]

    cf = int(max(0, crossfade_samples))
    if cf <= 1:
        mid = min_len // 2
        return np.concatenate((a[:mid], b[mid:]))

    cf = min(cf, min_len)

    mid = min_len // 2
    half = cf // 2
    start = mid - half
    end = start + cf

    if start < 0:
        start = 0
        end = cf
    if end > min_len:
        end = min_len
        start = max(0, end - cf)

    cf = end - start
    if cf <= 1:
        return np.concatenate((a[:start], b[start:]))

    fade_out = np.linspace(1.0, 0.0, cf)
    fade_in = 1.0 - fade_out

    cross = a[start:end] * fade_out + b[start:end] * fade_in

    result = np.concatenate((a[:start], cross, b[end:]))
    return result


def transition(config_a: MusicConfig, config_b: MusicConfig, duration: float = 60.0) -> FloatArray:
    """Generate with crossfade transition."""
    # Don't normalize individual tracks — normalize after crossfade
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
    config_a: MusicConfig,
    config_b: MusicConfig,
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

if __name__ == "__main__":
    print("VibeSynth V1/V2 - Pure Python Synthesis Engine")
    print("=" * 50)

    # Example 3: Bubblegum, sad, dead (from llm_to_synth.py, see @file_context_1)
    import json

    # These examples mirror the demo config blocks captured in @file_context_0:
    demo_configs: list[MusicConfigDict] = [
        {
            # "justification": "The 'Bubblegum'-style sound is characterized by a bright, slightly distorted, and playful melody with a prominent bass line and a smooth, evolving pad.",
            "tempo": 0.5,
            "root": "a",
            "mode": "major",
            "brightness": 0.75,
            "space": 0.5,
            "density": 2,
            "bass": "drone",
            "pad": "warm_slow",
            "melody": "contemplative",
            "rhythm": "none",
            "texture": "shimmer",
            "accent": "bells",
            "motion": 0.5,
            "attack": "medium",
            "stereo": 0.5,
            "depth": False,
            "echo": 0.25,
            "human": 0.5,
            "grain": "clean",
        },
        {
            # "justification": "The sad vibe is characterized by a low-key sound and a sense of melancholy. The low intensity and muted tones create a feeling of quiet sorrow. The 'low' intensity of the bass and the 'dark' tone of the pad support this feeling. The 'soft' attack and 'warm' tone of the pad create a sense of longing and a feeling of sadness.",
            "tempo": 0.25,
            "root": "e",
            "mode": "minor",
            "brightness": 0.25,
            "space": 0.25,
            "density": 5,
            "bass": "drone",
            "pad": "warm_slow",
            "melody": "contemplative",
            "rhythm": "soft_four",
            "texture": "shimmer_slow",
            "accent": "bells",
            "motion": 0.25,
            "attack": "soft",
            "stereo": 0.25,
            "depth": True,
            "echo": 0.5,
            "human": 0.5,
            "grain": "gritty",
        },
        {
            # "justification": "The 'dead', a low volume synth with a simple sine wave and a low pitch.",
            "tempo": 0.25,
            "root": "c",
            "mode": "minor",
            "brightness": 0.25,
            "space": 0.25,
            "density": 2,
            "bass": "drone",
            "pad": "warm_slow",
            "melody": "contemplative",
            "rhythm": "none",
            "texture": "shimmer_slow",
            "accent": "bells",
            "motion": 0.25,
            "attack": "soft",
            "stereo": 0.25,
            "depth": False,
            "echo": 0.25,
            "human": 0.25,
            "grain": "clean",
        },
    ]
    demo_names = ["bubblegum.wav", "sad.wav", "dead.wav"]
    demo_vibes = ["Bubblegum", "sad", "dead"]

    for vibe, fname, config_dict in zip(demo_vibes, demo_names, demo_configs):
        print(f"\n{'=' * 60}")
        print(f"Vibe: {vibe}")
        print("=" * 60)
        config = MusicConfig(**config_dict)
        print(json.dumps(config_dict, indent=2))
        config_to_audio(config, fname, duration=20.0)
        print(f"   Saved: {fname}")

    print("\nAll demo audio exported.")


# if __name__ == "__main__":
#     print("VibeSynth V1/V2 - Pure Python Synthesis Engine")
#     print("=" * 50)

#     # Example 1: Direct config
#     print("\n1. Generating from direct config (Indian Wedding)...")
#     config = MusicConfig(
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

#     morning = MusicConfig(
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

#     evening = MusicConfig(
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
