# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

"""
VibeSynth - Pure Python Synthesis Engine with Pedalboard DSP

Architecture:
1. Primitives: oscillators, envelopes (numpy) + filters/effects (pedalboard)
2. Patterns: pre-baked layer templates (bass, pad, melody, rhythm, texture, accent)
3. Assembler: config → audio conversion with V2 parameter transforms

Dependencies:
    pip install pedalboard numpy soundfile
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, TypeAlias, cast

import numpy as np
import soundfile as sf  # type: ignore[import]
from numpy.typing import NDArray

# Pedalboard for high-quality DSP
from pedalboard import (  # type: ignore[import]
    Chorus,
    Delay,
    HighpassFilter,
    Limiter,
    LowpassFilter,
    Reverb,
)

# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 44100

# Root note frequencies (octave 4)
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

# Mode intervals (semitones from root)
MODE_INTERVALS: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
}

# V2 parameter mappings
ATTACK_MULT: dict[str, float] = {"soft": 2.5, "medium": 1.0, "sharp": 0.3}
GRAIN_OSC: dict[str, str] = {"clean": "sine", "warm": "triangle", "gritty": "sawtooth"}

# Density → active layers
DENSITY_LAYERS: dict[int, tuple[str, ...]] = {
    2: ("bass", "pad"),
    3: ("bass", "pad", "melody"),
    4: ("bass", "pad", "melody", "rhythm"),
    5: ("bass", "pad", "melody", "rhythm", "texture"),
    6: ("bass", "pad", "melody", "rhythm", "texture", "accent"),
}

FloatArray: TypeAlias = NDArray[np.float32]
OscFn: TypeAlias = Callable[[float, float, int, float], FloatArray]


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


def _ensure_float32(signal: NDArray[Any]) -> FloatArray:
    """Ensure signal is float32 for Pedalboard compatibility."""
    return signal.astype(np.float32, copy=False)


def generate_sine(
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3
) -> FloatArray:
    """Generate sine wave."""
    t = np.linspace(0, duration, int(sr * duration), False, dtype=np.float32)
    return _ensure_float32(
        amp * np.sin(2 * np.pi * freq * t, dtype=np.float32)
    )  # numpy may upcast to float64.


def generate_triangle(
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3
) -> FloatArray:
    """Generate triangle wave."""
    t = np.linspace(0, duration, int(sr * duration), False, dtype=np.float32)
    phase = t * freq - np.floor(t * freq + 0.5)
    return _ensure_float32(amp * 2 * np.abs(2 * phase) - amp)


def generate_sawtooth(
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3, oversample: int = 2
) -> FloatArray:
    """Generate anti-aliased sawtooth using 4-point PolyBLEP + oversampling."""
    num_samples = int(sr * duration)
    sr_high = sr * oversample
    num_samples_high = num_samples * oversample
    dt = freq / sr_high

    t = np.linspace(0, duration * freq, num_samples_high, endpoint=False, dtype=np.float64)
    t = t % 1.0
    naive = 2.0 * t - 1.0
    correction = np.zeros(num_samples_high, dtype=np.float64)

    # 4-point PolyBLEP correction
    m1 = t < dt
    t1 = t[m1] / dt
    correction[m1] = t1 * t1 * (2 * t1 - 3) + 1

    m2 = (t >= dt) & (t < 2 * dt)
    t2 = t[m2] / dt - 1
    correction[m2] = t2 * t2 * (2 * t2 - 3)

    m3 = (t > 1 - 2 * dt) & (t <= 1 - dt)
    t3 = (t[m3] - 1) / dt + 1
    correction[m3] = t3 * t3 * (2 * t3 + 3)

    m4 = t > 1 - dt
    t4 = (t[m4] - 1) / dt
    correction[m4] = t4 * t4 * (2 * t4 + 3) + 1

    signal_high = naive - correction

    # Simple decimation (averaging)
    signal = signal_high.reshape(-1, oversample).mean(axis=1)

    if len(signal) > num_samples:
        signal = signal[:num_samples]
    elif len(signal) < num_samples:
        signal = np.pad(signal, (0, num_samples - len(signal)))

    return _ensure_float32(amp * signal)


def generate_square(
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3, oversample: int = 2
) -> FloatArray:
    """Generate anti-aliased square wave using 4-point PolyBLEP + oversampling."""
    num_samples = int(sr * duration)
    sr_high = sr * oversample
    num_samples_high = num_samples * oversample
    dt = freq / sr_high

    t = np.linspace(0, duration * freq, num_samples_high, endpoint=False, dtype=np.float64)
    t = t % 1.0
    naive = np.where(t < 0.5, 1.0, -1.0)
    correction = np.zeros(num_samples_high, dtype=np.float64)

    def apply_4pt_blep(phase: NDArray[Any], corr: NDArray[Any], sign: float) -> None:
        m1 = phase < dt
        t1 = phase[m1] / dt
        corr[m1] += sign * (t1 * t1 * (2 * t1 - 3) + 1)

        m2 = (phase >= dt) & (phase < 2 * dt)
        t2 = phase[m2] / dt - 1
        corr[m2] += sign * (t2 * t2 * (2 * t2 - 3))

        m3 = (phase > 1 - 2 * dt) & (phase <= 1 - dt)
        t3 = (phase[m3] - 1) / dt + 1
        corr[m3] += sign * (t3 * t3 * (2 * t3 + 3))

        m4 = phase > 1 - dt
        t4 = (phase[m4] - 1) / dt
        corr[m4] += sign * (t4 * t4 * (2 * t4 + 3) + 1)

    apply_4pt_blep(t, correction, 1.0)
    t_shifted = (t + 0.5) % 1.0
    apply_4pt_blep(t_shifted, correction, -1.0)

    signal_high = naive + correction
    signal = signal_high.reshape(-1, oversample).mean(axis=1)

    if len(signal) > num_samples:
        signal = signal[:num_samples]
    elif len(signal) < num_samples:
        signal = np.pad(signal, (0, num_samples - len(signal)))

    return _ensure_float32(amp * signal)


def generate_noise(duration: float, sr: int = SAMPLE_RATE, amp: float = 0.1) -> FloatArray:
    """Generate white noise."""
    return _ensure_float32(amp * np.random.randn(int(sr * duration)))


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
    attack = max(attack, 0.005)
    release = max(release, 0.01)

    total = len(signal)
    a_samples = int(attack * sr)
    d_samples = int(decay * sr)
    r_samples = int(release * sr)
    s_samples = max(0, total - a_samples - d_samples - r_samples)

    envelope = np.concatenate(
        (
            np.linspace(0, 1, max(1, a_samples), dtype=np.float32),
            np.linspace(1, sustain, max(1, d_samples), dtype=np.float32),
            np.full(max(1, s_samples), sustain, dtype=np.float32),
            np.linspace(sustain, 0, max(1, r_samples), dtype=np.float32),
        )
    )

    if len(envelope) < total:
        envelope = np.pad(envelope, (0, total - len(envelope)))
    else:
        envelope = envelope[:total]

    return _ensure_float32(signal * envelope)


# =============================================================================
# PEDALBOARD-BASED EFFECTS
# =============================================================================


def apply_lowpass(signal: FloatArray, cutoff: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply lowpass filter using Pedalboard."""
    cutoff = float(np.clip(cutoff, 20.0, sr / 2 - 100))
    lp = LowpassFilter(cutoff_frequency_hz=cutoff)
    return _ensure_float32(lp(signal, sr))


def apply_highpass(signal: FloatArray, cutoff: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply highpass filter using Pedalboard."""
    cutoff = float(np.clip(cutoff, 20.0, sr / 2 - 100))
    hp = HighpassFilter(cutoff_frequency_hz=cutoff)
    return _ensure_float32(hp(signal, sr))


def apply_delay(
    signal: FloatArray, delay_time: float, feedback: float, wet: float, sr: int = SAMPLE_RATE
) -> FloatArray:
    """Apply delay effect using Pedalboard."""
    delay_time = float(np.clip(delay_time, 0.01, 2.0))
    feedback = float(np.clip(feedback, 0.0, 0.95))
    wet = float(np.clip(wet, 0.0, 1.0))

    delay = Delay(
        delay_seconds=delay_time,
        feedback=feedback,
        mix=wet,
    )
    return _ensure_float32(delay(signal, sr))


def apply_reverb(signal: FloatArray, room: float, size: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply reverb using Pedalboard's high-quality reverb."""
    room = float(np.clip(room, 0.0, 1.0))
    size = float(np.clip(size, 0.0, 1.0))

    reverb = Reverb(
        room_size=size,
        damping=1.0 - room * 0.5,  # Higher room = less damping
        wet_level=room * 0.5,
        dry_level=1.0 - room * 0.3,
        width=0.8,
    )
    return _ensure_float32(reverb(signal, sr))


def apply_chorus(
    signal: FloatArray,
    rate: float = 1.0,
    depth: float = 0.25,
    mix: float = 0.5,
    sr: int = SAMPLE_RATE,
) -> FloatArray:
    """Apply chorus effect using Pedalboard."""
    chorus = Chorus(
        rate_hz=float(np.clip(rate, 0.1, 10.0)),
        depth=float(np.clip(depth, 0.0, 1.0)),
        mix=float(np.clip(mix, 0.0, 1.0)),
        centre_delay_ms=7.0,
        feedback=0.0,
    )
    return _ensure_float32(chorus(signal, sr))


def apply_final_limiter(signal: FloatArray, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply a limiter for clean final output."""
    limiter = Limiter(threshold_db=-1.0, release_ms=100.0)
    return _ensure_float32(limiter(signal, sr))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def generate_lfo(duration: float, rate: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Generate LFO signal (0 to 1 range)."""
    t = np.linspace(0, duration, int(sr * duration), False, dtype=np.float32)
    return _ensure_float32(0.5 + 0.5 * np.sin(2 * np.pi * rate * t))


def apply_humanize(signal: FloatArray, amount: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply subtle amplitude humanization."""
    if amount <= 0:
        return signal
    amp_lfo = 1.0 + (np.random.randn(len(signal)).astype(np.float32) * amount * 0.1)
    amp_lfo = np.clip(amp_lfo, 0.9, 1.1)
    return _ensure_float32(signal * amp_lfo)


def add_note(signal: FloatArray, note: FloatArray, start_index: int) -> None:
    """Safely adds a note to the signal buffer, clipping if necessary."""
    if start_index >= len(signal):
        return

    end_index = start_index + len(note)

    if end_index <= len(signal):
        signal[start_index:end_index] += note
    else:
        available = len(signal) - start_index
        clipped = note[:available].copy()

        fade_samples = min(int(SAMPLE_RATE * 0.01), available // 4)
        if fade_samples > 1:
            clipped[-fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)

        signal[start_index:] += clipped


# =============================================================================
# PART 2: PATTERN GENERATORS
# =============================================================================


@dataclass(frozen=True)
class ChordEvent:
    """A chord specified in scale degrees relative to the key, scheduled in beats."""

    start_beat: float
    duration_beats: float
    root_degree: int


@dataclass(frozen=True)
class HarmonyPlan:
    """A sequence of chord events spanning the render duration."""

    chords: Tuple[ChordEvent, ...] = ()
    beats_per_bar: int = 4

    def chord_at_beat(self, beat: float) -> ChordEvent:
        """Get the chord event active at the given beat."""
        if not self.chords:
            return ChordEvent(0.0, float(self.beats_per_bar), 0)

        for ch in self.chords:
            if ch.start_beat <= beat < (ch.start_beat + ch.duration_beats):
                return ch

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
    """A rendered melody note event."""

    start_sec: float
    dur_sec: float
    freq: float
    amp: float
    is_anchor: bool = False


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _tension_value(curve: str, x: float) -> float:
    x = _clamp01(x)
    if curve == "ramp":
        return x
    if curve == "waves":
        return 0.5 - 0.5 * float(np.cos(2.0 * np.pi * 2.0 * x))
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
    if not items:
        raise ValueError("No items to choose from")
    w = np.asarray(weights, dtype=np.float64)
    if np.all(w <= 0):
        w = np.ones_like(w)
    w = w / np.sum(w)
    idx = int(rng.choice(len(items), p=w))
    return int(items[idx])


def _chord_tones_ascending(params: "SynthParams", chord_root_degree: int) -> Tuple[int, ...]:
    d = int(chord_root_degree)
    tones: List[int] = [d, d + 2, d + 4]
    if params.chord_extensions in ("sevenths", "lush"):
        tones.append(d + 6)
    if params.chord_extensions == "lush":
        tones.append(d + 8)
    return tuple(tones)


@dataclass
class SynthParams:
    """Parameters passed to all synthesis functions."""

    root: str = "c"
    mode: str = "minor"
    brightness: float = 0.5
    space: float = 0.6
    duration: float = 16.0
    tempo: float = 0.35

    motion: float = 0.5
    attack: str = "medium"
    stereo: float = 0.5
    depth: bool = False
    echo: float = 0.5
    human: float = 0.0
    grain: str = "clean"

    harmony: HarmonyPlan | None = None
    chord_extensions: str = "triads"
    melody_policy: MelodyPolicy | None = None

    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)

    @property
    def attack_mult(self) -> float:
        return ATTACK_MULT.get(self.attack, 1.0)

    @property
    def osc_type(self) -> str:
        return GRAIN_OSC.get(self.grain, "sine")

    @property
    def echo_mult(self) -> float:
        return self.echo / 0.5

    @property
    def pan_width(self) -> float:
        return self.stereo * 0.5

    @property
    def bpm(self) -> float:
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

    def semitone_from_degree(self, degree: int) -> int:
        intervals = MODE_INTERVALS.get(self.mode, MODE_INTERVALS["minor"])
        return int(intervals[degree % len(intervals)] + (12 * (degree // len(intervals))))

    def get_scale_freq(self, degree: int, octave: int = 4) -> float:
        semitone = self.semitone_from_degree(degree)
        return freq_from_note(self.root, semitone, octave)

    def chord_tone_classes(self, chord_root_degree: int) -> Tuple[int, ...]:
        d = int(chord_root_degree) % 7
        tones: List[int] = [d, (d + 2) % 7, (d + 4) % 7]
        if self.chord_extensions in ("sevenths", "lush"):
            tones.append((d + 6) % 7)
        if self.chord_extensions == "lush":
            tones.append((d + 1) % 7)
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
    signal = np.zeros(int(sr * dur_total), dtype=np.float32)

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)
        freq = params.get_scale_freq(chord_root, 2)

        note = generate_sine(freq, note_dur, sr, 0.35)
        note = apply_lowpass(note, 80 * params.brightness + 20, sr)
        note = apply_adsr(note, 1.8 * params.attack_mult, 0.5, 0.95, 2.2 * params.attack_mult, sr)
        add_note(signal, note, int(start_sec * sr))

    signal = apply_reverb(signal, params.space * 0.5, params.space, sr)
    return signal


def bass_sustained(params: SynthParams) -> FloatArray:
    """Long sustained bass notes that follow chord roots."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    signal = np.zeros(int(sr * dur_total), dtype=np.float32)

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        pattern = (0, 0, 4, 0)
        note_dur = seg_dur / len(pattern)

        for i, rel_degree in enumerate(pattern):
            deg = chord_root + rel_degree
            freq = params.get_scale_freq(deg, 2)
            start = int((start_sec + i * note_dur) * sr)

            note = generate_sine(freq, note_dur * 0.95, sr, 0.32)
            note = apply_lowpass(note, 100 * params.brightness + 30, sr)
            note = apply_adsr(
                note, 0.7 * params.attack_mult, 0.3, 0.85, 1.2 * params.attack_mult, sr
            )
            add_note(signal, note, start)

    signal = apply_reverb(signal, params.space * 0.4, params.space, sr)
    return signal


def bass_pulsing(params: SynthParams) -> FloatArray:
    """Rhythmic pulsing bass (chord-aware)."""
    sr = SAMPLE_RATE
    dur = params.duration

    beats_total = max(1.0, params.beats_total)
    pulses_per_beat = 2.0
    num_pulses = int(np.ceil(beats_total * pulses_per_beat))
    pulse_beats = 1.0 / pulses_per_beat
    pulse_sec = pulse_beats * params.seconds_per_beat

    signal = np.zeros(int(sr * dur), dtype=np.float32)

    for i in range(num_pulses):
        start_sec = i * pulse_sec
        if start_sec >= dur:
            break

        beat = i * pulse_beats
        chord_root = params.chord_root_degree_at_beat(beat)
        freq = params.get_scale_freq(chord_root, 2)

        note = generate_sine(freq, pulse_sec * 0.82, sr, 0.35)
        note = apply_lowpass(note, 90 * params.brightness + 20, sr)
        note = apply_adsr(note, 0.02 * params.attack_mult, 0.08, 0.6, 0.25 * params.attack_mult, sr)
        add_note(signal, note, int(start_sec * sr))

    return signal


def bass_walking(params: SynthParams) -> FloatArray:
    """Walking bass line that follows the chord root."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    signal = np.zeros(int(sr * dur_total), dtype=np.float32)

    beats_total = int(np.ceil(params.beats_total))
    note_sec = params.seconds_per_beat

    for i in range(beats_total):
        start_sec = i * note_sec
        if start_sec >= dur_total:
            break

        chord_root = params.chord_root_degree_at_beat(float(i))
        rel = (0, 2, 4, 2)[i % 4]
        degree = chord_root + rel

        freq = params.get_scale_freq(degree, 2)
        note = generate_triangle(freq, note_sec * 0.92, sr, 0.30)
        note = apply_lowpass(note, 120 * params.brightness + 40, sr)
        note = apply_adsr(note, 0.04 * params.attack_mult, 0.12, 0.7, 0.22 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr)
        add_note(signal, note, int(start_sec * sr))

    signal = apply_reverb(signal, params.space * 0.3, params.space * 0.8, sr)
    return signal


def bass_fifth_drone(params: SynthParams) -> FloatArray:
    """Root + fifth drone that follows chord changes."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    signal = np.zeros(int(sr * dur_total), dtype=np.float32)

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
    signal = np.zeros(int(sr * dur), dtype=np.float32)

    beats_total = max(1.0, params.beats_total)
    pulses_per_bar = 2
    beats_per_pulse = 4.0 / pulses_per_bar
    pulse_sec = beats_per_pulse * params.seconds_per_beat

    num_pulses = int(np.ceil(beats_total / beats_per_pulse))
    for i in range(num_pulses):
        start_sec = i * pulse_sec
        if start_sec >= dur:
            break

        beat = i * beats_per_pulse
        chord_root = params.chord_root_degree_at_beat(beat)

        freq = params.get_scale_freq(chord_root, 1)
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

    signal = np.zeros(int(sr * dur_total), dtype=np.float32)

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        for degree in _chord_tones_ascending(params, chord_root)[:3]:
            freq = params.get_scale_freq(degree, 3)
            tone = osc(freq, note_dur, sr, 0.15)

            lfo_rate = 0.1 / (params.motion + 0.1)
            lfo = generate_lfo(note_dur, lfo_rate, sr)

            base_cutoff = 300 * params.brightness + 100
            tone_low = apply_lowpass(tone, base_cutoff * 0.5, sr)
            tone_high = apply_lowpass(tone, base_cutoff * 1.5, sr)
            tone = _ensure_float32(
                tone_low * (1 - lfo) + tone_high * lfo
            )  # normalize dtype for ADSR.

            tone = apply_adsr(
                tone, 1.5 * params.attack_mult, 0.8, 0.85, 2.5 * params.attack_mult, sr
            )
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, params.space * 0.7, params.space * 0.9, sr)
    signal = apply_delay(signal, 0.35, 0.3 * params.echo_mult, 0.25 * params.echo_mult, sr)
    signal = apply_humanize(signal, params.human, sr)

    return signal


def pad_dark_sustained(params: SynthParams) -> FloatArray:
    """Dark, heavy sustained pad (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    signal = np.zeros(int(sr * dur_total), dtype=np.float32)

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
    signal = np.zeros(int(sr * dur_total), dtype=np.float32)

    voicings = ((0, 3), (2, 3), (4, 3), (0, 4), (4, 4))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        note_dur = min(seg_dur * 1.10, dur_total - start_sec)

        for rel_degree, octave in voicings:
            degree = chord_root + rel_degree
            freq = params.get_scale_freq(degree, octave)

            tone = generate_sawtooth(freq, note_dur, sr, 0.08)
            tone += generate_triangle(freq * 1.002, note_dur, sr, 0.06)

            tone = apply_lowpass(tone, 400 * params.brightness + 150, sr)
            tone = apply_adsr(
                tone, 1.8 * params.attack_mult, 0.8, 0.88, 2.8 * params.attack_mult, sr
            )
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, params.space * 0.85, params.space, sr)
    signal = apply_delay(signal, 0.4, 0.35 * params.echo_mult, 0.3 * params.echo_mult, sr)

    # Add subtle chorus for width
    if params.stereo > 0.4:
        signal = apply_chorus(signal, rate=0.5, depth=0.15, mix=params.stereo * 0.3, sr=sr)

    return signal


def pad_thin_high(params: SynthParams) -> FloatArray:
    """Thin, high pad (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    signal = np.zeros(int(sr * dur_total), dtype=np.float32)

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        for rel_degree in (0, 4):
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
    """Slowly drifting ambient pad (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = params.duration
    signal = np.zeros(int(sr * dur_total), dtype=np.float32)

    for start_sec, seg_dur, chord_root in _iter_chord_segments(params):
        note_dur = min(seg_dur * 1.35, dur_total - start_sec)

        tones = list(_chord_tones_ascending(params, chord_root)[:3])

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

    signal = np.zeros(int(sr * dur_total), dtype=np.float32)

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
    p = params.melody_policy or MelodyPolicy()
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
    params: SynthParams,
) -> int:
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
    chord_root = chord_root_degree % 7
    pcs = params.chord_tone_classes(chord_root_degree)

    midpoint = 0.5 * (min_pitch + max_pitch)
    desired = midpoint + (tension - 0.5) * 0.35 * (max_pitch - min_pitch)

    items: List[int] = []
    weights: List[float] = []

    for pc in pcs:
        w_pc = 1.0
        if pc == chord_root:
            w_pc *= 1.0 + 1.8 * cadence_strength
        elif pc == (chord_root + 2) % 7:
            w_pc *= 1.0
        elif pc == (chord_root + 4) % 7:
            w_pc *= 0.95
        elif pc == (chord_root + 6) % 7:
            w_pc *= 0.85
        else:
            w_pc *= 0.75

        for oct_i in range((max_pitch // 7) + 1):
            cand = pc + 7 * oct_i
            if cand < min_pitch or cand > max_pitch:
                continue

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
    policy = _melody_policy_from_params(params)
    rng = params.rng

    beats_per_bar = 4
    spb = params.seconds_per_beat
    total_beats = max(1.0, params.beats_total)
    total_bars = int(np.ceil(total_beats / beats_per_bar))

    base_oct = policy.register_min_oct
    oct_span = max(0, policy.register_max_oct - policy.register_min_oct)
    min_pitch = 0
    max_pitch = oct_span * 7 + 6

    # Build anchor pitches (1 per bar)
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

    # Generate note events
    events: List[NoteEvent] = []

    step_beats = _grid_step_beats(policy.density)
    steps_per_bar = int(round(beats_per_bar / step_beats))
    steps_per_bar = max(1, min(64, steps_per_bar))

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

        use_motif = bool(motif) and is_phrase_start and (rng.random() < policy.motif_repeat_prob)

        need_resolve = False
        current_pitch = anchor_pitch
        bar_notes_for_motif: List[Tuple[int, int]] = []

        for step_idx in range(steps_per_bar):
            pos_beats = step_idx * step_beats
            beat = beat_bar + pos_beats
            start_sec = beat * spb

            if start_sec >= params.duration:
                break

            if step_beats == 0.5 and (step_idx % 2 == 1):
                start_sec += policy.swing * 0.12 * spb

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

                if is_cadence_bar and pos_beats >= 2.0:
                    p *= 0.35 + 0.40 * (1.0 - policy.cadence_strength)

                if events and (events[-1].start_sec > (start_sec - 0.001)):
                    p *= 0.6

                play = rng.random() < _clamp01(p)
                is_anchor = False

            if not play:
                continue

            # Pitch selection
            if step_idx == 0:
                pitch = anchor_pitch
            elif use_motif:
                offset = next((off for s, off in motif if s == step_idx), 0)
                pitch = int(np.clip(anchor_pitch + offset, min_pitch, max_pitch))
            else:
                delta = next_anchor - current_pitch
                if delta == 0:
                    direction = int(rng.choice([-1, 1]))
                else:
                    direction = 1 if delta > 0 else -1

                if need_resolve:
                    pitch = _nearest_chord_tone_pitch(
                        current_pitch, chord_root, min_pitch, max_pitch, params
                    )
                    need_resolve = False
                else:
                    leap_prob = (1.0 - policy.step_bias) * (0.25 + 0.35 * tension)
                    step_size = 1

                    if rng.random() < (0.06 + 0.10 * tension):
                        step_size = 0
                    elif abs(delta) >= 4 and (rng.random() < leap_prob):
                        step_size = int(rng.choice([2, 3, 4]))

                    pitch = int(
                        np.clip(current_pitch + direction * step_size, min_pitch, max_pitch)
                    )

                chord_pcs = params.chord_tone_classes(chord_root)
                if (pitch % 7) not in chord_pcs:
                    if rng.random() < (0.55 - 0.45 * tension):
                        pitch = _nearest_chord_tone_pitch(
                            pitch, chord_root, min_pitch, max_pitch, params
                        )
                    else:
                        need_resolve = True

            dur_beats = step_beats * (0.70 + 0.25 * float(rng.random()))
            if is_anchor:
                dur_beats = max(dur_beats, 1.0)
                if is_cadence_bar:
                    dur_beats = max(dur_beats, 1.6)

            dur_sec = dur_beats * spb
            dur_sec = min(dur_sec, max(0.02, params.duration - start_sec))

            jitter = 0.0
            if params.human > 0 and not is_anchor:
                jitter = float(rng.normal(0.0, params.human * 0.006 * spb))
                jitter = float(np.clip(jitter, -0.02 * spb, 0.02 * spb))
            start_sec_j = float(np.clip(start_sec + jitter, 0.0, max(0.0, params.duration - 0.01)))

            amp = 0.12 + 0.05 * tension
            if is_cadence_bar and pos_beats >= 2.0:
                amp *= 0.85
            if is_anchor:
                amp *= 1.10

            use_chromatic = (
                (policy.chromatic_prob > 0)
                and (not is_anchor)
                and (abs((next_anchor - pitch)) <= 2)
                and (rng.random() < (policy.chromatic_prob * (0.35 + 0.65 * tension)))
            )

            if use_chromatic:
                target = next_anchor
                sign = 1 if (target - pitch) >= 0 else -1
                sem = params.semitone_from_degree(target) - sign
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
                bar_notes_for_motif.append((step_idx, pitch - anchor_pitch))

            current_pitch = pitch

        if not motif and is_phrase_start and bar_notes_for_motif:
            motif = bar_notes_for_motif[: min(5, len(bar_notes_for_motif))]

    return events


def melody_procedural(params: SynthParams) -> FloatArray:
    """Procedural, chord-aware melody with phrases + tension/resolution."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    events = _generate_procedural_melody_events(params)

    signal = np.zeros(int(sr * dur), dtype=np.float32)

    cutoff = 700 * params.brightness + 180
    for ev in events:
        start_idx = int(ev.start_sec * sr)
        if start_idx >= len(signal):
            continue

        note = osc(ev.freq, ev.dur_sec, sr, ev.amp)
        note = apply_lowpass(note, cutoff, sr)

        atk = (0.05 if ev.is_anchor else 0.03) * params.attack_mult
        rel = (0.22 if ev.is_anchor else 0.16) * params.attack_mult
        note = apply_adsr(note, atk, 0.12, 0.55, rel, sr)
        note = apply_humanize(note, params.human * 0.6, sr)

        add_note(signal, note, start_idx)

    signal = apply_delay(signal, 0.33, 0.35 * params.echo_mult, 0.28 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.55, params.space * 0.75, sr)

    return signal


def melody_contemplative(params: SynthParams) -> FloatArray:
    """Slow, contemplative melody."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    pattern = (0, -1, 2, -1, 4, -1, 2, -1, 0, -1, -1, -1, 2, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur), dtype=np.float32)

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = params.get_scale_freq(degree, 5)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 1.5, sr, 0.18)
        note = apply_lowpass(note, 800 * params.brightness + 200, sr)
        note = apply_adsr(note, 0.1 * params.attack_mult, 0.3, 0.6, 0.8 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.35, 0.4 * params.echo_mult, 0.3 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.6, params.space * 0.8, sr)

    return signal


def melody_rising(params: SynthParams) -> FloatArray:
    """Ascending melodic line."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    pattern = (0, -1, 2, -1, 4, -1, -1, 5, -1, -1, 6, -1, -1, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur), dtype=np.float32)

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = params.get_scale_freq(degree, 5)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 1.8, sr, 0.16)
        note = apply_lowpass(note, 900 * params.brightness + 250, sr)
        note = apply_adsr(note, 0.08 * params.attack_mult, 0.25, 0.55, 0.9 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.33, 0.35 * params.echo_mult, 0.28 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.55, params.space * 0.75, sr)

    return signal


def melody_falling(params: SynthParams) -> FloatArray:
    """Descending melodic line."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    pattern = (6, -1, -1, 5, -1, -1, 4, -1, 2, -1, -1, 0, -1, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur), dtype=np.float32)

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = params.get_scale_freq(degree, 5)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 1.6, sr, 0.17)
        note = apply_lowpass(note, 850 * params.brightness + 220, sr)
        note = apply_adsr(note, 0.1 * params.attack_mult, 0.28, 0.52, 0.85 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.38, 0.38 * params.echo_mult, 0.3 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.58, params.space * 0.78, sr)

    return signal


def melody_minimal(params: SynthParams) -> FloatArray:
    """Very sparse, minimal melody."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_sine)

    pattern = (4, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur), dtype=np.float32)

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = params.get_scale_freq(degree, 5)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 2.5, sr, 0.20)
        note = apply_lowpass(note, 600 * params.brightness + 150, sr)
        note = apply_adsr(note, 0.15 * params.attack_mult, 0.4, 0.5, 1.2 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.5, 0.45 * params.echo_mult, 0.4 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.7, params.space * 0.9, sr)

    return signal


def melody_ornamental(params: SynthParams) -> FloatArray:
    """Ornamental melody with grace notes."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    main_notes = (0, 4, 2, 0)
    grace_offsets = (2, 5, 3, 2)
    note_dur = dur / (len(main_notes) * 2)
    signal = np.zeros(int(sr * dur), dtype=np.float32)

    for i, (main, grace) in enumerate(zip(main_notes, grace_offsets)):
        start = int(i * 2 * note_dur * sr)

        grace_freq = params.get_scale_freq(grace, 5)
        grace_note = osc(grace_freq, note_dur * 0.15, sr, 0.10)
        grace_note = apply_lowpass(grace_note, 1000 * params.brightness, sr)
        grace_note = apply_adsr(grace_note, 0.01, 0.05, 0.3, 0.1, sr)

        main_freq = params.get_scale_freq(main, 5)
        main_note = osc(main_freq, note_dur * 1.5, sr, 0.18)
        main_note = apply_lowpass(main_note, 900 * params.brightness + 200, sr)
        main_note = apply_adsr(
            main_note, 0.08 * params.attack_mult, 0.3, 0.55, 0.8 * params.attack_mult, sr
        )
        main_note = apply_humanize(main_note, params.human, sr)

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

    base_pattern = (0, 2, 4, 2)
    pattern = base_pattern * 8
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur), dtype=np.float32)

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
    return np.zeros(int(SAMPLE_RATE * params.duration), dtype=np.float32)


def rhythm_minimal(params: SynthParams) -> FloatArray:
    """Minimal rhythm - just occasional hits."""
    sr = SAMPLE_RATE
    dur = params.duration

    pattern = (1, 0, 0, 0, 0, 0, 0, 0) * 2
    hit_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur), dtype=np.float32)

    for i, hit in enumerate(pattern):
        if not hit:
            continue

        start = int(i * hit_dur * sr)

        kick_dur = 0.15
        kick = generate_sine(60, kick_dur, sr, 0.35)
        pitch_env = np.exp(-np.linspace(0, 8, len(kick))).astype(np.float32)
        kick = kick * pitch_env
        kick = apply_lowpass(kick, 150, sr)
        kick = apply_humanize(kick, params.human, sr)

        add_note(signal, kick, start)

    signal = apply_reverb(signal, params.space * 0.3, params.space * 0.5, sr)
    return signal


def rhythm_heartbeat(params: SynthParams) -> FloatArray:
    """Heartbeat-like rhythm."""
    sr = SAMPLE_RATE
    dur = params.duration

    pattern = (1, 1, 0, 0, 0, 0, 0, 0) * 2
    hit_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur), dtype=np.float32)

    for i, hit in enumerate(pattern):
        if not hit:
            continue

        start = int(i * hit_dur * sr)

        kick_dur = 0.12
        kick = generate_sine(55, kick_dur, sr, 0.32)
        pitch_env = np.exp(-np.linspace(0, 10, len(kick))).astype(np.float32)
        kick = kick * pitch_env
        kick = apply_lowpass(kick, 120, sr)

        add_note(signal, kick, start)

    return signal


def rhythm_soft_four(params: SynthParams) -> FloatArray:
    """Soft four-on-the-floor."""
    sr = SAMPLE_RATE
    dur = params.duration

    num_beats = 8
    beat_dur = dur / num_beats
    signal = np.zeros(int(sr * dur), dtype=np.float32)

    for i in range(num_beats):
        start = int(i * beat_dur * sr)

        kick_dur = 0.18
        kick = generate_sine(50, kick_dur, sr, 0.28)
        pitch_env = np.exp(-np.linspace(0, 6, len(kick))).astype(np.float32)
        kick = kick * pitch_env
        kick = apply_lowpass(kick, 100, sr)
        kick = apply_humanize(kick, params.human, sr)

        add_note(signal, kick, start)

    signal = apply_reverb(signal, params.space * 0.25, params.space * 0.4, sr)
    return signal


def rhythm_hats_only(params: SynthParams) -> FloatArray:
    """Just hi-hats."""
    sr = SAMPLE_RATE
    dur = params.duration

    num_hits = 32
    hit_dur = dur / num_hits
    signal = np.zeros(int(sr * dur), dtype=np.float32)

    for i in range(num_hits):
        start = int(i * hit_dur * sr)

        hat_dur = 0.05
        hat = generate_noise(hat_dur, sr, 0.08)
        hat = apply_highpass(hat, 6000, sr)
        hat = apply_adsr(hat, 0.001, 0.02, 0.1, 0.03, sr)
        hat = apply_humanize(hat, params.human, sr)

        add_note(signal, hat, start)

    return signal


def rhythm_electronic(params: SynthParams) -> FloatArray:
    """Electronic beat."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur), dtype=np.float32)
    beat_dur = dur / 16

    kick_pattern = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
    hat_pattern = (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)

    for i in range(16):
        start = int(i * beat_dur * sr)

        if kick_pattern[i]:
            kick = generate_sine(45, 0.2, sr, 0.35)
            pitch_env = np.exp(-np.linspace(0, 8, len(kick))).astype(np.float32)
            kick = kick * pitch_env
            kick = apply_lowpass(kick, 100, sr)
            add_note(signal, kick, start)

        if hat_pattern[i]:
            hat = generate_noise(0.06, sr, 0.06)
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
    return np.zeros(int(SAMPLE_RATE * params.duration), dtype=np.float32)


def texture_shimmer(params: SynthParams) -> FloatArray:
    """High, shimmering texture."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur), dtype=np.float32)

    for degree in (0, 2, 4):
        freq = params.get_scale_freq(degree, 6)
        tone = generate_sine(freq, dur, sr, 0.04)

        lfo_rate = 2.0 / (params.motion + 0.2)
        lfo = generate_lfo(dur, lfo_rate, sr)
        tone = _ensure_float32(tone * (0.5 + 0.5 * lfo))  # normalize dtype for ADSR.

        tone = apply_adsr(tone, 0.5 * params.attack_mult, 0.3, 0.8, 1.5 * params.attack_mult, sr)
        signal += tone

    signal = apply_delay(signal, 0.4, 0.5 * params.echo_mult, 0.4 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.8, params.space, sr)

    return signal


def texture_shimmer_slow(params: SynthParams) -> FloatArray:
    """Slow, gentle shimmer."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur), dtype=np.float32)

    for degree in (0, 4):
        freq = params.get_scale_freq(degree, 6)
        tone = generate_sine(freq, dur, sr, 0.035)

        lfo_rate = 0.5 / (params.motion + 0.2)
        lfo = generate_lfo(dur, lfo_rate, sr)
        tone = _ensure_float32(tone * (0.4 + 0.6 * lfo))  # normalize dtype for ADSR.

        tone = apply_adsr(tone, 1.0 * params.attack_mult, 0.5, 0.85, 2.0 * params.attack_mult, sr)
        signal += tone

    signal = apply_delay(signal, 0.5, 0.55 * params.echo_mult, 0.45 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.85, params.space, sr)

    return signal


def texture_vinyl_crackle(params: SynthParams) -> FloatArray:
    """Vinyl crackle texture."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur), dtype=np.float32)

    num_crackles = int(dur * 20)

    for _ in range(num_crackles):
        pos = np.random.randint(0, len(signal) - 100)
        crackle = generate_noise(0.002, sr, np.random.uniform(0.01, 0.04))
        crackle = apply_highpass(crackle, 2000, sr)

        add_note(signal, crackle, pos)

    hiss = generate_noise(dur, sr, 0.008)
    hiss = apply_lowpass(hiss, 8000, sr)
    hiss = apply_highpass(hiss, 1000, sr)
    signal += hiss

    return signal


def texture_breath(params: SynthParams) -> FloatArray:
    """Breathing texture."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = generate_noise(dur, sr, 0.06)

    freq = params.get_scale_freq(0, 3)
    signal = apply_lowpass(signal, freq * 2, sr)
    signal = apply_highpass(signal, freq * 0.5, sr)

    breath_rate = 0.2 / (params.motion + 0.1)
    lfo = generate_lfo(dur, breath_rate, sr)
    signal = signal * lfo

    signal = apply_reverb(signal, params.space * 0.6, params.space * 0.8, sr)

    return signal


def texture_stars(params: SynthParams) -> FloatArray:
    """Twinkling stars texture."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur), dtype=np.float32)

    num_stars = int(dur * 3)
    star_degrees = (0, 2, 4, 5)

    for _ in range(num_stars):
        pos = np.random.randint(0, len(signal) - sr)

        degree = np.random.choice(star_degrees)
        freq = params.get_scale_freq(degree, 6)

        star = generate_sine(freq, 0.3, sr, np.random.uniform(0.02, 0.05))
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
    return np.zeros(int(SAMPLE_RATE * params.duration), dtype=np.float32)


def accent_bells(params: SynthParams) -> FloatArray:
    """Bell-like accents."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur), dtype=np.float32)

    pattern = (0, -1, -1, -1, 4, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1)
    hit_dur = dur / len(pattern)

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        start = int(i * hit_dur * sr)
        freq = params.get_scale_freq(degree, 5)

        bell_dur = 0.8
        bell = generate_sine(freq, bell_dur, sr, 0.12)
        bell += generate_sine(freq * 2.0, bell_dur, sr, 0.06)
        bell += generate_sine(freq * 3.0, bell_dur, sr, 0.03)
        bell = apply_adsr(bell, 0.005 * params.attack_mult, 0.2, 0.1, 0.6 * params.attack_mult, sr)
        bell = apply_humanize(bell, params.human, sr)

        add_note(signal, bell, start)

    signal = apply_reverb(signal, params.space * 0.6, params.space * 0.8, sr)
    return signal


def accent_pluck(params: SynthParams) -> FloatArray:
    """Plucked string accents."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur), dtype=np.float32)

    pattern = (0, -1, -1, 4, -1, -1, 2, -1, -1, -1, 0, -1, -1, -1, -1, -1)
    hit_dur = dur / len(pattern)

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        start = int(i * hit_dur * sr)
        freq = params.get_scale_freq(degree, 4)

        pluck = generate_triangle(freq, 0.5, sr, 0.15)
        pluck = apply_lowpass(pluck, 1500 * params.brightness + 400, sr)
        pluck = apply_adsr(
            pluck, 0.003 * params.attack_mult, 0.15, 0.05, 0.4 * params.attack_mult, sr
        )
        pluck = apply_humanize(pluck, params.human, sr)

        add_note(signal, pluck, start)

    signal = apply_reverb(signal, params.space * 0.5, params.space * 0.7, sr)
    return signal


def accent_chime(params: SynthParams) -> FloatArray:
    """Wind chime accents."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur), dtype=np.float32)

    num_chimes = int(dur * 1.5)
    chime_degrees = (0, 2, 4, 5, 6)

    for _ in range(num_chimes):
        pos = np.random.randint(0, len(signal) - sr)

        degree = np.random.choice(chime_degrees)
        freq = params.get_scale_freq(degree, 5)

        chime_dur = 1.2
        chime = generate_sine(freq, chime_dur, sr, np.random.uniform(0.06, 0.12))
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

    bass: str = "drone"
    pad: str = "warm_slow"
    melody: str = "contemplative"
    rhythm: str = "minimal"
    texture: str = "shimmer"
    accent: str = "bells"

    motion: float = 0.5
    attack: str = "medium"
    stereo: float = 0.5
    depth: bool = False
    echo: float = 0.5
    human: float = 0.0
    grain: str = "clean"

    melody_engine: str = "pattern"

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
    tension_curve: str = "arc"

    harmony_style: str = "auto"
    chord_change_bars: int = 1
    chord_extensions: str = "triads"

    seed: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MusicConfig":
        """Create config from dict."""
        config = cls()

        for key, value in d.items():
            if key == "layers" and isinstance(value, dict):
                layers = cast(dict[str, Any], value)
                for layer_key, layer_value in layers.items():
                    if hasattr(config, layer_key):
                        setattr(config, layer_key, layer_value)

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
    """Generate a simple diatonic chord timeline in scale-degrees."""
    beats_per_bar = 4
    bars_total = int(np.ceil(params.beats_total / beats_per_bar))
    bars_total = max(1, bars_total)

    chord_change_bars = int(max(1, config.chord_change_bars))
    n_chords = int(np.ceil(bars_total / chord_change_bars))
    n_chords = max(1, n_chords)

    style = (config.harmony_style or "auto").lower()
    if style == "auto":
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

    pop_major = (
        (0, 5, 3, 4),
        (0, 3, 4, 0),
        (0, 4, 5, 3),
    )
    pop_minor = (
        (0, 5, 3, 6),
        (0, 3, 6, 4),
        (0, 6, 3, 5),
        (0, 4, 5, 3),
    )
    jazz_major = (
        (1, 4, 0, 0),
        (1, 4, 0, 5),
        (5, 1, 4, 0),
    )
    jazz_minor = (
        (1, 4, 0, 0),
        (5, 1, 4, 0),
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
    else:
        template = rng.choice(pop_major if majorish else pop_minor)

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


def assemble(config: MusicConfig, duration: float = 16.0, normalize: bool = True) -> FloatArray:
    """
    Assemble all layers into final audio.

    Args:
        config: MusicConfig object
        duration: Duration in seconds
        normalize: If True, applies limiter for clean output
    """
    sr = SAMPLE_RATE

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
        chord_extensions=config.chord_extensions,
        rng=np.random.default_rng(config.seed if config.seed else None),
    )

    params.harmony = build_harmony_plan(config, params)
    params.melody_policy = build_melody_policy(config)

    active_layers = DENSITY_LAYERS.get(config.density, DENSITY_LAYERS[5])

    output = np.zeros(int(sr * duration), dtype=np.float32)

    if "bass" in active_layers:
        bass_fn = BASS_PATTERNS.get(config.bass, bass_drone)
        output += bass_fn(params)

    if config.depth:
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

    # Use Pedalboard limiter for clean final output
    if normalize:
        # Pre-gain to bring levels up before limiting
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.9
        output = apply_final_limiter(output, sr)

    return output


def config_to_audio(config: MusicConfig, output_path: str, duration: float = 16.0) -> str:
    """Convert a MusicConfig to an audio file."""
    audio = assemble(config, duration)
    sf.write(output_path, audio, SAMPLE_RATE)
    return output_path


def dict_to_audio(config_dict: Dict[str, Any], output_path: str, duration: float = 16.0) -> str:
    """Convert a config dict directly to audio."""
    config = MusicConfig.from_dict(config_dict)
    return config_to_audio(config, output_path, duration)


# =============================================================================
# PART 4: CONVENIENCE FUNCTIONS
# =============================================================================


def generate_from_vibe(vibe: str, output_path: str = "output.wav", duration: float = 16.0) -> str:
    """Generate audio from a vibe description."""
    vibe_lower = vibe.lower()

    config = MusicConfig()

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

    if any(w in vibe_lower for w in ("slow", "calm", "meditation")):
        config.tempo = 0.30
    elif any(w in vibe_lower for w in ("fast", "energy", "drive")):
        config.tempo = 0.45
    else:
        config.tempo = 0.36

    if any(w in vibe_lower for w in ("vast", "space", "underwater", "cave")):
        config.space = 0.85
        config.echo = 0.75
    elif any(w in vibe_lower for w in ("intimate", "close")):
        config.space = 0.4
        config.echo = 0.3

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
    return a + (b - a) * t


def interpolate_configs(config_a: MusicConfig, config_b: MusicConfig, t: float) -> MusicConfig:
    """Interpolate between two configs."""
    return MusicConfig(
        tempo=lerp(config_a.tempo, config_b.tempo, t),
        root=config_a.root if t < 0.5 else config_b.root,
        mode=config_a.mode if t < 0.5 else config_b.mode,
        brightness=lerp(config_a.brightness, config_b.brightness, t),
        space=lerp(config_a.space, config_b.space, t),
        density=round(lerp(config_a.density, config_b.density, t)),
        bass=config_a.bass if t < 0.4 else config_b.bass,
        pad=config_a.pad if t < 0.5 else config_b.pad,
        melody=config_a.melody if t < 0.6 else config_b.melody,
        rhythm=config_a.rhythm if t < 0.5 else config_b.rhythm,
        texture=config_a.texture if t < 0.7 else config_b.texture,
        accent=config_a.accent if t < 0.8 else config_b.accent,
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
        seed=config_a.seed if t < 0.5 else config_b.seed,
    )


def morph_audio(
    config_a: MusicConfig, config_b: MusicConfig, duration: float = 60.0, segments: int = 8
) -> FloatArray:
    """Generate audio that morphs from config_a to config_b."""
    segment_duration = duration / segments
    output = []

    for i in range(segments):
        t = i / (segments - 1) if segments > 1 else 0.0
        interpolated = interpolate_configs(config_a, config_b, t)
        segment = assemble(interpolated, segment_duration, normalize=False)
        output.append(segment)

    result = _ensure_float32(np.concatenate(output))  # normalize dtype after concat.

    result = apply_final_limiter(result, SAMPLE_RATE)

    return result


def crossfade(audio_a: FloatArray, audio_b: FloatArray, crossfade_samples: int) -> FloatArray:
    """Crossfade between two audio arrays at the midpoint."""
    min_len = min(len(audio_a), len(audio_b))
    mid = min_len // 2
    half_cf = crossfade_samples // 2

    fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)

    result = _ensure_float32(
        np.concatenate(
            (
                audio_a[: mid - half_cf],
                audio_a[mid - half_cf : mid + half_cf] * fade_out
                + audio_b[mid - half_cf : mid + half_cf] * fade_in,
                audio_b[mid + half_cf : min_len],
            )
        )
    )  # normalize dtype after concat.

    return result


def transition(config_a: MusicConfig, config_b: MusicConfig, duration: float = 60.0) -> FloatArray:
    """Generate with crossfade transition."""
    audio_a = assemble(config_a, duration, normalize=False)
    audio_b = assemble(config_b, duration, normalize=False)

    crossfade_duration = 4.0
    crossfade_samples = int(crossfade_duration * SAMPLE_RATE)

    result = crossfade(audio_a, audio_b, crossfade_samples)
    result = apply_final_limiter(result, SAMPLE_RATE)

    return result


def generate_tween_with_automation(
    config_a: MusicConfig,
    config_b: MusicConfig,
    duration: float = 120.0,
    chunk_seconds: float = 2.0,
    overlap_seconds: float = 0.05,
) -> FloatArray:
    """Generate audio with automated parameters using Cached Block Processing."""
    sr = SAMPLE_RATE
    num_chunks = int(np.ceil(duration / chunk_seconds))
    overlap_samples = int(overlap_seconds * sr)

    PATTERN_LEN = 16.0
    chunks_per_pattern = int(PATTERN_LEN / chunk_seconds)

    chunk_len_sec = chunk_seconds + overlap_seconds
    chunk_samples = int(chunk_len_sec * sr)

    output = np.zeros(int(duration * sr), dtype=np.float32)

    cached_pattern: FloatArray | None = None
    cached_pattern_idx: int = -1

    for i in range(num_chunks):
        pattern_idx = i // chunks_per_pattern

        if pattern_idx != cached_pattern_idx:
            t = (pattern_idx * chunks_per_pattern) / max(1, num_chunks - 1)
            t = np.clip(t, 0.0, 1.0)
            t_eased = 0.5 - 0.5 * np.cos(t * np.pi)

            config = interpolate_configs(config_a, config_b, float(t_eased))
            cached_pattern = assemble(config, PATTERN_LEN, normalize=False)
            cached_pattern_idx = pattern_idx

        if cached_pattern is None:
            continue

        local_chunk_idx = i % chunks_per_pattern

        start_idx = int(local_chunk_idx * chunk_seconds * sr)
        end_idx = start_idx + chunk_samples

        pattern_len_samples = len(cached_pattern)

        if end_idx <= pattern_len_samples:
            chunk = cached_pattern[start_idx:end_idx].copy()
        else:
            part_a = cached_pattern[start_idx:]
            remainder = end_idx - pattern_len_samples
            part_b = cached_pattern[:remainder]
            chunk = np.concatenate((part_a, part_b))

        if i > 0:
            chunk[:overlap_samples] *= np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)

        if i < num_chunks - 1:
            chunk[-overlap_samples:] *= np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)

        out_start = int(i * chunk_seconds * sr)
        out_end = min(out_start + len(chunk), len(output))
        available = out_end - out_start

        if available > 0:
            output[out_start:out_end] += chunk[:available]

    output = apply_final_limiter(output, sr)

    return output


# =============================================================================
# MAIN - Demo
# =============================================================================

if __name__ == "__main__":
    import json

    print("VibeSynth - Pure Python Synthesis Engine with Pedalboard DSP")
    print("=" * 60)

    demo_configs = [
        {
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
        config = MusicConfig(
            **cast(Dict[str, Any], config_dict)
        )  # demo literals are fixed/validated.
        print(json.dumps(config_dict, indent=2))
        config_to_audio(config, fname, duration=20.0)
        print(f"   Saved: {fname}")

    print("\nAll demo audio exported.")
