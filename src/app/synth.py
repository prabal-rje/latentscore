# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

"""
Architecture:

1. Primitives: oscillators, envelopes, filters, effects
2. Patterns: pre-baked layer templates (bass, pad, melody, rhythm, texture, accent)
3. Procedural Melody: anchor+embellishment generation with phrase planning
4. Assembler: config → audio conversion with V2 parameter transforms
"""

from __future__ import annotations

from dataclasses import dataclass, replace, field
from enum import Enum, auto
from typing import Any, Callable, Dict, TypeAlias, cast, List, Tuple, Optional
import random

import numpy as np
import soundfile as sf  # type: ignore[import]
from numpy.typing import NDArray
from scipy.signal import butter, lfilter, decimate  # type: ignore[import]

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

# Chord tones for each chord quality (scale degrees that are chord tones)
CHORD_TONES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4),      # 1, 3, 5
    "minor": (0, 2, 4),      # 1, b3, 5
    "dom7": (0, 2, 4, 6),    # 1, 3, 5, b7
    "min7": (0, 2, 4, 6),    # 1, b3, 5, b7
    "maj7": (0, 2, 4, 6),    # 1, 3, 5, 7
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
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3,
    oversample: int = 2
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
    signal = decimate(signal_high, oversample, ftype='fir', zero_phase=True)
    
    # Ensure exact output length
    if len(signal) > num_samples:
        signal = signal[:num_samples]
    elif len(signal) < num_samples:
        signal = np.pad(signal, (0, num_samples - len(signal)))

    return cast(FloatArray, amp * signal)


def generate_square(
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3,
    oversample: int = 2
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
    signal = decimate(signal_high, oversample, ftype='fir', zero_phase=True)
    
    # Ensure exact output length
    if len(signal) > num_samples:
        signal = signal[:num_samples]
    elif len(signal) < num_samples:
        signal = np.pad(signal, (0, num_samples - len(signal)))

    return cast(FloatArray, amp * signal)


def generate_noise(duration: float, sr: int = SAMPLE_RATE, amp: float = 0.1) -> FloatArray:
    """Generate white noise."""
    return amp * np.random.randn(int(sr * duration))


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
    """Apply lowpass filter (causal, analog-style)."""
    nyquist = sr / 2
    normalized = min(max(cutoff / nyquist, 0.001), 0.99)
    b, a = cast(FilterCoefficients, butter(2, normalized, btype="low", output="ba"))
    return cast(FloatArray, lfilter(b, a, signal))


def apply_highpass(signal: FloatArray, cutoff: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply highpass filter (causal, analog-style)."""
    nyquist = sr / 2
    normalized = min(max(cutoff / nyquist, 0.001), 0.99)
    b, a = cast(FilterCoefficients, butter(2, normalized, btype="high", output="ba"))
    return cast(FloatArray, lfilter(b, a, signal))


def apply_delay(
    signal: FloatArray, delay_time: float, feedback: float, wet: float, sr: int = SAMPLE_RATE
) -> FloatArray:
    """Apply delay effect."""
    delay_samples = int(delay_time * sr)
    output = signal.copy()

    for i in range(1, 5):  # 5 delay taps
        offset = delay_samples * i
        if offset < len(signal):
            delayed = np.zeros_like(signal)
            delayed[offset:] = signal[:-offset] * (feedback**i) * wet
            output += delayed

    return output


def apply_reverb(signal: FloatArray, room: float, size: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Simple reverb via multiple delays."""
    output = signal.copy()

    # Multiple delay lines at prime-ish intervals (tuple)
    delays = (0.029, 0.037, 0.041, 0.053, 0.067)

    for i, delay in enumerate(delays):
        delay_samples = int(delay * size * sr)
        if delay_samples < len(signal) and delay_samples > 0:
            reverb = np.zeros_like(signal)
            reverb[delay_samples:] = signal[:-delay_samples] * room * (0.7**i)
            output += reverb

    return output


def generate_lfo(duration: float, rate: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Generate LFO signal (0 to 1 range)."""
    t = np.linspace(0, duration, int(sr * duration), False)
    return 0.5 + 0.5 * np.sin(2 * np.pi * rate * t)


def apply_humanize(signal: FloatArray, amount: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply subtle timing/amplitude humanization."""
    if amount <= 0:
        return signal

    # Subtle amplitude variation
    amp_lfo = 1.0 + (np.random.randn(len(signal)) * amount * 0.1)
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
# PART 1.5: PROCEDURAL MELODY SYSTEM
# =============================================================================


class MelodicRole(Enum):
    """FSM states for melodic note roles."""
    REST = auto()
    CHORD_TONE = auto()
    PASSING = auto()
    NEIGHBOR = auto()
    APPROACH = auto()
    LEAP = auto()
    CADENCE = auto()


class TensionCurve(Enum):
    """Tension envelope shapes for phrases."""
    ARC = "arc"           # low → high → resolve
    RAMP = "ramp"         # gradually increase tension
    WAVES = "waves"       # rise/fall every bar
    FLAT = "flat"         # constant tension


@dataclass
class Chord:
    """Represents a chord in the progression."""
    root_degree: int      # Scale degree of root (0-6)
    quality: str          # "major", "minor", "dom7", etc.
    duration_beats: float # Duration in beats
    
    def get_chord_tones(self) -> tuple[int, ...]:
        """Get chord tones as scale degrees relative to chord root."""
        return CHORD_TONES.get(self.quality, CHORD_TONES["major"])


@dataclass
class MelodyEvent:
    """A single melody note event."""
    start_seconds: float
    duration_seconds: float
    degree: int           # Scale degree
    octave: int
    velocity: float       # 0.0 to 1.0
    role: MelodicRole = MelodicRole.CHORD_TONE


@dataclass
class Motif:
    """A short melodic motif for repetition/variation."""
    intervals: tuple[int, ...]      # Intervals from first note
    durations: tuple[float, ...]    # Relative durations
    
    def transpose(self, semitones: int) -> "Motif":
        """Transpose the motif by semitones."""
        return Motif(
            intervals=tuple(i + semitones for i in self.intervals),
            durations=self.durations
        )
    
    def augment(self, factor: float = 2.0) -> "Motif":
        """Stretch durations."""
        return Motif(
            intervals=self.intervals,
            durations=tuple(d * factor for d in self.durations)
        )
    
    def diminish(self, factor: float = 0.5) -> "Motif":
        """Compress durations."""
        return self.augment(factor)
    
    def invert(self) -> "Motif":
        """Invert intervals."""
        return Motif(
            intervals=tuple(-i for i in self.intervals),
            durations=self.durations
        )
    
    def vary_note(self, idx: int, delta: int) -> "Motif":
        """Change one note by delta steps."""
        new_intervals = list(self.intervals)
        if 0 <= idx < len(new_intervals):
            new_intervals[idx] += delta
        return Motif(intervals=tuple(new_intervals), durations=self.durations)


@dataclass
class MelodyPolicy:
    """
    Policy knobs for procedural melody generation.
    These are the parameters the LLM outputs to control melody behavior.
    """
    # Phrase structure
    phrase_len_bars: int = 4            # 2, 4, or 8 bars per phrase
    
    # Rhythm
    note_density: float = 0.5           # 0.0 (sparse) to 1.0 (dense)
    syncopation: float = 0.2            # 0.0 (on-beat) to 1.0 (off-beat)
    swing: float = 0.0                  # 0.0 (straight) to 1.0 (heavy swing)
    
    # Pitch behavior
    step_vs_leap: float = 0.8           # 0.0 (all leaps) to 1.0 (all steps)
    chromatic_approach_prob: float = 0.1  # Probability of chromatic approach tones
    
    # Motif
    motif_repeat_prob: float = 0.3      # Probability of repeating/varying motif
    motif_length: int = 4               # Notes in motif (3-6)
    
    # Phrase shape
    cadence_strength: float = 0.7       # 0.0 (weak) to 1.0 (strong cadences)
    register_range: tuple[int, int] = (4, 5)  # Octave range
    tension_curve: TensionCurve = TensionCurve.ARC
    
    # Style modifiers
    rest_probability: float = 0.15      # Probability of rest on weak beats


@dataclass 
class ChordProgression:
    """A sequence of chords with timing."""
    chords: List[Chord] = field(default_factory=list)
    beats_per_bar: int = 4
    
    @classmethod
    def simple_progression(cls, mode: str = "minor", bars: int = 4) -> "ChordProgression":
        """Generate a simple chord progression for the given mode."""
        if mode == "major":
            # I - IV - V - I
            chords = [
                Chord(0, "major", 4),
                Chord(3, "major", 4),
                Chord(4, "major", 4),
                Chord(0, "major", 4),
            ]
        elif mode == "minor":
            # i - iv - VII - i
            chords = [
                Chord(0, "minor", 4),
                Chord(3, "minor", 4),
                Chord(6, "major", 4),
                Chord(0, "minor", 4),
            ]
        elif mode == "dorian":
            # i - IV - i - VII
            chords = [
                Chord(0, "minor", 4),
                Chord(3, "major", 4),
                Chord(0, "minor", 4),
                Chord(6, "major", 4),
            ]
        else:  # mixolydian
            # I - bVII - IV - I
            chords = [
                Chord(0, "major", 4),
                Chord(6, "major", 4),
                Chord(3, "major", 4),
                Chord(0, "major", 4),
            ]
        
        # Repeat to fill bars
        while len(chords) < bars:
            chords.extend(chords[:bars - len(chords)])
        
        return cls(chords=chords[:bars])
    
    def get_chord_at_beat(self, beat: float) -> Chord:
        """Get the chord active at a given beat."""
        current_beat = 0.0
        for chord in self.chords:
            if current_beat <= beat < current_beat + chord.duration_beats:
                return chord
            current_beat += chord.duration_beats
        return self.chords[-1] if self.chords else Chord(0, "major", 4)
    
    def total_beats(self) -> float:
        """Total duration in beats."""
        return sum(c.duration_beats for c in self.chords)


def get_tension_at_position(pos: float, curve: TensionCurve, phrase_len: float) -> float:
    """
    Get tension value (0-1) at a position within a phrase.
    pos: 0.0 to phrase_len
    """
    t = pos / phrase_len if phrase_len > 0 else 0.0
    t = min(max(t, 0.0), 1.0)
    
    if curve == TensionCurve.ARC:
        # Bell curve: low at start/end, high in middle
        return float(np.sin(t * np.pi))
    elif curve == TensionCurve.RAMP:
        # Linear increase
        return t
    elif curve == TensionCurve.WAVES:
        # Multiple peaks
        return float(0.5 + 0.5 * np.sin(t * 4 * np.pi))
    else:  # FLAT
        return 0.5


def generate_motif(policy: MelodyPolicy, scale_len: int = 7) -> Motif:
    """Generate a short melodic motif."""
    num_notes = policy.motif_length
    intervals = [0]  # Start on the root
    durations = []
    
    for i in range(num_notes - 1):
        # Prefer steps over leaps based on policy
        if random.random() < policy.step_vs_leap:
            # Step motion
            interval = random.choice([-1, 1])
        else:
            # Leap
            interval = random.choice([-3, -2, 2, 3])
        
        intervals.append(intervals[-1] + interval)
    
    # Generate durations (normalized)
    base_dur = 1.0 / num_notes
    for _ in range(num_notes):
        # Add some variation
        dur = base_dur * random.uniform(0.7, 1.3)
        durations.append(dur)
    
    # Normalize durations to sum to 1.0
    total = sum(durations)
    durations = [d / total for d in durations]
    
    return Motif(intervals=tuple(intervals), durations=tuple(durations))


def generate_anchor_notes(
    chord_prog: ChordProgression,
    policy: MelodyPolicy,
    beats_per_second: float,
    total_duration: float,
    mode_intervals: tuple[int, ...]
) -> List[MelodyEvent]:
    """
    Generate anchor notes (Pass A) - chord tones on strong beats.
    """
    anchors: List[MelodyEvent] = []
    
    total_beats = total_duration * beats_per_second
    beats_per_bar = chord_prog.beats_per_bar
    
    # Determine anchor positions (beat 1 of each bar, sometimes beat 3)
    beat = 0.0
    prev_degree = 0
    prev_octave = policy.register_range[0]
    
    while beat < total_beats:
        chord = chord_prog.get_chord_at_beat(beat % chord_prog.total_beats())
        chord_tones = chord.get_chord_tones()
        
        # Get phrase position for tension
        phrase_beats = policy.phrase_len_bars * beats_per_bar
        phrase_pos = beat % phrase_beats
        tension = get_tension_at_position(phrase_pos, policy.tension_curve, phrase_beats)
        
        # Voice leading: prefer small intervals from previous anchor
        best_degree = chord.root_degree
        best_distance = 999
        
        for ct in chord_tones:
            degree = (chord.root_degree + ct) % len(mode_intervals)
            distance = abs(degree - prev_degree)
            # Wrap around for voice leading
            distance = min(distance, len(mode_intervals) - distance)
            
            if distance < best_distance:
                best_distance = distance
                best_degree = degree
        
        # Determine octave based on register arc and tension
        octave_range = policy.register_range[1] - policy.register_range[0]
        if policy.tension_curve == TensionCurve.ARC:
            # Rise in middle of phrase, fall at end
            octave = policy.register_range[0] + int(tension * octave_range)
        else:
            octave = prev_octave
            # Small octave adjustments for voice leading
            if best_degree < prev_degree - 3 and octave < policy.register_range[1]:
                octave += 1
            elif best_degree > prev_degree + 3 and octave > policy.register_range[0]:
                octave -= 1
        
        # Check for cadence (end of phrase)
        is_cadence = phrase_pos >= phrase_beats - beats_per_bar
        
        # At cadence, prefer root or 5th
        if is_cadence and policy.cadence_strength > random.random():
            best_degree = chord.root_degree
            role = MelodicRole.CADENCE
        else:
            role = MelodicRole.CHORD_TONE
        
        # Calculate timing
        start_sec = beat / beats_per_second
        
        # Duration: longer at cadences, shorter at high tension
        base_dur = (beats_per_bar / 2) / beats_per_second  # Half bar
        if is_cadence:
            dur = base_dur * (1.0 + policy.cadence_strength)
        else:
            dur = base_dur * (1.0 - 0.3 * tension)
        
        # Velocity based on tension and beat position
        velocity = 0.6 + 0.3 * tension
        if beat % beats_per_bar == 0:  # Downbeat
            velocity = min(1.0, velocity + 0.1)
        
        anchors.append(MelodyEvent(
            start_seconds=start_sec,
            duration_seconds=dur,
            degree=best_degree,
            octave=octave,
            velocity=velocity,
            role=role
        ))
        
        prev_degree = best_degree
        prev_octave = octave
        
        # Move to next anchor position
        # Usually every bar, sometimes every half bar at high tension
        if tension > 0.7 and random.random() < tension:
            beat += beats_per_bar / 2
        else:
            beat += beats_per_bar
    
    return anchors


def fill_between_anchors(
    anchors: List[MelodyEvent],
    policy: MelodyPolicy,
    beats_per_second: float,
    mode_intervals: tuple[int, ...],
    chord_prog: ChordProgression
) -> List[MelodyEvent]:
    """
    Generate fill notes (Pass B) between anchors.
    Uses passing tones, neighbor tones, approach tones.
    """
    all_events: List[MelodyEvent] = []
    
    for i, anchor in enumerate(anchors):
        all_events.append(anchor)
        
        # Get next anchor (if exists)
        if i + 1 >= len(anchors):
            continue
        
        next_anchor = anchors[i + 1]
        gap_start = anchor.start_seconds + anchor.duration_seconds
        gap_end = next_anchor.start_seconds
        gap_duration = gap_end - gap_start
        
        if gap_duration < 0.1:  # Too short for fills
            continue
        
        # Determine number of fill notes based on density and gap size
        beats_in_gap = gap_duration * beats_per_second
        max_notes = int(beats_in_gap * 2 * policy.note_density)
        num_fills = random.randint(0, max(0, max_notes))
        
        if num_fills == 0:
            continue
        
        # Determine fill strategy based on interval between anchors
        interval = next_anchor.degree - anchor.degree
        
        fill_degrees: List[int] = []
        
        if abs(interval) <= 2:
            # Close anchors: use neighbor tones
            for _ in range(num_fills):
                if random.random() < 0.5:
                    # Upper neighbor
                    fill_degrees.append(anchor.degree + 1)
                else:
                    # Lower neighbor
                    fill_degrees.append(anchor.degree - 1)
        elif abs(interval) <= 4:
            # Medium distance: passing tones
            step = 1 if interval > 0 else -1
            current = anchor.degree + step
            while len(fill_degrees) < num_fills and current != next_anchor.degree:
                fill_degrees.append(current)
                current += step
        else:
            # Large leap: add approach tone before next anchor
            if random.random() < policy.chromatic_approach_prob:
                # Chromatic approach (half step)
                approach = next_anchor.degree - 1 if random.random() < 0.5 else next_anchor.degree + 1
            else:
                # Diatonic approach (whole step)
                approach = next_anchor.degree - 1
            fill_degrees.append(approach)
            
            # Maybe add one more fill
            if num_fills > 1:
                mid_degree = (anchor.degree + next_anchor.degree) // 2
                fill_degrees.insert(0, mid_degree)
        
        # Limit fills
        fill_degrees = fill_degrees[:num_fills]
        
        if not fill_degrees:
            continue
        
        # Distribute fills across the gap
        fill_duration = gap_duration / (len(fill_degrees) + 1)
        
        for j, degree in enumerate(fill_degrees):
            # Apply swing
            offset = (j + 1) * fill_duration
            if policy.swing > 0 and j % 2 == 1:
                offset += fill_duration * policy.swing * 0.3
            
            # Apply syncopation
            if policy.syncopation > 0 and random.random() < policy.syncopation:
                offset += fill_duration * random.uniform(-0.2, 0.2)
            
            start_sec = gap_start + offset
            
            # Shorter duration for fills
            dur = fill_duration * 0.8
            
            # Lower velocity for non-chord tones
            velocity = anchor.velocity * 0.7
            
            # Rest probability
            if random.random() < policy.rest_probability:
                continue
            
            # Determine role
            if abs(degree - anchor.degree) == 1:
                role = MelodicRole.NEIGHBOR
            elif degree in [next_anchor.degree - 1, next_anchor.degree + 1]:
                role = MelodicRole.APPROACH
            else:
                role = MelodicRole.PASSING
            
            all_events.append(MelodyEvent(
                start_seconds=start_sec,
                duration_seconds=dur,
                degree=degree,
                octave=anchor.octave,
                velocity=velocity,
                role=role
            ))
    
    # Sort by start time
    all_events.sort(key=lambda e: e.start_seconds)
    return all_events


def apply_motif_memory(
    events: List[MelodyEvent],
    policy: MelodyPolicy,
    mode_intervals: tuple[int, ...]
) -> List[MelodyEvent]:
    """
    Apply motif repetition and variation to make melody more memorable.
    """
    if len(events) < policy.motif_length * 2:
        return events
    
    # Extract first few notes as the motif
    motif_events = events[:policy.motif_length]
    base_degree = motif_events[0].degree
    
    motif = Motif(
        intervals=tuple(e.degree - base_degree for e in motif_events),
        durations=tuple(e.duration_seconds for e in motif_events)
    )
    
    result = list(events)
    
    # Find opportunities to insert motif variations
    i = policy.motif_length
    while i < len(result) - policy.motif_length:
        if random.random() < policy.motif_repeat_prob:
            # Choose a transformation
            transform_type = random.choice(['transpose', 'vary', 'invert', 'none'])
            
            if transform_type == 'transpose':
                # Transpose to current position's degree
                target_degree = result[i].degree
                delta = target_degree - base_degree
                varied = motif.transpose(delta)
            elif transform_type == 'vary':
                # Change one note
                idx = random.randint(0, len(motif.intervals) - 1)
                delta = random.choice([-1, 1])
                varied = motif.vary_note(idx, delta)
            elif transform_type == 'invert':
                varied = motif.invert()
            else:
                varied = motif
            
            # Apply the motif
            base_start = result[i].start_seconds
            base_octave = result[i].octave
            base_velocity = result[i].velocity
            
            for j, (interval, dur) in enumerate(zip(varied.intervals, varied.durations)):
                if i + j < len(result):
                    result[i + j] = MelodyEvent(
                        start_seconds=base_start + sum(varied.durations[:j]),
                        duration_seconds=dur,
                        degree=(base_degree + interval) % len(mode_intervals),
                        octave=base_octave,
                        velocity=base_velocity,
                        role=MelodicRole.CHORD_TONE
                    )
            
            i += policy.motif_length + random.randint(2, 6)
        else:
            i += 1
    
    return result


def generate_procedural_melody(
    params: "SynthParams",
    policy: MelodyPolicy,
    duration: float
) -> List[MelodyEvent]:
    """
    Main entry point for procedural melody generation.
    Combines anchor generation, fill notes, and motif memory.
    """
    mode_intervals = MODE_INTERVALS.get(params.mode, MODE_INTERVALS["minor"])
    
    # Create chord progression
    bars = int(duration / (4 / (params.tempo * 2)))  # Rough bar count
    bars = max(4, bars)
    chord_prog = ChordProgression.simple_progression(params.mode, bars)
    
    # Calculate tempo
    beats_per_second = params.tempo * 2  # Map tempo to reasonable BPS
    
    # Generate anchors (Pass A)
    anchors = generate_anchor_notes(
        chord_prog, policy, beats_per_second, duration, mode_intervals
    )
    
    # Generate fills (Pass B)
    events = fill_between_anchors(
        anchors, policy, beats_per_second, mode_intervals, chord_prog
    )
    
    # Apply motif memory
    events = apply_motif_memory(events, policy, mode_intervals)
    
    return events


def render_melody_events(
    events: List[MelodyEvent],
    params: "SynthParams",
    duration: float
) -> FloatArray:
    """Render melody events to audio buffer."""
    sr = SAMPLE_RATE
    signal = np.zeros(int(sr * duration))
    
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)
    mode_intervals = MODE_INTERVALS.get(params.mode, MODE_INTERVALS["minor"])
    
    for event in events:
        # Calculate frequency
        semitone = mode_intervals[event.degree % len(mode_intervals)]
        semitone += 12 * (event.degree // len(mode_intervals))
        freq = freq_from_note(params.root, semitone, event.octave)
        
        # Generate note
        note_dur = min(event.duration_seconds, duration - event.start_seconds)
        if note_dur <= 0:
            continue
        
        amp = 0.18 * event.velocity
        note = osc(freq, note_dur, sr, amp)
        
        # Apply envelope based on role
        if event.role == MelodicRole.CADENCE:
            note = apply_adsr(note, 0.1 * params.attack_mult, 0.3, 0.6, 0.8 * params.attack_mult, sr)
        elif event.role in [MelodicRole.PASSING, MelodicRole.NEIGHBOR, MelodicRole.APPROACH]:
            note = apply_adsr(note, 0.02 * params.attack_mult, 0.1, 0.4, 0.15 * params.attack_mult, sr)
        else:
            note = apply_adsr(note, 0.08 * params.attack_mult, 0.25, 0.55, 0.4 * params.attack_mult, sr)
        
        # Apply filter
        note = apply_lowpass(note, 800 * params.brightness + 200, sr)
        
        # Apply humanize
        note = apply_humanize(note, params.human, sr)
        
        start_idx = int(event.start_seconds * sr)
        add_note(signal, note, start_idx)
    
    return signal


# =============================================================================
# PART 2: PATTERN GENERATORS
# =============================================================================


@dataclass
class SynthParams:
    """Parameters passed to all synthesis functions."""

    root: str = "c"
    mode: str = "minor"
    brightness: float = 0.5
    space: float = 0.6
    duration: float = 16.0
    tempo: float = 0.35

    # V2 parameters
    motion: float = 0.5
    attack: str = "medium"
    stereo: float = 0.5
    depth: bool = False
    echo: float = 0.5
    human: float = 0.0
    grain: str = "clean"
    
    # Melody policy (optional, for procedural melody)
    melody_policy: Optional[MelodyPolicy] = None

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

    def get_scale_freq(self, degree: int, octave: int = 4) -> float:
        """Get frequency for a scale degree in the current mode."""
        intervals = MODE_INTERVALS.get(self.mode, MODE_INTERVALS["minor"])
        semitone = intervals[degree % len(intervals)] + (12 * (degree // len(intervals)))
        return freq_from_note(self.root, semitone, octave)


PatternFn: TypeAlias = Callable[[SynthParams], FloatArray]


# -----------------------------------------------------------------------------
# BASS PATTERNS
# -----------------------------------------------------------------------------


def bass_drone(params: SynthParams) -> FloatArray:
    """Sustained drone bass."""
    sr = SAMPLE_RATE
    dur = params.duration
    freq = params.get_scale_freq(0, 2)

    signal = generate_sine(freq, dur, sr, 0.35)
    signal = apply_lowpass(signal, 80 * params.brightness + 20, sr)
    signal = apply_adsr(signal, 2.0 * params.attack_mult, 0.5, 0.95, 3.0 * params.attack_mult, sr)
    signal = apply_reverb(signal, params.space * 0.5, params.space, sr)

    return signal


def bass_sustained(params: SynthParams) -> FloatArray:
    """Long sustained notes with slow movement."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Root and fifth alternating slowly (tuple)
    pattern = (0, 0, 4, 0)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        freq = params.get_scale_freq(degree, 2)
        start = int(i * note_dur * sr)

        note = generate_sine(freq, note_dur, sr, 0.32)
        note = apply_lowpass(note, 100 * params.brightness + 30, sr)
        note = apply_adsr(note, 0.8 * params.attack_mult, 0.3, 0.85, 1.5 * params.attack_mult, sr)

        add_note(signal, note, start)

    signal = apply_reverb(signal, params.space * 0.4, params.space, sr)
    return signal


def bass_pulsing(params: SynthParams) -> FloatArray:
    """Rhythmic pulsing bass."""
    sr = SAMPLE_RATE
    dur = params.duration

    # 8 pulses per cycle
    num_pulses = 8
    pulse_dur = dur / num_pulses
    signal = np.zeros(int(sr * dur))

    for i in range(num_pulses):
        freq = params.get_scale_freq(0, 2)
        start = int(i * pulse_dur * sr)

        note = generate_sine(freq, pulse_dur * 0.8, sr, 0.35)
        note = apply_lowpass(note, 90 * params.brightness + 20, sr)
        note = apply_adsr(note, 0.02 * params.attack_mult, 0.1, 0.6, 0.3 * params.attack_mult, sr)

        add_note(signal, note, start)

    return signal


def bass_walking(params: SynthParams) -> FloatArray:
    """Walking bass line."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Walking pattern (tuple)
    pattern = (0, 2, 4, 2, 0, 2, 4, 4)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        freq = params.get_scale_freq(degree, 2)
        start = int(i * note_dur * sr)

        note = generate_triangle(freq, note_dur * 0.9, sr, 0.30)
        note = apply_lowpass(note, 120 * params.brightness + 40, sr)
        note = apply_adsr(note, 0.05 * params.attack_mult, 0.15, 0.7, 0.25 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr)

        add_note(signal, note, start)

    signal = apply_reverb(signal, params.space * 0.3, params.space * 0.8, sr)
    return signal


def bass_fifth_drone(params: SynthParams) -> FloatArray:
    """Root + fifth drone."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Root
    root_freq = params.get_scale_freq(0, 2)
    root = generate_sine(root_freq, dur, sr, 0.28)
    root = apply_lowpass(root, 70 * params.brightness + 20, sr)
    root = apply_adsr(root, 2.5 * params.attack_mult, 0.5, 0.95, 3.0 * params.attack_mult, sr)

    # Fifth
    fifth_freq = params.get_scale_freq(4, 2)
    fifth = generate_sine(fifth_freq, dur, sr, 0.18)
    fifth = apply_lowpass(fifth, 100 * params.brightness + 30, sr)
    fifth = apply_adsr(fifth, 3.0 * params.attack_mult, 0.5, 0.9, 3.0 * params.attack_mult, sr)

    signal = root + fifth
    signal = apply_reverb(signal, params.space * 0.5, params.space, sr)
    return signal


def bass_sub_pulse(params: SynthParams) -> FloatArray:
    """Deep sub-bass pulse."""
    sr = SAMPLE_RATE
    dur = params.duration

    freq = params.get_scale_freq(0, 1)  # Very low octave

    # Slow pulse (4 per cycle)
    num_pulses = 4
    pulse_dur = dur / num_pulses
    signal = np.zeros(int(sr * dur))

    for i in range(num_pulses):
        start = int(i * pulse_dur * sr)

        note = generate_sine(freq, pulse_dur * 0.95, sr, 0.4)
        note = apply_lowpass(note, 50, sr)
        note = apply_adsr(note, 0.3 * params.attack_mult, 0.2, 0.9, 0.8 * params.attack_mult, sr)

        add_note(signal, note, start)

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
    """Warm, slowly evolving pad."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_sine)

    signal = np.zeros(int(sr * dur))

    # Stack root, 3rd, 5th (tuple)
    for degree in (0, 2, 4):
        freq = params.get_scale_freq(degree, 3)
        tone = osc(freq, dur, sr, 0.15)

        # Slow filter movement (affected by motion)
        lfo_rate = 0.1 / (params.motion + 0.1)
        lfo = generate_lfo(dur, lfo_rate, sr)

        # Apply moving filter
        base_cutoff = 300 * params.brightness + 100
        tone_low = apply_lowpass(tone, base_cutoff * 0.5, sr)
        tone_high = apply_lowpass(tone, base_cutoff * 1.5, sr)
        tone = tone_low * (1 - lfo) + tone_high * lfo

        tone = apply_adsr(tone, 1.5 * params.attack_mult, 0.8, 0.85, 2.5 * params.attack_mult, sr)
        signal += tone

    signal = apply_reverb(signal, params.space * 0.7, params.space * 0.9, sr)
    signal = apply_delay(signal, 0.35, 0.3 * params.echo_mult, 0.25 * params.echo_mult, sr)
    signal = apply_humanize(signal, params.human, sr)

    return signal


def pad_dark_sustained(params: SynthParams) -> FloatArray:
    """Dark, heavy sustained pad."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # Minor chord voicing (tuple)
    for degree in (0, 2, 4):
        freq = params.get_scale_freq(degree, 3)
        tone = generate_sawtooth(freq, dur, sr, 0.12)
        tone = apply_lowpass(tone, 200 * params.brightness + 80, sr)
        tone = apply_adsr(tone, 2.0 * params.attack_mult, 1.0, 0.9, 3.0 * params.attack_mult, sr)
        signal += tone

    signal = apply_reverb(signal, params.space * 0.8, params.space, sr)
    return signal


def pad_cinematic(params: SynthParams) -> FloatArray:
    """Big, cinematic pad with movement."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # Wider voicing with octave doubling (tuple of tuples)
    voicings = ((0, 3), (2, 3), (4, 3), (0, 4), (4, 4))

    for degree, octave in voicings:
        freq = params.get_scale_freq(degree, octave)

        # Mix oscillators
        tone = generate_sawtooth(freq, dur, sr, 0.08)
        tone += generate_triangle(freq * 1.002, dur, sr, 0.06)  # Slight detune

        tone = apply_lowpass(tone, 400 * params.brightness + 150, sr)
        tone = apply_adsr(tone, 1.8 * params.attack_mult, 0.8, 0.88, 2.8 * params.attack_mult, sr)
        signal += tone

    signal = apply_reverb(signal, params.space * 0.85, params.space, sr)
    signal = apply_delay(signal, 0.4, 0.35 * params.echo_mult, 0.3 * params.echo_mult, sr)

    return signal


def pad_thin_high(params: SynthParams) -> FloatArray:
    """Thin, high pad."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    for degree in (0, 4):  # Just root and fifth, high
        freq = params.get_scale_freq(degree, 4)
        tone = generate_sine(freq, dur, sr, 0.12)
        tone = apply_lowpass(tone, 800 * params.brightness + 200, sr)
        tone = apply_highpass(tone, 200, sr)
        tone = apply_adsr(tone, 1.2 * params.attack_mult, 0.6, 0.8, 2.0 * params.attack_mult, sr)
        signal += tone

    signal = apply_reverb(signal, params.space * 0.75, params.space * 0.95, sr)
    return signal


def pad_ambient_drift(params: SynthParams) -> FloatArray:
    """Slowly drifting ambient pad."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # Evolving chord (tuple of tuples)
    chord_dur = dur / 4
    chord_progressions = (
        (0, 2, 4),
        (0, 2, 5),
        (0, 3, 4),
        (0, 2, 4),
    )

    for i, chord in enumerate(chord_progressions):
        start = int(i * chord_dur * sr)

        for degree in chord:
            freq = params.get_scale_freq(degree, 3)
            tone = generate_sine(freq, chord_dur * 1.2, sr, 0.14)  # Overlap
            tone = apply_lowpass(tone, 350 * params.brightness + 100, sr)
            tone = apply_adsr(
                tone, 1.5 * params.attack_mult, 0.5, 0.85, 2.0 * params.attack_mult, sr
            )

            add_note(signal, tone, start)

    signal = apply_reverb(signal, params.space * 0.8, params.space, sr)
    signal = apply_delay(signal, 0.5, 0.4 * params.echo_mult, 0.35 * params.echo_mult, sr)

    return signal


def pad_stacked_fifths(params: SynthParams) -> FloatArray:
    """Fifths stacked for a powerful sound."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    signal = np.zeros(int(sr * dur))

    # Stack fifths (tuple of tuples)
    voicings = ((0, 3), (4, 3), (0, 4), (4, 4))
    for degree, octave in voicings:
        freq = params.get_scale_freq(degree, octave)
        tone = osc(freq, dur, sr, 0.10)
        tone = apply_lowpass(tone, 500 * params.brightness + 150, sr)
        tone = apply_adsr(tone, 1.3 * params.attack_mult, 0.7, 0.88, 2.2 * params.attack_mult, sr)
        signal += tone

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
# MELODY PATTERNS (Including Procedural)
# -----------------------------------------------------------------------------


def melody_procedural(params: SynthParams) -> FloatArray:
    """Procedurally generated melody using anchor+embellishment algorithm."""
    sr = SAMPLE_RATE
    dur = params.duration
    
    # Use melody policy if provided, otherwise create default
    policy = params.melody_policy if params.melody_policy else MelodyPolicy()
    
    # Generate melody events
    events = generate_procedural_melody(params, policy, dur)
    
    # Render to audio
    signal = render_melody_events(events, params, dur)
    
    # Apply effects
    signal = apply_delay(signal, 0.35, 0.4 * params.echo_mult, 0.3 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.6, params.space * 0.8, sr)
    
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
        kick = apply_humanize(kick, params.human, sr)

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
        kick = apply_humanize(kick, params.human, sr)

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

    # Sparse noise impulses
    signal = np.zeros(int(sr * dur))

    num_crackles = int(dur * 20)  # ~20 crackles per second

    for _ in range(num_crackles):
        pos = np.random.randint(0, len(signal) - 100)
        crackle = generate_noise(0.002, sr, np.random.uniform(0.01, 0.04))
        crackle = apply_highpass(crackle, 2000, sr)

        add_note(signal, crackle, pos)

    # Soft background hiss
    hiss = generate_noise(dur, sr, 0.008)
    hiss = apply_lowpass(hiss, 8000, sr)
    hiss = apply_highpass(hiss, 1000, sr)
    signal += hiss

    return signal


def texture_breath(params: SynthParams) -> FloatArray:
    """Breathing texture."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Filtered noise with slow envelope
    signal = generate_noise(dur, sr, 0.06)

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

    signal = np.zeros(int(sr * dur))

    # Random high plinks
    num_stars = int(dur * 3)  # ~3 per second

    # Scale degrees for stars (tuple)
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
        bell = apply_humanize(bell, params.human, sr)

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
        pluck = apply_humanize(pluck, params.human, sr)

        add_note(signal, pluck, start)

    signal = apply_reverb(signal, params.space * 0.5, params.space * 0.7, sr)
    return signal


def accent_chime(params: SynthParams) -> FloatArray:
    """Wind chime accents."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # Random chime hits
    num_chimes = int(dur * 1.5)

    # Chime degrees (tuple)
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
    
    # Melody policy parameters (for procedural melody)
    melody_phrase_len: int = 4
    melody_density: float = 0.5
    melody_syncopation: float = 0.2
    melody_swing: float = 0.0
    melody_step_vs_leap: float = 0.8
    melody_chromatic: float = 0.1
    melody_motif_repeat: float = 0.3
    melody_cadence_strength: float = 0.7
    melody_tension_curve: str = "arc"

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
            elif hasattr(config, key):
                setattr(config, key, value)

        return config
    
    def get_melody_policy(self) -> MelodyPolicy:
        """Convert config melody parameters to MelodyPolicy."""
        tension_map = {
            "arc": TensionCurve.ARC,
            "ramp": TensionCurve.RAMP,
            "waves": TensionCurve.WAVES,
            "flat": TensionCurve.FLAT,
        }
        return MelodyPolicy(
            phrase_len_bars=self.melody_phrase_len,
            note_density=self.melody_density,
            syncopation=self.melody_syncopation,
            swing=self.melody_swing,
            step_vs_leap=self.melody_step_vs_leap,
            chromatic_approach_prob=self.melody_chromatic,
            motif_repeat_prob=self.melody_motif_repeat,
            cadence_strength=self.melody_cadence_strength,
            tension_curve=tension_map.get(self.melody_tension_curve, TensionCurve.ARC),
        )


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
    
    # Get melody policy from config
    melody_policy = config.get_melody_policy()

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
        melody_policy=melody_policy,
    )

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
    
    # Melody style from vibe
    if any(w in vibe_lower for w in ("bouncy", "playful", "mario", "game")):
        config.melody = "procedural"
        config.melody_density = 0.8
        config.melody_motif_repeat = 0.6
        config.melody_step_vs_leap = 0.6
        config.melody_syncopation = 0.3
    elif any(w in vibe_lower for w in ("jazz", "bebop", "swing")):
        config.melody = "procedural"
        config.melody_chromatic = 0.3
        config.melody_swing = 0.5
        config.melody_syncopation = 0.4
    elif any(w in vibe_lower for w in ("ambient", "drone", "minimal")):
        config.melody = "minimal"
        config.melody_density = 0.2

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
        # Melody policy parameters
        melody_phrase_len=config_a.melody_phrase_len if t < 0.5 else config_b.melody_phrase_len,
        melody_density=lerp(config_a.melody_density, config_b.melody_density, t),
        melody_syncopation=lerp(config_a.melody_syncopation, config_b.melody_syncopation, t),
        melody_swing=lerp(config_a.melody_swing, config_b.melody_swing, t),
        melody_step_vs_leap=lerp(config_a.melody_step_vs_leap, config_b.melody_step_vs_leap, t),
        melody_chromatic=lerp(config_a.melody_chromatic, config_b.melody_chromatic, t),
        melody_motif_repeat=lerp(config_a.melody_motif_repeat, config_b.melody_motif_repeat, t),
        melody_cadence_strength=lerp(config_a.melody_cadence_strength, config_b.melody_cadence_strength, t),
        melody_tension_curve=config_a.melody_tension_curve if t < 0.5 else config_b.melody_tension_curve,
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


def crossfade(
    audio_a: FloatArray, audio_b: FloatArray, crossfade_samples: int
) -> FloatArray:
    """Crossfade between two audio arrays at the midpoint."""
    min_len = min(len(audio_a), len(audio_b))
    mid = min_len // 2
    half_cf = crossfade_samples // 2

    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples)

    result = np.concatenate((
        audio_a[: mid - half_cf],
        audio_a[mid - half_cf : mid + half_cf] * fade_out
        + audio_b[mid - half_cf : mid + half_cf] * fade_in,
        audio_b[mid + half_cf : min_len],
    ))

    return result


def transition(
    config_a: MusicConfig, config_b: MusicConfig, duration: float = 60.0
) -> FloatArray:
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
    num_chunks = int(np.ceil(duration / chunk_seconds)) # ceil ensures we cover full duration
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

    # Demo: Procedural melody generation
    print("\n1. Generating procedural melody demo...")
    
    # Mario-style bouncy config
    mario_config = MusicConfig(
        tempo=0.72,
        root="c",
        mode="major",
        brightness=0.85,
        space=0.15,
        density=4,
        bass="pulsing",
        pad="thin_high",
        melody="procedural",  # Use procedural melody!
        rhythm="electronic",
        texture="none",
        accent="bells",
        motion=0.65,
        attack="sharp",
        stereo=0.35,
        depth=False,
        echo=0.12,
        human=0.0,
        grain="gritty",
        # Melody policy for bouncy game music
        melody_phrase_len=4,
        melody_density=0.8,
        melody_syncopation=0.3,
        melody_swing=0.0,
        melody_step_vs_leap=0.6,
        melody_chromatic=0.05,
        melody_motif_repeat=0.6,
        melody_cadence_strength=0.8,
        melody_tension_curve="arc",
    )
    
    config_to_audio(mario_config, "mario_procedural.wav", duration=20.0)
    print("   Saved: mario_procedural.wav")
    
    # Lo-fi chill config
    print("\n2. Generating lo-fi chill demo...")
    lofi_config = MusicConfig(
        tempo=0.35,
        root="d",
        mode="dorian",
        brightness=0.4,
        space=0.6,
        density=4,
        bass="sustained",
        pad="warm_slow",
        melody="procedural",
        rhythm="minimal",
        texture="vinyl_crackle",
        accent="pluck",
        motion=0.3,
        attack="soft",
        stereo=0.5,
        depth=False,
        echo=0.5,
        human=0.3,
        grain="warm",
        # Lo-fi melody policy
        melody_phrase_len=8,
        melody_density=0.3,
        melody_syncopation=0.1,
        melody_swing=0.4,
        melody_step_vs_leap=0.9,
        melody_chromatic=0.15,
        melody_motif_repeat=0.4,
        melody_cadence_strength=0.5,
        melody_tension_curve="waves",
    )
    
    config_to_audio(lofi_config, "lofi_procedural.wav", duration=20.0)
    print("   Saved: lofi_procedural.wav")

    # Jazz-style config
    print("\n3. Generating jazz cafe demo...")
    jazz_config = MusicConfig(
        tempo=0.45,
        root="f",
        mode="dorian",
        brightness=0.5,
        space=0.5,
        density=5,
        bass="walking",
        pad="warm_slow",
        melody="procedural",
        rhythm="soft_four",
        texture="none",
        accent="pluck",
        motion=0.5,
        attack="medium",
        stereo=0.6,
        depth=False,
        echo=0.3,
        human=0.25,
        grain="warm",
        # Jazz melody policy
        melody_phrase_len=4,
        melody_density=0.6,
        melody_syncopation=0.4,
        melody_swing=0.5,
        melody_step_vs_leap=0.7,
        melody_chromatic=0.3,
        melody_motif_repeat=0.2,
        melody_cadence_strength=0.6,
        melody_tension_curve="arc",
    )
    
    config_to_audio(jazz_config, "jazz_procedural.wav", duration=20.0)
    print("   Saved: jazz_procedural.wav")

    print("\nDone! Generated 3 procedural melody audio files.")