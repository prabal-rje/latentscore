# """
# new_synth.py

# A "culture-general" procedural synth that is hard to break:
# - Canonical pitch set is scale_intervals (microtones allowed)
# - Harmony is policy-driven (drone/modal by default when uncertain)
# - Layers can have different rhythmic cycle lengths, and optional BPM overrides (quantized to safe ratios)
# - Uses Spotify's `pedalboard` for effects as much as possible when available

# Important licensing note:
# - `pedalboard` is GPLv3 on PyPI. If you intend to distribute a closed-source
#   commercial product, confirm license compatibility with counsel. This module
#   will run without pedalboard (with simpler built-in DSP fallbacks).

# Audio array convention:
# - Internally we use channels-first float32 arrays with shape (channels, samples),
#   matching pedalboard's conventions.
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

# import math
# import numpy as np

# try:
#     # Pedalboard is optional (GPLv3). We treat it as an optional backend.
#     from pedalboard import (
#         Pedalboard,
#         Chorus,
#         Clipping,
#         Compressor,
#         Delay,
#         Distortion,
#         Gain,
#         HighpassFilter,
#         LadderFilter,
#         Limiter,
#         LowpassFilter,
#         Phaser,
#         Reverb,
#     )

#     PEDALBOARD_AVAILABLE = True
# except Exception:  # pragma: no cover
#     PEDALBOARD_AVAILABLE = False



# # =============================================================================
# # Pedalboard requirement
# # =============================================================================

# def _require_pedalboard() -> None:
#     """Raise a clear error if pedalboard is not installed.

#     We intentionally rely on pedalboard for FX (no custom DSP fallbacks).
#     """
#     if not PEDALBOARD_AVAILABLE:
#         raise RuntimeError(
#             'pedalboard is required for rendering. Install it with: pip install pedalboard'
#         )

# # =============================================================================
# # Defaults + lookup tables
# # =============================================================================

# DEFAULT_SAMPLE_RATE = 44_100

# NoteName = Literal["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]

# MODE_INTERVALS: Dict[str, List[float]] = {
#     # "safe" defaults
#     "major": [0, 2, 4, 5, 7, 9, 11],
#     "minor": [0, 2, 3, 5, 7, 8, 10],
#     "dorian": [0, 2, 3, 5, 7, 9, 10],
#     "phrygian": [0, 1, 3, 5, 7, 8, 10],
#     "lydian": [0, 2, 4, 6, 7, 9, 11],
#     "mixolydian": [0, 2, 4, 5, 7, 9, 10],
#     "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
#     "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
#     "pentatonic_major": [0, 2, 4, 7, 9],
#     "pentatonic_minor": [0, 3, 5, 7, 10],
#     "blues": [0, 3, 5, 6, 7, 10],
#     "chromatic": [float(i) for i in range(12)],
# }

# ROOT_TO_SEMITONE: Dict[str, int] = {
#     "c": 0,
#     "c#": 1,
#     "d": 2,
#     "d#": 3,
#     "e": 4,
#     "f": 5,
#     "f#": 6,
#     "g": 7,
#     "g#": 8,
#     "a": 9,
#     "a#": 10,
#     "b": 11,
# }

# BrightnessName = Literal["very_dark", "dark", "medium", "bright", "very_bright"]
# SpaceName = Literal["dry", "intimate", "room", "hall", "cathedral", "infinite"]
# CharacterName = Literal["clean", "warm", "saturated", "distorted"]
# MaterialName = Literal["digital", "warm_analog", "wood", "metal", "glass", "breath", "skin"]
# StereoWidthName = Literal["mono", "narrow", "medium", "wide", "immersive"]
# HarmonicMotionName = Literal["drone", "minimal", "slow", "medium", "active"]
# OrnamentName = Literal["none", "subtle", "moderate", "expressive", "virtuosic"]

# LayerRole = Literal["bass", "pad", "lead", "rhythm", "texture", "voice", "accent"]

# # A small set of "safe" tempo ratios for per-layer BPM overrides.
# # (Keeps layers phase-related, avoids "trainwreck drift".)
# SAFE_TEMPO_RATIOS: List[Tuple[int, int]] = [
#     (1, 1),
#     (1, 2),
#     (2, 1),
#     (3, 2),
#     (2, 3),
#     (4, 3),
#     (3, 4),
# ]

# # =============================================================================
# # Public dataclasses (internal render spec)
# # =============================================================================


# @dataclass
# class GlobalSpec:
#     # Time
#     bpm: float = 96.0
#     cycle_beats: int = 16  # loop length in beats (not "beats per bar")
#     subdivision: int = 2  # pulses per beat (2=8ths, 3=triplet, 4=16ths)
#     swing: float = 0.0  # 0..1; applied to offbeats

#     # Pitch / tuning
#     root: str = "c"
#     mode: str = "minor"  # used only if scale_intervals not provided
#     scale_intervals: Optional[List[float]] = None  # semitones from root; microtones allowed
#     tuning_ref_hz: float = 440.0  # A4

#     # Harmony policy
#     harmonic_motion: str = "minimal"  # drone|minimal|slow|medium|active
#     chord_color: str = "open5"  # unison|open5|triad|seventh|cluster (engine uses heuristics)

#     # Vibe / timbre knobs
#     brightness: str = "medium"
#     space: str = "room"
#     character: str = "clean"
#     material: str = "warm_analog"
#     stereo_width: str = "medium"
#     density: int = 4

#     # Expression knobs
#     humanize: float = 0.15  # 0..1
#     ornament_intensity: str = "subtle"

#     # Render
#     seed: int = 0
#     sample_rate: int = DEFAULT_SAMPLE_RATE
#     duration_s: float = 16.0


# @dataclass
# class LayerSpec:
#     role: str = "pad"
#     pattern: str = "auto"
#     gain: float = 1.0  # relative, later normalized
#     pan: float = 0.0  # -1..+1

#     # Optional overrides (kept small on purpose)
#     bpm: Optional[float] = None
#     cycle_beats: Optional[int] = None

#     material: Optional[str] = None
#     brightness: Optional[str] = None
#     character: Optional[str] = None
#     space: Optional[str] = None
#     ornament_intensity: Optional[str] = None


# # =============================================================================
# # Small helpers
# # =============================================================================


# def _clamp(x: float, lo: float, hi: float) -> float:
#     try:
#         xf = float(x)
#     except Exception:
#         return lo
#     if math.isnan(xf) or math.isinf(xf):
#         return lo
#     return float(min(hi, max(lo, xf)))


# def _clamp_int(x: Any, lo: int, hi: int) -> int:
#     try:
#         xi = int(x)
#     except Exception:
#         return lo
#     return int(min(hi, max(lo, xi)))


# def _safe_str(x: Any, default: str) -> str:
#     if isinstance(x, str) and x.strip():
#         return x.strip()
#     return default


# def _as_list(x: Any) -> List[Any]:
#     if x is None:
#         return []
#     if isinstance(x, list):
#         return x
#     return [x]


# def _channels_first(audio: np.ndarray) -> np.ndarray:
#     """
#     Convert audio to channels-first float32:
#     - mono: (samples,) -> (1, samples)
#     - channels-last: (samples, ch) -> (ch, samples)
#     - channels-first already: (ch, samples)
#     """
#     a = np.asarray(audio)
#     if a.ndim == 1:
#         return a.astype(np.float32)[None, :]
#     if a.ndim != 2:
#         raise ValueError(f"Expected 1D or 2D audio array, got shape {a.shape}")
#     # Heuristic: if first dim is 1 or 2 and second is "large", assume channels-first already.
#     if a.shape[0] in (1, 2) and a.shape[1] > a.shape[0]:
#         return a.astype(np.float32)
#     return a.T.astype(np.float32)


# def _channels_last(audio_cf: np.ndarray) -> np.ndarray:
#     """(ch, samples) -> (samples, ch) for writing with soundfile."""
#     a = _channels_first(audio_cf)
#     return a.T


# def _normalize_peak(audio_cf: np.ndarray, peak: float = 0.98) -> np.ndarray:
#     a = _channels_first(audio_cf)
#     m = float(np.max(np.abs(a))) if a.size else 0.0
#     if m <= 1e-9:
#         return a
#     return (a * (peak / m)).astype(np.float32)


# def _soft_clip(audio_cf: np.ndarray, drive: float = 1.0) -> np.ndarray:
#     """
#     Cheap safety limiter for fallback mode.
#     drive>1 increases saturation.
#     """
#     a = _channels_first(audio_cf)
#     d = _clamp(drive, 0.1, 10.0)
#     return np.tanh(a * d).astype(np.float32)


# def _linear_pan(mono: np.ndarray, pan: float) -> np.ndarray:
#     """
#     Mono (samples,) -> stereo (2, samples) with equal-power-ish pan.
#     pan=-1 left, +1 right.
#     """
#     p = _clamp(pan, -1.0, 1.0)
#     # equal-power pan
#     left = math.cos((p + 1.0) * math.pi / 4.0)
#     right = math.sin((p + 1.0) * math.pi / 4.0)
#     out = np.vstack([mono * left, mono * right]).astype(np.float32)
#     return out


# # =============================================================================
# # Pitch math + safety
# # =============================================================================


# def _note_to_midi(note: str, octave: int) -> int:
#     n = note.lower().strip()
#     if n not in ROOT_TO_SEMITONE:
#         n = "c"
#     sem = ROOT_TO_SEMITONE[n]
#     # MIDI: C4=60; midi = 12*(octave+1) + semitone(C)
#     return 12 * (octave + 1) + sem


# def _freq_hz(root: str, semitone_offset: float, octave: int, a4_hz: float) -> float:
#     midi = _note_to_midi(root, octave)
#     # A4 MIDI 69
#     return float(a4_hz * (2.0 ** (((midi - 69) + float(semitone_offset)) / 12.0)))


# def _sanitize_scale_intervals(
#     scale_intervals: Optional[Sequence[Any]],
#     mode: str,
# ) -> List[float]:
#     """
#     Return a safe, sorted, unique set of intervals in [0, 12) with 0 included.

#     Self-healing behaviors:
#     - If intervals missing/invalid -> use MODE_INTERVALS[mode] (fallback to minor)
#     - If intervals contain garbage -> drop invalid entries
#     - If first element isn't 0 -> prepend 0
#     - If < 3 degrees -> fallback to pentatonic_minor
#     - If values outside [0, 24] -> wrap mod 12 (microtones preserved)
#     """
#     if not scale_intervals:
#         m = (mode or "minor").lower().strip()
#         base = MODE_INTERVALS.get(m, MODE_INTERVALS["minor"])
#         return [float(x) for x in base]

#     vals: List[float] = []
#     for x in scale_intervals:
#         try:
#             xf = float(x)
#         except Exception:
#             continue
#         if math.isnan(xf) or math.isinf(xf):
#             continue
#         # keep microtones, but constrain overall
#         # Wrap to one octave to improve harmonic safety (prevents "random 37 semitones").
#         xf = xf % 12.0
#         # Drop super-close duplicates later
#         vals.append(xf)

#     if not vals:
#         return _sanitize_scale_intervals(None, mode)

#     # Ensure 0 is present
#     vals.append(0.0)

#     # Sort and de-dup with a tolerance for microtones
#     vals = sorted(vals)
#     dedup: List[float] = []
#     for v in vals:
#         if not dedup or abs(v - dedup[-1]) > 0.05:
#             dedup.append(v)

#     # If too small, fallback
#     if len(dedup) < 3:
#         return [float(x) for x in MODE_INTERVALS["pentatonic_minor"]]

#     # Cap size (LLM might emit crazy long lists)
#     return dedup[:24]


# def _nearest_index(values: Sequence[float], target: float) -> int:
#     best_i = 0
#     best_d = float("inf")
#     for i, v in enumerate(values):
#         d = abs(float(v) - float(target))
#         if d < best_d:
#             best_d = d
#             best_i = i
#     return best_i


# def _quantize_to_scale(semitone: float, scale: Sequence[float]) -> float:
#     """Map any semitone offset to nearest allowed scale degree (within octave)."""
#     if not scale:
#         return float(semitone) % 12.0
#     t = float(semitone) % 12.0
#     i = _nearest_index(scale, t)
#     return float(scale[i])


# def _pick_open_fifth_or_octave(scale: Sequence[float]) -> float:
#     """
#     Pick an interval from the scale closest to a perfect fifth (7 semitones).
#     If none are close, return octave (12).
#     """
#     if not scale:
#         return 12.0
#     target = 7.0
#     i = _nearest_index(scale, target)
#     v = float(scale[i])
#     if abs(v - target) <= 1.0:  # within ~1 semitone
#         return v
#     return 12.0


# # =============================================================================
# # Low-level synthesis (numpy)
# # =============================================================================


# def _phase_from_freq(freq_hz: np.ndarray, sr: int) -> np.ndarray:
#     """Integrate frequency to phase."""
#     # phase increment per sample
#     inc = (2.0 * math.pi * freq_hz) / float(sr)
#     return np.cumsum(inc, dtype=np.float64)


# def _osc_from_phase(phase: np.ndarray, kind: str) -> np.ndarray:
#     kind_l = (kind or "sine").lower()
#     if kind_l == "sine":
#         return np.sin(phase)
#     if kind_l == "square":
#         return np.sign(np.sin(phase))
#     if kind_l == "saw":
#         # sawtooth from phase
#         x = (phase / (2.0 * math.pi)) % 1.0
#         return 2.0 * x - 1.0
#     if kind_l == "triangle":
#         x = (phase / (2.0 * math.pi)) % 1.0
#         return 2.0 * np.abs(2.0 * x - 1.0) - 1.0
#     # fallback
#     return np.sin(phase)


# def _adsr_env(
#     n: int,
#     sr: int,
#     attack_s: float,
#     decay_s: float,
#     sustain: float,
#     release_s: float,
# ) -> np.ndarray:
#     """
#     Basic ADSR envelope, always safe:
#     - clamps times
#     - ensures envelope length = n
#     """
#     a = max(0, int(sr * _clamp(attack_s, 0.0, 5.0)))
#     d = max(0, int(sr * _clamp(decay_s, 0.0, 5.0)))
#     r = max(0, int(sr * _clamp(release_s, 0.0, 10.0)))
#     s = _clamp(sustain, 0.0, 1.0)

#     # Ensure we have room for sustain
#     core = a + d + r
#     if core >= n:
#         # Just do an A/R-ish envelope
#         a2 = max(1, n // 10)
#         r2 = max(1, n // 5)
#         mid = max(0, n - (a2 + r2))
#         env = np.concatenate(
#             [
#                 np.linspace(0, 1, a2, endpoint=False),
#                 np.ones(mid),
#                 np.linspace(1, 0, r2, endpoint=True),
#             ]
#         )
#         return env.astype(np.float32)[:n]

#     sustain_len = n - core
#     env_a = np.linspace(0, 1, a, endpoint=False) if a > 0 else np.zeros(0)
#     env_d = np.linspace(1, s, d, endpoint=False) if d > 0 else np.zeros(0)
#     env_s = np.full(sustain_len, s, dtype=np.float32)
#     env_r = np.linspace(s, 0, r, endpoint=True) if r > 0 else np.zeros(0)

#     env = np.concatenate([env_a, env_d, env_s, env_r]).astype(np.float32)
#     if env.size < n:
#         env = np.pad(env, (0, n - env.size), mode="edge")
#     return env[:n]


# def _lowpass_onepole(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
#     """
#     Simple one-pole lowpass for fallback mode (not used when pedalboard is available).
#     """
#     c = _clamp(cutoff_hz, 30.0, sr * 0.45)
#     # RC filter
#     rc = 1.0 / (2.0 * math.pi * c)
#     dt = 1.0 / float(sr)
#     alpha = dt / (rc + dt)
#     y = np.empty_like(x, dtype=np.float32)
#     y0 = 0.0
#     for i, xi in enumerate(x.astype(np.float32)):
#         y0 = y0 + alpha * (float(xi) - y0)
#         y[i] = y0
#     return y


# def _simple_delay(x: np.ndarray, sr: int, delay_s: float, feedback: float, mix: float) -> np.ndarray:
#     d = int(sr * _clamp(delay_s, 0.01, 1.5))
#     fb = _clamp(feedback, 0.0, 0.95)
#     mx = _clamp(mix, 0.0, 1.0)
#     if d <= 0:
#         return x.astype(np.float32)
#     y = x.astype(np.float32).copy()
#     for i in range(d, len(y)):
#         y[i] += y[i - d] * fb
#     return (x * (1 - mx) + y * mx).astype(np.float32)


# # =============================================================================
# # Pattern generation (safe, culture-general primitives)
# # =============================================================================


# def _resolve_global(raw: Dict[str, Any]) -> GlobalSpec:
#     g = GlobalSpec()

#     g.bpm = _clamp(raw.get("bpm", raw.get("tempo_bpm", g.bpm)), 30.0, 220.0)
#     g.cycle_beats = _clamp_int(raw.get("cycle_beats", raw.get("beats_per_cycle", g.cycle_beats)), 2, 64)
#     g.subdivision = _clamp_int(raw.get("subdivision", g.subdivision), 1, 8)
#     if g.subdivision not in (2, 3, 4, 5):
#         # coerce into a "musical" subdivision
#         g.subdivision = int(min((2, 3, 4, 5), key=lambda v: abs(v - g.subdivision)))

#     g.swing = _clamp(raw.get("swing", g.swing), 0.0, 0.75)

#     g.root = _safe_str(raw.get("root", g.root), g.root).lower()
#     if g.root not in ROOT_TO_SEMITONE:
#         g.root = "c"

#     g.mode = _safe_str(raw.get("mode", g.mode), g.mode).lower()
#     g.scale_intervals = raw.get("scale_intervals", raw.get("scale", None))
#     g.tuning_ref_hz = _clamp(raw.get("tuning_ref_hz", raw.get("a4_hz", g.tuning_ref_hz)), 400.0, 480.0)

#     g.harmonic_motion = _safe_str(raw.get("harmonic_motion", g.harmonic_motion), g.harmonic_motion).lower()
#     if g.harmonic_motion not in ("drone", "minimal", "slow", "medium", "active"):
#         g.harmonic_motion = "minimal"

#     g.chord_color = _safe_str(raw.get("chord_color", g.chord_color), g.chord_color).lower()
#     if g.chord_color not in ("unison", "open5", "triad", "seventh", "cluster"):
#         g.chord_color = "open5"

#     g.brightness = _safe_str(raw.get("brightness", g.brightness), g.brightness).lower()
#     if g.brightness not in ("very_dark", "dark", "medium", "bright", "very_bright"):
#         g.brightness = "medium"

#     g.space = _safe_str(raw.get("space", g.space), g.space).lower()
#     if g.space not in ("dry", "intimate", "room", "hall", "cathedral", "infinite"):
#         g.space = "room"

#     g.character = _safe_str(raw.get("character", g.character), g.character).lower()
#     if g.character not in ("clean", "warm", "saturated", "distorted"):
#         g.character = "clean"

#     g.material = _safe_str(raw.get("material", g.material), g.material).lower()
#     if g.material not in ("digital", "warm_analog", "wood", "metal", "glass", "breath", "skin"):
#         g.material = "warm_analog"

#     g.stereo_width = _safe_str(raw.get("stereo_width", g.stereo_width), g.stereo_width).lower()
#     if g.stereo_width not in ("mono", "narrow", "medium", "wide", "immersive"):
#         g.stereo_width = "medium"

#     g.density = _clamp_int(raw.get("density", g.density), 1, 6)
#     g.humanize = _clamp(raw.get("humanize", g.humanize), 0.0, 1.0)

#     g.ornament_intensity = _safe_str(raw.get("ornament_intensity", raw.get("ornament", g.ornament_intensity)), g.ornament_intensity).lower()
#     if g.ornament_intensity not in ("none", "subtle", "moderate", "expressive", "virtuosic"):
#         g.ornament_intensity = "subtle"

#     g.seed = _clamp_int(raw.get("seed", g.seed), 0, 2_000_000_000)
#     g.sample_rate = _clamp_int(raw.get("sample_rate", g.sample_rate), 8_000, 192_000)
#     g.duration_s = _clamp(raw.get("duration_s", raw.get("duration", g.duration_s)), 2.0, 300.0)

#     return g


# def _default_layers_for_density(density: int) -> List[LayerSpec]:
#     # Always coherent and "pleasantly generic".
#     layers: List[LayerSpec] = [
#         LayerSpec(role="pad", pattern="drone", gain=0.9, pan=0.0),
#     ]
#     if density >= 2:
#         layers.append(LayerSpec(role="bass", pattern="drone", gain=0.65, pan=0.0))
#     if density >= 3:
#         layers.append(LayerSpec(role="lead", pattern="melody", gain=0.8, pan=0.1))
#     if density >= 4:
#         layers.append(LayerSpec(role="rhythm", pattern="pulse", gain=0.7, pan=0.0))
#     if density >= 5:
#         layers.append(LayerSpec(role="texture", pattern="air", gain=0.4, pan=-0.2))
#     if density >= 6:
#         layers.append(LayerSpec(role="accent", pattern="sparkle", gain=0.35, pan=0.2))
#     return layers


# def _resolve_layers(raw: Dict[str, Any], g: GlobalSpec) -> List[LayerSpec]:
#     """
#     Accepted input shapes:
#     - {"layers": [{"role": ..., "gain": ...}, ...]}
#     - {"layers": {"role": {...}, "role2": {...}}}  (LLM-friendly alt)
#     - missing layers -> auto layers based on density
#     """
#     layers_raw = raw.get("layers", None)

#     layers: List[LayerSpec] = []
#     if layers_raw is None:
#         layers = _default_layers_for_density(g.density)
#     elif isinstance(layers_raw, list):
#         for item in layers_raw:
#             if not isinstance(item, dict):
#                 continue
#             layers.append(_layer_from_dict(item))
#     elif isinstance(layers_raw, dict):
#         for role, item in layers_raw.items():
#             if isinstance(item, dict):
#                 d = dict(item)
#                 d.setdefault("role", role)
#                 layers.append(_layer_from_dict(d))

#     if not layers:
#         layers = _default_layers_for_density(g.density)

#     # Ensure roles aren't duplicated too much (keeps mix clear)
#     seen: Dict[str, int] = {}
#     for ly in layers:
#         r = ly.role
#         seen[r] = seen.get(r, 0) + 1
#         if seen[r] > 2:
#             ly.role = "texture"
#             ly.pattern = "air"

#     # Normalize gains safely
#     gains = np.array([max(0.0, float(l.gain)) for l in layers], dtype=np.float32)
#     if gains.size == 0:
#         return _default_layers_for_density(g.density)
#     if float(gains.sum()) <= 1e-9:
#         gains[:] = 1.0
#     gains = gains / float(gains.sum())
#     for l, gn in zip(layers, gains):
#         l.gain = float(gn)

#     return layers


# def _layer_from_dict(d: Dict[str, Any]) -> LayerSpec:
#     l = LayerSpec()
#     l.role = _safe_str(d.get("role", l.role), l.role).lower()
#     if l.role not in ("bass", "pad", "lead", "rhythm", "texture", "voice", "accent"):
#         l.role = "texture"

#     l.pattern = _safe_str(d.get("pattern", d.get("style", l.pattern)), l.pattern).lower()

#     # Safe numeric overrides
#     l.gain = float(_clamp(d.get("gain", l.gain), 0.0, 2.0))
#     l.pan = float(_clamp(d.get("pan", l.pan), -1.0, 1.0))

#     l.bpm = d.get("bpm", d.get("tempo_bpm", None))
#     if l.bpm is not None:
#         l.bpm = float(_clamp(l.bpm, 20.0, 260.0))

#     l.cycle_beats = d.get("cycle_beats", d.get("beats_per_cycle", None))
#     if l.cycle_beats is not None:
#         l.cycle_beats = int(_clamp_int(l.cycle_beats, 2, 64))

#     # Optional vibe overrides
#     for key in ("material", "brightness", "character", "space", "ornament_intensity"):
#         if key in d and isinstance(d[key], str):
#             setattr(l, key, d[key].lower().strip())

#     return l


# # =============================================================================
# # Musical event synthesis helpers
# # =============================================================================


# def _sine_tone(freq: np.ndarray, sr: int, kind: str) -> np.ndarray:
#     phase = _phase_from_freq(freq.astype(np.float64), sr)
#     wave = _osc_from_phase(phase, kind=kind).astype(np.float32)
#     return wave


# def _vibrato_curve(
#     base_freq: float,
#     n: int,
#     sr: int,
#     depth_cents: float,
#     rate_hz: float,
# ) -> np.ndarray:
#     """
#     Returns instantaneous frequency curve with vibrato.
#     depth_cents is peak deviation in cents.
#     """
#     depth = _clamp(depth_cents, 0.0, 200.0)
#     rate = _clamp(rate_hz, 0.1, 12.0)
#     t = np.arange(n, dtype=np.float32) / float(sr)
#     cents = depth * np.sin(2.0 * math.pi * rate * t)
#     ratio = 2.0 ** (cents / 1200.0)
#     return (base_freq * ratio).astype(np.float32)


# def _glide_curve(
#     f0: float,
#     f1: float,
#     n: int,
#     curve: str = "exp",
# ) -> np.ndarray:
#     if n <= 1:
#         return np.array([float(f1)], dtype=np.float32)
#     if curve == "linear":
#         return np.linspace(f0, f1, n, dtype=np.float32)
#     # exponential-ish (more natural portamento)
#     t = np.linspace(0.0, 1.0, n, dtype=np.float32)
#     a = 8.0
#     w = (1.0 - np.exp(-a * t)) / (1.0 - np.exp(-a))
#     return (f0 + (f1 - f0) * w).astype(np.float32)


# def _make_note(
#     freq_hz: float,
#     dur_s: float,
#     sr: int,
#     kind: str,
#     attack_s: float,
#     release_s: float,
#     vibrato: Tuple[float, float] = (0.0, 0.0),  # (depth_cents, rate_hz)
# ) -> np.ndarray:
#     n = max(1, int(sr * max(0.01, float(dur_s))))
#     base = np.full(n, float(freq_hz), dtype=np.float32)
#     if vibrato[0] > 0.0 and vibrato[1] > 0.0:
#         base = _vibrato_curve(freq_hz, n, sr, vibrato[0], vibrato[1])
#     wave = _sine_tone(base, sr, kind=kind)

#     # ADSR-ish: quick decay -> sustain, then release
#     env = _adsr_env(n, sr, attack_s=attack_s, decay_s=attack_s * 0.5, sustain=0.85, release_s=release_s)
#     return (wave * env).astype(np.float32)


# def _make_glide_note(
#     f0: float,
#     f1: float,
#     dur_s: float,
#     sr: int,
#     kind: str,
#     attack_s: float,
#     release_s: float,
#     vibrato: Tuple[float, float] = (0.0, 0.0),
# ) -> np.ndarray:
#     n = max(1, int(sr * max(0.01, float(dur_s))))
#     freq = _glide_curve(f0, f1, n, curve="exp")
#     if vibrato[0] > 0.0 and vibrato[1] > 0.0:
#         vib = _vibrato_curve(1.0, n, sr, vibrato[0], vibrato[1])  # ratio
#         freq = (freq * vib).astype(np.float32)

#     wave = _sine_tone(freq, sr, kind=kind)
#     env = _adsr_env(n, sr, attack_s=attack_s, decay_s=attack_s * 0.5, sustain=0.8, release_s=release_s)
#     return (wave * env).astype(np.float32)


# def _db_to_amp(db: float) -> float:
#     return float(10.0 ** (float(db) / 20.0))


# def _brightness_to_lp_cutoff(brightness: str, sr: int) -> float:
#     b = (brightness or "medium").lower()
#     mapping = {
#         "very_dark": 800.0,
#         "dark": 1800.0,
#         "medium": 4000.0,
#         "bright": 9000.0,
#         "very_bright": 16000.0,
#     }
#     return float(min(mapping.get(b, 4000.0), sr * 0.45))


# def _space_to_reverb(space: str) -> Tuple[float, float]:
#     """
#     Returns (room_size, wet_level) in 0..1.
#     """
#     s = (space or "room").lower()
#     mapping = {
#         "dry": (0.05, 0.05),
#         "intimate": (0.12, 0.12),
#         "room": (0.25, 0.22),
#         "hall": (0.55, 0.35),
#         "cathedral": (0.80, 0.50),
#         "infinite": (0.95, 0.70),
#     }
#     return mapping.get(s, (0.25, 0.22))


# def _stereo_width_to_chorus_mix(width: str) -> float:
#     w = (width or "medium").lower()
#     mapping = {
#         "mono": 0.0,
#         "narrow": 0.15,
#         "medium": 0.25,
#         "wide": 0.35,
#         "immersive": 0.45,
#     }
#     return float(mapping.get(w, 0.25))


# def _ornament_to_params(orn: str) -> Tuple[float, float, float]:
#     """
#     Map ornament_intensity to (vibrato_depth_cents, vibrato_rate_hz, glide_prob).
#     """
#     o = (orn or "subtle").lower()
#     if o == "none":
#         return (0.0, 0.0, 0.0)
#     if o == "subtle":
#         return (10.0, 4.5, 0.10)
#     if o == "moderate":
#         return (25.0, 5.5, 0.25)
#     if o == "expressive":
#         return (45.0, 6.0, 0.45)
#     if o == "virtuosic":
#         return (70.0, 6.5, 0.70)
#     return (10.0, 4.5, 0.10)


# def _material_to_osc(material: str, role: str) -> str:
#     """
#     Very crude mapping, but stable.
#     """
#     m = (material or "warm_analog").lower()
#     r = (role or "pad").lower()

#     if r in ("rhythm",):
#         return "sine"

#     if m == "digital":
#         return "saw"
#     if m == "warm_analog":
#         return "triangle" if r in ("pad", "bass") else "saw"
#     if m == "wood":
#         return "triangle"
#     if m == "metal":
#         return "saw"
#     if m == "glass":
#         return "sine"
#     if m == "breath":
#         return "sine"
#     if m == "skin":
#         return "sine"
#     return "sine"


# # =============================================================================
# # Layer generators
# # =============================================================================


# def _layer_seconds_per_beat(global_bpm: float, layer_bpm: Optional[float]) -> float:
#     gbpm = _clamp(global_bpm, 30.0, 220.0)
#     if layer_bpm is None:
#         return 60.0 / gbpm

#     # Quantize requested layer bpm into a safe small rational multiple of global bpm
#     ratio = _clamp(layer_bpm / gbpm, 0.25, 4.0)
#     best = (1, 1)
#     best_err = float("inf")
#     for p, q in SAFE_TEMPO_RATIOS:
#         r = p / q
#         err = abs(ratio - r)
#         if err < best_err:
#             best_err = err
#             best = (p, q)

#     # apply quantized ratio
#     effective = gbpm * (best[0] / best[1])
#     return 60.0 / effective


# def _gen_bass(
#     g: GlobalSpec,
#     layer: LayerSpec,
#     scale: List[float],
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     sr = g.sample_rate
#     n = int(g.duration_s * sr)
#     out = np.zeros(n, dtype=np.float32)

#     spb = _layer_seconds_per_beat(g.bpm, layer.bpm)
#     cycle_beats = layer.cycle_beats or g.cycle_beats
#     cycle_s = cycle_beats * spb

#     osc_kind = _material_to_osc(layer.material or g.material, "bass")
#     lp_cutoff = _brightness_to_lp_cutoff(layer.brightness or g.brightness, sr) * 0.35

#     # Bass policy: root + maybe fifth
#     fifth = _pick_open_fifth_or_octave(scale)
#     degrees = [0.0] if fifth == 12.0 else [0.0, fifth]

#     # Pattern decisions
#     pat = (layer.pattern or "auto").lower()
#     if pat in ("auto", "drone"):
#         # sustained
#         f = _freq_hz(g.root, 0.0, octave=2, a4_hz=g.tuning_ref_hz)
#         note = _make_note(f, g.duration_s, sr, kind=osc_kind, attack_s=0.02, release_s=0.4)
#         out += note * 0.35
#         if len(degrees) > 1 and degrees[1] != 12.0:
#             f2 = _freq_hz(g.root, degrees[1], octave=2, a4_hz=g.tuning_ref_hz)
#             note2 = _make_note(f2, g.duration_s, sr, kind=osc_kind, attack_s=0.02, release_s=0.4)
#             out += note2 * 0.18

#     else:
#         # pulsing / ostinato
#         beat = 0.0
#         while beat * spb < g.duration_s:
#             start_s = beat * spb
#             dur_s = spb * (1.0 if pat in ("pulse", "pulsing") else 0.5)
#             idx0 = int(start_s * sr)
#             idx1 = min(n, idx0 + int(dur_s * sr))

#             deg = float(rng.choice(degrees))
#             f = _freq_hz(g.root, deg, octave=2, a4_hz=g.tuning_ref_hz)
#             note = _make_note(f, (idx1 - idx0) / sr, sr, kind=osc_kind, attack_s=0.005, release_s=0.12)
#             out[idx0:idx0 + note.size] += note * 0.55
#             beat += 1.0 if pat in ("pulse", "pulsing") else 0.5

#     if PEDALBOARD_AVAILABLE:
#         board = Pedalboard(
#             [
#                 LowpassFilter(cutoff_frequency_hz=lp_cutoff),
#                 Compressor(threshold_db=-22, ratio=3.0, attack_ms=8, release_ms=120),
#             ]
#         )
#         mono_cf = _channels_first(out)
#         effected = board(mono_cf, sr)
#         return cast(np.ndarray, effected)[0].astype(np.float32)

#     # Fallback
#     out = _lowpass_onepole(out, lp_cutoff, sr)
#     return out.astype(np.float32)


# def _gen_pad(
#     g: GlobalSpec,
#     layer: LayerSpec,
#     scale: List[float],
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     sr = g.sample_rate
#     n = int(g.duration_s * sr)
#     out = np.zeros(n, dtype=np.float32)

#     spb = _layer_seconds_per_beat(g.bpm, layer.bpm)
#     cycle_beats = layer.cycle_beats or g.cycle_beats
#     cycle_s = cycle_beats * spb

#     osc_kind = _material_to_osc(layer.material or g.material, "pad")
#     lp_cutoff = _brightness_to_lp_cutoff(layer.brightness or g.brightness, sr)

#     # Harmony safety: in "drone"/"minimal" or microtonal-ish scales, avoid busy chord progressions.
#     microtonal = any(abs(x - round(x)) > 1e-6 for x in scale)
#     hm = (g.harmonic_motion or "minimal").lower()
#     safe_drone = (hm in ("drone", "minimal")) or microtonal

#     # Choose chord degrees
#     fifth = _pick_open_fifth_or_octave(scale)
#     chord_degs: List[float] = [0.0]
#     if g.chord_color in ("open5", "triad", "seventh", "cluster"):
#         chord_degs.append(fifth if fifth != 12.0 else 0.0)
#     chord_degs.append(12.0)  # octave always safe

#     # If triad-ish, try add a "third" by picking a scale step near 3 or 4 semitones.
#     if g.chord_color in ("triad", "seventh", "cluster") and len(scale) >= 4:
#         third = float(scale[_nearest_index(scale, 3.5)])
#         if third not in chord_degs:
#             chord_degs.append(third)

#     chord_degs = list(dict.fromkeys(chord_degs))  # preserve order, dedup

#     def chord_for_degree(shift: float) -> List[float]:
#         return [d + shift for d in chord_degs]

#     # Determine chord timeline
#     if safe_drone:
#         changes = [(0.0, chord_for_degree(0.0))]
#     else:
#         # Heuristic chord changes: based on harmonic_motion
#         bars = max(1, int(round(g.duration_s / (4.0 * spb))))
#         if hm == "slow":
#             change_every = 8
#         elif hm == "medium":
#             change_every = 4
#         else:
#             change_every = 2

#         # degree shifts within scale (0, 5, 3, 4 ...), but scaled to our scale indices.
#         prog = [0, 4, 5, 3] if "major" in (g.mode or "") else [0, 5, 3, 6]
#         # map into available scale degrees
#         prog = [p % max(1, len(scale)) for p in prog]
#         shifts = [float(scale[p]) for p in prog]
#         changes = []
#         for b in range(0, bars, change_every):
#             t0 = b * 4.0 * spb
#             shift = shifts[(b // change_every) % len(shifts)]
#             changes.append((t0, chord_for_degree(shift)))

#     # Render chords as sustained tones with slow attacks
#     for i, (t0, degs) in enumerate(changes):
#         t1 = changes[i + 1][0] if i + 1 < len(changes) else g.duration_s
#         idx0 = int(t0 * sr)
#         idx1 = int(t1 * sr)
#         dur = max(1, idx1 - idx0) / sr

#         for deg in degs:
#             if deg >= 12.0:
#                 # octave above
#                 f = _freq_hz(g.root, deg % 12.0, octave=4, a4_hz=g.tuning_ref_hz) * 2.0
#             else:
#                 f = _freq_hz(g.root, deg, octave=4, a4_hz=g.tuning_ref_hz)

#             note = _make_note(f, dur, sr, kind=osc_kind, attack_s=0.35, release_s=0.8)
#             out[idx0:idx0 + note.size] += note * (0.18 / max(1, len(degs)))

#     # Effects / coloration
#     space = layer.space or g.space
#     room_size, wet = _space_to_reverb(space)
#     chorus_mix = _stereo_width_to_chorus_mix(g.stereo_width)

#     if PEDALBOARD_AVAILABLE:
#         # Use LadderFilter for a more "synth" lowpass.
#         board = Pedalboard(
#             [
#                 LadderFilter(mode=LadderFilter.Mode.LPF12, cutoff_hz=lp_cutoff, resonance=0.2),
#                 Chorus(rate_hz=0.25, depth=chorus_mix, mix=chorus_mix),
#                 Reverb(room_size=room_size, wet_level=wet, dry_level=1.0 - wet),
#             ]
#         )
#         mono_cf = _channels_first(out)
#         effected = board(mono_cf, sr)
#         return cast(np.ndarray, effected)[0].astype(np.float32)

#     # Fallback: simple lowpass + delay-ish smear
#     out = _lowpass_onepole(out, lp_cutoff, sr)
#     if wet > 0.1:
#         out = _simple_delay(out, sr, delay_s=0.18 + room_size * 0.25, feedback=0.35 + 0.3 * room_size, mix=wet * 0.6)
#     return out.astype(np.float32)


# def _gen_melody_like(
#     g: GlobalSpec,
#     layer: LayerSpec,
#     scale: List[float],
#     rng: np.random.Generator,
#     voice_like: bool = False,
# ) -> np.ndarray:
#     sr = g.sample_rate
#     n = int(g.duration_s * sr)
#     out = np.zeros(n, dtype=np.float32)

#     spb = _layer_seconds_per_beat(g.bpm, layer.bpm)
#     cycle_beats = layer.cycle_beats or g.cycle_beats

#     osc_kind = _material_to_osc(layer.material or g.material, "voice" if voice_like else "lead")
#     lp_cutoff = _brightness_to_lp_cutoff(layer.brightness or g.brightness, sr)

#     orn = layer.ornament_intensity or g.ornament_intensity
#     vib_cents, vib_rate, glide_prob = _ornament_to_params(orn)

#     # Melody density heuristic:
#     # - In drone/minimal -> sparser, longer notes
#     hm = (g.harmonic_motion or "minimal").lower()
#     if hm in ("drone", "minimal"):
#         note_every_beats = 2.0
#         note_len_beats = 1.75
#     elif hm == "slow":
#         note_every_beats = 1.0
#         note_len_beats = 0.9
#     elif hm == "medium":
#         note_every_beats = 0.5
#         note_len_beats = 0.45
#     else:  # active
#         note_every_beats = 0.25
#         note_len_beats = 0.20

#     # Voice/chant tends to be more legato and ornamented
#     if voice_like:
#         note_every_beats *= 1.15
#         note_len_beats *= 1.35
#         glide_prob = min(0.9, glide_prob + 0.2)

#     # Register safety: keep within a musically pleasant range
#     base_oct = 5 if voice_like else 5
#     if layer.role == "lead":
#         base_oct = 5
#     if layer.role == "voice":
#         base_oct = 5

#     # Choose degrees with strong bias toward tonic
#     weights = np.ones(len(scale), dtype=np.float32)
#     weights[0] = 3.0
#     if len(scale) > 4:
#         weights[4 % len(scale)] += 0.8
#     weights = weights / float(weights.sum())

#     prev_deg = 0.0
#     beat = 0.0
#     while beat * spb < g.duration_s:
#         # Swing: shift offbeats later in time
#         t0 = beat * spb
#         if g.swing > 0 and (int(beat * 2) % 2 == 1):
#             t0 += (g.swing * 0.35) * spb

#         dur_s = max(0.05, note_len_beats * spb)
#         idx0 = int(t0 * sr)
#         if idx0 >= n:
#             break

#         # stepwise bias: pick nearby degrees
#         step = rng.choice([-2, -1, 0, 1, 2], p=[0.05, 0.20, 0.50, 0.20, 0.05])
#         # occasional leaps if "active"
#         if hm == "active" and rng.random() < 0.25:
#             step = int(rng.integers(-4, 5))
#         prev_i = _nearest_index(scale, prev_deg)
#         new_i = int((prev_i + step) % len(scale))
#         deg = float(scale[new_i])

#         # Optional drift toward tonic at phrase boundaries
#         if (beat % cycle_beats) < 1e-6 and rng.random() < 0.6:
#             deg = 0.0

#         f1 = _freq_hz(g.root, deg, octave=base_oct, a4_hz=g.tuning_ref_hz)

#         # decide glide from prev note
#         do_glide = (rng.random() < glide_prob) and (beat > 0)
#         if do_glide:
#             f0 = _freq_hz(g.root, prev_deg, octave=base_oct, a4_hz=g.tuning_ref_hz)
#             note = _make_glide_note(
#                 f0,
#                 f1,
#                 dur_s,
#                 sr,
#                 kind=osc_kind,
#                 attack_s=0.01 if not voice_like else 0.02,
#                 release_s=0.08 if not voice_like else 0.20,
#                 vibrato=(vib_cents, vib_rate),
#             )
#         else:
#             note = _make_note(
#                 f1,
#                 dur_s,
#                 sr,
#                 kind=osc_kind,
#                 attack_s=0.008 if not voice_like else 0.02,
#                 release_s=0.06 if not voice_like else 0.15,
#                 vibrato=(vib_cents, vib_rate),
#             )

#         # Amplitude + humanize
#         vel = 0.42 if voice_like else 0.35
#         vel *= float(0.85 + 0.3 * rng.random())
#         if g.humanize > 0:
#             jitter = (rng.random() - 0.5) * g.humanize * 0.06 * spb
#             idx0 = int((t0 + jitter) * sr)
#             idx0 = max(0, min(n - 1, idx0))

#         idx1 = min(n, idx0 + note.size)
#         out[idx0:idx1] += note[: idx1 - idx0] * vel

#         prev_deg = deg
#         beat += note_every_beats

#     if PEDALBOARD_AVAILABLE:
#         space = layer.space or g.space
#         room_size, wet = _space_to_reverb(space)
#         chorus_mix = _stereo_width_to_chorus_mix(g.stereo_width)
#         board = Pedalboard(
#             [
#                 LowpassFilter(cutoff_frequency_hz=lp_cutoff),
#                 Chorus(rate_hz=0.45, depth=chorus_mix * 0.8, mix=chorus_mix * 0.6),
#                 Delay(delay_seconds=0.18 if not voice_like else 0.10, feedback=0.22, mix=0.16 if not voice_like else 0.08),
#                 Reverb(room_size=room_size, wet_level=wet, dry_level=1.0 - wet),
#             ]
#         )
#         mono_cf = _channels_first(out)
#         effected = board(mono_cf, sr)
#         return cast(np.ndarray, effected)[0].astype(np.float32)

#     # Fallback
#     out = _lowpass_onepole(out, lp_cutoff, sr)
#     out = _simple_delay(out, sr, delay_s=0.16, feedback=0.25, mix=0.10)
#     return out.astype(np.float32)


# def _gen_rhythm(
#     g: GlobalSpec,
#     layer: LayerSpec,
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     sr = g.sample_rate
#     n = int(g.duration_s * sr)
#     out = np.zeros(n, dtype=np.float32)

#     spb = _layer_seconds_per_beat(g.bpm, layer.bpm)
#     cycle_beats = layer.cycle_beats or g.cycle_beats

#     pat = (layer.pattern or "pulse").lower()

#     # Onset grid in "subdivision" pulses
#     pulses_per_beat = g.subdivision
#     pulses_per_cycle = int(cycle_beats * pulses_per_beat)
#     if pulses_per_cycle <= 0:
#         pulses_per_cycle = 16

#     # Build a safe onset pattern:
#     # - pulse: simple kick on beat
#     # - groove: kick+snare+hat
#     # - polyrhythm: a second accent cycle
#     onsets_kick: List[int] = []
#     onsets_snare: List[int] = []
#     onsets_hat: List[int] = []

#     if pat in ("auto", "pulse", "heartbeat"):
#         for b in range(int(cycle_beats)):
#             onsets_kick.append(b * pulses_per_beat)
#         # occasional hat
#         for b in range(int(cycle_beats)):
#             if b % 2 == 1:
#                 onsets_hat.append(b * pulses_per_beat)
#     elif pat in ("four_floor", "driving"):
#         for b in range(int(cycle_beats)):
#             onsets_kick.append(b * pulses_per_beat)
#         for b in range(int(cycle_beats)):
#             onsets_hat.append(b * pulses_per_beat + pulses_per_beat // 2)
#         for b in range(int(cycle_beats)):
#             if b % 4 == 2:
#                 onsets_snare.append(b * pulses_per_beat)
#     elif pat in ("clave", "world", "syncopated"):
#         # Generic "clave-ish": asymmetric 5-hit cell across 2 bars (8 beats) if possible
#         base = [0, 3, 4, 6, 7]  # in 8 pulses (assuming subdivision=2)
#         cell_len = 8 * max(1, pulses_per_beat // 2)  # scale with subdivision
#         # map into pulses
#         for x in base:
#             onsets_snare.append(int(x * (pulses_per_beat / 2)))
#         # kick on 1 and 5
#         onsets_kick.extend([0, int(4 * pulses_per_beat)])
#         # hats in between
#         for p in range(0, cell_len, max(1, pulses_per_beat // 2)):
#             if p not in onsets_snare:
#                 onsets_hat.append(p)
#     elif pat in ("polyrhythm",):
#         # 3:2 accent relationship
#         # kick on 0 and halfway; hats do 3 evenly spaced
#         onsets_kick = [0, int(pulses_per_cycle / 2)]
#         onsets_hat = [int(i * pulses_per_cycle / 3) for i in range(3)]
#         onsets_snare = [int(pulses_per_cycle / 4), int(3 * pulses_per_cycle / 4)]
#     else:
#         # Unknown -> fallback
#         for b in range(int(cycle_beats)):
#             onsets_kick.append(b * pulses_per_beat)

#     def add_kick(t_s: float):
#         idx0 = int(t_s * sr)
#         if idx0 >= n:
#             return
#         dur = 0.18
#         nn = int(dur * sr)
#         t = np.arange(nn, dtype=np.float32) / float(sr)
#         # pitch drop
#         f0, f1 = 110.0, 45.0
#         f = f1 + (f0 - f1) * np.exp(-t * 18.0)
#         wave = np.sin(2.0 * math.pi * f * t)
#         env = np.exp(-t * 16.0)
#         kick = (wave * env).astype(np.float32)
#         idx1 = min(n, idx0 + kick.size)
#         out[idx0:idx1] += kick[: idx1 - idx0] * 0.9

#     def add_snare(t_s: float):
#         idx0 = int(t_s * sr)
#         if idx0 >= n:
#             return
#         dur = 0.14
#         nn = int(dur * sr)
#         noise = rng.standard_normal(nn).astype(np.float32)
#         t = np.arange(nn, dtype=np.float32) / float(sr)
#         env = np.exp(-t * 22.0)
#         sn = (noise * env).astype(np.float32)
#         idx1 = min(n, idx0 + sn.size)
#         out[idx0:idx1] += sn[: idx1 - idx0] * 0.25

#     def add_hat(t_s: float):
#         idx0 = int(t_s * sr)
#         if idx0 >= n:
#             return
#         dur = 0.05
#         nn = int(dur * sr)
#         noise = rng.standard_normal(nn).astype(np.float32)
#         t = np.arange(nn, dtype=np.float32) / float(sr)
#         env = np.exp(-t * 60.0)
#         hat = (noise * env).astype(np.float32)
#         idx1 = min(n, idx0 + hat.size)
#         out[idx0:idx1] += hat[: idx1 - idx0] * 0.10

#     # Render by looping cycles
#     total_cycles = int(math.ceil(g.duration_s / (cycle_beats * spb)))
#     for c in range(total_cycles):
#         base_t = c * cycle_beats * spb

#         for p in onsets_kick:
#             t = base_t + (p / pulses_per_beat) * spb
#             add_kick(t)
#         for p in onsets_snare:
#             t = base_t + (p / pulses_per_beat) * spb
#             add_snare(t)
#         for p in onsets_hat:
#             t = base_t + (p / pulses_per_beat) * spb
#             add_hat(t)

#     if PEDALBOARD_AVAILABLE:
#         # Add punch, tame highs
#         board = Pedalboard(
#             [
#                 HighpassFilter(cutoff_frequency_hz=40.0),
#                 Compressor(threshold_db=-18, ratio=4.0, attack_ms=3, release_ms=80),
#                 Clipping(threshold_db=-2.0),
#             ]
#         )
#         mono_cf = _channels_first(out)
#         effected = board(mono_cf, sr)
#         return cast(np.ndarray, effected)[0].astype(np.float32)

#     # Fallback
#     out = _soft_clip(out, drive=1.5)
#     return out.astype(np.float32)


# def _gen_texture(
#     g: GlobalSpec,
#     layer: LayerSpec,
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     """A background texture/noise layer.

#     We intentionally use pedalboard for filtering + space (no custom DSP fallbacks).
#     """
#     _require_pedalboard()

#     sr = g.sample_rate
#     n = int(g.duration_s * sr)

#     pat = (layer.pattern or "air").lower()
#     noise = rng.standard_normal(n).astype(np.float32) * 0.05

#     # Simple texture shaping (still synthesis-side)
#     lp = 4500.0
#     hp = 180.0
#     gain = 1.0

#     if pat in ("air", "breath"):
#         lp = 3500.0
#         hp = 220.0
#     elif pat in ("grain", "vinyl"):
#         lp = 6500.0
#         hp = 160.0
#         # add sparse clicks
#         clicks = int(max(0.0, g.duration_s) * 6)
#         for _ in range(clicks):
#             idx = int(rng.random() * max(1, n - 1))
#             j = min(n, idx + 20)
#             noise[idx:j] += (rng.random() * 0.6 - 0.3)
#     elif pat in ("noise", "storm"):
#         gain = 2.0
#         lp = 12_000.0
#         hp = 80.0
#     else:
#         lp = 4500.0

#     noise *= float(gain)

#     space = layer.space or g.space
#     room_size, wet = _space_to_reverb(space)
#     wet = float(min(0.95, max(0.0, wet + 0.15)))

#     board = Pedalboard(
#         [
#             HighpassFilter(cutoff_frequency_hz=hp),
#             LowpassFilter(cutoff_frequency_hz=lp),
#             Reverb(room_size=room_size, wet_level=wet, dry_level=1.0 - wet),
#         ]
#     )
#     mono_cf = _channels_first(noise)
#     effected = board(mono_cf, sr)
#     return cast(np.ndarray, effected)[0].astype(np.float32)




# def _gen_accent(
#     g: GlobalSpec,
#     layer: LayerSpec,
#     scale: List[float],
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     # A sparse bell/pluck layer.
#     sr = g.sample_rate
#     n = int(g.duration_s * sr)
#     out = np.zeros(n, dtype=np.float32)

#     spb = _layer_seconds_per_beat(g.bpm, layer.bpm)

#     osc_kind = "sine" if (layer.material or g.material) in ("glass", "metal") else "triangle"

#     # sparse hits every 2-4 beats
#     beat = 0.0
#     while beat * spb < g.duration_s:
#         t0 = beat * spb
#         idx0 = int(t0 * sr)
#         if idx0 >= n:
#             break

#         deg = float(rng.choice(scale))
#         f = _freq_hz(g.root, deg, octave=6, a4_hz=g.tuning_ref_hz)
#         dur_s = 0.25 + 0.15 * rng.random()
#         note = _make_note(f, dur_s, sr, kind=osc_kind, attack_s=0.002, release_s=0.25)
#         idx1 = min(n, idx0 + note.size)
#         out[idx0:idx1] += note[: idx1 - idx0] * 0.25

#         beat += float(rng.choice([2.0, 3.0, 4.0]))

#     if PEDALBOARD_AVAILABLE:
#         room_size, wet = _space_to_reverb(layer.space or g.space)
#         board = Pedalboard(
#             [
#                 Chorus(rate_hz=0.35, depth=0.25, mix=0.20),
#                 Reverb(room_size=room_size, wet_level=min(0.75, wet + 0.10), dry_level=1.0 - wet),
#             ]
#         )
#         mono_cf = _channels_first(out)
#         effected = board(mono_cf, sr)
#         return cast(np.ndarray, effected)[0].astype(np.float32)

#     out = _simple_delay(out, sr, delay_s=0.22, feedback=0.25, mix=0.18)
#     return out.astype(np.float32)


# # =============================================================================
# # Master + render entrypoints
# # =============================================================================


# def render(config: Union[Dict[str, Any], GlobalSpec], *, sample_rate: Optional[int] = None) -> np.ndarray:
#     """
#     Render config -> stereo audio (channels-first float32).

#     Accepted config forms:
#     - dict with keys described by GlobalSpec (+ optional "layers")
#     - GlobalSpec instance (no layers -> auto from density)
#     """
#     if isinstance(config, GlobalSpec):
#         raw: Dict[str, Any] = config.__dict__
#     else:
#         raw = cast(Dict[str, Any], config)

#     _require_pedalboard()

#     g = _resolve_global(raw)
#     if sample_rate is not None:
#         g.sample_rate = int(sample_rate)

#     # Canonical pitch set (single source of truth)
#     scale = _sanitize_scale_intervals(g.scale_intervals, g.mode)

#     # If the LLM gave a weird scale AND also asked for active harmony, self-heal toward safer harmony
#     microtonal = any(abs(x - round(x)) > 1e-6 for x in scale)
#     if microtonal and g.harmonic_motion in ("medium", "active"):
#         g.harmonic_motion = "minimal"

#     layers = _resolve_layers(raw, g)

#     # Deterministic RNG if seed != 0
#     if g.seed and g.seed != 0:
#         rng = np.random.default_rng(g.seed)
#     else:
#         rng = np.random.default_rng()

#     sr = g.sample_rate
#     n = int(g.duration_s * sr)
#     mix = np.zeros((2, n), dtype=np.float32)

#     # Layer rendering
#     for layer in layers:
#         role = layer.role
#         if role == "bass":
#             mono = _gen_bass(g, layer, scale, rng)
#         elif role == "pad":
#             mono = _gen_pad(g, layer, scale, rng)
#         elif role == "lead":
#             mono = _gen_melody_like(g, layer, scale, rng, voice_like=False)
#         elif role == "voice":
#             mono = _gen_melody_like(g, layer, scale, rng, voice_like=True)
#         elif role == "rhythm":
#             mono = _gen_rhythm(g, layer, rng)
#         elif role == "accent":
#             mono = _gen_accent(g, layer, scale, rng)
#         else:  # texture or unknown
#             mono = _gen_texture(g, layer, rng)

#         # Per-layer pan and gain
#         stereo = _linear_pan(mono, pan=layer.pan)
#         stereo *= float(layer.gain)

#         mix[:, : stereo.shape[1]] += stereo[:, : mix.shape[1]]
#     # Master bus (pedalboard-only)
#     _require_pedalboard()
#     room_size, wet = _space_to_reverb(g.space)
#     # Keep master reverb subtle; per-layer already has space
#     master_reverb_wet = min(0.20, wet * 0.35)
#     master_board = Pedalboard(
#         [
#             HighpassFilter(cutoff_frequency_hz=25.0),
#             Compressor(threshold_db=-16, ratio=2.0, attack_ms=10, release_ms=200),
#             Reverb(room_size=room_size, wet_level=master_reverb_wet, dry_level=1.0 - master_reverb_wet),
#             Limiter(threshold_db=-1.0),
#         ]
#     )
#     mix = cast(np.ndarray, master_board(mix, sr)).astype(np.float32)

#     # Final safety normalize
#     mix = _normalize_peak(mix, peak=0.98)

#     return mix.astype(np.float32)


# def write_wav(path: str, audio_cf: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
#     """Write channels-first audio to a wav file using soundfile."""
#     import soundfile as sf  # local import to keep module import light

#     a = _channels_last(audio_cf)  # (samples, channels)
#     sf.write(path, a, int(sample_rate))


# if __name__ == "__main__":  # pragma: no cover
#     import argparse
#     import json

#     parser = argparse.ArgumentParser(description="Render a JSON config with new_synth -> WAV")
#     parser.add_argument("--config", type=str, required=True, help="Path to JSON config (global dict or {global,layers}).")
#     parser.add_argument("--out", type=str, required=True, help="Output WAV path")
#     parser.add_argument("--duration", type=float, default=None, help="Override duration in seconds")
#     parser.add_argument("--sr", type=int, default=None, help="Override sample rate")
#     args = parser.parse_args()

#     with open(args.config, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # Accept a few common wrappers:
#     # - {"config": {...}}
#     # - {"global": {...}, "layers": [...]}
#     if isinstance(data, dict) and isinstance(data.get("config"), dict):
#         data = data["config"]

#     if isinstance(data, dict) and ("global" in data or "global_config" in data):
#         g = data.get("global") if isinstance(data.get("global"), dict) else None
#         if g is None and isinstance(data.get("global_config"), dict):
#             g = data.get("global_config")
#         if g is not None:
#             flat = dict(g)
#             if "layers" in data:
#                 flat["layers"] = data["layers"]
#             data = flat

#     if not isinstance(data, dict):
#         raise SystemExit("Config JSON must be an object/dict.")

#     # Apply CLI overrides (let _resolve_global clamp)
#     if args.duration is not None:
#         data["duration_s"] = float(args.duration)
#     if args.sr is not None:
#         data["sample_rate"] = int(args.sr)

#     # Resolve once for accurate SR/duration reporting
#     g = _resolve_global(data)

#     audio = render(data)
#     write_wav(args.out, audio, sample_rate=g.sample_rate)
#     print(f"Wrote {args.out} (sr={g.sample_rate}, duration_s={g.duration_s})")


##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

# """
# new_synth.py

# A "culture-general" procedural synth with Pedalboard effects.

# REQUIRES: pedalboard (pip install pedalboard)

# Audio array convention:
# - Internally we use channels-first float32 arrays with shape (channels, samples),
#   matching pedalboard's conventions.
# """

# from __future__ import annotations

# from dataclasses import dataclass
# import json
# import math
# from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

# import numpy as np

# # Pedalboard is REQUIRED - no fallback mode
# from pedalboard import (
#     Pedalboard,
#     Bitcrush,
#     Chorus,
#     Clipping,
#     Compressor,
#     Delay,
#     Distortion,
#     Gain,
#     HighpassFilter,
#     HighShelfFilter,
#     LadderFilter,
#     Limiter,
#     LowpassFilter,
#     LowShelfFilter,
#     Phaser,
#     Reverb,
# )


# # =============================================================================
# # Defaults + lookup tables
# # =============================================================================

# DEFAULT_SAMPLE_RATE = 44_100

# NoteName = Literal["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]

# MODE_INTERVALS: Dict[str, List[float]] = {
#     "major": [0, 2, 4, 5, 7, 9, 11],
#     "minor": [0, 2, 3, 5, 7, 8, 10],
#     "dorian": [0, 2, 3, 5, 7, 9, 10],
#     "phrygian": [0, 1, 3, 5, 7, 8, 10],
#     "lydian": [0, 2, 4, 6, 7, 9, 11],
#     "mixolydian": [0, 2, 4, 5, 7, 9, 10],
#     "locrian": [0, 1, 3, 5, 6, 8, 10],
#     "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
#     "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
#     "pentatonic_major": [0, 2, 4, 7, 9],
#     "pentatonic_minor": [0, 3, 5, 7, 10],
#     "blues": [0, 3, 5, 6, 7, 10],
#     "whole_tone": [0, 2, 4, 6, 8, 10],
#     "chromatic": [float(i) for i in range(12)],
# }

# ROOT_TO_SEMITONE: Dict[str, int] = {
#     "c": 0, "c#": 1, "d": 2, "d#": 3, "e": 4, "f": 5,
#     "f#": 6, "g": 7, "g#": 8, "a": 9, "a#": 10, "b": 11,
# }

# TEMPO_TO_BPM: Dict[str, float] = {
#     "glacial": 48.0,
#     "slow": 68.0,
#     "medium": 102.0,
#     "fast": 132.0,
#     "frenetic": 165.0,
# }

# SAFE_TEMPO_RATIOS: List[Tuple[int, int]] = [
#     (1, 1), (1, 2), (2, 1), (3, 2), (2, 3), (4, 3), (3, 4),
# ]


# # =============================================================================
# # Public dataclasses
# # =============================================================================


# @dataclass
# class GlobalSpec:
#     bpm: float = 96.0
#     cycle_beats: int = 16
#     subdivision: int = 2
#     swing: float = 0.0

#     root: str = "c"
#     mode: str = "minor"
#     scale_intervals: Optional[List[float]] = None
#     tuning_ref_hz: float = 440.0

#     harmonic_motion: str = "minimal"
#     chord_color: str = "open5"

#     brightness: str = "medium"
#     space: str = "room"
#     character: str = "clean"
#     material: str = "warm_analog"
#     stereo_width: str = "medium"
#     density: int = 4

#     humanize: float = 0.15
#     ornament_intensity: str = "subtle"

#     seed: int = 0
#     sample_rate: int = DEFAULT_SAMPLE_RATE
#     duration_s: float = 16.0


# @dataclass
# class LayerSpec:
#     role: str = "pad"
#     pattern: str = "auto"
#     gain: float = 1.0
#     pan: float = 0.0

#     bpm: Optional[float] = None
#     cycle_beats: Optional[int] = None

#     material: Optional[str] = None
#     brightness: Optional[str] = None
#     character: Optional[str] = None
#     space: Optional[str] = None
#     ornament_intensity: Optional[str] = None


# # =============================================================================
# # Small helpers
# # =============================================================================


# def _clamp(x: float, lo: float, hi: float) -> float:
#     try:
#         xf = float(x)
#     except Exception:
#         return lo
#     if math.isnan(xf) or math.isinf(xf):
#         return lo
#     return float(min(hi, max(lo, xf)))


# def _clamp_int(x: Any, lo: int, hi: int) -> int:
#     try:
#         xi = int(x)
#     except Exception:
#         return lo
#     return int(min(hi, max(lo, xi)))


# def _safe_str(x: Any, default: str) -> str:
#     if isinstance(x, str) and x.strip():
#         return x.strip()
#     return default


# def _channels_first(audio: np.ndarray) -> np.ndarray:
#     a = np.asarray(audio)
#     if a.ndim == 1:
#         return a.astype(np.float32)[None, :]
#     if a.ndim != 2:
#         raise ValueError(f"Expected 1D or 2D audio array, got shape {a.shape}")
#     if a.shape[0] in (1, 2) and a.shape[1] > a.shape[0]:
#         return a.astype(np.float32)
#     return a.T.astype(np.float32)


# def _channels_last(audio_cf: np.ndarray) -> np.ndarray:
#     a = _channels_first(audio_cf)
#     return a.T


# def _normalize_peak(audio_cf: np.ndarray, peak: float = 0.98) -> np.ndarray:
#     a = _channels_first(audio_cf)
#     m = float(np.max(np.abs(a))) if a.size else 0.0
#     if m <= 1e-9:
#         return a
#     return (a * (peak / m)).astype(np.float32)


# def _linear_pan(mono: np.ndarray, pan: float) -> np.ndarray:
#     p = _clamp(pan, -1.0, 1.0)
#     left = math.cos((p + 1.0) * math.pi / 4.0)
#     right = math.sin((p + 1.0) * math.pi / 4.0)
#     out = np.vstack([mono * left, mono * right]).astype(np.float32)
#     return out


# # =============================================================================
# # Pitch math
# # =============================================================================


# def _note_to_midi(note: str, octave: int) -> int:
#     n = note.lower().strip()
#     if n not in ROOT_TO_SEMITONE:
#         n = "c"
#     sem = ROOT_TO_SEMITONE[n]
#     return 12 * (octave + 1) + sem


# def _freq_hz(root: str, semitone_offset: float, octave: int, a4_hz: float) -> float:
#     midi = _note_to_midi(root, octave)
#     return float(a4_hz * (2.0 ** (((midi - 69) + float(semitone_offset)) / 12.0)))


# def _sanitize_scale_intervals(
#     scale_intervals: Optional[Sequence[Any]],
#     mode: str,
# ) -> List[float]:
#     if not scale_intervals:
#         m = (mode or "minor").lower().strip()
#         base = MODE_INTERVALS.get(m, MODE_INTERVALS["minor"])
#         return [float(x) for x in base]

#     vals: List[float] = []
#     for x in scale_intervals:
#         try:
#             xf = float(x)
#         except Exception:
#             continue
#         if math.isnan(xf) or math.isinf(xf):
#             continue
#         xf = xf % 12.0
#         vals.append(xf)

#     if not vals:
#         return _sanitize_scale_intervals(None, mode)

#     vals.append(0.0)
#     vals = sorted(vals)
#     dedup: List[float] = []
#     for v in vals:
#         if not dedup or abs(v - dedup[-1]) > 0.05:
#             dedup.append(v)

#     if len(dedup) < 3:
#         return [float(x) for x in MODE_INTERVALS["pentatonic_minor"]]

#     return dedup[:24]


# def _nearest_index(values: Sequence[float], target: float) -> int:
#     best_i = 0
#     best_d = float("inf")
#     for i, v in enumerate(values):
#         d = abs(float(v) - float(target))
#         if d < best_d:
#             best_d = d
#             best_i = i
#     return best_i


# def _pick_open_fifth_or_octave(scale: Sequence[float]) -> float:
#     if not scale:
#         return 12.0
#     target = 7.0
#     i = _nearest_index(scale, target)
#     v = float(scale[i])
#     if abs(v - target) <= 1.0:
#         return v
#     return 12.0


# # =============================================================================
# # Low-level synthesis (numpy)
# # =============================================================================


# def _phase_from_freq(freq_hz: np.ndarray, sr: int) -> np.ndarray:
#     inc = (2.0 * math.pi * freq_hz) / float(sr)
#     return np.cumsum(inc, dtype=np.float64)


# def _osc_from_phase(phase: np.ndarray, kind: str) -> np.ndarray:
#     kind_l = (kind or "sine").lower()
#     if kind_l == "sine":
#         return np.sin(phase)
#     if kind_l == "square":
#         return np.sign(np.sin(phase))
#     if kind_l == "saw":
#         x = (phase / (2.0 * math.pi)) % 1.0
#         return 2.0 * x - 1.0
#     if kind_l == "triangle":
#         x = (phase / (2.0 * math.pi)) % 1.0
#         return 2.0 * np.abs(2.0 * x - 1.0) - 1.0
#     return np.sin(phase)


# def _adsr_env(
#     n: int, sr: int,
#     attack_s: float, decay_s: float, sustain: float, release_s: float,
# ) -> np.ndarray:
#     a = max(0, int(sr * _clamp(attack_s, 0.0, 5.0)))
#     d = max(0, int(sr * _clamp(decay_s, 0.0, 5.0)))
#     r = max(0, int(sr * _clamp(release_s, 0.0, 10.0)))
#     s = _clamp(sustain, 0.0, 1.0)

#     core = a + d + r
#     if core >= n:
#         a2 = max(1, n // 10)
#         r2 = max(1, n // 5)
#         mid = max(0, n - (a2 + r2))
#         env = np.concatenate([
#             np.linspace(0, 1, a2, endpoint=False),
#             np.ones(mid),
#             np.linspace(1, 0, r2, endpoint=True),
#         ])
#         return env.astype(np.float32)[:n]

#     sustain_len = n - core
#     env_a = np.linspace(0, 1, a, endpoint=False) if a > 0 else np.zeros(0)
#     env_d = np.linspace(1, s, d, endpoint=False) if d > 0 else np.zeros(0)
#     env_s = np.full(sustain_len, s, dtype=np.float32)
#     env_r = np.linspace(s, 0, r, endpoint=True) if r > 0 else np.zeros(0)

#     env = np.concatenate([env_a, env_d, env_s, env_r]).astype(np.float32)
#     if env.size < n:
#         env = np.pad(env, (0, n - env.size), mode="edge")
#     return env[:n]


# # =============================================================================
# # Musical event synthesis helpers
# # =============================================================================


# def _sine_tone(freq: np.ndarray, sr: int, kind: str) -> np.ndarray:
#     phase = _phase_from_freq(freq.astype(np.float64), sr)
#     wave = _osc_from_phase(phase, kind=kind).astype(np.float32)
#     return wave


# def _vibrato_curve(
#     base_freq: float, n: int, sr: int,
#     depth_cents: float, rate_hz: float,
# ) -> np.ndarray:
#     depth = _clamp(depth_cents, 0.0, 200.0)
#     rate = _clamp(rate_hz, 0.1, 12.0)
#     t = np.arange(n, dtype=np.float32) / float(sr)
#     cents = depth * np.sin(2.0 * math.pi * rate * t)
#     ratio = 2.0 ** (cents / 1200.0)
#     return (base_freq * ratio).astype(np.float32)


# def _glide_curve(f0: float, f1: float, n: int, curve: str = "exp") -> np.ndarray:
#     if n <= 1:
#         return np.array([float(f1)], dtype=np.float32)
#     if curve == "linear":
#         return np.linspace(f0, f1, n, dtype=np.float32)
#     t = np.linspace(0.0, 1.0, n, dtype=np.float32)
#     a = 8.0
#     w = (1.0 - np.exp(-a * t)) / (1.0 - np.exp(-a))
#     return (f0 + (f1 - f0) * w).astype(np.float32)


# def _make_note(
#     freq_hz: float, dur_s: float, sr: int, kind: str,
#     attack_s: float, release_s: float,
#     vibrato: Tuple[float, float] = (0.0, 0.0),
# ) -> np.ndarray:
#     n = max(1, int(sr * max(0.01, float(dur_s))))
#     base = np.full(n, float(freq_hz), dtype=np.float32)
#     if vibrato[0] > 0.0 and vibrato[1] > 0.0:
#         base = _vibrato_curve(freq_hz, n, sr, vibrato[0], vibrato[1])
#     wave = _sine_tone(base, sr, kind=kind)
#     env = _adsr_env(n, sr, attack_s=attack_s, decay_s=attack_s * 0.5, sustain=0.85, release_s=release_s)
#     return (wave * env).astype(np.float32)


# def _make_glide_note(
#     f0: float, f1: float, dur_s: float, sr: int, kind: str,
#     attack_s: float, release_s: float,
#     vibrato: Tuple[float, float] = (0.0, 0.0),
# ) -> np.ndarray:
#     n = max(1, int(sr * max(0.01, float(dur_s))))
#     freq = _glide_curve(f0, f1, n, curve="exp")
#     if vibrato[0] > 0.0 and vibrato[1] > 0.0:
#         vib = _vibrato_curve(1.0, n, sr, vibrato[0], vibrato[1])
#         freq = (freq * vib).astype(np.float32)
#     wave = _sine_tone(freq, sr, kind=kind)
#     env = _adsr_env(n, sr, attack_s=attack_s, decay_s=attack_s * 0.5, sustain=0.8, release_s=release_s)
#     return (wave * env).astype(np.float32)


# def _brightness_to_lp_cutoff(brightness: str, sr: int) -> float:
#     b = (brightness or "medium").lower()
#     mapping = {
#         "very_dark": 800.0,
#         "dark": 1800.0,
#         "medium": 4000.0,
#         "bright": 9000.0,
#         "very_bright": 16000.0,
#     }
#     return float(min(mapping.get(b, 4000.0), sr * 0.45))


# def _space_to_reverb(space: str) -> Tuple[float, float]:
#     s = (space or "room").lower()
#     mapping = {
#         "dry": (0.05, 0.05),
#         "intimate": (0.12, 0.12),
#         "room": (0.25, 0.22),
#         "hall": (0.55, 0.35),
#         "cathedral": (0.80, 0.50),
#         "infinite": (0.95, 0.70),
#     }
#     return mapping.get(s, (0.25, 0.22))


# def _stereo_width_to_chorus_mix(width: str) -> float:
#     w = (width or "medium").lower()
#     mapping = {
#         "mono": 0.0,
#         "narrow": 0.15,
#         "medium": 0.25,
#         "wide": 0.35,
#         "immersive": 0.45,
#     }
#     return float(mapping.get(w, 0.25))


# def _character_to_saturation(character: str) -> Tuple[float, float]:
#     c = (character or "clean").lower()
#     mapping = {
#         "clean": (0.0, 0.0),
#         "warm": (3.0, 0.15),
#         "saturated": (8.0, 0.35),
#         "distorted": (18.0, 0.55),
#     }
#     return mapping.get(c, (0.0, 0.0))


# def _ornament_to_params(orn: str) -> Tuple[float, float, float]:
#     o = (orn or "subtle").lower()
#     if o == "none":
#         return (0.0, 0.0, 0.0)
#     if o == "subtle":
#         return (10.0, 4.5, 0.10)
#     if o == "moderate":
#         return (25.0, 5.5, 0.25)
#     if o == "expressive":
#         return (45.0, 6.0, 0.45)
#     if o == "virtuosic":
#         return (70.0, 6.5, 0.70)
#     return (10.0, 4.5, 0.10)


# def _material_to_osc(material: str, role: str) -> str:
#     m = (material or "warm_analog").lower()
#     r = (role or "pad").lower()

#     if r in ("rhythm",):
#         return "sine"
#     if m == "digital":
#         return "saw"
#     if m == "warm_analog":
#         return "triangle" if r in ("pad", "bass") else "saw"
#     if m == "wood":
#         return "triangle"
#     if m == "metal":
#         return "saw"
#     if m == "glass":
#         return "sine"
#     if m == "breath":
#         return "sine"
#     if m == "skin":
#         return "sine"
#     return "sine"


# # =============================================================================
# # Config resolution
# # =============================================================================


# def _resolve_global(raw: Dict[str, Any]) -> GlobalSpec:
#     g = GlobalSpec()

#     tempo_name = raw.get("tempo", None)
#     if isinstance(tempo_name, str) and tempo_name in TEMPO_TO_BPM:
#         g.bpm = TEMPO_TO_BPM[tempo_name]
#     else:
#         g.bpm = _clamp(raw.get("bpm", raw.get("tempo_bpm", g.bpm)), 30.0, 220.0)

#     g.cycle_beats = _clamp_int(raw.get("cycle_beats", raw.get("beats_per_cycle", g.cycle_beats)), 2, 64)
#     g.subdivision = _clamp_int(raw.get("subdivision", g.subdivision), 1, 8)
#     if g.subdivision not in (2, 3, 4, 5):
#         g.subdivision = int(min((2, 3, 4, 5), key=lambda v: abs(v - g.subdivision)))

#     g.swing = _clamp(raw.get("swing", g.swing), 0.0, 0.75)

#     g.root = _safe_str(raw.get("root", g.root), g.root).lower()
#     if g.root not in ROOT_TO_SEMITONE:
#         g.root = "c"

#     g.mode = _safe_str(raw.get("mode", raw.get("scale_preset", g.mode)), g.mode).lower()
#     g.scale_intervals = raw.get("scale_intervals", raw.get("scale", raw.get("scale_degrees", None)))
#     g.tuning_ref_hz = _clamp(raw.get("tuning_ref_hz", raw.get("a4_hz", g.tuning_ref_hz)), 400.0, 480.0)

#     g.harmonic_motion = _safe_str(raw.get("harmonic_motion", g.harmonic_motion), g.harmonic_motion).lower()
#     if g.harmonic_motion not in ("drone", "minimal", "slow", "medium", "active"):
#         g.harmonic_motion = "minimal"

#     g.chord_color = _safe_str(raw.get("chord_color", raw.get("voicing", g.chord_color)), g.chord_color).lower()
#     if g.chord_color not in ("unison", "open5", "triad", "seventh", "cluster"):
#         g.chord_color = "open5"

#     g.brightness = _safe_str(raw.get("brightness", g.brightness), g.brightness).lower()
#     if g.brightness not in ("very_dark", "dark", "medium", "bright", "very_bright"):
#         g.brightness = "medium"

#     g.space = _safe_str(raw.get("space", g.space), g.space).lower()
#     if g.space not in ("dry", "intimate", "room", "hall", "cathedral", "infinite"):
#         g.space = "room"

#     g.character = _safe_str(raw.get("character", g.character), g.character).lower()
#     if g.character not in ("clean", "warm", "saturated", "distorted"):
#         g.character = "clean"

#     g.material = _safe_str(raw.get("material", raw.get("timbre_family", g.material)), g.material).lower()

#     g.stereo_width = _safe_str(raw.get("stereo_width", raw.get("stereo", g.stereo_width)), g.stereo_width).lower()
#     if g.stereo_width not in ("mono", "narrow", "medium", "wide", "immersive"):
#         g.stereo_width = "medium"

#     g.density = _clamp_int(raw.get("density", g.density), 1, 6)
#     g.humanize = _clamp(raw.get("humanize", g.humanize), 0.0, 1.0)

#     g.ornament_intensity = _safe_str(raw.get("ornament_intensity", raw.get("ornament", g.ornament_intensity)), g.ornament_intensity).lower()
#     if g.ornament_intensity not in ("none", "subtle", "moderate", "expressive", "virtuosic"):
#         g.ornament_intensity = "subtle"

#     g.seed = _clamp_int(raw.get("seed", g.seed), 0, 2_000_000_000)
#     g.sample_rate = _clamp_int(raw.get("sample_rate", g.sample_rate), 8_000, 192_000)
#     g.duration_s = _clamp(raw.get("duration_s", raw.get("duration", g.duration_s)), 2.0, 300.0)

#     return g


# def _default_layers_for_density(density: int) -> List[LayerSpec]:
#     layers: List[LayerSpec] = [
#         LayerSpec(role="pad", pattern="drone", gain=0.9, pan=0.0),
#     ]
#     if density >= 2:
#         layers.append(LayerSpec(role="bass", pattern="drone", gain=0.65, pan=0.0))
#     if density >= 3:
#         layers.append(LayerSpec(role="lead", pattern="melody", gain=0.8, pan=0.1))
#     if density >= 4:
#         layers.append(LayerSpec(role="rhythm", pattern="pulse", gain=0.7, pan=0.0))
#     if density >= 5:
#         layers.append(LayerSpec(role="texture", pattern="air", gain=0.4, pan=-0.2))
#     if density >= 6:
#         layers.append(LayerSpec(role="accent", pattern="sparkle", gain=0.35, pan=0.2))
#     return layers


# def _resolve_layers(raw: Dict[str, Any], g: GlobalSpec) -> List[LayerSpec]:
#     layers_raw = raw.get("layers", None)

#     layers: List[LayerSpec] = []
#     if layers_raw is None:
#         layers = _default_layers_for_density(g.density)
#     elif isinstance(layers_raw, list):
#         for item in layers_raw:
#             if not isinstance(item, dict):
#                 continue
#             layers.append(_layer_from_dict(item))
#     elif isinstance(layers_raw, dict):
#         for role, item in layers_raw.items():
#             if isinstance(item, dict):
#                 d = dict(item)
#                 d.setdefault("role", role)
#                 layers.append(_layer_from_dict(d))

#     if not layers:
#         layers = _default_layers_for_density(g.density)

#     seen: Dict[str, int] = {}
#     for ly in layers:
#         r = ly.role
#         seen[r] = seen.get(r, 0) + 1
#         if seen[r] > 2:
#             ly.role = "texture"
#             ly.pattern = "air"

#     gains = np.array([max(0.0, float(l.gain)) for l in layers], dtype=np.float32)
#     if gains.size == 0:
#         return _default_layers_for_density(g.density)
#     if float(gains.sum()) <= 1e-9:
#         gains[:] = 1.0
#     gains = gains / float(gains.sum())
#     for l, gn in zip(layers, gains):
#         l.gain = float(gn)

#     return layers


# def _layer_from_dict(d: Dict[str, Any]) -> LayerSpec:
#     l = LayerSpec()
#     l.role = _safe_str(d.get("role", l.role), l.role).lower()
#     if l.role not in ("bass", "pad", "lead", "rhythm", "texture", "voice", "accent"):
#         l.role = "texture"

#     l.pattern = _safe_str(d.get("pattern", d.get("style", l.pattern)), l.pattern).lower()
#     l.gain = float(_clamp(d.get("gain", l.gain), 0.0, 2.0))
#     l.pan = float(_clamp(d.get("pan", l.pan), -1.0, 1.0))

#     l.bpm = d.get("bpm", d.get("tempo_bpm", None))
#     if l.bpm is not None:
#         l.bpm = float(_clamp(l.bpm, 20.0, 260.0))

#     l.cycle_beats = d.get("cycle_beats", d.get("beats_per_cycle", None))
#     if l.cycle_beats is not None:
#         l.cycle_beats = int(_clamp_int(l.cycle_beats, 2, 64))

#     for key in ("material", "brightness", "character", "space", "ornament_intensity"):
#         if key in d and isinstance(d[key], str):
#             setattr(l, key, d[key].lower().strip())

#     if "timbre_family" in d and isinstance(d["timbre_family"], str):
#         l.material = d["timbre_family"].lower().strip()

#     return l


# # =============================================================================
# # Layer generators (all use Pedalboard)
# # =============================================================================


# def _layer_seconds_per_beat(global_bpm: float, layer_bpm: Optional[float]) -> float:
#     gbpm = _clamp(global_bpm, 30.0, 220.0)
#     if layer_bpm is None:
#         return 60.0 / gbpm

#     ratio = _clamp(layer_bpm / gbpm, 0.25, 4.0)
#     best = (1, 1)
#     best_err = float("inf")
#     for p, q in SAFE_TEMPO_RATIOS:
#         r = p / q
#         err = abs(ratio - r)
#         if err < best_err:
#             best_err = err
#             best = (p, q)

#     effective = gbpm * (best[0] / best[1])
#     return 60.0 / effective


# def _gen_bass(
#     g: GlobalSpec,
#     layer: LayerSpec,
#     scale: List[float],
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     sr = g.sample_rate
#     n = int(g.duration_s * sr)
#     out = np.zeros(n, dtype=np.float32)

#     spb = _layer_seconds_per_beat(g.bpm, layer.bpm)
#     osc_kind = _material_to_osc(layer.material or g.material, "bass")
#     lp_cutoff = _brightness_to_lp_cutoff(layer.brightness or g.brightness, sr) * 0.35
#     character = layer.character or g.character
#     drive_db, _ = _character_to_saturation(character)

#     fifth = _pick_open_fifth_or_octave(scale)
#     degrees = [0.0] if fifth == 12.0 else [0.0, fifth]

#     pat = (layer.pattern or "auto").lower()
#     if pat in ("auto", "drone", "tanpura"):
#         f = _freq_hz(g.root, 0.0, octave=2, a4_hz=g.tuning_ref_hz)
#         note = _make_note(f, g.duration_s, sr, kind=osc_kind, attack_s=0.02, release_s=0.4)
#         out += note * 0.35
#         if len(degrees) > 1 and degrees[1] != 12.0:
#             f2 = _freq_hz(g.root, degrees[1], octave=2, a4_hz=g.tuning_ref_hz)
#             note2 = _make_note(f2, g.duration_s, sr, kind=osc_kind, attack_s=0.02, release_s=0.4)
#             out += note2 * 0.18
#     else:
#         beat = 0.0
#         while beat * spb < g.duration_s:
#             start_s = beat * spb
#             dur_s = spb * (1.0 if pat in ("pulse", "pulsing", "bounce") else 0.5)
#             idx0 = int(start_s * sr)
#             idx1 = min(n, idx0 + int(dur_s * sr))

#             deg = float(rng.choice(degrees))
#             f = _freq_hz(g.root, deg, octave=2, a4_hz=g.tuning_ref_hz)
#             note = _make_note(f, (idx1 - idx0) / sr, sr, kind=osc_kind, attack_s=0.005, release_s=0.12)
#             out[idx0:idx0 + note.size] += note * 0.55
#             beat += 1.0 if pat in ("pulse", "pulsing", "bounce") else 0.5

#     effects: List[Any] = [
#         LowpassFilter(cutoff_frequency_hz=lp_cutoff),
#         Compressor(threshold_db=-22, ratio=3.0, attack_ms=8, release_ms=120),
#     ]
#     if drive_db > 0:
#         effects.append(Distortion(drive_db=drive_db * 0.5))
#     effects.append(LowShelfFilter(cutoff_frequency_hz=120.0, gain_db=2.0))

#     board = Pedalboard(effects)
#     mono_cf = _channels_first(out)
#     effected = board(mono_cf, sr)
#     return cast(np.ndarray, effected)[0].astype(np.float32)


# def _gen_pad(
#     g: GlobalSpec,
#     layer: LayerSpec,
#     scale: List[float],
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     sr = g.sample_rate
#     n = int(g.duration_s * sr)
#     out = np.zeros(n, dtype=np.float32)

#     spb = _layer_seconds_per_beat(g.bpm, layer.bpm)
#     cycle_beats = layer.cycle_beats or g.cycle_beats

#     osc_kind = _material_to_osc(layer.material or g.material, "pad")
#     lp_cutoff = _brightness_to_lp_cutoff(layer.brightness or g.brightness, sr)
#     character = layer.character or g.character
#     drive_db, _ = _character_to_saturation(character)

#     microtonal = any(abs(x - round(x)) > 1e-6 for x in scale)
#     hm = (g.harmonic_motion or "minimal").lower()
#     safe_drone = (hm in ("drone", "minimal")) or microtonal

#     fifth = _pick_open_fifth_or_octave(scale)
#     chord_degs: List[float] = [0.0]
#     if g.chord_color in ("open5", "triad", "seventh", "cluster"):
#         chord_degs.append(fifth if fifth != 12.0 else 0.0)
#     chord_degs.append(12.0)

#     if g.chord_color in ("triad", "seventh", "cluster") and len(scale) >= 4:
#         third = float(scale[_nearest_index(scale, 3.5)])
#         if third not in chord_degs:
#             chord_degs.append(third)

#     chord_degs = list(dict.fromkeys(chord_degs))

#     def chord_for_degree(shift: float) -> List[float]:
#         return [d + shift for d in chord_degs]

#     if safe_drone:
#         changes = [(0.0, chord_for_degree(0.0))]
#     else:
#         bars = max(1, int(round(g.duration_s / (4.0 * spb))))
#         if hm == "slow":
#             change_every = 8
#         elif hm == "medium":
#             change_every = 4
#         else:
#             change_every = 2

#         prog = [0, 4, 5, 3] if "major" in (g.mode or "") else [0, 5, 3, 6]
#         prog = [p % max(1, len(scale)) for p in prog]
#         shifts = [float(scale[p]) for p in prog]
#         changes = []
#         for b in range(0, bars, change_every):
#             t0 = b * 4.0 * spb
#             shift = shifts[(b // change_every) % len(shifts)]
#             changes.append((t0, chord_for_degree(shift)))

#     for i, (t0, degs) in enumerate(changes):
#         t1 = changes[i + 1][0] if i + 1 < len(changes) else g.duration_s
#         idx0 = int(t0 * sr)
#         idx1 = int(t1 * sr)
#         dur = max(1, idx1 - idx0) / sr

#         for deg in degs:
#             if deg >= 12.0:
#                 f = _freq_hz(g.root, deg % 12.0, octave=4, a4_hz=g.tuning_ref_hz) * 2.0
#             else:
#                 f = _freq_hz(g.root, deg, octave=4, a4_hz=g.tuning_ref_hz)

#             note = _make_note(f, dur, sr, kind=osc_kind, attack_s=0.35, release_s=0.8)
#             out[idx0:idx0 + note.size] += note * (0.18 / max(1, len(degs)))

#     space = layer.space or g.space
#     room_size, wet = _space_to_reverb(space)
#     chorus_mix = _stereo_width_to_chorus_mix(g.stereo_width)

#     effects: List[Any] = [
#         LadderFilter(mode=LadderFilter.Mode.LPF12, cutoff_hz=lp_cutoff, resonance=0.2),
#     ]
#     if drive_db > 0:
#         effects.append(Distortion(drive_db=drive_db * 0.3))
#     effects.extend([
#         Chorus(rate_hz=0.25, depth=chorus_mix, mix=chorus_mix),
#         Phaser(rate_hz=0.15, depth=0.2, mix=chorus_mix * 0.3),
#         Delay(delay_seconds=0.3 * room_size, feedback=0.2, mix=wet * 0.2),
#         Reverb(room_size=room_size, wet_level=wet, dry_level=1.0 - wet),
#     ])

#     board = Pedalboard(effects)
#     mono_cf = _channels_first(out)
#     effected = board(mono_cf, sr)
#     return cast(np.ndarray, effected)[0].astype(np.float32)


# def _gen_melody_like(
#     g: GlobalSpec,
#     layer: LayerSpec,
#     scale: List[float],
#     rng: np.random.Generator,
#     voice_like: bool = False,
# ) -> np.ndarray:
#     sr = g.sample_rate
#     n = int(g.duration_s * sr)
#     out = np.zeros(n, dtype=np.float32)

#     spb = _layer_seconds_per_beat(g.bpm, layer.bpm)
#     cycle_beats = layer.cycle_beats or g.cycle_beats

#     osc_kind = _material_to_osc(layer.material or g.material, "voice" if voice_like else "lead")
#     lp_cutoff = _brightness_to_lp_cutoff(layer.brightness or g.brightness, sr)
#     character = layer.character or g.character
#     drive_db, _ = _character_to_saturation(character)

#     orn = layer.ornament_intensity or g.ornament_intensity
#     vib_cents, vib_rate, glide_prob = _ornament_to_params(orn)

#     hm = (g.harmonic_motion or "minimal").lower()
#     if hm in ("drone", "minimal"):
#         note_every_beats = 2.0
#         note_len_beats = 1.75
#     elif hm == "slow":
#         note_every_beats = 1.0
#         note_len_beats = 0.9
#     elif hm == "medium":
#         note_every_beats = 0.5
#         note_len_beats = 0.45
#     else:
#         note_every_beats = 0.25
#         note_len_beats = 0.20

#     if voice_like:
#         note_every_beats *= 1.15
#         note_len_beats *= 1.35
#         glide_prob = min(0.9, glide_prob + 0.2)

#     base_oct = 5
#     weights = np.ones(len(scale), dtype=np.float32)
#     weights[0] = 3.0
#     if len(scale) > 4:
#         weights[4 % len(scale)] += 0.8
#     weights = weights / float(weights.sum())

#     prev_deg = 0.0
#     beat = 0.0
#     while beat * spb < g.duration_s:
#         t0 = beat * spb
#         if g.swing > 0 and (int(beat * 2) % 2 == 1):
#             t0 += (g.swing * 0.35) * spb

#         dur_s = max(0.05, note_len_beats * spb)
#         idx0 = int(t0 * sr)
#         if idx0 >= n:
#             break

#         step = rng.choice([-2, -1, 0, 1, 2], p=[0.05, 0.20, 0.50, 0.20, 0.05])
#         if hm == "active" and rng.random() < 0.25:
#             step = int(rng.integers(-4, 5))
#         prev_i = _nearest_index(scale, prev_deg)
#         new_i = int((prev_i + step) % len(scale))
#         deg = float(scale[new_i])

#         if (beat % cycle_beats) < 1e-6 and rng.random() < 0.6:
#             deg = 0.0

#         f1 = _freq_hz(g.root, deg, octave=base_oct, a4_hz=g.tuning_ref_hz)

#         do_glide = (rng.random() < glide_prob) and (beat > 0)
#         if do_glide:
#             f0 = _freq_hz(g.root, prev_deg, octave=base_oct, a4_hz=g.tuning_ref_hz)
#             note = _make_glide_note(
#                 f0, f1, dur_s, sr, kind=osc_kind,
#                 attack_s=0.01 if not voice_like else 0.02,
#                 release_s=0.08 if not voice_like else 0.20,
#                 vibrato=(vib_cents, vib_rate),
#             )
#         else:
#             note = _make_note(
#                 f1, dur_s, sr, kind=osc_kind,
#                 attack_s=0.008 if not voice_like else 0.02,
#                 release_s=0.06 if not voice_like else 0.15,
#                 vibrato=(vib_cents, vib_rate),
#             )

#         vel = 0.42 if voice_like else 0.35
#         vel *= float(0.85 + 0.3 * rng.random())
#         if g.humanize > 0:
#             jitter = (rng.random() - 0.5) * g.humanize * 0.06 * spb
#             idx0 = int((t0 + jitter) * sr)
#             idx0 = max(0, min(n - 1, idx0))

#         idx1 = min(n, idx0 + note.size)
#         out[idx0:idx1] += note[: idx1 - idx0] * vel

#         prev_deg = deg
#         beat += note_every_beats

#     space = layer.space or g.space
#     room_size, wet = _space_to_reverb(space)
#     chorus_mix = _stereo_width_to_chorus_mix(g.stereo_width)

#     effects: List[Any] = [LowpassFilter(cutoff_frequency_hz=lp_cutoff)]
#     if drive_db > 0:
#         effects.append(Distortion(drive_db=drive_db * 0.4))
#     effects.extend([
#         Chorus(rate_hz=0.45, depth=chorus_mix * 0.8, mix=chorus_mix * 0.6),
#         Delay(delay_seconds=0.18 if not voice_like else 0.10, feedback=0.22, mix=0.16 if not voice_like else 0.08),
#         Reverb(room_size=room_size, wet_level=wet, dry_level=1.0 - wet),
#     ])
#     if voice_like:
#         effects.insert(1, Phaser(rate_hz=0.08, depth=0.15, mix=0.1))

#     board = Pedalboard(effects)
#     mono_cf = _channels_first(out)
#     effected = board(mono_cf, sr)
#     return cast(np.ndarray, effected)[0].astype(np.float32)


# def _gen_rhythm(
#     g: GlobalSpec,
#     layer: LayerSpec,
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     sr = g.sample_rate
#     n = int(g.duration_s * sr)
#     out = np.zeros(n, dtype=np.float32)

#     spb = _layer_seconds_per_beat(g.bpm, layer.bpm)
#     cycle_beats = layer.cycle_beats or g.cycle_beats
#     character = layer.character or g.character
#     drive_db, _ = _character_to_saturation(character)

#     pat = (layer.pattern or "pulse").lower()

#     pulses_per_beat = g.subdivision
#     pulses_per_cycle = int(cycle_beats * pulses_per_beat)
#     if pulses_per_cycle <= 0:
#         pulses_per_cycle = 16

#     onsets_kick: List[int] = []
#     onsets_snare: List[int] = []
#     onsets_hat: List[int] = []

#     if pat in ("auto", "pulse", "heartbeat"):
#         for b in range(int(cycle_beats)):
#             onsets_kick.append(b * pulses_per_beat)
#         for b in range(int(cycle_beats)):
#             if b % 2 == 1:
#                 onsets_hat.append(b * pulses_per_beat)
#     elif pat in ("fourfloor", "four_floor", "driving"):
#         for b in range(int(cycle_beats)):
#             onsets_kick.append(b * pulses_per_beat)
#         for b in range(int(cycle_beats)):
#             onsets_hat.append(b * pulses_per_beat + pulses_per_beat // 2)
#         for b in range(int(cycle_beats)):
#             if b % 4 == 2:
#                 onsets_snare.append(b * pulses_per_beat)
#     elif pat in ("clave", "world", "syncopated", "darbuka", "tabla"):
#         base = [0, 3, 4, 6, 7]
#         cell_len = 8 * max(1, pulses_per_beat // 2)
#         for x in base:
#             onsets_snare.append(int(x * (pulses_per_beat / 2)))
#         onsets_kick.extend([0, int(4 * pulses_per_beat)])
#         for p in range(0, cell_len, max(1, pulses_per_beat // 2)):
#             if p not in onsets_snare:
#                 onsets_hat.append(p)
#     elif pat in ("polyrhythm",):
#         onsets_kick = [0, int(pulses_per_cycle / 2)]
#         onsets_hat = [int(i * pulses_per_cycle / 3) for i in range(3)]
#         onsets_snare = [int(pulses_per_cycle / 4), int(3 * pulses_per_cycle / 4)]
#     else:
#         for b in range(int(cycle_beats)):
#             onsets_kick.append(b * pulses_per_beat)

#     def add_kick(t_s: float):
#         idx0 = int(t_s * sr)
#         if idx0 >= n:
#             return
#         dur = 0.18
#         nn = int(dur * sr)
#         t = np.arange(nn, dtype=np.float32) / float(sr)
#         f0, f1 = 110.0, 45.0
#         f = f1 + (f0 - f1) * np.exp(-t * 18.0)
#         wave = np.sin(2.0 * math.pi * f * t)
#         env = np.exp(-t * 16.0)
#         kick = (wave * env).astype(np.float32)
#         idx1 = min(n, idx0 + kick.size)
#         out[idx0:idx1] += kick[: idx1 - idx0] * 0.9

#     def add_snare(t_s: float):
#         idx0 = int(t_s * sr)
#         if idx0 >= n:
#             return
#         dur = 0.14
#         nn = int(dur * sr)
#         noise = rng.standard_normal(nn).astype(np.float32)
#         t = np.arange(nn, dtype=np.float32) / float(sr)
#         env = np.exp(-t * 22.0)
#         sn = (noise * env).astype(np.float32)
#         idx1 = min(n, idx0 + sn.size)
#         out[idx0:idx1] += sn[: idx1 - idx0] * 0.25

#     def add_hat(t_s: float):
#         idx0 = int(t_s * sr)
#         if idx0 >= n:
#             return
#         dur = 0.05
#         nn = int(dur * sr)
#         noise = rng.standard_normal(nn).astype(np.float32)
#         t = np.arange(nn, dtype=np.float32) / float(sr)
#         env = np.exp(-t * 60.0)
#         hat = (noise * env).astype(np.float32)
#         idx1 = min(n, idx0 + hat.size)
#         out[idx0:idx1] += hat[: idx1 - idx0] * 0.10

#     total_cycles = int(math.ceil(g.duration_s / (cycle_beats * spb)))
#     for c in range(total_cycles):
#         base_t = c * cycle_beats * spb

#         for p in onsets_kick:
#             t = base_t + (p / pulses_per_beat) * spb
#             add_kick(t)
#         for p in onsets_snare:
#             t = base_t + (p / pulses_per_beat) * spb
#             add_snare(t)
#         for p in onsets_hat:
#             t = base_t + (p / pulses_per_beat) * spb
#             add_hat(t)

#     effects: List[Any] = [
#         HighpassFilter(cutoff_frequency_hz=40.0),
#         Compressor(threshold_db=-18, ratio=4.0, attack_ms=3, release_ms=80),
#     ]
#     if drive_db > 0:
#         effects.append(Distortion(drive_db=drive_db * 0.6))
#     effects.append(Clipping(threshold_db=-2.0))

#     board = Pedalboard(effects)
#     mono_cf = _channels_first(out)
#     effected = board(mono_cf, sr)
#     return cast(np.ndarray, effected)[0].astype(np.float32)


# def _gen_texture(
#     g: GlobalSpec,
#     layer: LayerSpec,
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     sr = g.sample_rate
#     n = int(g.duration_s * sr)

#     pat = (layer.pattern or "air").lower()
#     noise = rng.standard_normal(n).astype(np.float32) * 0.05

#     space = layer.space or g.space
#     room_size, wet = _space_to_reverb(space)
#     brightness = layer.brightness or g.brightness
#     lp_cutoff = _brightness_to_lp_cutoff(brightness, sr)

#     if pat in ("air", "breath", "desert_wind"):
#         base_cutoff = 3500.0
#     elif pat in ("grain", "vinyl"):
#         base_cutoff = 6500.0
#         for _ in range(int(g.duration_s * 6)):
#             idx = int(rng.random() * n)
#             noise[idx: idx + 20] += (rng.random() * 0.6 - 0.3)
#     elif pat in ("noise", "storm"):
#         base_cutoff = 12000.0
#         noise *= 2.0
#     else:
#         base_cutoff = 4500.0

#     effects: List[Any] = [
#         LowpassFilter(cutoff_frequency_hz=min(base_cutoff, lp_cutoff)),
#         HighpassFilter(cutoff_frequency_hz=180.0),
#         Chorus(rate_hz=0.1, depth=0.3, mix=0.15),
#         Reverb(room_size=room_size, wet_level=min(0.85, wet + 0.15), dry_level=1.0 - wet),
#     ]
#     if pat in ("shimmer",):
#         effects.insert(2, Phaser(rate_hz=0.2, depth=0.4, mix=0.25))

#     board = Pedalboard(effects)
#     mono_cf = _channels_first(noise)
#     effected = board(mono_cf, sr)
#     return cast(np.ndarray, effected)[0].astype(np.float32)


# def _gen_accent(
#     g: GlobalSpec,
#     layer: LayerSpec,
#     scale: List[float],
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     sr = g.sample_rate
#     n = int(g.duration_s * sr)
#     out = np.zeros(n, dtype=np.float32)

#     spb = _layer_seconds_per_beat(g.bpm, layer.bpm)

#     material = layer.material or g.material
#     osc_kind = "sine" if material in ("glass", "metal", "metallic_like") else "triangle"

#     beat = 0.0
#     while beat * spb < g.duration_s:
#         t0 = beat * spb
#         idx0 = int(t0 * sr)
#         if idx0 >= n:
#             break

#         deg = float(rng.choice(scale))
#         f = _freq_hz(g.root, deg, octave=6, a4_hz=g.tuning_ref_hz)
#         dur_s = 0.25 + 0.15 * rng.random()
#         note = _make_note(f, dur_s, sr, kind=osc_kind, attack_s=0.002, release_s=0.25)
#         idx1 = min(n, idx0 + note.size)
#         out[idx0:idx1] += note[: idx1 - idx0] * 0.25

#         beat += float(rng.choice([2.0, 3.0, 4.0]))

#     space = layer.space or g.space
#     room_size, wet = _space_to_reverb(space)

#     effects: List[Any] = [
#         HighpassFilter(cutoff_frequency_hz=1000.0),
#         Chorus(rate_hz=0.35, depth=0.25, mix=0.20),
#         Delay(delay_seconds=0.22, feedback=0.25, mix=0.18),
#         Reverb(room_size=room_size, wet_level=min(0.75, wet + 0.10), dry_level=1.0 - wet),
#     ]

#     board = Pedalboard(effects)
#     mono_cf = _channels_first(out)
#     effected = board(mono_cf, sr)
#     return cast(np.ndarray, effected)[0].astype(np.float32)


# # =============================================================================
# # Master + render entrypoints
# # =============================================================================


# def render(config: Union[Dict[str, Any], GlobalSpec], *, sample_rate: Optional[int] = None) -> np.ndarray:
#     """Render config -> stereo audio (channels-first float32)."""
#     if isinstance(config, GlobalSpec):
#         raw: Dict[str, Any] = config.__dict__
#     else:
#         raw = cast(Dict[str, Any], config)

#     g = _resolve_global(raw)
#     if sample_rate is not None:
#         g.sample_rate = int(sample_rate)

#     scale = _sanitize_scale_intervals(g.scale_intervals, g.mode)

#     microtonal = any(abs(x - round(x)) > 1e-6 for x in scale)
#     if microtonal and g.harmonic_motion in ("medium", "active"):
#         g.harmonic_motion = "minimal"

#     layers = _resolve_layers(raw, g)

#     if g.seed and g.seed != 0:
#         rng = np.random.default_rng(g.seed)
#     else:
#         rng = np.random.default_rng()

#     sr = g.sample_rate
#     n = int(g.duration_s * sr)
#     mix = np.zeros((2, n), dtype=np.float32)

#     for layer in layers:
#         role = layer.role
#         if role == "bass":
#             mono = _gen_bass(g, layer, scale, rng)
#         elif role == "pad":
#             mono = _gen_pad(g, layer, scale, rng)
#         elif role == "lead":
#             mono = _gen_melody_like(g, layer, scale, rng, voice_like=False)
#         elif role == "voice":
#             mono = _gen_melody_like(g, layer, scale, rng, voice_like=True)
#         elif role == "rhythm":
#             mono = _gen_rhythm(g, layer, rng)
#         elif role == "accent":
#             mono = _gen_accent(g, layer, scale, rng)
#         else:
#             mono = _gen_texture(g, layer, rng)

#         stereo = _linear_pan(mono, pan=layer.pan)
#         stereo *= float(layer.gain)

#         mix[:, : stereo.shape[1]] += stereo[:, : mix.shape[1]]

#     room_size, wet = _space_to_reverb(g.space)
#     master_reverb_wet = min(0.20, wet * 0.35)
#     character = g.character
#     drive_db, _ = _character_to_saturation(character)

#     master_effects: List[Any] = [
#         HighpassFilter(cutoff_frequency_hz=25.0),
#         Compressor(threshold_db=-16, ratio=2.0, attack_ms=10, release_ms=200),
#     ]
#     if drive_db > 0:
#         master_effects.append(Distortion(drive_db=drive_db * 0.2))
#     master_effects.extend([
#         Reverb(room_size=room_size, wet_level=master_reverb_wet, dry_level=1.0 - master_reverb_wet),
#         Limiter(threshold_db=-1.0),
#     ])

#     master_board = Pedalboard(master_effects)
#     mix = cast(np.ndarray, master_board(mix, sr)).astype(np.float32)
#     mix = _normalize_peak(mix, peak=0.98)

#     return mix.astype(np.float32)


# def write_wav(path: str, audio_cf: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
#     """Write channels-first audio to a wav file."""
#     import soundfile as sf
#     a = _channels_last(audio_cf)
#     sf.write(path, a, int(sample_rate))


# def _load_json(path: str) -> Dict[str, Any]:
#     with open(path, "r", encoding="utf-8") as f:
#         return cast(Dict[str, Any], json.load(f))


# def _extract_first_config(obj: Any) -> Any:
#     if isinstance(obj, dict):
#         if isinstance(obj.get("config"), dict):
#             return obj["config"]
#         if isinstance(obj.get("plan"), dict):
#             return obj["plan"]
#         if isinstance(obj.get("global_config"), dict):
#             result = dict(obj["global_config"])
#             if "layers" in obj:
#                 result["layers"] = obj["layers"]
#             return result
#     return obj


# if __name__ == "__main__":
#     import argparse

#     p = argparse.ArgumentParser(description="Render a JSON config into a WAV.")
#     p.add_argument("--config", type=str, required=True, help="Path to JSON config file")
#     p.add_argument("--out", type=str, required=True, help="Output wav path")
#     p.add_argument("--duration", type=float, default=None, help="Override duration in seconds")
#     p.add_argument("--sample-rate", type=int, default=None, help="Override sample rate")
#     args = p.parse_args()

#     cfg_any = _extract_first_config(_load_json(args.config))
#     if not isinstance(cfg_any, dict):
#         raise SystemExit(f"Config JSON must be an object at top-level (got {type(cfg_any)})")

#     cfg: Dict[str, Any] = dict(cfg_any)
#     if args.duration is not None:
#         cfg["duration_s"] = float(args.duration)
#     if args.sample_rate is not None:
#         cfg["sample_rate"] = int(args.sample_rate)

#     sr = int(cfg.get("sample_rate", DEFAULT_SAMPLE_RATE))
#     audio = render(cfg, sample_rate=sr)
#     write_wav(args.out, audio, sample_rate=sr)
#     print(f"Wrote {args.out}")


##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

"""
new_synth_sf2.py

Hybrid synth that uses SF2 SoundFont samples for tonal instruments
while keeping the Pedalboard effects chain from new_synth.py.

This provides much more realistic instrument sounds compared to raw oscillators.

REQUIRES:
    pip install sf2_loader pedalboard numpy soundfile

USAGE:
    from new_synth_sf2 import render, write_wav
    
    config = {
        "bpm": 96,
        "root": "c",
        "mode": "minor",
        "use_sf2": True,  # Enable SF2 samples (default: True)
        "layers": [
            {"role": "bass", "pattern": "drone", "instrument": "acoustic_bass"},
            {"role": "pad", "pattern": "sustained", "instrument": "string_ensemble"},
            {"role": "lead", "pattern": "melodic", "instrument": "sitar"},
        ]
    }
    
    audio = render(config)
    write_wav("output.wav", audio)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

# Import Pedalboard
from pedalboard import (
    Pedalboard,
    Chorus,
    Compressor,
    Delay,
    Distortion,
    HighpassFilter,
    LadderFilter,
    Limiter,
    LowpassFilter,
    LowShelfFilter,
    Phaser,
    Reverb,
)

# Import SF2 integration
#try:
from .sf2_integration import (
    SF2Engine,
    get_engine,
    note_name_to_midi,
    midi_to_freq,
    GM_INSTRUMENTS,
    VIBE_TO_GM,
    SF2_AVAILABLE,
)
HAS_SF2 = SF2_AVAILABLE
# except ImportError:
#     HAS_SF2 = False
#     print("Warning: sf2_integration not found. Falling back to oscillator synthesis.")


# =============================================================================
# Constants
# =============================================================================

DEFAULT_SAMPLE_RATE = 44100

MODE_INTERVALS: Dict[str, List[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
}

ROOT_TO_SEMITONE: Dict[str, int] = {
    "c": 0, "c#": 1, "d": 2, "d#": 3, "e": 4, "f": 5,
    "f#": 6, "g": 7, "g#": 8, "a": 9, "a#": 10, "b": 11,
}

# Instrument mappings for each role
ROLE_DEFAULT_INSTRUMENTS: Dict[str, str] = {
    "bass": "acoustic_bass",
    "pad": "string_ensemble",
    "lead": "violin",
    "voice": "choir_aahs",
    "accent": "vibraphone",
    "rhythm": "taiko",  # Percussion handled separately
    "texture": "pad_choir",
}

# Ethnic instrument aliases for easier use
ETHNIC_ALIASES: Dict[str, str] = {
    # Indian
    "tanpura": "sitar",  # GM sitar can approximate
    "tabla": "taiko",    # Fallback
    "bansuri": "pan_flute",
    
    # Middle Eastern
    "oud": "acoustic_guitar_nylon",  # Closest GM equivalent
    "ney": "pan_flute",
    "darbuka": "taiko",
    "doumbek": "taiko",
    "kanun": "dulcimer",
    "qanun": "dulcimer",
    
    # East Asian
    "erhu": "violin",
    "pipa": "shamisen",
    "guzheng": "koto",
    
    # Latin/African
    "berimbau": "banjo",
    "djembe": "taiko",
}


# =============================================================================
# Config Dataclass
# =============================================================================

@dataclass
class SynthConfig:
    """Configuration for the hybrid synth."""
    
    # Timing
    bpm: float = 96.0
    duration_s: float = 16.0
    cycle_beats: int = 16
    swing: float = 0.0
    
    # Key/Scale
    root: str = "c"
    mode: str = "minor"
    tuning_ref_hz: float = 440.0
    
    # Sound character
    brightness: str = "medium"
    space: str = "room"
    character: str = "clean"
    stereo_width: str = "medium"
    
    # Layers
    density: int = 4
    humanize: float = 0.15
    
    # SF2 control
    use_sf2: bool = True
    
    # Technical
    sample_rate: int = DEFAULT_SAMPLE_RATE
    seed: int = 0


@dataclass
class LayerConfig:
    """Configuration for a single layer."""
    
    role: str = "pad"
    pattern: str = "sustained"
    instrument: str = ""  # Empty = use default for role
    gain: float = 1.0
    pan: float = 0.0
    
    # Overrides (None = use global)
    brightness: Optional[str] = None
    space: Optional[str] = None
    character: Optional[str] = None


# =============================================================================
# Utility Functions
# =============================================================================

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _freq_hz(root: str, semitone_offset: float, octave: int = 4, a4_hz: float = 440.0) -> float:
    """Convert root + semitone offset to frequency."""
    root_semi = ROOT_TO_SEMITONE.get(root.lower(), 0)
    midi = 12 * (octave + 1) + root_semi + semitone_offset
    return a4_hz * (2.0 ** ((midi - 69) / 12.0))


def _get_scale_notes(root: str, mode: str, octave: int = 4) -> List[int]:
    """Get MIDI note numbers for a scale."""
    intervals = MODE_INTERVALS.get(mode.lower(), MODE_INTERVALS["minor"])
    root_midi = note_name_to_midi(root, octave) if HAS_SF2 else (
        12 * (octave + 1) + ROOT_TO_SEMITONE.get(root.lower(), 0)
    )
    return [root_midi + interval for interval in intervals]


def _brightness_to_cutoff(brightness: str, sr: int) -> float:
    """Map brightness to lowpass cutoff frequency."""
    mapping = {
        "very_dark": 800.0,
        "dark": 1800.0,
        "medium": 4000.0,
        "bright": 9000.0,
        "very_bright": 16000.0,
    }
    return min(mapping.get(brightness.lower(), 4000.0), sr * 0.45)


def _space_to_reverb(space: str) -> Tuple[float, float]:
    """Map space to (room_size, wet_level)."""
    mapping = {
        "dry": (0.05, 0.05),
        "intimate": (0.15, 0.12),
        "room": (0.30, 0.22),
        "hall": (0.55, 0.35),
        "cathedral": (0.80, 0.50),
        "infinite": (0.95, 0.70),
    }
    return mapping.get(space.lower(), (0.30, 0.22))


def _resolve_instrument(role: str, instrument: str) -> str:
    """Resolve instrument name, applying aliases and defaults."""
    if not instrument:
        instrument = ROLE_DEFAULT_INSTRUMENTS.get(role, "piano")
    
    instrument = instrument.lower().strip()
    
    # Check ethnic aliases
    if instrument in ETHNIC_ALIASES:
        instrument = ETHNIC_ALIASES[instrument]
    
    # Check vibe-style mapping
    if HAS_SF2 and instrument in VIBE_TO_GM:
        instrument = VIBE_TO_GM[instrument]
    
    return instrument


# =============================================================================
# Oscillator Fallback (when SF2 not available)
# =============================================================================

def _osc_sine(freq: np.ndarray, sr: int) -> np.ndarray:
    """Generate sine wave from frequency array."""
    phase = np.cumsum(freq / sr) * 2 * np.pi
    return np.sin(phase).astype(np.float32)


def _osc_triangle(freq: np.ndarray, sr: int) -> np.ndarray:
    """Generate triangle wave."""
    phase = np.cumsum(freq / sr)
    return (2 * np.abs(2 * (phase - np.floor(phase + 0.5))) - 1).astype(np.float32)


def _osc_saw(freq: np.ndarray, sr: int) -> np.ndarray:
    """Generate sawtooth wave."""
    phase = np.cumsum(freq / sr)
    return (2 * (phase - np.floor(phase + 0.5))).astype(np.float32)


def _adsr_envelope(
    n: int,
    sr: int,
    attack: float = 0.01,
    decay: float = 0.1,
    sustain: float = 0.7,
    release: float = 0.2
) -> np.ndarray:
    """Generate ADSR envelope."""
    a = max(1, int(attack * sr))
    d = max(1, int(decay * sr))
    r = max(1, int(release * sr))
    s_len = max(0, n - a - d - r)
    
    env = np.concatenate([
        np.linspace(0, 1, a),
        np.linspace(1, sustain, d),
        np.full(s_len, sustain),
        np.linspace(sustain, 0, r),
    ])
    
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n].astype(np.float32)


def _fallback_note(
    freq: float,
    duration: float,
    sr: int,
    waveform: str = "sine",
    attack: float = 0.01,
    release: float = 0.1,
) -> np.ndarray:
    """Generate a note using oscillators (fallback when SF2 unavailable)."""
    n = int(duration * sr)
    freq_arr = np.full(n, freq, dtype=np.float32)
    
    if waveform == "triangle":
        wave = _osc_triangle(freq_arr, sr)
    elif waveform == "saw":
        wave = _osc_saw(freq_arr, sr)
    else:
        wave = _osc_sine(freq_arr, sr)
    
    env = _adsr_envelope(n, sr, attack=attack, release=release)
    return wave * env


# =============================================================================
# SF2-Powered Generators
# =============================================================================

def _render_sf2_note(
    midi_note: int,
    duration: float,
    velocity: int,
    instrument: str,
    sr: int,
) -> np.ndarray:
    """Render a note using SF2 engine."""
    if not HAS_SF2:
        freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        return _fallback_note(freq, duration, sr)
    
    engine = get_engine()
    audio = engine.render_note(
        midi_note=midi_note,
        duration=duration,
        velocity=velocity,
        instrument=instrument,
    )
    
    # Resample if needed
    if engine.sample_rate != sr:
        # Simple resampling (for production, use scipy.signal.resample)
        ratio = sr / engine.sample_rate
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio)
    
    return audio.astype(np.float32)


def _render_sf2_chord(
    midi_notes: List[int],
    duration: float,
    velocity: int,
    instrument: str,
    sr: int,
) -> np.ndarray:
    """Render a chord using SF2 engine."""
    if not HAS_SF2:
        result = np.zeros(int(duration * sr), dtype=np.float32)
        for midi_note in midi_notes:
            freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
            result += _fallback_note(freq, duration, sr) * 0.3
        return np.clip(result, -1, 1)
    
    engine = get_engine()
    audio = engine.render_chord(
        midi_notes=midi_notes,
        duration=duration,
        velocity=velocity,
        instrument=instrument,
    )
    
    return audio.astype(np.float32)


# =============================================================================
# Layer Generators
# =============================================================================

def _gen_bass(
    config: SynthConfig,
    layer: LayerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate bass layer."""
    sr = config.sample_rate
    n = int(config.duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    
    spb = 60.0 / config.bpm
    instrument = _resolve_instrument("bass", layer.instrument)
    brightness = layer.brightness or config.brightness
    lp_cutoff = _brightness_to_cutoff(brightness, sr) * 0.35
    
    # Get root note
    root_midi = _get_scale_notes(config.root, config.mode, octave=2)[0]
    fifth_midi = root_midi + 7  # Perfect fifth
    
    pat = layer.pattern.lower()
    
    if pat in ("drone", "tanpura", "sustained"):
        # Long sustained note
        note = _render_sf2_note(root_midi, config.duration_s, 90, instrument, sr)
        if len(note) > n:
            note = note[:n]
        out[:len(note)] += note * 0.6
        
        # Add fifth for tanpura-like sound
        if pat == "tanpura":
            fifth = _render_sf2_note(fifth_midi, config.duration_s, 70, instrument, sr)
            if len(fifth) > n:
                fifth = fifth[:n]
            out[:len(fifth)] += fifth * 0.3
    
    elif pat in ("pulse", "pulsing", "bounce"):
        # Rhythmic bass
        beat = 0.0
        while beat * spb < config.duration_s:
            start_s = beat * spb
            dur_s = spb * 0.8
            idx0 = int(start_s * sr)
            
            # Alternate root and fifth
            midi = root_midi if int(beat) % 2 == 0 else fifth_midi
            note = _render_sf2_note(midi, dur_s, 100, instrument, sr)
            
            end_idx = min(n, idx0 + len(note))
            out[idx0:end_idx] += note[:end_idx - idx0] * 0.7
            
            beat += 1.0
    
    elif pat in ("walking", "melodic"):
        # Walking bass line
        scale = _get_scale_notes(config.root, config.mode, octave=2)
        beat = 0.0
        prev_idx = 0
        
        while beat * spb < config.duration_s:
            start_s = beat * spb
            dur_s = spb * 0.9
            idx0 = int(start_s * sr)
            
            # Step-wise motion with occasional leaps
            step = rng.choice([-1, 0, 1, 2], p=[0.2, 0.3, 0.35, 0.15])
            note_idx = (prev_idx + step) % len(scale)
            midi = scale[note_idx]
            
            note = _render_sf2_note(midi, dur_s, 95, instrument, sr)
            end_idx = min(n, idx0 + len(note))
            out[idx0:end_idx] += note[:end_idx - idx0] * 0.65
            
            prev_idx = note_idx
            beat += 1.0
    
    else:
        # Default: simple sustained
        note = _render_sf2_note(root_midi, config.duration_s, 85, instrument, sr)
        if len(note) > n:
            note = note[:n]
        out[:len(note)] += note * 0.5
    
    # Apply effects
    effects = [
        LowpassFilter(cutoff_frequency_hz=lp_cutoff),
        Compressor(threshold_db=-20, ratio=3.0, attack_ms=10, release_ms=100),
        LowShelfFilter(cutoff_frequency_hz=120, gain_db=2.0),
    ]
    
    board = Pedalboard(effects)
    out_2d = out.reshape(1, -1)
    out = board(out_2d, sr)[0]
    
    return out.astype(np.float32)


def _gen_pad(
    config: SynthConfig,
    layer: LayerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate pad layer."""
    sr = config.sample_rate
    n = int(config.duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    
    spb = 60.0 / config.bpm
    instrument = _resolve_instrument("pad", layer.instrument)
    brightness = layer.brightness or config.brightness
    space = layer.space or config.space
    lp_cutoff = _brightness_to_cutoff(brightness, sr)
    room_size, wet = _space_to_reverb(space)
    
    # Build chord
    scale = _get_scale_notes(config.root, config.mode, octave=4)
    
    # Chord voicing: root, fifth, octave (open voicing)
    chord_midi = [scale[0], scale[0] + 7, scale[0] + 12]
    
    # Add third for richer sound
    if len(scale) >= 3:
        chord_midi.insert(1, scale[2])  # Add third
    
    pat = layer.pattern.lower()
    
    if pat in ("sustained", "drone", "ambient"):
        # One long chord
        chord = _render_sf2_chord(chord_midi, config.duration_s, 75, instrument, sr)
        if len(chord) > n:
            chord = chord[:n]
        out[:len(chord)] += chord * 0.5
    
    elif pat in ("evolving", "slow"):
        # Chord changes every few bars
        change_every = 4.0 * spb  # 4 beats
        t = 0.0
        chord_idx = 0
        
        # Simple progression: I - IV - V - I
        progressions = [0, 3, 4, 0]
        
        while t < config.duration_s:
            dur = min(change_every * 1.2, config.duration_s - t)
            idx0 = int(t * sr)
            
            # Transpose chord
            root_shift = scale[progressions[chord_idx % len(progressions)]] - scale[0]
            transposed = [m + root_shift for m in chord_midi]
            
            chord = _render_sf2_chord(transposed, dur, 70, instrument, sr)
            end_idx = min(n, idx0 + len(chord))
            out[idx0:end_idx] += chord[:end_idx - idx0] * 0.45
            
            t += change_every
            chord_idx += 1
    
    else:
        # Default sustained
        chord = _render_sf2_chord(chord_midi, config.duration_s, 70, instrument, sr)
        if len(chord) > n:
            chord = chord[:n]
        out[:len(chord)] += chord * 0.45
    
    # Apply effects
    chorus_mix = {"mono": 0, "narrow": 0.15, "medium": 0.25, "wide": 0.35}.get(
        config.stereo_width.lower(), 0.25
    )
    
    effects = [
        LadderFilter(mode=LadderFilter.Mode.LPF12, cutoff_hz=lp_cutoff, resonance=0.15),
        Chorus(rate_hz=0.3, depth=chorus_mix, mix=chorus_mix),
        Delay(delay_seconds=0.25, feedback=0.2, mix=wet * 0.15),
        Reverb(room_size=room_size, wet_level=wet, dry_level=1.0 - wet),
    ]
    
    board = Pedalboard(effects)
    out_2d = out.reshape(1, -1)
    out = board(out_2d, sr)[0]
    
    return out.astype(np.float32)


def _gen_lead(
    config: SynthConfig,
    layer: LayerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate lead/melody layer."""
    sr = config.sample_rate
    n = int(config.duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    
    spb = 60.0 / config.bpm
    instrument = _resolve_instrument("lead", layer.instrument)
    brightness = layer.brightness or config.brightness
    space = layer.space or config.space
    lp_cutoff = _brightness_to_cutoff(brightness, sr)
    room_size, wet = _space_to_reverb(space)
    
    scale = _get_scale_notes(config.root, config.mode, octave=5)
    
    pat = layer.pattern.lower()
    
    # Determine note density
    if pat in ("sparse", "minimal"):
        note_every = 2.0  # beats
    elif pat in ("melodic", "flowing"):
        note_every = 0.5
    elif pat in ("ornate", "virtuosic"):
        note_every = 0.25
    else:
        note_every = 1.0
    
    beat = 0.0
    prev_note_idx = 0
    
    while beat * spb < config.duration_s:
        # Occasional rest
        if rng.random() < 0.15:
            beat += note_every
            continue
        
        start_s = beat * spb
        
        # Swing
        if config.swing > 0 and int(beat * 2) % 2 == 1:
            start_s += config.swing * 0.3 * spb
        
        dur_s = note_every * spb * (0.7 + 0.3 * rng.random())
        idx0 = int(start_s * sr)
        
        if idx0 >= n:
            break
        
        # Melodic motion: mostly stepwise
        step = rng.choice([-2, -1, 0, 1, 2], p=[0.05, 0.25, 0.20, 0.35, 0.15])
        note_idx = (prev_note_idx + step) % len(scale)
        midi = scale[note_idx]
        
        # Emphasize root on downbeats
        if beat % config.cycle_beats < 0.1 and rng.random() < 0.6:
            note_idx = 0
            midi = scale[0]
        
        velocity = int(80 + 30 * rng.random())
        
        # Humanize timing
        if config.humanize > 0:
            jitter = (rng.random() - 0.5) * config.humanize * 0.05 * spb
            idx0 = max(0, min(n - 1, int((start_s + jitter) * sr)))
        
        note = _render_sf2_note(midi, dur_s, velocity, instrument, sr)
        end_idx = min(n, idx0 + len(note))
        out[idx0:end_idx] += note[:end_idx - idx0] * 0.55
        
        prev_note_idx = note_idx
        beat += note_every
    
    # Apply effects
    effects = [
        LowpassFilter(cutoff_frequency_hz=lp_cutoff),
        Chorus(rate_hz=0.5, depth=0.15, mix=0.2),
        Delay(delay_seconds=0.15, feedback=0.2, mix=0.15),
        Reverb(room_size=room_size, wet_level=wet * 0.7, dry_level=1.0 - wet * 0.7),
    ]
    
    board = Pedalboard(effects)
    out_2d = out.reshape(1, -1)
    out = board(out_2d, sr)[0]
    
    return out.astype(np.float32)


def _gen_accent(
    config: SynthConfig,
    layer: LayerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate accent layer (sparse melodic hits)."""
    sr = config.sample_rate
    n = int(config.duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    
    spb = 60.0 / config.bpm
    instrument = _resolve_instrument("accent", layer.instrument)
    space = layer.space or config.space
    room_size, wet = _space_to_reverb(space)
    
    scale = _get_scale_notes(config.root, config.mode, octave=5)
    
    # Sparse hits
    hits_per_cycle = 2 + int(rng.random() * 3)
    cycle_dur = config.cycle_beats * spb
    num_cycles = int(config.duration_s / cycle_dur) + 1
    
    for cycle in range(num_cycles):
        cycle_start = cycle * cycle_dur
        
        for _ in range(hits_per_cycle):
            # Random position within cycle
            t = cycle_start + rng.random() * cycle_dur
            if t >= config.duration_s:
                continue
            
            idx0 = int(t * sr)
            dur_s = 0.3 + 0.5 * rng.random()
            
            # Random scale note
            midi = rng.choice(scale)
            velocity = int(70 + 40 * rng.random())
            
            note = _render_sf2_note(midi, dur_s, velocity, instrument, sr)
            end_idx = min(n, idx0 + len(note))
            out[idx0:end_idx] += note[:end_idx - idx0] * 0.4
    
    # Apply effects
    effects = [
        Delay(delay_seconds=0.3, feedback=0.35, mix=0.3),
        Reverb(room_size=room_size, wet_level=wet, dry_level=1.0 - wet),
    ]
    
    board = Pedalboard(effects)
    out_2d = out.reshape(1, -1)
    out = board(out_2d, sr)[0]
    
    return out.astype(np.float32)


def _gen_texture(
    config: SynthConfig,
    layer: LayerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate texture/atmosphere layer."""
    sr = config.sample_rate
    n = int(config.duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    
    instrument = _resolve_instrument("texture", layer.instrument)
    space = layer.space or config.space
    room_size, wet = _space_to_reverb(space)
    
    scale = _get_scale_notes(config.root, config.mode, octave=6)
    
    pat = layer.pattern.lower()
    
    if pat in ("shimmer", "sparkle"):
        # Random high notes
        num_notes = int(config.duration_s * 2)
        
        for _ in range(num_notes):
            t = rng.random() * config.duration_s
            idx0 = int(t * sr)
            dur_s = 0.2 + 0.4 * rng.random()
            
            midi = rng.choice(scale)
            velocity = int(40 + 30 * rng.random())
            
            note = _render_sf2_note(midi, dur_s, velocity, instrument, sr)
            end_idx = min(n, idx0 + len(note))
            out[idx0:end_idx] += note[:end_idx - idx0] * 0.25
    
    elif pat in ("breath", "wind"):
        # Subtle sustained notes with lots of reverb
        chord = _render_sf2_chord(scale[:3], config.duration_s, 40, instrument, sr)
        if len(chord) > n:
            chord = chord[:n]
        out[:len(chord)] += chord * 0.2
    
    else:
        # Default ambient
        note = _render_sf2_note(scale[0], config.duration_s, 50, instrument, sr)
        if len(note) > n:
            note = note[:n]
        out[:len(note)] += note * 0.2
    
    # Heavy effects for texture
    effects = [
        LowpassFilter(cutoff_frequency_hz=8000),
        Chorus(rate_hz=0.2, depth=0.3, mix=0.4),
        Delay(delay_seconds=0.4, feedback=0.4, mix=0.4),
        Reverb(room_size=room_size * 1.2, wet_level=min(0.9, wet * 1.5), dry_level=0.3),
    ]
    
    board = Pedalboard(effects)
    out_2d = out.reshape(1, -1)
    out = board(out_2d, sr)[0]
    
    return out.astype(np.float32)


# =============================================================================
# Main Render Pipeline
# =============================================================================

def _parse_config(raw: Dict[str, Any]) -> SynthConfig:
    """Parse raw dict into SynthConfig."""
    config = SynthConfig()
    
    # Map common aliases
    if "tempo" in raw and isinstance(raw["tempo"], str):
        tempo_map = {"glacial": 48, "slow": 68, "medium": 102, "fast": 132, "frenetic": 165}
        config.bpm = tempo_map.get(raw["tempo"].lower(), 96)
    elif "bpm" in raw:
        config.bpm = float(raw["bpm"])
    elif "tempo" in raw:
        config.bpm = float(raw["tempo"])
    
    for key in ("duration_s", "cycle_beats", "swing", "root", "mode", 
                "brightness", "space", "character", "stereo_width",
                "density", "humanize", "use_sf2", "sample_rate", "seed"):
        if key in raw:
            setattr(config, key, raw[key])
    
    # Handle alternate names
    if "duration" in raw:
        config.duration_s = float(raw["duration"])
    if "tuning" in raw:
        config.tuning_ref_hz = float(raw["tuning"])
    
    return config


def _parse_layer(raw: Dict[str, Any]) -> LayerConfig:
    """Parse raw dict into LayerConfig."""
    layer = LayerConfig()
    
    for key in ("role", "pattern", "instrument", "gain", "pan",
                "brightness", "space", "character"):
        if key in raw:
            setattr(layer, key, raw[key])
    
    # Handle style as alias for pattern
    if "style" in raw and not raw.get("pattern"):
        layer.pattern = raw["style"]
    
    return layer


def _resolve_layers(raw: Dict[str, Any], config: SynthConfig) -> List[LayerConfig]:
    """Resolve layers from config, creating defaults if needed."""
    layers = []
    
    if "layers" in raw and isinstance(raw["layers"], list):
        for layer_raw in raw["layers"]:
            if isinstance(layer_raw, dict):
                layers.append(_parse_layer(layer_raw))
    
    # If no layers specified, create defaults based on density
    if not layers:
        default_roles = ["bass", "pad", "lead", "accent", "texture"][:config.density]
        for role in default_roles:
            layers.append(LayerConfig(role=role))
    
    return layers


def _linear_pan(mono: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """Convert mono to stereo with panning."""
    pan = _clamp(pan, -1.0, 1.0)
    left_gain = math.sqrt((1 - pan) / 2)
    right_gain = math.sqrt((1 + pan) / 2)
    return np.stack([mono * left_gain, mono * right_gain], axis=0)


def _normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Normalize audio to peak level."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio * (peak / max_val)
    return audio


def render(config: Union[Dict[str, Any], SynthConfig], **kwargs) -> np.ndarray:
    """
    Render audio from config.
    
    Args:
        config: Configuration dict or SynthConfig object
        **kwargs: Override config values
        
    Returns:
        Stereo audio as numpy array, shape (2, samples), float32
    """
    # Parse config
    if isinstance(config, SynthConfig):
        cfg = config
    else:
        cfg = _parse_config(config)
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    
    # Initialize SF2 engine if needed
    if cfg.use_sf2 and HAS_SF2:
        engine = get_engine()
        if not engine.soundfonts:
            print("Warning: No soundfonts loaded. Run download_soundfonts.py first.")
            print("Falling back to oscillator synthesis.")
    
    # Get layers
    layers = _resolve_layers(config if isinstance(config, dict) else {}, cfg)
    
    # Random generator
    rng = np.random.default_rng(cfg.seed if cfg.seed else None)
    
    # Generate each layer
    sr = cfg.sample_rate
    n = int(cfg.duration_s * sr)
    mix = np.zeros((2, n), dtype=np.float32)
    
    generators = {
        "bass": _gen_bass,
        "pad": _gen_pad,
        "lead": _gen_lead,
        "voice": _gen_lead,  # Same as lead with different default instrument
        "accent": _gen_accent,
        "texture": _gen_texture,
    }
    
    for layer in layers:
        gen_fn = generators.get(layer.role, _gen_texture)
        
        try:
            mono = gen_fn(cfg, layer, rng)
        except Exception as e:
            print(f"Error generating {layer.role}: {e}")
            mono = np.zeros(n, dtype=np.float32)
        
        # Ensure correct length
        if len(mono) < n:
            mono = np.pad(mono, (0, n - len(mono)))
        elif len(mono) > n:
            mono = mono[:n]
        
        # Pan and mix
        stereo = _linear_pan(mono, layer.pan)
        mix += stereo * layer.gain
    
    # Master processing
    space = cfg.space
    room_size, wet = _space_to_reverb(space)
    
    master_effects = [
        HighpassFilter(cutoff_frequency_hz=30),
        Compressor(threshold_db=-14, ratio=2.0, attack_ms=15, release_ms=150),
        Reverb(room_size=room_size * 0.5, wet_level=wet * 0.2, dry_level=0.9),
        Limiter(threshold_db=-1.0),
    ]
    
    master = Pedalboard(master_effects)
    mix = master(mix, sr)
    
    # Normalize
    mix = _normalize(mix, peak=0.95)
    
    return mix.astype(np.float32)


def write_wav(path: str, audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    """Write audio to WAV file."""
    import soundfile as sf
    
    # Ensure channels-last for soundfile
    if audio.ndim == 2 and audio.shape[0] == 2:
        audio = audio.T
    
    sf.write(path, audio, sample_rate)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Render audio from JSON config")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--out", type=str, default="output.wav", help="Output WAV path")
    parser.add_argument("--duration", type=float, help="Override duration (seconds)")
    parser.add_argument("--no-sf2", action="store_true", help="Disable SF2 samples")
    parser.add_argument("--test", action="store_true", help="Run test render")
    args = parser.parse_args()
    
    if args.test:
        print("Running test render...")
        print(f"SF2 available: {HAS_SF2}")
        
        test_config = {
            "bpm": 96,
            "duration_s": 10,
            "root": "d",
            "mode": "dorian",
            "space": "hall",
            "use_sf2": not args.no_sf2,
            "layers": [
                {"role": "bass", "pattern": "drone", "instrument": "acoustic_bass"},
                {"role": "pad", "pattern": "sustained", "instrument": "string_ensemble"},
                {"role": "lead", "pattern": "melodic", "instrument": "sitar"},
                {"role": "accent", "pattern": "sparse", "instrument": "vibraphone"},
            ]
        }
        
        print(f"Config: {json.dumps(test_config, indent=2)}")
        audio = render(test_config)
        write_wav(args.out, audio)
        print(f"Wrote: {args.out}")
    
    elif args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        
        if args.duration:
            config["duration_s"] = args.duration
        if args.no_sf2:
            config["use_sf2"] = False
        
        audio = render(config)
        write_wav(args.out, audio)
        print(f"Wrote: {args.out}")
    
    else:
        parser.print_help()


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
