# """
# new_llm_to_synth.py

# LLM-facing schema + "self-healing" parsing utilities for the culture-general synth in `new_synth.py`.

# Design goals:
# - Minimal surface area + high RoI
# - Canonical pitch set is `scale_intervals` (microtones allowed); `mode` is fallback only
# - Layering is first-class: multiple layers with per-layer gain + optional cycle/BPM overrides
# - Fail-safe: missing/invalid values coerce toward drone/modal + pentatonic/minor-ish defaults

# This file deliberately does NOT hard-code culture enums. A request like "Papua New Guinea"
# should be expressible by:
# - choosing a scale_intervals list (or leaving it null)
# - choosing per-layer material/ornament/harmonic_motion/rhythm cycle overrides
# - letting the synth render "pleasantly weird" if uncertain

# If you already have an LLM that emits JSON, just feed that JSON into:
#   - `LayeredSongConfig.model_validate_json(...)`
#   - then `render_layered_config(...)`

# We keep any LLM integration optional.
# """

# from __future__ import annotations

# from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

# import json
# import re

# from pydantic import BaseModel, Field, field_validator, model_validator

# import new_synth


# # =============================================================================
# # Schema primitives (LLM-friendly enums)
# # =============================================================================

# NoteName = Literal["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]

# ModeName = Literal[
#     "major",
#     "minor",
#     "dorian",
#     "phrygian",
#     "lydian",
#     "mixolydian",
#     "harmonic_minor",
#     "melodic_minor",
#     "pentatonic_major",
#     "pentatonic_minor",
#     "blues",
#     "chromatic",
# ]

# BrightnessName = Literal["very_dark", "dark", "medium", "bright", "very_bright"]
# SpaceName = Literal["dry", "intimate", "room", "hall", "cathedral", "infinite"]
# CharacterName = Literal["clean", "warm", "saturated", "distorted"]
# MaterialName = Literal["digital", "warm_analog", "wood", "metal", "glass", "breath", "skin"]
# StereoWidthName = Literal["mono", "narrow", "medium", "wide", "immersive"]
# HarmonicMotionName = Literal["drone", "minimal", "slow", "medium", "active"]
# OrnamentName = Literal["none", "subtle", "moderate", "expressive", "virtuosic"]

# LayerRole = Literal["bass", "pad", "lead", "rhythm", "texture", "voice", "accent"]


# # =============================================================================
# # Pydantic models
# # =============================================================================


# class GlobalMusicConfig(BaseModel):
#     """
#     Global (canonical) configuration.

#     Single source of truth (canonical):
#     - root
#     - scale_intervals (if present) else mode
#     - bpm (layers may override, but engine will quantize to safe ratios)
#     - cycle_beats (layers may override)

#     Derived (engine-side):
#     - actual chord voicings, phrase structure, etc.
#     """

#     # Meta
#     justification: str = Field(default="", max_length=2000)

#     # Render
#     seed: int = Field(default=0, ge=0, le=2_000_000_000)
#     duration_s: float = Field(default=16.0, ge=2.0, le=300.0)
#     sample_rate: int = Field(default=new_synth.DEFAULT_SAMPLE_RATE, ge=8_000, le=192_000)

#     # TIME
#     bpm: float = Field(default=96.0, ge=30.0, le=220.0, description="Pulse tempo in beats per minute.")
#     cycle_beats: int = Field(
#         default=16, ge=2, le=64, description="Loop length in beats (not time signature)."
#     )
#     subdivision: Literal[2, 3, 4, 5] = Field(
#         default=2, description="Pulses per beat: 2=8ths, 3=triplets, 4=16ths, 5=quintuplet-ish."
#     )
#     swing: float = Field(default=0.0, ge=0.0, le=0.75)

#     # PITCH
#     root: NoteName = Field(default="c")
#     mode: ModeName = Field(default="minor", description="Fallback if scale_intervals is not provided.")
#     scale_intervals: Optional[List[float]] = Field(
#         default=None,
#         description=(
#             "Canonical pitch set: semitone offsets from root (microtones allowed). "
#             "If provided, overrides `mode`."
#         ),
#     )
#     tuning_ref_hz: float = Field(default=440.0, ge=400.0, le=480.0, description="A4 reference frequency.")

#     # HARMONY POLICY
#     harmonic_motion: HarmonicMotionName = Field(
#         default="minimal", description="How much harmony moves. Drone/minimal is safest cross-culturally."
#     )
#     chord_color: Literal["unison", "open5", "triad", "seventh", "cluster"] = Field(
#         default="open5", description="Voicing complexity. open5 is the safest cross-culturally."
#     )

#     # TIMBRE / VIBE
#     brightness: BrightnessName = Field(default="medium")
#     space: SpaceName = Field(default="room")
#     stereo_width: StereoWidthName = Field(default="medium")
#     character: CharacterName = Field(default="clean")
#     material: MaterialName = Field(default="warm_analog")
#     density: Literal[1, 2, 3, 4, 5, 6] = Field(default=4)

#     # EXPRESSION
#     humanize: float = Field(
#         default=0.15, ge=0.0, le=1.0, description="Timing/velocity looseness."
#     )
#     ornament_intensity: OrnamentName = Field(
#         default="subtle",
#         description="Pitch slides/vibrato intensity. Critical for raga/maqam/chant-like styles.",
#     )

#     @field_validator("scale_intervals")
#     @classmethod
#     def _validate_scale_intervals(cls, v: Optional[List[float]]) -> Optional[List[float]]:
#         # Keep the schema permissive; engine does the heavy "self-heal".
#         if v is None:
#             return None
#         if not isinstance(v, list) or len(v) == 0:
#             return None
#         out: List[float] = []
#         for x in v:
#             try:
#                 out.append(float(x))
#             except Exception:
#                 continue
#         return out or None


# class LayerConfig(BaseModel):
#     """
#     A single layer/track.

#     Intent:
#     - Only a handful of overrides
#     - Gain/pan are first-class
#     - Optional BPM/cycle overrides allow polyrhythm & tempo ratios

#     NOTE: We do NOT allow per-layer root/scale overrides (too easy to create conflicts).
#     Harmony safety comes from shared pitch set in GlobalMusicConfig.
#     """

#     role: LayerRole = Field(default="pad")
#     pattern: str = Field(
#         default="auto",
#         max_length=48,
#         description=(
#             "Free-form string. The synth will treat unknown patterns as 'auto'. "
#             "Examples: drone, pulse, melody, chant, clave, polyrhythm, air, grain, sparkle."
#         ),
#     )

#     # Mixing
#     gain: float = Field(default=1.0, ge=0.0, le=2.0, description="Relative gain (normalized across layers).")
#     pan: float = Field(default=0.0, ge=-1.0, le=1.0)

#     # Safe time overrides
#     bpm: Optional[float] = Field(default=None, ge=20.0, le=260.0, description="If set, quantized to safe ratios.")
#     cycle_beats: Optional[int] = Field(default=None, ge=2, le=64)

#     # Optional vibe overrides (small, but high RoI)
#     material: Optional[MaterialName] = None
#     brightness: Optional[BrightnessName] = None
#     space: Optional[SpaceName] = None
#     character: Optional[CharacterName] = None
#     ornament_intensity: Optional[OrnamentName] = None

#     @model_validator(mode="after")
#     def _sanitize_text(self) -> "LayerConfig":
#         # Keep pattern from being pathological (LLM can be verbose).
#         if self.pattern:
#             self.pattern = self.pattern.strip().lower()
#             # truncate very long patterns; synth ignores unknown strings anyway
#             if len(self.pattern) > 48:
#                 self.pattern = self.pattern[:48]
#         return self


# class LayeredSongConfig(BaseModel):
#     """
#     Wrapper config:
#     - One global base
#     - 1..6 layers (kept small intentionally; >6 layers often degrades cohesion)
#     """

#     global_config: GlobalMusicConfig = Field(default_factory=GlobalMusicConfig, alias="global")
#     layers: List[LayerConfig] = Field(default_factory=list, min_length=0, max_length=8)

#     @model_validator(mode="after")
#     def _ensure_layers(self) -> "LayeredSongConfig":
#         # If the LLM omitted layers, we'll rely on synth auto-layering by leaving it empty.
#         # If layers were provided, normalize layer gains (self-heal).
#         if self.layers:
#             gains = [max(0.0, float(l.gain)) for l in self.layers]
#             s = sum(gains)
#             if s <= 1e-9:
#                 # all zeros -> reset to 1s
#                 for l in self.layers:
#                     l.gain = 1.0
#             else:
#                 for l, g in zip(self.layers, gains):
#                     l.gain = g / s
#         return self

#     def to_render_dict(self) -> Dict[str, Any]:
#         """
#         Convert into a dict accepted by new_synth.render().

#         We pass through enums as strings; new_synth has its own robust coercions.
#         """
#         g = self.global_config.model_dump()
#         # new_synth expects 'layers' at top-level
#         if self.layers:
#             g["layers"] = [ly.model_dump(exclude_none=True) for ly in self.layers]
#         return g


# class ScoredLayeredSongConfig(BaseModel):
#     config: LayeredSongConfig
#     score: int = Field(default=70, ge=0, le=100)


# # =============================================================================
# # Parsing helpers
# # =============================================================================


# _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", re.DOTALL)


# def extract_json_object(text: str) -> str:
#     """Extract the first JSON object/array from an LLM response.

#     Supports:
#     - Markdown code fences (```json ... ```)
#     - Raw text with leading/trailing commentary

#     We intentionally keep this permissive; Pydantic + engine self-heal will do the rest.
#     """
#     m = _JSON_FENCE_RE.search(text)
#     if m:
#         return m.group(1)

#     # Otherwise, find the earliest of '{' or '[' and the matching closing '}' or ']'.
#     i_obj = text.find("{")
#     i_arr = text.find("[")
#     starts = [i for i in (i_obj, i_arr) if i != -1]
#     if not starts:
#         return text

#     start = min(starts)
#     opener = text[start]
#     closer = "}" if opener == "{" else "]"
#     end = text.rfind(closer)
#     if end != -1 and end > start:
#         return text[start : end + 1]

#     return text[start:]


# def parse_layered_config_json(text: str) -> LayeredSongConfig:
#     """Parse LLM output into a validated LayeredSongConfig.

#     Self-healing behavior:
#     - Accepts either {"global": {...}, "layers": [...]} or a flat global dict.
#     - Accepts wrappers like {"config": {...}} or {"configs": [...]}.
#     - If given a list of candidates, picks the highest-scoring item if possible.
#     - Unknown extra keys are ignored by Pydantic.
#     """
#     js = extract_json_object(text)
#     data = json.loads(js)

#     # If the LLM returns a list of candidates, pick one.
#     if isinstance(data, list) and data:
#         def score_of(item):
#             if not isinstance(item, dict):
#                 return -1
#             # common fields used in different drafts
#             for k in ("score", "confidence"):
#                 if k in item:
#                     try:
#                         return float(item[k])
#                     except Exception:
#                         pass
#             return 0
#         best = max(data, key=score_of)
#         data = best

#     # Allow wrappers: {"configs": [...]} / {"candidates": [...]}.
#     if isinstance(data, dict):
#         for k in ("configs", "candidates", "variants"):
#             if k in data and isinstance(data[k], list) and data[k]:
#                 # recurse via the same selection logic
#                 data = data[k]
#                 if isinstance(data, list) and data:
#                     def score_of(item):
#                         if not isinstance(item, dict):
#                             return -1
#                         for kk in ("score", "confidence"):
#                             if kk in item:
#                                 try:
#                                     return float(item[kk])
#                                 except Exception:
#                                     pass
#                         return 0
#                     data = max(data, key=score_of)
#                 break

#     # Allow top-level "config": {...}
#     if isinstance(data, dict) and "config" in data and isinstance(data["config"], dict):
#         data = data["config"]

#     # At this point, data should be a dict.
#     if not isinstance(data, dict):
#         raise ValueError("Parsed JSON is not an object. Expected a config dict.")

#     # Accept either wrapper style {global,layers} OR flat global dict.
#     if "global" in data or "global_config" in data:
#         return LayeredSongConfig.model_validate(data)

#     # Flat: treat as global, and optionally lift top-level layers.
#     layers = None
#     if "layers" in data:
#         layers = data.get("layers")
#         # Remove to avoid pydantic complaining about unexpected keys on GlobalMusicConfig.
#         data = dict(data)
#         data.pop("layers", None)

#     cfg = LayeredSongConfig(global_config=GlobalMusicConfig.model_validate(data), layers=[])
#     if layers is not None:
#         cfg = LayeredSongConfig(global_config=cfg.global_config, layers=layers)  # pydantic will validate layers
#     return cfg


# # =============================================================================
# # Rendering helpers
# # =============================================================================


# def render_layered_config(cfg: LayeredSongConfig) -> new_synth.np.ndarray:
#     """
#     Render to stereo channels-first float32 array.
#     """
#     render_dict = cfg.to_render_dict()
#     return new_synth.render(render_dict, sample_rate=cfg.global_config.sample_rate)


# def write_wav(path: str, audio_cf: new_synth.np.ndarray, sr: int) -> None:
#     new_synth.write_wav(path, audio_cf, sr)


# # =============================================================================
# # OPTIONAL: rule-based fallback (useful for testing without an LLM)
# # =============================================================================


# _KEYWORD_PATCHES: List[Tuple[re.Pattern, Dict[str, Any]]] = [
#     # Drone/modal traditions
#     (re.compile(r"\braga\b|\bindian\b|\bsitar\b|\btanpura\b", re.I), {"harmonic_motion": "drone", "ornament_intensity": "virtuosic", "space": "hall"}),
#     (re.compile(r"\bmaqam\b|\barabic\b|\bou[d]?\\b|\badhan\b|\baazan\b|\bazan\b", re.I), {"harmonic_motion": "drone", "ornament_intensity": "expressive", "space": "cathedral", "brightness": "bright"}),
#     (re.compile(r"\bchant\b|\bmonophonic\b|\bprayer\b", re.I), {"harmonic_motion": "drone", "ornament_intensity": "expressive", "density": 3}),
#     (re.compile(r"\begypt\b|\bancient\b", re.I), {"mode": "phrygian", "harmonic_motion": "drone", "space": "cathedral", "brightness": "dark"}),
#     # Rhythmic cells
#     (re.compile(r"\bclave\b|\bcuban\b|\bsalsa\b", re.I), {"bpm": 120, "harmonic_motion": "medium", "density": 5}),
#     (re.compile(r"\bgamelan\b|\bindonesian\b|\bbali\b", re.I), {"material": "metal", "mode": "pentatonic_minor", "harmonic_motion": "minimal", "brightness": "bright", "space": "hall"}),
#     # Modern genres
#     (re.compile(r"\bcyberpunk\b|\bsynthwave\b", re.I), {"bpm": 100, "material": "digital", "character": "saturated", "brightness": "bright", "space": "room"}),
#     (re.compile(r"\bjazz\b", re.I), {"mode": "dorian", "harmonic_motion": "active", "chord_color": "seventh", "humanize": 0.25}),
#     (re.compile(r"\bmario\b|\b8-bit\b|\bchiptune\b", re.I), {"bpm": 140, "material": "digital", "brightness": "very_bright", "space": "intimate", "density": 4}),
# ]


# def vibe_to_layered_config_fallback(vibe: str, *, seed: int = 0) -> LayeredSongConfig:
#     """
#     A tiny heuristic mapper for when you don't want to run an LLM.
#     """
#     g = GlobalMusicConfig(seed=seed)

#     # Apply patches
#     for pat, patch in _KEYWORD_PATCHES:
#         if pat.search(vibe):
#             g = g.model_copy(update=patch)

#     # Suggest layers based on vibe (still optional; synth can auto-layer if omitted)
#     layers: List[LayerConfig] = []
#     v = vibe.lower()

#     if "aazan" in v or "adhan" in v or "chant" in v:
#         layers = [
#             LayerConfig(role="pad", pattern="drone", gain=0.9, pan=0.0, material="breath"),
#             LayerConfig(role="voice", pattern="chant", gain=0.85, pan=-0.05, material="breath"),
#             LayerConfig(role="texture", pattern="air", gain=0.35, pan=0.15),
#         ]
#     elif "gamelan" in v:
#         layers = [
#             LayerConfig(role="pad", pattern="drone", gain=0.8, pan=0.0, material="metal"),
#             LayerConfig(role="accent", pattern="sparkle", gain=0.55, pan=0.2, material="metal"),
#             LayerConfig(role="rhythm", pattern="polyrhythm", gain=0.55, pan=0.0),
#             LayerConfig(role="texture", pattern="air", gain=0.35, pan=-0.15),
#         ]
#     elif "clave" in v or "cuban" in v:
#         layers = [
#             LayerConfig(role="bass", pattern="pulse", gain=0.7, pan=0.0),
#             LayerConfig(role="pad", pattern="chordal", gain=0.75, pan=0.0),
#             LayerConfig(role="rhythm", pattern="clave", gain=0.8, pan=0.0, cycle_beats=8),
#             LayerConfig(role="lead", pattern="melody", gain=0.65, pan=0.15),
#         ]

#     return LayeredSongConfig(global_config=g, layers=layers)


# # =============================================================================
# # CLI
# # =============================================================================


# def _main() -> None:  # pragma: no cover
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Parse an LLM output text file (containing JSON) and render it with new_synth."
#     )
#     parser.add_argument(
#         "--in",
#         dest="in_path",
#         type=str,
#         default=None,
#         help="Path to a text file containing LLM output. (JSON can be fenced or embedded)",
#     )
#     parser.add_argument(
#         "--out",
#         dest="out_path",
#         type=str,
#         required=True,
#         help="Output WAV path.",
#     )
#     parser.add_argument("--duration", type=float, default=None, help="Override duration seconds")
#     parser.add_argument("--sr", type=int, default=None, help="Override sample rate")
#     parser.add_argument(
#         "--vibe",
#         type=str,
#         default=None,
#         help="Optional fallback: generate a config from a vibe string (no LLM).",
#     )
#     parser.add_argument("--seed", type=int, default=0, help="Seed for fallback mode")

#     args = parser.parse_args()

#     if args.in_path:
#         with open(args.in_path, "r", encoding="utf-8") as f:
#             txt = f.read()
#         cfg = parse_layered_config_json(txt)
#     else:
#         if not args.vibe:
#             raise SystemExit("Provide either --in <llm_output.txt> or --vibe <text prompt>.")
#         cfg = vibe_to_layered_config_fallback(args.vibe, seed=int(args.seed))

#     # Apply CLI overrides
#     if args.duration is not None:
#         cfg.global_config.duration_s = float(args.duration)
#     if args.sr is not None:
#         cfg.global_config.sample_rate = int(args.sr)

#     audio = render_layered_config(cfg)
#     write_wav(args.out_path, audio, cfg.global_config.sample_rate)
#     print(f"Wrote {args.out_path} (sr={cfg.global_config.sample_rate}, duration_s={cfg.global_config.duration_s})")


# if __name__ == "__main__":  # pragma: no cover
#     _main()

"""
new_llm_to_synth.py

LLM-facing schema + parsing helpers for the culture-general config used by new_synth.py.

Design goals
- LLM-friendly: small number of knobs, most categorical; pitch can be specified as:
  (a) preset name, (b) scale_intervals floats, or (c) EDO degrees + offsets (canonical).
- Hard to break: validation clamps, self-healing, and safe defaults.
- Layering: multiple layers with per-layer gain/pan, rational tempo ratios, and optional cycle overrides.

This module does NOT call any online services. It just validates + normalizes JSON-ish output from an LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple
import json
import re

from pydantic import BaseModel, Field, AliasChoices, model_validator

import new_synth

# =============================================================================
# Enums (keep aligned with new_synth)
# =============================================================================

TempoName = Literal["glacial", "slow", "medium", "fast", "frenetic"]

NoteName = Literal["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]

ScalePresetName = Literal[
    "major",
    "minor",
    "dorian",
    "phrygian",
    "lydian",
    "mixolydian",
    "aeolian",
    "locrian",
    "harmonic_minor",
    "melodic_minor",
    "pentatonic_major",
    "pentatonic_minor",
    "blues",
    "whole_tone",
    "chromatic",
]

HarmonyTextureName = Literal["mono", "drone_melody", "heterophony", "safe_polyphony", "chordal"]
ConsonanceName = Literal["strict", "gentle", "open"]
HarmonicMotionName = Literal["auto", "drone", "minimal", "slow", "medium", "active"]
VoicingName = Literal["unison", "open5", "triad", "seventh", "cluster"]

BrightnessName = Literal["very_dark", "dark", "medium", "bright", "very_bright"]
SpaceName = Literal["dry", "intimate", "room", "hall", "cathedral", "infinite"]
StereoWidthName = Literal["mono", "narrow", "medium", "wide", "immersive"]
CharacterName = Literal["clean", "warm", "saturated", "distorted"]

OrnamentName = Literal["none", "subtle", "moderate", "expressive", "virtuosic"]

TimbreFamilyName = Literal[
    "sine_like",
    "reed_like",
    "plucked_like",
    "bowed_like",
    "metallic_like",
    "membrane_like",
    "noise_like",
    "voice_like",
]

LayerRole = Literal["bass", "pad", "lead", "rhythm", "texture", "voice", "accent"]

TempoRatioName = Literal[
    "1/1",
    "2/1",
    "1/2",
    "3/2",
    "2/3",
    "4/3",
    "3/4",
    "5/4",
    "4/5",
]


# =============================================================================
# Schema
# =============================================================================


class GlobalMusicConfig(BaseModel):
    """
    Global/base configuration.

    Canonical pitch is (edo_divisions, period_cents, scale_degrees, degree_offsets_cents).
    Convenience fields:
      - scale_preset/mode (mapped to intervals)
      - scale_intervals (float semitone offsets; converted inside new_synth)
    """

    # META (optional; useful for debugging / tracing)
    justification: str = Field(default="", max_length=2000)

    # RENDER
    seed: int = Field(default=0, ge=0, le=2_000_000_000)
    duration_s: float = Field(default=16.0, ge=2.0, le=300.0)
    sample_rate: int = Field(default=new_synth.DEFAULT_SAMPLE_RATE, ge=8_000, le=192_000)

    # TIME
    tempo: TempoName = Field(default="medium", description="Categorical tempo (LLM-friendly).")
    bpm: Optional[float] = Field(
        default=None,
        ge=30.0,
        le=220.0,
        description="Optional BPM override. If omitted, derived from `tempo`.",
    )
    cycle_beats: int = Field(default=16, ge=2, le=64)
    subdivision: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Pulses per beat. 2=straight, 3=compound/swing grid, 4=16th grid.",
    )
    swing: float = Field(default=0.0, ge=0.0, le=0.75)

    # PITCH ANCHOR
    root: NoteName = Field(default="c", validation_alias=AliasChoices("root", "root_note"))
    tuning_ref_hz: float = Field(default=440.0, ge=400.0, le=480.0, description="A4 tuning reference.")

    # ADVANCED TUNING (optional; safe defaults)
    period_cents: float = Field(default=1200.0, ge=800.0, le=2400.0)
    edo_divisions: int = Field(default=12, ge=3, le=72)

    # SCALE (choose one; precedence handled by new_synth)
    scale_preset: ScalePresetName = Field(
        default="minor",
        validation_alias=AliasChoices("scale_preset", "mode", "preset"),
        description="Generic preset scales. For unseen cultures, provide `scale_intervals` or degrees+offsets.",
    )
    scale_intervals: Optional[List[float]] = Field(
        default=None,
        description="LLM-friendly semitone offsets (floats allowed for microtones). Overrides `scale_preset` if set.",
    )
    scale_degrees: Optional[List[int]] = Field(
        default=None,
        description="EDO degrees (0..edo_divisions-1). Overrides `scale_intervals` if set.",
    )
    degree_offsets_cents: Dict[int, float] = Field(
        default_factory=dict,
        description="Optional cents offsets per degree. Use for quarter-tone-ish tweaks without changing EDO.",
    )

    # ASYMMETRY + HIERARCHY (critical for raga/maqam-ish behaviour)
    ascending_degrees: Optional[List[int]] = Field(default=None)
    descending_degrees: Optional[List[int]] = Field(default=None)
    emphasis_degrees: Optional[List[int]] = Field(default=None)
    avoid_degrees: Optional[List[int]] = Field(default=None)

    # HARMONY POLICY (culture-general)
    texture: HarmonyTextureName = Field(default="drone_melody")
    consonance: ConsonanceName = Field(default="gentle")
    harmonic_motion: HarmonicMotionName = Field(
        default="auto",
        description="Horizontal harmonic movement. Auto chooses based on texture + density.",
    )
    voicing: VoicingName = Field(
        default="open5",
        validation_alias=AliasChoices("voicing", "chord_color"),
    )

    # TIMBRE / VIBE
    brightness: BrightnessName = Field(default="medium")
    space: SpaceName = Field(default="room")
    stereo_width: StereoWidthName = Field(default="medium", validation_alias=AliasChoices("stereo_width", "stereo"))
    character: CharacterName = Field(default="clean")
    timbre_family: TimbreFamilyName = Field(
        default="bowed_like",
        validation_alias=AliasChoices("timbre_family", "family", "material"),
        description="How the sound is excited (more universal than instrument lists).",
    )
    density: int = Field(default=4, ge=1, le=6)

    # EXPRESSION
    humanize: float = Field(default=0.15, ge=0.0, le=1.0)
    ornament_intensity: OrnamentName = Field(default="subtle")

    @model_validator(mode="after")
    def _self_heal(self) -> "GlobalMusicConfig":
        # Derive bpm from tempo if not provided
        if self.bpm is None:
            self.bpm = float(new_synth.TEMPO_TO_BPM.get(self.tempo, 102.0))

        # Clamp subdivision to nearby common values (keeps synth stable)
        if self.subdivision not in (2, 3, 4, 5):
            self.subdivision = int(min((2, 3, 4, 5), key=lambda v: abs(v - int(self.subdivision))))

        # If scale_degrees is provided, ensure root is implicitly present (the engine will also heal)
        if self.scale_degrees is not None and 0 not in self.scale_degrees:
            self.scale_degrees = [0] + list(self.scale_degrees)

        # Very conservative: if user sets chordal + non-12 EDO, steer texture away (engine will also).
        if self.texture == "chordal" and (self.edo_divisions != 12 or abs(self.period_cents - 1200.0) > 1e-6):
            self.texture = "drone_melody"
            self.voicing = "open5"
            self.harmonic_motion = "minimal"

        return self


class LayerConfig(BaseModel):
    """
    A single layer/track.

    Layers can override:
    - rhythmic cycle length (`cycle_beats`)
    - tempo via safe rational ratios (`tempo_ratio`) or a bpm override (quantized internally)
    - timbre/vibe knobs (small set)

    Layers should NOT override pitch. Harmony safety comes from every layer sharing the same global pitch system.
    """

    role: LayerRole = Field(default="pad")
    pattern: str = Field(default="auto", max_length=48)

    gain: float = Field(default=1.0, ge=0.0, le=2.0)
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)

    tempo_ratio: Optional[TempoRatioName] = Field(default=None)
    bpm: Optional[float] = Field(default=None, ge=20.0, le=260.0)
    cycle_beats: Optional[int] = Field(default=None, ge=2, le=64)

    # Optional timbre overrides (keep tiny)
    timbre_family: Optional[TimbreFamilyName] = Field(default=None, validation_alias=AliasChoices("timbre_family", "family", "material"))
    brightness: Optional[BrightnessName] = None
    space: Optional[SpaceName] = None
    character: Optional[CharacterName] = None
    ornament_intensity: Optional[OrnamentName] = None

    @model_validator(mode="after")
    def _self_heal(self) -> "LayerConfig":
        # Normalize pattern whitespace
        self.pattern = (self.pattern or "auto").strip().lower()[:48]
        return self


class LayeredSongConfig(BaseModel):
    """
    Global config + optional layers.

    If `layers` is omitted/empty, new_synth auto-layering kicks in using (texture, density).
    """

    global_config: GlobalMusicConfig = Field(
        default_factory=GlobalMusicConfig,
        validation_alias=AliasChoices("global_config", "global"),
    )
    layers: List[LayerConfig] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _normalize_layer_gains(self) -> "LayeredSongConfig":
        if not self.layers:
            return self

        gains = [max(0.0, float(l.gain)) for l in self.layers]
        s = sum(gains)
        if s <= 1e-9:
            gains = [1.0 for _ in gains]
            s = float(len(gains))
        gains = [g / s for g in gains]
        for l, gn in zip(self.layers, gains):
            l.gain = float(gn)
        return self

    def to_render_dict(self) -> Dict[str, Any]:
        g = self.global_config.model_dump(exclude_none=True)
        if self.layers:
            g["layers"] = [ly.model_dump(exclude_none=True) for ly in self.layers]
        return g


class ScoredLayeredSongConfig(BaseModel):
    config: LayeredSongConfig
    score: int = Field(default=70, ge=0, le=100)


# =============================================================================
# Parsing helpers
# =============================================================================


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> str:
    """
    LLMs sometimes wrap JSON in markdown fences.
    This pulls out the first {...} object if present.
    """
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_layered_config_json(text: str) -> LayeredSongConfig:
    """
    Parse LLM output into a validated LayeredSongConfig.

    Self-healing behaviour:
    - Accepts either {"global": {...}, "layers": [...]} or {"global_config": {...}, ...}
    - Accepts top-level {"config": {...}}
    - Unknown keys are ignored by pydantic (extra="ignore" default)
    """
    js = extract_json_object(text)
    data = json.loads(js)

    if isinstance(data, dict) and "config" in data and isinstance(data["config"], dict):
        data = data["config"]

    return LayeredSongConfig.model_validate(data)


# =============================================================================
# Rendering helpers
# =============================================================================


def render_layered_config(cfg: LayeredSongConfig) -> new_synth.np.ndarray:
    """
    Render to stereo channels-first float32 array.
    """
    render_dict = cfg.to_render_dict()
    return new_synth.render(render_dict, sample_rate=cfg.global_config.sample_rate)


def write_wav(path: str, audio_cf: new_synth.np.ndarray, sr: int) -> None:
    new_synth.write_wav(path, audio_cf, sr)


# =============================================================================
# CLI demo (rule-based, culture-neutral fallback)
# =============================================================================


_KEYWORD_PATCHES: List[Tuple[re.Pattern, Dict[str, Any]]] = [
    (re.compile(r"\bdrone\b|\bchant\b|\bmonophonic\b", re.I), {"texture": "mono", "harmonic_motion": "minimal", "ornament_intensity": "expressive"}),
    (re.compile(r"\bpolyrhythm\b|\bcomplex rhythm\b", re.I), {"density": 5, "harmonic_motion": "slow"}),
    (re.compile(r"\bbright\b|\bshimmer\b", re.I), {"brightness": "very_bright"}),
    (re.compile(r"\bdark\b|\bnoir\b", re.I), {"brightness": "dark", "space": "hall"}),
]


def vibe_to_layered_config_fallback(vibe: str, *, seed: int = 0) -> LayeredSongConfig:
    """
    A tiny heuristic mapper for when you don't want to run an LLM.

    Intentionally culture-neutral: no "raga"/"maqam"/etc hardcoding here.
    """
    g = GlobalMusicConfig(seed=seed)

    for pat, patch in _KEYWORD_PATCHES:
        if pat.search(vibe):
            g = g.model_copy(update=patch)

    layers: List[LayerConfig] = []
    v = vibe.lower()

    if "mono" in v or "chant" in v:
        layers = [
            LayerConfig(role="voice", pattern="chant", gain=0.9, pan=0.0, timbre_family="voice_like"),
            LayerConfig(role="texture", pattern="air", gain=0.35, pan=0.15),
        ]
    elif "polyrhythm" in v:
        layers = [
            LayerConfig(role="pad", pattern="drone", gain=0.7, pan=0.0),
            LayerConfig(role="rhythm", pattern="polyrhythm", gain=0.8, pan=0.0, cycle_beats=12, tempo_ratio="1/1"),
            LayerConfig(role="rhythm", pattern="pulse", gain=0.5, pan=0.0, cycle_beats=16, tempo_ratio="4/3"),
            LayerConfig(role="lead", pattern="melody", gain=0.6, pan=0.1),
        ]

    return LayeredSongConfig(global_config=g, layers=layers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render a vibe using the final schema + new_synth.")
    parser.add_argument("vibe", type=str, help="Text prompt / vibe description")
    parser.add_argument("--out", type=str, default="render.wav")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=16.0)
    args = parser.parse_args()

    cfg = vibe_to_layered_config_fallback(args.vibe, seed=args.seed)
    cfg.global_config.duration_s = args.duration

    audio = render_layered_config(cfg)
    write_wav(args.out, audio, cfg.global_config.sample_rate)
    print(f"Wrote {args.out}")