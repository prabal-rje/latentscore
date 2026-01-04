# """
# Vibe templates for new_synth.py / new_llm_to_synth.py

# Each template is a dict that can be passed directly to:
#   - new_llm_to_synth.LayeredSongConfig.model_validate({"global_config": template, "layers": [...]})
#   - new_synth.render(template)
# """

# TEMPLATES = {

#     # =========================================================================
#     # BUBBLE GUM - Sweet, pink, pop, sparkly, carefree
#     # =========================================================================
#     "bubble_gum": {
#         "global_config": {
#             "justification": "Sweet bubblegum pop: bright major key, bouncy tempo, clean synths, light and airy with sparkle accents.",
#             "tempo": "fast",
#             "bpm": 128.0,
#             "root": "f",
#             "scale_preset": "major",
#             "texture": "safe_polyphony",
#             "consonance": "strict",
#             "harmonic_motion": "active",
#             "voicing": "triad",
#             "brightness": "very_bright",
#             "space": "room",
#             "stereo_width": "wide",
#             "character": "clean",
#             "timbre_family": "plucked_like",
#             "density": 5,
#             "humanize": 0.08,
#             "ornament_intensity": "subtle",
#             "swing": 0.0,
#             "cycle_beats": 16,
#             "subdivision": 2,
#             "duration_s": 20.0,
#         },
#         "layers": [
#             {"role": "bass", "pattern": "bounce", "gain": 0.7, "pan": 0.0, "timbre_family": "sine_like"},
#             {"role": "pad", "pattern": "shimmer", "gain": 0.5, "pan": 0.0, "brightness": "very_bright"},
#             {"role": "lead", "pattern": "melody", "gain": 0.8, "pan": 0.1, "timbre_family": "plucked_like"},
#             {"role": "rhythm", "pattern": "fourfloor", "gain": 0.6, "pan": 0.0},
#             {"role": "accent", "pattern": "sparkle", "gain": 0.4, "pan": 0.2, "timbre_family": "metallic_like"},
#         ],
#     },

#     # =========================================================================
#     # INDIAN CULTURE - Raga-like, drone, ornamental, spiritual, meditative
#     # =========================================================================
#     "indian_culture": {
#         "global_config": {
#             "justification": "Indian classical inspired: D root (common for evening ragas), Yaman-like scale, drone texture, heavy ornamentation, spacious reverb.",
#             "tempo": "slow",
#             "bpm": 65.0,
#             "root": "d",
#             "scale_degrees": [0, 2, 4, 6, 7, 9, 11],  # Lydian-ish (Yaman raga)
#             "ascending_degrees": [0, 2, 4, 6, 7, 9, 11],
#             "descending_degrees": [11, 9, 7, 6, 4, 2, 0],
#             "emphasis_degrees": [0, 7, 4],  # Sa, Pa, Ma (tivra)
#             "degree_offsets_cents": {4: 10.0, 9: -8.0},  # Subtle shruti adjustments
#             "texture": "drone_melody",
#             "consonance": "strict",
#             "harmonic_motion": "drone",
#             "voicing": "open5",
#             "brightness": "medium",
#             "space": "cathedral",
#             "stereo_width": "medium",
#             "character": "warm",
#             "timbre_family": "plucked_like",
#             "density": 3,
#             "humanize": 0.25,
#             "ornament_intensity": "virtuosic",
#             "swing": 0.0,
#             "cycle_beats": 16,
#             "subdivision": 4,
#             "duration_s": 30.0,
#         },
#         "layers": [
#             {"role": "pad", "pattern": "drone", "gain": 0.8, "pan": 0.0, "timbre_family": "bowed_like", "brightness": "dark"},
#             {"role": "bass", "pattern": "tanpura", "gain": 0.6, "pan": 0.0, "timbre_family": "plucked_like", "tempo_ratio": "1/2"},
#             {"role": "lead", "pattern": "alap", "gain": 0.9, "pan": 0.05, "timbre_family": "plucked_like", "ornament_intensity": "virtuosic"},
#         ],
#     },

#     # =========================================================================
#     # LOVE - Warm, romantic, gentle, intimate, tender
#     # =========================================================================
#     "love": {
#         "global_config": {
#             "justification": "Romantic love: warm major key with 7ths for tenderness, slow tempo, intimate space, gentle dynamics, soft timbres.",
#             "tempo": "slow",
#             "bpm": 72.0,
#             "root": "e",
#             "scale_preset": "major",
#             "texture": "safe_polyphony",
#             "consonance": "gentle",
#             "harmonic_motion": "slow",
#             "voicing": "seventh",
#             "brightness": "medium",
#             "space": "intimate",
#             "stereo_width": "narrow",
#             "character": "warm",
#             "timbre_family": "bowed_like",
#             "density": 4,
#             "humanize": 0.18,
#             "ornament_intensity": "subtle",
#             "swing": 0.12,
#             "cycle_beats": 16,
#             "subdivision": 2,
#             "duration_s": 24.0,
#         },
#         "layers": [
#             {"role": "pad", "pattern": "swell", "gain": 0.7, "pan": 0.0, "timbre_family": "bowed_like", "character": "warm"},
#             {"role": "bass", "pattern": "gentle", "gain": 0.5, "pan": 0.0, "timbre_family": "sine_like"},
#             {"role": "lead", "pattern": "lyrical", "gain": 0.8, "pan": -0.1, "timbre_family": "bowed_like"},
#             {"role": "texture", "pattern": "air", "gain": 0.3, "pan": 0.15},
#         ],
#     },

#     # =========================================================================
#     # DEATH - Dark, slow, heavy, somber, funereal, oppressive
#     # =========================================================================
#     "death": {
#         "global_config": {
#             "justification": "Death and mortality: very slow glacial tempo, dark minor key, heavy drone, sparse texture, oppressive low frequencies, cavernous space.",
#             "tempo": "glacial",
#             "bpm": 48.0,
#             "root": "c#",
#             "scale_preset": "minor",
#             "emphasis_degrees": [0, 3, 7],  # Root, minor 3rd, 5th
#             "avoid_degrees": [6],  # Avoid tritone for bleakness not tension
#             "texture": "drone_melody",
#             "consonance": "gentle",
#             "harmonic_motion": "drone",
#             "voicing": "open5",
#             "brightness": "very_dark",
#             "space": "cathedral",
#             "stereo_width": "wide",
#             "character": "saturated",
#             "timbre_family": "bowed_like",
#             "density": 2,
#             "humanize": 0.10,
#             "ornament_intensity": "none",
#             "swing": 0.0,
#             "cycle_beats": 32,
#             "subdivision": 2,
#             "duration_s": 30.0,
#         },
#         "layers": [
#             {"role": "bass", "pattern": "drone", "gain": 1.0, "pan": 0.0, "timbre_family": "bowed_like", "brightness": "very_dark"},
#             {"role": "pad", "pattern": "swell", "gain": 0.6, "pan": 0.0, "timbre_family": "bowed_like", "space": "infinite"},
#             {"role": "texture", "pattern": "breath", "gain": 0.3, "pan": 0.1, "brightness": "dark"},
#         ],
#     },

#     # =========================================================================
#     # DRUNK AT 3 AM - Sloppy, loose timing, melancholic, disoriented, woozy
#     # =========================================================================
#     "drunk_3am": {
#         "global_config": {
#             "justification": "Drunk at 3am: stumbling tempo, heavy swing/humanize for sloppiness, minor key melancholy, warm/saturated character, intimate lonely space.",
#             "tempo": "slow",
#             "bpm": 68.0,
#             "root": "a",
#             "scale_preset": "dorian",  # Melancholic but not hopeless
#             "texture": "heterophony",
#             "consonance": "open",
#             "harmonic_motion": "minimal",
#             "voicing": "seventh",
#             "brightness": "dark",
#             "space": "intimate",
#             "stereo_width": "narrow",
#             "character": "saturated",
#             "timbre_family": "plucked_like",
#             "density": 3,
#             "humanize": 0.45,  # Very sloppy timing
#             "ornament_intensity": "moderate",
#             "swing": 0.35,  # Heavy swing = stumbling feel
#             "cycle_beats": 12,  # Odd cycle adds to disorientation
#             "subdivision": 3,  # Triplet feel
#             "duration_s": 24.0,
#         },
#         "layers": [
#             {"role": "pad", "pattern": "haze", "gain": 0.6, "pan": 0.0, "character": "saturated", "space": "room"},
#             {"role": "bass", "pattern": "walking", "gain": 0.5, "pan": 0.0, "timbre_family": "plucked_like"},
#             {"role": "lead", "pattern": "wander", "gain": 0.7, "pan": -0.15, "timbre_family": "plucked_like"},
#             {"role": "texture", "pattern": "vinyl", "gain": 0.25, "pan": 0.1},
#         ],
#     },

#     # =========================================================================
#     # DOMESTIC VIOLENCE: THE AGGRESSOR - Aggressive, distorted, chaotic, menacing
#     # =========================================================================
#     "domestic_violence_aggressor": {
#         "global_config": {
#             "justification": "Aggressor's internal chaos: fast aggressive tempo, distorted character, dissonant clusters, chaotic texture, oppressive brightness, relentless rhythm.",
#             "tempo": "fast",
#             "bpm": 145.0,
#             "root": "a",
#             "scale_preset": "locrian",  # Most unstable/tense mode
#             "emphasis_degrees": [0, 1, 6],  # Emphasize dissonant intervals
#             "texture": "chordal",
#             "consonance": "open",  # Allow dissonance
#             "harmonic_motion": "active",
#             "voicing": "cluster",
#             "brightness": "bright",
#             "space": "dry",
#             "stereo_width": "narrow",  # Claustrophobic
#             "character": "distorted",
#             "timbre_family": "noise_like",
#             "density": 6,
#             "humanize": 0.05,  # Mechanical, relentless
#             "ornament_intensity": "none",
#             "swing": 0.0,
#             "cycle_beats": 8,
#             "subdivision": 4,
#             "duration_s": 16.0,
#         },
#         "layers": [
#             {"role": "bass", "pattern": "pummel", "gain": 1.0, "pan": 0.0, "timbre_family": "membrane_like", "character": "distorted"},
#             {"role": "rhythm", "pattern": "assault", "gain": 0.9, "pan": 0.0, "timbre_family": "noise_like"},
#             {"role": "pad", "pattern": "grind", "gain": 0.7, "pan": 0.0, "character": "distorted", "brightness": "bright"},
#             {"role": "accent", "pattern": "stab", "gain": 0.6, "pan": 0.2, "timbre_family": "metallic_like"},
#         ],
#     },

#     # =========================================================================
#     # MARIO THE GAME - Bouncy 8-bit, bright, fast, playful, video game
#     # =========================================================================
#     "mario_game": {
#         "global_config": {
#             "justification": "Video game Mario: bright major key, bouncy fast tempo, clean square-wave-ish timbres, dry mix (no reverb like NES), tight timing, playful arpeggios.",
#             "tempo": "fast",
#             "bpm": 140.0,
#             "root": "c",
#             "scale_preset": "major",
#             "texture": "safe_polyphony",
#             "consonance": "strict",
#             "harmonic_motion": "active",
#             "voicing": "triad",
#             "brightness": "very_bright",
#             "space": "dry",  # NES had no reverb
#             "stereo_width": "narrow",  # Mono-ish like old games
#             "character": "clean",
#             "timbre_family": "sine_like",  # Chiptune-ish
#             "density": 5,
#             "humanize": 0.02,  # Very tight, machine-like
#             "ornament_intensity": "none",
#             "swing": 0.0,
#             "cycle_beats": 16,
#             "subdivision": 2,
#             "duration_s": 20.0,
#         },
#         "layers": [
#             {"role": "bass", "pattern": "bounce", "gain": 0.7, "pan": 0.0, "timbre_family": "sine_like"},
#             {"role": "lead", "pattern": "arpeggio", "gain": 0.9, "pan": 0.0, "timbre_family": "sine_like", "brightness": "very_bright"},
#             {"role": "rhythm", "pattern": "fourfloor", "gain": 0.5, "pan": 0.0, "timbre_family": "membrane_like"},
#             {"role": "accent", "pattern": "coin", "gain": 0.4, "pan": 0.1, "timbre_family": "metallic_like"},
#         ],
#     },

#     # =========================================================================
#     # EGYPTIAN NIGHTS - Ancient, mystical, Phrygian dominant, exotic, mysterious
#     # =========================================================================
#     "egyptian_nights": {
#         "global_config": {
#             "justification": "Ancient Egypt mystique: Phrygian dominant scale (augmented 2nd for exotic color), slow tempo, drone + melody texture, spacious reverb, ornamental melody.",
#             "tempo": "slow",
#             "bpm": 75.0,
#             "root": "e",
#             # Phrygian dominant / Hijaz-like: 1 b2 3 4 5 b6 b7
#             "scale_degrees": [0, 1, 4, 5, 7, 8, 10],
#             "emphasis_degrees": [0, 4, 7],  # Root, major 3rd, 5th
#             "texture": "drone_melody",
#             "consonance": "gentle",
#             "harmonic_motion": "minimal",
#             "voicing": "open5",
#             "brightness": "medium",
#             "space": "hall",
#             "stereo_width": "wide",
#             "character": "warm",
#             "timbre_family": "plucked_like",
#             "density": 4,
#             "humanize": 0.20,
#             "ornament_intensity": "expressive",
#             "swing": 0.08,
#             "cycle_beats": 16,
#             "subdivision": 4,
#             "duration_s": 24.0,
#         },
#         "layers": [
#             {"role": "bass", "pattern": "drone", "gain": 0.7, "pan": 0.0, "timbre_family": "bowed_like", "brightness": "dark"},
#             {"role": "pad", "pattern": "swell", "gain": 0.5, "pan": 0.0, "space": "cathedral"},
#             {"role": "lead", "pattern": "snake", "gain": 0.85, "pan": 0.05, "timbre_family": "reed_like", "ornament_intensity": "expressive"},
#             {"role": "rhythm", "pattern": "darbuka", "gain": 0.4, "pan": 0.0, "timbre_family": "membrane_like"},
#         ],
#     },

#     # =========================================================================
#     # ARABIAN DESERT - Maqam Hijaz, quarter-tones, ornamental, vast, mystical
#     # =========================================================================
#     "arabian_desert": {
#         "global_config": {
#             "justification": "Arabian desert: Maqam Hijaz with quarter-tone on 2nd degree, vast spacious reverb, ornamental melody, heterophonic texture, warm timbres.",
#             "tempo": "medium",
#             "bpm": 92.0,
#             "root": "d",
#             # Hijaz maqam: 1 b2(quarter-flat) 3 4 5 b6 b7
#             "scale_degrees": [0, 1, 4, 5, 7, 8, 10],
#             "degree_offsets_cents": {1: 50.0},  # Quarter-tone on flat 2nd
#             "ascending_degrees": [0, 1, 4, 5, 7, 8, 10],
#             "descending_degrees": [10, 8, 7, 5, 4, 1, 0],
#             "emphasis_degrees": [0, 4, 7],  # Tonic, 3rd, 5th
#             "avoid_degrees": [1],  # Approach the quarter-tone carefully
#             "texture": "heterophony",
#             "consonance": "gentle",
#             "harmonic_motion": "slow",
#             "voicing": "open5",
#             "brightness": "medium",
#             "space": "infinite",  # Vast desert
#             "stereo_width": "immersive",
#             "character": "warm",
#             "timbre_family": "plucked_like",
#             "density": 4,
#             "humanize": 0.22,
#             "ornament_intensity": "virtuosic",
#             "swing": 0.10,
#             "cycle_beats": 16,
#             "subdivision": 4,
#             "duration_s": 28.0,
#         },
#         "layers": [
#             {"role": "bass", "pattern": "drone", "gain": 0.6, "pan": 0.0, "timbre_family": "bowed_like", "tempo_ratio": "1/2"},
#             {"role": "pad", "pattern": "desert_wind", "gain": 0.4, "pan": 0.0, "timbre_family": "noise_like", "brightness": "dark"},
#             {"role": "lead", "pattern": "taqsim", "gain": 0.9, "pan": 0.0, "timbre_family": "plucked_like", "ornament_intensity": "virtuosic"},
#             {"role": "rhythm", "pattern": "tabla", "gain": 0.35, "pan": 0.0, "timbre_family": "membrane_like"},
#             {"role": "accent", "pattern": "ney", "gain": 0.5, "pan": -0.1, "timbre_family": "reed_like"},
#         ],
#     },

# }


# # =============================================================================
# # Quick render function
# # =============================================================================

# def render_template(name: str, output_path: str = None, duration_s: float = None):
#     """
#     Render a template by name.
    
#     Usage:
#         render_template("bubble_gum", "bubble_gum.wav")
#         render_template("arabian_desert", duration_s=60.0)
#     """
#     from . import new_synth

#     if name not in TEMPLATES:
#         raise ValueError(f"Unknown template: {name}. Available: {list(TEMPLATES.keys())}")

#     template = TEMPLATES[name]
#     config = {**template["global_config"]}
#     if "layers" in template:
#         config["layers"] = template["layers"]

#     if duration_s is not None:
#         config["duration_s"] = duration_s

#     if output_path is None:
#         output_path = f"{name}.wav"

#     print(f"Rendering '{name}' -> {output_path} ...")
#     audio = new_synth.render(config)
#     new_synth.write_wav(output_path, audio, config.get("sample_rate", new_synth.DEFAULT_SAMPLE_RATE))
#     print(f"Done: {output_path}")
#     return output_path


# def render_all(output_dir: str = ".", duration_s: float = 20.0):
#     """Render all templates."""
#     import os
#     os.makedirs(output_dir, exist_ok=True)

#     for name in TEMPLATES:
#         path = os.path.join(output_dir, f"{name}.wav")
#         render_template(name, path, duration_s=duration_s)


# # =============================================================================
# # CLI
# # =============================================================================

# if __name__ == "__main__":
#     import argparse
#     import os

#     parser = argparse.ArgumentParser(description="Render vibe templates")
#     parser.add_argument("template", nargs="?", default=None, help=f"Template name. Available: {list(TEMPLATES.keys())}")
#     parser.add_argument("--all", action="store_true", help="Render all templates")
#     parser.add_argument("--out", type=str, default=None, help="Output path")
#     parser.add_argument("--duration", type=float, default=20.0, help="Duration in seconds")
#     parser.add_argument("--list", action="store_true", help="List available templates")
#     args = parser.parse_args()

#     if args.list:
#         print("Available templates:")
#         for name, data in TEMPLATES.items():
#             justification = data["global_config"].get("justification", "")[:60]
#             print(f"  {name:30s} - {justification}...")
#         exit(0)

#     if args.all:
#         if args.out is not None:
#             # Interpret --out as output directory when rendering all
#             render_all(output_dir=args.out, duration_s=args.duration)
#         else:
#             render_all(duration_s=args.duration)
#     elif args.template:
#         # If --out is specified and is a directory, write into that dir named <template>.wav;
#         # otherwise --out is used as explicit output file path
#         out_path = args.out
#         if out_path is not None and os.path.isdir(out_path):
#             out_path = os.path.join(out_path, f"{args.template}.wav")
#         render_template(args.template, out_path, duration_s=args.duration)
#     else:
#         parser.print_help()
#!/usr/bin/env python3
"""
render_vibe_examples_sf2.py

Renders all FEW_SHOT_EXAMPLES from llm_to_synth.py using the SF2-based synth.
Produces higher-quality audio with real instrument samples.

Usage:
    python render_vibe_examples_sf2.py                    # Render all examples
    python render_vibe_examples_sf2.py --duration 20     # 20 seconds each
    python render_vibe_examples_sf2.py --indices 1 5 10  # Render specific examples
    python render_vibe_examples_sf2.py --list            # List all examples
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import the SF2 synth
# try:
from .new_synth import render, write_wav, HAS_SF2
# except ImportError:
#     print("Error: new_synth_sf2.py not found. Make sure it's in the same directory.")
#     sys.exit(1)
# except OSError as e:
#     print(f"Warning: SF2 library error: {e}")
#     print("Will continue with oscillator fallback.")
#     HAS_SF2 = False
#     from .new_synth import render, write_wav

# =============================================================================
# Mapping Constants (from llm_to_synth.py)
# =============================================================================

TEMPO_TO_BPM = {
    "very_slow": 55,
    "slow": 72,
    "medium": 96,
    "fast": 128,
    "very_fast": 160,
}

BRIGHTNESS_MAP_INV = {
    0.1: "very_dark",
    0.3: "dark", 
    0.5: "medium",
    0.7: "bright",
    0.9: "very_bright",
}

SPACE_MAP_INV = {
    0.1: "dry",
    0.3: "intimate",
    0.5: "room",
    0.7: "hall",
    0.95: "cathedral",
}

# Instrument mappings for layer types
BASS_INSTRUMENTS = {
    "drone": "acoustic_bass",
    "sustained": "acoustic_bass",
    "pulsing": "synth_bass",
    "walking": "acoustic_bass",
    "fifth_drone": "acoustic_bass",
    "sub_pulse": "synth_bass",
    "octave": "acoustic_bass",
    "arp_bass": "synth_bass",
}

PAD_INSTRUMENTS = {
    "warm_slow": "string_ensemble",
    "dark_sustained": "string_ensemble",
    "cinematic": "orchestra_hit",  # Use string for sustained
    "thin_high": "choir_aahs",
    "ambient_drift": "pad_atmosphere",
    "stacked_fifths": "string_ensemble",
    "bright_open": "choir_aahs",
}

MELODY_INSTRUMENTS = {
    "procedural": "violin",
    "contemplative": "violin",
    "contemplative_minor": "violin",
    "rising": "violin",
    "falling": "violin",
    "minimal": "flute",
    "ornamental": "sitar",
    "arp_melody": "vibraphone",
    "call_response": "violin",
    "heroic": "trumpet",
}

RHYTHM_INSTRUMENTS = {
    "none": None,
    "minimal": "taiko",
    "heartbeat": "taiko",
    "soft_four": "taiko",
    "hats_only": "taiko",
    "electronic": "taiko",
    "kit_light": "taiko",
    "kit_medium": "taiko",
    "military": "taiko",
    "tabla_essence": "taiko",
    "brush": "taiko",
}

TEXTURE_INSTRUMENTS = {
    "none": None,
    "shimmer": "celesta",
    "shimmer_slow": "celesta",
    "vinyl_crackle": None,  # No SF2 equivalent
    "breath": None,
    "stars": "celesta",
    "glitch": None,
    "noise_wash": None,
    "crystal": "celesta",
    "pad_whisper": "pad_atmosphere",
}

ACCENT_INSTRUMENTS = {
    "none": None,
    "bells": "tubular_bells",
    "bells_dense": "tubular_bells",
    "pluck": "acoustic_guitar_nylon",
    "chime": "tubular_bells",
    "blip": "vibraphone",
    "blip_random": "vibraphone",
    "brass_hit": "trumpet",
    "wind": "pan_flute",
    "arp_accent": "vibraphone",
    "piano_note": "acoustic_grand_piano",
}

# =============================================================================
# FEW_SHOT_EXAMPLES (copied from llm_to_synth.py)
# =============================================================================

# The examples are embedded in the docstring format in llm_to_synth.py
# We'll parse them dynamically from the file

def load_few_shot_examples(custom_path: Optional[str] = None) -> str:
    """Load FEW_SHOT_EXAMPLES from llm_to_synth.py"""
    # Try to find the file
    search_paths = [
        Path("llm_to_synth.py"),
        Path("src/app/llm_to_synth.py"),
        Path("app/llm_to_synth.py"),
        Path("../llm_to_synth.py"),
        Path("/mnt/project/llm_to_synth.py"),
    ]
    
    if custom_path:
        search_paths.insert(0, Path(custom_path))
    
    for path in search_paths:
        if path.exists():
            print(f"Found: {path}")
            content = path.read_text()
            # Extract FEW_SHOT_EXAMPLES string
            match = re.search(r'FEW_SHOT_EXAMPLES\s*=\s*"""(.*?)"""', content, re.DOTALL)
            if match:
                return match.group(1)
    
    searched = "\n  ".join(str(p) for p in search_paths)
    raise FileNotFoundError(f"Could not find llm_to_synth.py with FEW_SHOT_EXAMPLES.\nSearched:\n  {searched}")


def extract_examples(few_shot_text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract all example configs from FEW_SHOT_EXAMPLES string."""
    examples: List[Tuple[str, Dict[str, Any]]] = []
    
    # Pattern to match each example block
    example_pattern = re.compile(
        r'\*\*Example \d+\*\*\s*\nInput:\s*"([^"]+)"\s*\nOutput:\s*\n```json\s*\n(.*?)\n```',
        re.DOTALL
    )
    
    for match in example_pattern.finditer(few_shot_text):
        vibe_name = match.group(1)
        json_str = match.group(2)
        
        try:
            parsed = json.loads(json_str)
            config = parsed.get("config", parsed)
            examples.append((vibe_name, config))
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse example '{vibe_name}': {e}")
            continue
    
    return examples


# =============================================================================
# Config Conversion
# =============================================================================

def closest_key(value: float, mapping: Dict[float, str]) -> str:
    """Find the closest key in a mapping."""
    closest = min(mapping.keys(), key=lambda k: abs(k - value))
    return mapping[closest]


def vibe_to_sf2_config(vibe_config: Dict[str, Any], duration_s: float = 30.0) -> Dict[str, Any]:
    """Convert a VibeConfig to new_synth_sf2 config format."""
    
    # Extract values with defaults
    tempo = vibe_config.get("tempo", "medium")
    root = vibe_config.get("root", "c")
    mode = vibe_config.get("mode", "minor")
    brightness = vibe_config.get("brightness", "medium")
    space = vibe_config.get("space", "medium")
    density = vibe_config.get("density", 4)
    
    bass_type = vibe_config.get("bass", "drone")
    pad_type = vibe_config.get("pad", "warm_slow")
    melody_type = vibe_config.get("melody", "contemplative")
    rhythm_type = vibe_config.get("rhythm", "none")
    texture_type = vibe_config.get("texture", "none")
    accent_type = vibe_config.get("accent", "none")
    
    human = vibe_config.get("human", 0.3)
    if isinstance(human, str):
        human_map = {"robotic": 0.0, "tight": 0.15, "natural": 0.3, "loose": 0.5, "drunk": 0.8}
        human = human_map.get(human, 0.3)
    
    swing = vibe_config.get("swing", 0.0)
    if isinstance(swing, str):
        swing_map = {"none": 0.0, "light": 0.2, "medium": 0.5, "heavy": 0.8}
        swing = swing_map.get(swing, 0.0)
    
    # Map tempo to BPM
    if isinstance(tempo, str):
        bpm = TEMPO_TO_BPM.get(tempo, 96)
    else:
        # Numeric tempo (0-1) -> BPM
        bpm = int(55 + tempo * 110)
    
    # Map brightness (numeric or string)
    if isinstance(brightness, (int, float)):
        brightness_str = closest_key(float(brightness), BRIGHTNESS_MAP_INV)
    else:
        brightness_str = brightness
    
    # Map space (numeric or string)
    if isinstance(space, (int, float)):
        space_str = closest_key(float(space), SPACE_MAP_INV)
    else:
        space_str = space
    
    # Build layers based on density
    layers = []
    
    # Always include bass and pad
    if bass_type != "none":
        layers.append({
            "role": "bass",
            "pattern": bass_type,
            "instrument": BASS_INSTRUMENTS.get(bass_type, "acoustic_bass"),
            "gain": 0.8,
            "pan": 0.0,
        })
    
    if pad_type != "none":
        layers.append({
            "role": "pad",
            "pattern": "sustained",
            "instrument": PAD_INSTRUMENTS.get(pad_type, "string_ensemble"),
            "gain": 0.6,
            "pan": 0.0,
        })
    
    # Add melody for density >= 3
    if density >= 3 and melody_type != "none":
        layers.append({
            "role": "lead",
            "pattern": "melodic",
            "instrument": MELODY_INSTRUMENTS.get(melody_type, "violin"),
            "gain": 0.7,
            "pan": 0.1,
        })
    
    # Add rhythm for density >= 4
    if density >= 4 and rhythm_type != "none":
        inst = RHYTHM_INSTRUMENTS.get(rhythm_type)
        if inst:
            layers.append({
                "role": "accent",  # Use accent role for rhythm
                "pattern": "rhythmic",
                "instrument": inst,
                "gain": 0.4,
                "pan": -0.2,
            })
    
    # Add texture for density >= 5
    if density >= 5 and texture_type != "none":
        inst = TEXTURE_INSTRUMENTS.get(texture_type)
        if inst:
            layers.append({
                "role": "texture",
                "pattern": "ambient",
                "instrument": inst,
                "gain": 0.3,
                "pan": 0.3,
            })
    
    # Add accent for density >= 6
    if density >= 6 and accent_type != "none":
        inst = ACCENT_INSTRUMENTS.get(accent_type)
        if inst:
            layers.append({
                "role": "accent",
                "pattern": "sparse",
                "instrument": inst,
                "gain": 0.35,
                "pan": -0.3,
            })
    
    return {
        "bpm": bpm,
        "duration_s": duration_s,
        "root": root,
        "mode": mode,
        "brightness": brightness_str,
        "space": space_str,
        "humanize": human,
        "swing": swing,
        "density": density,
        "use_sf2": True,
        "layers": layers,
    }


def sanitize_filename(name: str) -> str:
    """Convert a vibe name to a safe filename."""
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower())
    return safe.strip("_")[:50]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Render vibe examples with SF2 synth")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration per example (seconds)")
    parser.add_argument("--indices", type=int, nargs="+", help="Specific example indices to render (1-based)")
    parser.add_argument("--list", action="store_true", help="List all examples without rendering")
    parser.add_argument("--out-dir", type=str, help="Output directory (default: timestamped)")
    parser.add_argument("--no-sf2", action="store_true", help="Disable SF2 samples (use oscillators)")
    parser.add_argument("--llm-path", type=str, help="Path to llm_to_synth.py")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Vibe Examples Renderer (SF2)")
    print("=" * 70)
    print(f"SF2 Available: {HAS_SF2}")
    
    # Load and parse examples
    try:
        few_shot_text = load_few_shot_examples(args.llm_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    examples = extract_examples(few_shot_text)
    print(f"Found {len(examples)} examples")
    
    if args.list:
        print("\nExamples:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"  {i:2d}. {name}")
        return
    
    # Filter examples if indices specified
    if args.indices:
        filtered = []
        for idx in args.indices:
            if 1 <= idx <= len(examples):
                filtered.append((idx, examples[idx - 1]))
            else:
                print(f"Warning: Index {idx} out of range (1-{len(examples)})")
        examples_to_render = filtered
    else:
        examples_to_render = [(i + 1, ex) for i, ex in enumerate(examples)]
    
    if not examples_to_render:
        print("No examples to render.")
        return
    
    # Setup output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = int(time.time())
        out_dir = Path(f"{timestamp}_vibe_examples_sf2")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # Render each example
    total = len(examples_to_render)
    for progress, (idx, (vibe_name, vibe_config)) in enumerate(examples_to_render, 1):
        print(f"\n{'=' * 70}")
        print(f"[{progress}/{total}] Example {idx}: {vibe_name}")
        print("=" * 70)
        
        # Convert config
        sf2_config = vibe_to_sf2_config(vibe_config, duration_s=args.duration)
        
        if args.no_sf2:
            sf2_config["use_sf2"] = False
        
        # Show config summary
        print(f"  BPM: {sf2_config['bpm']}, Root: {sf2_config['root']}, Mode: {sf2_config['mode']}")
        print(f"  Brightness: {sf2_config['brightness']}, Space: {sf2_config['space']}")
        print(f"  Layers: {len(sf2_config['layers'])}")
        for layer in sf2_config["layers"]:
            print(f"    - {layer['role']}: {layer.get('instrument', 'default')}")
        
        # Render
        try:
            audio = render(sf2_config)
            
            # Save
            safe_name = sanitize_filename(vibe_name)
            out_path = out_dir / f"{idx:02d}_{safe_name}.wav"
            write_wav(str(out_path), audio)
            print(f"  ✓ Saved: {out_path}")
            
            # Also save config JSON for reference
            config_path = out_dir / f"{idx:02d}_{safe_name}.json"
            with open(config_path, "w") as f:
                json.dump(sf2_config, f, indent=2)
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 70}")
    print(f"Done! {total} examples saved to: {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()