from __future__ import annotations

import json
from typing import Literal, List, Dict, Any
import numpy as np
import soundfile as sf
import outlines
import mlx_lm
from pydantic import BaseModel, Field
from .synth import MusicConfig, assemble, SAMPLE_RATE

MODEL_PATH = "./models/gemma-3-1b-it-qat-8bit"

# =============================================================================
# SCHEMA: LEAD SHEET (STRUCTURED SONG)
# =============================================================================

class GlobalVibe(BaseModel):
    """Global sound design and mix parameters."""
    tempo: Literal["very_slow", "slow", "medium", "fast", "very_fast"]
    key_root: Literal["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
    mode: Literal["major", "minor", "dorian", "mixolydian"]
    brightness: Literal["dark", "medium", "bright"]
    space: Literal["small", "medium", "large", "vast"]
    # Synth character
    grain: Literal["clean", "warm", "gritty"]
    human: Literal["robotic", "natural", "loose"]

class TimelineBlock(BaseModel):
    """A single chord/section in the song timeline."""
    chord_root: Literal["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
    chord_type: Literal["maj", "min", "maj7", "min7", "dom7", "dim", "sus4"]
    bars: int = Field(description="Duration in bars (usually 4 or 8)")
    
    # Dynamic shifts relative to global vibe
    density: Literal[2, 3, 4, 5, 6] = Field(description="Layer count. 2=sparse, 6=full")
    melody_style: Literal["smart", "minimal", "rising", "none"]

class LeadSheet(BaseModel):
    """A complete song arrangement."""
    justification: str = Field(description="Reasoning for the chord progression and vibe.")
    global_vibe: GlobalVibe
    timeline: List[TimelineBlock]

# =============================================================================
# MAPPINGS
# =============================================================================

TEMPO_MAP = {"very_slow": 0.15, "slow": 0.3, "medium": 0.5, "fast": 0.7, "very_fast": 0.9}
BRIGHTNESS_MAP = {"dark": 0.3, "medium": 0.5, "bright": 0.8}
SPACE_MAP = {"small": 0.3, "medium": 0.5, "large": 0.7, "vast": 0.95}

FEW_SHOT_EXAMPLES = """
Example 1:
Vibe: "Lo-fi hip hop study beats, relaxing, nostalgic"
{
  "justification": "Uses a jazz-influenced ii-V-I progression in C Major. Slow tempo and 'loose' humanization create the drunk/swung feel. Warm grain for vinyl texture.",
  "global_vibe": {
    "tempo": "slow", "key_root": "c", "mode": "major",
    "brightness": "dark", "space": "small", "grain": "warm", "human": "loose"
  },
  "timeline": [
    { "chord_root": "d", "chord_type": "min7", "bars": 4, "density": 4, "melody_style": "smart" },
    { "chord_root": "g", "chord_type": "dom7", "bars": 4, "density": 4, "melody_style": "smart" },
    { "chord_root": "c", "chord_type": "maj7", "bars": 8, "density": 3, "melody_style": "minimal" }
  ]
}

Example 2:
Vibe: "Epic cinematic battle, rising tension"
{
  "justification": "Minor key, building density. Starts with low drone, moves to dominant chord to build tension. 'Gritty' grain adds aggression.",
  "global_vibe": {
    "tempo": "fast", "key_root": "a", "mode": "minor",
    "brightness": "medium", "space": "vast", "grain": "gritty", "human": "robotic"
  },
  "timeline": [
    { "chord_root": "a", "chord_type": "min", "bars": 8, "density": 2, "melody_style": "none" },
    { "chord_root": "f", "chord_type": "maj", "bars": 4, "density": 4, "melody_style": "rising" },
    { "chord_root": "e", "chord_type": "dom7", "bars": 4, "density": 6, "melody_style": "rising" }
  ]
}
"""

SYSTEM_PROMPT = f"""
You are a session musician and composer.
Given a vibe description, generate a 'Lead Sheet' JSON.
This includes global sound settings and a specific chord progression timeline.
Use jazz chords (min7, maj7) for lo-fi/complex vibes. Use simple triads (maj, min) for pop/epic.
<examples>
{FEW_SHOT_EXAMPLES}
</examples>
"""

print(f"Loading model from {MODEL_PATH}...")
model = outlines.from_mlxlm(*mlx_lm.load(MODEL_PATH))

def generate_lead_sheet(vibe: str) -> LeadSheet:
    prompt = f"{SYSTEM_PROMPT}\nVibe: \"{vibe}\"\nOutput JSON:"
    result = model(prompt, output_type=LeadSheet, max_tokens=2048)
    return LeadSheet.model_validate_json(result)

def render_lead_sheet(sheet: LeadSheet, output_filename: str = "output.wav"):
    """
    Sequencer: Converts the Lead Sheet into audio by rendering blocks.
    """
    gv = sheet.global_vibe
    
    # Calculate seconds per bar (assuming 4/4)
    # Map abstract tempo to BPM-ish scaler
    bpm_approx = 60 + (TEMPO_MAP[gv.tempo] * 100)
    sec_per_bar = (60 / bpm_approx) * 4
    
    audio_segments = []
    
    print(f"Rendering {len(sheet.timeline)} blocks for vibe: {gv.tempo} / {gv.key_root}")
    
    for i, block in enumerate(sheet.timeline):
        duration = block.bars * sec_per_bar
        
        # Merge global settings with block-specific overrides
        config = MusicConfig(
            tempo=TEMPO_MAP[gv.tempo],
            root=gv.key_root,
            mode=gv.mode,
            brightness=BRIGHTNESS_MAP[gv.brightness],
            space=SPACE_MAP[gv.space],
            density=block.density,
            grain=gv.grain,
            human=0.3 if gv.human == "natural" else (0.6 if gv.human == "loose" else 0.0),
            
            # The Magic: Context-Aware Parameters
            chord_root=block.chord_root,
            chord_type=block.chord_type,
            melody=block.melody_style if block.melody_style != "none" else "none"
        )
        
        print(f"  Block {i+1}: {block.chord_root}{block.chord_type} ({block.bars} bars)")
        segment = assemble(config, duration=duration, normalize=False)
        
        # Simple crossfade envelope to prevent clicks between blocks
        fade_len = int(SAMPLE_RATE * 0.05)
        if len(segment) > fade_len * 2:
            segment[:fade_len] *= np.linspace(0, 1, fade_len)
            segment[-fade_len:] *= np.linspace(1, 0, fade_len)
            
        audio_segments.append(segment)
    
    full_audio = np.concatenate(audio_segments)
    
    # Final Normalize
    mx = np.max(np.abs(full_audio))
    if mx > 0: full_audio = full_audio / mx * 0.9
        
    sf.write(output_filename, full_audio, SAMPLE_RATE)
    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    EXAMPLES = [
        ("8-bit retro video game music, upbeat, bouncy, cheerful, platformer energy, bright and playful, fast-paced adventure theme like Mario", "mario_game.wav"),
        ("vast cosmic ambient, floating in zero gravity, ethereal pads, slow evolving textures, dark and mysterious, contemplative, starfield meditation", "ambient_space.wav"),
        ("smooth jazz cafe, warm evening atmosphere, sophisticated, relaxed, coffee shop vibes, brushed drums, walking bass, mellow trumpet feel", "jazz_cafe.wav"),
        ("lo-fi hip hop beats to study to, chill, nostalgic, tape hiss warmth, late night homework session, rainy window, mellow and repetitive", "lo-fi_study.wav"),
        ("video game music that gradually morphs into smooth jazz, starting bright and bouncy then evolving into sophisticated cafe jazz, playful to mellow transition", "mario_to_jazz_morph.wav"),
    ]
    
    for vibe, filename in EXAMPLES:
        print(f"\n{'='*60}")
        print(f"Generating: {filename}")
        print(f"Vibe: {vibe}")
        print(f"{'='*60}\n")
        
        sheet = generate_lead_sheet(vibe)
        print(f"Plan: {sheet.justification}")
        
        render_lead_sheet(sheet, filename)
        print(f"✓ Saved: {filename}")