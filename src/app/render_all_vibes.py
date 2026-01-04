"""
Render all FEW_SHOT_EXAMPLES from llm_to_synth.py using SF2 instruments.

Usage:
    python render_all_vibes.py                     # Render all 35 examples
    python render_all_vibes.py --indices 1 5 10   # Render specific ones
    python render_all_vibes.py --list             # Just list them
"""

import re
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import soundfile as sf

from .sf2_synth import SF2Synth, Note, SAMPLE_RATE

# ============================================================================
# CONFIG
# ============================================================================

SOUNDFONT_PATH = "/Users/prabal/Documents/Code/2025-12-29_latentscore/src/app/soundfonts/GeneralUser_GS.sf2"

# ============================================================================
# INSTRUMENT MAPPINGS
# ============================================================================

BASS_INSTRUMENTS = {
    "drone": "contrabass",
    "sustained": "acoustic_bass",
    "pulsing": "synth_bass",
    "walking": "acoustic_bass",
    "fifth_drone": "contrabass",
    "sub_pulse": "synth_bass",
    "octave": "electric_bass",
    "arp_bass": "synth_bass",
}

PAD_INSTRUMENTS = {
    "warm_slow": "strings",
    "dark_sustained": "slow_strings",
    "cinematic": "strings",
    "thin_high": "synth_strings",
    "ambient_drift": "choir_pad",
    "stacked_fifths": "strings",
    "bright_open": "synth_strings",
}

MELODY_INSTRUMENTS = {
    "procedural": "piano",
    "contemplative": "flute",
    "contemplative_minor": "oboe",
    "rising": "violin",
    "falling": "cello",
    "minimal": "vibraphone",
    "ornamental": "sitar",
    "arp_melody": "electric_piano",
    "call_response": "trumpet",
    "heroic": "brass_section",
}

ACCENT_INSTRUMENTS = {
    "none": None,
    "bells": "tubular_bells",
    "bells_dense": "glockenspiel",
    "pluck": "acoustic_guitar",
    "chime": "celesta",
    "blip": "music_box",
    "blip_random": "xylophone",
    "brass_hit": "brass_section",
    "wind": "pan_flute",
    "arp_accent": "marimba",
    "piano_note": "piano",
}

MODE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}

ROOT_TO_MIDI = {
    "c": 60, "c#": 61, "d": 62, "d#": 63, "e": 64, "f": 65,
    "f#": 66, "g": 67, "g#": 68, "a": 69, "a#": 70, "b": 71,
}

# Value mappings (same as llm_to_synth.py)
TEMPO_MAP = {"very_slow": 0.15, "slow": 0.3, "medium": 0.5, "fast": 0.7, "very_fast": 0.9}
BRIGHTNESS_MAP = {"very_dark": 0.1, "dark": 0.3, "medium": 0.5, "bright": 0.7, "very_bright": 0.9}
SPACE_MAP = {"dry": 0.1, "small": 0.3, "medium": 0.5, "large": 0.7, "vast": 0.95}


# ============================================================================
# PARSE FEW_SHOT_EXAMPLES
# ============================================================================

def find_llm_to_synth() -> Path:
    """Find llm_to_synth.py"""
    search_paths = [
        Path("llm_to_synth.py"),
        Path("src/app/llm_to_synth.py"),
        Path("app/llm_to_synth.py"),
        Path("../llm_to_synth.py"),
        Path("/mnt/project/llm_to_synth.py"),
    ]
    for p in search_paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"Can't find llm_to_synth.py. Searched: {search_paths}")


def extract_examples(llm_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract all examples from FEW_SHOT_EXAMPLES."""
    content = llm_path.read_text()
    
    # Find FEW_SHOT_EXAMPLES string
    match = re.search(r'FEW_SHOT_EXAMPLES\s*=\s*"""(.*?)"""', content, re.DOTALL)
    if not match:
        raise ValueError("Can't find FEW_SHOT_EXAMPLES in llm_to_synth.py")
    
    few_shot_text = match.group(1)
    
    # Parse each example
    example_pattern = re.compile(
        r'\*\*Example \d+\*\*\s*\nInput:\s*"([^"]+)"\s*\nOutput:\s*\n```json\s*\n(.*?)\n```',
        re.DOTALL
    )
    
    examples = []
    for m in example_pattern.finditer(few_shot_text):
        vibe_name = m.group(1)
        json_str = m.group(2)
        
        try:
            parsed = json.loads(json_str)
            config = parsed.get("config", parsed)
            
            # Convert string values to numeric where needed
            if isinstance(config.get("tempo"), str):
                config["tempo"] = TEMPO_MAP.get(config["tempo"], 0.5)
            if isinstance(config.get("brightness"), str):
                config["brightness"] = BRIGHTNESS_MAP.get(config["brightness"], 0.5)
            if isinstance(config.get("space"), str):
                config["space"] = SPACE_MAP.get(config["space"], 0.5)
            
            examples.append((vibe_name, config))
        except json.JSONDecodeError as e:
            print(f"  Warning: Failed to parse example '{vibe_name}': {e}")
    
    return examples


# ============================================================================
# NOTE GENERATION
# ============================================================================

def scale_degree_to_midi(root: str, mode: str, degree: int, octave: int = 4) -> int:
    """Convert scale degree to MIDI note."""
    root_midi = ROOT_TO_MIDI.get(root.lower(), 60)
    intervals = MODE_INTERVALS.get(mode.lower(), MODE_INTERVALS["minor"])
    
    octave_offset = degree // len(intervals)
    scale_index = degree % len(intervals)
    
    return root_midi + intervals[scale_index] + (12 * octave_offset) + (12 * (octave - 4))


def generate_bass_notes(config: Dict[str, Any], duration: float) -> List[Note]:
    """Generate bass line notes."""
    notes = []
    root = config.get("root", "c")
    mode = config.get("mode", "minor")
    tempo = config.get("tempo", 0.5)
    bass_type = config.get("bass", "sustained")
    
    bpm = 60 + tempo * 80
    beat_dur = 60 / bpm
    
    if bass_type in ("drone", "fifth_drone", "sub_pulse"):
        note_dur = min(8.0, duration / 2)
        t = 0.0
        while t < duration:
            notes.append(Note(
                pitch=scale_degree_to_midi(root, mode, 0, 2),
                velocity=70,
                start_sec=t,
                duration_sec=note_dur * 0.95,
                channel=0,
            ))
            if bass_type == "fifth_drone":
                notes.append(Note(
                    pitch=scale_degree_to_midi(root, mode, 4, 2),
                    velocity=60,
                    start_sec=t,
                    duration_sec=note_dur * 0.95,
                    channel=0,
                ))
            t += note_dur
    
    elif bass_type == "walking":
        pattern = [0, 2, 4, 2]
        t = 0.0
        i = 0
        while t < duration:
            degree = pattern[i % len(pattern)]
            notes.append(Note(
                pitch=scale_degree_to_midi(root, mode, degree, 2),
                velocity=80,
                start_sec=t,
                duration_sec=beat_dur * 0.9,
                channel=0,
            ))
            t += beat_dur
            i += 1
    
    elif bass_type in ("pulsing", "arp_bass"):
        t = 0.0
        while t < duration:
            notes.append(Note(
                pitch=scale_degree_to_midi(root, mode, 0, 2),
                velocity=75 if int(t / beat_dur) % 2 == 0 else 60,
                start_sec=t,
                duration_sec=beat_dur * 0.4,
                channel=0,
            ))
            t += beat_dur * 0.5
    
    else:
        t = 0.0
        while t < duration:
            notes.append(Note(
                pitch=scale_degree_to_midi(root, mode, 0, 2),
                velocity=75,
                start_sec=t,
                duration_sec=beat_dur * 2 * 0.9,
                channel=0,
            ))
            t += beat_dur * 2
    
    return notes


def generate_pad_notes(config: Dict[str, Any], duration: float) -> List[Note]:
    """Generate pad/chord notes."""
    notes = []
    root = config.get("root", "c")
    mode = config.get("mode", "minor")
    tempo = config.get("tempo", 0.5)
    
    bpm = 60 + tempo * 80
    chord_dur = 60 / bpm * 4
    
    progression = [0, 3, 4, 0]
    
    t = 0.0
    chord_idx = 0
    
    while t < duration:
        chord_root = progression[chord_idx % len(progression)]
        
        for degree_offset in [0, 2, 4]:
            degree = chord_root + degree_offset
            notes.append(Note(
                pitch=scale_degree_to_midi(root, mode, degree, 3),
                velocity=55,
                start_sec=t,
                duration_sec=chord_dur * 0.98,
                channel=1,
            ))
        
        t += chord_dur
        chord_idx += 1
    
    return notes


def generate_melody_notes(config: Dict[str, Any], duration: float) -> List[Note]:
    """Generate melody notes."""
    notes = []
    root = config.get("root", "c")
    mode = config.get("mode", "minor")
    tempo = config.get("tempo", 0.5)
    melody_type = config.get("melody", "contemplative")
    
    bpm = 60 + tempo * 80
    beat_dur = 60 / bpm
    
    rng = np.random.default_rng(42)
    
    if melody_type == "minimal":
        note_times = [0, 4, 8, 12]
        degrees = [4, 2, 4, 0]
    elif melody_type in ("rising", "heroic"):
        note_times = [0, 1, 2, 3, 4, 6]
        degrees = [0, 2, 4, 5, 6, 7]
    elif melody_type == "falling":
        note_times = [0, 1, 2, 3, 4, 6]
        degrees = [7, 6, 5, 4, 2, 0]
    elif melody_type == "arp_melody":
        note_times = list(range(16))
        degrees = [0, 2, 4, 7] * 4
    else:
        note_times = [0, 1.5, 3, 4, 5.5, 7]
        degrees = [4, 5, 4, 2, 4, 0]
    
    for i, beat_offset in enumerate(note_times):
        t = beat_offset * beat_dur
        if t >= duration:
            break
        
        degree = degrees[i % len(degrees)]
        note_length = beat_dur * (1.0 + rng.random())
        
        notes.append(Note(
            pitch=scale_degree_to_midi(root, mode, degree, 5),
            velocity=70 + int(rng.random() * 20),
            start_sec=t,
            duration_sec=min(note_length, duration - t - 0.1),
            channel=2,
        ))
    
    return notes


def generate_accent_notes(config: Dict[str, Any], duration: float) -> List[Note]:
    """Generate accent notes."""
    notes = []
    root = config.get("root", "c")
    mode = config.get("mode", "minor")
    tempo = config.get("tempo", 0.5)
    accent_type = config.get("accent", "none")
    
    if accent_type == "none" or ACCENT_INSTRUMENTS.get(accent_type) is None:
        return notes
    
    bpm = 60 + tempo * 80
    beat_dur = 60 / bpm
    
    rng = np.random.default_rng(123)
    
    t = 0.0
    while t < duration:
        if rng.random() < 0.3:
            degree = rng.choice([0, 2, 4, 5])
            notes.append(Note(
                pitch=scale_degree_to_midi(root, mode, degree, 5),
                velocity=60,
                start_sec=t,
                duration_sec=beat_dur * 0.8,
                channel=3,
            ))
        t += beat_dur * 2
    
    return notes


# ============================================================================
# RENDER
# ============================================================================

def render_config(config: Dict[str, Any], synth: SF2Synth, duration: float) -> np.ndarray:
    """Render a single config to audio."""
    
    bass_inst = BASS_INSTRUMENTS.get(config.get("bass", "sustained"), "acoustic_bass")
    pad_inst = PAD_INSTRUMENTS.get(config.get("pad", "warm_slow"), "strings")
    melody_inst = MELODY_INSTRUMENTS.get(config.get("melody", "contemplative"), "piano")
    accent_inst = ACCENT_INSTRUMENTS.get(config.get("accent", "none"))
    
    all_notes = []
    all_notes.extend(generate_bass_notes(config, duration))
    all_notes.extend(generate_pad_notes(config, duration))
    all_notes.extend(generate_melody_notes(config, duration))
    all_notes.extend(generate_accent_notes(config, duration))
    
    programs = {
        0: bass_inst,
        1: pad_inst,
        2: melody_inst,
    }
    if accent_inst:
        programs[3] = accent_inst
    
    print(f"    bass={bass_inst}, pad={pad_inst}, melody={melody_inst}", end="")
    if accent_inst:
        print(f", accent={accent_inst}", end="")
    print()
    
    return synth.render_notes(all_notes, duration, programs)


def sanitize_filename(name: str) -> str:
    """Make a safe filename."""
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower())
    return safe.strip("_")[:50]


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Render FEW_SHOT_EXAMPLES with SF2")
    parser.add_argument("--indices", type=int, nargs="+", help="Specific examples (1-based)")
    parser.add_argument("--list", action="store_true", help="Just list examples")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration per example")
    parser.add_argument("--out", type=str, default="rendered_vibes", help="Output directory")
    parser.add_argument("--sf2", type=str, default=SOUNDFONT_PATH, help="Soundfont path")
    args = parser.parse_args()
    
    print("=" * 70)
    print("FEW_SHOT_EXAMPLES -> SF2 Renderer")
    print("=" * 70)
    
    # Find and parse llm_to_synth.py
    try:
        llm_path = find_llm_to_synth()
        print(f"Found: {llm_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    examples = extract_examples(llm_path)
    print(f"Parsed {len(examples)} examples\n")
    
    # List mode
    if args.list:
        for i, (name, _) in enumerate(examples, 1):
            print(f"  {i:2d}. {name}")
        return
    
    # Check soundfont
    if not Path(args.sf2).exists():
        print(f"Error: Soundfont not found: {args.sf2}")
        sys.exit(1)
    
    # Filter examples
    if args.indices:
        selected = [(i, examples[i-1]) for i in args.indices if 0 < i <= len(examples)]
    else:
        selected = list(enumerate(examples, 1))
    
    # Create output dir
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    
    # Initialize synth
    synth = SF2Synth(args.sf2)
    
    # Render
    for idx, (name, config) in selected:
        print(f"\n[{idx}/{len(examples)}] {name}")
        print("-" * 60)
        
        try:
            audio = render_config(config, synth, args.duration)
            
            filename = f"{idx:02d}_{sanitize_filename(name)}.wav"
            out_path = out_dir / filename
            sf.write(str(out_path), audio, SAMPLE_RATE)
            print(f"  -> {out_path}")
        
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n{'=' * 70}")
    print(f"Done! Files saved to: {out_dir}/")


if __name__ == "__main__":
    main()