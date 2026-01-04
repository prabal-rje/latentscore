"""
sf2_integration.py

Clean SF2 SoundFont integration for new_synth.py.
Provides sampled instruments to replace raw oscillators.

REQUIRES: sf2_loader numpy
    pip install sf2_loader numpy

Usage:
    from sf2_integration import SF2Engine, note_name_to_midi
    
    engine = SF2Engine()
    engine.auto_load_soundfonts()  # Loads from ./soundfonts/
    
    audio = engine.render_note(
        note="c",
        octave=4,
        duration=1.0,
        instrument="piano"
    )
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Try to import sf2_loader
# try:
from sf2_loader import sf2_loader
SF2_AVAILABLE = True
# except ImportError:
#     SF2_AVAILABLE = False
#     print("Warning: sf2_loader not installed. Run: pip install sf2_loader")
#     print("SF2 instrument rendering will fall back to basic oscillators.")

FloatArray = NDArray[np.float32]

DEFAULT_SAMPLE_RATE = 44100


# =============================================================================
# Note/MIDI Utilities
# =============================================================================

NOTE_TO_SEMITONE: Dict[str, int] = {
    "c": 0, "c#": 1, "db": 1,
    "d": 2, "d#": 3, "eb": 3,
    "e": 4, "fb": 4, "e#": 5,
    "f": 5, "f#": 6, "gb": 6,
    "g": 7, "g#": 8, "ab": 8,
    "a": 9, "a#": 10, "bb": 10,
    "b": 11, "cb": 11, "b#": 0,
}


def note_name_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name to MIDI number. C4 = 60 (middle C)."""
    note_lower = note.lower().strip()
    semitone = NOTE_TO_SEMITONE.get(note_lower, 0)
    return semitone + (octave + 1) * 12


def midi_to_freq(midi_note: int, a4_hz: float = 440.0) -> float:
    """Convert MIDI note to frequency."""
    return a4_hz * (2.0 ** ((midi_note - 69) / 12.0))


def scale_to_midi_notes(
    root: str,
    mode: str,
    octave: int = 4,
    num_octaves: int = 1
) -> List[int]:
    """Get MIDI notes for a scale."""
    MODE_INTERVALS = {
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
    
    intervals = MODE_INTERVALS.get(mode.lower(), MODE_INTERVALS["minor"])
    root_midi = note_name_to_midi(root, octave)
    
    notes = []
    for oct in range(num_octaves):
        for interval in intervals:
            notes.append(root_midi + interval + (oct * 12))
    
    return notes


# =============================================================================
# GM Instrument Mapping
# =============================================================================

# Full General MIDI program numbers
GM_INSTRUMENTS: Dict[str, int] = {
    # Piano (0-7)
    "piano": 0, "acoustic_piano": 0, "bright_piano": 1,
    "electric_grand": 2, "honky_tonk": 3,
    "electric_piano": 4, "electric_piano_1": 4, "electric_piano_2": 5,
    "harpsichord": 6, "clavinet": 7,
    
    # Chromatic Percussion (8-15)
    "celesta": 8, "glockenspiel": 9, "music_box": 10,
    "vibraphone": 11, "marimba": 12, "xylophone": 13,
    "tubular_bells": 14, "bells": 14, "dulcimer": 15,
    
    # Organ (16-23)
    "organ": 16, "drawbar_organ": 16, "percussive_organ": 17,
    "rock_organ": 18, "church_organ": 19, "reed_organ": 20,
    "accordion": 21, "harmonica": 22, "tango_accordion": 23,
    
    # Guitar (24-31)
    "acoustic_guitar": 24, "acoustic_guitar_nylon": 24,
    "acoustic_guitar_steel": 25, "electric_guitar_jazz": 26,
    "electric_guitar_clean": 27, "electric_guitar": 27,
    "electric_guitar_muted": 28, "overdriven_guitar": 29,
    "distortion_guitar": 30, "guitar_harmonics": 31,
    
    # Bass (32-39)
    "acoustic_bass": 32, "bass": 32,
    "electric_bass_finger": 33, "electric_bass": 33,
    "electric_bass_pick": 34, "fretless_bass": 35,
    "slap_bass": 36, "slap_bass_1": 36, "slap_bass_2": 37,
    "synth_bass": 38, "synth_bass_1": 38, "synth_bass_2": 39,
    
    # Strings (40-47)
    "violin": 40, "viola": 41, "cello": 42, "contrabass": 43,
    "tremolo_strings": 44, "pizzicato_strings": 45,
    "orchestral_harp": 46, "harp": 46, "timpani": 47,
    
    # Ensemble (48-55)
    "string_ensemble": 48, "string_ensemble_1": 48, "strings": 48,
    "string_ensemble_2": 49, "synth_strings": 50, "synth_strings_1": 50,
    "synth_strings_2": 51, "choir_aahs": 52, "choir": 52,
    "voice_oohs": 53, "synth_voice": 54, "orchestra_hit": 55,
    
    # Brass (56-63)
    "trumpet": 56, "trombone": 57, "tuba": 58, "muted_trumpet": 59,
    "french_horn": 60, "brass_section": 61, "brass": 61,
    "synth_brass": 62, "synth_brass_1": 62, "synth_brass_2": 63,
    
    # Reed (64-71)
    "soprano_sax": 64, "alto_sax": 65, "tenor_sax": 66, "baritone_sax": 67,
    "oboe": 68, "english_horn": 69, "bassoon": 70, "clarinet": 71,
    
    # Pipe (72-79)
    "piccolo": 72, "flute": 73, "recorder": 74, "pan_flute": 75,
    "blown_bottle": 76, "shakuhachi": 77, "whistle": 78, "ocarina": 79,
    
    # Synth Lead (80-87)
    "lead_square": 80, "lead_sawtooth": 81, "lead_calliope": 82,
    "lead_chiff": 83, "lead_charang": 84, "lead_voice": 85,
    "lead_fifths": 86, "lead_bass_lead": 87,
    "synth_lead": 81,
    
    # Synth Pad (88-95)
    "pad_new_age": 88, "pad_warm": 89, "pad_polysynth": 90,
    "pad_choir": 91, "pad_bowed": 92, "pad_metallic": 93,
    "pad_halo": 94, "pad_sweep": 95,
    "pad": 89, "synth_pad": 89,
    
    # Synth Effects (96-103)
    "fx_rain": 96, "fx_soundtrack": 97, "fx_crystal": 98,
    "fx_atmosphere": 99, "fx_brightness": 100, "fx_goblins": 101,
    "fx_echoes": 102, "fx_scifi": 103,
    
    # Ethnic (104-111)
    "sitar": 104, "banjo": 105, "shamisen": 106, "koto": 107,
    "kalimba": 108, "bagpipe": 109, "fiddle": 110, "shanai": 111,
    
    # Percussive (112-119)
    "tinkle_bell": 112, "agogo": 113, "steel_drums": 114,
    "woodblock": 115, "taiko_drum": 116, "taiko": 116,
    "melodic_tom": 117, "synth_drum": 118, "reverse_cymbal": 119,
    
    # Sound Effects (120-127)
    "guitar_fret_noise": 120, "breath_noise": 121, "seashore": 122,
    "bird_tweet": 123, "telephone_ring": 124, "helicopter": 125,
    "applause": 126, "gunshot": 127,
}

# Map vibe-style names to GM instruments
VIBE_TO_GM: Dict[str, str] = {
    # Bass types
    "bass_drone": "acoustic_bass",
    "bass_sustained": "acoustic_bass",
    "bass_pulsing": "synth_bass_1",
    "bass_walking": "acoustic_bass",
    "bass_sub": "synth_bass_2",
    "bass_electric": "electric_bass_finger",
    
    # Pad types
    "pad_warm": "pad_warm",
    "pad_dark": "pad_bowed",
    "pad_bright": "pad_new_age",
    "pad_cinematic": "string_ensemble_1",
    "pad_ambient": "pad_choir",
    "pad_strings": "string_ensemble_1",
    "pad_choir": "choir_aahs",
    
    # Melody/Lead types
    "melody_piano": "piano",
    "melody_strings": "violin",
    "melody_flute": "flute",
    "melody_synth": "lead_sawtooth",
    "melody_bells": "vibraphone",
    "melody_voice": "voice_oohs",
    
    # World/Ethnic instruments
    "sitar": "sitar",
    "koto": "koto",
    "shamisen": "shamisen",
    "shanai": "shanai",
    "kalimba": "kalimba",
    "steel_drums": "steel_drums",
    "taiko": "taiko_drum",
    "shakuhachi": "shakuhachi",
    "pan_flute": "pan_flute",
    "bagpipe": "bagpipe",
    "fiddle": "fiddle",
    "banjo": "banjo",
    
    # Accents
    "bells": "tubular_bells",
    "chimes": "tubular_bells",
    "harp": "orchestral_harp",
    "pluck": "acoustic_guitar_nylon",
    "pizz": "pizzicato_strings",
    "marimba": "marimba",
    "vibes": "vibraphone",
    "glockenspiel": "glockenspiel",
}


# =============================================================================
# Ethnic Instrument Registry (for non-GM soundfonts)
# =============================================================================

@dataclass
class EthnicInstrument:
    """An ethnic instrument from a specialized soundfont."""
    name: str
    soundfont: str
    bank: int
    program: int
    note_range: Tuple[int, int] = (36, 96)  # MIDI range
    description: str = ""


# These will be populated when soundfonts are loaded
ETHNIC_INSTRUMENTS: Dict[str, EthnicInstrument] = {}


def register_ethnic_instrument(
    name: str,
    soundfont: str,
    bank: int,
    program: int,
    note_range: Tuple[int, int] = (36, 96),
    description: str = ""
) -> None:
    """Register an ethnic instrument from a specialty soundfont."""
    ETHNIC_INSTRUMENTS[name.lower()] = EthnicInstrument(
        name=name,
        soundfont=soundfont,
        bank=bank,
        program=program,
        note_range=note_range,
        description=description,
    )


# =============================================================================
# SF2 Engine
# =============================================================================

@dataclass
class SF2Engine:
    """
    SoundFont-based synthesis engine.
    
    Wraps sf2_loader to provide easy instrument rendering.
    Falls back to basic sine waves if sf2_loader is unavailable.
    """
    
    sample_rate: int = DEFAULT_SAMPLE_RATE
    soundfonts: Dict[str, Any] = field(default_factory=dict)
    default_soundfont: Optional[str] = None
    soundfont_dir: Optional[Path] = None
    
    def __post_init__(self):
        if self.soundfont_dir is None:
            # Default to ./soundfonts/ relative to this file
            self.soundfont_dir = Path(__file__).parent / "soundfonts"
    
    def load_soundfont(self, path: Union[str, Path], name: Optional[str] = None) -> bool:
        """
        Load a SoundFont file.
        
        Args:
            path: Path to .sf2 file
            name: Optional name to reference this soundfont (defaults to filename stem)
            
        Returns:
            True if loaded successfully
        """
        if not SF2_AVAILABLE:
            print("sf2_loader not available, skipping soundfont load")
            return False
        
        path = Path(path).resolve()
        
        if not path.exists():
            print(f"SoundFont not found: {path}")
            return False
        
        if name is None:
            name = path.stem
        
        try:
            loader = sf2_loader(str(path))
            self.soundfonts[name] = {
                "loader": loader,
                "path": str(path),
            }
            
            if self.default_soundfont is None:
                self.default_soundfont = name
            
            print(f"Loaded SoundFont: {name}")
            return True
            
        except Exception as e:
            print(f"Failed to load SoundFont {path}: {e}")
            return False
    
    def auto_load_soundfonts(self) -> int:
        """
        Automatically load all .sf2 files from the soundfont directory.
        
        Returns:
            Number of soundfonts loaded
        """
        if self.soundfont_dir is None or not self.soundfont_dir.exists():
            return 0
        
        count = 0
        for sf_file in sorted(self.soundfont_dir.glob("*.sf2")):
            if self.load_soundfont(sf_file):
                count += 1
        
        # Also try .SF2 extension (case-insensitive filesystems)
        for sf_file in sorted(self.soundfont_dir.glob("*.SF2")):
            if sf_file.stem not in self.soundfonts:
                if self.load_soundfont(sf_file):
                    count += 1
        
        return count
    
    def list_instruments(self, soundfont: Optional[str] = None) -> List[Tuple[int, int, str]]:
        """
        List all instruments in a soundfont.
        
        Returns:
            List of (bank, program, name) tuples
        """
        sf_name = soundfont or self.default_soundfont
        if sf_name is None or sf_name not in self.soundfonts:
            return []
        
        loader = self.soundfonts[sf_name]["loader"]
        
        try:
            return loader.all_instruments()
        except Exception as e:
            print(f"Error listing instruments: {e}")
            return []
    
    def get_program_for_instrument(self, instrument: str) -> Tuple[Optional[str], int, int]:
        """
        Get (soundfont_name, bank, program) for an instrument name.
        
        Args:
            instrument: Instrument name (e.g., "piano", "sitar", "oud")
            
        Returns:
            (soundfont_name, bank, program) or (None, 0, 0) if not found
        """
        inst_lower = instrument.lower().strip()
        
        # Check ethnic instruments first (specialty soundfonts)
        if inst_lower in ETHNIC_INSTRUMENTS:
            eth = ETHNIC_INSTRUMENTS[inst_lower]
            if eth.soundfont in self.soundfonts:
                return (eth.soundfont, eth.bank, eth.program)
        
        # Check vibe-style mapping
        if inst_lower in VIBE_TO_GM:
            gm_name = VIBE_TO_GM[inst_lower]
            program = GM_INSTRUMENTS.get(gm_name, 0)
            return (self.default_soundfont, 0, program)
        
        # Check direct GM mapping
        if inst_lower in GM_INSTRUMENTS:
            program = GM_INSTRUMENTS[inst_lower]
            return (self.default_soundfont, 0, program)
        
        # Default to piano
        return (self.default_soundfont, 0, 0)
    
    def render_note(
        self,
        midi_note: Optional[int] = None,
        note: Optional[str] = None,
        octave: int = 4,
        duration: float = 1.0,
        velocity: int = 100,
        instrument: str = "piano",
        channel: int = 0,
    ) -> FloatArray:
        """
        Render a single note.
        
        Args:
            midi_note: MIDI note number (60 = middle C). Takes precedence if provided.
            note: Note name (e.g., "c", "f#", "bb"). Used if midi_note is None.
            octave: Octave (4 = middle octave). Used with note name.
            duration: Note duration in seconds
            velocity: MIDI velocity (0-127)
            instrument: Instrument name
            channel: MIDI channel (0-15, 9 = drums)
            
        Returns:
            Audio as numpy array (mono, float32)
        """
        # Resolve MIDI note
        if midi_note is None:
            if note is None:
                midi_note = 60  # Default to middle C
            else:
                midi_note = note_name_to_midi(note, octave)
        
        # Get soundfont and program
        sf_name, bank, program = self.get_program_for_instrument(instrument)
        
        if sf_name is None or sf_name not in self.soundfonts:
            # Fallback: generate simple sine wave
            return self._fallback_note(midi_note, duration, velocity)
        
        loader = self.soundfonts[sf_name]["loader"]
        
        try:
            # Select instrument
            loader.change(channel=channel, bank=bank, preset=program)
            
            # Play note
            loader.noteon(channel=channel, note=midi_note, velocity=velocity)
            
            # Render audio
            num_samples = int(duration * self.sample_rate)
            # sf2_loader returns stereo, we'll convert to mono
            audio = loader.synth(num_samples)
            
            # Note off
            loader.noteoff(channel=channel, note=midi_note)
            
            # Convert to mono if stereo
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)
            
            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize if needed
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            
            return audio
            
        except Exception as e:
            print(f"SF2 render error: {e}")
            return self._fallback_note(midi_note, duration, velocity)
    
    def render_chord(
        self,
        midi_notes: List[int],
        duration: float = 1.0,
        velocity: int = 100,
        instrument: str = "piano",
    ) -> FloatArray:
        """
        Render a chord (multiple simultaneous notes).
        
        Args:
            midi_notes: List of MIDI note numbers
            duration: Chord duration in seconds
            velocity: MIDI velocity (0-127)
            instrument: Instrument name
            
        Returns:
            Audio as numpy array (mono, float32)
        """
        sf_name, bank, program = self.get_program_for_instrument(instrument)
        
        if sf_name is None or sf_name not in self.soundfonts:
            # Fallback: sum sine waves
            result = np.zeros(int(duration * self.sample_rate), dtype=np.float32)
            for midi_note in midi_notes:
                result += self._fallback_note(midi_note, duration, velocity // 2)
            return np.clip(result, -1.0, 1.0)
        
        loader = self.soundfonts[sf_name]["loader"]
        
        try:
            channel = 0
            loader.change(channel=channel, bank=bank, preset=program)
            
            # Play all notes
            for midi_note in midi_notes:
                loader.noteon(channel=channel, note=midi_note, velocity=velocity)
            
            # Render
            num_samples = int(duration * self.sample_rate)
            audio = loader.synth(num_samples)
            
            # Note off all
            for midi_note in midi_notes:
                loader.noteoff(channel=channel, note=midi_note)
            
            # Convert to mono if stereo
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)
            
            audio = audio.astype(np.float32)
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            
            return audio
            
        except Exception as e:
            print(f"SF2 chord render error: {e}")
            result = np.zeros(int(duration * self.sample_rate), dtype=np.float32)
            for midi_note in midi_notes:
                result += self._fallback_note(midi_note, duration, velocity // 2)
            return np.clip(result, -1.0, 1.0)
    
    def _fallback_note(
        self,
        midi_note: int,
        duration: float,
        velocity: int
    ) -> FloatArray:
        """Generate a simple sine wave as fallback."""
        freq = midi_to_freq(midi_note)
        t = np.linspace(0, duration, int(duration * self.sample_rate), dtype=np.float32)
        amp = (velocity / 127.0) * 0.5
        
        # Simple sine with ADSR-ish envelope
        wave = amp * np.sin(2 * np.pi * freq * t)
        
        # Quick attack, sustain, release envelope
        n = len(wave)
        attack = min(int(0.01 * self.sample_rate), n // 4)
        release = min(int(0.1 * self.sample_rate), n // 4)
        
        if attack > 0:
            wave[:attack] *= np.linspace(0, 1, attack)
        if release > 0:
            wave[-release:] *= np.linspace(1, 0, release)
        
        return wave.astype(np.float32)


# =============================================================================
# Convenience Functions
# =============================================================================

# Global engine instance (lazy-loaded)
_global_engine: Optional[SF2Engine] = None


def get_engine() -> SF2Engine:
    """Get or create the global SF2 engine."""
    global _global_engine
    if _global_engine is None:
        _global_engine = SF2Engine()
        _global_engine.auto_load_soundfonts()
    return _global_engine


def render_instrument_note(
    note: str,
    octave: int = 4,
    duration: float = 1.0,
    velocity: int = 100,
    instrument: str = "piano",
) -> FloatArray:
    """
    Convenience function to render a single instrument note.
    
    Args:
        note: Note name (e.g., "c", "f#", "bb")
        octave: Octave number (4 = middle octave)
        duration: Duration in seconds
        velocity: MIDI velocity (0-127)
        instrument: Instrument name
        
    Returns:
        Audio as numpy array
    """
    engine = get_engine()
    return engine.render_note(
        note=note,
        octave=octave,
        duration=duration,
        velocity=velocity,
        instrument=instrument,
    )


def render_instrument_chord(
    notes: List[Tuple[str, int]],  # List of (note_name, octave)
    duration: float = 1.0,
    velocity: int = 100,
    instrument: str = "piano",
) -> FloatArray:
    """
    Convenience function to render a chord.
    
    Args:
        notes: List of (note_name, octave) tuples
        duration: Duration in seconds
        velocity: MIDI velocity
        instrument: Instrument name
        
    Returns:
        Audio as numpy array
    """
    engine = get_engine()
    midi_notes = [note_name_to_midi(n, o) for n, o in notes]
    return engine.render_chord(
        midi_notes=midi_notes,
        duration=duration,
        velocity=velocity,
        instrument=instrument,
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("SF2 Integration Module")
    print("=" * 50)
    
    # Test note conversion
    print("\nNote conversions:")
    for note, oct in [("c", 4), ("a", 4), ("f#", 3), ("bb", 5)]:
        midi = note_name_to_midi(note, oct)
        freq = midi_to_freq(midi)
        print(f"  {note}{oct} -> MIDI {midi} -> {freq:.2f} Hz")
    
    # Test engine
    print("\nTesting SF2Engine...")
    engine = SF2Engine()
    loaded = engine.auto_load_soundfonts()
    print(f"Loaded {loaded} soundfonts")
    
    if loaded > 0:
        # List instruments from first soundfont
        for sf_name in engine.soundfonts:
            instruments = engine.list_instruments(sf_name)[:10]
            print(f"\nFirst 10 instruments in {sf_name}:")
            for bank, prog, name in instruments:
                print(f"  Bank {bank}, Program {prog}: {name}")
        
        # Render a test note
        print("\nRendering test note (C4, piano, 1 second)...")
        audio = engine.render_note(note="c", octave=4, duration=1.0, instrument="piano")
        print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")
        print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
        
        # Save test
        try:
            import soundfile as sf
            sf.write("test_sf2_note.wav", audio, engine.sample_rate)
            print("Saved: test_sf2_note.wav")
        except ImportError:
            print("soundfile not installed, skipping save")
    else:
        print("No soundfonts found. Run download_soundfonts.py first.")
        print("\nTesting fallback rendering...")
        audio = engine.render_note(note="c", octave=4, duration=1.0, instrument="piano")
        print(f"Fallback audio shape: {audio.shape}")