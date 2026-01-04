
"""
SF2 Synthesizer using pyfluidsynth for offline rendering.
Actually sounds like real instruments.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    import fluidsynth
    HAS_FLUIDSYNTH = True
except ImportError:
    HAS_FLUIDSYNTH = False
    print("WARNING: pyfluidsynth not installed. pip install pyfluidsynth")


SAMPLE_RATE = 44100

# GM Program Numbers (General MIDI standard)
GM_PROGRAMS = {
    # Piano
    "piano": 0,
    "bright_piano": 1,
    "electric_piano": 4,
    "honky_tonk": 3,
    
    # Chromatic Percussion
    "celesta": 8,
    "glockenspiel": 9,
    "music_box": 10,
    "vibraphone": 11,
    "marimba": 12,
    "xylophone": 13,
    "tubular_bells": 14,
    
    # Organ
    "organ": 19,
    "church_organ": 19,
    "reed_organ": 20,
    
    # Guitar
    "acoustic_guitar": 24,
    "electric_guitar": 27,
    "clean_guitar": 27,
    "muted_guitar": 28,
    "overdrive_guitar": 29,
    "distortion_guitar": 30,
    
    # Bass
    "acoustic_bass": 32,
    "electric_bass": 33,
    "finger_bass": 33,
    "pick_bass": 34,
    "fretless_bass": 35,
    "slap_bass": 36,
    "synth_bass": 38,
    
    # Strings
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "strings": 48,
    "slow_strings": 49,
    "synth_strings": 50,
    "orchestra_hit": 55,
    
    # Ensemble
    "choir": 52,
    "voice_oohs": 53,
    "synth_voice": 54,
    
    # Brass
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "french_horn": 60,
    "brass_section": 61,
    "synth_brass": 62,
    
    # Reed
    "soprano_sax": 64,
    "alto_sax": 65,
    "tenor_sax": 66,
    "baritone_sax": 67,
    "oboe": 68,
    "english_horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    
    # Pipe
    "piccolo": 72,
    "flute": 73,
    "recorder": 74,
    "pan_flute": 75,
    "shakuhachi": 77,
    "whistle": 78,
    "ocarina": 79,
    
    # Synth Lead
    "square_lead": 80,
    "saw_lead": 81,
    "calliope": 82,
    "chiff_lead": 83,
    "charang": 84,
    "voice_lead": 85,
    "fifth_lead": 86,
    "bass_lead": 87,
    
    # Synth Pad
    "new_age_pad": 88,
    "warm_pad": 89,
    "polysynth": 90,
    "choir_pad": 91,
    "bowed_pad": 92,
    "metallic_pad": 93,
    "halo_pad": 94,
    "sweep_pad": 95,
    
    # Synth Effects
    "rain": 96,
    "soundtrack": 97,
    "crystal": 98,
    "atmosphere": 99,
    "brightness": 100,
    "goblins": 101,
    "echoes": 102,
    "sci_fi": 103,
    
    # Ethnic
    "sitar": 104,
    "banjo": 105,
    "shamisen": 106,
    "koto": 107,
    "kalimba": 108,
    "bagpipe": 109,
    "fiddle": 110,
    "shanai": 111,
    
    # Percussive
    "tinkle_bell": 112,
    "agogo": 113,
    "steel_drums": 114,
    "woodblock": 115,
    "taiko": 116,
    "melodic_tom": 117,
    "synth_drum": 118,
    
    # Sound Effects
    "reverse_cymbal": 119,
    "breath": 121,
    "seashore": 122,
    "bird_tweet": 123,
    "telephone": 124,
    "helicopter": 125,
    "applause": 126,
    "gunshot": 127,
}


@dataclass
class Note:
    """A single note event."""
    pitch: int          # MIDI note number (60 = middle C)
    velocity: int       # 0-127
    start_sec: float    # Start time in seconds
    duration_sec: float # Duration in seconds
    channel: int = 0    # MIDI channel


class SF2Synth:
    """Offline SF2 synthesizer."""
    
    def __init__(self, soundfont_path: str):
        if not HAS_FLUIDSYNTH:
            raise RuntimeError("pyfluidsynth not installed")
        
        self.sf_path = Path(soundfont_path)
        if not self.sf_path.exists():
            raise FileNotFoundError(f"Soundfont not found: {soundfont_path}")
        
        self.fs: Optional[fluidsynth.Synth] = None
        self.sfid: int = -1
    
    def _init_synth(self):
        """Initialize FluidSynth."""
        if self.fs is not None:
            return
        
        self.fs = fluidsynth.Synth(samplerate=float(SAMPLE_RATE))
        self.sfid = self.fs.sfload(str(self.sf_path))
        if self.sfid < 0:
            raise RuntimeError(f"Failed to load soundfont: {self.sf_path}")
    
    def _cleanup(self):
        """Clean up FluidSynth."""
        if self.fs is not None:
            self.fs.delete()
            self.fs = None
            self.sfid = -1
    
    def set_program(self, channel: int, program: int | str, bank: int = 0):
        """Set instrument for a channel."""
        self._init_synth()
        
        if isinstance(program, str):
            program = GM_PROGRAMS.get(program.lower(), 0)
        
        self.fs.program_select(channel, self.sfid, bank, program)
    
    def render_notes(
        self,
        notes: List[Note],
        duration: float,
        programs: dict[int, int | str] = None,
    ) -> np.ndarray:
        """
        Render a list of notes to audio.
        
        Args:
            notes: List of Note objects
            duration: Total duration in seconds
            programs: Dict mapping channel -> program number/name
        
        Returns:
            Stereo audio array (samples, 2)
        """
        self._init_synth()
        
        # Set programs for each channel
        if programs:
            for ch, prog in programs.items():
                self.set_program(ch, prog)
        
        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda n: n.start_sec)
        
        # Build event list: (time_sec, is_note_on, pitch, velocity, channel)
        events: List[Tuple[float, bool, int, int, int]] = []
        for note in sorted_notes:
            events.append((note.start_sec, True, note.pitch, note.velocity, note.channel))
            events.append((note.start_sec + note.duration_sec, False, note.pitch, 0, note.channel))
        
        # Sort by time
        events.sort(key=lambda e: e[0])
        
        # Render in chunks
        total_samples = int(duration * SAMPLE_RATE)
        output = np.zeros((total_samples, 2), dtype=np.float32)
        
        current_sample = 0
        event_idx = 0
        
        # Process events and render between them
        while current_sample < total_samples:
            # Find next event time
            if event_idx < len(events):
                next_event_sample = int(events[event_idx][0] * SAMPLE_RATE)
            else:
                next_event_sample = total_samples
            
            # Render up to next event (or end)
            samples_to_render = min(next_event_sample - current_sample, total_samples - current_sample)
            
            if samples_to_render > 0:
                chunk = self.fs.get_samples(samples_to_render)
                # get_samples returns interleaved stereo
                chunk = np.array(chunk, dtype=np.float32).reshape(-1, 2)
                
                end_sample = current_sample + len(chunk)
                if end_sample > total_samples:
                    chunk = chunk[:total_samples - current_sample]
                    end_sample = total_samples
                
                output[current_sample:end_sample] = chunk
                current_sample = end_sample
            
            # Process events at this time
            while event_idx < len(events) and int(events[event_idx][0] * SAMPLE_RATE) <= current_sample:
                time_sec, is_on, pitch, vel, ch = events[event_idx]
                if is_on:
                    self.fs.noteon(ch, pitch, vel)
                else:
                    self.fs.noteoff(ch, pitch)
                event_idx += 1
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.85
        
        return output
    
    def render_chord_progression(
        self,
        chords: List[List[int]],
        chord_duration: float,
        program: int | str = "piano",
        velocity: int = 80,
        channel: int = 0,
    ) -> np.ndarray:
        """
        Render a simple chord progression.
        
        Args:
            chords: List of chords, each chord is a list of MIDI notes
            chord_duration: Duration of each chord in seconds
            program: Instrument
            velocity: Note velocity
            channel: MIDI channel
        
        Returns:
            Stereo audio array
        """
        notes = []
        for i, chord in enumerate(chords):
            start = i * chord_duration
            for pitch in chord:
                notes.append(Note(
                    pitch=pitch,
                    velocity=velocity,
                    start_sec=start,
                    duration_sec=chord_duration * 0.95,
                    channel=channel,
                ))
        
        total_duration = len(chords) * chord_duration
        return self.render_notes(notes, total_duration, {channel: program})
    
    def __del__(self):
        self._cleanup()


def demo():
    """Demo the SF2 synth."""
    import soundfile as sf
    
    # Find soundfont
    sf_paths = [
        Path("/Users/prabal/Documents/Code/2025-12-29_latentscore/src/app/soundfonts/GeneralUser_GS.sf2"),
        Path("soundfonts/GeneralUser_GS.sf2"),
        Path("GeneralUser_GS.sf2"),
    ]
    
    sf_path = None
    for p in sf_paths:
        if p.exists():
            sf_path = p
            break
    
    if not sf_path:
        print("No soundfont found!")
        return
    
    print(f"Using soundfont: {sf_path}")
    synth = SF2Synth(str(sf_path))
    
    # Demo 1: Piano chord progression
    print("\nRendering piano chords...")
    chords = [
        [60, 64, 67],      # C major
        [65, 69, 72],      # F major
        [67, 71, 74],      # G major
        [60, 64, 67],      # C major
    ]
    audio = synth.render_chord_progression(chords, 1.5, "piano", velocity=90)
    sf.write("demo_piano_chords.wav", audio, SAMPLE_RATE)
    print("  Saved: demo_piano_chords.wav")
    
    # Demo 2: Multi-instrument
    print("\nRendering multi-instrument piece...")
    notes = [
        # Bass line (channel 0)
        Note(36, 100, 0.0, 1.4, 0),
        Note(41, 100, 1.5, 1.4, 0),
        Note(43, 100, 3.0, 1.4, 0),
        Note(36, 100, 4.5, 1.4, 0),
        
        # Piano chords (channel 1)
        Note(60, 70, 0.0, 1.4, 1),
        Note(64, 70, 0.0, 1.4, 1),
        Note(67, 70, 0.0, 1.4, 1),
        Note(65, 70, 1.5, 1.4, 1),
        Note(69, 70, 1.5, 1.4, 1),
        Note(72, 70, 1.5, 1.4, 1),
        Note(67, 70, 3.0, 1.4, 1),
        Note(71, 70, 3.0, 1.4, 1),
        Note(74, 70, 3.0, 1.4, 1),
        Note(60, 70, 4.5, 1.4, 1),
        Note(64, 70, 4.5, 1.4, 1),
        Note(67, 70, 4.5, 1.4, 1),
        
        # Melody (channel 2)
        Note(72, 80, 0.0, 0.7, 2),
        Note(74, 80, 0.75, 0.7, 2),
        Note(76, 80, 1.5, 0.7, 2),
        Note(77, 80, 2.25, 0.7, 2),
        Note(79, 80, 3.0, 1.4, 2),
        Note(76, 80, 4.5, 0.7, 2),
        Note(72, 80, 5.25, 0.7, 2),
    ]
    
    programs = {
        0: "acoustic_bass",
        1: "piano",
        2: "flute",
    }
    
    audio = synth.render_notes(notes, 6.0, programs)
    sf.write("demo_multi_instrument.wav", audio, SAMPLE_RATE)
    print("  Saved: demo_multi_instrument.wav")
    
    # Demo 3: Strings pad
    print("\nRendering strings pad...")
    notes = [
        Note(48, 60, 0.0, 4.0, 0),  # C
        Note(55, 60, 0.0, 4.0, 0),  # G
        Note(60, 60, 0.0, 4.0, 0),  # C
        Note(64, 60, 0.0, 4.0, 0),  # E
    ]
    audio = synth.render_notes(notes, 5.0, {0: "strings"})
    sf.write("demo_strings.wav", audio, SAMPLE_RATE)
    print("  Saved: demo_strings.wav")
    
    print("\nDone! Check the .wav files.")


if __name__ == "__main__":
    demo()