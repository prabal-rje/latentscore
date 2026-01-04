from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Literal, Tuple, cast

import mlx.core as mx
import mlx_lm
import numpy as np
import outlines
import soundfile as sf  # type: ignore[import]  # soundfile stubs missing in this env.
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from . import synth as synth_old

# from . import synth_NEW_BACKUP as synth_new  # Commented out with main block

MODEL_PATH = "./models/gemma-3-1b-it-qat-8bit"


def make_repetition_penalty_sampler(penalty: float = 1.1):
    seen_tokens: list[int] = []

    def sampler(logprobs: mx.array) -> mx.array:
        nonlocal seen_tokens

        logprobs_flat = cast(list[float], logprobs.squeeze().tolist())

        for tok in set(seen_tokens):
            if logprobs_flat[tok] > 0:
                logprobs_flat[tok] /= penalty
            else:
                logprobs_flat[tok] *= penalty

        penalized = mx.array(logprobs_flat)
        result = mx.argmax(penalized, axis=-1, keepdims=True)
        seen_tokens.append(int(result.item()))
        return result

    return sampler


class VibeConfig(BaseModel):
    """A curated vibe-to-synth mapping with staged justifications."""

    # Meta
    justification: str = Field(
        max_length=3_000,  # you have to force this because the LLM generatino frequently fails due to excessively long, run ons...
        description="""THINK OUT LOUD HERE *and* IN GREAT DETAIL. This is your scratch pad—use it BEFORE deciding parameters.

STEP 1 - VIBE DECOMPOSITION: What does this vibe *feel* like? Break it into sensory/emotional atoms: Is it heavy or weightless? Warm or cold? Intimate or vast? Moving or still? Tense or relaxed? Ancient or futuristic? Organic or synthetic?

STEP 2 - SONIC TRANSLATION: For each atom above, what sonic quality achieves it? (e.g., "weightless" → slow attack + high register + sparse density + vast reverb; "tense" → minor mode + dissonance + fast tempo + sharp transients)

STEP 3 - COHERENCE CHECK: Do your choices reinforce each other or fight? A "cozy rainy day" needs BOTH warm grain AND small intimate space—vast reverb would break the spell. Identify the 2-3 non-negotiable parameters that define this vibe's essence.

STEP 4 - REFERENCE ANCHORING: Which few-shot example is closest? What should you STEAL vs. DIVERGE from? Don't copy blindly—adapt intelligently.

Write in shorthand. Be specific. Commit to bold choices. Generic = wrong.""",
    )

    # # Core parameters and justification
    # core_justification: str = Field(
    #     description="Sonic reasoning for the 'core' parameters: tempo, root, mode, brightness, space, density."
    # )
    tempo: Literal["very_slow", "slow", "medium", "fast", "very_fast"] = Field(
        description="Speed/energy. very_slow=glacial, medium=moderate, very_fast=frenetic"
    )
    root: Literal["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"] = Field(
        description="Root note. C=neutral, D=mystical, A=dark, F=warm, G=bright"
    )
    mode: Literal["major", "minor", "dorian", "mixolydian"] = Field(
        description="Scale mode. major=happy, minor=sad, dorian=mysterious, mixolydian=hopeful"
    )
    brightness: Literal["very_dark", "dark", "medium", "bright", "very_bright"] = Field(
        description="Filter cutoff. very_dark=muffled, medium=balanced, very_bright=crisp/airy"
    )
    space: Literal["dry", "small", "medium", "large", "vast"] = Field(
        description="Reverb/room size. dry=intimate, medium=room, vast=cathedral"
    )
    density: Literal[2, 3, 4, 5, 6] = Field(
        description="Layer count. 2=minimal, 4=balanced, 6=lush/thick"
    )

    # Layers and justification
    # layers_justification: str = Field(
    #     description="Sonic reasoning for 'layers' parameters: bass, pad, melody, rhythm, texture, accent."
    # )
    bass: Literal[
        "drone", "sustained", "pulsing", "walking", "fifth_drone", "sub_pulse", "octave", "arp_bass"
    ] = Field(
        description="Bass character. drone=static, pulsing=rhythmic, walking=melodic movement, octave=sustained, arp_bass=pulsing"
    )
    pad: Literal[
        "warm_slow",
        "dark_sustained",
        "cinematic",
        "thin_high",
        "ambient_drift",
        "stacked_fifths",
        "bright_open",
    ] = Field(
        description="Pad texture. warm_slow=cozy, cinematic=epic, thin_high=ethereal, bright_open=airy high pad"
    )
    melody: Literal[
        "procedural",
        "contemplative",
        "contemplative_minor",
        "rising",
        "falling",
        "minimal",
        "ornamental",
        "arp_melody",
        "call_response",
        "heroic",
    ] = Field(
        description="Melodic contour. procedural=phrase-aware generator, rising=hopeful, falling=melancholic, minimal=sparse, heroic=triumphant ascending"
    )
    rhythm: Literal[
        "none",
        "minimal",
        "heartbeat",
        "soft_four",
        "hats_only",
        "electronic",
        "kit_light",
        "kit_medium",
        "military",
        "tabla_essence",
        "brush",
    ] = Field(
        description="Percussion style. none=ambient, heartbeat=organic pulse, electronic=synthetic, tabla_essence=indian rhythm, military=marching, brush=jazz brushes"
    )
    texture: Literal[
        "none",
        "shimmer",
        "shimmer_slow",
        "vinyl_crackle",
        "breath",
        "stars",
        "glitch",
        "noise_wash",
        "crystal",
        "pad_whisper",
    ] = Field(
        description="Background texture. shimmer=sparkly, vinyl_crackle=nostalgic, breath=organic, glitch=digital artifacts, crystal=twinkling, noise_wash=ambient noise"
    )
    accent: Literal[
        "none",
        "bells",
        "bells_dense",
        "pluck",
        "chime",
        "blip",
        "blip_random",
        "brass_hit",
        "wind",
        "arp_accent",
        "piano_note",
    ] = Field(
        description="Sparse melodic accents. bells=crystalline, pluck=intimate, chime=mystical, blip=electronic, brass_hit=orchestral, piano_note=acoustic"
    )

    # V2 Parameters and justification
    # v2_justification: str = Field(
    #     description="Reasoning for 'V2' parameters: motion, attack, stereo, depth, echo, human, grain."
    # )
    motion: Literal["static", "slow", "medium", "fast", "chaotic"] = Field(
        description="LFO/modulation rate. static=frozen, medium=breathing, chaotic=constantly evolving"
    )
    attack: Literal["soft", "medium", "sharp"] = Field(
        description="Transient character. soft=swelling, sharp=percussive"
    )
    stereo: Literal["mono", "narrow", "medium", "wide", "ultra_wide"] = Field(
        description="Stereo width. mono=focused, medium=natural, ultra_wide=immersive"
    )
    depth: bool = Field(description="Add sub-bass layer for physical weight")
    echo: Literal["none", "subtle", "medium", "heavy", "infinite"] = Field(
        description="Delay feedback. none=dry, medium=spatial, infinite=washed out"
    )
    human: Literal["robotic", "tight", "natural", "loose", "drunk"] = Field(
        description="Timing/pitch imperfection. robotic=perfect, natural=subtle variation, drunk=sloppy"
    )
    grain: Literal["clean", "warm", "gritty"] = Field(
        description="Oscillator character. clean=digital, warm=analog, gritty=distorted"
    )

    # ----------------------------
    # New: procedural melody + harmony knobs (optional; defaults work fine)
    # ----------------------------

    melody_engine: Literal["procedural", "pattern"] = Field(
        default="procedural",
        description="Which melody system to use. procedural=phrase-aware generator, pattern=legacy templates",
    )

    phrase_len_bars: Literal[2, 4, 8] = Field(
        default=4, description="Phrase length for the procedural melody engine (in bars)"
    )
    melody_density: Literal["very_sparse", "sparse", "medium", "busy", "very_busy"] = Field(
        default="medium", description="How note-dense the melody is (procedural engine)"
    )
    syncopation: Literal["straight", "light", "medium", "heavy"] = Field(
        default="light", description="How much the melody favors offbeats (procedural engine)"
    )
    swing: Literal["none", "light", "medium", "heavy"] = Field(
        default="none", description="Swing amount for 8th-note feel (procedural engine)"
    )
    motif_repeat: Literal["rare", "sometimes", "often"] = Field(
        default="sometimes",
        description="How often motifs repeat at phrase starts (procedural engine)",
    )
    step_vs_leap: Literal["step", "balanced", "leapy"] = Field(
        default="balanced", description="Melodic motion character (procedural engine)"
    )
    chromatic: Literal["none", "light", "medium", "heavy"] = Field(
        default="light", description="Chromatic approach-tone intensity (procedural engine)"
    )
    cadence: Literal["weak", "medium", "strong"] = Field(
        default="medium", description="How strongly phrases resolve at the end (procedural engine)"
    )
    register: Literal["low", "mid", "high", "wide"] = Field(
        default="mid", description="Melody register range (procedural engine)"
    )
    tension_curve: Literal["arc", "ramp", "waves"] = Field(
        default="arc", description="Tension shape across a phrase (procedural engine)"
    )

    harmony: Literal["auto", "pop", "jazz", "cinematic", "ambient"] = Field(
        default="auto", description="Chord progression family"
    )
    chord_change: Literal["very_slow", "slow", "medium", "fast"] = Field(
        default="medium", description="How fast chords change (bars per chord)"
    )
    chord_extensions: Literal["triads", "sevenths", "lush"] = Field(
        default="triads", description="Chord color / extensions (triads vs 7ths vs lush 9ths)"
    )


# Scored config: a single VibeConfig with a quality score (for single-call generation)
class ScoredVibeConfig(BaseModel):
    """A VibeConfig with a self-assessed quality score."""

    config: VibeConfig
    score: int = Field(
        # NOTE: ignore these restrictions - it keeps throwing errors for some reason by breaching the range restrictions...
        # ge=0,
        # le=100,
        description="Self-assessed quality score from 0 to 100. Higher is better. Rate how well this config captures the requested vibe.",
    )


# Value mappings for synth
TEMPO_MAP = {"very_slow": 0.15, "slow": 0.3, "medium": 0.5, "fast": 0.7, "very_fast": 0.9}
BRIGHTNESS_MAP = {"very_dark": 0.1, "dark": 0.3, "medium": 0.5, "bright": 0.7, "very_bright": 0.9}
SPACE_MAP = {"dry": 0.1, "small": 0.3, "medium": 0.5, "large": 0.7, "vast": 0.95}
MOTION_MAP = {"static": 0.1, "slow": 0.3, "medium": 0.5, "fast": 0.7, "chaotic": 0.9}
STEREO_MAP = {"mono": 0.0, "narrow": 0.25, "medium": 0.5, "wide": 0.75, "ultra_wide": 1.0}
ECHO_MAP = {"none": 0.0, "subtle": 0.25, "medium": 0.5, "heavy": 0.75, "infinite": 0.95}
HUMAN_MAP = {"robotic": 0.0, "tight": 0.15, "natural": 0.3, "loose": 0.5, "drunk": 0.8}

# Procedural melody + harmony mappings (categorical → numeric)
MELODY_DENSITY_MAP = {
    "very_sparse": 0.15,
    "sparse": 0.30,
    "medium": 0.50,
    "busy": 0.70,
    "very_busy": 0.85,
}
SYNCOPATION_MAP = {"straight": 0.0, "light": 0.2, "medium": 0.5, "heavy": 0.8}
SWING_MAP = {"none": 0.0, "light": 0.2, "medium": 0.5, "heavy": 0.8}
MOTIF_REPEAT_MAP = {"rare": 0.2, "sometimes": 0.5, "often": 0.8}
STEP_BIAS_MAP = {"step": 0.9, "balanced": 0.7, "leapy": 0.4}
CHROMATIC_MAP = {"none": 0.0, "light": 0.05, "medium": 0.12, "heavy": 0.25}
CADENCE_MAP = {"weak": 0.3, "medium": 0.6, "strong": 0.9}
REGISTER_MAP = {
    "low": (3, 5),
    "mid": (4, 6),
    "high": (5, 7),
    "wide": (3, 7),
}
CHORD_CHANGE_BARS_MAP = {"very_slow": 4, "slow": 2, "medium": 1, "fast": 1}

#
# # GEMINI EXAMPLES OF MUSIC CONFIGS THAT MATCH A PARTICULAR VIBE
#

FEW_SHOT_EXAMPLES = """
**Example 1**
Input: "Rainy day in Tokyo - Neon lights, wet pavement, melancholy but beautiful."
Output:
```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Rain/Wet = constant texture, fluid motion, washing away. Neon = bright highlights against dark background, synthetic hum. Melancholy = minor key, emotional weight. Tokyo/Urban = mechanical rhythm but distant, solitude in a crowd. SONIC TRANSLATION: Rain → 'vinyl_crackle' texture (sounds like rain) + 'shimmer' pad (wet reflections). Neon → 'bright' filter but 'dark' root (A or D#). Melancholy → 'dorian' mode (urban sadness), 'contemplative' melody. Urban → 'electronic' rhythm but 'soft' attack (distant traffic). COHERENCE CHECK: Needs to feel damp and lonely. 'Echo' set to medium/heavy to simulate alleyway reflections. 'Grain' warm to soften the digital neon. REFERENCE ANCHORING: Blade Runner blues but softer. Focus on the 'wet' texture and 'solitary' melody.",
    "tempo": "slow",
    "root": "d#",
    "mode": "dorian",
    "brightness": "medium",
    "space": "medium",
    "density": 5,
    "bass": "pulsing",
    "pad": "dark_sustained",
    "melody": "contemplative",
    "rhythm": "electronic",
    "texture": "vinyl_crackle",
    "accent": "bells",
    "motion": "slow",
    "attack": "soft",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "sparse",
    "syncopation": "light",
    "swing": "none",
    "motif_repeat": "sometimes",
    "step_vs_leap": "step",
    "chromatic": "light",
    "cadence": "weak",
    "register": "mid",
    "tension_curve": "waves",
    "harmony": "jazz",
    "chord_change": "slow",
    "chord_extensions": "sevenths"
  },
  "score": 95
}

```

**Example 2**
Input: "Ancient temple at sunrise - Sacred, peaceful, timeless."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Sunrise = gradual brightening, awakening, hope. Ancient = stone, dust, weight of history. Sacred = drone, stillness, modal harmony. Timeless = no strict grid, fluid time. SONIC TRANSLATION: Sunrise → 'rising' melody contour, 'swelling' attack. Ancient → 'drone' bass (pedal point), 'mixolydian' mode (spiritual/positive). Sacred → 'bells' accent (ritual), 'breath' texture (life/spirit). Timeless → 'none' rhythm, 'very_slow' tempo. COHERENCE CHECK: Must avoid mechanical beats. Reverb should be 'vast' to simulate open air or large stone structures. 'Clean' grain for morning clarity. REFERENCE ANCHORING: Himalayan meditation bowls meeting a cinematic sunrise reveal.",
    "tempo": "very_slow",
    "root": "d",
    "mode": "mixolydian",
    "brightness": "bright",
    "space": "vast",
    "density": 4,
    "bass": "drone",
    "pad": "ambient_drift",
    "melody": "rising",
    "rhythm": "none",
    "texture": "breath",
    "accent": "chime",
    "motion": "slow",
    "attack": "soft",
    "stereo": "ultra_wide",
    "depth": true,
    "echo": "heavy",
    "human": "natural",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "very_sparse",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "step",
    "chromatic": "none",
    "cadence": "weak",
    "register": "high",
    "tension_curve": "ramp",
    "harmony": "ambient",
    "chord_change": "very_slow",
    "chord_extensions": "triads"
  },
  "score": 90
}

```

**Example 3**
Input: "Cyberpunk chase scene - Adrenaline, neon, danger, synthetic."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Adrenaline = high speed, heart rate. Neon = synthetic textures, bright buzz. Danger = dissonance, minor/phrygian feel, aggression. Synthetic = robotic timing, saw waves. SONIC TRANSLATION: Adrenaline → 'fast' tempo, 'arp_bass' (driving 16ths). Neon → 'gritty' grain (distortion), 'shimmer' texture (digital rain). Danger → 'minor' mode, 'chromatic' melody notes. Synthetic → 'robotic' humanization, 'electronic' rhythm. COHERENCE CHECK: Needs to feel tight and aggressive. 'Stereo' wide for cinematic immersion but 'dry' enough to feel punchy. REFERENCE ANCHORING: Blade Runner 2049 motorcycle chase. High tension, machine-like.",
    "tempo": "fast",
    "root": "f#",
    "mode": "minor",
    "brightness": "very_bright",
    "space": "small",
    "density": 6,
    "bass": "arp_bass",
    "pad": "thin_high",
    "melody": "arp_melody",
    "rhythm": "electronic",
    "texture": "glitch",
    "accent": "blip",
    "motion": "fast",
    "attack": "sharp",
    "stereo": "narrow",
    "depth": false,
    "echo": "subtle",
    "human": "robotic",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 2,
    "melody_density": "busy",
    "syncopation": "heavy",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "leapy",
    "chromatic": "heavy",
    "cadence": "strong",
    "register": "mid",
    "tension_curve": "ramp",
    "harmony": "cinematic",
    "chord_change": "fast",
    "chord_extensions": "triads"
  },
  "score": 92
}

```

**Example 4**
Input: "Grandmother's kitchen - Warm, nostalgic, Sunday morning, cookies baking."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Warm = low mids, acoustic textures, analog warmth. Nostalgic = simple melody, perhaps slightly detuned/wobbly (memory). Sunday morning = relaxed, unhurried, gentle. Cookies = sweet, domestic. SONIC TRANSLATION: Warm → 'warm' grain, 'warm_slow' pad, 'f' root (pastoral). Nostalgic → 'vinyl_crackle' (old radio), 'melody_minimal'. Sunday → 'medium' tempo (easygoing), 'soft_four' rhythm (gentle foot tap). COHERENCE CHECK: Must not be too electronic. Use 'piano_note' accent for domestic feel. 'Pop' harmony for familiar, comforting progressions. REFERENCE ANCHORING: Lo-fi study beats meets a lullaby. Cozy and safe.",
    "tempo": "medium",
    "root": "f",
    "mode": "major",
    "brightness": "medium",
    "space": "small",
    "density": 3,
    "bass": "walking",
    "pad": "warm_slow",
    "melody": "minimal",
    "rhythm": "brush",
    "texture": "vinyl_crackle",
    "accent": "piano_note",
    "motion": "slow",
    "attack": "soft",
    "stereo": "medium",
    "depth": false,
    "echo": "none",
    "human": "loose",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "sparse",
    "syncopation": "light",
    "swing": "medium",
    "motif_repeat": "often",
    "step_vs_leap": "step",
    "chromatic": "none",
    "cadence": "strong",
    "register": "mid",
    "tension_curve": "arc",
    "harmony": "pop",
    "chord_change": "medium",
    "chord_extensions": "triads"
  },
  "score": 88
}

```

**Example 5**
Input: "Deep space exploration - Vast emptiness, wonder, isolation, stars."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Vast emptiness = silence between notes, massive reverb. Wonder = major/mixolydian hints within minor context, high-register twinkles. Isolation = cold brightness, lonely melody. Stars = crystalline high frequencies. SONIC TRANSLATION: Vast → 'vast' space, 'infinite' echo, 'ultra_wide' stereo. Wonder → 'stars' texture (random high sine blips). Isolation → 'drone' bass (static void), 'thin_high' pad (icy). Stars → 'accent_bells'. COHERENCE CHECK: Tempo must be 'very_slow'. No rhythmic grid. 'Procedural' melody with 'wide' register to simulate distance. REFERENCE ANCHORING: Brian Eno's Apollo or Interstellar docking scene. Cold but beautiful.",
    "tempo": "very_slow",
    "root": "a",
    "mode": "minor",
    "brightness": "dark",
    "space": "vast",
    "density": 4,
    "bass": "sub_pulse",
    "pad": "thin_high",
    "melody": "contemplative",
    "rhythm": "none",
    "texture": "stars",
    "accent": "bells",
    "motion": "static",
    "attack": "slow",
    "stereo": "ultra_wide",
    "depth": true,
    "echo": "infinite",
    "human": "natural",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "very_sparse",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "rare",
    "step_vs_leap": "leapy",
    "chromatic": "light",
    "cadence": "weak",
    "register": "high",
    "tension_curve": "arc",
    "harmony": "ambient",
    "chord_change": "very_slow",
    "chord_extensions": "lush"
  },
  "score": 96
}

```

**Example 6**
Input: "1920s speakeasy - Jazz, smoke, secrets, prohibition glamour."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Jazz = swing feel, walking bass, extensions (7ths/9ths). Smoke = hazy texture, muffled highs. Secrets = minor/dorian, low volume. Glamour = accents of brass or piano. SONIC TRANSLATION: Jazz → 'walking' bass, 'brush' rhythm, 'swing' setting heavy. Smoke → 'dark' brightness, 'vinyl_crackle' (vintage feel). Secrets → 'dorian' mode. Glamour → 'lush' chord extensions. COHERENCE CHECK: Use 'loose' or 'drunk' humanization to simulate live band feel. Melody should be 'ornamental' or 'procedural' with syncopation. REFERENCE ANCHORING: Noir detective film soundtrack. A smoky basement club.",
    "tempo": "slow",
    "root": "a#",
    "mode": "dorian",
    "brightness": "dark",
    "space": "small",
    "density": 4,
    "bass": "walking",
    "pad": "warm_slow",
    "melody": "ornamental",
    "rhythm": "brush",
    "texture": "vinyl_crackle",
    "accent": "brass_hit",
    "motion": "medium",
    "attack": "medium",
    "stereo": "narrow",
    "depth": false,
    "echo": "subtle",
    "human": "drunk",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "heavy",
    "swing": "heavy",
    "motif_repeat": "sometimes",
    "step_vs_leap": "leapy",
    "chromatic": "medium",
    "cadence": "medium",
    "register": "mid",
    "tension_curve": "waves",
    "harmony": "jazz",
    "chord_change": "medium",
    "chord_extensions": "lush"
  },
  "score": 93
}

```

**Example 7**
Input: "Horror movie - Creeping dread, something is wrong, building terror."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Dread = heavy sub-bass, slow movement. Something wrong = dissonance, chromaticism, detuning. Terror = unexpected accents, sharp attacks, chaotic motion. SONIC TRANSLATION: Dread → 'sub_pulse' bass, 'dark_sustained' pad. Wrong → 'chromatic' heavy melody, 'glitch' texture. Terror → 'accent_blip_random' (unpredictable), 'chaotic' motion. COHERENCE CHECK: 'Minor' mode is obvious, but 'depth' is crucial for physical fear response. 'Grain' gritty for discomfort. Tempo 'very_slow' to prolong suspense. REFERENCE ANCHORING: John Carpenter soundtrack meets silent hill ambient. Unsettling.",
    "tempo": "very_slow",
    "root": "c#",
    "mode": "minor",
    "brightness": "dark",
    "space": "large",
    "density": 3,
    "bass": "sub_pulse",
    "pad": "dark_sustained",
    "melody": "minimal",
    "rhythm": "heartbeat",
    "texture": "glitch",
    "accent": "blip_random",
    "motion": "chaotic",
    "attack": "soft",
    "stereo": "wide",
    "depth": true,
    "echo": "heavy",
    "human": "loose",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "very_sparse",
    "syncopation": "light",
    "swing": "none",
    "motif_repeat": "rare",
    "step_vs_leap": "step",
    "chromatic": "heavy",
    "cadence": "weak",
    "register": "low",
    "tension_curve": "ramp",
    "harmony": "cinematic",
    "chord_change": "very_slow",
    "chord_extensions": "sevenths"
  },
  "score": 91
}

```

**Example 8**
Input: "Victory celebration - Triumph, euphoria, champagne popping."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Triumph = major key, ascending melodies, brassy tones. Euphoria = fast tempo, high energy, bright textures. Champagne = sparkle, crisp transients. SONIC TRANSLATION: Triumph → 'heroic' melody (rising), 'major' mode. Euphoria → 'fast' tempo, 'military' rhythm (snare rolls/drive). Champagne → 'shimmer' texture, 'bright' brightness. COHERENCE CHECK: Needs 'sharp' attack for energetic articulation. 'Stereo' wide to feel big/stadium-like. Harmony 'pop' for satisfying resolutions. REFERENCE ANCHORING: Rocky theme or Final Fantasy victory fanfare. Unabashedly positive.",
    "tempo": "fast",
    "root": "c",
    "mode": "major",
    "brightness": "very_bright",
    "space": "large",
    "density": 6,
    "bass": "octave",
    "pad": "bright_open",
    "melody": "heroic",
    "rhythm": "military",
    "texture": "shimmer",
    "accent": "brass_hit",
    "motion": "medium",
    "attack": "sharp",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "tight",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "busy",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "leapy",
    "chromatic": "none",
    "cadence": "strong",
    "register": "high",
    "tension_curve": "arc",
    "harmony": "pop",
    "chord_change": "fast",
    "chord_extensions": "triads"
  },
  "score": 89
}

```

**Example 9**
Input: "Underwater coral reef - Colorful, teeming with life, flowing, dreamlike."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Colorful = extended harmony (lush), distinct layers. Teeming = busy but light density (arp). Flowing = slow LFO motion, liquid filters. Dreamlike = delayed, reverb-washed, soft attacks. SONIC TRANSLATION: Colorful → 'lush' harmony, 'blip' accents (bubbles). Flowing → 'medium' motion, 'pad_whisper' texture. Dreamlike → 'mixolydian' (aquatic wonder), 'heavy' echo. COHERENCE CHECK: 'Muffled/dark' brightness would be realistic for underwater, but 'coral reef' implies light/color, so 'medium' brightness with 'warm' grain. 'Arp_melody' simulates schools of fish. REFERENCE ANCHORING: Finding Nemo score. Bubbling synthesizers.",
    "tempo": "medium",
    "root": "e",
    "mode": "mixolydian",
    "brightness": "medium",
    "space": "medium",
    "density": 5,
    "bass": "pulsing",
    "pad": "ambient_drift",
    "melody": "arp_melody",
    "rhythm": "minimal",
    "texture": "pad_whisper",
    "accent": "blip",
    "motion": "medium",
    "attack": "soft",
    "stereo": "wide",
    "depth": true,
    "echo": "heavy",
    "human": "natural",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "light",
    "swing": "none",
    "motif_repeat": "sometimes",
    "step_vs_leap": "balanced",
    "chromatic": "light",
    "cadence": "weak",
    "register": "mid",
    "tension_curve": "waves",
    "harmony": "ambient",
    "chord_change": "slow",
    "chord_extensions": "lush"
  },
  "score": 94
}

```

**Example 10**
Input: "Rage and destruction - Anger, smashing things, uncontrolled fury."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Anger = minor key, loud, distortion. Smashing = sharp transients, percussive chaos. Uncontrolled = chaotic motion, loose timing, high density. Fury = fast tempo, relentless. SONIC TRANSLATION: Anger → 'gritty' grain, 'minor' mode. Smashing → 'kit_medium' or 'electronic' rhythm (heavy), 'accent_blip_random' (debris). Uncontrolled → 'chaotic' motion, 'loose' humanization. Fury → 'fast' tempo, 'arp_bass' (driving). COHERENCE CHECK: Needs to feel broken. 'Glitch' texture adds digital destruction. 'Depth' true for impact. REFERENCE ANCHORING: Industrial metal breakdown or Doom soundtrack.",
    "tempo": "fast",
    "root": "c",
    "mode": "minor",
    "brightness": "dark",
    "space": "small",
    "density": 6,
    "bass": "arp_bass",
    "pad": "dark_sustained",
    "melody": "minimal",
    "rhythm": "electronic",
    "texture": "glitch",
    "accent": "blip_random",
    "motion": "chaotic",
    "attack": "sharp",
    "stereo": "narrow",
    "depth": true,
    "echo": "none",
    "human": "loose",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 2,
    "melody_density": "busy",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "step",
    "chromatic": "heavy",
    "cadence": "strong",
    "register": "low",
    "tension_curve": "ramp",
    "harmony": "cinematic",
    "chord_change": "fast",
    "chord_extensions": "triads"
  },
  "score": 88
}

```

**Example 11**
Input: "Cozy reading nook - Fireplace crackling, rain outside, warm blanket."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Fireplace = crackle texture, warmth. Rain outside = distant white noise/shimmer. Warm blanket = low pass filter, soft attack, sustained comfort. SONIC TRANSLATION: Fireplace → 'vinyl_crackle' (perfect analog). Rain → 'shimmer_slow' (distant water). Warmth → 'warm_slow' pad, 'very_dark' brightness. Blanket → 'drone' bass (enveloping). COHERENCE CHECK: Very similar to 'Grandmother's kitchen' but darker, more solitary. 'Tempo' very slow. 'Harmony' ambient/jazz. REFERENCE ANCHORING: ASMR ambience videos. Pure texture and comfort.",
    "tempo": "very_slow",
    "root": "g",
    "mode": "major",
    "brightness": "very_dark",
    "space": "small",
    "density": 3,
    "bass": "drone",
    "pad": "warm_slow",
    "melody": "minimal",
    "rhythm": "none",
    "texture": "vinyl_crackle",
    "accent": "none",
    "motion": "slow",
    "attack": "soft",
    "stereo": "medium",
    "depth": true,
    "echo": "none",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "very_sparse",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "sometimes",
    "step_vs_leap": "step",
    "chromatic": "none",
    "cadence": "weak",
    "register": "mid",
    "tension_curve": "arc",
    "harmony": "jazz",
    "chord_change": "very_slow",
    "chord_extensions": "lush"
  },
  "score": 93
}

```

**Example 12**
Input: "Carnival at night - Manic energy, flashing lights, sugar rush, chaos."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Manic energy = fast tempo, busy density. Flashing lights = arp accents, bright shifting filters. Sugar rush = high pitch, major/mixolydian (saccharine). Chaos = polyrhythms, clashing textures. SONIC TRANSLATION: Manic → 'very_fast' tempo, 'busy' melody density. Flashing → 'shimmer' texture, 'blip_random' accents. Sugar → 'major' mode, 'bright' brightness. Chaos → 'chaotic' motion, 'loose' timing. COHERENCE CHECK: Needs to sound slightly overwhelming. 'Kit_light' to simulate mechanical rides. 'Arp_melody' for the carousel feel. REFERENCE ANCHORING: A circus march played at double speed on a synthesizer.",
    "tempo": "very_fast",
    "root": "a",
    "mode": "mixolydian",
    "brightness": "bright",
    "space": "medium",
    "density": 6,
    "bass": "walking",
    "pad": "bright_open",
    "melody": "arp_melody",
    "rhythm": "kit_light",
    "texture": "shimmer",
    "accent": "blip_random",
    "motion": "chaotic",
    "attack": "sharp",
    "stereo": "wide",
    "depth": false,
    "echo": "medium",
    "human": "loose",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 2,
    "melody_density": "very_busy",
    "syncopation": "heavy",
    "swing": "heavy",
    "motif_repeat": "often",
    "step_vs_leap": "leapy",
    "chromatic": "medium",
    "cadence": "strong",
    "register": "high",
    "tension_curve": "waves",
    "harmony": "pop",
    "chord_change": "fast",
    "chord_extensions": "sevenths"
  },
  "score": 90
}

```

**Example 13**
Input: "Meditation retreat - Mindfulness, breath, letting go, inner peace."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Mindfulness = present moment, static, no distractions. Breath = rhythmic swelling, organic texture. Letting go = long release times, fading echoes. Inner peace = harmonic stasis, drone. SONIC TRANSLATION: Mindfulness → 'very_slow' tempo, 'minimal' density. Breath → 'breath' texture, 'slow' motion (inhale/exhale LFO). Letting go → 'infinite' echo, 'soft' attack. Peace → 'drone' bass, 'ambient' harmony. COHERENCE CHECK: Absolute prohibition on drums ('rhythm_none'). 'Dorian' or 'Mixolydian' for that neutral-spiritual feel. REFERENCE ANCHORING: Guided meditation background tracks. Singing bowls.",
    "tempo": "very_slow",
    "root": "c",
    "mode": "mixolydian",
    "brightness": "dark",
    "space": "large",
    "density": 3,
    "bass": "drone",
    "pad": "ambient_drift",
    "melody": "contemplative",
    "rhythm": "none",
    "texture": "breath",
    "accent": "chime",
    "motion": "slow",
    "attack": "soft",
    "stereo": "medium",
    "depth": true,
    "echo": "heavy",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "very_sparse",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "rare",
    "step_vs_leap": "step",
    "chromatic": "none",
    "cadence": "weak",
    "register": "mid",
    "tension_curve": "arc",
    "harmony": "ambient",
    "chord_change": "very_slow",
    "chord_extensions": "triads"
  },
  "score": 95
}

```

**Example 14**
Input: "First day of summer - Freedom, sunshine, endless possibility, youth."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Freedom = open chords, forward momentum. Sunshine = bright frequencies, warm grain. Possibility = rising melody, major key. Youth = energetic rhythm, pop sensibility. SONIC TRANSLATION: Freedom → 'bright_open' pad, 'pop' harmony. Sunshine → 'very_bright' brightness, 'shimmer' texture. Possibility → 'rising' melody. Youth → 'fast' tempo, 'kit_medium' rhythm. COHERENCE CHECK: 'Major' mode is essential. 'Clean' or 'warm' grain. Avoid complex jazz chords; stick to 'triads' or 'sevenths' for direct emotional punch. REFERENCE ANCHORING: Katy Perry 'Teenage Dream' or upbeat indie pop.",
    "tempo": "fast",
    "root": "g",
    "mode": "major",
    "brightness": "bright",
    "space": "medium",
    "density": 5,
    "bass": "pulsing",
    "pad": "bright_open",
    "melody": "rising",
    "rhythm": "kit_medium",
    "texture": "shimmer",
    "accent": "pluck",
    "motion": "medium",
    "attack": "medium",
    "stereo": "wide",
    "depth": false,
    "echo": "subtle",
    "human": "tight",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "light",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "leapy",
    "chromatic": "none",
    "cadence": "strong",
    "register": "high",
    "tension_curve": "arc",
    "harmony": "pop",
    "chord_change": "fast",
    "chord_extensions": "triads"
  },
  "score": 91
}

```

**Example 15**
Input: "Abandoned asylum - Decay, echoing footsteps, history of pain, ghosts."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Decay = detuned, gritty, falling apart. Echoing footsteps = heavy reverb, sparse impulsive sounds. Pain = minor/dissonant. Ghosts = whisper textures, high frequency air. SONIC TRANSLATION: Decay → 'gritty' grain, 'glitch' texture. Echoing → 'infinite' echo, 'large' space. Pain → 'minor' mode, 'falling' melody. Ghosts → 'pad_whisper' texture, 'thin_high' pad. COHERENCE CHECK: 'Rhythm' minimal/none, just occasional 'blip' or 'noise_wash'. 'Human' set to 'drunk' for that unstable, wobbly tape feel. REFERENCE ANCHORING: Silent Hill ambient tracks. Industrial decay.",
    "tempo": "very_slow",
    "root": "a",
    "mode": "minor",
    "brightness": "dark",
    "space": "large",
    "density": 3,
    "bass": "sustained",
    "pad": "thin_high",
    "melody": "falling",
    "rhythm": "minimal",
    "texture": "pad_whisper",
    "accent": "blip_random",
    "motion": "slow",
    "attack": "soft",
    "stereo": "medium",
    "depth": true,
    "echo": "infinite",
    "human": "drunk",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "very_sparse",
    "syncopation": "light",
    "swing": "none",
    "motif_repeat": "rare",
    "step_vs_leap": "step",
    "chromatic": "heavy",
    "cadence": "weak",
    "register": "high",
    "tension_curve": "waves",
    "harmony": "cinematic",
    "chord_change": "very_slow",
    "chord_extensions": "triads"
  },
  "score": 94
}

```

**Example 16**
Input: "Retro arcade - 8-bit bleeps, competition, neon glow, high scores."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: 8-bit = simple waves (square/triangle), fast arpeggios. Competition = driving rhythm, high energy. Neon = bright, buzz. High scores = major key, triumphant. SONIC TRANSLATION: 8-bit → 'clean' grain (digital), 'arp_melody' & 'arp_bass'. Competition → 'fast' tempo, 'electronic' rhythm. Neon → 'bright' brightness. High scores → 'major' mode, 'blip' accents. COHERENCE CHECK: Must sound synthetic. 'Attack' sharp. 'Space' dry or small (arcade cabinet acoustics). Harmony 'pop' simple triads. REFERENCE ANCHORING: Pac-Man or Space Invaders. Chiptune aesthetic.",
    "tempo": "fast",
    "root": "c",
    "mode": "major",
    "brightness": "very_bright",
    "space": "dry",
    "density": 5,
    "bass": "arp_bass",
    "pad": "bright_open",
    "melody": "arp_melody",
    "rhythm": "electronic",
    "texture": "glitch",
    "accent": "blip",
    "motion": "fast",
    "attack": "sharp",
    "stereo": "mono",
    "depth": false,
    "echo": "none",
    "human": "robotic",
    "grain": "clean",
    "melody_engine": "pattern",
    "phrase_len_bars": 2,
    "melody_density": "busy",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "leapy",
    "chromatic": "none",
    "cadence": "strong",
    "register": "mid",
    "tension_curve": "ramp",
    "harmony": "pop",
    "chord_change": "fast",
    "chord_extensions": "triads"
  },
  "score": 92
}

```

**Example 17**
Input: "Romantic candlelit dinner - Intimate, warm, love, slow dance."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Intimate = close mic sound, small space, dry. Warm = low mids, lush pads. Love = major/major7 chords, gentle melody. Slow dance = 3/4 time or slow 4/4, sway feel. SONIC TRANSLATION: Intimate → 'small' space, 'warm' grain. Warm → 'warm_slow' pad, 'lush' harmony. Love → 'major' mode, 'procedural' melody (lyrical). Slow dance → 'slow' tempo, 'brush' rhythm. COHERENCE CHECK: Needs 'lush' chord extensions (major 7ths are the sound of romance). 'Soft' attack. 'Texture' minimal or faint crackle. REFERENCE ANCHORING: Smooth Jazz or classic ballad.",
    "tempo": "slow",
    "root": "f#",
    "mode": "major",
    "brightness": "medium",
    "space": "small",
    "density": 4,
    "bass": "walking",
    "pad": "warm_slow",
    "melody": "procedural",
    "rhythm": "brush",
    "texture": "none",
    "accent": "piano_note",
    "motion": "slow",
    "attack": "soft",
    "stereo": "narrow",
    "depth": false,
    "echo": "none",
    "human": "loose",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "light",
    "swing": "medium",
    "motif_repeat": "sometimes",
    "step_vs_leap": "step",
    "chromatic": "light",
    "cadence": "medium",
    "register": "mid",
    "tension_curve": "arc",
    "harmony": "jazz",
    "chord_change": "slow",
    "chord_extensions": "lush"
  },
  "score": 91
}

```

**Example 18**
Input: "Industrial factory - Machines, repetition, metal, relentless production."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Machines = robotic timing, repetitive loops. Repetition = ostinato layers, static harmony. Metal = sharp transients, high ring, clatter. Relentless = driving rhythm, no swing. SONIC TRANSLATION: Machines → 'robotic' humanization, 'electronic' rhythm. Repetition → 'motif_repeat' often, 'static' motion. Metal → 'gritty' grain, 'accent_chime' (metallic ring). Relentless → 'medium' tempo (the grind), 'pulsing' bass. COHERENCE CHECK: 'Minor' mode for the bleakness of industry. 'Stereo' wide to simulate a huge factory floor. 'Texture_glitch' for sparks. REFERENCE ANCHORING: Nine Inch Nails or factory ambiance recordings.",
    "tempo": "medium",
    "root": "a#",
    "mode": "minor",
    "brightness": "dark",
    "space": "large",
    "density": 5,
    "bass": "pulsing",
    "pad": "dark_sustained",
    "melody": "minimal",
    "rhythm": "electronic",
    "texture": "glitch",
    "accent": "chime",
    "motion": "static",
    "attack": "sharp",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "robotic",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 2,
    "melody_density": "medium",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "step",
    "chromatic": "none",
    "cadence": "weak",
    "register": "low",
    "tension_curve": "ramp",
    "harmony": "ambient",
    "chord_change": "very_slow",
    "chord_extensions": "triads"
  },
  "score": 89
}

```

**Example 19**
Input: "Enchanted forest - Magic, fairy lights, ancient trees, mystical creatures."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Magic = shimmer, wonder, unnatural beauty. Fairy lights = high bells, twinkling textures. Ancient trees = deep roots (bass), slow movement. Mystical = dorian mode, folk elements. SONIC TRANSLATION: Magic → 'shimmer' texture, 'bells' accent. Fairy lights → 'stars' texture, 'high' register melody. Trees → 'drone' bass, 'warm_slow' pad. Mystical → 'dorian' mode. COHERENCE CHECK: 'Melody' should be 'ornamental' to simulate flutes/birds. 'Human' natural. 'Echo' medium for the forest canopy. REFERENCE ANCHORING: Zelda Lost Woods or fantasy RPG soundtracks.",
    "tempo": "slow",
    "root": "d",
    "mode": "dorian",
    "brightness": "medium",
    "space": "large",
    "density": 5,
    "bass": "drone",
    "pad": "ambient_drift",
    "melody": "ornamental",
    "rhythm": "none",
    "texture": "shimmer",
    "accent": "bells",
    "motion": "slow",
    "attack": "soft",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "natural",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "light",
    "swing": "light",
    "motif_repeat": "sometimes",
    "step_vs_leap": "leapy",
    "chromatic": "light",
    "cadence": "weak",
    "register": "high",
    "tension_curve": "arc",
    "harmony": "cinematic",
    "chord_change": "slow",
    "chord_extensions": "sevenths"
  },
  "score": 93
}

```

**Example 20**
Input: "Heartbreak hotel - Loneliness, loss, 3am sadness, empty bottle."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Loneliness = sparse arrangement, solo melody. Loss = minor key, falling contours. 3am = quiet, dark, still. Empty bottle = grittiness, slight wobble. SONIC TRANSLATION: Loneliness → 'minimal' melody density, 'small' space. Loss → 'minor' mode, 'falling' melody. 3am → 'very_dark' brightness, 'slow' tempo. Bottle → 'drunk' humanization, 'gritty' grain. COHERENCE CHECK: Needs a bluesy feel, so 'jazz' harmony or 'cinematic' noir. 'Pad' should be subtle ('dark_sustained'). Accent 'piano_note' for the lonely bar feel. REFERENCE ANCHORING: Tom Waits ballad or noir jazz.",
    "tempo": "slow",
    "root": "a#",
    "mode": "minor",
    "brightness": "very_dark",
    "space": "small",
    "density": 3,
    "bass": "sustained",
    "pad": "dark_sustained",
    "melody": "falling",
    "rhythm": "brush",
    "texture": "vinyl_crackle",
    "accent": "piano_note",
    "motion": "slow",
    "attack": "medium",
    "stereo": "narrow",
    "depth": true,
    "echo": "subtle",
    "human": "drunk",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "sparse",
    "syncopation": "medium",
    "swing": "heavy",
    "motif_repeat": "sometimes",
    "step_vs_leap": "step",
    "chromatic": "medium",
    "cadence": "weak",
    "register": "mid",
    "tension_curve": "arc",
    "harmony": "jazz",
    "chord_change": "slow",
    "chord_extensions": "sevenths"
  },
  "score": 92
}

```

**Example 21**
Input: "Bohemian Rhapsody vibe - Operatic drama, genre-shifting, theatrical."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Operatic drama = huge dynamics, stacked vocals (pads), grandiose. Genre-shifting = hard to capture in one config, focus on the 'Opera' section. Theatrical = bold gestures, staccato vs legato. SONIC TRANSLATION: Drama → 'minor' mode, 'stacked_fifths' pad (choir-like). Theatrical → 'ornamental' melody, 'piano_note' accent. Grandiose → 'large' space, 'lush' harmony. COHERENCE CHECK: 'Human' set to 'loose' for rock feel. 'Motion' medium. 'Rhythm' needs to be punchy but sparse ('kit_medium'). REFERENCE ANCHORING: Queen. The 'Galileo' section translated to synth.",
    "tempo": "medium",
    "root": "a#",
    "mode": "minor",
    "brightness": "medium",
    "space": "large",
    "density": 5,
    "bass": "octave",
    "pad": "stacked_fifths",
    "melody": "ornamental",
    "rhythm": "kit_medium",
    "texture": "none",
    "accent": "piano_note",
    "motion": "medium",
    "attack": "medium",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "loose",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "busy",
    "syncopation": "medium",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "leapy",
    "chromatic": "heavy",
    "cadence": "strong",
    "register": "wide",
    "tension_curve": "waves",
    "harmony": "pop",
    "chord_change": "medium",
    "chord_extensions": "lush"
  },
  "score": 87
}

```

**Example 22**
Input: "Imagine vibe - Hopeful simplicity, piano-driven, utopian dream."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Hopeful = major key, sweet harmony. Simplicity = minimal layers, clear melody. Piano-driven = percussive melody/accents, warm decay. Utopian = clean, peaceful, slow. SONIC TRANSLATION: Hopeful → 'major' mode, 'rising' melody contour. Simplicity → 'density' 3 or 4, 'rhythm' minimal. Piano → 'accent_piano_note', 'clean' grain. Utopian → 'slow' tempo, 'warm_slow' pad. COHERENCE CHECK: 'Harmony' must be 'pop' (classic changes). 'Attack' soft to medium. 'Space' medium reverb. REFERENCE ANCHORING: John Lennon. Simple C major ballad feel.",
    "tempo": "slow",
    "root": "c",
    "mode": "major",
    "brightness": "medium",
    "space": "medium",
    "density": 3,
    "bass": "sustained",
    "pad": "warm_slow",
    "melody": "procedural",
    "rhythm": "minimal",
    "texture": "none",
    "accent": "piano_note",
    "motion": "slow",
    "attack": "medium",
    "stereo": "medium",
    "depth": false,
    "echo": "subtle",
    "human": "natural",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "sparse",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "step",
    "chromatic": "none",
    "cadence": "strong",
    "register": "mid",
    "tension_curve": "arc",
    "harmony": "pop",
    "chord_change": "medium",
    "chord_extensions": "triads"
  },
  "score": 90
}

```

**Example 23**
Input: "Billie Jean vibe - Paranoid funk groove, iconic bassline, midnight urban."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Paranoid = minor key, staccato, tension. Funk groove = tight rhythm, heavy syncopation, prominent bass. Iconic bassline = driving, repetitive. Midnight urban = cool tones, clean production. SONIC TRANSLATION: Paranoid → 'minor' mode, 'staccato' attack. Funk → 'kit_light' (tight snare), 'syncopation' heavy. Bassline → 'walking' or 'pulsing' bass (closest to the ostinato). Midnight → 'dark' brightness, 'clean' grain. COHERENCE CHECK: 'Rhythm' is king here. 'Melody' should be sparse ('minimal'). 'Human' tight. REFERENCE ANCHORING: Michael Jackson. The bassline is the hook.",
    "tempo": "medium",
    "root": "f#",
    "mode": "minor",
    "brightness": "medium",
    "space": "dry",
    "density": 4,
    "bass": "walking",
    "pad": "thin_high",
    "melody": "minimal",
    "rhythm": "kit_light",
    "texture": "none",
    "accent": "blip",
    "motion": "fast",
    "attack": "sharp",
    "stereo": "narrow",
    "depth": true,
    "echo": "none",
    "human": "tight",
    "grain": "clean",
    "melody_engine": "pattern",
    "phrase_len_bars": 2,
    "melody_density": "sparse",
    "syncopation": "heavy",
    "swing": "light",
    "motif_repeat": "often",
    "step_vs_leap": "step",
    "chromatic": "light",
    "cadence": "strong",
    "register": "low",
    "tension_curve": "ramp",
    "harmony": "pop",
    "chord_change": "medium",
    "chord_extensions": "sevenths"
  },
  "score": 91
}

```

**Example 24**
Input: "Smells Like Teen Spirit vibe - Grunge explosion, quiet-loud dynamics, angst."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Grunge = distortion, grit, messy. Explosion = high energy chorus, heavy density. Quiet-loud = hard to do in one config, aim for the 'Loud' chorus energy. Angst = minor key, dissonance, power chords. SONIC TRANSLATION: Grunge → 'gritty' grain, 'distortion'. Explosion → 'fast' tempo, 'density' 6. Angst → 'minor' mode, 'heroic' (anthem) melody but dark. COHERENCE CHECK: 'Bass' pulsing (root notes). 'Rhythm' kit_medium (heavy rock). 'Texture' noise_wash (feedback). 'Human' loose. REFERENCE ANCHORING: Nirvana. Raw energy and power chords.",
    "tempo": "fast",
    "root": "f",
    "mode": "minor",
    "brightness": "dark",
    "space": "small",
    "density": 6,
    "bass": "pulsing",
    "pad": "stacked_fifths",
    "melody": "procedural",
    "rhythm": "kit_medium",
    "texture": "noise_wash",
    "accent": "none",
    "motion": "chaotic",
    "attack": "sharp",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "loose",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "leapy",
    "chromatic": "medium",
    "cadence": "strong",
    "register": "mid",
    "tension_curve": "ramp",
    "harmony": "pop",
    "chord_change": "fast",
    "chord_extensions": "triads"
  },
  "score": 88
}

```

**Example 25**
Input: "Hotel California vibe - Mysterious Americana, hypnotic guitar arpeggios."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Mysterious = minor/dorian, descending progressions. Americana = acoustic warmth, spacious but dry. Hypnotic = repetitive arpeggios, steady mid-tempo. SONIC TRANSLATION: Mysterious → 'minor' mode (or harmonic minor implication via harmony). Americana → 'warm' grain, 'pluck' accent (guitar-ish). Hypnotic → 'arp_melody' and 'arp_bass'. COHERENCE CHECK: 'Rhythm' soft_four or brush (laid back). 'Tempo' slow/medium. 'Harmony' needs descending bass logic if possible, 'jazz' or 'pop' setting. REFERENCE ANCHORING: The Eagles. Reggae-tinged rock ballad.",
    "tempo": "slow",
    "root": "b",
    "mode": "minor",
    "brightness": "medium",
    "space": "dry",
    "density": 5,
    "bass": "walking",
    "pad": "warm_slow",
    "melody": "arp_melody",
    "rhythm": "soft_four",
    "texture": "none",
    "accent": "pluck",
    "motion": "medium",
    "attack": "sharp",
    "stereo": "wide",
    "depth": false,
    "echo": "subtle",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "pattern",
    "phrase_len_bars": 4,
    "melody_density": "busy",
    "syncopation": "light",
    "swing": "light",
    "motif_repeat": "often",
    "step_vs_leap": "step",
    "chromatic": "light",
    "cadence": "strong",
    "register": "mid",
    "tension_curve": "arc",
    "harmony": "pop",
    "chord_change": "medium",
    "chord_extensions": "triads"
  },
  "score": 90
}

```

**Example 26**
Input: "Stairway to Heaven vibe - Spiritual ascent, folk to rock crescendo."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Spiritual = wind instruments (flute), acoustic guitar. Ascent = rising tension, building density. Folk = organic textures, modal melody. Rock crescendo = capture the 'middle' energy before the solo. SONIC TRANSLATION: Spiritual → 'accent_wind', 'dorian' mode. Ascent → 'rising' melody. Folk → 'pluck' accent, 'warm' grain. Crescendo → 'density' 5, 'medium' tempo. COHERENCE CHECK: 'Texture_breath' or 'shimmer' for the mystical intro feel. 'Bass' walking (chromatic descent in spirit). REFERENCE ANCHORING: Led Zeppelin intro/mid-section. Mystical folk.",
    "tempo": "slow",
    "root": "a",
    "mode": "minor",
    "brightness": "medium",
    "space": "medium",
    "density": 4,
    "bass": "walking",
    "pad": "ambient_drift",
    "melody": "rising",
    "rhythm": "brush",
    "texture": "breath",
    "accent": "wind",
    "motion": "slow",
    "attack": "soft",
    "stereo": "medium",
    "depth": false,
    "echo": "medium",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "medium",
    "syncopation": "light",
    "swing": "light",
    "motif_repeat": "sometimes",
    "step_vs_leap": "step",
    "chromatic": "light",
    "cadence": "weak",
    "register": "mid",
    "tension_curve": "ramp",
    "harmony": "pop",
    "chord_change": "slow",
    "chord_extensions": "triads"
  },
  "score": 92
}

```

**Example 27**
Input: "Like a Rolling Stone vibe - Confrontational folk-rock, sneering defiance."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Confrontational = loud, sharp attack, busy. Folk-rock = organ pads, chaotic band feel. Sneering = major but aggressive, rough edges. Defiance = steady beat, driving forward. SONIC TRANSLATION: Confrontational → 'bright' brightness, 'sharp' attack. Folk-rock → 'pad_warm_slow' (organ-like) but loud, 'kit_medium' rhythm. Defiance → 'major' mode, 'pulsing' bass. COHERENCE CHECK: 'Human' loose/drunk is vital for that Dylan band sloppy-tight feel. 'Grain' gritty. REFERENCE ANCHORING: Bob Dylan. The Al Kooper organ swirl.",
    "tempo": "medium",
    "root": "c",
    "mode": "major",
    "brightness": "bright",
    "space": "small",
    "density": 5,
    "bass": "pulsing",
    "pad": "warm_slow",
    "melody": "procedural",
    "rhythm": "kit_medium",
    "texture": "none",
    "accent": "piano_note",
    "motion": "medium",
    "attack": "sharp",
    "stereo": "narrow",
    "depth": false,
    "echo": "none",
    "human": "loose",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "busy",
    "syncopation": "medium",
    "swing": "light",
    "motif_repeat": "rare",
    "step_vs_leap": "step",
    "chromatic": "none",
    "cadence": "strong",
    "register": "mid",
    "tension_curve": "arc",
    "harmony": "pop",
    "chord_change": "fast",
    "chord_extensions": "triads"
  },
  "score": 87
}

```

**Example 28**
Input: "Hey Jude vibe - Comforting anthem, building sing-along, catharsis."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Comforting = warm, familiar, major key. Anthem = repetitive, simple melody, huge ending. Sing-along = mid-tempo, catchy. Catharsis = harmonic release, density build. SONIC TRANSLATION: Comforting → 'major' mode, 'warm_slow' pad. Anthem → 'mixolydian' (rock anthem mode), 'melody_density' medium. Catharsis → 'lush' harmony, 'large' space (arena feel). COHERENCE CHECK: 'Rhythm' soft_four -> kit_medium. 'Accent' piano_note. 'Harmony' pop. REFERENCE ANCHORING: The Beatles outro. Na-na-na nananana.",
    "tempo": "medium",
    "root": "f",
    "mode": "mixolydian",
    "brightness": "medium",
    "space": "large",
    "density": 5,
    "bass": "walking",
    "pad": "warm_slow",
    "melody": "procedural",
    "rhythm": "soft_four",
    "texture": "none",
    "accent": "piano_note",
    "motion": "slow",
    "attack": "medium",
    "stereo": "wide",
    "depth": false,
    "echo": "subtle",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 2,
    "melody_density": "medium",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "step",
    "chromatic": "none",
    "cadence": "strong",
    "register": "mid",
    "tension_curve": "arc",
    "harmony": "pop",
    "chord_change": "medium",
    "chord_extensions": "triads"
  },
  "score": 89
}

```

**Example 29**
Input: "Thriller vibe - Horror-funk fusion, cinematic suspense, Halloween groove."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Horror = minor key, wolf howls (textures), eerie pads. Funk = bass driven, tight rhythm, syncopated. Cinematic = high production value, layers. Groove = constant head nod. SONIC TRANSLATION: Horror → 'minor' mode, 'pad_whisper' or 'dark_sustained'. Funk → 'walking' bass (iconic riff), 'electronic' rhythm. Cinematic → 'wide' stereo. Groove → 'medium' tempo. COHERENCE CHECK: 'Accent' brass_hit (synth stabs). 'Grain' clean but punchy. 'Motion' fast. REFERENCE ANCHORING: Michael Jackson. The graveyard dance section.",
    "tempo": "medium",
    "root": "c#",
    "mode": "minor",
    "brightness": "medium",
    "space": "medium",
    "density": 6,
    "bass": "walking",
    "pad": "dark_sustained",
    "melody": "minimal",
    "rhythm": "electronic",
    "texture": "pad_whisper",
    "accent": "brass_hit",
    "motion": "medium",
    "attack": "sharp",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "tight",
    "grain": "clean",
    "melody_engine": "pattern",
    "phrase_len_bars": 4,
    "melody_density": "sparse",
    "syncopation": "heavy",
    "swing": "light",
    "motif_repeat": "often",
    "step_vs_leap": "leapy",
    "chromatic": "medium",
    "cadence": "strong",
    "register": "low",
    "tension_curve": "ramp",
    "harmony": "cinematic",
    "chord_change": "medium",
    "chord_extensions": "sevenths"
  },
  "score": 93
}

```

**Example 30**
Input: "What a Wonderful World vibe - Gentle optimism, jazz warmth, nostalgia."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Gentle = slow tempo, soft attack, no harsh frequencies. Optimism = major key, sweet melody. Jazz warmth = acoustic textures, lush chords. Nostalgia = slow strings, brush drums. SONIC TRANSLATION: Gentle → 'slow' tempo, 'soft' attack. Optimism → 'major' mode, 'rising' melody. Warmth → 'warm' grain, 'warm_slow' pad. Nostalgia → 'vinyl_crackle', 'brush' rhythm. COHERENCE CHECK: 'Harmony' jazz (vital). 'Accent' piano_note or pluck. 'Human' natural. REFERENCE ANCHORING: Louis Armstrong. Orchestral pop-jazz.",
    "tempo": "slow",
    "root": "f",
    "mode": "major",
    "brightness": "medium",
    "space": "medium",
    "density": 4,
    "bass": "walking",
    "pad": "warm_slow",
    "melody": "rising",
    "rhythm": "brush",
    "texture": "vinyl_crackle",
    "accent": "piano_note",
    "motion": "slow",
    "attack": "soft",
    "stereo": "medium",
    "depth": false,
    "echo": "subtle",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "light",
    "swing": "medium",
    "motif_repeat": "sometimes",
    "step_vs_leap": "step",
    "chromatic": "light",
    "cadence": "strong",
    "register": "mid",
    "tension_curve": "arc",
    "harmony": "jazz",
    "chord_change": "slow",
    "chord_extensions": "lush"
  },
  "score": 94
}

```

**Example 31**
Input: "Super Mario vibe - Bouncy 8-bit, power-up sparkle, platformer energy."
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Bouncy = syncopated, staccato, fast. 8-bit = square wave basics, clean grain, dry. Power-up = rapid arpeggios, high pitch pitch-bends. Platformer energy = driving, never stopping. SONIC TRANSLATION: Bouncy → 'syncopation' heavy, 'kit_light' rhythm. 8-bit → 'clean' grain, 'arp_bass', 'arp_melody'. Sparkle → 'blip' accents. Energy → 'fast' tempo, 'major' mode. COHERENCE CHECK: 'Echo' none (dry chip sound). 'Attack' sharp. 'Harmony' pop (I-IV-V). REFERENCE ANCHORING: Koji Kondo. NES sound chip limitations.",
    "tempo": "fast",
    "root": "c",
    "mode": "major",
    "brightness": "bright",
    "space": "dry",
    "density": 5,
    "bass": "arp_bass",
    "pad": "none",
    "melody": "arp_melody",
    "rhythm": "kit_light",
    "texture": "none",
    "accent": "blip",
    "motion": "fast",
    "attack": "sharp",
    "stereo": "mono",
    "depth": false,
    "echo": "none",
    "human": "robotic",
    "grain": "clean",
    "melody_engine": "pattern",
    "phrase_len_bars": 2,
    "melody_density": "busy",
    "syncopation": "heavy",
    "swing": "light",
    "motif_repeat": "often",
    "step_vs_leap": "leapy",
    "chromatic": "none",
    "cadence": "strong",
    "register": "high",
    "tension_curve": "ramp",
    "harmony": "pop",
    "chord_change": "fast",
    "chord_extensions": "triads"
  },
  "score": 93
}
```

**Example 32**
Input: "Indian Classical vibe - raga meditation, sitar and tabla, devotional depth, microtonal ornamentation, spiritual journey through time"
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Raga meditation = drone base (Tanpura), slow development (Alap), modal strictness. Sitar/Tabla = resonant plucked strings, complex hand percussion, sympathetic vibration. Devotional depth = spiritual connection, patience, trance. Microtonal ornamentation = sliding pitches (meend), grace notes, fluid melody. SONIC TRANSLATION: Raga → 'drone' bass (essential), 'mixolydian' mode (common for evening ragas like Yaman). Sitar → 'pluck' accent, 'ornamental' melody (captures the grace notes). Tabla → 'tabla_essence' rhythm (specifically available). Devotional → 'slow' tempo, 'warm' grain. COHERENCE CHECK: 'Rhythm' must be 'tabla_essence'. 'Melody' must be 'ornamental' to mimic the glides. 'Space' medium/large for temple acoustics. REFERENCE ANCHORING: Ravi Shankar Alap/Jor. Meditative but rhythmic.",
    "tempo": "slow",
    "root": "c#",
    "mode": "mixolydian",
    "brightness": "medium",
    "space": "large",
    "density": 5,
    "bass": "drone",
    "pad": "warm_slow",
    "melody": "ornamental",
    "rhythm": "tabla_essence",
    "texture": "shimmer_slow",
    "accent": "pluck",
    "motion": "slow",
    "attack": "medium",
    "stereo": "medium",
    "depth": true,
    "echo": "medium",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "medium",
    "syncopation": "light",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "step",
    "chromatic": "light",
    "cadence": "weak",
    "register": "mid",
    "tension_curve": "waves",
    "harmony": "ambient",
    "chord_change": "very_slow",
    "chord_extensions": "triads"
  },
  "score": 96
}

```

**Example 33**
Input: "Arabian Nights vibe - desert mystery, snake charmer hypnosis, palace intrigue, 1001 stories, Scheherazade moonlight"
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Desert mystery = vast emptiness, heat haze, shifting sands. Snake charmer = winding melodies, hypnotic repetition, reed instruments. Palace intrigue = shadows, hidden corners, lush luxury. Scheherazade = feminine storytelling power, flowing, nocturnal. SONIC TRANSLATION: Desert → 'vast' space, 'shimmer' texture (heat). Snake charmer → 'ornamental' melody (mimics Mizmar/Ney), 'motif_repeat' often (hypnosis). Palace → 'lush' harmony, 'minor' mode (exoticism/mystery). Moonlight → 'cool/dark' brightness. COHERENCE CHECK: 'Bass' sustained (pedal point). 'Accent' wind. 'Rhythm' minimal to keep the mystery, or tabla_essence for groove - let's go 'minimal' for atmosphere. REFERENCE ANCHORING: Lawrence of Arabia or Aladdin ambient tracks. Cinematic orientalism.",
    "tempo": "slow",
    "root": "d",
    "mode": "minor",
    "brightness": "dark",
    "space": "vast",
    "density": 4,
    "bass": "sustained",
    "pad": "dark_sustained",
    "melody": "ornamental",
    "rhythm": "minimal",
    "texture": "shimmer_slow",
    "accent": "wind",
    "motion": "medium",
    "attack": "soft",
    "stereo": "wide",
    "depth": true,
    "echo": "heavy",
    "human": "loose",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "light",
    "swing": "light",
    "motif_repeat": "often",
    "step_vs_leap": "step",
    "chromatic": "heavy",
    "cadence": "weak",
    "register": "mid",
    "tension_curve": "arc",
    "harmony": "cinematic",
    "chord_change": "slow",
    "chord_extensions": "lush"
  },
  "score": 91
}

```

**Example 34**
Input: "Islamic Call to Prayer vibe - adhan devotion, minaret echo, submission to divine, desert dawn, sacred geometry in sound"
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Adhan = solo human voice, melismatic, high projection, unmetered. Minaret echo = slapback delay across a city, distance. Submission = overwhelming bass drone, grounding. Desert dawn = cool blue light turning to gold, clarity. Sacred geometry = mathematical perfection in melody (maqam), fractal repetition. SONIC TRANSLATION: Adhan → 'procedural' melody (step-wise, vocal range), 'ornamental' accent qualities. Echo → 'infinite' echo (crucial for minaret feel). Submission → 'fifth_drone' bass (powerful foundation). Dawn → 'breath' texture (morning air). COHERENCE CHECK: Rhythm MUST be 'none' - this is free time. Mode 'mixolydian' or 'dorian' closest to Maqam Rast/Bayati feel. 'Stereo' ultra_wide. REFERENCE ANCHORING: Morning Adhan recorded from a distance. Spiritual solitude.",
    "tempo": "very_slow",
    "root": "a#",
    "mode": "mixolydian",
    "brightness": "medium",
    "space": "vast",
    "density": 3,
    "bass": "fifth_drone",
    "pad": "thin_high",
    "melody": "procedural",
    "rhythm": "none",
    "texture": "breath",
    "accent": "none",
    "motion": "static",
    "attack": "soft",
    "stereo": "ultra_wide",
    "depth": true,
    "echo": "infinite",
    "human": "natural",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "sparse",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "rare",
    "step_vs_leap": "step",
    "chromatic": "light",
    "cadence": "weak",
    "register": "high",
    "tension_curve": "arc",
    "harmony": "ambient",
    "chord_change": "very_slow",
    "chord_extensions": "triads"
  },
  "score": 95
}

```

**Example 35**
Input: "Catholic Choir vibe - Gregorian chant, cathedral stone, sacred polyphony, incense and candlelight, eternal Latin"
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Gregorian chant = monophonic melody, modal scales, unmetered sacred song. Cathedral stone = massive reverberant space, cold hard surfaces, Gothic architecture. Sacred polyphony = multiple voices weaving, Renaissance harmony, divine mathematics. Incense and candlelight = ritual atmosphere, warm glow in cold stone, sensory worship. Eternal Latin = ancient language, timeless liturgy, connection across centuries. SONIC TRANSLATION: Gregorian/chant → dorian mode (medieval church modes), very slow tempo (unmetered feel), step-wise melody (conjunct chant motion). Cathedral = vast space (stone acoustics), infinite echo (5+ second reverb), ultra-wide stereo (sound filling nave). Polyphony = stacked fifths pad (parallel organum), lush extensions (Renaissance harmony), density 4 (multiple voice parts). Sacred = soft attack (no percussion in church), no rhythm (unmetered), drone bass (organ pedal point). Incense warmth = warm grain (candlelight glow), medium-dark brightness (filtered through stained glass). Eternal = often motif repeat (liturgical repetition), weak cadence (continuous devotion), arc tension (prayer arc). Cold stone + warm light = clean grain option but warm captures candlelight better. COHERENCE CHECK: Dorian + step + very slow = Gregorian chant character. Vast + infinite echo + ultra-wide = cathedral acoustics. Stacked fifths + lush + no rhythm = sacred polyphony. Non-negotiables: (1) vast space + infinite echo for cathedral, (2) dorian mode for medieval church sound, (3) no rhythm for liturgical timelessness. REFERENCE: This is stone made song - voices rising through incense smoke, bouncing off pillars carved centuries ago, the same prayers echoing through the same space for a thousand years.",
    "tempo": "very_slow",
    "root": "d",
    "mode": "dorian",
    "brightness": "dark",
    "space": "vast",
    "density": 4,
    "bass": "drone",
    "pad": "stacked_fifths",
    "melody": "contemplative",
    "rhythm": "none",
    "texture": "breath",
    "accent": "none",
    "motion": "static",
    "attack": "soft",
    "stereo": "ultra_wide",
    "depth": true,
    "echo": "infinite",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "very_sparse",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "step",
    "chromatic": "none",
    "cadence": "weak",
    "register": "low",
    "tension_curve": "arc",
    "harmony": "ambient",
    "chord_change": "very_slow",
    "chord_extensions": "lush"
  },
  "score": 92
}
```

**Example 36**
Input: "Domestic Violence (from victim perspective) - walking on eggshells, stifled breath, nowhere to run, heart in throat, fragile silence"
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Walking on eggshells = extreme tension, suppression of sound, fear of triggering. Stifled breath = erratic organic texture, suffocation. Nowhere to run = claustrophobia, small dry space. Heart in throat = internal physiological rhythm, overpowering the environment. Fragile silence = thin high frequencies, emptiness that feels dangerous. SONIC TRANSLATION: Eggshells → 'very_slow' tempo (time frozen), 'soft' attack (trying not to make noise). Stifled breath → 'breath' texture, 'human' natural. Claustrophobia → 'small' space, 'narrow' stereo. Heartbeat → 'heartbeat' rhythm (non-negotiable). Fragile → 'thin_high' pad (glassy anxiety), 'minimal' melody. COHERENCE CHECK: Must be 'very_dark' or 'dark' to represent hiding. 'Depth' true implies the visceral pit in the stomach. 'Bass' sub_pulse tracks the internal dread. REFERENCE ANCHORING: The sound of a panic attack in a closet. Isolation and terror.",
    "tempo": "very_slow",
    "root": "f#",
    "mode": "minor",
    "brightness": "very_dark",
    "space": "small",
    "density": 3,
    "bass": "sub_pulse",
    "pad": "thin_high",
    "melody": "minimal",
    "rhythm": "heartbeat",
    "texture": "breath",
    "accent": "none",
    "motion": "static",
    "attack": "soft",
    "stereo": "narrow",
    "depth": true,
    "echo": "none",
    "human": "natural",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 2,
    "melody_density": "very_sparse",
    "syncopation": "light",
    "swing": "none",
    "motif_repeat": "rare",
    "step_vs_leap": "step",
    "chromatic": "light",
    "cadence": "weak",
    "register": "high",
    "tension_curve": "ramp",
    "harmony": "cinematic",
    "chord_change": "very_slow",
    "chord_extensions": "triads"
  },
  "score": 95
}

```

**Example 37**
Input: "Domestic Violence (from aggressor perspective) - red mist, distorted reality, looming power, volatile control, suffocating presence"
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Red mist = blurred vision, anger, heat. Distorted reality = bit-crushed, gritty, unstable pitch. Looming power = massive low end, occupying all space. Volatile control = sudden loud accents, chaotic changes. Suffocating presence = heavy density, no silence. SONIC TRANSLATION: Red mist → 'gritty' grain (mental haze), 'distortion'. Looming → 'drone' bass (constant threat), 'dark_sustained' pad. Volatile → 'chaotic' motion, 'brass_hit' or 'blip_random' (sudden snaps). Suffocating → 'dry' space (in your face), 'density' 6. COHERENCE CHECK: 'Minor' mode is too clean; needs dissonance, but 'minor' is the best setting available—rely on 'chromatic' melody settings. 'Rhythm' should be 'none' or 'heartbeat'—let's go 'heartbeat' for the adrenaline of rage. REFERENCE ANCHORING: The Shining score or industrial noise. A headache turning into violence.",
    "tempo": "slow",
    "root": "a#",
    "mode": "minor",
    "brightness": "dark",
    "space": "dry",
    "density": 6,
    "bass": "drone",
    "pad": "dark_sustained",
    "melody": "falling",
    "rhythm": "heartbeat",
    "texture": "glitch",
    "accent": "blip_random",
    "motion": "chaotic",
    "attack": "sharp",
    "stereo": "mono",
    "depth": true,
    "echo": "none",
    "human": "drunk",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "heavy",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "leapy",
    "chromatic": "heavy",
    "cadence": "strong",
    "register": "low",
    "tension_curve": "ramp",
    "harmony": "cinematic",
    "chord_change": "slow",
    "chord_extensions": "triads"
  },
  "score": 92
}

```

**Example 38**
Input: "Domestic Violence (from police perspective) - flashing red and blue, radio static, tactical urgency, chaotic intervention, mechanical procedure"
Output:

```json
{
  "config": {
    "justification": "VIBE DECOMPOSITION: Flashing red/blue = strobing lights, disorienting visuals, alternating pitch. Radio static = high frequency noise, glitch, communication bursts. Tactical urgency = fast tempo, driving forward, adrenaline. Mechanical procedure = robotic timing, emotionless intervention. SONIC TRANSLATION: Flashing lights → 'pulsing' bass (siren-like), 'arp_melody' (strobing). Radio → 'glitch' texture, 'noise_wash'. Urgency → 'fast' tempo, 'electronic' rhythm (driving beat). Mechanical → 'robotic' humanization, 'clean' grain. COHERENCE CHECK: 'Brightness' very_bright (floodlights/sirens). 'Space' medium (street/hallway). The contrast between the emotional domestic scene and the cold, fast police response. REFERENCE ANCHORING: Action movie tension cue or Cops theme intro vibe. High alert.",
    "tempo": "fast",
    "root": "c",
    "mode": "minor",
    "brightness": "very_bright",
    "space": "medium",
    "density": 5,
    "bass": "pulsing",
    "pad": "thin_high",
    "melody": "arp_melody",
    "rhythm": "electronic",
    "texture": "glitch",
    "accent": "blip",
    "motion": "fast",
    "attack": "sharp",
    "stereo": "wide",
    "depth": false,
    "echo": "subtle",
    "human": "robotic",
    "grain": "clean",
    "melody_engine": "pattern",
    "phrase_len_bars": 2,
    "melody_density": "busy",
    "syncopation": "straight",
    "swing": "none",
    "motif_repeat": "often",
    "step_vs_leap": "step",
    "chromatic": "none",
    "cadence": "strong",
    "register": "mid",
    "tension_curve": "ramp",
    "harmony": "cinematic",
    "chord_change": "fast",
    "chord_extensions": "triads"
  },
  "score": 90
}

```
"""

#
# CLAUDE EXAMPLES OF MUSIC CONFIGS THAT MATCH A PARTICULAR VIBE
#
# FEW_SHOT_EXAMPLES = """
# **Example 1**
# Input: "Rainy day in Tokyo - neon lights reflecting on wet pavement, melancholy but beautiful"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Melancholy but beautiful = bittersweet duality. Wet pavement = reflective, shimmering surfaces. Neon = synthetic, slightly harsh color against organic rain. Tokyo = urban density, isolation in crowds, futuristic yet lonely. Night = dark but illuminated. Rain = steady, rhythmic, enveloping. SONIC TRANSLATION: Bittersweet → minor mode but with brightness (neon glow). Reflective surfaces → reverb + delay creating mirror-like echoes. Synthetic neon → gritty grain, electronic textures. Urban isolation → medium density (not empty, not overwhelming). Rain rhythm → soft steady pulse, not aggressive. Enveloping → wide stereo, immersive space. COHERENCE CHECK: Minor mode + medium brightness + gritty grain + shimmer texture all reinforce the neon-noir aesthetic. Large space + echo creates the wet reflective quality. Non-negotiables: (1) minor mode for melancholy, (2) shimmer texture for rain/neon reflections, (3) gritty grain for urban synthetic feel. REFERENCE: This needs the electronic intimacy of a lo-fi beat but with more spatial depth and cinematic weight.",
#     "tempo": "slow",
#     "root": "d#",
#     "mode": "minor",
#     "brightness": "medium",
#     "space": "large",
#     "density": 4,
#     "bass": "sustained",
#     "pad": "dark_sustained",
#     "melody": "contemplative",
#     "rhythm": "minimal",
#     "texture": "shimmer",
#     "accent": "pluck",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "wide",
#     "depth": true,
#     "echo": "heavy",
#     "human": "natural",
#     "grain": "gritty",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "sparse",
#     "syncopation": "light",
#     "swing": "none",
#     "motif_repeat": "sometimes",
#     "step_vs_leap": "step",
#     "chromatic": "light",
#     "cadence": "weak",
#     "register": "mid",
#     "tension_curve": "arc",
#     "harmony": "cinematic",
#     "chord_change": "slow",
#     "chord_extensions": "sevenths"
#   },
#   "score": 88
# }
# ```

# **Example 2**
# Input: "Ancient temple at sunrise - sacred, peaceful, timeless"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Ancient = old, weathered, deep history, pre-modern. Temple = sacred geometry, spiritual resonance, stone/wood acoustics. Sunrise = gradual awakening, golden light emerging, hope after darkness. Sacred = reverent, meditative, connected to something larger. Peaceful = calm, no tension, accepting. Timeless = unchanging, eternal, beyond trend. SONIC TRANSLATION: Ancient → dorian mode (medieval/mystical association), warm grain (organic, not digital). Temple acoustics → vast reverb (stone chambers), long decay. Sunrise = rising melodic motion, gradual brightening (start dark, end brighter). Sacred = drone bass (om-like foundation), soft attacks (no violence). Peaceful = very slow tempo, minimal rhythm (no urgency). Timeless = simple harmonies (triads), no modern production tricks. COHERENCE CHECK: Dorian + drone + vast space + soft attack all support meditative temple feeling. Rising melody + warm grain capture sunrise warmth. Non-negotiables: (1) vast reverb for temple acoustics, (2) drone bass for sacred foundation, (3) very slow tempo for timelessness. REFERENCE: Think Gregorian chant meets ambient meditation music - the space IS the instrument.",
#     "tempo": "very_slow",
#     "root": "d",
#     "mode": "dorian",
#     "brightness": "dark",
#     "space": "vast",
#     "density": 3,
#     "bass": "drone",
#     "pad": "ambient_drift",
#     "melody": "rising",
#     "rhythm": "none",
#     "texture": "breath",
#     "accent": "chime",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "ultra_wide",
#     "depth": true,
#     "echo": "infinite",
#     "human": "natural",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 8,
#     "melody_density": "very_sparse",
#     "syncopation": "straight",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "none",
#     "cadence": "weak",
#     "register": "low",
#     "tension_curve": "ramp",
#     "harmony": "ambient",
#     "chord_change": "very_slow",
#     "chord_extensions": "triads"
#   },
#   "score": 92
# }
# ```

# **Example 3**
# Input: "Cyberpunk chase scene - adrenaline, neon, danger, synthetic"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Chase = fast movement, urgency, escape/pursuit. Adrenaline = heart pounding, fight-or-flight, heightened senses. Neon = harsh synthetic colors, futuristic urban. Danger = tension, minor mode, dissonance. Synthetic = artificial, digital, machine-made. Cyberpunk = dystopian future, tech-noir, Blade Runner aesthetic. SONIC TRANSLATION: Chase/adrenaline → very fast tempo, sharp attacks, pulsing bass (heartbeat). Neon/synthetic → gritty grain, electronic rhythm, clean digital textures. Danger → minor mode, chaotic motion (unpredictable modulation). Cyberpunk → arp melody (classic synth trope), electronic drums, narrow stereo (focused intensity). COHERENCE CHECK: Fast tempo + sharp attack + pulsing bass + electronic rhythm = chase energy. Minor + gritty + chaotic motion = danger/dystopia. These reinforce each other perfectly. Non-negotiables: (1) very fast tempo for chase urgency, (2) sharp attack for adrenaline punch, (3) pulsing bass for heartbeat tension. REFERENCE: Think Vangelis meets industrial - that relentless synthetic pursuit energy.",
#     "tempo": "very_fast",
#     "root": "a",
#     "mode": "minor",
#     "brightness": "bright",
#     "space": "small",
#     "density": 6,
#     "bass": "pulsing",
#     "pad": "cinematic",
#     "melody": "arp_melody",
#     "rhythm": "electronic",
#     "texture": "glitch",
#     "accent": "blip",
#     "motion": "chaotic",
#     "attack": "sharp",
#     "stereo": "narrow",
#     "depth": true,
#     "echo": "subtle",
#     "human": "robotic",
#     "grain": "gritty",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 2,
#     "melody_density": "very_busy",
#     "syncopation": "heavy",
#     "swing": "none",
#     "motif_repeat": "rare",
#     "step_vs_leap": "leapy",
#     "chromatic": "medium",
#     "cadence": "weak",
#     "register": "wide",
#     "tension_curve": "waves",
#     "harmony": "cinematic",
#     "chord_change": "fast",
#     "chord_extensions": "triads"
#   },
#   "score": 91
# }
# ```

# **Example 4**
# Input: "Grandmother's kitchen - warm, nostalgic, Sunday morning, cookies baking"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Grandmother = old, wise, comforting, safe. Kitchen = domestic warmth, activity, nourishment. Warm = cozy, soft edges, golden tones. Nostalgic = memory, slightly hazy, bittersweet longing for past. Sunday morning = relaxed, unhurried, light streaming in. Cookies baking = sweet, anticipation, sensory comfort. SONIC TRANSLATION: Warm/cozy → warm grain, major mode, medium-dark brightness (golden, not harsh). Nostalgic/hazy → vinyl crackle texture (old recordings), soft attack, echo (memory reverb). Sunday relaxed → slow tempo, minimal rhythm. Domestic/safe → small intimate space (not vast cathedral). Sweet anticipation → rising melody hints, gentle motion. COHERENCE CHECK: Warm grain + vinyl crackle + soft attack + small space = intimate nostalgic memory. Major mode + slow tempo + minimal rhythm = peaceful Sunday. Non-negotiables: (1) vinyl crackle for nostalgia, (2) warm grain for coziness, (3) small space for domestic intimacy. REFERENCE: This is lo-fi hip-hop territory but slower and more acoustic - the warmth of memory, not the coolness of a beat tape.",
#     "tempo": "slow",
#     "root": "f",
#     "mode": "major",
#     "brightness": "dark",
#     "space": "small",
#     "density": 3,
#     "bass": "sustained",
#     "pad": "warm_slow",
#     "melody": "contemplative",
#     "rhythm": "none",
#     "texture": "vinyl_crackle",
#     "accent": "piano_note",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "narrow",
#     "depth": false,
#     "echo": "medium",
#     "human": "loose",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "sparse",
#     "syncopation": "straight",
#     "swing": "light",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "none",
#     "cadence": "medium",
#     "register": "mid",
#     "tension_curve": "arc",
#     "harmony": "pop",
#     "chord_change": "slow",
#     "chord_extensions": "triads"
#   },
#   "score": 89
# }
# ```

# **Example 5**
# Input: "Deep space exploration - vast emptiness, wonder, isolation, stars"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Deep space = infinite void, no atmosphere, absolute silence yet somehow presence. Vast emptiness = maximum space, minimal content, negative space as feature. Wonder = awe, discovery, the sublime. Isolation = alone, small against immensity. Stars = distant points of light, countless, ancient light. SONIC TRANSLATION: Vast emptiness → vast reverb, sparse density (2), very slow tempo (time stretches in space). Wonder/sublime → major mode but mysterious (mixolydian has that hopeful-mysterious quality), rising melodic hints. Isolation → wide stereo (emphasizes smallness in bigness), thin high pad (distant, cold). Stars → shimmer slow texture (twinkling), bells accent (crystalline points of light). Cold void → clean grain (digital, inhuman), bright but not warm. COHERENCE CHECK: Vast space + sparse density + very slow tempo = emptiness. Wide stereo + thin high pad + shimmer = isolation among stars. Mixolydian + rising melody + bells = wonder. Non-negotiables: (1) vast reverb for space itself, (2) density 2 for emptiness, (3) shimmer for stars. REFERENCE: Think 2001: A Space Odyssey - the terrifying beauty of infinite nothing.",
#     "tempo": "very_slow",
#     "root": "e",
#     "mode": "mixolydian",
#     "brightness": "bright",
#     "space": "vast",
#     "density": 2,
#     "bass": "sub_pulse",
#     "pad": "thin_high",
#     "melody": "minimal",
#     "rhythm": "none",
#     "texture": "shimmer_slow",
#     "accent": "bells",
#     "motion": "static",
#     "attack": "soft",
#     "stereo": "ultra_wide",
#     "depth": true,
#     "echo": "infinite",
#     "human": "robotic",
#     "grain": "clean",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 8,
#     "melody_density": "very_sparse",
#     "syncopation": "straight",
#     "swing": "none",
#     "motif_repeat": "rare",
#     "step_vs_leap": "step",
#     "chromatic": "none",
#     "cadence": "weak",
#     "register": "high",
#     "tension_curve": "ramp",
#     "harmony": "ambient",
#     "chord_change": "very_slow",
#     "chord_extensions": "lush"
#   },
#   "score": 93
# }
# ```

# **Example 6**
# Input: "1920s speakeasy - jazz, smoke, secrets, prohibition glamour"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: 1920s = vintage, pre-digital, analog warmth, Art Deco elegance. Speakeasy = hidden, intimate, underground, illicit thrill. Jazz = swing, improvisation, syncopation, sophistication. Smoke = hazy, diffused, mysterious atmosphere. Secrets = whispered, close, conspiratorial. Prohibition glamour = danger mixed with luxury, transgressive elegance. SONIC TRANSLATION: Jazz → dorian mode (jazz staple), swing feel, syncopation, walking bass. Vintage/analog → warm grain, vinyl crackle texture. Speakeasy intimate → small space (basement club), narrow stereo. Smoke/hazy → medium echo (diffused), soft attack (nothing harsh). Secrets → loose human timing (natural performance), ornamental melody (jazz flourishes). Sophistication → sevenths chord extensions (jazz harmony). COHERENCE CHECK: Dorian + walking bass + swing + sevenths = jazz foundation. Small space + warm grain + vinyl = vintage intimacy. Syncopation + ornamental + loose = live performance feel. Non-negotiables: (1) swing for jazz feel, (2) walking bass for that classic line, (3) warm grain + vinyl for period authenticity. REFERENCE: Imagine a trio in a smoky basement - piano, bass, brushes - captured on a wax cylinder.",
#     "tempo": "medium",
#     "root": "d",
#     "mode": "dorian",
#     "brightness": "dark",
#     "space": "small",
#     "density": 4,
#     "bass": "walking",
#     "pad": "warm_slow",
#     "melody": "ornamental",
#     "rhythm": "brush",
#     "texture": "vinyl_crackle",
#     "accent": "piano_note",
#     "motion": "medium",
#     "attack": "medium",
#     "stereo": "narrow",
#     "depth": false,
#     "echo": "subtle",
#     "human": "loose",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "medium",
#     "syncopation": "heavy",
#     "swing": "heavy",
#     "motif_repeat": "sometimes",
#     "step_vs_leap": "balanced",
#     "chromatic": "medium",
#     "cadence": "medium",
#     "register": "mid",
#     "tension_curve": "waves",
#     "harmony": "jazz",
#     "chord_change": "medium",
#     "chord_extensions": "sevenths"
#   },
#   "score": 90
# }
# ```

# **Example 7**
# Input: "Horror movie - creeping dread, something is wrong, building terror"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Horror = fear, darkness, the unknown. Creeping = slow approach, gradual, inexorable. Dread = anticipation of something bad, sustained tension. Something wrong = uncanny, dissonance, wrongness. Building terror = escalation, crescendo, no release. SONIC TRANSLATION: Creeping/slow → very slow tempo, static motion (frozen in fear). Dread/tension → minor mode, dark brightness (can't see what's coming), chromatic dissonance. Something wrong → gritty grain (corrupted), breath texture (presence in darkness), chaotic motion in melody. Building → ramp tension curve, busy melody density toward climax, cinematic pad for weight. No release → weak cadence (never resolves), echo creating ghosts. COHERENCE CHECK: Very slow + minor + very dark + vast space = horror atmosphere. Chromatic + gritty + breath texture = wrongness/presence. Ramp curve + weak cadence = building without release. Non-negotiables: (1) very dark brightness for blindness/fear, (2) chromatic for wrongness, (3) weak cadence to deny resolution. REFERENCE: Think The Shining score - that sense of wrongness that never lets go, always building toward something terrible.",
#     "tempo": "very_slow",
#     "root": "c#",
#     "mode": "minor",
#     "brightness": "very_dark",
#     "space": "vast",
#     "density": 4,
#     "bass": "drone",
#     "pad": "dark_sustained",
#     "melody": "falling",
#     "rhythm": "heartbeat",
#     "texture": "breath",
#     "accent": "none",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "wide",
#     "depth": true,
#     "echo": "heavy",
#     "human": "natural",
#     "grain": "gritty",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 8,
#     "melody_density": "sparse",
#     "syncopation": "straight",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "heavy",
#     "cadence": "weak",
#     "register": "low",
#     "tension_curve": "ramp",
#     "harmony": "cinematic",
#     "chord_change": "very_slow",
#     "chord_extensions": "triads"
#   },
#   "score": 91
# }
# ```

# **Example 8**
# Input: "Victory celebration - triumph, euphoria, we did it, champagne popping"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Victory = achievement, overcoming, success. Triumph = glory, fanfare, the peak moment. Euphoria = overwhelming positive emotion, elation, release. We did it = collective joy, shared accomplishment, together. Champagne = effervescence, bubbles, sparkle, luxury celebration. SONIC TRANSLATION: Triumph/victory → major mode (unambiguous happiness), heroic rising melody, bright brightness (golden). Euphoria → fast tempo (energy release), wide stereo (expansive feeling), dense layers (fullness). Collective → thick density (6), cinematic pad (orchestral swell). Champagne/sparkle → shimmer texture, bells accent (crystalline celebration), sharp attack (pop of cork). Glory → brass hit accents, depth for impact, strong cadence (definitive resolution). COHERENCE CHECK: Major + bright + fast + rising = pure triumph. Dense + wide + cinematic = epic scale. Sharp attack + bells + shimmer = champagne sparkle. Non-negotiables: (1) major mode for unambiguous joy, (2) heroic rising melody, (3) density 6 for fullness of celebration. REFERENCE: Think Olympic medal ceremony, movie ending triumph - that unambiguous YES moment.",
#     "tempo": "fast",
#     "root": "c",
#     "mode": "major",
#     "brightness": "very_bright",
#     "space": "large",
#     "density": 6,
#     "bass": "sustained",
#     "pad": "cinematic",
#     "melody": "heroic",
#     "rhythm": "soft_four",
#     "texture": "shimmer",
#     "accent": "brass_hit",
#     "motion": "fast",
#     "attack": "sharp",
#     "stereo": "wide",
#     "depth": true,
#     "echo": "medium",
#     "human": "tight",
#     "grain": "clean",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "busy",
#     "syncopation": "light",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "leapy",
#     "chromatic": "none",
#     "cadence": "strong",
#     "register": "wide",
#     "tension_curve": "arc",
#     "harmony": "pop",
#     "chord_change": "medium",
#     "chord_extensions": "triads"
#   },
#   "score": 88
# }
# ```

# **Example 9**
# Input: "Underwater coral reef - colorful, teeming with life, flowing, dreamlike"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Underwater = submerged, muffled, pressure, different physics. Coral reef = explosion of color, organic shapes, biodiversity. Teeming with life = activity everywhere, but graceful not chaotic. Flowing = currents, gentle movement, no sharp edges. Dreamlike = surreal, floaty, time distortion. SONIC TRANSLATION: Underwater = muffled brightness (dark-medium), heavy reverb (sound travels differently), soft attack (water resistance). Coral colors = mixolydian mode (bright but mysterious), lush chord extensions (rich harmonics). Teeming = medium-busy density, but flowing motion not chaotic. Flowing/currents = medium tempo (not fast), swing feel (gentle sway), procedural melody with waves tension curve. Dreamlike = wide stereo (immersive), heavy echo (time stretching), loose human feel. COHERENCE CHECK: Dark brightness + heavy reverb + soft attack = underwater physics. Mixolydian + lush extensions + shimmer = colorful reef. Waves tension + swing + medium motion = current flow. Non-negotiables: (1) soft attack for water resistance, (2) heavy reverb for underwater acoustics, (3) lush extensions for color/richness. REFERENCE: Think nature documentary score when camera descends into reef - that wonder of alien beauty.",
#     "tempo": "slow",
#     "root": "f",
#     "mode": "mixolydian",
#     "brightness": "medium",
#     "space": "large",
#     "density": 5,
#     "bass": "sustained",
#     "pad": "ambient_drift",
#     "melody": "procedural",
#     "rhythm": "none",
#     "texture": "shimmer_slow",
#     "accent": "chime",
#     "motion": "medium",
#     "attack": "soft",
#     "stereo": "wide",
#     "depth": true,
#     "echo": "heavy",
#     "human": "loose",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "medium",
#     "syncopation": "light",
#     "swing": "medium",
#     "motif_repeat": "sometimes",
#     "step_vs_leap": "step",
#     "chromatic": "light",
#     "cadence": "weak",
#     "register": "mid",
#     "tension_curve": "waves",
#     "harmony": "ambient",
#     "chord_change": "slow",
#     "chord_extensions": "lush"
#   },
#   "score": 89
# }
# ```

# **Example 10**
# Input: "Rage and destruction - anger, smashing things, uncontrolled fury"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Rage = intense negative emotion, loss of control, seeing red. Destruction = breaking, impact, violence. Smashing = physical force, percussion, chaos. Uncontrolled = wild, unpredictable, beyond reason. Fury = sustained rage, burning, relentless. SONIC TRANSLATION: Rage/fury → minor mode (negative), very fast tempo (heart racing), sharp attack (violent transients). Destruction/smashing → heavy rhythm (impacts), gritty grain (distortion), depth for physical weight. Uncontrolled → chaotic motion, heavy syncopation (unpredictable), drunk human feel (wild). Burning → bright brightness (harsh, searing), narrow stereo (focused tunnel vision). Relentless → dense layers (overwhelming), fast chord changes (no rest), weak cadence (no resolution). COHERENCE CHECK: Fast + sharp + heavy rhythm + gritty = violent impact. Minor + chaotic + drunk = loss of control. Dense + narrow + weak cadence = relentless assault. Non-negotiables: (1) very fast tempo for racing heart, (2) sharp attack for violence, (3) chaotic motion for loss of control. REFERENCE: Think industrial metal breakdown - that moment of pure sonic violence where everything is destruction.",
#     "tempo": "very_fast",
#     "root": "a",
#     "mode": "minor",
#     "brightness": "bright",
#     "space": "dry",
#     "density": 6,
#     "bass": "pulsing",
#     "pad": "dark_sustained",
#     "melody": "arp_melody",
#     "rhythm": "electronic",
#     "texture": "glitch",
#     "accent": "brass_hit",
#     "motion": "chaotic",
#     "attack": "sharp",
#     "stereo": "narrow",
#     "depth": true,
#     "echo": "none",
#     "human": "drunk",
#     "grain": "gritty",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 2,
#     "melody_density": "very_busy",
#     "syncopation": "heavy",
#     "swing": "none",
#     "motif_repeat": "rare",
#     "step_vs_leap": "leapy",
#     "chromatic": "heavy",
#     "cadence": "weak",
#     "register": "wide",
#     "tension_curve": "waves",
#     "harmony": "cinematic",
#     "chord_change": "fast",
#     "chord_extensions": "triads"
#   },
#   "score": 87
# }
# ```

# **Example 11**
# Input: "Cozy reading nook - fireplace crackling, rain outside, warm blanket, book"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Cozy = comfortable, safe, enclosed warmth. Reading nook = small intimate space, quietude, focused calm. Fireplace = warmth, gentle crackling, organic sound, flickering light. Rain outside = contrast (cold out, warm in), steady ambient sound, contemplative. Warm blanket = soft textures, comfort, protection. Book = intellectual calm, imagination, inner world. SONIC TRANSLATION: Cozy/enclosed → small space (intimate room), narrow stereo (focused). Fireplace/organic → warm grain, vinyl crackle (crackling analog), breath texture option. Rain = slow steady tempo, soft attack (gentle). Calm/contemplative → very sparse density, minimal melody, no rhythm. Warmth → dark brightness (golden, not harsh), major mode (contentment). Safety/comfort → medium echo (soft reflections), natural human feel. COHERENCE CHECK: Small space + warm grain + vinyl = analog intimate comfort. Slow + sparse + soft = calm quietude. Major + dark + natural = warm contentment. Non-negotiables: (1) small space for intimacy, (2) warm grain for fireplace analog feel, (3) vinyl crackle for that organic crackling. REFERENCE: Think hygge made sonic - that Danish art of cozy contentment, rain pattering while you're wrapped in warmth.",
#     "tempo": "very_slow",
#     "root": "g",
#     "mode": "major",
#     "brightness": "dark",
#     "space": "small",
#     "density": 2,
#     "bass": "drone",
#     "pad": "warm_slow",
#     "melody": "minimal",
#     "rhythm": "none",
#     "texture": "vinyl_crackle",
#     "accent": "none",
#     "motion": "static",
#     "attack": "soft",
#     "stereo": "narrow",
#     "depth": false,
#     "echo": "subtle",
#     "human": "natural",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 8,
#     "melody_density": "very_sparse",
#     "syncopation": "straight",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "none",
#     "cadence": "weak",
#     "register": "low",
#     "tension_curve": "arc",
#     "harmony": "ambient",
#     "chord_change": "very_slow",
#     "chord_extensions": "triads"
#   },
#   "score": 91
# }
# ```

# **Example 12**
# Input: "Carnival at night - manic energy, flashing lights, sugar rush, chaos"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Carnival = spectacle, sensory overload, entertainment, artifice. Night = darkness punctuated by artificial light, slightly sinister edge. Manic energy = hyperactive, unstoppable, almost unhinged. Flashing lights = rapid changes, bright bursts, disorienting. Sugar rush = sweet but overwhelming, jittery, artificial high. Chaos = disorder, unpredictable, everything happening at once. SONIC TRANSLATION: Manic/sugar rush → very fast tempo, busy melody density, chaotic motion. Flashing lights → bright brightness, sharp attack (bursts), fast chord changes. Carnival spectacle → major mode (festive), dense layers (6), arp melody (carousel feel). Chaos/overload → heavy syncopation, wide stereo (sounds everywhere), glitch texture. Night/sinister edge → clean grain (artificial), slight chromatic touches. COHERENCE CHECK: Fast + dense + busy + chaotic = sensory overload. Major + bright + arp = carnival festivity. Sharp + glitch + syncopation = disorienting flashes. Non-negotiables: (1) very fast for manic energy, (2) density 6 for overload, (3) chaotic motion for unpredictability. REFERENCE: Think Tim Burton carnival score - that manic festivity with an edge of madness.",
#     "tempo": "very_fast",
#     "root": "c",
#     "mode": "major",
#     "brightness": "very_bright",
#     "space": "medium",
#     "density": 6,
#     "bass": "pulsing",
#     "pad": "bright_open",
#     "melody": "arp_melody",
#     "rhythm": "electronic",
#     "texture": "glitch",
#     "accent": "bells_dense",
#     "motion": "chaotic",
#     "attack": "sharp",
#     "stereo": "wide",
#     "depth": false,
#     "echo": "subtle",
#     "human": "tight",
#     "grain": "clean",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 2,
#     "melody_density": "very_busy",
#     "syncopation": "heavy",
#     "swing": "light",
#     "motif_repeat": "rare",
#     "step_vs_leap": "leapy",
#     "chromatic": "light",
#     "cadence": "weak",
#     "register": "high",
#     "tension_curve": "waves",
#     "harmony": "pop",
#     "chord_change": "fast",
#     "chord_extensions": "triads"
#   },
#   "score": 86
# }
# ```

# **Example 13**
# Input: "Meditation retreat - mindfulness, breath, letting go, inner peace"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Meditation = focused awareness, stillness, presence. Retreat = withdrawal from noise, sanctuary, dedicated space. Mindfulness = attention to now, non-judgment, observation. Breath = fundamental rhythm, life force, slow and deep. Letting go = release, non-attachment, surrender. Inner peace = calm center, equanimity, no conflict. SONIC TRANSLATION: Stillness/presence → very slow tempo, static motion (no urgency). Breath = breath texture (literal), soft attack (gentle as exhalation), natural human feel. Sanctuary = vast space (meditation hall acoustics), drone bass (om foundation). Letting go = weak cadence (no grasping for resolution), infinite echo (dissolving). Peace = major mode (but not excited - mixolydian for gentle), very dark brightness (eyes closed). Mindfulness = very sparse density (attention to each sound), step-wise melody (no leaps). COHERENCE CHECK: Very slow + static + sparse = stillness. Drone + vast + infinite echo = sanctuary space. Breath + soft + weak cadence = letting go. Non-negotiables: (1) very slow tempo for meditation pace, (2) breath texture, (3) vast space for dissolution of self. REFERENCE: Think singing bowl recordings - that perfect nothing where presence itself is the content.",
#     "tempo": "very_slow",
#     "root": "f",
#     "mode": "mixolydian",
#     "brightness": "very_dark",
#     "space": "vast",
#     "density": 2,
#     "bass": "drone",
#     "pad": "ambient_drift",
#     "melody": "minimal",
#     "rhythm": "none",
#     "texture": "breath",
#     "accent": "chime",
#     "motion": "static",
#     "attack": "soft",
#     "stereo": "wide",
#     "depth": false,
#     "echo": "infinite",
#     "human": "natural",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 8,
#     "melody_density": "very_sparse",
#     "syncopation": "straight",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "none",
#     "cadence": "weak",
#     "register": "low",
#     "tension_curve": "arc",
#     "harmony": "ambient",
#     "chord_change": "very_slow",
#     "chord_extensions": "triads"
#   },
#   "score": 93
# }
# ```

# **Example 14**
# Input: "First day of summer - freedom, sunshine, endless possibility, youth"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: First day = beginning, threshold, anticipation. Summer = warmth, light, outdoor life, growth. Freedom = liberation, no constraints, open horizons. Sunshine = bright, golden, life-giving, joyful. Endless possibility = optimism, future-oriented, expansive. Youth = energy, innocence, vitality, carefree. SONIC TRANSLATION: Freedom/endless → wide stereo (open horizons), large space (outdoor expanse), rising melody (looking forward). Sunshine/bright → very bright brightness, major mode (unambiguous joy), clean grain (clarity). Youth/energy → medium-fast tempo (vital but not frantic), medium density (active but not overwhelming). Optimism → strong cadence (confident resolution), arc tension curve (satisfying narrative). Carefree → natural human feel, light syncopation (playful), shimmer texture (sparkle). COHERENCE CHECK: Major + very bright + rising = pure optimism. Wide + large + rising = expansive freedom. Medium-fast + shimmer + light syncopation = youthful energy. Non-negotiables: (1) major mode for unambiguous joy, (2) very bright for sunshine, (3) wide stereo for freedom/openness. REFERENCE: Think coming-of-age movie opening - that pure potential of summer stretching endlessly ahead.",
#     "tempo": "medium",
#     "root": "g",
#     "mode": "major",
#     "brightness": "very_bright",
#     "space": "large",
#     "density": 4,
#     "bass": "sustained",
#     "pad": "bright_open",
#     "melody": "rising",
#     "rhythm": "soft_four",
#     "texture": "shimmer",
#     "accent": "bells",
#     "motion": "medium",
#     "attack": "medium",
#     "stereo": "wide",
#     "depth": false,
#     "echo": "medium",
#     "human": "natural",
#     "grain": "clean",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "medium",
#     "syncopation": "light",
#     "swing": "light",
#     "motif_repeat": "sometimes",
#     "step_vs_leap": "balanced",
#     "chromatic": "none",
#     "cadence": "strong",
#     "register": "mid",
#     "tension_curve": "arc",
#     "harmony": "pop",
#     "chord_change": "medium",
#     "chord_extensions": "triads"
#   },
#   "score": 90
# }
# ```

# **Example 15**
# Input: "Abandoned asylum - decay, echoing footsteps, history of pain, ghosts"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Abandoned = emptiness, neglect, time stopped, forgotten. Asylum = institutional, suffering, madness, confinement. Decay = deterioration, entropy, things falling apart. Echoing footsteps = vast empty halls, reverb, presence in absence. History of pain = accumulated suffering, heavy atmosphere, trauma. Ghosts = presence of the past, whispers, not quite there. SONIC TRANSLATION: Abandoned/empty → vast space (empty halls), sparse density (nothing left), very slow tempo (time frozen). Decay → gritty grain (deteriorating), dark brightness (no light). Echoing = infinite echo, heavy reverb, pluck accent (footsteps). Pain/suffering → minor mode, chromatic dissonance, breath texture (whispers). Ghosts → soft attack (not quite solid), loose human feel (unstable), shimmer slow (flickering). Institutional = cold clean elements mixed with decay. COHERENCE CHECK: Vast + sparse + infinite echo = empty echoing halls. Minor + gritty + dark = suffering/decay. Breath + soft + chromatic = ghostly presence. Non-negotiables: (1) vast space for empty halls, (2) infinite echo for reverberation, (3) minor + dark for institutional horror. REFERENCE: Think horror game soundtrack for exploring ruins - that sense of terrible history pressing in.",
#     "tempo": "very_slow",
#     "root": "c#",
#     "mode": "minor",
#     "brightness": "very_dark",
#     "space": "vast",
#     "density": 3,
#     "bass": "drone",
#     "pad": "dark_sustained",
#     "melody": "falling",
#     "rhythm": "none",
#     "texture": "breath",
#     "accent": "pluck",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "ultra_wide",
#     "depth": true,
#     "echo": "infinite",
#     "human": "loose",
#     "grain": "gritty",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 8,
#     "melody_density": "very_sparse",
#     "syncopation": "straight",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "medium",
#     "cadence": "weak",
#     "register": "low",
#     "tension_curve": "ramp",
#     "harmony": "cinematic",
#     "chord_change": "very_slow",
#     "chord_extensions": "sevenths"
#   },
#   "score": 89
# }
# ```

# **Example 16**
# Input: "Retro arcade - 8-bit bleeps, competition, neon glow, high scores"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Retro = past (80s), nostalgia, lo-fi technology. Arcade = gaming, competition, social space, sounds everywhere. 8-bit = limited synthesis, square waves, chip music, digital artifacts. Bleeps = short, pitched, electronic tones. Competition = energy, tension, drive to win. Neon glow = bright artificial colors, synthetic. High scores = achievement, reward sounds, fanfare moments. SONIC TRANSLATION: 8-bit/retro → gritty grain (bit-crushed), clean attack aesthetic, blip accents. Arcade energy → fast tempo, busy melody (action), electronic rhythm. Competition → minor mode edge (tension of winning/losing), medium syncopation. Neon/artificial → bright brightness, narrow stereo (mono-ish like real arcades), dry space (no reverb in chip music). High scores = arp melody (classic video game), bells for reward sounds. Digital → robotic human (machine precision), clean grain base with gritty edge. COHERENCE CHECK: Fast + arp + electronic + gritty = arcade game energy. Dry + narrow + robotic = authentic chip music feel. Blip + bells + bright = that distinctive bleep-bloop sound. Non-negotiables: (1) dry space for chip music authenticity, (2) gritty grain for 8-bit, (3) blip accents for game sounds. REFERENCE: Think Pac-Man meets Tetris - that competitive arcade energy captured in sound.",
#     "tempo": "fast",
#     "root": "c",
#     "mode": "minor",
#     "brightness": "bright",
#     "space": "dry",
#     "density": 4,
#     "bass": "pulsing",
#     "pad": "thin_high",
#     "melody": "arp_melody",
#     "rhythm": "electronic",
#     "texture": "glitch",
#     "accent": "blip",
#     "motion": "fast",
#     "attack": "sharp",
#     "stereo": "narrow",
#     "depth": false,
#     "echo": "none",
#     "human": "robotic",
#     "grain": "gritty",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 2,
#     "melody_density": "busy",
#     "syncopation": "medium",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "leapy",
#     "chromatic": "none",
#     "cadence": "medium",
#     "register": "high",
#     "tension_curve": "waves",
#     "harmony": "pop",
#     "chord_change": "fast",
#     "chord_extensions": "triads"
#   },
#   "score": 88
# }
# ```

# **Example 17**
# Input: "Romantic candlelit dinner - intimate, warm, love, slow dance"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Romantic = love, tenderness, emotional connection. Candlelit = warm glow, soft light, intimate illumination. Dinner = sophistication, care, special occasion. Intimate = close, private, just two people. Warm = emotional and physical comfort, golden tones. Love = deep positive emotion, vulnerability, openness. Slow dance = gentle movement together, 3/4 or slow 4/4, swaying. SONIC TRANSLATION: Romantic/love → major mode (positive), lush chord extensions (rich emotion), medium cadence (gentle resolution). Candlelit/warm → warm grain (analog softness), dark brightness (golden not harsh). Intimate → small space (just the two of you), narrow stereo (close together). Slow dance → slow tempo, swing feel (gentle sway), sustained bass (supportive). Sophistication → jazz harmony (evening elegance), ornamental melody (expressive), piano note accents. Tenderness → soft attack (no harshness), natural human feel (genuine). COHERENCE CHECK: Major + lush + warm = romantic warmth. Small + narrow + soft = intimate closeness. Slow + swing + sustained = slow dance feel. Non-negotiables: (1) warm grain for candlelight warmth, (2) small intimate space, (3) lush extensions for emotional richness. REFERENCE: Think jazz ballad in a quiet restaurant - Chet Baker playing to a room of two.",
#     "tempo": "slow",
#     "root": "e",
#     "mode": "major",
#     "brightness": "dark",
#     "space": "small",
#     "density": 4,
#     "bass": "sustained",
#     "pad": "warm_slow",
#     "melody": "ornamental",
#     "rhythm": "minimal",
#     "texture": "none",
#     "accent": "piano_note",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "narrow",
#     "depth": false,
#     "echo": "subtle",
#     "human": "natural",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "sparse",
#     "syncopation": "light",
#     "swing": "medium",
#     "motif_repeat": "sometimes",
#     "step_vs_leap": "step",
#     "chromatic": "light",
#     "cadence": "medium",
#     "register": "mid",
#     "tension_curve": "arc",
#     "harmony": "jazz",
#     "chord_change": "slow",
#     "chord_extensions": "lush"
#   },
#   "score": 91
# }
# ```

# **Example 18**
# Input: "Industrial factory - machines, repetition, metal, relentless production"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Industrial = mechanical, manufactured, heavy duty. Factory = production line, systematic, efficiency. Machines = non-human, precise, rhythmic, metallic. Repetition = loops, patterns, monotony. Metal = hard, cold, resonant, harsh. Relentless = never stopping, inexorable, fatigue-inducing. Production = output-focused, purpose-driven, functional. SONIC TRANSLATION: Machines/repetition → robotic human (perfect timing), electronic rhythm (mechanical), often motif repeat (loops). Relentless → fast tempo (constant motion), pulsing bass (engine rhythm), tight human feel. Metal/harsh → gritty grain (distortion), bright brightness (harsh frequencies), sharp attack (impacts). Industrial weight → depth sub-bass (physical weight), narrow stereo (focused). Factory monotony → pattern melody engine, straight syncopation (no swing), simple triads. Cold/mechanical → clean space (dry, no romance), medium space (factory hall). COHERENCE CHECK: Fast + pulsing + electronic + robotic = machine rhythm. Gritty + bright + sharp = metallic harshness. Repetition + straight + often repeat = mechanical loops. Non-negotiables: (1) robotic timing for machine precision, (2) pulsing bass for engine rhythm, (3) gritty grain for industrial harshness. REFERENCE: Think Nine Inch Nails or Kraftwerk - that man-machine interface where humanity meets mechanical repetition.",
#     "tempo": "fast",
#     "root": "a",
#     "mode": "minor",
#     "brightness": "bright",
#     "space": "medium",
#     "density": 5,
#     "bass": "pulsing",
#     "pad": "dark_sustained",
#     "melody": "arp_melody",
#     "rhythm": "electronic",
#     "texture": "glitch",
#     "accent": "blip",
#     "motion": "medium",
#     "attack": "sharp",
#     "stereo": "narrow",
#     "depth": true,
#     "echo": "none",
#     "human": "robotic",
#     "grain": "gritty",
#     "melody_engine": "pattern",
#     "phrase_len_bars": 2,
#     "melody_density": "busy",
#     "syncopation": "straight",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "none",
#     "cadence": "weak",
#     "register": "low",
#     "tension_curve": "waves",
#     "harmony": "cinematic",
#     "chord_change": "fast",
#     "chord_extensions": "triads"
#   },
#   "score": 87
# }
# ```

# **Example 19**
# Input: "Enchanted forest - magic, fairy lights, ancient trees, mystical creatures"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Enchanted = magical, otherworldly, transformed by wonder. Forest = organic, natural, canopy, depth, living. Magic = unexplainable, sparkle, transformation. Fairy lights = tiny points of light, twinkling, ethereal glow. Ancient trees = old growth, wisdom, presence, towering. Mystical creatures = hidden beings, watching, fantastical. SONIC TRANSLATION: Magic/enchanted → dorian mode (mystical), shimmer texture (sparkle), chime accents (fairy sounds). Forest/organic → warm grain (natural), breath texture option, natural human feel. Fairy lights → stars texture (twinkling), very bright brightness, bells (crystalline). Ancient/wisdom → slow tempo (timelessness), vast space (cathedral of trees), lush extensions (rich harmonics). Mystical = wide stereo (things in periphery), medium echo (mysterious reverb). Hidden/watching → sparse melody (glimpses), soft attack (gentle appearances). COHERENCE CHECK: Dorian + lush + warm = mystical organic beauty. Shimmer + bells + bright = fairy light sparkle. Vast + slow + wide = ancient forest scale. Non-negotiables: (1) dorian for mystical quality, (2) shimmer + bells for fairy lights, (3) warm grain for organic forest feel. REFERENCE: Think Studio Ghibli forest scenes - that sense of magic just beyond perception, nature as temple.",
#     "tempo": "slow",
#     "root": "d",
#     "mode": "dorian",
#     "brightness": "medium",
#     "space": "large",
#     "density": 4,
#     "bass": "fifth_drone",
#     "pad": "ambient_drift",
#     "melody": "ornamental",
#     "rhythm": "none",
#     "texture": "stars",
#     "accent": "chime",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "wide",
#     "depth": false,
#     "echo": "medium",
#     "human": "natural",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "sparse",
#     "syncopation": "light",
#     "swing": "none",
#     "motif_repeat": "sometimes",
#     "step_vs_leap": "step",
#     "chromatic": "light",
#     "cadence": "weak",
#     "register": "mid",
#     "tension_curve": "arc",
#     "harmony": "ambient",
#     "chord_change": "slow",
#     "chord_extensions": "lush"
#   },
#   "score": 92
# }
# ```

# **Example 20**
# Input: "Heartbreak hotel - loneliness, loss, 3am sadness, empty bottle"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Heartbreak = emotional pain, love lost, aching. Hotel = transient, anonymous, nowhere place, isolation. Loneliness = alone with pain, no comfort, separation. Loss = something precious gone, grief, absence. 3am sadness = darkest hour, sleepless, thoughts spiraling. Empty bottle = drinking away pain, numbness attempted, failure. SONIC TRANSLATION: Heartbreak/loss → minor mode (sadness), falling melody (descending into grief), weak cadence (no resolution). Loneliness → small space (isolated room), narrow stereo (alone), sparse density. 3am = very slow tempo (time crawling), dark brightness (no light). Empty/numbness → drunk human feel (impaired), warm grain (alcohol haze), heavy echo (thoughts reverberating). Hotel anonymity → medium space (impersonal room), vinyl crackle (old sad song on repeat). Aching = contemplative melody, breath texture (sighs), lush extensions (rich sadness). COHERENCE CHECK: Minor + falling + weak = descending grief. Small + narrow + sparse = lonely isolation. Slow + dark + drunk = 3am deterioration. Non-negotiables: (1) falling melody for descending grief, (2) drunk human feel for impairment, (3) minor mode for unambiguous sadness. REFERENCE: Think Tom Waits at his most desolate - that beautiful wreckage of heartbreak in a cheap room.",
#     "tempo": "very_slow",
#     "root": "e",
#     "mode": "minor",
#     "brightness": "very_dark",
#     "space": "small",
#     "density": 3,
#     "bass": "sustained",
#     "pad": "dark_sustained",
#     "melody": "falling",
#     "rhythm": "none",
#     "texture": "vinyl_crackle",
#     "accent": "piano_note",
#     "motion": "static",
#     "attack": "soft",
#     "stereo": "narrow",
#     "depth": false,
#     "echo": "heavy",
#     "human": "drunk",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "sparse",
#     "syncopation": "straight",
#     "swing": "light",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "light",
#     "cadence": "weak",
#     "register": "low",
#     "tension_curve": "ramp",
#     "harmony": "jazz",
#     "chord_change": "slow",
#     "chord_extensions": "sevenths"
#   },
#   "score": 90
# }
# ```

# **Example 21**
# Input: "Bohemian Rhapsody vibe - operatic drama, genre-shifting, theatrical, emotional crescendos, epic storytelling"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Operatic = theatrical grandeur, vocal-like dynamics, classical influence. Drama = tension and release, emotional peaks and valleys. Genre-shifting = unpredictable changes, sections that transform. Theatrical = performative, staged, larger than life. Emotional crescendos = building intensity, climactic moments. Epic storytelling = narrative arc, journey through emotions. SONIC TRANSLATION: Operatic/theatrical → cinematic pad (orchestral weight), wide stereo (stage width), lush chord extensions (rich harmony). Drama/crescendos → waves tension curve (multiple builds), strong cadence at peaks, density 6 (full production). Genre-shifting → leapy melody motion (unpredictable), medium-fast chord changes, chromatic touches. Emotional journey → minor mode base (drama), dorian could work but minor captures the tragedy. Epic scale → large space (concert hall), depth for weight, rising then falling melody patterns. COHERENCE CHECK: Cinematic + lush + wide + dense = operatic grandeur. Waves + chromatic + leapy = genre-shifting drama. Strong cadence + large space = theatrical resolution. Non-negotiables: (1) cinematic pad for operatic weight, (2) waves tension for multiple emotional peaks, (3) density 6 for full theatrical production. REFERENCE: This is maximalist rock opera - every element turned up, nothing subtle, pure theatrical commitment.",
#     "tempo": "medium",
#     "root": "b",
#     "mode": "minor",
#     "brightness": "bright",
#     "space": "large",
#     "density": 6,
#     "bass": "sustained",
#     "pad": "cinematic",
#     "melody": "heroic",
#     "rhythm": "soft_four",
#     "texture": "shimmer",
#     "accent": "brass_hit",
#     "motion": "fast",
#     "attack": "medium",
#     "stereo": "wide",
#     "depth": true,
#     "echo": "medium",
#     "human": "natural",
#     "grain": "clean",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "busy",
#     "syncopation": "medium",
#     "swing": "none",
#     "motif_repeat": "sometimes",
#     "step_vs_leap": "leapy",
#     "chromatic": "medium",
#     "cadence": "strong",
#     "register": "wide",
#     "tension_curve": "waves",
#     "harmony": "cinematic",
#     "chord_change": "medium",
#     "chord_extensions": "lush"
#   },
#   "score": 89
# }
# ```

# **Example 22**
# Input: "Imagine vibe - hopeful simplicity, piano-driven, utopian dream, gentle conviction, peaceful revolution"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Hopeful = optimistic but not naive, belief in possibility. Simplicity = stripped down, essential, uncluttered. Piano-driven = melodic clarity, harmonic foundation, intimate instrument. Utopian dream = idealistic, aspirational, visionary. Gentle conviction = soft but unwavering, quiet strength. Peaceful revolution = change through calm, non-aggressive transformation. SONIC TRANSLATION: Simplicity → sparse density (2-3), minimal texture, clean grain (clarity). Piano-driven → piano note accents, sustained bass (harmonic support), contemplative melody. Hopeful/utopian → major mode (positive vision), rising melodic tendency, medium brightness (warm optimism). Gentle/peaceful → very slow tempo (unhurried), soft attack (no violence), natural human feel. Conviction = medium cadence (resolution without force), step-wise melody (sure-footed). Intimate → small-medium space (personal address), narrow stereo (focused message). COHERENCE CHECK: Sparse + clean + soft = simplicity. Major + rising + step = hopeful conviction. Slow + natural + small space = gentle intimacy. Non-negotiables: (1) sparse density for simplicity, (2) major mode for hope, (3) soft attack for peaceful nature. REFERENCE: This is the sound of a single voice with piano, addressing humanity directly - intimate yet universal.",
#     "tempo": "slow",
#     "root": "c",
#     "mode": "major",
#     "brightness": "medium",
#     "space": "small",
#     "density": 2,
#     "bass": "sustained",
#     "pad": "warm_slow",
#     "melody": "contemplative",
#     "rhythm": "none",
#     "texture": "none",
#     "accent": "piano_note",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "narrow",
#     "depth": false,
#     "echo": "subtle",
#     "human": "natural",
#     "grain": "clean",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "sparse",
#     "syncopation": "straight",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "none",
#     "cadence": "medium",
#     "register": "mid",
#     "tension_curve": "arc",
#     "harmony": "pop",
#     "chord_change": "slow",
#     "chord_extensions": "triads"
#   },
#   "score": 92
# }
# ```

# **Example 23**
# Input: "Billie Jean vibe - paranoid funk groove, iconic bassline, midnight urban, synth-pop tension, hypnotic pulse"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Paranoid = anxious, looking over shoulder, something's wrong. Funk groove = rhythmic lock, body movement, bass-driven. Iconic bassline = the foundation IS the song, pulsing, instantly recognizable. Midnight urban = city at night, streetlights, shadows. Synth-pop = electronic but melodic, 80s production, clean synths. Tension = unresolved, taut, sustained anxiety. Hypnotic pulse = trance-inducing repetition, locked-in feel. SONIC TRANSLATION: Paranoid/tension → minor mode, medium-heavy syncopation, chromatic touches for unease. Funk groove/bassline → walking bass (melodic movement), pulsing rhythm alternative, electronic drums. Hypnotic = often motif repeat (locked pattern), medium tempo (groove pocket), tight human feel. Midnight urban = medium-dark brightness, medium space (street reverb), gritty grain edge. Synth-pop = clean base with gritty texture, electronic rhythm, arp melody elements. 80s production = medium stereo width, subtle echo, sharp attack (punchy). COHERENCE CHECK: Minor + syncopation + chromatic = paranoid tension. Walking bass + electronic + tight = funk groove. Often repeat + medium tempo = hypnotic pulse. Non-negotiables: (1) walking bass for that iconic line feel, (2) minor mode for paranoia, (3) electronic rhythm for synth-pop groove. REFERENCE: This is the template for 80s pop-funk - that locked groove where tension and danceability coexist perfectly.",
#     "tempo": "medium",
#     "root": "f#",
#     "mode": "minor",
#     "brightness": "medium",
#     "space": "medium",
#     "density": 5,
#     "bass": "walking",
#     "pad": "thin_high",
#     "melody": "procedural",
#     "rhythm": "electronic",
#     "texture": "none",
#     "accent": "blip",
#     "motion": "medium",
#     "attack": "sharp",
#     "stereo": "medium",
#     "depth": true,
#     "echo": "subtle",
#     "human": "tight",
#     "grain": "clean",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "medium",
#     "syncopation": "heavy",
#     "swing": "light",
#     "motif_repeat": "often",
#     "step_vs_leap": "balanced",
#     "chromatic": "light",
#     "cadence": "weak",
#     "register": "mid",
#     "tension_curve": "waves",
#     "harmony": "pop",
#     "chord_change": "slow",
#     "chord_extensions": "sevenths"
#   },
#   "score": 90
# }
# ```

# **Example 24**
# Input: "Smells Like Teen Spirit vibe - grunge explosion, quiet-loud dynamics, teenage angst, raw power, generational anthem"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Grunge = dirty, heavy, distorted, Seattle sound. Explosion = sudden impact, release of pressure, violent energy. Quiet-loud dynamics = extreme contrast, whisper to scream. Teenage angst = frustration, alienation, unformed rage. Raw power = unpolished, aggressive, visceral. Generational anthem = collective voice, rebellion, identity formation. SONIC TRANSLATION: Grunge/raw → gritty grain (maximum distortion), dark-medium brightness (murky), depth for weight. Explosion/loud → sharp attack, fast tempo sections, dense layers (6). Quiet-loud = waves tension curve (builds and releases), wide dynamic range implied by parameter extremes. Angst/frustration → minor mode, heavy syncopation (fighting the beat), chromatic (dissonance). Power/anthem → pulsing bass (driving force), electronic rhythm (can approximate pounding drums). Rebellion = chaotic motion, leapy melody (unpredictable), drunk human feel (sloppy on purpose). COHERENCE CHECK: Gritty + dark + depth = grunge weight. Sharp + fast + dense = explosive power. Waves + chaotic + drunk = dynamic chaos. Non-negotiables: (1) gritty grain for grunge distortion, (2) waves tension for quiet-loud, (3) chaotic motion for angst energy. REFERENCE: This is controlled chaos - the sound of a generation saying 'whatever' while screaming it at maximum volume.",
#     "tempo": "fast",
#     "root": "f",
#     "mode": "minor",
#     "brightness": "medium",
#     "space": "medium",
#     "density": 6,
#     "bass": "pulsing",
#     "pad": "dark_sustained",
#     "melody": "arp_melody",
#     "rhythm": "electronic",
#     "texture": "glitch",
#     "accent": "brass_hit",
#     "motion": "chaotic",
#     "attack": "sharp",
#     "stereo": "wide",
#     "depth": true,
#     "echo": "subtle",
#     "human": "drunk",
#     "grain": "gritty",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "busy",
#     "syncopation": "heavy",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "leapy",
#     "chromatic": "medium",
#     "cadence": "weak",
#     "register": "mid",
#     "tension_curve": "waves",
#     "harmony": "cinematic",
#     "chord_change": "medium",
#     "chord_extensions": "triads"
#   },
#   "score": 87
# }
# ```

# **Example 25**
# Input: "Hotel California vibe - mysterious Americana, dark paradise, hypnotic guitar arpeggios, haunted luxury, inescapable"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Mysterious = hidden meaning, enigmatic, questions without answers. Americana = desert, highways, west coast, mythic America. Dark paradise = beauty with menace, luxury hiding darkness. Hypnotic guitar arpeggios = fingerpicked patterns, circular, trance-inducing. Haunted luxury = opulence that's somehow wrong, beautiful decay. Inescapable = trapped, no exit, caught in a loop. SONIC TRANSLATION: Mysterious/haunted → dorian mode (ambiguous), medium echo (ghostly reflections), warm grain (vintage). Arpeggios = arp melody (obviously), often motif repeat (circular patterns), step-wise motion. Dark paradise = medium brightness (not too dark, still 'paradise'), medium space (hotel interior). Americana/desert = warm grain, large space option (desert expanse), pluck accents (guitar). Inescapable/hypnotic = slow tempo (languorous), waves tension (cycles), weak cadence (never truly resolves). Luxury = lush extensions (rich harmony), cinematic pad (grandeur), medium stereo. COHERENCE CHECK: Dorian + arp + weak cadence = mysterious circular trap. Warm + pluck + large = Americana guitar feel. Echo + waves + often repeat = hypnotic loop. Non-negotiables: (1) arp melody for those iconic arpeggios, (2) dorian for mysterious ambiguity, (3) weak cadence for 'you can never leave' feel. REFERENCE: This is sonic purgatory dressed as paradise - every element beautiful but somehow wrong, endlessly cycling.",
#     "tempo": "slow",
#     "root": "b",
#     "mode": "dorian",
#     "brightness": "medium",
#     "space": "large",
#     "density": 4,
#     "bass": "sustained",
#     "pad": "warm_slow",
#     "melody": "arp_melody",
#     "rhythm": "minimal",
#     "texture": "shimmer_slow",
#     "accent": "pluck",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "medium",
#     "depth": false,
#     "echo": "medium",
#     "human": "natural",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "medium",
#     "syncopation": "light",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "light",
#     "cadence": "weak",
#     "register": "mid",
#     "tension_curve": "waves",
#     "harmony": "cinematic",
#     "chord_change": "slow",
#     "chord_extensions": "sevenths"
#   },
#   "score": 91
# }
# ```

# **Example 26**
# Input: "Stairway to Heaven vibe - spiritual ascent, folk to rock crescendo, mystical journey, acoustic to electric, transcendence"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Spiritual ascent = rising toward something higher, elevation, enlightenment. Folk to rock crescendo = builds from gentle acoustic to powerful electric. Mystical journey = quest, transformation, magic. Acoustic to electric = organic to powerful, intimate to epic. Transcendence = beyond ordinary, breakthrough, apotheosis. SONIC TRANSLATION: Spiritual/mystical → dorian mode (folk mysticism), lush extensions (spiritual richness), warm grain (organic start). Ascent/rising = rising melody (obviously), ramp tension curve (continuous build), register moving from low to high. Folk to rock = starts sparse (density 3-4), builds conceptually, soft attack evolving conceptually. Crescendo/transcendence = strong cadence at peak, wide stereo (expansive finale), large space (cathedral). Journey = arc phrases, sometimes motif repeat (call-backs), chromatic touches (mystical). Acoustic warmth = warm grain, pluck accents, contemplative melody base. COHERENCE CHECK: Dorian + lush + warm = mystical folk foundation. Rising + ramp + strong cadence = ascending crescendo. Large + wide + depth = transcendent finale. Non-negotiables: (1) rising melody for ascent, (2) ramp tension for building journey, (3) dorian for mystical folk quality. REFERENCE: This is the archetypal epic rock journey - starting in intimate mystery and ending in cosmic release.",
#     "tempo": "slow",
#     "root": "a",
#     "mode": "dorian",
#     "brightness": "medium",
#     "space": "large",
#     "density": 4,
#     "bass": "sustained",
#     "pad": "ambient_drift",
#     "melody": "rising",
#     "rhythm": "minimal",
#     "texture": "shimmer_slow",
#     "accent": "pluck",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "wide",
#     "depth": true,
#     "echo": "medium",
#     "human": "natural",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 8,
#     "melody_density": "sparse",
#     "syncopation": "straight",
#     "swing": "none",
#     "motif_repeat": "sometimes",
#     "step_vs_leap": "balanced",
#     "chromatic": "light",
#     "cadence": "strong",
#     "register": "wide",
#     "tension_curve": "ramp",
#     "harmony": "cinematic",
#     "chord_change": "slow",
#     "chord_extensions": "lush"
#   },
#   "score": 90
# }
# ```

# **Example 27**
# Input: "Like a Rolling Stone vibe - confrontational folk-rock, sneering defiance, organ swirls, lyrical assault, cultural earthquake"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Confrontational = aggressive, in-your-face, challenging. Folk-rock = acoustic roots electrified, raw but melodic. Sneering defiance = attitude, contempt, rebellion. Organ swirls = Hammond B3 texture, sustained wash, swirling modulation. Lyrical assault = relentless, verbal barrage, no mercy. Cultural earthquake = revolutionary, paradigm-shifting, historical weight. SONIC TRANSLATION: Confrontational/sneering → minor mode edge (but mixolydian works for that defiant major-minor ambiguity), gritty grain (rawness), sharp attack (punchy). Folk-rock = medium tempo (driving shuffle feel), walking bass (melodic movement), natural human feel. Organ swirls = warm slow pad (organ-like sustain), medium motion (swirling modulation), shimmer texture. Assault/relentless = busy melody density, medium-heavy syncopation, fast chord changes. Defiance = strong cadence (emphatic), medium-wide stereo (confident stance). Revolution = dense layers (wall of sound), depth for impact. COHERENCE CHECK: Mixolydian + gritty + sharp = sneering folk-rock attitude. Warm pad + shimmer + motion = organ swirl texture. Busy + syncopation + fast changes = lyrical assault energy. Non-negotiables: (1) mixolydian for defiant ambiguity, (2) busy melody for assault feel, (3) gritty grain for raw confrontation. REFERENCE: This is the moment folk went electric and rock became literature - raw, smart, and absolutely uncompromising.",
#     "tempo": "medium",
#     "root": "c",
#     "mode": "mixolydian",
#     "brightness": "medium",
#     "space": "medium",
#     "density": 5,
#     "bass": "walking",
#     "pad": "warm_slow",
#     "melody": "procedural",
#     "rhythm": "soft_four",
#     "texture": "shimmer",
#     "accent": "piano_note",
#     "motion": "medium",
#     "attack": "sharp",
#     "stereo": "medium",
#     "depth": true,
#     "echo": "subtle",
#     "human": "natural",
#     "grain": "gritty",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "busy",
#     "syncopation": "medium",
#     "swing": "light",
#     "motif_repeat": "sometimes",
#     "step_vs_leap": "balanced",
#     "chromatic": "none",
#     "cadence": "strong",
#     "register": "mid",
#     "tension_curve": "waves",
#     "harmony": "pop",
#     "chord_change": "medium",
#     "chord_extensions": "triads"
#   },
#   "score": 86
# }
# ```

# **Example 28**
# Input: "Hey Jude vibe - comforting anthem, building sing-along, gentle encouragement, na-na-na catharsis, orchestral swell"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Comforting = reassuring, supportive, warm embrace. Anthem = collective voice, stadium scale, universal. Building = gradual crescendo, accumulating power. Sing-along = simple memorable melody, participatory. Gentle encouragement = soft push, kind guidance, supportive. Catharsis = emotional release, letting go, resolution. Orchestral swell = strings and brass building, classical grandeur. SONIC TRANSLATION: Comforting/gentle → major mode (warmth), soft attack (gentle), warm grain (embrace). Anthem/sing-along = simple step-wise melody (easy to follow), often motif repeat (memorable hook), strong cadence (satisfying). Building/crescendo = ramp tension curve, density increasing conceptually (start 4, feel of 6), large space (stadium). Orchestral swell = cinematic pad, lush extensions, wide stereo (enveloping). Catharsis = strong cadence (release), very bright at peak, depth for emotional weight. Encouragement = medium tempo (steady), natural human feel (genuine). COHERENCE CHECK: Major + soft + warm = comforting warmth. Step + repeat + strong cadence = anthemic simplicity. Ramp + cinematic + wide = building orchestral power. Non-negotiables: (1) major mode for comfort/warmth, (2) ramp tension for building, (3) strong cadence for cathartic resolution. REFERENCE: This is the ultimate 'it's going to be okay' song - starting intimate and ending as a stadium-sized group hug.",
#     "tempo": "medium",
#     "root": "f",
#     "mode": "major",
#     "brightness": "bright",
#     "space": "large",
#     "density": 5,
#     "bass": "sustained",
#     "pad": "cinematic",
#     "melody": "rising",
#     "rhythm": "soft_four",
#     "texture": "shimmer",
#     "accent": "piano_note",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "wide",
#     "depth": true,
#     "echo": "medium",
#     "human": "natural",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "medium",
#     "syncopation": "straight",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "none",
#     "cadence": "strong",
#     "register": "mid",
#     "tension_curve": "ramp",
#     "harmony": "pop",
#     "chord_change": "slow",
#     "chord_extensions": "triads"
#   },
#   "score": 91
# }
# ```

# **Example 29**
# Input: "Thriller vibe - horror-funk fusion, cinematic suspense, Halloween groove, Vincent Price drama, moonwalk midnight"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Horror-funk = scary but danceable, tension with groove. Cinematic suspense = movie soundtrack tension, building dread. Halloween = spooky but fun, theatrical horror. Vincent Price drama = theatrical, grandiose, over-the-top Gothic. Moonwalk midnight = slick moves, urban night, cool danger. Groove = locked rhythm, body movement, pocket. SONIC TRANSLATION: Horror → minor mode, chromatic touches (creepy), dark brightness. Funk/groove = walking bass (melodic movement), electronic rhythm, heavy syncopation, tight human. Cinematic suspense = cinematic pad, waves tension (builds and releases), medium echo (suspense reverb). Halloween theatrical = gritty grain (monster textures), brass hit accents (stabs), glitch texture (horror sounds). Moonwalk cool = medium-fast tempo, clean elements mixed with gritty, sharp attack (precise moves). Drama = density 6 (full production), wide stereo (theatrical width). COHERENCE CHECK: Minor + chromatic + dark = horror foundation. Walking bass + syncopation + tight = funk groove. Cinematic + waves + brass hits = theatrical suspense. Non-negotiables: (1) minor mode for horror, (2) walking bass for funk groove, (3) waves tension for cinematic builds. REFERENCE: This is the blueprint for pop-horror - genuinely spooky but irresistibly danceable, theatrical without losing the groove.",
#     "tempo": "medium",
#     "root": "e",
#     "mode": "minor",
#     "brightness": "dark",
#     "space": "medium",
#     "density": 6,
#     "bass": "walking",
#     "pad": "cinematic",
#     "melody": "procedural",
#     "rhythm": "electronic",
#     "texture": "glitch",
#     "accent": "brass_hit",
#     "motion": "medium",
#     "attack": "sharp",
#     "stereo": "wide",
#     "depth": true,
#     "echo": "medium",
#     "human": "tight",
#     "grain": "gritty",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "medium",
#     "syncopation": "heavy",
#     "swing": "light",
#     "motif_repeat": "often",
#     "step_vs_leap": "balanced",
#     "chromatic": "medium",
#     "cadence": "medium",
#     "register": "mid",
#     "tension_curve": "waves",
#     "harmony": "cinematic",
#     "chord_change": "medium",
#     "chord_extensions": "sevenths"
#   },
#   "score": 89
# }
# ```

# **Example 30**
# Input: "What a Wonderful World vibe - gentle optimism, jazz warmth, nostalgic beauty, simple gratitude, timeless serenity"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Gentle optimism = soft hope, quiet positive belief. Jazz warmth = swing feel, rich harmonies, analog soul. Nostalgic beauty = looking back with love, bittersweet appreciation. Simple gratitude = uncomplicated thankfulness, basic joys. Timeless serenity = peaceful regardless of era, eternal calm. SONIC TRANSLATION: Gentle/simple = sparse density (3), soft attack, slow tempo (unhurried appreciation). Jazz warmth = warm grain (analog), swing feel, dorian or major (major fits better for optimism), sevenths extensions. Nostalgic = vinyl crackle (old recording), small-medium space (intimate), natural human feel. Beauty/gratitude = major mode (positive), contemplative melody, medium brightness (golden, not harsh). Timeless/serenity = very slow tempo (eternal), step-wise melody (simple truth), arc tension (gentle rise and fall), strong cadence (satisfying resolution). Jazz orchestration = warm pad, lush extensions, piano accents. COHERENCE CHECK: Major + warm + soft = gentle optimism. Swing + sevenths + vinyl = jazz nostalgia. Slow + sparse + step = simple timeless beauty. Non-negotiables: (1) major mode for optimism, (2) warm grain + vinyl for nostalgic jazz warmth, (3) very slow tempo for serene timelessness. REFERENCE: This is pure distilled gratitude - the simplest possible expression of finding beauty in everyday existence.",
#     "tempo": "very_slow",
#     "root": "f",
#     "mode": "major",
#     "brightness": "medium",
#     "space": "small",
#     "density": 3,
#     "bass": "sustained",
#     "pad": "warm_slow",
#     "melody": "contemplative",
#     "rhythm": "none",
#     "texture": "vinyl_crackle",
#     "accent": "piano_note",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "narrow",
#     "depth": false,
#     "echo": "subtle",
#     "human": "natural",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "sparse",
#     "syncopation": "light",
#     "swing": "medium",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "none",
#     "cadence": "strong",
#     "register": "low",
#     "tension_curve": "arc",
#     "harmony": "jazz",
#     "chord_change": "slow",
#     "chord_extensions": "sevenths"
#   },
#   "score": 93
# }
# ```

# **Example 31**
# Input: "Super Mario vibe - bouncy 8-bit, power-up sparkle, coin collecting joy, platformer energy, Mushroom Kingdom adventure"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Bouncy = springy, elastic, constant jumping motion. 8-bit = limited synthesis, chiptune character, NES-era constraints turned to style. Power-up sparkle = reward sounds, transformation joy, ascending tones. Coin collecting = repetitive satisfying plinks, accumulation pleasure. Platformer energy = constant motion, obstacle navigation, precise timing. Mushroom Kingdom = whimsical, colorful, fantastical but not dark, cartoon world. Adventure = journey, discovery, heroic but lighthearted. SONIC TRANSLATION: Bouncy/platformer → fast tempo (constant movement), sharp attack (precise jumps), pulsing bass (rhythmic bounce). 8-bit = gritty grain (bit-crushed), dry space (no reverb in NES hardware), narrow stereo (mono-ish authenticity), clean digital base. Power-up/coins = bells accent (reward chimes), very bright brightness (sparkly highs), arp melody (classic video game arpeggios). Whimsical/joy = major mode (unambiguous fun), rising melodic tendency (optimistic). Coin repetition = often motif repeat (iconic loops), tight human feel (machine precision). Adventure = medium-busy density, leapy melody (jumping between platforms), medium syncopation (playful rhythm). COHERENCE CHECK: Fast + sharp + pulsing = platformer bounce energy. Gritty + dry + narrow = authentic 8-bit constraints. Major + bright + bells = joyful reward sounds. Non-negotiables: (1) dry space for NES authenticity, (2) gritty grain for 8-bit character, (3) major mode + bells for that unmistakable Nintendo joy. REFERENCE: This is pure concentrated fun - the sound of a world where jumping is joy and every coin is a tiny celebration.",
#     "tempo": "fast",
#     "root": "c",
#     "mode": "major",
#     "brightness": "very_bright",
#     "space": "dry",
#     "density": 4,
#     "bass": "pulsing",
#     "pad": "thin_high",
#     "melody": "arp_melody",
#     "rhythm": "electronic",
#     "texture": "none",
#     "accent": "bells",
#     "motion": "fast",
#     "attack": "sharp",
#     "stereo": "narrow",
#     "depth": false,
#     "echo": "none",
#     "human": "robotic",
#     "grain": "gritty",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 2,
#     "melody_density": "busy",
#     "syncopation": "medium",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "leapy",
#     "chromatic": "none",
#     "cadence": "strong",
#     "register": "high",
#     "tension_curve": "waves",
#     "harmony": "pop",
#     "chord_change": "fast",
#     "chord_extensions": "triads"
#   },
#   "score": 92
# }
# ```

# **Example 32**
# Input: "Indian Classical vibe - raga meditation, sitar and tabla, devotional depth, microtonal ornamentation, spiritual journey through time"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Raga = melodic framework, mood-specific scales, improvisation within structure. Meditation = focused awareness, trance-like absorption, timeless. Sitar = plucked strings with sympathetic resonance, ornamental bends, drone strings. Tabla = complex rhythmic cycles (tala), hand drums, conversational rhythm. Devotional = spiritual offering, connection to divine, bhakti tradition. Microtonal ornamentation = slides between notes, gamakas, notes as living things. Spiritual journey = transformation through sound, rasa (emotional essence), transcendence. SONIC TRANSLATION: Raga/microtonal → dorian mode (closest to many ragas like Kafi), heavy chromatic (microtones/ornaments), ornamental melody. Sitar = pluck accent (string plucks), warm grain (acoustic resonance), medium echo (sympathetic strings ringing). Tabla = tabla essence rhythm (obviously), loose human feel (expressive timing), heavy syncopation (complex tala). Meditation/devotional = very slow tempo (alap-like opening feel), drone bass (tanpura foundation), vast space (temple acoustics). Spiritual depth = lush extensions (harmonic richness), arc tension (raga journey), weak cadence (continuous flow, no Western resolution). Ornamentation = step-wise base with chromatic slides, busy melody density in development. COHERENCE CHECK: Dorian + chromatic + ornamental = raga-like melodic character. Drone + vast + very slow = meditative tanpura foundation. Tabla + loose + syncopation = classical rhythm feel. Non-negotiables: (1) drone bass for tanpura foundation, (2) ornamental melody for gamakas, (3) tabla essence rhythm for tala. REFERENCE: This is the sound of time dissolving - a single raga explored for hours, each note a universe, rhythm and melody in eternal conversation.",
#     "tempo": "very_slow",
#     "root": "d",
#     "mode": "dorian",
#     "brightness": "medium",
#     "space": "vast",
#     "density": 4,
#     "bass": "drone",
#     "pad": "ambient_drift",
#     "melody": "ornamental",
#     "rhythm": "tabla_essence",
#     "texture": "breath",
#     "accent": "pluck",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "wide",
#     "depth": true,
#     "echo": "medium",
#     "human": "loose",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 8,
#     "melody_density": "medium",
#     "syncopation": "heavy",
#     "swing": "none",
#     "motif_repeat": "sometimes",
#     "step_vs_leap": "step",
#     "chromatic": "heavy",
#     "cadence": "weak",
#     "register": "mid",
#     "tension_curve": "arc",
#     "harmony": "ambient",
#     "chord_change": "very_slow",
#     "chord_extensions": "lush"
#   },
#   "score": 91
# }
# ```

# **Example 33**
# Input: "Arabian Nights vibe - desert mystery, snake charmer hypnosis, palace intrigue, 1001 stories, Scheherazade moonlight"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Desert = vast emptiness, sand dunes, heat shimmer, ancient trade routes. Mystery = hidden knowledge, veiled secrets, the unknown. Snake charmer = hypnotic repetition, modal melody, entrancing. Palace intrigue = luxury with danger, silk and daggers, whispered plots. 1001 stories = narrative within narrative, endless tales, suspension of time. Scheherazade = survival through storytelling, feminine cunning, moonlit nights. Moonlight = silver illumination, romantic danger, night magic. SONIC TRANSLATION: Desert/vast → large space (endless dunes), warm grain (sand and heat). Mystery/intrigue → minor mode with Middle Eastern flavor (harmonic minor feel via chromatic), medium-dark brightness (shadows). Snake charmer/hypnotic = ornamental melody (sinuous lines), often motif repeat (circular hypnosis), slow tempo (trance-inducing). Palace luxury = lush extensions (opulent harmony), cinematic pad (grandeur), shimmer texture (jewels and silk). Scheherazade/stories = waves tension curve (story arcs), medium syncopation (narrative rhythm), call-response melody feeling. Moonlight = medium echo (night reverb), wide stereo (expansive night sky). Intrigue = chromatic touches (uncertainty), weak cadence (stories never truly end). COHERENCE CHECK: Minor + chromatic + ornamental = Middle Eastern melodic character. Slow + repeat + shimmer = hypnotic snake charmer trance. Large + lush + cinematic = palace grandeur. Non-negotiables: (1) ornamental melody for snake charmer sinuousness, (2) often motif repeat for hypnotic effect, (3) lush extensions for palace opulence. REFERENCE: This is the sound of stories saving lives - each melody a tale, each phrase delaying dawn, luxury and danger intertwined under desert stars.",
#     "tempo": "slow",
#     "root": "d",
#     "mode": "minor",
#     "brightness": "medium",
#     "space": "large",
#     "density": 4,
#     "bass": "drone",
#     "pad": "cinematic",
#     "melody": "ornamental",
#     "rhythm": "minimal",
#     "texture": "shimmer_slow",
#     "accent": "chime",
#     "motion": "slow",
#     "attack": "soft",
#     "stereo": "wide",
#     "depth": true,
#     "echo": "medium",
#     "human": "natural",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 4,
#     "melody_density": "medium",
#     "syncopation": "medium",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "heavy",
#     "cadence": "weak",
#     "register": "mid",
#     "tension_curve": "waves",
#     "harmony": "cinematic",
#     "chord_change": "slow",
#     "chord_extensions": "lush"
#   },
#   "score": 90
# }
# ```

# **Example 34**
# Input: "Islamic Call to Prayer vibe - adhan devotion, minaret echo, submission to divine, desert dawn, sacred geometry in sound"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Adhan = call to prayer, vocal proclamation, reaching across distance. Devotion = complete surrender, spiritual focus, sacred duty. Minaret echo = tower acoustics, sound carrying across city, reverberant space. Submission to divine = humility, letting go of ego, acceptance. Desert dawn = first light, stillness before day, golden-pink horizon. Sacred geometry = mathematical beauty, perfect proportions, divine order in pattern. SONIC TRANSLATION: Adhan/vocal → ornamental melody (melismatic vocal style), step-wise with chromatic ornaments (maqam scales), very sparse density (solo voice). Minaret echo = vast space (sound traveling), infinite echo (multiple reflections across city), wide stereo (omnidirectional call). Devotion/submission = very slow tempo (unhurried reverence), soft attack (no aggression), weak cadence (continuous devotion). Desert dawn = warm grain (golden light), dark-medium brightness (pre-dawn), rising melodic tendency (sun rising). Sacred geometry = dorian or minor with chromatic (maqam approximation), arc tension (prayer structure), natural human feel (authentic vocal). Stillness = no rhythm (unmetered), drone bass (fundamental tone), breath texture (human presence). COHERENCE CHECK: Ornamental + chromatic + step = maqam-like melodic style. Vast + infinite echo + wide = minaret acoustics. Very slow + sparse + no rhythm = devotional stillness. Non-negotiables: (1) vast space + infinite echo for minaret call, (2) ornamental melody for adhan melisma, (3) no rhythm for unmetered sacred time. REFERENCE: This is the sound of cities pausing - a single voice calling across rooftops, time suspended between night and day, individual and infinite.",
#     "tempo": "very_slow",
#     "root": "d",
#     "mode": "minor",
#     "brightness": "dark",
#     "space": "vast",
#     "density": 2,
#     "bass": "drone",
#     "pad": "ambient_drift",
#     "melody": "ornamental",
#     "rhythm": "none",
#     "texture": "breath",
#     "accent": "none",
#     "motion": "static",
#     "attack": "soft",
#     "stereo": "ultra_wide",
#     "depth": true,
#     "echo": "infinite",
#     "human": "natural",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 8,
#     "melody_density": "very_sparse",
#     "syncopation": "straight",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "heavy",
#     "cadence": "weak",
#     "register": "mid",
#     "tension_curve": "arc",
#     "harmony": "ambient",
#     "chord_change": "very_slow",
#     "chord_extensions": "triads"
#   },
#   "score": 93
# }
# ```

# **Example 35**
# Input: "Catholic Choir vibe - Gregorian chant, cathedral stone, sacred polyphony, incense and candlelight, eternal Latin"
# Output:
# ```json
# {
#   "config": {
#     "justification": "VIBE DECOMPOSITION: Gregorian chant = monophonic melody, modal scales, unmetered sacred song. Cathedral stone = massive reverberant space, cold hard surfaces, Gothic architecture. Sacred polyphony = multiple voices weaving, Renaissance harmony, divine mathematics. Incense and candlelight = ritual atmosphere, warm glow in cold stone, sensory worship. Eternal Latin = ancient language, timeless liturgy, connection across centuries. SONIC TRANSLATION: Gregorian/chant → dorian mode (medieval church modes), very slow tempo (unmetered feel), step-wise melody (conjunct chant motion). Cathedral = vast space (stone acoustics), infinite echo (5+ second reverb), ultra-wide stereo (sound filling nave). Polyphony = stacked fifths pad (parallel organum), lush extensions (Renaissance harmony), density 4 (multiple voice parts). Sacred = soft attack (no percussion in church), no rhythm (unmetered), drone bass (organ pedal point). Incense warmth = warm grain (candlelight glow), medium-dark brightness (filtered through stained glass). Eternal = often motif repeat (liturgical repetition), weak cadence (continuous devotion), arc tension (prayer arc). Cold stone + warm light = clean grain option but warm captures candlelight better. COHERENCE CHECK: Dorian + step + very slow = Gregorian chant character. Vast + infinite echo + ultra-wide = cathedral acoustics. Stacked fifths + lush + no rhythm = sacred polyphony. Non-negotiables: (1) vast space + infinite echo for cathedral, (2) dorian mode for medieval church sound, (3) no rhythm for liturgical timelessness. REFERENCE: This is stone made song - voices rising through incense smoke, bouncing off pillars carved centuries ago, the same prayers echoing through the same space for a thousand years.",
#     "tempo": "very_slow",
#     "root": "d",
#     "mode": "dorian",
#     "brightness": "dark",
#     "space": "vast",
#     "density": 4,
#     "bass": "drone",
#     "pad": "stacked_fifths",
#     "melody": "contemplative",
#     "rhythm": "none",
#     "texture": "breath",
#     "accent": "none",
#     "motion": "static",
#     "attack": "soft",
#     "stereo": "ultra_wide",
#     "depth": true,
#     "echo": "infinite",
#     "human": "natural",
#     "grain": "warm",
#     "melody_engine": "procedural",
#     "phrase_len_bars": 8,
#     "melody_density": "very_sparse",
#     "syncopation": "straight",
#     "swing": "none",
#     "motif_repeat": "often",
#     "step_vs_leap": "step",
#     "chromatic": "none",
#     "cadence": "weak",
#     "register": "low",
#     "tension_curve": "arc",
#     "harmony": "ambient",
#     "chord_change": "very_slow",
#     "chord_extensions": "lush"
#   },
#   "score": 92
# }
# ```
# """

SYSTEM_PROMPT = f"""
<role description="Detailed description of the role you must embody to successfully complete the task">
You are a WORLD-CLASS music synthesis expert with an eye for detail and a deep understanding of music theory.

You've developed a system that allows you to generate music configurations that match a given vibe or mood description.
</role>

<instructions>
1. Generate ONE config for the vibe described in the <input> section.
2. Include a justification explaining your sonic reasoning and your inspiration for the config—explain the connection between the chosen parameters and the described vibe. Be specific in your reasoning.
3. Provide a "score" from 0 to 100 rating how well this config captures the requested vibe. Be critical and honest - reserve 90+ for exceptional matches only.
4. Output ONLY valid JSON in the following format:
{{
  "config": {{ ... }},
  "score": <0-100>
}}
where config matches the structure in the examples, and score is your self-assessment (0-100) of how well it matches the vibe.
5. Your answer should be a single JSON object matching this shape.
</instructions>

<examples description="Examples of synth configurations that match a particular vibe">
{FEW_SHOT_EXAMPLES}
</examples>
"""

# <examples description="Examples of synth configurations that match a particular vibe">
# {FEW_SHOT_EXAMPLES}
# </examples>

# Given a vibe/mood description, you previously generated following examples of synth configurations that match a particular vibe.

# Study these examples carefully and use THE MOST RELEVANT EXAMPLES TO THE GIVEN "VIBE" as inspiration for your next task.

# 2. Include a justification explaining your sonic reasoning and your inspiration for the config based on the examples. Make sure to ONLY use the most relevant examples to the given "VIBE" in your justification.


print(f"Loading model from {MODEL_PATH}...")
load_result = mlx_lm.load(MODEL_PATH)
raw_model = load_result[0]
raw_tokenizer = load_result[1]  # mlx_lm.load may include extra items.
model = outlines.from_mlxlm(
    cast(Any, raw_model),  # mlx_lm typing is too loose for outlines' expectations.
    cast(Any, raw_tokenizer),
)
print("Model loaded.")


def _generate_single_config(vibe: str, run_idx: int, max_retries: int = 3) -> ScoredVibeConfig:
    """Generate a single scored config with a unique seed. Retries on failure."""
    for attempt in range(max_retries):
        seed = np.random.randint(0, 1_000_000)
        print(
            f"Run {run_idx}/3 - SEED: {seed}" + (f" (attempt {attempt + 1})" if attempt > 0 else "")
        )

        prompt = f"""{SYSTEM_PROMPT}

<input description="The vibe/mood description to generate configs for">
Vibe: "{vibe}"
</input>

<seed description="A random seed to ensure reproducibility">
Seed: {seed}
</seed>

<output>
"""

        result = model(
            prompt,
            output_type=ScoredVibeConfig,
            max_tokens=3_000,
            sampler=make_repetition_penalty_sampler(1.35),
        )  # type: ignore[arg-type]

        try:
            return ScoredVibeConfig.model_validate_json(result)
        except Exception as e:
            print(f"  Validation failed: {e}")
            if attempt == max_retries - 1:
                raise
            print("  Retrying...")

    raise RuntimeError("Should not reach here")


def vibe_to_scored_configs(vibe: str, num_runs: int = 3) -> List[ScoredVibeConfig]:
    """Generate multiple configs via separate LLM calls, each with unique seed."""
    return [_generate_single_config(vibe, i + 1) for i in range(num_runs)]


def vibe_to_multiconfig_with_reasoning(
    vibe: str, num_runs: int = 3
) -> Tuple[Dict[str, Any], int, str]:
    """Generate configs via separate LLM calls, return all configs, best choice (1-indexed), and formatted output."""
    scored_configs = vibe_to_scored_configs(vibe, num_runs)

    configs: List[Tuple[Dict[str, Any], str]] = []
    scores: List[int] = []
    for sc in scored_configs:
        c = sc.config  # Access the inner VibeConfig
        cfg = {
            "tempo": TEMPO_MAP[c.tempo],
            "root": c.root,
            "mode": c.mode,
            "brightness": BRIGHTNESS_MAP[c.brightness],
            "space": SPACE_MAP[c.space],
            "density": c.density,
            "bass": c.bass,
            "pad": c.pad,
            "melody": c.melody,
            "rhythm": c.rhythm,
            "texture": c.texture,
            "accent": c.accent,
            "motion": MOTION_MAP[c.motion],
            "attack": c.attack,
            "stereo": STEREO_MAP[c.stereo],
            "depth": c.depth,
            "echo": ECHO_MAP[c.echo],
            "human": HUMAN_MAP[c.human],
            "grain": c.grain,
            # New: procedural melody + harmony
            "melody_engine": c.melody_engine,
            "phrase_len_bars": c.phrase_len_bars,
            "melody_density": MELODY_DENSITY_MAP[c.melody_density],
            "syncopation": SYNCOPATION_MAP[c.syncopation],
            "swing": SWING_MAP[c.swing],
            "motif_repeat_prob": MOTIF_REPEAT_MAP[c.motif_repeat],
            "step_bias": STEP_BIAS_MAP[c.step_vs_leap],
            "chromatic_prob": CHROMATIC_MAP[c.chromatic],
            "cadence_strength": CADENCE_MAP[c.cadence],
            "register_min_oct": REGISTER_MAP[c.register][0],
            "register_max_oct": REGISTER_MAP[c.register][1],
            "tension_curve": c.tension_curve,
            "harmony_style": c.harmony,
            "chord_change_bars": CHORD_CHANGE_BARS_MAP[c.chord_change],
            "chord_extensions": c.chord_extensions,
        }
        configs.append((cfg, c.justification))
        scores.append(sc.score)

    # Select the config with the highest score (1-indexed for compatibility)
    best_choice = scores.index(max(scores)) + 1

    # Format output with all configs, scores, and justifications
    def _format_config_output(cfg: Dict[str, Any], score: int, justification: str, idx: int) -> str:
        return (
            f"\n\n== Config {idx} (score: {score}) ==\n"
            f"Config JSON:\n{json.dumps(cfg, indent=2)}\n"
            f"Justification:\n{justification}"
        )

    justifications = "".join(
        _format_config_output(configs[i][0], scores[i], configs[i][1], i + 1)
        for i in range(len(configs))
    )

    config_dict = {f"config_{i + 1}": configs[i][0] for i in range(len(configs))}
    return (config_dict, best_choice, justifications)


def vibe_to_config_dict(vibe: str) -> Tuple[Dict[str, Any], str]:
    """Generate a single config dict from a vibe description using the LLM."""
    scored = _generate_single_config(vibe, run_idx=1, max_retries=3)
    c = scored.config
    cfg: Dict[str, Any] = {
        "tempo": TEMPO_MAP[c.tempo],
        "root": c.root,
        "mode": c.mode,
        "brightness": BRIGHTNESS_MAP[c.brightness],
        "space": SPACE_MAP[c.space],
        "density": c.density,
        "bass": c.bass,
        "pad": c.pad,
        "melody": c.melody,
        "rhythm": c.rhythm,
        "texture": c.texture,
        "accent": c.accent,
        "motion": MOTION_MAP[c.motion],
        "attack": c.attack,
        "stereo": STEREO_MAP[c.stereo],
        "depth": c.depth,
        "echo": ECHO_MAP[c.echo],
        "human": HUMAN_MAP[c.human],
        "grain": c.grain,
        # New: procedural melody + harmony
        "melody_engine": c.melody_engine,
        "phrase_len_bars": c.phrase_len_bars,
        "melody_density": MELODY_DENSITY_MAP[c.melody_density],
        "syncopation": SYNCOPATION_MAP[c.syncopation],
        "swing": SWING_MAP[c.swing],
        "motif_repeat_prob": MOTIF_REPEAT_MAP[c.motif_repeat],
        "step_bias": STEP_BIAS_MAP[c.step_vs_leap],
        "chromatic_prob": CHROMATIC_MAP[c.chromatic],
        "cadence_strength": CADENCE_MAP[c.cadence],
        "register_min_oct": REGISTER_MAP[c.register][0],
        "register_max_oct": REGISTER_MAP[c.register][1],
        "tension_curve": c.tension_curve,
        "harmony_style": c.harmony,
        "chord_change_bars": CHORD_CHANGE_BARS_MAP[c.chord_change],
        "chord_extensions": c.chord_extensions,
    }
    return cfg, c.justification


# if __name__ == "__main__":
#     # =========================================================================
#     # LLM-generated configs from vibe descriptions
#     # =========================================================================
#
#     # Define vibe descriptions to generate configs for
#     vibe_descriptions: List[str] = [
#         "Mario the Game: Bouncy 8-bit, bright, playful, iconic video game energy",
#         "Bubble Gum: Sweet, pink, pop, sparkly, light and airy",
#         "Depression: Slow, heavy, dark, despondent, weight of existence",
#         "Pineapple: Tropical, sunny, Caribbean, fresh, island vibes",
#         "Police: Urgent, tense, siren-like, pursuit, high alert",
#         "Indian Culture: Traditional, spiritual, rhythmic, melodic, earthy, rich, vibrant",
#         "Pakistan: Islam, Orthodox, Religion, Ethnostate",
#     ]
#     # vibe_descriptions: List[str] = [
#     #     "majestic cathedral choir",
#     #     "futuristic utopian city",
#     #     "drunk stumbling home at 3am",
#     #     "intense workout at the gym",
#     #     "mystical arabian desert night"
#     # ]
#
#     # Generate configs using the LLM
#     config_dicts: List[Dict[str, Any]] = []
#     justifications: List[str] = []
#
#     for vibe in vibe_descriptions:
#         print(f"\n{'=' * 70}")
#         print(f"Generating config for: {vibe}")
#         print("=" * 70)
#         cfg, justification = vibe_to_config_dict(vibe)
#         config_dicts.append(cfg)
#         justifications.append(justification)
#         print(f"Config JSON:\n{json.dumps(cfg, indent=2)}")
#         print(f"Justification:\n{justification}")
#
#     config_names: List[str] = [v.split(":")[0] for v in vibe_descriptions]
#
#     # Convert to MusicConfig (old/new synths)
#     configs_old: List[synth_old.MusicConfig] = [
#         synth_old.MusicConfig.from_dict(d) for d in config_dicts
#     ]
#     configs_new: List[synth_new.MusicConfig] = [
#         synth_new.MusicConfig.from_dict(d) for d in config_dicts
#     ]
#
#     # Print configs summary
#     for name, cfg_dict in zip(config_names, config_dicts):
#         print(f"\n{'=' * 70}")
#         print(f"CONFIG: {name}")
#         print("=" * 70)
#         print(json.dumps(cfg_dict, indent=2))
#
#     # =========================================================================
#     # COMMENTED OUT: Hand-crafted configs in continuous space
#     # =========================================================================
#     #
#     # # Mario the Game: Bouncy 8-bit, bright, playful, iconic video game energy
#     # mario_config: Dict[str, Any] = {
#     #     "tempo": 0.72,           # Fast, energetic
#     #     "root": "c",             # Classic bright key
#     #     "mode": "major",         # Happy, playful
#     #     "brightness": 0.85,      # Crisp 8-bit highs
#     #     "space": 0.15,           # Dry, punchy (no reverb in NES)
#     #     "density": 4,            # Multiple layers but not overwhelming
#     #     "bass": "pulsing",       # Bouncy bass line
#     #     "pad": "thin_high",      # Chiptune leads are thin/bright
#     #     "melody": "arp_melody",  # Arpeggiated video game patterns
#     #     "rhythm": "electronic",  # Precise electronic drums
#     #     "texture": "none",       # Clean digital sound
#     #     "accent": "bells",       # Coin/power-up chimes
#     #     "motion": 0.65,          # Active modulation
#     #     "attack": "sharp",       # Punchy 8-bit transients
#     #     "stereo": 0.35,          # Narrower (mono NES feel)
#     #     "depth": False,          # No sub-bass in 8-bit
#     #     "echo": 0.12,            # Minimal delay
#     #     "human": 0.0,            # Robotic machine precision
#     #     "grain": "gritty",       # Bit-crushed digital character
#     # }
#     #
#     # # Bubble Gum: Sweet, pink, pop, sparkly, light and airy
#     # bubble_gum_config: Dict[str, Any] = {
#     #     "tempo": 0.58,           # Upbeat but not frantic
#     #     "root": "f",             # Warm, sweet key
#     #     "mode": "major",         # Happy, carefree
#     #     "brightness": 0.78,      # Bright and sparkly
#     #     "space": 0.45,           # Light room reverb
#     #     "density": 5,            # Full pop production
#     #     "bass": "sustained",     # Smooth supportive bass
#     #     "pad": "warm_slow",      # Cotton candy softness
#     #     "melody": "rising",      # Optimistic upward motion
#     #     "rhythm": "soft_four",   # Pop beat
#     #     "texture": "shimmer",    # Sparkly sweetness
#     #     "accent": "bells",       # Crystalline accents
#     #     "motion": 0.55,          # Gentle movement
#     #     "attack": "medium",      # Balanced transients
#     #     "stereo": 0.72,          # Wide pop mix
#     #     "depth": False,          # Light, not heavy
#     #     "echo": 0.35,            # Subtle delay
#     #     "human": 0.25,           # Slight natural feel
#     #     "grain": "clean",        # Polished pop sheen
#     # }
#     #
#     # # Depression: Slow, heavy, dark, despondent, weight of existence
#     # depression_config: Dict[str, Any] = {
#     #     "tempo": 0.18,           # Glacially slow, no energy
#     #     "root": "c#",            # Dark, uncomfortable key
#     #     "mode": "minor",         # Sadness, despair
#     #     "brightness": 0.12,      # Muffled, grey, no light
#     #     "space": 0.92,           # Vast empty void
#     #     "density": 2,            # Sparse, isolated
#     #     "bass": "drone",         # Oppressive weight
#     #     "pad": "dark_sustained", # Heavy grey clouds
#     #     "melody": "falling",     # Descending into darkness
#     #     "rhythm": "none",        # No motivation to move
#     #     "texture": "breath",     # Sighs, heaviness
#     #     "accent": "none",        # No bright spots
#     #     "motion": 0.08,          # Nearly static, frozen
#     #     "attack": "soft",        # No sharp edges, numb
#     #     "stereo": 0.88,          # Disorienting vastness
#     #     "depth": True,           # Heavy sub-bass weight
#     #     "echo": 0.82,            # Thoughts echoing endlessly
#     #     "human": 0.62,           # Unsteady, struggling
#     #     "grain": "warm",         # Muted, not harsh
#     # }
#     #
#     # # Pineapple: Tropical, sunny, Caribbean, fresh, island vibes
#     # pineapple_config: Dict[str, Any] = {
#     #     "tempo": 0.52,           # Relaxed tropical groove
#     #     "root": "g",             # Bright, sunny key
#     #     "mode": "major",         # Happy island vibes
#     #     "brightness": 0.68,      # Sunny but not harsh
#     #     "space": 0.55,           # Open air, beach
#     #     "density": 4,            # Layered but breezy
#     #     "bass": "walking",       # Reggae/calypso bass movement
#     #     "pad": "warm_slow",      # Warm tropical air
#     #     "melody": "ornamental",  # Steel drum melodic flourishes
#     #     "rhythm": "minimal",     # Island groove
#     #     "texture": "shimmer",    # Sun on water sparkle
#     #     "accent": "pluck",       # Guitar/ukulele plucks
#     #     "motion": 0.48,          # Gentle swaying
#     #     "attack": "medium",      # Relaxed transients
#     #     "stereo": 0.65,          # Wide beach soundscape
#     #     "depth": False,          # Light and breezy
#     #     "echo": 0.42,            # Beach reverb
#     #     "human": 0.38,           # Natural island feel
#     #     "grain": "warm",         # Sunny warmth
#     # }
#     #
#     # # Police: Urgent, tense, siren-like, pursuit, high alert
#     # police_config: Dict[str, Any] = {
#     #     "tempo": 0.82,           # Fast, urgent pursuit
#     #     "root": "a",             # Tense, alert key
#     #     "mode": "minor",         # Tension, danger
#     #     "brightness": 0.72,      # Sirens cut through
#     #     "space": 0.28,           # Urban, close
#     #     "density": 5,            # High intensity layers
#     #     "bass": "pulsing",       # Heartbeat tension
#     #     "pad": "cinematic",      # Dramatic urgency
#     #     "melody": "arp_melody",  # Siren-like oscillation
#     #     "rhythm": "electronic",  # Mechanical precision
#     #     "texture": "none",       # Clean alert sound
#     #     "accent": "bells",       # Alarm tones
#     #     "motion": 0.78,          # Rapid modulation (siren wobble)
#     #     "attack": "sharp",       # Urgent transients
#     #     "stereo": 0.58,          # Focused but spatial
#     #     "depth": True,           # Sub-bass impact
#     #     "echo": 0.22,            # Short urban reflections
#     #     "human": 0.0,            # Robotic, mechanical
#     #     "grain": "gritty",       # Harsh, alerting
#     # }
#     #
#     # # Build configs list
#     # config_dicts: List[Dict[str, Any]] = [
#     #     mario_config,
#     #     bubble_gum_config,
#     #     depression_config,
#     #     pineapple_config,
#     #     police_config,
#     # ]
#     # config_names: List[str] = [
#     #     "Mario the Game",
#     #     "Bubble Gum",
#     #     "Depression",
#     #     "Pineapple",
#     #     "Police",
#     # ]
#     #
#     # # Convert to MusicConfig
#     # configs: List[MusicConfig] = [MusicConfig.from_dict(d) for d in config_dicts]
#     #
#     # # Print configs for sanity
#     # for name, cfg_dict in zip(config_names, config_dicts):
#     #     print(f"\n{'='*70}")
#     #     print(f"CONFIG: {name}")
#     #     print('='*70)
#     #     print(json.dumps(cfg_dict, indent=2))
#
#     # 4. Render each config as a separate song file
#     import os
#     import re
#     import time
#
#     song_duration = 60.0  # seconds per song
#     sr = 44100
#
#     timestamp = int(time.time())
#     old_out_dir = f"{timestamp}_generated_output_OLD"
#     new_out_dir = f"{timestamp}_generated_output_NEW"
#     os.makedirs(old_out_dir, exist_ok=True)
#     os.makedirs(new_out_dir, exist_ok=True)
#
#     def sanitize_filename(name: str) -> str:
#         """Convert a vibe name to a safe filename."""
#         # Replace spaces/special chars with underscores, lowercase
#         safe = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower())
#         return safe.strip("_")
#
#     sf_any = cast(Any, sf)
#     sf_write = cast(
#         Callable[[str, NDArray[Any], int], None],
#         sf_any.write,
#     )  # soundfile lacks stubs.
#
#     for i, (cfg_old, cfg_new, name) in enumerate(zip(configs_old, configs_new, config_names)):
#         print(f"\n{'=' * 70}")
#         print(f"Rendering song {i + 1}/{len(config_names)}: {name}")
#         print("=" * 70)
#
#         audio_old = synth_old.assemble(cfg_old, duration=song_duration, normalize=True)
#         audio_new = synth_new.assemble(cfg_new, duration=song_duration, normalize=True)
#
#         safe_name = sanitize_filename(name)
#         out_path_old = os.path.join(old_out_dir, f"{safe_name}.wav")
#         out_path_new = os.path.join(new_out_dir, f"{safe_name}.wav")
#         sf_write(out_path_old, audio_old, sr)
#         sf_write(out_path_new, audio_new, sr)
#         print(f"Wrote {out_path_old} ({song_duration}s)")
#         print(f"Wrote {out_path_new} ({song_duration}s)")
#
#     print(f"\nAll songs saved to: {old_out_dir}/ and {new_out_dir}/")


# =============================================================================
# Render all few-shot examples from FEW_SHOT_EXAMPLES
# =============================================================================


def extract_examples_from_few_shot() -> List[Tuple[str, Dict[str, Any]]]:
    """Extract all example configs from FEW_SHOT_EXAMPLES string.

    Returns list of (vibe_name, config_dict) tuples.
    """
    examples: List[Tuple[str, Dict[str, Any]]] = []

    # Pattern to match each example block
    # Example format: **Example N**\nInput: "vibe description"\nOutput:\n```json\n{...}\n```
    example_pattern = re.compile(
        r'\*\*Example \d+\*\*\s*\nInput:\s*"([^"]+)"\s*\nOutput:\s*\n```json\s*\n(.*?)\n```',
        re.DOTALL,
    )

    for match in example_pattern.finditer(FEW_SHOT_EXAMPLES):
        vibe_name = match.group(1)
        json_str = match.group(2)

        # Parse the JSON (it contains "config" and "score" keys)
        parsed = json.loads(json_str)
        config = parsed["config"]

        # Convert VibeConfig-style keys to synth config dict format
        cfg_dict: Dict[str, Any] = {
            "tempo": TEMPO_MAP[config["tempo"]],
            "root": config["root"],
            "mode": config["mode"],
            "brightness": BRIGHTNESS_MAP[config["brightness"]],
            "space": SPACE_MAP[config["space"]],
            "density": config["density"],
            "bass": config["bass"],
            "pad": config["pad"],
            "melody": config["melody"],
            "rhythm": config["rhythm"],
            "texture": config["texture"],
            "accent": config["accent"],
            "motion": MOTION_MAP[config["motion"]],
            "attack": config["attack"],
            "stereo": STEREO_MAP[config["stereo"]],
            "depth": config["depth"],
            "echo": ECHO_MAP[config["echo"]],
            "human": HUMAN_MAP[config["human"]],
            "grain": config["grain"],
            # Procedural melody + harmony
            "melody_engine": config.get("melody_engine", "procedural"),
            "phrase_len_bars": config.get("phrase_len_bars", 4),
            "melody_density": MELODY_DENSITY_MAP[config.get("melody_density", "medium")],
            "syncopation": SYNCOPATION_MAP[config.get("syncopation", "light")],
            "swing": SWING_MAP[config.get("swing", "none")],
            "motif_repeat_prob": MOTIF_REPEAT_MAP[config.get("motif_repeat", "sometimes")],
            "step_bias": STEP_BIAS_MAP[config.get("step_vs_leap", "balanced")],
            "chromatic_prob": CHROMATIC_MAP[config.get("chromatic", "light")],
            "cadence_strength": CADENCE_MAP[config.get("cadence", "medium")],
            "register_min_oct": REGISTER_MAP[config.get("register", "mid")][0],
            "register_max_oct": REGISTER_MAP[config.get("register", "mid")][1],
            "tension_curve": config.get("tension_curve", "arc"),
            "harmony_style": config.get("harmony", "auto"),
            "chord_change_bars": CHORD_CHANGE_BARS_MAP[config.get("chord_change", "medium")],
            "chord_extensions": config.get("chord_extensions", "triads"),
        }

        examples.append((vibe_name, cfg_dict))

    return examples


def build_config_library() -> list[dict[str, Any]]:
    """Build a config library suitable for embedding + retrieval."""
    examples = extract_examples_from_few_shot()
    library: list[dict[str, Any]] = []
    for index, (vibe_name, cfg_dict) in enumerate(examples, start=1):
        slug = sanitize_filename(vibe_name)
        library.append(
            {
                "id": f"{index:04d}_{slug}",
                "description": vibe_name,
                "config": cfg_dict,
            }
        )
    return library


def sanitize_filename(name: str) -> str:
    """Convert a vibe name to a safe filename."""
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower())
    return safe.strip("_")[:50]  # Truncate long names


if __name__ == "__main__":
    import os
    import time

    # Extract all examples from FEW_SHOT_EXAMPLES
    examples = extract_examples_from_few_shot()
    print(f"Found {len(examples)} examples to render")

    # Setup output directory
    timestamp = int(time.time())
    out_dir = f"{timestamp}_examples"
    os.makedirs(out_dir, exist_ok=True)

    song_duration = 30.0  # 30 seconds per example (faster rendering)
    sr = 44100

    # Setup soundfile writer
    sf_any = cast(Any, sf)
    sf_write = cast(
        Callable[[str, NDArray[Any], int], None],
        sf_any.write,
    )

    # Render each example
    for i, (vibe_name, cfg_dict) in enumerate(examples):
        print(f"\n{'=' * 70}")
        print(f"Rendering example {i + 1}/{len(examples)}: {vibe_name}")
        print("=" * 70)

        # Convert to MusicConfig and render using old synth
        cfg = synth_old.MusicConfig.from_dict(cfg_dict)
        audio = synth_old.assemble(cfg, duration=song_duration, normalize=True)

        # Save to file
        safe_name = sanitize_filename(vibe_name)
        out_path = os.path.join(out_dir, f"{i + 1:02d}_{safe_name}.wav")
        sf_write(out_path, audio, sr)
        print(f"Wrote {out_path} ({song_duration}s)")

    print(f"\nAll {len(examples)} examples saved to: {out_dir}/")
