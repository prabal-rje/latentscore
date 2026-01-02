
from __future__ import annotations

import json
from typing import Literal, Dict, Any, Tuple, List
import numpy as np
from numpy.typing import NDArray
import soundfile as sf

import outlines
import mlx_lm
from pydantic import BaseModel, Field
from .synth import MusicConfig, interpolate_configs, assemble

MODEL_PATH = "./models/gemma-3-1b-it-qat-8bit"

# ---- Output schema for multiple configs + best_choice ----

class VibeConfig(BaseModel):
    """A curated vibe-to-synth mapping with staged justifications."""

    # Meta
    justification: str = Field(
        description="Ultimate reasoning tying all parameter choices together for this config and how it matches the requested vibe. Talk about the CORE, the LAYERS, MELODY POLICY, and other details."
    )    

    # Core parameters
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

    # Layers
    bass: Literal["drone", "sustained", "pulsing", "walking", "fifth_drone", "sub_pulse"] = Field(
        description="Bass character. drone=static, pulsing=rhythmic, walking=melodic movement"
    )
    pad: Literal["warm_slow", "dark_sustained", "cinematic", "thin_high", "ambient_drift", "stacked_fifths"] = Field(
        description="Pad texture. warm_slow=cozy, cinematic=epic, thin_high=ethereal"
    )
    melody: Literal["procedural", "contemplative", "rising", "falling", "minimal", "ornamental", "arp_melody"] = Field(
        description="Melodic style. procedural=generated with policy knobs, contemplative=slow sparse, arp_melody=fast arpeggios"
    )
    rhythm: Literal["none", "minimal", "heartbeat", "soft_four", "hats_only", "electronic"] = Field(
        description="Percussion style. none=ambient, heartbeat=organic pulse, electronic=synthetic"
    )
    texture: Literal["none", "shimmer", "shimmer_slow", "vinyl_crackle", "breath", "stars"] = Field(
        description="Background texture. shimmer=sparkly, vinyl_crackle=nostalgic, breath=organic"
    )
    accent: Literal["none", "bells", "pluck", "chime"] = Field(
        description="Sparse melodic accents. bells=crystalline, pluck=intimate, chime=mystical"
    )
    
    # V2 Parameters
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
    
    # ===== MELODY POLICY KNOBS (NEW!) =====
    # These control the procedural melody generator when melody="procedural"
    
    melody_phrase_len: Literal[2, 4, 8] = Field(
        default=4,
        description="Phrase length in bars. 2=short punchy phrases, 4=balanced, 8=long flowing phrases"
    )
    melody_density: Literal["sparse", "light", "medium", "dense", "very_dense"] = Field(
        default="medium",
        description="Note density. sparse=few notes, medium=balanced, very_dense=many notes per bar"
    )
    melody_syncopation: Literal["none", "light", "medium", "heavy"] = Field(
        default="light",
        description="Off-beat emphasis. none=straight on-beat, heavy=lots of anticipation/delay"
    )
    melody_swing: Literal["straight", "light", "medium", "heavy"] = Field(
        default="straight",
        description="Swing feel. straight=even 8ths, heavy=jazz triplet feel"
    )
    melody_step_vs_leap: Literal["jumpy", "mixed", "smooth"] = Field(
        default="smooth",
        description="Interval preference. jumpy=big leaps, smooth=stepwise motion"
    )
    melody_chromatic: Literal["diatonic", "light", "moderate", "jazzy"] = Field(
        default="diatonic",
        description="Chromatic approach tones. diatonic=in-scale only, jazzy=bebop-style chromaticism"
    )
    melody_motif_repeat: Literal["never", "rare", "sometimes", "often", "always"] = Field(
        default="sometimes",
        description="Motif repetition. never=always new, always=high repetition for memorable themes"
    )
    melody_cadence_strength: Literal["weak", "medium", "strong"] = Field(
        default="medium",
        description="Phrase ending strength. weak=ambiguous, strong=clear resolution to root"
    )
    melody_tension_curve: Literal["flat", "arc", "ramp", "waves"] = Field(
        default="arc",
        description="Tension shape within phrase. arc=build-release, ramp=building, waves=oscillating"
    )


# Scored config: a single VibeConfig with a quality score
class ScoredVibeConfig(BaseModel):
    """A VibeConfig with a self-assessed quality score."""
    config: VibeConfig
    score: int = Field(
        ge=0, le=100,
        description="Self-assessed quality score from 0 to 100. Higher is better. Rate how well this config captures the requested vibe."
    )

# Value mappings for synth
TEMPO_MAP = {"very_slow": 0.15, "slow": 0.3, "medium": 0.5, "fast": 0.7, "very_fast": 0.9}
BRIGHTNESS_MAP = {"very_dark": 0.1, "dark": 0.3, "medium": 0.5, "bright": 0.7, "very_bright": 0.9}
SPACE_MAP = {"dry": 0.1, "small": 0.3, "medium": 0.5, "large": 0.7, "vast": 0.95}
MOTION_MAP = {"static": 0.1, "slow": 0.3, "medium": 0.5, "fast": 0.7, "chaotic": 0.9}
STEREO_MAP = {"mono": 0.0, "narrow": 0.25, "medium": 0.5, "wide": 0.75, "ultra_wide": 1.0}
ECHO_MAP = {"none": 0.0, "subtle": 0.25, "medium": 0.5, "heavy": 0.75, "infinite": 0.95}
HUMAN_MAP = {"robotic": 0.0, "tight": 0.15, "natural": 0.3, "loose": 0.5, "drunk": 0.8}

# NEW: Melody policy mappings
MELODY_DENSITY_MAP = {"sparse": 0.2, "light": 0.35, "medium": 0.5, "dense": 0.7, "very_dense": 0.9}
MELODY_SYNCOPATION_MAP = {"none": 0.0, "light": 0.15, "medium": 0.3, "heavy": 0.5}
MELODY_SWING_MAP = {"straight": 0.0, "light": 0.2, "medium": 0.4, "heavy": 0.6}
MELODY_STEP_VS_LEAP_MAP = {"jumpy": 0.3, "mixed": 0.6, "smooth": 0.9}
MELODY_CHROMATIC_MAP = {"diatonic": 0.0, "light": 0.1, "moderate": 0.2, "jazzy": 0.35}
MELODY_MOTIF_REPEAT_MAP = {"never": 0.0, "rare": 0.15, "sometimes": 0.3, "often": 0.5, "always": 0.7}
MELODY_CADENCE_MAP = {"weak": 0.3, "medium": 0.6, "strong": 0.9}

FEW_SHOT_EXAMPLES = """
Example 1:
Vibe: "dark ambient underwater cave with bioluminescence"
{
  "justification": "Low tempo for stillness. Dorian mode adds mystery. High space simulates underwater reverb. Shimmer texture for bioluminescent sparkles. Sub-bass depth for pressure. Low brightness = murky water filtering light. Procedural melody with sparse density and long phrases for ambient feel.",
  "tempo": "very_slow",
  "root": "d",
  "mode": "dorian",
  "brightness": "very_dark",
  "space": "vast",
  "density": 4,
  "bass": "drone",
  "pad": "dark_sustained",
  "melody": "procedural",
  "rhythm": "none",
  "texture": "shimmer_slow",
  "accent": "none",
  "motion": "slow",
  "attack": "soft",
  "stereo": "wide",
  "depth": true,
  "echo": "heavy",
  "human": "tight",
  "grain": "warm",
  "melody_phrase_len": 8,
  "melody_density": "sparse",
  "melody_syncopation": "none",
  "melody_swing": "straight",
  "melody_step_vs_leap": "smooth",
  "melody_chromatic": "diatonic",
  "melody_motif_repeat": "rare",
  "melody_cadence_strength": "weak",
  "melody_tension_curve": "flat"
}

Example 2:
Vibe: "bouncy 8-bit video game like Mario"
{
  "justification": "Fast tempo for energy. Major mode = playful happiness. Sharp attack = punchy 8-bit transients. Gritty grain = bit-crushed character. Procedural melody with high density, motif repetition for memorable themes, and strong cadences for clear phrase endings.",
  "tempo": "fast",
  "root": "c",
  "mode": "major",
  "brightness": "very_bright",
  "space": "dry",
  "density": 4,
  "bass": "pulsing",
  "pad": "thin_high",
  "melody": "procedural",
  "rhythm": "electronic",
  "texture": "none",
  "accent": "bells",
  "motion": "fast",
  "attack": "sharp",
  "stereo": "narrow",
  "depth": false,
  "echo": "subtle",
  "human": "robotic",
  "grain": "gritty",
  "melody_phrase_len": 4,
  "melody_density": "dense",
  "melody_syncopation": "medium",
  "melody_swing": "straight",
  "melody_step_vs_leap": "mixed",
  "melody_chromatic": "diatonic",
  "melody_motif_repeat": "often",
  "melody_cadence_strength": "strong",
  "melody_tension_curve": "arc"
}

Example 3:
Vibe: "jazz cafe at night, smooth and sophisticated"
{
  "justification": "Medium tempo for relaxed swing. Dorian mode = jazzy sophistication. Walking bass = classic jazz movement. Procedural melody with swing, chromatic approaches for bebop flavor, varied motifs like improvisation.",
  "tempo": "medium",
  "root": "f",
  "mode": "dorian",
  "brightness": "medium",
  "space": "medium",
  "density": 5,
  "bass": "walking",
  "pad": "warm_slow",
  "melody": "procedural",
  "rhythm": "soft_four",
  "texture": "none",
  "accent": "pluck",
  "motion": "medium",
  "attack": "medium",
  "stereo": "medium",
  "depth": false,
  "echo": "subtle",
  "human": "natural",
  "grain": "warm",
  "melody_phrase_len": 4,
  "melody_density": "medium",
  "melody_syncopation": "medium",
  "melody_swing": "medium",
  "melody_step_vs_leap": "mixed",
  "melody_chromatic": "jazzy",
  "melody_motif_repeat": "rare",
  "melody_cadence_strength": "medium",
  "melody_tension_curve": "arc"
}

Example 4:
Vibe: "peaceful meditation in a zen garden"
{
  "justification": "Very slow tempo for mindfulness. Minimal density reduces mental clutter. Soft attacks = non-jarring. Natural human feel = organic imperfection. Chime accents = temple bells. Minimal procedural melody with sparse notes and weak cadences for continuous flow.",
  "tempo": "very_slow",
  "root": "f",
  "mode": "major",
  "brightness": "medium",
  "space": "large",
  "density": 2,
  "bass": "drone",
  "pad": "ambient_drift",
  "melody": "procedural",
  "rhythm": "none",
  "texture": "breath",
  "accent": "chime",
  "motion": "static",
  "attack": "soft",
  "stereo": "medium",
  "depth": false,
  "echo": "medium",
  "human": "natural",
  "grain": "clean",
  "melody_phrase_len": 8,
  "melody_density": "sparse",
  "melody_syncopation": "none",
  "melody_swing": "straight",
  "melody_step_vs_leap": "smooth",
  "melody_chromatic": "diatonic",
  "melody_motif_repeat": "sometimes",
  "melody_cadence_strength": "weak",
  "melody_tension_curve": "flat"
}

Example 5:
Vibe: "epic cinematic battle scene"
{
  "justification": "Fast tempo for urgency. Max density for wall of sound. Minor mode = tension. Sharp attacks = aggressive hits. Cinematic pad for orchestral weight. Procedural melody with ramp tension building to climax, strong cadences for impact.",
  "tempo": "very_fast",
  "root": "d",
  "mode": "minor",
  "brightness": "medium",
  "space": "medium",
  "density": 6,
  "bass": "pulsing",
  "pad": "cinematic",
  "melody": "procedural",
  "rhythm": "soft_four",
  "texture": "none",
  "accent": "bells",
  "motion": "fast",
  "attack": "sharp",
  "stereo": "ultra_wide",
  "depth": true,
  "echo": "subtle",
  "human": "robotic",
  "grain": "warm",
  "melody_phrase_len": 4,
  "melody_density": "dense",
  "melody_syncopation": "light",
  "melody_swing": "straight",
  "melody_step_vs_leap": "mixed",
  "melody_chromatic": "light",
  "melody_motif_repeat": "often",
  "melody_cadence_strength": "strong",
  "melody_tension_curve": "ramp"
}

Example 6:
Vibe: "lo-fi hip hop beats to study to"
{
  "justification": "Slow tempo for relaxation. Warm grain = vintage vibe. Vinyl crackle = nostalgic lo-fi character. Procedural melody with swing for groove, sparse-to-medium density, waves tension for gentle movement.",
  "tempo": "slow",
  "root": "d",
  "mode": "dorian",
  "brightness": "dark",
  "space": "small",
  "density": 4,
  "bass": "sustained",
  "pad": "warm_slow",
  "melody": "procedural",
  "rhythm": "minimal",
  "texture": "vinyl_crackle",
  "accent": "pluck",
  "motion": "slow",
  "attack": "soft",
  "stereo": "medium",
  "depth": false,
  "echo": "medium",
  "human": "natural",
  "grain": "warm",
  "melody_phrase_len": 8,
  "melody_density": "light",
  "melody_syncopation": "light",
  "melody_swing": "medium",
  "melody_step_vs_leap": "smooth",
  "melody_chromatic": "light",
  "melody_motif_repeat": "sometimes",
  "melody_cadence_strength": "weak",
  "melody_tension_curve": "waves"
}

Example 7:
Vibe: "80s synthwave driving at night"
{
  "justification": "Fast tempo = highway speed. Gritty grain = analog synths. Arp melody = classic synthwave. Electronic rhythm = drum machine. Bright = neon lights. Procedural could work but arp_melody fits the genre better.",
  "tempo": "fast",
  "root": "a",
  "mode": "minor",
  "brightness": "bright",
  "space": "medium",
  "density": 5,
  "bass": "pulsing",
  "pad": "stacked_fifths",
  "melody": "arp_melody",
  "rhythm": "electronic",
  "texture": "none",
  "accent": "none",
  "motion": "fast",
  "attack": "sharp",
  "stereo": "wide",
  "depth": true,
  "echo": "medium",
  "human": "robotic",
  "grain": "gritty",
  "melody_phrase_len": 4,
  "melody_density": "dense",
  "melody_syncopation": "light",
  "melody_swing": "straight",
  "melody_step_vs_leap": "mixed",
  "melody_chromatic": "diatonic",
  "melody_motif_repeat": "often",
  "melody_cadence_strength": "medium",
  "melody_tension_curve": "arc"
}

Example 8:
Vibe: "haunted victorian mansion at midnight"
{
  "justification": "Very slow tempo = creeping dread. Minor mode = horror. Very dark brightness = shadows. Vast space = cavernous halls. Procedural melody with chaotic motion, sparse notes, chromatic approaches for unsettling dissonance.",
  "tempo": "very_slow",
  "root": "g#",
  "mode": "minor",
  "brightness": "very_dark",
  "space": "vast",
  "density": 3,
  "bass": "walking",
  "pad": "dark_sustained",
  "melody": "procedural",
  "rhythm": "none",
  "texture": "breath",
  "accent": "chime",
  "motion": "chaotic",
  "attack": "soft",
  "stereo": "wide",
  "depth": true,
  "echo": "heavy",
  "human": "loose",
  "grain": "warm",
  "melody_phrase_len": 8,
  "melody_density": "sparse",
  "melody_syncopation": "none",
  "melody_swing": "straight",
  "melody_step_vs_leap": "jumpy",
  "melody_chromatic": "moderate",
  "melody_motif_repeat": "never",
  "melody_cadence_strength": "weak",
  "melody_tension_curve": "ramp"
}

Example 9:
Vibe: "cheerful summer beach party"
{
  "justification": "Fast tempo = party energy. Major mode = happiness. Very bright = sunshine. Small space = outdoor open air. Procedural melody with high density, strong motif repetition for catchy hooks, strong cadences.",
  "tempo": "fast",
  "root": "c",
  "mode": "major",
  "brightness": "very_bright",
  "space": "small",
  "density": 5,
  "bass": "pulsing",
  "pad": "warm_slow",
  "melody": "procedural",
  "rhythm": "hats_only",
  "texture": "shimmer",
  "accent": "bells",
  "motion": "fast",
  "attack": "medium",
  "stereo": "wide",
  "depth": false,
  "echo": "subtle",
  "human": "natural",
  "grain": "clean",
  "melody_phrase_len": 4,
  "melody_density": "dense",
  "melody_syncopation": "medium",
  "melody_swing": "light",
  "melody_step_vs_leap": "mixed",
  "melody_chromatic": "diatonic",
  "melody_motif_repeat": "often",
  "melody_cadence_strength": "strong",
  "melody_tension_curve": "arc"
}

Example 10:
Vibe: "3AM coding session deep focus"
{
  "justification": "Slow tempo = sustained concentration. Dorian = mysterious but not sad. Sparse density = minimal distraction. Procedural melody with very low density, long phrases, no syncopation for unobtrusive background.",
  "tempo": "slow",
  "root": "e",
  "mode": "dorian",
  "brightness": "dark",
  "space": "medium",
  "density": 3,
  "bass": "drone",
  "pad": "ambient_drift",
  "melody": "procedural",
  "rhythm": "none",
  "texture": "stars",
  "accent": "none",
  "motion": "slow",
  "attack": "soft",
  "stereo": "medium",
  "depth": false,
  "echo": "medium",
  "human": "tight",
  "grain": "clean",
  "melody_phrase_len": 8,
  "melody_density": "sparse",
  "melody_syncopation": "none",
  "melody_swing": "straight",
  "melody_step_vs_leap": "smooth",
  "melody_chromatic": "diatonic",
  "melody_motif_repeat": "sometimes",
  "melody_cadence_strength": "weak",
  "melody_tension_curve": "flat"
}
"""

SYSTEM_PROMPT = f"""
<role description="Detailed description of the role you must embody to successfully complete the task">
You are a WORLD-CLASS music synthesis expert with an eye for detail and a deep understanding of music theory.

You've developed a system that allows you to generate music configurations that match a given vibe/mood description.

The system now includes a PROCEDURAL MELODY GENERATOR with policy knobs that control:
- Phrase structure (phrase_len)
- Note density (how many notes per bar)
- Syncopation (off-beat emphasis)
- Swing (jazz triplet feel)
- Step vs leap preference (smooth stepwise vs jumpy intervals)
- Chromatic approach tones (diatonic vs jazzy)
- Motif repetition (memorable themes vs constant variation)
- Cadence strength (clear phrase endings vs ambiguous)
- Tension curve (arc, ramp, waves, flat)

When melody="procedural", these policy knobs control the generated melody.
For other melody types (contemplative, rising, falling, etc.), the policy knobs are ignored.

Study these examples carefully and use THE MOST RELEVANT EXAMPLES TO THE GIVEN "VIBE" as inspiration.
</role>

<examples description="Examples of synth configurations that match a particular vibe">
{FEW_SHOT_EXAMPLES}
</examples>

<instructions>
1. Generate ONE config for the vibe described in the <input> section.
2. Include a justification explaining your sonic reasoning and your inspiration for the config based on the examples.
3. When the vibe suggests melodic interest (games, jazz, pop), use melody="procedural" and set appropriate policy knobs.
4. When the vibe is ambient/drone/minimal, you can use melody="minimal" or "procedural" with sparse settings.
5. Provide a "score" from 0 to 100 rating how well this config captures the requested vibe.
6. Output ONLY valid JSON in the following format:
{{
  "config": {{ ... }},
  "score": <0-100>
}}
where config matches the structure in the examples, and score is your self-assessment (0-100).
</instructions>
"""

print(f"Loading model from {MODEL_PATH}...")
model = outlines.from_mlxlm(*mlx_lm.load(MODEL_PATH))
print("Model loaded.")


def _generate_single_config(vibe: str, run_idx: int, max_retries: int = 3) -> ScoredVibeConfig:
    """Generate a single scored config with a unique seed. Retries on failure."""
    for attempt in range(max_retries):
        seed = np.random.randint(0, 1_000_000)
        print(f"Run {run_idx}/3 - SEED: {seed}" + (f" (attempt {attempt + 1})" if attempt > 0 else ""))
        
        prompt = f"""{SYSTEM_PROMPT}

<input description="The vibe/mood description to generate configs for">
Vibe: "{vibe}"
</input>

<seed description="A random seed to ensure reproducibility">
Seed: {seed}
</seed>

<output>
"""

        result = model(prompt, output_type=ScoredVibeConfig, max_tokens=3_000)
        
        try:
            return ScoredVibeConfig.model_validate_json(result)
        except Exception as e:
            print(f"  Validation failed: {e}")
            if attempt == max_retries - 1:
                raise
            print(f"  Retrying...")
    
    raise RuntimeError("Should not reach here")


def vibe_to_scored_configs(vibe: str, num_runs: int = 3) -> List[ScoredVibeConfig]:
    """Generate multiple configs via separate LLM calls, each with unique seed."""
    return [_generate_single_config(vibe, i + 1) for i in range(num_runs)]


def vibe_config_to_music_config(c: VibeConfig) -> Dict[str, Any]:
    """Convert a VibeConfig to a MusicConfig-compatible dict."""
    return {
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
        # Melody policy parameters
        "melody_phrase_len": c.melody_phrase_len,
        "melody_density": MELODY_DENSITY_MAP[c.melody_density],
        "melody_syncopation": MELODY_SYNCOPATION_MAP[c.melody_syncopation],
        "melody_swing": MELODY_SWING_MAP[c.melody_swing],
        "melody_step_vs_leap": MELODY_STEP_VS_LEAP_MAP[c.melody_step_vs_leap],
        "melody_chromatic": MELODY_CHROMATIC_MAP[c.melody_chromatic],
        "melody_motif_repeat": MELODY_MOTIF_REPEAT_MAP[c.melody_motif_repeat],
        "melody_cadence_strength": MELODY_CADENCE_MAP[c.melody_cadence_strength],
        "melody_tension_curve": c.melody_tension_curve,
    }


def vibe_to_multiconfig_with_reasoning(vibe: str, num_runs: int = 3) -> Tuple[Dict[str, Any], int, str]:
    """Generate configs via separate LLM calls, return all configs, best choice (1-indexed), and formatted output."""
    scored_configs = vibe_to_scored_configs(vibe, num_runs)
    
    configs: List[Tuple[Dict[str, Any], str]] = []
    scores: List[int] = []
    for sc in scored_configs:
        c = sc.config
        cfg = vibe_config_to_music_config(c)
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
    
    config_dict = {f"config_{i+1}": configs[i][0] for i in range(len(configs))}
    return (config_dict, best_choice, justifications)


def vibe_to_audio(vibe: str, output_path: str = "output.wav", duration: float = 20.0) -> str:
    """Generate audio from a vibe description using LLM config generation."""
    scored_configs = vibe_to_scored_configs(vibe, num_runs=1)
    best_config = scored_configs[0].config
    
    cfg_dict = vibe_config_to_music_config(best_config)
    config = MusicConfig.from_dict(cfg_dict)
    
    from .synth import config_to_audio
    return config_to_audio(config, output_path, duration)


if __name__ == "__main__":
    # =========================================================================
    # Hand-crafted configs in continuous space with melody policy
    # =========================================================================

    # Mario the Game: Bouncy 8-bit, bright, playful, iconic video game energy
    mario_config: Dict[str, Any] = {
        "tempo": 0.72,
        "root": "c",
        "mode": "major",
        "brightness": 0.85,
        "space": 0.15,
        "density": 4,
        "bass": "pulsing",
        "pad": "thin_high",
        "melody": "procedural",  # Use procedural melody
        "rhythm": "electronic",
        "texture": "none",
        "accent": "bells",
        "motion": 0.65,
        "attack": "sharp",
        "stereo": 0.35,
        "depth": False,
        "echo": 0.12,
        "human": 0.0,
        "grain": "gritty",
        # Melody policy for bouncy game music
        "melody_phrase_len": 4,
        "melody_density": 0.8,
        "melody_syncopation": 0.3,
        "melody_swing": 0.0,
        "melody_step_vs_leap": 0.6,
        "melody_chromatic": 0.05,
        "melody_motif_repeat": 0.6,
        "melody_cadence_strength": 0.8,
        "melody_tension_curve": "arc",
    }

    # Lo-fi Study: Chill, vinyl crackle, relaxed
    lofi_config: Dict[str, Any] = {
        "tempo": 0.35,
        "root": "d",
        "mode": "dorian",
        "brightness": 0.4,
        "space": 0.3,
        "density": 4,
        "bass": "sustained",
        "pad": "warm_slow",
        "melody": "procedural",
        "rhythm": "minimal",
        "texture": "vinyl_crackle",
        "accent": "pluck",
        "motion": 0.3,
        "attack": "soft",
        "stereo": 0.5,
        "depth": False,
        "echo": 0.5,
        "human": 0.3,
        "grain": "warm",
        # Lo-fi melody policy
        "melody_phrase_len": 8,
        "melody_density": 0.35,
        "melody_syncopation": 0.15,
        "melody_swing": 0.4,
        "melody_step_vs_leap": 0.9,
        "melody_chromatic": 0.15,
        "melody_motif_repeat": 0.4,
        "melody_cadence_strength": 0.5,
        "melody_tension_curve": "waves",
    }

    # Jazz Cafe: Sophisticated, swinging, chromatic
    jazz_config: Dict[str, Any] = {
        "tempo": 0.45,
        "root": "f",
        "mode": "dorian",
        "brightness": 0.5,
        "space": 0.5,
        "density": 5,
        "bass": "walking",
        "pad": "warm_slow",
        "melody": "procedural",
        "rhythm": "soft_four",
        "texture": "none",
        "accent": "pluck",
        "motion": 0.5,
        "attack": "medium",
        "stereo": 0.6,
        "depth": False,
        "echo": 0.3,
        "human": 0.25,
        "grain": "warm",
        # Jazz melody policy
        "melody_phrase_len": 4,
        "melody_density": 0.6,
        "melody_syncopation": 0.4,
        "melody_swing": 0.5,
        "melody_step_vs_leap": 0.7,
        "melody_chromatic": 0.35,
        "melody_motif_repeat": 0.2,
        "melody_cadence_strength": 0.6,
        "melody_tension_curve": "arc",
    }

    # Ambient Space: Vast, minimal, ethereal
    ambient_config: Dict[str, Any] = {
        "tempo": 0.2,
        "root": "b",
        "mode": "minor",
        "brightness": 0.25,
        "space": 0.95,
        "density": 3,
        "bass": "drone",
        "pad": "ambient_drift",
        "melody": "procedural",
        "rhythm": "none",
        "texture": "stars",
        "accent": "chime",
        "motion": 0.1,
        "attack": "soft",
        "stereo": 1.0,
        "depth": True,
        "echo": 0.9,
        "human": 0.0,
        "grain": "clean",
        # Ambient melody policy
        "melody_phrase_len": 8,
        "melody_density": 0.15,
        "melody_syncopation": 0.0,
        "melody_swing": 0.0,
        "melody_step_vs_leap": 0.95,
        "melody_chromatic": 0.0,
        "melody_motif_repeat": 0.2,
        "melody_cadence_strength": 0.2,
        "melody_tension_curve": "flat",
    }

    # Build configs list
    config_dicts: List[Dict[str, Any]] = [
        mario_config,
        lofi_config,
        jazz_config,
        ambient_config,
    ]
    config_names: List[str] = [
        "Mario Game",
        "Lo-fi Study",
        "Jazz Cafe",
        "Ambient Space",
    ]

    # Convert to MusicConfig
    configs: List[MusicConfig] = [MusicConfig.from_dict(d) for d in config_dicts]

    # Print configs
    for name, cfg_dict in zip(config_names, config_dicts):
        print(f"\n{'='*70}")
        print(f"CONFIG: {name}")
        print('='*70)
        print(json.dumps(cfg_dict, indent=2))

    # Generate audio for each
    sr = 44100
    for name, config in zip(config_names, configs):
        filename = name.lower().replace(" ", "_") + ".wav"
        print(f"\nGenerating: {filename}")
        audio = assemble(config, duration=20.0, normalize=True)
        sf.write(filename, audio, sr)
        print(f"  Saved: {filename}")

    # Morph demo: Mario → Jazz transition
    print("\n" + "="*70)
    print("Generating morph: Mario → Jazz (60s)")
    print("="*70)
    
    all_outputs: List[NDArray[np.float64]] = []
    chunk_duration = 15.0
    num_tween_chunks = 4
    
    for j in range(num_tween_chunks):
        t = j / (num_tween_chunks - 1)
        tweened = interpolate_configs(configs[0], configs[2], t)
        chunk = assemble(tweened, duration=chunk_duration, normalize=False)
        all_outputs.append(chunk)

    full_audio: NDArray[np.float64] = np.concatenate(all_outputs)
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / max_val * 0.85
    
    sf.write("mario_to_jazz_morph.wav", full_audio, sr)
    print("  Saved: mario_to_jazz_morph.wav")

    print("\nAll demos generated!")