from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Literal, Tuple, cast

import mlx_lm
import numpy as np
import outlines
import soundfile as sf
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from .synth import MusicConfig, assemble, interpolate_configs

MODEL_PATH = "./models/gemma-3-1b-it-qat-8bit"

# ---- Output schema for multiple configs + best_choice ----


class VibeConfig(BaseModel):
    """A curated vibe-to-synth mapping with staged justifications."""

    # Meta
    justification: str = Field(
        description="Ultimate reasoning tying all parameter choices together for this config and how it matches the requested vibe. Talk about the CORE, the LAYERS, and other details."
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
    bass: Literal["drone", "sustained", "pulsing", "walking", "fifth_drone", "sub_pulse"] = Field(
        description="Bass character. drone=static, pulsing=rhythmic, walking=melodic movement"
    )
    pad: Literal[
        "warm_slow", "dark_sustained", "cinematic", "thin_high", "ambient_drift", "stacked_fifths"
    ] = Field(description="Pad texture. warm_slow=cozy, cinematic=epic, thin_high=ethereal")
    melody: Literal["contemplative", "rising", "falling", "minimal", "ornamental", "arp_melody"] = (
        Field(description="Melodic contour. rising=hopeful, falling=melancholic, minimal=sparse")
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


# Scored config: a single VibeConfig with a quality score (for single-call generation)
class ScoredVibeConfig(BaseModel):
    """A VibeConfig with a self-assessed quality score."""

    config: VibeConfig
    score: int = Field(
        ge=0,
        le=100,
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

FEW_SHOT_EXAMPLES = """
Example 1:
Vibe: "dark ambient underwater cave with bioluminescence"
{
  "justification": "Low tempo for stillness. Dorian mode adds mystery. High space simulates underwater reverb. Shimmer texture for bioluminescent sparkles. Sub-bass depth for pressure. Low brightness = murky water filtering light.",
  "tempo": "very_slow",
  "root": "d",
  "mode": "dorian",
  "brightness": "very_dark",
  "space": "vast",
  "density": 4,
  "bass": "drone",
  "pad": "dark_sustained",
  "melody": "minimal",
  "rhythm": "none",
  "texture": "shimmer_slow",
  "accent": "none",
  "motion": "slow",
  "attack": "soft",
  "stereo": "wide",
  "depth": true,
  "echo": "heavy",
  "human": "tight",
  "grain": "warm"
}

Example 2:
Vibe: "uplifting sunrise over mountains"
{
  "justification": "Rising melody mirrors sun ascending. Major mode = hope. Increasing brightness like dawn light. Medium tempo = gentle awakening. Bells accent = morning clarity. Wide stereo = vast landscape.",
  "tempo": "medium",
  "root": "c",
  "mode": "major",
  "brightness": "bright",
  "space": "large",
  "density": 4,
  "bass": "sustained",
  "pad": "warm_slow",
  "melody": "rising",
  "rhythm": "minimal",
  "texture": "shimmer",
  "accent": "bells",
  "motion": "medium",
  "attack": "soft",
  "stereo": "wide",
  "depth": false,
  "echo": "medium",
  "human": "natural",
  "grain": "clean"
}

Example 3:
Vibe: "cyberpunk nightclub in tokyo"
{
  "justification": "Fast tempo for dance energy. Electronic rhythm = synthetic. Gritty grain = distorted club speakers. Sharp attack = punchy. Low space = claustrophobic club. High brightness = neon cutting through haze.",
  "tempo": "fast",
  "root": "a",
  "mode": "minor",
  "brightness": "bright",
  "space": "small",
  "density": 6,
  "bass": "pulsing",
  "pad": "cinematic",
  "melody": "arp_melody",
  "rhythm": "electronic",
  "texture": "none",
  "accent": "none",
  "motion": "fast",
  "attack": "sharp",
  "stereo": "wide",
  "depth": true,
  "echo": "subtle",
  "human": "robotic",
  "grain": "gritty"
}

Example 4:
Vibe: "peaceful meditation in a zen garden"
{
  "justification": "Very slow tempo for mindfulness. Minimal density reduces mental clutter. Soft attacks = non-jarring. Natural human feel = organic imperfection. Chime accents = temple bells. Clean grain = clarity of mind.",
  "tempo": "very_slow",
  "root": "f",
  "mode": "major",
  "brightness": "medium",
  "space": "large",
  "density": 2,
  "bass": "drone",
  "pad": "ambient_drift",
  "melody": "minimal",
  "rhythm": "none",
  "texture": "breath",
  "accent": "chime",
  "motion": "static",
  "attack": "soft",
  "stereo": "medium",
  "depth": false,
  "echo": "medium",
  "human": "natural",
  "grain": "clean"
}

Example 5:
Vibe: "epic cinematic battle scene"
{
  "justification": "High tempo for urgency. Max density for wall of sound. Minor mode = tension. Sharp attacks = aggressive. Cinematic pad = orchestral weight. Depth for visceral sub-bass hits. Robotic human = mechanical precision.",
  "tempo": "very_fast",
  "root": "d",
  "mode": "minor",
  "brightness": "medium",
  "space": "medium",
  "density": 6,
  "bass": "pulsing",
  "pad": "cinematic",
  "melody": "rising",
  "rhythm": "soft_four",
  "texture": "none",
  "accent": "bells",
  "motion": "fast",
  "attack": "sharp",
  "stereo": "ultra_wide",
  "depth": true,
  "echo": "subtle",
  "human": "robotic",
  "grain": "warm"
}

Example 6:
Vibe: "rainy afternoon in a cozy coffee shop"
{
  "justification": "Slow tempo for relaxation. Warm grain = vintage speakers. Vinyl crackle = nostalgic ambiance. Medium brightness = soft indoor light. Pluck accent = acoustic guitar feel. Natural human = live performance intimacy.",
  "tempo": "slow",
  "root": "g",
  "mode": "major",
  "brightness": "medium",
  "space": "small",
  "density": 3,
  "bass": "sustained",
  "pad": "warm_slow",
  "melody": "contemplative",
  "rhythm": "minimal",
  "texture": "vinyl_crackle",
  "accent": "pluck",
  "motion": "slow",
  "attack": "soft",
  "stereo": "narrow",
  "depth": false,
  "echo": "subtle",
  "human": "natural",
  "grain": "warm"
}

Example 7:
Vibe: "floating through deep space alone"
{
  "justification": "Very slow tempo = weightlessness. Vast space = infinite void. Thin high pad = cold emptiness. Stars texture = distant galaxies. Ultra wide stereo = disorientation. No rhythm = timelessness. Minor mode = isolation.",
  "tempo": "very_slow",
  "root": "b",
  "mode": "minor",
  "brightness": "dark",
  "space": "vast",
  "density": 3,
  "bass": "sub_pulse",
  "pad": "thin_high",
  "melody": "falling",
  "rhythm": "none",
  "texture": "stars",
  "accent": "none",
  "motion": "static",
  "attack": "soft",
  "stereo": "ultra_wide",
  "depth": true,
  "echo": "infinite",
  "human": "robotic",
  "grain": "clean"
}

Example 8:
Vibe: "ancient forest at twilight with fireflies"
{
  "justification": "Slow tempo = nature's pace. Dorian mode = ancient mystery. Shimmer = firefly blinks. Breath texture = wind through leaves. Large space = forest canopy. Natural human = organic life. Warm grain = earthy tones.",
  "tempo": "slow",
  "root": "e",
  "mode": "dorian",
  "brightness": "dark",
  "space": "large",
  "density": 4,
  "bass": "drone",
  "pad": "ambient_drift",
  "melody": "ornamental",
  "rhythm": "none",
  "texture": "shimmer_slow",
  "accent": "chime",
  "motion": "slow",
  "attack": "soft",
  "stereo": "wide",
  "depth": false,
  "echo": "medium",
  "human": "natural",
  "grain": "warm"
}

Example 9:
Vibe: "80s synthwave driving at night"
{
  "justification": "Fast tempo = highway speed. Gritty grain = analog synths. Arp melody = classic synthwave. Electronic rhythm = drum machine. Bright = neon lights. Pulsing bass = driving force. Sharp attack = punchy gates.",
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
  "grain": "gritty"
}

Example 10:
Vibe: "sad piano in an empty concert hall"
{
  "justification": "Slow tempo = grief. Minor mode = sadness. Vast space = empty hall reverb. Minimal density = solo instrument. Falling melody = descending tears. Clean grain = acoustic clarity. Loose human = emotional rubato.",
  "tempo": "slow",
  "root": "c#",
  "mode": "minor",
  "brightness": "medium",
  "space": "vast",
  "density": 2,
  "bass": "sustained",
  "pad": "thin_high",
  "melody": "falling",
  "rhythm": "none",
  "texture": "none",
  "accent": "pluck",
  "motion": "static",
  "attack": "soft",
  "stereo": "medium",
  "depth": false,
  "echo": "heavy",
  "human": "loose",
  "grain": "clean"
}

Example 11:
Vibe: "tribal drums around a bonfire"
{
  "justification": "Medium tempo = dancing pulse. Heartbeat rhythm = primal drums. Warm grain = fire crackle. Dorian mode = ancient ritual. Tight human = synchronized tribe. Medium space = outdoor clearing. Sharp attack = drum hits.",
  "tempo": "medium",
  "root": "d",
  "mode": "dorian",
  "brightness": "dark",
  "space": "medium",
  "density": 4,
  "bass": "pulsing",
  "pad": "dark_sustained",
  "melody": "minimal",
  "rhythm": "heartbeat",
  "texture": "breath",
  "accent": "none",
  "motion": "medium",
  "attack": "sharp",
  "stereo": "medium",
  "depth": true,
  "echo": "subtle",
  "human": "tight",
  "grain": "warm"
}

Example 12:
Vibe: "haunted victorian mansion at midnight"
{
  "justification": "Very slow tempo = creeping dread. Minor mode = horror. Very dark brightness = shadows. Vast space = cavernous halls. Chime accent = grandfather clock. Walking bass = footsteps. Chaotic motion = paranormal activity.",
  "tempo": "very_slow",
  "root": "g#",
  "mode": "minor",
  "brightness": "very_dark",
  "space": "vast",
  "density": 3,
  "bass": "walking",
  "pad": "dark_sustained",
  "melody": "falling",
  "rhythm": "none",
  "texture": "breath",
  "accent": "chime",
  "motion": "chaotic",
  "attack": "soft",
  "stereo": "wide",
  "depth": true,
  "echo": "heavy",
  "human": "loose",
  "grain": "warm"
}

Example 13:
Vibe: "cheerful summer beach party"
{
  "justification": "Fast tempo = party energy. Major mode = happiness. Very bright = sunshine. Small space = outdoor open air. Hats rhythm = upbeat. Rising melody = good vibes. Wide stereo = crowd spread out.",
  "tempo": "fast",
  "root": "c",
  "mode": "major",
  "brightness": "very_bright",
  "space": "small",
  "density": 5,
  "bass": "pulsing",
  "pad": "warm_slow",
  "melody": "rising",
  "rhythm": "hats_only",
  "texture": "shimmer",
  "accent": "bells",
  "motion": "fast",
  "attack": "medium",
  "stereo": "wide",
  "depth": false,
  "echo": "subtle",
  "human": "natural",
  "grain": "clean"
}

Example 14:
Vibe: "lonely robot contemplating existence"
{
  "justification": "Slow tempo = processing thoughts. Minor mode = melancholy. Clean grain = digital precision. Robotic human = mechanical perfection. Arp melody = algorithmic patterns. Thin high pad = cold circuits. Medium echo = memory loops.",
  "tempo": "slow",
  "root": "f#",
  "mode": "minor",
  "brightness": "medium",
  "space": "medium",
  "density": 3,
  "bass": "drone",
  "pad": "thin_high",
  "melody": "arp_melody",
  "rhythm": "minimal",
  "texture": "none",
  "accent": "bells",
  "motion": "slow",
  "attack": "medium",
  "stereo": "narrow",
  "depth": false,
  "echo": "medium",
  "human": "robotic",
  "grain": "clean"
}

Example 15:
Vibe: "mystical arabian desert night"
{
  "justification": "Medium tempo = camel caravan pace. Dorian mode = middle eastern feel. Ornamental melody = arabic scales. Warm grain = sand and heat. Large space = open desert. Stars texture = night sky. Breath texture = desert wind.",
  "tempo": "medium",
  "root": "d",
  "mode": "dorian",
  "brightness": "dark",
  "space": "large",
  "density": 4,
  "bass": "fifth_drone",
  "pad": "ambient_drift",
  "melody": "ornamental",
  "rhythm": "minimal",
  "texture": "stars",
  "accent": "chime",
  "motion": "medium",
  "attack": "soft",
  "stereo": "wide",
  "depth": false,
  "echo": "medium",
  "human": "natural",
  "grain": "warm"
}

Example 16:
Vibe: "intense workout at the gym"
{
  "justification": "Very fast tempo = high BPM motivation. Sharp attack = punchy transients. Electronic rhythm = four on floor. Gritty grain = distorted energy. Bright = adrenaline. Max density = wall of sound. Robotic = machine-like precision.",
  "tempo": "very_fast",
  "root": "e",
  "mode": "minor",
  "brightness": "bright",
  "space": "dry",
  "density": 6,
  "bass": "pulsing",
  "pad": "cinematic",
  "melody": "rising",
  "rhythm": "electronic",
  "texture": "none",
  "accent": "none",
  "motion": "chaotic",
  "attack": "sharp",
  "stereo": "wide",
  "depth": true,
  "echo": "none",
  "human": "robotic",
  "grain": "gritty"
}

Example 17:
Vibe: "gentle lullaby for a sleeping baby"
{
  "justification": "Very slow tempo = soothing. Major mode = comfort and safety. Very soft attack = non-startling. Minimal density = simple and calming. Clean grain = pure tones. Narrow stereo = intimate closeness. Natural human = mother's touch.",
  "tempo": "very_slow",
  "root": "f",
  "mode": "major",
  "brightness": "dark",
  "space": "small",
  "density": 2,
  "bass": "drone",
  "pad": "warm_slow",
  "melody": "falling",
  "rhythm": "none",
  "texture": "breath",
  "accent": "none",
  "motion": "static",
  "attack": "soft",
  "stereo": "narrow",
  "depth": false,
  "echo": "subtle",
  "human": "natural",
  "grain": "clean"
}

Example 18:
Vibe: "futuristic utopian city"
{
  "justification": "Medium tempo = efficient society. Major mode = optimism. Very bright = gleaming architecture. Clean grain = advanced technology. Rising melody = progress. Stacked fifths = harmonic perfection. Wide stereo = expansive cityscape.",
  "tempo": "medium",
  "root": "c",
  "mode": "major",
  "brightness": "very_bright",
  "space": "large",
  "density": 5,
  "bass": "sustained",
  "pad": "stacked_fifths",
  "melody": "rising",
  "rhythm": "minimal",
  "texture": "shimmer",
  "accent": "bells",
  "motion": "medium",
  "attack": "medium",
  "stereo": "wide",
  "depth": false,
  "echo": "medium",
  "human": "tight",
  "grain": "clean"
}

Example 19:
Vibe: "drunk stumbling home at 3am"
{
  "justification": "Slow tempo = impaired movement. Drunk human = sloppy timing. Minor mode = regret. Dark brightness = streetlights. Walking bass = unsteady footsteps. Loose motion = swaying. Warm grain = blurred perception.",
  "tempo": "slow",
  "root": "a#",
  "mode": "minor",
  "brightness": "dark",
  "space": "medium",
  "density": 3,
  "bass": "walking",
  "pad": "dark_sustained",
  "melody": "falling",
  "rhythm": "minimal",
  "texture": "vinyl_crackle",
  "accent": "none",
  "motion": "chaotic",
  "attack": "soft",
  "stereo": "wide",
  "depth": false,
  "echo": "heavy",
  "human": "drunk",
  "grain": "warm"
}

Example 20:
Vibe: "majestic cathedral choir"
{
  "justification": "Slow tempo = reverence. Major mode = divine glory. Vast space = cathedral acoustics. Stacked fifths pad = choir harmonies. Rising melody = ascending to heaven. Heavy echo = stone walls. Soft attack = voices swelling.",
  "tempo": "slow",
  "root": "c",
  "mode": "major",
  "brightness": "medium",
  "space": "vast",
  "density": 5,
  "bass": "drone",
  "pad": "stacked_fifths",
  "melody": "rising",
  "rhythm": "none",
  "texture": "breath",
  "accent": "bells",
  "motion": "slow",
  "attack": "soft",
  "stereo": "ultra_wide",
  "depth": true,
  "echo": "heavy",
  "human": "natural",
  "grain": "clean"
}
"""

SYSTEM_PROMPT = f"""
<role description="Detailed description of the role you must embody to successfully complete the task">
You are a WORLD-CLASS music synthesis expert with an eye for detail and a deep understanding of music theory.

You've developed a system that allows you to generate music configurations that match a given vibe/mood description.

Given a vibe/mood description, you previously generated following examples of synth configurations that match a particular vibe.

Study these examples carefully and use THE MOST RELEVANT EXAMPLES TO THE GIVEN "VIBE" as inspiration for your next task.
</role>

<examples description="Examples of synth configurations that match a particular vibe">
{FEW_SHOT_EXAMPLES}
</examples>

<instructions>
1. Generate ONE config for the vibe described in the <input> section.
2. Include a justification explaining your sonic reasoning and your inspiration for the config based on the examples. Make sure to ONLY use the most relevant examples to the given "VIBE" in your justification.
3. Provide a "score" from 0 to 100 rating how well this config captures the requested vibe. Be critical and honest - reserve 90+ for exceptional matches only.
4. Output ONLY valid JSON in the following format:
{{
  "config": {{ ... }},
  "score": <0-100>
}}
where config matches the structure in the examples, and score is your self-assessment (0-100) of how well it matches the vibe.
5. Your answer should be a single JSON object matching this shape.
</instructions>
"""

print(f"Loading model from {MODEL_PATH}...")
loaded = mlx_lm.load(MODEL_PATH)
model_obj, tokenizer_obj, *_ = loaded
model = outlines.from_mlxlm(cast(Any, model_obj), cast(Any, tokenizer_obj))
print("Model loaded.")


def _generate_single_config(vibe: str, run_idx: int, max_retries: int = 3) -> ScoredVibeConfig:
    """Generate a single scored config with a unique seed. Retries on failure."""
    rng = np.random.default_rng()
    for attempt in range(max_retries):
        # numpy Generator typing is incomplete for integers().
        seed = int(rng.integers(0, 1_000_000))  # type: ignore[reportUnknownMemberType]
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

        result = model(prompt, output_type=ScoredVibeConfig, max_tokens=3_000)

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


if __name__ == "__main__":
    # =========================================================================
    # Hand-crafted configs in continuous space
    # =========================================================================

    # Mario the Game: Bouncy 8-bit, bright, playful, iconic video game energy
    mario_config: Dict[str, Any] = {
        "tempo": 0.72,  # Fast, energetic
        "root": "c",  # Classic bright key
        "mode": "major",  # Happy, playful
        "brightness": 0.85,  # Crisp 8-bit highs
        "space": 0.15,  # Dry, punchy (no reverb in NES)
        "density": 4,  # Multiple layers but not overwhelming
        "bass": "pulsing",  # Bouncy bass line
        "pad": "thin_high",  # Chiptune leads are thin/bright
        "melody": "arp_melody",  # Arpeggiated video game patterns
        "rhythm": "electronic",  # Precise electronic drums
        "texture": "none",  # Clean digital sound
        "accent": "bells",  # Coin/power-up chimes
        "motion": 0.65,  # Active modulation
        "attack": "sharp",  # Punchy 8-bit transients
        "stereo": 0.35,  # Narrower (mono NES feel)
        "depth": False,  # No sub-bass in 8-bit
        "echo": 0.12,  # Minimal delay
        "human": 0.0,  # Robotic machine precision
        "grain": "gritty",  # Bit-crushed digital character
    }

    # Bubble Gum: Sweet, pink, pop, sparkly, light and airy
    bubble_gum_config: Dict[str, Any] = {
        "tempo": 0.58,  # Upbeat but not frantic
        "root": "f",  # Warm, sweet key
        "mode": "major",  # Happy, carefree
        "brightness": 0.78,  # Bright and sparkly
        "space": 0.45,  # Light room reverb
        "density": 5,  # Full pop production
        "bass": "sustained",  # Smooth supportive bass
        "pad": "warm_slow",  # Cotton candy softness
        "melody": "rising",  # Optimistic upward motion
        "rhythm": "soft_four",  # Pop beat
        "texture": "shimmer",  # Sparkly sweetness
        "accent": "bells",  # Crystalline accents
        "motion": 0.55,  # Gentle movement
        "attack": "medium",  # Balanced transients
        "stereo": 0.72,  # Wide pop mix
        "depth": False,  # Light, not heavy
        "echo": 0.35,  # Subtle delay
        "human": 0.25,  # Slight natural feel
        "grain": "clean",  # Polished pop sheen
    }

    # Depression: Slow, heavy, dark, despondent, weight of existence
    depression_config: Dict[str, Any] = {
        "tempo": 0.18,  # Glacially slow, no energy
        "root": "c#",  # Dark, uncomfortable key
        "mode": "minor",  # Sadness, despair
        "brightness": 0.12,  # Muffled, grey, no light
        "space": 0.92,  # Vast empty void
        "density": 2,  # Sparse, isolated
        "bass": "drone",  # Oppressive weight
        "pad": "dark_sustained",  # Heavy grey clouds
        "melody": "falling",  # Descending into darkness
        "rhythm": "none",  # No motivation to move
        "texture": "breath",  # Sighs, heaviness
        "accent": "none",  # No bright spots
        "motion": 0.08,  # Nearly static, frozen
        "attack": "soft",  # No sharp edges, numb
        "stereo": 0.88,  # Disorienting vastness
        "depth": True,  # Heavy sub-bass weight
        "echo": 0.82,  # Thoughts echoing endlessly
        "human": 0.62,  # Unsteady, struggling
        "grain": "warm",  # Muted, not harsh
    }

    # Pineapple: Tropical, sunny, Caribbean, fresh, island vibes
    pineapple_config: Dict[str, Any] = {
        "tempo": 0.52,  # Relaxed tropical groove
        "root": "g",  # Bright, sunny key
        "mode": "major",  # Happy island vibes
        "brightness": 0.68,  # Sunny but not harsh
        "space": 0.55,  # Open air, beach
        "density": 4,  # Layered but breezy
        "bass": "walking",  # Reggae/calypso bass movement
        "pad": "warm_slow",  # Warm tropical air
        "melody": "ornamental",  # Steel drum melodic flourishes
        "rhythm": "minimal",  # Island groove
        "texture": "shimmer",  # Sun on water sparkle
        "accent": "pluck",  # Guitar/ukulele plucks
        "motion": 0.48,  # Gentle swaying
        "attack": "medium",  # Relaxed transients
        "stereo": 0.65,  # Wide beach soundscape
        "depth": False,  # Light and breezy
        "echo": 0.42,  # Beach reverb
        "human": 0.38,  # Natural island feel
        "grain": "warm",  # Sunny warmth
    }

    # Police: Urgent, tense, siren-like, pursuit, high alert
    police_config: Dict[str, Any] = {
        "tempo": 0.82,  # Fast, urgent pursuit
        "root": "a",  # Tense, alert key
        "mode": "minor",  # Tension, danger
        "brightness": 0.72,  # Sirens cut through
        "space": 0.28,  # Urban, close
        "density": 5,  # High intensity layers
        "bass": "pulsing",  # Heartbeat tension
        "pad": "cinematic",  # Dramatic urgency
        "melody": "arp_melody",  # Siren-like oscillation
        "rhythm": "electronic",  # Mechanical precision
        "texture": "none",  # Clean alert sound
        "accent": "bells",  # Alarm tones
        "motion": 0.78,  # Rapid modulation (siren wobble)
        "attack": "sharp",  # Urgent transients
        "stereo": 0.58,  # Focused but spatial
        "depth": True,  # Sub-bass impact
        "echo": 0.22,  # Short urban reflections
        "human": 0.0,  # Robotic, mechanical
        "grain": "gritty",  # Harsh, alerting
    }

    # Build configs list
    config_dicts: List[Dict[str, Any]] = [
        mario_config,
        bubble_gum_config,
        depression_config,
        pineapple_config,
        police_config,
    ]
    config_names: List[str] = [
        "Mario the Game",
        "Bubble Gum",
        "Depression",
        "Pineapple",
        "Police",
    ]

    # Convert to MusicConfig
    configs: List[MusicConfig] = [MusicConfig.from_dict(d) for d in config_dicts]

    # Print configs for sanity
    for name, cfg_dict in zip(config_names, config_dicts):
        print(f"\n{'=' * 70}")
        print(f"CONFIG: {name}")
        print("=" * 70)
        print(json.dumps(cfg_dict, indent=2))

    # 4. Interpolate between the configs with tweening, concatenate output
    all_outputs: List[NDArray[np.float64]] = []
    chunk_duration = 30.0  # seconds per section
    sr = 44100

    if len(configs) == 1:
        # Only one config, just render it
        single_audio = assemble(configs[0], duration=chunk_duration, normalize=False)
        all_outputs.append(single_audio)
    else:
        for i in range(len(configs) - 1):
            config_a: MusicConfig = configs[i]
            config_b: MusicConfig = configs[i + 1]

            # Tween from config_a to config_b (slowly, ~16s fade)
            num_tween_chunks = 8
            for j in range(num_tween_chunks):
                t = j / (num_tween_chunks - 1)
                tweened = interpolate_configs(config_a, config_b, t)
                # Note: Set normalize=False to prevent jumps, let final global normalize
                chunk = assemble(tweened, duration=chunk_duration, normalize=False)
                all_outputs.append(chunk)

    full_audio: NDArray[np.float64] = np.concatenate(all_outputs)
    # Save output to "test.wav"
    write_audio = cast(Callable[[str, NDArray[np.float64], int], None], sf.write)  # type: ignore[reportUnknownMemberType]
    write_audio("test.wav", full_audio, sr)  # type: ignore[reportUnknownMemberType]
    print("\nConcatenated tweened audio saved to test.wav")
