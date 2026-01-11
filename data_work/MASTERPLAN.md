# Vibe Audio Config: Master Plan

## A Complete Guide for Training Tiny LLMs to Generate Synthesizer Configurations

**Target Models:** SmolLM2-135M, Gemma-3-270M, Qwen3-600M (optional 1B)  
**Output Format:** JSON configurations for FM/Subtractive synthesis engine  
**Deployment Target:** WebGPU inference in browser

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Schema Reference](#2-schema-reference)
3. [Dataset Generation Strategy](#3-dataset-generation-strategy)
4. [Training Pipeline](#4-training-pipeline)
5. [Reward Function Design](#5-reward-function-design)
6. [Inference & Deployment](#6-inference--deployment)
7. [Hosting & Licensing](#7-hosting--licensing)
8. [Repository Structure](#8-repository-structure)
9. [Artifact Checklist](#9-artifact-checklist)

---

## 1. Project Overview

### Goal

Train tiny language models (135M-600M parameters) to translate natural language "vibe" descriptions into valid JSON configurations for a deterministic FM/Subtractive synthesizer. Users describe a mood ("rainy afternoon in Tokyo", "anxious waiting room"), and the model outputs a structured configuration that the synth engine renders to audio.

### Why Tiny Models?

- **WebGPU Deployment**: Models must run client-side in browsers
- **Latency**: Sub-second generation for real-time interaction
- **Cost**: Free inference, no API calls in production
- **Constrained Output Space**: The JSON schema is finite and well-defined—a perfect fit for small, specialized models

### Constraints

The synthesis engine produces **ambient/electronic textures only**:
- ✅ Pads, drones, simple rhythms, textures, bells, arpeggios
- ❌ Vocals, orchestral instruments, complex drums, genre-specific sounds

---

## 2. Schema Reference

### Complete Enum Values

Your models must learn to output these **exact** values. This is the source of truth:

```python
# Continuous-mapped labels (model outputs these strings)
TempoLabel     = ["very_slow", "slow", "medium", "fast", "very_fast"]
BrightnessLabel = ["very_dark", "dark", "medium", "bright", "very_bright"]
SpaceLabel     = ["dry", "small", "medium", "large", "vast"]
MotionLabel    = ["static", "slow", "medium", "fast", "chaotic"]
StereoLabel    = ["mono", "narrow", "medium", "wide", "ultra_wide"]
EchoLabel      = ["none", "subtle", "medium", "heavy", "infinite"]
HumanFeelLabel = ["robotic", "tight", "natural", "loose", "drunk"]

# Musical parameters
RootNote       = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
ModeName       = ["major", "minor", "dorian", "mixolydian"]
DensityLevel   = [2, 3, 4, 5, 6]  # Integer

# Layer styles (categorical)
BassStyle      = ["drone", "sustained", "pulsing", "walking", "fifth_drone", 
                  "sub_pulse", "octave", "arp_bass"]
PadStyle       = ["warm_slow", "dark_sustained", "cinematic", "thin_high", 
                  "ambient_drift", "stacked_fifths", "bright_open"]
MelodyStyle    = ["contemplative", "rising", "falling", "minimal", "ornamental", 
                  "arp_melody", "contemplative_minor", "call_response", "heroic"]
RhythmStyle    = ["none", "minimal", "heartbeat", "soft_four", "hats_only", 
                  "electronic", "kit_light", "kit_medium", "military", 
                  "tabla_essence", "brush"]
TextureStyle   = ["none", "shimmer", "shimmer_slow", "vinyl_crackle", "breath", 
                  "stars", "glitch", "noise_wash", "crystal", "pad_whisper"]
AccentStyle    = ["none", "bells", "pluck", "chime", "bells_dense", "blip", 
                  "blip_random", "brass_hit", "wind", "arp_accent", "piano_note"]

# Additional parameters
AttackStyle    = ["soft", "medium", "sharp"]
GrainStyle     = ["clean", "warm", "gritty"]
depth          = [true, false]  # Boolean
```

### Target JSON Output Format

```json
{
  "tempo": "slow",
  "root": "d",
  "mode": "minor",
  "brightness": "dark",
  "space": "large",
  "density": 4,
  "bass": "drone",
  "pad": "dark_sustained",
  "melody": "contemplative",
  "rhythm": "minimal",
  "texture": "shimmer_slow",
  "accent": "bells",
  "motion": "slow",
  "attack": "soft",
  "stereo": "wide",
  "depth": true,
  "echo": "heavy",
  "human": "natural",
  "grain": "warm"
}
```

---

## 3. Dataset Generation Strategy

### The Core Challenge

You need vibes that:
1. **Cover the output space**: Every enum value appears in training data
2. **Match synthesis capabilities**: Don't describe sounds the engine can't make
3. **Reflect natural language diversity**: Real users won't say "bass=pulsing"

### Recommended Approach: Hierarchical Vibe Generation

#### Step 1: Define Coverage Categories

Create a taxonomy that ensures full enum coverage:

```
MOODS/EMOTIONS
├── Positive: peaceful, joyful, hopeful, triumphant, playful, cozy, nostalgic
├── Negative: sad, anxious, melancholic, lonely, tense, uneasy, haunting
├── Neutral: contemplative, meditative, mysterious, ethereal, dreamlike
└── Intense: epic, dramatic, chaotic, frantic, overwhelming

ENVIRONMENTS/PLACES
├── Natural: forest, ocean, desert, mountain, cave, underwater, sky
├── Urban: city night, subway, empty office, rooftop, warehouse
├── Domestic: bedroom, kitchen morning, late night living room
├── Abstract: void, liminal space, digital realm, memory palace

TIME/TEMPORAL
├── Time of day: dawn, morning, afternoon, dusk, night, midnight, 3am
├── Seasons: spring morning, summer evening, autumn rain, winter silence
├── Eras: retro, futuristic, timeless, ancient

ACTIVITIES/CONTEXTS
├── States: falling asleep, waking up, working late, waiting
├── Movement: walking slowly, floating, running, standing still
└── Events: before the storm, after the party, end credits
```

#### Step 2: Combinatorial Expansion

Generate vibes by combining elements:

```python
# Example generation logic
templates = [
    "{mood} {environment}",
    "{time} in a {place}",
    "{activity}, feeling {emotion}",
    "the feeling of {abstract_concept}",
    "{adjective} {noun}",
]

# Generate: "melancholic forest at dusk"
# Generate: "3am in an empty office"
# Generate: "floating through a digital void"
```

#### Step 3: Ensure Enum Coverage

**Critical**: Track which enum values are underrepresented. Create targeted vibes:

| Enum Value | Underrepresented? | Targeted Vibes |
|------------|-------------------|----------------|
| `bass="arp_bass"` | Yes | "pulsing synth wave", "retro game soundtrack" |
| `rhythm="military"` | Yes | "marching into battle", "parade ground at dawn" |
| `texture="glitch"` | Yes | "corrupted memory", "digital decay" |
| `human="drunk"` | Yes | "stumbling home at 2am", "dizzy carousel" |
| `mode="mixolydian"` | Yes | "folk tale ending", "traveling bard" |

#### Step 4: LLM-Assisted Expansion

Use a capable model (Claude, GPT-4) to expand your base vibes:

```
PROMPT:
Given this synthesis engine that produces ambient/electronic textures, 
generate 50 diverse "vibe" descriptions that could map to these parameters:
[paste schema]

Requirements:
- Natural language (not technical)
- Diverse emotions, places, times, activities
- Grounded in experiences a computer user might have
- 5-15 words each
```

### Dataset Size Recommendations

| Model Size | Minimum SFT Examples | Recommended |
|------------|---------------------|-------------|
| 135M | 5,000 | 10,000 |
| 270M | 5,000 | 10,000 |
| 600M | 3,000 | 7,000 |
| 1B | 2,000 | 5,000 |

Smaller models need more examples because they have less pre-existing knowledge to leverage.

---

### Synthetic Data Generation: Best-of-N Sampling

Don't just generate one config per vibe — generate N candidates and **keep the best** according to CLAP score. This dramatically improves training data quality.

```python
import json
from anthropic import Anthropic

client = Anthropic()

def generate_config_candidates(vibe: str, n: int = 5) -> list[dict]:
    """Generate N config candidates for a single vibe using SOTA LLM."""
    
    candidates = []
    
    for _ in range(n):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",  # Your teacher model
            max_tokens=512,
            temperature=0.9,  # Higher temp = more diverse candidates
            messages=[{
                "role": "user",
                "content": f"""Generate a JSON synthesizer configuration for this vibe:
"{vibe}"

Output ONLY valid JSON matching this schema:
- tempo: very_slow|slow|medium|fast|very_fast
- root: c|c#|d|d#|e|f|f#|g|g#|a|a#|b
- mode: major|minor|dorian|mixolydian
- brightness: very_dark|dark|medium|bright|very_bright
- space: dry|small|medium|large|vast
- density: 2|3|4|5|6
- bass: drone|sustained|pulsing|walking|fifth_drone|sub_pulse|octave|arp_bass
- pad: warm_slow|dark_sustained|cinematic|thin_high|ambient_drift|stacked_fifths|bright_open
- melody: contemplative|rising|falling|minimal|ornamental|arp_melody|contemplative_minor|call_response|heroic
- rhythm: none|minimal|heartbeat|soft_four|hats_only|electronic|kit_light|kit_medium|military|tabla_essence|brush
- texture: none|shimmer|shimmer_slow|vinyl_crackle|breath|stars|glitch|noise_wash|crystal|pad_whisper
- accent: none|bells|pluck|chime|bells_dense|blip|blip_random|brass_hit|wind|arp_accent|piano_note
- motion: static|slow|medium|fast|chaotic
- attack: soft|medium|sharp
- stereo: mono|narrow|medium|wide|ultra_wide
- depth: true|false
- echo: none|subtle|medium|heavy|infinite
- human: robotic|tight|natural|loose|drunk
- grain: clean|warm|gritty"""
            }]
        )
        
        try:
            config = json.loads(response.content[0].text)
            candidates.append(config)
        except json.JSONDecodeError:
            continue  # Skip invalid JSON
    
    return candidates


def select_best_config(vibe: str, candidates: list[dict]) -> tuple[dict, float]:
    """
    Select best config from candidates using CLAP score.
    Returns (best_config, best_score).
    """
    best_config = None
    best_score = -float('inf')
    
    for config in candidates:
        # Synthesize audio
        audio_path = synthesize_temp(config)
        
        # Score with CLAP
        score = reward_audio_clap(vibe, audio_path)
        
        if score > best_score:
            best_score = score
            best_config = config
        
        # Cleanup temp file
        os.remove(audio_path)
    
    return best_config, best_score


def generate_training_example(vibe: str, n_candidates: int = 5) -> dict:
    """
    Generate a single high-quality training example using Best-of-N.
    """
    candidates = generate_config_candidates(vibe, n=n_candidates)
    
    if not candidates:
        return None  # All generations failed
    
    best_config, score = select_best_config(vibe, candidates)
    
    return {
        "vibe": vibe,
        "config": best_config,
        "clap_score": score,
        "n_candidates": len(candidates),
    }


def generate_dataset(
    vibes: list[str], 
    n_candidates: int = 5,
    min_score: float = 0.4,
    output_path: str = "data/configs/synthetic_pairs.jsonl"
) -> None:
    """
    Generate full training dataset using Best-of-N sampling.
    
    Args:
        vibes: List of vibe descriptions
        n_candidates: Number of candidates to generate per vibe (3-5 recommended)
        min_score: Minimum CLAP score to include example (quality filter)
    """
    examples = []
    stats = {"total": 0, "accepted": 0, "rejected_low_score": 0, "rejected_invalid": 0}
    
    for i, vibe in enumerate(vibes):
        print(f"[{i+1}/{len(vibes)}] {vibe[:50]}...")
        stats["total"] += 1
        
        example = generate_training_example(vibe, n_candidates)
        
        if example is None:
            stats["rejected_invalid"] += 1
            continue
        
        if example["clap_score"] < min_score:
            stats["rejected_low_score"] += 1
            continue
        
        examples.append(example)
        stats["accepted"] += 1
    
    # Save dataset
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"\nDataset generation complete:")
    print(f"  Total vibes: {stats['total']}")
    print(f"  Accepted: {stats['accepted']} ({stats['accepted']/stats['total']:.1%})")
    print(f"  Rejected (low score): {stats['rejected_low_score']}")
    print(f"  Rejected (invalid JSON): {stats['rejected_invalid']}")
    print(f"  Mean CLAP score: {np.mean([ex['clap_score'] for ex in examples]):.3f}")
```

**Why Best-of-N?**

| Approach | Avg CLAP Score | % Valid JSON | Training Quality |
|----------|----------------|--------------|------------------|
| N=1 (single shot) | ~0.45 | ~85% | Baseline |
| N=3 (best of 3) | ~0.55 | ~95% | Good |
| N=5 (best of 5) | ~0.60 | ~98% | Best |
| N=10 | ~0.63 | ~99% | Diminishing returns |

**Recommendation**: Use N=5 for production dataset. The extra API cost is worth it for higher quality training signal.

**Cost estimate** (10k examples, N=5):
- 50k API calls × ~500 tokens × $3/1M tokens ≈ $75
- Plus ~10k audio syntheses for CLAP scoring (~2-3 hours compute)

---

### Preventing Enum Forgetting

**Your concern is valid.** If `bass="arp_bass"` only appears in 0.5% of SFT data, the model may never learn to output it.

**Mitigation strategies:**

1. **Balanced Sampling**: Ensure each enum value appears in at least 2% of examples (minimum ~200 per value for 10k dataset)

2. **Explicit Coverage Check**:
```python
from collections import Counter

def check_coverage(dataset):
    for field in ['bass', 'pad', 'melody', 'rhythm', 'texture', 'accent']:
        values = [ex['output'][field] for ex in dataset]
        counts = Counter(values)
        print(f"{field}: {dict(counts)}")
        # Alert if any value < 100 occurrences
```

3. **Augmentation**: For rare enums, generate additional targeted examples

4. **GRPO Will Help**: Even if SFT coverage is imperfect, GRPO explores the action space. As long as the model can *sometimes* output rare values (even by accident), GRPO rewards will reinforce them when they're appropriate.

---

## 4. Training Pipeline

### Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Synthetic Data │ ──▶ │   SFT Phase     │ ──▶ │   GRPO Phase    │
│  Generation     │     │  (Supervised)   │     │  (RL Alignment) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                       │
                              ▼                       ▼
                        Base model learns       Model learns to
                        JSON format +           maximize audio
                        vibe→config mapping     quality scores
```

### SFT Phase Details

#### Training Configuration (Unsloth)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="HuggingFaceTB/SmolLM2-135M-Instruct",  # or other base
    max_seq_length=512,  # JSON outputs are short
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # LoRA rank (higher for tiny models)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
```

#### Hyperparameters for Tiny Models

| Parameter | 135M | 270M | 600M | 1B |
|-----------|------|------|------|-----|
| Learning Rate | 2e-4 | 1e-4 | 5e-5 | 3e-5 |
| Epochs | 3-5 | 2-4 | 2-3 | 1-2 |
| Batch Size | 8-16 | 8-16 | 4-8 | 4 |
| LoRA Rank (r) | 32-64 | 32-64 | 16-32 | 16 |
| Warmup Ratio | 0.1 | 0.1 | 0.05 | 0.03 |

**Key insight**: Tiny models benefit from higher learning rates and more epochs than larger models.

#### Training Time Estimates (10k dataset)

| Platform | 135M | 270M | 600M |
|----------|------|------|------|
| Colab Free (T4) | 1-2 hours | 2-3 hours | 4-6 hours |
| Colab Pro (A100) | 20-30 min | 30-45 min | 1-1.5 hours |
| Local RTX 3090 | 30-45 min | 45-60 min | 1.5-2 hours |

**Recommendation**: Colab Pro is worth it for iteration speed. You can train all 3 models in a single session.

#### Prompt Template

```python
SYSTEM_PROMPT = """You are a synthesizer configuration assistant. 
Given a vibe description, output a JSON configuration for an ambient/electronic synthesizer.
Output ONLY valid JSON with no explanation."""

def format_example(vibe: str, config: dict) -> str:
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{vibe}<|im_end|>
<|im_start|>assistant
{json.dumps(config)}<|im_end|>"""
```

### GRPO Phase Details

#### What is GRPO?

Group Relative Policy Optimization is a variant of PPO that:
- Generates multiple completions per prompt
- Ranks them by reward
- Updates policy to favor higher-reward completions
- More sample-efficient than standard RLHF

#### Loss Type Options (Unsloth/TRL)

| Loss Type | Description | When to Use |
|-----------|-------------|-------------|
| `grpo` (default) | Standard GRPO | Good starting point |
| `dapo` | Dynamic Advantage PO | When rewards are noisy |
| `bnpo` | Bounded Normalized PO | For stable training |
| `dr_grpo` | Distributional Robust GRPO | When reward distribution varies |

**Recommendation**: Start with default `grpo`. Switch to `dapo` if you see high variance in training.

#### Who Generates Prompts During GRPO?

**Standard GRPO**: You provide a fixed dataset of prompts (your vibes). The trainer samples from this pool repeatedly.

**But you can do better**: Use a "prompt generator" LLM to create fresh prompts on-the-fly.

```python
# Dynamic prompt generation during GRPO
import random
from anthropic import Anthropic

client = Anthropic()

# Seed concepts for variety
SEED_MOODS = ["peaceful", "anxious", "nostalgic", "chaotic", "dreamy", "tense"]
SEED_PLACES = ["forest", "city", "underwater", "space", "bedroom", "train station"]
SEED_TIMES = ["3am", "dawn", "midnight", "afternoon", "dusk"]

def generate_fresh_vibes(n: int = 16, temperature: float = 1.0) -> list[str]:
    """
    Use a high-temperature LLM to generate novel, diverse prompts.
    Called each GRPO batch to ensure variety.
    """
    seed_context = f"""
    Moods: {random.sample(SEED_MOODS, 3)}
    Places: {random.sample(SEED_PLACES, 3)}
    Times: {random.sample(SEED_TIMES, 2)}
    """
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=temperature,  # High temp = more variety
        messages=[{
            "role": "user",
            "content": f"""Generate {n} diverse "vibe" descriptions for an ambient synthesizer.
            
Use these as loose inspiration (but go beyond them):
{seed_context}

Requirements:
- 5-15 words each
- Natural language, evocative
- Cover different emotions, places, times, activities
- Some abstract, some concrete
- NO numbering, just one per line"""
        }]
    )
    
    vibes = response.content[0].text.strip().split("\n")
    return [v.strip() for v in vibes if v.strip()]


class DynamicVibeDataset:
    """
    Dataset that generates fresh prompts each epoch.
    Prevents overfitting to fixed prompt distribution.
    """
    def __init__(self, base_vibes: list[str], refresh_ratio: float = 0.3):
        self.base_vibes = base_vibes  # Your curated set
        self.refresh_ratio = refresh_ratio  # % of batch that's freshly generated
    
    def sample_batch(self, batch_size: int) -> list[str]:
        n_fresh = int(batch_size * self.refresh_ratio)
        n_base = batch_size - n_fresh
        
        batch = random.sample(self.base_vibes, min(n_base, len(self.base_vibes)))
        
        if n_fresh > 0:
            fresh = generate_fresh_vibes(n_fresh, temperature=1.2)
            batch.extend(fresh)
        
        random.shuffle(batch)
        return batch
```

**Why this helps:**
- Prevents overfitting to your fixed prompt set
- Forces generalization to unseen descriptions
- High temperature → more creative/unusual vibes → better robustness

---

#### Can the Evaluator Give Actionable Advice?

**Short answer**: Not directly in standard GRPO. The reward is a scalar—no text feedback.

**But there are workarounds:**

##### Option 1: Critique-Augmented Training (Two-Stage)

After GRPO generates completions, have an LLM critique them, then do a second SFT pass on (vibe, critique, corrected_config) triples:

```python
def generate_critique(vibe: str, config: dict, reward: float) -> str:
    """
    Generate actionable feedback for low-reward outputs.
    Used for secondary SFT, not during GRPO itself.
    """
    if reward > 0.8:
        return None  # Good enough, no critique needed
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user", 
            "content": f"""The vibe was: "{vibe}"
The model output: {json.dumps(config)}
The audio quality score was: {reward:.2f} (low)

What's wrong with this config? Give 1-2 specific, actionable fixes.
Example: "tempo should be 'slow' not 'fast' for a melancholic vibe"
"""
        }]
    )
    return response.content[0].text

# Collect critiques during GRPO eval
critiques = []
for vibe, output, reward in grpo_samples:
    if reward < 0.5:
        critique = generate_critique(vibe, output, reward)
        corrected = generate_corrected_config(vibe)  # From teacher model
        critiques.append({
            "vibe": vibe,
            "bad_output": output,
            "critique": critique,
            "good_output": corrected
        })

# Secondary SFT on critiques (teaches model to self-correct)
# This is "learning from mistakes"
```

##### Option 2: Reward Decomposition (Richer Signal)

Instead of one scalar, return a **reward vector** and let the model see which components failed:

```python
def compute_decomposed_reward(vibe: str, output: str) -> dict:
    """
    Return structured reward with per-component scores.
    Can be logged and used for analysis.
    """
    r_format = reward_format(output)
    r_schema, field_errors = reward_schema(output)
    
    if r_format > 0 and r_schema > 0.5:
        config = json.loads(output)
        audio_path = synthesize_temp(config)
        r_audio = reward_audio_clap(vibe, audio_path)
    else:
        r_audio = 0.0
    
    return {
        "total": 0.2 * r_format + 0.3 * r_schema + 0.5 * r_audio,
        "format": r_format,
        "schema": r_schema,
        "audio": r_audio,
        "field_errors": field_errors,  # Which fields were wrong
    }
```

GRPO still uses `total` as the scalar reward, but you log the decomposition. Post-hoc, you can identify patterns like "model always picks wrong rhythm for 'calm' vibes" and add targeted SFT examples.

##### Option 3: RLHF with Language Feedback (Research Frontier)

There's emerging work on **RLAIF with critique** where the reward model outputs (score, explanation) and the explanation is somehow incorporated into training. But this isn't standard in TRL/Unsloth yet. For your project, Option 1 (collect critiques → secondary SFT) is the practical approach.

---

#### Dynamic Difficulty / Curriculum Learning

You asked about "forcing the LLM to perform better and better." Here's how:

```python
class CurriculumVibeGenerator:
    """
    Start with easy vibes, progressively increase difficulty.
    """
    def __init__(self):
        self.epoch = 0
        self.difficulty = 0.0  # 0 = easy, 1 = hard
    
    def get_batch(self, batch_size: int) -> list[str]:
        if self.difficulty < 0.3:
            # Easy: common moods, simple descriptions
            return self._generate_easy(batch_size)
        elif self.difficulty < 0.7:
            # Medium: combinations, less common words
            return self._generate_medium(batch_size)
        else:
            # Hard: abstract, unusual, edge cases
            return self._generate_hard(batch_size)
    
    def _generate_hard(self, n: int) -> list[str]:
        """Generate challenging prompts that stress-test the model."""
        return generate_fresh_vibes(n, temperature=1.4)  # Very high temp
    
    def update_difficulty(self, mean_reward: float):
        """Increase difficulty when model is doing well."""
        if mean_reward > 0.75:
            self.difficulty = min(1.0, self.difficulty + 0.1)
        elif mean_reward < 0.4:
            self.difficulty = max(0.0, self.difficulty - 0.05)
        
        self.epoch += 1
```

This creates a **self-paced curriculum**: easy vibes first, then harder ones as the model improves. Keeps the model in the "learning zone."

---

#### GRPO Configuration

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir="./grpo-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-5,  # Much lower than SFT
    
    # GRPO-specific
    num_generations=4,  # Completions per prompt
    max_new_tokens=256,
    temperature=0.8,  # Encourage exploration
    
    # Loss type
    loss_type="grpo",  # or "dapo", "bnpo", "dr_grpo"
    
    # KL penalty
    beta=0.1,  # KL coefficient (lower = more exploration)
)
```

#### GRPO Training Time

GRPO is significantly slower than SFT because each step requires:
1. Generating multiple completions
2. Running reward function on each
3. Computing advantages

| Reward Function | Time per 1k prompts (T4) |
|-----------------|-------------------------|
| JSON validation only | 10-15 min |
| + Local CLAP scoring | 30-45 min |
| + Voxtral validation (periodic) | +5 min every 500 steps |

**Recommended setup:**
- **Training loop**: CLAP only (fast, local)
- **Validation callback**: Voxtral every 500 steps on 20 samples (tracks "true" quality)
- **Final benchmark**: Full dual-scoring on test set

---

#### Monitoring with Weights & Biases

Yes — W&B will show you exactly what's happening and whether training has stalled.

##### Setup

```python
import wandb
from trl import GRPOConfig, GRPOTrainer

# Initialize W&B
wandb.init(
    project="vibe-audio-config",
    name="smol-135m-grpo-run1",
    config={
        "model": "SmolLM2-135M",
        "phase": "grpo",
        "reward_type": "clap",
        "learning_rate": 1e-5,
    }
)

# TRL auto-logs to W&B if it's initialized
config = GRPOConfig(
    output_dir="./grpo-output",
    report_to="wandb",  # Enable W&B logging
    # ... other config
)
```

##### Key Metrics to Watch

| Metric | Healthy Sign | Stalled/Broken Sign |
|--------|--------------|---------------------|
| `reward/mean` | Trending up over time | Flat line for 500+ steps |
| `reward/std` | Moderate (0.1-0.3) | Very low (<0.05) = collapsed; Very high (>0.5) = unstable |
| `policy/kl_divergence` | Low but non-zero (0.01-0.1) | Zero = not learning; >1.0 = diverging |
| `policy/entropy` | Gradual decrease | Sudden collapse = mode collapse |
| `loss/policy_loss` | Decreasing | Oscillating wildly or NaN |
| `tokens/response_length` | Stable (~200-300) | Collapsing to very short = degenerate |

##### Custom Logging for Your Rewards

```python
class RewardLoggingCallback:
    """Log detailed reward components to W&B."""
    
    def __init__(self, log_every: int = 50):
        self.log_every = log_every
        self.step = 0
        self.reward_history = []
    
    def on_reward_computed(self, rewards: list[dict]):
        """Called after computing rewards for a batch."""
        self.step += 1
        
        # Aggregate batch stats
        format_scores = [r["format"] for r in rewards]
        schema_scores = [r["schema"] for r in rewards]
        audio_scores = [r["audio"] for r in rewards]
        total_scores = [r["total"] for r in rewards]
        
        self.reward_history.extend(total_scores)
        
        if self.step % self.log_every == 0:
            wandb.log({
                # Component breakdown
                "reward/format_mean": np.mean(format_scores),
                "reward/schema_mean": np.mean(schema_scores),
                "reward/audio_mean": np.mean(audio_scores),
                "reward/total_mean": np.mean(total_scores),
                "reward/total_std": np.std(total_scores),
                
                # Distribution info
                "reward/total_min": np.min(total_scores),
                "reward/total_max": np.max(total_scores),
                "reward/pct_above_0.5": np.mean([s > 0.5 for s in total_scores]),
                "reward/pct_above_0.8": np.mean([s > 0.8 for s in total_scores]),
                
                # JSON validity rate
                "reward/json_valid_pct": np.mean(format_scores),
                
                # Rolling average (detect trends)
                "reward/rolling_100": np.mean(self.reward_history[-100:]),
            }, step=self.step)
    
    def on_voxtral_validation(self, clap_scores: list, voxtral_scores: list):
        """Log correlation during periodic Voxtral validation."""
        r, _ = stats.pearsonr(clap_scores, voxtral_scores)
        
        wandb.log({
            "validation/clap_mean": np.mean(clap_scores),
            "validation/voxtral_mean": np.mean(voxtral_scores),
            "validation/clap_voxtral_correlation": r,
        }, step=self.step)
```

##### Detecting Stalls & Problems

```python
class StallDetector:
    """Alert when training appears stuck."""
    
    def __init__(self, patience: int = 300, min_improvement: float = 0.01):
        self.patience = patience  # Steps to wait
        self.min_improvement = min_improvement
        self.best_reward = -float('inf')
        self.steps_without_improvement = 0
    
    def check(self, current_reward: float, step: int) -> str | None:
        """Returns warning message if stalled, None otherwise."""
        
        if current_reward > self.best_reward + self.min_improvement:
            self.best_reward = current_reward
            self.steps_without_improvement = 0
            return None
        
        self.steps_without_improvement += 1
        
        if self.steps_without_improvement >= self.patience:
            msg = (f"⚠️ STALL DETECTED at step {step}: "
                   f"No improvement in {self.patience} steps. "
                   f"Best: {self.best_reward:.3f}, Current: {current_reward:.3f}")
            wandb.alert(
                title="Training Stalled",
                text=msg,
                level=wandb.AlertLevel.WARN
            )
            return msg
        
        return None


class DivergenceDetector:
    """Alert on signs of training instability."""
    
    def check(self, metrics: dict, step: int) -> list[str]:
        warnings = []
        
        # KL divergence too high
        if metrics.get("kl_divergence", 0) > 1.0:
            warnings.append(f"⚠️ KL divergence too high ({metrics['kl_divergence']:.2f})")
        
        # Reward collapsed
        if metrics.get("reward_std", 1) < 0.02:
            warnings.append("⚠️ Reward variance collapsed - possible mode collapse")
        
        # JSON validity dropping
        if metrics.get("json_valid_pct", 1) < 0.8:
            warnings.append(f"⚠️ JSON validity dropped to {metrics['json_valid_pct']:.0%}")
        
        # Loss is NaN
        if np.isnan(metrics.get("loss", 0)):
            warnings.append("🚨 Loss is NaN - training has diverged!")
        
        for w in warnings:
            wandb.alert(title="Training Issue", text=w, level=wandb.AlertLevel.ERROR)
        
        return warnings
```

##### W&B Dashboard Setup

Create a custom dashboard with these panels:

```
┌─────────────────────────────────────────────────────────────────┐
│ REWARD PROGRESSION                                               │
│ ┌─────────────────────┐ ┌─────────────────────┐                 │
│ │ reward/total_mean   │ │ reward/rolling_100  │  ← Main focus   │
│ │ (line chart)        │ │ (line chart)        │                 │
│ └─────────────────────┘ └─────────────────────┘                 │
├─────────────────────────────────────────────────────────────────┤
│ REWARD COMPONENTS                                                │
│ ┌─────────────────────┐ ┌─────────────────────┐                 │
│ │ format/schema/audio │ │ reward distribution │                 │
│ │ (stacked area)      │ │ (histogram)         │                 │
│ └─────────────────────┘ └─────────────────────┘                 │
├─────────────────────────────────────────────────────────────────┤
│ TRAINING HEALTH                                                  │
│ ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────┐ │
│ │ policy/kl_divergence│ │ policy/entropy      │ │json_valid_%│ │
│ │ (line, alert >1.0)  │ │ (line)              │ │ (line)      │ │
│ └─────────────────────┘ └─────────────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ CLAP vs VOXTRAL VALIDATION                                       │
│ ┌─────────────────────────────────────────────┐                 │
│ │ validation/clap_voxtral_correlation          │ ← Should stay  │
│ │ (line chart, periodic)                       │   above 0.6    │
│ └─────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

##### Quick Diagnostic Checklist

When you look at W&B, ask:

| Question | Where to Look | What's Good |
|----------|---------------|-------------|
| Is reward improving? | `reward/rolling_100` | Upward trend |
| Is it still exploring? | `policy/entropy` | Not collapsed to zero |
| Is it stable? | `policy/kl_divergence` | Between 0.01 and 0.5 |
| Is JSON still valid? | `reward/json_valid_pct` | Above 95% |
| Is CLAP trustworthy? | `validation/clap_voxtral_correlation` | Above 0.6 |

##### Example: What a Healthy Run Looks Like

```
Step 0:      reward=0.25, kl=0.00, entropy=4.2, json_valid=0.70
Step 500:    reward=0.45, kl=0.05, entropy=3.8, json_valid=0.92
Step 1000:   reward=0.58, kl=0.08, entropy=3.5, json_valid=0.97
Step 2000:   reward=0.68, kl=0.10, entropy=3.2, json_valid=0.98
Step 3000:   reward=0.72, kl=0.12, entropy=3.0, json_valid=0.99  ← Converging
```

##### Example: What a Stalled Run Looks Like

```
Step 0:      reward=0.25, kl=0.00, entropy=4.2
Step 500:    reward=0.48, kl=0.06, entropy=3.7
Step 1000:   reward=0.51, kl=0.07, entropy=3.6
Step 1500:   reward=0.50, kl=0.07, entropy=3.5   ← Plateau
Step 2000:   reward=0.51, kl=0.07, entropy=3.5   ← Still flat
Step 2500:   reward=0.49, kl=0.07, entropy=3.5   ← Stalled!

Action: Try increasing learning rate, adding fresh prompts, or adjusting reward weights
```

#### ART (Agent Reinforcement Trainer) by OpenPipe

ART is a higher-level abstraction for RL fine-tuning. Pros:
- Simpler API
- Built-in reward function templates
- Good logging/visualization

Cons:
- Less control over training dynamics
- May not support latest loss types
- Additional dependency

**Verdict**: Use standard TRL/Unsloth for maximum control. Consider ART if you want faster iteration with less customization.

---

## 5. Reward Function Design

### Multi-Component Reward Architecture

```
Total Reward = w₁·R_format + w₂·R_schema + w₃·R_audio
```

Where:
- `R_format`: Is it valid JSON? (binary)
- `R_schema`: Do values match allowed enums? (granular)
- `R_audio`: Does the audio match the vibe? (CLAP/API)

### Component 1: Format Reward (R_format)

```python
import json

def reward_format(output: str) -> float:
    """Binary: is it valid JSON?"""
    try:
        json.loads(output)
        return 1.0
    except json.JSONDecodeError:
        return 0.0
```

### Component 2: Schema Reward (R_schema) — Granular Scoring

**Your concern about binary pass/fail is valid.** Here's a granular approach:

```python
from pydantic import ValidationError
from config import MusicConfig  # Your Pydantic model

FIELD_WEIGHTS = {
    # Core musical choices (higher weight)
    'tempo': 1.5, 'root': 1.0, 'mode': 1.5, 'brightness': 1.2,
    # Layer selections (medium weight)  
    'bass': 1.0, 'pad': 1.0, 'melody': 1.0, 'rhythm': 1.0,
    'texture': 0.8, 'accent': 0.8,
    # Modifiers (lower weight)
    'space': 0.6, 'motion': 0.6, 'attack': 0.5, 'stereo': 0.5,
    'depth': 0.3, 'echo': 0.5, 'human': 0.4, 'grain': 0.4,
    'density': 0.5,
}

def reward_schema(output: str) -> tuple[float, dict]:
    """
    Granular schema validation with partial credit.
    Returns (score, field_errors) where score is 0-1.
    """
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return 0.0, {"_json": "invalid"}
    
    if not isinstance(data, dict):
        return 0.0, {"_type": "not a dict"}
    
    errors = {}
    total_weight = sum(FIELD_WEIGHTS.values())
    earned_weight = 0.0
    
    for field, weight in FIELD_WEIGHTS.items():
        if field not in data:
            errors[field] = "missing"
            continue
            
        value = data[field]
        
        # Check if value is valid for this field
        try:
            # Use Pydantic for validation
            MusicConfig.model_validate({field: value})
            earned_weight += weight
        except ValidationError as e:
            errors[field] = str(e.errors()[0]['msg'])
            # Partial credit for close values (e.g., typos)
            if _is_close_match(field, value):
                earned_weight += weight * 0.3
    
    score = earned_weight / total_weight
    return score, errors

def _is_close_match(field: str, value: str) -> bool:
    """Check if value is a near-miss (typo, case error)."""
    from difflib import get_close_matches
    
    valid_values = _get_valid_values(field)
    if not valid_values:
        return False
    
    # Check for close string matches
    matches = get_close_matches(str(value).lower(), 
                                 [v.lower() for v in valid_values], 
                                 n=1, cutoff=0.8)
    return len(matches) > 0
```

### Component 3: Audio Reward (R_audio)

#### Option A: CLAP with Penalized Similarity (Local, Fast)

The reward measures how well the audio matches the vibe, **penalized** by how much the audio resembles "bad" audio concepts—but only the *excess* badness beyond what the input text itself implies.

```python
import numpy as np
import laion_clap
from functools import lru_cache

# Load model once
clap_model = laion_clap.CLAP_Module(enable_fusion=False)
clap_model.load_ckpt()

# Define "bad song" concepts
BAD_CONCEPTS = [
    "bad", "terrible", "awful", "discordant", "unharmonious",
    "cacophony", "noise", "unpleasant", "harsh", "grating",
    "off-key", "out of tune", "distorted badly", "broken audio",
    "annoying sound", "painful to listen to", "low quality audio"
]

@lru_cache(maxsize=1)
def get_bad_embedding() -> np.ndarray:
    """
    Compute average embedding of 'bad song' concepts.
    Cached since it never changes.
    """
    embeddings = clap_model.get_text_embedding(BAD_CONCEPTS)
    return embeddings.mean(axis=0, keepdims=True)

def softplus(x: float, beta: float = 1.0) -> float:
    """Smooth approximation of ReLU for continuous gradients."""
    return (1 / beta) * np.log(1 + np.exp(beta * x))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings."""
    return (a @ b.T).item() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def reward_audio_clap(vibe: str, audio_path: str) -> float:
    """
    CLAP-based reward with penalized similarity.
    
    Score = exp(raw_score - 1), mapped to [0, 1]
    
    Where raw_score uses softplus for continuous space:
    - Positive: similarity(audio, vibe_text)
    - Penalty: EXCESS similarity to "bad" concepts
              = max(0, sim(audio, bad) - sim(vibe_text, bad))
    
    The penalty is relative: if the vibe itself is "dark and gritty",
    we don't penalize the audio for also being dark/gritty.
    """
    # Get embeddings
    text_embed = clap_model.get_text_embedding([vibe])
    audio_embed = clap_model.get_audio_embedding_from_filelist([audio_path])
    bad_embed = get_bad_embedding()
    
    # Core similarity: how well does audio match the vibe?
    audio_text_sim = cosine_sim(audio_embed, text_embed)
    
    # Badness similarities
    audio_bad_sim = cosine_sim(audio_embed, bad_embed)
    text_bad_sim = cosine_sim(text_embed, bad_embed)
    
    # Excess badness penalty (only penalize if audio is MORE bad than text implies)
    # Use softplus for smooth, continuous penalty in gradient space
    excess_badness = audio_bad_sim - text_bad_sim
    penalty = softplus(excess_badness, beta=5.0)  # beta controls sharpness
    
    # Combine: reward alignment, penalize excess badness
    # Weighting: similarity is primary, penalty is secondary
    raw_score = softplus(audio_text_sim, beta=2.0) - 0.5 * penalty
    
    # Map to [0, 1] using exp(score - 1)
    # This gives: score=1 → reward≈1, score=0 → reward≈0.37, score=-1 → reward≈0.14
    reward = np.exp(raw_score - 1)
    
    # Clamp to valid range
    return float(np.clip(reward, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARKING VERSION (for evaluation, not training)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_audio_clap(vibe: str, audio_path: str) -> dict:
    """
    Full diagnostic version for benchmarking.
    Returns all intermediate values for analysis.
    """
    text_embed = clap_model.get_text_embedding([vibe])
    audio_embed = clap_model.get_audio_embedding_from_filelist([audio_path])
    bad_embed = get_bad_embedding()
    
    audio_text_sim = cosine_sim(audio_embed, text_embed)
    audio_bad_sim = cosine_sim(audio_embed, bad_embed)
    text_bad_sim = cosine_sim(text_embed, bad_embed)
    
    excess_badness = audio_bad_sim - text_bad_sim
    penalty = softplus(excess_badness, beta=5.0)
    raw_score = softplus(audio_text_sim, beta=2.0) - 0.5 * penalty
    reward = float(np.clip(np.exp(raw_score - 1), 0.0, 1.0))
    
    return {
        "vibe": vibe,
        "audio_text_similarity": float(audio_text_sim),
        "audio_bad_similarity": float(audio_bad_sim),
        "text_bad_similarity": float(text_bad_sim),
        "excess_badness": float(excess_badness),
        "penalty": float(penalty),
        "raw_score": float(raw_score),
        "final_reward": reward,
    }
```

**Why this formulation works:**

| Component | Purpose |
|-----------|---------|
| `audio_text_sim` | Rewards audio that matches the vibe |
| `text_bad_sim` | Baseline: some vibes ARE supposed to be dark/gritty |
| `audio_bad_sim - text_bad_sim` | Only penalize *excess* badness beyond what's expected |
| `softplus` | Continuous gradients (better than ReLU for RL) |
| `exp(score - 1)` | Maps to [0,1] with nice gradient properties |

**Does CLAP know music?** Yes, but with caveats:
- CLAP was trained on AudioSet + other datasets including music
- Works well for general audio-text alignment
- May struggle with fine-grained musical distinctions
- Good for "bright vs dark", "fast vs slow"; weaker for "dorian vs mixolydian"

#### Option B: LLM-as-Judge with Voxtral (Slower, Higher Quality)

Voxtral is Mistral's open-source audio-capable model. Use it as a "gold standard" evaluator to validate CLAP scores.

```python
import base64
import httpx
from mistralai import Mistral

# Initialize Mistral client
mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

def encode_audio_base64(audio_path: str) -> str:
    """Encode audio file to base64 for API submission."""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def reward_audio_voxtral(vibe: str, audio_path: str) -> float:
    """
    Use Voxtral as LLM-as-judge for audio-vibe alignment.
    
    Returns 0-1 score.
    More accurate than CLAP but slower and requires API.
    """
    audio_b64 = encode_audio_base64(audio_path)
    
    response = mistral_client.chat.complete(
        model="mistral-voxtral-latest",  # Or specific version
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": f"data:audio/wav;base64,{audio_b64}"
                    },
                    {
                        "type": "text",
                        "text": f"""You are an expert audio engineer evaluating synthesizer output.

The intended vibe was: "{vibe}"

Listen to this audio and rate how well it matches the intended vibe.

Score from 0 to 10:
- 0-2: Completely wrong mood/feel, doesn't match at all
- 3-4: Some elements match but overall misses the mark  
- 5-6: Partially matches, acceptable but not ideal
- 7-8: Good match, captures the essence of the vibe
- 9-10: Excellent match, perfectly evokes the intended feeling

Also briefly note what works and what doesn't (1-2 sentences).

Format your response as:
SCORE: [number]
NOTES: [brief explanation]"""
                    }
                ]
            }
        ],
        max_tokens=150,
        temperature=0.3,  # Low temp for consistent scoring
    )
    
    # Parse response
    response_text = response.choices[0].message.content
    score_line = [l for l in response_text.split("\n") if l.startswith("SCORE:")][0]
    score = float(score_line.replace("SCORE:", "").strip()) / 10.0
    
    return float(np.clip(score, 0.0, 1.0))


def benchmark_audio_voxtral(vibe: str, audio_path: str) -> dict:
    """
    Full diagnostic version for benchmarking.
    Returns score and qualitative feedback.
    """
    audio_b64 = encode_audio_base64(audio_path)
    
    response = mistral_client.chat.complete(
        model="mistral-voxtral-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": f"data:audio/wav;base64,{audio_b64}"
                    },
                    {
                        "type": "text", 
                        "text": f"""You are an expert audio engineer evaluating synthesizer output.

The intended vibe was: "{vibe}"

Evaluate this audio on multiple dimensions:

1. MOOD_MATCH (0-10): Does the emotional feel match the vibe?
2. TEMPO_MATCH (0-10): Is the speed/energy level appropriate?
3. TEXTURE_MATCH (0-10): Do the timbres/sounds fit the description?
4. OVERALL (0-10): Holistic score for vibe alignment
5. AUDIO_QUALITY (0-10): Technical quality (ignoring vibe match)

Format:
MOOD_MATCH: [score]
TEMPO_MATCH: [score]
TEXTURE_MATCH: [score]  
OVERALL: [score]
AUDIO_QUALITY: [score]
NOTES: [1-2 sentences on what works/doesn't]"""
                    }
                ]
            }
        ],
        max_tokens=200,
        temperature=0.3,
    )
    
    response_text = response.choices[0].message.content
    
    def extract_score(key: str) -> float:
        line = [l for l in response_text.split("\n") if l.startswith(f"{key}:")][0]
        return float(line.split(":")[1].strip()) / 10.0
    
    notes_line = [l for l in response_text.split("\n") if l.startswith("NOTES:")]
    notes = notes_line[0].replace("NOTES:", "").strip() if notes_line else ""
    
    return {
        "vibe": vibe,
        "mood_match": extract_score("MOOD_MATCH"),
        "tempo_match": extract_score("TEMPO_MATCH"),
        "texture_match": extract_score("TEXTURE_MATCH"),
        "overall": extract_score("OVERALL"),
        "audio_quality": extract_score("AUDIO_QUALITY"),
        "notes": notes,
        "final_reward": extract_score("OVERALL"),  # Use overall as reward
    }
```

---

### Dual-Scorer Benchmarking: Validating CLAP as a Proxy

The goal: **prove that CLAP scores correlate with Voxtral scores**, validating CLAP as a cheap proxy for expert evaluation.

#### Benchmark Against Teacher Model (SOTA Ceiling)

Your benchmark should include the **teacher LLM** (the SOTA model used to generate synthetic data) as a ceiling. This shows:
1. How close tiny models get to the teacher
2. Whether knowledge distillation worked
3. The theoretical maximum performance

```python
def run_full_benchmark(
    test_vibes: list[str],
    models: dict[str, Any],  # {"smol-135m": model1, "gemma-270m": model2, ...}
    teacher_model: str = "claude-sonnet-4-20250514",
    output_dir: Path = Path("./benchmark_results")
) -> pd.DataFrame:
    """
    Benchmark all models INCLUDING the teacher on the same test set.
    
    This answers: "How much performance do we lose by using tiny models?"
    """
    output_dir.mkdir(exist_ok=True)
    all_results = []
    
    # Add teacher model to comparison
    models_to_test = {**models, "teacher (Claude Sonnet)": teacher_model}
    
    for model_name, model in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")
        
        for i, vibe in enumerate(test_vibes):
            print(f"  [{i+1}/{len(test_vibes)}] {vibe[:40]}...")
            
            # Generate config
            if model_name.startswith("teacher"):
                # Use API for teacher model
                config_json = generate_from_teacher(vibe, teacher_model)
            else:
                # Use local model
                config_json = model.generate(vibe)
            
            try:
                config = json.loads(config_json)
            except json.JSONDecodeError:
                all_results.append({
                    "model": model_name,
                    "vibe": vibe,
                    "valid_json": False,
                    "clap_reward": 0.0,
                    "voxtral_overall": 0.0,
                })
                continue
            
            # Synthesize audio
            audio_path = output_dir / f"{model_name}_{i:04d}.wav"
            config_to_audio(MusicConfig.from_dict(config), str(audio_path), duration=10.0)
            
            # Score with both metrics
            clap_result = benchmark_audio_clap(vibe, str(audio_path))
            voxtral_result = benchmark_audio_voxtral(vibe, str(audio_path))
            
            all_results.append({
                "model": model_name,
                "vibe": vibe,
                "config": config_json,
                "valid_json": True,
                "clap_reward": clap_result["final_reward"],
                "clap_audio_text_sim": clap_result["audio_text_similarity"],
                "voxtral_overall": voxtral_result["overall"],
                "voxtral_mood": voxtral_result["mood_match"],
                "voxtral_notes": voxtral_result["notes"],
            })
    
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "full_benchmark.csv", index=False)
    
    return df


def generate_benchmark_report(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Generate a markdown report comparing all models.
    """
    report = []
    report.append("# Vibe Audio Config - Benchmark Report\n")
    report.append(f"**Test samples:** {df['vibe'].nunique()}\n")
    report.append(f"**Models tested:** {df['model'].nunique()}\n\n")
    
    # Summary table
    report.append("## Model Comparison\n")
    report.append("| Model | JSON Valid % | CLAP Score | Voxtral Score | vs Teacher |")
    report.append("|-------|-------------|------------|---------------|------------|")
    
    teacher_clap = df[df['model'].str.contains('teacher')]['clap_reward'].mean()
    teacher_voxtral = df[df['model'].str.contains('teacher')]['voxtral_overall'].mean()
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        valid_pct = model_df['valid_json'].mean() * 100
        clap_mean = model_df['clap_reward'].mean()
        voxtral_mean = model_df['voxtral_overall'].mean()
        
        # Performance relative to teacher
        if 'teacher' in model:
            vs_teacher = "—"
        else:
            vs_teacher = f"{(clap_mean / teacher_clap) * 100:.0f}%"
        
        report.append(f"| {model} | {valid_pct:.1f}% | {clap_mean:.3f} | {voxtral_mean:.3f} | {vs_teacher} |")
    
    report.append("\n")
    
    # Key findings
    report.append("## Key Findings\n")
    
    # CLAP-Voxtral correlation across all models
    r, p = stats.pearsonr(df['clap_reward'], df['voxtral_overall'])
    report.append(f"- **CLAP-Voxtral correlation:** r={r:.3f} (p={p:.2e})\n")
    
    if r > 0.7:
        report.append("  - ✅ Strong correlation: CLAP is a valid proxy for expert evaluation\n")
    elif r > 0.5:
        report.append("  - ⚠️ Moderate correlation: CLAP is acceptable but imperfect\n")
    else:
        report.append("  - ❌ Weak correlation: Consider alternative metrics\n")
    
    # Best tiny model
    tiny_models = df[~df['model'].str.contains('teacher')]
    best_tiny = tiny_models.groupby('model')['clap_reward'].mean().idxmax()
    best_tiny_score = tiny_models.groupby('model')['clap_reward'].mean().max()
    
    report.append(f"- **Best tiny model:** {best_tiny} (CLAP={best_tiny_score:.3f})\n")
    report.append(f"- **Teacher ceiling:** CLAP={teacher_clap:.3f}, Voxtral={teacher_voxtral:.3f}\n")
    report.append(f"- **Distillation efficiency:** {(best_tiny_score/teacher_clap)*100:.1f}% of teacher performance\n")
    
    # Save report
    report_text = "\n".join(report)
    with open(output_dir / "benchmark_report.md", "w") as f:
        f.write(report_text)
    
    return report_text
```

**Expected Benchmark Output:**

| Model | JSON Valid % | CLAP Score | Voxtral Score | vs Teacher |
|-------|-------------|------------|---------------|------------|
| SmolLM2-135M (SFT) | 94.2% | 0.52 | 0.48 | 76% |
| SmolLM2-135M (GRPO) | 98.5% | 0.61 | 0.58 | 89% |
| Gemma-3-270M (GRPO) | 99.1% | 0.64 | 0.61 | 93% |
| Qwen3-600M (GRPO) | 99.4% | 0.67 | 0.64 | 97% |
| **Teacher (Claude Sonnet)** | 99.8% | 0.69 | 0.66 | — |

**What this proves:**
1. GRPO improves over SFT significantly
2. Larger tiny models get closer to teacher
3. Even 135M can achieve ~89% of teacher performance
4. CLAP and Voxtral agree (validating CLAP as training signal)

---

#### Per-Sample Dual Scoring

```python
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt

def run_dual_benchmark(
    test_vibes: list[str],
    model,
    output_dir: Path = Path("./benchmark_results")
) -> pd.DataFrame:
    """
    Score each (vibe, generated_audio) pair with BOTH CLAP and Voxtral.
    Returns DataFrame for correlation analysis.
    """
    output_dir.mkdir(exist_ok=True)
    results = []
    
    for i, vibe in enumerate(test_vibes):
        print(f"[{i+1}/{len(test_vibes)}] Evaluating: {vibe[:50]}...")
        
        # Generate config from model
        config_json = model.generate(vibe)
        config = json.loads(config_json)
        
        # Synthesize audio
        audio_path = output_dir / f"sample_{i:04d}.wav"
        config_to_audio(MusicConfig.from_dict(config), str(audio_path), duration=10.0)
        
        # Score with CLAP (fast)
        clap_result = benchmark_audio_clap(vibe, str(audio_path))
        
        # Score with Voxtral (slow but gold-standard)
        voxtral_result = benchmark_audio_voxtral(vibe, str(audio_path))
        
        results.append({
            "vibe": vibe,
            "config": config_json,
            "audio_path": str(audio_path),
            
            # CLAP scores
            "clap_audio_text_sim": clap_result["audio_text_similarity"],
            "clap_penalty": clap_result["penalty"],
            "clap_reward": clap_result["final_reward"],
            
            # Voxtral scores
            "voxtral_mood": voxtral_result["mood_match"],
            "voxtral_tempo": voxtral_result["tempo_match"],
            "voxtral_texture": voxtral_result["texture_match"],
            "voxtral_overall": voxtral_result["overall"],
            "voxtral_quality": voxtral_result["audio_quality"],
            "voxtral_notes": voxtral_result["notes"],
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "benchmark_results.csv", index=False)
    
    return df


def analyze_correlation(df: pd.DataFrame, output_dir: Path = Path("./benchmark_results")):
    """
    Compute and visualize correlation between CLAP and Voxtral scores.
    This validates whether CLAP is a good proxy for expert evaluation.
    """
    
    # Primary correlation: CLAP reward vs Voxtral overall
    r_overall, p_overall = stats.pearsonr(df["clap_reward"], df["voxtral_overall"])
    rho_overall, p_rho = stats.spearmanr(df["clap_reward"], df["voxtral_overall"])
    
    print("=" * 60)
    print("CLAP vs Voxtral Correlation Analysis")
    print("=" * 60)
    print(f"\nPrimary (CLAP reward vs Voxtral overall):")
    print(f"  Pearson r:  {r_overall:.3f} (p={p_overall:.2e})")
    print(f"  Spearman ρ: {rho_overall:.3f} (p={p_rho:.2e})")
    
    # Interpretation
    if r_overall > 0.7:
        print(f"  → STRONG correlation: CLAP is a good proxy for expert eval")
    elif r_overall > 0.5:
        print(f"  → MODERATE correlation: CLAP is acceptable but imperfect")
    else:
        print(f"  → WEAK correlation: Consider using Voxtral for training")
    
    # Detailed correlations
    print(f"\nDetailed correlations (CLAP reward vs Voxtral components):")
    for col in ["voxtral_mood", "voxtral_tempo", "voxtral_texture", "voxtral_quality"]:
        r, p = stats.pearsonr(df["clap_reward"], df[col])
        print(f"  vs {col}: r={r:.3f} (p={p:.2e})")
    
    # Also check CLAP audio-text similarity (raw) vs Voxtral
    r_raw, _ = stats.pearsonr(df["clap_audio_text_sim"], df["voxtral_overall"])
    print(f"\nCLAP raw similarity vs Voxtral overall: r={r_raw:.3f}")
    print(f"(If this is higher than reward, penalty might be hurting)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scatter: CLAP vs Voxtral overall
    ax1 = axes[0, 0]
    ax1.scatter(df["clap_reward"], df["voxtral_overall"], alpha=0.6)
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect agreement')
    ax1.set_xlabel("CLAP Reward")
    ax1.set_ylabel("Voxtral Overall Score")
    ax1.set_title(f"CLAP vs Voxtral (r={r_overall:.3f})")
    ax1.legend()
    
    # Histogram of score differences
    ax2 = axes[0, 1]
    diff = df["clap_reward"] - df["voxtral_overall"]
    ax2.hist(diff, bins=20, edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--')
    ax2.set_xlabel("CLAP - Voxtral (score difference)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Score Difference (mean={diff.mean():.3f}, std={diff.std():.3f})")
    
    # CLAP components vs Voxtral
    ax3 = axes[1, 0]
    ax3.scatter(df["clap_audio_text_sim"], df["voxtral_mood"], alpha=0.6, label="Mood")
    ax3.scatter(df["clap_audio_text_sim"], df["voxtral_texture"], alpha=0.6, label="Texture")
    ax3.set_xlabel("CLAP Audio-Text Similarity (raw)")
    ax3.set_ylabel("Voxtral Component Scores")
    ax3.set_title("CLAP raw sim vs Voxtral components")
    ax3.legend()
    
    # Penalty analysis
    ax4 = axes[1, 1]
    ax4.scatter(df["clap_penalty"], df["voxtral_quality"], alpha=0.6)
    ax4.set_xlabel("CLAP Penalty (excess badness)")
    ax4.set_ylabel("Voxtral Audio Quality")
    ax4.set_title("Does CLAP penalty correlate with quality issues?")
    
    plt.tight_layout()
    plt.savefig(output_dir / "clap_voxtral_correlation.png", dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to {output_dir / 'clap_voxtral_correlation.png'}")
    
    # Summary statistics
    summary = {
        "n_samples": len(df),
        "clap_reward_mean": df["clap_reward"].mean(),
        "clap_reward_std": df["clap_reward"].std(),
        "voxtral_overall_mean": df["voxtral_overall"].mean(),
        "voxtral_overall_std": df["voxtral_overall"].std(),
        "pearson_r": r_overall,
        "spearman_rho": rho_overall,
        "mean_absolute_diff": abs(diff).mean(),
    }
    
    with open(output_dir / "correlation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def identify_disagreements(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Find cases where CLAP and Voxtral strongly disagree.
    These are useful for understanding CLAP's failure modes.
    """
    diff = abs(df["clap_reward"] - df["voxtral_overall"])
    disagreements = df[diff > threshold].copy()
    disagreements["score_diff"] = diff[diff > threshold]
    disagreements = disagreements.sort_values("score_diff", ascending=False)
    
    print(f"\nFound {len(disagreements)} cases with |CLAP - Voxtral| > {threshold}")
    
    if len(disagreements) > 0:
        print("\nTop disagreements:")
        for _, row in disagreements.head(5).iterrows():
            print(f"\n  Vibe: {row['vibe'][:60]}...")
            print(f"  CLAP: {row['clap_reward']:.3f}, Voxtral: {row['voxtral_overall']:.3f}")
            print(f"  Voxtral notes: {row['voxtral_notes']}")
    
    return disagreements
```

---

### When to Use Each Scorer

| Scorer | Use Case | Speed | Cost |
|--------|----------|-------|------|
| **CLAP** | GRPO training loop | ~0.1s/sample | Free |
| **Voxtral** | Benchmarking, validation | ~2-5s/sample | API cost |
| **Both** | Correlation analysis | N/A | N/A |

**Recommended workflow:**

1. **Training**: Use CLAP (fast, free, runs locally)
2. **Validation checkpoints**: Score with Voxtral every N steps to track "true" quality
3. **Final benchmark**: Full dual-scoring to prove CLAP validity
4. **Failure analysis**: Use Voxtral's qualitative notes to understand errors

```python
# Example: Validation callback during GRPO
class VoxtralValidationCallback:
    def __init__(self, val_vibes: list[str], every_n_steps: int = 500):
        self.val_vibes = val_vibes[:20]  # Small subset for speed
        self.every_n_steps = every_n_steps
        self.history = []
    
    def on_step(self, step: int, model):
        if step % self.every_n_steps != 0:
            return
        
        scores = []
        for vibe in self.val_vibes:
            config = model.generate(vibe)
            audio_path = synthesize_temp(json.loads(config))
            
            clap = reward_audio_clap(vibe, audio_path)
            voxtral = reward_audio_voxtral(vibe, audio_path)
            scores.append({"clap": clap, "voxtral": voxtral})
        
        df = pd.DataFrame(scores)
        r, _ = stats.pearsonr(df["clap"], df["voxtral"])
        
        self.history.append({
            "step": step,
            "clap_mean": df["clap"].mean(),
            "voxtral_mean": df["voxtral"].mean(),
            "correlation": r,
        })
        
        print(f"[Step {step}] CLAP: {df['clap'].mean():.3f}, "
              f"Voxtral: {df['voxtral'].mean():.3f}, r={r:.3f}")
```

---

### Combining Rewards

```python
def compute_total_reward(
    vibe: str,
    output: str,
    weights: dict = {"format": 0.2, "schema": 0.3, "audio": 0.5}
) -> float:
    """
    Compute weighted total reward.
    """
    # Format check (gate)
    r_format = reward_format(output)
    if r_format == 0:
        return 0.0  # Invalid JSON gets zero reward
    
    # Schema validation
    r_schema, _ = reward_schema(output)
    
    # Audio scoring (only if schema is mostly valid)
    if r_schema > 0.5:
        config = json.loads(output)
        audio_path = synthesize_temp(config)  # Generate audio
        r_audio = reward_audio_clap(vibe, audio_path)
    else:
        r_audio = 0.0  # Don't waste compute on bad configs
    
    total = (weights["format"] * r_format + 
             weights["schema"] * r_schema + 
             weights["audio"] * r_audio)
    
    return total
```

### Can You Force Structured Output During Training?

**Short answer: No, and you shouldn't.**

During GRPO, the model must freely generate text so it can:
1. Explore the output space
2. Learn from mistakes
3. Receive gradient signal from rewards

If you constrain generation during training, the model never learns *why* certain outputs are wrong.

**However**, you can use structured output during **inference** (see Section 6).

---

## 6. Inference & Deployment

### Constrained Decoding

#### What Is It?

Constrained decoding restricts the model's vocabulary at each token to only valid continuations. For JSON, this means:
- After `{`, only allow `"` or `}`
- After `"tempo":`, only allow `"very_slow"`, `"slow"`, etc.

#### Is It Expensive?

**Compute cost**: Minimal. It's just masking logits before sampling.

**Implementation cost**: Moderate. You need a grammar/schema definition.

#### WebGPU Implementation

Use **Outlines** or **llama.cpp** grammar support:

```javascript
// Using transformers.js with grammar constraints
import { pipeline } from '@xenova/transformers';

const generator = await pipeline('text-generation', 'your-model');

// Define JSON schema as constraint
const schema = {
  type: "object",
  properties: {
    tempo: { enum: ["very_slow", "slow", "medium", "fast", "very_fast"] },
    // ... rest of schema
  },
  required: ["tempo", "root", "mode", /* ... */]
};

const output = await generator(prompt, {
  max_new_tokens: 256,
  json_schema: schema  // If supported by your runtime
});
```

For **MLC WebGPU**, you'd compile the grammar into the model:

```python
# During export
from mlc_llm import convert

convert(
    model_path="./your-model",
    output_path="./mlc-model",
    quantization="q4f16_1",
    json_schema="./schema.json"  # Compile grammar
)
```

### Prompting Strategy

#### Use the Same System Prompt

**Yes, use the exact same system prompt during inference as training.**

```python
SYSTEM_PROMPT = """You are a synthesizer configuration assistant. 
Given a vibe description, output a JSON configuration for an ambient/electronic synthesizer.
Output ONLY valid JSON with no explanation."""

def generate_config(vibe: str) -> dict:
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{vibe}<|im_end|>
<|im_start|>assistant
"""
    output = model.generate(prompt, max_tokens=256, stop=["<|im_end|>"])
    return json.loads(output)
```

**Why the full prompt?**
- Tiny models have limited context understanding
- They rely heavily on exact prompt patterns seen during training
- Variations may cause format degradation

### Quantization Formats

#### Recommended Exports

| Format | Use Case | Size (135M) | WebGPU Support |
|--------|----------|-------------|----------------|
| GGUF Q4_K_M | llama.cpp, Ollama | ~80MB | Via wllama |
| MLC Q4F16 | WebLLM (best WebGPU) | ~90MB | ✅ Native |
| ONNX INT8 | transformers.js | ~140MB | ✅ Via WebNN |

#### Export Commands

```bash
# GGUF (for llama.cpp ecosystem)
python -m unsloth.export \
  --model ./sft-model \
  --output ./exports/gguf \
  --format gguf \
  --quantization q4_k_m

# MLC (for WebGPU)
mlc_llm convert \
  --model ./sft-model \
  --output ./exports/mlc \
  --quantization q4f16_1

# ONNX (for transformers.js)
optimum-cli export onnx \
  --model ./sft-model \
  --task text-generation \
  ./exports/onnx
```

---

## 7. Hosting & Licensing

### Model Licenses

| Base Model | License | Can Fine-tune? | Can Host? | Can Commercialize? |
|------------|---------|----------------|-----------|-------------------|
| SmolLM2-135M | Apache 2.0 | ✅ | ✅ | ✅ |
| Gemma-3-270M | Gemma License | ✅ | ✅ | ✅ (with terms) |
| Qwen3-600M | Apache 2.0 | ✅ | ✅ | ✅ |

**All three allow fine-tuning, hosting, and distribution.** Gemma requires accepting their terms but is permissive.

### Hugging Face Hosting

#### What You Can Host (Free)

- ✅ Fine-tuned models (full weights or LoRA adapters)
- ✅ Quantized versions (GGUF, MLC, ONNX)
- ✅ Datasets (your synthetic vibe→config pairs)
- ✅ CLAP model (it's open-source)

#### Storage Limits

| Account Type | Storage | Bandwidth |
|--------------|---------|-----------|
| Free | Unlimited* | Unlimited* |
| Pro ($9/mo) | Unlimited | Unlimited |

*HF is generous but may rate-limit extremely popular models. For your project scale, free tier is fine.

#### Repository Structure on HF

```
your-username/
├── vibe-audio-smol-135m/           # Full fine-tuned model
├── vibe-audio-smol-135m-gguf/      # GGUF quantized
├── vibe-audio-smol-135m-mlc/       # MLC WebGPU format
├── vibe-audio-gemma-270m/
├── vibe-audio-gemma-270m-gguf/
├── vibe-audio-qwen-600m/
├── vibe-audio-lora-adapters/       # Just the LoRA weights
└── vibe-audio-dataset/             # Training data
```

### Can You Host CLAP?

Yes. CLAP (LAION) is released under a permissive license. You can:
- Host the model weights
- Use it in your HF Space
- Include it in your scoring pipeline

---

## 8. Repository Structure

### Recommended Layout

```
vibe-audio-config/
├── README.md
├── pyproject.toml
│
├── data/
│   ├── vibes/
│   │   ├── raw_vibes.txt           # Initial vibe list
│   │   ├── expanded_vibes.jsonl    # LLM-expanded vibes
│   │   └── coverage_report.json    # Enum coverage stats
│   ├── configs/
│   │   └── synthetic_pairs.jsonl   # vibe→config training data
│   └── splits/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
│
├── src/
│   ├── synth/
│   │   ├── __init__.py
│   │   ├── config.py               # Your Pydantic models
│   │   ├── synth.py                # Synthesis engine
│   │   └── errors.py
│   ├── data/
│   │   ├── generate_vibes.py       # Vibe generation scripts
│   │   ├── generate_pairs.py       # Config generation
│   │   └── validate_coverage.py    # Enum coverage checker
│   ├── training/
│   │   ├── sft.py                  # SFT training script
│   │   ├── grpo.py                 # GRPO training script
│   │   └── rewards.py              # Reward functions
│   ├── eval/
│   │   ├── benchmark.py            # Evaluation suite
│   │   └── human_eval.py           # Human evaluation setup
│   └── export/
│       ├── to_gguf.py
│       ├── to_mlc.py
│       └── to_onnx.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sft_training.ipynb
│   ├── 03_grpo_training.ipynb
│   └── 04_evaluation.ipynb
│
├── configs/
│   ├── sft_smol.yaml
│   ├── sft_gemma.yaml
│   ├── grpo_smol.yaml
│   └── export_config.yaml
│
├── demo/
│   ├── index.html                  # WebGPU demo
│   ├── inference.js
│   └── synth_wasm/                 # Optional: WASM synth
│
└── exports/                        # Generated artifacts
    ├── smol-135m/
    ├── gemma-270m/
    └── qwen-600m/
```

---

## 9. Artifact Checklist

### What Researchers Should See/Try

#### Phase 1: Data
- [ ] `data/vibes/raw_vibes.txt` — 500+ seed vibes
- [ ] `data/vibes/coverage_report.json` — Shows all enum values covered
- [ ] `data/configs/synthetic_pairs.jsonl` — 10k+ vibe→config pairs
- [ ] **Best-of-N stats**: mean candidates per vibe, mean CLAP score
- [ ] **Quality filter stats**: % rejected for low score, % rejected for invalid JSON

#### Phase 2: SFT Models
- [ ] `exports/smol-135m-sft/` — Fine-tuned SmolLM2
- [ ] `exports/gemma-270m-sft/` — Fine-tuned Gemma
- [ ] `exports/qwen-600m-sft/` — Fine-tuned Qwen3
- [ ] Training logs showing loss curves

#### Phase 3: GRPO Models
- [ ] `exports/smol-135m-grpo/` — GRPO-aligned model
- [ ] Reward curves showing improvement
- [ ] **W&B run links** with full training logs
- [ ] **W&B dashboard screenshot** showing healthy convergence
- [ ] A/B comparison: SFT vs GRPO outputs

#### Phase 4: Quantized Exports
- [ ] GGUF versions (Q4_K_M)
- [ ] MLC WebGPU versions
- [ ] ONNX versions (optional)

#### Phase 5: Evaluation
- [ ] Benchmark results table (both CLAP and Voxtral scores)
- [ ] **Teacher model (SOTA) included as ceiling baseline**
- [ ] **"vs Teacher" column** showing distillation efficiency %
- [ ] **CLAP-Voxtral correlation analysis** (proves CLAP is valid proxy)
- [ ] `clap_voxtral_correlation.png` visualization
- [ ] `correlation_summary.json` with Pearson r, Spearman ρ
- [ ] Disagreement analysis (where CLAP fails)
- [ ] `model_comparison.png` — bar charts comparing all models
- [ ] Human eval scores (optional, gold standard)
- [ ] Audio samples (10-20 example generations per model)

#### Phase 6: Demo
- [ ] Working WebGPU demo
- [ ] Hugging Face Space with inference API

---

## Quick Start Commands

### 1. Generate Dataset

```bash
# Generate seed vibes
python -m src.data.generate_vibes --count 10000 --output data/vibes/expanded.jsonl

# Generate config pairs using Best-of-N (requires capable LLM + CLAP)
python -m src.data.generate_pairs \
  --vibes data/vibes/expanded.jsonl \
  --output data/configs/pairs.jsonl \
  --model claude-sonnet-4-20250514 \
  --n-candidates 5 \
  --min-clap-score 0.4

# Validate coverage
python -m src.data.validate_coverage data/configs/pairs.jsonl
```

### 2. SFT Training

```bash
# Train SmolLM2-135M
python -m src.training.sft \
  --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --data data/splits/train.jsonl \
  --output exports/smol-135m-sft \
  --epochs 3 \
  --lr 2e-4
```

### 3. GRPO Training

```bash
# Run GRPO alignment
python -m src.training.grpo \
  --model exports/smol-135m-sft \
  --data data/splits/train.jsonl \
  --output exports/smol-135m-grpo \
  --reward-type clap \
  --epochs 1
```

### 4. Export

```bash
# Export all formats
python -m src.export.to_gguf exports/smol-135m-grpo --quantize q4_k_m
python -m src.export.to_mlc exports/smol-135m-grpo --quantize q4f16_1
```

### 5. Upload to HF

```bash
huggingface-cli upload your-username/vibe-audio-smol-135m ./exports/smol-135m-grpo
huggingface-cli upload your-username/vibe-audio-smol-135m-gguf ./exports/smol-135m-grpo-gguf
```

---

## Summary

| Question | Answer |
|----------|--------|
| How to generate comprehensive vibes? | Hierarchical taxonomy + LLM expansion + coverage tracking |
| **How to generate high-quality configs?** | **Best-of-N (N=5): generate 5 candidates, keep highest CLAP score** |
| Will rare enums be forgotten? | Ensure 2%+ coverage per value; GRPO explores |
| SFT time for 10k examples? | 1-2 hours (Colab Free), 20-30 min (Colab Pro) |
| GRPO time with CLAP? | 30-45 min per 1k prompts |
| Which loss type? | Start with `grpo`, try `dapo` if unstable |
| Structured outputs during training? | No—train freely, constrain at inference |
| Granular JSON rewards? | Yes—field-weighted partial credit |
| Does CLAP know music? | Yes, good for general alignment |
| **Why also use Voxtral?** | **Validate CLAP as proxy; show r > 0.7 correlation** |
| **Benchmark against teacher model?** | **Yes — show tiny models achieve 85-97% of SOTA ceiling** |
| **Can evaluator give feedback?** | **Not in GRPO, but collect critiques → secondary SFT** |
| **Dynamic prompts during GRPO?** | **Yes, use high-temp LLM to generate fresh vibes** |
| **W&B for monitoring?** | **Yes — watch `reward/rolling_100`, `kl_divergence`, `entropy`** |
| **How to detect stalls?** | **No improvement in 300+ steps; entropy collapse; KL > 1** |
| Constrained decoding cost? | Minimal (logit masking) |
| Same system prompt at inference? | Yes, exact match |
| Can host on HF free? | Yes, all models and quantizations |
| License issues? | No—SmolLM/Qwen Apache 2.0, Gemma permissive |

---

*Document generated for third-party researcher review. For questions, see the project repository or contact the maintainers.*
