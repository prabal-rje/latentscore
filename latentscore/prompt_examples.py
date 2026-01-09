from __future__ import annotations

FEW_SHOT_EXAMPLES = """
**Example 1**
Input: "Rainy day in Tokyo - Neon lights, wet pavement, melancholy but beautiful."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Rain/Wet = constant texture, fluid motion, washing away. Neon = bright highlights against dark background, synthetic hum. Melancholy = minor key, emotional weight. Tokyo/Urban = mechanical rhythm but distant, solitude in a crowd. SONIC TRANSLATION: Rain \u2192 'vinyl_crackle' texture (sounds like rain) + 'shimmer' pad (wet reflections). Neon \u2192 'bright' filter but 'dark' root (A or D#). Melancholy \u2192 'dorian' mode (urban sadness), 'contemplative' melody. Urban \u2192 'electronic' rhythm but 'soft' attack (distant traffic). COHERENCE CHECK: Needs to feel damp and lonely. 'Echo' set to medium/heavy to simulate alleyway reflections. 'Grain' warm to soften the digital neon. REFERENCE ANCHORING: Blade Runner blues but softer. Focus on the 'wet' texture and 'solitary' melody.",
  "config": {
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
    "melody_density": 0.3,
    "syncopation": 0.2,
    "swing": 0.0,
    "tension_curve": "waves",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": 0.5,
    "step_bias": 0.9,
    "chromatic_prob": 0.05,
    "cadence_strength": 0.3,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "jazz",
    "chord_change_bars": 2
  }
}
```


**Example 2**
Input: "Ancient temple at sunrise - Sacred, peaceful, timeless."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Sunrise = gradual brightening, awakening, hope. Ancient = stone, dust, weight of history. Sacred = drone, stillness, modal harmony. Timeless = no strict grid, fluid time. SONIC TRANSLATION: Sunrise \u2192 'rising' melody contour, 'swelling' attack. Ancient \u2192 'drone' bass (pedal point), 'mixolydian' mode (spiritual/positive). Sacred \u2192 'bells' accent (ritual), 'breath' texture (life/spirit). Timeless \u2192 'none' rhythm, 'very_slow' tempo. COHERENCE CHECK: Must avoid mechanical beats. Reverb should be 'vast' to simulate open air or large stone structures. 'Clean' grain for morning clarity. REFERENCE ANCHORING: Himalayan meditation bowls meeting a cinematic sunrise reveal.",
  "config": {
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
    "melody_density": 0.15,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.9,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.3,
    "register_min_oct": 5,
    "register_max_oct": 7,
    "harmony_style": "ambient",
    "chord_change_bars": 4
  }
}
```


**Example 3**
Input: "Cyberpunk chase scene - Adrenaline, neon, danger, synthetic."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Adrenaline = high speed, heart rate. Neon = synthetic textures, bright buzz. Danger = dissonance, minor/phrygian feel, aggression. Synthetic = robotic timing, saw waves. SONIC TRANSLATION: Adrenaline \u2192 'fast' tempo, 'arp_bass' (driving 16ths). Neon \u2192 'gritty' grain (distortion), 'shimmer' texture (digital rain). Danger \u2192 'minor' mode, 'chromatic' melody notes. Synthetic \u2192 'robotic' humanization, 'electronic' rhythm. COHERENCE CHECK: Needs to feel tight and aggressive. 'Stereo' wide for cinematic immersion but 'dry' enough to feel punchy. REFERENCE ANCHORING: Blade Runner 2049 motorcycle chase. High tension, machine-like.",
  "config": {
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
    "melody_density": 0.7,
    "syncopation": 0.8,
    "swing": 0.0,
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.4,
    "chromatic_prob": 0.25,
    "cadence_strength": 0.9,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "cinematic",
    "chord_change_bars": 1
  }
}
```


**Example 4**
Input: "Grandmother's kitchen - Warm, nostalgic, Sunday morning, cookies baking."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Warm = low mids, acoustic textures, analog warmth. Nostalgic = simple melody, perhaps slightly detuned/wobbly (memory). Sunday morning = relaxed, unhurried, gentle. Cookies = sweet, domestic. SONIC TRANSLATION: Warm \u2192 'warm' grain, 'warm_slow' pad, 'f' root (pastoral). Nostalgic \u2192 'vinyl_crackle' (old radio), 'melody_minimal'. Sunday \u2192 'medium' tempo (easygoing), 'soft_four' rhythm (gentle foot tap). COHERENCE CHECK: Must not be too electronic. Use 'piano_note' accent for domestic feel. 'Pop' harmony for familiar, comforting progressions. REFERENCE ANCHORING: Lo-fi study beats meets a lullaby. Cozy and safe.",
  "config": {
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
    "melody_density": 0.3,
    "syncopation": 0.2,
    "swing": 0.5,
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.9,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.9,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "pop",
    "chord_change_bars": 1
  }
}
```


**Example 5**
Input: "Deep space exploration - Vast emptiness, wonder, isolation, stars."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Vast emptiness = silence between notes, massive reverb. Wonder = major/mixolydian hints within minor context, high-register twinkles. Isolation = cold brightness, lonely melody. Stars = crystalline high frequencies. SONIC TRANSLATION: Vast \u2192 'vast' space, 'infinite' echo, 'ultra_wide' stereo. Wonder \u2192 'stars' texture (random high sine blips). Isolation \u2192 'drone' bass (static void), 'thin_high' pad (icy). Stars \u2192 'accent_bells'. COHERENCE CHECK: Tempo must be 'very_slow'. No rhythmic grid. 'Procedural' melody with 'wide' register to simulate distance. REFERENCE ANCHORING: Brian Eno's Apollo or Interstellar docking scene. Cold but beautiful.",
  "config": {
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
    "attack": "soft",
    "stereo": "ultra_wide",
    "depth": true,
    "echo": "infinite",
    "human": "natural",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": 0.15,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "arc",
    "chord_extensions": "lush",
    "motif_repeat_prob": 0.2,
    "step_bias": 0.4,
    "chromatic_prob": 0.05,
    "cadence_strength": 0.3,
    "register_min_oct": 5,
    "register_max_oct": 7,
    "harmony_style": "ambient",
    "chord_change_bars": 4
  }
}
```


**Example 6**
Input: "1920s speakeasy - Jazz, smoke, secrets, prohibition glamour."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Jazz = swing feel, walking bass, extensions (7ths/9ths). Smoke = hazy texture, muffled highs. Secrets = minor/dorian, low volume. Glamour = accents of brass or piano. SONIC TRANSLATION: Jazz \u2192 'walking' bass, 'brush' rhythm, 'swing' setting heavy. Smoke \u2192 'dark' brightness, 'vinyl_crackle' (vintage feel). Secrets \u2192 'dorian' mode. Glamour \u2192 'lush' chord extensions. COHERENCE CHECK: Use 'loose' or 'drunk' humanization to simulate live band feel. Melody should be 'ornamental' or 'procedural' with syncopation. REFERENCE ANCHORING: Noir detective film soundtrack. A smoky basement club.",
  "config": {
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
    "melody_density": 0.5,
    "syncopation": 0.8,
    "swing": 0.8,
    "tension_curve": "waves",
    "chord_extensions": "lush",
    "motif_repeat_prob": 0.5,
    "step_bias": 0.4,
    "chromatic_prob": 0.12,
    "cadence_strength": 0.6,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "jazz",
    "chord_change_bars": 1
  }
}
```


**Example 7**
Input: "Horror movie - Creeping dread, something is wrong, building terror."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Dread = heavy sub-bass, slow movement. Something wrong = dissonance, chromaticism, detuning. Terror = unexpected accents, sharp attacks, chaotic motion. SONIC TRANSLATION: Dread \u2192 'sub_pulse' bass, 'dark_sustained' pad. Wrong \u2192 'chromatic' heavy melody, 'glitch' texture. Terror \u2192 'accent_blip_random' (unpredictable), 'chaotic' motion. COHERENCE CHECK: 'Minor' mode is obvious, but 'depth' is crucial for physical fear response. 'Grain' gritty for discomfort. Tempo 'very_slow' to prolong suspense. REFERENCE ANCHORING: John Carpenter soundtrack meets silent hill ambient. Unsettling.",
  "config": {
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
    "melody_density": 0.15,
    "syncopation": 0.2,
    "swing": 0.0,
    "tension_curve": "ramp",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": 0.2,
    "step_bias": 0.9,
    "chromatic_prob": 0.25,
    "cadence_strength": 0.3,
    "register_min_oct": 3,
    "register_max_oct": 5,
    "harmony_style": "cinematic",
    "chord_change_bars": 4
  }
}
```


**Example 8**
Input: "Victory celebration - Triumph, euphoria, champagne popping."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Triumph = major key, ascending melodies, brassy tones. Euphoria = fast tempo, high energy, bright textures. Champagne = sparkle, crisp transients. SONIC TRANSLATION: Triumph \u2192 'heroic' melody (rising), 'major' mode. Euphoria \u2192 'fast' tempo, 'military' rhythm (snare rolls/drive). Champagne \u2192 'shimmer' texture, 'bright' brightness. COHERENCE CHECK: Needs 'sharp' attack for energetic articulation. 'Stereo' wide to feel big/stadium-like. Harmony 'pop' for satisfying resolutions. REFERENCE ANCHORING: Rocky theme or Final Fantasy victory fanfare. Unabashedly positive.",
  "config": {
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
    "melody_density": 0.7,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.4,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.9,
    "register_min_oct": 5,
    "register_max_oct": 7,
    "harmony_style": "pop",
    "chord_change_bars": 1
  }
}
```


**Example 9**
Input: "Underwater coral reef - Colorful, teeming with life, flowing, dreamlike."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Colorful = extended harmony (lush), distinct layers. Teeming = busy but light density (arp). Flowing = slow LFO motion, liquid filters. Dreamlike = delayed, reverb-washed, soft attacks. SONIC TRANSLATION: Colorful \u2192 'lush' harmony, 'blip' accents (bubbles). Flowing \u2192 'medium' motion, 'pad_whisper' texture. Dreamlike \u2192 'mixolydian' (aquatic wonder), 'heavy' echo. COHERENCE CHECK: 'Muffled/dark' brightness would be realistic for underwater, but 'coral reef' implies light/color, so 'medium' brightness with 'warm' grain. 'Arp_melody' simulates schools of fish. REFERENCE ANCHORING: Finding Nemo score. Bubbling synthesizers.",
  "config": {
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
    "melody_density": 0.5,
    "syncopation": 0.2,
    "swing": 0.0,
    "tension_curve": "waves",
    "chord_extensions": "lush",
    "motif_repeat_prob": 0.5,
    "step_bias": 0.7,
    "chromatic_prob": 0.05,
    "cadence_strength": 0.3,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "ambient",
    "chord_change_bars": 2
  }
}
```


**Example 10**
Input: "Rage and destruction - Anger, smashing things, uncontrolled fury."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Anger = minor key, loud, distortion. Smashing = sharp transients, percussive chaos. Uncontrolled = chaotic motion, loose timing, high density. Fury = fast tempo, relentless. SONIC TRANSLATION: Anger \u2192 'gritty' grain, 'minor' mode. Smashing \u2192 'kit_medium' or 'electronic' rhythm (heavy), 'accent_blip_random' (debris). Uncontrolled \u2192 'chaotic' motion, 'loose' humanization. Fury \u2192 'fast' tempo, 'arp_bass' (driving). COHERENCE CHECK: Needs to feel broken. 'Glitch' texture adds digital destruction. 'Depth' true for impact. REFERENCE ANCHORING: Industrial metal breakdown or Doom soundtrack.",
  "config": {
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
    "melody_density": 0.7,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.9,
    "chromatic_prob": 0.25,
    "cadence_strength": 0.9,
    "register_min_oct": 3,
    "register_max_oct": 5,
    "harmony_style": "cinematic",
    "chord_change_bars": 1
  }
}
```


**Example 11**
Input: "Cozy reading nook - Fireplace crackling, rain outside, warm blanket."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Fireplace = crackle texture, warmth. Rain outside = distant white noise/shimmer. Warm blanket = low pass filter, soft attack, sustained comfort. SONIC TRANSLATION: Fireplace \u2192 'vinyl_crackle' (perfect analog). Rain \u2192 'shimmer_slow' (distant water). Warmth \u2192 'warm_slow' pad, 'very_dark' brightness. Blanket \u2192 'drone' bass (enveloping). COHERENCE CHECK: Very similar to 'Grandmother's kitchen' but darker, more solitary. 'Tempo' very slow. 'Harmony' ambient/jazz. REFERENCE ANCHORING: ASMR ambience videos. Pure texture and comfort.",
  "config": {
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
    "melody_density": 0.15,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "arc",
    "chord_extensions": "lush",
    "motif_repeat_prob": 0.5,
    "step_bias": 0.9,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.3,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "jazz",
    "chord_change_bars": 4
  }
}
```


**Example 12**
Input: "Carnival at night - Manic energy, flashing lights, sugar rush, chaos."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Manic energy = fast tempo, busy density. Flashing lights = arp accents, bright shifting filters. Sugar rush = high pitch, major/mixolydian (saccharine). Chaos = polyrhythms, clashing textures. SONIC TRANSLATION: Manic \u2192 'very_fast' tempo, 'busy' melody density. Flashing \u2192 'shimmer' texture, 'blip_random' accents. Sugar \u2192 'major' mode, 'bright' brightness. Chaos \u2192 'chaotic' motion, 'loose' timing. COHERENCE CHECK: Needs to sound slightly overwhelming. 'Kit_light' to simulate mechanical rides. 'Arp_melody' for the carousel feel. REFERENCE ANCHORING: A circus march played at double speed on a synthesizer.",
  "config": {
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
    "melody_density": 0.85,
    "syncopation": 0.8,
    "swing": 0.8,
    "tension_curve": "waves",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.4,
    "chromatic_prob": 0.12,
    "cadence_strength": 0.9,
    "register_min_oct": 5,
    "register_max_oct": 7,
    "harmony_style": "pop",
    "chord_change_bars": 1
  }
}
```


**Example 13**
Input: "Meditation retreat - Mindfulness, breath, letting go, inner peace."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Mindfulness = present moment, static, no distractions. Breath = rhythmic swelling, organic texture. Letting go = long release times, fading echoes. Inner peace = harmonic stasis, drone. SONIC TRANSLATION: Mindfulness \u2192 'very_slow' tempo, 'minimal' density. Breath \u2192 'breath' texture, 'slow' motion (inhale/exhale LFO). Letting go \u2192 'infinite' echo, 'soft' attack. Peace \u2192 'drone' bass, 'ambient' harmony. COHERENCE CHECK: Absolute prohibition on drums ('rhythm_none'). 'Dorian' or 'Mixolydian' for that neutral-spiritual feel. REFERENCE ANCHORING: Guided meditation background tracks. Singing bowls.",
  "config": {
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
    "melody_density": 0.15,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.2,
    "step_bias": 0.9,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.3,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "ambient",
    "chord_change_bars": 4
  }
}
```


**Example 14**
Input: "First day of summer - Freedom, sunshine, endless possibility, youth."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Freedom = open chords, forward momentum. Sunshine = bright frequencies, warm grain. Possibility = rising melody, major key. Youth = energetic rhythm, pop sensibility. SONIC TRANSLATION: Freedom \u2192 'bright_open' pad, 'pop' harmony. Sunshine \u2192 'very_bright' brightness, 'shimmer' texture. Possibility \u2192 'rising' melody. Youth \u2192 'fast' tempo, 'kit_medium' rhythm. COHERENCE CHECK: 'Major' mode is essential. 'Clean' or 'warm' grain. Avoid complex jazz chords; stick to 'triads' or 'sevenths' for direct emotional punch. REFERENCE ANCHORING: Katy Perry 'Teenage Dream' or upbeat indie pop.",
  "config": {
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
    "melody_density": 0.5,
    "syncopation": 0.2,
    "swing": 0.0,
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.4,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.9,
    "register_min_oct": 5,
    "register_max_oct": 7,
    "harmony_style": "pop",
    "chord_change_bars": 1
  }
}
```


**Example 15**
Input: "Abandoned asylum - Decay, echoing footsteps, history of pain, ghosts."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Decay = detuned, gritty, falling apart. Echoing footsteps = heavy reverb, sparse impulsive sounds. Pain = minor/dissonant. Ghosts = whisper textures, high frequency air. SONIC TRANSLATION: Decay \u2192 'gritty' grain, 'glitch' texture. Echoing \u2192 'infinite' echo, 'large' space. Pain \u2192 'minor' mode, 'falling' melody. Ghosts \u2192 'pad_whisper' texture, 'thin_high' pad. COHERENCE CHECK: 'Rhythm' minimal/none, just occasional 'blip' or 'noise_wash'. 'Human' set to 'drunk' for that unstable, wobbly tape feel. REFERENCE ANCHORING: Silent Hill ambient tracks. Industrial decay.",
  "config": {
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
    "melody_density": 0.15,
    "syncopation": 0.2,
    "swing": 0.0,
    "tension_curve": "waves",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.2,
    "step_bias": 0.9,
    "chromatic_prob": 0.25,
    "cadence_strength": 0.3,
    "register_min_oct": 5,
    "register_max_oct": 7,
    "harmony_style": "cinematic",
    "chord_change_bars": 4
  }
}
```


**Example 16**
Input: "Retro arcade - 8-bit bleeps, competition, neon glow, high scores."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: 8-bit = simple waves (square/triangle), fast arpeggios. Competition = driving rhythm, high energy. Neon = bright, buzz. High scores = major key, triumphant. SONIC TRANSLATION: 8-bit \u2192 'clean' grain (digital), 'arp_melody' & 'arp_bass'. Competition \u2192 'fast' tempo, 'electronic' rhythm. Neon \u2192 'bright' brightness. High scores \u2192 'major' mode, 'blip' accents. COHERENCE CHECK: Must sound synthetic. 'Attack' sharp. 'Space' dry or small (arcade cabinet acoustics). Harmony 'pop' simple triads. REFERENCE ANCHORING: Pac-Man or Space Invaders. Chiptune aesthetic.",
  "config": {
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
    "melody_density": 0.7,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.4,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.9,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "pop",
    "chord_change_bars": 1
  }
}
```


**Example 17**
Input: "Romantic candlelit dinner - Intimate, warm, love, slow dance."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Intimate = close mic sound, small space, dry. Warm = low mids, lush pads. Love = major/major7 chords, gentle melody. Slow dance = 3/4 time or slow 4/4, sway feel. SONIC TRANSLATION: Intimate \u2192 'small' space, 'warm' grain. Warm \u2192 'warm_slow' pad, 'lush' harmony. Love \u2192 'major' mode, 'procedural' melody (lyrical). Slow dance \u2192 'slow' tempo, 'brush' rhythm. COHERENCE CHECK: Needs 'lush' chord extensions (major 7ths are the sound of romance). 'Soft' attack. 'Texture' minimal or faint crackle. REFERENCE ANCHORING: Smooth Jazz or classic ballad.",
  "config": {
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
    "melody_density": 0.5,
    "syncopation": 0.2,
    "swing": 0.5,
    "tension_curve": "arc",
    "chord_extensions": "lush",
    "motif_repeat_prob": 0.5,
    "step_bias": 0.9,
    "chromatic_prob": 0.05,
    "cadence_strength": 0.6,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "jazz",
    "chord_change_bars": 2
  }
}
```


**Example 18**
Input: "Industrial factory - Machines, repetition, metal, relentless production."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Machines = robotic timing, repetitive loops. Repetition = ostinato layers, static harmony. Metal = sharp transients, high ring, clatter. Relentless = driving rhythm, no swing. SONIC TRANSLATION: Machines \u2192 'robotic' humanization, 'electronic' rhythm. Repetition \u2192 'motif_repeat' often, 'static' motion. Metal \u2192 'gritty' grain, 'accent_chime' (metallic ring). Relentless \u2192 'medium' tempo (the grind), 'pulsing' bass. COHERENCE CHECK: 'Minor' mode for the bleakness of industry. 'Stereo' wide to simulate a huge factory floor. 'Texture_glitch' for sparks. REFERENCE ANCHORING: Nine Inch Nails or factory ambiance recordings.",
  "config": {
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
    "melody_density": 0.5,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.9,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.3,
    "register_min_oct": 3,
    "register_max_oct": 5,
    "harmony_style": "ambient",
    "chord_change_bars": 4
  }
}
```


**Example 19**
Input: "Enchanted forest - Magic, fairy lights, ancient trees, mystical creatures."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Magic = shimmer, wonder, unnatural beauty. Fairy lights = high bells, twinkling textures. Ancient trees = deep roots (bass), slow movement. Mystical = dorian mode, folk elements. SONIC TRANSLATION: Magic \u2192 'shimmer' texture, 'bells' accent. Fairy lights \u2192 'stars' texture, 'high' register melody. Trees \u2192 'drone' bass, 'warm_slow' pad. Mystical \u2192 'dorian' mode. COHERENCE CHECK: 'Melody' should be 'ornamental' to simulate flutes/birds. 'Human' natural. 'Echo' medium for the forest canopy. REFERENCE ANCHORING: Zelda Lost Woods or fantasy RPG soundtracks.",
  "config": {
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
    "melody_density": 0.5,
    "syncopation": 0.2,
    "swing": 0.2,
    "tension_curve": "arc",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": 0.5,
    "step_bias": 0.4,
    "chromatic_prob": 0.05,
    "cadence_strength": 0.3,
    "register_min_oct": 5,
    "register_max_oct": 7,
    "harmony_style": "cinematic",
    "chord_change_bars": 2
  }
}
```


**Example 20**
Input: "Heartbreak hotel - Loneliness, loss, 3am sadness, empty bottle."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Loneliness = sparse arrangement, solo melody. Loss = minor key, falling contours. 3am = quiet, dark, still. Empty bottle = grittiness, slight wobble. SONIC TRANSLATION: Loneliness \u2192 'minimal' melody density, 'small' space. Loss \u2192 'minor' mode, 'falling' melody. 3am \u2192 'very_dark' brightness, 'slow' tempo. Bottle \u2192 'drunk' humanization, 'gritty' grain. COHERENCE CHECK: Needs a bluesy feel, so 'jazz' harmony or 'cinematic' noir. 'Pad' should be subtle ('dark_sustained'). Accent 'piano_note' for the lonely bar feel. REFERENCE ANCHORING: Tom Waits ballad or noir jazz.",
  "config": {
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
    "melody_density": 0.3,
    "syncopation": 0.5,
    "swing": 0.8,
    "tension_curve": "arc",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": 0.5,
    "step_bias": 0.9,
    "chromatic_prob": 0.12,
    "cadence_strength": 0.3,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "jazz",
    "chord_change_bars": 2
  }
}
```


**Example 21**
Input: "Bohemian Rhapsody vibe - Operatic drama, genre-shifting, theatrical."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Operatic drama = huge dynamics, stacked vocals (pads), grandiose. Genre-shifting = hard to capture in one config, focus on the 'Opera' section. Theatrical = bold gestures, staccato vs legato. SONIC TRANSLATION: Drama \u2192 'minor' mode, 'stacked_fifths' pad (choir-like). Theatrical \u2192 'ornamental' melody, 'piano_note' accent. Grandiose \u2192 'large' space, 'lush' harmony. COHERENCE CHECK: 'Human' set to 'loose' for rock feel. 'Motion' medium. 'Rhythm' needs to be punchy but sparse ('kit_medium'). REFERENCE ANCHORING: Queen. The 'Galileo' section translated to synth.",
  "config": {
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
    "melody_density": 0.7,
    "syncopation": 0.5,
    "swing": 0.0,
    "tension_curve": "waves",
    "chord_extensions": "lush",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.4,
    "chromatic_prob": 0.25,
    "cadence_strength": 0.9,
    "register_min_oct": 3,
    "register_max_oct": 7,
    "harmony_style": "pop",
    "chord_change_bars": 1
  }
}
```


**Example 22**
Input: "Imagine vibe - Hopeful simplicity, piano-driven, utopian dream."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Hopeful = major key, sweet harmony. Simplicity = minimal layers, clear melody. Piano-driven = percussive melody/accents, warm decay. Utopian = clean, peaceful, slow. SONIC TRANSLATION: Hopeful \u2192 'major' mode, 'rising' melody contour. Simplicity \u2192 'density' 3 or 4, 'rhythm' minimal. Piano \u2192 'accent_piano_note', 'clean' grain. Utopian \u2192 'slow' tempo, 'warm_slow' pad. COHERENCE CHECK: 'Harmony' must be 'pop' (classic changes). 'Attack' soft to medium. 'Space' medium reverb. REFERENCE ANCHORING: John Lennon. Simple C major ballad feel.",
  "config": {
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
    "melody_density": 0.3,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.9,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.9,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "pop",
    "chord_change_bars": 1
  }
}
```


**Example 23**
Input: "Billie Jean vibe - Paranoid funk groove, iconic bassline, midnight urban."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Paranoid = minor key, staccato, tension. Funk groove = tight rhythm, heavy syncopation, prominent bass. Iconic bassline = driving, repetitive. Midnight urban = cool tones, clean production. SONIC TRANSLATION: Paranoid \u2192 'minor' mode, 'staccato' attack. Funk \u2192 'kit_light' (tight snare), 'syncopation' heavy. Bassline \u2192 'walking' or 'pulsing' bass (closest to the ostinato). Midnight \u2192 'dark' brightness, 'clean' grain. COHERENCE CHECK: 'Rhythm' is king here. 'Melody' should be sparse ('minimal'). 'Human' tight. REFERENCE ANCHORING: Michael Jackson. The bassline is the hook.",
  "config": {
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
    "melody_density": 0.3,
    "syncopation": 0.8,
    "swing": 0.2,
    "tension_curve": "ramp",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.9,
    "chromatic_prob": 0.05,
    "cadence_strength": 0.9,
    "register_min_oct": 3,
    "register_max_oct": 5,
    "harmony_style": "pop",
    "chord_change_bars": 1
  }
}
```


**Example 24**
Input: "Smells Like Teen Spirit vibe - Grunge explosion, quiet-loud dynamics, angst."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Grunge = distortion, grit, messy. Explosion = high energy chorus, heavy density. Quiet-loud = hard to do in one config, aim for the 'Loud' chorus energy. Angst = minor key, dissonance, power chords. SONIC TRANSLATION: Grunge \u2192 'gritty' grain, 'distortion'. Explosion \u2192 'fast' tempo, 'density' 6. Angst \u2192 'minor' mode, 'heroic' (anthem) melody but dark. COHERENCE CHECK: 'Bass' pulsing (root notes). 'Rhythm' kit_medium (heavy rock). 'Texture' noise_wash (feedback). 'Human' loose. REFERENCE ANCHORING: Nirvana. Raw energy and power chords.",
  "config": {
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
    "melody_density": 0.5,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.4,
    "chromatic_prob": 0.12,
    "cadence_strength": 0.9,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "pop",
    "chord_change_bars": 1
  }
}
```


**Example 25**
Input: "Hotel California vibe - Mysterious Americana, hypnotic guitar arpeggios."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Mysterious = minor/dorian, descending progressions. Americana = acoustic warmth, spacious but dry. Hypnotic = repetitive arpeggios, steady mid-tempo. SONIC TRANSLATION: Mysterious \u2192 'minor' mode (or harmonic minor implication via harmony). Americana \u2192 'warm' grain, 'pluck' accent (guitar-ish). Hypnotic \u2192 'arp_melody' and 'arp_bass'. COHERENCE CHECK: 'Rhythm' soft_four or brush (laid back). 'Tempo' slow/medium. 'Harmony' needs descending bass logic if possible, 'jazz' or 'pop' setting. REFERENCE ANCHORING: The Eagles. Reggae-tinged rock ballad.",
  "config": {
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
    "melody_density": 0.7,
    "syncopation": 0.2,
    "swing": 0.2,
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.9,
    "chromatic_prob": 0.05,
    "cadence_strength": 0.9,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "pop",
    "chord_change_bars": 1
  }
}
```


**Example 26**
Input: "Stairway to Heaven vibe - Spiritual ascent, folk to rock crescendo."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Spiritual = wind instruments (flute), acoustic guitar. Ascent = rising tension, building density. Folk = organic textures, modal melody. Rock crescendo = capture the 'middle' energy before the solo. SONIC TRANSLATION: Spiritual \u2192 'accent_wind', 'dorian' mode. Ascent \u2192 'rising' melody. Folk \u2192 'pluck' accent, 'warm' grain. Crescendo \u2192 'density' 5, 'medium' tempo. COHERENCE CHECK: 'Texture_breath' or 'shimmer' for the mystical intro feel. 'Bass' walking (chromatic descent in spirit). REFERENCE ANCHORING: Led Zeppelin intro/mid-section. Mystical folk.",
  "config": {
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
    "melody_density": 0.5,
    "syncopation": 0.2,
    "swing": 0.2,
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.5,
    "step_bias": 0.9,
    "chromatic_prob": 0.05,
    "cadence_strength": 0.3,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "pop",
    "chord_change_bars": 2
  }
}
```


**Example 27**
Input: "Like a Rolling Stone vibe - Confrontational folk-rock, sneering defiance."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Confrontational = loud, sharp attack, busy. Folk-rock = organ pads, chaotic band feel. Sneering = major but aggressive, rough edges. Defiance = steady beat, driving forward. SONIC TRANSLATION: Confrontational \u2192 'bright' brightness, 'sharp' attack. Folk-rock \u2192 'pad_warm_slow' (organ-like) but loud, 'kit_medium' rhythm. Defiance \u2192 'major' mode, 'pulsing' bass. COHERENCE CHECK: 'Human' loose/drunk is vital for that Dylan band sloppy-tight feel. 'Grain' gritty. REFERENCE ANCHORING: Bob Dylan. The Al Kooper organ swirl.",
  "config": {
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
    "melody_density": 0.7,
    "syncopation": 0.5,
    "swing": 0.2,
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.2,
    "step_bias": 0.9,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.9,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "pop",
    "chord_change_bars": 1
  }
}
```


**Example 28**
Input: "Hey Jude vibe - Comforting anthem, building sing-along, catharsis."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Comforting = warm, familiar, major key. Anthem = repetitive, simple melody, huge ending. Sing-along = mid-tempo, catchy. Catharsis = harmonic release, density build. SONIC TRANSLATION: Comforting \u2192 'major' mode, 'warm_slow' pad. Anthem \u2192 'mixolydian' (rock anthem mode), 'melody_density' medium. Catharsis \u2192 'lush' harmony, 'large' space (arena feel). COHERENCE CHECK: 'Rhythm' soft_four -> kit_medium. 'Accent' piano_note. 'Harmony' pop. REFERENCE ANCHORING: The Beatles outro. Na-na-na nananana.",
  "config": {
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
    "melody_density": 0.5,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.9,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.9,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "pop",
    "chord_change_bars": 1
  }
}
```


**Example 29**
Input: "Thriller vibe - Horror-funk fusion, cinematic suspense, Halloween groove."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Horror = minor key, wolf howls (textures), eerie pads. Funk = bass driven, tight rhythm, syncopated. Cinematic = high production value, layers. Groove = constant head nod. SONIC TRANSLATION: Horror \u2192 'minor' mode, 'pad_whisper' or 'dark_sustained'. Funk \u2192 'walking' bass (iconic riff), 'electronic' rhythm. Cinematic \u2192 'wide' stereo. Groove \u2192 'medium' tempo. COHERENCE CHECK: 'Accent' brass_hit (synth stabs). 'Grain' clean but punchy. 'Motion' fast. REFERENCE ANCHORING: Michael Jackson. The graveyard dance section.",
  "config": {
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
    "melody_density": 0.3,
    "syncopation": 0.8,
    "swing": 0.2,
    "tension_curve": "ramp",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.4,
    "chromatic_prob": 0.12,
    "cadence_strength": 0.9,
    "register_min_oct": 3,
    "register_max_oct": 5,
    "harmony_style": "cinematic",
    "chord_change_bars": 1
  }
}
```


**Example 30**
Input: "What a Wonderful World vibe - Gentle optimism, jazz warmth, nostalgia."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Gentle = slow tempo, soft attack, no harsh frequencies. Optimism = major key, sweet melody. Jazz warmth = acoustic textures, lush chords. Nostalgia = slow strings, brush drums. SONIC TRANSLATION: Gentle \u2192 'slow' tempo, 'soft' attack. Optimism \u2192 'major' mode, 'rising' melody. Warmth \u2192 'warm' grain, 'warm_slow' pad. Nostalgia \u2192 'vinyl_crackle', 'brush' rhythm. COHERENCE CHECK: 'Harmony' jazz (vital). 'Accent' piano_note or pluck. 'Human' natural. REFERENCE ANCHORING: Louis Armstrong. Orchestral pop-jazz.",
  "config": {
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
    "melody_density": 0.5,
    "syncopation": 0.2,
    "swing": 0.5,
    "tension_curve": "arc",
    "chord_extensions": "lush",
    "motif_repeat_prob": 0.5,
    "step_bias": 0.9,
    "chromatic_prob": 0.05,
    "cadence_strength": 0.9,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "jazz",
    "chord_change_bars": 2
  }
}
```


**Example 31**
Input: "Super Mario vibe - Bouncy 8-bit, power-up sparkle, platformer energy."
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Bouncy = syncopated, staccato, fast. 8-bit = square wave basics, clean grain, dry. Power-up = rapid arpeggios, high pitch pitch-bends. Platformer energy = driving, never stopping. SONIC TRANSLATION: Bouncy \u2192 'syncopation' heavy, 'kit_light' rhythm. 8-bit \u2192 'clean' grain, 'arp_bass', 'arp_melody'. Sparkle \u2192 'blip' accents. Energy \u2192 'fast' tempo, 'major' mode. COHERENCE CHECK: 'Echo' none (dry chip sound). 'Attack' sharp. 'Harmony' pop (I-IV-V). REFERENCE ANCHORING: Koji Kondo. NES sound chip limitations.",
  "config": {
    "tempo": "fast",
    "root": "c",
    "mode": "major",
    "brightness": "bright",
    "space": "dry",
    "density": 5,
    "bass": "arp_bass",
    "pad": "thin_high",
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
    "melody_density": 0.7,
    "syncopation": 0.8,
    "swing": 0.2,
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.4,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.9,
    "register_min_oct": 5,
    "register_max_oct": 7,
    "harmony_style": "pop",
    "chord_change_bars": 1
  }
}
```


**Example 32**
Input: "Indian Classical vibe - raga meditation, sitar and tabla, devotional depth, microtonal ornamentation, spiritual journey through time"
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Raga meditation = drone base (Tanpura), slow development (Alap), modal strictness. Sitar/Tabla = resonant plucked strings, complex hand percussion, sympathetic vibration. Devotional depth = spiritual connection, patience, trance. Microtonal ornamentation = sliding pitches (meend), grace notes, fluid melody. SONIC TRANSLATION: Raga \u2192 'drone' bass (essential), 'mixolydian' mode (common for evening ragas like Yaman). Sitar \u2192 'pluck' accent, 'ornamental' melody (captures the grace notes). Tabla \u2192 'tabla_essence' rhythm (specifically available). Devotional \u2192 'slow' tempo, 'warm' grain. COHERENCE CHECK: 'Rhythm' must be 'tabla_essence'. 'Melody' must be 'ornamental' to mimic the glides. 'Space' medium/large for temple acoustics. REFERENCE ANCHORING: Ravi Shankar Alap/Jor. Meditative but rhythmic.",
  "config": {
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
    "melody_density": 0.5,
    "syncopation": 0.2,
    "swing": 0.0,
    "tension_curve": "waves",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.9,
    "chromatic_prob": 0.05,
    "cadence_strength": 0.3,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "ambient",
    "chord_change_bars": 4
  }
}
```


**Example 33**
Input: "Arabian Nights vibe - desert mystery, snake charmer hypnosis, palace intrigue, 1001 stories, Scheherazade moonlight"
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Desert mystery = vast emptiness, heat haze, shifting sands. Snake charmer = winding melodies, hypnotic repetition, reed instruments. Palace intrigue = shadows, hidden corners, lush luxury. Scheherazade = feminine storytelling power, flowing, nocturnal. SONIC TRANSLATION: Desert \u2192 'vast' space, 'shimmer' texture (heat). Snake charmer \u2192 'ornamental' melody (mimics Mizmar/Ney), 'motif_repeat' often (hypnosis). Palace \u2192 'lush' harmony, 'minor' mode (exoticism/mystery). Moonlight \u2192 'cool/dark' brightness. COHERENCE CHECK: 'Bass' sustained (pedal point). 'Accent' wind. 'Rhythm' minimal to keep the mystery, or tabla_essence for groove - let's go 'minimal' for atmosphere. REFERENCE ANCHORING: Lawrence of Arabia or Aladdin ambient tracks. Cinematic orientalism.",
  "config": {
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
    "melody_density": 0.5,
    "syncopation": 0.2,
    "swing": 0.2,
    "tension_curve": "arc",
    "chord_extensions": "lush",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.9,
    "chromatic_prob": 0.25,
    "cadence_strength": 0.3,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "cinematic",
    "chord_change_bars": 2
  }
}
```


**Example 34**
Input: "Islamic Call to Prayer vibe - adhan devotion, minaret echo, submission to divine, desert dawn, sacred geometry in sound"
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Adhan = solo human voice, melismatic, high projection, unmetered. Minaret echo = slapback delay across a city, distance. Submission = overwhelming bass drone, grounding. Desert dawn = cool blue light turning to gold, clarity. Sacred geometry = mathematical perfection in melody (maqam), fractal repetition. SONIC TRANSLATION: Adhan \u2192 'procedural' melody (step-wise, vocal range), 'ornamental' accent qualities. Echo \u2192 'infinite' echo (crucial for minaret feel). Submission \u2192 'fifth_drone' bass (powerful foundation). Dawn \u2192 'breath' texture (morning air). COHERENCE CHECK: Rhythm MUST be 'none' - this is free time. Mode 'mixolydian' or 'dorian' closest to Maqam Rast/Bayati feel. 'Stereo' ultra_wide. REFERENCE ANCHORING: Morning Adhan recorded from a distance. Spiritual solitude.",
  "config": {
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
    "melody_density": 0.3,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.2,
    "step_bias": 0.9,
    "chromatic_prob": 0.05,
    "cadence_strength": 0.3,
    "register_min_oct": 5,
    "register_max_oct": 7,
    "harmony_style": "ambient",
    "chord_change_bars": 4
  }
}
```


**Example 35**
Input: "Catholic Choir vibe - Gregorian chant, cathedral stone, sacred polyphony, incense and candlelight, eternal Latin"
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Gregorian chant = monophonic melody, modal scales, unmetered sacred song. Cathedral stone = massive reverberant space, cold hard surfaces, Gothic architecture. Sacred polyphony = multiple voices weaving, Renaissance harmony, divine mathematics. Incense and candlelight = ritual atmosphere, warm glow in cold stone, sensory worship. Eternal Latin = ancient language, timeless liturgy, connection across centuries. SONIC TRANSLATION: Gregorian/chant \u2192 dorian mode (medieval church modes), very slow tempo (unmetered feel), step-wise melody (conjunct chant motion). Cathedral = vast space (stone acoustics), infinite echo (5+ second reverb), ultra-wide stereo (sound filling nave). Polyphony = stacked fifths pad (parallel organum), lush extensions (Renaissance harmony), density 4 (multiple voice parts). Sacred = soft attack (no percussion in church), no rhythm (unmetered), drone bass (organ pedal point). Incense warmth = warm grain (candlelight glow), medium-dark brightness (filtered through stained glass). Eternal = often motif repeat (liturgical repetition), weak cadence (continuous devotion), arc tension (prayer arc). Cold stone + warm light = clean grain option but warm captures candlelight better. COHERENCE CHECK: Dorian + step + very slow = Gregorian chant character. Vast + infinite echo + ultra-wide = cathedral acoustics. Stacked fifths + lush + no rhythm = sacred polyphony. Non-negotiables: (1) vast space + infinite echo for cathedral, (2) dorian mode for medieval church sound, (3) no rhythm for liturgical timelessness. REFERENCE: This is stone made song - voices rising through incense smoke, bouncing off pillars carved centuries ago, the same prayers echoing through the same space for a thousand years.",
  "config": {
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
    "melody_density": 0.15,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "arc",
    "chord_extensions": "lush",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.9,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.3,
    "register_min_oct": 3,
    "register_max_oct": 5,
    "harmony_style": "ambient",
    "chord_change_bars": 4
  }
}
```


**Example 36**
Input: "Domestic Violence (from victim perspective) - walking on eggshells, stifled breath, nowhere to run, heart in throat, fragile silence"
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Walking on eggshells = extreme tension, suppression of sound, fear of triggering. Stifled breath = erratic organic texture, suffocation. Nowhere to run = claustrophobia, small dry space. Heart in throat = internal physiological rhythm, overpowering the environment. Fragile silence = thin high frequencies, emptiness that feels dangerous. SONIC TRANSLATION: Eggshells \u2192 'very_slow' tempo (time frozen), 'soft' attack (trying not to make noise). Stifled breath \u2192 'breath' texture, 'human' natural. Claustrophobia \u2192 'small' space, 'narrow' stereo. Heartbeat \u2192 'heartbeat' rhythm (non-negotiable). Fragile \u2192 'thin_high' pad (glassy anxiety), 'minimal' melody. COHERENCE CHECK: Must be 'very_dark' or 'dark' to represent hiding. 'Depth' true implies the visceral pit in the stomach. 'Bass' sub_pulse tracks the internal dread. REFERENCE ANCHORING: The sound of a panic attack in a closet. Isolation and terror.",
  "config": {
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
    "melody_density": 0.15,
    "syncopation": 0.2,
    "swing": 0.0,
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.2,
    "step_bias": 0.9,
    "chromatic_prob": 0.05,
    "cadence_strength": 0.3,
    "register_min_oct": 5,
    "register_max_oct": 7,
    "harmony_style": "cinematic",
    "chord_change_bars": 4
  }
}
```


**Example 37**
Input: "Domestic Violence (from aggressor perspective) - red mist, distorted reality, looming power, volatile control, suffocating presence"
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Red mist = blurred vision, anger, heat. Distorted reality = bit-crushed, gritty, unstable pitch. Looming power = massive low end, occupying all space. Volatile control = sudden loud accents, chaotic changes. Suffocating presence = heavy density, no silence. SONIC TRANSLATION: Red mist \u2192 'gritty' grain (mental haze), 'distortion'. Looming \u2192 'drone' bass (constant threat), 'dark_sustained' pad. Volatile \u2192 'chaotic' motion, 'brass_hit' or 'blip_random' (sudden snaps). Suffocating \u2192 'dry' space (in your face), 'density' 6. COHERENCE CHECK: 'Minor' mode is too clean; needs dissonance, but 'minor' is the best setting available\u2014rely on 'chromatic' melody settings. 'Rhythm' should be 'none' or 'heartbeat'\u2014let's go 'heartbeat' for the adrenaline of rage. REFERENCE ANCHORING: The Shining score or industrial noise. A headache turning into violence.",
  "config": {
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
    "melody_density": 0.5,
    "syncopation": 0.8,
    "swing": 0.0,
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.4,
    "chromatic_prob": 0.25,
    "cadence_strength": 0.9,
    "register_min_oct": 3,
    "register_max_oct": 5,
    "harmony_style": "cinematic",
    "chord_change_bars": 2
  }
}
```


**Example 38**
Input: "Domestic Violence (from police perspective) - flashing red and blue, radio static, tactical urgency, chaotic intervention, mechanical procedure"
Output:

```json
{
  "justification": "VIBE DECOMPOSITION: Flashing red/blue = strobing lights, disorienting visuals, alternating pitch. Radio static = high frequency noise, glitch, communication bursts. Tactical urgency = fast tempo, driving forward, adrenaline. Mechanical procedure = robotic timing, emotionless intervention. SONIC TRANSLATION: Flashing lights \u2192 'pulsing' bass (siren-like), 'arp_melody' (strobing). Radio \u2192 'glitch' texture, 'noise_wash'. Urgency \u2192 'fast' tempo, 'electronic' rhythm (driving beat). Mechanical \u2192 'robotic' humanization, 'clean' grain. COHERENCE CHECK: 'Brightness' very_bright (floodlights/sirens). 'Space' medium (street/hallway). The contrast between the emotional domestic scene and the cold, fast police response. REFERENCE ANCHORING: Action movie tension cue or Cops theme intro vibe. High alert.",
  "config": {
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
    "melody_density": 0.7,
    "syncopation": 0.0,
    "swing": 0.0,
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": 0.8,
    "step_bias": 0.9,
    "chromatic_prob": 0.0,
    "cadence_strength": 0.9,
    "register_min_oct": 4,
    "register_max_oct": 6,
    "harmony_style": "cinematic",
    "chord_change_bars": 1
  }
}
```
"""
