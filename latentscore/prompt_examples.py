from __future__ import annotations

FEW_SHOT_EXAMPLES = """
**Example 1**
Input: "Rainy day in Tokyo - neon lights reflecting on wet pavement, melancholy but beautiful"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Melancholy but beautiful = bittersweet duality. Wet pavement = reflective, shimmering surfaces. Neon = synthetic, slightly harsh color against organic rain. Tokyo = urban density, isolation in crowds, futuristic yet lonely. Night = dark but illuminated. Rain = steady, rhythmic, enveloping. SONIC TRANSLATION: Bittersweet \u2192 minor mode but with brightness (neon glow). Reflective surfaces \u2192 reverb + delay creating mirror-like echoes. Synthetic neon \u2192 gritty grain, electronic textures. Urban isolation \u2192 medium density (not empty, not overwhelming). Rain rhythm \u2192 soft steady pulse, not aggressive. Enveloping \u2192 wide stereo, immersive space. COHERENCE CHECK: Minor mode + medium brightness + gritty grain + shimmer texture all reinforce the neon-noir aesthetic. Large space + echo creates the wet reflective quality. Non-negotiables: (1) minor mode for melancholy, (2) shimmer texture for rain/neon reflections, (3) gritty grain for urban synthetic feel. REFERENCE: This needs the electronic intimacy of a lo-fi beat but with more spatial depth and cinematic weight.",
  "config": {
    "tempo": "slow",
    "root": "d#",
    "mode": "minor",
    "brightness": "medium",
    "space": "large",
    "density": 4,
    "bass": "sustained",
    "pad": "dark_sustained",
    "melody": "contemplative",
    "rhythm": "minimal",
    "texture": "shimmer",
    "accent": "pluck",
    "motion": "slow",
    "attack": "soft",
    "stereo": "wide",
    "depth": true,
    "echo": "heavy",
    "human": "natural",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "sparse",
    "syncopation": "light",
    "swing": "none",
    "tension_curve": "arc",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": "sometimes",
    "step_bias": "step",
    "chromatic_prob": "light",
    "cadence_strength": "weak",
    "harmony_style": "cinematic",
    "chord_change_bars": "slow",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 2**
Input: "Ancient temple at sunrise - sacred, peaceful, timeless"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Ancient = old, weathered, deep history, pre-modern. Temple = sacred geometry, spiritual resonance, stone/wood acoustics. Sunrise = gradual awakening, golden light emerging, hope after darkness. Sacred = reverent, meditative, connected to something larger. Peaceful = calm, no tension, accepting. Timeless = unchanging, eternal, beyond trend. SONIC TRANSLATION: Ancient \u2192 dorian mode (medieval/mystical association), warm grain (organic, not digital). Temple acoustics \u2192 vast reverb (stone chambers), long decay. Sunrise = rising melodic motion, gradual brightening (start dark, end brighter). Sacred = drone bass (om-like foundation), soft attacks (no violence). Peaceful = very slow tempo, minimal rhythm (no urgency). Timeless = simple harmonies (triads), no modern production tricks. COHERENCE CHECK: Dorian + drone + vast space + soft attack all support meditative temple feeling. Rising melody + warm grain capture sunrise warmth. Non-negotiables: (1) vast reverb for temple acoustics, (2) drone bass for sacred foundation, (3) very slow tempo for timelessness. REFERENCE: Think Gregorian chant meets ambient meditation music - the space IS the instrument.",
  "config": {
    "tempo": "very_slow",
    "root": "d",
    "mode": "dorian",
    "brightness": "dark",
    "space": "vast",
    "density": 3,
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
    "echo": "infinite",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "very_sparse",
    "syncopation": "straight",
    "swing": "none",
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "none",
    "cadence_strength": "weak",
    "harmony_style": "ambient",
    "chord_change_bars": "very_slow",
    "register_min_oct": 3,
    "register_max_oct": 5
  }
}
```

**Example 3**
Input: "Cyberpunk chase scene - adrenaline, neon, danger, synthetic"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Chase = fast movement, urgency, escape/pursuit. Adrenaline = heart pounding, fight-or-flight, heightened senses. Neon = harsh synthetic colors, futuristic urban. Danger = tension, minor mode, dissonance. Synthetic = artificial, digital, machine-made. Cyberpunk = dystopian future, tech-noir, Blade Runner aesthetic. SONIC TRANSLATION: Chase/adrenaline \u2192 very fast tempo, sharp attacks, pulsing bass (heartbeat). Neon/synthetic \u2192 gritty grain, electronic rhythm, clean digital textures. Danger \u2192 minor mode, chaotic motion (unpredictable modulation). Cyberpunk \u2192 arp melody (classic synth trope), electronic drums, narrow stereo (focused intensity). COHERENCE CHECK: Fast tempo + sharp attack + pulsing bass + electronic rhythm = chase energy. Minor + gritty + chaotic motion = danger/dystopia. These reinforce each other perfectly. Non-negotiables: (1) very fast tempo for chase urgency, (2) sharp attack for adrenaline punch, (3) pulsing bass for heartbeat tension. REFERENCE: Think Vangelis meets industrial - that relentless synthetic pursuit energy.",
  "config": {
    "tempo": "very_fast",
    "root": "a",
    "mode": "minor",
    "brightness": "bright",
    "space": "small",
    "density": 6,
    "bass": "pulsing",
    "pad": "cinematic",
    "melody": "arp_melody",
    "rhythm": "electronic",
    "texture": "glitch",
    "accent": "blip",
    "motion": "chaotic",
    "attack": "sharp",
    "stereo": "narrow",
    "depth": true,
    "echo": "subtle",
    "human": "robotic",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 2,
    "melody_density": "very_busy",
    "syncopation": "heavy",
    "swing": "none",
    "tension_curve": "waves",
    "chord_extensions": "triads",
    "motif_repeat_prob": "rare",
    "step_bias": "leapy",
    "chromatic_prob": "medium",
    "cadence_strength": "weak",
    "harmony_style": "cinematic",
    "chord_change_bars": "fast",
    "register_min_oct": 3,
    "register_max_oct": 7
  }
}
```

**Example 4**
Input: "Grandmother's kitchen - warm, nostalgic, Sunday morning, cookies baking"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Grandmother = old, wise, comforting, safe. Kitchen = domestic warmth, activity, nourishment. Warm = cozy, soft edges, golden tones. Nostalgic = memory, slightly hazy, bittersweet longing for past. Sunday morning = relaxed, unhurried, light streaming in. Cookies baking = sweet, anticipation, sensory comfort. SONIC TRANSLATION: Warm/cozy \u2192 warm grain, major mode, medium-dark brightness (golden, not harsh). Nostalgic/hazy \u2192 vinyl crackle texture (old recordings), soft attack, echo (memory reverb). Sunday relaxed \u2192 slow tempo, minimal rhythm. Domestic/safe \u2192 small intimate space (not vast cathedral). Sweet anticipation \u2192 rising melody hints, gentle motion. COHERENCE CHECK: Warm grain + vinyl crackle + soft attack + small space = intimate nostalgic memory. Major mode + slow tempo + minimal rhythm = peaceful Sunday. Non-negotiables: (1) vinyl crackle for nostalgia, (2) warm grain for coziness, (3) small space for domestic intimacy. REFERENCE: This is lo-fi hip-hop territory but slower and more acoustic - the warmth of memory, not the coolness of a beat tape.",
  "config": {
    "tempo": "slow",
    "root": "f",
    "mode": "major",
    "brightness": "dark",
    "space": "small",
    "density": 3,
    "bass": "sustained",
    "pad": "warm_slow",
    "melody": "contemplative",
    "rhythm": "none",
    "texture": "vinyl_crackle",
    "accent": "piano_note",
    "motion": "slow",
    "attack": "soft",
    "stereo": "narrow",
    "depth": false,
    "echo": "medium",
    "human": "loose",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "sparse",
    "syncopation": "straight",
    "swing": "light",
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "none",
    "cadence_strength": "medium",
    "harmony_style": "pop",
    "chord_change_bars": "slow",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 5**
Input: "Deep space exploration - vast emptiness, wonder, isolation, stars"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Deep space = infinite void, no atmosphere, absolute silence yet somehow presence. Vast emptiness = maximum space, minimal content, negative space as feature. Wonder = awe, discovery, the sublime. Isolation = alone, small against immensity. Stars = distant points of light, countless, ancient light. SONIC TRANSLATION: Vast emptiness \u2192 vast reverb, sparse density (2), very slow tempo (time stretches in space). Wonder/sublime \u2192 major mode but mysterious (mixolydian has that hopeful-mysterious quality), rising melodic hints. Isolation \u2192 wide stereo (emphasizes smallness in bigness), thin high pad (distant, cold). Stars \u2192 shimmer slow texture (twinkling), bells accent (crystalline points of light). Cold void \u2192 clean grain (digital, inhuman), bright but not warm. COHERENCE CHECK: Vast space + sparse density + very slow tempo = emptiness. Wide stereo + thin high pad + shimmer = isolation among stars. Mixolydian + rising melody + bells = wonder. Non-negotiables: (1) vast reverb for space itself, (2) density 2 for emptiness, (3) shimmer for stars. REFERENCE: Think 2001: A Space Odyssey - the terrifying beauty of infinite nothing.",
  "config": {
    "tempo": "very_slow",
    "root": "e",
    "mode": "mixolydian",
    "brightness": "bright",
    "space": "vast",
    "density": 2,
    "bass": "sub_pulse",
    "pad": "thin_high",
    "melody": "minimal",
    "rhythm": "none",
    "texture": "shimmer_slow",
    "accent": "bells",
    "motion": "static",
    "attack": "soft",
    "stereo": "ultra_wide",
    "depth": true,
    "echo": "infinite",
    "human": "robotic",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "very_sparse",
    "syncopation": "straight",
    "swing": "none",
    "tension_curve": "ramp",
    "chord_extensions": "lush",
    "motif_repeat_prob": "rare",
    "step_bias": "step",
    "chromatic_prob": "none",
    "cadence_strength": "weak",
    "harmony_style": "ambient",
    "chord_change_bars": "very_slow",
    "register_min_oct": 5,
    "register_max_oct": 7
  }
}
```

**Example 6**
Input: "1920s speakeasy - jazz, smoke, secrets, prohibition glamour"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: 1920s = vintage, pre-digital, analog warmth, Art Deco elegance. Speakeasy = hidden, intimate, underground, illicit thrill. Jazz = swing, improvisation, syncopation, sophistication. Smoke = hazy, diffused, mysterious atmosphere. Secrets = whispered, close, conspiratorial. Prohibition glamour = danger mixed with luxury, transgressive elegance. SONIC TRANSLATION: Jazz \u2192 dorian mode (jazz staple), swing feel, syncopation, walking bass. Vintage/analog \u2192 warm grain, vinyl crackle texture. Speakeasy intimate \u2192 small space (basement club), narrow stereo. Smoke/hazy \u2192 medium echo (diffused), soft attack (nothing harsh). Secrets \u2192 loose human timing (natural performance), ornamental melody (jazz flourishes). Sophistication \u2192 sevenths chord extensions (jazz harmony). COHERENCE CHECK: Dorian + walking bass + swing + sevenths = jazz foundation. Small space + warm grain + vinyl = vintage intimacy. Syncopation + ornamental + loose = live performance feel. Non-negotiables: (1) swing for jazz feel, (2) walking bass for that classic line, (3) warm grain + vinyl for period authenticity. REFERENCE: Imagine a trio in a smoky basement - piano, bass, brushes - captured on a wax cylinder.",
  "config": {
    "tempo": "medium",
    "root": "d",
    "mode": "dorian",
    "brightness": "dark",
    "space": "small",
    "density": 4,
    "bass": "walking",
    "pad": "warm_slow",
    "melody": "ornamental",
    "rhythm": "brush",
    "texture": "vinyl_crackle",
    "accent": "piano_note",
    "motion": "medium",
    "attack": "medium",
    "stereo": "narrow",
    "depth": false,
    "echo": "subtle",
    "human": "loose",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "heavy",
    "swing": "heavy",
    "tension_curve": "waves",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": "sometimes",
    "step_bias": "balanced",
    "chromatic_prob": "medium",
    "cadence_strength": "medium",
    "harmony_style": "jazz",
    "chord_change_bars": "medium",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 7**
Input: "Horror movie - creeping dread, something is wrong, building terror"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Horror = fear, darkness, the unknown. Creeping = slow approach, gradual, inexorable. Dread = anticipation of something bad, sustained tension. Something wrong = uncanny, dissonance, wrongness. Building terror = escalation, crescendo, no release. SONIC TRANSLATION: Creeping/slow \u2192 very slow tempo, static motion (frozen in fear). Dread/tension \u2192 minor mode, dark brightness (can't see what's coming), chromatic dissonance. Something wrong \u2192 gritty grain (corrupted), breath texture (presence in darkness), chaotic motion in melody. Building \u2192 ramp tension curve, busy melody density toward climax, cinematic pad for weight. No release \u2192 weak cadence (never resolves), echo creating ghosts. COHERENCE CHECK: Very slow + minor + very dark + vast space = horror atmosphere. Chromatic + gritty + breath texture = wrongness/presence. Ramp curve + weak cadence = building without release. Non-negotiables: (1) very dark brightness for blindness/fear, (2) chromatic for wrongness, (3) weak cadence to deny resolution. REFERENCE: Think The Shining score - that sense of wrongness that never lets go, always building toward something terrible.",
  "config": {
    "tempo": "very_slow",
    "root": "c#",
    "mode": "minor",
    "brightness": "very_dark",
    "space": "vast",
    "density": 4,
    "bass": "drone",
    "pad": "dark_sustained",
    "melody": "falling",
    "rhythm": "heartbeat",
    "texture": "breath",
    "accent": "none",
    "motion": "slow",
    "attack": "soft",
    "stereo": "wide",
    "depth": true,
    "echo": "heavy",
    "human": "natural",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "sparse",
    "syncopation": "straight",
    "swing": "none",
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "heavy",
    "cadence_strength": "weak",
    "harmony_style": "cinematic",
    "chord_change_bars": "very_slow",
    "register_min_oct": 3,
    "register_max_oct": 5
  }
}
```

**Example 8**
Input: "Victory celebration - triumph, euphoria, we did it, champagne popping"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Victory = achievement, overcoming, success. Triumph = glory, fanfare, the peak moment. Euphoria = overwhelming positive emotion, elation, release. We did it = collective joy, shared accomplishment, together. Champagne = effervescence, bubbles, sparkle, luxury celebration. SONIC TRANSLATION: Triumph/victory \u2192 major mode (unambiguous happiness), heroic rising melody, bright brightness (golden). Euphoria \u2192 fast tempo (energy release), wide stereo (expansive feeling), dense layers (fullness). Collective \u2192 thick density (6), cinematic pad (orchestral swell). Champagne/sparkle \u2192 shimmer texture, bells accent (crystalline celebration), sharp attack (pop of cork). Glory \u2192 brass hit accents, depth for impact, strong cadence (definitive resolution). COHERENCE CHECK: Major + bright + fast + rising = pure triumph. Dense + wide + cinematic = epic scale. Sharp attack + bells + shimmer = champagne sparkle. Non-negotiables: (1) major mode for unambiguous joy, (2) heroic rising melody, (3) density 6 for fullness of celebration. REFERENCE: Think Olympic medal ceremony, movie ending triumph - that unambiguous YES moment.",
  "config": {
    "tempo": "fast",
    "root": "c",
    "mode": "major",
    "brightness": "very_bright",
    "space": "large",
    "density": 6,
    "bass": "sustained",
    "pad": "cinematic",
    "melody": "heroic",
    "rhythm": "soft_four",
    "texture": "shimmer",
    "accent": "brass_hit",
    "motion": "fast",
    "attack": "sharp",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "tight",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "busy",
    "syncopation": "light",
    "swing": "none",
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": "often",
    "step_bias": "leapy",
    "chromatic_prob": "none",
    "cadence_strength": "strong",
    "harmony_style": "pop",
    "chord_change_bars": "medium",
    "register_min_oct": 3,
    "register_max_oct": 7
  }
}
```

**Example 9**
Input: "Underwater coral reef - colorful, teeming with life, flowing, dreamlike"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Underwater = submerged, muffled, pressure, different physics. Coral reef = explosion of color, organic shapes, biodiversity. Teeming with life = activity everywhere, but graceful not chaotic. Flowing = currents, gentle movement, no sharp edges. Dreamlike = surreal, floaty, time distortion. SONIC TRANSLATION: Underwater = muffled brightness (dark-medium), heavy reverb (sound travels differently), soft attack (water resistance). Coral colors = mixolydian mode (bright but mysterious), lush chord extensions (rich harmonics). Teeming = medium-busy density, but flowing motion not chaotic. Flowing/currents = medium tempo (not fast), swing feel (gentle sway), procedural melody with waves tension curve. Dreamlike = wide stereo (immersive), heavy echo (time stretching), loose human feel. COHERENCE CHECK: Dark brightness + heavy reverb + soft attack = underwater physics. Mixolydian + lush extensions + shimmer = colorful reef. Waves tension + swing + medium motion = current flow. Non-negotiables: (1) soft attack for water resistance, (2) heavy reverb for underwater acoustics, (3) lush extensions for color/richness. REFERENCE: Think nature documentary score when camera descends into reef - that wonder of alien beauty.",
  "config": {
    "tempo": "slow",
    "root": "f",
    "mode": "mixolydian",
    "brightness": "medium",
    "space": "large",
    "density": 5,
    "bass": "sustained",
    "pad": "ambient_drift",
    "melody": "procedural",
    "rhythm": "none",
    "texture": "shimmer_slow",
    "accent": "chime",
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
    "swing": "medium",
    "tension_curve": "waves",
    "chord_extensions": "lush",
    "motif_repeat_prob": "sometimes",
    "step_bias": "step",
    "chromatic_prob": "light",
    "cadence_strength": "weak",
    "harmony_style": "ambient",
    "chord_change_bars": "slow",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 10**
Input: "Rage and destruction - anger, smashing things, uncontrolled fury"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Rage = intense negative emotion, loss of control, seeing red. Destruction = breaking, impact, violence. Smashing = physical force, percussion, chaos. Uncontrolled = wild, unpredictable, beyond reason. Fury = sustained rage, burning, relentless. SONIC TRANSLATION: Rage/fury \u2192 minor mode (negative), very fast tempo (heart racing), sharp attack (violent transients). Destruction/smashing \u2192 heavy rhythm (impacts), gritty grain (distortion), depth for physical weight. Uncontrolled \u2192 chaotic motion, heavy syncopation (unpredictable), drunk human feel (wild). Burning \u2192 bright brightness (harsh, searing), narrow stereo (focused tunnel vision). Relentless \u2192 dense layers (overwhelming), fast chord changes (no rest), weak cadence (no resolution). COHERENCE CHECK: Fast + sharp + heavy rhythm + gritty = violent impact. Minor + chaotic + drunk = loss of control. Dense + narrow + weak cadence = relentless assault. Non-negotiables: (1) very fast tempo for racing heart, (2) sharp attack for violence, (3) chaotic motion for loss of control. REFERENCE: Think industrial metal breakdown - that moment of pure sonic violence where everything is destruction.",
  "config": {
    "tempo": "very_fast",
    "root": "a",
    "mode": "minor",
    "brightness": "bright",
    "space": "dry",
    "density": 6,
    "bass": "pulsing",
    "pad": "dark_sustained",
    "melody": "arp_melody",
    "rhythm": "electronic",
    "texture": "glitch",
    "accent": "brass_hit",
    "motion": "chaotic",
    "attack": "sharp",
    "stereo": "narrow",
    "depth": true,
    "echo": "none",
    "human": "drunk",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 2,
    "melody_density": "very_busy",
    "syncopation": "heavy",
    "swing": "none",
    "tension_curve": "waves",
    "chord_extensions": "triads",
    "motif_repeat_prob": "rare",
    "step_bias": "leapy",
    "chromatic_prob": "heavy",
    "cadence_strength": "weak",
    "harmony_style": "cinematic",
    "chord_change_bars": "fast",
    "register_min_oct": 3,
    "register_max_oct": 7
  }
}
```

**Example 11**
Input: "Cozy reading nook - fireplace crackling, rain outside, warm blanket, book"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Cozy = comfortable, safe, enclosed warmth. Reading nook = small intimate space, quietude, focused calm. Fireplace = warmth, gentle crackling, organic sound, flickering light. Rain outside = contrast (cold out, warm in), steady ambient sound, contemplative. Warm blanket = soft textures, comfort, protection. Book = intellectual calm, imagination, inner world. SONIC TRANSLATION: Cozy/enclosed \u2192 small space (intimate room), narrow stereo (focused). Fireplace/organic \u2192 warm grain, vinyl crackle (crackling analog), breath texture option. Rain = slow steady tempo, soft attack (gentle). Calm/contemplative \u2192 very sparse density, minimal melody, no rhythm. Warmth \u2192 dark brightness (golden, not harsh), major mode (contentment). Safety/comfort \u2192 medium echo (soft reflections), natural human feel. COHERENCE CHECK: Small space + warm grain + vinyl = analog intimate comfort. Slow + sparse + soft = calm quietude. Major + dark + natural = warm contentment. Non-negotiables: (1) small space for intimacy, (2) warm grain for fireplace analog feel, (3) vinyl crackle for that organic crackling. REFERENCE: Think hygge made sonic - that Danish art of cozy contentment, rain pattering while you're wrapped in warmth.",
  "config": {
    "tempo": "very_slow",
    "root": "g",
    "mode": "major",
    "brightness": "dark",
    "space": "small",
    "density": 2,
    "bass": "drone",
    "pad": "warm_slow",
    "melody": "minimal",
    "rhythm": "none",
    "texture": "vinyl_crackle",
    "accent": "none",
    "motion": "static",
    "attack": "soft",
    "stereo": "narrow",
    "depth": false,
    "echo": "subtle",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "very_sparse",
    "syncopation": "straight",
    "swing": "none",
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "none",
    "cadence_strength": "weak",
    "harmony_style": "ambient",
    "chord_change_bars": "very_slow",
    "register_min_oct": 3,
    "register_max_oct": 5
  }
}
```

**Example 12**
Input: "Carnival at night - manic energy, flashing lights, sugar rush, chaos"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Carnival = spectacle, sensory overload, entertainment, artifice. Night = darkness punctuated by artificial light, slightly sinister edge. Manic energy = hyperactive, unstoppable, almost unhinged. Flashing lights = rapid changes, bright bursts, disorienting. Sugar rush = sweet but overwhelming, jittery, artificial high. Chaos = disorder, unpredictable, everything happening at once. SONIC TRANSLATION: Manic/sugar rush \u2192 very fast tempo, busy melody density, chaotic motion. Flashing lights \u2192 bright brightness, sharp attack (bursts), fast chord changes. Carnival spectacle \u2192 major mode (festive), dense layers (6), arp melody (carousel feel). Chaos/overload \u2192 heavy syncopation, wide stereo (sounds everywhere), glitch texture. Night/sinister edge \u2192 clean grain (artificial), slight chromatic touches. COHERENCE CHECK: Fast + dense + busy + chaotic = sensory overload. Major + bright + arp = carnival festivity. Sharp + glitch + syncopation = disorienting flashes. Non-negotiables: (1) very fast for manic energy, (2) density 6 for overload, (3) chaotic motion for unpredictability. REFERENCE: Think Tim Burton carnival score - that manic festivity with an edge of madness.",
  "config": {
    "tempo": "very_fast",
    "root": "c",
    "mode": "major",
    "brightness": "very_bright",
    "space": "medium",
    "density": 6,
    "bass": "pulsing",
    "pad": "bright_open",
    "melody": "arp_melody",
    "rhythm": "electronic",
    "texture": "glitch",
    "accent": "bells_dense",
    "motion": "chaotic",
    "attack": "sharp",
    "stereo": "wide",
    "depth": false,
    "echo": "subtle",
    "human": "tight",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 2,
    "melody_density": "very_busy",
    "syncopation": "heavy",
    "swing": "light",
    "tension_curve": "waves",
    "chord_extensions": "triads",
    "motif_repeat_prob": "rare",
    "step_bias": "leapy",
    "chromatic_prob": "light",
    "cadence_strength": "weak",
    "harmony_style": "pop",
    "chord_change_bars": "fast",
    "register_min_oct": 5,
    "register_max_oct": 7
  }
}
```

**Example 13**
Input: "Meditation retreat - mindfulness, breath, letting go, inner peace"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Meditation = focused awareness, stillness, presence. Retreat = withdrawal from noise, sanctuary, dedicated space. Mindfulness = attention to now, non-judgment, observation. Breath = fundamental rhythm, life force, slow and deep. Letting go = release, non-attachment, surrender. Inner peace = calm center, equanimity, no conflict. SONIC TRANSLATION: Stillness/presence \u2192 very slow tempo, static motion (no urgency). Breath = breath texture (literal), soft attack (gentle as exhalation), natural human feel. Sanctuary = vast space (meditation hall acoustics), drone bass (om foundation). Letting go = weak cadence (no grasping for resolution), infinite echo (dissolving). Peace = major mode (but not excited - mixolydian for gentle), very dark brightness (eyes closed). Mindfulness = very sparse density (attention to each sound), step-wise melody (no leaps). COHERENCE CHECK: Very slow + static + sparse = stillness. Drone + vast + infinite echo = sanctuary space. Breath + soft + weak cadence = letting go. Non-negotiables: (1) very slow tempo for meditation pace, (2) breath texture, (3) vast space for dissolution of self. REFERENCE: Think singing bowl recordings - that perfect nothing where presence itself is the content.",
  "config": {
    "tempo": "very_slow",
    "root": "f",
    "mode": "mixolydian",
    "brightness": "very_dark",
    "space": "vast",
    "density": 2,
    "bass": "drone",
    "pad": "ambient_drift",
    "melody": "minimal",
    "rhythm": "none",
    "texture": "breath",
    "accent": "chime",
    "motion": "static",
    "attack": "soft",
    "stereo": "wide",
    "depth": false,
    "echo": "infinite",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "very_sparse",
    "syncopation": "straight",
    "swing": "none",
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "none",
    "cadence_strength": "weak",
    "harmony_style": "ambient",
    "chord_change_bars": "very_slow",
    "register_min_oct": 3,
    "register_max_oct": 5
  }
}
```

**Example 14**
Input: "First day of summer - freedom, sunshine, endless possibility, youth"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: First day = beginning, threshold, anticipation. Summer = warmth, light, outdoor life, growth. Freedom = liberation, no constraints, open horizons. Sunshine = bright, golden, life-giving, joyful. Endless possibility = optimism, future-oriented, expansive. Youth = energy, innocence, vitality, carefree. SONIC TRANSLATION: Freedom/endless \u2192 wide stereo (open horizons), large space (outdoor expanse), rising melody (looking forward). Sunshine/bright \u2192 very bright brightness, major mode (unambiguous joy), clean grain (clarity). Youth/energy \u2192 medium-fast tempo (vital but not frantic), medium density (active but not overwhelming). Optimism \u2192 strong cadence (confident resolution), arc tension curve (satisfying narrative). Carefree \u2192 natural human feel, light syncopation (playful), shimmer texture (sparkle). COHERENCE CHECK: Major + very bright + rising = pure optimism. Wide + large + rising = expansive freedom. Medium-fast + shimmer + light syncopation = youthful energy. Non-negotiables: (1) major mode for unambiguous joy, (2) very bright for sunshine, (3) wide stereo for freedom/openness. REFERENCE: Think coming-of-age movie opening - that pure potential of summer stretching endlessly ahead.",
  "config": {
    "tempo": "medium",
    "root": "g",
    "mode": "major",
    "brightness": "very_bright",
    "space": "large",
    "density": 4,
    "bass": "sustained",
    "pad": "bright_open",
    "melody": "rising",
    "rhythm": "soft_four",
    "texture": "shimmer",
    "accent": "bells",
    "motion": "medium",
    "attack": "medium",
    "stereo": "wide",
    "depth": false,
    "echo": "medium",
    "human": "natural",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "light",
    "swing": "light",
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": "sometimes",
    "step_bias": "balanced",
    "chromatic_prob": "none",
    "cadence_strength": "strong",
    "harmony_style": "pop",
    "chord_change_bars": "medium",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 15**
Input: "Abandoned asylum - decay, echoing footsteps, history of pain, ghosts"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Abandoned = emptiness, neglect, time stopped, forgotten. Asylum = institutional, suffering, madness, confinement. Decay = deterioration, entropy, things falling apart. Echoing footsteps = vast empty halls, reverb, presence in absence. History of pain = accumulated suffering, heavy atmosphere, trauma. Ghosts = presence of the past, whispers, not quite there. SONIC TRANSLATION: Abandoned/empty \u2192 vast space (empty halls), sparse density (nothing left), very slow tempo (time frozen). Decay \u2192 gritty grain (deteriorating), dark brightness (no light). Echoing = infinite echo, heavy reverb, pluck accent (footsteps). Pain/suffering \u2192 minor mode, chromatic dissonance, breath texture (whispers). Ghosts \u2192 soft attack (not quite solid), loose human feel (unstable), shimmer slow (flickering). Institutional = cold clean elements mixed with decay. COHERENCE CHECK: Vast + sparse + infinite echo = empty echoing halls. Minor + gritty + dark = suffering/decay. Breath + soft + chromatic = ghostly presence. Non-negotiables: (1) vast space for empty halls, (2) infinite echo for reverberation, (3) minor + dark for institutional horror. REFERENCE: Think horror game soundtrack for exploring ruins - that sense of terrible history pressing in.",
  "config": {
    "tempo": "very_slow",
    "root": "c#",
    "mode": "minor",
    "brightness": "very_dark",
    "space": "vast",
    "density": 3,
    "bass": "drone",
    "pad": "dark_sustained",
    "melody": "falling",
    "rhythm": "none",
    "texture": "breath",
    "accent": "pluck",
    "motion": "slow",
    "attack": "soft",
    "stereo": "ultra_wide",
    "depth": true,
    "echo": "infinite",
    "human": "loose",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "very_sparse",
    "syncopation": "straight",
    "swing": "none",
    "tension_curve": "ramp",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "medium",
    "cadence_strength": "weak",
    "harmony_style": "cinematic",
    "chord_change_bars": "very_slow",
    "register_min_oct": 3,
    "register_max_oct": 5
  }
}
```

**Example 16**
Input: "Retro arcade - 8-bit bleeps, competition, neon glow, high scores"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Retro = past (80s), nostalgia, lo-fi technology. Arcade = gaming, competition, social space, sounds everywhere. 8-bit = limited synthesis, square waves, chip music, digital artifacts. Bleeps = short, pitched, electronic tones. Competition = energy, tension, drive to win. Neon glow = bright artificial colors, synthetic. High scores = achievement, reward sounds, fanfare moments. SONIC TRANSLATION: 8-bit/retro \u2192 gritty grain (bit-crushed), clean attack aesthetic, blip accents. Arcade energy \u2192 fast tempo, busy melody (action), electronic rhythm. Competition \u2192 minor mode edge (tension of winning/losing), medium syncopation. Neon/artificial \u2192 bright brightness, narrow stereo (mono-ish like real arcades), dry space (no reverb in chip music). High scores = arp melody (classic video game), bells for reward sounds. Digital \u2192 robotic human (machine precision), clean grain base with gritty edge. COHERENCE CHECK: Fast + arp + electronic + gritty = arcade game energy. Dry + narrow + robotic = authentic chip music feel. Blip + bells + bright = that distinctive bleep-bloop sound. Non-negotiables: (1) dry space for chip music authenticity, (2) gritty grain for 8-bit, (3) blip accents for game sounds. REFERENCE: Think Pac-Man meets Tetris - that competitive arcade energy captured in sound.",
  "config": {
    "tempo": "fast",
    "root": "c",
    "mode": "minor",
    "brightness": "bright",
    "space": "dry",
    "density": 4,
    "bass": "pulsing",
    "pad": "thin_high",
    "melody": "arp_melody",
    "rhythm": "electronic",
    "texture": "glitch",
    "accent": "blip",
    "motion": "fast",
    "attack": "sharp",
    "stereo": "narrow",
    "depth": false,
    "echo": "none",
    "human": "robotic",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 2,
    "melody_density": "busy",
    "syncopation": "medium",
    "swing": "none",
    "tension_curve": "waves",
    "chord_extensions": "triads",
    "motif_repeat_prob": "often",
    "step_bias": "leapy",
    "chromatic_prob": "none",
    "cadence_strength": "medium",
    "harmony_style": "pop",
    "chord_change_bars": "fast",
    "register_min_oct": 5,
    "register_max_oct": 7
  }
}
```

**Example 17**
Input: "Romantic candlelit dinner - intimate, warm, love, slow dance"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Romantic = love, tenderness, emotional connection. Candlelit = warm glow, soft light, intimate illumination. Dinner = sophistication, care, special occasion. Intimate = close, private, just two people. Warm = emotional and physical comfort, golden tones. Love = deep positive emotion, vulnerability, openness. Slow dance = gentle movement together, 3/4 or slow 4/4, swaying. SONIC TRANSLATION: Romantic/love \u2192 major mode (positive), lush chord extensions (rich emotion), medium cadence (gentle resolution). Candlelit/warm \u2192 warm grain (analog softness), dark brightness (golden not harsh). Intimate \u2192 small space (just the two of you), narrow stereo (close together). Slow dance \u2192 slow tempo, swing feel (gentle sway), sustained bass (supportive). Sophistication \u2192 jazz harmony (evening elegance), ornamental melody (expressive), piano note accents. Tenderness \u2192 soft attack (no harshness), natural human feel (genuine). COHERENCE CHECK: Major + lush + warm = romantic warmth. Small + narrow + soft = intimate closeness. Slow + swing + sustained = slow dance feel. Non-negotiables: (1) warm grain for candlelight warmth, (2) small intimate space, (3) lush extensions for emotional richness. REFERENCE: Think jazz ballad in a quiet restaurant - Chet Baker playing to a room of two.",
  "config": {
    "tempo": "slow",
    "root": "e",
    "mode": "major",
    "brightness": "dark",
    "space": "small",
    "density": 4,
    "bass": "sustained",
    "pad": "warm_slow",
    "melody": "ornamental",
    "rhythm": "minimal",
    "texture": "none",
    "accent": "piano_note",
    "motion": "slow",
    "attack": "soft",
    "stereo": "narrow",
    "depth": false,
    "echo": "subtle",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "sparse",
    "syncopation": "light",
    "swing": "medium",
    "tension_curve": "arc",
    "chord_extensions": "lush",
    "motif_repeat_prob": "sometimes",
    "step_bias": "step",
    "chromatic_prob": "light",
    "cadence_strength": "medium",
    "harmony_style": "jazz",
    "chord_change_bars": "slow",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 18**
Input: "Industrial factory - machines, repetition, metal, relentless production"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Industrial = mechanical, manufactured, heavy duty. Factory = production line, systematic, efficiency. Machines = non-human, precise, rhythmic, metallic. Repetition = loops, patterns, monotony. Metal = hard, cold, resonant, harsh. Relentless = never stopping, inexorable, fatigue-inducing. Production = output-focused, purpose-driven, functional. SONIC TRANSLATION: Machines/repetition \u2192 robotic human (perfect timing), electronic rhythm (mechanical), often motif repeat (loops). Relentless \u2192 fast tempo (constant motion), pulsing bass (engine rhythm), tight human feel. Metal/harsh \u2192 gritty grain (distortion), bright brightness (harsh frequencies), sharp attack (impacts). Industrial weight \u2192 depth sub-bass (physical weight), narrow stereo (focused). Factory monotony \u2192 pattern melody engine, straight syncopation (no swing), simple triads. Cold/mechanical \u2192 clean space (dry, no romance), medium space (factory hall). COHERENCE CHECK: Fast + pulsing + electronic + robotic = machine rhythm. Gritty + bright + sharp = metallic harshness. Repetition + straight + often repeat = mechanical loops. Non-negotiables: (1) robotic timing for machine precision, (2) pulsing bass for engine rhythm, (3) gritty grain for industrial harshness. REFERENCE: Think Nine Inch Nails or Kraftwerk - that man-machine interface where humanity meets mechanical repetition.",
  "config": {
    "tempo": "fast",
    "root": "a",
    "mode": "minor",
    "brightness": "bright",
    "space": "medium",
    "density": 5,
    "bass": "pulsing",
    "pad": "dark_sustained",
    "melody": "arp_melody",
    "rhythm": "electronic",
    "texture": "glitch",
    "accent": "blip",
    "motion": "medium",
    "attack": "sharp",
    "stereo": "narrow",
    "depth": true,
    "echo": "none",
    "human": "robotic",
    "grain": "gritty",
    "melody_engine": "pattern",
    "phrase_len_bars": 2,
    "melody_density": "busy",
    "syncopation": "straight",
    "swing": "none",
    "tension_curve": "waves",
    "chord_extensions": "triads",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "none",
    "cadence_strength": "weak",
    "harmony_style": "cinematic",
    "chord_change_bars": "fast",
    "register_min_oct": 3,
    "register_max_oct": 5
  }
}
```

**Example 19**
Input: "Enchanted forest - magic, fairy lights, ancient trees, mystical creatures"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Enchanted = magical, otherworldly, transformed by wonder. Forest = organic, natural, canopy, depth, living. Magic = unexplainable, sparkle, transformation. Fairy lights = tiny points of light, twinkling, ethereal glow. Ancient trees = old growth, wisdom, presence, towering. Mystical creatures = hidden beings, watching, fantastical. SONIC TRANSLATION: Magic/enchanted \u2192 dorian mode (mystical), shimmer texture (sparkle), chime accents (fairy sounds). Forest/organic \u2192 warm grain (natural), breath texture option, natural human feel. Fairy lights \u2192 stars texture (twinkling), very bright brightness, bells (crystalline). Ancient/wisdom \u2192 slow tempo (timelessness), vast space (cathedral of trees), lush extensions (rich harmonics). Mystical = wide stereo (things in periphery), medium echo (mysterious reverb). Hidden/watching \u2192 sparse melody (glimpses), soft attack (gentle appearances). COHERENCE CHECK: Dorian + lush + warm = mystical organic beauty. Shimmer + bells + bright = fairy light sparkle. Vast + slow + wide = ancient forest scale. Non-negotiables: (1) dorian for mystical quality, (2) shimmer + bells for fairy lights, (3) warm grain for organic forest feel. REFERENCE: Think Studio Ghibli forest scenes - that sense of magic just beyond perception, nature as temple.",
  "config": {
    "tempo": "slow",
    "root": "d",
    "mode": "dorian",
    "brightness": "medium",
    "space": "large",
    "density": 4,
    "bass": "fifth_drone",
    "pad": "ambient_drift",
    "melody": "ornamental",
    "rhythm": "none",
    "texture": "stars",
    "accent": "chime",
    "motion": "slow",
    "attack": "soft",
    "stereo": "wide",
    "depth": false,
    "echo": "medium",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "sparse",
    "syncopation": "light",
    "swing": "none",
    "tension_curve": "arc",
    "chord_extensions": "lush",
    "motif_repeat_prob": "sometimes",
    "step_bias": "step",
    "chromatic_prob": "light",
    "cadence_strength": "weak",
    "harmony_style": "ambient",
    "chord_change_bars": "slow",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 20**
Input: "Heartbreak hotel - loneliness, loss, 3am sadness, empty bottle"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Heartbreak = emotional pain, love lost, aching. Hotel = transient, anonymous, nowhere place, isolation. Loneliness = alone with pain, no comfort, separation. Loss = something precious gone, grief, absence. 3am sadness = darkest hour, sleepless, thoughts spiraling. Empty bottle = drinking away pain, numbness attempted, failure. SONIC TRANSLATION: Heartbreak/loss \u2192 minor mode (sadness), falling melody (descending into grief), weak cadence (no resolution). Loneliness \u2192 small space (isolated room), narrow stereo (alone), sparse density. 3am = very slow tempo (time crawling), dark brightness (no light). Empty/numbness \u2192 drunk human feel (impaired), warm grain (alcohol haze), heavy echo (thoughts reverberating). Hotel anonymity \u2192 medium space (impersonal room), vinyl crackle (old sad song on repeat). Aching = contemplative melody, breath texture (sighs), lush extensions (rich sadness). COHERENCE CHECK: Minor + falling + weak = descending grief. Small + narrow + sparse = lonely isolation. Slow + dark + drunk = 3am deterioration. Non-negotiables: (1) falling melody for descending grief, (2) drunk human feel for impairment, (3) minor mode for unambiguous sadness. REFERENCE: Think Tom Waits at his most desolate - that beautiful wreckage of heartbreak in a cheap room.",
  "config": {
    "tempo": "very_slow",
    "root": "e",
    "mode": "minor",
    "brightness": "very_dark",
    "space": "small",
    "density": 3,
    "bass": "sustained",
    "pad": "dark_sustained",
    "melody": "falling",
    "rhythm": "none",
    "texture": "vinyl_crackle",
    "accent": "piano_note",
    "motion": "static",
    "attack": "soft",
    "stereo": "narrow",
    "depth": false,
    "echo": "heavy",
    "human": "drunk",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "sparse",
    "syncopation": "straight",
    "swing": "light",
    "tension_curve": "ramp",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "light",
    "cadence_strength": "weak",
    "harmony_style": "jazz",
    "chord_change_bars": "slow",
    "register_min_oct": 3,
    "register_max_oct": 5
  }
}
```

**Example 21**
Input: "Bohemian Rhapsody vibe - operatic drama, genre-shifting, theatrical, emotional crescendos, epic storytelling"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Operatic = theatrical grandeur, vocal-like dynamics, classical influence. Drama = tension and release, emotional peaks and valleys. Genre-shifting = unpredictable changes, sections that transform. Theatrical = performative, staged, larger than life. Emotional crescendos = building intensity, climactic moments. Epic storytelling = narrative arc, journey through emotions. SONIC TRANSLATION: Operatic/theatrical \u2192 cinematic pad (orchestral weight), wide stereo (stage width), lush chord extensions (rich harmony). Drama/crescendos \u2192 waves tension curve (multiple builds), strong cadence at peaks, density 6 (full production). Genre-shifting \u2192 leapy melody motion (unpredictable), medium-fast chord changes, chromatic touches. Emotional journey \u2192 minor mode base (drama), dorian could work but minor captures the tragedy. Epic scale \u2192 large space (concert hall), depth for weight, rising then falling melody patterns. COHERENCE CHECK: Cinematic + lush + wide + dense = operatic grandeur. Waves + chromatic + leapy = genre-shifting drama. Strong cadence + large space = theatrical resolution. Non-negotiables: (1) cinematic pad for operatic weight, (2) waves tension for multiple emotional peaks, (3) density 6 for full theatrical production. REFERENCE: This is maximalist rock opera - every element turned up, nothing subtle, pure theatrical commitment.",
  "config": {
    "tempo": "medium",
    "root": "b",
    "mode": "minor",
    "brightness": "bright",
    "space": "large",
    "density": 6,
    "bass": "sustained",
    "pad": "cinematic",
    "melody": "heroic",
    "rhythm": "soft_four",
    "texture": "shimmer",
    "accent": "brass_hit",
    "motion": "fast",
    "attack": "medium",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "natural",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "busy",
    "syncopation": "medium",
    "swing": "none",
    "tension_curve": "waves",
    "chord_extensions": "lush",
    "motif_repeat_prob": "sometimes",
    "step_bias": "leapy",
    "chromatic_prob": "medium",
    "cadence_strength": "strong",
    "harmony_style": "cinematic",
    "chord_change_bars": "medium",
    "register_min_oct": 3,
    "register_max_oct": 7
  }
}
```

**Example 22**
Input: "Imagine vibe - hopeful simplicity, piano-driven, utopian dream, gentle conviction, peaceful revolution"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Hopeful = optimistic but not naive, belief in possibility. Simplicity = stripped down, essential, uncluttered. Piano-driven = melodic clarity, harmonic foundation, intimate instrument. Utopian dream = idealistic, aspirational, visionary. Gentle conviction = soft but unwavering, quiet strength. Peaceful revolution = change through calm, non-aggressive transformation. SONIC TRANSLATION: Simplicity \u2192 sparse density (2-3), minimal texture, clean grain (clarity). Piano-driven \u2192 piano note accents, sustained bass (harmonic support), contemplative melody. Hopeful/utopian \u2192 major mode (positive vision), rising melodic tendency, medium brightness (warm optimism). Gentle/peaceful \u2192 very slow tempo (unhurried), soft attack (no violence), natural human feel. Conviction = medium cadence (resolution without force), step-wise melody (sure-footed). Intimate \u2192 small-medium space (personal address), narrow stereo (focused message). COHERENCE CHECK: Sparse + clean + soft = simplicity. Major + rising + step = hopeful conviction. Slow + natural + small space = gentle intimacy. Non-negotiables: (1) sparse density for simplicity, (2) major mode for hope, (3) soft attack for peaceful nature. REFERENCE: This is the sound of a single voice with piano, addressing humanity directly - intimate yet universal.",
  "config": {
    "tempo": "slow",
    "root": "c",
    "mode": "major",
    "brightness": "medium",
    "space": "small",
    "density": 2,
    "bass": "sustained",
    "pad": "warm_slow",
    "melody": "contemplative",
    "rhythm": "none",
    "texture": "none",
    "accent": "piano_note",
    "motion": "slow",
    "attack": "soft",
    "stereo": "narrow",
    "depth": false,
    "echo": "subtle",
    "human": "natural",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "sparse",
    "syncopation": "straight",
    "swing": "none",
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "none",
    "cadence_strength": "medium",
    "harmony_style": "pop",
    "chord_change_bars": "slow",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 23**
Input: "Billie Jean vibe - paranoid funk groove, iconic bassline, midnight urban, synth-pop tension, hypnotic pulse"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Paranoid = anxious, looking over shoulder, something's wrong. Funk groove = rhythmic lock, body movement, bass-driven. Iconic bassline = the foundation IS the song, pulsing, instantly recognizable. Midnight urban = city at night, streetlights, shadows. Synth-pop = electronic but melodic, 80s production, clean synths. Tension = unresolved, taut, sustained anxiety. Hypnotic pulse = trance-inducing repetition, locked-in feel. SONIC TRANSLATION: Paranoid/tension \u2192 minor mode, medium-heavy syncopation, chromatic touches for unease. Funk groove/bassline \u2192 walking bass (melodic movement), pulsing rhythm alternative, electronic drums. Hypnotic = often motif repeat (locked pattern), medium tempo (groove pocket), tight human feel. Midnight urban = medium-dark brightness, medium space (street reverb), gritty grain edge. Synth-pop = clean base with gritty texture, electronic rhythm, arp melody elements. 80s production = medium stereo width, subtle echo, sharp attack (punchy). COHERENCE CHECK: Minor + syncopation + chromatic = paranoid tension. Walking bass + electronic + tight = funk groove. Often repeat + medium tempo = hypnotic pulse. Non-negotiables: (1) walking bass for that iconic line feel, (2) minor mode for paranoia, (3) electronic rhythm for synth-pop groove. REFERENCE: This is the template for 80s pop-funk - that locked groove where tension and danceability coexist perfectly.",
  "config": {
    "tempo": "medium",
    "root": "f#",
    "mode": "minor",
    "brightness": "medium",
    "space": "medium",
    "density": 5,
    "bass": "walking",
    "pad": "thin_high",
    "melody": "procedural",
    "rhythm": "electronic",
    "texture": "none",
    "accent": "blip",
    "motion": "medium",
    "attack": "sharp",
    "stereo": "medium",
    "depth": true,
    "echo": "subtle",
    "human": "tight",
    "grain": "clean",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "heavy",
    "swing": "light",
    "tension_curve": "waves",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": "often",
    "step_bias": "balanced",
    "chromatic_prob": "light",
    "cadence_strength": "weak",
    "harmony_style": "pop",
    "chord_change_bars": "slow",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 24**
Input: "Smells Like Teen Spirit vibe - grunge explosion, quiet-loud dynamics, teenage angst, raw power, generational anthem"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Grunge = dirty, heavy, distorted, Seattle sound. Explosion = sudden impact, release of pressure, violent energy. Quiet-loud dynamics = extreme contrast, whisper to scream. Teenage angst = frustration, alienation, unformed rage. Raw power = unpolished, aggressive, visceral. Generational anthem = collective voice, rebellion, identity formation. SONIC TRANSLATION: Grunge/raw \u2192 gritty grain (maximum distortion), dark-medium brightness (murky), depth for weight. Explosion/loud \u2192 sharp attack, fast tempo sections, dense layers (6). Quiet-loud = waves tension curve (builds and releases), wide dynamic range implied by parameter extremes. Angst/frustration \u2192 minor mode, heavy syncopation (fighting the beat), chromatic (dissonance). Power/anthem \u2192 pulsing bass (driving force), electronic rhythm (can approximate pounding drums). Rebellion = chaotic motion, leapy melody (unpredictable), drunk human feel (sloppy on purpose). COHERENCE CHECK: Gritty + dark + depth = grunge weight. Sharp + fast + dense = explosive power. Waves + chaotic + drunk = dynamic chaos. Non-negotiables: (1) gritty grain for grunge distortion, (2) waves tension for quiet-loud, (3) chaotic motion for angst energy. REFERENCE: This is controlled chaos - the sound of a generation saying 'whatever' while screaming it at maximum volume.",
  "config": {
    "tempo": "fast",
    "root": "f",
    "mode": "minor",
    "brightness": "medium",
    "space": "medium",
    "density": 6,
    "bass": "pulsing",
    "pad": "dark_sustained",
    "melody": "arp_melody",
    "rhythm": "electronic",
    "texture": "glitch",
    "accent": "brass_hit",
    "motion": "chaotic",
    "attack": "sharp",
    "stereo": "wide",
    "depth": true,
    "echo": "subtle",
    "human": "drunk",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "busy",
    "syncopation": "heavy",
    "swing": "none",
    "tension_curve": "waves",
    "chord_extensions": "triads",
    "motif_repeat_prob": "often",
    "step_bias": "leapy",
    "chromatic_prob": "medium",
    "cadence_strength": "weak",
    "harmony_style": "cinematic",
    "chord_change_bars": "medium",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 25**
Input: "Hotel California vibe - mysterious Americana, dark paradise, hypnotic guitar arpeggios, haunted luxury, inescapable"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Mysterious = hidden meaning, enigmatic, questions without answers. Americana = desert, highways, west coast, mythic America. Dark paradise = beauty with menace, luxury hiding darkness. Hypnotic guitar arpeggios = fingerpicked patterns, circular, trance-inducing. Haunted luxury = opulence that's somehow wrong, beautiful decay. Inescapable = trapped, no exit, caught in a loop. SONIC TRANSLATION: Mysterious/haunted \u2192 dorian mode (ambiguous), medium echo (ghostly reflections), warm grain (vintage). Arpeggios = arp melody (obviously), often motif repeat (circular patterns), step-wise motion. Dark paradise = medium brightness (not too dark, still 'paradise'), medium space (hotel interior). Americana/desert = warm grain, large space option (desert expanse), pluck accents (guitar). Inescapable/hypnotic = slow tempo (languorous), waves tension (cycles), weak cadence (never truly resolves). Luxury = lush extensions (rich harmony), cinematic pad (grandeur), medium stereo. COHERENCE CHECK: Dorian + arp + weak cadence = mysterious circular trap. Warm + pluck + large = Americana guitar feel. Echo + waves + often repeat = hypnotic loop. Non-negotiables: (1) arp melody for those iconic arpeggios, (2) dorian for mysterious ambiguity, (3) weak cadence for 'you can never leave' feel. REFERENCE: This is sonic purgatory dressed as paradise - every element beautiful but somehow wrong, endlessly cycling.",
  "config": {
    "tempo": "slow",
    "root": "b",
    "mode": "dorian",
    "brightness": "medium",
    "space": "large",
    "density": 4,
    "bass": "sustained",
    "pad": "warm_slow",
    "melody": "arp_melody",
    "rhythm": "minimal",
    "texture": "shimmer_slow",
    "accent": "pluck",
    "motion": "slow",
    "attack": "soft",
    "stereo": "medium",
    "depth": false,
    "echo": "medium",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "light",
    "swing": "none",
    "tension_curve": "waves",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "light",
    "cadence_strength": "weak",
    "harmony_style": "cinematic",
    "chord_change_bars": "slow",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 26**
Input: "Stairway to Heaven vibe - spiritual ascent, folk to rock crescendo, mystical journey, acoustic to electric, transcendence"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Spiritual ascent = rising toward something higher, elevation, enlightenment. Folk to rock crescendo = builds from gentle acoustic to powerful electric. Mystical journey = quest, transformation, magic. Acoustic to electric = organic to powerful, intimate to epic. Transcendence = beyond ordinary, breakthrough, apotheosis. SONIC TRANSLATION: Spiritual/mystical \u2192 dorian mode (folk mysticism), lush extensions (spiritual richness), warm grain (organic start). Ascent/rising = rising melody (obviously), ramp tension curve (continuous build), register moving from low to high. Folk to rock = starts sparse (density 3-4), builds conceptually, soft attack evolving conceptually. Crescendo/transcendence = strong cadence at peak, wide stereo (expansive finale), large space (cathedral). Journey = arc phrases, sometimes motif repeat (call-backs), chromatic touches (mystical). Acoustic warmth = warm grain, pluck accents, contemplative melody base. COHERENCE CHECK: Dorian + lush + warm = mystical folk foundation. Rising + ramp + strong cadence = ascending crescendo. Large + wide + depth = transcendent finale. Non-negotiables: (1) rising melody for ascent, (2) ramp tension for building journey, (3) dorian for mystical folk quality. REFERENCE: This is the archetypal epic rock journey - starting in intimate mystery and ending in cosmic release.",
  "config": {
    "tempo": "slow",
    "root": "a",
    "mode": "dorian",
    "brightness": "medium",
    "space": "large",
    "density": 4,
    "bass": "sustained",
    "pad": "ambient_drift",
    "melody": "rising",
    "rhythm": "minimal",
    "texture": "shimmer_slow",
    "accent": "pluck",
    "motion": "slow",
    "attack": "soft",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "sparse",
    "syncopation": "straight",
    "swing": "none",
    "tension_curve": "ramp",
    "chord_extensions": "lush",
    "motif_repeat_prob": "sometimes",
    "step_bias": "balanced",
    "chromatic_prob": "light",
    "cadence_strength": "strong",
    "harmony_style": "cinematic",
    "chord_change_bars": "slow",
    "register_min_oct": 3,
    "register_max_oct": 7
  }
}
```

**Example 27**
Input: "Like a Rolling Stone vibe - confrontational folk-rock, sneering defiance, organ swirls, lyrical assault, cultural earthquake"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Confrontational = aggressive, in-your-face, challenging. Folk-rock = acoustic roots electrified, raw but melodic. Sneering defiance = attitude, contempt, rebellion. Organ swirls = Hammond B3 texture, sustained wash, swirling modulation. Lyrical assault = relentless, verbal barrage, no mercy. Cultural earthquake = revolutionary, paradigm-shifting, historical weight. SONIC TRANSLATION: Confrontational/sneering \u2192 minor mode edge (but mixolydian works for that defiant major-minor ambiguity), gritty grain (rawness), sharp attack (punchy). Folk-rock = medium tempo (driving shuffle feel), walking bass (melodic movement), natural human feel. Organ swirls = warm slow pad (organ-like sustain), medium motion (swirling modulation), shimmer texture. Assault/relentless = busy melody density, medium-heavy syncopation, fast chord changes. Defiance = strong cadence (emphatic), medium-wide stereo (confident stance). Revolution = dense layers (wall of sound), depth for impact. COHERENCE CHECK: Mixolydian + gritty + sharp = sneering folk-rock attitude. Warm pad + shimmer + motion = organ swirl texture. Busy + syncopation + fast changes = lyrical assault energy. Non-negotiables: (1) mixolydian for defiant ambiguity, (2) busy melody for assault feel, (3) gritty grain for raw confrontation. REFERENCE: This is the moment folk went electric and rock became literature - raw, smart, and absolutely uncompromising.",
  "config": {
    "tempo": "medium",
    "root": "c",
    "mode": "mixolydian",
    "brightness": "medium",
    "space": "medium",
    "density": 5,
    "bass": "walking",
    "pad": "warm_slow",
    "melody": "procedural",
    "rhythm": "soft_four",
    "texture": "shimmer",
    "accent": "piano_note",
    "motion": "medium",
    "attack": "sharp",
    "stereo": "medium",
    "depth": true,
    "echo": "subtle",
    "human": "natural",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "busy",
    "syncopation": "medium",
    "swing": "light",
    "tension_curve": "waves",
    "chord_extensions": "triads",
    "motif_repeat_prob": "sometimes",
    "step_bias": "balanced",
    "chromatic_prob": "none",
    "cadence_strength": "strong",
    "harmony_style": "pop",
    "chord_change_bars": "medium",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 28**
Input: "Hey Jude vibe - comforting anthem, building sing-along, gentle encouragement, na-na-na catharsis, orchestral swell"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Comforting = reassuring, supportive, warm embrace. Anthem = collective voice, stadium scale, universal. Building = gradual crescendo, accumulating power. Sing-along = simple memorable melody, participatory. Gentle encouragement = soft push, kind guidance, supportive. Catharsis = emotional release, letting go, resolution. Orchestral swell = strings and brass building, classical grandeur. SONIC TRANSLATION: Comforting/gentle \u2192 major mode (warmth), soft attack (gentle), warm grain (embrace). Anthem/sing-along = simple step-wise melody (easy to follow), often motif repeat (memorable hook), strong cadence (satisfying). Building/crescendo = ramp tension curve, density increasing conceptually (start 4, feel of 6), large space (stadium). Orchestral swell = cinematic pad, lush extensions, wide stereo (enveloping). Catharsis = strong cadence (release), very bright at peak, depth for emotional weight. Encouragement = medium tempo (steady), natural human feel (genuine). COHERENCE CHECK: Major + soft + warm = comforting warmth. Step + repeat + strong cadence = anthemic simplicity. Ramp + cinematic + wide = building orchestral power. Non-negotiables: (1) major mode for comfort/warmth, (2) ramp tension for building, (3) strong cadence for cathartic resolution. REFERENCE: This is the ultimate 'it's going to be okay' song - starting intimate and ending as a stadium-sized group hug.",
  "config": {
    "tempo": "medium",
    "root": "f",
    "mode": "major",
    "brightness": "bright",
    "space": "large",
    "density": 5,
    "bass": "sustained",
    "pad": "cinematic",
    "melody": "rising",
    "rhythm": "soft_four",
    "texture": "shimmer",
    "accent": "piano_note",
    "motion": "slow",
    "attack": "soft",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "straight",
    "swing": "none",
    "tension_curve": "ramp",
    "chord_extensions": "triads",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "none",
    "cadence_strength": "strong",
    "harmony_style": "pop",
    "chord_change_bars": "slow",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 29**
Input: "Thriller vibe - horror-funk fusion, cinematic suspense, Halloween groove, Vincent Price drama, moonwalk midnight"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Horror-funk = scary but danceable, tension with groove. Cinematic suspense = movie soundtrack tension, building dread. Halloween = spooky but fun, theatrical horror. Vincent Price drama = theatrical, grandiose, over-the-top Gothic. Moonwalk midnight = slick moves, urban night, cool danger. Groove = locked rhythm, body movement, pocket. SONIC TRANSLATION: Horror \u2192 minor mode, chromatic touches (creepy), dark brightness. Funk/groove = walking bass (melodic movement), electronic rhythm, heavy syncopation, tight human. Cinematic suspense = cinematic pad, waves tension (builds and releases), medium echo (suspense reverb). Halloween theatrical = gritty grain (monster textures), brass hit accents (stabs), glitch texture (horror sounds). Moonwalk cool = medium-fast tempo, clean elements mixed with gritty, sharp attack (precise moves). Drama = density 6 (full production), wide stereo (theatrical width). COHERENCE CHECK: Minor + chromatic + dark = horror foundation. Walking bass + syncopation + tight = funk groove. Cinematic + waves + brass hits = theatrical suspense. Non-negotiables: (1) minor mode for horror, (2) walking bass for funk groove, (3) waves tension for cinematic builds. REFERENCE: This is the blueprint for pop-horror - genuinely spooky but irresistibly danceable, theatrical without losing the groove.",
  "config": {
    "tempo": "medium",
    "root": "e",
    "mode": "minor",
    "brightness": "dark",
    "space": "medium",
    "density": 6,
    "bass": "walking",
    "pad": "cinematic",
    "melody": "procedural",
    "rhythm": "electronic",
    "texture": "glitch",
    "accent": "brass_hit",
    "motion": "medium",
    "attack": "sharp",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "tight",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "heavy",
    "swing": "light",
    "tension_curve": "waves",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": "often",
    "step_bias": "balanced",
    "chromatic_prob": "medium",
    "cadence_strength": "medium",
    "harmony_style": "cinematic",
    "chord_change_bars": "medium",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 30**
Input: "What a Wonderful World vibe - gentle optimism, jazz warmth, nostalgic beauty, simple gratitude, timeless serenity"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Gentle optimism = soft hope, quiet positive belief. Jazz warmth = swing feel, rich harmonies, analog soul. Nostalgic beauty = looking back with love, bittersweet appreciation. Simple gratitude = uncomplicated thankfulness, basic joys. Timeless serenity = peaceful regardless of era, eternal calm. SONIC TRANSLATION: Gentle/simple = sparse density (3), soft attack, slow tempo (unhurried appreciation). Jazz warmth = warm grain (analog), swing feel, dorian or major (major fits better for optimism), sevenths extensions. Nostalgic = vinyl crackle (old recording), small-medium space (intimate), natural human feel. Beauty/gratitude = major mode (positive), contemplative melody, medium brightness (golden, not harsh). Timeless/serenity = very slow tempo (eternal), step-wise melody (simple truth), arc tension (gentle rise and fall), strong cadence (satisfying resolution). Jazz orchestration = warm pad, lush extensions, piano accents. COHERENCE CHECK: Major + warm + soft = gentle optimism. Swing + sevenths + vinyl = jazz nostalgia. Slow + sparse + step = simple timeless beauty. Non-negotiables: (1) major mode for optimism, (2) warm grain + vinyl for nostalgic jazz warmth, (3) very slow tempo for serene timelessness. REFERENCE: This is pure distilled gratitude - the simplest possible expression of finding beauty in everyday existence.",
  "config": {
    "tempo": "very_slow",
    "root": "f",
    "mode": "major",
    "brightness": "medium",
    "space": "small",
    "density": 3,
    "bass": "sustained",
    "pad": "warm_slow",
    "melody": "contemplative",
    "rhythm": "none",
    "texture": "vinyl_crackle",
    "accent": "piano_note",
    "motion": "slow",
    "attack": "soft",
    "stereo": "narrow",
    "depth": false,
    "echo": "subtle",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "sparse",
    "syncopation": "light",
    "swing": "medium",
    "tension_curve": "arc",
    "chord_extensions": "sevenths",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "none",
    "cadence_strength": "strong",
    "harmony_style": "jazz",
    "chord_change_bars": "slow",
    "register_min_oct": 3,
    "register_max_oct": 5
  }
}
```

**Example 31**
Input: "Super Mario vibe - bouncy 8-bit, power-up sparkle, coin collecting joy, platformer energy, Mushroom Kingdom adventure"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Bouncy = springy, elastic, constant jumping motion. 8-bit = limited synthesis, chiptune character, NES-era constraints turned to style. Power-up sparkle = reward sounds, transformation joy, ascending tones. Coin collecting = repetitive satisfying plinks, accumulation pleasure. Platformer energy = constant motion, obstacle navigation, precise timing. Mushroom Kingdom = whimsical, colorful, fantastical but not dark, cartoon world. Adventure = journey, discovery, heroic but lighthearted. SONIC TRANSLATION: Bouncy/platformer \u2192 fast tempo (constant movement), sharp attack (precise jumps), pulsing bass (rhythmic bounce). 8-bit = gritty grain (bit-crushed), dry space (no reverb in NES hardware), narrow stereo (mono-ish authenticity), clean digital base. Power-up/coins = bells accent (reward chimes), very bright brightness (sparkly highs), arp melody (classic video game arpeggios). Whimsical/joy = major mode (unambiguous fun), rising melodic tendency (optimistic). Coin repetition = often motif repeat (iconic loops), tight human feel (machine precision). Adventure = medium-busy density, leapy melody (jumping between platforms), medium syncopation (playful rhythm). COHERENCE CHECK: Fast + sharp + pulsing = platformer bounce energy. Gritty + dry + narrow = authentic 8-bit constraints. Major + bright + bells = joyful reward sounds. Non-negotiables: (1) dry space for NES authenticity, (2) gritty grain for 8-bit character, (3) major mode + bells for that unmistakable Nintendo joy. REFERENCE: This is pure concentrated fun - the sound of a world where jumping is joy and every coin is a tiny celebration.",
  "config": {
    "tempo": "fast",
    "root": "c",
    "mode": "major",
    "brightness": "very_bright",
    "space": "dry",
    "density": 4,
    "bass": "pulsing",
    "pad": "thin_high",
    "melody": "arp_melody",
    "rhythm": "electronic",
    "texture": "none",
    "accent": "bells",
    "motion": "fast",
    "attack": "sharp",
    "stereo": "narrow",
    "depth": false,
    "echo": "none",
    "human": "robotic",
    "grain": "gritty",
    "melody_engine": "procedural",
    "phrase_len_bars": 2,
    "melody_density": "busy",
    "syncopation": "medium",
    "swing": "none",
    "tension_curve": "waves",
    "chord_extensions": "triads",
    "motif_repeat_prob": "often",
    "step_bias": "leapy",
    "chromatic_prob": "none",
    "cadence_strength": "strong",
    "harmony_style": "pop",
    "chord_change_bars": "fast",
    "register_min_oct": 5,
    "register_max_oct": 7
  }
}
```

**Example 32**
Input: "Indian Classical vibe - raga meditation, sitar and tabla, devotional depth, microtonal ornamentation, spiritual journey through time"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Raga = melodic framework, mood-specific scales, improvisation within structure. Meditation = focused awareness, trance-like absorption, timeless. Sitar = plucked strings with sympathetic resonance, ornamental bends, drone strings. Tabla = complex rhythmic cycles (tala), hand drums, conversational rhythm. Devotional = spiritual offering, connection to divine, bhakti tradition. Microtonal ornamentation = slides between notes, gamakas, notes as living things. Spiritual journey = transformation through sound, rasa (emotional essence), transcendence. SONIC TRANSLATION: Raga/microtonal \u2192 dorian mode (closest to many ragas like Kafi), heavy chromatic (microtones/ornaments), ornamental melody. Sitar = pluck accent (string plucks), warm grain (acoustic resonance), medium echo (sympathetic strings ringing). Tabla = tabla essence rhythm (obviously), loose human feel (expressive timing), heavy syncopation (complex tala). Meditation/devotional = very slow tempo (alap-like opening feel), drone bass (tanpura foundation), vast space (temple acoustics). Spiritual depth = lush extensions (harmonic richness), arc tension (raga journey), weak cadence (continuous flow, no Western resolution). Ornamentation = step-wise base with chromatic slides, busy melody density in development. COHERENCE CHECK: Dorian + chromatic + ornamental = raga-like melodic character. Drone + vast + very slow = meditative tanpura foundation. Tabla + loose + syncopation = classical rhythm feel. Non-negotiables: (1) drone bass for tanpura foundation, (2) ornamental melody for gamakas, (3) tabla essence rhythm for tala. REFERENCE: This is the sound of time dissolving - a single raga explored for hours, each note a universe, rhythm and melody in eternal conversation.",
  "config": {
    "tempo": "very_slow",
    "root": "d",
    "mode": "dorian",
    "brightness": "medium",
    "space": "vast",
    "density": 4,
    "bass": "drone",
    "pad": "ambient_drift",
    "melody": "ornamental",
    "rhythm": "tabla_essence",
    "texture": "breath",
    "accent": "pluck",
    "motion": "slow",
    "attack": "soft",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "loose",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 8,
    "melody_density": "medium",
    "syncopation": "heavy",
    "swing": "none",
    "tension_curve": "arc",
    "chord_extensions": "lush",
    "motif_repeat_prob": "sometimes",
    "step_bias": "step",
    "chromatic_prob": "heavy",
    "cadence_strength": "weak",
    "harmony_style": "ambient",
    "chord_change_bars": "very_slow",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 33**
Input: "Arabian Nights vibe - desert mystery, snake charmer hypnosis, palace intrigue, 1001 stories, Scheherazade moonlight"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Desert = vast emptiness, sand dunes, heat shimmer, ancient trade routes. Mystery = hidden knowledge, veiled secrets, the unknown. Snake charmer = hypnotic repetition, modal melody, entrancing. Palace intrigue = luxury with danger, silk and daggers, whispered plots. 1001 stories = narrative within narrative, endless tales, suspension of time. Scheherazade = survival through storytelling, feminine cunning, moonlit nights. Moonlight = silver illumination, romantic danger, night magic. SONIC TRANSLATION: Desert/vast \u2192 large space (endless dunes), warm grain (sand and heat). Mystery/intrigue \u2192 minor mode with Middle Eastern flavor (harmonic minor feel via chromatic), medium-dark brightness (shadows). Snake charmer/hypnotic = ornamental melody (sinuous lines), often motif repeat (circular hypnosis), slow tempo (trance-inducing). Palace luxury = lush extensions (opulent harmony), cinematic pad (grandeur), shimmer texture (jewels and silk). Scheherazade/stories = waves tension curve (story arcs), medium syncopation (narrative rhythm), call-response melody feeling. Moonlight = medium echo (night reverb), wide stereo (expansive night sky). Intrigue = chromatic touches (uncertainty), weak cadence (stories never truly end). COHERENCE CHECK: Minor + chromatic + ornamental = Middle Eastern melodic character. Slow + repeat + shimmer = hypnotic snake charmer trance. Large + lush + cinematic = palace grandeur. Non-negotiables: (1) ornamental melody for snake charmer sinuousness, (2) often motif repeat for hypnotic effect, (3) lush extensions for palace opulence. REFERENCE: This is the sound of stories saving lives - each melody a tale, each phrase delaying dawn, luxury and danger intertwined under desert stars.",
  "config": {
    "tempo": "slow",
    "root": "d",
    "mode": "minor",
    "brightness": "medium",
    "space": "large",
    "density": 4,
    "bass": "drone",
    "pad": "cinematic",
    "melody": "ornamental",
    "rhythm": "minimal",
    "texture": "shimmer_slow",
    "accent": "chime",
    "motion": "slow",
    "attack": "soft",
    "stereo": "wide",
    "depth": true,
    "echo": "medium",
    "human": "natural",
    "grain": "warm",
    "melody_engine": "procedural",
    "phrase_len_bars": 4,
    "melody_density": "medium",
    "syncopation": "medium",
    "swing": "none",
    "tension_curve": "waves",
    "chord_extensions": "lush",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "heavy",
    "cadence_strength": "weak",
    "harmony_style": "cinematic",
    "chord_change_bars": "slow",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 34**
Input: "Islamic Call to Prayer vibe - adhan devotion, minaret echo, submission to divine, desert dawn, sacred geometry in sound"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Adhan = call to prayer, vocal proclamation, reaching across distance. Devotion = complete surrender, spiritual focus, sacred duty. Minaret echo = tower acoustics, sound carrying across city, reverberant space. Submission to divine = humility, letting go of ego, acceptance. Desert dawn = first light, stillness before day, golden-pink horizon. Sacred geometry = mathematical beauty, perfect proportions, divine order in pattern. SONIC TRANSLATION: Adhan/vocal \u2192 ornamental melody (melismatic vocal style), step-wise with chromatic ornaments (maqam scales), very sparse density (solo voice). Minaret echo = vast space (sound traveling), infinite echo (multiple reflections across city), wide stereo (omnidirectional call). Devotion/submission = very slow tempo (unhurried reverence), soft attack (no aggression), weak cadence (continuous devotion). Desert dawn = warm grain (golden light), dark-medium brightness (pre-dawn), rising melodic tendency (sun rising). Sacred geometry = dorian or minor with chromatic (maqam approximation), arc tension (prayer structure), natural human feel (authentic vocal). Stillness = no rhythm (unmetered), drone bass (fundamental tone), breath texture (human presence). COHERENCE CHECK: Ornamental + chromatic + step = maqam-like melodic style. Vast + infinite echo + wide = minaret acoustics. Very slow + sparse + no rhythm = devotional stillness. Non-negotiables: (1) vast space + infinite echo for minaret call, (2) ornamental melody for adhan melisma, (3) no rhythm for unmetered sacred time. REFERENCE: This is the sound of cities pausing - a single voice calling across rooftops, time suspended between night and day, individual and infinite.",
  "config": {
    "tempo": "very_slow",
    "root": "d",
    "mode": "minor",
    "brightness": "dark",
    "space": "vast",
    "density": 2,
    "bass": "drone",
    "pad": "ambient_drift",
    "melody": "ornamental",
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
    "tension_curve": "arc",
    "chord_extensions": "triads",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "heavy",
    "cadence_strength": "weak",
    "harmony_style": "ambient",
    "chord_change_bars": "very_slow",
    "register_min_oct": 4,
    "register_max_oct": 6
  }
}
```

**Example 35**
Input: "Catholic Choir vibe - Gregorian chant, cathedral stone, sacred polyphony, incense and candlelight, eternal Latin"
Output:

```json
{
  "thinking": "VIBE DECOMPOSITION: Gregorian chant = monophonic melody, modal scales, unmetered sacred song. Cathedral stone = massive reverberant space, cold hard surfaces, Gothic architecture. Sacred polyphony = multiple voices weaving, Renaissance harmony, divine mathematics. Incense and candlelight = ritual atmosphere, warm glow in cold stone, sensory worship. Eternal Latin = ancient language, timeless liturgy, connection across centuries. SONIC TRANSLATION: Gregorian/chant \u2192 dorian mode (medieval church modes), very slow tempo (unmetered feel), step-wise melody (conjunct chant motion). Cathedral = vast space (stone acoustics), infinite echo (5+ second reverb), ultra-wide stereo (sound filling nave). Polyphony = stacked fifths pad (parallel organum), lush extensions (Renaissance harmony), density 4 (multiple voice parts). Sacred = soft attack (no percussion in church), no rhythm (unmetered), drone bass (organ pedal point). Incense warmth = warm grain (candlelight glow), medium-dark brightness (filtered through stained glass). Eternal = often motif repeat (liturgical repetition), weak cadence (continuous devotion), arc tension (prayer arc). Cold stone + warm light = clean grain option but warm captures candlelight better. COHERENCE CHECK: Dorian + step + very slow = Gregorian chant character. Vast + infinite echo + ultra-wide = cathedral acoustics. Stacked fifths + lush + no rhythm = sacred polyphony. Non-negotiables: (1) vast space + infinite echo for cathedral, (2) dorian mode for medieval church sound, (3) no rhythm for liturgical timelessness. REFERENCE: This is stone made song - voices rising through incense smoke, bouncing off pillars carved centuries ago, the same prayers echoing through the same space for a thousand years.",
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
    "melody_density": "very_sparse",
    "syncopation": "straight",
    "swing": "none",
    "tension_curve": "arc",
    "chord_extensions": "lush",
    "motif_repeat_prob": "often",
    "step_bias": "step",
    "chromatic_prob": "none",
    "cadence_strength": "weak",
    "harmony_style": "ambient",
    "chord_change_bars": "very_slow",
    "register_min_oct": 3,
    "register_max_oct": 5
  }
}
```

"""
