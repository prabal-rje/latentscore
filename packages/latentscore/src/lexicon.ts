/**
 * Bundled copy of spec/mood_lexicon.json for zero-I/O execution across Node,
 * Bun, Deno, browsers, and bundlers (CI tests enforce parity with spec/).
 */

export interface LexiconEntry {
  readonly keywords: readonly string[];
  readonly set: Readonly<Record<string, string | number | boolean>>;
}

export interface LexiconGroup {
  readonly group: string;
  readonly first_match: boolean;
  readonly entries: readonly LexiconEntry[];
}

export const MOOD_LEXICON_SPEC_VERSION = 1 as const;

export const MOOD_LEXICON_RULES: readonly LexiconGroup[] = [
  {
    group: "mood_color",
    first_match: true,
    entries: [
      {
        keywords: ["dark", "sad", "night", "mysterious", "shadow", "melancholy", "grief", "midnight"],
        set: { root: "d", mode: "dorian", brightness: "dark", pad: "dark_sustained", melody: "contemplative" },
      },
      {
        keywords: ["happy", "bright", "joy", "sunrise", "morning", "sunlight", "cheerful", "golden"],
        set: { root: "c", mode: "major", brightness: "bright", pad: "warm_slow", melody: "rising" },
      },
      {
        keywords: ["epic", "cinematic", "powerful", "dramatic", "storm", "thunder", "heroic"],
        set: { root: "d", mode: "minor", brightness: "medium", pad: "cinematic", depth: true, density: 6 },
      },
      {
        keywords: ["meditation", "zen", "spiritual", "peaceful", "calm", "tranquil", "forest", "garden"],
        set: { root: "d", mode: "dorian", brightness: "medium", human: "natural", grain: "warm", rhythm: "none" },
      },
      {
        keywords: ["rain", "rainy", "drizzle", "cafe", "coffee", "vinyl", "lofi", "cozy", "hearth", "fireplace"],
        set: { root: "f", mode: "dorian", brightness: "dark", texture: "vinyl_crackle", grain: "warm", human: "natural" },
      },
      {
        keywords: ["cyberpunk", "neon", "synthwave", "electronic", "techno", "city", "arcade"],
        set: { root: "a", mode: "minor", brightness: "bright", rhythm: "electronic", bass: "pulsing", grain: "gritty", attack: "sharp" },
      },
    ],
  },
  {
    group: "tempo",
    first_match: true,
    entries: [
      { keywords: ["still", "glacial", "frozen", "sleep", "drift", "floating"], set: { tempo: "very_slow", motion: "slow" } },
      { keywords: ["slow", "calm", "meditation", "zen", "quiet", "gentle", "rain", "cozy", "coffee"], set: { tempo: "slow", motion: "slow" } },
      { keywords: ["fast", "energy", "drive", "run", "cyberpunk", "neon", "urgent", "rush"], set: { tempo: "fast", motion: "fast" } },
    ],
  },
  {
    group: "space",
    first_match: true,
    entries: [
      { keywords: ["vast", "space", "ocean", "deep", "underwater", "cave", "nebula", "cosmos", "stars", "cathedral"], set: { space: "vast", echo: "heavy", stereo: "wide" } },
      { keywords: ["intimate", "close", "desk", "room", "cafe", "coffee", "fireplace", "study"], set: { space: "small", echo: "subtle", stereo: "narrow" } },
    ],
  },
  {
    group: "texture_override",
    first_match: true,
    entries: [
      { keywords: ["stars", "crystal", "snow", "frost", "ice", "twinkle"], set: { texture: "stars", accent: "chime" } },
      { keywords: ["ocean", "wind", "breath", "mist", "fog", "cloud"], set: { texture: "breath" } },
      { keywords: ["vinyl", "lofi", "tape", "cafe", "coffee", "fireplace", "rain"], set: { texture: "vinyl_crackle" } },
    ],
  },
  {
    group: "density_override",
    first_match: true,
    entries: [
      { keywords: ["minimal", "sparse", "solo", "quiet", "whisper", "empty"], set: { density: 3, melody: "minimal", rhythm: "none" } },
      { keywords: ["lush", "rich", "full", "orchestral", "grand", "dense"], set: { density: 6, chord_extensions: "lush" } },
    ],
  },
];
