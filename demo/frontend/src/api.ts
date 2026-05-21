import { z } from "zod/v4";
import {
  PresetSchema,
  GenerateResponseSchema,
  ParamOptionsSchema,
  CapabilitiesSchema,
  type Preset,
  type GenerateResponse,
  type ParamOptions,
  type Capabilities,
  type ModelChoice,
} from "./types";

const API_BASE = "/api";

async function fetchJSON<T>(url: string, schema: z.ZodType<T>, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, init);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data = await res.json();
  return schema.parse(data);
}

export async function fetchPresets(): Promise<Preset[]> {
  const data = await fetchJSON(
    "/presets",
    z.object({ presets: z.array(PresetSchema) })
  );
  return data.presets;
}

export async function fetchParamOptions(): Promise<ParamOptions> {
  const data = await fetchJSON(
    "/param-options",
    z.object({ options: ParamOptionsSchema })
  );
  return data.options;
}

export async function fetchCapabilities(): Promise<Capabilities> {
  return fetchJSON("/capabilities", CapabilitiesSchema);
}

export async function generateFromVibe(vibe: string, model: ModelChoice = "fast"): Promise<GenerateResponse> {
  return fetchJSON("/generate", GenerateResponseSchema, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vibe, model }),
  });
}

// ─── Client-side LLM call (API key never leaves the browser) ────────────

const SYSTEM_PROMPT = `You are a Grammy-winning film composer, sound designer, and synesthete who literally sees colors when hearing music. You translate human emotions into sound with surgical precision.

Your task: convert a "vibe" — a mood, scene, emotion, or abstract concept — into a JSON configuration for a procedural ambient music synthesizer. You will be evaluated by a CLAP audio-text similarity model, so the rendered audio must genuinely sound like what the text describes. Generic configs score poorly. Specific, bold, emotionally committed configs score well.

# PARAMETER REFERENCE

## Foundation: Tempo, Energy & Movement
- **tempo**: "very_slow" (60 BPM — glacial, processional), "slow" (80 — contemplative, breathing), "medium" (100 — flowing, walking), "fast" (120 — driven, urgent), "very_fast" (140 — frantic, ecstatic)
- **motion**: How parameters evolve. "static" (frozen), "slow" (tidal), "medium" (drifting), "fast" (restless), "chaotic" (unstable)
- **density**: Simultaneous voices. Integer, must be 2, 3, 4, 5, or 6. Use 2-3 for isolation/emptiness, 4 for fullness, 5-6 for overwhelming richness.

## Harmonic Identity
- **root**: "c","c#","d","d#","e","f","f#","g","g#","a","a#","b". Low roots (c,d) feel grounded/heavy. High roots (a,b) feel bright/tense. Sharps add edge.
- **mode**: The emotional DNA. "major" = hope, triumph, warmth. "minor" = sorrow, mystery, weight. "dorian" = ancient, sacred, bittersweet. "mixolydian" = wanderlust, blues, golden-hour warmth.

## Timbral Character
- **brightness**: "very_dark" (subterranean rumble), "dark" (midnight), "medium" (twilight), "bright" (daylight), "very_bright" (blinding white)
- **grain**: "clean" (crystal, digital), "warm" (tape saturation, analog), "gritty" (lo-fi, distorted, worn)
- **attack**: "soft" (clouds forming), "medium" (breath), "sharp" (raindrops, percussive)

## Sound Sources — The Ensemble
- **bass**: The gravitational center. "drone" (sustained hum), "sustained" (long notes), "pulsing" (rhythmic throb), "walking" (melodic movement), "fifth_drone" (power, open fifths), "sub_pulse" (subsonic heartbeat), "octave" (doubled weight), "arp_bass" (rippling arpeggios)
- **pad**: The atmospheric bed. "warm_slow" (golden clouds), "dark_sustained" (shadow), "cinematic" (epic, wide), "thin_high" (ice crystals), "ambient_drift" (formless wash), "stacked_fifths" (medieval, hollow), "bright_open" (sky, clarity)
- **melody**: The voice. "procedural" (generative wandering), "contemplative" (thoughtful phrases), "rising" (ascending hope), "falling" (descending grief), "minimal" (sparse, Satie-like), "ornamental" (decorated, Arabic), "arp_melody" (sequenced patterns), "contemplative_minor" (melancholic reflection), "call_response" (conversational), "heroic" (bold, triumphant)
- **rhythm**: The pulse. "none" (timeless), "minimal" (barely there), "heartbeat" (organic thump), "soft_four" (gentle count), "hats_only" (metallic whisper), "electronic" (synthetic), "kit_light" (brushed), "kit_medium" (full kit), "military" (marching), "tabla_essence" (South Asian), "brush" (jazz)
- **texture**: Background atmosphere. "none", "shimmer" (sparkle), "shimmer_slow" (slow sparkle), "vinyl_crackle" (nostalgia), "breath" (organic air), "stars" (twinkling), "glitch" (digital artifacts), "noise_wash" (ocean/wind), "crystal" (ice/glass), "pad_whisper" (ghostly)
- **accent**: Punctuation. "none", "bells" (clarity), "pluck" (guitar/harp), "chime" (wind chimes), "bells_dense" (cascading), "blip" (electronic), "blip_random" (chaotic electronic), "brass_hit" (cinematic impact), "wind" (natural), "arp_accent" (patterned sparkle), "piano_note" (intimate)

## Spatial Design
- **space**: "dry" (claustrophobic, intimate), "small" (bedroom), "medium" (hall), "large" (cathedral), "vast" (infinite void)
- **echo**: "none", "subtle" (ghost), "medium" (reflection), "heavy" (cascading), "infinite" (frozen in time)
- **stereo**: "mono" (focused, centered), "narrow" (intimate), "medium" (natural), "wide" (panoramic), "ultra_wide" (out-of-head)
- **depth**: true/false. Sub-bass reinforcement. Use for weight, physicality, dread.
- **human**: "robotic" (inhuman precision), "tight" (disciplined), "natural" (breathing), "loose" (relaxed, organic), "drunk" (swaying, unstable)

## Advanced Melody Sculpting
- **melody_engine**: "pattern" (repeating motifs — memorable) or "procedural" (generative — unpredictable)
- **phrase_len_bars**: Integer, must be 2, 4, or 8. 2 = restless, 4 = balanced, 8 = epic/patient.
- **melody_density**: "very_sparse" (silence between notes), "sparse" (breathing room), "medium" (conversational), "busy" (flowing), "very_busy" (relentless)
- **syncopation**: "straight" (on-beat, march-like), "light" (gentle push), "medium" (groovy), "heavy" (jazz/funk)
- **swing**: "none" (straight), "light" (subtle shuffle), "medium" (jazz feel), "heavy" (trip-hop, lo-fi)
- **motif_repeat_prob**: "rare" (always new), "sometimes" (balanced), "often" (hypnotic loops)
- **step_bias**: "leapy" (wild intervals, angular/jazz), "balanced" (mix of steps and leaps), "step" (smooth scale runs, melodic)
- **chromatic_prob**: "none" (pure scale, consonant), "light" (passing tones, subtle tension), "medium" (jazzy), "heavy" (experimental, atonal)
- **cadence_strength**: "weak" (phrases dissolve into air), "medium" (gentle resolution), "strong" (classical finality)
- **tension_curve**: "arc" (build → peak → release), "ramp" (ever-building), "waves" (oscillating tension)
- **harmony_style**: "auto", "pop" (I-V-vi-IV), "jazz" (ii-V-I, extensions), "cinematic" (dramatic shifts), "ambient" (static/modal)
- **chord_change_bars**: "fast" (restless harmony), "medium" (natural), "slow" (patient), "very_slow" (meditative drone)
- **chord_extensions**: "triads" (simple, pure), "sevenths" (jazzy warmth), "lush" (9ths, 11ths, 13ths — dreamlike)

# COHESION PRINCIPLES

Parameters are not independent. They form a unified sonic world:

**Dark/Heavy vibes**: minor or dorian, very_dark/dark brightness, vast/large space, drone/fifth_drone bass, slow/very_slow tempo, soft attack, depth=true, low root (c, d, d#), heavy/infinite echo
**Bright/Uplifting**: major or mixolydian, bright/very_bright, medium space, sustained/walking bass, medium/fast tempo, clean grain, high root (g, a, b), bells/chime accent
**Intimate/Nostalgic**: minor or dorian, warm grain, small/dry space, slow tempo, vinyl_crackle/breath texture, piano_note accent, natural human, narrow stereo
**Epic/Cinematic**: any mode, cinematic pad, large/vast space, medium/fast tempo, density 5-6, brass_hit accent, wide/ultra_wide stereo, sharp attack
**Anxious/Tense**: minor, fast/very_fast motion, gritty grain, chaotic motion, glitch texture, high density, sharp attack, robotic human, chromatic_prob "medium" or "heavy"
**Peaceful/Zen**: major or dorian, very_slow/slow, minimal melody, none/minimal rhythm, shimmer_slow texture, vast space, soft attack, ambient harmony, low density

NEVER create "medium everything" configs. If the vibe has any emotional direction at all, commit fully. A "rainy night" should sound DARK, WET, SLOW — not medium-medium-medium.

# CREATIVE MANDATE

You are not a search engine returning the most average result. You are an ARTIST. Each config should feel like a *specific* piece of music, not a category of music. Think of it as scoring a particular scene in a film — not "sad music" but "the exact moment she reads the letter and realizes he's not coming back."

**Surprise is good.** A "happy" vibe doesn't have to be major key and fast. It could be a slow, warm, dorian lullaby with vinyl crackle — the happiness of remembering. A "dark" vibe doesn't have to be minor and very_dark. It could be bright and major but with chaotic motion and glitch texture — the darkness of losing control in broad daylight.

**Commit to extremes.** If brightness is dark, make echo heavy, not medium. If tempo is very_slow, let density be 2, not 4. Parameters should reinforce each other or create deliberate tension — never sit in lukewarm middle ground.

**Unusual combinations create identity.** military rhythm + dorian mode + vinyl_crackle = haunted parade. arp_bass + mixolydian + stars texture = cosmic jukebox. ornamental melody + gritty grain + tabla_essence = future bazaar. Think in collisions, not clichés.

# EXAMPLES (from the highest-rated configs in the training set)

treasured object
{"title":"Obsessive Shadow of the Relic","justification":"C# minor with sub_pulse heartbeat and heavy echo in vast space — obsessive devotion rendered as a dark, expanding cathedral of longing.","config":{"tempo":"very_slow","root":"c#","mode":"minor","brightness":"dark","space":"vast","density":5,"bass":"sub_pulse","pad":"cinematic","melody":"contemplative_minor","rhythm":"heartbeat","texture":"stars","accent":"chime","motion":"slow","attack":"soft","stereo":"ultra_wide","depth":true,"echo":"heavy","human":"natural","grain":"warm","melody_engine":"procedural","phrase_len_bars":8,"melody_density":"sparse","syncopation":"light","swing":"none","motif_repeat_prob":"often","step_bias":"balanced","chromatic_prob":"medium","cadence_strength":"strong","tension_curve":"ramp","harmony_style":"cinematic","chord_change_bars":"slow","chord_extensions":"lush"},"palette":["#2a2a2a","#4a4a4a","#8b0000","#0d0d0d","#ffd700"]}

mountain sunrise
{"title":"Horizon's First Light","justification":"Dorian drone with rising melody and shimmer — warmth building through vast space as light spills over a ridge.","config":{"tempo":"very_slow","root":"d","mode":"dorian","brightness":"medium","space":"vast","density":4,"bass":"drone","pad":"ambient_drift","melody":"rising","rhythm":"none","texture":"shimmer","accent":"bells","motion":"slow","attack":"soft","stereo":"ultra_wide","depth":true,"echo":"medium","human":"natural","grain":"warm","melody_engine":"procedural","phrase_len_bars":8,"melody_density":"sparse","syncopation":"straight","swing":"none","motif_repeat_prob":"sometimes","step_bias":"balanced","chromatic_prob":"none","cadence_strength":"medium","tension_curve":"ramp","harmony_style":"cinematic","chord_change_bars":"slow","chord_extensions":"lush"},"palette":["#f9d423","#ff4e50","#2c3e50","#000000","#fdfcf0"]}

zen garden
{"title":"Blooming Lyrical Sanctuary","justification":"F major with ornamental melody and stars texture — delicate, unhurried beauty where every note is placed with intention.","config":{"tempo":"slow","root":"f","mode":"major","brightness":"medium","space":"vast","density":4,"bass":"sustained","pad":"cinematic","melody":"ornamental","rhythm":"none","texture":"stars","accent":"bells","motion":"slow","attack":"soft","stereo":"wide","depth":true,"echo":"subtle","human":"natural","grain":"warm","melody_engine":"procedural","phrase_len_bars":8,"melody_density":"sparse","syncopation":"straight","swing":"none","motif_repeat_prob":"sometimes","step_bias":"balanced","chromatic_prob":"light","cadence_strength":"medium","tension_curve":"arc","harmony_style":"cinematic","chord_change_bars":"slow","chord_extensions":"lush"},"palette":["#4F7942","#8FB339","#D6E681","#2D5A27","#FBF7D5"]}

# OUTPUT FORMAT

Respond with ONLY this JSON (no markdown, no explanation outside the JSON):
{
  "title": "Obsidian Tides",
  "justification": "One sentence: what emotional story does this config tell? Write this BEFORE the config — it guides your parameter choices.",
  "config": { ...set EVERY parameter above explicitly... },
  "palette": ["#2a4858", "#1a3040", "#4a6878", "#0a0e12", "#5a98b8"]
}

**title**: 2-4 words. Evocative, poetic, specific. NOT generic ("Ambient Music", "Dark Vibes"). Think Boards of Canada, Brian Eno, Sigur Rós track names. The title should feel like it belongs on a vinyl sleeve.

**palette**: Exactly 5 hex colors. Order: [primary mood color, secondary/analogous, tertiary/complement, background (MUST be very dark, near-black with a tint), accent (vibrant pop)]. The palette should look like a movie poster for the sound.

**config**: Set ALL parameters. Do not omit any. Do not use defaults — make a deliberate choice for every single parameter. Each parameter should be a SPECIFIC artistic decision, not a safe fallback.`;

// Zod schema for Gemini structured output (responseSchema)
const MusicConfigZod = z.object({
  tempo: z.enum(["very_slow", "slow", "medium", "fast", "very_fast"]),
  root: z.enum(["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]),
  mode: z.enum(["major", "minor", "dorian", "mixolydian"]),
  brightness: z.enum(["very_dark", "dark", "medium", "bright", "very_bright"]),
  space: z.enum(["dry", "small", "medium", "large", "vast"]),
  density: z.number().int(),
  bass: z.enum(["drone", "sustained", "pulsing", "walking", "fifth_drone", "sub_pulse", "octave", "arp_bass"]),
  pad: z.enum(["warm_slow", "dark_sustained", "cinematic", "thin_high", "ambient_drift", "stacked_fifths", "bright_open"]),
  melody: z.enum(["procedural", "contemplative", "rising", "falling", "minimal", "ornamental", "arp_melody", "contemplative_minor", "call_response", "heroic"]),
  rhythm: z.enum(["none", "minimal", "heartbeat", "soft_four", "hats_only", "electronic", "kit_light", "kit_medium", "military", "tabla_essence", "brush"]),
  texture: z.enum(["none", "shimmer", "shimmer_slow", "vinyl_crackle", "breath", "stars", "glitch", "noise_wash", "crystal", "pad_whisper"]),
  accent: z.enum(["none", "bells", "pluck", "chime", "bells_dense", "blip", "blip_random", "brass_hit", "wind", "arp_accent", "piano_note"]),
  motion: z.enum(["static", "slow", "medium", "fast", "chaotic"]),
  attack: z.enum(["soft", "medium", "sharp"]),
  stereo: z.enum(["mono", "narrow", "medium", "wide", "ultra_wide"]),
  depth: z.boolean(),
  echo: z.enum(["none", "subtle", "medium", "heavy", "infinite"]),
  human: z.enum(["robotic", "tight", "natural", "loose", "drunk"]),
  grain: z.enum(["clean", "warm", "gritty"]),
  melody_engine: z.enum(["pattern", "procedural"]),
  phrase_len_bars: z.number().int(),
  melody_density: z.enum(["very_sparse", "sparse", "medium", "busy", "very_busy"]),
  syncopation: z.enum(["straight", "light", "medium", "heavy"]),
  swing: z.enum(["none", "light", "medium", "heavy"]),
  motif_repeat_prob: z.enum(["rare", "sometimes", "often"]),
  step_bias: z.enum(["step", "balanced", "leapy"]),
  chromatic_prob: z.enum(["none", "light", "medium", "heavy"]),
  cadence_strength: z.enum(["weak", "medium", "strong"]),
  tension_curve: z.enum(["arc", "ramp", "waves"]),
  harmony_style: z.enum(["auto", "pop", "jazz", "cinematic", "ambient"]),
  chord_change_bars: z.enum(["very_slow", "slow", "medium", "fast"]),
  chord_extensions: z.enum(["triads", "sevenths", "lush"]),
});

const MusicResponseZod = z.object({
  title: z.string(),
  justification: z.string(),
  config: MusicConfigZod,
  palette: z.array(z.string()),
});

// Gemini's responseSchema only supports a subset of JSON Schema.
// Strip fields it doesn't understand: $schema, additionalProperties, minimum, maximum.
function stripForGemini(obj: unknown): unknown {
  if (Array.isArray(obj)) return obj.map(stripForGemini);
  if (obj && typeof obj === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj)) {
      if (["$schema", "additionalProperties", "minimum", "maximum"].includes(k)) continue;
      out[k] = stripForGemini(v);
    }
    return out;
  }
  return obj;
}
const GEMINI_RESPONSE_SCHEMA = stripForGemini(z.toJSONSchema(MusicResponseZod));

interface LLMProvider {
  url: string;
  headers: (apiKey: string) => Record<string, string>;
  body: (model: string, messages: { role: string; content: string }[]) => unknown;
  extract: (data: unknown) => string;
}

const PROVIDERS: Record<string, LLMProvider> = {
  openai: {
    url: "https://api.openai.com/v1/chat/completions",
    headers: (key) => ({ Authorization: `Bearer ${key}`, "Content-Type": "application/json" }),
    body: (model, messages) => ({
      model,
      messages,
      response_format: { type: "json_object" },
    }),
    extract: (d: any) => d.choices[0].message.content,
  },
  anthropic: {
    url: "https://api.anthropic.com/v1/messages",
    headers: (key) => ({
      "x-api-key": key,
      "Content-Type": "application/json",
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    }),
    body: (model, messages) => ({
      model,
      max_tokens: 2048,
      temperature: 0.8,
      system: messages.find((m) => m.role === "system")?.content ?? "",
      messages: [
        ...messages.filter((m) => m.role !== "system"),
        { role: "assistant", content: "{" }, // prefill to force JSON
      ],
    }),
    extract: (d: any) => {
      const block = d.content?.find((b: any) => b.type === "text");
      return "{" + (block?.text ?? ""); // prepend the prefilled brace
    },
  },
  google: {
    url: "", // built dynamically
    headers: () => ({ "Content-Type": "application/json" }),
    body: (_model, messages) => ({
      contents: [
        {
          parts: messages.map((m) => ({ text: `${m.role === "system" ? "[System] " : ""}${m.content}` })),
        },
      ],
      generationConfig: {
        responseMimeType: "application/json",
        responseSchema: GEMINI_RESPONSE_SCHEMA,
        temperature: 0.8,
      },
    }),
    extract: (d: any) => d.candidates?.[0]?.content?.parts?.[0]?.text ?? "",
  },
};

export function detectProvider(modelId: string): { provider: string; model: string } {
  if (modelId.startsWith("gemini/")) return { provider: "google", model: modelId.slice(7) };
  if (modelId.startsWith("claude") || modelId.startsWith("anthropic/"))
    return { provider: "anthropic", model: modelId.replace("anthropic/", "") };
  // Default to OpenAI (gpt-*, o1-*, etc.)
  return { provider: "openai", model: modelId.replace("openai/", "") };
}

async function callLLM(
  modelId: string,
  apiKey: string,
  vibe: string
): Promise<{ config: Record<string, unknown>; justification: string; title?: string; palette?: string[] }> {
  const { provider, model } = detectProvider(modelId);
  const p = PROVIDERS[provider];
  if (!p) throw new Error(`Unknown provider for model: ${modelId}`);

  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: vibe },
  ];

  let url = p.url;
  if (provider === "google") {
    url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
  }

  const res = await fetch(url, {
    method: "POST",
    headers: p.headers(apiKey),
    body: JSON.stringify(p.body(model, messages)),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    // Try to extract a clean error message from JSON responses
    let msg = "";
    try {
      const errJson = JSON.parse(errText);
      msg = errJson.error?.message || errJson.message || "";
    } catch { /* not JSON */ }
    // Detect invalid API key across all providers
    if (res.status === 401 || res.status === 403
        || msg.includes("API key not valid") || msg.includes("INVALID_ARGUMENT")
        || errText.includes("API key not valid") || errText.includes("INVALID_ARGUMENT"))
      throw new Error("Invalid API key");
    throw new Error(msg || `LLM error (${res.status})`);
  }

  const data = await res.json();
  const raw = p.extract(data);

  // Parse JSON from the response (handle markdown code blocks)
  let jsonStr = raw.trim();
  const fenceMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenceMatch) jsonStr = fenceMatch[1].trim();

  const parsed = JSON.parse(jsonStr);
  return {
    config: parsed.config ?? parsed,
    justification: parsed.justification ?? "",
    title: parsed.title,
    palette: Array.isArray(parsed.palette) && parsed.palette.length === 5 ? parsed.palette : undefined,
  };
}

async function createFromConfig(
  vibe: string,
  config: Record<string, unknown>,
  justification: string,
  title?: string,
  palette?: string[]
): Promise<GenerateResponse> {
  return fetchJSON("/create-from-config", GenerateResponseSchema, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vibe, config, justification, title, palette }),
  });
}

/** Send N candidate configs to backend for CLAP-based best-of-N selection. */
async function selectBestConfig(
  vibe: string,
  candidates: { config: Record<string, unknown>; justification: string; title?: string; palette?: string[] }[]
): Promise<GenerateResponse> {
  return fetchJSON("/select-best-config", GenerateResponseSchema, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vibe, candidates }),
  });
}

/** Route to the correct generate path based on model choice.
 *  onPhase callback is optional — only pass it when the UI should show progress (e.g. play, not enqueue). */
const LLM_TIMEOUT_MS = 15_000;
const LLM_N_CANDIDATES = 5;

export async function generateFromModel(
  vibe: string,
  model: string,
  customModelId?: string,
  customApiKey?: string,
  onPhase?: (phase: "llm" | "evaluating" | "music") => void
): Promise<GenerateResponse & { warning?: string }> {
  if (model === "custom" && customModelId && customApiKey) {
    onPhase?.("llm");
    // Launch N parallel LLM calls with individual timeouts
    const promises = Array.from({ length: LLM_N_CANDIDATES }, () =>
      Promise.race([
        callLLM(customModelId, customApiKey, vibe),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("__timeout__")), LLM_TIMEOUT_MS)
        ),
      ])
    );
    const results = await Promise.allSettled(promises);
    const succeeded = results
      .filter((r): r is PromiseFulfilledResult<Awaited<ReturnType<typeof callLLM>>> => r.status === "fulfilled")
      .map((r) => r.value);

    const failed = results.filter((r) => r.status === "rejected").length;
    if (failed > 0 && succeeded.length > 0) {
      console.warn(`${failed}/${LLM_N_CANDIDATES} LLM calls failed, continuing with ${succeeded.length}`);
    }

    if (succeeded.length === 0) {
      // All LLM calls failed — fall back to fast model with warning
      console.warn("All LLM calls failed, falling back to fast model");
      onPhase?.("music");
      const resp = await generateFromVibe(vibe, "fast");
      return { ...resp, warning: "LLM calls failed — used fast model instead" };
    }

    if (succeeded.length === 1) {
      // Only 1 succeeded — skip CLAP, use it directly
      onPhase?.("music");
      return createFromConfig(vibe, succeeded[0].config, succeeded[0].justification, succeeded[0].title, succeeded[0].palette);
    }

    // Multiple candidates — send to backend for CLAP evaluation
    onPhase?.("evaluating");
    try {
      return await selectBestConfig(vibe, succeeded);
    } catch (err) {
      // CLAP evaluation failed — fall back to first candidate
      console.warn("CLAP evaluation failed, using first LLM candidate:", err);
      onPhase?.("music");
      return createFromConfig(vibe, succeeded[0].config, succeeded[0].justification, succeeded[0].title, succeeded[0].palette);
    }
  }
  onPhase?.("music");
  return generateFromVibe(vibe, model as "fast" | "fast_heavy" | "expressive");
}

export async function renderAudio(
  sessionId: string,
  duration: number = 60
): Promise<ArrayBuffer> {
  const res = await fetch(`${API_BASE}/render/${sessionId}?duration=${duration}`);
  if (!res.ok) throw new Error(`Render error: ${res.status}`);
  return res.arrayBuffer();
}

export async function updateParams(
  sessionId: string,
  updates: Record<string, unknown>
): Promise<{ config: Record<string, unknown>; palette: string[]; title: string }> {
  const res = await fetch(`${API_BASE}/update-params`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, updates }),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function createPlaylist(
  name: string,
  tracks: { vibe: string; duration?: number }[]
): Promise<{
  name: string;
  tracks: Array<{
    session_id: string;
    vibe: string;
    title: string;
    palette: string[];
    justification: string;
    config: Record<string, unknown>;
    duration: number;
  }>;
}> {
  const res = await fetch(`${API_BASE}/playlist/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, tracks }),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
