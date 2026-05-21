import { z } from "zod/v4";

export const PresetSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  vibe: z.string(),
  icon: z.string(),
});

export type Preset = z.infer<typeof PresetSchema>;

export const GenerateResponseSchema = z.object({
  session_id: z.string(),
  title: z.string(),
  justification: z.string(),
  palette: z.array(z.string()),
  config: z.record(z.string(), z.unknown()),
});

export type GenerateResponse = z.infer<typeof GenerateResponseSchema>;

export const ParamOptionsSchema = z.record(z.string(), z.array(z.string()));
export type ParamOptions = z.infer<typeof ParamOptionsSchema>;

export type ModelChoice = "fast" | "fast_heavy" | "expressive" | "custom";

export const CapabilitiesSchema = z.object({
  models: z.array(z.string()),
  expressive_available: z.boolean(),
});

export type Capabilities = z.infer<typeof CapabilitiesSchema>;

export interface PlaylistTrack {
  id: string;                         // unique client-side ID for tracking
  sessionId: string;
  vibe: string;
  title: string;
  palette: string[];
  justification: string;
  config: Record<string, unknown>;
  duration: number;
  status: "pending" | "ready" | "error";
  errorMessage?: string;
}

export interface Playlist {
  name: string;
  tracks: PlaylistTrack[];
}

// The tunable parameters (the ones that make sense as sliders/selects)
export const TUNABLE_PARAMS = [
  "tempo",
  "brightness",
  "space",
  "density",
  "motion",
  "echo",
  "stereo",
  "human",
  "grain",
  "attack",
  "bass",
  "pad",
  "melody",
  "rhythm",
  "texture",
  "accent",
  "root",
  "mode",
] as const;

export type TunableParam = (typeof TUNABLE_PARAMS)[number];
