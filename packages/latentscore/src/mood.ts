/**
 * Zero-dependency deterministic text-to-mood mapping (matching spec/mood_lexicon.json),
 * optional external int8 embedding-index loader, and momentum-guided AutoDrift.
 */

import {
  ConfigInput,
  DEFAULT_CONFIG,
  ModeName,
  MusicConfig,
  RootNote,
  SynthConfig,
  interpolateConfigs,
  toInternalConfig,
  toPublicConfig,
} from "./config.js";
import { MOOD_LEXICON_RULES } from "./lexicon.js";

const ROOT_CIRCLE: readonly RootNote[] = [
  "c",
  "g",
  "d",
  "a",
  "e",
  "b",
  "f#",
  "c#",
  "g#",
  "d#",
  "a#",
  "f",
];

const DEFAULT_DRIFT_MOODS: readonly string[] = [
  "quiet morning desk",
  "warm cedar zen meditation",
  "rainy midnight coffee shop",
  "vast deep space nebula with stars",
  "soft ocean mist and slow drift",
  "gentle golden sunrise",
];

const INFLECTION_SUFFIXES = new Set(["s", "es", "y", "ed", "ing", "ly", "er", "ness"]);

/**
 * Map natural-language text deterministically to a MusicConfig with zero downloads.
 * Uses word-boundary + inflection matching (matching Python and Rust) so ordinary
 * words like "office", "breakfast", or "window" never trigger unrelated mood rules.
 */
export function moodFromText(text: string, base: MusicConfig = DEFAULT_CONFIG): MusicConfig {
  const tokens = text.toLowerCase().match(/[a-z]+/g) ?? [];
  const next: Record<string, unknown> = { ...base, schema_version: 1 };

  for (const group of MOOD_LEXICON_RULES) {
    for (const entry of group.entries) {
      let matched = false;
      for (const kw of entry.keywords) {
        const allowInflection = kw.length >= 4;
        for (const tok of tokens) {
          if (
            tok === kw ||
            (allowInflection &&
              tok.startsWith(kw) &&
              INFLECTION_SUFFIXES.has(tok.slice(kw.length)))
          ) {
            matched = true;
            break;
          }
        }
        if (matched) break;
      }
      if (matched) {
        for (const [field, value] of Object.entries(entry.set)) {
          next[field] = value;
        }
        if (group.first_match) break;
      }
    }
  }

  return next as unknown as MusicConfig;
}

export interface EmbeddingIndexManifest {
  readonly version: number;
  readonly model: string;
  readonly rows: number;
  readonly dims: number;
  readonly embeddings: string;
  readonly scales: string;
  readonly metadata: string;
}

export interface IndexedMood {
  readonly index: number;
  readonly title: string;
  readonly palette: readonly string[];
  readonly config: MusicConfig;
  readonly score: number;
}

interface RawMetadata {
  readonly fields: readonly string[];
  readonly vocabs: readonly (readonly (string | number | boolean)[])[];
  readonly rows: readonly [string, readonly string[], readonly number[]][];
}

/**
 * Optional loader for the quantized 14,035-mood int8 embedding map
 * (not bundled in the npm package; load from a hosted /data/ directory or local path).
 */
export class EmbeddingIndex {
  readonly rows: number;
  readonly dims: number;
  private readonly quantized: Int8Array;
  private readonly scales: Float32Array;
  private readonly metadata: RawMetadata;

  constructor(
    rows: number,
    dims: number,
    quantized: Int8Array,
    scales: Float32Array,
    metadata: RawMetadata,
  ) {
    if (quantized.length !== rows * dims) {
      throw new Error(`Expected ${rows * dims} int8 weights, got ${quantized.length}`);
    }
    if (scales.length !== rows) {
      throw new Error(`Expected ${rows} float32 scales, got ${scales.length}`);
    }
    this.rows = rows;
    this.dims = dims;
    this.quantized = quantized;
    this.scales = scales;
    this.metadata = metadata;
  }

  /** Fetch manifest, embeddings, scales, and metadata from a base URL (e.g. "/data"). */
  static async loadFromUrl(baseUrl: string): Promise<EmbeddingIndex> {
    const root = baseUrl.replace(/\/+$/, "");
    const manifestRes = await fetch(`${root}/manifest.json`);
    if (!manifestRes.ok) {
      throw new Error(`Failed to fetch manifest.json: HTTP ${manifestRes.status}`);
    }
    const manifest = (await manifestRes.json()) as EmbeddingIndexManifest;

    const [embRes, scaleRes, metaRes] = await Promise.all([
      fetch(`${root}/${manifest.embeddings}`),
      fetch(`${root}/${manifest.scales}`),
      fetch(`${root}/${manifest.metadata}`),
    ]);
    if (!embRes.ok || !scaleRes.ok || !metaRes.ok) {
      throw new Error("Failed to fetch embedding index binary/metadata files");
    }

    const [embBuf, scaleBuf, metadata] = await Promise.all([
      embRes.arrayBuffer(),
      scaleRes.arrayBuffer(),
      metaRes.json() as Promise<RawMetadata>,
    ]);

    return new EmbeddingIndex(
      manifest.rows,
      manifest.dims,
      new Int8Array(embBuf),
      new Float32Array(scaleBuf),
      metadata,
    );
  }

  /** Decode a row's MusicConfig, title, and palette by row index. */
  get(rowIndex: number, score = 1.0): IndexedMood {
    const safeIdx = ((rowIndex % this.rows) + this.rows) % this.rows;
    const row = this.metadata.rows[safeIdx]!;
    const [title, palette, indices] = row;
    const obj: Record<string, unknown> = { ...DEFAULT_CONFIG, schema_version: 1 };
    for (let i = 0; i < this.metadata.fields.length; i++) {
      const field = this.metadata.fields[i]!;
      const vocab = this.metadata.vocabs[i]!;
      const vIdx = indices[i]!;
      if (vIdx >= 0 && vIdx < vocab.length) {
        obj[field] = vocab[vIdx];
      }
    }
    return {
      index: safeIdx,
      title,
      palette,
      config: obj as unknown as MusicConfig,
      score,
    };
  }

  /** Decode a row's unit vector into a Float32Array. */
  vectorAt(rowIndex: number): Float32Array {
    const safeIdx = ((rowIndex % this.rows) + this.rows) % this.rows;
    const scale = this.scales[safeIdx]!;
    const offset = safeIdx * this.dims;
    const out = new Float32Array(this.dims);
    for (let d = 0; d < this.dims; d++) {
      out[d] = this.quantized[offset + d]! * scale;
    }
    return out;
  }

  /** Find the nearest row to a unit query embedding vector. */
  searchVector(query: Float32Array): IndexedMood {
    if (query.length !== this.dims) {
      throw new Error(`Query dim ${query.length} !== index dim ${this.dims}`);
    }
    let bestIdx = 0;
    let bestScore = -Infinity;
    for (let r = 0; r < this.rows; r++) {
      const off = r * this.dims;
      let dot = 0;
      for (let d = 0; d < this.dims; d++) {
        dot += this.quantized[off + d]! * query[d]!;
      }
      const sim = dot * this.scales[r]!;
      if (sim > bestScore) {
        bestScore = sim;
        bestIdx = r;
      }
    }
    return this.get(bestIdx, bestScore);
  }

  /**
   * Pick a smooth similarity-gradient neighbor from `currentIndex` with momentum,
   * avoiding recently visited moods.
   */
  pickSmoothNeighbor(
    currentIndex: number,
    prevVector: Float32Array | null,
    visited: ReadonlySet<number>,
    rng: () => number = Math.random,
  ): { mood: IndexedMood; vector: Float32Array } {
    const curVec = this.vectorAt(currentIndex);
    const probe = new Float32Array(this.dims);
    let normSq = 0;
    for (let d = 0; d < this.dims; d++) {
      const momentum = prevVector ? 0.35 * (curVec[d]! - prevVector[d]!) : 0;
      const v = curVec[d]! + momentum;
      probe[d] = v;
      normSq += v * v;
    }
    const invNorm = normSq > 1e-12 ? 1 / Math.sqrt(normSq) : 1;
    for (let d = 0; d < this.dims; d++) probe[d]! *= invNorm;

    const topK: { idx: number; sim: number }[] = [];
    const K = 40;
    for (let r = 0; r < this.rows; r++) {
      if (r === currentIndex || visited.has(r)) continue;
      const off = r * this.dims;
      let dot = 0;
      for (let d = 0; d < this.dims; d++) {
        dot += this.quantized[off + d]! * probe[d]!;
      }
      const sim = dot * this.scales[r]!;
      if (sim > 0.96) continue;
      if (topK.length < K) {
        topK.push({ idx: r, sim });
        if (topK.length === K) topK.sort((a, b) => b.sim - a.sim);
      } else if (sim > topK[K - 1]!.sim) {
        topK[K - 1] = { idx: r, sim };
        for (let i = K - 1; i > 0 && topK[i]!.sim > topK[i - 1]!.sim; i--) {
          const tmp = topK[i]!;
          topK[i] = topK[i - 1]!;
          topK[i - 1] = tmp;
        }
      }
    }

    if (topK.length === 0) {
      const fallbackIdx = Math.floor(rng() * this.rows);
      return { mood: this.get(fallbackIdx, 0.5), vector: this.vectorAt(fallbackIdx) };
    }

    const pick = topK[Math.min(topK.length - 1, Math.floor(rng() * Math.min(18, topK.length)))]!;
    return { mood: this.get(pick.idx, pick.sim), vector: this.vectorAt(pick.idx) };
  }
}

function makeRng(seed: number): () => number {
  let state = (seed >>> 0) || 0x9e3779b9;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function configDistance(a: SynthConfig, b: SynthConfig): number {
  let d =
    Math.abs(a.tempo - b.tempo) * 2.0 +
    Math.abs(a.brightness - b.brightness) * 1.5 +
    Math.abs(a.space - b.space) * 1.5 +
    Math.abs(a.motion - b.motion) +
    Math.abs(a.density - b.density) * 0.4;
  if (a.mode !== b.mode) d += 1.2;
  if (a.root !== b.root) d += 0.5;
  if (a.rhythm !== b.rhythm) d += 0.6;
  if (a.texture !== b.texture) d += 0.5;
  return d;
}

/**
 * Endless smooth gradient mood drifter.
 * Works out of the box along harmonic circle-of-fifths + mood-anchor paths,
 * or walks the full 14,035-mood similarity graph when given an EmbeddingIndex.
 */
export class AutoDrift {
  private readonly index: EmbeddingIndex | null;
  private readonly rng: () => number;
  private currentRow = 0;
  private prevVector: Float32Array | null = null;
  private readonly visited = new Set<number>();
  private stepCount = 0;
  private currentConfig: SynthConfig;

  constructor(options?: {
    initial?: ConfigInput;
    index?: EmbeddingIndex;
    seed?: number;
  }) {
    this.index = options?.index ?? null;
    this.rng = makeRng(options?.seed ?? 2026);
    this.currentConfig = toInternalConfig(options?.initial ?? DEFAULT_CONFIG);
    if (this.index && this.index.rows > 0) {
      this.anchorToConfig(this.currentConfig);
    }
  }

  /** Re-anchor the drift cursor to the nearest row for `cfg` and reset momentum. */
  anchorToConfig(cfg: ConfigInput): void {
    const target = toInternalConfig(cfg);
    this.currentConfig = target;
    this.prevVector = null;
    if (!this.index || this.index.rows === 0) return;

    const stride = Math.max(1, Math.floor(this.index.rows / 256));
    let bestRow = 0;
    let bestDist = Infinity;
    for (let r = 0; r < this.index.rows; r += stride) {
      const cand = toInternalConfig(this.index.get(r).config);
      const d = configDistance(target, cand);
      if (d < bestDist) {
        bestDist = d;
        bestRow = r;
      }
    }
    this.currentRow = bestRow;
    this.visited.clear();
    this.visited.add(bestRow);
  }

  /** Advance one smooth step along the similarity / harmonic gradient. */
  next(from?: ConfigInput): {
    config: MusicConfig;
    synthConfig: SynthConfig;
    title?: string;
    palette?: readonly string[];
  } {
    if (from) {
      this.currentConfig = toInternalConfig(from);
    }
    this.stepCount += 1;

    if (this.index && this.index.rows > 0) {
      if (this.visited.size > 160) {
        this.visited.clear();
        this.visited.add(this.currentRow);
      }
      const { mood } = this.index.pickSmoothNeighbor(
        this.currentRow,
        this.prevVector,
        this.visited,
        this.rng,
      );
      this.prevVector = this.index.vectorAt(this.currentRow);
      this.currentRow = mood.index;
      this.visited.add(mood.index);
      const synthConfig = toInternalConfig(mood.config);
      this.currentConfig = synthConfig;
      return { config: mood.config, synthConfig, title: mood.title, palette: mood.palette };
    }

    const cur = this.currentConfig;
    const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));
    const rootIdx = ROOT_CIRCLE.indexOf(cur.root);
    const stepDir = this.rng() >= 0.5 ? 1 : -1;
    const nextRoot: RootNote =
      this.rng() < 0.6
        ? ROOT_CIRCLE[((rootIdx < 0 ? 0 : rootIdx) + stepDir + ROOT_CIRCLE.length) % ROOT_CIRCLE.length]!
        : cur.root;
    const modes: readonly ModeName[] = ["minor", "dorian", "major", "mixolydian"];
    const nextMode: ModeName =
      this.rng() < 0.35 ? modes[Math.floor(this.rng() * modes.length)]! : cur.mode;

    let nextSynth: SynthConfig;
    if (this.stepCount % 4 === 0) {
      const anchorName =
        DEFAULT_DRIFT_MOODS[Math.floor(this.stepCount / 4) % DEFAULT_DRIFT_MOODS.length]!;
      const anchor = toInternalConfig(moodFromText(anchorName));
      nextSynth = interpolateConfigs(cur, anchor, 0.45);
    } else {
      nextSynth = {
        ...cur,
        root: nextRoot,
        mode: nextMode,
        tempo: clamp(cur.tempo + (this.rng() * 0.16 - 0.08), 0.15, 0.85),
        brightness: clamp(cur.brightness + (this.rng() * 0.28 - 0.14), 0.12, 0.88),
        space: clamp(cur.space + (this.rng() * 0.24 - 0.12), 0.2, 0.95),
        motion: clamp(cur.motion + (this.rng() * 0.24 - 0.12), 0.12, 0.85),
        echo: clamp(cur.echo + (this.rng() * 0.24 - 0.12), 0.15, 0.88),
      };
    }

    this.currentConfig = nextSynth;
    return { config: toPublicConfig(nextSynth), synthConfig: nextSynth };
  }
}
