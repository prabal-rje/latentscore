/**
 * Dedicated Web Worker for browser-side MiniLM text embeddings, quantized int8
 * nearest-neighbor retrieval, and semantic similarity-gradient Auto Drift over
 * 14,035 curated musical soundscapes.
 */

import { env, pipeline } from "@huggingface/transformers";

// Browser cache API keeps the ~23 MB quantized ONNX weights locally after first visit.
env.allowLocalModels = false;
env.useBrowserCache = true;

interface Manifest {
  model: string;
  rows: number;
  dims: number;
  embeddings: string;
  scales: string;
  metadata: string;
  embeddings_sha256?: string;
}

interface MetadataPayload {
  fields: string[];
  vocabs: (string | number | boolean)[][];
  rows: [string, string[], number[]][];
}

interface MatchRequest {
  type: "match";
  id: number;
  vibe: string;
}

interface DriftNextRequest {
  type: "drift_next";
  id: number;
  currentRowIndex: number | null;
  currentConfig?: Record<string, unknown>;
  visitedIndices: number[];
  visitedTitles: string[];
  reset?: boolean;
}

interface WarmupRequest {
  type: "warmup";
}

type WorkerRequest = MatchRequest | DriftNextRequest | WarmupRequest;

interface ExtractorOutput {
  data: Float32Array;
}

type FeatureExtractor = (
  text: string,
  options: { pooling: "mean"; normalize: boolean }
) => Promise<ExtractorOutput>;

interface LoadedIndex {
  embeddings: Int8Array;
  scales: Float32Array;
  metadata: MetadataPayload;
  rows: number;
  dims: number;
}

interface LoadedResources extends LoadedIndex {
  extractor: FeatureExtractor;
}

let indexPromise: Promise<LoadedIndex> | null = null;
let extractorPromise: Promise<FeatureExtractor> | null = null;
let driftMomentum: Float32Array | null = null;

function postStatus(stage: string, progress: number | null, detail: string) {
  self.postMessage({ type: "status", source: "vibe", stage, progress, detail });
}

async function fetchIndex(): Promise<LoadedIndex> {
  postStatus("loading", 0.05, "Downloading vibe index (4.5 MB, cached locally)");
  const base = new URL("/data/", self.location.origin);
  const manifestRes = await fetch(new URL("manifest.json", base), { cache: "no-cache" });
  if (!manifestRes.ok) throw new Error(`Failed to fetch manifest: ${manifestRes.status}`);
  const manifest = (await manifestRes.json()) as Manifest;
  if (
    !manifest ||
    typeof manifest.rows !== "number" ||
    typeof manifest.dims !== "number" ||
    manifest.rows <= 0 ||
    manifest.dims <= 0
  ) {
    throw new Error("Invalid embedding manifest");
  }
  const versionQuery = manifest.embeddings_sha256
    ? `?v=${encodeURIComponent(manifest.embeddings_sha256)}`
    : "";

  const [embBuf, scaleBuf, metaRes] = await Promise.all([
    fetch(new URL(`${manifest.embeddings}${versionQuery}`, base)).then((r) => {
      if (!r.ok) throw new Error(`Failed to fetch embeddings: ${r.status}`);
      return r.arrayBuffer();
    }),
    fetch(new URL(`${manifest.scales}${versionQuery}`, base)).then((r) => {
      if (!r.ok) throw new Error(`Failed to fetch scales: ${r.status}`);
      return r.arrayBuffer();
    }),
    fetch(new URL(`${manifest.metadata}${versionQuery}`, base)),
  ]);
  if (!metaRes.ok) throw new Error(`Failed to fetch metadata: ${metaRes.status}`);
  const metadata = (await metaRes.json()) as MetadataPayload;

  const embeddings = new Int8Array(embBuf);
  const scales = new Float32Array(scaleBuf);
  if (embeddings.length !== manifest.rows * manifest.dims) {
    throw new Error("Embedding matrix size does not match manifest");
  }
  if (scales.length !== manifest.rows) {
    throw new Error("Scale vector size does not match manifest");
  }
  if (!Array.isArray(metadata.rows) || metadata.rows.length !== manifest.rows) {
    throw new Error("Metadata row count does not match manifest");
  }

  return {
    embeddings,
    scales,
    metadata,
    rows: manifest.rows,
    dims: manifest.dims,
  };
}

function ensureIndexLoaded(): Promise<LoadedIndex> {
  if (!indexPromise) {
    indexPromise = fetchIndex().catch((err) => {
      indexPromise = null;
      throw err;
    });
  }
  return indexPromise;
}

function ensureExtractorLoaded(): Promise<FeatureExtractor> {
  if (!extractorPromise) {
    postStatus("loading", 0.15, "Loading MiniLM encoder (23 MB, cached locally)");
    extractorPromise = (
      pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
        dtype: "q8",
        progress_callback: (info: { status?: string; progress?: number }) => {
          if (typeof info.progress === "number") {
            const pct = Math.min(0.95, 0.15 + (info.progress / 100) * 0.8);
            postStatus("loading", pct, `Loading vibe encoder (${Math.round(info.progress)}%)`);
          }
        },
      }) as Promise<unknown>
    )
      .then((raw) => {
        postStatus("ready", 1.0, "Vibe encoder ready");
        return raw as FeatureExtractor;
      })
      .catch((err) => {
        extractorPromise = null;
        postStatus("error", null, err instanceof Error ? err.message : "Vibe encoder failed");
        throw err;
      });
  }
  return extractorPromise;
}

async function ensureAllLoaded(): Promise<LoadedResources> {
  const [index, extractor] = await Promise.all([
    ensureIndexLoaded(),
    ensureExtractorLoaded(),
  ]);
  return { ...index, extractor };
}

function findBestRow(
  embeddings: Int8Array,
  scales: Float32Array,
  rows: number,
  dims: number,
  queryVec: Float32Array
): number {
  let bestIdx = 0;
  let bestScore = -Infinity;
  for (let r = 0; r < rows; r++) {
    const base = r * dims;
    let dot = 0;
    for (let d = 0; d < dims; d++) {
      dot += embeddings[base + d] * queryVec[d];
    }
    const score = dot * scales[r];
    if (score > bestScore) {
      bestScore = score;
      bestIdx = r;
    }
  }
  return bestIdx;
}

function decodeRow(metadata: MetadataPayload, rowIdx: number) {
  const [title, palette, indices] = metadata.rows[rowIdx];
  const config: Record<string, unknown> = {};
  for (let i = 0; i < metadata.fields.length; i++) {
    config[metadata.fields[i]] = metadata.vocabs[i][indices[i]];
  }
  return { title, palette, config };
}

function dequantizeRow(
  embeddings: Int8Array,
  scales: Float32Array,
  dims: number,
  rowIdx: number
): Float32Array {
  const vec = new Float32Array(dims);
  const base = rowIdx * dims;
  const scale = scales[rowIdx];
  let normSq = 0;
  for (let d = 0; d < dims; d++) {
    const v = embeddings[base + d] * scale;
    vec[d] = v;
    normSq += v * v;
  }
  const invNorm = normSq > 1e-12 ? 1 / Math.sqrt(normSq) : 1;
  for (let d = 0; d < dims; d++) vec[d] *= invNorm;
  return vec;
}

function findAnchorRowFromConfig(
  metadata: MetadataPayload,
  currentConfig?: Record<string, unknown>
): number {
  if (!currentConfig || Object.keys(currentConfig).length === 0) {
    return Math.floor(Math.random() * metadata.rows.length);
  }
  const weightedFields: [string, number][] = [
    ["mode", 3.0],
    ["tempo", 2.5],
    ["brightness", 2.0],
    ["space", 2.0],
    ["pad", 1.8],
    ["bass", 1.5],
    ["rhythm", 1.5],
    ["texture", 1.3],
    ["melody", 1.2],
    ["density", 1.0],
  ];
  const fieldIndices: { idx: number; weight: number; target: string }[] = [];
  for (const [field, weight] of weightedFields) {
    const idx = metadata.fields.indexOf(field);
    if (idx !== -1 && currentConfig[field] !== undefined) {
      fieldIndices.push({ idx, weight, target: String(currentConfig[field]) });
    }
  }
  if (fieldIndices.length === 0) {
    return Math.floor(Math.random() * metadata.rows.length);
  }

  let bestScore = -1;
  const topCandidates: number[] = [];
  for (let r = 0; r < metadata.rows.length; r++) {
    const [, , indices] = metadata.rows[r];
    let score = 0;
    for (const f of fieldIndices) {
      if (String(metadata.vocabs[f.idx][indices[f.idx]]) === f.target) {
        score += f.weight;
      }
    }
    if (score > bestScore) {
      bestScore = score;
      topCandidates.length = 0;
      topCandidates.push(r);
    } else if (score === bestScore && topCandidates.length < 32) {
      topCandidates.push(r);
    }
  }
  return topCandidates[Math.floor(Math.random() * topCandidates.length)] ?? 0;
}

interface Candidate {
  idx: number;
  sim: number;
  score: number;
}

function updateDriftMomentum(curVec: Float32Array, nextVec: Float32Array): void {
  const dims = curVec.length;
  if (!driftMomentum || driftMomentum.length !== dims) {
    driftMomentum = new Float32Array(dims);
  }
  let normSq = 0;
  for (let d = 0; d < dims; d++) {
    const step = nextVec[d] - curVec[d];
    const blended = 0.72 * driftMomentum[d] + 0.28 * step;
    driftMomentum[d] = blended;
    normSq += blended * blended;
  }
  if (normSq > 1e-12) {
    const inv = 1 / Math.sqrt(normSq);
    for (let d = 0; d < dims; d++) driftMomentum[d] *= inv;
  }
}

/**
 * Step along a smooth similarity gradient across 14,035 curated soundscapes.
 * Prefers neighbors around cosine similarity ~0.68 with forward semantic
 * momentum so genres evolve fluidly without jumping harshly or orbiting one spot.
 */
function pickNextGradientRow(
  index: LoadedIndex,
  currentRowIndex: number | null,
  currentConfig: Record<string, unknown> | undefined,
  visitedIndices: number[],
  visitedTitles: string[],
  reset: boolean
): { rowIndex: number; similarity: number | null; title: string; palette: string[]; config: Record<string, unknown> } {
  const { embeddings, scales, metadata, rows, dims } = index;
  if (reset) driftMomentum = null;

  const visitedSet = new Set(visitedIndices);
  const titleSet = new Set(visitedTitles.map((t) => t.trim().toLowerCase()).filter(Boolean));

  // If no track is anchored yet, anchor on current config or pick a fresh seed row.
  let anchorIdx = currentRowIndex;
  if (anchorIdx === null || anchorIdx < 0 || anchorIdx >= rows) {
    anchorIdx = findAnchorRowFromConfig(metadata, currentConfig);
    if (!currentConfig || Object.keys(currentConfig).length === 0) {
      // Fresh Auto Drift launch from idle: play the seed row directly!
      const decoded = decodeRow(metadata, anchorIdx);
      return { rowIndex: anchorIdx, similarity: null, ...decoded };
    }
  }

  visitedSet.add(anchorIdx);
  const curVec = dequantizeRow(embeddings, scales, dims, anchorIdx);
  const [, , curCfgIdx] = metadata.rows[anchorIdx];
  const modeField = metadata.fields.indexOf("mode");
  const padField = metadata.fields.indexOf("pad");
  const bassField = metadata.fields.indexOf("bass");
  const tempoField = metadata.fields.indexOf("tempo");

  const mom = driftMomentum;
  const targetSim = 0.68;
  const invTwoSigmaSq = 1 / (2 * 0.14 * 0.14);
  const topK: Candidate[] = [];
  const MAX_K = 16;

  let fallbackIdx = (anchorIdx + 1) % rows;
  let fallbackSim = -Infinity;

  for (let r = 0; r < rows; r++) {
    if (visitedSet.has(r)) continue;
    const [rawTitle, , rCfgIdx] = metadata.rows[r];
    const normTitle = rawTitle.trim().toLowerCase();
    if (normTitle && titleSet.has(normTitle)) continue;

    // Require at least one core sonic element to evolve across the step.
    if (
      modeField !== -1 &&
      padField !== -1 &&
      bassField !== -1 &&
      tempoField !== -1 &&
      rCfgIdx[modeField] === curCfgIdx[modeField] &&
      rCfgIdx[padField] === curCfgIdx[padField] &&
      rCfgIdx[bassField] === curCfgIdx[bassField] &&
      rCfgIdx[tempoField] === curCfgIdx[tempoField]
    ) {
      continue;
    }

    const base = r * dims;
    const scale = scales[r];
    let dot = 0;
    let momDot = 0;
    if (mom) {
      for (let d = 0; d < dims; d++) {
        const val = embeddings[base + d];
        dot += val * curVec[d];
        momDot += val * mom[d];
      }
      dot *= scale;
      momDot *= scale;
    } else {
      for (let d = 0; d < dims; d++) {
        dot += embeddings[base + d] * curVec[d];
      }
      dot *= scale;
    }

    if (dot > fallbackSim) {
      fallbackSim = dot;
      fallbackIdx = r;
    }

    if (dot < 0.42 || dot > 0.89) continue;

    const diff = dot - targetSim;
    const simAffinity = Math.exp(-diff * diff * invTwoSigmaSq);
    const score = simAffinity + (mom ? 0.34 * momDot : 0);

    if (topK.length < MAX_K || score > topK[topK.length - 1].score) {
      let insertAt = topK.length;
      while (insertAt > 0 && topK[insertAt - 1].score < score) {
        insertAt--;
      }
      topK.splice(insertAt, 0, { idx: r, sim: dot, score });
      if (topK.length > MAX_K) topK.pop();
    }
  }

  let chosen: Candidate;
  if (topK.length > 0) {
    // Weighted softmax sample among the top-16 smooth gradient neighbors.
    const maxScore = topK[0].score;
    let weightSum = 0;
    const weights = topK.map((c) => {
      const w = Math.exp((c.score - maxScore) * 4.5);
      weightSum += w;
      return w;
    });
    let pick = Math.random() * weightSum;
    chosen = topK[0];
    for (let i = 0; i < topK.length; i++) {
      pick -= weights[i];
      if (pick <= 0) {
        chosen = topK[i];
        break;
      }
    }
  } else {
    chosen = { idx: fallbackIdx, sim: Math.max(0, fallbackSim), score: 0 };
  }

  const nextVec = dequantizeRow(embeddings, scales, dims, chosen.idx);
  updateDriftMomentum(curVec, nextVec);
  const decoded = decodeRow(metadata, chosen.idx);
  return {
    rowIndex: chosen.idx,
    similarity: Math.max(0, Math.min(1, chosen.sim)),
    ...decoded,
  };
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  if (msg.type === "warmup") {
    try {
      await ensureAllLoaded();
    } catch {
      // Status already posted; subsequent match calls can retry or fall back.
    }
    return;
  }

  if (msg.type === "drift_next") {
    try {
      const index = await ensureIndexLoaded();
      const result = pickNextGradientRow(
        index,
        msg.currentRowIndex,
        msg.currentConfig,
        msg.visitedIndices,
        msg.visitedTitles,
        Boolean(msg.reset)
      );
      self.postMessage({ type: "drift_result", id: msg.id, result });
    } catch (err) {
      self.postMessage({
        type: "drift_error",
        id: msg.id,
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (msg.type === "match") {
    try {
      const res = await ensureAllLoaded();
      const tensor = await res.extractor(msg.vibe, { pooling: "mean", normalize: true });
      const bestIdx = findBestRow(res.embeddings, res.scales, res.rows, res.dims, tensor.data);
      const matched = decodeRow(res.metadata, bestIdx);
      self.postMessage({
        type: "match_result",
        id: msg.id,
        result: { ...matched, rowIndex: bestIdx },
      });
    } catch (err) {
      self.postMessage({
        type: "match_error",
        id: msg.id,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }
};
