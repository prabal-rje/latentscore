/**
 * Dedicated Web Worker running the exact latentscore DSP synthesizer inside
 * Pyodide + NumPy, with a fast JS Float64Array biquad filter hook so SciPy is
 * never downloaded. Supports request cancellation so rapid track switches skip
 * obsolete queued renders.
 */

const PYODIDE_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.5/full/";
const PYODIDE_MJS_URL = `${PYODIDE_INDEX_URL}pyodide.mjs`;

interface PyProxyLike {
  toJs: () => Float32Array;
  destroy: () => void;
}

interface PyodideLike {
  loadPackage: (packages: string | string[]) => Promise<void>;
  runPythonAsync: (code: string) => Promise<unknown>;
  globals: {
    get: (name: string) => unknown;
  };
}

type RenderFn = (
  configJson: string,
  duration: number,
  tOffset: number,
  normalize: boolean,
  seed: number | null
) => PyProxyLike;

interface RenderRequest {
  type: "render";
  id: number;
  configJson: string;
  duration: number;
  tOffset: number;
  normalize: boolean;
  seed: number | null;
}

interface CancelRequest {
  type: "cancel";
  id: number;
}

interface WarmupRequest {
  type: "warmup";
}

type SynthWorkerRequest = RenderRequest | CancelRequest | WarmupRequest;

// Fast JS Direct-Form-II Transposed biquad called by latentscore_synth.py via `import js`.
function latentscoreBiquad(
  x: Float64Array,
  b0: number,
  b1: number,
  b2: number,
  a1: number,
  a2: number
): Float64Array {
  const n = x.length;
  const out = new Float64Array(n);
  let z1 = 0.0;
  let z2 = 0.0;
  for (let i = 0; i < n; i++) {
    const xi = x[i];
    const yi = b0 * xi + z1;
    z1 = b1 * xi - a1 * yi + z2;
    z2 = b2 * xi - a2 * yi;
    out[i] = yi;
  }
  return out;
}

(globalThis as unknown as { _latentscore_biquad?: typeof latentscoreBiquad })._latentscore_biquad =
  latentscoreBiquad;

let cachedPyodide: PyodideLike | null = null;
let initPromise: Promise<RenderFn> | null = null;
const cancelledIds = new Set<number>();

interface WorkerScopeLike {
  postMessage: (message: unknown, transfer?: Transferable[]) => void;
  onmessage: ((event: MessageEvent<SynthWorkerRequest>) => void) | null;
}

const workerSelf = self as unknown as WorkerScopeLike;

function postStatus(stage: string, progress: number | null, detail: string) {
  workerSelf.postMessage({ type: "status", source: "synth", stage, progress, detail });
}

async function initSynth(): Promise<RenderFn> {
  if (!cachedPyodide) {
    postStatus("loading", 0.1, "Booting browser synth engine (first load)");
    const pyodideModule = (await import(/* @vite-ignore */ PYODIDE_MJS_URL)) as {
      loadPyodide: (opts: { indexURL: string }) => Promise<PyodideLike>;
    };

    postStatus("loading", 0.35, "Loading WebAssembly Python runtime");
    cachedPyodide = await pyodideModule.loadPyodide({ indexURL: PYODIDE_INDEX_URL });
  }

  postStatus("loading", 0.65, "Loading NumPy DSP package");
  const synthUrl = new URL("/py/latentscore_synth.py", self.location.origin);
  const [, synthCode] = await Promise.all([
    cachedPyodide.loadPackage("numpy"),
    fetch(synthUrl, { cache: "no-cache" }).then((r) => {
      if (!r.ok) throw new Error(`Failed to fetch synth module: ${r.status}`);
      return r.text();
    }),
  ]);

  postStatus("loading", 0.9, "Initializing LatentScore synthesizer");
  await cachedPyodide.runPythonAsync(synthCode);
  const renderFn = cachedPyodide.globals.get("render_pcm_f32") as RenderFn;

  // Quick 50ms warmup so caches and numpy ufuncs settle before first track.
  const warmProxy = renderFn("{}", 0.05, 0.0, true, 1);
  try {
    // Discard result; only exercising code path.
  } finally {
    warmProxy.destroy();
  }

  postStatus("ready", 1.0, "Browser synth ready");
  return renderFn;
}

function ensureInit(): Promise<RenderFn> {
  if (!initPromise) {
    initPromise = initSynth().catch((err) => {
      initPromise = null;
      postStatus("error", null, err instanceof Error ? err.message : "Synth init failed");
      throw err;
    });
  }
  return initPromise;
}

workerSelf.onmessage = async (event: MessageEvent<SynthWorkerRequest>) => {
  const msg = event.data;
  if (msg.type === "cancel") {
    cancelledIds.add(msg.id);
    if (cancelledIds.size > 256) {
      const oldest = cancelledIds.values().next().value;
      if (oldest !== undefined) cancelledIds.delete(oldest);
    }
    return;
  }

  if (msg.type === "warmup") {
    try {
      await ensureInit();
    } catch {
      // Status message already emitted; next render call will retry.
    }
    return;
  }

  if (msg.type === "render") {
    if (cancelledIds.delete(msg.id)) {
      workerSelf.postMessage({ type: "render_cancelled", id: msg.id });
      return;
    }
    try {
      const renderFn = await ensureInit();
      if (cancelledIds.delete(msg.id)) {
        workerSelf.postMessage({ type: "render_cancelled", id: msg.id });
        return;
      }
      const proxy = renderFn(msg.configJson, msg.duration, msg.tOffset, msg.normalize, msg.seed);
      let pcm: Float32Array;
      try {
        // Pyodide 0.27.5 toJs() returns a fresh copied Float32Array for NumPy float32 arrays.
        const converted = proxy.toJs();
        pcm = converted instanceof Float32Array ? converted : new Float32Array(converted);
      } finally {
        proxy.destroy();
      }
      const buffer =
        pcm.buffer instanceof ArrayBuffer &&
        pcm.byteOffset === 0 &&
        pcm.byteLength === pcm.buffer.byteLength
          ? pcm.buffer
          : pcm.slice().buffer;
      workerSelf.postMessage({ type: "render_result", id: msg.id, buffer }, [buffer]);
    } catch (err) {
      workerSelf.postMessage({
        type: "render_error",
        id: msg.id,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }
};
