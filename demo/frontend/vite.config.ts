import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// Transformers.js v4 loads ONNX Runtime WASM from jsDelivr CDN via wasmPaths,
// but Rollup still emits a dead-code fallback .wasm asset (~26 MB) that exceeds
// Cloudflare Pages' 25 MiB per-file limit. Strip it from the build output.
function stripUnusedOrtWasm(): Plugin {
  return {
    name: "strip-unused-ort-wasm",
    generateBundle(_options, bundle) {
      for (const fileName of Object.keys(bundle)) {
        if (fileName.includes("ort-wasm-") && fileName.endsWith(".wasm")) {
          delete bundle[fileName];
        }
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), stripUnusedOrtWasm()],
  worker: {
    format: "es",
    plugins: () => [stripUnusedOrtWasm()],
  },
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
