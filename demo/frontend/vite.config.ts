import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
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

const DEMO_TITLE = "LatentScore Live Demo: Free Browser Ambient Music Studio from Text";
const DEMO_DESCRIPTION =
  "Sketch procedural ambient music from text prompts directly in your browser. Zero GPU, 14,035 moods, live 18-parameter tuning, and endless similarity-gradient Auto Drift.";
const DEMO_URL = "https://latentscore.com/demo";

const DEMO_LD_JSON = JSON.stringify(
  {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebApplication",
        "@id": "https://latentscore.com/demo#webapp",
        name: "LatentScore Live Browser Studio",
        url: DEMO_URL,
        applicationCategory: "MultimediaApplication",
        operatingSystem: "Any",
        browserRequirements: "Requires Web Audio API and WebAssembly",
        isAccessibleForFree: true,
        offers: {
          "@type": "Offer",
          price: "0",
          priceCurrency: "USD",
        },
        description: DEMO_DESCRIPTION,
        creator: {
          "@type": "Person",
          name: "Prabal Gupta",
          url: "https://prabal.ca",
        },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          {
            "@type": "ListItem",
            position: 1,
            name: "LatentScore",
            item: "https://latentscore.com/",
          },
          {
            "@type": "ListItem",
            position: 2,
            name: "Live Browser Studio",
            item: DEMO_URL,
          },
        ],
      },
    ],
  },
  null,
  2,
);

const DEMO_SEO_FALLBACK = `
      <div style="max-width:740px;margin:0 auto;padding:48px 24px;font-family:'Plus Jakarta Sans',system-ui,sans-serif;color:#111318;line-height:1.6">
        <header style="display:flex;justify-content:space-between;align-items:center;margin-bottom:48px">
          <a href="/" style="font-family:'Fraunces',Georgia,serif;font-size:22px;color:#111318;text-decoration:none">Latent<em>Score</em></a>
          <nav aria-label="Studio navigation" style="display:flex;gap:18px;font-size:14px">
            <a href="/" style="color:#111318">Home</a>
            <a href="https://github.com/prabal-rje/latentscore" style="color:#111318">GitHub</a>
          </nav>
        </header>
        <main>
          <p style="font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:#4B5563">Live Browser Studio · Local WebAssembly DSP</p>
          <h1 style="font-family:'Fraunces',Georgia,serif;font-size:44px;font-weight:400;line-height:1.1;margin:12px 0 18px">Sketch ambient music from <em>plain words</em>.</h1>
          <p style="font-size:17px;color:#2D333E;margin:0 0 24px">${DEMO_DESCRIPTION}</p>
          <section aria-label="Studio features">
            <h2 style="font-size:16px;margin:20px 0 6px">Instant Presets and Text Prompts</h2>
            <p style="margin:0 0 14px;color:#4B5563">Type any mood or pick a preset to synthesize seamless procedural ambient audio locally in your browser.</p>
            <h2 style="font-size:16px;margin:20px 0 6px">18 Live Musical Controls</h2>
            <p style="margin:0 0 14px;color:#4B5563">Shape tempo, root key, mode, brightness, space, echo, density, bass, pad, melody, rhythm, texture, and stereo width in real time.</p>
            <h2 style="font-size:16px;margin:20px 0 6px">Endless Auto Drift Across 14,035 Moods</h2>
            <p style="margin:0 0 14px;color:#4B5563">Enable Auto mode to glide continuously from one mood to nearby soundscapes along an int8 embedding similarity gradient.</p>
          </section>
          <noscript><p style="margin-top:24px;padding:12px 16px;background:#F3F4F6;border-radius:8px">JavaScript and Web Audio are required to play live audio in the browser studio.</p></noscript>
        </main>
      </div>
    `;

// Emit dist/demo/index.html with dedicated /demo title, description, canonical,
// Open Graph tags, JSON-LD, and crawlable semantic HTML for Cloudflare Pages & Nginx.
function emitDemoSeoRoute(): Plugin {
  let outDir = "dist";
  return {
    name: "emit-demo-seo-route",
    apply: "build",
    configResolved(config) {
      outDir = config.build.outDir;
    },
    closeBundle() {
      const rootHtmlPath = resolve(outDir, "index.html");
      let html = readFileSync(rootHtmlPath, "utf-8");

      html = html
        .replace(/<title>[\s\S]*?<\/title>/, `<title>${DEMO_TITLE}</title>`)
        .replace(
          /<meta\s+name="description"\s+content="[\s\S]*?"\s*\/>/,
          `<meta name="description" content="${DEMO_DESCRIPTION}" />`,
        )
        .replace(
          /<link\s+rel="canonical"\s+href="https:\/\/latentscore\.com\/"\s*\/>/,
          `<link rel="canonical" href="${DEMO_URL}" />`,
        )
        .replace(
          /<meta\s+property="og:url"\s+content="https:\/\/latentscore\.com\/"\s*\/>/,
          `<meta property="og:url" content="${DEMO_URL}" />`,
        )
        .replace(
          /<meta\s+property="og:title"\s+content="[\s\S]*?"\s*\/>/,
          `<meta property="og:title" content="${DEMO_TITLE}" />`,
        )
        .replace(
          /<meta\s+property="og:description"\s+content="[\s\S]*?"\s*\/>/,
          `<meta property="og:description" content="${DEMO_DESCRIPTION}" />`,
        )
        .replace(
          /<meta\s+name="twitter:title"\s+content="[\s\S]*?"\s*\/>/,
          `<meta name="twitter:title" content="${DEMO_TITLE}" />`,
        )
        .replace(
          /<meta\s+name="twitter:description"\s+content="[\s\S]*?"\s*\/>/,
          `<meta name="twitter:description" content="${DEMO_DESCRIPTION}" />`,
        )
        .replace(
          /<script id="ld-json" type="application\/ld\+json">[\s\S]*?<\/script>/,
          `<script id="ld-json" type="application/ld+json">\n${DEMO_LD_JSON}\n    </script>`,
        )
        .replace(
          /<!--SEO_ROOT_START-->[\s\S]*?<!--SEO_ROOT_END-->/,
          `<!--SEO_ROOT_START-->${DEMO_SEO_FALLBACK}<!--SEO_ROOT_END-->`,
        );

      const demoDir = resolve(outDir, "demo");
      mkdirSync(demoDir, { recursive: true });
      writeFileSync(resolve(demoDir, "index.html"), html, "utf-8");
    },
  };
}

export default defineConfig({
  plugins: [react(), stripUnusedOrtWasm(), emitDemoSeoRoute()],
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
