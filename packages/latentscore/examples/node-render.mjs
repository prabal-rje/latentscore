import { writeFileSync } from "node:fs";
import { createScore, encodeWavBytes } from "../dist/esm/index.js";

// Create a stateful ambient score and morph smoothly from morning desk to midnight rain.
const score = createScore("quiet morning desk", { seed: 42, glideSeconds: 4.0 });
const partA = score.fill(44100 * 4);

score.steer("rainy midnight coffee shop");
const partB = score.fill(44100 * 4);

const combined = new Float32Array(partA.length + partB.length);
combined.set(partA, 0);
combined.set(partB, partA.length);

const wav = encodeWavBytes(combined, { channels: 2, stereoWidth: 0.6, fadeOutSeconds: 0.5 });
writeFileSync("morning-to-rain.wav", wav);
console.log("Wrote morning-to-rain.wav", wav.byteLength, "bytes");
