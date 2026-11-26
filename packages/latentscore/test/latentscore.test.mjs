import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import {
  DEFAULT_CONFIG,
  SAMPLE_RATE,
  applyAdsr,
  applyHumanize,
  applyMasterHallReverb,
  applyUpdate,
  assemble,
  createScore,
  encodeWavBytes,
  fadeIntoNothingness,
  generateSine,
  moodFromText,
  render,
  step,
  toInternalConfig,
} from "../dist/esm/index.js";
import { MOOD_LEXICON_RULES } from "../dist/esm/lexicon.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const pkgDir = join(__dirname, "..");
const repoSpecDir = join(pkgDir, "..", "..", "spec");

test("bundled lexicon rules match root spec/mood_lexicon.json exactly", () => {
  const rootSpec = JSON.parse(readFileSync(join(repoSpecDir, "mood_lexicon.json"), "utf8"));
  const pkgSpec = JSON.parse(readFileSync(join(pkgDir, "spec", "mood_lexicon.json"), "utf8"));
  assert.deepEqual(pkgSpec, rootSpec);
  assert.deepEqual(MOOD_LEXICON_RULES, rootSpec.rules);
});

test("moodFromText matches every golden_texts case including word boundaries", () => {
  const spec = JSON.parse(readFileSync(join(repoSpecDir, "mood_lexicon.json"), "utf8"));
  for (const item of spec.golden_texts) {
    const cfg = moodFromText(item.text);
    for (const [k, expected] of Object.entries(item.expect)) {
      assert.equal(cfg[k], expected, `${item.text} -> ${k}`);
    }
  }
});

test("applyUpdate supports Step deltas and rejects unknown keys", () => {
  const next = applyUpdate(DEFAULT_CONFIG, { tempo: step(2), space: step(-1) });
  assert.equal(next.tempo, "very_fast");
  assert.equal(next.space, "small");
  assert.throws(() => applyUpdate(DEFAULT_CONFIG, { bogus: "nope" }), /Unknown/);
});

test("ADSR cases always start and end at exact zero", () => {
  const spec = JSON.parse(readFileSync(join(repoSpecDir, "golden_cases.json"), "utf8"));
  for (const c of spec.adsr_cases) {
    const ones = new Float64Array(c.samples).fill(1.0);
    const out = applyAdsr(ones, c.attack, c.decay, c.sustain, c.release, SAMPLE_RATE);
    assert.equal(out.length, c.samples);
    assert.equal(out[0], 0.0, `${c.name} start`);
    assert.equal(out[out.length - 1], 0.0, `${c.name} end`);
  }
});

test("humanize introduces zero high-frequency hiss above 4 kHz", () => {
  const spec = JSON.parse(readFileSync(join(repoSpecDir, "golden_cases.json"), "utf8"));
  const tone = generateSine(220, 1.0, SAMPLE_RATE, 0.5);
  const out = applyHumanize(tone, 0.8, SAMPLE_RATE);
  let energyTotal = 0;
  let energyDiff = 0;
  let origDiff = 0;
  for (let i = 1; i < out.length - 1; i++) {
    const x = out[i];
    const lap = out[i + 1] - 2 * out[i] + out[i - 1];
    const origLap = tone[i + 1] - 2 * tone[i] + tone[i - 1];
    energyTotal += x * x;
    energyDiff += lap * lap;
    origDiff += origLap * origLap;
  }
  const addedHfRatio = Math.abs(energyDiff - origDiff) / (energyTotal + 1e-12);
  assert.ok(addedHfRatio < spec.tolerances.humanize_hf_ratio_above_4khz * 10, `addedHfRatio=${addedHfRatio}`);
});

test("calibrated master hall reverb keeps 0.35-amp pure tone below clipping", () => {
  const spec = JSON.parse(readFileSync(join(repoSpecDir, "golden_cases.json"), "utf8"));
  const tone = generateSine(330, 3.0, SAMPLE_RATE, 0.35);
  const wet = applyMasterHallReverb(tone, 0.95, SAMPLE_RATE);
  let peak = 0;
  for (let i = 0; i < wet.length; i++) peak = Math.max(peak, Math.abs(wet[i]));
  assert.ok(peak <= spec.tolerances.hall_reverb_max_tone_peak, `peak=${peak}`);
});

test("golden_cases.json audio invariants pass across all multi-voice configs", () => {
  const spec = JSON.parse(readFileSync(join(repoSpecDir, "golden_cases.json"), "utf8"));
  const tol = spec.tolerances;

  for (const item of spec.render_cases) {
    const cfg = toInternalConfig(item.config);
    const audio = assemble(cfg, { duration: item.duration, normalize: true, seed: item.seed, tOffset: 0 });
    let peak = 0;
    let sum = 0;
    let maxJump = 0;
    for (let i = 0; i < audio.length; i++) {
      const v = audio[i];
      assert.ok(Number.isFinite(v), `${item.name} finite`);
      const a = Math.abs(v);
      if (a > peak) peak = a;
      sum += v;
      if (i > 0) {
        const j = Math.abs(v - audio[i - 1]);
        if (j > maxJump) maxJump = j;
      }
    }
    assert.ok(peak >= tol.min_audible_peak && peak <= tol.max_peak, `${item.name} peak=${peak}`);
    assert.ok(Math.abs(sum / audio.length) < tol.max_abs_dc, `${item.name} dc`);
    assert.ok(maxJump < tol.max_sample_jump, `${item.name} maxJump=${maxJump}`);
  }
});

test("Score converges discrete fields over small blocks, stays click-free, and exports silent-end stereo WAV", () => {
  const score = createScore("quiet morning desk", { seed: 11, glideSeconds: 2.0, blockSeconds: 0.5 });
  assert.equal(score.currentConfig.mode, "major");

  const t1 = score.steer("rainy midnight coffee shop");
  const t2 = score.steer("rainy midnight coffee shop");
  assert.deepEqual(t1, t2);
  assert.equal(t1.texture, "vinyl_crackle");

  // 4 x 0.5 s = 2.0 s must converge discrete fields cleanly with no seam clicks.
  const blocks = [score.nextBlock(0.5), score.nextBlock(0.5), score.nextBlock(0.5), score.nextBlock(0.5)];
  for (let i = 1; i < blocks.length; i++) {
    const jump = Math.abs(blocks[i][0] - blocks[i - 1][blocks[i - 1].length - 1]);
    assert.ok(jump < 0.08, `seam jump ${jump}`);
  }
  assert.equal(score.currentConfig.root, "d");
  assert.equal(score.currentConfig.mode, "dorian");
  assert.equal(score.currentConfig.texture, "vinyl_crackle");

  // Identical re-steer with a longer glideSeconds after convergence must NOT rewind state.
  const settledBrightness = score.internalConfig.brightness;
  score.steer("rainy midnight coffee shop", { glideSeconds: 10.0 });
  score.nextBlock(0.5);
  assert.equal(score.internalConfig.brightness, settledBrightness);

  // Mixing fill() and nextBlock() stays strictly sequential without jumps.
  const a = score.fill(1024);
  const b = score.nextBlock(0.25);
  const c = score.fill(new Float32Array(2048));
  assert.equal(a.length, 1024);
  assert.equal(b.length, Math.round(0.25 * SAMPLE_RATE));
  assert.equal(c.length, 2048);
  assert.ok(Math.abs(b[0] - a[a.length - 1]) < 0.08);
  assert.ok(Math.abs(c[0] - b[b.length - 1]) < 0.08);

  // Nudging space preserves untouched continuous tempo.
  const s2 = createScore({ ...toInternalConfig(DEFAULT_CONFIG), tempo: 0.42 }, { glideSeconds: 0, blockSeconds: 0.5 });
  s2.nudge({ space: step(1) });
  s2.nextBlock(0.5);
  assert.ok(Math.abs(s2.internalConfig.tempo - 0.42) < 1e-6);
  assert.throws(() => score.nextBlock(Number.NaN), /finite/);

  // Stereo WAV must end in exact silence (0, 0) on both L and R channels.
  const wavStereo = encodeWavBytes(blocks[3], { channels: 2, stereoWidth: 0.9, fadeOutSeconds: 0.1 });
  assert.equal(Buffer.from(wavStereo.subarray(0, 4)).toString("ascii"), "RIFF");
  assert.equal(Buffer.from(wavStereo.subarray(8, 12)).toString("ascii"), "WAVE");
  const dv = new DataView(wavStereo.buffer, wavStereo.byteOffset, wavStereo.byteLength);
  assert.equal(dv.getUint32(24, true), SAMPLE_RATE);
  assert.equal(dv.getInt16(wavStereo.byteLength - 4, true), 0);
  assert.equal(dv.getInt16(wavStereo.byteLength - 2, true), 0);

  // Score.toWavBytes enforces 44,100 Hz sample rate header.
  const scoreWav = score.toWavBytes(0.5, { channels: 2 });
  const dv2 = new DataView(scoreWav.buffer, scoreWav.byteOffset, scoreWav.byteLength);
  assert.equal(dv2.getUint32(24, true), SAMPLE_RATE);

  const faded = fadeIntoNothingness(new Float32Array(4410).fill(1), 0.05);
  assert.equal(faded[faded.length - 1], 0);
  const oneShot = render("warm cedar zen meditation", { duration: 1.0, fadeOutSeconds: 0.1 });
  assert.equal(oneShot.length, SAMPLE_RATE);
  assert.equal(oneShot[oneShot.length - 1], 0);
  score.dispose();
  s2.dispose();
});

test("npm pack tarball works for ESM, CJS runtime, AND TypeScript node16 .cts/.mts consumers", () => {
  const workDir = mkdtempSync(join(tmpdir(), "latentscore-pack-"));
  try {
    const packOut = execFileSync("npm", ["pack", "--pack-destination", workDir, "--silent"], {
      cwd: pkgDir,
      encoding: "utf8",
    }).trim();
    const tgzName = packOut.split("\n").pop();
    assert.ok(tgzName && tgzName.endsWith(".tgz"), `Unexpected npm pack output: ${packOut}`);
    const tgzPath = join(workDir, tgzName);

    const consumerDir = join(workDir, "consumer");
    execFileSync("mkdir", ["-p", consumerDir]);
    writeFileSync(
      join(consumerDir, "package.json"),
      JSON.stringify({ name: "smoke", private: true }, null, 2),
    );
    execFileSync("npm", ["install", "--no-audit", "--no-fund", tgzPath], {
      cwd: consumerDir,
      stdio: "pipe",
    });

    // Runtime ESM check
    const esmOut = execFileSync(
      "node",
      [
        "--input-type=module",
        "-e",
        'import { createScore } from "latentscore"; const s = createScore("quiet morning desk", { blockSeconds: 0.5 }); console.log(s.fill(256).length);',
      ],
      { cwd: consumerDir, encoding: "utf8" },
    ).trim();
    assert.equal(esmOut, "256");

    // Runtime CJS check
    const cjsOut = execFileSync(
      "node",
      [
        "-e",
        'const { createScore } = require("latentscore"); const s = createScore("quiet morning desk", { blockSeconds: 0.5 }); console.log(s.fill(256).length);',
      ],
      { cwd: consumerDir, encoding: "utf8" },
    ).trim();
    assert.equal(cjsOut, "256");

    // TypeScript node16 consumer check for BOTH .mts (ESM) and .cts (CJS)!
    writeFileSync(
      join(consumerDir, "esm_check.mts"),
      'import { createScore } from "latentscore";\nexport const n: number = createScore("quiet morning desk").fill(16).length;\n',
    );
    writeFileSync(
      join(consumerDir, "cjs_check.cts"),
      'import { createScore } from "latentscore";\nexport const n: number = createScore("quiet morning desk").fill(16).length;\n',
    );
    writeFileSync(
      join(consumerDir, "tsconfig.json"),
      JSON.stringify({
        compilerOptions: {
          module: "Node16",
          moduleResolution: "Node16",
          strict: true,
          noEmit: true,
          skipLibCheck: true,
        },
        include: ["esm_check.mts", "cjs_check.cts"],
      }),
    );
    const tscBin = join(pkgDir, "node_modules", ".bin", "tsc");
    execFileSync(tscBin, ["-p", join(consumerDir, "tsconfig.json")], {
      cwd: consumerDir,
      stdio: "pipe",
    });
  } finally {
    rmSync(workDir, { recursive: true, force: true });
  }
});
