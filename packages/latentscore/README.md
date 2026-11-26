# latentscore (JavaScript and TypeScript)

Generate calm, reactive ambient music locally in JavaScript and TypeScript. The package needs no GPU, no server calls, and zero runtime dependencies.

[Quickstart](#quickstart) | [Type to morph in the browser](#type-to-morph-in-the-browser) | [Node and Bun usage](#node-and-bun-usage) | [API overview](#api-overview) | [Optional 14,035-mood index](#optional-14035-mood-index)

## Quickstart

Install the package from npm:

```bash
npm install latentscore
```

The root entry point (`latentscore`) uses no browser DOM APIs. It runs in Node.js, Bun, Deno, Web Workers, SSR servers, and browsers.

The browser entry point (`latentscore/browser`) wraps the engine with a Web Audio player for interactive local web apps.

[Back to top](#latentscore-javascript-and-typescript)

## Type to morph in the browser

Build a local writing app where music glides smoothly as the user types:

```ts
import { createBrowserScore } from "latentscore/browser";

const music = createBrowserScore("quiet morning desk", { glideSeconds: 3.5 });

// Call play() inside a user click handler so browser autoplay policies allow sound.
startButton.addEventListener("click", async () => {
  await music.play();
});

// Music glides toward the mood of the text as the user types.
textarea.addEventListener("input", () => {
  music.steerOnType(textarea.value);
});

// Enable endless smooth drift across neighboring moods.
const drift = music.startAutoDrift({ intervalMs: 12000 });

// Stop smoothly with a cubic fade into complete silence.
stopButton.addEventListener("click", () => {
  drift.stop();
  music.stop(0.5);
});
```

Calling `steer("rainy midnight coffee shop")` is idempotent. Calling it once or ten times aims for the same target configuration, while audio glides smoothly from the current audible state over `glideSeconds`. For example, at 100 BPM one bar lasts 2.4 seconds, so a `3.5`-second glide shifts key, filter, and instrument layers across roughly one and a half bars.

Manual calls to `steer()` or `nudge()` always take priority and pause active auto-drift.

To try the single-file browser demo at `examples/type-to-morph.html`, build the package and serve the folder over a local HTTP server (browsers block ES module imports over `file://`):

```bash
npm run build
npx serve .
```

Then open `http://localhost:3000/examples/type-to-morph.html`.

[Back to top](#latentscore-javascript-and-typescript)

## Node and Bun usage

Render seamless PCM audio or write a stereo WAV file in Node.js or Bun:

```ts
import { writeFileSync } from "node:fs";
import { createScore, encodeWavBytes, step } from "latentscore";

const score = createScore("quiet morning desk", { seed: 42, glideSeconds: 4.0 });
const firstHalf = score.fill(44100 * 4);

score.steer("rainy midnight coffee shop");
score.nudge({ space: step(+1) });
const secondHalf = score.fill(44100 * 4);

const full = new Float32Array(firstHalf.length + secondHalf.length);
full.set(firstHalf, 0);
full.set(secondHalf, firstHalf.length);

// channels: 2 builds a mono-compatible Mid-Side stereo WAV file.
const wav = encodeWavBytes(full, { channels: 2, stereoWidth: 0.65, fadeOutSeconds: 0.5 });
writeFileSync("morning-to-rain.wav", wav);
```

[Back to top](#latentscore-javascript-and-typescript)

## API overview

The core entry point (`import { ... } from "latentscore"`) exports these functions:

- `createScore(initial, options)` creates a stateful reactive `Score`.

- `score.steer(promptOrConfig, options)` sets a new target mood idempotently.

- `score.nudge(update)` adjusts fields with absolute values or `step(+1)` / `step(-1)`.

- `score.driftNext()` advances one smooth step along a harmonic or similarity gradient.

- `score.startAutoDrift(options)` schedules periodic drift steps and returns a `{ stop() }` handle.

- `score.fill(framesOrBuffer)` pulls any exact number of mono Float32 samples at 44,100 Hz.

- `score.nextBlock(seconds)` renders one overlap-crossfaded Float32 block.

- `render(promptOrConfig, options)` performs a one-shot Float32 PCM render.

- `moodFromText(text)` runs instant deterministic keyword matching from `spec/mood_lexicon.json`.

- `encodeWavBytes(samples, options)` encodes 16-bit mono or M-S stereo WAV bytes.

- `computeBarAlignedDuration(tempo, targetSeconds)` snaps clip length to a 4-bar phrase boundary. For example, at `tempo = 0.5`, BPM is 110 and 4 bars equal 8.73 seconds.

The browser entry point (`import { createBrowserScore } from "latentscore/browser"`) exports `createBrowserScore` with these methods:

- `music.play()` creates and resumes `AudioContext` inside a user gesture, then streams audio.

- `music.steerOnType(text)` applies debounced steering for live text editors.

- `music.setVolume(value)` adjusts master volume smoothly between `0` and `1`.

- `music.stop(fadeOutSeconds)` runs a single cubic fade into exact silence.

- `music.dispose()` stops audio, removes visibility listeners, and closes the `AudioContext`.

[Back to top](#latentscore-javascript-and-typescript)

## Optional 14,035-mood index

The npm package stays small and does not bundle the 4.5 MB embedding map. To walk the full 14,035-mood similarity graph, host the quantized index files (`manifest.json`, `embeddings.int8.bin`, `scales.f32.bin`, and `metadata.json`) and load them at runtime:

```ts
import { EmbeddingIndex, createScore } from "latentscore";

const index = await EmbeddingIndex.loadFromUrl("/data");
const score = createScore("quiet morning desk", { index });
score.driftNext();
```

[Back to top](#latentscore-javascript-and-typescript)
