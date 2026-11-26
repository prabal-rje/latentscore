# latentscore (Rust)

Generate calm, reactive ambient music locally in pure Rust. The crate needs no GPU, no Python runtime, and no external runtime assets.

[Quickstart](#quickstart) | [Live playback and steering](#live-playback-and-steering) | [Pull-based audio and WAV export](#pull-based-audio-and-wav-export) | [Optional 14,035-mood index](#optional-14035-mood-index)

## Quickstart

Add `latentscore` to `Cargo.toml` for offline synthesis and WAV export:

```toml
[dependencies]
latentscore = "0.1"
```

Enable live system audio output via `cpal` with the `playback` feature:

```toml
[dependencies]
latentscore = { version = "0.1", features = ["playback"] }
```

On Ubuntu or Debian Linux, install ALSA development headers before compiling with `playback`:

```bash
sudo apt-get update && sudo apt-get install -y libasound2-dev pkg-config
```

[Back to top](#latentscore-rust)

## Live playback and steering

Build a local app where ambient music responds as someone types:

```rust
use latentscore::{FieldOrStep, MusicConfigUpdate, Score, Step};
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Starts non-blocking background playback with lock-free audio output.
    let handle = Score::from_text("quiet morning desk")
        .with_glide_seconds(3.5)
        .play()?;

    thread::sleep(Duration::from_secs(6));

    // Steer idempotently whenever your app state or editor text changes.
    handle.steer("rainy midnight coffee shop");
    thread::sleep(Duration::from_secs(6));

    // Nudge one parameter up by a step.
    handle.nudge(MusicConfigUpdate {
        space: Some(FieldOrStep::Delta(Step(1))),
        ..Default::default()
    });
    thread::sleep(Duration::from_secs(4));

    // Drift smoothly to a harmonically neighboring mood.
    handle.drift_next();
    thread::sleep(Duration::from_secs(6));

    // Fade smoothly into complete silence and release the audio device.
    handle.stop(0.5);
    Ok(())
}
```

The `cpal` audio callback reads from a preallocated lock-free SPSC ring buffer (`AtomicUsize`). It runs with zero mutex locks and zero heap allocations on the OS audio callback thread. It also resamples automatically from 44,100 Hz to your output device rate, such as 48,000 Hz on macOS or Linux.

Run the interactive terminal example:

```bash
cargo run --example type_to_morph --features playback
```

[Back to top](#latentscore-rust)

## Pull-based audio and WAV export

Use `Score::fill(&mut [f32])` for offline rendering or inside a dedicated producer thread that feeds your own audio callback ring buffer. Do not call `Score::fill()` directly on a hard audio callback thread because synthesis allocates intermediate DSP buffers.

```rust
use latentscore::{Score, WavOptions};

let mut score = Score::from_text("quiet morning desk")
    .with_seed(42)
    .with_glide_seconds(4.0);

let mut buf = vec![0.0_f32; 44_100 * 4];
score.fill(&mut buf);

score.steer("vast deep space nebula with stars");
score.save_wav(
    "space.wav",
    6.0,
    WavOptions {
        channels: 2,
        stereo_width: 0.7,
        fade_out_seconds: 0.5,
        ..WavOptions::default()
    },
)?;
```

Calling `steer()` is idempotent. Calling it once or ten times with the same phrase aims for the same target configuration, while audio glides smoothly from the current audible state over `glide_seconds`. For example, at 110 BPM one bar lasts 2.18 seconds, so a `4.0`-second glide spans just under two bars.

[Back to top](#latentscore-rust)

## Optional 14,035-mood index

The crate embeds only the compact text lexicon (`spec/mood_lexicon.json`). To walk the full 14,035-mood similarity graph, point `EmbeddingIndex::load_from_dir` at a local folder holding `manifest.json`, `embeddings.int8.bin`, `scales.f32.bin`, and `metadata.json`:

```rust
use latentscore::{EmbeddingIndex, Score};

let index = EmbeddingIndex::load_from_dir("demo/frontend/public/data")?;
let mut score = Score::from_text("quiet morning desk").with_embedding_index(index);
let next_mood = score.drift_next();
```

[Back to top](#latentscore-rust)
