# Sample App

Minimal tracer-bullet slice for async Python using `uvloop` by default (when available) and `dependency_injector` wiring.

## LatentScore Library

```python
from latentscore import MusicConfigUpdate, render, save_wav

update = MusicConfigUpdate(tempo="medium")
audio = render("warm sunrise over water", duration=8, model="fast", update=update)
save_wav("output.wav", audio)
```

- Local-first by default (no API keys required).
- `latentscore demo` renders a short clip to `demo.wav`.
- `latentscore download expressive` prefetches local LLM weights (~1.2GB).
- `docs/latentscore-dx.md` has the full API and audio contract.
- Streaming supports prefetching, preview playback while configs load, and fallback policies; see `docs/latentscore-dx.md`.
- First-time expressive downloads show a spinner; run `latentscore doctor` and prefetch missing models in production to avoid runtime downloads.

## Quickstart
- Apple Silicon recommended; use an arm64 Conda (e.g. Miniforge) when possible.
- `conda create -n latentscore python=3.10` and `conda activate latentscore`.
- `pip install -r requirements.txt`.
- Run the demo: `PYTHONPATH=src python -m app` (or `pip install -e .` once).

## UI demos
- Textual TUI: `python -m app.tui` renders a centered `Hello world!` message in the terminal.
- macOS menu bar: `PYTHONPATH=src python -m app.menubar` adds a status bar item. The "Open Sample App" entry launches a Textual web UI inside a native window (install `textual-serve` + `pywebview`). Use "Open Logs Folder" to inspect logs, or "See Diagnostics" to open the Textual log viewer in a native window (falls back to browser if `pywebview` is missing). Diagnostics copy uses `pyperclip` for the system clipboard.
- Menu bar subprocesses watch a parent-death signal (pipe/PID) and exit when the menu bar app exits.

## Tooling
- `make check`: ruff lint, ruff format --check, pyright --strict, pytest with coverage.
- `make format`: apply `ruff format`.
- `make run`: same as `python -m app`.

## Project layout
- `src/app/branding.py`: app name + derived identifiers used across UI/logging.
- `src/app/loop.py`: installs `uvloop` policy when available (falls back to asyncio on Windows) and wraps `asyncio.run`.
- `src/app/app.py`: DI container and demo entrypoint.
- `src/app/computation.py`: pure async helper used by the demo.
- `src/app/tui.py`: Textual hello world app (`python -m app.tui`).
- `src/app/menubar.py`: macOS status bar button that greets via dialog (`python -m app.menubar` on macOS).
- `tests/`: pytest + pytest-asyncio coverage-backed smoke tests.
- `docs/examples.md`: extra patterns and snippets.

## Contributing
- Keep commits small and focused; split refactors/formatting from behavior.
- Stay async-first; avoid mixing threads unless bounded and intentional.
- Add tests for new behavior and keep coverage high; prefer strict typing (`pyright --strict`).

# LatentScore - Strudel Audio in Textual Webview

## What This Does

Plays Strudel audio **directly in the Textual webview** - no separate browser windows.

```
┌─────────────────────────────────────────────────────────────────┐
│                    pywebview Window                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Textual TUI (via textual-serve)              │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │             🎵 LatentScore                          │  │  │
│  │  │                                                     │  │  │
│  │  │  [🔇 Click to Enable Audio]  ← Required first click │  │  │
│  │  │                                                     │  │  │
│  │  │  [🇮🇳 Indian] [🌑 Dark] [☁️ Ambient]               │  │  │
│  │  │  [🤖 Sad Robot] [🛋️ Haunted IKEA]                  │  │  │
│  │  │                                                     │  │  │
│  │  │  [◼ Stop]                                           │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                     ↑                                      │  │
│  │       Injected Strudel JS (Web Audio API)                  │  │
│  │       Listens for button clicks, plays audio               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## How It Works

1. **`strudel_bridge.py`**: Contains JavaScript that gets injected into the webview
   - Loads Strudel SDK from CDN (esm.sh)
   - Exposes `strudelPlay(vibe)` and `strudelStop()` functions
   - Listens for clicks on buttons with IDs starting with `vibe-`

2. **`webview_app.py`**: Modified to call `install_strudel_bridge(window)`
   - Injects the Strudel JS when the page loads
   - Uses pywebview's `events.loaded` hook

3. **`textual_app.py`**: Has vibe buttons with specific IDs
   - `id="vibe-indian_wedding"`, `id="vibe-dark_electronic"`, etc.
   - `id="vibe-stop"` for the stop button
   - `id="audio-unlock"` for the initial audio unlock

## Installation

Copy these files to `src/app/`:

```bash
cp patch5/strudel_bridge.py src/app/strudel_bridge.py
cp patch5/webview_app.py src/app/webview_app.py
cp patch5/textual_app.py src/app/textual_app.py
```

## Usage

```bash
conda activate latentscore
PYTHONPATH=src python -m app.menubar
```

1. Click **"Open Sample App"** in menubar
2. Click **"🔇 Click to Enable Audio"** (required by browsers)
3. Click any vibe button → **Audio plays in the same window!**

## How the JS Bridge Works

The injected JavaScript:

```javascript
// Listens for clicks on elements with id="vibe-*"
document.addEventListener('click', async (e) => {
    let target = e.target;
    while (target) {
        if (target.id?.startsWith('vibe-')) {
            const vibe = target.id.replace('vibe-', '');
            if (vibe === 'stop') {
                window.strudelStop();
            } else {
                await window.strudelPlay(vibe);  // Plays audio!
            }
        }
        target = target.parentElement;
    }
}, true);
```

## Adding New Vibes

1. Add pattern to `PATTERNS` dict in `strudel_bridge.py`
2. Add button in `textual_app.py` with `id="vibe-your_vibe_name"`

## Architecture

```
Textual Button (Python)
    ↓ renders to
DOM Element (Browser) with id="vibe-indian_wedding"
    ↓ clicked
Injected JS catches click event
    ↓ extracts vibe name from id
strudelPlay("indian_wedding")
    ↓ loads Strudel SDK (once)
    ↓ evaluates pattern code
    ↓ starts scheduler
Web Audio API → 🔊 Sound
```

## Files Changed

| File | Changes |
|------|---------|
| `strudel_bridge.py` | **NEW** - JS bridge module |
| `webview_app.py` | Added `install_strudel_bridge()` call |
| `textual_app.py` | New UI with vibe buttons |

## Dependencies

No new Python dependencies! The Strudel SDK loads from CDN in the browser.

## Troubleshooting

**No sound?**
- Make sure you clicked "Enable Audio" first
- Check browser console (if accessible) for errors
- The Strudel SDK loads from `esm.sh` CDN - needs internet

**Buttons don't work?**
- Check that button IDs start with `vibe-`
- The JS uses event bubbling - should work even if Textual wraps buttons

**SDK won't load?**
- Check network connectivity
- esm.sh might be slow on first load (caches after)

# Activity-Driven Ambient Music Generator

## Design Document v1.0

---

## 1. Executive Summary

A lightweight, fully local application that generates continuous ambient music based on user activity patterns. The system monitors keyboard/mouse activity, active applications, and time of day to infer user "mood" or work state, then steers a neural audio synthesizer in real-time to produce contextually appropriate background music.

**Key Properties:**
- Fully local (no cloud dependencies)
- Runs on Apple Silicon (M1 MacBook Air target)
- Distributable via PyInstaller
- Continuous, non-repetitive audio generation
- Smooth transitions between moods (no jarring changes)

---

## 2. Problem Statement

### User Need
Background music that adapts to work context without manual intervention. Users want:
- Music that matches their energy level
- No repetitive loops
- No subscription services
- No internet requirement
- Minimal system resource usage

### Technical Challenge
Mapping semantic concepts ("focused coding at 2am") to audio generation parameters, while maintaining:
- Musical coherence
- Smooth transitions
- Low latency response to state changes
- Minimal compute footprint

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ACTIVITY MONITOR                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  Keyboard   │  │   Mouse     │  │   Active    │                  │
│  │  Velocity   │  │  Movement   │  │    App      │                  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │
│         └────────────────┼────────────────┘                         │
│                          ▼                                          │
│                 ┌─────────────────┐                                 │
│                 │  State Calc     │                                 │
│                 │  (every 10-30s) │                                 │
│                 └────────┬────────┘                                 │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SEMANTIC BRIDGE                                │
│                                                                     │
│  Activity State ──► Text Description ──► CLAP Text Encoder          │
│                                                │                    │
│                                                ▼                    │
│                                    ┌────────────────────┐           │
│                                    │  Nearest Neighbor  │           │
│                                    │  Search (5000 pts) │           │
│                                    └─────────┬──────────┘           │
│                                              │                      │
│               Pre-computed                   │                      │
│               CLAP Audio ◄───────────────────┘                      │
│               Embeddings                                            │
│                   │                                                 │
│                   ▼                                                 │
│            Matched Latent Vector (z)                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       AUDIO ENGINE                                  │
│                                                                     │
│    ┌───────────────────────────────────────────────────-───┐        │
│    │              LATENT INTERPOLATOR                      │        │
│    │                                                       │        │
│    │   current_z ────► exponential ────► smoothed_z        │        │
│    │       ▲            smoothing            │             │        │
│    │       │                                 │             │        │
│    │   target_z ◄── from Semantic Bridge     │             │        │
│    └─────────────────────────────────────────┼─────────────┘        │
│                                              │                      │
│                                              ▼                      │
│                                    ┌─────────────────┐              │
│                                    │  RAVE Decoder   │              │
│                                    │  (TorchScript)  │              │
│                                    └────────┬────────┘              │
│                                             │                       │
│                                             ▼                       │
│                                    ┌─────────────────┐              │
│                                    │  Audio Stream   │              │
│                                    │   (PyAudio)     │              │
│                                    └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Specifications

### 4.1 Activity Monitor

**Purpose:** Capture user activity signals without logging sensitive content.

**Implementation:** macOS-specific using PyObjC

| Signal | Method | Update Rate | Privacy |
|--------|--------|-------------|---------|
| Keyboard velocity | Quartz CGEventTap | Event-driven | Keystroke count only, no content |
| Mouse movement | Quartz CGEventTap | Event-driven | Movement delta only |
| Active application | NSWorkspace notifications | On change | App name only |
| Time of day | System clock | On state calc | N/A |

**Required Permissions:**
- Accessibility (Input Monitoring)
- Potentially Screen Recording (for window titles)

**Output:** `ActivityState` dataclass updated every 10-30 seconds

```python
@dataclass
class ActivityState:
    typing_wpm: float          # Words per minute estimate
    mouse_activity: float      # Movement integral (0-1)
    active_app: str            # e.g., "VS Code", "Chrome"
    app_category: str          # "code", "browser", "creative", "communication"
    hour: int                  # 0-23
    idle_seconds: float        # Time since last input
```

---

### 4.2 State-to-Text Mapper

**Purpose:** Convert `ActivityState` to natural language description for CLAP.

**Logic:**

```python
def state_to_description(state: ActivityState) -> str:
    descriptors = []
    
    # Energy level
    if state.typing_wpm > 60:
        descriptors.append("energetic")
    elif state.typing_wpm > 30:
        descriptors.append("focused")
    elif state.typing_wpm > 10:
        descriptors.append("relaxed")
    else:
        descriptors.append("calm")
    
    # Time of day
    if 22 <= state.hour or state.hour < 5:
        descriptors.append("late night")
    elif 5 <= state.hour < 9:
        descriptors.append("morning")
    elif 17 <= state.hour < 22:
        descriptors.append("evening")
    
    # Context
    if state.app_category == "code":
        descriptors.append("concentration")
    elif state.app_category == "creative":
        descriptors.append("creative")
    elif state.app_category == "browser":
        descriptors.append("browsing")
    
    # Idle handling
    if state.idle_seconds > 300:
        descriptors = ["ambient", "background"]
    
    return ", ".join(descriptors)
```

**Example Outputs:**
- `"energetic, late night, concentration"` → Coding at 2am, typing fast
- `"calm, morning"` → Just opened laptop, no activity yet
- `"focused, evening, creative"` → Working in design app after dinner

---

### 4.3 CLAP Semantic Bridge

**Purpose:** Map text descriptions to RAVE latent vectors.

**Model:** `laion/clap-htsat-unfused` (MIT License)

**Pre-computation (One-time setup, ~2-3 hours on M1):**

1. Sample 5,000 random latent vectors from RAVE's latent space
2. Decode each to ~3 seconds of audio
3. Embed each audio clip using CLAP's audio encoder
4. Store: `latent_vectors.npy` (5000, 16) + `clap_embeddings.npy` (5000, 512)
5. Build nearest-neighbor index

**Runtime Query:**

1. Text → CLAP text encoder → 512-dim embedding
2. Find k=3 nearest CLAP audio embeddings (cosine similarity)
3. Weighted average of corresponding latent vectors
4. Return target `z` vector

**Storage:**
- Pre-computed embeddings: ~5 MB
- CLAP model (text encoder only at runtime): ~300 MB

---

### 4.4 RAVE Audio Engine

**Purpose:** Generate continuous audio from latent vectors.

**Model:** Tangible Music Lab's `freesound_loops` (MIT License)

| Property | Value |
|----------|-------|
| Latent dimensions | 16 |
| Sample rate | 48 kHz |
| Block size | 2048 samples (~43ms) |
| Compression ratio | ~2048:1 |

**Streaming Architecture:**

```python
class RAVEStreamer:
    def __init__(self, model_path: str):
        self.model = torch.jit.load(model_path).eval()
        self.current_z = torch.zeros(1, 16, 1)
        self.target_z = torch.zeros(1, 16, 1)
        self.smoothing = 0.02  # ~2-3 second transition
        
    def set_target(self, z: torch.Tensor):
        """Called when semantic bridge returns new target."""
        self.target_z = z
        
    def generate_block(self) -> np.ndarray:
        """Generate one audio block, smoothly interpolating."""
        # Exponential moving average toward target
        self.current_z = (
            self.current_z * (1 - self.smoothing) + 
            self.target_z * self.smoothing
        )
        
        with torch.no_grad():
            audio = self.model.decode(self.current_z)
        
        return audio.squeeze().numpy()
```

**Smoothing Behavior:**

| Smoothing Factor | Transition Time | Use Case |
|------------------|-----------------|----------|
| 0.01 | ~5-7 seconds | Very gradual, barely perceptible |
| 0.02 | ~2-3 seconds | Default, smooth but responsive |
| 0.05 | ~1 second | Noticeable but not jarring |
| 0.10 | ~0.5 seconds | Quick response, may be audible |

---

## 5. Data Flow

### 5.1 Startup Sequence

```
1. Load RAVE model (.ts file)
2. Load CLAP text encoder
3. Load pre-computed embeddings + build NN index
4. Initialize audio stream (PyAudio)
5. Start activity monitor (register event taps)
6. Start audio generation thread
7. Begin main loop
```

### 5.2 Runtime Loop

```
┌─────────────────────────────────────────────────────┐
│ Activity Monitor Thread                             │
│                                                     │
│   while True:                                       │
│       collect events for 10-30 seconds              │
│       compute ActivityState                         │
│       if state changed significantly:               │
│           notify main thread                        │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ Main Thread                                         │
│                                                     │
│   on activity_state_changed(state):                 │
│       description = state_to_description(state)    │
│       target_z = clap_bridge.text_to_latent(desc)  │
│       audio_engine.set_target(target_z)            │
└─────────────────────────────────────────────────────┘
                          
┌─────────────────────────────────────────────────────┐
│ Audio Thread (continuous)                           │
│                                                     │
│   while True:                                       │
│       block = audio_engine.generate_block()        │
│       audio_stream.write(block)                    │
│       # Automatically interpolates toward target   │
└─────────────────────────────────────────────────────┘
```

### 5.3 Latency Budget

| Stage | Latency | Notes |
|-------|---------|-------|
| Activity state calculation | 0 ms | Async, not in audio path |
| Text → CLAP embedding | ~50 ms | Once per state change |
| NN search (5000 points) | <1 ms | Cosine similarity |
| RAVE decode (per block) | ~5 ms | 2048 samples |
| Audio buffer | ~43 ms | 2048 @ 48kHz |
| **Total audio latency** | **~48 ms** | Imperceptible |
| **State change → audible** | **~2-3 sec** | Smoothing, intentional |

---

## 6. Licensing & Distribution

### 6.1 Component Licenses

| Component | License | Commercial Use | Notes |
|-----------|---------|----------------|-------|
| RAVE (code) | MIT | ✅ Yes | acids-ircam/RAVE |
| Tangible Music Lab model | MIT | ✅ Yes | Trained on Freesound Loop Dataset |
| CLAP (code) | MIT | ✅ Yes | laion/clap |
| CLAP (weights) | CC-BY-4.0 | ✅ Yes | Attribution required |
| PyTorch | BSD | ✅ Yes | |
| PyAudio | MIT | ✅ Yes | |
| PyObjC | MIT | ✅ Yes | |

### 6.2 Attribution Requirements

Include in application "About" or documentation:

```
Audio generation powered by:
- RAVE by Antoine Caillon and Philippe Esling (IRCAM)
- CLAP by LAION
- Freesound Loop Dataset by Antonio Ramires et al.
```

### 6.3 Training Data Considerations

The Tangible Music Lab model is trained on the Freesound Loop Dataset (FSL10K), which contains ~9,500 loops under various Creative Commons licenses. The model weights themselves are MIT licensed.

**Risk Assessment:** Low. Model weights are generally considered transformative and not direct copies of training data. However, consult legal counsel if distributing commercially at scale.

---

## 7. Performance Specifications

### 7.1 Target Hardware

- **Primary:** Apple M1 MacBook Air (8GB RAM)
- **Secondary:** Any Apple Silicon Mac

### 7.2 Resource Budget

| Resource | Budget | Notes |
|----------|--------|-------|
| CPU (continuous) | <15% | Audio generation + monitoring |
| CPU (peak) | <30% | During CLAP embedding |
| RAM | <1 GB | Models + buffers |
| Disk (app bundle) | ~800 MB | Models + dependencies |
| Disk (runtime) | ~50 MB | Embeddings + cache |

### 7.3 Measured Performance (Estimated)

| Operation | M1 Air | Notes |
|-----------|--------|-------|
| RAVE decode (per block) | ~3-5 ms | Well under real-time |
| CLAP text embed | ~50-80 ms | Once per state change |
| Activity monitor | <1% CPU | Event-driven |
| Total idle | ~8-12% CPU | Continuous audio generation |

---

## 8. File Structure

```
ActivityMusic.app/
├── Contents/
│   ├── MacOS/
│   │   └── ActivityMusic          # PyInstaller executable
│   ├── Resources/
│   │   ├── models/
│   │   │   ├── rave_freesound.ts  # RAVE TorchScript (~80 MB)
│   │   │   └── clap_text/         # CLAP text encoder (~300 MB)
│   │   ├── embeddings/
│   │   │   └── rave_clap_bridge.npz  # Pre-computed (~5 MB)
│   │   └── config/
│   │       └── app_categories.json   # App → category mapping
│   ├── Info.plist
│   └── entitlements.plist         # Accessibility permissions
└── ...
```

---

## 9. Configuration

### 9.1 User-Configurable Settings

```yaml
# config.yaml
audio:
  volume: 0.7                    # 0.0 - 1.0
  transition_speed: "medium"     # slow, medium, fast
  
activity:
  sensitivity: "medium"          # low, medium, high
  update_interval_seconds: 15    # How often to check state
  
mood_mapping:
  # Override default state-to-text mapping
  late_night_coding: "dark ambient electronic minimal"
  morning_email: "calm peaceful piano"
  
excluded_apps:
  - "Spotify"                    # Don't generate when music already playing
  - "Music"
  - "VLC"
```

### 9.2 App Category Mapping

```json
{
  "code": ["VS Code", "Xcode", "PyCharm", "Terminal", "iTerm"],
  "creative": ["Figma", "Sketch", "Photoshop", "Logic Pro", "Ableton"],
  "browser": ["Safari", "Chrome", "Firefox", "Arc"],
  "communication": ["Slack", "Discord", "Messages", "Mail", "Zoom"],
  "writing": ["Notion", "Obsidian", "Word", "Pages", "Bear"]
}
```

---

## 10. Risk Assessment

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CLAP-RAVE semantic mismatch | Medium | Medium | Manual override zones, user feedback |
| PyInstaller + PyTorch issues | Medium | High | Test thoroughly, fallback to .app bundle |
| Accessibility permission denied | Low | High | Clear onboarding, graceful degradation |
| Audio glitches on load | Low | Medium | Pre-warm model, fade-in on start |
| Memory pressure on 8GB Mac | Low | Medium | Lazy loading, memory monitoring |

### 10.2 UX Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Music doesn't match user's taste | High | High | Multiple RAVE models, style presets |
| Transitions feel unnatural | Medium | Medium | Tunable smoothing, crossfade options |
| Latent space has "bad regions" | Medium | Medium | Constrain to known-good embeddings |
| Users want more control | Medium | Low | Manual mood override, favorite zones |

---

## 11. Future Enhancements

### Phase 2
- [ ] Multiple RAVE models (different genres/vibes)
- [ ] User-trainable preferences (like/dislike feedback)
- [ ] Calendar integration (meeting mode, focus time)
- [ ] iOS companion (sync mood across devices)

### Phase 3
- [ ] Train custom RAVE model on curated ambient/lo-fi audio
- [ ] Fine-tune CLAP bridge with user feedback
- [ ] Biometric integration (heart rate from Apple Watch)

---

## 12. Open Questions

1. **Model Selection:** Is the Freesound Loops model the right aesthetic? Should we train a custom model on more ambient/atmospheric content?

2. **Embedding Coverage:** Are 5,000 sampled points sufficient to cover the meaningful regions of RAVE's latent space?

3. **Privacy:** Should we offer an "ultra-private" mode that doesn't even track app names, only input velocity?

4. **Audio Conflicts:** How do we detect when the user is already playing music and should we auto-pause?

5. **Multiple Output Devices:** Should we support routing to specific audio devices (e.g., headphones only)?

---

## 13. Appendix

### A. RAVE Latent Space Visualization

The RAVE model's 16-dimensional latent space can be partially understood through these observations:

- **Dimensions 0-3:** Tend to control overall energy/density
- **Dimensions 4-7:** Affect timbral characteristics
- **Dimensions 8-11:** Influence rhythmic patterns
- **Dimensions 12-15:** Fine texture and noise characteristics

*(Note: These are empirical observations and may vary by model)*

### B. CLAP Embedding Space

CLAP's 512-dimensional joint audio-text space clusters semantically:

- Similar concepts cluster together ("calm piano" near "peaceful keys")
- Opposing concepts are distant ("energetic" far from "relaxed")
- The text encoder generalizes to unseen phrases

### C. References

1. Caillon, A., & Esling, P. (2021). RAVE: A variational autoencoder for fast and high-quality neural audio synthesis. arXiv:2111.05011

2. Wu, Y., et al. (2023). Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation. ICASSP 2023.

3. Ramires, A., et al. (2020). The Freesound Loop Dataset and Annotation Tool. ISMIR 2020.

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Author: [Your Name]*
