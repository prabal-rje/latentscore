# Sample App

Minimal tracer-bullet slice for async Python using `uvloop` by default (when available) and `dependency_injector` wiring.

## Quickstart
- Apple Silicon recommended; use an arm64 Conda (e.g. Miniforge) when possible.
- `conda create -n sample_app_env python=3.13` and `conda activate sample_app_env`.
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
conda activate sample_app_env
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