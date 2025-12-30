# latentscore

Minimal tracer-bullet slice for async Python using `uvloop` by default and `dependency_injector` wiring.

## Quickstart
- Apple Silicon recommended; use an arm64 Conda (e.g. Miniforge) when possible.
- `conda create -n latentscore python=3.10` and `conda activate latentscore`.
- `pip install -r requirements.txt`.
- Run the demo: `PYTHONPATH=src python -m latentscore` (or `pip install -e .` once).

## UI demos
- Textual TUI: `python -m latentscore.tui` renders a centered `Hello world!` message in the terminal.
- macOS menu bar: `PYTHONPATH=src python -m latentscore.menubar` adds a status bar item. The "Open LatentScore" entry launches a Textual web UI inside a native window (install `textual-serve` + `pywebview`). Use "Open Logs Folder" to inspect logs, or "See Diagnostics" to open the Textual log viewer in a native window (falls back to browser if `pywebview` is missing). Logs default to `~/Library/Logs/LatentScore` and can be overridden with `LATENTSCORE_LOG_DIR`.

## Tooling
- `make check`: ruff lint, ruff format --check, pyright --strict, pytest with coverage.
- `make format`: apply `ruff format`.
- `make run`: same as `python -m latentscore`.

## Project layout
- `src/latentscore/loop.py`: installs `uvloop` policy and wraps `asyncio.run`.
- `src/latentscore/app.py`: DI container and demo entrypoint.
- `src/latentscore/computation.py`: pure async helper used by the demo.
- `src/latentscore/tui.py`: Textual hello world app (`python -m latentscore.tui`).
- `src/latentscore/menubar.py`: macOS status bar button that greets via dialog (`python -m latentscore.menubar` on macOS).
- `tests/`: pytest + pytest-asyncio coverage-backed smoke tests.
- `docs/examples.md`: extra patterns and snippets.

## Contributing
- Keep commits small and focused; split refactors/formatting from behavior.
- Stay async-first; avoid mixing threads unless bounded and intentional.
- Add tests for new behavior and keep coverage high; prefer strict typing (`pyright --strict`).
