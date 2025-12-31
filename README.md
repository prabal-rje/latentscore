# Sample App

Minimal tracer-bullet slice for async Python using `uvloop` by default and `dependency_injector` wiring.

## Quickstart
- Apple Silicon recommended; use an arm64 Conda (e.g. Miniforge) when possible.
- `conda create -n sample_app_env python=3.10` and `conda activate sample_app_env`.
- `pip install -r requirements.txt`.
- Run the demo: `PYTHONPATH=src python -m app` (or `pip install -e .` once).

## UI demos
- Textual TUI: `python -m app.tui` renders a centered `Hello world!` message in the terminal.
- macOS menu bar: `PYTHONPATH=src python -m app.menubar` adds a status bar item. The "Open Sample App" entry launches a Textual web UI inside a native window (install `textual-serve` + `pywebview`). Use "Open Logs Folder" to inspect logs, or "See Diagnostics" to open the Textual log viewer in a native window (falls back to browser if `pywebview` is missing)..

## Tooling
- `make check`: ruff lint, ruff format --check, pyright --strict, pytest with coverage.
- `make format`: apply `ruff format`.
- `make run`: same as `python -m app`.

## Project layout
- `src/app/branding.py`: app name + derived identifiers used across UI/logging.
- `src/app/loop.py`: installs `uvloop` policy and wraps `asyncio.run`.
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
