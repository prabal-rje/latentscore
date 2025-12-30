# latentscore

Minimal tracer-bullet slice for async Python using `uvloop` by default and `dependency_injector` wiring.

## Quickstart
- `conda create -n latentspace python=3.10` and `conda activate latentspace`.
- `pip install -r requirements.txt`.
- Run the demo: `python -m latentscore` (computes a sample mean via the container).

## UI demos
- Textual TUI: `python -m latentscore.tui` renders a centered `Hello world!` message in the terminal.
- macOS menu bar: `python -m latentscore.menubar` adds a status bar item with a single `Say hi` menu entry that pops a `Hi there!` dialog (macOS only).

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
