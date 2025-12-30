# Repository Notes for Agents

- Current contents are minimal (`README.md` was empty; `CONTRIBUTE.md` documented Conda setup). Keep additions concise and self-contained.
- Environment: use Conda env `latentspace` with Python 3.10; if `requirements.txt` appears, install via `pip install -r requirements.txt`.
- Git hygiene: prefer small, reviewable commits (1–3 per PR). Separate refactors/formatting from behavior changes; squash noisy WIP commits before merging.
- Tests/tooling: none present initially. If you add code, include at least smoke tests or runnable examples and note how to execute them.
- Defaults: stay ASCII unless a file already uses Unicode; add comments only when logic is non-obvious; avoid destructive git commands.
- Tooling: canonical flow is `make check` (ruff check, ruff format --check, pyright --strict, pytest with coverage). Use `make format` to apply formatting (`ruff format`); lint with `ruff check`.
- Async: install `uvloop` as the default event loop policy; favor async-first APIs and avoid mixing threads unless bounded and deliberate. `python -m latentscore` runs a demo slice using uvloop.
- Dependency injection: wire services with `dependency-injector`; override providers in tests for determinism.
- Testing: design from UX/DX outward—define intended usage, write tests that capture that DX/UX, then implement to satisfy tests. Target 95%+ coverage with both unit and integration tests; invest in tooling/visibility that makes debugging easy.
- Complexity: if design feels tangled, prune aggressively and ask the human loop for direction before pushing forward.
- Libraries: grab and skim docs for popular dependencies before changing code; don’t guess. When stuck, inspect library internals to understand behavior rather than patching blindly.
- Functional FP/CAII: aim for commutativity, associativity, idempotence, immutability; prefer pure helpers, lazy generators where helpful, and avoid hidden state.
- Testing workflow examples: for quick spikes, drop assertions under an `if __name__ == "__main__":` or early-return block; for integrated pieces, run the main entrypoint with injected dependencies. Use `.inputs`/`.outputs` dirs for sample IO when present.
- Docs hygiene: when behavior/tooling shifts, update all relevant Markdown (README, CONTRIBUTE, AGENTS, docs/templates) in the same change.
- Examples: now live in `docs/examples.md` for quick reference.
