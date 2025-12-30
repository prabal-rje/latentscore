# Repository Notes for Agents

## ⚠️ MANDATORY: Shell Environment Setup

**BEFORE running ANY shell command** (Python, pip, pytest, make, etc.), you MUST first activate the conda environment:

```bash
conda activate latentscore
```

This is NON-NEGOTIABLE. Every terminal session requires this. Do NOT run `python`, `pip`, `pytest`, `make check`, or any Python tooling without first activating the environment. If unsure whether the env is active, run `conda activate latentscore` anyway—it's idempotent.

---

## ⚠️ MANDATORY: Run Type Checking with Pyright

**You MUST run Pyright in strict mode** (`pyright --strict`) every single time you change _any_ Python file. This ensures that type errors are caught as early as possible.

- Run the following command from the project root AFTER ANY PYTHON FILE CHANGE, before any PR or commit:
  ```bash
  conda activate latentscore
  pyright
  ```
- Do **not** ignore or silence type errors unless you have a justified, documented reason.
- If you are unsure why Pyright is failing, ask for help immediately or investigate with the `--verbose` flag.

Maintaining strict type integrity is **required for all merges**.

---

## General Guidelines

- Current contents are minimal (`README.md` was empty; `CONTRIBUTE.md` documented Conda setup). Keep additions concise and self-contained.
- Environment: Conda env `latentscore` with Python 3.10. If `requirements.txt` appears, install via `pip install -r requirements.txt`.
- Git hygiene: prefer small, reviewable commits (1–3 per PR). Separate refactors/formatting from behavior changes; squash noisy WIP commits before merging.
- Tests/tooling: none present initially. If you add code, include at least smoke tests or runnable examples and note how to execute them.
- Defaults: stay ASCII unless a file already uses Unicode; add comments only when logic is non-obvious; avoid destructive git commands.
- Exception handling: never add bare `try/except` with `pass` or silent swallowing. If handling errors, be explicit about the exception types and the behavior; otherwise let it raise.
- Tooling: canonical flow is `make check` (ruff check, ruff format --check, pyright --strict, pytest with coverage). Use `make format` to apply formatting (`ruff format`); lint with `ruff check`.
- Type checking: run `pyright --strict` every single time you change any Python file.
- Async: install `uvloop` as the default event loop policy; favor async-first APIs and avoid mixing threads unless bounded and deliberate. `python -m latentscore` runs a demo slice using uvloop.
- Dependency injection: wire services with `dependency-injector`; override providers in tests for determinism.
- Testing: design from UX/DX outward—define intended usage, write tests that capture that DX/UX, then implement to satisfy tests. Target 95%+ coverage with both unit and integration tests; invest in tooling/visibility that makes debugging easy.
- Complexity: if design feels tangled, prune aggressively and ask the human loop for direction before pushing forward.
- Libraries: grab and skim docs for popular dependencies before changing code; don’t guess. When stuck, inspect library internals to understand behavior rather than patching blindly.
- Functional FP/CAII: aim for commutativity, associativity, idempotence, immutability; prefer pure helpers, lazy generators where helpful, and avoid hidden state.
- Testing workflow examples: for quick spikes, drop assertions under an `if __name__ == "__main__":` or early-return block; for integrated pieces, run the main entrypoint with injected dependencies. Use `.inputs`/`.outputs` dirs for sample IO when present.
- Docs hygiene: when behavior/tooling shifts, update all relevant Markdown (README, CONTRIBUTE, AGENTS, docs/templates) in the same change.
- Examples: now live in `docs/examples.md` for quick reference.
