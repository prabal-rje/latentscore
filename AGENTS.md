# Repository Notes for Agents

## ⚠️ MANDATORY: Shell Environment Setup

**BEFORE running ANY shell command** (Python, pip, pytest, make, etc.), you MUST first activate the conda environment:

```bash
conda activate sample_app_env
```

This is NON-NEGOTIABLE. Every terminal session requires this. Do NOT run `python`, `pip`, `pytest`, `make check`, or any Python tooling without first activating the environment. If unsure whether the env is active, run `conda activate sample_app_env` anyway—it's idempotent.

*Exception: You may skip this requirement only if the user explicitly instructs you to do otherwise.*

---

## ⚠️ MANDATORY: Run Type Checking with Pyright

**You MUST run Pyright** (`pyright`) every single time you change _any_ Python file. This ensures that type errors are caught as early as possible.

- Run the following command from the project root AFTER ANY PYTHON FILE CHANGE, before any PR or commit:
  ```bash
  conda activate sample_app_env
  pyright
  ```
- Do **not** ignore or silence type errors unless you have a justified, documented reason.
- If you are unsure why Pyright is failing, ask for help immediately or investigate with the `--verbose` flag.

Maintaining strict type integrity is **required for all merges**.

---

## ⚠️ MANDATORY: Write Plan First

**You MUST always use the `write-plan` skill** before executing any task.

- **Plan first, then act.**
- Do not start writing code or running shell commands until you have established a plan.
- This applies to every task, no matter how small.

---

## ⚠️ MANDATORY: Use Context-Appropriate Skills

**You MUST invoke skills based on task context**, in addition to the `write-plan` requirement.

- Use `systematic-debugging` for bug investigations, regressions, or unclear behavior before making changes.
- Use `writing-plans` when the user asks for a plan or when multi-step planning is needed.
- Use `writing-skills` when creating or updating any skill definitions or workflows.
- State which skill(s) you are using and why; if multiple apply, use the minimal set needed.

---

## ⚠️ MANDATORY: Use Agent Memory Before and After Every Action

**You MUST scan `.chats/context/agent_memory.md`** before taking any action (including reading files, running commands, or editing).
**You MUST update `.chats/context/agent_memory.md`** after every action if anything changed, a new gotcha appeared, or a new decision was made.

- Keep updates succinct and linked to the relevant file/folder/concept.
- If the memory file is missing, create it from `skills/memory-use/references/memory-template.md`.

This is NON-NEGOTIABLE. No exceptions.

---

## ⚠️ MANDATORY: Docs Context for Every Library Docs Lookup

**You MUST update memory and the docs context** whenever you need to consult or search a library's documentation.

- First, check `.chats/context/docs/index.md` for existing captured docs.
- If missing, capture the docs into `.chats/context/docs/<library>.md` and add an index entry.
- After using the docs, update `.chats/context/agent_memory.md` with any gotchas, triggers, or mandatory actions.

This is NON-NEGOTIABLE. No exceptions.

---

## ⚠️ MANDATORY: Run `make check` Before Presenting Final Work

**You MUST run `make check`** before presenting any completed work to the user. This ensures all quality gates pass.

- Run the following command from the project root BEFORE presenting final work:
  ```bash
  conda activate sample_app_env
  make check
  ```
- Do **not** present work as complete if `make check` fails.
- If any check fails, fix the issues based on the errors, then re-run `make check` until it passes.
- Use `make fix` for automatic syntax/formatting fixes.
- See the `Makefile` for details on what checks are run.
- Only after a clean `make check` should you present the final work to the user.

This is NON-NEGOTIABLE. No exceptions.

---

## General Guidelines

- Current contents are minimal (`README.md` was empty; `CONTRIBUTE.md` documented Conda setup). Keep additions concise and self-contained.
- Environment: Conda env `sample_app_env` with Python 3.13. If `requirements.txt` appears, install via `pip install -r requirements.txt`.
- Git hygiene: prefer small, reviewable commits (1–3 per PR). Separate refactors/formatting from behavior changes; squash noisy WIP commits before merging.
- Tests/tooling: none present initially. If you add code, include at least smoke tests or runnable examples and note how to execute them.
- Defaults: stay ASCII unless a file already uses Unicode; add comments only when logic is non-obvious; avoid destructive git commands.
- Exception handling: never add bare `try/except` with `pass` or silent swallowing. If handling errors, be explicit about the exception types and the behavior; otherwise let it raise.
- Tooling: canonical flow is `make check` (ruff check, ruff format --check, pyright, pytest with coverage). Use `make format` to apply formatting (`ruff format`); lint with `ruff check`.
- Type checking: run `pyright` every single time you change any Python file.
- Async: install `uvloop` as the default event loop policy; favor async-first APIs and avoid mixing threads unless bounded and deliberate. `python -m sample_app_env` runs a demo slice using uvloop.
- Dependency injection: wire services with `dependency-injector`; override providers in tests for determinism.
- Testing: design from UX/DX outward—define intended usage, write tests that capture that DX/UX, then implement to satisfy tests. Target 95%+ coverage with both unit and integration tests; invest in tooling/visibility that makes debugging easy.
- Complexity: if design feels tangled, prune aggressively and ask the human loop for direction before pushing forward.
- Libraries: grab and skim docs for popular dependencies before changing code; don’t guess. When stuck, inspect library internals to understand behavior rather than patching blindly.
- Functional FP/CAII: aim for commutativity, associativity, idempotence, immutability; prefer pure helpers, lazy generators where helpful, and avoid hidden state.
- Testing workflow examples: for quick spikes, drop assertions under an `if __name__ == "__main__":` or early-return block; for integrated pieces, run the main entrypoint with injected dependencies. Use `.inputs`/`.outputs` dirs for sample IO when present.
- Docs hygiene: when behavior/tooling shifts, update all relevant Markdown (README, CONTRIBUTE, AGENTS, docs/templates) in the same change.
- Examples: now live in `docs/examples.md` for quick reference.
