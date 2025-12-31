# RTFM Examples

## Example 1: Unfamiliar library (Textual)
Trigger: need to modify the TUI or add widgets.

Steps:
1. Check local docs (`README.md`, `docs/examples.md`).
2. Search `.chats/context/docs/index.md` for existing Textual docs.
3. If missing, repomix capture and store `.chats/context/docs/textual-docs.md`.
4. Add index entry with source, scope, tags, triggers.
5. Use the captured docs to implement changes; cite sections.
6. Add memory entry if any gotcha appears (e.g., requires `textual-serve`).

## Example 2: Dependency injection details
Trigger: wiring or overrides need exact API.

Steps:
1. Search repo for `dependency_injector` usage.
2. Check `.chats/context/docs/index.md` for dependency-injector docs.
3. If missing, capture docs (README + docs + src) into `.chats/context/docs/dependency-injector-docs.md`.
4. Update index and memory with any hard rules (override patterns, provider lifetimes).

## Example 3: Local docs are enough
Trigger: need to adjust type checking or linting.

Steps:
1. Read `AGENTS.md`, `pyrightconfig.json`, and `Makefile`.
2. No external docs needed; proceed and update memory if new gotcha is found.
