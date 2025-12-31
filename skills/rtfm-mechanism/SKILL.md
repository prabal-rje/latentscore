---
name: rtfm-mechanism
description: Docs-first workflow that prioritizes reading and capturing authoritative documentation into `.chats/context/docs/`. Use when behavior is unclear, when working with unfamiliar libraries/APIs, when accuracy depends on a specific version, or when a request implies "read the docs".
---

# RTFM Mechanism

## Core rule
- Prefer local docs: repo files and `.chats/context/docs/index.md`.
- Do not guess; cite docs or capture them.
- If docs are missing, capture them into `.chats/context/docs/` and update the index.

## Workflow
1. Define the exact behavior or question to verify.
2. Search local sources (`README.md`, `docs/`, `pyproject.toml`, `.chats/context/docs/index.md`).
3. If still unclear, acquire docs with repomix or manual capture.
4. Save as Markdown under `.chats/context/docs/` with metadata header.
5. Update `.chats/context/docs/index.md` with source, scope, tags, and triggers.
6. If a new gotcha appears, update `.chats/context/agent_memory.md`.

## Capture guidance
- Use `references/docs-capture.md` for naming, metadata, and repomix patterns.
- Use `references/examples.md` for high-quality examples.
