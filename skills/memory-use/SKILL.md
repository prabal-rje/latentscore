---
name: memory-use
description: Maintain and consult agent memory in `.chats/context/agent_memory.md` and the docs index in `.chats/context/docs/index.md`. Use when starting or finishing any action, when touching risky files or unfamiliar concepts, when new gotchas appear, or when capturing external documentation.
---

# Memory Use

## Core rule
- Before any action: open and scan `.chats/context/agent_memory.md`.
- After any action: update memory if you learned something new or changed assumptions.

## Where memory lives
- Primary memory: `.chats/context/agent_memory.md`
- Docs index: `.chats/context/docs/index.md`
- Captured docs: `.chats/context/docs/*.md`

## How to update memory
- Add one section per file/folder/concept.
- Include tags, triggers/hooks, and mandatory actions.
- Keep entries short; link to sources instead of repeating them.
- If the memory file is missing, create it using `references/memory-template.md`.

## Mandatory checks
- If touching a file mentioned in memory, follow its Do/Check items.
- If adding docs, update the docs index and add a memory entry pointing to them.

## Examples
- See `references/memory-template.md` for the section template and high-quality examples.
