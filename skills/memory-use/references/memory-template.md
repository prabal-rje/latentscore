# Agent Memory Template

Keep entries short (3-6 lines). One section per file/folder/concept.

## Section template
## <File/Folder/Concept> — <short label>
Tags: [tag1, tag2]
Triggers: <when to use>
Do/Check: <mandatory actions>
Notes: <gotchas, constraints>
Refs: <paths, docs, or links>

## Seed entries (edit to fit)

## AGENTS.md — mandatory workflow
Tags: [process, env, typing, planning]
Triggers: before any shell command; after changing Python; when starting a task.
Do/Check: run `conda activate sample_app_env` before any shell command; run `pyright` after any Python file change; use create-plan skill before tasks.
Notes: env may be missing; still attempt activation; avoid Python tooling without env.
Refs: AGENTS.md

## skills/ — repo shared skills
Tags: [skills, repo]
Triggers: creating/updating skills.
Do/Check: store raw skills under `skills/`; keep create-plan here too.
Notes: favor manual edits; package only when asked.
Refs: skills/

## .chats/context/docs — local doc cache
Tags: [docs, rtfm]
Triggers: when you need external docs, API details, or version-specific behavior.
Do/Check: store captured docs in `.chats/context/docs/`; update `.chats/context/docs/index.md`.
Notes: use repomix to snapshot repos; record scope and source in index.
Refs: .chats/context/docs/index.md
