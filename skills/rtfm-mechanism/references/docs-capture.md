# Docs Capture Guide

## Naming
- Use kebab-case and include project + scope.
- Example: `textual-docs.md`, `dependency-injector-api.md`.

## Metadata header (mandatory)
Use this at the top of every captured doc file:

```
# <Doc Title>
Source: <url or repo>
Version/Commit: <hash or tag or date>
Captured: <YYYY-MM-DD>
Scope: <include globs or pages captured>
Notes: <optional, 1 line>
```

## Repomix capture patterns
- Choose include globs intentionally; list extensions in the index.
- Prefer docs + README + relevant source folders.
- Keep output in `.chats/context/docs/` as Markdown.

Examples:

```
repomix https://github.com/Textualize/textual \
  --include "README.md,docs/**/*.md,src/**/*.py" \
  --exclude "node_modules,dist" \
  --output .chats/context/docs/textual-docs.md
```

```
repomix https://github.com/ets-labs/python-dependency-injector \
  --include "README.rst,docs/**/*.rst,src/**/*.py" \
  --output .chats/context/docs/dependency-injector-docs.md
```

## If docs are a website
- Capture the minimum authoritative pages.
- Save as Markdown and include the source link in the header and index.

## Search usage
- Use `rg` inside `.chats/context/docs/` before asking to fetch more.
