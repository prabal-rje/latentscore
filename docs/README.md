# LatentScore Docs

Top-level [`README.md`](../README.md) covers install, the headline API,
and where to find everything else. This folder contains longer-form
user-facing docs.

## User-facing

- [`library.md`](library.md) — library guide: model selection
  (`fast`, `fast_heavy`, `expressive`, `external:*`), the audio
  contract, streaming semantics, and `MusicConfig`.
- [`FAQ.md`](FAQ.md) — first-call hang, system deps, Windows / WSL2,
  citation.

## Contributor-facing

- [`contribute/coding-guidelines.md`](contribute/coding-guidelines.md) —
  style rules (strict typing, functional style, pattern matching, no
  silent excepts). Mandatory pre-merge checklist lives here.
- [`contribute/examples.md`](contribute/examples.md) — concrete code
  samples illustrating the style.
- [`contribute/README_TEMPLATE.md`](contribute/README_TEMPLATE.md),
  [`contribute/DESIGN_DOC_TEMPLATE.md`](contribute/DESIGN_DOC_TEMPLATE.md),
  [`contribute/CHANGELOG_TEMPLATE.md`](contribute/CHANGELOG_TEMPLATE.md) —
  templates for new components.

## Research / data pipeline

Moved to [`data_work/docs/`](../data_work/docs/) so it lives next to
the code it describes:

- [`data_work/docs/architecture.md`](../data_work/docs/architecture.md) —
  pipeline flow (download → process → train → benchmark → export).
- [`data_work/docs/ablation-guide.md`](../data_work/docs/ablation-guide.md) —
  parameter axes + evaluation harness for ablation studies.

## Local-only

`plans/` is gitignored — design docs and implementation plans stay
on disk for reference but never ship with the repo.
