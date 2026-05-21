# LatentScore Docs

Top-level [`README.md`](../README.md) covers install, the headline API,
and where to find everything else. This folder contains longer-form docs.

## Reading docs (reviewer + user-facing)

- [`latentscore-dx.md`](latentscore-dx.md) — library DX: model selection
  (`fast`, `fast_heavy`, `expressive`, `external:*`), the audio contract,
  streaming semantics, and `MusicConfig`.
- [`architecture.md`](architecture.md) — system architecture overview.
- [`ablation-guide.md`](ablation-guide.md) — research ablation methodology
  (parameter axes, evaluation harness). Companion to `data_work/`.

## Contributing

See [`contribute/`](contribute/):

- [`contribute/coding-guidelines.md`](contribute/coding-guidelines.md) —
  the project's style rules (strict typing, functional style, pattern
  matching, no silent excepts). The mandatory pre-merge checklist lives
  here.
- [`contribute/examples.md`](contribute/examples.md) — concrete code
  samples illustrating the style.
- [`contribute/CHANGELOG_TEMPLATE.md`](contribute/CHANGELOG_TEMPLATE.md)
- [`contribute/DESIGN_DOC_TEMPLATE.md`](contribute/DESIGN_DOC_TEMPLATE.md)
- [`contribute/README_TEMPLATE.md`](contribute/README_TEMPLATE.md)

## Local-only

`plans/` is gitignored — design docs and historical implementation plans
stay on disk for reference but never ship with the repo.
