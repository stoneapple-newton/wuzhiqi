---
name: import-raw-md
description: Promote Markdown files from docs/raw into their final documentation destination for this Wuzhiqi project. Use when Codex needs to import, reorganize, rename, summarize, split, or place raw Markdown research/planning files into docs/wiki for project context pages or docs/SAFe for SAFe planning artifacts while keeping docs indexes and links coherent.
---

# Import Raw Markdown

Use this skill to turn raw Markdown under `docs/raw/` into organized project documentation.

## Destination Rules

- Put project context pages under `docs/wiki/`.
- Put SAFe planning artifacts under `docs/SAFe/`.
- Leave source material in `docs/raw/` unless the user explicitly asks to move or delete it.
- Keep generated training data, checkpoints, and experiment logs out of docs.

## Workflow

1. Inspect the raw Markdown and identify its purpose.
2. Choose the destination:
   - Gomoku rules, state encoding, network design, MCTS, training, evaluation, datasets, checkpoints, reproducibility, and project background go to `docs/wiki/`.
   - Epics, features, user stories, PI planning, acceptance criteria, and Definition of Done go to `docs/SAFe/`.
3. Prefer focused pages over large omnibus pages. Split raw files when one file covers multiple concepts.
4. Normalize file names to existing project style:
   - Wiki pages use PascalCase, such as `StateRepresentation.md`.
   - SAFe pages follow the existing names in `docs/SAFe/`.
5. Update the relevant index after adding or renaming pages:
   - `docs/wiki/README.md` for project context pages.
   - `docs/README.md` when top-level navigation changes.
6. Preserve useful citations and caveats from raw research, but remove raw prompt text, duplicated planning boilerplate, and speculative content that is not ready to be source of truth.
7. Keep links relative to the file location.

## Helper Script

For straightforward imports, run:

```powershell
uv run python .agents/skills/import-raw-md/scripts/import_raw_md.py docs/raw/source.md --dest wiki --title "Page Title"
```

Use `--dry-run` first when the destination path is uncertain. Use `--move` only when the user explicitly wants the raw file removed after import.
