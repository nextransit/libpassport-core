# Vendored source pin

This directory is a **read-only vendored snapshot** of an upstream skill library, installed into
the `passport` project per `AGENTS.md` (section 「模型训练与识别 Skill 强制使用」).

| Field | Value |
|---|---|
| Upstream | https://github.com/Orchestra-Research/AI-Research-SKILLs.git |
| Pinned commit | `773a52944ba4747a18bd4ae9ade53fff041adcbc` |
| Upstream version | v1.7.2 (`release: v1.7.2 — ship Qoder agent auto-detection to npm`) |
| Upstream commit date | 2026-06-15 21:36:40 -0400 |
| Installed on | 2026-11-05 |
| Skills installed | 98 `SKILL.md` across 23 category directories |
| Installed size | 8.4 MB (excluding `.git`) |
| License | MIT (see `LICENSE`); individual skills may reference third-party libraries under other licenses |

## What was copied

- All 23 numbered category directories (`0-autoresearch-skill`, `01-*` … `22-*`), including
  each skill's `references/`, `scripts/` and `assets/`.
- Top-level `scripts/`, `README.md`, `WELCOME.md`, `CLAUDE.md`, `CONTRIBUTING.md`,
  `CITATION.cff`, `LICENSE`.

## What was deliberately excluded

`.git/`, `demos/`, `dev_data/`, `docs/`, `packages/`, `anthropic_official_docs/`,
`video-promo/` — promotional, demo, npm-installer and duplicate-doc material not needed to
use the skills. Excluding them keeps the vendored tree to 8.4 MB without losing any
`SKILL.md` or `references/`.

## Updating

Do not edit files in this directory. To upgrade, re-clone the upstream at the desired tag,
diff the 23 category directories, then update the pin fields above and re-run the
acceptance checks listed in `.workbuddy-ai/skills/ai-research-skills-mrz/SKILL.md`.

## Project entry point

The mandatory project-level skill that routes to this library is
`.workbuddy-ai/skills/ai-research-skills-mrz/SKILL.md`. Read it before any model training
or recognition work in this repository.
