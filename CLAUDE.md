# Claude Instructions

Follow the repository rules in [AGENTS.md](AGENTS.md).

This repository is a Chinese-first LLM systems paper library. Paper summaries and analysis should live in handwritten Markdown notes, not in frontend code or duplicated YAML fields.

## Index

- `README.md`: home page, category navigation, reading priority, dashboard link, and maintenance notes.
- `data/papers.yml`: structured metadata consumed by the dashboard and validation script.
- `papers/*.md`: primary paper notes and the source of long-form summaries.
- `categories/*.md`: category-level reading indexes by system layer.
- `skills/paper-analysis/SKILL.md`: default paper analysis workflow for adding, summarizing, revising, and answering questions about papers.
- `web/`: read-only React + Vite dashboard for browsing metadata and rendering Markdown notes.
- `.github/workflows/pages.yml`: GitHub Pages workflow with note validation and dashboard deployment.
- `AGENTS.md`: full agent guidance. Prefer it when instructions conflict.

## Default Skill

For paper analysis tasks, first follow `skills/paper-analysis/SKILL.md`, then apply the repository rules below.

## Note Standard

When updating notes, keep professional terms in English and preserve this exact section order:

- Metadata
- 一句话结论
- Related Works
- 问题与背景
- 方法与系统设计
- 创新点
- 实验与结论
- 局限性与风险
- 对 LLM Systems 的启发
- 复现/阅读建议

The note title must match `data/papers.yml`. The `Metadata` section must include `Year`, `Authors`, `Category`, `Priority`, `Links`, and `Keywords`.

## Adding Papers

When adding a paper:

1. Create `papers/YYYY-short-topic.md` with the standard sections.
2. Add metadata to `data/papers.yml`.
3. Update `README.md`.
4. Update the matching `categories/*.md` page, or create a new category page and link it from README.
5. Run `cd web && npm run validate:notes && npm run build`.

If facts or metrics are unclear, write `待核对` instead of guessing.
