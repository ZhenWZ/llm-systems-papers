# Repository Instructions

This repository is a Chinese-first paper library for LLM systems research. Keep professional terms in English when that makes search and comparison with papers easier, for example `prefill`, `decode`, `KV cache`, `GEMM`, `epilogue`, `sparse attention`, `kernel`, and `throughput`.

## Index

- `README.md`: repository home page, categorized reading map, reading priority table, dashboard link, and maintenance notes.
- `data/papers.yml`: structured metadata used by the dashboard and validation scripts.
- `papers/*.md`: handwritten paper notes. This is the primary source for summaries and analysis.
- `categories/*.md`: system-layer category indexes. Each paper must appear in the category page matching its primary `category`.
- `web/`: React + Vite Notion Database-style dashboard. It reads `data/papers.yml` and renders full Markdown notes.
- `skills/paper-analysis/SKILL.md`: default paper analysis skill for agents summarizing, adding, updating, or answering questions about papers.
- `.github/workflows/pages.yml`: GitHub Pages workflow. It installs frontend dependencies, validates note format, builds the dashboard, and deploys Pages.
- `CLAUDE.md`: Claude-compatible instruction entry point.
- `AGNETS.md`: compatibility pointer for the earlier misspelled filename request.

## Scope

- Maintain the home page as a categorized reading map with priority, reading links, and note links.
- Keep paper notes under `papers/`.
- Keep system-layer category indexes under `categories/`.
- Keep structured metadata in `data/papers.yml` when adding or changing paper entries.
- Keep the dashboard as a read-only view over Markdown notes and YAML metadata. Do not duplicate paper summaries into frontend code.

## Default Paper Analysis Skill

When an agent is asked to summarize, analyze, add, revise, or answer questions about a paper, first load and follow `skills/paper-analysis/SKILL.md`. Treat it as the default workflow for paper analysis in this repository.

## Paper Note Standard

Each paper note must use this exact section order:

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

The Markdown note title must match `data/papers.yml` exactly. The `Metadata` section must include `Year`, `Authors`, `Category`, `Priority`, `Links`, and `Keywords`. If a section is not ready, keep the section heading and write `待补充` or `待核对`; do not remove the section.

## Adding A Paper

1. Create a note under `papers/YYYY-short-topic.md`.
2. Use the standard note sections above, Chinese-first prose, and English technical terms where useful.
3. Add an entry to `data/papers.yml` with stable lower-kebab-case `id`, `title`, `year`, `priority`, `category`, `phase`, `links`, `note`, and non-empty `tags`.
4. Add the paper to the reading priority table and current collection table in `README.md`.
5. Add the paper to the matching `categories/*.md` page. If the system layer is new, create a category page and add it to the README category navigation.
6. Keep `note` paths relative to the repository root, for example `papers/2026-example-paper.md`.
7. Run `cd web && npm run validate:notes && npm run build` before committing.

## Classification

Use system-layer categories. Prefer categories that describe systems bottlenecks or implementation layers, not only model families.

Current categories:

- `Kernel & Operator Systems`
- `Long-Context Attention & KV Cache`
- `Inference Serving & Speculative Decoding`
- `Agent Skill Optimization & Procedural Memory`

## Priority Convention

- `P0`: near-term must-read or reproduction target.
- `P1`: important direction worth detailed reading.
- `P2`: background or tracking item.

## Editing Style

- Chinese prose, concise but technically precise.
- Keep names, acronyms, benchmark names, and core systems terms in English.
- Prefer tables for homepage indexes.
- Do not invent paper results. If a metric is unclear, mark it as `待核对`.
- Preserve existing public links and add new external links only when they are primary sources or clearly useful for reading/reproduction.
